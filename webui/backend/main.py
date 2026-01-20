from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from typing import List, Optional, Dict, Tuple
import sys
import threading
import shutil
import json
from datetime import datetime

# Add the scripts directory to path to import existing logic
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'scripts')))

try:
    from auto_labeler import YOLOInference
    from crop_resize import process_image, get_supported_image_files
    from split_folder import split_folder
except ImportError as e:
    print(f"Error importing scripts: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Progress Tracking
class TaskStatus(BaseModel):
    status: str
    current: int
    total: int
    message: str = ""
    result: Optional[dict] = None

app.state.task_progress = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "message": "",
    "result": None
}

class DatasetInfo(BaseModel):
    path: str

class CropParams(BaseModel):
    x: int
    y: int
    width: int
    height: int

class PreprocessRequest(BaseModel):
    dataset_path: str
    crop: Optional[CropParams] = None
    resize_width: int = 640
    resize_height: int = 640
    output_dir: Optional[str] = None

class AutoLabelRequest(BaseModel):
    dataset_path: str
    model_path: str
    confidence: float = 0.5
    save_masked: bool = False

class FilterRequest(BaseModel):
    image_name: str
    source_dir: str # Path to masked or images dir
    labeled_dir: Optional[str] = None # Path to original labeled directory
    target_name: str = "filtered"

class MergeRequest(BaseModel):
    source_paths: List[str]
    output_path: str

class ExtractRequest(BaseModel):
    source_path: str
    selected_classes: List[str] # List of class names
    output_path: str

class GenerateMaskRequest(BaseModel):
    dataset_path: str
    labels_path: Optional[str] = None
    output_path: Optional[str] = None

class ProjectConfig(BaseModel):
    dataset_path: str
    processed_dir: Optional[str] = None
    labeled_dir: Optional[str] = None
    masked_dir: Optional[str] = None
    model_path: Optional[str] = None
    crop: Optional[CropParams] = None
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    last_modified: str = ""

class ExtractEmptyRequest(BaseModel):
    dataset_path: str
    labeled_root: str

class SplitRequest(BaseModel):
    input_path: str
    num_splits: int

def find_project_root(path: Path) -> Path:
    """Heuristic to find the 'main' project directory by traversing up from known subfolders."""
    p = path.absolute()
    
    # If parent already has a config (prefixed or generic), that's a very strong indicator
    parent_dir, identity_name = get_config_identity(p)
    generic_conf = parent_dir / "yolo_project_config.json"
    prefixed_conf = parent_dir / f"{identity_name}_yolo_project_config.json"
    if generic_conf.exists() or prefixed_conf.exists():
        return parent_dir

    # Special check: if p itself is a common subfolder, check sibling configs
    # But find_project_root is usually called on something that MIGHT be the root
    
    while True:
        name_lower = p.name.lower()
        # Check if current folder name matches known subfolders or common patterns
        if name_lower.endswith("_processed") or \
           name_lower.endswith("_images") or \
           name_lower.endswith("_labels") or \
           name_lower.endswith("_labeled") or \
           name_lower in ["labeled", "masked_images", "images", "labels", "train", "valid", "test", "raw", "data", "masked"]:
            if p.parent == p: break
            p = p.parent
        else:
            break
    return p

def get_config_identity(path: Path) -> Tuple[Path, str]:
    """Finds the directory where config should live and the base name for the config file."""
    p = path.absolute()
    # Climb up known subfolders
    while True:
        name_lower = p.name.lower()
        if name_lower in ["labels", "images", "masked", "masked_images", "train", "val", "test", "data"]:
             if p.parent == p: break
             p = p.parent
        else:
             break
    
    # Now p is the "dataset folder indentifier" (e.g. test_images, or test_images_processed)
    # Refine identity by stripping task suffixes to get the "Base Identity"
    identity = p.name
    # Including common image folder names to strip
    # we do NOT strip _images or _masks from the identity itself, because "test_images" is a valid identity
    task_suffixes = ["_processed", "_labeled", "_masked_images", "_filtered", "_empty"]
    while True:
        changed = False
        for s in task_suffixes:
            if identity.endswith(s):
                identity = identity[:-len(s)]
                changed = True
        if not changed:
            break
            
    return p.parent, identity

def save_project_config(dataset_path: Path, config: dict):
    """Updates the project config file using a unified Base Identity approach."""
    root, identity = get_config_identity(dataset_path)
    # The canonical config file name
    base_config_path = root / f"{identity}_yolo_project_config.json"
    
    print(f"DEBUG: Saving config to {base_config_path} (ID: {identity})")
    
    # Load and merge all existing related configs to ensure no data loss during transition
    existing = load_project_config(dataset_path)
            
    # Update with new values
    existing.update(config)
    existing['last_modified'] = datetime.now().isoformat()
    existing['dataset_path'] = str(root.absolute())
    
    try:
        with open(base_config_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"DEBUG: Successfully consolidated project config to {base_config_path}")
    except Exception as e:
        print(f"ERROR: saving config: {e}")

def load_project_config(dataset_path: Path) -> dict:
    """Loads and merges all related config files for a given dataset identity."""
    root, identity = get_config_identity(dataset_path)
    
    merged_data = {}
    
    # 1. Try legacy generic name (lowest priority)
    legacy_path = root / "yolo_project_config.json"
    if legacy_path.exists():
        try:
            with open(legacy_path, 'r') as f:
                merged_data.update(json.load(f))
        except: pass

    # 2. Find all files matching the identity prefix
    # identity_yolo_project_config.json, identity_processed_yolo_project_config.json, etc.
    config_files = list(root.glob(f"{identity}*_yolo_project_config.json"))
    # Sort by modification time to preserve most recent updates during merge
    config_files.sort(key=lambda x: x.stat().st_mtime)
    
    if not config_files and not legacy_path.exists():
        return {}

    for cp in config_files:
        try:
            with open(cp, 'r') as f:
                merged_data.update(json.load(f))
        except: pass
        
    print(f"DEBUG: Loaded unified config for {identity} from {len(config_files)} files")
    return merged_data

def discover_data_yaml(start_path: Path) -> Optional[Path]:
    """Climb up and search for data.yaml in common locations."""
    dataset_name = start_path.name
    # Strip only 'task' suffixes to find immediate prefix context
    # e.g. test_images_processed_masked_images -> test_images_processed
    prefix = dataset_name
    for s in ["_masked_images", "_labeled", "_filtered", "_empty"]:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
            break

    curr = start_path.absolute()
    # Check current and 3 parents
    for _ in range(4):
        # 1. Check siblings that match the prefix
        if curr.parent and curr.parent.exists():
            for item in curr.parent.iterdir():
                if item.is_dir() and item.name.startswith(prefix):
                    # Check in root of the sibling or common subfolders
                    for cand in [item / "data.yaml", item / "labeled" / "data.yaml", item / "processed" / "data.yaml"]:
                        if cand.exists():
                            return cand

        # 2. Traditional candidates
        candidates = [
            curr / "data.yaml",
            curr / f"{dataset_name}_labeled" / "data.yaml",
            curr / "labeled" / "data.yaml",
            curr / f"{dataset_name}_processed" / "data.yaml",
            curr / "processed" / "data.yaml",
            curr / "dataset" / "data.yaml"
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        if curr.parent == curr:
            break
        curr = curr.parent
    return None

@app.get("/api/dataset/classes")
def get_dataset_classes(path: str):
    p = Path(path).absolute()
    yaml_path = discover_data_yaml(p)
    
    if not yaml_path:
        raise HTTPException(status_code=400, detail=f"data.yaml not found near {path}")
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            if isinstance(names, dict):
                # Ensure keys are sorted if it's a dict
                sorted_keys = sorted([int(k) for k in names.keys()])
                names = [names.get(str(k) if str(k) in names else k) for k in sorted_keys]
            return {"classes": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading yaml: {str(e)}")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/progress")
def get_progress():
    return app.state.task_progress

@app.post("/api/dataset/info")
def get_dataset_info(request: DatasetInfo):
    path = Path(request.path)
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    
    image_files = get_supported_image_files(path)
    return {
        "count": len(image_files),
        "path": str(path.absolute())
    }

@app.get("/api/dataset/status")
def get_dataset_status(path: str):
    p = Path(path).absolute()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid path")
    
    print(f"Checking status for: {p}")
    dataset_name = p.name
    # Check for processed folder
    processed_dir = p.parent / f"{dataset_name}_processed"
    has_processed = processed_dir.exists() and processed_dir.is_dir()
    
    # Priority: prefixed folder > legacy non-prefixed folder
    # NEW: Robust discovery using glob to find folders starting with dataset_name and ending with suffixes
    candidates = []
    
    # 1. Search siblings of dataset
    for item in p.parent.iterdir():
        if item.is_dir() and item.name.startswith(dataset_name):
            if any(item.name.endswith(s) for s in ["_labeled", "_masked_images", "_processed_labeled", "_processed_masked_images"]):
                candidates.append(item)
    
    # 2. Add legacy and explicit paths
    labeled_dirs = candidates + [
        p.parent / f"{dataset_name}_masked_images",
        processed_dir / f"{dataset_name}_masked_images",
        p / f"{dataset_name}_masked_images",
        p.parent / f"{dataset_name}_labeled",
        processed_dir / f"{dataset_name}_labeled",
        p / f"{dataset_name}_labeled",
        # Legacy non-prefixed versions
        p.parent / "masked_images",
        processed_dir / "masked_images",
        p / "masked_images",
        p.parent / "labeled",
        processed_dir / "labeled",
        p / "labeled"
    ]
    
    # Deduplicate while preserving order
    seen = set()
    labeled_dirs = [x for x in labeled_dirs if not (x in seen or seen.add(x))]

    found_labeled = None
    labels_root = None
    has_masks = False

    for ld in labeled_dirs:
        if ld.exists() and ld.is_dir():
            # Check for files in root
            files = get_supported_image_files(ld)
            if files:
                found_labeled = str(ld.absolute())
                has_masks = True
                break
            
            # Check for files in known subdirs
            for sub in ["masked_images", "images", "masked"]:
                sub_dir = ld / sub
                if sub_dir.exists() and sub_dir.is_dir():
                    sub_files = get_supported_image_files(sub_dir)
                    if sub_files:
                        found_labeled = str(sub_dir.absolute())
                        has_masks = True
                        break
            if found_labeled:
                break
                
    # Also check specifically for labels directory to suggest mask generation
    labels_root = None
    
    # NEW: Try to find labels in the robustly discovered candidate directories first
    for ld in labeled_dirs:
        # If ld is literally a labels folder
        if ld.name == "labels" and ld.is_dir():
            labels_root = str(ld.absolute())
            break
        # If it's a labeled root, check common subfolders
        for sub in ["labels", "labels/train", "labels/val", "labeled/labels"]:
            sub_p = ld / sub
            if sub_p.exists() and sub_p.is_dir():
                labels_root = str(sub_p.absolute())
                break
        if labels_root:
            break

    # Fallback to current/parent if not found in candidates
    if not labels_root:
        # Determine identifying prefix by stripping ALL technical suffixes
        p_path = Path(p).absolute()
        _, prefix = get_config_identity(p_path)
        
        # Climb up and search
        curr = p_path
        for _ in range(3):
            candidates = [
                curr / "labels", 
                curr / "labeling" / "labels", 
                curr / "labeled" / "labels",
                curr / f"{prefix}_labeled" / "labels",
                curr.parent / f"{prefix}_labeled" / "labels"
            ]
            for cand in candidates:
                if cand.exists() and cand.is_dir():
                    labels_root = str(cand.absolute())
                    break
            if labels_root or curr.parent == curr:
                break
            curr = curr.parent

    if labels_root and not found_labeled:
         found_labeled = str(p.absolute())

    config = load_project_config(p)
    
    return {
        "processed_dir": str(processed_dir.absolute()) if has_processed else None,
        "labeled_dir": found_labeled,
        "labels_root": labels_root,
        "has_labels": labels_root is not None,
        "has_masks": has_masks,
        "config": config
    }

@app.get("/api/dataset/sample")
def get_sample_image(path: str):
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid path")
    
    image_files = get_supported_image_files(p)
    if not image_files:
        raise HTTPException(status_code=404, detail="No images found")
    
    # Get original dimensions
    img = cv2.imread(str(image_files[0]))
    height, width = img.shape[:2]
    
    return {
        "sample_url": f"/static/dataset/{image_files[0].name}", 
        "filename": image_files[0].name,
        "width": width,
        "height": height
    }

def run_preprocess_task(request: PreprocessRequest):
    app.state.task_progress["status"] = "processing"
    app.state.task_progress["message"] = "Starting pre-processing..."
    
    input_dir = Path(request.dataset_path)
    output_dir = Path(request.output_dir) if request.output_dir else input_dir.parent / f"{input_dir.name}_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_supported_image_files(input_dir)
    app.state.task_progress["total"] = len(image_files)
    app.state.task_progress["current"] = 0
    
    crop_params = (request.crop.x, request.crop.y, request.crop.width, request.crop.height) if request.crop else None
    resize_dims = (request.resize_width, request.resize_height)

    success_count = 0
    for i, img_file in enumerate(image_files):
        output_path = output_dir / img_file.name
        if process_image(img_file, output_path, crop_params, resize_dims):
            success_count += 1
        app.state.task_progress["current"] = i + 1
    
    app.state.task_progress["status"] = "idle"
    app.state.task_progress["message"] = f"Finished. Processed {success_count} images."
    app.state.task_progress["result"] = {
        "processed_count": success_count,
        "output_dir": str(output_dir.absolute())
    }
    
    # Save to config
    save_project_config(Path(request.dataset_path), {
        "processed_dir": request.output_dir or str(output_dir.absolute()),
        "crop": request.crop.dict() if request.crop else None,
        "resize_width": request.resize_width,
        "resize_height": request.resize_height
    })

@app.post("/api/preprocess")
def preprocess_dataset(request: PreprocessRequest, background_tasks: BackgroundTasks):
    input_dir = Path(request.dataset_path)
    if not input_dir.exists():
        raise HTTPException(status_code=400, detail="Input directory does not exist")

    # Set status before queueing
    app.state.task_progress["status"] = "processing"
    app.state.task_progress["message"] = "Initializing pre-processing..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0
    
    background_tasks.add_task(run_preprocess_task, request)
    return {"status": "started"}

def run_autolabel_task(request: AutoLabelRequest):
    try:
        app.state.task_progress["status"] = "labeling"
        app.state.task_progress["message"] = "Loading YOLO model..."
        
        img_dir = Path(request.dataset_path)
        inferencer = YOLOInference(
            model_path=request.model_path,
            image_dir=str(img_dir),
            confidence=request.confidence,
            save_masked_images=request.save_masked,
            single_folder=True
        )
        
        # We need to monkey patch or modify YOLOInference to report progress
        # For now, we'll monitor the output directory or just simulate if we can't easily hook into tqdm
        # Since YOLOInference uses tqdm, we could try to capture it, but simple way is to use its logic here
        
        image_paths = list(inferencer.image_dir_.glob('*.png')) + list(inferencer.image_dir_.glob('*.jpg')) + list(inferencer.image_dir_.glob('*.tif'))
        app.state.task_progress["total"] = len(image_paths)
        app.state.task_progress["current"] = 0
        app.state.task_progress["message"] = "Starting auto-labeling..."

        def progress_callback(current, total):
            app.state.task_progress["current"] = current
            app.state.task_progress["total"] = total
            
        inferencer.run_inference(callback=progress_callback)
        
        app.state.task_progress["current"] = len(image_paths)
        app.state.task_progress["message"] = "Generating data.yaml..."

        # Generate data.yaml
        nc = len(inferencer.model_.names)
        names = list(inferencer.model_.names.values())
        
        data_yaml = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': nc,
            'names': names
        }
        
        # Prefixed labeled directory
        labeled_dir_path = img_dir.parent / f"{img_dir.name}_labeled"
        labeled_dir_path.mkdir(parents=True, exist_ok=True)
        
        yaml_path = labeled_dir_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
            
        # Move generated files if YOLOInference saved them to 'labeled'
        # Note: YOLOInference currently hardcodes 'labeled'. We'll need to update it too.
        # But if we update YOLOInference, it will use the prefixed name directly.
        
        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = "Auto-labeling complete!"
        app.state.task_progress["result"] = {
            "labeled_dir": str(labeled_dir_path.absolute()),
            "masked_dir": str(inferencer.masked_images_dir_.absolute()),
            "yaml_path": str(yaml_path.absolute()),
            "classes": names
        }
        
        # Save to config
        config_update = {
            "model_path": request.model_path,
            "labeled_dir": str(labeled_dir_path.absolute())
        }
        if request.save_masked:
             config_update["masked_dir"] = str(inferencer.masked_images_dir_.absolute())
             
        save_project_config(Path(request.dataset_path), config_update)
    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Error: {str(e)}"

@app.post("/api/autolabel")
def autolabel_dataset(request: AutoLabelRequest, background_tasks: BackgroundTasks):
    img_dir = Path(request.dataset_path)
    if not img_dir.exists():
        raise HTTPException(status_code=400, detail="Dataset path does not exist")
    
    if not request.model_path or not Path(request.model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {request.model_path}")

    # Set status before queueing
    app.state.task_progress["status"] = "labeling"
    app.state.task_progress["message"] = "Initializing auto-labeling..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0

    background_tasks.add_task(run_autolabel_task, request)
    return {"status": "started"}

def run_merge_task(request: MergeRequest):
    try:
        app.state.task_progress["status"] = "merging"
        app.state.task_progress["message"] = "Verifying datasets..."
        app.state.task_progress["current"] = 0
        app.state.task_progress["total"] = 0
        
        source_paths = [Path(p).absolute() for p in request.source_paths]
        output_dir = Path(request.output_path).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Sanity Checks & Collection
        valid_datasets = []
        all_class_names = []
        class_mappings = [] # List of dicts
        
        for p in source_paths:
            if not p.exists() or not p.is_dir():
                 raise ValueError(f"Dataset path does not exist or is not a directory: {p}")
            
            yaml_path = p / "data.yaml"
            if not yaml_path.exists():
                raise ValueError(f"Missing data.yaml in: {p}")
            
            # Check for images and labels
            # Could be images/labels in root or inside splits
            # We will support both but simplified: check if it has 'images' and 'labels' anywhere
            # Or assume standard structure: p/images and p/labels or p/train/images etc.
            # Let's look for images and labels subfolders in the provided path
            if not (p / "images").exists() or not (p / "labels").exists():
                # Check if it's a split-based dataset (train/valid/test)
                has_split = False
                for split in ["train", "valid", "test"]:
                    if (p / split / "images").exists() and (p / split / "labels").exists():
                        has_split = True
                        break
                if not has_split:
                    raise ValueError(f"Dataset {p} must contain 'images' and 'labels' subfolders (or standard splits)")

            # Read labels
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', [])
                if isinstance(names, dict):
                    names = [names.get(i, f"class_{i}") for i in range(max(names.keys()) + 1)]
                
                # Update global class names
                local_to_global = {}
                for local_idx, name in enumerate(names):
                    if name not in all_class_names:
                        global_idx = len(all_class_names)
                        all_class_names.append(name)
                    else:
                        global_idx = all_class_names.index(name)
                    local_to_global[local_idx] = global_idx
                
                class_mappings.append(local_to_global)
                valid_datasets.append(p)

        # 2. Count total files
        total_files = 0
        file_list = [] # List of (img_path, label_path, dataset_idx)
        for i, p in enumerate(valid_datasets):
            # Find all images and their labels
            # Handle both flat and split structure
            for root, dirs, files in os.walk(p):
                if "images" in root:
                    for f in files:
                        if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                            img_path = Path(root) / f
                            # Find corresponding label in 'labels' sibling folder
                            # images -> root, labels -> root.parent / 'labels'
                            label_root = Path(root).parent / "labels"
                            label_path = label_root / f"{img_path.stem}.txt"
                            if label_path.exists():
                                file_list.append((img_path, label_path, i))
        
        app.state.task_progress["total"] = len(file_list)
        app.state.task_progress["message"] = f"Found {len(file_list)} images to merge. Starting copy..."

        # 3. Copy with renaming and remapping
        os.makedirs(output_dir / "images", exist_ok=True)
        os.makedirs(output_dir / "labels", exist_ok=True)
        
        name_counts = {} # To handle collisions
        
        for i, (img_src, label_src, dataset_idx) in enumerate(file_list):
            img_name = img_src.name
            target_name = img_name
            
            # Conflict handling: prepend dataset index if collision
            if img_name in name_counts:
                 target_name = f"d{dataset_idx}_{img_name}"
            name_counts[target_name] = True
            
            # Copy image
            shutil.copy2(img_src, output_dir / "images" / target_name)
            
            # Remap labels
            mapping = class_mappings[dataset_idx]
            remapped_lines = []
            with open(label_src, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        local_cls = int(parts[0])
                        global_cls = mapping.get(local_cls, local_cls)
                        parts[0] = str(global_cls)
                        remapped_lines.append(" ".join(parts) + "\n")
            
            with open(output_dir / "labels" / f"{Path(target_name).stem}.txt", 'w') as f:
                f.writelines(remapped_lines)
            
            app.state.task_progress["current"] = i + 1

        # 4. Create data.yaml
        new_yaml = {
            'train': './images', # Simplified output structure
            'val': './images',
            'nc': len(all_class_names),
            'names': all_class_names
        }
        with open(output_dir / "data.yaml", 'w') as f:
            yaml.dump(new_yaml, f)

        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Merge complete! Consolidated {len(file_list)} images into {output_dir}"
        app.state.task_progress["result"] = {
            "output_dir": str(output_dir.absolute()),
            "total_images": len(file_list),
            "classes": all_class_names
        }

    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Merge Error: {str(e)}"
        print(f"Merge Error: {e}")

@app.post("/api/merge_datasets")
def merge_datasets(request: MergeRequest, background_tasks: BackgroundTasks):
    if not request.source_paths:
        raise HTTPException(status_code=400, detail="No source paths provided")
    
    # Initialize status
    app.state.task_progress["status"] = "merging"
    app.state.task_progress["message"] = "Initializing merge..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0

    background_tasks.add_task(run_merge_task, request)
    return {"status": "started"}

def run_extract_task(request: ExtractRequest):
    try:
        app.state.task_progress["status"] = "extracting"
        app.state.task_progress["message"] = "Starting extraction..."
        app.state.task_progress["current"] = 0
        app.state.task_progress["total"] = 0
        
        src_path = Path(request.source_path).absolute()
        out_path = Path(request.output_path).resolve()
        
        # 1. Get classes and indices
        yaml_path = src_path / "data.yaml"
        if not yaml_path.exists():
            root = find_project_root(src_path)
            yaml_path = root / "data.yaml"
        
        if not yaml_path.exists():
             raise ValueError(f"data.yaml not found in {src_path}")
             
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            all_names = data.get('names', [])
            if isinstance(all_names, dict):
                all_names = [all_names.get(i, f"class_{i}") for i in range(max(all_names.keys()) + 1)]
                
        target_indices = []
        for name in request.selected_classes:
            if name in all_names:
                target_indices.append(all_names.index(name))
        
        if not target_indices:
            raise ValueError("None of the selected classes were found in the dataset")

        # 2. Collect files
        # We search in splits if they exist, or root images/labels
        file_pairs = []
        for root, dirs, files in os.walk(src_path):
            if "images" in root:
                for f in files:
                    if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        img_p = Path(root) / f
                        label_root = Path(root).parent / "labels"
                        lbl_p = label_root / f"{img_p.stem}.txt"
                        if lbl_p.exists():
                            file_pairs.append((img_p, lbl_p))
        
        app.state.task_progress["total"] = len(file_pairs)
        app.state.task_progress["message"] = f"Found {len(file_pairs)} images to scan..."

        # 3. Process
        os.makedirs(out_path / "images", exist_ok=True)
        os.makedirs(out_path / "labels", exist_ok=True)
        
        extracted_count = 0
        name_counts = {}
        
        for i, (img_src, lbl_src) in enumerate(file_pairs):
            match = False
            with open(lbl_src, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_idx = int(parts[0])
                        if cls_idx in target_indices:
                            match = True
                            break
            
            if match:
                img_name = img_src.name
                target_img_name = img_name
                # Basic collision handling (different splits might have same name)
                if img_name in name_counts:
                     name_counts[img_name] += 1
                     target_img_name = f"{img_src.stem}_{name_counts[img_name]}{img_src.suffix}"
                else:
                    name_counts[img_name] = 0
                
                shutil.copy2(img_src, out_path / "images" / target_img_name)
                shutil.copy2(lbl_src, out_path / "labels" / f"{Path(target_img_name).stem}.txt")
                extracted_count += 1
            
            app.state.task_progress["current"] = i + 1
            if (i+1) % 10 == 0:
                app.state.task_progress["message"] = f"Processed {i+1}/{len(file_pairs)}... Extracted: {extracted_count}"

        # 4. Create new data.yaml
        # For simplicity, keep all original names or just selected? 
        # Usually it's better to keep indices the same if we copy the label files without editing.
        # But here we copy FULL labels, so names must match original NC.
        new_yaml = {
            'train': './images',
            'val': './images',
            'nc': len(all_names),
            'names': all_names
        }
        with open(out_path / "data.yaml", 'w') as f:
            yaml.dump(new_yaml, f)

        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Extraction complete! Saved {extracted_count} matching images to {out_path}"
        app.state.task_progress["result"] = {
            "output_dir": str(out_path.absolute()),
            "extracted_count": extracted_count,
            "total_scanned": len(file_pairs)
        }

    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Extraction Error: {str(e)}"

def run_extract_empty_task(request: ExtractEmptyRequest):
    try:
        app.state.task_progress["status"] = "extracting_empty"
        app.state.task_progress["message"] = "Initializing extraction of empty images..."
        app.state.task_progress["current"] = 0
        app.state.task_progress["total"] = 0
        
        raw_path = Path(request.dataset_path).absolute()
        labeled_root = Path(request.labeled_root).absolute()
        
        # Determine labels directory
        labels_dir = None
        if (labeled_root / "labels").exists():
            labels_dir = labeled_root / "labels"
        else:
            labels_dir = labeled_root
            
        # Determine images directory to pull from (prioritize processed if available)
        img_src_path = raw_path
        if (labeled_root / "images").exists():
            img_src_path = labeled_root / "images"
            
        # Determine output directory (sibling to raw_path)
        output_dir = raw_path.parent / f"{raw_path.name}_empty"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = get_supported_image_files(img_src_path)
        app.state.task_progress["total"] = len(image_files)
        
        empty_count = 0
        for i, img_p in enumerate(image_files):
            # Check for label file
            lbl_p = labels_dir / f"{img_p.stem}.txt"
            
            is_empty = False
            if not lbl_p.exists():
                is_empty = True
            else:
                try:
                    with open(lbl_p, 'r') as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                        if not lines:
                            is_empty = True
                except:
                    is_empty = True
            
            if is_empty:
                shutil.copy2(img_p, output_dir / img_p.name)
                empty_count += 1
            
            app.state.task_progress["current"] = i + 1
            if (i+1) % 10 == 0:
                app.state.task_progress["message"] = f"Processed {i+1}/{len(image_files)}... Found {empty_count} empty."
                
        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Extraction complete! Saved {empty_count} missing/empty label images to {output_dir}"
        app.state.task_progress["result"] = {"output_dir": str(output_dir.absolute()), "count": empty_count}
        
    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Extraction Error: {str(e)}"
        print(f"Extraction Error: {e}")

@app.post("/api/dataset/extract_empty")
def extract_empty_images(request: ExtractEmptyRequest, background_tasks: BackgroundTasks):
    app.state.task_progress["status"] = "extracting_empty"
    app.state.task_progress["message"] = "Starting extraction task..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0
    background_tasks.add_task(run_extract_empty_task, request)
    return {"status": "started"}

@app.post("/api/extract_by_class")
def extract_by_class(request: ExtractRequest, background_tasks: BackgroundTasks):
    if not request.selected_classes:
        raise HTTPException(status_code=400, detail="No classes selected")
    
    # Initialize status
    app.state.task_progress["status"] = "extracting"
    app.state.task_progress["message"] = "Initializing extraction..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0

    background_tasks.add_task(run_extract_task, request)
    return {"status": "started"}

@app.post("/api/generate_masked")
def generate_masked_images_endpoint(request: GenerateMaskRequest, background_tasks: BackgroundTasks):
    # Initialize progress
    app.state.task_progress["status"] = "generating_masks"
    app.state.task_progress["message"] = "Initializing mask generation..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0
    
    background_tasks.add_task(run_mask_generation_task, request)
    return {"status": "started"}

def run_mask_generation_task(request: GenerateMaskRequest):
    try:
        app.state.task_progress["status"] = "generating_masks"
        
        raw_path = Path(request.dataset_path).absolute()
        
        # Determine labels path if not provided
        labels_path = None
        if request.labels_path:
            labels_path = Path(request.labels_path).absolute()
        else:
            # Try to find labels sibling or inside
            potential = [raw_path.parent / "labels", raw_path / "labels"]
            for p in potential:
                if p.exists():
                    labels_path = p
                    break
        
        if not labels_path:
             raise ValueError("Could not find labels directory automatically. Please provide it.")

        # Heuristic: if labels_path doesn't contain .txt files directly, but has a 'labels' subfolder, use that
        if not any(labels_path.glob("*.txt")):
            if (labels_path / "labels").exists():
                labels_path = labels_path / "labels"
            elif (labels_path.parent / "labels").exists():
                labels_path = labels_path.parent / "labels"

        # Determine which images to use as base
        # If labels_path is .../labeled/labels, check if .../labeled/images exists
        img_src_path = raw_path
        if labels_path.name == "labels":
            potential_imgs = labels_path.parent / "images"
            if potential_imgs.exists() and any(get_supported_image_files(potential_imgs)):
                img_src_path = potential_imgs
                print(f"DEBUG: Found processed images in {img_src_path}, using as base for masks.")

        # Determine output path
        out_path = None
        if request.output_path:
            out_path = Path(request.output_path).absolute()
        else:
            # We want it in project_root/prefix_masked_images
            root = find_project_root(raw_path)
            out_path = root / f"{raw_path.name}_masked_images"
            
        os.makedirs(out_path, exist_ok=True)

        # Load class names
        class_names = {}
        yaml_path = discover_data_yaml(raw_path)
        if yaml_path and yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', [])
                if isinstance(names, list):
                    class_names = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}

        image_files = get_supported_image_files(img_src_path)
        app.state.task_progress["total"] = len(image_files)
        app.state.task_progress["message"] = f"Generating masks for {len(image_files)} images from {img_src_path.name}..."

        saved_count = 0
        for i, img_p in enumerate(image_files):
            lbl_p = labels_path / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                app.state.task_progress["current"] = i + 1
                continue
            
            img = cv2.imread(str(img_p))
            if img is None:
                app.state.task_progress["current"] = i + 1
                continue
                
            h, w = img.shape[:2]
            
            with open(lbl_p, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    color = (0, 255, 0) # Green default
                    # Simple color scheme
                    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0), (255,0,255)]
                    color = colors[cls_id % len(colors)]
                    
                    if len(coords) == 4:
                        # Bounding Box (center_x, center_y, width, height)
                        cx, cy, dw, dh = coords
                        x1 = int((cx - dw/2) * w)
                        y1 = int((cy - dh/2) * h)
                        x2 = int((cx + dw/2) * w)
                        y2 = int((cy + dh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        label = class_names.get(cls_id, f"class_{cls_id}")
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif len(coords) >= 6:
                        # Polygon (x1, y1, x2, y2, ...)
                        pts = []
                        for j in range(0, len(coords), 2):
                            pts.append([int(coords[j] * w), int(coords[j+1] * h)])
                        pts = np.array(pts, np.int32)
                        cv2.polylines(img, [pts], True, color, 2)
                        
                        label = class_names.get(cls_id, f"class_{cls_id}")
                        cv2.putText(img, label, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imwrite(str(out_path / (img_p.stem + ".jpg")), img)
            saved_count += 1
            
            app.state.task_progress["current"] = i + 1
            if (i+1) % 10 == 0:
                app.state.task_progress["message"] = f"Processed {i+1}/{len(image_files)} images... Saved: {saved_count}"

        if saved_count == 0:
             raise ValueError(f"No corresponding label files found in {labels_path}. Verified {len(image_files)} images.")

        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Mask generation complete! Saved {saved_count} images to {out_path}"
        app.state.task_progress["result"] = {
            "masked_dir": str(out_path),
            "labeled_dir": str(labels_path.parent.absolute() if labels_path.name == "labels" else labels_path.absolute())
        }
        
        # Update config if possible
        save_project_config(raw_path, {"masked_dir": str(out_path)})

    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Mask Gen Error: {str(e)}"

# This allows us to serve images for the crop UI or verification
@app.post("/api/mount")
def mount_path(name: str, path: str):
    p = Path(path).absolute()
    if p.exists():
        # Remove existing route with this name to allow "unmounting" / updating
        for i, route in enumerate(app.routes):
            if getattr(route, "name", None) == f"static_{name}":
                print(f"Replacing existing mount: static_{name}")
                app.routes.pop(i)
                break
                
        app.mount(f"/static/{name}", StaticFiles(directory=str(p)), name=f"static_{name}")
        return {"status": "mounted", "url": f"/static/{name}"}
    return {"status": "error", "message": f"Path does not exist: {path}"}


def discover_labels_dir(p: Path) -> Optional[Path]:
    """Robustly find the labels directory for a given dataset path."""
    root, identity = get_config_identity(p)
    
    candidates = [
        p / "labels", # Current path's labels subfolder
        root / f"{identity}_labeled" / "labels",
        root / "labeled" / "labels",
        root / "labels" # Root's labels
    ]
    
    # Check candidates
    for lp in candidates:
        if lp.exists() and lp.is_dir():
            return lp
            
    # Search for any directory ending with _labeled/labels
    if root.exists():
        for item in root.iterdir():
            if item.is_dir() and item.name.startswith(identity) and item.name.endswith("_labeled"):
                if (item / "labels").exists():
                    return item / "labels"
                    
    # Final fallback if nothing found but root/labels is a standard guess
    return None

@app.get("/api/labeled/images")
def list_labeled_images(path: str, limit: int = 20, offset: int = 0, classes: str = None):
    p = Path(path).absolute()
    if not p.exists() or not p.is_dir():
        return {"images": [], "total": 0, "offset": offset, "limit": limit}
    
    # Define root early so it's available for fallback and yaml lookups
    root, _ = get_config_identity(p)
    
    labels_dir = discover_labels_dir(p)
    if not labels_dir:
        labels_dir = root / "labels"
    
    print(f"Using labels directory: {labels_dir}")
    
    # Load class names from data.yaml if available
    class_names_map = {}
    yaml_paths = [
        labels_dir.parent / "data.yaml",
        labels_dir.parent.parent / "data.yaml",
        root / "data.yaml" # Check project root
    ]
    
    for yp in yaml_paths:
        if yp.exists():
            print(f"Searching for class names in: {yp}")
            try:
                with open(yp, 'r') as f:
                    data = yaml.safe_load(f)
                    names = data.get('names', [])
                    if isinstance(names, list):
                        class_names_map = {i: n for i, n in enumerate(names)}
                    elif isinstance(names, dict):
                        class_names_map = {int(k): v for k, v in names.items()}
                    if class_names_map:
                        print(f"Loaded {len(class_names_map)} class names")
                        break
            except Exception as e:
                print(f"Error reading {yp}: {e}")

    files = get_supported_image_files(p)
    
    # If no files found in root, check if we should look in 'images' or 'masked_images' subdir
    if not files:
        if (p / "masked_images").exists():
            print(f"No images in root, redirecting to {p / 'masked_images'}")
            p = p / "masked_images"
            files = get_supported_image_files(p)
        elif (p / "images").exists():
             print(f"No images in root, redirecting to {p / 'images'}")
             p = p / "images"
             files = get_supported_image_files(p)
    
    target_classes = []
    if classes:
        target_classes = [c.strip() for c in classes.split(',')]
        print(f"Filtering by classes: {target_classes}")

    all_filtered_results = []
    # If we have class filtering, we unfortunately have to scan ALL files to know the total count
    # and to paginate correctly. 
    # Optimization: if no classes, we can just paginate 'files' and process only those.
    
    if not target_classes:
        total = len(files)
        paged_files = files[offset:offset + limit]
    else:
        # Scan all to filter
        print(f"Scanning {len(files)} files for class filtering...")
        paged_files = files # We'll filter and then paginate manually
            
    results = []
    for f in paged_files:
        stats = {}
        # Try to read corresponding label file
        label_file = labels_dir / f"{f.stem}.txt"
        if label_file.exists():
            try:
                with open(label_file, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            cls_name = class_names_map.get(cls_id, f"class_{cls_id}")
                            stats[cls_name] = stats.get(cls_name, 0) + 1
            except:
                pass
        
        item = {"name": f.name, "stats": stats}
        
        if target_classes:
            # Check if this item matches
            has_match = any(cls in target_classes for cls in stats.keys())
            if has_match:
                results.append(item)
        else:
            results.append(item)

    # Now handle pagination based on whether filtering occurred
    if target_classes:
        total = len(results)
        results = results[offset:offset + limit]
    else:
        total = len(files)
        # results already contains the paged items, so no need to slice again with offset
        # results = results[offset:offset + limit] <--- This was the BUG (double slicing)
        pass

    return {
        "images": results,
        "total": total,
        "offset": offset,
        "limit": limit
    }

@app.post("/api/labeled/filter")
def filter_labeled_image(request: FilterRequest):
    source_p = Path(request.source_dir).absolute()
    print(f"Filter request: {request.image_name} from {source_p}")
    
    # Use unified identity logic
    root, identity = get_config_identity(source_p)
    
    # Determine the project root (where filtered folder should go)
    target_name = request.target_name
    if target_name == "filtered":
        target_name = f"{identity}_filtered"
    target_root = root / target_name
    
    target_images = target_root / "images"
    target_labels = target_root / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    
    img_name = request.image_name
    
    # Determine the source for COPYING (Original Image)
    # If labeled_dir is provided, use that as the source of truth for images and labels.
    # Otherwise, fallback to heuristic discovery.
    
    src_img = None
    src_label = None
    
    if request.labeled_dir:
        l_root = Path(request.labeled_dir)
        # Check standard YOLO structure first
        if (l_root / "images").exists():
             src_img = l_root / "images" / img_name
             if not src_img.exists():
                 # Fallback: maybe images are in root?
                 src_img = l_root / img_name
        else:
            src_img = l_root / img_name
            
        # Try to find specific label
        if (l_root / "labels").exists():
            src_label = l_root / "labels" / f"{Path(img_name).stem}.txt"
        else:
            # Fallback check same dir
            src_label = l_root / f"{Path(img_name).stem}.txt"
            
    else:
        # Fallback to old behavior: use discovery
        print("Warning: No labeled_dir provided. Attempting to discover...")
        # ... (simplified discovery from previous code if needed, or primarily rely on source_dir if it looks like an image dir)
        # If source_dir IS the masked dir, we want to find the original.
        # But 'root' / "labeled" / "images" is a good guess.
        
        potential_roots = [
            root / f"{identity}_labeled",
            root / "labeled",
            source_p.parent.parent if source_p.name == "masked_images" or source_p.name == "images" else source_p,
            root
        ]
        
        labeled_root = None
        for pr in potential_roots:
            if (pr / "images").exists() and (pr / "labels").exists():
                labeled_root = pr
                break
        
        if labeled_root:
            src_img = labeled_root / "images" / img_name
            src_label = labeled_root / "labels" / f"{Path(img_name).stem}.txt"
        else:
             # Last resort: just use the source file itself (e.g. if filtering from raw images)
             src_img = source_p / img_name

    # Validate existence
    if not src_img or not src_img.exists():
        # Fallback: maybe the user IS looking at the image they want to filter (e.g. raw dataset)
        fallback_img = source_p / img_name
        if fallback_img.exists():
            print(f"Original not found in labeled_dir, falling back to source: {fallback_img}")
            src_img = fallback_img
            # Look for label in source too
            fallback_lbl = source_p.parent / "labels" / f"{Path(img_name).stem}.txt"
            if fallback_lbl.exists(): src_label = fallback_lbl
        else:
            print(f"Error: Source image not found. Looked for {img_name}")
            raise HTTPException(status_code=404, detail=f"Source image {img_name} not found")

    print(f"Copying {src_img} to {target_images / img_name}")
    shutil.copy2(src_img, target_images / img_name)
    
    if src_label and src_label.exists():
        print(f"Copying {src_label} to {target_labels / src_label.name}")
        shutil.copy2(src_label, target_labels / src_label.name)
        
    return {"status": "ok", "message": f"Copied to {target_root}"}

@app.get("/api/labeled/stats")
def get_dataset_stats(path: str):
    p = Path(path).absolute()
    print(f"Calculating stats for path: {p}")
    
    if not p.exists():
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Resolve labels_dir
    labels_dir = None
    if p.name == "labels":
        labels_dir = p
    elif (p / "labels").exists():
        labels_dir = p / "labels"
    elif (p / "txt").exists(): # Sometimes people name it txt
        labels_dir = p / "txt"
    else:
        # Maybe p itself contains .txt files?
        if list(p.glob("*.txt")):
            labels_dir = p
            
    if not labels_dir:
        print(f"Labels directory not found in {p}")
        raise HTTPException(status_code=400, detail="Labels directory not found")
        
    print(f"Resolved labels_dir: {labels_dir}")

    # Load class names
    class_names_map = {}
    # Look for data.yaml in current dir, parent, or grandparent
    potential_yamls = [
        labels_dir / "data.yaml",
        labels_dir / "classes.txt", # Check for classes.txt too
        labels_dir.parent / "data.yaml",
        labels_dir.parent / "classes.txt",
        labels_dir.parent.parent / "data.yaml"
    ]
    
    for yp in potential_yamls:
        if yp.exists():
            print(f"Found class config at: {yp}")
            try:
                if yp.suffix == '.yaml':
                    with open(yp, 'r') as f:
                        data = yaml.safe_load(f)
                        names = data.get('names', [])
                        if isinstance(names, list):
                            class_names_map = {i: n for i, n in enumerate(names)}
                        elif isinstance(names, dict):
                            class_names_map = {int(k): v for k, v in names.items()}
                else: # classes.txt
                    with open(yp, 'r') as f:
                        names = [l.strip() for l in f.readlines() if l.strip()]
                        class_names_map = {i: n for i, n in enumerate(names)}
                
                if class_names_map: 
                    print(f"Loaded {len(class_names_map)} classes")
                    break
            except Exception as e:
                print(f"Error loading {yp}: {e}")

    # Total images: try to find 'images' sibling, or 'masked_images', or count in p if mixed
    # Try to strip task suffixes to find the base name for sibling discovery
    # Use unified identity to find siblings and potential labels
    root, identity = get_config_identity(p)
    
    # Find siblings that might contain processed or labeled data
    processed_dir = None
    labeled_dirs = []
    has_processed = False
    
    # Generic candidates
    if (root / "processed").exists():
        processed_dir = root / "processed"
        has_processed = True
    if (root / "labeled").exists():
        labeled_dirs.append(root / "labeled")

    # Prefix-based candidates (the preferred approach)
    if root.exists():
        for item in root.iterdir():
            if not item.is_dir(): continue
            if item.name.startswith(identity):
                if item.name.endswith("_processed"):
                    processed_dir = item
                    has_processed = True
                elif item.name.endswith("_labeled"):
                    labeled_dirs.append(item)
                elif item.name.endswith("_masked_images"):
                     labeled_dirs.append(item)

    total_images = 0
    potential_images = [
        labels_dir.parent / "images", # legacy root
        labels_dir.parent / "masked_images",
        root / f"{identity}_labeled" / "images", # prefixed root
        root / f"{identity}_masked_images",
        root / "images",
        root / "masked_images"
    ]
    
    # Try to search for ANY subdirectory starting with identity and ending with _labeled/images
    if root.exists():
        for item in root.iterdir():
            if item.is_dir() and item.name.startswith(identity) and item.name.endswith("_labeled"):
                potential_images.append(item / "images")
            if item.is_dir() and item.name.startswith(identity) and item.name.endswith("_masked_images"):
                potential_images.append(item)

    images_dir = None
    for pid in potential_images:
        if pid.exists() and pid.is_dir():
            # Verify it has images
            if get_supported_image_files(pid):
                images_dir = pid
                break
            
    if images_dir:
         total_images = len(get_supported_image_files(images_dir))
    else:
         # Maybe images are in same folder?
         total_images = len(get_supported_image_files(labels_dir))
         
    print(f"Counting objects in {labels_dir}")
    empty_images = []
    class_counts = {}
    total_objects = 0
    
    for label_file in labels_dir.glob("*.txt"):
        if label_file.name == "classes.txt": continue # Skip classes file
        
        has_obj = False
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        has_obj = True
                        cls_id = int(parts[0])
                        cls_name = class_names_map.get(cls_id, f"class_{cls_id}")
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        total_objects += 1
        except:
            pass
            
        if not has_obj:
            empty_images.append(label_file.stem)

    # Calculate percentages
    stats_data = []
    for cls_name, count in class_counts.items():
        stats_data.append({
            "name": cls_name,
            "count": count,
            "percentage": round((count / total_objects * 100), 2) if total_objects > 0 else 0
        })

    # Sort by count descending
    stats_data.sort(key=lambda x: x['count'], reverse=True)
    
    result = {
        "total_images": total_images,
        "total_objects": total_objects,
        "class_stats": stats_data,
        "empty_images": empty_images,
        "empty_count": len(empty_images)
    }
    print(f"Stats result: {result}")
    return result

@app.get("/api/fs/list")
def list_fs(path: str = "/", only_dirs: bool = False):
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            # Try to handle partial paths or defaults
            if path == "/":
                 p = Path(os.getcwd()).resolve()
            else:
                raise HTTPException(status_code=404, detail="Path not found")

        items = []
        # Sort so directories come first
        for entry in sorted(os.scandir(p), key=lambda e: (not e.is_dir(), e.name.lower())):
            if only_dirs and not entry.is_dir():
                continue
            
            # For model selection, we might want to see .pt files
            # For now, let's just show all or just dirs
            items.append({
                "name": entry.name,
                "path": str(Path(entry.path).absolute()),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if not entry.is_dir() else 0
            })

        return {
            "current_path": str(p.absolute()),
            "parent_path": str(p.parent.absolute()) if p.parent != p else None,
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_split_task(request: SplitRequest):
    try:
        app.state.task_progress["status"] = "splitting"
        app.state.task_progress["message"] = "Initializing split..."
        app.state.task_progress["current"] = 0
        app.state.task_progress["total"] = 0
        
        def progress_callback(current, total):
            app.state.task_progress["current"] = current
            app.state.task_progress["total"] = total
            
        processed_count = split_folder(request.input_path, request.num_splits, callback=progress_callback)
        
        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Split complete! Processed {processed_count} images."
        app.state.task_progress["result"] = {
            "processed_count": processed_count,
            "input_path": request.input_path
        }
        
    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Split Error: {str(e)}"
        print(f"Split Error: {e}")

@app.post("/api/split-dataset")
def split_dataset(request: SplitRequest, background_tasks: BackgroundTasks):
    if not request.input_path or not Path(request.input_path).exists():
        raise HTTPException(status_code=400, detail="Invalid input path")
    
    app.state.task_progress["status"] = "splitting"
    app.state.task_progress["message"] = "Initializing split request..."
    app.state.task_progress["current"] = 0
    app.state.task_progress["total"] = 0
    
    background_tasks.add_task(run_split_task, request)
    return {"status": "started"}

class AnnotationItem(BaseModel):
    class_id: int
    points: List[float] # YOLO normalized coordinates
    type: str # 'box' or 'polygon'

class SaveAnnotationRequest(BaseModel):
    dataset_path: str
    image_name: str
    annotations: List[AnnotationItem]

def find_image_file(dataset_root: Path, image_name: str) -> Optional[Path]:
    """Robustly find an image file within a dataset structure."""
    # 1. Check direct subfolders
    candidates = [
        dataset_root / image_name,
        dataset_root / "images" / image_name,
        dataset_root / "masked_images" / image_name,
        dataset_root / "train" / "images" / image_name,
        dataset_root / "val" / "images" / image_name,
        dataset_root / "test" / "images" / image_name
    ]
    for c in candidates:
        if c.exists():
            return c
            
    # 2. Search recursively if not in obvious places (max depth 3)
    # Only do this if image_name doesn't already look like a subpath
    if "/" not in image_name and "\\" not in image_name:
        for ext in ["", ".jpg", ".png", ".jpeg", ".WEBP", ".JPG"]:
            name_to_find = image_name if not ext else f"{Path(image_name).stem}{ext}"
            # Check if we can find it by walking (limit walk)
            for root, dirs, files in os.walk(str(dataset_root)):
                if name_to_find in files:
                    return Path(root) / name_to_find
                if root.count(os.sep) - str(dataset_root).count(os.sep) > 3:
                     dirs[:] = [] # stop recursion
    
    return None

def get_label_path(dataset_root: Path, image_name: str) -> Path:
    """Consistently find or determine the label path for an image."""
    img_path = find_image_file(dataset_root, image_name)
    
    if img_path:
        # Standard YOLO structure replacement: /images/ -> /labels/
        parent = img_path.parent
        if parent.name == "images":
            return parent.parent / "labels" / f"{img_path.stem}.txt"
        elif parent.name == "masked_images":
            # If viewing masked images, try to find original labels folder
            found = discover_labels_dir(dataset_root)
            if found: return found / f"{img_path.stem}.txt"
            return parent.parent / "labels" / f"{img_path.stem}.txt"
            
        # Check sibling labels folder
        if (parent / "labels").exists():
            return parent / "labels" / f"{img_path.stem}.txt"
        elif (parent.parent / "labels").exists():
             return parent.parent / "labels" / f"{img_path.stem}.txt"
             
        # Flat structure
        return parent / f"{img_path.stem}.txt"

    # Fallback to older logic if image not found (guess based on dataset_root)
    labels_dir = discover_labels_dir(dataset_root)
    if labels_dir:
        return labels_dir / f"{Path(image_name).stem}.txt"
        
    return dataset_root / f"{Path(image_name).stem}.txt"

@app.get("/api/annotation/data")
def get_annotation_data(path: str, image_name: str):
    """
    Get existing annotations for a specific image.
    Path is the DATASET path. We need to find the specific image and its label file.
    """
    p = Path(path).absolute()
    label_path = get_label_path(p, image_name)
    
    annotations = []
    if label_path.exists():
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    # Heuristic detection
                    atype = 'polygon'
                    if len(coords) > 4: # Changed from == 4 to > 4 for polygon detection
                        atype = 'polygon'
                    else: # Assuming 4 coords means box
                        atype = 'box'
                        
                    annotations.append({
                        "class_id": cls_id,
                        "points": coords,
                        "type": atype
                    })
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
            
    return {"annotations": annotations}

@app.get("/api/annotation/image_file")
def serve_annotation_image_file(path: str, image_name: str):
    p = Path(path).absolute()
    img_path = find_image_file(p, image_name)
    
    if not img_path:
        print(f"DEBUG: Image NOT FOUND: {image_name} in {path}")
        raise HTTPException(status_code=404, detail=f"Image {image_name} not found in {path}")
        
    print(f"DEBUG: Serving annotation image: {img_path}")
    from fastapi.responses import FileResponse
    return FileResponse(str(img_path))

@app.post("/api/annotation/save")
def save_annotation_data(request: SaveAnnotationRequest):
    p = Path(request.dataset_path).absolute()
    label_path = get_label_path(p, request.image_name)
    
    lines = []
    for ann in request.annotations:
        # Validation
        if not ann.points: continue
        
        # Format string
        coords_str = " ".join([f"{x:.6f}" for x in ann.points])
        lines.append(f"{ann.class_id} {coords_str}\n")
        
    try:
        # Ensure parent exists
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write labels: {e}")
        
    return {"status": "ok", "message": "Saved"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
