from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from typing import List, Optional, Dict, Tuple, Any, Union
import sys
import threading
import shutil
import json
import asyncio
from datetime import datetime
from project_manager import ProjectManager
from training_manager import training_manager
from export_manager import export_manager
from dataset_tools_manager import DatasetToolsManager
from data_augmentation_manager import DataAugmentationManager
from terminal_manager import terminal_manager
from simple_augmentation_manager import SimpleAugmentationManager
from fastapi.responses import StreamingResponse
from workflow_manager import WorkflowManager, Job
import io
import torch

# Add the scripts directory to path to import existing logic
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'scripts')))

try:
    from auto_labeler import YOLOInference
    from crop_resize import process_image, get_supported_image_files
    from split_folder import split_folder
    from sample_yolo_dataset import DatasetSampler
    from ultralytics import SAM
except ImportError as e:
    print(f"Error importing scripts/ultralytics: {e}")

app = FastAPI()
print("DEBUG: main.py STARTING - VERIFICATION ID 999")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AutoLabelSingleRequest(BaseModel):
    dataset_path: str
    image_name: str
    model_path: str
    confidence: float = 0.5
    save_masked: bool = True

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
    images_list: Optional[List[str]] = None

class SplitRequest(BaseModel):
    input_path: str
    num_splits: int

class CreateProjectRequest(BaseModel):
    name: str
    parent_dir: str
    raw_images_dir: str
    classes: List[str]

class CreateProjectFromSplitRequest(BaseModel):
    name: str
    parent_dir: str
    split_dataset_path: str

class LoadProjectRequest(BaseModel):
    path: str

class CreateDatasetRequest(BaseModel):
    project_path: str
    name: str # Dataset name
    strategy: str = "all"
    split_ratios: Optional[List[float]] = [0.7, 0.2, 0.1]

class SampleRequest(BaseModel):
    dataset_path: str
    output_path: Optional[str] = None
    count: Optional[int] = None
    percentage: Optional[float] = None
    class_percentages: Optional[str] = None # JSON string or "id:pct,id:pct"
    seed: int = 42

class UpdateConfigRequest(BaseModel):
    path: str
    updates: Dict[str, Any]

class SplitStatsRequest(BaseModel):
    dataset_path: str

class RebalanceRequest(BaseModel):
    dataset_path: str
    train_pct: float
    val_pct: float
    test_pct: float
    delete_original: bool = False

class FlattenRequest(BaseModel):
    dataset_path: str


class FlattenRequest(BaseModel):
    dataset_path: str

class AugmentationPreviewRequest(BaseModel):
    dataset_path: str
    background_path: str # Path to temp file or existing file
    class_ids: List[int]
    rotation_range: Optional[List[float]] = None
    blur_range: Optional[List[int]] = None
    scaling_range: Optional[List[float]] = None
    scaling_range: Optional[List[float]] = None
    contrast_range: Optional[List[float]] = None
    brightness_range: Optional[List[int]] = None
    region_scale: float = 0.8
    roi: Optional[List[float]] = None # [x, y, w, h] normalized
    
class AugmentationGenerateRequest(AugmentationPreviewRequest):
    output_path: Optional[str] = None
    num_augmentations: int = 3
    augment_together: bool = False
    composition_mode: bool = False
    total_images: int = 10
    composition_mode: bool = False
    total_images: int = 10
    objects_per_image: int = 3
    objects_per_image: int = 3
    custom_output_name: Optional[str] = None

class SimpleAugmentationPreviewRequest(BaseModel):
    dataset_path: str
    config: Dict

class SimpleAugmentationScanRequest(BaseModel):
    dataset_path: str

class SimpleAugmentationRunRequest(BaseModel):
    dataset_path: str
    output_name: str
    multiplier: int
    config: Dict
    selected_splits: Optional[List[str]] = None

def find_project_root(path: Path) -> Path:
    """Heuristic to find the 'main' project directory by traversing up from known subfolders."""
    p = path.absolute()
    
    # If parent already has a config (prefixed or generic), that's a very strong indicator
    parent_dir, identity_name = get_config_identity(p)
    generic_conf = parent_dir / "yolo_project_config.json"
    project_conf = parent_dir / "project_config.json" # NEW: Strict Project Config
    prefixed_conf = parent_dir / f"{identity_name}_yolo_project_config.json"
    
    if generic_conf.exists() or prefixed_conf.exists() or project_conf.exists():
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
    
    # 0. Try standard project config (Primary)
    # Check in resolved root AND in the dataset_path itself (in case get_config_identity went up too far)
    # AND check in dataset_path / identity (nested structure case)
    potential_paths = [
        root / "project_config.json",
        dataset_path / "project_config.json",
        dataset_path / identity / "project_config.json"
    ]
    
    standard_conf = None
    standard_conf_root = None
    for p in potential_paths:
        if p.exists():
            standard_conf = p
            standard_conf_root = p.parent
            break
            
    if standard_conf:
        try:
            with open(standard_conf, 'r') as f:
                merged_data.update(json.load(f))
                if standard_conf_root:
                    merged_data["_project_root"] = str(standard_conf_root.absolute())
        except Exception as e:
            print(f"Error loading project_config.json: {e}")

    # 1. Try legacy generic name (Secondary)
    legacy_path = root / "yolo_project_config.json"
    if legacy_path.exists():
        try:
            with open(legacy_path, 'r') as f:
                # Merge, but prioritize standard if keys conflict? 
                # Actually, standard should likely override legacy.
                # So we load legacy first? Or update with legacy?
                # Let's assume standard is source of truth.
                # If we want standard to win, we should have loaded legacy first.
                # But typically legacy is old. 
                # Let's load legacy data into a temp dict and update with existing merged_data
                legacy_data = json.load(f)
                # We want merged_data (standard) to overwrite legacy_data
                legacy_data.update(merged_data)
                merged_data = legacy_data
        except: pass

    # 2. Find all files matching the identity prefix
    # identity_yolo_project_config.json, identity_processed_yolo_project_config.json, etc.
    config_files = list(root.glob(f"{identity}*_yolo_project_config.json"))
    # Sort by modification time to preserve most recent updates during merge
    config_files.sort(key=lambda x: x.stat().st_mtime)
    
    if not config_files and not legacy_path.exists() and not standard_conf:
        return {}

    for cp in config_files:
        try:
            with open(cp, 'r') as f:
                merged_data.update(json.load(f))
        except: pass
        
    print(f"DEBUG: Loaded unified config for {identity} from {len(config_files) + (1 if standard_conf else 0)} files")
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

def load_classes_from_path(path: Path) -> List[str]:
    """
    Robustly load classes from project structure (classes.txt, project_config) or legacy data.yaml.
    """
    p = path.absolute()
    
    # 1. Check for classes.txt (User requested priority/feature)
    # Check current dir AND parent dir
    for candidate in [p / "classes.txt", p.parent / "classes.txt"]:
        if candidate.exists():
            try:
                with open(candidate, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]
                if classes:
                    return classes
            except Exception as e:
                print(f"Error reading classes.txt at {candidate}: {e}")

    # 2. Check for Project Config
    # Check current dir AND parent dir
    for candidate in [p / "project_config.json", p.parent / "project_config.json"]:
        if candidate.exists():
            try:
                with open(candidate, 'r') as f:
                    pc = json.load(f)
                    if "classes" in pc and pc["classes"]:
                        return pc["classes"]
            except Exception as e:
                print(f"Error reading project config at {candidate}: {e}")

    # 3. Fallback to data.yaml discovery
    yaml_path = discover_data_yaml(p)
    if yaml_path:
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', [])
                if isinstance(names, dict):
                    sorted_keys = sorted([int(k) for k in names.keys()])
                    return [names.get(str(k) if str(k) in names else k) for k in sorted_keys]
                elif isinstance(names, list):
                    return names
        except Exception as e:
             print(f"Error reading yaml {yaml_path}: {e}")
             
    return []


@app.get("/api/dataset/classes")
def get_dataset_classes(path: str):
    p = Path(path).absolute()
    names = load_classes_from_path(p)
    if not names:
        # Try one more fallback: maybe it's just 'classes' list in a yaml nearby without strict structure
        # (Handled by discover_data_yaml inside load_classes_from_path already)
        raise HTTPException(status_code=400, detail=f"Classes not found. No classes.txt, project_config.json or data.yaml near {path}")
    
    return {"classes": names}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/progress")
def get_progress():
    return app.state.task_progress

@app.post("/api/project/create")
def create_project(request: CreateProjectRequest):
    try:
        return ProjectManager.create_project(
            request.name, 
            request.parent_dir, 
            request.raw_images_dir, 
            request.classes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/create_from_split")
def create_project_from_split(request: CreateProjectFromSplitRequest):
    try:
        return ProjectManager.create_project_from_split(
            request.name,
            request.parent_dir,
            request.split_dataset_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/load")
def load_project(request: LoadProjectRequest):
    try:
        return ProjectManager.load_project(request.path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/dataset/create")
def create_dataset(request: CreateDatasetRequest):
    try:
        return ProjectManager.create_dataset(
            request.project_path,
            request.name,
            request.strategy,
            request.split_ratios
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/list")
def list_datasets(project_path: str):
    """List all created datasets in the project."""
    try:
        return ProjectManager.list_datasets(project_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/config")
def update_project_config(request: UpdateConfigRequest):
    try:
        return ProjectManager.update_project_config(request.path, request.updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # 0. Load Project Config First
    config = load_project_config(p)
    
    # Check for processed folder using config or default
    processed_dir = None
    if config and config.get("dirs", {}).get("processed"):
        # Resolve relative to project root
        if config.get("_project_root"):
             root = Path(config["_project_root"])
        else:
             # Fallback
             root, _ = get_config_identity(p)
             
        potential_proc = root / config["dirs"]["processed"]
        if potential_proc.exists():
            processed_dir = potential_proc
    
    if not processed_dir:
        # Legacy check
        processed_dir = p.parent / f"{dataset_name}_processed"
        if not processed_dir.exists():
            processed_dir = None
            
    has_processed = processed_dir is not None
    
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
        processed_dir / f"{dataset_name}_masked_images" if processed_dir else None,
        p / f"{dataset_name}_masked_images",
        p.parent / f"{dataset_name}_labeled",
        processed_dir / f"{dataset_name}_labeled" if processed_dir else None,
        p / f"{dataset_name}_labeled",
        # Legacy non-prefixed versions
        p.parent / "masked_images",
        processed_dir / "masked_images" if processed_dir else None,
        p / "masked_images",
        p.parent / "labeled",
        processed_dir / "labeled" if processed_dir else None,
        p / "labeled"
    ]
    # Filter out None values
    labeled_dirs = [x for x in labeled_dirs if x is not None]
    
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
            if len(files) > 0:
                found_labeled = str(ld.absolute())
                has_masks = True
                break
            
            # Check for files in known subdirs
            for sub in ["masked_images", "images", "masked"]:
                sub_dir = ld / sub
                if sub_dir.exists() and sub_dir.is_dir():
                    sub_files = get_supported_image_files(sub_dir)
                    if len(sub_files) > 0:
                        found_labeled = str(sub_dir.absolute())
                        has_masks = True
                        break
            if found_labeled:
                break
                
    # Also check specifically for labels directory to suggest mask generation
    labels_root = None
    
    # A. Check Config for Labels
    if config and config.get("dirs", {}).get("annotations"):
        # Assuming relative to project root
        if config.get("_project_root"):
             root = Path(config["_project_root"])
        else:
             root, _ = get_config_identity(p)
             
        p_ann = root / config["dirs"]["annotations"]
        if p_ann.exists():
            labels_root = str(p_ann.absolute())

    # B. Should we detect Split Dataset (train/val/test)?
    # Boolean logic: if we don't find "flat" labels, look for splits.
    has_splits = False
    possible_splits = ["train", "valid", "test", "val"]
    for s in possible_splits:
        if (p / s / "images").exists() and (p / s / "labels").exists():
            has_splits = True
            break
            
    # B. If not in config, try robustness using discover_labels_dir helper
    if not labels_root and not has_splits:
        discovered = discover_labels_dir(p)
        if discovered:
            labels_root = str(discovered.absolute())
            
    
    # C. Fallback to current/parent if not found in candidates or config
    if not labels_root and not has_splits:
        # Determine identifying prefix by stripping ALL technical suffixes
        p_path = Path(p).absolute()
        _, prefix = get_config_identity(p_path)
        
        # Climb up and search
        curr = p_path
        for _ in range(3):
            candidates_lbl = [
                curr / "labels",
                curr / "annotations", # Standard project structure
                curr / "labeling" / "labels", 
                curr / "labeled" / "labels",
                curr / f"{prefix}_labeled" / "labels",
                curr.parent / f"{prefix}_labeled" / "labels"
            ]
            for cand in candidates_lbl:
                if cand.exists() and cand.is_dir():
                    # Verify it has txt files or subdirs
                    if any(cand.glob("*.txt")) or (cand / "labels").exists():
                        labels_root = str(cand.absolute())
                        break
            if labels_root or curr.parent == curr:
                break
            curr = curr.parent
            
    # CRITICAL: If we have labels but no masked images (found_labeled is None), 
    # we MUST set found_labeled to something (e.g. the labels root itself, or the dataset path)
    # so the frontend thinks we have a "labeled directory" and allows mask generation.
    # The frontend checks `if (!labelResult?.labeled_dir)` to block generation.
    # We return `labeled_dir` key as `found_labeled`.
    
    if (labels_root or has_splits) and not found_labeled:
         # Use labels_root as the handle for "labeled directory" if no visual masks exist yet
         found_labeled = labels_root if labels_root else str(p.absolute())
    
    # Override from config if masked dir is explicitly set
    if config and config.get("dirs", {}).get("masked"):
         if config.get("_project_root"):
             root = Path(config["_project_root"])
         else:
             root, _ = get_config_identity(p)
             
         p_masked = root / config["dirs"]["masked"]
         if p_masked.exists() and any(get_supported_image_files(p_masked)):
             found_labeled = str(p_masked.absolute())
             has_masks = True

    return {
        "processed_dir": str(processed_dir.absolute()) if has_processed else None,
        "labeled_dir": found_labeled,
        "labels_root": labels_root,
        "has_labels": labels_root is not None or has_splits,
        "has_masks": has_masks,
        "config": config,
        "is_split": has_splits
    }

@app.post("/api/dataset/split-stats")
def get_split_stats(request: SplitStatsRequest):
    try:
        return DatasetToolsManager.get_split_stats(request.dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/rebalance")
def rebalance_dataset(request: RebalanceRequest):
    try:
        return DatasetToolsManager.rebalance_dataset(
            request.dataset_path, 
            request.train_pct, 
            request.val_pct, 
            request.test_pct, 
            request.delete_original
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/flatten")
def flatten_dataset(request: FlattenRequest):
    try:
        return DatasetToolsManager.flatten_dataset(request.dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augmentation/upload_background")
async def upload_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = DataAugmentationManager.save_temp_background(contents, file.filename)
        return {"path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augmentation/preview")
def preview_augmentation(request: AugmentationPreviewRequest):
    try:
        return DataAugmentationManager.generate_preview(
            request.dataset_path,
            request.background_path,
            request.class_ids,
            request.rotation_range,
            request.blur_range,
            request.scaling_range,
            request.contrast_range,
            request.brightness_range,
            request.region_scale,
            request.roi
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment/simple/preview")
def simple_augment_preview(request: SimpleAugmentationPreviewRequest):
    try:
        return SimpleAugmentationManager.generate_preview(request.dataset_path, request.config)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment/simple/run")
def simple_augment_run(request: SimpleAugmentationRunRequest, background_tasks: BackgroundTasks):
    if app.state.task_progress["status"] == "running":
         raise HTTPException(status_code=400, detail="A background task is already running")
         
    try:
        app.state.task_progress = {
            "status": "idle",
            "current": 0,
            "total": 0,
            "message": "Starting...",
            "result": None
        }
        background_tasks.add_task(
            SimpleAugmentationManager.run_augmentation_job,
            request.dataset_path,
            request.output_name,
            request.multiplier,
            request.config,
            app.state.task_progress
        )
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AugmentationSampleRequest(BaseModel):
    dataset_path: str
    class_ids: List[int]
    background_path: str

class AugmentationApplyPreviewRequest(BaseModel):
    background_path: str
    rotation_range: Optional[List[float]] = None
    blur_range: Optional[List[int]] = None # Kernel size
    scaling_range: Optional[List[float]] = None
    contrast_range: Optional[List[float]] = None
    brightness_range: Optional[List[int]] = None
    region_scale: float = 0.8
    roi: Optional[List[float]] = None

@app.get("/api/augmentation/stats")
def get_augmentation_stats(path: str):
    try:
        return DataAugmentationManager.get_dataset_stats(path)
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {"total_objects": 0, "class_counts": {}, "class_percentages": {}}

@app.websocket("/api/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    print("[TERMINAL_DEBUG] WS: Connection request received", flush=True)
    try:
        await websocket.accept()
        print("[TERMINAL_DEBUG] WS: Connection accepted", flush=True)
    except Exception as e:
        print(f"[TERMINAL_DEBUG] WS: Error accepting connection: {e}", flush=True)
        return
    
    try:
        session_id = str(id(websocket))
        print(f"[TERMINAL_DEBUG] WS: Starting session {session_id}", flush=True)
        master_fd, pid = terminal_manager.start_session(session_id)
        print(f"[TERMINAL_DEBUG] WS: Session started, master_fd={master_fd}, pid={pid}", flush=True)
    except Exception as e:
        print(f"[TERMINAL_DEBUG] WS: Error starting session: {e}", flush=True)
        import traceback
        traceback.print_exc()
        await websocket.close()
        return

    # Create a loop to read from pty and send to websocket
    async def read_from_pty():
        print(f"[TERMINAL_DEBUG] WS: Reader task started for {session_id}", flush=True)
        try:
            while True:
                data = await asyncio.to_thread(terminal_manager.read_output, session_id)
                if data:
                    try:
                        await websocket.send_text(data.decode(errors='replace'))
                    except Exception as e:
                        print(f"[TERMINAL_DEBUG] WS: Error sending text: {e}", flush=True)
                        break
                else:
                    await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[TERMINAL_DEBUG] WS: Error reading from pty: {e}", flush=True)
        print(f"[TERMINAL_DEBUG] WS: Reader task ended for {session_id}", flush=True)

    read_task = asyncio.create_task(read_from_pty())
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'input':
                terminal_manager.write_input(session_id, message['data'])
            elif message['type'] == 'resize':
                terminal_manager.resize_terminal(session_id, message['cols'], message['rows'])
                
    except WebSocketDisconnect:
        print("[TERMINAL_DEBUG] WS: Terminal WebSocket disconnected", flush=True)
    except Exception as e:
        print(f"[TERMINAL_DEBUG] WS: Terminal WebSocket error: {e}", flush=True)
    finally:
        print(f"[TERMINAL_DEBUG] WS: Cleaning up session {session_id}", flush=True)
        read_task.cancel()
        terminal_manager.close_session(session_id)



@app.post("/api/augmentation/sample")
def sample_augmentation_object(request: AugmentationSampleRequest):
    try:
        return DataAugmentationManager.sample_object(
            request.dataset_path,
            request.class_ids,
            request.background_path
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augmentation/apply_preview")
def apply_augmentation_preview(request: AugmentationApplyPreviewRequest):
    try:
        return DataAugmentationManager.apply_preview_to_sample(
            request.background_path,
            request.rotation_range,
            request.blur_range,
            request.scaling_range,
            request.contrast_range,
            request.brightness_range,
            request.region_scale,
            request.roi
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augmentation/generate")
def generate_augmentation(req: AugmentationGenerateRequest, background_tasks: BackgroundTasks):
    print(f"DEBUG: /api/augmentation/generate received: {req}")
    # Validate paths
    try:
        # Check if busy
        if app.state.task_progress["status"] == "running":
            raise HTTPException(status_code=400, detail="A task is already running")
            
        app.state.task_progress = {
             "status": "starting",
             "current": 0,
             "total": 0,
             "message": "Initializing augmentation...",
             "result": None
        }
        
        # Determine output path if not set
        # Determine output path structure
        if req.custom_output_name:
             # Subdirectory with custom name
             p = Path(req.dataset_path)
             req.output_path = str(p / req.custom_output_name)
        elif not req.output_path:
             # Default: subfolder 'augmented'
             p = Path(req.dataset_path)
             req.output_path = str(p / "augmented")

        background_tasks.add_task(
            DataAugmentationManager.run_augmentation_task,
            dataset_path=req.dataset_path,
            background_path=req.background_path,
            output_path=req.output_path, 
            class_ids=req.class_ids,
            num_augmentations=req.num_augmentations,
            rotation_range=req.rotation_range,
            blur_range=req.blur_range,
            scaling_range=req.scaling_range,
            contrast_range=req.contrast_range,
            brightness_range=req.brightness_range,
            region_scale=req.region_scale,
            augment_together=req.augment_together,
            progress_tracker=app.state.task_progress,
            roi=req.roi,
            composition_mode=req.composition_mode,
            total_images=req.total_images,
            objects_per_image=req.objects_per_image
        )
        
        return {"status": "started", "output_path": req.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/sample")
def get_sample_image(path: str):
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Invalid path")
    
    image_files = get_supported_image_files(p)
    
    if not image_files:
        # Fallback: check standard subdirectories
        potential_dirs = [
            p / "train" / "images",
            p / "val" / "images",
            p / "valid" / "images",
            p / "test" / "images",
            p / "images",
            p / "processed_images",
            p / "masked_images"
        ]
        
        for d in potential_dirs:
            if d.exists() and d.is_dir():
                sub_imgs = get_supported_image_files(d)
                if sub_imgs:
                    image_files = sub_imgs
                    break
        
        # If still nothing, do a shallow recursive search
        if not image_files:
            for root, dirs, files in os.walk(str(p)):
                # Limit depth
                rel_root = Path(root).relative_to(p)
                if len(rel_root.parts) > 3:
                    dirs[:] = []
                    continue
                sub_imgs = get_supported_image_files(Path(root))
                if sub_imgs:
                    image_files = sub_imgs
                    break
    
    if not image_files:
        raise HTTPException(status_code=404, detail="No images found")
    
    # Get original dimensions and encode to base64
    img_path = str(image_files[0])
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Encode
    _, buffer = cv2.imencode('.jpg', img)
    import base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "sample_url": f"data:image/jpeg;base64,{img_base64}", 
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
    # Check for new project structure (project_config.json in parent)
    project_config_path = input_dir.parent / "project_config.json"
    if project_config_path.exists():
        try:
            with open(project_config_path, 'r') as f:
                p_config = json.load(f)
            
            p_config["crop"] = request.crop.dict() if request.crop else None
            p_config["resize_width"] = request.resize_width
            p_config["resize_height"] = request.resize_height
            # Update processed dir if not set or different
            if "processed" not in p_config.get("dirs", {}):
                 if "dirs" not in p_config: p_config["dirs"] = {}
                 p_config["dirs"]["processed"] = output_dir.name

            with open(project_config_path, 'w') as f:
                json.dump(p_config, f, indent=4)
            print(f"Updated project config at {project_config_path}")
            return # Exit early, do not use legacy config save
        except Exception as e:
            print(f"Error updating project config: {e}")

    # Fallback to legacy config if not a Project
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
        app.state.task_progress["message"] = "Initializing..."
        
        img_dir = Path(request.dataset_path)
        # Prepare class names map
        names_list = load_classes_from_path(img_dir)
        class_names_map = {i: n for i, n in enumerate(names_list)} if names_list else None

        # Determine output directories based on Project existence
        project_config_path = img_dir.parent / "project_config.json"
        
        output_labels_dir = None
        output_masked_dir = None
        save_labeled_images = True
        labeled_dir_path = None
        masked_images_dir = None
        
        if project_config_path.exists():
            print(f"Project config found at {project_config_path}. Using Project structure.")
            with open(project_config_path, 'r') as f:
                p_config = json.load(f)
            
            # Use Project definitions
            labeled_dir_path = img_dir.parent / p_config.get("dirs", {}).get("annotations", "annotations")
            masked_images_dir = img_dir.parent / p_config.get("dirs", {}).get("masked", f"{p_config.get('name')}_masked_images")
            
            output_labels_dir = labeled_dir_path
            output_masked_dir = masked_images_dir
            save_labeled_images = False
            
            # Ensure directories exist
            labeled_dir_path.mkdir(parents=True, exist_ok=True)
            masked_images_dir.mkdir(parents=True, exist_ok=True)
        
        app.state.task_progress["message"] = "Loading YOLO model..."

        inferencer = YOLOInference(
            model_path=request.model_path,
            image_dir=str(img_dir),
            confidence=request.confidence,
            save_masked_images=request.save_masked,
            single_folder=True,
            class_names_map=class_names_map,
            output_labels_dir=output_labels_dir,
            output_masked_dir=output_masked_dir,
            save_labeled_images=save_labeled_images
        )
        
        image_paths = list(inferencer.image_dir_.glob('*.png')) + list(inferencer.image_dir_.glob('*.jpg')) + list(inferencer.image_dir_.glob('*.tif'))
        app.state.task_progress["total"] = len(image_paths)
        app.state.task_progress["current"] = 0
        app.state.task_progress["message"] = "Starting auto-labeling..."

        def progress_callback(current, total):
            app.state.task_progress["current"] = current
            app.state.task_progress["total"] = total
            
        inferencer.run_inference(callback=progress_callback)
        
        app.state.task_progress["current"] = len(image_paths)
        app.state.task_progress["message"] = "Finalizing..."

        # data.yaml generation skipped for Project workflow (handled by dataset creation)
        
        if project_config_path.exists():
            # Update Project Config with Model Path (Reloading to be safe)
            with open(project_config_path, 'r') as f:
                p_config = json.load(f)
                
            p_config["model_path"] = request.model_path
            with open(project_config_path, 'w') as f:
                json.dump(p_config, f, indent=4)
                
            # Update result for frontend    
            app.state.task_progress["status"] = "idle"
            app.state.task_progress["message"] = "Auto-labeling complete!"
            app.state.task_progress["result"] = {
                "labeled_dir": str(labeled_dir_path.absolute()),
                "masked_dir": str(masked_images_dir.absolute()),
                "yaml_path": str(project_config_path),
                "classes": list(inferencer.model_.names.values())
            }
            return

        else:
             # Legacy Fallback
            config_update = {"model_path": request.model_path}
            save_project_config(Path(request.dataset_path), config_update)
            
            # For legacy, inferred dirs might be different if we didn't pass explicit ones (we didn't for this branch)
            # Default behavior of inferencer applies
            
            app.state.task_progress["status"] = "idle"
            app.state.task_progress["message"] = "Auto-labeling complete!"
            app.state.task_progress["result"] = {
                "labeled_dir": str(inferencer.labeled_labels_dir_.parent.absolute()) if inferencer.labeled_labels_dir_ else None, # approximate
                "masked_dir": str(inferencer.masked_images_dir_.absolute()),
                "yaml_path": "", # No yaml path guaranteed
                "classes": list(inferencer.model_.names.values())
            }

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

@app.post("/api/autolabel/single")
def autolabel_single_image(request: AutoLabelSingleRequest):
    img_dir = Path(request.dataset_path)
    if not img_dir.exists():
        raise HTTPException(status_code=400, detail="Dataset path does not exist")
    
    if not request.model_path or not Path(request.model_path).exists():
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {request.model_path}")

    # Verify image exists
    p = img_dir # Use 'p' for consistency with other functions that might modify it
    files = get_supported_image_files(p)

    # If no files found in root, check standard subdirectories
    if not files:
        root_p, identity = get_config_identity(p)
        potential_dirs = [
            p / "images",
            p / "masked_images",
            p / f"{identity}_processed_images",
            p / "processed_images"
        ]
        
        for d in potential_dirs:
            if d.exists() and d.is_dir():
                sub_files = get_supported_image_files(d)
                if sub_files:
                    files = sub_files
                    p = d # Update path to point to where images were found
                    break
    
    if not files:
        raise HTTPException(status_code=404, detail="No images found in dataset directory")

    try:
        # Initialize inferencer for single folder structure (usually what we want for webui)
        inferencer = YOLOInference(
            model_path=request.model_path,
            image_dir=str(img_dir),
            confidence=request.confidence,
            save_masked_images=request.save_masked,
            single_folder=True
        )
        
        # Inject the specific image to process
        inferencer.specific_images_ = [request.image_name]
        
        # Run inference synchronously since it's just one image
        inferencer.run_inference()
        
        # Determine output directories based on Project existence
        project_config_path = img_dir.parent / "project_config.json"
        
        if project_config_path.exists():
            with open(project_config_path, 'r') as f:
                p_config = json.load(f)
            
            # Use Project definitions
            labeled_dir_path = img_dir.parent / p_config.get("dirs", {}).get("annotations", "annotations")
            masked_images_dir = img_dir.parent / p_config.get("dirs", {}).get("masked", f"{p_config.get('name')}_masked_images")
            
            # Ensure directories exist
            labeled_dir_path.mkdir(parents=True, exist_ok=True)
            masked_images_dir.mkdir(parents=True, exist_ok=True)

            # NOTE: YOLOInference currently outputs to legacy folders. Move them.
            # inferencer.masked_images_dir_ -> masked_images_dir
            if request.save_masked and inferencer.masked_images_dir_.exists():
                for item in inferencer.masked_images_dir_.iterdir():
                     # Only move the specific image if we can identify it, or just all content since its single inode?
                     # Since we ran inference on specific image, it should only have that one.
                    if item.is_file():
                        shutil.move(str(item), str(masked_images_dir / item.name))
                # Cleanup legacy dir (careful if parallel, but this is single)
                try:
                    shutil.rmtree(inferencer.masked_images_dir_)
                except:
                    pass

            # legacy_labeled_dir -> labeled_dir_path
            legacy_labeled_dir = img_dir.parent / f"{img_dir.name}_labeled"
            if legacy_labeled_dir.exists():
                # Check for 'labels' subdir or flat txts
                # YOLOInference structure usually: labeled_dir / labels / file.txt
                source_labels = legacy_labeled_dir / "labels"
                if source_labels.exists():
                    for item in source_labels.iterdir():
                        if item.is_file():
                             shutil.move(str(item), str(labeled_dir_path / item.name))
                else:
                     for item in legacy_labeled_dir.iterdir():
                        if item.is_file() and item.suffix == '.txt':
                            shutil.move(str(item), str(labeled_dir_path / item.name))
                try:
                    shutil.rmtree(legacy_labeled_dir)
                except:
                    pass

            # Update Project Config
            p_config["model_path"] = request.model_path
            with open(project_config_path, 'w') as f:
                json.dump(p_config, f, indent=4)
                
            # Mount result (masked dir)
            app.mount("/static/single_label_result", StaticFiles(directory=str(masked_images_dir)), name="single_label_result")

            return {
                "status": "success",
                "image_name": request.image_name,
                "masked_url": f"/static/single_label_result/{Path(request.image_name).stem}.jpg", 
                "masked_dir": str(masked_images_dir.absolute()),
                "labeled_dir": str(labeled_dir_path.absolute())
            }

        else:
            # Legacy Fallback
            # Construct result paths
            masked_dir = img_dir.parent / f"{img_dir.name}_masked_images"
            labeled_dir = img_dir.parent / f"{img_dir.name}_labeled"
            
            # Update project config so Verification page knows where to look
            config_update = {
                "model_path": request.model_path,
                "labeled_dir": str(labeled_dir.absolute()),
                "masked_dir": str(masked_dir.absolute())
            }
            save_project_config(Path(request.dataset_path), config_update)
            
            # Mount the result directory to ensure it's accessible
            app.mount("/static/single_label_result", StaticFiles(directory=str(masked_dir)), name="single_label_result")

            return {
                "status": "success",
                "image_name": request.image_name,
                "masked_url": f"/static/single_label_result/{Path(request.image_name).stem}.jpg", 
                "masked_dir": str(masked_dir.absolute()),
                "labeled_dir": str(labeled_dir.absolute())
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Labeling failed: {str(e)}")

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
        
        processed_count = 0
        name_counts = {} # To handle collisions
        
        # Track output splits
        preserved_splits = set()
        
        for i, (img_src, label_src, dataset_idx) in enumerate(file_list):
            try:
                # Detect split from source path
                parts = img_src.parts
                split_name = "train" # Default
                
                if "train" in parts: split_name = "train"
                elif "valid" in parts: split_name = "valid"
                elif "val" in parts: split_name = "valid"
                elif "test" in parts: split_name = "test"
                else: 
                     # Flat structure -> train
                     split_name = "train"
                
                preserved_splits.add(split_name)
                
                # Output directories
                out_img_dir = output_dir / split_name / "images"
                out_lbl_dir = output_dir / split_name / "labels"
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_lbl_dir, exist_ok=True)

                img_name = img_src.name
                target_name = img_name
                
                # Conflict handling: prepend dataset index if collision
                # Key must be per-split
                key = f"{split_name}_{img_name}"
                
                if key in name_counts:
                     target_name = f"d{dataset_idx}_{img_name}"
                # Update collision tracking
                name_counts[key] = True
                # Actually we should track target names to ensure uniqueness
                # Simpler: if file exists in target? No, standard collision logic
                # Just stick to appending d{idx} if collision logic from before, 
                # but "name_counts" logic was a bit weird in original code (just checked existence in dict then set True)
                # It didn't actually check if *target* name existed, just if *source* name appeared before.
                
                # Let's improve collision handling
                # We need to ensure target_name is unique in output dir
                while (out_img_dir / target_name).exists():
                     target_name = f"d{dataset_idx}_{target_name}"
                
                # Copy image
                shutil.copy2(img_src, out_img_dir / target_name)
                
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
                
                with open(out_lbl_dir / f"{Path(target_name).stem}.txt", 'w') as f:
                    f.writelines(remapped_lines)
                
                processed_count += 1
                app.state.task_progress["current"] = i + 1
            except Exception as e:
                print(f"Error merging {img_src}: {e}")
                
        # 4. Create data.yaml
        new_yaml = {
            'path': str(output_dir.absolute()),
            'nc': len(all_class_names),
            'names': all_class_names
        }
        
        # Populate splits
        for split in preserved_splits:
             key = 'val' if split == 'valid' else split
             new_yaml[key] = f"{split}/images"
             
        # Defaults
        if 'train' not in new_yaml: new_yaml['train'] = 'train/images'
        if 'val' not in new_yaml: new_yaml['val'] = 'valid/images'
            
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

def run_sampling_task(request: SampleRequest):
    try:
        app.state.task_progress["status"] = "sampling"
        app.state.task_progress["message"] = "Starting sampling..."
        
        # Parse class percentages
        class_pcts = None
        if request.class_percentages:
            try:
                # Try JSON first
                parsed = json.loads(request.class_percentages)
                class_pcts = {int(k): float(v) for k,v in parsed.items()}
            except:
                # Try simple format
                 class_pcts = {}
                 for part in request.class_percentages.split(','):
                     k,v = part.split(':')
                     class_pcts[int(k)] = float(v)

        def progress_callback(current, total, msg=""):
            app.state.task_progress["current"] = current
            app.state.task_progress["total"] = total
            app.state.task_progress["message"] = msg
            
        sampler = DatasetSampler(
            request.dataset_path, 
            request.output_path, 
            verbose=True
        )
        
        summary = sampler.sample(
            global_percentage=request.percentage,
            class_percentages=class_pcts,
            count=request.count,
            seed=request.seed,
            progress_callback=progress_callback
        )
        
        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Sampling complete! Sampled {summary['total_sampled']} from {summary['total_original']} images."
        app.state.task_progress["result"] = {
            "output_dir": str(sampler.output_path),
            "stats": summary
        }
        
    except Exception as e:
        app.state.task_progress["status"] = "error"
        app.state.task_progress["message"] = f"Sampling Error: {str(e)}"
        print(f"Sampling Error: {e}")

@app.post("/api/dataset/sample_task")
def sample_dataset_task(request: SampleRequest, background_tasks: BackgroundTasks):
    p = Path(request.dataset_path)
    if not p.exists():
        raise HTTPException(status_code=400, detail="Dataset path does not exist")
        
    background_tasks.add_task(run_sampling_task, request)
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
        
        created_splits = set()
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
                # Detect split from path
                # Heuristic: check if "train", "valid", "test", "val" is in the parent parts
                # or strictly check known split names
                
                # Default to "train" if purely flat, but if flat we might want to keep it flat?
                # Actually, if we are extracting to a new dataset, it's safer to standardize on train/val/test if possible,
                # OR just keep flat if input is flat.
                
                # Let's try to detect current split
                parts = img_src.parts
                split_name = "train" # Default
                
                if "train" in parts: split_name = "train"
                elif "valid" in parts: split_name = "valid"
                elif "val" in parts: split_name = "valid"
                elif "test" in parts: split_name = "test"
                else: 
                     # If flat, maybe everything goes to "train"?
                     # Or stick to flat structure?
                     # Existing code outputted flat "images"/"labels".
                     # If we always output "train/images", then we change behavior for flat datasets.
                     # But split datasets MUST have splits.
                     # Let's check if we detected any split structure in the SOURCE scan?
                     pass
                     
                created_splits.add(split_name)

                # Output dirs
                out_img_dir = out_path / split_name / "images"
                out_lbl_dir = out_path / split_name / "labels"
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_lbl_dir, exist_ok=True)
                
                img_name = img_src.name
                target_img_name = img_name
                
                key = f"{split_name}_{img_name}"
                if key in name_counts:
                     name_counts[key] += 1
                     target_img_name = f"{img_src.stem}_{name_counts[key]}{img_src.suffix}"
                else:
                    name_counts[key] = 0
                
                shutil.copy2(img_src, out_img_dir / target_img_name)
                shutil.copy2(lbl_src, out_lbl_dir / f"{Path(target_img_name).stem}.txt")
                extracted_count += 1
            
            app.state.task_progress["current"] = i + 1
            if (i+1) % 10 == 0:
                app.state.task_progress["message"] = f"Processed {i+1}/{len(file_pairs)}... Extracted: {extracted_count}"

        # 4. Create new data.yaml
        new_yaml = {
            'path': str(out_path.absolute()),
            'nc': len(all_names),
            'names': all_names
        }
        
        # Populate splits in yaml
        for split in created_splits:
            # Map 'valid' to 'val' in yaml keys
            key = 'val' if split == 'valid' else split
            new_yaml[key] = f"{split}/images"
            
        # Ensure minimal keys
        if 'train' not in new_yaml: new_yaml['train'] = 'train/images'
        if 'val' not in new_yaml: new_yaml['val'] = 'valid/images'
        
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

        # 1. FIND PROJECT ROOT & IDENTITY STRICTLY
        # Traverse up to find project_config.json
        project_conf = None
        curr = labeled_root
        root = None
        project_name = None
        
        # Traverse up from labeled_root to find project_config.json
        # Limit depth to avoid infinite loop
        for _ in range(5): 
            if (curr / "project_config.json").exists():
                project_conf = curr / "project_config.json"
                root = curr
                break
            if curr.parent == curr: break
            curr = curr.parent
            
        if not root:
             # Fallback to heuristic if we can't find config
             # Try to guess root from "annotations" or parent of "_labeled"
             root, identity = get_config_identity(labeled_root)
             project_name = identity
        
        # Read config if found
        dirs_config = {}
        if project_conf:
             try:
                 with open(project_conf, 'r') as f:
                     pc = json.load(f)
                     project_name = pc.get("name", root.name)
                     dirs_config = pc.get("dirs", {})
             except:
                 project_name = root.name

        if not project_name: project_name = "project"

        # Determine Output Directory: ALWAYS <root>/<proj_name>_empty_images
        output_dir = root / f"{project_name}_empty_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. DETERMINE IMAGE SOURCE
        # Distinguish between "Entire Project" (annotations) vs "Dataset"
        img_src_path = None
        
        is_project_annotations = labeled_root.name == "annotations" or (labeled_root / "annotations").exists() # Heuristic
        dataset_root = None
        
        # If user passed "annotations" folder directly:
        if labeled_root.name == "annotations":
             # Branch 1: Project Annotations
             labels_dir = labeled_root
             
             # USE PROCESSED IMAGES
             if "processed" in dirs_config:
                  img_src_path = root / dirs_config["processed"]
             elif (root / f"{project_name}_processed_images").exists():
                  img_src_path = root / f"{project_name}_processed_images"
             else:
                  # Fallback
                  img_src_path = root / "processed_images"
        else:
             # Branch 2: Local Dataset
             # labeled_root might be "labels" folder or the dataset folder
             dataset_root = None
             if labeled_root.name == "labels":
                  labels_dir = labeled_root
                  dataset_root = labeled_root.parent
             else:
                  labels_dir = labeled_root / "labels"
                  dataset_root = labeled_root
             
             # Look for images sibling
             if (dataset_root / "images").exists():
                  img_src_path = dataset_root / "images"
             else:
                  img_src_path = dataset_root # Mixed dir?

        # Validation
        image_files = []
        is_split = False
        
        # Check for splits if img_src_path not explicitly found or we want to be robust
        possible_splits = ["train", "valid", "test", "val"]
        found_splits = []
        if dataset_root:
             found_splits = [s for s in possible_splits if (dataset_root / s / "images").exists()]
        
        if found_splits:
             print(f"Extraction detected split dataset: {found_splits}")
             is_split = True
             for s in found_splits:
                 split_imgs = get_supported_image_files(dataset_root / s / "images")
                 image_files.extend(split_imgs)
             print(f"Collected {len(image_files)} images from splits")
        else:
            if not img_src_path or not img_src_path.exists():
                 print(f"Warning: Could not pinpoint image source. Trying raw path: {raw_path}")
                 img_src_path = raw_path

            print(f"Extracting empty from {img_src_path} -> {output_dir} using labels from {labels_dir}")
            if img_src_path and img_src_path.exists():
                image_files = get_supported_image_files(img_src_path)
            else:
                image_files = []
        
        # Filter if specific images requested
        if request.images_list:
            target_names = set(request.images_list)
            # Fix: Compare stems, as stats API returns stems
            image_files = [p for p in image_files if p.stem in target_names]
            print(f"Filtered extraction to {len(image_files)} specific images")

        app.state.task_progress["total"] = len(image_files)
        
        empty_count = 0
        empty_count = 0
        for i, img_p in enumerate(image_files):
            # Check for label file
            # If split, we need to respect the split structure for label lookup?
            # Or does the user just want "empty images"?
            # The label_dir resolution before was simplistic for splits.
            # If split, labels are in dataset_root/split/labels.
            
            if is_split:
                 # Find which split this image belongs to
                 # img_p.parent is .../split/images
                 # labels should be .../split/labels
                 # This assumes standard YOLO structure
                 split_labels_dir = img_p.parent.parent / "labels"
                 lbl_p = split_labels_dir / f"{img_p.stem}.txt"
            else:
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

def draw_single_mask(img_p: Path, lbl_p: Path, out_path: Path, class_names: dict):
    """Draws masks on a single image and saves it to out_path."""
    img = cv2.imread(str(img_p))
    if img is None:
        return False
        
    h, w = img.shape[:2]
    
    if not lbl_p.exists():
        return False

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

    # out_file should be the full target path including subdirectory structure
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), img)
    return True

def run_mask_generation_task(request: GenerateMaskRequest):
    try:
        app.state.task_progress["status"] = "generating_masks"
        
        raw_path = Path(request.dataset_path).absolute()
        
        # 1. FIND PROJECT CONFIG
        # Traverse up to find project_config.json or deduce identity
        project_conf = None
        curr = raw_path
        root = None
        project_name = None
        
        for _ in range(4):
            if (curr / "project_config.json").exists():
                project_conf = curr / "project_config.json"
                root = curr
                break
            if curr.parent == curr: break
            curr = curr.parent
            
        # Fallback root determination
        if not root:
             root, identity = get_config_identity(raw_path)
             project_name = identity
        
        # Read config if found
        dirs_config = {}
        if project_conf:
             try:
                 with open(project_conf, 'r') as f:
                     pc = json.load(f)
                     project_name = pc.get("name", root.name)
                     dirs_config = pc.get("dirs", {})
             except:
                 project_name = root.name

        if not project_name: project_name = "project"


        # 2. DETERMINE LABELS PATH
        labels_path = None
        
        # User override?
        if request.labels_path:
            labels_path = Path(request.labels_path).absolute()
        
        # Config?
        if not labels_path and "annotations" in dirs_config:
            labels_path = root / dirs_config["annotations"]
            
        # Common locations
        if not labels_path:
            candidates = [
                root / "annotations",
                root / f"{project_name}_annotations",
                root / "labels",
                raw_path / "labels",
                raw_path.parent / "labels"
            ]
            for c in candidates:
                if c.exists() and (any(c.glob("*.txt")) or (c/"labels").exists()):
                   labels_path = c
                   break

        if not labels_path or not labels_path.exists():
             raise ValueError(f"Could not find labels directory. Searched in common locations within {root}")

        # Handle 'labels' subdirectory structure
        # If labels_path is a root like 'labeled' or 'annotations', check inside
        if (labels_path / "labels").exists():
            labels_path = labels_path / "labels"
            
        # 3. DETERMINE IMAGE SOURCE
        # Where are the images we want to mask?
        # Usually 'processed' or 'dataset/images'
        
        img_src_path = None
        
        # Config?
        if "processed" in dirs_config:
             potential = root / dirs_config["processed"]
             if potential.exists(): img_src_path = potential
             
        # Deduced?
        if not img_src_path:
            # If current raw_path has images, use it (unless it IS the output dir we are trying to fill)
            # Check if raw_path is likely the output dir (e.g. ends with _masked_images)
            is_output_target = raw_path.name.endswith("_masked_images")
            
            if not is_output_target and any(get_supported_image_files(raw_path)):
                img_src_path = raw_path
            else:
                # Look for siblings
                candidates = [
                    root / f"{project_name}_processed",
                    root / f"{project_name}_processed_images",
                    root / "processed",
                    root / "images",
                    root / f"{project_name}_images"
                ]
                for c in candidates:
                    if c.exists() and any(get_supported_image_files(c)):
                        img_src_path = c
                        break
        
        if not img_src_path or not img_src_path.exists():
             raise ValueError(f"Could not find source images (processed or raw). Searched common locations.")

        print(f"DEBUG: Mask Gen - Source: {img_src_path}, Labels: {labels_path}")

        # 4. DETERMINE OUTPUT PATH
        out_path = None
        if request.output_path:
            out_path = Path(request.output_path).absolute()
        elif "masked" in dirs_config:
            out_path = root / dirs_config["masked"]
        else:
            # Logic: [Project]_masked_images
            if raw_path.name.endswith("_masked_images"):
                out_path = raw_path # Overwrite/Fill current
            else:
                out_path = root / f"{project_name}_masked_images"
            
        os.makedirs(out_path, exist_ok=True)

        # Load class names
        class_names = {}
        # Try both root and dataset for yaml/config
        names_list = load_classes_from_path(img_src_path)
        if not names_list: names_list = load_classes_from_path(root)
        
        if names_list:
             class_names = {i: n for i, n in enumerate(names_list)}

        image_files = get_supported_image_files(img_src_path)
        app.state.task_progress["total"] = len(image_files)
        app.state.task_progress["message"] = f"Generating masks for {len(image_files)} images from {img_src_path.name}..."

        saved_count = 0
        for i, img_p in enumerate(image_files):
            # Calculate where to save: preserve relative structure from img_src_path
            try:
                rel_p = img_p.relative_to(img_src_path)
            except:
                rel_p = Path(img_p.name)
            
            # Ensure output is .jpg
            out_file = out_path / rel_p.with_suffix(".jpg")
            
            # Label might be local to labels_path, or nested if we want to mirror
            lbl_p = labels_path / f"{img_p.stem}.txt"
            
            # Try to handle nested labels if they exist mirroring img_src_path structure
            if not lbl_p.exists():
                potential_nested = labels_path / rel_p.with_suffix(".txt")
                if potential_nested.exists():
                    lbl_p = potential_nested
            
            if draw_single_mask(img_p, lbl_p, out_file, class_names):
                saved_count += 1
            
            app.state.task_progress["current"] = i + 1
            if (i+1) % 10 == 0:
                app.state.task_progress["message"] = f"Processed {i+1}/{len(image_files)} images... Saved: {saved_count}"

        if saved_count == 0:
             raise ValueError(f"No corresponding label files found in {labels_path}. Verified {len(image_files)} images from {img_src_path}.")

        app.state.task_progress["status"] = "idle"
        app.state.task_progress["message"] = f"Mask generation complete! Saved {saved_count} images to {out_path}"
        app.state.task_progress["result"] = {
            "masked_dir": str(out_path),
            "labeled_dir": str(labels_path.parent.absolute() if labels_path.name == "labels" else labels_path.absolute())
        }
        
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
        p / "annotations", # Current path's annotations subfolder
        root / "annotations", # Project-standard annotations folder
        root / f"{identity}_datasets" / "labels",
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
            if item.is_dir() and item.name.startswith(identity) and (item.name.endswith("_labeled") or item.name.endswith("_datasets")):
                if (item / "labels").exists():
                    return item / "labels"
                    
    # Final fallback if nothing found but root/labels is a standard guess
    return None

@app.get("/api/labeled/images")
def get_labeled_images(path: str, limit: int = 50, offset: int = 0, classes: Optional[str] = None, search: Optional[str] = None, split: Optional[str] = "all"):
    print(f"Listing images in {path} with limit={limit}, offset={offset}, classes={classes}, search={search}, split={split}")
    p = Path(path).absolute()
    if not p.exists() or not p.is_dir():
        return {"images": [], "total": 0, "offset": offset, "limit": limit}
    
    # Define root early so it's available for fallback and yaml lookups
    original_p = p
    root, _ = get_config_identity(p)
    
    labels_dir = discover_labels_dir(p)
    if not labels_dir:
        labels_dir = root / "labels"
    
    print(f"Using labels directory: {labels_dir}")  
    
    # Load class names using the unified helper
    names_list = load_classes_from_path(p) 
    class_names_map = {}
    if names_list:
        class_names_map = {i: n for i, n in enumerate(names_list)}
        print(f"Loaded {len(class_names_map)} class names using unified loader")

    # Image Collection Strategy
    files = []
    
    # Check for Split Dataset
    possible_splits = ["train", "valid", "test", "val"]
    is_split_dataset = False
    for s in possible_splits:
        if (p / s / "images").exists():
            is_split_dataset = True
            break
            
    if is_split_dataset:
         print(f"Detected split dataset. Split filter: {split}")
         for s in possible_splits:
             # Skip if we are filtering for a specific split
             if split and split != "all" and split != s:
                 # Special case: normalize val/valid
                 if split in ["val", "valid"] and s in ["val", "valid"]:
                     pass # Don't skip if both are validation
                 else:
                     continue
                     
             s_img_dir = p / s / "images"
             if s_img_dir.exists():
                 # Collect images keeping relative path structure
                 sub_files = get_supported_image_files(s_img_dir)
                 files.extend(sub_files)
    else:
        # Standard flat dataset: check root, then standard subdirs
        files = get_supported_image_files(p)
        
        # If no files found in root, check standard subdirectories (legacy behavior)
        if not files:
            if (p / "images").exists():
                 print(f"No images in root, using {p / 'images'}")
                 files = get_supported_image_files(p / "images")
            elif (p / "masked_images").exists():
                print(f"No images in root, using {p / 'masked_images'}")
                files = get_supported_image_files(p / "masked_images")
    

    target_classes = []
    if classes:
        target_classes = [c.strip() for c in classes.split(',')]
        print(f"Filtering by classes: {target_classes}")

    # SEARCH FILTERING
    if search:
        search_lower = search.lower()
        files = [f for f in files if search_lower in f.name.lower()]
        print(f"Filtered by search '{search}': {len(files)} files remaining")

    # SORTING: Always sort for consistency across different requests
    files.sort(key=lambda x: str(x))

    # PAGINATION & PROJECTION
    # Note: files contains Path objects (absolute)
    
    # If filtering by class is ON, we must scan ALL to paginate correctly
    if target_classes:
        print(f"Scanning {len(files)} files for class filtering...")
        paged_files = files # We'll filter all then paginate
    else:
        # Optimization: Slice first, then process stats
        total = len(files)
        paged_files = files[offset:offset + limit]

    results = []
    
    # Helper to resolve label for an image path
    def resolve_label_file(img_path):
        # If split, label is in sibling 'labels' folder
        # e.g. .../train/images/foo.jpg -> .../train/labels/foo.txt
        if is_split_dataset:
            # img_path parent is 'images'
            # img_path parent parent is split dir (e.g. 'train')
            split_root = img_path.parent.parent
            return split_root / "labels" / f"{img_path.stem}.txt"
        else:
            return labels_dir / f"{img_path.stem}.txt"

    for f in paged_files:
        stats = {}
        # Try to read corresponding label file
        label_file = resolve_label_file(f)
        
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
        
        # Always return path relative to the input 'path' (original_p)
        try:
            rel_name = str(f.relative_to(original_p))
        except:
            rel_name = f.name
        
        item = {"name": rel_name, "stats": stats}
        
        if target_classes:
            # Check if this item matches
            # Handle "Empty" special case
            is_empty = not stats
            
            match_found = False
            
            # 1. Check if "Empty" is requested and this image is empty
            if "Empty" in target_classes and is_empty:
                match_found = True
            
            # 2. Check if any other requested class is present in stats
            if not match_found and stats:
                real_targets = [c for c in target_classes if c != "Empty"]
                if any(cls in real_targets for cls in stats.keys()):
                    match_found = True
            
            if match_found:
                results.append(item)
        else:
            results.append(item)

    # Now handle pagination based on whether filtering occurred
    if target_classes:
        total = len(results)
        results = results[offset:offset + limit]

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
    
    # 1. Determine Project Root and Config
    # We can try to look up project config from the source path tree
    root = find_project_root(source_p) # Helper function exists? Or use parent traversal
    if not root:
        root = source_p.parent.parent # Heuristic for now if find_project_root fails or isn't robust
    
    # Try to load project config
    project_conf = root / "project_config.json"
    dirs_config = {}
    project_name = root.name
    
    if project_conf.exists():
        try:
             with open(project_conf, 'r') as f:
                pc = json.load(f)
                dirs_config = pc.get("dirs", {})
                project_name = pc.get("name", root.name)
        except Exception as e:
            print(f"Error reading project config: {e}")
            
    # 2. Determine Source (Processed Image)
    # User said: "copy the corresponding image (from the _processed_images folder)"
    # Config key 'processed' or default
    rel_processed = dirs_config.get("processed", f"{project_name}_processed_images")
    src_processed_dir = root / rel_processed
    
    src_img = src_processed_dir / request.image_name
    
    if not src_img.exists():
        # Fallback: Check if we can find it where the user is looking?
        # If user is in _masked_images, maybe we want to fallback to raw?
        # But request is specific. Let's try to be helpful if processed fails.
        if (source_p / request.image_name).exists():
            print(f"Warning: Processed image {src_img} not found. Falling back to source {source_p}")
            src_img = source_p / request.image_name
        else:
             # NEW: Relative path support check?
             # request.image_name might be "train/images/foo.jpg"
             if (source_p / request.image_name).exists():
                 src_img = source_p / request.image_name
             else:
                 raise HTTPException(status_code=404, detail=f"Source image {request.image_name} not found in {src_processed_dir} or {source_p}")

    # 3. Determine Target (Filtered Folder)
    # User said: "to the _filtered folder. there is no need to create subfolders"
    if root.name == project_name: # Simple heuristic: if root.name IS the project name, we are likely in dataset root
         # Stick to existing logic
         rel_filtered = dirs_config.get("filtered", f"{project_name}_filtered")
    else:
         # Standalone fallback. Create filtered_images in "source_p" (the dataset root)
         rel_filtered = "filtered_images"
         if not (root / rel_filtered).exists() and not (root / f"{project_name}_filtered").exists():
              # If no existing project folder, use simple one
              target_dir = root / "filtered_images"
         else:
              target_dir = root / dirs_config.get("filtered", f"{project_name}_filtered")

    if not 'target_dir' in locals():
         target_dir = root / rel_filtered

    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Copy
    dest_path = target_dir / request.image_name
    
    # Ensure nested subdirectories exist (e.g. train/images/)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {src_img} to {dest_path}")
    shutil.copy2(src_img, dest_path)
    
    return {"status": "ok", "message": f"Copied to {target_dir.name}"}

@app.get("/api/dataset/render_image")
def render_dataset_image(path: str, name: str):
    """
    Render a single image with annotations on-the-fly.
    path: The directory containing the image (e.g., images/ or processed/)
    name: The filename of the image (or relative path for splits e.g. train/images/foo.jpg)
    """
    img_dir = Path(path).absolute()
    img_path = img_dir / name
    
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    # 1. Resolve Labels Directory / Path
    lbl_path = None
    
    # Check if this is a split structure (e.g. train/images/foo.jpg)
    parts = Path(name).parts
    if len(parts) > 1 and "images" in parts:
         # Resolve label: replace 'images' with 'labels' in the parent path
         try:
             parent = img_path.parent
             if parent.name == "images":
                  lbl_path = parent.parent / "labels" / f"{img_path.stem}.txt"
         except:
             pass

    if not lbl_path:
        # Fallback to standard flattened logic
        root, _ = get_config_identity(img_dir)
        labels_dir = discover_labels_dir(img_dir)
        if not labels_dir:
            labels_dir = root / "labels"
        
        # Check if 'name' itself has subfolders (e.g. images/foo.jpg or subset/foo.jpg)
        name_path = Path(name)
        if len(name_path.parts) > 1:
            # Try to resolve mirroring the structure
            # e.g. path=root, name=images/foo.jpg -> root/labels/foo.txt
            # Or if name=subset/foo.jpg -> root/labels/subset/foo.txt
            
            # If it starts with 'images/', strip it for the label path
            if name_path.parts[0] == "images":
                lbl_rel_path = Path(*name_path.parts[1:]).with_suffix(".txt")
            else:
                lbl_rel_path = name_path.with_suffix(".txt")
                
            lbl_path = labels_dir / lbl_rel_path
        else:
            lbl_path = labels_dir / f"{img_path.stem}.txt"
        
        # Final desperate fallback
        if not lbl_path.exists() and not (len(name_path.parts) > 1 and name_path.parts[0] == "images"):
             # If we failed and it's flat name, try prefixing labels/ just in case
             lbl_path = labels_dir / f"{img_path.stem}.txt"

    # 2. Load Class Names
    names_list = load_classes_from_path(img_dir)
    class_names = {}
    if names_list:
        class_names = {i: n for i, n in enumerate(names_list)}
        
    # 3. Read Image
    img = cv2.imread(str(img_path))
    if img is None:
         raise HTTPException(status_code=500, detail="Failed to read image")
         
    h, w = img.shape[:2]
    
    # 4. Draw Annotations
    if lbl_path and lbl_path.exists():
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    # Color
                    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,0), (255,0,255)]
                    color = colors[cls_id % len(colors)]
                    
                    label = class_names.get(cls_id, f"class_{cls_id}")
                    
                    if len(coords) == 4:
                        # Bounding Box
                        cx, cy, dw, dh = coords
                        x1 = int((cx - dw/2) * w)
                        y1 = int((cy - dh/2) * h)
                        x2 = int((cx + dw/2) * w)
                        y2 = int((cy + dh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif len(coords) >= 6:
                        # Polygon
                        pts = []
                        for j in range(0, len(coords), 2):
                            pts.append([int(coords[j] * w), int(coords[j+1] * h)])
                        pts = np.array(pts, np.int32)
                        cv2.polylines(img, [pts], True, color, 2)
                        cv2.putText(img, label, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except ValueError:
                    pass

    # 5. Encode to JPEG
    res, im_jpg = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")

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
            
            
    # Check for splits (train, valid, test)
    splits = ["train", "valid", "test", "val"]
    found_splits = [s for s in splits if (p / s).exists() and (p / s / "labels").exists()]

    if not labels_dir and not found_splits:
        print(f"Labels directory not found in {p}")
        raise HTTPException(status_code=400, detail="Labels directory not found")
        
    print(f"Resolved labels_dir: {labels_dir} | Found splits: {found_splits}")

    # Load class names
    # We pass 'root' because that's where config likely is, but p (dataset dir) works too
    names_list = load_classes_from_path(p) 
    class_names_map = {}
    if names_list:
        class_names_map = {i: n for i, n in enumerate(names_list)}
        print(f"Loaded {len(class_names_map)} class names using unified loader")

    # Helper to aggregate stats
    def aggregate_stats(target_labels_dir: Path, target_images_dir: Optional[Path]):
        l_counts = {}
        l_total_obj = 0
        l_empty = []
        l_total_img = 0
        
        if target_images_dir and target_images_dir.exists():
            l_total_img = len(get_supported_image_files(target_images_dir))
        elif target_labels_dir.exists():
             # Fallback if images dir not separate
             l_total_img = len(get_supported_image_files(target_labels_dir))
             
        # Count objects
        labeled_stems = set()
        if target_labels_dir.exists():
            for label_file in target_labels_dir.glob("*.txt"):
                if label_file.name == "classes.txt": continue
                
                labeled_stems.add(label_file.stem)
                has_obj = False
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                has_obj = True
                                cls_id = int(parts[0])
                                cls_name = class_names_map.get(cls_id, f"class_{cls_id}")
                                l_counts[cls_name] = l_counts.get(cls_name, 0) + 1
                                l_total_obj += 1
                except: pass
                
                if not has_obj:
                    l_empty.append(label_file.stem)
        
        # Check non-labeled images
        if target_images_dir and target_images_dir.exists():
            for img_p in get_supported_image_files(target_images_dir):
                if img_p.stem not in labeled_stems:
                    l_empty.append(img_p.stem)

        return l_counts, l_total_obj, l_empty, l_total_img

    if found_splits:
        print(f"Detected split dataset with: {found_splits}")
        total_images = 0
        total_objects = 0
        class_counts = {}
        empty_images_list = []
        
    per_split_stats = {}
    if found_splits:
        print(f"Detected split dataset with: {found_splits}")
        total_images = 0
        total_objects = 0
        class_counts = {}
        empty_images_list = []
        
        for s in found_splits:
            s_labels = p / s / "labels"
            s_images = p / s / "images"
            if not s_images.exists(): s_images = None
            
            c, o, e, i = aggregate_stats(s_labels, s_images)
            
            # Format individual split stats
            split_data = []
            for cls_name, count in c.items():
                split_data.append({
                    "class": cls_name,
                    "count": count,
                    "percentage": round((count / o) * 100, 2) if o > 0 else 0
                })
            split_data.sort(key=lambda x: x["count"], reverse=True)
            
            per_split_stats[s] = {
                "total_images": i,
                "total_objects": o,
                "class_counts": split_data,
                "empty_images_count": len(e),
                "empty_images": e
            }

            # Merge
            total_images += i
            total_objects += o
            empty_images_list.extend(e)
            for k, v in c.items():
                class_counts[k] = class_counts.get(k, 0) + v
                
        empty_images = empty_images_list # Flattened list
        
    else:
        # ---- EXISTING FLAT LOGIC ----
        
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
                    elif item.name.endswith("_labeled") or item.name.endswith("_datasets"):
                        labeled_dirs.append(item)
                    elif item.name.endswith("_masked_images"):
                         labeled_dirs.append(item)
    
        # Try to load exact processed dir from config
        config_processed = None
        try:
            if (root / "project_config.json").exists():
                with open(root / "project_config.json") as f:
                    pc = json.load(f)
                    if "dirs" in pc and "processed" in pc["dirs"]:
                         config_processed = root / pc["dirs"]["processed"]
        except: pass
    
        potential_images = [
            labels_dir.parent / "images", # legacy root
            labels_dir.parent / "masked_images",
            root / f"{identity}_datasets" / "images", # prefixed root
            root / f"{identity}_labeled" / "images", # prefixed root
            root / f"{identity}_masked_images",
            root / f"{identity}_processed_images", # Add processed images
            root / "images",
            root / "masked_images",
            root / "processed_images", # Add legacy/standard processed name
            processed_dir, # Add discovered processed dir
            config_processed # Add configured processed dir
        ]
        
        # Try to search for ANY subdirectory starting with identity and ending with _labeled/images
        if root.exists():
            for item in root.iterdir():
                if item.is_dir() and item.name.startswith(identity) and (item.name.endswith("_labeled") or item.name.endswith("_datasets")):
                    potential_images.append(item / "images")
                if item.is_dir() and item.name.startswith(identity) and item.name.endswith("_masked_images"):
                    potential_images.append(item)
    
        images_dir = None
        for pid in potential_images:
            if pid and pid.exists() and pid.is_dir():
                # Verify it has images
                if get_supported_image_files(pid):
                    images_dir = pid
                    break
                
        c, o, e, i = aggregate_stats(labels_dir, images_dir)
        class_counts = c
        total_objects = o
        empty_images = e
        total_images = i
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
        "empty_count": len(empty_images),
        "per_split_stats": per_split_stats if 'per_split_stats' in locals() else None
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
        
        # Safe iteration and collection
        with os.scandir(p) as entries:
            for entry in entries:
                try:
                    # Robust check for directory
                    try:
                        is_dir = entry.is_dir()
                    except OSError:
                        continue # Skip unreadable

                    if only_dirs and not is_dir:
                        continue
                    
                    # Robust size retrieval
                    size = 0
                    if not is_dir:
                        try:
                            size = entry.stat().st_size
                        except (FileNotFoundError, PermissionError, OSError):
                            size = 0 
                    
                    items.append({
                        "name": entry.name,
                        "path": str(Path(entry.path).absolute()),
                        "is_dir": is_dir,
                        "size": size
                    })
                except Exception:
                    continue
        
        # Sort safe items: Directories first, then alphabetical
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

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

class SAMPredictRequest(BaseModel):
    model_path: str
    image_path: str
    image_name: str
    points: List[List[float]] # Normalized [[x,y], ...]
    labels: List[int] # [1, 0, ...]
    epsilon: float = 1.0 # Approximation error for polygon simplification

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
            print(f"find_image_file: Found at {c}")
            return c
            
    # 2. Search recursively if not in obvious places (max depth 3)
    # Only do this if image_name doesn't already look like a subpath
    if "/" not in image_name and "\\" not in image_name:
        print(f"find_image_file: Not found in candidates, searching recursively for {image_name}...")
        for ext in ["", ".jpg", ".png", ".jpeg", ".WEBP", ".JPG"]:
            name_to_find = image_name if not ext else f"{Path(image_name).stem}{ext}"
            # Check if we can find it by walking (limit walk)
            for root, dirs, files in os.walk(str(dataset_root)):
                if name_to_find in files:
                    found_p = Path(root) / name_to_find
                    print(f"find_image_file: Found via recursion at {found_p}")
                    return found_p
                if root.count(os.sep) - str(dataset_root).count(os.sep) > 3:
                     dirs[:] = [] # stop recursion
    
    print(f"find_image_file: FAILED to find {image_name} in {dataset_root}")
    return None

def get_label_path(dataset_root: Path, image_name: str) -> Path:
    """Consistently find or determine the label path for an image."""
    # NEW: Check if this is part of a Project Structure first
    try:
        project_root = find_project_root(dataset_root)
        project_conf = project_root / "project_config.json"
        if project_conf.exists():
            # If we are in a project, annotations are centralized
            with open(project_conf, 'r') as f:
                conf_data = json.load(f)
            
            # Helper to get annotation dir from config or default
            rel_ann = conf_data.get("dirs", {}).get("annotations", "annotations")
            ann_dir = project_root / rel_ann
            if ann_dir.exists():
                return ann_dir / f"{Path(image_name).stem}.txt"
    except: 
        pass

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
    print(f"DEBUG: Request to serve image {image_name} from {path}")
    img_path = find_image_file(p, image_name)
    print(f"DEBUG: Found image path: {img_path}")
    
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
            
        # PROACTIVE: Update masked image if possible
        try:
            masked_dir = None
            
            # Check Project Config first
            project_conf = p.parent / "project_config.json"
            if project_conf.exists():
                with open(project_conf, 'r') as f:
                    pc = json.load(f)
                masked_name = pc.get("dirs", {}).get("masked", f"{pc.get('name')}_masked_images")
                masked_dir = p.parent / masked_name
            else:
                # Legacy fallback
                root, identity = get_config_identity(p)
                masked_dir = root / f"{identity}_masked_images"

            if masked_dir and masked_dir.exists():
                # Find original image
                img_path = find_image_file(p, request.image_name)
                if img_path:
                    # Load classes
                    # Load classes using unified helper
                    names_list = load_classes_from_path(p) 
                    class_names = {}
                    if names_list:
                         class_names = {i: n for i, n in enumerate(names_list)}
                    
                    draw_single_mask(img_path, label_path, masked_dir, class_names)
                    print(f"DEBUG: Automatically updated mask for {request.image_name} in {masked_dir}")
        except Exception as e:
            print(f"Warning: Failed to auto-update mask: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write labels: {e}")
        
    return {"status": "ok", "message": "Saved"}

@app.post("/api/annotation/sam_predict")
def sam_predict(request: SAMPredictRequest):
    """Run SAM3 inference for a given point/box prompt."""
    p = Path(request.image_path).absolute()
    img_path = find_image_file(p, request.image_name)
    
    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image {request.image_name} not found")
        
    # Model storage/caching
    if not hasattr(app.state, 'sam_model') or app.state.sam_model_path != request.model_path:
        print(f"Loading SAM model from {request.model_path}...")
        try:
            app.state.sam_model = SAM(request.model_path)
            app.state.sam_model_path = request.model_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load SAM model: {e}")

    # Prepare points
    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        raise HTTPException(status_code=500, detail="Failed to read image for SAM")
    h, w = img_cv.shape[:2]
    
    # Scale points from normalized to absolute
    abs_points = [[p[0] * w, p[1] * h] for p in request.points]
    
    try:
        # Run inference
        # SAM 3 predict: Source can be path or ndarray
        results = app.state.sam_model.predict(
            source=img_path, 
            points=[abs_points], 
            labels=[request.labels],
            verbose=False
        )
        
        if not results or len(results) == 0:
            return {"points": []}
            
        # Get the mask polygons
        masks = results[0].masks
        if masks is not None and len(masks.xy) > 0:
            # Take the first mask (pixel coordinates)
            poly_pixels = masks.xy[0].astype(np.float32)
            
            # Simplify polygon using Douglas-Peucker
            if request.epsilon > 0:
                # poly_pixels is (N, 2)
                poly_pixels = cv2.approxPolyDP(poly_pixels, request.epsilon, True)
                # Reshape back to (N, 2)
                poly_pixels = poly_pixels.reshape(-1, 2)
            
            # Normalize points
            poly_norm = []
            for pt in poly_pixels:
                poly_norm.append(float(pt[0] / w))
                poly_norm.append(float(pt[1] / h))
                
            return {"points": poly_norm}
            
        return {"points": []}
    except Exception as e:
        print(f"SAM Predict Error: {e}")
        raise HTTPException(status_code=500, detail=f"SAM Inference error: {e}")



# -------------------------------------------------------------------------
# Training Endpoints
# -------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    model: str = "yolov8n.pt"
    data: str
    epochs: int = 100
    batch: Union[int, float] = 16 # Supports int or float (though usually int) or -1
    imgsz: int = 640
    device: Optional[str] = None
    workers: int = 8
    project: Optional[str] = None
    name: Optional[str] = None
    exist_ok: bool = False
    optimizer: str = "auto"
    verbose: bool = True
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    resume: bool = False
    amp: bool = True
    fraction: float = 1.0
    profile: bool = False
    freeze: Optional[int] = None
    # Add generic overrides
    overrides: Optional[Dict[str, Any]] = None

class ExportConfig(BaseModel):
    model: str
    format: str
    imgsz: Union[int, List[int]] = 640
    batch: int = 1
    device: Optional[str] = None
    half: bool = False
    int8: bool = False
    dynamic: bool = False
    simplify: bool = False
    opset: Optional[int] = None
    workspace: Optional[int] = None # GB
    nms: bool = False
    data: Optional[str] = None # Required for INT8 calibration
    overrides: Optional[Dict[str, Any]] = None

@app.post("/api/augment/simple/scan")
def simple_augmentation_scan(request: SimpleAugmentationScanRequest):
    try:
        if not os.path.exists(request.dataset_path):
             raise HTTPException(status_code=400, detail="Dataset path not found")
        
        return SimpleAugmentationManager.scan_dataset(request.dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment/simple/preview")
def simple_augmentation_preview(request: SimpleAugmentationPreviewRequest):
    try:
        return SimpleAugmentationManager.generate_preview(request.dataset_path, request.config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment/simple/run")
def simple_augmentation_run(request: SimpleAugmentationRunRequest, background_tasks: BackgroundTasks):
    try:
        if app.state.task_progress["status"] != "idle" and app.state.task_progress["status"] != "error":
             raise HTTPException(status_code=400, detail="A task is already running")
             
        app.state.task_progress["status"] = "starting"
        app.state.task_progress["current"] = 0
        app.state.task_progress["total"] = 0
        app.state.task_progress["message"] = "Starting..."
        app.state.task_progress["result"] = None
        
        background_tasks.add_task(
            SimpleAugmentationManager.run_augmentation_job,
            request.dataset_path,
            request.output_name,
            request.multiplier,
            request.config,
            app.state.task_progress,
            request.selected_splits
        )
        
        return {"status": "started", "output_path": request.output_name}
    except Exception as e:
        app.state.task_progress["status"] = "error"
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/models")
def get_training_models():
    """Return list of featured models supported by Ultralytics."""
    return {
        "YOLOv8": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        "YOLOv9": ["yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e"],
        "YOLOv10": ["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"],
        "YOLO11": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        "YOLOv5": ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
        "YOLO26": ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
    }

@app.get("/api/training/devices")
def get_training_devices():
    """Return list of available devices (CPU + GPUs)."""
    devices = [{"id": "cpu", "name": "CPU"}]
    
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for i in range(count):
            try:
                name = torch.cuda.get_device_name(i)
                devices.append({"id": str(i), "name": f"GPU {i}: {name}"})
            except:
                devices.append({"id": str(i), "name": f"GPU {i}"})
                
    return devices

@app.post("/api/training/start")
def start_training(config: TrainingConfig):
    # Flatten config to dict for ultralytics
    # Handle 'overrides'
    base_config = config.dict(exclude={"overrides"})
    
    # Merge overrides if any
    final_config = base_config.copy()
    if config.overrides:
        final_config.update(config.overrides)
        
    try:
        return training_manager.start_training(final_config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/training/stop")
def stop_training():
    return training_manager.stop_training()

@app.get("/api/training/status")
def get_training_status():
    return training_manager.get_state()

@app.post("/api/export/start")
def start_export(config: ExportConfig):
    # Flatten config to dict for ultralytics
    # Handle 'overrides'
    base_config = config.dict(exclude={"overrides"})
    
    # Merge overrides if any
    final_config = base_config.copy()
    if config.overrides:
        final_config.update(config.overrides)
        
    try:
        return export_manager.start_export(final_config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/export/stop")
def stop_export():
    return export_manager.stop_export()

@app.get("/api/export/status")
def get_export_status():
    return export_manager.get_state()




from workflow_manager import WorkflowManager, Job

# ... existing code ...

# Workflow API Models
class WorkflowJobCreateRequest(BaseModel):
    project_path: str
    batch_size: int
    annotator: str
    reviewer: str
    source: str = "unassigned"
    include_annotated: Optional[bool] = False
    source_job_id: Optional[str] = None

class WorkflowJobActionRequest(BaseModel):
    project_path: str

class WorkflowAnnotationSaveRequest(BaseModel):
    project_path: str
    image_name: str
    content: str

class WorkflowImageActionRequest(BaseModel):
    project_path: str
    image_name: str
    comment: Optional[str] = None

class UserRegisterRequest(BaseModel):
    project_path: str
    name: str

# Workflow Endpoints

@app.get("/api/workflow/unassigned")
def get_workflow_unassigned_count(project_path: str):
    try:
        wm = WorkflowManager(project_path)
        count = wm.get_unassigned_count()
        return {"count": count}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/dataset")
def get_workflow_dataset_count(project_path: str):
    try:
        wm = WorkflowManager(project_path)
        count = wm.get_dataset_count()
        return {"count": count}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/jobs")
def get_workflow_jobs(project_path: str):
    try:
        wm = WorkflowManager(project_path)
        return wm.get_jobs()
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/create")
def create_workflow_job(request: WorkflowJobCreateRequest):
    try:
        wm = WorkflowManager(request.project_path)
        return wm.create_job(
            request.batch_size,
            request.annotator,
            request.reviewer,
            request.source,
            request.include_annotated,
            request.source_job_id
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/unassign")
def unassign_workflow_job(job_id: str, request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.unassign_job(job_id)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/job/{job_id}")
def get_workflow_job(job_id: str, project_path: str):
    try:
        wm = WorkflowManager(project_path)
        job = wm.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/submit")
def submit_workflow_job(job_id: str, request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.submit_job(job_id)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/approve_all")
def approve_workflow_job(job_id: str, request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.approve_job(job_id)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/return")
def return_workflow_job(job_id: str, request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.return_job(job_id)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/annotation")
def save_workflow_annotation(job_id: str, request: WorkflowAnnotationSaveRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.save_annotation(job_id, request.image_name, request.content)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/job/{job_id}/annotation")
def get_workflow_annotation(job_id: str, image_name: str, project_path: str):
    try:
        wm = WorkflowManager(project_path)
        return wm.get_annotation(job_id, image_name)
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/dataset/reset")
def reset_workflow_dataset(request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.reset_dataset()
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/approve")
def approve_workflow_image(job_id: str, request: WorkflowImageActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.approve_image(job_id, request.image_name)
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/job/{job_id}/reject")
def reject_workflow_image(job_id: str, request: WorkflowImageActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.reject_image(job_id, request.image_name, request.comment or "")
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflow/users")
def get_workflow_users(project_path: str):
    try:
        wm = WorkflowManager(project_path)
        return wm.get_users()
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/users/register")
def register_workflow_user(request: UserRegisterRequest):
    try:
        wm = WorkflowManager(request.project_path)
        return wm.register_user(request.name)
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/dataset/reset")
def reset_workflow_dataset(request: WorkflowJobActionRequest):
    try:
        wm = WorkflowManager(request.project_path)
        wm.reset_dataset()
        return {"status": "success"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
