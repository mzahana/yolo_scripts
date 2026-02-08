import os
import shutil
import json
import yaml
import random
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel
from workflow_manager import WorkflowManager

class ProjectConfig(BaseModel):
    name: str
    created_at: str
    description: str = ""
    description: str = ""
    classes: List[str] = []
    sam_model_path: Optional[str] = None
    model_path: Optional[str] = None # For YOLO model
    
    # Paths relative to project root
    dirs: Dict[str, str] = {
        "raw": "images_raw",
        "processed": "images_processed",
        "annotations": "annotations",
        "masked": "images_masked",
        "filtered": "images_filtered",
        "labeled": "datasets"
    }

class ProjectManager:
    CONFIG_FILENAME = "project_config.json"
    CLASSES_FILENAME = "classes.txt"

    @staticmethod
    def create_project(name: str, parent_dir: str, raw_images_source: str, classes: List[str]) -> Dict:
        """
        Creates a new project structure.
        """
        project_root = Path(parent_dir) / name
        
        if project_root.exists():
            raise FileExistsError(f"Project directory '{project_root}' already exists.")
        
        # Create directories
        dirs = {
            "raw": project_root / f"{name}_raw_images",
            "processed": project_root / f"{name}_processed_images",
            "annotations": project_root / "annotations",
            "masked": project_root / f"{name}_masked_images",
            "filtered": project_root / f"{name}_filtered_images",
            "labeled": project_root / f"{name}_datasets"
        }
        
        for k, d in dirs.items():
            if k == "raw": continue
            d.mkdir(parents=True, exist_ok=True)
            
        # Move raw images folder instead of copying (Efficiency)
        if raw_images_source and raw_images_source.strip():
            source_path = Path(raw_images_source)
            target_path = dirs["raw"]
            
            # Ensure we are not moving the parent dir or current dir
            # Also catch if source_path resolves to target_path parent
            try:
                if source_path.exists() and source_path.is_dir() and source_path.resolve() != Path('.').resolve():
                     shutil.move(str(source_path), str(target_path))
                else:
                     target_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not move raw images: {e}")
                target_path.mkdir(parents=True, exist_ok=True)
        else:
            dirs["raw"].mkdir(parents=True, exist_ok=True)

        # Create classes.txt
        with open(project_root / ProjectManager.CLASSES_FILENAME, 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
                
        # Create Config
        from datetime import datetime
        config = ProjectConfig(
            name=name,
            created_at=datetime.now().isoformat(),
            classes=classes,
            dirs={k: v.name for k, v in dirs.items()}
        )
        
        with open(project_root / ProjectManager.CONFIG_FILENAME, 'w') as f:
            f.write(config.model_dump_json(indent=4))
            
        return {
            "path": str(project_root),
            "config": config.model_dump()
        }

    @staticmethod
    def create_project_from_split(name: str, parent_dir: str, split_dataset_path: str) -> Dict:
        """
        Creates a new project structure from an existing split dataset.
        """
        split_path = Path(split_dataset_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Split dataset path '{split_path}' does not exist.")

        data_yaml = split_path / "data.yaml"
        if not data_yaml.exists():
            # Try to find it in 1 level deep (common case)
            found = list(split_path.glob("**/data.yaml"))
            if found:
                data_yaml = found[0]
            else:
                 raise FileNotFoundError(f"data.yaml not found in '{split_path}'. Required for split import.")
        
        # Parse classes from data.yaml
        classes = []
        try:
            with open(data_yaml, 'r') as f:
                y = yaml.safe_load(f)
                names = y.get('names', [])
                if isinstance(names, dict):
                    sorted_keys = sorted([int(k) for k in names.keys()])
                    classes = [names.get(k) if isinstance(k, str) else names.get(k) for k in sorted_keys] # Handle potential mix? usually dict keys are ints
                    # Fix: yaml loaded dict keys could be int or str. 
                    # If yaml is {0: 'a'}, keys are 0.
                    classes = [names[k] for k in sorted(names.keys())]
                elif isinstance(names, list):
                    classes = names
        except Exception as e:
            raise ValueError(f"Failed to parse classes from data.yaml: {e}")

        project_root = Path(parent_dir) / name
        if project_root.exists():
             raise FileExistsError(f"Project directory '{project_root}' already exists.")

        # Create directories
        # Note: 'raw' is created but empty as per requirements
        dirs = {
            "raw": project_root / f"{name}_raw_images",
            "processed": project_root / f"{name}_processed_images",
            "annotations": project_root / "annotations",
            "masked": project_root / f"{name}_masked_images",
            "filtered": project_root / f"{name}_filtered_images",
            "labeled": project_root / f"{name}_datasets"
        }
        
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # 1. Copy original dataset to _datasets subfolder
        # We start with this to ensure we have the source
        target_split_dir = dirs["labeled"] / split_path.name
        if target_split_dir.exists():
            shutil.rmtree(target_split_dir)
        shutil.copytree(split_path, target_split_dir)
        
        # 2. Flatten images and annotations
        # We traverse the COPIED split dataset to avoid touching original? 
        # Actually requirements say "copy original... to _datasets", and "copy all images... to _processed".
        # We can scan the target_split_dir we just created.
        
        # Common split structures:
        # A) root/train/images/*.jpg, root/train/labels/*.txt
        # B) root/images/train/*.jpg, root/labels/train/*.txt
        # C) root/*.jpg (flat - unlikely with "split" desc but possible)
        
        # Robust Walker
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        processed_count = 0
        
        # Walk through the COPIED dataset to find all images
        for root, _, files in os.walk(target_split_dir):
            for file in files:
                if Path(file).suffix.lower() in image_exts:
                    img_src = Path(root) / file
                    
                    # Copy image to processed
                    shutil.copy2(img_src, dirs["processed"] / file)
                    processed_count += 1
                    
                    # Look for corresponding label
                    # Heuristic: check sibling 'labels' folder or parallel 'labels' folder
                    # 1. Sibling 'labels' (e.g. dataset/train/images/a.jpg -> dataset/train/labels/a.txt)
                    # 2. Parallel 'labels' (e.g. dataset/images/train/a.jpg -> dataset/labels/train/a.txt)
                    # 3. Same folder
                    
                    lbl_name = f"{img_src.stem}.txt"
                    lbl_src = None
                    
                    # Check same folder
                    if (img_src.parent / lbl_name).exists():
                         lbl_src = img_src.parent / lbl_name
                    
                    # Check sibling 'labels' folder ( ../labels )
                    elif (img_src.parent.parent / "labels" / lbl_name).exists():
                         lbl_src = img_src.parent.parent / "labels" / lbl_name
                         
                    # Check parallel structure with subfolder retention?
                    # dataset/images/train -> dataset/labels/train
                    elif "images" in img_src.parent.parts:
                        # Replace 'images' with 'labels' in path parts
                        # careful with indices
                        parts = list(img_src.parent.parts)
                        # Find right-most 'images'
                        try:
                            idx = len(parts) - 1 - parts[::-1].index("images")
                            parts[idx] = "labels"
                            potential_lbl = Path(*parts) / lbl_name
                            if potential_lbl.exists():
                                lbl_src = potential_lbl
                        except ValueError:
                            pass
                            
                    if lbl_src:
                        shutil.copy2(lbl_src, dirs["annotations"] / lbl_name)

        # Create classes.txt
        with open(project_root / ProjectManager.CLASSES_FILENAME, 'w') as f:
            for cls in classes:
                f.write(f"{cls}\n")
                
        # Create Config
        from datetime import datetime
        config = ProjectConfig(
            name=name,
            created_at=datetime.now().isoformat(),
            classes=classes,
            dirs={k: v.name for k, v in dirs.items()}
        )
        # Force model_path if best.pt exists in split? No requirement for that.
        
        with open(project_root / ProjectManager.CONFIG_FILENAME, 'w') as f:
            f.write(config.model_dump_json(indent=4))
            
        return {
            "path": str(project_root),
            "config": config.model_dump(),
            "imported_count": processed_count
        }

    @staticmethod
    def load_project(project_path: str) -> Dict:
        """
        Loads an existing project.
        """
        root = Path(project_path)
        config_path = root / ProjectManager.CONFIG_FILENAME
        
        if not config_path.exists():
            raise FileNotFoundError(f"Not a valid project. Missing {ProjectManager.CONFIG_FILENAME}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Validate/Reload classes from text file if it exists, as it might be easier to edit manually
        classes_path = root / ProjectManager.CLASSES_FILENAME
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
                config_data['classes'] = classes

        # Construct absolute paths for frontend convenience
        dirs = config_data.get("dirs", {})
        abs_dirs = {}
        for k, v in dirs.items():
            abs_dirs[k] = str(root / v)
            
        return {
            "path": str(root),
            "config": config_data,
            "paths": abs_dirs
        }


    @staticmethod
    def update_project_config(project_path: str, updates: Dict) -> Dict:
        """
        Updates the project configuration.
        """
        root = Path(project_path)
        config_path = root / ProjectManager.CONFIG_FILENAME
        
        if not config_path.exists():
             raise FileNotFoundError(f"Config not found at {config_path}")
             
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Update allowed fields
        for k, v in updates.items():
            if k in ['sam_model_path', 'model_path', 'classes', 'description']:
                config_data[k] = v
                
        # Save back
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
            
        return config_data
        return config_data

    @staticmethod
    def create_dataset(project_path: str, dataset_name: str, strategy: str = "all", split_ratios: List[float] = [0.7, 0.2, 0.1]) -> Dict:
        """
        Creates a YOLO dataset from processed images and annotations.
        strategy: 'all' (copy all found pairs), 'random' (not implemented yet, defaults to all)
        """
        root = Path(project_path)
        config = ProjectManager.load_project(project_path)["config"]
        
        rel_processed = config.get("dirs", {}).get("processed", f"{config.get('name')}_processed_images")
        rel_annotations = config.get("dirs", {}).get("annotations", "annotations")
        rel_labeled = config.get("dirs", {}).get("labeled", f"{config.get('name')}_datasets")
        
        source_images = root / rel_processed
        source_labels = root / rel_annotations
        target_dataset = root / rel_labeled / dataset_name
        
        if not source_images.exists():
            raise FileNotFoundError(f"Source images dir not found: {source_images}")
        if not source_labels.exists():
            raise FileNotFoundError(f"Source annotations dir not found: {source_labels}")
            
        target_dataset.mkdir(parents=True, exist_ok=True)
        target_dataset.mkdir(parents=True, exist_ok=True)
        # We delay subfolder creation until we know the strategy
        
        # Find matching pairs
        processed_count = 0
        
        # Get all images
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images = []
        for item in source_images.iterdir():
            if item.is_file() and item.suffix.lower() in image_exts:
                images.append(item)
                
        # Filter those that have annotations AND are approved in workflow
        # Load workflow state
        workflow_state_path = root / "workflow_state.json"
        approved_images = set()
        if workflow_state_path.exists():
            try:
                with open(workflow_state_path, 'r') as f:
                     state = json.load(f)
                     for img_name, data in state.get("images", {}).items():
                         if data.get("status") == "dataset":
                             approved_images.add(img_name)
            except Exception as e:
                print(f"Error reading workflow state: {e}")

        valid_pairs = []
        for img in images:
            # Check if image is approved
            if workflow_state_path.exists() and img.name not in approved_images:
                continue

            txt_path = source_labels / f"{img.stem}.txt"
            if txt_path.exists():
                valid_pairs.append((img, txt_path))
        
        # Apply strategy
        selected_pairs = valid_pairs
        if strategy == "split":
            random.shuffle(selected_pairs)
            
            total_pairs = len(selected_pairs)
            train_ratio = split_ratios[0]
            val_ratio = split_ratios[1] if len(split_ratios) > 1 else 0.0
            test_ratio = split_ratios[2] if len(split_ratios) > 2 else 0.0
            
            # Re-normalize if needed or just use as is (assuming sum=1)
            # Use counts to be safe
            train_count = int(total_pairs * train_ratio)
            val_count = int(total_pairs * val_ratio)
            # test_count gets the remainder
            test_count = total_pairs - train_count - val_count
            
            train_pairs = selected_pairs[:train_count]
            val_pairs = selected_pairs[train_count:train_count+val_count]
            test_pairs = selected_pairs[train_count+val_count:]
            
            sets = [
                ("train", train_pairs),
                ("valid", val_pairs),
                ("test", test_pairs)
            ]
            
            data_yaml_paths = {}
            
            for split_name, pairs in sets:
                if not pairs:
                    # If empty split (e.g. test=0)
                    continue
                    
                split_dir = target_dataset / split_name
                (split_dir / "images").mkdir(parents=True, exist_ok=True)
                (split_dir / "labels").mkdir(parents=True, exist_ok=True)
                
                for img, txt in pairs:
                    shutil.copy2(img, split_dir / "images" / img.name)
                    shutil.copy2(txt, split_dir / "labels" / txt.name)
                    
                data_yaml_paths[split_name] = f"{split_name}/images"
                
            processed_count = len(selected_pairs)
            
            # Create data.yaml for SPLIT
            classes = config.get("classes", [])
            data_yaml = {
                "path": str(target_dataset.absolute()), 
                "train": data_yaml_paths.get("train", "train/images"),
                "val": data_yaml_paths.get("valid", data_yaml_paths.get("train", "train/images")), # Ultralytics uses 'val' key for validation data
                "test": data_yaml_paths.get("test", None),
                "nc": len(classes),
                "names": classes
            }
            # Remove test key if None
            if data_yaml["test"] is None:
                del data_yaml["test"]
                
        else:
            # Default 'all'
            (target_dataset / "images").mkdir(exist_ok=True)
            (target_dataset / "labels").mkdir(exist_ok=True)
            
            for img, txt in selected_pairs:
                shutil.copy2(img, target_dataset / "images" / img.name)
                shutil.copy2(txt, target_dataset / "labels" / txt.name)
                processed_count += 1
                
            # Create data.yaml for FLAT
            classes = config.get("classes", [])
            data_yaml = {
                "path": str(target_dataset.absolute()),
                "train": "images",
                "val": "images",
                "nc": len(classes),
                "names": classes
            }

        with open(target_dataset / "data.yaml", 'w') as f:
             yaml.dump(data_yaml, f, sort_keys=False)
             
        return {
            "name": dataset_name,
            "path": str(target_dataset),
            "count": processed_count,
            "yaml": str(target_dataset / "data.yaml")
        }

    @staticmethod
    def list_datasets(project_path: str) -> List[Dict]:
        """
        Lists all datasets created in the project.
        """
        root = Path(project_path)
        try:
            config = ProjectManager.load_project(project_path)["config"]
        except:
             return []
             
        rel_labeled = config.get("dirs", {}).get("labeled", f"{config.get('name')}_datasets")
        labeled_root = root / rel_labeled
        
        datasets = []
        if labeled_root.exists() and labeled_root.is_dir():
            for item in labeled_root.iterdir():
                if item.is_dir() and (item / "data.yaml").exists():
                    # It's a valid YOLO dataset
                    # Count images
                    splits_counts = {}
                    img_count = 0
                    if (item / "images").exists():
                        img_count = len([x for x in (item / "images").iterdir() if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']])
                    else:
                        # Count splits
                        for sub in ["train", "valid", "val", "test"]:
                             if (item / sub / "images").exists():
                                 c = len([x for x in (item / sub / "images").iterdir() if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']])
                                 splits_counts[sub] = c
                                 img_count += c
                        
                    datasets.append({
                        "name": item.name,
                        "path": str(item),
                        "image_count": img_count,
                        "splits": splits_counts,
                        "created_at": item.stat().st_ctime # Approx
                    })
        
        return sorted(datasets, key=lambda x: x['created_at'], reverse=True)

