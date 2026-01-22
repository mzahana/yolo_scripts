import os
import shutil
import json
import yaml
import random
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    name: str
    created_at: str
    description: str = ""
    classes: List[str] = []
    
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
        
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # Copy raw images (Scanning and Copying)
        # Note: In a real scenario with huge datasets, we might want to symlink or just reference.
        # But the request implies the folder structure is self-contained.
        # "subfolder with the raw images"
        # We will copy for now to be safe and self-contained as requested.
        source_path = Path(raw_images_source)
        if source_path.exists() and source_path.is_dir():
             for item in source_path.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(item, dirs["raw"] / item.name)

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
                
        # Filter those that have annotations
        valid_pairs = []
        for img in images:
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

