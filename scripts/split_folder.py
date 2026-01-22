import os
import shutil
import math
import random
from pathlib import Path
from typing import List, Tuple

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def get_image_files(directory: Path) -> List[Path]:
    """Get all supported image files from a directory."""
    images = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(f)
    return images

def split_folder(input_dir: str, num_splits: int, callback=None):
    """
    Splits images in the input directory into num_splits equal parts.
    Creates sibling directories named {input_dir_name}_split{i}.
    Copies corresponding label files if they exist.
    
    Args:
        input_dir: Path to the input directory
        num_splits: Number of splits to create
        callback: Optional callback(current, total) for progress tracking
    """
    if num_splits < 2:
        raise ValueError("Number of splits must be at least 2")

    input_path = Path(input_dir).resolve()
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Detect structure
    splits_found = []
    for split in ["train", "valid", "test", "val"]:
        if (input_path / split / "images").exists():
            splits_found.append(split)
            
    has_splits = len(splits_found) > 0
    has_structure = has_splits or ((input_path / "images").exists() and (input_path / "images").is_dir())
    
    # Cleaning up previous splits
    parent_dir = input_path.parent
    base_name = input_path.name
    print(f"Checking for previous splits in {parent_dir}...")
    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith(f"{base_name}_split"):
            try:
                print(f"Removing previous split: {item}")
                shutil.rmtree(item)
            except Exception as e:
                print(f"Error removing {item}: {e}")

    processed_count = 0
    
    # Function to distribute a list of images into N splits
    def distribute_images(image_list, subpath=""):
        nonlocal processed_count
        random.shuffle(image_list)
        
        # Create chunks
        chunks = [[] for _ in range(num_splits)]
        for i, img in enumerate(image_list):
            chunks[i % num_splits].append(img)
            
        for i, chunk in enumerate(chunks):
            split_idx = i + 1
            output_dir = parent_dir / f"{base_name}_split{split_idx}"
            
            # Destination path logic
            if subpath: # Split dataset case: _splitX / train / images
                out_images_dir = output_dir / subpath / "images"
                out_labels_dir = output_dir / subpath / "labels"
            elif has_structure: # Flat but structured: _splitX / images
                out_images_dir = output_dir / "images"
                out_labels_dir = output_dir / "labels"
            else: # Completely Flat: _splitX
                out_images_dir = output_dir
                out_labels_dir = output_dir
            
            out_images_dir.mkdir(parents=True, exist_ok=True)
            out_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy data.yaml if exists and not already there
            yaml_path = input_path / "data.yaml"
            dst_yaml = output_dir / "data.yaml"
            if yaml_path.exists() and not dst_yaml.exists():
                shutil.copy2(yaml_path, output_dir)

            for img_path in chunk:
                # Copy Image
                shutil.copy2(img_path, out_images_dir)
                
                # Copy Label
                label_name = f"{img_path.stem}.txt"
                
                # Search strategy
                candidates = []
                
                # 1. Sibling labels folder (relative to image parent)
                if (img_path.parent.parent / "labels" / label_name).exists():
                    candidates.append(img_path.parent.parent / "labels" / label_name)
                    
                # 2. Input root labels (for flat structured)
                if (input_path / "labels" / label_name).exists():
                    candidates.append(input_path / "labels" / label_name)
                    
                # 3. Same folder
                if (img_path.with_suffix(".txt")).exists():
                     candidates.append(img_path.with_suffix(".txt"))
                     
                for lp in candidates:
                    if lp.exists():
                        shutil.copy2(lp, out_labels_dir)
                        break
                        
                processed_count += 1
                if callback:
                    # We pass processed_count so far
                    # Total is roughly known via pre-calculation below
                    pass

    # Execution Flow
    if has_splits:
        total_images = 0
        all_work = []
        for split in splits_found:
            imgs = get_image_files(input_path / split / "images")
            all_work.append((split, imgs))
            total_images += len(imgs)
            
        for split, imgs in all_work:
            distribute_images(imgs, subpath=split)
            # Roughly update progress? 
            # distribute matches processed_count
            if callback: callback(processed_count, total_images)
            
    else:
        # Flat or "images/labels" root
        source_dir = input_path / "images" if has_structure else input_path
        images = get_image_files(source_dir)
        if not images:
             raise ValueError(f"No objects found in {source_dir}")
        distribute_images(images, subpath="")
        if callback: callback(processed_count, len(images))

    return processed_count
