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
    has_structure = (input_path / "images").exists() and (input_path / "images").is_dir()
    source_dir = input_path / "images" if has_structure else input_path
    
    # Gather images
    images = get_image_files(source_dir)
    if not images:
        raise ValueError(f"No objects found in {source_dir}")
    
    # Shuffle for random distribution
    random.shuffle(images)
    
    total_images = len(images)
    
    # Distribute images
    parent_dir = input_path.parent
    base_name = input_path.name

    # Cleaning up previous splits
    print(f"Checking for previous splits in {parent_dir}...")
    for item in parent_dir.iterdir():
        if item.is_dir() and item.name.startswith(f"{base_name}_split"):
            try:
                print(f"Removing previous split: {item}")
                shutil.rmtree(item)
            except Exception as e:
                print(f"Error removing {item}: {e}")

    processed_count = 0
    splits = [[] for _ in range(num_splits)]
    for idx, img in enumerate(images):
        splits[idx % num_splits].append(img)

    for i, split_images in enumerate(splits):
        split_idx = i + 1
        output_dir = parent_dir / f"{base_name}_split{split_idx}"
        output_dir.mkdir(exist_ok=True)
        
        # Prepare output structure
        out_images_dir = output_dir
        out_labels_dir = output_dir
        
        if has_structure:
            out_images_dir = output_dir / "images"
            out_labels_dir = output_dir / "labels"
            out_images_dir.mkdir(exist_ok=True)
            out_labels_dir.mkdir(exist_ok=True)
            
            # Copy data.yaml if exists (only to the first split or all? All makes them independent datasets)
            yaml_path = input_path / "data.yaml"
            if yaml_path.exists():
                shutil.copy2(yaml_path, output_dir)

        for img_path in split_images:
            # Copy image
            shutil.copy2(img_path, out_images_dir)
            
            # Handle labels
            label_found = False
            
            # 1. Check if structured (look in sibling labels folder)
            if has_structure:
                # input was .../dataset/images/img.jpg
                # looking for .../dataset/labels/img.txt
                src_labels_dir = input_path / "labels"
                if src_labels_dir.exists():
                    label_path = src_labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, out_labels_dir)
                        label_found = True
            
            # 2. If not structured or label not found yet, check same directory (flat)
            if not label_found:
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    # If structured output, put in labels folder, else flat
                    shutil.copy2(label_path, out_labels_dir)

            processed_count += 1
            if callback:
                callback(processed_count, total_images)

    return processed_count
