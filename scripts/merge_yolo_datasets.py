"""
Enhanced Merge Datasets Script with Progress Bars

This script merges multiple YOLO datasets into one consolidated dataset with two modes:
1. Preserve mode: Maintains original train/valid/test splits from each dataset
2. Re-split mode: Merges all data and creates new random splits with custom ratios

Each input dataset must have the same structure, with subdirectories for train, valid, and test splits,
each containing images and corresponding labels. The merged dataset will have a unified data.yaml file with 
merged class names and proper class remapping.

Features:
- Visual progress bars for all operations
- Handles filename conflicts by adding unique suffixes
- Merges class names from all datasets and remaps class indices
- Optional re-splitting of merged data with custom ratios
- Validates dataset compatibility before merging
- Provides detailed logging of the merge process

Requirements:
    pip install pyyaml tqdm

Usage:
    # Preserve original splits
    python merge_datasets.py /path/to/dataset1 /path/to/dataset2 /path/to/output_dir
    
    # Re-split merged data with default ratios (80/10/10)
    python merge_yolo_datasets.py /path/to/dataset1 /path/to/dataset2 /path/to/output_dir --resplit
    
    # Re-split with custom ratios
    python merge_yolo_datasets.py /path/to/dataset1 /path/to/dataset2 /path/to/output_dir --resplit --ratios 0.7 0.2 0.1

Copyright (c) Mohamed Abdelakder 2024
Enhanced version with progress bars 2024
"""

import os
import shutil
import argparse
import yaml
from collections import defaultdict
from pathlib import Path
import hashlib
import warnings
import random
from typing import List, Tuple, Dict
import sys

# Try to import tqdm, provide fallback if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  Warning: tqdm not installed. Install it for progress bars: pip install tqdm")
    print("   Continuing without progress bars...\n")
    
    # Fallback wrapper that acts like tqdm but does nothing
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, unit='it', leave=True, 
                     bar_format=None, ncols=None, colour=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            self.n = 0
            if desc:
                print(f"{desc}...")
        
        def __iter__(self):
            return iter(self.iterable) if self.iterable else self
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            self.n += n
        
        def set_description(self, desc):
            self.desc = desc
        
        def close(self):
            pass

def generate_unique_suffix(dataset_idx, original_path):
    """Generate a unique suffix based on dataset index and file hash."""
    # Create a short hash of the original file path
    path_hash = hashlib.md5(original_path.encode()).hexdigest()[:6]
    return f"_d{dataset_idx}_{path_hash}"

def validate_datasets(dataset_dirs):
    """Validate that all datasets have compatible structures."""
    print("\n🔍 Validating datasets...")
    issues = []
    warnings_list = []
    
    with tqdm(dataset_dirs, desc="Checking datasets", unit="dataset", 
              bar_format='{l_bar}{bar:30}{r_bar}', colour='cyan') as pbar:
        for dataset_dir in pbar:
            dataset_name = Path(dataset_dir).name
            pbar.set_description(f"Checking {dataset_name}")
            
            if not os.path.exists(dataset_dir):
                issues.append(f"Dataset directory does not exist: {dataset_dir}")
                continue
                
            # Check for data.yaml
            yaml_path = os.path.join(dataset_dir, "data.yaml")
            if not os.path.exists(yaml_path):
                issues.append(f"Missing data.yaml in: {dataset_dir}")
            
            # Check for standard splits
            splits_found = False
            for split in ["train", "valid", "test"]:
                if os.path.exists(os.path.join(dataset_dir, split, "images")):
                    splits_found = True
            
            # Check for flat structure
            flat_images = os.path.join(dataset_dir, "images")
            flat_labels = os.path.join(dataset_dir, "labels")
            
            if not splits_found and not (os.path.exists(flat_images) and os.path.exists(flat_labels)):
                 issues.append(f"Invalid structure in {dataset_name}: Needs either train/valid/test splits OR images/labels subdirectories")
            
            if splits_found:
                 # Warn if some splits missing but not fatal
                 pass
    
    # Show warnings if any
    if warnings_list:
        print("\n⚠️  Warnings:")
        for warning in warnings_list:
            print(f"   - {warning}")
    
    if issues:
        print("\n❌ Validation errors found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ All datasets validated successfully!")
    return True

def merge_class_names(dataset_dirs):
    """Merge class names from all datasets and create mapping."""
    print("\n🏷️  Merging class names from all datasets...")
    all_classes = []  # List to maintain YOLO format (index = class ID)
    all_classes_set = set()  # Set for quick lookup of existing classes
    class_mappings = {}  # Store mapping for each dataset
    
    with tqdm(enumerate(dataset_dirs), total=len(dataset_dirs), 
              desc="Processing class names", unit="dataset",
              bar_format='{l_bar}{bar:30}{r_bar}', colour='green') as pbar:
        for dataset_idx, dataset_dir in pbar:
            dataset_name = Path(dataset_dir).name
            pbar.set_description(f"Processing {dataset_name}")
            yaml_path = os.path.join(dataset_dir, "data.yaml")
            
            try:
                with open(yaml_path, 'r') as file:
                    data_yaml = yaml.safe_load(file)
                
                # Handle different formats of class names in YOLO
                names = data_yaml.get('names', [])
                
                # Ensure we're working with a list format
                if isinstance(names, dict):
                    # Convert dict to list, ensuring correct order by index
                    max_idx = max(names.keys())
                    names = [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
                
                # Create mapping for this dataset
                dataset_mapping = {}
                for old_idx, class_name in enumerate(names):
                    if class_name not in all_classes_set:
                        # Add new class with next available index
                        new_idx = len(all_classes)
                        all_classes.append(class_name)
                        all_classes_set.add(class_name)
                    else:
                        # Find existing index for this class
                        new_idx = all_classes.index(class_name)
                    
                    dataset_mapping[old_idx] = new_idx
                
                class_mappings[dataset_idx] = dataset_mapping
                
            except Exception as e:
                print(f"\n⚠️  Error reading classes from {dataset_dir}: {e}")
                class_mappings[dataset_idx] = {}
    
    print(f"✅ Total unique classes found: {len(all_classes)}")
    return all_classes, class_mappings

def remap_labels(label_file_path, class_mapping, output_path):
    """Remap class indices in a label file according to the mapping."""
    try:
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        
        remapped_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # YOLO format: class_id x y w h
                old_class_id = int(parts[0])
                new_class_id = class_mapping.get(old_class_id, old_class_id)
                parts[0] = str(new_class_id)
                remapped_lines.append(' '.join(parts) + '\n')
            else:
                remapped_lines.append(line)  # Keep line as is if format is unexpected
        
        with open(output_path, 'w') as f:
            f.writelines(remapped_lines)
        
        return True
    except Exception as e:
        return False

def collect_all_files(dataset_dirs, class_mappings) -> Tuple[List[Tuple[str, str, int]], int]:
    """
    Collect all image-label pairs from all datasets and splits.
    Returns: List of tuples (image_path, label_path, dataset_idx) and conflict count
    """
    all_files = []
    file_tracking = set()
    conflict_count = 0
    
    print("\n📁 Collecting all files from datasets...")
    
    # First, count total files for progress bar
    total_files = 0
    for dataset_dir in dataset_dirs:
        for split in ["train", "valid", "test", "flat"]:
            if split == "flat":
                 images_src = os.path.join(dataset_dir, "images")
            else:
                 images_src = os.path.join(dataset_dir, split, "images")
            
            if os.path.exists(images_src):
                total_files += len([f for f in os.listdir(images_src) 
                                  if os.path.isfile(os.path.join(images_src, f))])
    
    with tqdm(total=total_files, desc="Scanning files", unit="files",
              bar_format='{l_bar}{bar:30}{r_bar}', colour='blue') as pbar:
        for dataset_idx, dataset_dir in enumerate(dataset_dirs):
            dataset_name = Path(dataset_dir).name
            
            # Check if flat
            is_flat = os.path.exists(os.path.join(dataset_dir, "images")) and not os.path.exists(os.path.join(dataset_dir, "train"))
            # Iterate splits or flat
            splits_to_check = ["flat"] if is_flat else ["train", "valid", "test"]

            for split in splits_to_check:
                if split == "flat":
                     images_src = os.path.join(dataset_dir, "images")
                     labels_src = os.path.join(dataset_dir, "labels")
                else:
                     images_src = os.path.join(dataset_dir, split, "images")
                     labels_src = os.path.join(dataset_dir, split, "labels")
                
                if os.path.exists(images_src) and os.path.exists(labels_src):
                    image_files = [f for f in os.listdir(images_src) 
                                 if os.path.isfile(os.path.join(images_src, f))]
                    
                    for img_file in image_files:
                        pbar.update(1)
                        img_path = os.path.join(images_src, img_file)
                        
                        # Find corresponding label file
                        label_name = os.path.splitext(img_file)[0] + '.txt'
                        label_path = os.path.join(labels_src, label_name)
                        
                        if os.path.exists(label_path):
                            # Check for filename conflict
                            if img_file in file_tracking:
                                conflict_count += 1
                            else:
                                file_tracking.add(img_file)
                            
                            all_files.append((img_path, label_path, dataset_idx))
    
    print(f"✅ Total files collected: {len(all_files)}")
    if conflict_count > 0:
        print(f"⚠️  Filename conflicts detected: {conflict_count} (will be handled)")
    return all_files, conflict_count

def split_dataset(all_files: List[Tuple[str, str, int]], ratios: List[float], seed: int = 42) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    Split the collected files into train/valid/test sets according to ratios.
    """
    print(f"\n🎲 Shuffling and splitting data (seed={seed})...")
    random.seed(seed)
    random.shuffle(all_files)
    
    total = len(all_files)
    train_size = int(total * ratios[0])
    valid_size = int(total * ratios[1])
    
    splits = {
        'train': all_files[:train_size],
        'valid': all_files[train_size:train_size + valid_size],
        'test': all_files[train_size + valid_size:]
    }
    
    print(f"✅ Data split complete: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    return splits

def copy_files_with_progress(file_list, split_name, output_dir, class_mappings):
    """Copy files to output directory with progress bar."""
    split_dir = "valid" if split_name == "valid" else split_name
    images_dst = os.path.join(output_dir, split_dir, "images")
    labels_dst = os.path.join(output_dir, split_dir, "labels")
    
    file_name_tracking = set()
    
    with tqdm(file_list, desc=f"  {split_name.capitalize():5}", unit="pairs",
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}',
              leave=True, colour='yellow') as pbar:
        for img_src, label_src, dataset_idx in pbar:
            # Get original filename
            img_filename = os.path.basename(img_src)
            label_filename = os.path.basename(label_src)
            
            # Handle filename conflicts
            if img_filename in file_name_tracking:
                # Generate unique filename
                name, ext = os.path.splitext(img_filename)
                suffix = generate_unique_suffix(dataset_idx, img_src)
                img_filename = f"{name}{suffix}{ext}"
                label_filename = f"{name}{suffix}.txt"
            else:
                file_name_tracking.add(img_filename)
            
            # Copy image
            img_dst = os.path.join(images_dst, img_filename)
            shutil.copy2(img_src, img_dst)
            
            # Copy and remap label
            label_dst = os.path.join(labels_dst, label_filename)
            if dataset_idx in class_mappings and class_mappings[dataset_idx]:
                remap_labels(label_src, class_mappings[dataset_idx], label_dst)
            else:
                shutil.copy2(label_src, label_dst)

def merge_datasets_with_resplit(dataset_dirs, output_dir, ratios, seed=42):
    """Merge datasets and create new random splits."""
    
    # Validate datasets first
    if not validate_datasets(dataset_dirs):
        print("\n❌ Dataset validation failed. Please fix the issues and try again.")
        return False
    
    # Merge class names and get mappings
    merged_classes, class_mappings = merge_class_names(dataset_dirs)
    
    # Collect all files from all datasets
    all_files, conflict_count = collect_all_files(dataset_dirs, class_mappings)
    
    if not all_files:
        print("❌ No valid image-label pairs found in the datasets!")
        return False
    
    # Split the data according to ratios
    print(f"\n📊 Split ratios: train={ratios[0]:.0%}, valid={ratios[1]:.0%}, test={ratios[2]:.0%}")
    splits = split_dataset(all_files, ratios, seed)
    
    # Create output directories
    print("\n📂 Creating output directories...")
    for split in ["train", "valid", "test"]:
        images_path = os.path.join(output_dir, split, "images")
        labels_path = os.path.join(output_dir, split, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)
    
    # Copy files to new splits
    print("\n📋 Copying files to new splits...")
    split_sizes = {}
    for split_name, file_list in splits.items():
        copy_files_with_progress(file_list, split_name, output_dir, class_mappings)
        split_sizes[split_name] = len(file_list)
    
    # Print summary
    print("\n" + "="*50)
    print("MERGE AND RE-SPLIT SUMMARY")
    print("="*50)
    print(f"Datasets merged: {len(dataset_dirs)}")
    print(f"Total files processed: {len(all_files)}")
    print(f"Filename conflicts handled: {conflict_count}")
    print(f"Total unique classes: {len(merged_classes)}")
    print(f"\nNew split sizes:")
    for split, size in split_sizes.items():
        percentage = (size / len(all_files)) * 100 if all_files else 0
        print(f"  {split}: {size} files ({percentage:.1f}%)")
    
    # Create merged data.yaml
    create_data_yaml(output_dir, merged_classes, dataset_dirs, class_mappings, 
                    conflict_count, mode="resplit", ratios=ratios, seed=seed)
    
    print(f"\n✅ Dataset successfully merged and re-split at: {output_dir}")
    return True

def merge_datasets_preserve_splits(dataset_dirs, output_dir):
    """Original merge function that preserves existing splits."""
    
    # Validate datasets first
    if not validate_datasets(dataset_dirs):
        print("\n❌ Dataset validation failed. Please fix the issues and try again.")
        return False
    
    # Merge class names and get mappings
    merged_classes, class_mappings = merge_class_names(dataset_dirs)
    
    # Initialize tracking variables
    split_sizes = defaultdict(int)
    file_tracking = defaultdict(set)
    conflict_count = 0
    
    print("\n📂 Creating output directories...")
    # Create output directories
    for split in ["train", "valid", "test"]:
        images_path = os.path.join(output_dir, split, "images")
        labels_path = os.path.join(output_dir, split, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

    print("\n🔀 Merging datasets (preserving original splits)...")
    
    # Count total files for progress bar
    total_operations = 0
    for dataset_dir in dataset_dirs:
        for split in ["train", "valid", "test"]:
            images_src = os.path.join(dataset_dir, split, "images")
            if os.path.exists(images_src):
                total_operations += len([f for f in os.listdir(images_src) 
                                       if os.path.isfile(os.path.join(images_src, f))]) * 2  # x2 for images and labels
    
    with tqdm(total=total_operations, desc="Copying files", unit="files",
              bar_format='{l_bar}{bar:30}{r_bar}', colour='magenta') as main_pbar:
        
        for dataset_idx, dataset_dir in enumerate(dataset_dirs):
            dataset_name = Path(dataset_dir).name
            main_pbar.set_description(f"Dataset {dataset_name}")
            
            # Determine hierarchy
            branches = ["train", "valid", "test"]
            is_flat = os.path.exists(os.path.join(dataset_dir, "images")) and not os.path.exists(os.path.join(dataset_dir, "train"))
            if is_flat:
                 branches = ["flat"]

            for split in branches:
                if split == "flat":
                     images_src = os.path.join(dataset_dir, "images")
                     labels_src = os.path.join(dataset_dir, "labels")
                     # Map flat to train for preservation mode?
                     # Or maybe we need a dedicated "train" target anyway
                     # Let's map flat -> train by default
                     images_dst = os.path.join(output_dir, "train", "images")
                     labels_dst = os.path.join(output_dir, "train", "labels")
                     # Tracking key uses "train" effectively
                     split_key = "train"
                else:
                     images_src = os.path.join(dataset_dir, split, "images")
                     labels_src = os.path.join(dataset_dir, split, "labels")
                     images_dst = os.path.join(output_dir, split, "images")
                     labels_dst = os.path.join(output_dir, split, "labels")
                     split_key = split

                if os.path.exists(images_src) and os.path.exists(labels_src):
                    # Get list of files
                    image_files = [f for f in os.listdir(images_src) 
                                 if os.path.isfile(os.path.join(images_src, f))]
                    label_files = [f for f in os.listdir(labels_src) 
                                 if os.path.isfile(os.path.join(labels_src, f))]
                    
                    # Copy images with conflict handling
                    for img_file in image_files:
                        src_path = os.path.join(images_src, img_file)
                        
                        # Check for filename conflict
                        if img_file in file_tracking[f"{split_key}_images"]:
                            # Generate unique filename
                            name, ext = os.path.splitext(img_file)
                            suffix = generate_unique_suffix(dataset_idx, src_path)
                            new_img_file = f"{name}{suffix}{ext}"
                            dst_path = os.path.join(images_dst, new_img_file)
                            conflict_count += 1
                        else:
                            dst_path = os.path.join(images_dst, img_file)
                            file_tracking[f"{split_key}_images"].add(img_file)
                        
                        shutil.copy2(src_path, dst_path)
                        # Update size for correct split key
                        split_sizes[split_key] += 1
                        main_pbar.update(1)

                    # Copy and remap labels with conflict handling
                    for label_file in label_files:
                        src_path = os.path.join(labels_src, label_file)
                        
                        # Check for filename conflict
                        if label_file in file_tracking[f"{split_key}_labels"]:
                            # Generate unique filename (matching the image rename)
                            name, ext = os.path.splitext(label_file)
                            suffix = generate_unique_suffix(dataset_idx, src_path)
                            new_label_file = f"{name}{suffix}{ext}"
                            dst_path = os.path.join(labels_dst, new_label_file)
                        else:
                            dst_path = os.path.join(labels_dst, label_file)
                            file_tracking[f"{split_key}_labels"].add(label_file)
                        
                        # Remap class indices if needed
                        if dataset_idx in class_mappings and class_mappings[dataset_idx]:
                            remap_labels(src_path, class_mappings[dataset_idx], dst_path)
                        else:
                            shutil.copy2(src_path, dst_path)
                        main_pbar.update(1)

    # Print summary statistics
    print("\n" + "="*50)
    print("MERGE SUMMARY (Preserved Splits)")
    print("="*50)
    print(f"Datasets merged: {len(dataset_dirs)}")
    print(f"Filename conflicts resolved: {conflict_count}")
    print(f"Total unique classes: {len(merged_classes)}")
    print("\nSplit sizes:")
    for split, size in split_sizes.items():
        print(f"  {split}: {size} images")
    
    # Create merged data.yaml
    create_data_yaml(output_dir, merged_classes, dataset_dirs, class_mappings, 
                    conflict_count, mode="preserve")
    
    print(f"\n✅ Merged dataset successfully created at: {output_dir}")
    return True

def create_data_yaml(output_dir, merged_classes, dataset_dirs, class_mappings, 
                    conflict_count, mode="preserve", ratios=None, seed=None):
    """Create the data.yaml file for the merged dataset."""
    
    print("\n📝 Creating merged data.yaml file...")
    
    # Get first dataset's yaml as template for other fields
    yaml_path = os.path.join(dataset_dirs[0], "data.yaml")
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    
    # Update with merged information - using list format for names
    data_yaml['names'] = merged_classes  # This is now a list
    data_yaml['nc'] = len(merged_classes)
    
    # Update paths to be relative
    data_yaml['train'] = './train/images'
    data_yaml['val'] = './valid/images'
    data_yaml['test'] = './test/images'
    
    # Add merge metadata
    merge_info = {
        'source_datasets': [str(Path(d).name) for d in dataset_dirs],
        'total_datasets': len(dataset_dirs),
        'conflicts_resolved': conflict_count,
        'merge_mode': mode,
        'class_mappings': {f"dataset_{i}": mapping for i, mapping in class_mappings.items()}
    }
    
    if mode == "resplit":
        merge_info['split_ratios'] = {
            'train': ratios[0],
            'valid': ratios[1],
            'test': ratios[2]
        }
        merge_info['random_seed'] = seed
    
    data_yaml['merge_info'] = merge_info
    
    output_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(output_yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, default_flow_style=False, sort_keys=False)

    # Print class names with a nice format
    print("\n🏷️  Merged class names:")
    for idx, name in enumerate(merged_classes):
        print(f"   {idx}: {name}")

    print(f"\n📄 Configuration saved to: {output_yaml_path}")

def validate_ratios(ratios):
    """Validate that ratios sum to 1.0 and are positive."""
    if len(ratios) != 3:
        raise ValueError("Exactly 3 ratios required (train, valid, test)")
    
    if any(r < 0 for r in ratios):
        raise ValueError("All ratios must be non-negative")
    
    total = sum(ratios)
    if abs(total - 1.0) > 0.001:  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    # Check for required packages
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    if missing:
        print("❌ Missing required dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    # Check for optional packages
    if not TQDM_AVAILABLE:
        print("💡 Tip: Install tqdm for better progress bars: pip install tqdm\n")

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO datasets with options to preserve or re-split data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Preserve original splits (default):
    python merge_datasets.py dataset1/ dataset2/ merged_output/
  
  Re-split with default ratios (80/10/10):
    python merge_datasets.py dataset1/ dataset2/ merged_output/ --resplit
  
  Re-split with custom ratios:
    python merge_datasets.py dataset1/ dataset2/ merged_output/ --resplit --ratios 0.7 0.2 0.1
  
  Re-split with custom seed for reproducibility:
    python merge_datasets.py data/cars/ data/trucks/ output/ --resplit --seed 123
        """
    )
    parser.add_argument("datasets", nargs='+', 
                       help="Paths to dataset directories followed by output directory")
    
    parser.add_argument("--resplit", action="store_true",
                       help="Merge all data and create new random splits instead of preserving original splits")
    
    parser.add_argument("--ratios", nargs=3, type=float, metavar=('TRAIN', 'VALID', 'TEST'),
                       default=[0.8, 0.1, 0.1],
                       help="Split ratios for train, valid, and test sets (default: 0.8 0.1 0.1)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Parse input and output directories
    if len(args.datasets) < 2:
        parser.error("At least one input dataset and one output directory are required")
    
    dataset_dirs = args.datasets[:-1]
    output_dir = args.datasets[-1]
    
    # Validate ratios if resplit is enabled
    if args.resplit:
        try:
            validate_ratios(args.ratios)
        except ValueError as e:
            parser.error(f"Invalid ratios: {e}")
    
    # Check if user might have made a mistake with arguments
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        print("❌ Error: Output path exists but is not a directory!")
        return
    
    # Confirm with user if output directory already exists
    if os.path.exists(output_dir) and os.listdir(output_dir):
        response = input(f"⚠️  Output directory '{output_dir}' already exists and is not empty. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Merge cancelled.")
            return
    
    print(f"\n{'='*50}")
    print(f"🚀 YOLO DATASET MERGER")
    print(f"{'='*50}")
    print(f"📥 Input datasets: {', '.join([Path(d).name for d in dataset_dirs])}")
    print(f"📤 Output directory: {output_dir}")
    
    if args.resplit:
        print(f"🎯 Mode: Re-split with ratios {args.ratios[0]:.0%}/{args.ratios[1]:.0%}/{args.ratios[2]:.0%}")
        print(f"🎲 Random seed: {args.seed}")
        success = merge_datasets_with_resplit(dataset_dirs, output_dir, args.ratios, args.seed)
    else:
        print(f"🎯 Mode: Preserve original splits")
        success = merge_datasets_preserve_splits(dataset_dirs, output_dir)
    
    if not success:
        print("\n❌ Merge process failed!")
        exit(1)
    else:
        print(f"\n{'='*50}")
        print(f"✨ MERGE COMPLETE!")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()