#!/usr/bin/env python3
"""
YOLO Dataset Random Sampler

This script randomly samples images from a YOLO format dataset while maintaining 
the original folder structure. It supports sampling by number of images or percentage
and preserves corresponding label files and data.yaml configuration.

Features:
- Random sampling by count or percentage
- Maintains train/valid/test split structure
- Copies corresponding label files
- Preserves data.yaml configuration
- Command line interface with argument validation

Requirements:
- Dataset must have at least 'train' and 'valid' folders
- Each split folder must contain 'images' and 'labels' subfolders
- data.yaml file must be present in the dataset root

Author: Assistant
Version: 1.0

Usage Examples:
    # Sample 1000 random images from each split
    python sample_yolo_dataset.py /path/to/dataset --count 1000
    
    # Sample 15% of images from each split
    python sample_yolo_dataset.py /path/to/dataset --percentage 15
    
    # Sample with custom output directory
    python sample_yolo_dataset.py /path/to/dataset --count 500 --output_dir /path/to/output
    
    # Sample 20% with verbose output
    python sample_yolo_dataset.py /path/to/dataset --percentage 20 --verbose
"""

import os
import argparse
import random
import shutil
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable, Union

class DatasetSampler:
    def __init__(self, dataset_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, verbose: bool = False):
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        self.structure_type = self._detect_structure()
        
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.dataset_path.parent / f"{self.dataset_path.name}_sampled"

    def _detect_structure(self) -> str:
        """Detect if dataset is 'split' (train/val folders) or 'flat' (images/labels folders or root)."""
        if (self.dataset_path / 'train').exists() and (self.dataset_path / 'valid').exists():
            return 'split'
        # Check for images folder or flat root (files directly in root)
        # Note: A flat dataset usually has 'images' and 'labels' folders.
        if (self.dataset_path / 'images').exists():
            return 'flat'
        
        # If deeply flat (files in root), we treat as flat but need ot be careful. 
        # For safety, we assume standard YOLO flat format: images/ and labels/ dirs.
        # If not present, we can look for image files in root.
        has_images = any(self.dataset_path.glob('*.jpg')) or any(self.dataset_path.glob('*.png'))
        if has_images:
            return 'flat_root'
            
        raise ValueError(f"Unknown dataset structure at {self.dataset_path}. Expected 'train/val' splits or 'images' folder.")

    def _get_pairs(self, directory: Path) -> List[Tuple[Path, Path]]:
        """Find image-label pairs in a directory (handling flat 'images'/'labels' or simple root)."""
        images_dir = directory / 'images'
        labels_dir = directory / 'labels'
        
        # fallback for flat_root
        if not images_dir.exists() and self.structure_type == 'flat_root':
            images_dir = directory
            labels_dir = directory

        if not images_dir.exists():
            return []

        pairs = []
        # Support common extensions
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        
        # Scan images
        files = list(images_dir.iterdir())
        
        for img_file in files:
            if img_file.suffix.lower() in exts:
                # Expect label in labels_dir with same stem
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    pairs.append((img_file, label_file))
                
        return pairs

    def _scan_classes(self, pairs: List[Tuple[Path, Path]]) -> Dict[int, List[Tuple[Path, Path]]]:
        """Scan all pairs and map class_id -> list of pairs containing that class."""
        class_map = {}
        
        for img_path, label_path in pairs:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Get unique classes in this image
                classes_in_img = set()
                for line in lines:
                    if not line.strip(): continue
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        classes_in_img.add(cls_id)
                
                for cls_id in classes_in_img:
                    if cls_id not in class_map:
                        class_map[cls_id] = []
                    class_map[cls_id].append((img_path, label_path))
            except Exception as e:
                if self.verbose:
                    print(f"Error reading label {label_path}: {e}")
                    
        return class_map

    def sample(self, 
               global_percentage: Optional[float] = None, 
               class_percentages: Optional[Dict[int, float]] = None,
               count: Optional[int] = None,
               seed: int = 42,
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, int]:
        
        random.seed(seed)
        
        splits = ['train', 'valid', 'test'] if self.structure_type == 'split' else ['.']
        if self.structure_type == 'flat': splits = ['.'] # treat as root relative
        if self.structure_type == 'flat_root': splits = ['.']

        total_files = 0
        # First pass to count total for progress
        # (Approximation, real scanning takes time)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy config if exists
        config_src = self.dataset_path / 'data.yaml'
        if config_src.exists():
            shutil.copy2(config_src, self.output_path / 'data.yaml')
        
        summary = {"total_original": 0, "total_sampled": 0}

        for split_name in splits:
            if progress_callback: progress_callback(0, 0, f"Scanning {split_name}...")
            
            src_dir = self.dataset_path / split_name if split_name != '.' else self.dataset_path
            
            # For output, we maintain structure
            if split_name != '.':
                dest_dir = self.output_path / split_name
            else:
                dest_dir = self.output_path

            pairs = self._get_pairs(src_dir)
            summary["total_original"] += len(pairs)
            
            if not pairs:
                continue

            selected_pairs = set()

            # Strategy 1: Global Count
            if count is not None:
                k = min(count, len(pairs))
                selected_pairs = set(random.sample(pairs, k))

            # Strategy 2: Global Percentage
            elif global_percentage is not None:
                k = int(len(pairs) * (global_percentage / 100.0))
                selected_pairs = set(random.sample(pairs, k))

            # Strategy 3: Per-Class Percentage
            elif class_percentages is not None:
                # Scan classes
                if progress_callback: progress_callback(0, len(pairs), f"Analyzing classes in {split_name}...")
                class_map = self._scan_classes(pairs)
                
                # Union strategy: For each class, sample X%. Add to set.
                for cls_id, cls_pairs in class_map.items():
                    pct = class_percentages.get(cls_id, 0) # Default to 0? Or 100? Assuming 0 if not specified implies "don't care about this class", but usually means 0.
                    # Wait, if user specifies some classes, others should be 0? 
                    # Let's assume user provides map for ALL classes they want. 
                    if pct > 0:
                        k = int(len(cls_pairs) * (pct / 100.0))
                        if k > 0:
                            selected_pairs.update(random.sample(cls_pairs, k))
            
            # Copy Files
            if split_name != '.':
                (dest_dir / 'images').mkdir(parents=True, exist_ok=True)
                (dest_dir / 'labels').mkdir(parents=True, exist_ok=True)
            else:
                # Flat structure
                if (src_dir / 'images').exists():
                    (dest_dir / 'images').mkdir(parents=True, exist_ok=True)
                    (dest_dir / 'labels').mkdir(parents=True, exist_ok=True)
                else:
                    # Flat root output - messy but preserves input style
                    dest_dir.mkdir(parents=True, exist_ok=True)

            total_to_copy = len(selected_pairs)
            summary["total_sampled"] += total_to_copy
            
            processed = 0
            for img_path, label_path in selected_pairs:
                processed += 1
                if progress_callback and processed % 10 == 0:
                    progress_callback(processed, total_to_copy, f"Copying {split_name}: {processed}/{total_to_copy}")

                # Determine relative structure for copy target
                # If structure is standard (images/labels), copy to those folders
                if (src_dir / 'images').exists():
                   shutil.copy2(img_path, dest_dir / 'images' / img_path.name)
                   shutil.copy2(label_path, dest_dir / 'labels' / label_path.name)
                else:
                   # Flat root
                   shutil.copy2(img_path, dest_dir / img_path.name)
                   shutil.copy2(label_path, dest_dir / label_path.name)

        return summary

def main():
    parser = argparse.ArgumentParser(description="YOLO Dataset Sampler")
    parser.add_argument('dataset_path', help="Input dataset path")
    parser.add_argument('--output_dir', help="Output directory")
    parser.add_argument('--count', type=int, help="Fixed number of images")
    parser.add_argument('--percentage', type=float, help="Global percentage (0-100)")
    parser.add_argument('--class_percentages', type=str, help="JSON string or 'id:pct,id:pct' for class percentages")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Parse class percentages if string
    class_pcts = None
    if args.class_percentages:
        import json
        try:
            class_pcts = json.loads(args.class_percentages)
            # Ensure keys are ints
            class_pcts = {int(k): float(v) for k,v in class_pcts.items()}
        except:
             # Try simple format 0:50,1:20
             class_pcts = {}
             for part in args.class_percentages.split(','):
                 k,v = part.split(':')
                 class_pcts[int(k)] = float(v)

    sampler = DatasetSampler(args.dataset_path, args.output_dir, args.verbose)
    
    def console_progress(current, total, msg):
        if total > 0:
            print(f"\r{msg} [{current}/{total}]", end='')
        else:
            print(f"\r{msg}", end='')

    summary = sampler.sample(
        global_percentage=args.percentage,
        class_percentages=class_pcts,
        count=args.count,
        seed=args.seed,
        progress_callback=console_progress if args.verbose else None
    )
    print("\nDone.")
    print(f"Original: {summary['total_original']}")
    print(f"Sampled: {summary['total_sampled']}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    exit(main())