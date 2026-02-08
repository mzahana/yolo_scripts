#!/usr/bin/env python3
"""
Dataset Splitter for Images and Labels - YOLO Compatible CLI (with output dir control + progress bars)

Changes vs. original:
- New optional --output argument for the destination root directory.
- Default output directory is <input_dir_name>_splits (sibling of the input).
- Non-destructive: copies files; does not modify the source dataset.
- Simple in-terminal progress bars per split (no external deps).

Usage:
  python split_dataset.py /path/to/dataset
  python split_dataset.py /path/to/dataset --output /path/to/my_splits
  python split_dataset.py /path/to/dataset --train 0.8 --valid 0.15 --test 0.05
  python split_dataset.py /path/to/dataset --seed 42 --yaml --verbose
"""

import os
import shutil
from pathlib import Path
import random
import argparse
import json
from typing import List, Tuple, Dict
import sys
import time


def _print_progress(prefix: str, current: int, total: int, bar_len: int = 30):
    """Render a simple progress bar to stdout."""
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    filled = int(bar_len * ratio)
    bar = "█" * filled + "░" * (bar_len - filled)
    percent = int(ratio * 100)
    sys.stdout.write(f"\r{prefix} [{bar}] {percent:3d}% ({current}/{total})")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


class YOLODatasetSplitter:
    """Class to handle YOLO dataset splitting operations."""

    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    def __init__(self, parent_dir: Path, train_pct: float = 0.7,
                 valid_pct: float = 0.2, test_pct: float = 0.1,
                 seed: int = None, verbose: bool = False,
                 output_dir: Path | None = None):
        """
        Initialize the dataset splitter.

        Args:
            parent_dir: Path to parent directory containing 'images' and 'labels'
            train_pct: Percentage for training set
            valid_pct: Percentage for validation set
            test_pct: Percentage for test set
            seed: Random seed for reproducible splits
            verbose: Whether to print detailed information
            output_dir: Optional explicit root directory for the split dataset.
                        If None, defaults to sibling '<input_dir_name>_splits'.
        """
        self.parent_dir = Path(parent_dir).resolve()
        self.train_pct = train_pct
        self.valid_pct = valid_pct
        self.test_pct = test_pct
        self.seed = seed
        self.verbose = verbose

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Validate percentages
        total = train_pct + valid_pct + test_pct
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split percentages must sum to 1.0, got {total:.3f}")

        # Setup input directories
        self.images_dir = self.parent_dir / "images"
        self.labels_dir = self.parent_dir / "labels"

        # Setup output root
        if output_dir is None:
            # Default: create sibling named "<input_dir_name>_splits"
            base_name = self.parent_dir.name + "_splits"
            self.output_root = self.parent_dir.parent / base_name
        else:
            self.output_root = Path(output_dir).resolve()

        # Check if input is already split
        self.is_already_split = False
        if (self.parent_dir / "train").exists() or (self.parent_dir / "valid").exists():
            self.is_already_split = True
            print(f"Detected split dataset structure in {self.parent_dir}")
        else:
            self.images_dir = self.parent_dir / "images"
            self.labels_dir = self.parent_dir / "labels"

        # Statistics tracking
        self.stats = {
            'train': {'images': 0, 'labels': 0, 'missing_labels': []},
            'valid': {'images': 0, 'labels': 0, 'missing_labels': []},
            'test':  {'images': 0, 'labels': 0, 'missing_labels': []}
        }

    def validate_directories(self):
        """Validate that required input directories exist."""
        if self.is_already_split:
            # At least one split should exist
            splits_found = False
            for split in ['train', 'valid', 'test', 'val']:
                if (self.parent_dir / split / 'images').exists():
                    splits_found = True
                    break
            if not splits_found:
                raise FileNotFoundError(f"No valid split subdirectories (train/valid/test) found in {self.parent_dir}")
        else:
            if not self.images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            if not self.labels_dir.exists():
                raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

    def create_output_directories(self) -> Dict[str, Dict[str, Path]]:
        """Create output directory structure for train/valid/test splits."""
        dirs = {}
        for split in ['train', 'valid', 'test']:
            dirs[split] = {
                'images': self.output_root / split / 'images',
                'labels': self.output_root / split / 'labels'
            }
            dirs[split]['images'].mkdir(parents=True, exist_ok=True)
            dirs[split]['labels'].mkdir(parents=True, exist_ok=True)
        return dirs

    def get_image_files(self) -> List[Path]:
        """Get all image files from the images directory."""
        image_files = []
        if self.is_already_split:
            for split in ['train', 'valid', 'test', 'val']:
                split_img_dir = self.parent_dir / split / 'images'
                if split_img_dir.exists():
                    for ext in self.SUPPORTED_IMAGE_FORMATS:
                        image_files.extend(split_img_dir.glob(f'*{ext}'))
                        image_files.extend(split_img_dir.glob(f'*{ext.upper()}'))
        else:
            for ext in self.SUPPORTED_IMAGE_FORMATS:
                image_files.extend(self.images_dir.glob(f'*{ext}'))
                image_files.extend(self.images_dir.glob(f'*{ext.upper()}'))

        if not image_files:
            raise ValueError(f"No image files found in {self.parent_dir}")

        return sorted(image_files)  # Sort for consistency

    def split_files(self, files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split files into train, valid, and test sets."""
        files_copy = files.copy()
        random.shuffle(files_copy)

        n = len(files_copy)
        train_idx = int(n * self.train_pct)
        valid_idx = train_idx + int(n * self.valid_pct)

        train_files = files_copy[:train_idx]
        valid_files = files_copy[train_idx:valid_idx]
        test_files  = files_copy[valid_idx:]

        return train_files, valid_files, test_files

    def _copy_split_with_progress(self, files: List[Path], split_name: str,
                                  target_dirs: Dict[str, Path]) -> int:
        """
        Copy image and label files to target directories with a progress bar.

        Returns:
            Number of successfully copied label files.
        """
        labels_copied = 0
        missing_labels = []

        total = len(files)
        _print_progress(f"Copying {split_name:<5}", 0, total)
        for i, img_file in enumerate(files, start=1):
            # Copy image
            shutil.copy2(img_file, target_dirs['images'])
            self.stats[split_name]['images'] += 1

            # Find and copy corresponding label file
            label_name = img_file.stem + '.txt'
            
            if self.is_already_split:
                # If already split, label is likely in sibling "labels" dir of the image's parent
                # images parent is <split>/images, so we want <split>/labels
                # img_file.parent = .../split/images
                # img_file.parent.parent = .../split
                label_file = img_file.parent.parent / "labels" / label_name
            else:
                label_file = self.labels_dir / label_name

            if label_file.exists():
                shutil.copy2(label_file, target_dirs['labels'])
                labels_copied += 1
                self.stats[split_name]['labels'] += 1
            else:
                missing_labels.append(img_file.name)
                self.stats[split_name]['missing_labels'].append(img_file.name)

            # Update progress
            _print_progress(f"Copying {split_name:<5}", i, total)

        if missing_labels and self.verbose:
            print(f"  Warning: {len(missing_labels)} missing label files in {split_name} set")

        return labels_copied

    def _load_source_yaml_info(self) -> Dict:
        """Attempt to load nc and names from a source data.yaml if it exists."""
        source_yaml = self.parent_dir / 'data.yaml'
        if not source_yaml.exists():
            # Try parent if we are in a split folder structure
            source_yaml = self.parent_dir.parent / 'data.yaml'
        
        if source_yaml.exists():
            try:
                import yaml
                with open(source_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    return {
                        'nc': data.get('nc'),
                        'names': data.get('names')
                    }
            except Exception as e:
                print(f"Warning: Could not load source data.yaml: {e}")
        return {}

    def generate_yaml_config(self, num_classes: int = None):
        """Generate YOLO configuration YAML file rooted at the output directory."""
        yaml_content = {
            'path': str(self.output_root.absolute()),
            'train': 'train/images',
            'val':   'valid/images',
            'test':  'test/images',
        }

        # Priority 1: Use provided num_classes (arg)
        # Priority 2: Use information from source data.yaml
        # Priority 3: Detect from label files
        
        source_info = self._load_source_yaml_info()
        
        final_nc = num_classes
        final_names = None

        if final_nc is None:
             final_nc = source_info.get('nc')
        
        if source_info.get('names'):
            final_names = source_info.get('names')
            # If names provided but nc not, deduce nc
            if final_nc is None:
                final_nc = len(final_names)

        # Fallback detection if still None
        if final_nc is None:
            print("Detecting classes from labels...")
            final_nc = self._detect_num_classes()

        if final_nc:
            yaml_content['nc'] = final_nc
            # If names available from source, use them. Otherwise generate placeholders.
            if final_names:
                # Ensure length matches nc
                if len(final_names) != final_nc:
                    print(f"Warning: Source class names count ({len(final_names)}) does not match nc ({final_nc}).")
                    # Should we resize? Stick to names list usually.
                    # If names < nc, extend.
                    if len(final_names) < final_nc:
                        final_names.extend([f'class_{i}' for i in range(len(final_names), final_nc)])
                    else:
                        final_names = final_names[:final_nc]
                yaml_content['names'] = final_names
            else:
                 yaml_content['names'] = [f'class_{i}' for i in range(final_nc)]

        yaml_path = self.output_root / 'data.yaml'
        try:
            import yaml
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            print(f"\nGenerated YOLO config file: {yaml_path}")
        except ImportError:
            # Fallback
            with open(yaml_path, 'w') as f:
                f.write(f"# YOLO Dataset Configuration\n")
                for k, v in yaml_content.items():
                    f.write(f"{k}: {v}\n")
            print(f"\nGenerated YOLO config file (simple format): {yaml_path}")

    def _detect_num_classes(self) -> int:
        """Detect number of classes from ALL label files."""
        max_class = -1
        # Scan ALL label files instead of just first 100 to ensure we catch rare classes
        label_files = list(self.labels_dir.glob('*.txt'))
        if self.is_already_split:
            for split in ['train', 'valid', 'test', 'val']:
                d = self.parent_dir / split / 'labels'
                if d.exists():
                    label_files.extend(list(d.glob('*.txt')))

        # Remove duplicates if any (though glob paths are unique)
        label_files = list(set(label_files))

        if not label_files:
             return None

        # To avoid being too slow on massive datasets, maybe limit to 1000 or 5000? 
        # But user wants correctness. Let's scan all but be efficient.
        # Check every 10th file if > 1000 files?
        if len(label_files) > 5000:
             label_files = label_files[::5] # 20% sample

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                if class_id > max_class:
                                    max_class = class_id
                            except ValueError:
                                pass
            except Exception:
                continue

        return max_class + 1 if max_class >= 0 else None

    def print_statistics(self):
        """Print detailed statistics about the split."""
        print("\n" + "="*60)
        print("DATASET SPLIT STATISTICS")
        print("="*60)

        total_images = sum(s['images'] for s in self.stats.values())
        total_labels = sum(s['labels'] for s in self.stats.values())

        for split_name in ['train', 'valid', 'test']:
            split_stats = self.stats[split_name]
            pct = getattr(self, f'{split_name}_pct' if split_name != 'valid' else 'valid_pct')

            print(f"\n{split_name.upper()} SET ({pct:.1%}):")
            print(f"  Images: {split_stats['images']:,} files")
            print(f"  Labels: {split_stats['labels']:,} files")

            if split_stats['missing_labels']:
                print(f"  Missing labels: {len(split_stats['missing_labels'])} files")
                if self.verbose and len(split_stats['missing_labels']) <= 10:
                    for fname in split_stats['missing_labels'][:10]:
                        print(f"    - {fname}")

        print(f"\nTOTAL:")
        print(f"  Images: {total_images:,} files")
        print(f"  Labels: {total_labels:,} files")
        if total_images > 0:
            print(f"  Label coverage: {total_labels/total_images:.1%}")

        if self.seed is not None:
            print(f"\nRandom seed: {self.seed}")

        print("="*60)

    def run(self, generate_yaml: bool = False):
        """Execute the dataset splitting process."""
        print(f"Input dataset: {self.parent_dir}")
        print(f"Output root  : {self.output_root}")

        # Validate input
        self.validate_directories()

        # Get image files
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} image files")

        # Create output directories
        output_dirs = self.create_output_directories()

        # Split files
        train_files, valid_files, test_files = self.split_files(image_files)

        # Copy files to respective directories with progress bars
        print("\nCopying files...")
        self._copy_split_with_progress(train_files, 'train', output_dirs['train'])
        self._copy_split_with_progress(valid_files, 'valid', output_dirs['valid'])
        self._copy_split_with_progress(test_files,  'test',  output_dirs['test'])

        # Generate YAML if requested
        if generate_yaml:
            self.generate_yaml_config()

        # Print statistics
        self.print_statistics()

        print("\n✓ Dataset split completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Split image dataset into train/valid/test sets for YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dataset
  %(prog)s /path/to/dataset --output /path/to/my_splits
  %(prog)s /path/to/dataset --train 0.8 --valid 0.15 --test 0.05
  %(prog)s /path/to/dataset --seed 42 --yaml
  %(prog)s /path/to/dataset --train 0.6 --valid 0.2 --test 0.2 --verbose
        """
    )

    parser.add_argument(
        'parent_dir',
        type=str,
        help='Path to parent directory containing "images" and "labels" subdirectories'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=("Destination root directory for the split dataset. "
              "If omitted, a sibling directory named '<input_dir_name>_splits' is used.")
    )

    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        metavar='PCT',
        help='Percentage for training set (default: 0.7)'
    )

    parser.add_argument(
        '--valid',
        type=float,
        default=0.2,
        metavar='PCT',
        help='Percentage for validation set (default: 0.2)'
    )

    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        metavar='PCT',
        help='Percentage for test set (default: 0.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        metavar='INT',
        help='Random seed for reproducible splits'
    )

    parser.add_argument(
        '--yaml',
        action='store_true',
        help='Generate YOLO configuration YAML file (dataset.yaml in the output root)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information during processing'
    )

    args = parser.parse_args()

    # Validate percentages
    for pct_name, pct_value in [('train', args.train), ('valid', args.valid), ('test', args.test)]:
        if not 0.0 < pct_value <= 1.0:
            parser.error(f"{pct_name} percentage must be between 0.0 and 1.0")

    total = args.train + args.valid + args.test
    if abs(total - 1.0) > 0.001:
        parser.error(f"Percentages must sum to 1.0, got {total:.3f}")

    try:
        out = Path(args.output).resolve() if args.output else None
        splitter = YOLODatasetSplitter(
            parent_dir=Path(args.parent_dir),
            train_pct=args.train,
            valid_pct=args.valid,
            test_pct=args.test,
            seed=args.seed,
            verbose=args.verbose,
            output_dir=out
        )
        splitter.run(generate_yaml=args.yaml)
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
