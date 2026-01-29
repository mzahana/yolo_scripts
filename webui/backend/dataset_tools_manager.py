
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add scripts to path to import split_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

from split_dataset import YOLODatasetSplitter

class DatasetToolsManager:
    @staticmethod
    def get_split_stats(dataset_path: str) -> Dict[str, Dict[str, float]]:
        """
        Analyze the dataset to get current split statistics.
        Returns counts and percentages for train, val, test.
        """
        path = Path(dataset_path)
        stats = {
            'total': 0,
            'splits': {}
        }

        if not path.exists():
            return stats

        # Check for YOLO format splits
        for split in ['train', 'valid', 'test', 'val']:
            # Handle 'val' vs 'valid' normalization
            display_name = 'val' if split == 'valid' else split
            
            split_dir = path / split / 'images'
            if split_dir.exists():
                count = len([f for f in split_dir.glob('*') if f.is_file() and f.suffix.lower() in YOLODatasetSplitter.SUPPORTED_IMAGE_FORMATS])
                stats['splits'][display_name] = count
                stats['total'] += count
            elif (path / split).exists() and not (path / split / 'images').exists():
                 # Maybe flat split folder?
                 count = len([f for f in (path / split).glob('*') if f.is_file() and f.suffix.lower() in YOLODatasetSplitter.SUPPORTED_IMAGE_FORMATS])
                 if count > 0:
                     stats['splits'][display_name] = count
                     stats['total'] += count

        # If no splits found, maybe it's just 'images' folder?
        if stats['total'] == 0 and (path / 'images').exists():
            count = len([f for f in (path / 'images').glob('*') if f.is_file() and f.suffix.lower() in YOLODatasetSplitter.SUPPORTED_IMAGE_FORMATS])
            stats['splits']['all'] = count
            stats['total'] = count

        # Calculate percentages
        if stats['total'] > 0:
            for k in stats['splits']:
                stats['splits'][k] = {
                    'count': stats['splits'][k],
                    'percentage': round((stats['splits'][k] / stats['total']) * 100, 1)
                }

        return stats

    @staticmethod
    def rebalance_dataset(dataset_path: str, train_pct: float, val_pct: float, test_pct: float, delete_original: bool = False) -> Dict:
        """
        Rebalance the dataset splits.
        """
        input_path = Path(dataset_path).resolve()
        
        # Determine output path
        # If we are effectively replacing, we use a temp folder first
        output_path = input_path.parent / f"{input_path.name}_rebalanced"
        
        # Ensure output doesn't exist to avoid mixing
        if output_path.exists():
            shutil.rmtree(output_path)

        try:
            # 1. Run Splitter
            # The splitter handles reading from existing splits if they exist
            splitter = YOLODatasetSplitter(
                parent_dir=input_path,
                train_pct=train_pct,
                valid_pct=val_pct,
                test_pct=test_pct,
                output_dir=output_path,
                verbose=False
            )
            splitter.run(generate_yaml=True)
            
            # 2. Handle Delete Original (Move replacement)
            if delete_original:
                # Backup original just in case? Or just delete as requested.
                # To be safe, let's rename original to _trash, move new to original, then delete _trash
                trash_path = input_path.parent / f"{input_path.name}_trash_{random.randint(1000, 9999)}"
                
                print(f"Renaming original {input_path} to {trash_path}")
                os.rename(input_path, trash_path)
                
                print(f"Renaming new {output_path} to {input_path}")
                os.rename(output_path, input_path)
                
                print(f"Removing trash {trash_path}")
                shutil.rmtree(trash_path)
                
                final_path = input_path
            else:
                final_path = output_path

            return {
                "status": "success",
                "message": "Dataset rebalanced successfully",
                "new_path": str(final_path),
                "stats": DatasetToolsManager.get_split_stats(str(final_path))
            }

        except Exception as e:
            # Cleanup output if failed
            if output_path.exists() and output_path != input_path:
                shutil.rmtree(output_path)
            raise e

    @staticmethod
    def flatten_dataset(dataset_path: str) -> Dict:
        """
        Convert a split dataset into a flat dataset (images/ and labels/ folders).
        """
        input_path = Path(dataset_path).resolve()
        output_path = input_path.parent / f"{input_path.name}_flat"
        
        if output_path.exists():
            shutil.rmtree(output_path)
            
        images_out = output_path / 'images'
        labels_out = output_path / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        counts = {'images': 0, 'labels': 0}
        
        # Traverse splits
        for split in ['train', 'valid', 'test', 'val']:
            split_dir = input_path / split
            if not split_dir.exists():
                continue
                
            # Handle images
            if (split_dir / 'images').exists():
                for img_file in (split_dir / 'images').glob('*'):
                    if img_file.is_file() and img_file.suffix.lower() in YOLODatasetSplitter.SUPPORTED_IMAGE_FORMATS:
                        # Handle collision? Overwrite seems acceptable for flattening same dataset
                        shutil.copy2(img_file, images_out)
                        counts['images'] += 1
                        
            # Handle labels
            # Standard YOLO: labels in split/labels
            if (split_dir / 'labels').exists():
                for lbl_file in (split_dir / 'labels').glob('*.txt'):
                    shutil.copy2(lbl_file, labels_out)
                    counts['labels'] += 1
            elif (input_path / 'labels' / split).exists():
                 # Sometimes labels are in root/labels/train
                 for lbl_file in (input_path / 'labels' / split).glob('*.txt'):
                    shutil.copy2(lbl_file, labels_out)
                    counts['labels'] += 1

        # Generate data.yaml
        # Reuse logic from YOLODatasetSplitter to read source yaml
        # We can temporarily instantiate a splitter just to use its helper if we made it public?
        # Or just reimplement simple yaml reading here.
        
        source_yaml = input_path / 'data.yaml'
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images',
            'val': 'images', # Flat dataset technically doesn't have splits, but we point to images
            'test': '',
        }
        
        if source_yaml.exists():
            try:
                import yaml
                with open(source_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'nc' in data: yaml_content['nc'] = data['nc']
                    if 'names' in data: yaml_content['names'] = data['names']
            except Exception as e:
                print(f"Warning: Could not read source yaml: {e}")
        
        # Write yaml
        yaml_out = output_path / 'data.yaml'
        try:
            import yaml
            with open(yaml_out, 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False)
        except ImportError:
            with open(yaml_out, 'w') as f:
                for k, v in yaml_content.items():
                    f.write(f"{k}: {v}\n")
                    
        return {
            "status": "success",
            "message": "Dataset flattened successfully",
            "new_path": str(output_path),
            "stats": counts
        }
