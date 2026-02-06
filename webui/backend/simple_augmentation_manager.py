
import os
import cv2
import base64
import random
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

# Import the augmentation logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
from simple_augmentation import SimpleAugmentor

class SimpleAugmentationManager:
    
    _cached_sample: Optional[Dict] = None

    @staticmethod
    def get_random_sample(dataset_path: str):
        """
        Picks a random image and its label from the dataset.
        """
        p = Path(dataset_path)
        
        # Simple discovery logic matching script's logic
        # We need to find an image.
        # Check standard dirs
        search_paths = [
            p / 'images', 
            p / 'train' / 'images', 
            p / 'valid' / 'images', 
            p # Flat
        ]
        
        valid_extensions = {'.jpg', '.png', '.jpeg', '.bmp', '.webp'}
        
        found_images = []
        for sp in search_paths:
            if sp.exists():
                for f in os.listdir(sp):
                    if os.path.splitext(f)[1].lower() in valid_extensions:
                        found_images.append(os.path.join(sp, f))
                if found_images: break # Stop if we found a dir with images
                
        if not found_images:
            raise ValueError(f"No images found in {dataset_path}")
            
        selected_image = random.choice(found_images)
        
        # Find label
        # Logic: try to substitute 'images' with 'labels' in path, or same dir with .txt
        img_p = Path(selected_image)
        label_path = None
        
        parts = list(img_p.parts)
        if 'images' in parts:
             idx = len(parts) - 1 - parts[::-1].index('images')
             parts[idx] = 'labels'
             potential = Path(*parts).with_suffix('.txt')
             if potential.exists():
                 label_path = str(potential)
                 
        if not label_path:
            potential = img_p.with_suffix('.txt')
            if potential.exists():
                label_path = str(potential)
                
        return selected_image, label_path

    @staticmethod
    def generate_preview(dataset_path: str, config: Dict) -> Dict:
        """
        Generates a preview of the augmentation.
        Uses cached sample if available, or picks new one.
        """
        augmentor = SimpleAugmentor()
        
        image_path = None
        label_path = None
        
        # Use cached sample if valid? 
        # For simplicity, let's just pick one or maybe allow user to request "New Sample"?
        # Usually user wants to see effect on same image.
        if SimpleAugmentationManager._cached_sample and SimpleAugmentationManager._cached_sample['dataset'] == dataset_path:
             image_path = SimpleAugmentationManager._cached_sample['image']
             label_path = SimpleAugmentationManager._cached_sample['label']
        else:
             image_path, label_path = SimpleAugmentationManager.get_random_sample(dataset_path)
             SimpleAugmentationManager._cached_sample = {
                 'dataset': dataset_path,
                 'image': image_path,
                 'label': label_path
             }
             
        # If no label found, we pass None to augmentor? Augmentor handles empty list but expects a path.
        # Actually augmentor.read_yolo_label requires path.
        # If label_path is None, we need to handle it.
        # The script `read_yolo_label` checks if path exists. So if None passed, it might crash or we pass dummy.
        # Let's verify script. `if not os.path.exists(label_path): return [], []`
        # So passing a non-existent path is fine.
        
        if label_path is None:
             label_path = "dummy_non_existent.txt"
             
        aug_image = augmentor.generate_preview(image_path, label_path, config)
        
        if aug_image is None:
             raise ValueError("Failed to generate preview")
             
        _, buffer = cv2.imencode('.jpg', aug_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Also return original for comparison?
        # Frontend can fetch original? No, better to return it here to ensure sync.
        
        orig_img = cv2.imread(image_path)
        _, buffer_orig = cv2.imencode('.jpg', orig_img)
        orig_base64 = base64.b64encode(buffer_orig).decode('utf-8')
        
        return {
            "augmented_image": f"data:image/jpeg;base64,{img_base64}",
            "original_image": f"data:image/jpeg;base64,{orig_base64}",
            "sample_name": os.path.basename(image_path)
        }

    @staticmethod
    def scan_dataset(dataset_path: str) -> Dict:
        """Wrapper for scan_dataset"""
        augmentor = SimpleAugmentor()
        return augmentor.scan_dataset(dataset_path)

    @staticmethod
    def run_augmentation_job(
        dataset_path: str,
        output_name: str,
        multiplier: int,
        config: Dict,
        progress_tracker: Dict,
        selected_splits: Optional[list] = None
    ):
        """
        Background task wrapper.
        """
        print(f"DEBUG: Starting Simple Augmentation Job for {dataset_path} with splits {selected_splits}")
        augmentor = SimpleAugmentor()
        
        def update_progress(current, total, msg=""):
            progress_tracker["status"] = "running"
            progress_tracker["current"] = current
            progress_tracker["total"] = total
            progress_tracker["message"] = msg
            
        try:
            output_dir = augmentor.process_dataset(
                dataset_path, 
                output_name, 
                multiplier, 
                config, 
                selected_splits=selected_splits,
                progress_callback=update_progress
            )
            
            progress_tracker["status"] = "idle"
            progress_tracker["total"] = progress_tracker["current"] # Ensure 100%
            progress_tracker["message"] = f"Finished! Output saved to {output_dir}"
            progress_tracker["result"] = {"output_dir": output_dir}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            progress_tracker["status"] = "error"
            progress_tracker["message"] = str(e)
