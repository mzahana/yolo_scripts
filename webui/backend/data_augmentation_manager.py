
import os
import shutil
import tempfile
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import List, Optional, Dict
import threading
import random

# Import the augmentation logic
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
from data_augmentation import DataAugmentor

class DataAugmentationManager:
    
    _cached_sample: Optional[Dict] = None

    @staticmethod
    def save_temp_background(file_bytes: bytes, filename: str) -> str:
        """Saves an uploaded background image to a temp location."""
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"yolo_aug_bg_{filename}")
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        return file_path

    @staticmethod
    def get_dataset_stats(dataset_path: str) -> Dict:
        """
        Scans all label files in the dataset to count objects per class.
        Returns: { total_objects: int, class_counts: { class_id: count }, class_percentages: { class_id: pct } }
        """
        p = Path(dataset_path)
        labels_dir = p / 'labels'
        
        if not labels_dir.exists():
            if (p / 'train' / 'labels').exists(): labels_dir = p / 'train' / 'labels'
            elif (p / 'valid' / 'labels').exists(): labels_dir = p / 'valid' / 'labels'
            else: labels_dir = p # Check fallback

        if not labels_dir.exists():
            return {"total_objects": 0, "class_counts": {}, "class_percentages": {}}

        class_counts = {}
        total_objects = 0
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'classes.txt']
        
        # Limit scan for performance if massive? 1000 files is fast enough.
        # But user wants accurate stats. Let's scan all.
        for lf in label_files:
            try:
                with open(os.path.join(labels_dir, lf), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cid = int(parts[0])
                                class_counts[cid] = class_counts.get(cid, 0) + 1
                                total_objects += 1
                            except: pass
            except: pass
            
        percentages = {k: (v / total_objects * 100) for k, v in class_counts.items()} if total_objects > 0 else {}
        
        return {
            "total_objects": total_objects,
            "class_counts": class_counts,
            "class_percentages": percentages
        }

    @staticmethod
    def sample_object(
        dataset_path: str,
        class_ids: List[int],
        background_path: str,
    ) -> Dict:
        """
        Picks a random object, caches it, and returns the 'Original' preview (on background, no augs).
        """
        # Find images/labels logic (reused)
        p = Path(dataset_path)
        images_dir = p / 'images'
        labels_dir = p / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
             if (p / 'train' / 'images').exists() and (p / 'train' / 'labels').exists():
                 images_dir = p / 'train' / 'images'
                 labels_dir = p / 'train' / 'labels'
             elif (p / 'valid' / 'images').exists() and (p / 'valid' / 'labels').exists():
                 images_dir = p / 'valid' / 'images'
                 labels_dir = p / 'valid' / 'labels'
             else:
                 images_dir = p
                 labels_dir = p

        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        np.random.shuffle(label_files)
        
        selected_file = None
        selected_line = None
        
        for lf in label_files[:100]:
            with open(os.path.join(labels_dir, lf), 'r') as f:
                lines = f.readlines()
                valid_lines = [l for l in lines if l.strip() and int(l.split()[0]) in class_ids]
                if valid_lines:
                    selected_file = lf
                    selected_line = random.choice(valid_lines)
                    break
        
        if not selected_file:
            raise ValueError(f"No objects found for classes {class_ids}")

        base_name = os.path.splitext(selected_file)[0]
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            cand = os.path.join(images_dir, base_name + ext)
            if os.path.exists(cand):
                image_path = cand
                break
        
        if not image_path:
            raise ValueError(f"Image not found for label {selected_file}")

        # Cache the sample
        DataAugmentationManager._cached_sample = {
            "image_path": str(image_path),
            "label_line": selected_line
        }

        # Generate 'Original' preview (No augs)
        bg_img = cv2.imread(background_path)
        if bg_img is None: raise ValueError("Background not found")
        
        # Call generate with NO augs (default params)
        preview_img = DataAugmentor.generate_single_preview(
            str(image_path), selected_line, bg_img,
            None, None, None, None, None, 1.0, None
        )
        
        if preview_img is None: raise ValueError("Failed to extract object")

        _, buffer = cv2.imencode('.jpg', preview_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "sample_info": {"file": selected_file, "class_id": int(selected_line.split()[0])}
        }

    @staticmethod
    def apply_preview_to_sample(
        background_path: str,
        rotation_range: Optional[List[float]],
        blur_range: Optional[List[int]],
        scaling_range: Optional[List[float]],
        contrast_range: Optional[List[float]],
        brightness_range: Optional[List[int]],
        region_scale: float,
        roi: Optional[List[float]] = None
    ) -> Dict:
        """
        Applies augmentation to the currently cached sample object.
        """
        if not DataAugmentationManager._cached_sample:
            raise ValueError("No sample object selected. Please draw a sample first.")
            
        sample = DataAugmentationManager._cached_sample
        bg_img = cv2.imread(background_path)
        if bg_img is None: raise ValueError("Background not found")
        
        roi_tuple = tuple(roi) if roi and len(roi) == 4 else None

        preview_img = DataAugmentor.generate_single_preview(
            sample["image_path"], sample["label_line"], bg_img,
            rotation_range, blur_range, scaling_range, contrast_range, brightness_range, region_scale, roi_tuple
        )
        
        if preview_img is None: raise ValueError("Failed to generate preview")

        _, buffer = cv2.imencode('.jpg', preview_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}"
        }

    # Backward compatibility for 'generate_preview' if needed, or remove it.
    # We'll just keep the run_task method.
    
    @staticmethod
    def run_augmentation_task(
        dataset_path: str,
        background_path: str,
        output_path: str, 
        class_ids: List[int],
        num_augmentations: int,
        rotation_range: Optional[List[float]],
        blur_range: Optional[List[int]],
        scaling_range: Optional[List[float]],
        contrast_range: Optional[List[float]],
        brightness_range: Optional[List[int]],
        region_scale: float,
        augment_together: bool,
        progress_tracker: dict,
        roi: Optional[List[float]] = None,
        composition_mode: bool = False,
        total_images: int = 10,
        objects_per_image: int = 3
    ):
        """
        Runs the full augmentation. Designed to be run in a thread.
        """
        print(f"DEBUG: Manager.run_augmentation_task called with composition_mode={composition_mode}, total_images={total_images}")
        def update_progress(current, total):
            progress_tracker["current"] = current
            progress_tracker["total"] = total
            progress_tracker["status"] = "running"
            progress_tracker["message"] = f"Processing {current}/{total} files..."

        try:
            # Copy data.yaml if exists
            os.makedirs(output_path, exist_ok=True)
            src_yaml = os.path.join(dataset_path, 'data.yaml')
            if os.path.exists(src_yaml):
                shutil.copy(src_yaml, os.path.join(output_path, 'data.yaml'))
            
            roi_tuple = tuple(roi) if roi and len(roi) == 4 else None
            count = DataAugmentor.run(
                image_dir=dataset_path,
                class_ids=class_ids,
                background_img_path=background_path,
                output_dir=output_path,
                num_augmentations=num_augmentations,
                rotation_range=rotation_range,
                blur_range=blur_range,
                scaling_range=scaling_range,
                contrast_range=contrast_range,
                brightness_range=brightness_range,
                region_scale=region_scale,
                augment_together=augment_together,
                progress_callback=update_progress,
                roi=roi_tuple,
                composition_mode=composition_mode,
                total_images=total_images,
                objects_per_image=objects_per_image
            )
            progress_tracker["status"] = "idle"
            progress_tracker["result"] = {"count": count, "output_path": output_path}
            progress_tracker["message"] = f"Successfully generated {count} objects."
        except Exception as e:
            progress_tracker["status"] = "error"
            progress_tracker["message"] = str(e)
            print(f"Augmentation Error: {e}")

