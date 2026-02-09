
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
    
    _cached_samples = [] # List of {image_path, label_line, preview_location}

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
    def sample_for_preview(
        dataset_path: str,
        class_ids: List[int],
        background_path: str,
        composition_mode: bool = False,
        objects_per_image: int = 3,
        roi: Optional[List[float]] = None
    ) -> Dict:
        """
        Picks random object(s), caches them, and returns the 'Original' preview (on background, no augs).
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
        
        
        picked_samples = []
        # Pick more samples than needed to handle placement failures
        target_count = objects_per_image if composition_mode else 1
        num_to_pick = target_count * 5 if composition_mode else 1
        
        # Try to pick N objects
        # Robust sampling: iterate through files and pick all valid objects until num_to_pick is reached
        for lf in label_files:
            if len(picked_samples) >= num_to_pick:
                break
                
            label_path = os.path.join(labels_dir, lf)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                valid_lines = [l for l in lines if l.strip() and int(l.split()[0]) in class_ids]
                if not valid_lines:
                    continue
                
                # Check if image exists
                base_name = os.path.splitext(lf)[0]
                image_path = None
                for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG']:
                    cand = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(cand):
                        image_path = cand
                        break
                
                if image_path:
                    # Pick ALL valid objects from this file until we reach num_to_pick
                    random.shuffle(valid_lines)
                    for line in valid_lines:
                        if len(picked_samples) >= num_to_pick:
                            break
                        picked_samples.append({
                            "image_path": str(image_path),
                            "label_line": line,
                            "file": lf,
                            "class_id": int(line.split()[0])
                        })

        if not picked_samples:
            raise ValueError(f"No objects found for classes {class_ids}")

        # Cache the samples
        DataAugmentationManager._cached_samples = picked_samples

        # Generate 'Original' preview (No augs)
        bg_img = cv2.imread(background_path)
        if bg_img is None: raise ValueError("Background not found")
        
        if composition_mode:
            # Call composition preview with NO augs, pass limit=target_count
            preview_img, placed_indices, locations = DataAugmentor.generate_composition_preview(
                picked_samples, bg_img,
                None, None, None, None, None, 1.0, roi,
                max_objects=target_count
            )
            # IMPORTANT: Filter picked_samples to only those that were successfully placed
            final_samples = []
            for i, p_idx in enumerate(placed_indices):
                sample = picked_samples[p_idx]
                sample["preview_location"] = locations[i]
                final_samples.append(sample)
            
            picked_samples = final_samples
        else:
            # Call single preview with NO augs
            sample = picked_samples[0]
            preview_result = DataAugmentor.generate_single_preview(
                sample["image_path"], sample["label_line"], bg_img,
                None, None, None, None, None, 1.0, roi
            )
            if preview_result:
                preview_img, loc = preview_result
                sample["preview_location"] = loc
                picked_samples = [sample]
            else:
                raise ValueError("Failed to place object in preview")
        
        # Cache the samples (NOW with locations and filtered to placed ones)
        DataAugmentationManager._cached_samples = picked_samples

        _, buffer = cv2.imencode('.jpg', preview_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "sample_info": picked_samples[0]["file"] if not composition_mode else f"{len(picked_samples)} objects"
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
        roi: Optional[List[float]] = None,
        min_width: int = 0,
        min_height: int = 0,
        composition_mode: bool = False
    ) -> Dict:
        """
        Applies augmentation to the currently cached sample objects.
        """
        if not DataAugmentationManager._cached_samples:
            raise ValueError("No cached samples. Draw a sample first.")
        
        samples = DataAugmentationManager._cached_samples
        bg_img = cv2.imread(background_path)
        if bg_img is None: raise ValueError("Background not found")
        
        roi_tuple = tuple(roi) if roi else None

        if composition_mode:
            # Use cached locations for stability
            fixed_locs = [s.get("preview_location") for s in samples]
            preview_img, _, _ = DataAugmentor.generate_composition_preview(
                samples, bg_img,
                rotation_range, blur_range, scaling_range, contrast_range, brightness_range, region_scale, roi_tuple,
                min_width=min_width, min_height=min_height,
                fixed_locations=fixed_locs
            )
        else:
            sample = samples[0]
            fixed_loc = sample.get("preview_location")
            preview_result = DataAugmentor.generate_single_preview(
                sample["image_path"], sample["label_line"], bg_img,
                rotation_range, blur_range, scaling_range, contrast_range, brightness_range, region_scale, roi_tuple,
                min_width=min_width, min_height=min_height,
                fixed_location=fixed_loc
            )
            if preview_result:
                preview_img, _ = preview_result
            else:
                 raise ValueError("Failed to apply preview to object")
        
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
        objects_per_image: int = 3,
        min_width: int = 0,
        min_height: int = 0
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
                objects_per_image=objects_per_image,
                min_width=min_width,
                min_height=min_height
            )
            progress_tracker["status"] = "idle"
            progress_tracker["result"] = {"count": count, "output_path": output_path}
            progress_tracker["message"] = f"Successfully generated {count} objects."
        except Exception as e:
            progress_tracker["status"] = "error"
            progress_tracker["message"] = str(e)
            print(f"Augmentation Error: {e}")

