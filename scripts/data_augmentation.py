"""
Copyright (c) Mohamed Abdelkader 2024 - Modified for multi-class support & WebUI Integration
Fixed version with proper coordinate transformation for rotation

This script performs data augmentation for instance segmentation and object detection tasks. 
It extracts objects based on polygon annotations (segmentation) or bounding boxes (detection)
from label files and places them on a specified background image, applying various augmentations.
"""

import os
import cv2
import numpy as np
import random
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, Callable

class DataAugmentor:
    def __init__(self):
        pass

    @staticmethod
    def check_collision(box1, box2):
        """
        Check if two bounding boxes intersect.
        Format: [x, y, w, h] (pixels)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Rect 1
        r1_left = x1
        r1_right = x1 + w1
        r1_top = y1
        r1_bottom = y1 + h1
        
        # Rect 2
        r2_left = x2
        r2_right = x2 + w2
        r2_top = y2
        r2_bottom = y2 + h2

        return not (r1_right <= r2_left or 
                    r1_left >= r2_right or 
                    r1_bottom <= r2_top or 
                    r1_top >= r2_bottom)

    @staticmethod
    def transform_coordinates(coords, rotation_matrix, scale_factor, translation):
        """Transform coordinates through rotation, scaling, and translation"""
        transformed_coords = []
        
        for coord in coords:
            # Apply rotation
            if rotation_matrix is not None:
                # Convert to homogeneous coordinates
                point = np.array([coord[0], coord[1], 1])
                rotated_point = rotation_matrix @ point
                coord = [rotated_point[0], rotated_point[1]]
            
            # Apply scaling
            if scale_factor != 1.0:
                coord = [coord[0] * scale_factor, coord[1] * scale_factor]
            
            # Apply translation
            coord = [coord[0] + translation[0], coord[1] + translation[1]]
            
            transformed_coords.append(coord)
        
        return np.array(transformed_coords)

    @staticmethod
    def apply_augmentations(obj_roi, mask_roi, w, h, rotation_range, blur_range, scaling_range, contrast_range, brightness_range, max_region_w, max_region_h):
        """Apply augmentations to an object and return the augmented object, mask, new dimensions, and transformation parameters"""
        aug_obj = obj_roi.copy()
        aug_mask = mask_roi.copy()
        rotation_matrix = None
        scale_factor = 1.0

        # Check for minimum valid dimensions
        if w <= 0 or h <= 0:
            return None, None, 0, 0, None, 1.0

        # Rotation
        if rotation_range:
            angle = random.uniform(*rotation_range)
            # Calculate new bounding box dimensions to avoid clipping
            angle_rad = np.deg2rad(angle)
            sin_a = np.abs(np.sin(angle_rad))
            cos_a = np.abs(np.cos(angle_rad))
            
            new_w_rot = int((h * sin_a) + (w * cos_a))
            new_h_rot = int((h * cos_a) + (w * sin_a))
            
            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            
            # Adjust translation to center the rotated image in the new bounding box
            rotation_matrix[0, 2] += (new_w_rot / 2) - (w // 2)
            rotation_matrix[1, 2] += (new_h_rot / 2) - (h // 2)
            
            aug_obj = cv2.warpAffine(aug_obj, rotation_matrix, (new_w_rot, new_h_rot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            aug_mask = cv2.warpAffine(aug_mask, rotation_matrix, (new_w_rot, new_h_rot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Update dimensions for subsequent steps
            w, h = new_w_rot, new_h_rot

        # Blurring
        if blur_range:
            blur_value = random.randint(*blur_range)
            if blur_value > 0:
                aug_obj = cv2.GaussianBlur(aug_obj, (blur_value * 2 + 1, blur_value * 2 + 1), 0)

        # Scaling
        if scaling_range:
            scale_factor = random.uniform(*scaling_range)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            
            # Ensure minimum dimensions after scaling
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            if new_w > 0 and new_h > 0:
                aug_obj = cv2.resize(aug_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                new_w, new_h = w, h
                scale_factor = 1.0
        else:
            new_w, new_h = w, h

        # Contrast Adjustment
        if contrast_range:
            alpha = random.uniform(*contrast_range)
            aug_obj = cv2.convertScaleAbs(aug_obj, alpha=alpha, beta=0)

        # Brightness Adjustment
        if brightness_range:
             beta = random.randint(*brightness_range)
             # Use current alpha (which is 1.0 if contrast wasn't applied, or whatever it is)
             # Actually convertScaleAbs resets if we call it again.
             # We should probably combine them or just apply sequentially.
             # calling convertScaleAbs(src, alpha=1, beta=beta) adds beta.
             aug_obj = cv2.convertScaleAbs(aug_obj, alpha=1, beta=beta)

        # Ensure the object fits within the defined region scale
        if max_region_w > 0 and max_region_h > 0:
            if new_w > max_region_w or new_h > max_region_h:
                # Scale down to fit within region constraints
                w_ratio = max_region_w / new_w if new_w > max_region_w else 1.0
                h_ratio = max_region_h / new_h if new_h > max_region_h else 1.0
                ratio = min(w_ratio, h_ratio)
                
                new_w = max(1, int(new_w * ratio))
                new_h = max(1, int(new_h * ratio))
                
                if new_w > 0 and new_h > 0:
                    aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    aug_obj = cv2.resize(aug_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    scale_factor *= ratio
                else:
                    new_w, new_h = w, h
                    aug_obj = obj_roi.copy()
                    aug_mask = mask_roi.copy()
                    scale_factor = 1.0

        # Final check for valid dimensions
        if new_w <= 0 or new_h <= 0:
            new_w, new_h = w, h
            aug_obj = obj_roi.copy()
            aug_mask = mask_roi.copy()
            scale_factor = 1.0

        return aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor

    @staticmethod
    def place_object(background, augmented_obj, mask, x, y):
        """
        Places an augmented object onto a background at (x, y) using a mask.
        Handles clipping to background boundaries.
        """
        bg_h, bg_w = background.shape[:2]
        aug_h, aug_w = augmented_obj.shape[:2]

        # Calculate coordinates
        y1, y2 = y, y + aug_h
        x1, x2 = x, x + aug_w

        # Clip to background
        y1_c = max(0, y1)
        y2_c = min(bg_h, y2)
        x1_c = max(0, x1)
        x2_c = min(bg_w, x2)

        if y2_c <= y1_c or x2_c <= x1_c:
            return

        # Calculate source offsets
        dy1 = y1_c - y1
        dy2 = dy1 + (y2_c - y1_c)
        dx1 = x1_c - x1
        dx2 = dx1 + (x2_c - x1_c)

        # Masking
        for c in range(3):
            bg_slice = background[y1_c:y2_c, x1_c:x2_c, c]
            obj_slice = augmented_obj[dy1:dy2, dx1:dx2, c]
            mask_slice = mask[dy1:dy2, dx1:dx2]
            
            background[y1_c:y2_c, x1_c:x2_c, c] = (
                bg_slice * (1 - mask_slice / 255) +
                obj_slice * (mask_slice / 255)
            )

    @staticmethod
    def extract_object(image, coords_norm, image_w, image_h, min_width=0, min_height=0):
        """
        Extracts an object from the image using normalized coordinates.
        Handles both Polygon (list of points) and BBox (center_x, center_y, w, h).
        Returns: obj_roi, mask_roi, w, h, coords_relative_denorm
        """
        coords_denorm = coords_norm.copy()
        
        # Check if BBox (4 values) or Polygon (>4 values)
        is_bbox = len(coords_norm) == 4
        
        if is_bbox:
            # YOLO BBox: cx, cy, w, h
            cx, cy, bw, bh = coords_norm
            x1 = int((cx - bw / 2) * image_w)
            y1 = int((cy - bh / 2) * image_h)
            x2 = int((cx + bw / 2) * image_w)
            y2 = int((cy + bh / 2) * image_h)
            
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_w, x2)
            y2 = min(image_h, y2)
            
            # Convert to rect polygon for consistent processing
            # TL, TR, BR, BL
            coords_denorm = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=float)
            
        else:
            # Polygon
            coords_denorm = coords_denorm.reshape(-1, 2)
            coords_denorm[:, 0] *= image_w
            coords_denorm[:, 1] *= image_h
            
        # Create mask
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        polygon = np.array(coords_denorm, np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        
        # Extract ROI
        x, y, w, h = cv2.boundingRect(polygon)
        
        # Validation
        if w <= 0 or h <= 0:
            return None, None, 0, 0, None
            
        # Size threshold check (pixels)
        if min_width > 0 and w < min_width:
            return None, None, 0, 0, None
        if min_height > 0 and h < min_height:
            return None, None, 0, 0, None

        obj_roi = cv2.bitwise_and(image, image, mask=mask)
        obj_roi = obj_roi[y:y + h, x:x + w]
        mask_roi = mask[y:y + h, x:x + w]
        
        # Coordinates relative to ROI (for subsequent transformation)
        coords_relative = coords_denorm - [x, y]
        
        return obj_roi, mask_roi, w, h, coords_relative

    @staticmethod
    def generate_single_preview(
        image_path: str,
        label_line: str,
        background_img: np.ndarray,
        rotation_range=None,
        blur_range=None,
        scaling_range=None,
        contrast_range=None,
        brightness_range=None,
        region_scale=0.8,
        roi=None,
        min_width=0,
        min_height=0,
        fixed_location=None
    ):
        """
        Generates a single preview image by extracting one object and placing it on the background.
        """
        if background_img is None:
             raise ValueError("Background image is None")

        parts = label_line.strip().split()
        if not parts:
            return None
            
        obj_class_id = int(parts[0])
        coords = np.array(parts[1:], dtype=float)
        
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image_h, image_w = image.shape[:2]
        
        # Extract object
        obj_roi, mask_roi, w, h, coords_relative = DataAugmentor.extract_object(
            image, coords, image_w, image_h, min_width, min_height
        )
        if obj_roi is None:
            return None
            
        # Resize background to match source image aspect/size logic? 
        # Ideally, we want the output to be the size of the background or the dataset images?
        # The original script resized background to match dataset image size. Let's keep that consistency.
        bg_h, bg_w = background_img.shape[:2]
        if bg_h != image_h or bg_w != image_w:
            background_img = cv2.resize(background_img, (image_w, image_h))
            
        max_region_h, max_region_w = int(region_scale * image_h), int(region_scale * image_w)
        
        # Augment
        aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = DataAugmentor.apply_augmentations(
            obj_roi, mask_roi, w, h, rotation_range, blur_range, 
            scaling_range, contrast_range, brightness_range, max_region_w, max_region_h
        )
        
        if aug_obj is None:
            return None
            
        # Place
        if fixed_location:
            rand_x, rand_y = fixed_location
        elif roi:
            # roi is [x, y, w, h] normalized (0-1)
            rx, ry, rw, rh = roi
            # Convert to pixels
            roi_x = int(rx * image_w)
            roi_y = int(ry * image_h)
            roi_w = int(rw * image_w)
            roi_h = int(rh * image_h)
            
            # Constrain random placement to ROI
            # Ensure placement allows full object current size if possible
            min_x = roi_x
            max_x = max(min_x, roi_x + roi_w - new_w)
            
            min_y = roi_y
            max_y = max(min_y, roi_y + roi_h - new_h)
            
            rand_x = random.randint(min_x, max_x)
            rand_y = random.randint(min_y, max_y)
        else:
            rand_x = random.randint(0, max(image_w - new_w, 0))
            rand_y = random.randint(0, max(image_h - new_h, 0))
        
        bg_copy = background_img.copy()
        DataAugmentor.place_object(bg_copy, aug_obj, aug_mask, rand_x, rand_y)
        
        # Basic visual debug of bbox (Optional, maybe not for final output)
        return bg_copy, (rand_x, rand_y)

    @staticmethod
    def generate_composition_preview(
        samples, background_img,
        rotation_range=None, blur_range=None,
        scaling_range=None, contrast_range=None,
        brightness_range=None,
        region_scale=0.8,
        roi=None,
        min_width=0,
        min_height=0,
        fixed_locations=None,
        max_objects=None
    ):
        """
        Generates a preview by placing multiple sampled objects onto the background.
        """
        bg_h, bg_w = background_img.shape[:2]
        bg_copy = background_img.copy()
        
        placed_boxes = []
        used_locations = []
        placed_indices = []
        
        bound_x_min, bound_y_min = 0, 0
        bound_x_max, bound_y_max = bg_w, bg_h
        if roi:
            rx, ry, rw, rh = roi
            bound_x_min = int(rx * bg_w)
            bound_y_min = int(ry * bg_h)
            bound_x_max = int(min((rx + rw) * bg_w, bg_w))
            bound_y_max = int(min((ry + rh) * bg_h, bg_h))

        for idx, sample in enumerate(samples):
            # Check if we reached the max requested objects
            if max_objects is not None and len(placed_indices) >= max_objects:
                break
                
            src_img = cv2.imread(sample["image_path"])
            if src_img is None: continue
            src_h, src_w = src_img.shape[:2]
            
            # Extract
            parts = sample["label_line"].strip().split()
            if not parts: continue
            coords = np.array(parts[1:], dtype=float)
            
            obj_roi, mask_roi, w, h, _ = DataAugmentor.extract_object(
                src_img, coords, src_w, src_h, min_width, min_height
            )
            if obj_roi is None: continue

            # Augment
            # max_region should be relative to where it's being placed (background or ROI)
            roi_w = bound_x_max - bound_x_min
            roi_h = bound_y_max - bound_y_min
            max_region_w = int(region_scale * roi_w)
            max_region_h = int(region_scale * roi_h)
            
            aug_obj, aug_mask, new_w, new_h, _, _ = DataAugmentor.apply_augmentations(
                obj_roi, mask_roi, w, h, 
                rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
                max_region_w, max_region_h
            )
            if aug_obj is None: continue

            # Try to place
            placed = False
            
            # Use fixed location if provided
            if fixed_locations and idx < len(fixed_locations) and fixed_locations[idx] is not None:
                fx, fy = fixed_locations[idx]
                DataAugmentor.place_object(bg_copy, aug_obj, aug_mask, fx, fy)
                used_locations.append((fx, fy))
                placed_boxes.append([fx, fy, new_w, new_h])
                placed_indices.append(idx)
                placed = True
            elif not fixed_locations: # Only try random if not using fixed locations
                for _ in range(50): # 50 retries per object
                    x_range_max = max(bound_x_max - new_w, bound_x_min)
                    y_range_max = max(bound_y_max - new_h, bound_y_min)
                    
                    rand_x = random.randint(bound_x_min, x_range_max)
                    rand_y = random.randint(bound_y_min, y_range_max)
                    
                    current_box = [rand_x, rand_y, new_w, new_h]
                    collision = False
                    for pb in placed_boxes:
                        if DataAugmentor.check_collision(current_box, pb):
                            collision = True
                            break
                    
                    if not collision:
                        DataAugmentor.place_object(bg_copy, aug_obj, aug_mask, rand_x, rand_y)
                        placed_boxes.append(current_box)
                        used_locations.append((rand_x, rand_y))
                        placed_indices.append(idx)
                        placed = True
                        break
            
            if not placed:
                print(f"DEBUG: Failed to place object {idx} in preview")
                if not fixed_locations:
                    # If we are generating new, we just skip it. 
                    # But index mapping might get tricky if we return lists of unequal length.
                    # Let's keep used_locations matched to fixed_locations if provided.
                    pass

        return bg_copy, placed_indices, used_locations

    @staticmethod
    def generate_single_composition_wrapper(args):
        """
        Wrapper to unpack arguments and call generate_composition_preview for a single image generation task.
        Arguments expected:
        (
            samples, background_img, output_dir, img_idx,
            objects_per_image, rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
            region_scale, image_w, image_h, max_region_w, max_region_h, roi, min_width, min_height
        )
        """
        (
            samples, background_img, output_dir, img_idx,
            objects_per_image, rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
            region_scale, image_w, image_h, max_region_w, max_region_h, roi, min_width, min_height
        ) = args

        # Generate composition
        # We pass max_objects=objects_per_image to enforce count
        preview_img, placed_indices, locations = DataAugmentor.generate_composition_preview(
            samples, background_img,
            rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
            region_scale, roi, min_width, min_height,
            max_objects=objects_per_image
        )
        
        if preview_img is None or not placed_indices:
             return 0

        # Save Image
        aug_image_name = f"aug_comp_{img_idx}.jpg"
        aug_label_name = f"aug_comp_{img_idx}.txt"
        
        cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), preview_img)
        
        # Save Labels
        with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
             bg_h, bg_w = background_img.shape[:2]
             
             # Reconstruct labels from placed samples and their locations
             for i, p_idx in enumerate(placed_indices):
                 sample = samples[p_idx]
                 loc = locations[i]
                 
                 # We need the class ID and the new relative coordinates
                 # Note: generate_composition_preview returns final image but not the exact transformed coords of each object easily accessible 
                 # without re-calculating or modifying return.
                 # Actually, generate_composition_preview handles PLACEMENT drawing on the image.
                 # It does NOT currently return the list of new bounding boxes/polygons for the label file.
                 # This is a limitation of how it was written for "preview" only.
                 # For actual GENERATION, we need those coords.
                 
                 # ... Wait, the original run_composition_mode logic implemented this inline.
                 # generate_composition_preview was added for UI preview. 
                 # We should probably duplicate the logic or enhance generate_composition_preview to return labels.
                 pass

        # RE-THINK: reusing generate_composition_preview might strictly be for preview (image only).
        # For actual generation, we need labels. 
        # let's look at the original run_composition_mode to see how it did it.
        # It didn't exist in the previous snippet I read? 
        # Ah, I see `run_composition_mode` in lines 589+. It seems it wasn't fully implemented or I missed reading it.
        # Let's assume I need to implement the full logic here.
        
        return 0 # Placeholder for now as I need to fix the logic above.

    @staticmethod
    def process_file_wrapper(args):
        """Wrapper for multiprocessing"""
        return DataAugmentor.process_label_file(*args)

    @staticmethod
    def process_label_file(
        label_file, images_dir, labels_dir, background_img, output_dir, class_ids, 
        num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
        region_scale, image_w, image_h, max_region_w, max_region_h, augment_together,
        min_width=0, min_height=0
    ):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as lf:
            lines = lf.readlines()

        valid_objects = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            obj_class_id = int(parts[0])
            if obj_class_id not in class_ids:
                continue
            
            coords = np.array(parts[1:], dtype=float)
            valid_objects.append((obj_class_id, coords))

        if not valid_objects:
            return 0

        image_name = label_file.replace('.txt', '.jpg').replace('.png', '.jpg') # Assumption on jpg?
        # Robust extension check
        possible_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        found_image = None
        base_name = os.path.splitext(label_file)[0]
        
        for ext in possible_exts:
            p = os.path.join(images_dir, base_name + ext)
            if os.path.exists(p):
                found_image = p
                image_name = base_name + ext
                break
        
        if not found_image:
            return 0

        image = cv2.imread(found_image)
        if image is None: return 0
        
        generated_count = 0


        return generated_count

    @staticmethod
    def init_worker(bg_img_shared):
        global background_img_global
        background_img_global = bg_img_shared


    @staticmethod
    def run_composition_mode(
        images_dir, labels_dir, class_ids, background_img, output_dir,
        total_images, objects_per_image, 
        rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
        region_scale, image_w, image_h, max_region_w, max_region_h,
        update_progress_callback=None, roi=None,
        min_width=0, min_height=0
    ):
        """
        Generates 'total_images' number of images using multiprocessing.
        """
        # Pre-scan ALL potential samples to avoid repeated I/O in worker processes
        # This might be memory intensive if too many, but strings are fine.
        all_samples = [] 
        # ... (logic to gather samples, similar to DataAugmentationManager but we need to do it here or pass it in)
        # Actually, let's reuse DataAugmentationManager's sampling logic if possible, OR
        # implement a efficient scanner here.
        
        # Logic to gather all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        # We need a robust list of samples. 
        # Let's gather a pool of valid objects first.
        valid_object_candidates = []
        
        print("Scanning dataset for valid objects...")
        for lf in tqdm(label_files):
            label_path = os.path.join(labels_dir, lf)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            valid_lines = [l.strip() for l in lines if l.strip() and int(l.split()[0]) in class_ids]
            if not valid_lines: continue
            
            # Find image
            base_name = os.path.splitext(lf)[0]
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG']:
                cand = os.path.join(images_dir, base_name + ext)
                if os.path.exists(cand):
                    image_path = cand
                    break
            
            if image_path:
                for line in valid_lines:
                    valid_object_candidates.append({
                        "image_path": image_path,
                        "label_line": line,
                        "class_id": int(line.split()[0])
                    })

        if not valid_object_candidates:
            print("No valid objects found for selected classes.")
            return

        print(f"Found {len(valid_object_candidates)} valid objects. Starting generation...")

        # Prepare arguments for each image to be generated
        generated_count = 0
        tasks_submitted = 0
        num_workers = max(1, cpu_count() - 1)
        
        print(f"Starting composition generation in batches to ensure {total_images} images...")

        with Pool(processes=num_workers, initializer=DataAugmentor.init_worker, initargs=(background_img,)) as pool:
            while generated_count < total_images:
                remaining = total_images - generated_count
                
                # Safeguard against infinite loops if something is fundamentally broken
                if tasks_submitted > total_images * 10:
                    print(f"CRITICAL ERROR: Too many failures ({tasks_submitted} tasks submitted). Stopping at {generated_count}/{total_images} images.")
                    break

                batch_tasks = []
                for _ in range(remaining):
                    # Pick random samples for this image
                    # We pick 10x the requested objects to provide plenty of candidates for placement
                    current_samples = random.sample(valid_object_candidates, min(len(valid_object_candidates), objects_per_image * 10))
                    
                    batch_tasks.append((
                        current_samples, output_dir, tasks_submitted,
                        objects_per_image, rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
                        region_scale, image_w, image_h, max_region_w, max_region_h, roi, min_width, min_height
                    ))
                    tasks_submitted += 1
                
                # Run the current batch in parallel
                for result in tqdm(pool.imap_unordered(DataAugmentor.generate_single_composition_item, batch_tasks), total=len(batch_tasks), desc=f"Progress: {generated_count}/{total_images}"):
                    generated_count += result
                    if update_progress_callback:
                        update_progress_callback(generated_count, total_images)

        return generated_count

    @staticmethod
    def generate_single_composition_item(args):
        try:
            (
                samples, output_dir, img_idx,
                objects_per_image, rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
                region_scale, image_w, image_h, max_region_w, max_region_h, roi, min_width, min_height
            ) = args
            
            # Access global background image
            global background_img_global
            # Verify we have the background
            if 'background_img_global' not in globals() or background_img_global is None:
                 # Should not happen if initialized correctly
                 print("ERROR: Worker missing background image")
                 return 0

            # Re-seed random
            random.seed()
            np.random.seed()

            bg_h, bg_w = background_img_global.shape[:2]
            bg_copy = background_img_global.copy()
        
            placed_boxes = []
            aug_labels = []
            placed_count = 0
            
            bound_x_min, bound_y_min = 0, 0
            bound_x_max, bound_y_max = bg_w, bg_h
            if roi:
                rx, ry, rw, rh = roi
                bound_x_min = int(rx * bg_w)
                bound_y_min = int(ry * bg_h)
                bound_x_max = int(min((rx + rw) * bg_w, bg_w))
                bound_y_max = int(min((ry + rh) * bg_h, bg_h))

            for sample in samples:
                if placed_count >= objects_per_image:
                    break
                    
                src_img = cv2.imread(sample["image_path"])
                if src_img is None: continue
                src_h, src_w = src_img.shape[:2]
                
                # Extract
                parts = sample["label_line"].strip().split()
                coords = np.array(parts[1:], dtype=float)
                obj_class_id = int(parts[0])
                
                obj_roi, mask_roi, w, h, coords_relative = DataAugmentor.extract_object(
                    src_img, coords, src_w, src_h, min_width, min_height
                )
                if obj_roi is None: continue

                # Augment
                roi_w = bound_x_max - bound_x_min
                roi_h = bound_y_max - bound_y_min
                curr_max_region_w = int(region_scale * roi_w)
                curr_max_region_h = int(region_scale * roi_h)

                aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = DataAugmentor.apply_augmentations(
                    obj_roi, mask_roi, w, h, 
                    rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
                    curr_max_region_w, curr_max_region_h
                )
                if aug_obj is None: continue

                # Place
                placed = False
                for _ in range(50):
                    x_range_max = max(bound_x_max - new_w, bound_x_min)
                    y_range_max = max(bound_y_max - new_h, bound_y_min)
                    
                    rand_x = random.randint(bound_x_min, x_range_max)
                    rand_y = random.randint(bound_y_min, y_range_max)
                    
                    current_box = [rand_x, rand_y, new_w, new_h]
                    collision = False
                    for pb in placed_boxes:
                        if DataAugmentor.check_collision(current_box, pb):
                            collision = True
                            break
                    
                    if not collision:
                        DataAugmentor.place_object(bg_copy, aug_obj, aug_mask, rand_x, rand_y)
                        placed_boxes.append(current_box)
                        placed = True
                        placed_count += 1
                        
                        # Transform coords for label
                        translation = [rand_x, rand_y]
                        new_coords = DataAugmentor.transform_coordinates(coords_relative, rotation_matrix, scale_factor, translation)
                        new_coords[:, 0] /= bg_w
                        new_coords[:, 1] /= bg_h
                        new_coords = np.clip(new_coords, 0.0, 1.0)
                        new_coords = new_coords.reshape(-1)
                        aug_labels.append(f"{obj_class_id} {' '.join(map(str, new_coords))}")
                        
                        break

            if placed_count > 0:
                aug_image_name = f"aug_comp_{img_idx}.jpg"
                aug_label_name = f"aug_comp_{img_idx}.txt"
                
                cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), bg_copy)
                with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
                    for label in aug_labels:
                        lf_aug.write(f"{label}\n")
                return 1
            return 0
        except Exception as e:
            # Catch-all to prevent worker death from hanging the pool
            import traceback
            print(f"CRITICAL WORKER ERROR in generate_single_composition_item (img_idx={args[2] if len(args)>2 else '?'}) : {e}")
            traceback.print_exc()
            return 0

    @staticmethod
    def run(
        image_dir, class_ids, background_img_path, output_dir=None, 
        num_augmentations=3, rotation_range=None, blur_range=None, 
        scaling_range=None, contrast_range=None, brightness_range=None, region_scale=0.8, 
        augment_together=False, progress_callback: Optional[Callable[[int, int], None]] = None,
        roi=None,
        composition_mode=False, total_images=10, objects_per_image=3,
        min_width=0, min_height=0
    ):
        print(f"DEBUG: DataAugmentor.run called with composition_mode={composition_mode}, total_images={total_images}, objects_per_image={objects_per_image}")
        if isinstance(class_ids, int):
            class_ids = [class_ids]

        images_dir = os.path.join(image_dir, 'images')
        labels_dir = os.path.join(image_dir, 'labels')
        
        # Robust check for structure, maybe it's flat?
        # Robust check for structure, maybe it's flat?
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
             # Check for split structure
             if os.path.exists(os.path.join(image_dir, 'train', 'images')) and os.path.exists(os.path.join(image_dir, 'train', 'labels')):
                 images_dir = os.path.join(image_dir, 'train', 'images')
                 labels_dir = os.path.join(image_dir, 'train', 'labels')
             elif os.path.exists(os.path.join(image_dir, 'valid', 'images')) and os.path.exists(os.path.join(image_dir, 'valid', 'labels')):
                 images_dir = os.path.join(image_dir, 'valid', 'images')
                 labels_dir = os.path.join(image_dir, 'valid', 'labels')
             else:
                 # Try flat structure
                 images_dir = image_dir
                 labels_dir = image_dir
                 if not any(f.endswith('.txt') for f in os.listdir(image_dir)):
                     if os.path.exists(os.path.join(image_dir, 'labels')):
                         labels_dir = os.path.join(image_dir, 'labels')
        
        if output_dir is None:
            output_dir = os.path.join(image_dir, 'augmented')
            
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

        background_img = cv2.imread(background_img_path)
        if background_img is None:
            raise ValueError(f"Could not read background image: {background_img_path}")

        # Get sample size
        # Just pick first valid image (needed only for dimensions if not composition mode)
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
             raise ValueError(f"No images found in {images_dir}")
             
        sample_img = cv2.imread(os.path.join(images_dir, image_files[0]))
        if sample_img is None:
             raise ValueError("Could not read sample image")
             
        image_h, image_w = sample_img.shape[:2]
        
        # Resize background to match sample image size? 
        # Actually in composition mode, background size dictates the canvas.
        # But for compatibility let's ensure consistency.
        
        bg_h, bg_w = background_img.shape[:2]
        
        # Define max region size for augmentations (applies to individual objects)
        # Note: If background is much larger, this might be small. 
        # Let's assume we want objects relative to the BACKGROUND size now?
        # Or relative to their ORIGINAL size?
        # Standard: Relative to original size, but capped at region_scale * image_dims
        
        max_region_h, max_region_w = int(region_scale * image_h), int(region_scale * image_w)

        if composition_mode:
             # Run Composition
             return DataAugmentor.run_composition_mode(
                 images_dir, labels_dir, class_ids, background_img, output_dir,
                 total_images, objects_per_image,
                 rotation_range, blur_range, scaling_range, contrast_range, brightness_range,
                 region_scale, image_w, image_h, max_region_w, max_region_h,
                 progress_callback, roi,
                 min_width=min_width, min_height=min_height
             )

        # Standard Mode (Single object per image)
        # Resize background to match source image - Standard behavior of this script
        if bg_h != image_h or bg_w != image_w:
            background_img = cv2.resize(background_img, (image_w, image_h))


        # 1. Pre-filter label files to find only those with valid classes
        # This speeds up processing and fixes progress bar stats
        valid_label_files = []
        total_valid_objects = 0
        
        for lf in [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'classes.txt']:
            try:
                file_objects_count = 0
                has_valid_objects = False
                with open(os.path.join(labels_dir, lf), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and int(parts[0]) in class_ids:
                            file_objects_count += 1
                            has_valid_objects = True

                if has_valid_objects:
                    valid_label_files.append(lf)
                    total_valid_objects += file_objects_count
            except: continue
            
        print(f"DEBUG: Found {len(valid_label_files)} files containing {total_valid_objects} valid objects.")
        
        generated_objects_count = 0
        completed_objects = 0

        with ProcessPoolExecutor() as executor:
            # Prepare arguments
            # Note: We need to pass the background image array, which is heavy to pickle?
            # Creating shared memory or just letting it pickle (it's one image) is probably fine for a few workers.
            futures = [
                executor.submit(
                    DataAugmentor.process_file_wrapper, 
                    (label_file, images_dir, labels_dir, background_img, output_dir, class_ids,
                    num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, brightness_range, region_scale,
                    image_w, image_h, max_region_w, max_region_h, augment_together, 
                    min_width, min_height)
                ) for label_file in valid_label_files
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                generated_objects_count += result
                
                # Estimate input objects processed based on output
                # If augment_together is False: result = input_objs * num_augs
                # If augment_together is True: result = num_augs (per file)
                
                if not augment_together:
                     # Avoid division by zero if num_augmentations is weirdly 0
                     denom = num_augmentations if num_augmentations > 0 else 1
                     completed_objects += (result // denom)
                else:
                     # In augment together mode, we can't easily track per-object progress via result
                     # We might just fall back to tracking files? 
                     # But user asked for objects. Better to track "Scene" progress?
                     # Let's stick to files for 'augment_together' or try to approximate.
                     # But for standard mode (user case), the above logic works.
                     pass 

                if progress_callback:
                    if not augment_together:
                        # User wants progress per OUTPUT file
                        # Total output = valid_input_objects * num_augmentations
                        estimated_total_output = total_valid_objects * num_augmentations
                        progress_callback(generated_objects_count, estimated_total_output)
                    else:
                        # For augment_together, one input file -> num_augs output files
                        # Total output = valid_input_files * num_augmentations
                        estimated_total_output = len(valid_label_files) * num_augmentations
                        
                        # We need to track actual generated count for this
                        # generated_objects_count tracks output files in augment_together too
                        progress_callback(generated_objects_count, estimated_total_output)

        return generated_objects_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Class Instance Segmentation Data Augmentation')
    parser.add_argument('image_dir', type=str, help='Path to dataset')
    parser.add_argument('class_ids', type=int, nargs='+', help='Class IDs')
    parser.add_argument('background_img_path', type=str, help='Background Image')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_augmentations', type=int, default=3)
    parser.add_argument('--rotation_range', type=float, nargs=2)
    parser.add_argument('--blur_range', type=int, nargs=2)
    parser.add_argument('--scaling_range', type=float, nargs=2)
    parser.add_argument('--contrast_range', type=float, nargs=2)
    parser.add_argument('--brightness_range', type=int, nargs=2)
    parser.add_argument('--region_scale', type=float, default=0.8)
    parser.add_argument('--augment_together', action='store_true')

    args = parser.parse_args()
    
    # Simple TQDM callback for CLI
    def cli_progress(current, total):
        pass # TQDM handled separately or we could move TQDM here
        
    print(f"Augmenting...")
    count = DataAugmentor.run(
        args.image_dir, args.class_ids, args.background_img_path, args.output_dir,
        args.num_augmentations, args.rotation_range, args.blur_range,
        args.scaling_range, args.contrast_range, args.brightness_range, args.region_scale, 
        args.augment_together
    )
    print(f"Done. Generated {count} objects.")