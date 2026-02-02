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
    def apply_augmentations(obj_roi, mask_roi, w, h, rotation_range, blur_range, scaling_range, contrast_range, max_region_w, max_region_h):
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
            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aug_obj = cv2.warpAffine(aug_obj, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            aug_mask = cv2.warpAffine(aug_mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

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
    def extract_object(image, coords_norm, image_w, image_h):
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
        region_scale=0.8,
        roi=None
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
        obj_roi, mask_roi, w, h, coords_relative = DataAugmentor.extract_object(image, coords, image_w, image_h)
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
            scaling_range, contrast_range, max_region_w, max_region_h
        )
        
        if aug_obj is None:
            return None
            
        # Place
        if roi:
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
        
        for c in range(3):
            bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] = (
                bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] * (1 - aug_mask / 255) +
                aug_obj[:, :, c] * (aug_mask / 255)
            )
            
        # Basic visual debug of bbox (Optional, maybe not for final output)
        return bg_copy


    @staticmethod
    def process_file_wrapper(args):
        """Wrapper for multiprocessing"""
        return DataAugmentor.process_label_file(*args)

    @staticmethod
    def process_label_file(
        label_file, images_dir, labels_dir, background_img, output_dir, class_ids, 
        num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, 
        region_scale, image_w, image_h, max_region_w, max_region_h, augment_together
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

        if augment_together:
            for i in range(num_augmentations):
                bg_copy = background_img.copy()
                aug_labels = []
                
                for obj_class_id, coords in valid_objects:
                    obj_roi, mask_roi, w, h, coords_relative = DataAugmentor.extract_object(image, coords, image_w, image_h)
                    if obj_roi is None: continue

                    aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = DataAugmentor.apply_augmentations(
                        obj_roi, mask_roi, w, h, rotation_range, blur_range, 
                        scaling_range, contrast_range, max_region_w, max_region_h
                    )
                    if aug_obj is None: continue

                    rand_x = random.randint(0, max(image_w - new_w, 0))
                    rand_y = random.randint(0, max(image_h - new_h, 0))

                    for c in range(3):
                        bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] = (
                            bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] * (1 - aug_mask / 255) +
                            aug_obj[:, :, c] * (aug_mask / 255)
                        )

                    translation = [rand_x, rand_y]
                    new_coords = DataAugmentor.transform_coordinates(coords_relative, rotation_matrix, scale_factor, translation)
                    
                    new_coords[:, 0] /= image_w
                    new_coords[:, 1] /= image_h
                    new_coords = np.clip(new_coords, 0.0, 1.0)
                    new_coords = new_coords.reshape(-1)
                    aug_labels.append(f"{obj_class_id} {' '.join(map(str, new_coords))}")

                if aug_labels:
                    aug_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg"
                    aug_label_name = f"{os.path.splitext(label_file)[0]}_aug_{i}.txt"
                    cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), bg_copy)
                    with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
                        for label in aug_labels:
                            lf_aug.write(f"{label}\n")
                    generated_count += 1
        else:
            for obj_idx, (obj_class_id, coords) in enumerate(valid_objects):
                obj_roi, mask_roi, w, h, coords_relative = DataAugmentor.extract_object(image, coords, image_w, image_h)
                if obj_roi is None: continue

                for i in range(num_augmentations):
                    aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = DataAugmentor.apply_augmentations(
                        obj_roi, mask_roi, w, h, rotation_range, blur_range, 
                        scaling_range, contrast_range, max_region_w, max_region_h
                    )
                    if aug_obj is None: continue

                    rand_x = random.randint(0, max(image_w - new_w, 0))
                    rand_y = random.randint(0, max(image_h - new_h, 0))

                    bg_copy = background_img.copy()
                    for c in range(3):
                        bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] = (
                            bg_copy[rand_y:rand_y + new_h, rand_x:rand_x + new_w, c] * (1 - aug_mask / 255) +
                            aug_obj[:, :, c] * (aug_mask / 255)
                        )

                    translation = [rand_x, rand_y]
                    new_coords = DataAugmentor.transform_coordinates(coords_relative, rotation_matrix, scale_factor, translation)
                    new_coords[:, 0] /= image_w
                    new_coords[:, 1] /= image_h
                    new_coords = np.clip(new_coords, 0.0, 1.0)
                    new_coords = new_coords.reshape(-1)

                    aug_image_name = f"{os.path.splitext(image_name)[0]}_c{obj_class_id}_obj{obj_idx}_aug_{i}.jpg"
                    aug_label_name = f"{os.path.splitext(label_file)[0]}_c{obj_class_id}_obj{obj_idx}_aug_{i}.txt"
                    
                    cv2.imwrite(os.path.join(output_dir, 'images', aug_image_name), bg_copy)
                    with open(os.path.join(output_dir, 'labels', aug_label_name), 'w') as lf_aug:
                        lf_aug.write(f"{obj_class_id} {' '.join(map(str, new_coords))}\n")
                    
                    generated_count += 1

        return generated_count

    @staticmethod
    def run_composition_mode(
        images_dir, labels_dir, class_ids, background_img, output_dir,
        total_images, objects_per_image, 
        rotation_range, blur_range, scaling_range, contrast_range, 
        region_scale, image_w, image_h, max_region_w, max_region_h,
        update_progress_callback=None, roi=None
    ):
        """
        Generates 'total_images' number of images.
        Each image contains 'objects_per_image' randomly selected objects.
        Ensures no collision between objects.
        """
        # 1. Collect all valid objects first
        all_valid_objects = [] # list of (image_path, class_id, coords)
        
        # Scan all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        for lf in label_files:
            try:
                with open(os.path.join(labels_dir, lf), 'r') as f:
                    lines = f.readlines()
                    
                # Find corresponding image
                base_name = os.path.splitext(lf)[0]
                found_image = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    p = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(p):
                        found_image = p
                        break
                
                if not found_image: continue

                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    cid = int(parts[0])
                    if cid in class_ids:
                        coords = np.array(parts[1:], dtype=float)
                        all_valid_objects.append((found_image, cid, coords))
            except: continue

        if not all_valid_objects:
            print("No valid objects found for composition.")
            return 0
            
        generated_count = 0
        
        for img_idx in range(total_images):
            bg_copy = background_img.copy()
            bg_h, bg_w = bg_copy.shape[:2]
            
            aug_labels_for_this_image = []
            placed_boxes = [] # List of [x, y, w, h] in pixels
            
            # Try to place M objects
            objects_placed = 0
            retries = 0
            max_retries = 200 # increased from 50 to allow finding space
            
            while objects_placed < objects_per_image and retries < max_retries:
                # Pick random object
                try:
                    src_img_path, obj_cid, obj_coords = random.choice(all_valid_objects)
                    
                    src_img = cv2.imread(src_img_path)
                    if src_img is None: 
                        retries += 1
                        continue
                    
                    src_h, src_w = src_img.shape[:2]
                    
                    # Extract
                    obj_roi, mask_roi, w, h, coords_relative = DataAugmentor.extract_object(src_img, obj_coords, src_w, src_h)
                    if obj_roi is None:
                        retries += 1
                        continue

                    # Augment
                    aug_obj, aug_mask, new_w, new_h, rotation_matrix, scale_factor = DataAugmentor.apply_augmentations(
                        obj_roi, mask_roi, w, h, rotation_range, blur_range, 
                        scaling_range, contrast_range, max_region_w, max_region_h
                    )
                    if aug_obj is None:
                        retries += 1
                        continue

                    # Define placement bounds based on ROI
                    bound_x_min, bound_y_min = 0, 0
                    bound_x_max, bound_y_max = bg_w, bg_h
                    
                    if roi:
                        # roi is [x, y, w, h] normalized (0-1)
                        rx, ry, rw, rh = roi
                        bound_x_min = int(rx * bg_w)
                        bound_y_min = int(ry * bg_h)
                        bound_x_max = int(min((rx + rw) * bg_w, bg_w))
                        bound_y_max = int(min((ry + rh) * bg_h, bg_h))

                    # Ensure object fits in bounds
                    x_range_max = max(bound_x_max - new_w, bound_x_min)
                    y_range_max = max(bound_y_max - new_h, bound_y_min)
                    
                    # Random Position within bounds
                    rand_x = random.randint(bound_x_min, x_range_max)
                    rand_y = random.randint(bound_y_min, y_range_max)
                    
                    # Collision Check
                    current_box = [rand_x, rand_y, new_w, new_h]
                    collision = False
                    for pb in placed_boxes:
                        if DataAugmentor.check_collision(current_box, pb):
                            collision = True
                            break
                    
                    if collision:
                        retries += 1
                        continue
                    
                    
                    # Safe Slicing: Ensure we don't exceed image bounds or object bounds
                    y1, y2 = rand_y, rand_y + new_h
                    x1, x2 = rand_x, rand_x + new_w
                    
                    # Clip coordinates to background
                    y1_c = max(0, y1)
                    y2_c = min(bg_h, y2)
                    x1_c = max(0, x1)
                    x2_c = min(bg_w, x2)
                    
                    # If heavily clipped (invisible), skip
                    if y2_c <= y1_c or x2_c <= x1_c:
                         retries += 1
                         continue

                    # Calculate corresponding offsets in the object mask
                    obj_y1 = y1_c - y1
                    obj_y2 = obj_y1 + (y2_c - y1_c)
                    obj_x1 = x1_c - x1
                    obj_x2 = obj_x1 + (x2_c - x1_c)

                    # Place it
                    for c in range(3):
                        bg_slice = bg_copy[y1_c:y2_c, x1_c:x2_c, c]
                        obj_slice = aug_obj[obj_y1:obj_y2, obj_x1:obj_x2, c]
                        mask_slice = aug_mask[obj_y1:obj_y2, obj_x1:obj_x2]
                        
                        bg_copy[y1_c:y2_c, x1_c:x2_c, c] = (
                            bg_slice * (1 - mask_slice / 255) +
                            obj_slice * (mask_slice / 255)
                        )
                    
                    # Transform Coords
                    translation = [rand_x, rand_y]
                    new_coords = DataAugmentor.transform_coordinates(coords_relative, rotation_matrix, scale_factor, translation)
                    new_coords[:, 0] /= bg_w
                    new_coords[:, 1] /= bg_h
                    new_coords = np.clip(new_coords, 0.0, 1.0)
                    new_coords = new_coords.reshape(-1)
                    
                    aug_labels_for_this_image.append(f"{obj_cid} {' '.join(map(str, new_coords))}")
                    placed_boxes.append(current_box)
                    objects_placed += 1
                except Exception as e:
                    import traceback
                    print(f"DEBUG: Error processing object: {e}")
                    traceback.print_exc()
                    retries += 1
                    continue
            
            # Save Image & Label
            if aug_labels_for_this_image:
                out_name = f"comp_img_{img_idx}.jpg"
                out_label = f"comp_img_{img_idx}.txt"
                
                cv2.imwrite(os.path.join(output_dir, 'images', out_name), bg_copy)
                with open(os.path.join(output_dir, 'labels', out_label), 'w') as f:
                    for l in aug_labels_for_this_image:
                        f.write(l + "\n")
                
                generated_count += 1
                if update_progress_callback:
                    update_progress_callback(generated_count, total_images)

        return generated_count

    @staticmethod
    def run(
        image_dir, class_ids, background_img_path, output_dir=None, 
        num_augmentations=3, rotation_range=None, blur_range=None, 
        scaling_range=None, contrast_range=None, region_scale=0.8, 
        augment_together=False, progress_callback: Optional[Callable[[int, int], None]] = None,
        roi=None,
        composition_mode=False, total_images=10, objects_per_image=3
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
                 rotation_range, blur_range, scaling_range, contrast_range,
                 region_scale, image_w, image_h, max_region_w, max_region_h,
                 progress_callback, roi
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
                    num_augmentations, rotation_range, blur_range, scaling_range, contrast_range, region_scale,
                    image_w, image_h, max_region_w, max_region_h, augment_together)
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
        args.scaling_range, args.contrast_range, args.region_scale, 
        args.augment_together
    )
    print(f"Done. Generated {count} objects.")