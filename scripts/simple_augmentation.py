
import cv2
import numpy as np
import os
import albumentations as A
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil

class SimpleAugmentor:
    def __init__(self):
        self.supported_augments = {
            'horizontal_flip': A.HorizontalFlip,
            'vertical_flip': A.VerticalFlip,
            'rotate': A.SafeRotate, 
            'brightness_contrast': A.RandomBrightnessContrast,
            'blur': A.Blur,
            'motion_blur': A.MotionBlur,
            'gaussian_noise': A.GaussNoise,
            'grayscale': A.ToGray,
            'rgb_shift': A.RGBShift,
            'clahe': A.CLAHE,
            'shear': A.Affine
        }

    def get_pipeline(self, config, has_bboxes=False, has_keypoints=False):
        """
        Builds albumentations Compose pipeline from config dict.
        """
        transforms = []
        
        # Mapping UI params to Albumentations params
        if config.get('horizontal_flip', {}).get('enabled'):
            transforms.append(A.HorizontalFlip(p=config['horizontal_flip'].get('p', 0.5)))
            
        if config.get('vertical_flip', {}).get('enabled'):
            transforms.append(A.VerticalFlip(p=config['vertical_flip'].get('p', 0.5)))
            
        if config.get('rotate', {}).get('enabled'):
            limit = config['rotate'].get('limit', 15)
            # border_mode=0 (CONSTANT), value=0 (Black)
            transforms.append(A.SafeRotate(limit=limit, p=config['rotate'].get('p', 0.5), border_mode=0, value=0))

        if config.get('shear', {}).get('enabled'):
             shear = config['shear'].get('shear', 15)
             transforms.append(A.Affine(shear={'x': (-shear, shear), 'y': (-shear, shear)}, p=config['shear'].get('p', 0.5), mode=0, cval=0))

        if config.get('brightness', {}).get('enabled'):
            brightness_limit = config['brightness'].get('brightness_limit', 0.2)
            contrast_limit = config['brightness'].get('contrast_limit', 0.0)
            transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=config['brightness'].get('p', 0.5)))

        if config.get('blur', {}).get('enabled'):
             limit = config['blur'].get('blur_limit', 7)
             if limit % 2 == 0: limit += 1
             transforms.append(A.Blur(blur_limit=limit, p=config['blur'].get('p', 0.5)))

        if config.get('noise', {}).get('enabled'):
             var_limit = config['noise'].get('var_limit', (10.0, 50.0))
             transforms.append(A.GaussNoise(var_limit=var_limit, p=config['noise'].get('p', 0.5)))

        if config.get('grayscale', {}).get('enabled'):
             transforms.append(A.ToGray(p=config['grayscale'].get('p', 0.5)))

        # Define extra params based on data content
        bbox_params = None
        keypoint_params = None
        
        if has_bboxes:
            # YOLO format: [x, y, w, h] normalized
            bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)
            
        if has_keypoints:
            # Keypoints: [x, y] absolute (pixels)
            # remove_invisible=False ensuring we keep points even if rotated out? 
            # Actually for polygons, if points go out, the polygon is clipped?
            # It's better to clip them to image bounds later manually if needed.
            keypoint_params = A.KeypointParams(format='xy', remove_invisible=False)
        
        return A.Compose(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params)

    def clamp_bbox(self, coords):
        """Clamps YOLO format bbox within [0, 1]."""
        x_c, y_c, w, h = coords
        x1 = max(0.0, min(1.0, x_c - w / 2))
        y1 = max(0.0, min(1.0, y_c - h / 2))
        x2 = max(0.0, min(1.0, x_c + w / 2))
        y2 = max(0.0, min(1.0, y_c + h / 2))
        return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

    def read_label_file(self, label_path):
        """
        Parses label file.
        Returns:
            bboxes: list of [x,y,w,h] (normalized)
            polygons: list of list of [x, y] (normalized)
            bbox_classes: list of class_ids for bboxes
            poly_classes: list of class_ids for polygons
        """
        bboxes = []
        polygons = []
        bbox_classes = []
        poly_classes = []
        
        if not os.path.exists(label_path):
            return [], [], [], []
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            if len(coords) == 4:
                # BBox
                coords = self.clamp_bbox(coords)
                if coords[2] > 0 and coords[3] > 0:
                    bboxes.append(coords)
                    bbox_classes.append(class_id)
            else:
                # Polygon (>= 6 coords usually, pairs of x,y)
                # Group into (x,y) pairs
                points = []
                for i in range(0, len(coords), 2):
                    if i+1 < len(coords):
                        points.append([coords[i], coords[i+1]])
                if points:
                    polygons.append(points)
                    poly_classes.append(class_id)
                    
        return bboxes, polygons, bbox_classes, poly_classes

    def apply_augmentation(self, image, label_path, config):
        """Apply augmentation to image and labels."""
        h, w = image.shape[:2]
        bboxes, polygons, bbox_classes, poly_classes = self.read_label_file(label_path)
        
        # Prepare data for Albumentations
        # 1. BBoxes keys are [x, y, w, h] normalized
        
        # 2. Polygons keys: flatten to list of keypoints, convert to absolute (pixels)
        all_keypoints = []
        poly_lengths = [] # Track how many points per polygon to reconstruct
        
        for poly in polygons:
            poly_lengths.append(len(poly))
            for pt in poly:
                # Denormalize
                all_keypoints.append([pt[0] * w, pt[1] * h])
                
        has_bboxes = len(bboxes) > 0
        has_keypoints = len(all_keypoints) > 0
        
        pipeline = self.get_pipeline(config, has_bboxes=has_bboxes, has_keypoints=has_keypoints)
        
        # Prepare kwargs
        kwargs = {'image': image}
        if has_bboxes:
            kwargs['bboxes'] = bboxes
            kwargs['class_labels'] = bbox_classes
        if has_keypoints:
            kwargs['keypoints'] = all_keypoints
            
        res = pipeline(**kwargs)
        
        aug_img = res['image']
        aug_bboxes = res.get('bboxes', [])
        aug_bbox_classes = res.get('class_labels', []) # Used for bboxes only in our config
        aug_keypoints = res.get('keypoints', [])
        
        # Reconstruct outputs
        final_bboxes = []
        final_bbox_classes = list(aug_bbox_classes) # assuming order preserved or matching filtering
        
        # Albumentations filtering might remove bboxes, so class_labels are updated automatically.
        # But for keypoints, it doesn't automatically group them back to polygons.
        # We need to assume that if remove_invisible=False, keypoints count remains same?
        # Yes, with remove_invisible=False, invisible points remain.
        
        final_bboxes = aug_bboxes
        
        final_polygons = []
        final_poly_classes = []
        
        if has_keypoints:
            # Slice back
            current_idx = 0
            new_h, new_w = aug_img.shape[:2]
            
            for i, length in enumerate(poly_lengths):
                poly_pts = aug_keypoints[current_idx : current_idx + length]
                current_idx += length
                
                # Check if this polygon is still valid (e.g. inside image)
                # Maybe clamp points?
                norm_poly = []
                valid_pts = 0
                for pt in poly_pts:
                    # Normalize back
                    nx = pt[0] / new_w
                    ny = pt[1] / new_h
                    # Clamp
                    nx = max(0.0, min(1.0, nx))
                    ny = max(0.0, min(1.0, ny))
                    norm_poly.append([nx, ny])
                    valid_pts += 1
                
                # We keep the polygon if it has points? 
                # With minimal geometric augs, it should.
                final_polygons.append(norm_poly)
                final_poly_classes.append(poly_classes[i])
                
        return aug_img, final_bboxes, final_bbox_classes, final_polygons, final_poly_classes

    def generate_preview(self, image_path, label_path, config):
        image = cv2.imread(image_path)
        if image is None: raise ValueError("Invalid image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        aug_img, boxes, box_cls, polys, poly_cls = self.apply_augmentation(image, label_path, config)
        
        # Visualise
        vis_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        h, w = vis_img.shape[:2]
        
        # Draw Boxes
        for box in boxes:
            x_c, y_c, bw, bh = box
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)
            cv2.rectangle(vis_img, (x1,y1), (x2,y2), (0,255,0), 2)
            
        # Draw Polygons
        for poly in polys:
            pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in poly], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], True, (0, 0, 255), 2)
            
        return vis_img

    def scan_dataset(self, dataset_path):
        """
        Scans dataset to determine structure (flat vs split) and count images.
        """
        info = {
            'type': 'flat',
            'splits': [],
            'total_images': 0,
            'split_counts': {},
            'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        }
        
        if not os.path.isdir(dataset_path):
            return info

        # Check for standard YOLO splits
        potential_splits = ['train', 'val', 'valid', 'test']
        found_splits = []
        
        for split in potential_splits:
            split_dir = os.path.join(dataset_path, split)
            if os.path.isdir(split_dir):
                # Check for images subdir
                img_dir = os.path.join(split_dir, 'images')
                if os.path.isdir(img_dir):
                    found_splits.append(split)
                elif any(os.path.splitext(f)[1].lower() in info['image_extensions'] for f in os.listdir(split_dir)):
                     # Maybe images are directly in split folder (less common for YOLO but possible)
                     found_splits.append(split)
        
        if found_splits:
            info['type'] = 'split'
            info['splits'] = found_splits
            for split in found_splits:
                count = 0
                # Look in images subdir first
                target_dir = os.path.join(dataset_path, split, 'images')
                if not os.path.isdir(target_dir):
                    target_dir = os.path.join(dataset_path, split)
                
                for root, _, files in os.walk(target_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in info['image_extensions']:
                            count += 1
                info['split_counts'][split] = count
                info['total_images'] += count
        else:
            # Check for flat structure (images/ labels/ or just images)
            count = 0
            # If 'images' folder exists, count there
            img_dir = os.path.join(dataset_path, 'images')
            target_dir = img_dir if os.path.isdir(img_dir) else dataset_path
            
            for root, _, files in os.walk(target_dir):
                 for f in files:
                    if os.path.splitext(f)[1].lower() in info['image_extensions']:
                        count += 1
            info['total_images'] = count
            
        return info

    def process_dataset(self, dataset_path, output_name, multiplier, config, selected_splits=None, progress_callback=None):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        parent_dir = os.path.dirname(dataset_path.rstrip(os.sep))
        output_dir = os.path.join(parent_dir, output_name)
        
        # Collect images based on structure and selection
        images_to_process = []
        
        # Scan first to know structure (or rely on what UI passed? safer to re-scan or just walk relevant paths)
        # If selected_splits is provided, we assume split structure.
        
        if selected_splits:
            # Split structure
            for split in selected_splits:
                # Try standard YOLO path: split/images
                split_img_dir = os.path.join(dataset_path, split, 'images')
                if not os.path.isdir(split_img_dir):
                    split_img_dir = os.path.join(dataset_path, split) # fallback
                
                if os.path.isdir(split_img_dir):
                    for root, dirs, files in os.walk(split_img_dir):
                        for file in files:
                            if os.path.splitext(file)[1].lower() in image_extensions:
                                images_to_process.append(os.path.join(root, file))
        else:
            # Flat or auto-detect all
            # If flat, just walk everything (excluding output dir if it ends up inside, but we write to sibling)
            for root, dirs, files in os.walk(dataset_path):
                # Avoid recursing into the output directory if it happens to be created inside (though we aim for sibling)
                if os.path.abspath(output_dir).startswith(os.path.abspath(root)):
                    continue
                    
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        images_to_process.append(os.path.join(root, file))
                    
        total_ops = len(images_to_process) * multiplier
        current_op = 0
        if progress_callback: progress_callback(0, total_ops, f"Starting... Found {len(images_to_process)} images.")
        
        for img_path in images_to_process:
            # Find label path logic (needs to be robust for splits)
            # Standard YOLO split: dataset/train/images/img.jpg -> dataset/train/labels/img.txt
            # Flat: dataset/images/img.jpg -> dataset/labels/img.txt
            
            p = Path(img_path)
            parts = list(p.parts)
            label_path = None
            
            # 1. Swap images -> labels
            try:
                # Find the right-most 'images' occurrence to swap
                if 'images' in parts:
                    # rindex equivalent
                    idx = len(parts) - 1 - parts[::-1].index('images')
                    parts[idx] = 'labels'
                    pot = Path(*parts).with_suffix('.txt')
                    if pot.exists(): label_path = str(pot)
            except: pass
            
            # 2. Same dir
            if not label_path:
                pot = p.with_suffix('.txt')
                if pot.exists(): label_path = str(pot)
            
            # Load Image
            image = cv2.imread(img_path)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process N times
            for i in range(multiplier):
                if not label_path:
                    pipeline = self.get_pipeline(config, has_bboxes=False, has_keypoints=False)
                    res = pipeline(image=image)
                    aug_img = res['image']
                    final_bboxes, final_polygons = [], []
                    final_bbox_classes, final_poly_classes = [], []
                else:
                    aug_img, final_bboxes, final_bbox_classes, final_polygons, final_poly_classes = self.apply_augmentation(image, label_path, config)
                
                # Determine relative path for output to maintain structure
                # If split: dataset/train/images/img.jpg -> output/train/images/img_aug_0.jpg
                rel_path = os.path.relpath(img_path, dataset_path)
                out_img_path = os.path.join(output_dir, rel_path)
                
                suffix = f"_aug_{i}" if multiplier > 1 else "_aug"
                name, ext = os.path.splitext(out_img_path)
                final_img_path = f"{name}{suffix}{ext}"
                
                os.makedirs(os.path.dirname(final_img_path), exist_ok=True)
                cv2.imwrite(final_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                # Save Label
                final_lines = []
                for b, c in zip(final_bboxes, final_bbox_classes):
                    final_lines.append(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                for p, c in zip(final_polygons, final_poly_classes):
                    pts_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in p])
                    final_lines.append(f"{c} {pts_str}")
                
                # Logic to determine output label path
                # mirroring input structure
                final_label_path = None
                
                # If we found a label path originally, we try to mirror that relative structure?
                # Or just assume standard YOLO structure in output?
                # Simplest: apply same relative path transformation to label path if it exists
                if label_path:
                    l_rel = os.path.relpath(label_path, dataset_path)
                    l_base = os.path.join(output_dir, l_rel)
                    ln, le = os.path.splitext(l_base)
                    final_label_path = f"{ln}{suffix}{le}"
                else:
                     # Infer from output image path
                     if 'images' in final_img_path.split(os.sep):
                         p_parts = list(Path(final_img_path).parts)
                         try:
                             idx = len(p_parts) - 1 - p_parts[::-1].index('images')
                             p_parts[idx] = 'labels'
                             final_label_path = str(Path(*p_parts).with_suffix('.txt'))
                         except: pass

                if final_label_path:
                    os.makedirs(os.path.dirname(final_label_path), exist_ok=True)
                    if final_lines or label_path: # Write if we have content OR if original existed (empty file)
                         with open(final_label_path, 'w') as f:
                            f.write('\n'.join(final_lines))

                current_op += 1
                if progress_callback and current_op % 10 == 0:
                     progress_callback(current_op, total_ops, f"Processed {current_op}/{total_ops}")

        # Copy data.yaml if exists and update it?
        # For now just copy it.
        src_yaml = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(src_yaml):
             try:
                shutil.copy(src_yaml, os.path.join(output_dir, 'data.yaml'))
             except: pass

        if progress_callback: progress_callback(total_ops, total_ops, "Done.")
        return output_dir
