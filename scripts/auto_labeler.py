"""
Auto Labeler Script for YOLOv8

This script performs inference on a set of images using a YOLOv8 model to detect objects and segment them. 
It then saves the images and their corresponding labels into different directories based on the detected classes.

Functionality:
1. Loads a YOLOv8 model and performs segmentation or detection on images.
2. Categorizes images into 'bundles' and 'single' based on detected object classes OR saves all in a single folder:
   - 'Bundle' class (0) images are saved in the 'bundles' directory.
   - 'Single-Item' class (1) images are saved in the 'single' directory.
   - If --single-folder is used, all images are saved in 'labeled/images' and 'labeled/labels'.
3. Optionally, saves images with drawn masks or bounding boxes indicating detected objects.
4. If masks are available, simplifies the segmentation mask polygons to reduce the number of points while preserving the shape.
5. If masks are not available, generates bounding box labels in the YOLOv8 format.
6. Outputs the processed images and corresponding label files in the respective directories.
7. Supports reading .png, .jpg, and .tif images.
8. Allows optional resizing of images to specified dimensions before passing them to the model.
9. Always saves images in the individual folders as `.jpg`.
10. Displays class labels (text) next to detected objects in masked images.

Usage:
    python3 auto_labeler.py <image_dir> <model_path> [--save-masked-images] [--epsilon <value>] [--resize-height <height>] [--resize-width <width>] [--single-folder]

Arguments:
    image_dir: Path to the directory containing images (PNG, JPG, or TIF format).
    model_path: Path to the YOLOv8 .pt model file.
    --save-masked-images: Optional flag to save images with drawn masks or bounding boxes.
    --epsilon: Optional value for polygon simplification (default: 0.01).
    --resize-height: Optional height to resize the images before passing them to the model.
    --resize-width: Optional width to resize the images before passing them to the model.
    --single-folder: Optional flag to save all images in a single 'labeled' folder instead of separating by class.

Example:
    python3 auto_labeler.py /path/to/images /path/to/model.pt --save-masked-images --epsilon 0.02 --resize-height 640 --resize-width 480 --single-folder
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2
import numpy as np

class YOLOInference:
    def __init__(self, model_path, image_dir, confidence=0.5, iou=0.45, save_masked_images=False, epsilon=0.01, resize_height=None, resize_width=None, single_folder=False, class_names_map=None, output_labels_dir=None, output_masked_dir=None, save_labeled_images=True):
        self.model_path_ = model_path
        self.image_dir_ = Path(image_dir)
        self.confidence_ = confidence
        self.iou_ = iou
        self.save_masked_images_ = save_masked_images
        self.epsilon_ = epsilon
        self.resize_height_ = resize_height
        self.resize_width_ = resize_width
        self.single_folder_ = single_folder
        self.save_labeled_images_ = save_labeled_images
        self.specific_images_ = None  # List of specific filenames to process

        # Define parent directory and output directories for images and labels
        self.parent_dir_ = self.image_dir_.parent
        dataset_name = self.image_dir_.name
        
        if self.single_folder_:
            # Single folder structure
            if save_labeled_images:
                self.labeled_images_dir_ = self.parent_dir_ / f"{dataset_name}_labeled/images"
                self.labeled_images_dir_.mkdir(parents=True, exist_ok=True)
            else:
                 self.labeled_images_dir_ = None

            if output_labels_dir:
                self.labeled_labels_dir_ = Path(output_labels_dir)
            else:
                self.labeled_labels_dir_ = self.parent_dir_ / f"{dataset_name}_labeled/labels"
            self.labeled_labels_dir_.mkdir(parents=True, exist_ok=True)

            if output_masked_dir:
                self.masked_images_dir_ = Path(output_masked_dir)
            else:
                self.masked_images_dir_ = self.parent_dir_ / f"{dataset_name}_masked_images"
            if self.save_masked_images_:
                self.masked_images_dir_.mkdir(parents=True, exist_ok=True)
            
        else:
            # Separate folders by class
            # Note: Custom output dirs not fully implemented for bundle/single split mode yet as per current requirements
            self.bundle_images_dir_ = self.parent_dir_ / f"{dataset_name}_bundles/images"
            self.bundle_labels_dir_ = self.parent_dir_ / f"{dataset_name}_bundles/labels"
            self.single_images_dir_ = self.parent_dir_ / f"{dataset_name}_single/images"
            self.single_labels_dir_ = self.parent_dir_ / f"{dataset_name}_single/labels"
            
            if output_masked_dir:
                self.masked_images_dir_ = Path(output_masked_dir)
            else:
                self.masked_images_dir_ = self.parent_dir_ / f"{dataset_name}_masked_images"
            
            # Ensure all output directories exist
            for dir in [self.bundle_images_dir_, self.bundle_labels_dir_, self.single_images_dir_, self.single_labels_dir_]:
                dir.mkdir(parents=True, exist_ok=True)
            if self.save_masked_images_:
                self.masked_images_dir_.mkdir(parents=True, exist_ok=True)
        
        # Define class names mapping
        if class_names_map:
            self.class_names_ = class_names_map
        else:
            self.class_names_ = {
                0: "Bundle",
                1: "Single-Item"
            }
        
        # Load model with explicit task definition
        # self.model_ = YOLO(self.model_path_, task='segment' if 'segment' in self.model_path_ else 'detect')
        self.model_ = YOLO(self.model_path_)
        self.device_ = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def get_text_position(self, mask=None, box=None):
        """Calculate optimal text position based on mask or bounding box"""
        if mask is not None:
            # For masks, find the top-left corner of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the bounding rectangle of the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y - 10 if y > 20 else y + h + 20)
        elif box is not None:
            # For bounding boxes, use top-left corner
            x1, y1, x2, y2 = map(int, box[:4])
            return (x1, y1 - 10 if y1 > 20 else y2 + 20)
        
        return (10, 30)  # Default position

    def draw_masks(self, image, masks, classes, boxes=None):
        """Draw masks with class labels"""
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            class_id = int(cls)
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            text_color = (255, 255, 255)  # White text
            
            # Prepare mask
            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
            
            # Get class name and confidence
            class_name = self.class_names_.get(class_id, f"Class_{class_id}")
            
            # Calculate text position
            text_pos = self.get_text_position(mask=mask)
            
            # Draw text background rectangle for better visibility
            text = class_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(image, 
                         (text_pos[0] - 2, text_pos[1] - text_height - 2),
                         (text_pos[0] + text_width + 2, text_pos[1] + baseline + 2),
                         color, -1)
            
            # Draw text
            cv2.putText(image, text, text_pos, font, font_scale, text_color, thickness)
            
        return image

    def draw_bounding_boxes(self, image, boxes, classes):
        """Draw bounding boxes with class labels"""
        for box, cls in zip(boxes, classes):
            class_id = int(cls)
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Bundle, Red for Single-Item
            text_color = (255, 255, 255)  # White text
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Get class name
            class_name = self.class_names_.get(class_id, f"Class_{class_id}")
            
            # Calculate text position
            text_pos = self.get_text_position(box=box)
            
            # Draw text background rectangle for better visibility
            text = class_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(image, 
                         (text_pos[0] - 2, text_pos[1] - text_height - 2),
                         (text_pos[0] + text_width + 2, text_pos[1] + baseline + 2),
                         color, -1)
            
            # Draw text
            cv2.putText(image, text, text_pos, font, font_scale, text_color, thickness)
            
        return image

    def simplify_contour(self, contour):
        epsilon = self.epsilon_ * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def run_inference(self, callback=None):
        total_images = 0
        bundle_count = 0
        single_count = 0
        labeled_count = 0

        # Iterate over images in the image directory
        image_paths = list(self.image_dir_.glob('*.png')) + list(self.image_dir_.glob('*.jpg')) + list(self.image_dir_.glob('*.tif'))
        
        # Filter for specific images if requested
        if self.specific_images_:
            image_paths = [p for p in image_paths if p.name in self.specific_images_]
            
        total_images = len(image_paths)
        
        for i, img_file in enumerate(tqdm(image_paths, desc="Processing Images")):
            if callback:
                callback(i, total_images)
            try:
                # Read the image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"WARNING ⚠️ Unable to read image {img_file}")
                    continue

                # Optionally resize the image
                if self.resize_height_ is not None and self.resize_width_ is not None:
                    img = cv2.resize(img, (self.resize_width_, self.resize_height_))

                # Perform inference on each image
                results = self.model_.predict(
                    source=img,
                    verbose=False,
                    stream=False,
                    conf=self.confidence_,
                    iou=self.iou_,
                    device=self.device_
                )
            except Exception as e:
                print(f"WARNING ⚠️ Image Read Error {img_file}: {e}")
                continue
            
            results = results[0].cpu()
            contains_bundle = False
            all_single = True
            label_data = []

            # If masks are available, generate mask labels
            if hasattr(results, 'masks') and results.masks:
                for box_data, mask_data in zip(results.boxes, results.masks.data):
                    class_id = int(box_data.cls)
                    if class_id == 0:  # Bundle class
                        contains_bundle = True
                        all_single = False
                    elif class_id == 1:  # Single-Item class
                        all_single = True
                    
                    # Extract the segmentation mask points
                    mask = mask_data.cpu().numpy().astype(np.uint8)
                    mask = cv2.resize(mask, (results.orig_shape[1], results.orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        simplified_contour = self.simplify_contour(contour)
                        if len(simplified_contour) < 3:
                            continue  # Skip contours that are too small to be valid polygons
                        normalized_points = [(point[0][0] / results.orig_shape[1], point[0][1] / results.orig_shape[0]) for point in simplified_contour]
                        label_data.append((class_id, normalized_points))

            # If no masks, generate bounding box labels
            elif hasattr(results, 'boxes') and results.boxes:
                for box_data in results.boxes:
                    class_id = int(box_data.cls)
                    if class_id == 0:  # Bundle class
                        contains_bundle = True
                        all_single = False
                    elif class_id == 1:  # Single-Item class
                        all_single = True

                    # YOLOv8 format: class_id center_x center_y width height (normalized)
                    x1, y1, x2, y2 = box_data.xyxy[0]
                    width = x2 - x1
                    height = y2 - y1
                    center_x = x1 + width / 2
                    center_y = y1 + height / 2
                    normalized_bbox = [
                        class_id,
                        center_x / img.shape[1],
                        center_y / img.shape[0],
                        width / img.shape[1],
                        height / img.shape[0]
                    ]
                    label_data.append(normalized_bbox)

            # Prepare label content
            label_file_content = ""
            if hasattr(results, 'masks') and results.masks:
                for class_id, points in label_data:
                    points_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                    label_file_content += f"{int(class_id)} {points_str}\n"
            else:
                for bbox in label_data:
                    label_file_content += f"{int(bbox[0])} " + " ".join(f"{x:.6f}" for x in bbox[1:]) + "\n"
                
            # Determine the corresponding label file
            label_file_path = img_file.with_suffix('.txt')
            
            # Save image and label based on the mode
            if self.single_folder_:
                # Save all in single folder
                if self.labeled_images_dir_:
                    image_save_path = self.labeled_images_dir_ / (img_file.stem + ".jpg")
                else:
                    image_save_path = None
                    
                with open(self.labeled_labels_dir_ / label_file_path.name, 'w') as f:
                    f.write(label_file_content)
                labeled_count += 1
            else:
                # Save in separate folders by class
                if contains_bundle:
                    image_save_path = self.bundle_images_dir_ / (img_file.stem + ".jpg")
                    bundle_count += 1
                    with open(self.bundle_labels_dir_ / label_file_path.name, 'w') as f:
                        f.write(label_file_content)
                elif all_single:
                    image_save_path = self.single_images_dir_ / (img_file.stem + ".jpg")
                    single_count += 1
                    with open(self.single_labels_dir_ / label_file_path.name, 'w') as f:
                        f.write(label_file_content)

            # Save the image as .jpg
            if image_save_path:
                cv2.imwrite(str(image_save_path), img)

            # Optionally save the masked image or with bounding boxes
            if self.save_masked_images_:
                # Create a copy of the image for visualization
                viz_img = img.copy()
                
                if hasattr(results, 'masks') and results.masks:
                    masked_img = self.draw_masks(viz_img, results.masks.data, results.boxes.cls, results.boxes.xyxy)
                else:
                    masked_img = self.draw_bounding_boxes(viz_img, results.boxes.xyxy, results.boxes.cls)
                
                masked_img_path = self.masked_images_dir_ / (img_file.stem + ".jpg")
                cv2.imwrite(str(masked_img_path), masked_img)

        # Print the summary
        print(f"Total images processed: {total_images}")
        if self.single_folder_:
            print(f"Images saved in 'labeled' folder: {labeled_count}")
        else:
            print(f"Images saved in 'bundles': {bundle_count}")
            print(f"Images saved in 'single': {single_count}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images")
    parser.add_argument("model_path", type=str, help="Path to the YOLOv8 .pt model")
    parser.add_argument("--save-masked-images", action="store_true", help="Save images with drawn masks or bounding boxes")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon value for polygon simplification")
    parser.add_argument("--resize-height", type=int, help="Height to resize images before processing")
    parser.add_argument("--resize-width", type=int, help="Width to resize images before processing")
    parser.add_argument("--single-folder", action="store_true", help="Save all images in a single 'labeled' folder instead of separating by class")

    args = parser.parse_args()

    yolo_inference = YOLOInference(
        model_path=args.model_path,
        image_dir=args.image_dir,
        confidence=0.5,
        save_masked_images=args.save_masked_images,
        epsilon=args.epsilon,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
        single_folder=args.single_folder
    )

    yolo_inference.run_inference()