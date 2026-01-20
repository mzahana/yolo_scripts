# yolo_scripts

This repository contains a collection of Python scripts for working with YOLO (You Only Look Once) models, supported by the [Ultralytics](https://ultralytics.com/) framework. These scripts are designed to be flexible and can be used with any YOLO version compatible with the `ultralytics` package.

# Installation

To use these scripts, you need to have Python 3 installed, along with the `ultralytics` package and other dependencies.

```bash
pip install ultralytics torch numpy matplotlib seaborn pyyaml
```

# Ultralytics YOLO Usage

The `ultralytics` package provides a command-line interface (CLI) for training, validating, and running inference with YOLO models.

## Training

You can train a YOLO model from scratch or by using a pretrained model.

```bash
# Train a new model from a YAML configuration
yolo train data=your_dataset.yaml model=yolov8n.yaml epochs=100 imgsz=640

# Train from a pretrained model
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## Prediction

Run inference on images or videos using a trained model.

```bash
# Predict using an official model
yolo predict model=yolov8n.pt source='path/to/your/image.jpg'

# Predict using a custom trained model
yolo predict model='path/to/your/best.pt' source='path/to/your/image.jpg'
```

## Validation

Validate your model's performance on a dataset.

```bash
yolo val model='path/to/your/best.pt' data=your_dataset.yaml
```

# Scripts

This repository includes a variety of scripts to help with your YOLO projects. For more detailed information on any script, feel free to ask.

*   **`auto_labeler.py`**: (TODO: Add description)
*   **`bn_recalibrate.py`**: (TODO: Add description)
*   **`check_yolo_annotation_types.py`**: (TODO: Add description)
*   **`convert_yolo2efficient.py`**: Converts a YOLO semantic segmentation dataset to the EfficientViT format. It transforms YOLO polygon annotations into segmentation masks.
*   **`crop_resize.py`**: This script crops and/or resizes images in a directory.
*   **`data_augmentation.py`**: This script performs data augmentation for instance segmentation. It extracts objects from images based on polygon annotations, applies various augmentations (rotation, blur, scaling, contrast), and places them on a background image.
*   **`dataset_analytics.py`**: Analyzes a YOLO dataset, providing statistics such as image counts, object counts per class, and heatmaps of object locations.
*   **`display_and_filter_images.py`**: This script provides a GUI to display and compare detection results with raw images side-by-side. It allows the user to navigate through images and save selected raw images to a specified directory.
*   **`draw_annotations_yolov4.py`**: This script draws bounding box annotations on images from a dataset in YOLOv4 format (where annotations are in a single `_annotations.txt` file).
*   **`export_segmentation_dataset_to_classification.py`**: This script converts an instance segmentation dataset to a classification dataset. It extracts the bounding box of each segmented object, crops it, and saves it into a class-specific folder.
*   **`export.py`**: This script provides an example of how to export a YOLO model to the TensorRT `.engine` format. The recommended way to export models is to use the `yolo` command-line tool.
*   **`extract_yolo_by_class.py`**: (TODO: Add description)
*   **`filter_labeled_dataset.py`**: This script filters a YOLO dataset based on a given set of class IDs. It copies the images and labels that contain the specified classes to a new directory.
*   **`hyperparameter_tuning.py`**: (TODO: Add description)
*   **`image_saver_gui.py`**: (TODO: Add description)
*   **`measure_convexity.py`**: (TODO: Add description)
*   **`merge_yolo_datasets.py`**: (TODO: Add description)
*   **`polygon_to_bbox_converter.py`**: (TODO: Add description)
*   **`sample_yolo_dataset.py`**: (TODO: Add description)
*   **`seg_to_bbx_dataset.py`**: (TODO: Add description)
*   **`split_dataset.py`**: (TODO: Add description)
*   **`split_parcels.py`**: (TODO: Add description)
*   **`yolo_advanced_inference.py`**: (TODO: Add description)
*   **`yolo_inference.py`**: Performs object detection on a directory of images using a trained YOLO model and saves images that contain objects of specified classes.
*   **`yolo_visualizer.py`**: (TODO: Add description)

## Script Usage Examples

### `auto_labeler.py`

This script is run from the command line.

```bash
python3 scripts/auto_labeler.py /path/to/images /path/to/model.pt --save-masked-images --epsilon 0.02 --resize-height 640 --resize-width 480 --single-folder
```

**Arguments:**

*   `image_dir`: Path to the directory containing images.
*   `model_path`: Path to the YOLO .pt model file.
*   `--save-masked-images`: Optional flag to save images with drawn masks or bounding boxes.
*   `--epsilon`: Optional value for polygon simplification (default: 0.01).
*   `--resize-height`: Optional height to resize the images.
*   `--resize-width`: Optional width to resize the images.
*   `--single-folder`: Optional flag to save all images in a single 'labeled' folder.

### `yolo_inference.py`

This script can be used as a class within another Python script. To use it, you can import the `YOLOInference` class and instantiate it with the required parameters.

```python
from scripts.yolo_inference import YOLOInference

desired_classes = ['class1', 'class2']
yolo_inference = YOLOInference(
    model_path='/path/to/your/model.pt',
    input_dir='/path/to/your/input/directory',
    output_dir='/path/to/your/output/directory',  # Optional
    confidence=0.5,
    desired_classes=desired_classes
)

yolo_inference.run_inference()
```

### `dataset_analytics.py`

This script is run from the command line and requires the path to the dataset directory.

```bash
python scripts/dataset_analytics.py --data_dir /path/to/your/dataset
```

### `bn_recalibrate.py`

This script is run from the command line.

```bash
python scripts/bn_recalibrate.py --model /path/to/your/model.pt --images /path/to/new/images
```

**Arguments:**

*   `--model`: Path to the trained YOLO .pt model.
*   `--images`: Path to the folder with the new images.
*   `--imgsz`: Inference size (default: 640).
*   `--batch`: Batch size (default: 16).
*   `--limit`: Maximum number of images to use (default: 1000).
*   `--save`: Output path for the recalibrated weights (optional).

### `check_yolo_annotation_types.py`

This script is run from the command line.

```bash
# Scan a dataset and report annotation types
python scripts/check_yolo_annotation_types.py /path/to/your/dataset

# Scan and list background images
python scripts/check_yolo_annotation_types.py /path/to/your/dataset --list-background

# Remove all bounding box annotations from the dataset
python scripts/check_yolo_annotation_types.py /path/to/your/dataset --remove-type bbox --apply --yes
```

**Arguments:**

*   `dataset_root`: Path to the dataset folder.
*   `--splits`: Comma-separated subset of splits to process (e.g., `train,valid`).
*   `--list-background`: List background image file paths.
*   `--remove-type`: Set to `bbox` or `seg` to remove that type of annotation.
*   `--delete-mixed`: Also delete mixed label files when using `--remove-type`.
*   `--apply`: Actually perform the deletion (dry run by default).
*   `--yes`: Skip the confirmation prompt when using `--apply`.


# References
* [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
* [Tips for Best Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
* [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)