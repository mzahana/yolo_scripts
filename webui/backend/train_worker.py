import sys
import json
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import threading

def run_training(config):
    print(f"Starting training with config: {config}")
    try:
        # Change CWD to dataset directory so checks/downloads happen there
        data_config_path = config.get("data")
        dataset_dir = None
        if data_config_path:
            try:
                # data_config_path is likely /path/to/dataset/data.yaml
                dataset_dir = Path(data_config_path).parent.absolute()
                print(f"Changing working directory to: {dataset_dir}")
                os.chdir(dataset_dir)
            except Exception as e:
                print(f"Warning: Failed to change working directory to dataset path: {e}")

        # Load model (pretrained or custom)
        model_name = config.get("model", "yolov8n.pt")
        # If it's a path to a file that exists, use it. Otherwise assume it's a model name downloaded by ultralytics
        print(f"Loading using model: {model_name}")
        model = YOLO(model_name)
        
        # Verify model path if possible
        if Path(model_name).exists():
             print(f"Model file confirm at: {Path(model_name).absolute()}")
        else:
             print(f"Model '{model_name}' (pretrained) will be downloaded to: {os.getcwd()}")

        # Prepare arguments
        # Filter out arguments that are not for training or handled separately
        train_args = {k: v for k, v in config.items() if k not in ["model", "task", "mode"]}
        
        # Ensure project and name are set if not provided, to keep results organized
        if "project" not in train_args:
            # FORCE ABSOLUTE PATH based on dataset_dir if available
            task_name = config.get("task", "detect")
            if dataset_dir:
                 train_args["project"] = str(dataset_dir / "runs" / task_name)
            else:
                 train_args["project"] = f"runs/{task_name}"
        if "name" not in train_args:
            train_args["name"] = "train"
            
        # Run training
        # Ultralytics prints to stdout/stderr automatically
        results = model.train(**train_args)
        
        # Print the result directory for the manager to parse
        try:
             # results.save_dir is a Path object usually
             print(f"RESULT_DIR: {results.save_dir}")
             
             # Extract metrics
             # For detection/segmentation/pose/obb: results.box.map50, map, etc.
             # results objects have different attributes based on task
             metrics = {}
             
             # Try generic generic dictionary access if available or specific attributes
             # Ultralytics results APIs have evolved, safe attribute check
             
             # Box metrics (Detect, Segment, OBB)
             if hasattr(results, 'box') and results.box:
                 metrics['box_map50'] = results.box.map50
                 metrics['box_map'] = results.box.map
                 metrics['box_precision'] = results.box.mp
                 metrics['box_recall'] = results.box.mr
                 
             # Mask metrics (Segment)
             if hasattr(results, 'seg') and results.seg:
                 metrics['mask_map50'] = results.seg.map50
                 metrics['mask_map'] = results.seg.map
                 
             # Classify metrics
             if hasattr(results, 'top1'):
                 metrics['top1'] = results.top1
                 metrics['top5'] = results.top5
                 
             # Pose metrics
             if hasattr(results, 'pose') and results.pose:
                 metrics['pose_map50'] = results.pose.map50
                 metrics['pose_map'] = results.pose.map

             if metrics:
                 print(f"RESULT_SUMMARY: {json.dumps(metrics)}")
                 
        except Exception as e:
             # Fallback if extraction fails
             print(f"Error extracting metrics: {e}")
             pass
        
        print("Training completed successfully.")
        # We could print results summary or path to best.pt here
        
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="JSON string of training configuration")
    args = parser.parse_args()
    
    try:
        config = json.loads(args.config)
        run_training(config)
    except json.JSONDecodeError:
        print("Invalid JSON config provided", file=sys.stderr)
        sys.exit(1)
