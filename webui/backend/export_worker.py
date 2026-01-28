import sys
import json
import argparse
from ultralytics import YOLO

def run_export(config):
    print(f"Starting export with config: {config}")
    try:
        # Load model
        model_path = config.get("model")
        if not model_path:
            raise ValueError("Model path is required for export")
            
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        # Prepare arguments
        # Filter out 'model' and ensure correct types
        export_args = {k: v for k, v in config.items() if k != "model"}
        
        # Ensure 'format' is present
        if "format" not in export_args:
            raise ValueError("Export format is required")

        # Run export
        # Ultralytics prints to stdout/stderr automatically
        print(f"Exporting to {export_args['format']} with args: {export_args}")
        export_path = model.export(**export_args)
        
        # Print the success message and path for the manager to parse
        print(f"EXPORT_SUCCESS: {export_path}")
        
    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="JSON string of export configuration")
    args = parser.parse_args()
    
    try:
        config = json.loads(args.config)
        run_export(config)
    except json.JSONDecodeError:
        print("Invalid JSON config provided", file=sys.stderr)
        sys.exit(1)
