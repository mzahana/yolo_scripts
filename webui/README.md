# YOLO Dataset Web UI

This subproject provides a modern web interface for pre-processing and auto-labeling YOLO image datasets.

## Structure
- `backend/`: FastAPI server integrating `crop_resize.py` and `auto_labeler.py`.
- `frontend/`: Vite-based React application for the user interface.

## Prerequisites
- Python 3.8+
- Node.js & npm
- Virtual Environment (recommended): `/home/mzahana/venv/ultralytics-venv`

## How to Run

### 1. Backend
Open a terminal, activate your virtual environment, and run the backend:
```bash
source /home/mzahana/venv/ultralytics-venv/bin/activate
cd webui/backend
pip install -r requirements.txt
python main.py
```

### 2. Frontend
Open another terminal and run the frontend:
```bash
cd webui/frontend
npm install
npm run dev
```
The UI will be available at `http://localhost:3000`.

## Features
- **Dataset Selection & Persistence**: Provide the local path to your raw images. The app automatically saves your progress (paths, model choice) to `yolo_project_config.json`, so you can reload your project instantly.
- **Interactive Cropping**: Select a region on a sample image to apply cropping to the entire dataset.
- **Bulk Processing**: Efficiently resize and crop all images.
- **Auto-Labeling**: Use any local YOLOv8/v11 model to automatically generate labels.
- **YOLO Config Generation**: Automatically creates a `data.yaml` file compatible with YOLO training.
- **Verification Gallery**: Inspect masked images side-by-side with object counts. filter mechanism allows for quickly filtering out bad images and copying the original or processed version to a separate folder.
- **Dataset Statistics**: View detailed analytics including object class distribution and counts.
- **Auto-Detection**: The app automatically detects existing `_processed` folders and labeled data, populating the UI without manual reconfiguration.
