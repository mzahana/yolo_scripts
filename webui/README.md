# YOLO Dataset Web UI

This project provides a modern, high-performance web interface for managing, pre-processing, and auto-labeling YOLO image datasets. It integrates several backend scripts into a unified workflow, making it easier to prepare data for training.

## 🚀 Quick Start

### 1. Prerequisite: Virtual Environment
It is highly recommended to use the project's dedicated virtual environment:
```bash
source /home/mzahana/venv/ultralytics-venv/bin/activate
```

### 2. Run the Backend (FastAPI)
```bash
cd webui/backend
pip install -r requirements.txt
python main.py
```

### 3. Run the Frontend (React + Vite)
```bash
cd webui/frontend
npm install
npm run dev
```
Access the UI at: `http://localhost:3000`

---

## 🏗 Project Structure & Workflow

The web UI enforces a structured project layout to ensure data integrity and traceability.

- **Raw Images**: `[project_name]_raw_images` - The immutable source of truth.
- **Processed Images**: `[project_name]_processed_images` - Cropped/Resized versions used for labeling.
- **Annotations**: `annotations/` - Central repository for project-wide labels.
- **Masked Images**: `[project_name]_masked_images` - Visual verification of labels.
- **Generated Datasets**: `[project_name]_datasets/` - Exported YOLO-format datasets (split into train/val).

## 🛠 Features & Functionality

### 1. 🏠 Project Management
- **Create/Load Project**: Initialize a new project with a unified config file (`project_config.json`) or load an existing one.
- **State Persistence**: All settings (crop, resize, model) and directory paths are persisted automatically.

### 2. ✂️ Pre-processing
- **Interactive Cropping**: precise crop tool (using percentage-based coordinates) to define the region of interest for all images.
- **Bulk Resizing**: Standardize image dimensions for training (e.g., 640x640).
- **Batch Processing**: Generates `_processed_images` from the raw source.

### 3. 🤖 Auto-Labeling
- **YOLO/SAM Integration**: Use trained `.pt` models to automatically generate bounding boxes.
- **Confidence Control**: Adjust thresholds to balance precision and recall.
- **Single Image Test**: Verify model performance on a sample before processing the batch.
- **Mask Generation**: Creates visualization overlays for verification.

### 4. ✏️ Manual Annotation
- **Full-Featured Editor**: Add, move, resize, and delete bounding boxes.
- **SAM Assisted**: Use "Segment Anything" to generate masks/boxes with point clicks.
- **Jump-to-Edit**: Seamlessly transition from inspection/verification to editing specific images.

### 5. 🔍 Data Inspection (Verification)
- **Visual Inspection**: Review `masked_images` to verify label accuracy.
- **Filter Mechanism**: Copy valid/important images to a flat `_filtered_images` certified folder.
- **Direct Edit**: One-click jump to correct annotations for any problematic image.

### 6. 📊 Statistics
- **Dual Scope**: View analytics for the **Entire Project** (Annotations + Processed Images) OR specific **Generated Datasets**.
- **Metrics**: Class distribution charts, total object counts, and empty image counts.
- **Extract Empty**: One-click extraction of unannotated images to `[project_name]_empty_images` (useful for negative mining).

### 7. 📦 Dataset Generation
- **Create YOLO Dataset**: Export processed images and annotations into a strictly formatted YOLO dataset (`images/`, `labels/`, `data.yaml`).
- **Strategies**: Support for "All" (export all valid pairs) or "Random" (future support).
- **Dataset List**: Manage and track all generated versions (e.g., `v1`, `v2`).

---

## 🔧 backend/main.py Key Components

- **Project Manager**: Handles directory structure creation and config persistence.
- **Heuristic Discovery**: Intelligently discovers `labels`, `images`, and config files even if folder structures vary slightly.
- **Analysis Engine**: Calculates statistics dynamically based on the selected scope.

## 🔧 Troubleshooting
- **Missing Images in Stats?**: Ensure you have run **Pre-processing** and **Auto-Labeling**. The "Entire Project" stats rely on the `_processed_images` folder being present.
- **Crop Tool Offset?**: The crop tool uses percentage-based coordinates to handle display scaling. Use the inputs or drag handles to set precise crops.
- **Path Errors**: The system relies on `project_config.json`. If you manually rename folders, ensure you update the config file (or use the provided Refactoring tools if available).
