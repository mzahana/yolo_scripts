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

## 🛠 Features & Functionality

### 📂 Dataset Management
- **Dataset Selection**: Enter a local path to your raw images. The UI automatically validates the path and counts images.
- **Project Persistence**: All settings (dataset path, model choice, crop coordinates, resize dimensions) are automatically saved to `<dataset_base>_yolo_project_config.json`.
- **Smart Auto-Detection**: When loading a dataset, the app automatically detects existing `_processed`, `_labeled`, or `masked_images` folders and restores the project state.

### ✂️ Pre-processing
- **Interactive Cropping**: Select a region of interest on a sample image using a visual interface. The crop coordinates are applied to the entire dataset.
- **Bulk Resizing**: Set uniform dimensions (e.g., 640x640) for all images.
- **Batch Processing**: High-performance bulk processing of the entire directory.

### 🤖 Auto-Labeling
- **YOLOv8/v11 Support**: Use any local `.pt` weight file for automated labeling.
- **Customizable Confidence**: Fine-tune the detection threshold.
- **Masked Image Generation**: Automatically generates masked versions of labeled images for quick verification.
- **YOLO Config Generation**: Automatically creates a `data.yaml` file compatible with standard YOLO training.

### ✏️ Manual Annotation
- **Built-in Editor**: A full-featured manual annotation tool with bounding box support.
- **SAM Integration**: (Segment Anything Model) support for assisted annotation (requires SAM weights).
- **Jump to Annotation**: Directly open any image from the Verification gallery in the manual editor for quick corrections.

### ⚙️ Data Processing Tools
- **Merge Datasets**: Combine multiple YOLO datasets into one, automatically handling class remapping and file name collisions.
- **Extract by Class**: Filter a dataset to only include specific object classes.
- **Split Dataset**: Divide a dataset into multiple shuffled splits for training/validation.
- **Extract Empty**: Isolate images that contain no detections (useful for negative samples).

### ✅ Verification & Quality Control
- **Verification Gallery**: Browse masked images side-by-side with object counts.
- **Filtering**: Quickly filter the gallery by object class.
- **Flag & Copy**: Use the filter mechanism to copy high-quality original or processed images to a "filtered" directory for final training sets.

### 📊 Analytics & Statistics
- **Class Distribution**: Interactive bar and pie charts showing the frequency of each object class.
- **Dataset Health**: Get a high-level overview of object counts and image counts across your dataset.

---

## 📖 Step-by-Step Tutorials

### 1. Project Setup
1. Open the **Dataset Info** tab.
2. Enter the full local path to your directory of raw images.
3. Click **Load Dataset**. The app will report the image count and automatically look for any previous progress.

### 2. Image Preparation (Pre-processing)
1. Navigate to the **Pre-processing** tab.
2. Use your mouse to draw a crop box on the sample image to define the region of interest.
3. (Optional) Adjust the target **Resize Width** and **Height**.
4. Click **Start Pre-processing**. A new folder suffixed with `_processed` will be created.

### 3. Automated Labeling
1. Go to the **Auto-Labeling** tab.
2. Click **Browse** to select your trained YOLO model (`.pt` file).
3. Adjust the **Confidence Threshold** (default is 0.5).
4. Click **Start Auto-Labeling**. The app will generate YOLO-format text labels and a `data.yaml` config.

### 4. Quality Control (Verification)
1. Open the **Verification** tab.
2. Browse the generated masked images to check detection accuracy.
3. Use the **Filter by Class** buttons to focus on specific objects.
4. Click the **✅ Filter** button under any image to copy it to a "filtered" directory for your final training set.

### 5. Manual Refinement
1. If an image in the Verification gallery needs better labeling, click the **✏️ Edit** button to "Jump to Annotation".
2. You can also manually open images in the **Manual Annotation** tab.
3. Use the shortcut keys or the sidebar to add/adjust bounding boxes.
4. (Assisted) If a SAM model is loaded, use point-clicks to generate masks that automatically convert to bounding boxes.

### 6. Data Management Tools
- **Merge**: In the **Data Processing** tab, select two or more datasets and an output folder to consolidate them.
- **Extract by Class**: Select a labeled dataset, pick your target classes, and click **Extract** to create a subset.
- **Split**: Use the **Split Dataset** tool to divide your data into training/validation sets (randomly shuffled).
- **Extract Empty**: Use this to find and isolate images with no detections to use as background/negative samples.

### 7. Insights
1. Open the **Statistics** tab after labeling.
2. View the **Class Distribution** bar chart to see which objects are most frequent.
3. Check the **Pie Chart** for a percentage breakdown of your dataset composition.

---

## 🏗 Project Structure

- `backend/`: FastAPI server providing REST APIs for all heavy-lifting tasks.
- `frontend/`: React application using Vite, Recharts for analytics, and React Crop for interactive tools.
- `scripts/`: Powering the backend with optimized Python logic for YOLO operations.

## 🔧 Troubleshooting
- **Path Issues**: Ensure the backend has read/write permissions for the dataset directories.
- **Model Loading**: Verify your `.pt` files are compatible with the Ultralytics framework.
- **Environment**: If scripts fail, ensure the `ultralytics-venv` is activated.
