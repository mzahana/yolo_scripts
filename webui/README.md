# YOLO Dataset Web UI

This project provides a modern, high-performance web interface for managing, pre-processing, and auto-labeling YOLO image datasets. It integrates several backend scripts into a unified workflow, making it easier to prepare data for training.

## 🚀 Getting Started

### 1. Prerequisites & Installation

Before running the application, ensure you have the necessary system dependencies installed.

#### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip nodejs npm
```

### 2. Automated Installation (Recommended)
You can use the provided helper script to install all dependencies and optionally create a virtual environment (`yolovenv`).

```bash
cd webui
./install.sh
```
Follow the on-screen prompts.

### 3. Manual Installation
If you prefer to install dependencies manually:

#### A. Set up Virtual Environment
It is **highly recommended** to use a python virtual environment to manage dependencies and avoid conflicts.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

#### B. Install Backend Dependencies
With your virtual environment activated, install the required Python packages.

```bash
cd webui/backend
pip install -r requirements.txt
```

#### C. Install Frontend Dependencies
Install the Node.js packages for the React frontend.

```bash
cd webui/frontend
npm install
```

### 4. Run the Application

You can run the backend and frontend separately, or use the helper script.

#### Option A: Using the Helper Script (Recommended)
This script manages both services for you.

```bash
cd webui
./webui.sh start --port 4000
```
Then access the UI at `http://localhost:4000`.

#### Option B: Manual Start

**Terminal 1 (Backend):**
```bash
source venv/bin/activate
cd webui/backend
python main.py
```

**Terminal 2 (Frontend):**
```bash
cd webui/frontend
npm run dev
```
Then access the UI at `http://localhost:3000`.

### 5. Remote Access
To access the WebUI from a different machine (e.g., accessing a remote server from your local laptop), you can use SSH port forwarding.

```bash
# Forward remote port 4000 to local port 4000
ssh -L 4000:localhost:4000 user@remote-server-ip
```
Now, simply open `http://localhost:4000` in your local browser.

---

## 🏗 Project Structure

The web UI enforces a structured project layout to ensure data integrity and traceability.

*   **`[project]_raw_images`**: The immutable source of truth. Original images are kept here.
*   **`[project]_processed_images`**: Cropped and resized versions used for labeling and training.
*   **`annotations/`**: Central repository for project-wide labels (`.txt` files).
*   **`[project]_masked_images`**: Visual verification images with bounding boxes/masks drawn.
*   **`[project]_datasets/`**: Exported YOLO-format datasets (split into `train`/`val`/`test`).

---

## 🛠 Features & Functionality

### 1. 🏠 Project Management
*   **Create Project**: Initialize a new project from a folder of raw images or an existing split dataset.
*   **Load Project**: Resume work on any existing project. The system intelligently detects config files.
*   **State Persistence**: All settings (crop coordinates, resize dims, model paths) are saved automatically in `project_config.json`.

### 2. ✂️ Pre-processing
Prepare your raw images for labeling and training.
*   **Interactive Cropping**: Use the visual crop tool to define the region of interest. Coordinates are percentage-based for consistency.
*   **Bulk Resizing**: Standardize image dimensions (e.g., 640x640).
*   **Batch Processing**: Applies crop and resize operations to generate the `_processed_images` folder.

### 3. 🤖 Auto-Labeling
Accelerate annotation using AI models.
*   **YOLO Inference**: Run any `.pt` model to automatically detect objects and generate bounding boxes.
*   **Confidence Control**: Adjust thresholds to filter out weak detections.
*   **Single Image Test**: Dry-run settings on a sample image before processing the entire dataset.

### 4. ✏️ Manual Annotation & Labeling
*   **Mask/Box Generation**: Generate visual overlays (`masked_images`) to verify that your textual labels match the images.
*   **SAM (Segment Anything)**: Use SAM models to generate segmentation masks quickly.

### 5. 🔍 Data Inspection (Verification)
Review your dataset quality.
*   **Gallery View**: Browse through `masked_images` to visually verify labels.
*   **Filtering**:
    *   **By Class**: Show only images containing specific classes.
    *   **Empty Images**: Find images with no detections (useful for background/negative mining).
*   **Dynamic Rendering**: If static masked images don't exist, the UI can render bounding boxes on the fly.
*   **Filtered Extraction**: Select images and copy them to a "Filtered" folder for curation.

### 6. ⚙️ Dataset Tools
A suite of utilities for managing dataset structure.
*   **Merge**: Combine multiple datasets into one.
*   **Extract**: Filter specific classes (e.g., only "person" and "car") into a new dataset.
*   **Rebalance**: Redistribution of data into Train/Val/Test splits (e.g., 70/20/10) with random shuffling.
*   **Flatten**: Convert a Split dataset (train/val/test folders) into a Flat structure (all images in one folder, all labels in another).
*   **Split Stats**: Visualize the distribution of images across splits.

### 7. 🏋️ Training & Export
*   **Train Model**: Configure and start YOLO training runs directly from the UI.
*   **Model Export**: Export your trained `.pt` models to deployment formats.
    *   **Formats**: ONNX, TensorRT, OpenVINO, CoreML, TFLite.
    *   **Settings**: Configure Image Size, Batch Size, Half-Precision (FP16), Int8 Quantization, and Dynamic Axes.
    *   **Live Logs**: Monitor the export process in real-time.

### 8. 🖼 Data Augmentation
Expand your dataset using synthetic generation.
*   **Background Upload**: Upload custom background images for object placement.
*   **Preview**: Visualize augmentations (Rotation, Blur, Color Jitter, Scaling) in real-time.
*   **Generate**: Create a new augmented dataset with multiplied samples.

### 9. 💻 Web Terminal
Manage your environment directly from the browser.
*   **Integrated Shell**: Full-featured terminal emulator using `xterm.js`.
*   **Real-time Interaction**: Execute system commands, run scripts, and manage processes via WebSockets.
*   **Auto-Resize**: Terminal responsive to window size changes.
*   **Direct Connection**: Connects directly to the backend process for low-latency interaction.

---

## 🔧 Troubleshooting

*   **Missing Images in Stats?**: Ensure you have run **Pre-processing** and **Auto-Labeling**. The "Entire Project" stats rely on the `_processed_images` folder.
*   **Crop Tool Offset?**: The crop tool uses percentage-based coordinates. Ensure the sample image is loaded correctly.
*   **"No Labels Found"**: Check if your `annotations` folder is correctly linked in `project_config.json`.
*   **Export Failures**: Ensure you have the necessary libraries installed for specific formats (e.g., `onnx` for ONNX export, `tensorrt` for TRT).
