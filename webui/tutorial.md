# YOLO Dataset WebUI Tutorial

This tutorial will guide you through the complete lifecycle of creating, labeling, verifying, and training a YOLO object detection project using the WebUI.

---

## 1. Prerequisites
Ensure you have installed the application using the `install.sh` script as described in the [README](README.md).
Start the application:

```bash
cd webui
./webui.sh start --port 4000
```
Open your browser at `http://localhost:4000`.

---

## 2. The Workflow
The WebUI follows a linear workflow designed to ensure data quality:
1.  **Import Raw Images**: Start with a source folder.
2.  **Pre-process**: Crop and resize images for consistency.
3.  **Auto-Label**: Use an Initial AI model to guess labels.
4.  **Verify & Correct**: Fix mistakes visually.
5.  **Generate Dataset**: Export to YOLO format.
6.  **Train**: Fine-tune a new model.

---

## 3. Step-by-Step Guide

### Step 1: Create a Project 🏠
1.  Navigate to the **Project Info** tab (Home).
2.  In the "Create New Project" section:
    *   **Project Name**: Enter a unique name (e.g., `cardetection`).
    *   **Parent Directory**: Where to save the project (e.g., `/home/user/projects`).
    *   **Raw Images Path**: The absolute path to your folder of collected images.
    *   **Class Names**: Enter classes separated by commas (e.g., `car, bus, truck`).
3.  Click **Create Project**.
    *   *Note*: The system will create a standard folder structure (`cardetection_raw_images`, `cardetection_processed_images`, etc.).

### Step 2: Pre-processing ✂️
1.  Go to the **Pre-processing** tab.
2.  **Crop Configuration**:
    *   You will see a sample image.
    *   Drag the crop box to define the Region of Interest (ROI). This is useful if your object is always in a specific part of the camera frame.
    *   *Tip*: If you want to use the full image, ensure the crop box covers 100%.
3.  **Resize Dimensions**: Set the target size (e.g., `640` x `640`).
4.  Click **Process All Images**.
    *   This generates the `_processed_images` folder.

### Step 3: Auto-Labeling 🤖
1.  Go to the **Auto-Labeling** tab.
2.  **Select Model**:
    *   Enter the path to a specialized `.pt` model if you have one.
    *   Or use a standard `yolov8n.pt`.
3.  **Confidence Threshold**: Set to `0.4` or `0.5`. Higher values reduce false positives but might miss objects.
4.  Click **Test on Single Image** to verify settings.
5.  Click **Label All Images** to run batch inference.
    *   This generates `.txt` label files in the `annotations` folder.

### Step 4: Verification (Crucial Step!) ✅
1.  Go to the **Data Inspection** tab.
2.  **Generate Masks**: If you don't see images, confirm "Masked Images" are generated. The auto-label step usually does this, or you can click a button to generate them.
3.  **Review Gallery**: Scroll through the images. Look for:
    *   **Missing Boxes**: Objects not detected.
    *   **Wrong Labels**: A "car" labeled as a "bus".
    *   **Bad Crops**: Images cut off incorrectly.
4.  **Actions**:
    *   **Filter**: Use the Sidebar to copy  images to a folder for further inspection.
    *   **Edit**: Use the **Manual Annotation** tab (✏️) to fix specific images.


### Step 5: Dataset Creation 📦
1.  Go to the **Create Dataset** tab.
2.  **Dataset Name**: Give a versioned name (e.g., `v1_rebalanced`).
3.  **Split Ratios**: Define percentage for Train/Validation/Test (e.g., `70 / 20 / 10`).
4.  Click **Create Dataset**.
    *   This exports a `data.yaml` and separate folders (`train/images`, `train/labels`, etc.) ready for training.

### Step 6: Dataset Tools (Optional) ⚙️
If you have complex needs:
*   **Merge**: Combine `v1` and `new_data` into `v2`.
*   **Rebalance**: If your dataset isn't split correctly, use the Rebalance tool to reshuffle it.
*   **Split Stats**: Check if you have enough training data vs validation data.

### Step 7: Training 🏋️
1.  Go to the **Training** tab.
2.  **Model**: Select `yolov8n.pt` (Nano) for speed or `yolov8m.pt` (Medium) for accuracy.
3.  **Dataset**: Enter the path to the `data.yaml` generated in Step 5.
4.  **Hyperparameters**:
    *   `Epochs`: 50-100 is a good start.
    *   `Image Size`: 640.
5.  Click **Start Training**.
    *   Watch the live logs.
    *   Once finished, the new model is saved in `runs/detect/trainX/weights/best.pt`.

### Step 8: Export 📤
1.  Still in the **Training** tab (scroll down).
2.  Select your best model (`best.pt`).
3.  Choose a format (e.g., **TensorRT** for Jetson, **ONNX** for general deployment).
4.  Click **Export Model**.

---

## 4. Tips & Tricks
*   **Empty Images**: Don't just delete images with no objects! Keep some as "background" images (empty labels) to teach the model what *not* to detect.
*   **Data Augmentation**: Use the Augmentation tab to create variations (blur, rotate) of your training data to make the model more robust.
