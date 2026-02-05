# Project Overview

This project, `yolo_scripts`, is a comprehensive toolkit for working with YOLO (You Only Look Once) object detection models, powered by the `ultralytics` Python package. It consists of two main parts: a collection of command-line Python scripts for various data manipulation and model management tasks, and a full-featured web-based user interface for a more interactive workflow.

## Main Technologies

*   **Backend:** Python, FastAPI, `ultralytics`
*   **Frontend:** JavaScript, React, Vite, `axios`, `recharts`, `react-image-crop`
*   **Data:** YAML for dataset configuration, JSON for project configuration.

## Architecture

The project is divided into two main components:

1.  **`scripts/`**: A collection of standalone Python scripts that can be run from the command line. These scripts cover a wide range of functionalities, including data augmentation, dataset analysis, model exporting, and more.
2.  **`webui/`**: A modern web application with a FastAPI backend and a React frontend. The web UI provides a user-friendly interface for managing the entire YOLO workflow, from data pre-processing and auto-labeling to dataset generation and analysis.

The backend of the web UI leverages the scripts in the `scripts/` directory to perform its tasks.

# Building and Running

## Command-Line Scripts

To use the command-line scripts, you need to have Python 3 and the required dependencies installed.

1.  **Install dependencies:**

    ```bash
    pip install ultralytics torch numpy matplotlib seaborn pyyaml
    ```

2.  **Run a script:**

    ```bash
    python3 scripts/<script_name>.py [arguments]
    ```

    For example, to run the dataset analytics script:

    ```bash
    python3 scripts/dataset_analytics.py --data_dir /path/to/your/dataset
    ```

## Web UI

To run the web UI, you need to have Python 3 and Node.js installed.

1.  **Run the Backend (FastAPI):**

    ```bash
    cd webui/backend
    pip install -r requirements.txt
    python main.py
    ```

2.  **Run the Frontend (React + Vite):**

    ```bash
    cd webui/frontend
    npm install
    npm run dev
    ```

3.  **Access the UI:**

    Open your web browser and navigate to `http://localhost:3000`.

Alternatively, you can use the `webui.sh` script to manage the web UI services:

```bash
cd webui
./webui.sh start --port 4000
```

# Development Conventions

*   **Python:** The Python code follows standard Python conventions. The scripts are designed to be modular and reusable.
*   **JavaScript/React:** The frontend code is written in modern JavaScript (ESM) and follows the standard React project structure.
*   **Configuration:** The project uses YAML files for dataset configuration (`data.yaml`) and a JSON file (`project_config.json`) for web UI project configuration.
*   **Virtual Environment:** It is highly recommended to use a dedicated virtual environment for this project to avoid dependency conflicts.
