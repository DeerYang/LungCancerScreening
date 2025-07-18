<div align="right">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</div>

# AI-Assisted Lung Cancer Screening System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2+-000000.svg)](https://flask.palletsprojects.com/)

An end-to-end open-source system for AI-assisted lung cancer screening. This project features a 3-stage deep learning pipeline (Segmentation, Classification, Malignancy Analysis) trained on the full **LUNA16 dataset**. It includes all training scripts, a Flask backend, and a React frontend with an interactive CT viewer.

---

## 🌟 Features

*   **End-to-End Pipeline**: A complete workflow from raw CT scans to diagnostic reports.
*   **3-Stage AI Analysis**: A sophisticated pipeline for accurate diagnostics.
    *   **Segmentation**: UNet model to identify nodule candidates on 2D CT slices.
    *   **Classification**: 3D-CNN to confirm if candidates are true nodules.
    *   **Malignancy Prediction**: A fine-tuned 3D-CNN to assess the malignancy risk of confirmed nodules.
*   **Interactive CT Viewer**: An intuitive web-based viewer to scroll through CT slices and visualize AI-detected nodules with bounding boxes.
*   **Full-Stack Application**: A robust Flask backend serves the AI models via a RESTful API, consumed by a modern React + Ant Design frontend.
*   **Data Persistence**: Diagnostic results and uploaded CT files are saved persistently using a MySQL database.
*   **LLM-Powered Chatbot**: An integrated AI assistant to help doctors interpret diagnostic reports and answer questions.
*   **Complete Training Scripts**: The full PyTorch training pipeline is provided, allowing for model reproduction and further research.


## 🛠️ Tech Stack

*   **Backend**: Python, Flask, PyTorch, SQLAlchemy, PyMySQL, SimpleITK, NumPy, SciPy
*   **Frontend**: React, Ant Design, ECharts, Axios
*   **Database**: MySQL
*   **AI Models**: UNet, 3D-CNN
*   **LLM**: Qwen (via Dashscope API)

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

*   **Python**: 3.9 or higher.
*   **Node.js**: 16.x or higher.
*   **MySQL Server**: 5.7 or higher (8.x recommended).
*   **Git**

### 2. Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/YourUsername/LungCancerScreening.git
cd LungCancerScreening
```
*(Please replace `YourUsername` with your GitHub username.)*

**Step 2: Set up the Python backend**
```bash
# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows, use `.venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt
```

**Step 3: Set up the MySQL database**
1.  Log in to your MySQL server.
2.  Create the database for this project.
    ```sql
    CREATE DATABASE lung_cancer_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    ```
3.  Open the backend configuration file `web_app/app.py`.
4.  Find and update the database connection string with your MySQL credentials.
    ```python
    # web_app/app.py
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://YOUR_USER:YOUR_PASSWORD@localhost/lung_cancer_db'
    ```

**Step 4: Set up the React frontend**
```bash
# Navigate to the frontend directory
cd web_app/frontend

# Install npm dependencies (this might take a few minutes)
npm install
```

**Step 5: Configure API Key (Required for Chatbot)**

The integrated AI Chatbot is powered by Alibaba Cloud's Dashscope service. To enable this feature, you must configure your API Key. Please refer to the [Chinese README (中文文档)](README_zh.md#step-5-configure-api-key--%E9%85%8D%E7%BD%AEapi%E5%AF%86%E9%92%A5-required-for-chatbot--%E8%81%8A%E5%A4%A9%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%8A%9F%E8%83%BD%E5%BF%85%E9%9C%80) for detailed instructions.

### 3. Running the Application

A convenience script is provided to start both the backend and frontend servers simultaneously.

From the project **root directory** (`LungCancerScreening/`), run:```bash
python start_system.py

The system will be available at:
*   **Frontend URL**: [http://localhost:3000](http://localhost:3000)
*   **Backend API**: [http://localhost:5000](http://localhost:5000)

## 🧠 Training Your Own Models

If you wish to train the models yourself, you can use the provided training scripts.

### 1. Dataset Setup

1.  Download the LUNA16 dataset from the official website: [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/Download/).
2.  Place the `subset0` through `subset9` folders inside the `data-unversioned/data/` directory.
3.  Place `annotations.csv` and `candidates.csv` into the `data/` directory.

Your data directory structure should look like this:
```
LungCancerScreening/
├── data/
│   ├── annotations.csv
│   └── candidates.csv
└── data-unversioned/
    └── data/
        ├── subset0/
        ├── subset1/
        └── ...
```
### 2. Training Workflow

For each model (segmentation, classification, malignancy), there is a two-step process: **caching** and **training**.

1.  **Segmentation Model**:
    ```bash
    # Step 1: Cache the data for segmentation
    python prep_seg_cache.py
    # Step 2: Start training
    python segmentTraining.py --epochs 10 --batch-size 16
    ```
2.  **Classification Model**:
    ```bash
    # Step 1: Cache the data for classification
    python prepcache.py
    # Step 2: Start training
    python training.py --epochs 5 --batch-size 8 --balanced
    ```
3.  **Malignancy Model**:
    ```bash
    # Step 1: Cache the data for malignancy analysis
    python TumorPrepCache.py
    # Step 2: Start training (fine-tuning from the classification model)
    python TumorTraining.py --epochs 10 --batch-size 8 --finetune "path/to/your/best_classification_model.state"
    ```
    *(Note: Adjust the training parameters like epochs, batch size, etc., as needed.)*

### 3. Model Evaluation

After training, you can evaluate the performance of your entire pipeline using `model_eval.py`.

```bash
python model_eval.py --run-validation \
    --segmentation-path "path/to/your/best_segmentation_model.state" \
    --classification-path "path/to/your/best_classification_model.state" \
    --malignancy-path "path/to/your/best_malignancy_model.state"
```
You can also evaluate a single CT scan by providing its `series_uid`:
```bash
python model_eval.py a.b.c.123  # Replace with a valid series_uid
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YourUsername/LungCancerScreening/issues).

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
