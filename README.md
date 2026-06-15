# 🫁 Early-Stage Lung Cancer Detection & Classification - MLOps Pipeline

End-to-End Deep Learning project for detecting and assessing the malignancy risk of pulmonary nodules directly from DICOM CT scans.

---

## 📊 Project Overview

This project builds and serves a two-stage computer vision pipeline:
- **Stage 1 (Detection):** YOLOv8m custom-trained on 2.5D slices to locate nodules.
- **Stage 2 (Classification):** R(2+1)D Dual-Attention 3D CNN to classify nodule malignancy.
- Integrated into a FastAPI backend and a React 3D viewer frontend.

---

## 🛠️ Tech Stack

| Layer                  | Tool / Library                |
|------------------------|-------------------------------|
| **Programming**        | Python 3.10, JavaScript       |
| **Detection Model**    | Ultralytics YOLOv8 (PyTorch)  |
| **Classification Model**| Custom R(2+1)D (PyTorch)     |
| **Data Processing**    | Pydicom, SimpleITK, OpenCV    |
| **Backend API**        | FastAPI, Uvicorn              |
| **Frontend UI**        | React, Three.js (3D viewer)   |

---

## 📁 Project Structure

```
CRISP-ML(Q)/
├── 01_Business_and_Data_Understanding/ # EDA & Project Scoping
├── 02_Model_Development/
│   ├── data_engineering/               # DICOM to YOLO/3D patches preprocessing
│   └── ml_model_engineering/
│       ├── models/                     # Best weights (best.pt, best_model.pth)
│       └── kgl_yolo_v3.py              # YOLO training logic
├── 03_Model_Operations/
│   └── deployment/
│       ├── api.py                      # FastAPI Backend
│       ├── start_api.bat               # API launch script
│       └── frontend/
│           ├── src/                    # React source code
│           ├── package.json            # Node dependencies
│           └── start_frontend.bat      # UI launch script
├── data/                               # Raw datasets and visualizations
└── README.md
```

---

## 🚀 Quickstart (Local)

### 1. Clone the repository

```bash
git clone "https://github.com/omar22kormadi/lung-nodule-mlops.git"
cd "lung-nodule-mlops/CRISP-ML(Q)"
```

### 2. Start the API (FastAPI)

```bash
cd "03_Model_Operations/deployment"
# Run using the batch file:
start_api.bat

# OR run manually:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: http://127.0.0.1:8000/docs

### 3. Start the UI (React)

Open a **new** terminal:
```bash
cd "03_Model_Operations/deployment/frontend"
npm install

# Run using the batch file:
start_frontend.bat

# OR run manually:
npm run dev
```
- UI: http://localhost:8080 (or `5173`)

Upload a complete DICOM study (10+ slices) to view the bounding boxes, risk assessment, and interactive 3D lung reconstruction!

---

## 📈 Visual Results & Model Performance

### Ground Truth vs. Predictions (2.5D YOLOv8m)
*Notice how the model successfully isolates pulmonary nodules, ignoring complex anatomical distractors such as blood vessels and airways.*

| Ground Truth (Expert Annotations) | Model Predictions (YOLOv8m) |
|:---:|:---:|
| <img src="02_Model_Development/ml_model_evaluation/yolo_results/val_batch2_labels.jpg" alt="Ground Truth" width="100%"> | <img src="02_Model_Development/ml_model_evaluation/yolo_results/val_batch2_pred.jpg" alt="Model Predictions" width="100%"> |

### Training Convergence & Metrics
The training curves demonstrate exceptional stability. The 2.5D spatial context combined with the optimal SGD learning rate allowed the detection model to steadily drive down bounding box and classification losses while pushing `mAP@50` smoothly above 0.70.

<p align="center">
  <img src="02_Model_Development/ml_model_evaluation/yolo_results/results.png" alt="Training Convergence Curves" width="100%" style="border-radius: 8px;">
</p>

### Final Metrics

| Stage | Model | Task | Key Metric | Value |
|-------|-------|------|------------|-------|
| **Detection** | YOLOv8m (2.5D) | Bounding Box | mAP@50 | **0.708** |
| **Detection** | YOLOv8m (2.5D) | Bounding Box | Precision | **0.737** |
| **Classification**| R(2+1)D Dual-Attention | Binary Malignancy | ROC-AUC | **0.953** |
| **Classification**| R(2+1)D Dual-Attention | Binary Malignancy | Sensitivity | **0.927** |

---

## 👤 Author

- **Name**: Amor Kormadi 
- **Email**: amor.kormadi@polytechnicien.tn 
- **Project**: CRISP-ML(Q) Pipeline for Pulmonary Nodule Detection and Classification 
