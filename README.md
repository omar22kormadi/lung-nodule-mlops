# Lung Nodule Detection and Classification

An end-to-end pipeline for detecting and classifying lung nodules from CT scans. This project leverages state-of-the-art Deep Learning models to provide 3D visual reconstruction and binary malignancy risk assessment from DICOM medical imaging.

## Datasets & Preprocessing

The project utilizes two major medical imaging datasets:

*   **LUNA16 (Detection):** Used to train the object detection model to locate nodules.
    *   **Preprocessing (2.5D Slicing):** Stacked slices with a gap (`Z-2`, `Z`, `Z+2`) mapped into RGB channels. This inter-slice context drastically reduced false positives on blood vessels.
    *   **Medical Augmentations:** Banned all anatomy-breaking augmentations (e.g., vertical flip, mixup, copy-paste, shear).
*   **LIDC-IDRI (Classification):** Used to train a 3D CNN to classify nodule malignancy.
    *   **Preprocessing:** Extracted dense 64x64x64 3D voxel patches centered perfectly around the detected nodules.
    *   **Simplification:** Dropped auxiliary tasks (margin/spiculation) to focus strictly on a robust binary malignancy objective.

## Models & Architecture

### 1. Nodule Detection: YOLOv8m (2.5D)
A modified 2D YOLOv8 model adapted to read 2.5D medical slices.
*   **Hyperparameters & Training:**
    *   Optimizer: SGD (lr0=0.001) - *Proven to beat AdamW by +7.82% mAP in ablations.*
*   **Final Metrics:**
    *   mAP@50: **0.708** | Precision: **0.737** | Recall: **0.681**

### 2. Nodule Classification: R(2+1)D + Dual Attention
A powerful 3D Convolutional Neural Network built on ResNet3D-18, enhanced with custom Dual Channel-Attention layers.
*   **Task:** Binary Malignancy Classification
*   **Final Metrics:**
    *   ROC-AUC: **0.953** | Accuracy: **0.881** | Sensitivity: **0.927** | F1: **0.872**

## Visual Results & Model Performance

### Ground Truth vs. Predictions (2.5D YOLOv8m)
*Notice how the model successfully isolates pulmonary nodules, ignoring complex anatomical distractors such as blood vessels, airways, and the pleural wall.*

| Ground Truth (Expert Annotations) | Model Predictions (YOLOv8m) |
|:---:|:---:|
| <img src="02_Model_Development/ml_model_evaluation/yolo_results/val_batch2_labels.jpg" alt="Ground Truth" width="100%"> | <img src="02_Model_Development/ml_model_evaluation/yolo_results/val_batch2_pred.jpg" alt="Model Predictions" width="100%"> |

### Training Convergence & Metrics
The training curves demonstrate exceptional stability. The 2.5D spatial context combined with the optimal SGD learning rate allowed the detection model to steadily drive down bounding box and classification losses while pushing `mAP@50` smoothly above 0.70.

<p align="center">
  <img src="02_Model_Development/ml_model_evaluation/yolo_results/results.png" alt="Training Convergence Curves" width="100%" style="border-radius: 8px;">
</p>
## How to Run

To test the models locally, you need to run the FastAPI backend and the React frontend.

### 1. Start the API (Backend)
Navigate to the deployment folder and start the FastAPI server:
```bash
cd 03_Model_Operations/deployment
# Run using the batch file:
start_api.bat
# OR run manually:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
*Note: Ensure your Python environment is activated and the trained model weights are in the correct models folder.*

### 2. Start the UI (Frontend)
Open a new terminal, navigate to the frontend folder, and start the development server:
```bash
cd 03_Model_Operations/deployment/frontend
npm install
# Run using the batch file:
start_frontend.bat
# OR run manually:
npm run dev
```
The application will launch in your browser (typically at `http://localhost:8080`). Upload a complete DICOM study (10+ slices) to view the bounding boxes, risk assessment, and interactive 3D lung reconstruction!
