# Project Evolution & Version History

4-month iterative development of the Lung Nodule AI pipeline. All metrics below are sourced directly from MLflow experiment logs (`mlflow_all_results.txt`, generated 2026-04-29).

---

## Detection Stage Evolution

| Version | Model | Input | mAP@50 | Precision | Recall | Notes |
|---------|-------|-------|:------:|:---------:|:------:|-------|
| V2 | V-Net 3D segmentation | 3D volume | — | — | — | 17k fwd passes/scan, HoughCircles fallback |
| V3.1 Trial 1 | YOLOv8x (AdamW) | 2D slice | 0.561 | 0.653 | 0.530 | flipud=0.5 (wrong) |
| V3.1 Trial 2 | YOLOv8x (AdamW) | 2D slice | 0.525 | 0.622 | 0.488 | mosaic=0, underfit |
| V3.1 Trial 3 | YOLOv8x (AdamW) | 2D slice | 0.539 | 0.614 | 0.512 | scale=0.3 too aggressive |
| V3.1 Best | YOLOv8x (AdamW) | 2D slice | 0.587 | — | — | Best Optuna result |
| V3.2 | YOLOv8m (AdamW) | 2D slice | 0.556 | 0.623 | 0.566 | No inter-slice context |
| **V3.3 (Final)** | **YOLOv8m (SGD)** | **2.5D (Z−2,Z,Z+2)** | **0.708** | **0.737** | **0.681** | Medical augmentations |

## Classification Stage Evolution

| Version | Backbone | Task | Val AUC | Val Accuracy | Val F1 | Notes |
|---------|----------|------|:-------:|:------------:|:------:|-------|
| V1 | Single 2D CNN | Binary | 0.543 | 0.524 | 0.333 | Near-random |
| V2 | ResNet50 | 3-class | 0.525 | 0.371 | 0.265 | 3-class confusion |
| V3.0a | ResNet50 | Binary, 10 ep | 0.826 | 0.781 | 0.693 | Under-trained |
| V3.0b | r2plus1d-18 | Multi-Task (3 heads) | 0.874 | 0.813 | 0.679 | Spiculation/Margin collapsed |
| V3.0c | r2plus1d-18 | Multi-Task + Focal + MixUp | 0.870 | 0.786 | 0.731 | Spiculation F1: 0.464 |
| V3.2 | r2plus1d-18 | Binary only | 0.917 | — | — | Dropped aux heads |
| **V3.3 (Final)** | **r2plus1d-18 + Dual Attention** | **Binary** | **0.953** | **0.881** | **0.872** | **Sensitivity: 0.927** |

---

## Version 1: Single-Model Baseline *(March 2026)*

| Property | Detail |
|----------|--------|
| **Script** | `archive/preprocess.py` + `archive/train.py` |
| **Dataset** | LIDC-IDRI only, 2D slices |
| **Architecture** | Single CNN doing both localization and classification |
| **MLflow Experiment** | `Lung Nodule Multi-Output Classification` (earliest run, March 16) |
| **Metrics** | Val Accuracy: **0.524**, Val AUC: **0.543**, Val F1: **0.333** |
| **Why it failed** | Near-random AUC. No dedicated detection. Vessels constantly misclassified. |

---

## Version 2: V-Net Detection + 3-Class ResNet50 *(Early April 2026)*

| Property | Detail |
|----------|--------|
| **Detection** | V-Net 3D segmentation sliding window (`best_vnet.pth`, 703 MB) |
| **Classification** | ResNet50, 3-class (benign / indeterminate / malignant) |
| **MLflow Experiment** | `Lung Nodule Multi-Output Classification` — ResNet50 3-class run |
| **Metrics** | Val Accuracy: **0.371**, Val AUC: **0.525**, Val F1: **0.265** |
| **Why it failed** | 3-class formulation confused the network. V-Net needed ~17,000 forward passes per scan. Required HoughCircles fallback when V-Net failed. |

---

## Version 3: Two-Stage YOLO + Deep Classifier

### V3.0a — ResNet50 Binary, 10 Epochs *(April 6, 2026)*

| Metric | Value |
|--------|-------|
| Backbone | ResNet50 (pretrained) |
| Epochs | 10 |
| Batch | 8, LR: 0.001, WD: 1e-5 |
| Val Accuracy | **0.781** |
| Val AUC | **0.826** |
| Val F1 | **0.693** |

First real sign of life. Switched to binary classification (benign vs malignant). 10 epochs not enough to converge — large gap between train AUC (0.817) and val AUC (0.826) hinting at lucky initialization.

---

### V3.0b — r2plus1d-18 Multi-Task (Malignancy + Spiculation + Margin), 60 Epochs *(April 7, 2026)*

| Metric | Value |
|--------|-------|
| Backbone | r2plus1d-18 (best run of 4 trials) |
| Epochs | 60, Batch: 8, LR: 0.0001, WD: 0.0001 |
| Val Accuracy | **0.824** |
| Val AUC | **0.809** |
| Val F1 | **0.679** |
| Spiculation AUC | **0.782** (collapsed — F1: 0.483) |
| Margin AUC | **0.603** (collapsed — Specificity: 0.18) |

The multi-task architecture with uncertainty-weighted loss actually achieved decent accuracy, but the spiculation and margin heads were degenerate — they predicted only the majority class. This was traced to fake labels silently set to 0 during preprocessing.

---

### V3.0c — r2plus1d-18 + Uncertainty-Weighted Loss + v2 Data *(April 15, 2026)*

| Metric | Value |
|--------|-------|
| Backbone | r2plus1d-18, Multi-Task |
| Key changes | Focal loss (γ=2.0), label smoothing 0.05, MixUp α=0.2, differential LR |
| Val Accuracy | **0.786** |
| Val AUC | **0.870** |
| Val F1 (malignancy) | **0.731** |
| Val Sensitivity | **0.731** |
| Val Specificity | **0.822** |
| Spiculation F1 | **0.464** |

AUC improved but still struggled with the auxiliary heads. Introduced Optuna-style hyperparameter search. Decided to drop spiculation/margin entirely and focus on binary malignancy.

---

### V3.1 — Optuna Hyperparameter Search for YOLOv8x (2D), *(April 20, 2026)*

5 Optuna trials on Kaggle GPU (YOLOv8x, 2D slices, 640px). All used `flipud=0.5` — later identified as a medical mistake.

| Trial | Optimizer | lr0 | Mosaic | mAP50 | Precision | Recall |
|-------|-----------|-----|--------|-------|-----------|--------|
| Trial 1 | AdamW | 0.000477 | 0.25 | **0.561** | 0.653 | 0.530 |
| Trial 2 | AdamW | 0.001330 | 0.00 | **0.525** | 0.622 | 0.488 |
| Trial 3 | AdamW | 0.006758 | 0.50 | **0.539** | 0.614 | 0.512 |
| Best overall | AdamW | — | — | **0.587** | — | — |
| Final long run | AdamW | 0.000174 | 0.25 | **0.556** | 0.623 | 0.566 |

Best Optuna trial achieved mAP50 = 0.587. Not satisfactory — identified augmentation issues (vertical flip, aggressive scale=0.3, copy_paste=0.2 all breaking medical anatomy).

---

### V3.2 — Binary r2plus1d-18 + 2D YOLOv8m *(Late April 2026)*

| Property | Detail |
|----------|--------|
| **Key change** | Removed Spiculation + Margin heads entirely |
| **Classification (from metrics.json)** | Val AUC: **0.917** (best val, epoch 39), Final AUC: **0.953** |
| **Detection** | Still 2D single-slice (no inter-slice context) |
| **Remaining problem** | 2D detection: adjacent slices not visible → high false positives on vessels |

---

### V3.3 — 2.5D YOLOv8m + Binary r2plus1d-18 + Dual Attention *(Final)*

| Property | Detail |
|----------|--------|
| **Scripts** | `kgl_yolo_v3.py` + `kgl_classifier.py` |
| **Key change** | 2.5D: stack slices Z−2, Z, Z+2 as RGB channels |
| **Forbidden augmentations** | flipud=0, mixup=0, copy_paste=0, perspective=0, shear=0 |
| **Optimizer** | SGD (lr0=0.001) — defeated AdamW by +7.82% mAP in ablation |
| **Architecture** | r2plus1d-18 + Dual Channel-Attention, 64³ patches |

#### Final Test Metrics

| Stage | Metric | Value |
|-------|--------|-------|
| Detection | mAP@50 | **0.708** |
| Detection | Precision | **0.737** |
| Detection | Recall | **0.681** |
| Detection | F1 | **0.708** |
| Classification | ROC-AUC | **0.953** |
| Classification | PR-AUC | **0.952** |
| Classification | Accuracy | **0.881** |
| Classification | Sensitivity | **0.927** |
| Classification | Specificity | **0.845** |
| Classification | F1 | **0.872** |

---

## Key Decisions Timeline

| Date | Decision | Trigger |
|------|----------|---------|
| March 2026 | Split into two stages | V1: AUC 0.543, random-level performance |
| Early April | Switched 3-class → binary malignancy | V2: 3-class accuracy 0.371, model confused |
| April 6–7 | Replaced ResNet50 → r2plus1d-18 | +AUC 0.045, better 3D feature capture |
| April 7 | Tried Multi-Task (+ spiculation, margin) | MULTITASK_FIX_SUMMARY: heads always predict class 0 |
| April 15 | Added Uncertainty-Weighted Loss + Focal Loss | Partial fix: spiculation F1 0.464 still poor |
| April 15 | Dropped aux heads entirely | AUC jumped from 0.870 → 0.917 (val) |
| April 20 | Ran 5 Optuna trials for YOLO hyperparams | Baseline mAP50 0.556 not good enough |
| April 20 | Banned medical-unsafe augmentations | flipud, copy_paste, mixup identified as anatomy-breaking |
| April 20 | Switched AdamW → SGD | Ablation: SGD +7.82% mAP over AdamW |
| Late April | Introduced 2.5D input (Z−2, Z, Z+2) | 2D missed inter-slice context → vessel FPs |
