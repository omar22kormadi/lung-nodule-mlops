"""
Lung Nodule AI API — Detection (YOLOv8) + Classification (R2Plus1D)
=====================================================================
Aligned with:
  - luna16_preprocessing_v2.py  (2.5D RGB, lung_default HU window)
  - lidc_preprocessing.py         (64³ patches, HU [-1000, 400])
  - kgl_yolo_v3.py                (conf=0.25, iou=0.40)
  - kgl_classifier.py             (LungNoduleClassifier multi-task)

Run:  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import csv
import datetime
import io
import json
import os
import random
import shutil
import tempfile
import uuid
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models.video import r2plus1d_18
from ultralytics import YOLO
import skimage.measure

# =============================================================================
# Paths
# =============================================================================
DEPLOYMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent.parent

MODEL_DIR = PROJECT_ROOT / "02_Model_Development" / "ml_model_engineering" / "models"
YOLO_PATH = MODEL_DIR / "best.pt"
CLASSIFIER_PATH = MODEL_DIR / "best_model.pth"
BACKBONE_WEIGHTS = PROJECT_ROOT / "02_Model_Development" / "ml_model_engineering" / "r2plus1d_18_weights.pth"

CLASSIFICATION_TEST_DIR = PROJECT_ROOT / "data" / "classification_dataset" / "test"
CLASSIFICATION_META = PROJECT_ROOT / "data" / "classification_dataset" / "metadata.csv"

YOLO_DATASET_DIR = PROJECT_ROOT / "data" / "luna16_yolo_dataset_v4"
DETECTION_TEST_IMAGES = YOLO_DATASET_DIR / "images" / "test"
DETECTION_TEST_LABELS = YOLO_DATASET_DIR / "labels" / "test"

# Monitoring & labeled data paths
MONITORING_LOG  = PROJECT_ROOT / "data" / "monitoring" / "predictions_log.csv"
LABELED_DIR     = PROJECT_ROOT / "data" / "labeled"
LABELED_META    = LABELED_DIR / "metadata.csv"

# Model version string (update when you retrain)
MODEL_VERSION = "yolov8m_v3.3 + r2plus1d_v3.3"

# Preprocessing constants (must match training)
HU_WINDOW_CENTER = -600.0
HU_WINDOW_WIDTH = 1500.0
HU_MIN = HU_WINDOW_CENTER - HU_WINDOW_WIDTH / 2.0   # -1350
HU_MAX = HU_WINDOW_CENTER + HU_WINDOW_WIDTH / 2.0   # 150
HU_RANGE = HU_MAX - HU_MIN

PATCH_HU_MIN = -1000
PATCH_HU_MAX = 400
PATCH_SIZE = 64

SLICE_GAP_25D = 2
YOLO_CONF = 0.25
YOLO_IOU = 0.40
SOFTMAX_TEMPERATURE = 5.0

# =============================================================================
# FastAPI
# =============================================================================
app = FastAPI(title="Lung Nodule AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device: torch.device | None = None
classifier: nn.Module | None = None
yolo_model: YOLO | None = None


# =============================================================================
# Model (matches kgl_classifier.py)
# =============================================================================
class _ClsConfig:
    NUM_CLASSES_MAL = 2
    NUM_CLASSES_SPI = 2
    NUM_CLASSES_MAR = 2
    WEIGHTS_PATH = BACKBONE_WEIGHTS if BACKBONE_WEIGHTS.exists() else None


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x.unsqueeze(-1)).squeeze(-1))
        max_out = self.fc(self.max_pool(x.unsqueeze(-1)).squeeze(-1))
        return x * self.sigmoid(avg_out + max_out)


class DualAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.refine = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.channel_att(x))


class LungNoduleClassifier(nn.Module):
    def __init__(self, config: _ClsConfig = _ClsConfig()):
        super().__init__()
        backbone = r2plus1d_18(weights=None)
        if config.WEIGHTS_PATH and Path(config.WEIGHTS_PATH).exists():
            state = torch.load(config.WEIGHTS_PATH, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)

        feat_dim = backbone.fc.in_features
        self.backbone = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        for name, param in self.backbone.named_parameters():
            if name.startswith("0") or name.startswith("1"):
                param.requires_grad = False

        self.attention = DualAttention(feat_dim, reduction=16)
        self.bn_adapt = nn.BatchNorm1d(feat_dim)
        self.malignancy_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, config.NUM_CLASSES_MAL),
        )
        self.spiculation_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, config.NUM_CLASSES_SPI),
        )
        self.margin_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, config.NUM_CLASSES_MAR),
        )

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        features = self.backbone(x).view(x.size(0), -1)
        features = self.attention(self.bn_adapt(features))
        return (
            self.malignancy_head(features),
            self.spiculation_head(features),
            self.margin_head(features),
        )


# =============================================================================
# Preprocessing helpers
# =============================================================================
def _hu_window_slice(slice_hu: np.ndarray) -> np.ndarray:
    clipped = np.clip(slice_hu.astype(np.float32), HU_MIN, HU_MAX)
    return ((clipped - HU_MIN) / HU_RANGE * 255.0).astype(np.uint8)


def create_25d_image(volume_hu: np.ndarray, z_center: int) -> np.ndarray:
    """R=Z-gap, G=Z, B=Z+gap — same as luna16_preprocessing_v2._create_25d_image."""
    d = volume_hu.shape[0]
    gap = SLICE_GAP_25D
    z_m = max(0, z_center - gap)
    z_p = min(d - 1, z_center + gap)
    lo, hi = HU_MIN, HU_MAX
    channels = []
    for z in (z_m, z_center, z_p):
        s = np.clip(volume_hu[z], lo, hi).astype(np.float32)
        channels.append((s - lo) / max(hi - lo, 1e-6))
    rgb = (np.stack(channels, axis=-1) * 255.0).astype(np.uint8)
    return rgb


def normalize_hu_patch(patch: np.ndarray) -> np.ndarray:
    """lidc_preprocessing.PatchExtractor.normalize_hu"""
    clipped = np.clip(patch, PATCH_HU_MIN, PATCH_HU_MAX)
    return ((clipped - PATCH_HU_MIN) / (PATCH_HU_MAX - PATCH_HU_MIN)).astype(np.float32)


def extract_patch_3d(volume_hu: np.ndarray, z: int, y: int, x: int, size: int = PATCH_SIZE) -> np.ndarray:
    half = size // 2
    d, h, w = volume_hu.shape
    z1, z2 = max(0, z - half), min(d, z + half)
    y1, y2 = max(0, y - half), min(h, y + half)
    x1, x2 = max(0, x - half), min(w, x + half)

    patch = np.full((size, size, size), PATCH_HU_MIN, dtype=np.float32)
    pz, py, px = z1 - (z - half), y1 - (y - half), x1 - (x - half)
    patch[pz : pz + (z2 - z1), py : py + (y2 - y1), px : px + (x2 - x1)] = volume_hu[z1:z2, y1:y2, x1:x2]
    return normalize_hu_patch(patch)


def load_dicom_series(file_paths: list[str]) -> tuple[np.ndarray | None, list[float], float]:
    slices = []
    for fp in file_paths:
        try:
            ds = pydicom.dcmread(fp, force=True)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            continue

    if len(slices) < 10:
        return None, [1.0, 1.0], 1.0

    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    volume = volume * slope + intercept

    try:
        ps = [float(x) for x in slices[0].PixelSpacing]
    except Exception:
        ps = [1.0, 1.0]
    try:
        st = float(slices[0].SliceThickness)
    except Exception:
        st = 1.0
    return volume, ps, st


def volume_tensor_from_npz(volume: np.ndarray) -> torch.Tensor:
    """NPZ volumes are already normalized to [0, 1] during LIDC preprocessing."""
    v = volume.astype(np.float32)
    if v.max() > 1.5:
        v = normalize_hu_patch(v)
    t = torch.from_numpy(v)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def encode_image_b64(img_bgr: np.ndarray, fmt: str = ".jpg") -> str:
    ok, buf = cv2.imencode(fmt, img_bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def label_name(idx: int, task: str) -> str:
    maps = {
        "malignancy": {0: "Benign", 1: "Malignant"},
        "spiculation": {0: "Absent", 1: "Present"},
        "margin": {0: "Smooth", 1: "Irregular"},
    }
    return maps.get(task, {}).get(idx, str(idx))


def format_classification(mal_o, spi_o, mar_o) -> dict[str, Any]:
    mal_p = torch.softmax(mal_o / SOFTMAX_TEMPERATURE, dim=1).cpu().numpy()[0]
    spi_p = torch.softmax(spi_o / SOFTMAX_TEMPERATURE, dim=1).cpu().numpy()[0]
    mar_p = torch.softmax(mar_o / SOFTMAX_TEMPERATURE, dim=1).cpu().numpy()[0]
    return {
        "malignancy": {
            "pred": int(np.argmax(mal_p)),
            "label": label_name(int(np.argmax(mal_p)), "malignancy"),
            "probs": mal_p.tolist(),
            "probability_malignant": float(mal_p[1]),
        },
        "spiculation": {
            "pred": int(np.argmax(spi_p)),
            "label": label_name(int(np.argmax(spi_p)), "spiculation"),
            "probs": spi_p.tolist(),
        },
        "margin": {
            "pred": int(np.argmax(mar_p)),
            "label": label_name(int(np.argmax(mar_p)), "margin"),
            "probs": mar_p.tolist(),
        },
    }


@torch.inference_mode()
def predict_volume_tensor(volume: torch.Tensor) -> dict[str, Any]:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)
    volume = volume.to(device)
    mal_o, spi_o, mar_o = classifier(volume)
    return format_classification(mal_o, spi_o, mar_o)


def yolo_detect_rgb(img_rgb: np.ndarray) -> list[dict[str, Any]]:
    if yolo_model is None:
        return []
    res = yolo_model(img_rgb, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    out = []
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        out.append({
            "bbox_xyxy": [x1, y1, x2, y2],
            "x": cx, "y": cy,
            "radius": max((x2 - x1) // 2, (y2 - y1) // 2),
            "confidence": float(conf),
        })
    return out


def draw_overlay(
    img_bgr: np.ndarray,
    pred_detections: list[dict],
    gt_center: tuple[int, int] | None = None,
    gt_label: str | None = None,
    pred_label: str | None = None,
    pred_prob: float | None = None,
) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    if gt_center is not None:
        gx, gy = gt_center
        cv2.circle(vis, (gx, gy), 28, (80, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(vis, f"GT: {gt_label or '?'}", (max(5, gx - 40), max(20, gy - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 120), 1, cv2.LINE_AA)

    for i, det in enumerate(pred_detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        prob = det.get("mal_prob")
        if prob is None:
            color = (100, 180, 255)
        elif prob > 0.6:
            color = (80, 80, 255)
        elif prob > 0.4:
            color = (50, 180, 255)
        else:
            color = (255, 180, 80)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        tag = f"Pred#{i+1} {det['confidence']:.2f}"
        if prob is not None:
            tag += f" | {pred_label or ''} {prob*100:.0f}%"
        cv2.putText(vis, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if pred_label and gt_label:
        banner = f"True: {gt_label}  |  Pred: {pred_label}"
        if pred_prob is not None:
            banner += f" ({pred_prob*100:.1f}%)"
        cv2.rectangle(vis, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, banner, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def generate_nodule_mesh(patch_normalized: np.ndarray, threshold: float = 0.4) -> dict[str, list]:
    """Generate a lightweight 3D mesh from the 64x64x64 nodule patch for frontend rendering."""
    try:
        # Pad with 0 (air) to ensure closed surfaces
        padded = np.pad(patch_normalized, pad_width=1, mode='constant', constant_values=0)
        verts, faces, normals, values = skimage.measure.marching_cubes(
            padded, level=threshold, step_size=1
        )
        # Shift vertices to center them around the origin (patch size is 64, padded is 66)
        verts = verts - 33.0
        return {
            "vertices": verts.flatten().tolist(),
            "faces": faces.flatten().tolist()
        }
    except Exception as e:
        print(f"[WARN] Failed to generate 3D mesh: {e}")
        return {"vertices": [], "faces": []}


def generate_lung_mesh(volume_hu: np.ndarray, threshold: float = -400.0, step_size: int = 4) -> dict[str, list]:
    """Generate a lightweight 3D mesh of the entire lung volume using marching cubes."""
    try:
        # Pad volume slightly to close boundaries
        padded = np.pad(volume_hu, pad_width=1, mode='constant', constant_values=-1000)
        verts, faces, normals, values = skimage.measure.marching_cubes(
            padded, level=threshold, step_size=step_size
        )
        # Shift vertices back
        verts = verts - 1.0
        return {
            "vertices": verts.flatten().tolist(),
            "faces": faces.flatten().tolist()
        }
    except Exception as e:
        print(f"[WARN] Failed to generate full lung 3D mesh: {e}")
        return {"vertices": [], "faces": []}


# =============================================================================
# Startup
# =============================================================================
def load_models() -> None:
    global device, classifier, yolo_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")

    ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
    classifier = LungNoduleClassifier()
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device).eval()

    if YOLO_PATH.exists():
        yolo_model = YOLO(str(YOLO_PATH))
    else:
        yolo_model = None
        print(f"[WARN] YOLO weights missing: {YOLO_PATH}")


@app.on_event("startup")
async def startup():
    load_models()
    print("[OK] Models loaded")


# =============================================================================
# Routes
# =============================================================================
@app.get("/")
def root():
    return {"service": "Lung Nodule AI API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
        "classifier": CLASSIFIER_PATH.exists(),
        "yolo": YOLO_PATH.exists() and yolo_model is not None,
        "classification_test_samples": len(list(CLASSIFICATION_TEST_DIR.glob("*.npz"))) if CLASSIFICATION_TEST_DIR.exists() else 0,
        "detection_test_images": len(list(DETECTION_TEST_IMAGES.glob("*.png"))) if DETECTION_TEST_IMAGES.exists() else 0,
    }


@app.get("/api/config")
def api_config():
    return {
        "patch_size": PATCH_SIZE,
        "hu_window": {"center": HU_WINDOW_CENTER, "width": HU_WINDOW_WIDTH, "min": HU_MIN, "max": HU_MAX},
        "yolo": {"conf": YOLO_CONF, "iou": YOLO_IOU},
        "paths": {
            "classifier": str(CLASSIFIER_PATH),
            "yolo": str(YOLO_PATH),
            "classification_test": str(CLASSIFICATION_TEST_DIR),
            "detection_test": str(DETECTION_TEST_IMAGES),
        },
    }


# ── Classification test set ───────────────────────────────────────────────────

@app.get("/api/test/classification/samples")
def list_classification_samples():
    files = sorted(CLASSIFICATION_TEST_DIR.glob("*.npz"))
    return {"count": len(files), "samples": [f.name for f in files[:200]]}


@app.get("/api/test/classification/sample")
def classification_test_sample(filename: str | None = Query(None)):
    """Random (or named) test .npz with ground truth + prediction + overlay."""
    files = list(CLASSIFICATION_TEST_DIR.glob("*.npz"))
    if not files:
        raise HTTPException(404, "No test samples found")

    if filename:
        path = CLASSIFICATION_TEST_DIR / filename
        if not path.exists():
            raise HTTPException(404, f"Not found: {filename}")
    else:
        path = random.choice(files)

    data = np.load(path)
    volume = data["volume"].astype(np.float32)
    true_mal = int(data["label"]) if "label" in data.files else int(data.get("malignancy_label", -1))
    true_label = "malignant" if true_mal == 1 else "benign"

    tensor = volume_tensor_from_npz(volume)
    pred = predict_volume_tensor(tensor)

    mid = volume.shape[0] // 2
    sl = (volume[mid] * 255).astype(np.uint8)
    sl = cv2.cvtColor(cv2.resize(sl, (512, 512)), cv2.COLOR_GRAY2BGR)

    mal_prob = pred["malignancy"]["probability_malignant"]
    pred_label = pred["malignancy"]["label"]
    vis = draw_overlay(sl, [], gt_center=(256, 256), gt_label=true_label.title(),
                     pred_label=pred_label, pred_prob=mal_prob)
    cv2.circle(vis, (256, 256), 90, (80, 220, 120), 2, cv2.LINE_AA)

    correct = (pred["malignancy"]["pred"] == true_mal)
    return {
        "filename": path.name,
        "true_label": true_label,
        "true_malignancy": true_mal,
        "prediction": pred,
        "correct": correct,
        "visualization": encode_image_b64(vis),
        "volume_shape": list(volume.shape),
    }


@app.post("/api/evaluate/classification")
def evaluate_classification(limit: int = Query(0, ge=0, le=500)):
    """Evaluate classifier on all (or first `limit`) test .npz patches."""
    files = sorted(CLASSIFICATION_TEST_DIR.glob("*.npz"))
    if limit > 0:
        files = files[:limit]

    tp = fp = tn = fn = 0
    for path in files:
        data = np.load(path)
        y_true = int(data["label"])
        vol = volume_tensor_from_npz(data["volume"].astype(np.float32))
        y_pred = predict_volume_tensor(vol)["malignancy"]["pred"]
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        else:
            fn += 1

    n = len(files) or 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / n
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "samples": len(files),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


# ── Detection test set ────────────────────────────────────────────────────────

@app.post("/api/evaluate/detection")
def evaluate_detection(limit: int = Query(100, ge=1, le=2000)):
    """Subset evaluation on YOLO test images with label files."""
    if yolo_model is None:
        raise HTTPException(503, "YOLO model not loaded")

    images = sorted(DETECTION_TEST_IMAGES.glob("*.png"))[:limit]
    tp = fp = fn = 0
    iou_thr = 0.5

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = parse_yolo_label(DETECTION_TEST_LABELS / f"{img_path.stem}.txt", w, h)
        preds = yolo_detect_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred_boxes = [tuple(d["bbox_xyxy"]) for d in preds]

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "images_evaluated": len(images),
        "iou_threshold": iou_thr,
        "yolo_conf": YOLO_CONF,
        "yolo_iou": YOLO_IOU,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


@app.get("/api/test/detection/sample")
def detection_test_sample(filename: str | None = Query(None)):
    """Random test 2.5D slice with GT box + YOLO prediction overlay."""
    images = sorted(DETECTION_TEST_IMAGES.glob("*.png"))
    if not images:
        raise HTTPException(404, "No detection test images")

    img_path = DETECTION_TEST_IMAGES / filename if filename else random.choice(images)
    if not img_path.exists():
        raise HTTPException(404, f"Not found: {filename}")

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = yolo_detect_rgb(rgb)

    gt_boxes = parse_yolo_label(DETECTION_TEST_LABELS / f"{img_path.stem}.txt", w, h)
    vis = img.copy()
    for gb in gt_boxes:
        cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (80, 220, 120), 2)
        cv2.putText(vis, "GT nodule", (gb[0], max(15, gb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120), 1)

    for d in preds:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 255), 2)
        cv2.putText(vis, f"{d['confidence']:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)

    return {
        "filename": img_path.name,
        "has_ground_truth": len(gt_boxes) > 0,
        "num_gt": len(gt_boxes),
        "num_predictions": len(preds),
        "predictions": preds,
        "visualization": encode_image_b64(vis),
    }


# ── DICOM upload (10+ slices) ─────────────────────────────────────────────────

@app.post("/api/predict/dicom")
async def predict_dicom(
    files: list[UploadFile] = File(...),
    true_label: str | None = Form(None, description="Optional: benign | malignant for overlay"),
    gt_x: int | None = Form(None),
    gt_y: int | None = Form(None),
    gt_z: int | None = Form(None),
):
    """
    Upload ≥10 DICOM slices (slices before/after nodule).
    Runs 2.5D YOLO detection + 64³ classification on detected nodules.
    Optional ground-truth center (gt_x, gt_y, gt_z) and label for validation overlay.
    """
    if len(files) < 10:
        raise HTTPException(400, "Upload at least 10 DICOM (.dcm) files")

    tmp_paths: list[str] = []
    try:
        for f in files:
            if not f.filename.lower().endswith(".dcm"):
                raise HTTPException(400, f"Only .dcm allowed, got: {f.filename}")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
            tmp.write(await f.read())
            tmp.close()
            tmp_paths.append(tmp.name)

        volume_hu, pixel_spacing, slice_thickness = load_dicom_series(tmp_paths)
        if volume_hu is None:
            raise HTTPException(400, "Could not build volume from DICOM series")

        d, h, w = volume_hu.shape
        candidates: list[dict[str, Any]] = []

        for z in range(d):
            rgb = create_25d_image(volume_hu, z)
            for det in yolo_detect_rgb(rgb):
                det["z"] = z
                det["diameter_mm"] = det["radius"] * 2 * float(pixel_spacing[0])
                candidates.append(det)

        # Merge nearby detections
        merged: list[dict[str, Any]] = []
        used: set[int] = set()
        for i, c in enumerate(candidates):
            if i in used:
                continue
            group = [c]
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                o = candidates[j]
                if abs(c["z"] - o["z"]) <= 10 and abs(c["y"] - o["y"]) <= 30 and abs(c["x"] - o["x"]) <= 30:
                    group.append(o)
                    used.add(j)
            used.add(i)
            best = max(group, key=lambda x: x["confidence"])
            merged.append(best)
        merged.sort(key=lambda x: x["confidence"], reverse=True)
        merged = merged[:5]

        nodule_results = []
        for det in merged:
            patch = extract_patch_3d(volume_hu, det["z"], det["y"], det["x"])
            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
            cls = predict_volume_tensor(tensor)
            
            det_out = {**det, "classification": cls}
            det_out["mal_prob"] = cls["malignancy"]["probability_malignant"]
            nodule_results.append(det_out)

        # Visualization on center slice (or gt_z)
        vis_z = int(gt_z) if gt_z is not None and 0 <= gt_z < d else d // 2
        sl_gray = _hu_window_slice(volume_hu[vis_z])
        vis = cv2.cvtColor(cv2.resize(sl_gray, (512, 512)), cv2.COLOR_GRAY2BGR)
        sx, sy = 512 / w, 512 / h

        gt_center_vis = None
        if gt_x is not None and gt_y is not None:
            gt_center_vis = (int(gt_x * sx), int(gt_y * sy))

        dets_for_draw = []
        for r in nodule_results:
            if r["z"] == vis_z:
                x1, y1, x2, y2 = r["bbox_xyxy"]
                dets_for_draw.append({
                    "bbox_xyxy": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                    "confidence": r["confidence"],
                    "mal_prob": r["mal_prob"],
                })

        top = nodule_results[0] if nodule_results else None
        pred_lbl = top["classification"]["malignancy"]["label"] if top else None
        pred_prob = top["mal_prob"] if top else None
        vis = draw_overlay(
            vis, dets_for_draw,
            gt_center=gt_center_vis,
            gt_label=true_label.title() if true_label else None,
            pred_label=pred_lbl,
            pred_prob=pred_prob,
        )

        slice_images = []
        for zi in range(d):
            sg = _hu_window_slice(volume_hu[zi])
            bgr = cv2.cvtColor(cv2.resize(sg, (384, 384)), cv2.COLOR_GRAY2BGR)
            for r in nodule_results:
                if r["z"] == zi:
                    x1, y1, x2, y2 = r["bbox_xyxy"]
                    sc = 384 / w
                    cv2.rectangle(bgr, (int(x1*sc), int(y1*sc)), (int(x2*sc), int(y2*sc)), (80, 80, 255), 1)
            slice_images.append({"index": zi, "image": encode_image_b64(bgr)})

        # Save session data for 3D mesh endpoint
        session_id = str(uuid.uuid4())
        session_path = os.path.join(tempfile.gettempdir(), f"lung_session_{session_id}.pkl")
        with open(session_path, "wb") as f:
            pickle.dump({
                "volume_hu": volume_hu,
                "nodule_results": nodule_results
            }, f)

        # Log prediction to monitoring CSV
        _log_prediction(session_id, [d, h, w], nodule_results, user_saved=False)

        return {
            "success": True,
            "session_id": session_id,
            "volume_shape": [d, h, w],
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
            "num_nodules": len(nodule_results),
            "nodules": nodule_results,
            "visualization": encode_image_b64(vis),
            "slices": slice_images,
            "true_label": true_label,
        }
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# Legacy aliases for older frontend paths
@app.get("/api/test/sample")
def legacy_test_sample():
    return classification_test_sample()


# =============================================================================
# Monitoring helpers (inline — no import needed)
# =============================================================================
_LOG_COLUMNS = [
    "timestamp", "scan_id", "model_version", "num_nodules",
    "nodule_index", "yolo_confidence", "mal_probability", "mal_label",
    "diameter_mm", "volume_depth", "volume_height", "volume_width", "user_saved",
]


def _log_prediction(
    scan_id: str,
    volume_shape: list[int],
    nodule_results: list[dict],
    user_saved: bool = False,
) -> None:
    """Append prediction rows to data/monitoring/predictions_log.csv."""
    MONITORING_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not MONITORING_LOG.exists()
    timestamp = datetime.datetime.utcnow().isoformat()
    d, h, w = volume_shape if len(volume_shape) == 3 else (0, 0, 0)

    rows = []
    if not nodule_results:
        rows.append({
            "timestamp": timestamp, "scan_id": scan_id,
            "model_version": MODEL_VERSION, "num_nodules": 0,
            "nodule_index": -1, "yolo_confidence": None,
            "mal_probability": None, "mal_label": "No Nodule",
            "diameter_mm": None,
            "volume_depth": d, "volume_height": h, "volume_width": w,
            "user_saved": user_saved,
        })
    else:
        for i, nd in enumerate(nodule_results):
            cls = nd.get("classification", {})
            mal = cls.get("malignancy", {})
            rows.append({
                "timestamp": timestamp, "scan_id": scan_id,
                "model_version": MODEL_VERSION,
                "num_nodules": len(nodule_results),
                "nodule_index": i,
                "yolo_confidence": round(float(nd.get("confidence", 0)), 4),
                "mal_probability": round(float(nd.get("mal_prob", 0)), 4),
                "mal_label": mal.get("label", "Unknown"),
                "diameter_mm": round(float(nd.get("diameter_mm", 0)), 2),
                "volume_depth": d, "volume_height": h, "volume_width": w,
                "user_saved": user_saved,
            })

    with open(MONITORING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Save labeled data endpoint
# =============================================================================


@app.post("/api/save/labeled")
async def save_labeled(
    session_id: str = Form(...),
    confirmed_indices: str = Form(..., description="JSON array of confirmed nodule indices, e.g. [0,2]"),
    dicom_files: list[UploadFile] = File(...),
):
    """
    Called when the user confirms their nodule feedback in the frontend.

    - Receives the original DICOM files again.
    - Receives which nodule predictions the user verified as correct.
    - Saves the raw DICOM files to data/labeled/<scan_id>/dicoms/
    - Saves a YOLO-format labels.txt for each confirmed nodule.
    - Appends a row to data/labeled/metadata.csv.
    - Updates the predictions_log.csv user_saved flag.
    """
    try:
        indices: list[int] = json.loads(confirmed_indices)
    except Exception:
        raise HTTPException(400, "confirmed_indices must be a valid JSON array, e.g. [0,1]")

    # Load the session to get nodule results
    session_path = os.path.join(tempfile.gettempdir(), f"lung_session_{session_id}.pkl")
    if not os.path.exists(session_path):
        raise HTTPException(
            404,
            "Session not found or expired. Note: the 3D mesh endpoint deletes sessions. "
            "Please re-run prediction and save before requesting the 3D mesh."
        )

    with open(session_path, "rb") as f:
        session = pickle.load(f)

    nodule_results: list[dict] = session["nodule_results"]
    volume_hu: np.ndarray = session["volume_hu"]
    d, h, w = volume_hu.shape

    # Validate indices
    valid_indices = [i for i in indices if 0 <= i < len(nodule_results)]
    if not valid_indices:
        raise HTTPException(400, "No valid nodule indices provided.")

    # Create scan folder under data/labeled/<scan_id>/
    scan_id = session_id  # reuse session_id as unique scan folder name
    scan_dir = LABELED_DIR / scan_id
    dicom_dir = scan_dir / "dicoms"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    # Save raw DICOM files
    for f in dicom_files:
        if not f.filename.lower().endswith(".dcm"):
            continue
        dest = dicom_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)

    # Save YOLO-format labels.txt for each confirmed nodule
    # Format per line: 0 cx_norm cy_norm w_norm h_norm
    labels_lines = []
    for idx in valid_indices:
        nd = nodule_results[idx]
        x1, y1, x2, y2 = nd["bbox_xyxy"]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        labels_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    labels_path = scan_dir / "labels.txt"
    labels_path.write_text("\n".join(labels_lines), encoding="utf-8")

    # Append to labeled/metadata.csv
    LABELED_META.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LABELED_META.exists() or LABELED_META.stat().st_size == 0
    timestamp = datetime.datetime.utcnow().isoformat()
    with open(LABELED_META, "a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["scan_id", "timestamp", "model_version", "source", "num_nodules", "notes"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "scan_id": scan_id,
            "timestamp": timestamp,
            "model_version": MODEL_VERSION,
            "source": "user_upload",
            "num_nodules": len(valid_indices),
            "notes": f"confirmed_indices={valid_indices}",
        })

    # Update monitoring log: mark this scan as user_saved=True
    _log_prediction(scan_id, [d, h, w], [nodule_results[i] for i in valid_indices], user_saved=True)

    return {
        "success": True,
        "scan_id": scan_id,
        "saved_to": str(scan_dir),
        "num_dicoms_saved": len(list(dicom_dir.glob("*.dcm"))),
        "confirmed_nodules": len(valid_indices),
        "labels_file": str(labels_path),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


@app.get("/api/mesh/dicom")
def get_dicom_mesh(session_id: str = Query(...)):
    """
    Retrieves the full lung mesh and nodule meshes generated from a previous prediction session.
    The session data is deleted after the meshes are generated.
    """
    session_path = os.path.join(tempfile.gettempdir(), f"lung_session_{session_id}.pkl")
    if not os.path.exists(session_path):
        raise HTTPException(404, "Session not found or expired")
    
    try:
        with open(session_path, "rb") as f:
            data = pickle.load(f)
        
        volume_hu = data["volume_hu"]
        nodule_results = data["nodule_results"]
        
        # Generate full lung mesh
        lung_mesh = generate_lung_mesh(volume_hu, step_size=4)
        
        # Generate individual nodule meshes
        nodules_with_mesh = []
        for det in nodule_results:
            patch = extract_patch_3d(volume_hu, det["z"], det["y"], det["x"])
            mesh = generate_nodule_mesh(patch)
            det_with_mesh = {**det, "mesh3D": mesh}
            nodules_with_mesh.append(det_with_mesh)
            
        return {
            "success": True,
            "lung_mesh": lung_mesh,
            "nodules": nodules_with_mesh
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to generate meshes: {e}")
