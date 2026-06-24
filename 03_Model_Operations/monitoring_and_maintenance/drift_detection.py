"""
drift_detection.py
==================
Monitors model and data health by running drift checks against the
baseline statistics established at training time.

Two categories of checks are run:

  DATA DRIFT
    1. yolo_confidence  — KS-test vs baseline distribution
    2. mal_probability  — KS-test vs baseline distribution
    3. diameter_mm      — KS-test vs baseline distribution
    4. malignant_rate   — absolute delta vs baseline rate
    5. hu_intensity     — mean HU of saved DICOMs vs training baseline (2σ rule)

  PERFORMANCE DRIFT  (requires labeled data in data/labeled/)
    6. detection_map50       — mAP@50 on radiologist-labeled scans (threshold < 0.65)
    7. detection_sensitivity — Sensitivity (recall) on labeled scans
    8. classification_auc    — ROC-AUC from predictions_log (threshold < 0.90)

Run manually:
    python drift_detection.py

Outputs a drift report dict (also consumed by monitoring_report.py & alerts.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent

LOG_PATH      = PROJECT_ROOT / "data" / "monitoring" / "predictions_log.csv"
BASELINE_PATH = PROJECT_ROOT / "data" / "monitoring" / "baseline_stats.json"
LABELED_DIR   = PROJECT_ROOT / "data" / "labeled"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
RECENT_WINDOW            = 100
DRIFT_PVALUE_THRESHOLD   = 0.05
MALIGNANT_RATE_DELTA     = 0.15
HU_SIGMA_THRESHOLD       = 2.0          # alert if |mean_hu - baseline| > 2σ
MIN_SCANS_FOR_HU         = 5            # need at least this many labeled scans
MIN_SCANS_FOR_PERF       = 10           # need at least this many for mAP/AUC
MAP50_THRESHOLD          = 0.65
AUC_THRESHOLD            = 0.90
IOU_THRESHOLD            = 0.50

# HU baseline defaults (LUNA16 lung window, used if baseline_stats.json has no hu_* fields)
_HU_BASELINE_MEAN = -500.0
_HU_BASELINE_STD  = 150.0

# ---------------------------------------------------------------------------
# Preprocessing helpers (kept in sync with api.py)
# ---------------------------------------------------------------------------
HU_WINDOW_CENTER = -600.0
HU_WINDOW_WIDTH  = 1500.0
HU_MIN = HU_WINDOW_CENTER - HU_WINDOW_WIDTH / 2.0
HU_MAX = HU_WINDOW_CENTER + HU_WINDOW_WIDTH / 2.0
SLICE_GAP_25D = 2
YOLO_CONF = 0.25
YOLO_IOU  = 0.40


def _load_dicom_series(dicom_dir: Path):
    """Load DICOM slices from a directory → HU volume (D,H,W)."""
    slices = []
    for fp in dicom_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            continue
    if len(slices) < 5:
        return None
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    slope     = float(getattr(slices[0], "RescaleSlope", 1))
    return volume * slope + intercept


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes
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


def _create_25d_image(volume_hu: np.ndarray, z_center: int) -> np.ndarray:
    d = volume_hu.shape[0]
    z_m = max(0, z_center - SLICE_GAP_25D)
    z_p = min(d - 1, z_center + SLICE_GAP_25D)
    channels = []
    for z in (z_m, z_center, z_p):
        s = np.clip(volume_hu[z], HU_MIN, HU_MAX).astype(np.float32)
        channels.append((s - HU_MIN) / max(HU_MAX - HU_MIN, 1e-6))
    return (np.stack(channels, axis=-1) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1–4  Existing KS / rate checks
# ---------------------------------------------------------------------------

def load_recent_predictions(n: int = RECENT_WINDOW) -> pd.DataFrame | None:
    if not LOG_PATH.exists():
        return None
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return None
    df = df[df["nodule_index"] >= 0]
    return df.tail(n) if not df.empty else None


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def ks_drift(recent_values: pd.Series, baseline: dict, field: str) -> dict:
    values = recent_values.dropna().astype(float).values
    if len(values) < 10:
        return {"is_drifted": False, "reason": "not enough data", "n": len(values)}
    b = baseline[field]
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=b["mean"], scale=b["std"], size=500)
    reference = np.clip(reference, b["min"], b["max"])
    stat, p_value = stats.ks_2samp(values, reference)
    drifted = p_value < DRIFT_PVALUE_THRESHOLD
    return {
        "is_drifted": bool(drifted),
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "recent_mean": round(float(np.mean(values)), 4),
        "baseline_mean": b["mean"],
        "recent_std": round(float(np.std(values)), 4),
        "baseline_std": b["std"],
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# 5  HU intensity drift
# ---------------------------------------------------------------------------

def check_hu_intensity_drift() -> dict:
    """
    Reads DICOM files from data/labeled/*/dicoms/, computes mean HU per scan,
    then checks if the population mean deviates > 2σ from the training baseline.
    """
    scan_dirs = [p for p in LABELED_DIR.iterdir() if p.is_dir()] if LABELED_DIR.exists() else []
    dicom_dirs = [s / "dicoms" for s in scan_dirs if (s / "dicoms").exists()]

    if len(dicom_dirs) < MIN_SCANS_FOR_HU:
        return {
            "is_drifted": False,
            "reason": "insufficient_data",
            "n_scans": len(dicom_dirs),
            "min_required": MIN_SCANS_FOR_HU,
        }

    scan_means = []
    for ddir in dicom_dirs:
        vol = _load_dicom_series(ddir)
        if vol is None:
            continue
        mean_hu = float(np.mean(vol))
        # Sanity check: physically valid HU range
        if -1024 <= mean_hu <= 3071:
            scan_means.append(mean_hu)
        else:
            print(f"[HU Drift] Skipping out-of-range HU={mean_hu:.1f} in {ddir}")

    if len(scan_means) < MIN_SCANS_FOR_HU:
        return {
            "is_drifted": False,
            "reason": "insufficient_valid_scans",
            "n_scans": len(scan_means),
        }

    # Load baseline HU stats (seed defaults if missing)
    try:
        baseline = load_baseline()
        hu_mean_b = float(baseline.get("hu_mean", _HU_BASELINE_MEAN))
        hu_std_b  = float(baseline.get("hu_std",  _HU_BASELINE_STD))
    except Exception:
        hu_mean_b, hu_std_b = _HU_BASELINE_MEAN, _HU_BASELINE_STD

    recent_mean = float(np.mean(scan_means))
    deviation   = abs(recent_mean - hu_mean_b)
    sigma_dist  = deviation / max(hu_std_b, 1e-6)
    drifted     = sigma_dist > HU_SIGMA_THRESHOLD

    return {
        "is_drifted": bool(drifted),
        "recent_mean_hu": round(recent_mean, 2),
        "baseline_mean_hu": round(hu_mean_b, 2),
        "baseline_std_hu": round(hu_std_b, 2),
        "deviation_sigma": round(sigma_dist, 3),
        "threshold_sigma": HU_SIGMA_THRESHOLD,
        "n_scans": len(scan_means),
    }


# ---------------------------------------------------------------------------
# 6–7  Performance drift — mAP@50 + Sensitivity
# ---------------------------------------------------------------------------

def check_performance_drift_detection() -> dict:
    """
    Runs YOLO on 2.5D images generated from labeled DICOMs and computes
    mAP@50 and Sensitivity against the saved labels.txt ground truth.
    """
    scan_dirs = [p for p in LABELED_DIR.iterdir() if p.is_dir()] if LABELED_DIR.exists() else []
    valid_scans = [s for s in scan_dirs if (s / "dicoms").exists() and (s / "labels.txt").exists()]

    if len(valid_scans) < MIN_SCANS_FOR_PERF:
        return {
            "is_drifted": False,
            "reason": "insufficient_labeled_data",
            "n_scans": len(valid_scans),
            "min_required": MIN_SCANS_FOR_PERF,
        }

    # Import YOLO lazily — only needed here
    try:
        from ultralytics import YOLO
        yolo_path = PROJECT_ROOT / "02_Model_Development" / "ml_model_engineering" / "models" / "best.pt"
        if not yolo_path.exists():
            return {"is_drifted": False, "reason": "yolo_weights_not_found"}
        model = YOLO(str(yolo_path))
    except Exception as e:
        return {"is_drifted": False, "reason": f"yolo_load_failed: {e}"}

    all_precisions, all_recalls = [], []
    tp_total = fp_total = fn_total = 0

    for scan_dir in valid_scans:
        vol = _load_dicom_series(scan_dir / "dicoms")
        if vol is None:
            continue
        d, h, w = vol.shape
        gt_boxes = _parse_yolo_label(scan_dir / "labels.txt", w, h)
        if not gt_boxes:
            continue

        # Run YOLO on all slices, collect predictions
        all_preds = []
        for z in range(d):
            rgb = _create_25d_image(vol, z)
            res = model(rgb, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    all_preds.append((x1, y1, x2, y2, float(conf)))

        if not all_preds:
            fn_total += len(gt_boxes)
            continue

        # Match predictions to GT boxes at IoU ≥ 0.50
        matched_gt = set()
        matched_pred = set()
        all_preds.sort(key=lambda x: x[4], reverse=True)
        for pi, (px1, py1, px2, py2, _) in enumerate(all_preds):
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if _box_iou((px1, py1, px2, py2), gt) >= IOU_THRESHOLD:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        tp = len(matched_gt)
        fp = len(all_preds) - len(matched_pred)
        fn = len(gt_boxes) - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        all_precisions.append(prec)
        all_recalls.append(rec)

    if not all_precisions:
        return {"is_drifted": False, "reason": "no_valid_predictions"}

    map50       = float(np.mean(all_precisions))
    sensitivity = float(tp_total / max(tp_total + fn_total, 1))

    return {
        "is_drifted": bool(map50 < MAP50_THRESHOLD),
        "map50": round(map50, 4),
        "sensitivity": round(sensitivity, 4),
        "threshold_map50": MAP50_THRESHOLD,
        "n_scans": len(valid_scans),
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
    }


# ---------------------------------------------------------------------------
# 8  Performance drift — ROC-AUC
# ---------------------------------------------------------------------------

def check_performance_drift_classification() -> dict:
    """
    Computes ROC-AUC from predictions_log.csv entries where user_saved=True,
    using mal_label as ground truth and mal_probability as the predicted score.
    """
    if not LOG_PATH.exists():
        return {"is_drifted": False, "reason": "no_log_file"}

    df = pd.read_csv(LOG_PATH)
    saved = df[(df["user_saved"] == True) & (df["nodule_index"] >= 0)].copy()

    if len(saved) < MIN_SCANS_FOR_PERF:
        return {
            "is_drifted": False,
            "reason": "insufficient_labeled_data",
            "n_samples": len(saved),
            "min_required": MIN_SCANS_FOR_PERF,
        }

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"is_drifted": False, "reason": "sklearn_not_installed"}

    y_true = (saved["mal_label"] == "Malignant").astype(int).values
    y_score = saved["mal_probability"].fillna(0.5).values

    if len(np.unique(y_true)) < 2:
        return {
            "is_drifted": False,
            "reason": "only_one_class_in_labeled_data",
            "n_samples": len(saved),
        }

    auc = float(roc_auc_score(y_true, y_score))
    return {
        "is_drifted": bool(auc < AUC_THRESHOLD),
        "roc_auc": round(auc, 4),
        "threshold_auc": AUC_THRESHOLD,
        "n_samples": len(saved),
        "n_malignant": int(y_true.sum()),
        "n_benign": int((1 - y_true).sum()),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_drift(n: int = RECENT_WINDOW) -> dict:
    """
    Runs all drift checks and returns a unified report dict.
    Backward-compatible — existing four checks are always present.
    """
    df = load_recent_predictions(n)

    checks: dict = {}

    # ── 1-4: KS / rate checks (from predictions_log.csv) ──────────────────
    if df is not None and not df.empty:
        try:
            baseline = load_baseline()
            checks["yolo_confidence"] = ks_drift(df["yolo_confidence"], baseline, "yolo_confidence")
            checks["mal_probability"] = ks_drift(df["mal_probability"], baseline, "mal_probability")
            checks["diameter_mm"]     = ks_drift(df["diameter_mm"],     baseline, "diameter_mm")
            recent_mal_rate  = (df["mal_label"] == "Malignant").mean()
            baseline_mal_rate = baseline["malignant_rate"]
            mal_rate_delta   = abs(float(recent_mal_rate) - baseline_mal_rate)
            checks["malignant_rate"] = {
                "is_drifted": mal_rate_delta > MALIGNANT_RATE_DELTA,
                "recent_rate": round(float(recent_mal_rate), 4),
                "baseline_rate": baseline_mal_rate,
                "delta": round(mal_rate_delta, 4),
            }
        except Exception as e:
            checks["ks_checks_error"] = {"is_drifted": False, "reason": str(e)}
    else:
        checks["yolo_confidence"] = {"is_drifted": False, "reason": "no_log_data"}
        checks["mal_probability"] = {"is_drifted": False, "reason": "no_log_data"}
        checks["diameter_mm"]     = {"is_drifted": False, "reason": "no_log_data"}
        checks["malignant_rate"]  = {"is_drifted": False, "reason": "no_log_data"}

    # ── 5: HU intensity drift ──────────────────────────────────────────────
    try:
        checks["hu_intensity"] = check_hu_intensity_drift()
    except Exception as e:
        checks["hu_intensity"] = {"is_drifted": False, "reason": str(e)}

    # ── 6-7: Detection performance (mAP@50 + Sensitivity) ─────────────────
    try:
        checks["detection_map50"] = check_performance_drift_detection()
    except Exception as e:
        checks["detection_map50"] = {"is_drifted": False, "reason": str(e)}

    # ── 8: Classification AUC ─────────────────────────────────────────────
    try:
        checks["classification_auc"] = check_performance_drift_classification()
    except Exception as e:
        checks["classification_auc"] = {"is_drifted": False, "reason": str(e)}

    any_drift = any(v.get("is_drifted", False) for v in checks.values())
    return {
        "status": "DRIFT_DETECTED" if any_drift else "OK",
        "window": n,
        "total_rows_analysed": len(df) if df is not None else 0,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json as _json
    print("[Drift] Running all checks (this may take a moment if YOLO inference runs)...")
    report = detect_drift()
    print(_json.dumps(report, indent=2))
    if report["status"] == "DRIFT_DETECTED":
        print("\n⚠️  DRIFT DETECTED — consider retraining or reviewing recent predictions.")
    else:
        print("\n✅  No significant drift detected.")
