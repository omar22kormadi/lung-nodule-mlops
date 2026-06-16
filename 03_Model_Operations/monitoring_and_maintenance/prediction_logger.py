"""
prediction_logger.py
====================
Logs every prediction made by the API to a running CSV file.
Called by api.py after each /api/predict/dicom response.

Log file: data/monitoring/predictions_log.csv
"""

from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths (relative to project root — two levels up from this file)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "monitoring" / "predictions_log.csv"

# CSV columns
_COLUMNS = [
    "timestamp",
    "scan_id",
    "model_version",
    "num_nodules",
    "nodule_index",
    "yolo_confidence",
    "mal_probability",
    "mal_label",
    "diameter_mm",
    "volume_depth",
    "volume_height",
    "volume_width",
    "user_saved",          # True if user confirmed and saved labeled data
]


def _ensure_log_file() -> None:
    """Create log directory and CSV header if they don't exist."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()


def log_prediction(
    scan_id: str,
    model_version: str,
    volume_shape: list[int],
    nodule_results: list[dict[str, Any]],
    user_saved: bool = False,
) -> None:
    """
    Append one row per detected nodule to predictions_log.csv.

    Parameters
    ----------
    scan_id        : Unique ID for this scan (session_id from api.py).
    model_version  : e.g. "yolov8m_v3.3 + r2plus1d_v3.3"
    volume_shape   : [depth, height, width] of the DICOM volume.
    nodule_results : List of nodule dicts returned by predict_dicom().
    user_saved     : Whether the user confirmed and saved this scan.
    """
    _ensure_log_file()
    timestamp = datetime.datetime.utcnow().isoformat()
    d, h, w = volume_shape if len(volume_shape) == 3 else (0, 0, 0)

    rows = []
    if not nodule_results:
        # Log a "no nodule" row so we track clean scans too
        rows.append({
            "timestamp": timestamp,
            "scan_id": scan_id,
            "model_version": model_version,
            "num_nodules": 0,
            "nodule_index": -1,
            "yolo_confidence": None,
            "mal_probability": None,
            "mal_label": "No Nodule",
            "diameter_mm": None,
            "volume_depth": d,
            "volume_height": h,
            "volume_width": w,
            "user_saved": user_saved,
        })
    else:
        for i, nodule in enumerate(nodule_results):
            cls = nodule.get("classification", {})
            mal = cls.get("malignancy", {})
            rows.append({
                "timestamp": timestamp,
                "scan_id": scan_id,
                "model_version": model_version,
                "num_nodules": len(nodule_results),
                "nodule_index": i,
                "yolo_confidence": round(float(nodule.get("confidence", 0)), 4),
                "mal_probability": round(float(nodule.get("mal_prob", 0)), 4),
                "mal_label": mal.get("label", "Unknown"),
                "diameter_mm": round(float(nodule.get("diameter_mm", 0)), 2),
                "volume_depth": d,
                "volume_height": h,
                "volume_width": w,
                "user_saved": user_saved,
            })

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writerows(rows)

    print(f"[Logger] Logged {len(rows)} row(s) for scan_id={scan_id}")
