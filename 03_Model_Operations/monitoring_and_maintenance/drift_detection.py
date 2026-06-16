"""
drift_detection.py
==================
Compares recent prediction statistics from predictions_log.csv
against baseline_stats.json and flags distribution drift.

Run manually or on a schedule:
    python drift_detection.py

Outputs a drift report dict (also used by monitoring_report.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent

LOG_PATH      = PROJECT_ROOT / "data" / "monitoring" / "predictions_log.csv"
BASELINE_PATH = PROJECT_ROOT / "data" / "monitoring" / "baseline_stats.json"

# How many recent predictions to compare against baseline
RECENT_WINDOW = 100

# p-value threshold for KS-test: flag drift if p < this
DRIFT_PVALUE_THRESHOLD = 0.05

# Absolute threshold for malignant_rate drift
MALIGNANT_RATE_DELTA = 0.15


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_recent_predictions(n: int = RECENT_WINDOW) -> pd.DataFrame | None:
    """Load the last `n` rows from the prediction log."""
    if not LOG_PATH.exists():
        print(f"[Drift] Log not found: {LOG_PATH}")
        return None
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        print("[Drift] Log is empty — no predictions to analyse yet.")
        return None
    # Only rows with actual nodule detections
    df = df[df["nodule_index"] >= 0]
    return df.tail(n)


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)


def ks_drift(recent_values: pd.Series, baseline: dict, field: str) -> dict:
    """
    Kolmogorov-Smirnov test against a Gaussian approximation of the baseline.
    Returns a dict with is_drifted, p_value, recent_mean, baseline_mean.
    """
    values = recent_values.dropna().astype(float).values
    if len(values) < 10:
        return {"is_drifted": False, "reason": "not enough data", "n": len(values)}

    b = baseline[field]
    # Generate reference sample from baseline distribution
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


def detect_drift(n: int = RECENT_WINDOW) -> dict:
    """
    Main function. Returns a full drift report dict.
    """
    df = load_recent_predictions(n)
    if df is None or df.empty:
        return {"status": "no_data", "checks": {}}

    baseline = load_baseline()

    checks = {}

    # 1. YOLO confidence drift
    checks["yolo_confidence"] = ks_drift(df["yolo_confidence"], baseline, "yolo_confidence")

    # 2. Malignancy probability drift
    checks["mal_probability"] = ks_drift(df["mal_probability"], baseline, "mal_probability")

    # 3. Nodule diameter drift
    checks["diameter_mm"] = ks_drift(df["diameter_mm"], baseline, "diameter_mm")

    # 4. Malignant rate drift (simple absolute delta)
    recent_mal_rate = (df["mal_label"] == "Malignant").mean()
    baseline_mal_rate = baseline["malignant_rate"]
    mal_rate_delta = abs(float(recent_mal_rate) - baseline_mal_rate)
    checks["malignant_rate"] = {
        "is_drifted": mal_rate_delta > MALIGNANT_RATE_DELTA,
        "recent_rate": round(float(recent_mal_rate), 4),
        "baseline_rate": baseline_mal_rate,
        "delta": round(mal_rate_delta, 4),
    }

    # Overall status
    any_drift = any(v.get("is_drifted", False) for v in checks.values())
    report = {
        "status": "DRIFT_DETECTED" if any_drift else "OK",
        "window": n,
        "total_rows_analysed": len(df),
        "checks": checks,
    }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json as _json
    report = detect_drift()
    print(_json.dumps(report, indent=2))
    if report["status"] == "DRIFT_DETECTED":
        print("\n⚠️  DRIFT DETECTED — consider retraining or reviewing recent predictions.")
    else:
        print("\n✅  No significant drift detected.")
