"""
monitoring_report.py
====================
Generates a human-readable monitoring report from the prediction log.
Includes drift detection results, summary statistics, and saved scan counts.

Run manually or on a schedule:
    python monitoring_report.py

Output: prints to stdout + writes report to data/monitoring/latest_report.txt
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd

from drift_detection import detect_drift

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent

LOG_PATH    = PROJECT_ROOT / "data" / "monitoring" / "predictions_log.csv"
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
REPORT_PATH = PROJECT_ROOT / "data" / "monitoring" / "latest_report.txt"

SEP = "=" * 70


def _load_log() -> pd.DataFrame | None:
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return None
    df = pd.read_csv(LOG_PATH)
    return df if not df.empty else None


def _count_labeled_scans() -> int:
    if not LABELED_DIR.exists():
        return 0
    return sum(1 for p in LABELED_DIR.iterdir() if p.is_dir())


def generate_report() -> str:
    lines = []
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append(SEP)
    lines.append("   LUNG NODULE AI — MONITORING REPORT")
    lines.append(f"   Generated: {now}")
    lines.append(SEP)

    df = _load_log()

    # ── Summary Stats ────────────────────────────────────────────────────────
    lines.append("\n📊  PREDICTION LOG SUMMARY")
    lines.append("-" * 40)

    if df is None:
        lines.append("  No predictions logged yet.")
    else:
        total_scans = df["scan_id"].nunique()
        total_nodule_rows = df[df["nodule_index"] >= 0]
        total_nodules = len(total_nodule_rows)
        no_nodule_scans = df[df["nodule_index"] == -1]["scan_id"].nunique()
        malignant_count = (total_nodule_rows["mal_label"] == "Malignant").sum()
        mal_rate = malignant_count / max(total_nodules, 1)
        avg_conf = total_nodule_rows["yolo_confidence"].mean()
        avg_mal_prob = total_nodule_rows["mal_probability"].mean()
        avg_diam = total_nodule_rows["diameter_mm"].mean()
        saved_count = df[df["user_saved"] == True]["scan_id"].nunique()

        lines.append(f"  Total scans processed   : {total_scans}")
        lines.append(f"  Scans with no nodule    : {no_nodule_scans}")
        lines.append(f"  Total nodules detected  : {total_nodules}")
        lines.append(f"  Malignant rate          : {mal_rate:.1%}")
        lines.append(f"  Avg YOLO confidence     : {avg_conf:.3f}")
        lines.append(f"  Avg malignancy prob     : {avg_mal_prob:.3f}")
        lines.append(f"  Avg nodule diameter     : {avg_diam:.1f} mm")
        lines.append(f"  User-saved scans        : {saved_count}")

        # Last 5 scans
        lines.append("\n  Last 5 scan IDs:")
        for sid in df["scan_id"].unique()[-5:]:
            lines.append(f"    · {sid}")

    # ── Labeled Data ─────────────────────────────────────────────────────────
    lines.append("\n🏷️   LABELED DATA SUMMARY")
    lines.append("-" * 40)
    n_labeled = _count_labeled_scans()
    lines.append(f"  Labeled scan folders in data/labeled/ : {n_labeled}")
    if LABELED_DIR.exists():
        meta = LABELED_DIR / "metadata.csv"
        if meta.exists():
            mdf = pd.read_csv(meta)
            if not mdf.empty:
                lines.append(f"  Metadata rows             : {len(mdf)}")
                for _, row in mdf.tail(3).iterrows():
                    lines.append(f"    · [{row.get('timestamp','?')}] {row.get('scan_id','?')} — {row.get('num_nodules','?')} nodule(s)")

    # ── Drift Detection ──────────────────────────────────────────────────────
    lines.append("\n🔍  DRIFT DETECTION (last 100 predictions)")
    lines.append("-" * 40)
    try:
        drift = detect_drift(n=100)
        status = drift.get("status", "unknown")
        icon = "⚠️ " if status == "DRIFT_DETECTED" else "✅ "
        lines.append(f"  Overall Status : {icon} {status}")
        lines.append(f"  Rows analysed  : {drift.get('total_rows_analysed', 0)}")

        # ── KS / rate checks ──
        ks_keys = ["yolo_confidence", "mal_probability", "diameter_mm", "malignant_rate"]
        for check_name in ks_keys:
            result = drift.get("checks", {}).get(check_name, {})
            flag = "⚠️ DRIFT" if result.get("is_drifted") else "  OK   "
            rec  = result.get("recent_mean", result.get("recent_rate", result.get("reason", "N/A")))
            base = result.get("baseline_mean", result.get("baseline_rate", "N/A"))
            lines.append(f"  [{flag}] {check_name:<22} recent={rec}  baseline={base}")

        # ── HU intensity drift ──
        lines.append("")
        lines.append("  📡  DATA DRIFT — HU Intensity")
        hu = drift.get("checks", {}).get("hu_intensity", {})
        if hu.get("reason"):
            lines.append(f"  [  N/A  ] hu_intensity           {hu['reason']} (need ≥{hu.get('min_required', 5)} scans, have {hu.get('n_scans', 0)})")
        else:
            flag = "⚠️ DRIFT" if hu.get("is_drifted") else "  OK   "
            lines.append(
                f"  [{flag}] hu_intensity           "
                f"recent={hu.get('recent_mean_hu')} HU  "
                f"baseline={hu.get('baseline_mean_hu')} HU  "
                f"({hu.get('deviation_sigma')}σ, threshold {hu.get('threshold_sigma')}σ)  "
                f"n={hu.get('n_scans')} scans"
            )

        # ── Performance drift ──
        lines.append("")
        lines.append("  📈  PERFORMANCE DRIFT")
        det = drift.get("checks", {}).get("detection_map50", {})
        if det.get("reason"):
            lines.append(f"  [  N/A  ] detection_map50        {det['reason']} (need ≥{det.get('min_required', 10)} scans, have {det.get('n_scans', 0)})")
        else:
            flag = "⚠️ DRIFT" if det.get("is_drifted") else "  OK   "
            lines.append(
                f"  [{flag}] detection_map50        "
                f"mAP@50={det.get('map50')}  "
                f"sensitivity={det.get('sensitivity')}  "
                f"threshold={det.get('threshold_map50')}  "
                f"n={det.get('n_scans')} scans"
            )

        cls = drift.get("checks", {}).get("classification_auc", {})
        if cls.get("reason"):
            lines.append(f"  [  N/A  ] classification_auc     {cls['reason']} (need ≥{cls.get('min_required', 10)} samples, have {cls.get('n_samples', 0)})")
        else:
            flag = "⚠️ DRIFT" if cls.get("is_drifted") else "  OK   "
            lines.append(
                f"  [{flag}] classification_auc     "
                f"ROC-AUC={cls.get('roc_auc')}  "
                f"threshold={cls.get('threshold_auc')}  "
                f"n={cls.get('n_samples')} samples"
            )

    except Exception as e:
        lines.append(f"  Drift check failed: {e}")

    # ── Footer ───────────────────────────────────────────────────────────────
    lines.append(f"\n{SEP}")
    lines.append("  Run `python drift_detection.py` for full drift JSON.")
    lines.append(SEP)

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[Report] Saved to: {REPORT_PATH}")
