"""
Business Understanding — CRISP-ML(Q) Phase 01
Documents WHY the project exists: objectives, KPIs, success criteria vs current performance.

Outputs:
  output/luna16/raw_data/          — detection clinical context
  output/luna16/preprocessed_data/ — detection KPI charts
  output/LIDC-IDRI/raw_data/       — classification clinical context
  output/LIDC-IDRI/preprocessed_data/ — classification KPI charts
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared style from phase-01 root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import PALETTE, apply_pro_style  # noqa: E402

apply_pro_style()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUSINESS_DIR = Path(__file__).resolve().parent

# LUNA16 / YOLO detection — latest evaluation (update after each run)
LUNA16_DETECTION_METRICS = {
    "mAP@50": {"current": 0.7077681710452679, "target": 0.70},
    "mAP@50-95": {"current": 0.35621387507985136, "target": 0.40},
    "Precision": {"current": 0.7372654155495979, "target": 0.75},
    "Recall": {"current": 0.6806930693069307, "target": 0.85},
    "F1": {"current": 0.7078507028586668, "target": 0.75},
    "F2": {"current": 0.691302159985855, "target": 0.80},
    "Dice": {"current": 0.7078507028586668, "target": 0.75},
    "FDR": {"current": 0.2627345844504021, "target": 0.20, "lower_is_better": True},
    "FPPI": {"current": 26.27345844504021, "target": 4.0, "lower_is_better": True},
}

CLASSIFICATION_KPIS = {
    "Malignancy Recall": {"target": 0.90, "current": 0.927},
    "Malignancy ROC-AUC": {"target": 0.90, "current": 0.952},
    "Spiculation F1": {"target": 0.65, "current": 0.482},
    "Margin Specificity": {"target": 0.70, "current": 0.181},
}


def _out_dirs(dataset: str) -> tuple[Path, Path]:
    base = BUSINESS_DIR / "output" / dataset
    raw = base / "raw_data"
    pre = base / "preprocessed_data"
    raw.mkdir(parents=True, exist_ok=True)
    pre.mkdir(parents=True, exist_ok=True)
    return raw, pre


def plot_luna16_detection_dashboard(path: Path):
    """
    Two-panel KPI chart: (1) detection quality on 0–1 scale,
    (2) clinical workload metrics (FPPI, FDR) on their own scale.
    Avoids mixing incomparable metrics on one axis.
    """
    quality = {k: v for k, v in LUNA16_DETECTION_METRICS.items() if not v.get("lower_is_better")}
    workload = {k: v for k, v in LUNA16_DETECTION_METRICS.items() if v.get("lower_is_better")}

    fig, (ax_q, ax_w) = plt.subplots(1, 2, figsize=(13, 6),
                                     gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("LUNA16 Detection — KPIs vs Targets (YOLOv8 Evaluation)",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.02)

    # ── Panel 1: quality metrics (higher is better) ──
    names = list(quality.keys())
    y = np.arange(len(names))
    currents = [quality[n]["current"] for n in names]
    targets = [quality[n]["target"] for n in names]
    met = [c >= t for c, t in zip(currents, targets)]
    bar_colors = [PALETTE["green"] if m else PALETTE["accent2"] for m in met]

    ax_q.barh(y, currents, height=0.55, color=bar_colors, edgecolor=PALETTE["bg"], linewidth=1.2,
              label="Current", zorder=2)
    for i, (c, t, name) in enumerate(zip(currents, targets, names)):
        ax_q.plot(t, i, marker="|", markersize=22, markeredgewidth=3,
                  color=PALETTE["accent5"], zorder=3, label="Target" if i == 0 else "")
        ax_q.text(c + 0.02, i, f"{c:.3f}", va="center", fontsize=9, fontweight="bold", color=PALETTE["text"])

    ax_q.set_yticks(y)
    ax_q.set_yticklabels(names)
    ax_q.invert_yaxis()
    ax_q.set_xlim(0, 1.05)
    ax_q.set_xlabel("Score (0 – 1)")
    ax_q.set_title("Detection Quality", pad=10)
    ax_q.axvline(1.0, color=PALETTE["grid"], ls=":", lw=0.8, alpha=0.5)
    ax_q.grid(axis="x")
    ax_q.legend(loc="lower right", framealpha=0.85)

    # ── Panel 2: workload (lower is better) ──
    w_names = list(workload.keys())
    yw = np.arange(len(w_names))
    w_curr = [workload[n]["current"] for n in w_names]
    w_targ = [workload[n]["target"] for n in w_names]
    w_met = [c <= t for c, t in zip(w_curr, w_targ)]
    w_colors = [PALETTE["green"] if m else PALETTE["accent2"] for m in w_met]

    ax_w.barh(yw, w_curr, height=0.45, color=w_colors, edgecolor=PALETTE["bg"], linewidth=1.2)
    for i, (c, t, name) in enumerate(zip(w_curr, w_targ, w_names)):
        ax_w.plot(t, i, marker="|", markersize=22, markeredgewidth=3, color=PALETTE["accent5"], zorder=3)
        label = f"{c:.3f}" if name == "FDR" else f"{c:.2f}"
        ax_w.text(c + max(w_curr) * 0.03, i, label, va="center", fontsize=10,
                  fontweight="bold", color=PALETTE["text"])

    ax_w.set_yticks(yw)
    ax_w.set_yticklabels([f"{n}  (↓ better)" for n in w_names])
    ax_w.invert_yaxis()
    ax_w.set_xlim(0, max(w_curr) * 1.25)
    ax_w.set_xlabel("Rate / count per image")
    ax_w.set_title("Clinical Workload", pad=10)
    ax_w.grid(axis="x")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_kpi_comparison(kpis: dict, title: str, path: Path, invert_keys: set | None = None):
    """Grouped bar: target vs current for each KPI."""
    invert_keys = invert_keys or set()
    names = list(kpis.keys())
    targets = []
    currents = []
    colors_t = []
    colors_c = []

    for name in names:
        t = kpis[name]["target"]
        c = kpis[name]["current"]
        inv = kpis[name].get("invert") or name in invert_keys
        targets.append(t)
        currents.append(c)
        if inv:
            met = c <= t
        else:
            met = c >= t
        colors_t.append(PALETTE["accent4"])
        colors_c.append(PALETTE["green"] if met else PALETTE["accent2"])

    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, targets, w, label="Target", color=colors_t, edgecolor=PALETTE["bg"], linewidth=1.2)
    ax.bar(x + w / 2, currents, w, label="Current", color=colors_c, edgecolor=PALETTE["bg"], linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Score (metric-specific scale)")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(axis="y")
    ax.set_ylim(0, max(max(targets), max(currents)) * 1.15)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_pipeline_overview(path: Path):
    """High-level dual-pipeline diagram for stakeholders."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 2.2, "Raw CT\nLUNA16 / LIDC"),
        (2.5, 2.2, "Preprocess\nHU · Mask · Patches"),
        (5.0, 3.0, "YOLO\nDetection\n(LUNA16)"),
        (5.0, 1.2, "3D CNN\nClassification\n(LIDC-IDRI)"),
        (7.5, 2.2, "Evaluate\nRecall · FPPI · AUC"),
        (9.0, 2.2, "Radiologist\nDecision Support"),
    ]
    for x, y, text in boxes:
        ax.text(
            x, y, text, ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc=PALETTE["panel"], ec=PALETTE["accent1"], lw=1.5),
            color=PALETTE["text"],
        )

    arrow_kw = dict(arrowstyle="->", color=PALETTE["accent3"], lw=1.8)
    ax.annotate("", xy=(2.3, 2.2), xytext=(1.4, 2.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.7, 3.0), xytext=(3.5, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.7, 1.2), xytext=(3.5, 1.9), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.2, 2.2), xytext=(6.2, 2.2), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.7, 2.2), xytext=(8.0, 2.2), arrowprops=arrow_kw)

    ax.set_title("Project Pipeline — Business View", fontsize=14, fontweight="bold", color=PALETTE["text"], pad=16)
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def plot_clinical_priorities(path: Path, dataset_label: str, priorities: list[str]):
    """Raw-data folder: clinical priority ranking (business context)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(priorities))
    scores = np.linspace(1.0, 0.55, len(priorities))
    colors = [PALETTE["gradient"][i % len(PALETTE["gradient"])] for i in range(len(priorities))]

    ax.barh(y, scores, color=colors, height=0.55, edgecolor=PALETTE["bg"])
    ax.set_yticks(y)
    ax.set_yticklabels(priorities)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Relative Priority (normalized)")
    ax.set_title(f"Clinical Priorities — {dataset_label}")
    ax.grid(axis="x")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    print("[INFO] Business Understanding — generating outputs")

    # LUNA16 / detection
    luna_raw, luna_pre = _out_dirs("luna16")
    plot_clinical_priorities(
        luna_raw / "clinical_priorities.png",
        "LUNA16 Detection",
        [
            "Minimize missed nodules (high recall)",
            "Reduce false positives per scan (low FPPI)",
            "Accurate localization (mAP)",
            "Assist triage, not replace radiologist",
        ],
    )
    plot_pipeline_overview(luna_raw / "pipeline_overview.png")
    plot_luna16_detection_dashboard(luna_pre / "kpi_targets_vs_current.png")

    # LIDC-IDRI / classification
    lidc_raw, lidc_pre = _out_dirs("LIDC-IDRI")
    plot_clinical_priorities(
        lidc_raw / "clinical_priorities.png",
        "LIDC-IDRI Classification",
        [
            "Reliable benign vs malignant stratification",
            "Capture malignant cases (high recall)",
            "Morphology features (spiculation, margin)",
            "Transparent limitations & human oversight",
        ],
    )
    plot_kpi_comparison(
        CLASSIFICATION_KPIS,
        "Classification KPIs — Target vs Current (LIDC-IDRI)",
        lidc_pre / "kpi_targets_vs_current.png",
    )

    print(f"[DONE] Outputs under {BUSINESS_DIR / 'output'}")


if __name__ == "__main__":
    main()
