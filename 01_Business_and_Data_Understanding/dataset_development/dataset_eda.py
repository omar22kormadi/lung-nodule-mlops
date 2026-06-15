"""
Dataset Development EDA — CRISP-ML(Q) Phase 01
Documents WHAT the data is: raw-source and preprocessed-dataset analysis.

Datasets:
  - LUNA16  → lung nodule detection (YOLO 2.5D slices)
  - LIDC-IDRI → benign / malignant classification (64³ patches)

Outputs:
  output/luna16/raw_data/ | preprocessed_data/
  output/LIDC-IDRI/raw_data/ | preprocessed_data/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import PALETTE, apply_pro_style  # noqa: E402

apply_pro_style()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DEV_DIR = Path(__file__).resolve().parent

# Data paths
LIDC_RAW_CANDIDATES = [
    Path(r"G:\manifest-1600709154662\LIDC-IDRI"),
    Path(r"C:\Users\amork\Desktop\data\manifest-1600709154662\LIDC-IDRI"),
]


def _resolve_lidc_raw_dir() -> Path | None:
    for p in LIDC_RAW_CANDIDATES:
        if p.exists():
            return p
    return None
CLASSIFICATION_DIR = PROJECT_ROOT / "data" / "classification_dataset"
CLASSIFICATION_META = CLASSIFICATION_DIR / "metadata.csv"

LUNA16_BASES = [Path("H:/luna16"), Path("G:/luna16")]
YOLO_DIR = PROJECT_ROOT / "data" / "luna16_yolo_dataset_v4"
YOLO_META = YOLO_DIR / "metadata" / "dataset_metadata.csv"


def out_paths(dataset: str) -> dict[str, Path]:
    base = DATASET_DEV_DIR / "output" / dataset
    paths = {
        "base": base,
        "raw": base / "raw_data",
        "pre": base / "preprocessed_data",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _find_luna_annotations() -> Path | None:
    for base in LUNA16_BASES:
        p = base / "annotations.csv"
        if p.exists():
            return p
    return None


# ─── LUNA16 RAW ───────────────────────────────────────────────────────────────

def luna16_raw_eda(out: dict[str, Path]) -> dict:
    print("[LUNA16] Raw data EDA …")
    stats: dict = {}
    ann_path = _find_luna_annotations()

    if ann_path is None:
        print("  [WARN] annotations.csv not found on G:/ or H:/ — skipping raw LUNA16 plots")
        return stats

    ann = pd.read_csv(ann_path)
    stats["nodules_annotated"] = len(ann)
    stats["series_with_nodules"] = ann["seriesuid"].nunique()
    diameters = ann["diameter_mm"].dropna().values

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(diameters, bins=40, kde=True, color=PALETTE["accent3"],
                 edgecolor=PALETTE["bg"], linewidth=0.5, alpha=0.75, line_kws={"lw": 2}, ax=ax)
    ax.axvline(3.0, color=PALETTE["accent5"], ls="--", lw=1.5, label="3 mm clinical threshold")
    ax.axvline(np.mean(diameters), color=PALETTE["green"], ls="-", lw=1.8,
               label=f"Mean {np.mean(diameters):.1f} mm")
    ax.set_title("LUNA16 Raw — Nodule Diameter Distribution (annotations.csv)")
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Count")
    ax.legend(framealpha=0.8)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out["raw"] / "nodule_diameter_distribution.png")
    plt.close(fig)

    # Scans per subset (count .mhd files)
    subset_counts = []
    for base in LUNA16_BASES:
        if not base.exists():
            continue
        for subset_dir in sorted(base.glob("subset*")):
            if subset_dir.is_dir():
                n = len(list(subset_dir.glob("*.mhd")))
                if n:
                    subset_counts.append((subset_dir.name, n))

    if subset_counts:
        names, counts = zip(*subset_counts)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(names, counts, color=PALETTE["gradient"][: len(names)],
                      edgecolor=PALETTE["bg"], linewidth=1.2)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                    str(val), ha="center", fontweight="bold", fontsize=9)
        ax.set_title("LUNA16 Raw — CT Scans per Subset (.mhd count)")
        ax.set_ylabel("Number of scans")
        ax.set_xlabel("Subset")
        ax.grid(axis="y")
        fig.tight_layout()
        fig.savefig(out["raw"] / "scans_per_subset.png")
        plt.close(fig)
        stats["total_scans_mhd"] = int(sum(counts))

    pd.DataFrame([stats]).to_csv(out["raw"] / "data_statistics.csv", index=False)
    print(f"  [OK] {out['raw']}")
    return stats


# ─── LUNA16 PREPROCESSED ─────────────────────────────────────────────────────

def luna16_preprocessed_eda(out: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    print("[LUNA16] Preprocessed data EDA …")
    if not YOLO_META.exists():
        print(f"  [ERROR] Missing {YOLO_META}")
        return pd.DataFrame(), {}

    df = pd.read_csv(YOLO_META)
    stats = {
        "total_slices": len(df),
        "total_scans": df["series_uid"].nunique(),
        "positive_slices": int((df["is_positive"] == True).sum()),
        "negative_slices": int((df["is_positive"] == False).sum()),
    }

    # Split donut
    ordered = ["train", "val", "test"]
    split_counts = df["split"].value_counts()
    vals = [split_counts.get(s, 0) for s in ordered]
    colours = [PALETTE["accent1"], PALETTE["accent5"], PALETTE["accent2"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(vals, labels=None, autopct="%1.1f%%", colors=colours, startangle=140,
           pctdistance=0.75, wedgeprops=dict(width=0.48, edgecolor=PALETTE["bg"], linewidth=2.5))
    for t in ax.texts:
        if hasattr(t, "set_fontweight"):
            t.set_fontweight("bold")
    ax.legend([f"Train ({vals[0]})", f"Val ({vals[1]})", f"Test ({vals[2]})"],
              loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    ax.set_title("Preprocessed — Scan-Level 2.5D Slice Split", pad=14)
    ax.text(0, 0, f"{sum(vals)}\nslices", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["muted"])
    fig.tight_layout()
    fig.savefig(out["pre"] / "dataset_split.png")
    plt.close(fig)

    # Class imbalance
    imb = df["is_positive"].value_counts()
    neg, pos = imb.get(False, 0), imb.get(True, 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(["Positive\n(Nodule)", "Negative\n(Background)"], [pos, neg],
                   color=[PALETTE["accent2"], PALETTE["accent1"]], height=0.55,
                   edgecolor=PALETTE["bg"], linewidth=1.5)
    for bar, val in zip(bars, [pos, neg]):
        ax.text(bar.get_width() + max(pos, neg) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontweight="bold", fontsize=12)
    ratio = neg / pos if pos else 0
    ax.text(0.98, 0.95, f"Ratio {ratio:.1f} : 1", transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=PALETTE["accent5"],
            bbox=dict(boxstyle="round,pad=0.4", fc=PALETTE["panel"], ec=PALETTE["accent5"], lw=1.2))
    ax.set_title("Preprocessed — Positive vs Negative Slices")
    ax.set_xlabel("Slice count")
    ax.invert_yaxis()
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out["pre"] / "detection_class_imbalance.png")
    plt.close(fig)

    # Nodule size (positive only)
    pos_df = df[df["is_positive"] == True]
    diameters = pos_df["diameter_mm"].replace(0, np.nan).dropna().values
    mean_d = float(np.mean(diameters)) if len(diameters) else 0.0
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(diameters, bins=40, kde=True, color=PALETTE["accent4"],
                 edgecolor=PALETTE["bg"], linewidth=0.5, alpha=0.7, line_kws={"lw": 2}, ax=ax)
    ax.axvline(3.0, color=PALETTE["accent5"], ls="--", lw=1.5, label="3 mm")
    ax.axvline(mean_d, color=PALETTE["green"], ls="-", lw=1.8, label=f"Mean {mean_d:.1f} mm")
    ax.set_title("Preprocessed — Nodule Diameter (YOLO slices)")
    ax.set_xlabel("Diameter (mm)")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out["pre"] / "nodule_size_distribution.png")
    plt.close(fig)

    # Lung coverage
    plot_df = df[["is_positive", "lung_coverage"]].copy()
    plot_df["Class"] = plot_df["is_positive"].map({True: "Positive", False: "Negative"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=plot_df, x="Class", y="lung_coverage",
                   palette=[PALETTE["accent1"], PALETTE["accent2"]],
                   inner=None, linewidth=0, alpha=0.35, ax=ax, order=["Negative", "Positive"])
    sns.boxplot(data=plot_df, x="Class", y="lung_coverage",
                palette=[PALETTE["accent1"], PALETTE["accent2"]], width=0.18, ax=ax,
                order=["Negative", "Positive"],
                boxprops=dict(edgecolor="white", alpha=0.9),
                medianprops=dict(color=PALETTE["accent5"], linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_title("Preprocessed — Lung Tissue Coverage")
    ax.set_ylabel("Lung coverage fraction")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out["pre"] / "lung_coverage_analysis.png")
    plt.close(fig)

    stats["mean_diameter_mm"] = mean_d
    df.describe().to_csv(out["pre"] / "data_statistics.csv")
    _write_luna16_report(out["pre"], df, stats, split_counts.to_dict())
    print(f"  [OK] {out['pre']}")
    return df, stats


def _write_luna16_report(out_pre: Path, df: pd.DataFrame, stats: dict, splits: dict):
    neg = int((df["is_positive"] == False).sum())
    pos = int((df["is_positive"] == True).sum())
    ratio = neg / pos if pos else 0
    md = f"""# LUNA16 Dataset Report — Detection (YOLO)

**Phase:** CRISP-ML(Q) — Dataset Development  
**Task:** Pulmonary nodule detection on 2.5D slices

## Raw Data (`output/luna16/raw_data/`)

- Source: LUNA16 challenge subsets (`.mhd` volumes + `annotations.csv`)
- See `nodule_diameter_distribution.png`, `scans_per_subset.png`, `data_statistics.csv`

## Preprocessed Data (`output/luna16/preprocessed_data/`)

| Statistic | Value |
|-----------|-------|
| CT scans (series) | {stats.get('total_scans', 'N/A')} |
| 2.5D slices | {stats.get('total_slices', len(df))} |
| Positive slices | {pos} |
| Negative slices | {neg} |
| Imbalance ratio | {ratio:.1f} : 1 |
| Mean nodule diameter | {stats.get('mean_diameter_mm', 0):.2f} mm |

### Train / Val / Test
![Split](./dataset_split.png)

- Train: {splits.get('train', 0)} | Val: {splits.get('val', 0)} | Test: {splits.get('test', 0)}

### Class imbalance
![Imbalance](./detection_class_imbalance.png)

### Nodule size & lung coverage
![Size](./nodule_size_distribution.png)
![Coverage](./lung_coverage_analysis.png)

## Data Quality Notes

- Scan-level split (subsets 0–7 train, 8 val, 9 test) prevents patient leakage
- HU window [-1000, 400] and lung masking applied during preprocessing
- Corrupted `.mhd` files skipped per audit log in `data/luna16_yolo_dataset_v4/audit/`
"""
    (out_pre / "dataset_report.md").write_text(md, encoding="utf-8")


# ─── LIDC RAW ─────────────────────────────────────────────────────────────────

def lidc_raw_eda(out: dict[str, Path]) -> dict:
    print("[LIDC-IDRI] Raw data EDA …")
    stats: dict = {"patients_on_disk": 0}

    lidc_raw = _resolve_lidc_raw_dir()
    if lidc_raw is None:
        print(f"  [WARN] LIDC raw dir not found in {LIDC_RAW_CANDIDATES}")
        return stats

    patients = [n for n in os.listdir(lidc_raw)
                if n.startswith("LIDC-IDRI-") and (lidc_raw / n).is_dir()]
    stats["patients_on_disk"] = len(patients)

    thicknesses = []
    pixel_spacings = []

    try:
        import pylidc as pl
        for np_fix, alias in [(int, "int"), (float, "float"), (bool, "bool")]:
            if not hasattr(np, alias):
                setattr(np, alias, np_fix)

        for pid in patients:
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            if scan is None:
                continue
            if scan.slice_thickness:
                thicknesses.append(float(scan.slice_thickness))
            if scan.pixel_spacing is not None:
                ps = scan.pixel_spacing
                pixel_spacings.append(float(ps[0] if hasattr(ps, "__getitem__") else ps))
    except ImportError:
        print("  [WARN] pylidc not installed — using patient folder count only")

    fig, axes = plt.subplots(1, 2 if thicknesses else 1, figsize=(12 if thicknesses else 6, 4.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax0 = axes[0]
    ax0.bar(["Downloaded patients"], [stats["patients_on_disk"]],
            color=PALETTE["accent1"], width=0.4, edgecolor=PALETTE["bg"])
    ax0.set_title("LIDC-IDRI Raw — Patients Available on Disk")
    ax0.set_ylabel("Count")
    ax0.grid(axis="y")

    if thicknesses and len(axes) > 1:
        sns.histplot(thicknesses, bins=20, kde=True, color=PALETTE["accent4"],
                     ax=axes[1], edgecolor=PALETTE["bg"])
        axes[1].set_title("Raw — Slice Thickness Distribution (mm)")
        axes[1].set_xlabel("Slice thickness (mm)")
        axes[1].grid(axis="y")
        stats["mean_slice_thickness_mm"] = float(np.mean(thicknesses))

    if pixel_spacings:
        stats["mean_pixel_spacing_mm"] = float(np.mean(pixel_spacings))

    fig.tight_layout()
    fig.savefig(out["raw"] / "raw_cohort_overview.png")
    plt.close(fig)

    pd.DataFrame([stats]).to_csv(out["raw"] / "data_statistics.csv", index=False)
    print(f"  [OK] {out['raw']}")
    return stats


# ─── LIDC PREPROCESSED ───────────────────────────────────────────────────────

def lidc_preprocessed_eda(out: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    print("[LIDC-IDRI] Preprocessed data EDA …")
    if not CLASSIFICATION_META.exists():
        print(f"  [ERROR] Missing {CLASSIFICATION_META}")
        return pd.DataFrame(), {}

    df = pd.read_csv(CLASSIFICATION_META)

    # Class imbalance donuts
    tasks = [
        ("malignancy_label", "Malignancy", ["Benign", "Malignant"]),
        ("spiculation_label", "Spiculation", ["Absent", "Present"]),
        ("margin_label", "Margin", ["Smooth", "Irregular"]),
    ]
    colors_pair = [PALETTE["accent1"], PALETTE["accent2"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Preprocessed — Multi-Task Label Distribution", fontsize=15,
                 fontweight="bold", color=PALETTE["text"], y=1.02)
    imb = {}
    for ax, (col, title, labels) in zip(axes, tasks):
        counts = df[col].value_counts().sort_index()
        ax.pie(counts, autopct="%1.1f%%", colors=colors_pair, startangle=90,
               pctdistance=0.78, wedgeprops=dict(width=0.44, edgecolor=PALETTE["bg"], linewidth=2))
        for t in ax.texts:
            t.set_color("white")
            t.set_fontweight("bold")
        ax.set_title(title, pad=10)
        ax.legend(labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.08))
        ax.text(0, 0, f"n={counts.sum()}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=PALETTE["muted"])
        imb[f"{col}_pos"] = int(counts.get(1, 0))
        imb[f"{col}_neg"] = int(counts.get(0, 0))
    fig.tight_layout()
    fig.savefig(out["pre"] / "classification_class_imbalance.png")
    plt.close(fig)

    # Clinical attributes
    attrs = ["malignancy_score", "spiculation_score", "margin_score", "lobulation_score", "texture"]
    nice = ["Malignancy", "Spiculation", "Margin", "Lobulation", "Texture"]
    melted = df[attrs].melt(var_name="Attribute", value_name="Score")
    melted["Attribute"] = melted["Attribute"].map(dict(zip(attrs, nice)))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colours = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
               PALETTE["accent4"], PALETTE["accent5"]]
    sns.violinplot(data=melted, x="Attribute", y="Score", palette=colours,
                   inner=None, linewidth=0, alpha=0.35, ax=ax)
    sns.boxplot(data=melted, x="Attribute", y="Score", palette=colours, width=0.25, ax=ax,
                medianprops=dict(color=PALETTE["accent5"], linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.set_title("Preprocessed — Radiologist Consensus Scores (1–5)")
    ax.set_ylim(0.5, 5.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out["pre"] / "clinical_attributes_distribution.png")
    plt.close(fig)

    # Nodule size
    diameters = df["diameter_mm"].values
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(diameters, bins=35, kde=True, color=PALETTE["accent1"],
                 edgecolor=PALETTE["bg"], alpha=0.75, line_kws={"lw": 2}, ax=ax)
    ax.axvline(3.0, color=PALETTE["accent5"], ls="--", lw=1.5, label="3 mm")
    ax.axvline(10.0, color=PALETTE["accent2"], ls="--", lw=1.5, label="10 mm")
    ax.axvline(np.mean(diameters), color=PALETTE["green"], ls="-", lw=1.8,
               label=f"Mean {np.mean(diameters):.1f} mm")
    ax.set_title("Preprocessed — Nodule Diameter Distribution")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out["pre"] / "nodule_size_distribution.png")
    plt.close(fig)

    sizes = {
        "mean_dia": float(np.mean(diameters)),
        "sub_3mm": float(np.sum(diameters < 3) / len(diameters) * 100),
        "3_to_10mm": float(np.sum((diameters >= 3) & (diameters <= 10)) / len(diameters) * 100),
        "over_10mm": float(np.sum(diameters > 10) / len(diameters) * 100),
    }

    # Train/val/test split
    dirs = ["train", "val", "test"]
    counts = []
    for d in dirs:
        p = CLASSIFICATION_DIR / d
        counts.append(len(list(p.glob("*.npz"))) if p.exists() else 0)
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = [PALETTE["accent1"], PALETTE["accent5"], PALETTE["accent2"]]
    ax.pie(counts, autopct="%1.1f%%", colors=colours, startangle=140,
           pctdistance=0.75, wedgeprops=dict(width=0.48, edgecolor=PALETTE["bg"], linewidth=2.5))
    ax.legend([f"Train ({counts[0]})", f"Val ({counts[1]})", f"Test ({counts[2]})"],
              loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    ax.set_title("Preprocessed — Patient-Level 3D Patch Split")
    ax.text(0, 0, f"{sum(counts)}\npatches", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["muted"])
    fig.tight_layout()
    fig.savefig(out["pre"] / "dataset_split_distribution.png")
    plt.close(fig)

    df.describe().to_csv(out["pre"] / "data_statistics.csv")
    _write_lidc_report(out["pre"], df, imb, sizes, counts)
    stats = {"patches": len(df), "patients": df["patient_id"].nunique(), **sizes}
    print(f"  [OK] {out['pre']}")
    return df, stats


def _write_lidc_report(out_pre: Path, df, imb, sizes, splits):
    md = f"""# LIDC-IDRI Dataset Report — Classification

**Phase:** CRISP-ML(Q) — Dataset Development  
**Task:** Benign vs malignant (+ spiculation, margin)

## Raw Data (`output/LIDC-IDRI/raw_data/`)

- Source: LIDC-IDRI DICOM via pylidc (downloaded patient subset)
- See `raw_cohort_overview.png`, `data_statistics.csv`

## Preprocessed Data (`output/LIDC-IDRI/preprocessed_data/`)

| Statistic | Value |
|-----------|-------|
| Patients | {df['patient_id'].nunique()} |
| Patches | {len(df)} |
| Mean diameter | {sizes['mean_dia']:.2f} mm |

### Labels
![Imbalance](./classification_class_imbalance.png)

- Malignancy: {imb['malignancy_label_neg']} benign / {imb['malignancy_label_pos']} malignant
- Spiculation: {imb['spiculation_label_neg']} absent / {imb['spiculation_label_pos']} present
- Margin: {imb['margin_label_neg']} smooth / {imb['margin_label_pos']} irregular

### Clinical scores & size
![Attributes](./clinical_attributes_distribution.png)
![Size](./nodule_size_distribution.png)

- Micro (<3mm): {sizes['sub_3mm']:.1f}% | 3–10mm: {sizes['3_to_10mm']:.1f}% | >10mm: {sizes['over_10mm']:.1f}%

### Split
![Split](./dataset_split_distribution.png)

Train / Val / Test patches: {splits[0]} / {splits[1]} / {splits[2]}

## Data Quality

- Patient-level split prevents leakage
- Duplicate annotations merged (5 mm spatial matching)
- Indeterminate malignancy score (3) excluded during preprocessing
"""
    (out_pre / "dataset_report.md").write_text(md, encoding="utf-8")


def write_root_dataset_report():
    root_md = DATASET_DEV_DIR / "dataset_report.md"
    root_md.write_text("""# Dataset Development — Overview

This folder documents **what** the data is and how it was analyzed (EDA).

| Dataset | Task | Raw outputs | Preprocessed outputs |
|---------|------|-------------|----------------------|
| LUNA16 | Nodule detection | `output/luna16/raw_data/` | `output/luna16/preprocessed_data/` |
| LIDC-IDRI | Malignancy classification | `output/LIDC-IDRI/raw_data/` | `output/LIDC-IDRI/preprocessed_data/` |

## Run EDA

```bash
cd 01_Business_and_Data_Understanding/dataset_development
python dataset_eda.py
```

## Dependencies

See `requirements.txt`.
""", encoding="utf-8")


def main():
    write_root_dataset_report()
    luna_out = out_paths("luna16")
    lidc_out = out_paths("LIDC-IDRI")

    luna16_raw_eda(luna_out)
    luna16_preprocessed_eda(luna_out)
    lidc_raw_eda(lidc_out)
    lidc_preprocessed_eda(lidc_out)

    print(f"[DONE] All outputs under {DATASET_DEV_DIR / 'output'}")


if __name__ == "__main__":
    main()
