"""
evidently_report.py
===================
Generates an interactive HTML monitoring report using Evidently AI v0.7+.

Covers:
  - Data Drift       : distribution shift in yolo_confidence, mal_probability, diameter_mm
  - Data Summary     : column stats, value ranges, missing values
  - Classification   : quality metrics on user-saved (labeled) predictions

The log is split into two halves:
  reference = first half  (older predictions, closer to training time)
  current   = second half (most recent predictions)

Run:
    python evidently_report.py

Output:
    data/monitoring/evidently_report.html  (open in any browser)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent

LOG_PATH    = PROJECT_ROOT / "data" / "monitoring" / "predictions_log.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "monitoring" / "evidently_report.html"

NUMERICAL_COLS   = ["yolo_confidence", "mal_probability", "diameter_mm"]
CATEGORICAL_COLS = ["mal_label", "model_version"]


def _load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log not found: {LOG_PATH}")

    df = pd.read_csv(LOG_PATH)
    df = df[df["nodule_index"] >= 0].copy()
    df = df.dropna(subset=NUMERICAL_COLS)

    if len(df) < 10:
        raise ValueError(
            f"Not enough data ({len(df)} rows). Need at least 10 predictions."
        )

    # Binary columns for classification metrics
    df["target"]     = (df["mal_label"] == "Malignant").astype(int)
    df["prediction"] = (df["mal_probability"] >= 0.5).astype(int)

    split     = max(len(df) // 2, 5)
    reference = df.iloc[:split].reset_index(drop=True)
    current   = df.iloc[split:].reset_index(drop=True)

    print(f"[Evidently] Reference: {len(reference)} rows | Current: {len(current)} rows")
    return reference, current


def generate_evidently_report() -> None:
    try:
        import evidently
        from evidently import Report, Dataset, DataDefinition, ColumnType, BinaryClassification
        from evidently.presets import DataDriftPreset, DataSummaryPreset, ClassificationPreset
    except ImportError as e:
        print(f"[Evidently] Import error: {e}")
        print("[Evidently] Run: pip install evidently>=0.7.0")
        return

    reference, current = _load_and_prepare()

    # Define column schema
    data_def = DataDefinition(
        numerical_columns=NUMERICAL_COLS,
        categorical_columns=CATEGORICAL_COLS,
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas="mal_probability",
                pos_label=1,
            )
        ],
    )

    ref_dataset = Dataset.from_pandas(reference, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current,   data_definition=data_def)

    report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    result = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(OUTPUT_PATH))

    print(f"[Evidently] ✅ Report saved to: {OUTPUT_PATH}")
    print(f"[Evidently] Open in browser: file:///{OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    generate_evidently_report()
