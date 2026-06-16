# LUNA16 Preprocessing v2 — Report

**Date:** 2026-05-20 17:48
**Output:** `C:\Users\amork\Desktop\data\manifest-1600709154662\CRISP-ML(Q)\data\luna16_yolo_dataset_v4`

## Configuration snapshot

| Parameter | Value |
|-----------|-------|
| HU window | lung_default (C=-600.0, W=1500.0) |
| Min lung coverage | 8% |
| Negative ratio | 1.25 |
| Nodule exclusion | 12.0 mm |
| Min bbox contrast | -8.0 |
| Random seed | 42 |

## Processing summary

- Scans processed: 601
- Scans failed: 0
- Positive slices: 3536
- Negative slices: 4254
- v2 total (pre-export): 7790
- Pos/neg ratio: 0.831

See `V1_VS_V2_COMPARISON.md` for legacy comparison and expected metric impact.
See `qa_report.json`, `bbox_qa_overlays.png`, `v2_distributions.png` for QA artifacts.