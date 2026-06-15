# Dataset Development — Overview

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
