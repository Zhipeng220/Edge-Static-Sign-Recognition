# Data

Raw data are not included in this repository due to dataset licensing and file size constraints.

Use `data/raw_datasets/` as the only raw-image staging area before coordinate extraction:

```text
data/raw_datasets/
  asl_dataset/
  indian_sign_language/
  nus_hand_posture/Color/
  asl_large_dataset/
```

Run MediaPipe coordinate extraction from the repository root:

```bash
python src/extract_landmarks.py
```

The extractor writes reproducible landmark files to each dataset folder:

```text
data/<dataset_name>/processed_landmarks/
  X_data.npy
  y_labels.npy
  class_mapping.npy
  metadata.csv
  failed_images.csv
  per_class_accounting.csv
  dataset_accounting_summary.csv
```

Training reads these `processed_landmarks` folders through `configs/experiment_config.json`.
