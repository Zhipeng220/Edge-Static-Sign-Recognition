# Dataset Card

## Overview

This repository uses MediaPipe hand landmarks extracted from public static sign-language or hand-posture image datasets. Raw images are not redistributed in this repository unless explicitly permitted by the original dataset license.

Configured datasets:

```text
asl_dataset
indian_sign_language
nus_hand_posture
asl_large_dataset
```

All datasets are converted to the same single-hand landmark protocol:

```text
21 MediaPipe hand landmarks x 3 coordinates = 63 features
```

Processed landmark arrays are generated locally by:

```bash
python src/extract_landmarks.py
```

## Directory Layout

Raw downloaded datasets should be staged under:

```text
data/raw_datasets/
  asl_dataset/
  indian_sign_language/
  nus_hand_posture/Color/
  asl_large_dataset/
```

Generated landmark arrays are written to:

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

Exact split CSV files, when exported, are written to:

```text
data/<dataset_name>/split_files/
  seed42_train.csv
  seed42_val.csv
  seed42_test.csv
  ...
  split_summary.csv
```

Generate them with:

```bash
python src/export_split_files.py
```

## Dataset Entries

### `asl_dataset`

Type: folder-structured static sign image dataset.

Expected raw layout:

```text
data/raw_datasets/asl_dataset/<class_name>/<image_file>
```

Processed output:

```text
data/asl_dataset/processed_landmarks/
```

License and redistribution: must be verified against the original dataset source before public redistribution. Raw images are not included in this repository.

### `indian_sign_language`

Type: folder-structured static sign image dataset.

Expected raw layout:

```text
data/raw_datasets/indian_sign_language/<class_name>/<image_file>
```

Processed output:

```text
data/indian_sign_language/processed_landmarks/
```

License and redistribution: must be verified against the original dataset source before public redistribution. Raw images are not included in this repository.

### `nus_hand_posture`

Type: flat image dataset with labels parsed from filenames.

Expected raw layout:

```text
data/raw_datasets/nus_hand_posture/Color/<image_file>
```

The extractor parses labels from filenames such as `g1.jpg`, `_g1.jpg`, and `g1 (1).jpg`.

Processed output:

```text
data/nus_hand_posture/processed_landmarks/
```

License and redistribution: must be verified against the original dataset source before public redistribution. Raw images are not included in this repository.

### `asl_large_dataset`

Type: folder-structured ASL alphabet image dataset.

Expected raw layout:

```text
data/raw_datasets/asl_large_dataset/<class_name>/<image_file>
```

Processed output:

```text
data/asl_large_dataset/processed_landmarks/
```

License and redistribution: must be verified against the original dataset source before public redistribution. Raw images are not included in this repository.

## Preprocessing Protocol

For each image:

1. Read the image with OpenCV.
2. Run MediaPipe Hands in static-image mode.
3. For the single-hand protocol, select the detected hand with the largest 2D bounding-box area.
4. Export landmark coordinates as `[x, y, z]` for 21 landmarks.
5. Save a 63-dimensional row to `X_data.npy`.
6. Save numeric labels to `y_labels.npy`.
7. Save the class-name mapping to `class_mapping.npy`.
8. Record success and failure metadata in CSV files.

Training then applies:

- conversion of 126-dimensional two-hand exports to 63-dimensional single-hand features when needed;
- all-zero failed-row filtering when enabled;
- wrist-centering and palm-scale normalization;
- label canonicalization;
- small-class filtering using `min_samples_per_class`;
- stratified 70/10/20 train/validation/test split;
- train-only `StandardScaler`;
- train-only SMOTE when enabled.

## Split Protocol

The split protocol is deterministic for a given dataset and seed:

```text
train = 70%
validation = 10%
test = 20%
stratify = labels
```

Configured full protocol seeds:

```text
42, 2024, 2025, 3407, 1234
```

The checked-in lightweight config may use seed `42` only. For the full protocol, set `seeds` in `configs/experiment_config.json` to the full five-seed list.

## Final Holdout Policy

The independent image-level final holdout is separated before MediaPipe landmark extraction. Final holdout images are never included in `X_data.npy`, and they are used only after training for final image-level inference.

For `asl_large_dataset`, an external test folder can be staged directly under:

```text
data/asl_large_dataset/final_holdout/
```

## Known Gaps Requiring Manual Review

- Original source URLs should be added for each dataset before public release.
- Original dataset licenses and redistribution terms must be reviewed.
- Class counts and sample counts should be regenerated from the actual staged raw data or extraction summaries.
- Raw images should not be committed unless the original license explicitly permits redistribution.
