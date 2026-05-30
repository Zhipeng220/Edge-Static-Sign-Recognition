# Data Provenance

## Source Policy

Raw datasets are obtained from their original public sources by the repository user. This repository does not redistribute raw third-party images by default. Users are responsible for confirming the original dataset licenses, citation requirements, and redistribution terms.

Raw images should be staged only under:

```text
data/raw_datasets/
```

The repository keeps only lightweight documentation, selected processed artifacts, summary tables, and placeholders unless a dataset license explicitly permits redistribution.

## Raw Data Staging

Expected staging layout:

```text
data/raw_datasets/
  asl_dataset/<class_name>/<image_file>
  indian_sign_language/<class_name>/<image_file>
  nus_hand_posture/Color/<image_file>
  asl_large_dataset/<class_name>/<image_file>
```

This layout is consumed by:

```bash
python src/extract_landmarks.py
```

## Landmark Extraction

The extraction script uses MediaPipe Hands in static-image mode. The default public protocol is single-hand:

```text
max_num_hands = 1
feature dimension = 21 landmarks x 3 coordinates = 63
```

For each dataset, the script writes:

```text
data/<dataset_name>/processed_landmarks/X_data.npy
data/<dataset_name>/processed_landmarks/y_labels.npy
data/<dataset_name>/processed_landmarks/class_mapping.npy
data/<dataset_name>/processed_landmarks/metadata.csv
data/<dataset_name>/processed_landmarks/failed_images.csv
data/<dataset_name>/processed_landmarks/per_class_accounting.csv
data/<dataset_name>/processed_landmarks/dataset_accounting_summary.csv
```

`metadata.csv` records per-image extraction status. `failed_images.csv` records unreadable images or MediaPipe detection failures. `per_class_accounting.csv` and `dataset_accounting_summary.csv` provide an audit trail for retained and failed samples.

## Label Handling

Folder-structured datasets use folder names as class labels. The NUS Hand Posture configuration uses a flat folder and parses labels from filenames.

During training, labels are canonicalized before filtering and splitting. Examples include uppercasing single-letter ASL labels and mapping common aliases such as `space`, `blank`, `del`, and `nothing` to canonical names.

## Filtering and Conversion

The training loader applies the following data-cleaning rules:

- convert supported 126-dimensional two-hand exports to the 63-dimensional single-hand protocol;
- filter all-zero landmark rows when enabled;
- filter classes with fewer than `min_samples_per_class` retained samples;
- preserve a label-canonicalization report when label names are changed.

The stable `sample_id` used by split export is the post-filter landmark-row index after these same training-loader rules.

## Split Timing

The landmark-level split is generated after landmark extraction and training-loader filtering. The split is generated before `StandardScaler` fitting and before SMOTE.

Protocol:

```text
train = 70%
validation = 10%
test = 20%
stratified by label
random_state = configured seed
```

Configured full seeds:

```text
42, 2024, 2025, 3407, 1234
```

Export exact split CSV files with:

```bash
python src/export_split_files.py
```

## Anti-Leakage Rules

- `StandardScaler` is fitted only on the training split.
- Validation and test features are transformed with the training-fitted scaler.
- SMOTE is applied only to the training split.
- Validation is used for early stopping and model selection.
- The internal test split is used only for landmark-level final evaluation.
- Final image holdout data is excluded before landmark extraction and is not included in `X_data.npy`.

## Final Holdout

For folder-structured datasets where `extract_final_test=True`, the extractor moves one image per class to:

```text
data/<dataset_name>/final_holdout/
```

This class-level move is idempotent: if a class already has a final-holdout image, it is skipped on later runs.

For `asl_large_dataset`, the current configuration expects a separately staged holdout folder and does not move images from the training folder by default.

## Distribution Notes

Recommended distribution:

- source code and Markdown: GitHub repository;
- small summary CSV files: GitHub repository;
- exact split CSV files: GitHub repository;
- model weights and ONNX files: Git LFS, GitHub release assets, or Zenodo;
- raw third-party images: not redistributed unless license permits;
- large processed landmark arrays: Zenodo if redistribution is permitted.
