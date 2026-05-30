# Reproducibility

## Environment

Use either:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate edge-static-sign-recognition
```

The Conda environment pins Python to 3.11. Core dependencies include PyTorch, NumPy, pandas, scikit-learn, imbalanced-learn, OpenCV, MediaPipe, matplotlib, seaborn, and joblib.

## Data Locations

Raw images should be staged only under:

```text
data/raw_datasets/
```

MediaPipe extraction writes processed landmark arrays to:

```text
data/<dataset_name>/processed_landmarks/
```

Training reads these processed landmark folders through:

```text
configs/experiment_config.json
```

## Reproduction Commands

Run from the repository root:

```bash
python src/extract_landmarks.py
python src/export_split_files.py
python src/train_single_hand.py
python src/predict.py --dataset asl_large_dataset --input data/asl_large_dataset/final_holdout --no-tsne
```

The split exporter is optional for training but recommended for reviewer audit because it materializes the deterministic sample-level split CSV files.

## Random Seeds

Full repeated-experiment seed list:

```text
42, 2024, 2025, 3407, 1234
```

The checked-in `configs/experiment_config.json` may use:

```json
"seeds": [42]
```

for a lightweight reproduction run. For the full five-seed protocol, set:

```json
"seeds": [42, 2024, 2025, 3407, 1234]
```

The same seed is used for the stratified split and for experiment-level reproducibility controls.

## Landmark Input Protocol

The main protocol uses single-hand MediaPipe landmarks:

```text
21 landmarks x 3 coordinates = 63 features
model tensor shape = [batch, 21, 3]
```

Supported preprocessing behavior:

- `[N, 63]` inputs are treated as single-hand features;
- `[N, 126]` inputs are converted to the single-hand protocol by selecting the stronger non-zero hand;
- wrist-centering is applied;
- palm-scale normalization uses the wrist-to-middle-MCP distance;
- all-zero failed rows can be filtered before training.

## Split Protocol

The landmark-level experiments use a fixed stratified split:

```text
train = 70%
validation = 10%
test = 20%
stratify = labels
```

The split is generated before `StandardScaler` fitting and before SMOTE.

The split exporter writes:

```text
data/<dataset_name>/split_files/seed<seed>_train.csv
data/<dataset_name>/split_files/seed<seed>_val.csv
data/<dataset_name>/split_files/seed<seed>_test.csv
data/<dataset_name>/split_files/split_summary.csv
```

Each split CSV records:

```text
sample_id
label_id
label_name
split
seed
dataset
```

`sample_id` is the post-filter landmark-row identifier after the same filtering rules used by training.

## Anti-Leakage Controls

The repository follows these rules:

- split before scaler fitting;
- `StandardScaler` fitted only on training features;
- validation and test transformed only with the training-fitted scaler;
- SMOTE applied only on training features;
- validation used for early stopping/model selection;
- internal test used only for final landmark-level evaluation;
- final image-level holdout separated before MediaPipe extraction;
- final holdout images never included in `X_data.npy`.

## Evaluation Artifacts

Training writes result files to:

```text
multi_dataset_results_fixed_labels/
```

Selected public summary tables are copied to:

```text
results/tables/
```

Standalone prediction CSV evaluation:

```bash
python src/evaluate.py --predictions path/to/predictions.csv --output-dir results/evaluation
```

The evaluator exports:

```text
metrics_summary.csv
classification_report.txt
confusion_matrix.csv
confusion_matrix.png
```

## Operations Not Required for Lightweight Verification

These operations are intentionally not run during lightweight repository checks:

- full model retraining;
- full MediaPipe extraction on raw datasets;
- ONNX runtime inference;
- TensorRT build;
- Jetson hardware benchmarking.

Run them only when the corresponding data, hardware, and dependencies are available.
