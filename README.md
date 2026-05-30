# Edge-Static-Sign-Recognition

This repository contains training, inference, evaluation, ONNX export, and optional edge-deployment notes for static sign recognition using MediaPipe hand landmarks.

Large artifacts such as processed landmarks, trained weights, and optional deployment outputs may be distributed through Git LFS, GitHub release assets, or Zenodo.

## Overview

Edge-Static-Sign-Recognition is a lightweight static sign recognition pipeline designed for reproducible landmark-based experiments. The project uses MediaPipe hand landmarks as input features, trains landmark-based classifiers, exports model artifacts, and provides scripts for image-level inference and t-SNE visualization.

The repository is organized for reproducible experiments across multiple static sign datasets:

- `asl_dataset`
- `indian_sign_language`
- `nus_hand_posture`
- `asl_large_dataset`

The training protocol uses a fixed stratified 70/10/20 train/validation/test split. The split is created before fitting `StandardScaler` and before applying SMOTE, so preprocessing and balancing are fitted only on the training subset.

## Repository Structure

```text
configs/        Experiment and deployment configuration files
data/           Dataset documentation, raw-data staging folders, and processed-landmark layout
src/            Landmark extraction, training, inference, export, and original utility code
models/         Selected model artifacts and model documentation
results/        Selected summary tables, figures, and manifests
deployment/     Optional deployment notes
sh/             Shell wrappers for training, evaluation, export, and packaging
scripts/        Windows batch wrappers
demo/           Demo runner plus sample image staging directory
tests/          Lightweight pytest smoke and leakage tests
```

The original utility package is preserved under:

```text
src/utils_original/gesture_experiment/
```

It is preserved as the experiment implementation source, while `src/_path_setup.py` lets the top-level scripts run directly without manually setting `PYTHONPATH`.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Or use the Conda environment file:

```bash
conda env create -f environment.yml
conda activate edge-static-sign-recognition
```

Main dependencies include:

- PyTorch
- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- OpenCV
- MediaPipe
- matplotlib
- seaborn

## Data Preparation

Raw datasets are not included in this repository due to dataset licensing and file size constraints. Download each dataset from its original source, then generate MediaPipe landmark files.

Put downloaded raw images under the unified raw-data staging folder:

```text
data/raw_datasets/
  asl_dataset/<class_name>/*.jpg
  indian_sign_language/<class_name>/*.jpg
  nus_hand_posture/Color/*.jpg
  asl_large_dataset/<class_name>/*.jpg
```

The landmark extraction entry is:

```bash
python src/extract_landmarks.py
```

Expected processed landmark files:

```text
X_data.npy
y_labels.npy
class_mapping.npy
```

Each dataset folder follows this layout:

```text
data/<dataset_name>/
  processed_landmarks/
  final_holdout/
  split_files/
```

Important anti-leakage rule:

For landmark-level experiments, landmarks are extracted from the training image pool and then split into train/validation/test subsets. For the independent image-level final holdout test, one image per class is removed before landmark extraction and is never included in `X_data.npy`.

See [data/DATA_PROVENANCE.md](data/DATA_PROVENANCE.md) for details.

## Full Reproduction Workflow

Run all commands from the repository root.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Stage raw image datasets

Put downloaded raw images under `data/raw_datasets/`:

```text
data/raw_datasets/
  asl_dataset/
    A/
      image.jpg
    B/
      image.jpg
  indian_sign_language/
    A/
      image.jpg
    B/
      image.jpg
  nus_hand_posture/
    Color/
      g1.jpg
      g1 (1).jpg
  asl_large_dataset/
    A/
      image.jpg
    B/
      image.jpg
```

Folder-structured datasets use class-name subfolders. NUS Hand Posture uses the flat `Color/` folder and parses labels from filenames.

### 3. Extract MediaPipe landmark coordinates

```bash
python src/extract_landmarks.py
```

The extractor writes processed landmark files to:

```text
data/asl_dataset/processed_landmarks/
data/indian_sign_language/processed_landmarks/
data/nus_hand_posture/processed_landmarks/
data/asl_large_dataset/processed_landmarks/
```

Each processed folder should contain:

```text
X_data.npy
y_labels.npy
class_mapping.npy
metadata.csv
failed_images.csv
per_class_accounting.csv
dataset_accounting_summary.csv
```

`X_data.npy` stores one 63-dimensional vector per image by default: 21 MediaPipe hand landmarks times 3 coordinates.

### 4. Check the training configuration

The training entry reads `configs/experiment_config.json`. The checked-in configuration points to:

```text
data/<dataset_name>/processed_landmarks/
```

The default configuration uses seed 42:

```json
"seeds": [42]
```

For the full five-seed protocol, set:

```json
"seeds": [42, 2024, 2025, 3407, 1234]
```

### 5. Train and evaluate models

```bash
python src/train_single_hand.py
```

The training script performs:

- label canonicalization and class filtering
- fixed stratified 70/10/20 train/validation/test splitting
- train-only `StandardScaler` fitting
- train-only SMOTE balancing when enabled
- baseline model training
- full model training
- ablation experiments
- test-split evaluation
- optional final image-holdout evaluation

Results are written to:

```text
multi_dataset_results_fixed_labels/
```

Important result files include:

```text
multi_dataset_results_fixed_labels/<dataset_name>/baseline_comparison_summary.csv
multi_dataset_results_fixed_labels/<dataset_name>/ablation_summary.csv
multi_dataset_results_fixed_labels/<dataset_name>/final_holdout_summary_seed42.csv
multi_dataset_results_fixed_labels/<dataset_name>/split_accounting_all_variants.csv
```

For multi-dataset runs, combined tables are written as:

```text
multi_dataset_results_fixed_labels/ALL_DATASETS_baseline_comparison_summary.csv
multi_dataset_results_fixed_labels/ALL_DATASETS_ablation_summary.csv
multi_dataset_results_fixed_labels/ALL_DATASETS_final_holdout_summary.csv
```

### 6. Run final prediction tests

Predict from a final-holdout image folder:

```bash
python src/predict.py --dataset asl_large_dataset --input data/asl_large_dataset/final_holdout --no-tsne
```

Predict from a processed landmark folder:

```bash
python src/predict.py --dataset asl_large_dataset --input data/asl_large_dataset/processed_landmarks --no-tsne
```

Prediction outputs are written to:

```text
multi_dataset_results_fixed_labels/<dataset_name>/predict_outputs/
```

Typical files:

```text
predictions_<dataset>_<variant>_seed<seed>.csv
tsne_<dataset>_<variant>_seed<seed>.pdf
tsne_<dataset>_<variant>_seed<seed>_coordinates.csv
```

t-SNE files are created only when t-SNE is enabled and at least three valid samples are available.

Minimal command sequence:

```bash
pip install -r requirements.txt
python src/extract_landmarks.py
python src/train_single_hand.py
python src/predict.py --dataset asl_large_dataset --input data/asl_large_dataset/final_holdout --no-tsne
```

## Training

The main training entry is:

```bash
python src/train_single_hand.py
```

The default experiment configuration is:

```text
configs/experiment_config.json
```

The configured seeds are:

```text
42 in configs/experiment_config.json
```

For the five-seed protocol, copy the seed list from `configs/train_all_seeds.yaml` into `configs/experiment_config.json`.

Training produces per-dataset result folders with:

- classification reports
- prediction CSV files
- confusion matrices
- ablation summaries
- baseline comparison summaries
- model checkpoints
- scaler and label encoder artifacts
- optional ONNX deployment files

## Inference

The inference entry is:

```bash
python src/predict.py --dataset asl_large_dataset --input path/to/input
```

Supported inputs:

- a single image
- a directory of images
- a `.npy` feature file
- a directory containing `X_data.npy`

Example using processed landmark files:

```bash
python src/predict.py --dataset asl_large_dataset --input data/asl_large_dataset/processed_landmarks
```

Example using an image folder:

```bash
python src/predict.py --dataset asl_dataset --input data/asl_dataset/final_holdout
```

The script writes:

```text
predictions_<dataset>_<variant>_seed<seed>.csv
tsne_<dataset>_<variant>_seed<seed>.pdf
tsne_<dataset>_<variant>_seed<seed>_coordinates.csv
```

t-SNE generation is skipped automatically when fewer than three valid samples are available.

## Quickstart: 5-minute prediction smoke run

The repository includes seed-42 model artifacts under `models/`. To verify that the inference entry point is wired correctly, place one or more images in `demo/sample_images/`, then run:

```bash
python demo/run_demo.py --input demo/sample_images --no-tsne
```

For landmark arrays, pass a `.npy` file or a directory containing `X_data.npy`:

```bash
python src/predict.py --dataset asl_large_dataset --input data/asl_large_dataset/processed_landmarks --no-tsne
```

Prediction CSV files are written to `demo/outputs/` for the demo runner, or to the configured result folder when using `src/predict.py` directly.

## Tests and CI

Run the local test suite with:

```bash
python -m pytest -q
```

The GitHub Actions workflow in `.github/workflows/ci.yml` installs the pinned dependency ranges from `requirements.txt` and runs the same pytest command.

## Model Artifacts

The `models/` folder may contain selected artifacts such as:

```text
full_model_seed42.pth
full_model_seed42.onnx
scaler_full_model_seed42.pkl
label_encoder_full_model_seed42.pkl
```

For image-level inference, both the scaler and label encoder are required. The packaged seed-42 model artifacts are only a few MB and can be included directly in the GitHub repository. See the [Model Card](models/model_card.md) for the model protocol and limitations.

## Results

Selected summary tables are stored in:

```text
results/tables/
```

The repository also includes file manifests:

```text
results/manifests/repository_file_manifest.csv
results/manifests/large_artifacts_manifest.csv
```

These manifests record paths, file sizes, extensions, and artifact categories without parsing large result files.

## TensorRT Deployment and Benchmarking on Jetson Xavier NX

Jetson and TensorRT deployment notes are placed under:

```text
deployment/jetson/
```

The GitHub repository keeps Linux shell wrappers under:

```text
sh/
```

For the Jetson-side deployment package, the working directory is:

```text
~/paper_hand_deploy
```

The Jetson-side workflow is command-oriented. TensorRT engine building, worker compilation, parity checks, classifier-only benchmarking, offline stored-image-to-label benchmarking, peak-RAM measurement, and log packaging are exposed through shell scripts. Users do not need to manually invoke auxiliary Python utilities unless they want to inspect or customize the post-processing workflow.

TensorRT engine binaries are target-environment specific. The repository provides ONNX artifacts and shell scripts required to rebuild engines on the Jetson Xavier NX. The `engines/` directory in the Jetson-side package may contain locally generated artifacts for the verified environment.

## Shell Script Index

| Category | Script | Purpose |
|---|---|---|
| TensorRT build | `sh/build_trt_worker.sh` | Build the persistent TensorRT inference worker |
| TensorRT build | `sh/build_all_trt_engines.sh` | Build FP32 and FP16 TensorRT engines |
| Parity validation | `sh/run_parity_test.sh` | Run ONNX / TensorRT parity checks |
| Small-ASL parity | `sh/run_small_asl_trt_parity.sh` | Run retained-sample TensorRT parity tests |
| Full Small-ASL parity | `sh/run_small_asl_full_parity.sh` | Evaluate ONNX, TensorRT FP32, and TensorRT FP16 consistency |
| Classifier benchmark | `sh/run_trtexec_5rounds.sh` | Run five-round classifier-only TensorRT benchmarks |
| RAM benchmark | `sh/run_peak_ram_trtexec.sh` | Measure classifier-only peak RAM and collect telemetry |
| Pipeline benchmark | `sh/run_pipeline_5rounds.sh` | Run the offline stored-image-to-label pipeline benchmark |
| Pipeline wrapper | `sh/run_formal_pipeline_5rounds.sh` | Run formal five-round FP16 and FP32 pipeline benchmarks |
| Log packaging | `sh/pack_required_table_logs.sh` | Package logs used for paper tables |
| Full packaging | `sh/pack_scanned_paper_data.sh` | Create reproducibility archives and checksums |
| Training wrapper | `sh/run_train_all_seeds.sh` | Run the configured training workflow |
| Evaluation wrapper | `sh/run_evaluate_all.sh` | Evaluate a prediction CSV with the standalone evaluator |
| Split export | `sh/run_export_splits.sh` | Export deterministic train/validation/test split CSV files |
| Scaler export | `sh/run_export_scaler_params.sh` | Export scaler mean/scale arrays for deployment |
| ONNX export | `sh/run_export_onnx.sh` | Call the ONNX export entry point |
| Packaging wrapper | `sh/package_release.sh` | Package available reproducibility materials |

### Jetson Commands

Run these commands from `~/paper_hand_deploy` on the Jetson-side deployment package. In this GitHub repository, equivalent wrappers are stored under `sh/`.

| Command | Purpose | Main outputs |
|---|---|---|
| `bash build_all_trt_engines.sh` | Build TensorRT engines on the target Jetson device | `engines/` |
| `bash build_trt_worker.sh` | Build the persistent TensorRT inference worker | worker executable |
| `bash run_parity_test.sh` | Run ONNX / TensorRT parity checks | `parity_results/` |
| `bash run_small_asl_trt_parity.sh` | Run the retained-sample TensorRT parity test | `parity_results/` |
| `bash run_small_asl_full_parity.sh` | Run ONNX, TensorRT FP32, and TensorRT FP16 consistency checks | `parity_results/` |
| `bash run_trtexec_5rounds.sh` | Run five-round classifier-only TensorRT benchmarks | `trtexec_final_results/` |
| `bash run_peak_ram_trtexec.sh` | Measure peak RAM and collect telemetry | `peak_ram_results/` |
| `bash run_pipeline_5rounds.sh` | Run the offline stored-image-to-label pipeline benchmark | pipeline result directory |
| `bash run_formal_pipeline_5rounds.sh` | Run the formal five-round FP16 and FP32 pipeline benchmarks | `pipeline_final_results/` |
| `bash pack_required_table_logs.sh` | Package the logs used for paper tables | timestamped archive |
| `bash pack_scanned_paper_data.sh` | Package reproducibility evidence and checksums | `paper_hand_packages/` |

Auxiliary Python tools used by the TensorRT workflow are invoked automatically by the shell wrappers. They are not listed as required manual Jetson commands.

### Jetson-side File Layout

The Jetson-side deployment package is organized as:

```text
paper_hand_deploy/
|-- build_all_trt_engines.sh
|-- build_trt_worker.sh
|-- run_parity_test.sh
|-- run_small_asl_trt_parity.sh
|-- run_trtexec_5rounds.sh
|-- run_peak_ram_trtexec.sh
|-- run_pipeline_5rounds.sh
|-- run_formal_pipeline_5rounds.sh
|-- pack_required_table_logs.sh
|-- models/
|-- engines/
|-- test_data/
|-- parity_inputs_small_asl/
|-- parity_results/
|-- trtexec_final_results/
|-- pipeline_final_results/
|-- peak_ram_results/
|-- logs/
`-- trt_infer_worker
```

| Path | Purpose |
|---|---|
| `models/` | ONNX models and deployment artifacts |
| `engines/` | TensorRT engines rebuilt on the Xavier NX target device |
| `test_data/` | Held-out images used by the offline pipeline benchmark |
| `parity_inputs_small_asl/` | Prepared parity-test inputs |
| `parity_results/` | Numerical agreement and prediction-consistency results |
| `trtexec_final_results/` | Classifier-only TensorRT benchmark logs and summaries |
| `pipeline_final_results/` | Offline stored-image-to-label pipeline results |
| `peak_ram_results/` | Peak-RAM and telemetry logs |
| `logs/` | Additional raw logs |
| `trt_infer_worker` | Persistent TensorRT inference worker used by the pipeline |

### Verified Jetson Xavier NX Environment

| Item | Verified setting |
|---|---|
| Device | NVIDIA Jetson Xavier NX Developer Kit |
| L4T | R35.4.1 |
| Kernel | Linux 5.10.120-tegra, aarch64 |
| TensorRT | 8.5.2.2 |
| MediaPipe | 0.10.9 |
| OpenCV | 4.8.1 |
| Power mode | `MODE_10W_DESKTOP`, mode 5 |
| Online CPU cores | CPU 0-5 |
| CPU frequency | 1.9072 GHz |
| GPU frequency | 510 MHz |
| EMC frequency | 1600 MHz |
| Telemetry tool | NVIDIA `tegrastats` |
| Telemetry interval | 100 ms |
| Idle system RAM mean | 1294.03 MB |

### Benchmark Scope

Classifier-only TensorRT benchmark:

```bash
bash run_trtexec_5rounds.sh
```

Scope:

```text
TensorRT engine execution only
```

Not included:

```text
image loading
colour conversion
MediaPipe
camera capture
display rendering
speech synthesis
STM32 communication
actuator response
```

Offline stored-image-to-label pipeline benchmark:

```bash
bash run_formal_pipeline_5rounds.sh
```

Scope:

```text
stored image loading
-> colour conversion
-> MediaPipe landmark extraction
-> normalization/scaler
-> persistent TensorRT worker
-> prediction decoding
```

Formal configuration:

```text
Images per round: 1789 held-out images
Covered classes: 26 letter classes, A-Z
Excluded classes: space and nothing
FP16 repetitions: 5 rounds
FP32 repetitions: 5 rounds
```

Not included:

```text
camera capture
display rendering
speech synthesis
STM32 communication
actuator response
```

This is an offline stored-image benchmark, not a live-camera FPS benchmark.

The exact warm-up value used for the formal benchmark remains to be verified from the archived formal commands. The earlier 30-image smoke-test setting is not reported as a formal benchmark parameter unless it matches the archived commands.

## Demo Asset

A short GIF demonstration will be added to the repository README. No GIF file is currently included in this repository snapshot.

Recommended repository reproduction order:

```text
bash sh/run_export_splits.sh
        ->
bash sh/run_train_all_seeds.sh
        ->
bash sh/run_evaluate_all.sh
        ->
bash sh/run_export_scaler_params.sh
```

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the seed list, split protocol, preprocessing order, and independent final holdout policy.

## Data and License Notes

Raw datasets are subject to their original licenses and are not redistributed in this repository unless explicitly permitted.

See:

- [LICENSE](LICENSE)
- [LICENSE_DATA.md](LICENSE_DATA.md)
- [data/dataset_card.md](data/dataset_card.md)

## Citation

If you use this repository, please cite it using [CITATION.cff](CITATION.cff). Update the placeholder author, DOI, and repository URL before public release.
