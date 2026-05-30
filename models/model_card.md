# Model Card

## Model Details

Model name: `full_model`

Task: static sign recognition from MediaPipe hand-landmark coordinates.

Repository artifact examples:

```text
models/full_model_seed42.pth
models/full_model_seed42.onnx
models/scaler_full_model_seed42.pkl
models/label_encoder_full_model_seed42.pkl
models/checksums.txt
```

The PyTorch checkpoint, ONNX model, scaler, and label encoder are all required for a complete inference workflow. For public release, large model artifacts should be stored with Git LFS, GitHub release assets, or Zenodo.

## Intended Use

This model is intended for reproducible research on lightweight static sign recognition and edge-deployment experiments. It is designed for image or landmark inputs that can be converted into MediaPipe single-hand landmark coordinates.

Appropriate uses:

- reproducing the reported landmark-based training protocol;
- comparing lightweight landmark classifiers under the same split and preprocessing rules;
- testing image-level inference through MediaPipe landmark extraction;
- exporting classifier artifacts for ONNX or Jetson/TensorRT deployment experiments.

Out-of-scope uses:

- continuous sign-language translation;
- sentence-level or temporal gesture recognition;
- clinical, legal, accessibility-critical, or safety-critical decision making;
- identity recognition or biometric identification.

## Input and Output

Input protocol:

```text
single hand: 21 MediaPipe landmarks x 3 coordinates = 63 features
flattened feature shape: [batch, 63]
model tensor shape: [batch, 21, 3]
```

The training and inference code can accept 126-dimensional two-hand exports in some preprocessing paths. When 126-dimensional features are provided, the code converts them to the single-hand 63-dimensional protocol by selecting the stronger non-zero hand representation. The main protocol for this repository is single-hand.

Preprocessing:

- convert image input to MediaPipe hand landmarks;
- choose the largest detected hand when extracting single-hand features;
- wrist-center the hand landmarks;
- normalize by wrist-to-middle-MCP palm scale;
- apply `StandardScaler` fitted only on the training split.

Output:

- predicted class ID;
- predicted class label from the fitted label encoder;
- confidence score for neural-network models when softmax probabilities are available.

## Architecture Summary

The `full_model` is a lightweight attention-based landmark classifier. It embeds each landmark coordinate into a learned representation, optionally adds positional encoding, applies multi-head self-attention, uses a residual connection, and classifies the flattened landmark representation. The full model also includes the anatomical regularization/reconstruction branch during training.

The repository also includes same-protocol baselines:

```text
baseline_mlp
baseline_svm
baseline_pointnet
baseline_gcn
baseline_transformer_lite
full_model_wo_anatomical_loss
full_model
```

These baselines are trained and evaluated under the same split, preprocessing, and seed settings.

## Training Protocol

The training entry point is:

```bash
python src/train_single_hand.py
```

The default runtime configuration is:

```text
configs/experiment_config.json
```

The checked-in config uses seed `42` for a lightweight reproduction run. The full repeated-experiment seed list is:

```text
42, 2024, 2025, 3407, 1234
```

Training protocol:

- landmark-level split: fixed stratified 70/10/20 train/validation/test;
- split generated before `StandardScaler` fitting;
- `StandardScaler` fitted only on training features;
- SMOTE applied only to the training split when enabled;
- validation split used for early stopping/model selection;
- test split used only for final landmark-level evaluation;
- independent image-level final holdout excluded before landmark extraction.

## Evaluation

The training workflow exports per-dataset and combined result tables under:

```text
multi_dataset_results_fixed_labels/
results/tables/
```

Included summary files may include:

```text
ALL_DATASETS_baseline_comparison_summary.csv
ALL_DATASETS_baseline_comparison_all_seeds.csv
ALL_DATASETS_ablation_summary.csv
ALL_DATASETS_ablation_all_seeds.csv
ALL_DATASETS_clean_anatomical_ablation_summary.csv
ALL_DATASETS_final_holdout_summary.csv
ALL_DATASETS_mcnemar.csv
```

The standalone evaluator can be used on prediction CSV files:

```bash
python src/evaluate.py --predictions path/to/predictions.csv --output-dir results/evaluation
```

It exports accuracy, macro-F1, weighted-F1, macro precision, macro recall, classification report, and confusion matrix files.

## Deployment Notes

The repository includes ONNX export and Jetson/TensorRT deployment materials under:

```text
deployment/jetson/
```

The packaged ONNX artifact is classifier-level. End-to-end image latency must include image loading, MediaPipe landmark extraction, preprocessing, classifier inference, and postprocessing. Classifier-only TensorRT benchmarks should not be reported as full pipeline latency.

## Limitations

- The model recognizes static signs only.
- Performance depends on MediaPipe hand detection quality.
- Occlusion, motion blur, unusual hand poses, background clutter, and camera viewpoint shifts can reduce accuracy.
- The model uses landmark geometry rather than full image appearance.
- Raw third-party datasets are not redistributed in this repository.
- Dataset licenses and permitted redistribution terms must be checked against the original sources before public artifact release.

## Ethical Considerations

This model should not be presented as a complete sign-language communication system. Static sign classification is not equivalent to natural sign-language understanding. Users should avoid deploying it in contexts where errors could affect access, safety, or rights without substantial domain validation and user-centered testing.
