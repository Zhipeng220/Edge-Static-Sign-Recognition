import argparse
import os
from pathlib import Path

from _path_setup import add_original_utils_to_path

add_original_utils_to_path()

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from gesture_experiment.common import *
from gesture_experiment.config import *
from gesture_experiment.data import *
from gesture_experiment.experiments import load_state_dict_safely
from gesture_experiment.models import build_model
from gesture_experiment.robustness import extract_label_from_image_path, mediapipe_features_from_image
from gesture_experiment.training import pdf_output_path, save_tsne_plot


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def experiment_config_by_name(name: str) -> ExperimentConfig:
    configs = {}
    for cfg in [*BASELINE_CONFIGS, *ABLATION_CONFIGS]:
        configs[cfg.name] = cfg
    if name not in configs:
        raise ValueError(f"Unknown variant '{name}'. Available: {sorted(configs)}")
    return configs[name]


def resolve_dataset_paths(dataset_name: str, config_path: str) -> PathConfig:
    runtime_cfg = load_runtime_config(config_path)
    for spec in dataset_specs_from_config(runtime_cfg):
        if spec.dataset_name == dataset_name:
            return build_paths_from_spec(spec, result_root=runtime_cfg["result_root"])
    names = [spec.dataset_name for spec in dataset_specs_from_config(runtime_cfg)]
    raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {names}")


def default_artifact_paths(paths: PathConfig, variant: str, seed: int):
    if variant == "baseline_svm":
        return {
            "model": os.path.join(paths.model_dir, f"{variant}_seed{seed}.pkl"),
            "scaler": "",
            "encoder": "",
        }

    return {
        "model": os.path.join(paths.model_dir, f"{variant}_seed{seed}.pth"),
        "scaler": os.path.join(paths.result_dir, f"scaler_{variant}_seed{seed}.pkl"),
        "encoder": os.path.join(paths.result_dir, f"label_encoder_{variant}_seed{seed}.pkl"),
    }


def load_predictor(paths: PathConfig, variant: str, seed: int, model_path: str, scaler_path: str, encoder_path: str):
    cfg = experiment_config_by_name(variant)
    defaults = default_artifact_paths(paths, variant, seed)
    model_path = model_path or defaults["model"]
    scaler_path = scaler_path or defaults["scaler"]
    encoder_path = encoder_path or defaults["encoder"]

    if cfg.model_type == "svm":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing SVM artifact: {model_path}")
        bundle = joblib.load(model_path)
        return {
            "cfg": cfg,
            "model": bundle["model"],
            "scaler": bundle.get("scaler"),
            "label_encoder": bundle["label_encoder"],
            "device": None,
            "model_path": model_path,
        }

    missing = [p for p in [model_path, scaler_path, encoder_path] if not p or not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing trained artifacts. Run training first or pass explicit paths. Missing: "
            + "; ".join(missing)
        )

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    model = build_model(cfg, num_classes=len(label_encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(load_state_dict_safely(model_path, device))
    model = model.to(device).eval()

    return {
        "cfg": cfg,
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "device": device,
        "model_path": model_path,
    }


def image_files_from_path(input_path: str):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [str(path)]
    if path.is_dir():
        files = []
        for root, _, names in os.walk(path):
            for name in names:
                if name.startswith("._"):
                    continue
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    files.append(os.path.join(root, name))
        return sorted(files)
    return []


def load_features_from_images(input_path: str, label_encoder):
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(f"MediaPipe is required for image input: {exc}") from exc

    files = image_files_from_path(input_path)
    if not files:
        raise ValueError(f"No image files found under: {input_path}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )

    rows = []
    features = []
    true_labels = []
    try:
        for image_path in files:
            true_label = extract_label_from_image_path(image_path, label_encoder)
            image = cv2.imread(image_path)
            if image is None:
                rows.append({"source": image_path, "status": "unreadable_image", "true_label": true_label or ""})
                continue

            row_features = mediapipe_features_from_image(image, hands)
            if row_features is None:
                rows.append({"source": image_path, "status": "mediapipe_detection_failed", "true_label": true_label or ""})
                continue

            features.append(row_features.reshape(-1))
            true_labels.append(true_label)
            rows.append({"source": image_path, "status": "ok", "true_label": true_label or ""})
    finally:
        hands.close()

    if not features:
        raise ValueError("No valid MediaPipe hand landmarks were extracted from the image input.")

    return np.asarray(features, dtype=np.float32), rows, true_labels


def load_features_from_npy(input_path: str, label_encoder):
    path = Path(input_path)
    source = str(path)
    y_path = None
    mapping_path = None

    if path.is_dir():
        x_path = path / "X_data.npy"
        y_candidate = path / "y_labels.npy"
        mapping_candidate = path / "class_mapping.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing X_data.npy in directory: {path}")
        source = str(x_path)
        y_path = y_candidate if y_candidate.exists() else None
        mapping_path = mapping_candidate if mapping_candidate.exists() else None
    else:
        x_path = path
        sibling_y = path.with_name("y_labels.npy")
        sibling_mapping = path.with_name("class_mapping.npy")
        y_path = sibling_y if sibling_y.exists() else None
        mapping_path = sibling_mapping if sibling_mapping.exists() else None

    X = np.load(x_path)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    true_labels = [None] * len(X)
    if y_path is not None:
        raw_y = np.load(y_path, allow_pickle=True)
        if mapping_path is not None:
            class_mapping = np.load(mapping_path, allow_pickle=True).item()
            true_labels = canonicalize_labels(labels_to_names(raw_y, class_mapping)).tolist()
        else:
            true_labels = canonicalize_labels(raw_y).tolist()

    rows = [
        {
            "source": source,
            "source_index": idx,
            "status": "ok",
            "true_label": "" if true_labels[idx] is None else str(true_labels[idx]),
        }
        for idx in range(len(X))
    ]
    return X.astype(np.float32), rows, true_labels


def load_input_features(input_path: str, label_encoder):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".npy":
        return load_features_from_npy(input_path, label_encoder)
    if path.is_dir() and (path / "X_data.npy").exists():
        return load_features_from_npy(input_path, label_encoder)
    return load_features_from_images(input_path, label_encoder)


def preprocess_for_model(X_raw, scaler):
    X = normalize_single_hand_landmarks(to_single_hand_features(X_raw))
    if scaler is not None:
        X = scaler.transform(X)
    return X.astype(np.float32)


def predict_features(bundle, X):
    cfg = bundle["cfg"]
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    if cfg.model_type == "svm":
        pred_ids = model.predict(X).astype(int)
        confidences = np.full(len(pred_ids), np.nan)
    else:
        x_tensor = torch.tensor(
            X.reshape(-1, cfg.num_landmarks, cfg.coord_dim),
            dtype=torch.float32,
            device=bundle["device"],
        )
        with torch.no_grad():
            logits, _, _ = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_ids = probs.argmax(axis=1).astype(int)
        confidences = probs[np.arange(len(pred_ids)), pred_ids]

    pred_labels = label_encoder.inverse_transform(pred_ids)
    return pred_ids, pred_labels, confidences


def add_predictions_to_rows(rows, valid_indices, true_labels, pred_ids, pred_labels, confidences, label_encoder):
    valid_pos = 0
    for idx in valid_indices:
        row = rows[idx]
        true_label = true_labels[valid_pos]
        true_id = ""
        is_correct = ""
        if true_label:
            try:
                true_id = int(label_encoder.transform([true_label])[0])
                is_correct = bool(true_id == int(pred_ids[valid_pos]))
            except ValueError:
                true_id = ""
                is_correct = ""

        row.update({
            "true_id": true_id,
            "pred_id": int(pred_ids[valid_pos]),
            "pred_label": str(pred_labels[valid_pos]),
            "confidence": float(confidences[valid_pos]) if not np.isnan(confidences[valid_pos]) else "",
            "is_correct": is_correct,
        })
        valid_pos += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run trained gesture model inference and save prediction CSV plus t-SNE PDF."
    )
    parser.add_argument("--input", required=True, help="Image file/dir, .npy file, or directory containing X_data.npy.")
    parser.add_argument("--dataset", default="asl_large_dataset", help="Dataset name from configs/experiment_config.json.")
    parser.add_argument("--config", default="configs/experiment_config.json", help="Runtime config JSON path.")
    parser.add_argument("--variant", default="full_model", help="Trained variant name, default: full_model.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed used by the saved artifacts.")
    parser.add_argument("--output-dir", default="", help="Output directory. Default: dataset result_dir/predict_outputs.")
    parser.add_argument("--model-path", default="", help="Optional explicit .pth or SVM .pkl model path.")
    parser.add_argument("--scaler-path", default="", help="Optional explicit scaler .pkl path.")
    parser.add_argument("--encoder-path", default="", help="Optional explicit label encoder .pkl path.")
    parser.add_argument("--no-tsne", action="store_true", help="Disable t-SNE figure generation.")
    args = parser.parse_args()

    paths = resolve_dataset_paths(args.dataset, args.config)
    output_dir = args.output_dir or os.path.join(paths.result_dir, "predict_outputs")
    os.makedirs(output_dir, exist_ok=True)

    bundle = load_predictor(
        paths=paths,
        variant=args.variant,
        seed=args.seed,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        encoder_path=args.encoder_path,
    )
    label_encoder = bundle["label_encoder"]

    X_raw, rows, true_labels = load_input_features(args.input, label_encoder)
    valid_indices = [idx for idx, row in enumerate(rows) if row["status"] == "ok"]
    X_model = preprocess_for_model(X_raw, bundle["scaler"])
    pred_ids, pred_labels, confidences = predict_features(bundle, X_model)
    rows = add_predictions_to_rows(
        rows=rows,
        valid_indices=valid_indices,
        true_labels=true_labels,
        pred_ids=pred_ids,
        pred_labels=pred_labels,
        confidences=confidences,
        label_encoder=label_encoder,
    )

    stem = f"predictions_{args.dataset}_{args.variant}_seed{args.seed}"
    pred_path = os.path.join(output_dir, f"{stem}.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions: {pred_path}")

    if not args.no_tsne and len(X_model) >= 3:
        tsne_labels = []
        for true_label, pred_id in zip(true_labels, pred_ids):
            if true_label:
                try:
                    tsne_labels.append(int(label_encoder.transform([true_label])[0]))
                    continue
                except ValueError:
                    pass
            tsne_labels.append(int(pred_id))

        tsne_path = pdf_output_path(os.path.join(output_dir, f"tsne_{args.dataset}_{args.variant}_seed{args.seed}.pdf"))
        plot_path, coords_path = save_tsne_plot(
            X=X_model,
            y=np.asarray(tsne_labels, dtype=int),
            label_encoder=label_encoder,
            out_path=tsne_path,
            title="",
            seed=args.seed,
        )
        print(f"Saved t-SNE plot: {plot_path}")
        print(f"Saved t-SNE coordinates: {coords_path}")
    elif not args.no_tsne:
        print("Skipped t-SNE: at least 3 valid samples are required.")


if __name__ == "__main__":
    main()
