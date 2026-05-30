import os
import numpy as np
import joblib

from train_single_hand import normalize_single_hand_landmarks


def make_jetson_samples():
    data_dir = "./asl_npy_dataset_single_hand"
    result_dir = "./results_single_hand"

    x_path = os.path.join(data_dir, "X_data.npy")
    y_path = os.path.join(data_dir, "y_labels.npy")
    scaler_path = os.path.join(result_dir, "scaler_full_model_seed42.pkl")

    out_raw = os.path.join(result_dir, "sample_landmarks_raw.npy")
    out_norm = os.path.join(result_dir, "sample_landmarks_norm.npy")
    out_scaled = os.path.join(result_dir, "sample_landmarks_scaled.npy")
    out_labels = os.path.join(result_dir, "sample_labels.npy")

    X_raw = np.load(x_path)
    y = np.load(y_path)

    print("Loaded raw X:", X_raw.shape)
    print("Loaded y:", y.shape)

    if X_raw.shape[1] != 63:
        raise ValueError(
            f"Expected single-hand 63-dim features, got {X_raw.shape}. "
            "Please check whether you are loading the correct single-hand dataset."
        )

    # 取前 1000 个样本即可，不需要传整个数据集到 Jetson
    n = min(1000, len(X_raw))

    X_raw_sample = X_raw[:n].astype(np.float32)
    y_sample = y[:n]

    # 1. wrist-centered + palm-scale normalization
    X_norm_sample = normalize_single_hand_landmarks(X_raw_sample).astype(np.float32)

    # 2. StandardScaler transform
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Please make sure full_model seed=42 has been trained and saved."
        )

    scaler = joblib.load(scaler_path)
    X_scaled_sample = scaler.transform(X_norm_sample).astype(np.float32)

    np.save(out_raw, X_raw_sample)
    np.save(out_norm, X_norm_sample)
    np.save(out_scaled, X_scaled_sample)
    np.save(out_labels, y_sample)

    print("\nSaved Jetson sample files:")
    print(out_raw, X_raw_sample.shape)
    print(out_norm, X_norm_sample.shape)
    print(out_scaled, X_scaled_sample.shape)
    print(out_labels, y_sample.shape)

    print("\nFor TensorRT classifier-only testing, use:")
    print(out_scaled)


if __name__ == "__main__":
    make_jetson_samples()