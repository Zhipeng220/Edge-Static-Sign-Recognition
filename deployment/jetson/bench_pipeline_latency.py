import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(image_dir):
    image_paths = []

    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                image_paths.append(os.path.join(root, f))

    image_paths.sort()
    return image_paths


def normalize_single_hand_landmarks(x_raw, eps=1e-6):
    """
    x_raw: [1, 63], raw MediaPipe single-hand landmarks.
    return: [1, 63], wrist-centered and palm-scale normalized landmarks.
    """
    X = x_raw.reshape(1, 21, 3).astype(np.float32)

    wrist = X[:, 0:1, :]       # landmark 0
    middle_mcp = X[:, 9:10, :] # landmark 9

    X = X - wrist

    palm_size = np.linalg.norm(middle_mcp - wrist, axis=-1, keepdims=True)
    palm_size = np.maximum(palm_size, eps)

    X = X / palm_size

    return X.reshape(1, 63).astype(np.float32)


def apply_standard_scaler_numpy(x_norm, scaler_mean, scaler_scale, eps=1e-12):
    """
    Manual StandardScaler transform:
        x_scaled = (x_norm - mean) / scale

    x_norm: [1, 63]
    scaler_mean: [63]
    scaler_scale: [63]
    return: [1, 63]
    """
    scaler_scale = np.maximum(scaler_scale, eps)
    return ((x_norm - scaler_mean) / scaler_scale).astype(np.float32)


def extract_single_hand_landmarks(results):
    """
    Extract the first detected hand as 63-dim features.
    """
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    row = []
    for lm in hand.landmark:
        row.extend([lm.x, lm.y, lm.z])

    if len(row) != 63:
        return None

    return np.asarray(row, dtype=np.float32).reshape(1, 63)


def load_scaler_arrays(scaler_prefix):
    """
    Load scaler mean and scale arrays.

    If scaler_prefix is:
        models/scaler_full_model_seed42

    This function loads:
        models/scaler_full_model_seed42_mean.npy
        models/scaler_full_model_seed42_scale.npy

    If user passes:
        models/scaler_full_model_seed42.pkl

    It will automatically strip .pkl and load the same two .npy files.
    """
    base = scaler_prefix

    if base.endswith(".pkl"):
        base = base[:-4]

    if base.endswith(".npy"):
        base = base[:-4]

    mean_path = base + "_mean.npy"
    scale_path = base + "_scale.npy"

    if not os.path.exists(mean_path):
        raise FileNotFoundError(
            f"Scaler mean not found: {mean_path}\n"
            f"Expected file name: {mean_path}"
        )

    if not os.path.exists(scale_path):
        raise FileNotFoundError(
            f"Scaler scale not found: {scale_path}\n"
            f"Expected file name: {scale_path}"
        )

    scaler_mean = np.load(mean_path).astype(np.float32)
    scaler_scale = np.load(scale_path).astype(np.float32)

    if scaler_mean.shape[0] != 63:
        raise ValueError(f"Expected scaler_mean shape [63], got {scaler_mean.shape}")

    if scaler_scale.shape[0] != 63:
        raise ValueError(f"Expected scaler_scale shape [63], got {scaler_scale.shape}")

    print(f"Loaded scaler mean: {mean_path}, shape={scaler_mean.shape}")
    print(f"Loaded scaler scale: {scale_path}, shape={scaler_scale.shape}")

    return scaler_mean, scaler_scale


def summarize_latency(df, out_path):
    latency_cols = [
        "image_read_ms",
        "bgr_to_rgb_ms",
        "mediapipe_ms",
        "landmark_norm_scaler_ms",
        "classifier_ms",
        "postprocess_ms",
        "total_pipeline_ms",
    ]

    rows = []

    for col in latency_cols:
        if col not in df.columns:
            continue

        values = df[col].dropna().values

        if len(values) == 0:
            continue

        rows.append({
            "stage": col,
            "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median_ms": float(np.median(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "n": int(len(values)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    return summary


def benchmark_image_folder(
    image_dir,
    scaler_prefix,
    classifier_ms,
    warmup,
    max_images,
    out_prefix,
    mediapipe_conf,
):
    image_paths = collect_images(image_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    if max_images > 0:
        image_paths = image_paths[:max_images]

    print("\n================ Pipeline Benchmark Config ================")
    print(f"Found images: {len(image_paths)}")
    print(f"Image dir: {image_dir}")
    print(f"Scaler prefix: {scaler_prefix}")
    print(f"Classifier latency inserted: {classifier_ms:.6f} ms")
    print(f"Warm-up images: {warmup}")
    print(f"MediaPipe min_detection_confidence: {mediapipe_conf}")
    print(f"Output prefix: {out_prefix}")

    scaler_mean, scaler_scale = load_scaler_arrays(scaler_prefix)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=mediapipe_conf,
    )

    rows = []

    # Warm-up MediaPipe and OpenCV.
    warmup_paths = image_paths[:min(warmup, len(image_paths))]

    print("\nWarming up...")
    for img_path in warmup_paths:
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _ = hands.process(rgb)

    print("Warm-up done.\n")

    for idx, img_path in enumerate(image_paths):
        row = {
            "idx": idx,
            "image_path": img_path,
            "success": 0,
            "read_failed": 0,
            "mediapipe_failed": 0,
        }

        total_start = time.perf_counter()

        # 1. Image read
        t0 = time.perf_counter()
        img = cv2.imread(img_path)
        t1 = time.perf_counter()

        row["image_read_ms"] = (t1 - t0) * 1000.0

        if img is None:
            row["read_failed"] = 1
            row["bgr_to_rgb_ms"] = np.nan
            row["mediapipe_ms"] = np.nan
            row["landmark_norm_scaler_ms"] = np.nan
            row["classifier_ms"] = np.nan
            row["postprocess_ms"] = np.nan
            row["total_pipeline_ms"] = np.nan
            row["total_wallclock_without_inserted_classifier_ms"] = np.nan
            rows.append(row)
            continue

        # 2. BGR to RGB preprocessing
        t2 = time.perf_counter()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t3 = time.perf_counter()

        row["bgr_to_rgb_ms"] = (t3 - t2) * 1000.0

        # 3. MediaPipe landmark extraction
        t4 = time.perf_counter()
        results = hands.process(rgb)
        t5 = time.perf_counter()

        row["mediapipe_ms"] = (t5 - t4) * 1000.0

        x_raw = extract_single_hand_landmarks(results)

        if x_raw is None:
            row["mediapipe_failed"] = 1
            row["landmark_norm_scaler_ms"] = np.nan
            row["classifier_ms"] = np.nan
            row["postprocess_ms"] = np.nan
            row["total_pipeline_ms"] = np.nan
            row["total_wallclock_without_inserted_classifier_ms"] = np.nan
            rows.append(row)
            continue

        # 4. Landmark normalization + manual scaler
        t6 = time.perf_counter()

        x_norm = normalize_single_hand_landmarks(x_raw)
        x_scaled = apply_standard_scaler_numpy(
            x_norm=x_norm,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

        # Final model input shape would be [1, 21, 3].
        _ = x_scaled.reshape(1, 21, 3)

        t7 = time.perf_counter()

        row["landmark_norm_scaler_ms"] = (t7 - t6) * 1000.0

        # 5. TensorRT classifier stage
        # The classifier-only latency has already been measured using trtexec.
        # We insert that value here to avoid requiring PyCUDA / TensorRT Python loop.
        row["classifier_ms"] = float(classifier_ms)

        # 6. Post-processing
        # Simulates argmax on logits.
        t8 = time.perf_counter()
        fake_logits = np.zeros((1, 28), dtype=np.float32)
        pred_id = int(np.argmax(fake_logits, axis=1)[0])
        t9 = time.perf_counter()

        row["postprocess_ms"] = (t9 - t8) * 1000.0
        row["pred_id_dummy"] = pred_id

        total_end = time.perf_counter()

        measured_without_classifier = (
            row["image_read_ms"]
            + row["bgr_to_rgb_ms"]
            + row["mediapipe_ms"]
            + row["landmark_norm_scaler_ms"]
            + row["postprocess_ms"]
        )

        row["total_pipeline_ms"] = measured_without_classifier + float(classifier_ms)
        row["total_wallclock_without_inserted_classifier_ms"] = (
            total_end - total_start
        ) * 1000.0

        row["success"] = 1
        rows.append(row)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")

    hands.close()

    df = pd.DataFrame(rows)

    raw_path = f"{out_prefix}_raw.csv"
    summary_path = f"{out_prefix}_summary.csv"
    report_path = f"{out_prefix}_report.csv"

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    df.to_csv(raw_path, index=False)

    success_df = df[df["success"] == 1].copy()
    summary = summarize_latency(success_df, summary_path)

    total_images = len(df)
    success_images = int(df["success"].sum())
    read_failed = int(df["read_failed"].sum())
    mediapipe_failed = int(df["mediapipe_failed"].sum())

    report = pd.DataFrame([{
        "total_images": total_images,
        "success_images": success_images,
        "read_failed": read_failed,
        "mediapipe_failed": mediapipe_failed,
        "success_rate": success_images / total_images if total_images > 0 else 0.0,
        "read_failure_rate": read_failed / total_images if total_images > 0 else 0.0,
        "mediapipe_failure_rate": mediapipe_failed / total_images if total_images > 0 else 0.0,
        "inserted_classifier_ms": float(classifier_ms),
        "successful_latency_samples": len(success_df),
    }])

    report.to_csv(report_path, index=False)

    print("\n================ Pipeline Latency Summary ================")
    if len(summary) > 0:
        print(summary)
    else:
        print("No successful samples. Summary is empty.")

    print("\n================ Pipeline Report ================")
    print(report.T)

    print("\nSaved:")
    print(raw_path)
    print(summary_path)
    print(report_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        required=True,
        help="Folder containing test images."
    )

    parser.add_argument(
        "--scaler",
        default="models/scaler_full_model_seed42",
        help=(
            "Scaler prefix. If you pass models/scaler_full_model_seed42, "
            "the script loads models/scaler_full_model_seed42_mean.npy and "
            "models/scaler_full_model_seed42_scale.npy. "
            "Passing .pkl is also allowed; it will be stripped."
        )
    )

    parser.add_argument(
        "--classifier_ms",
        type=float,
        required=True,
        help="TensorRT classifier-only latency in ms. Example: FP16 0.196, FP32 0.222"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warm-up images."
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=1000,
        help="Maximum number of images to evaluate. Use -1 for all."
    )

    parser.add_argument(
        "--mediapipe_conf",
        type=float,
        default=0.5,
        help="MediaPipe min_detection_confidence."
    )

    parser.add_argument(
        "--out_prefix",
        default="results/pipeline_latency",
        help="Output prefix."
    )

    args = parser.parse_args()

    benchmark_image_folder(
        image_dir=args.image_dir,
        scaler_prefix=args.scaler,
        classifier_ms=args.classifier_ms,
        warmup=args.warmup,
        max_images=args.max_images,
        out_prefix=args.out_prefix,
        mediapipe_conf=args.mediapipe_conf,
    )


if __name__ == "__main__":
    main()