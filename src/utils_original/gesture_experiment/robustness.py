from .common import *
from .config import *
from .data import *
from .models import *
from .training import *
from .experiments import *


def perturb_landmarks(X_raw, mode="gaussian", severity=0.01, seed=42):
    """
    Single-hand landmark perturbation.

    X_raw: [N, 63], raw MediaPipe coordinates before scaling.
    Return: [N, 63]
    """
    rng = np.random.default_rng(seed)
    X = X_raw.copy().reshape(-1, 21, 3)

    if mode == "none":
        pass

    elif mode == "gaussian":
        noise = rng.normal(0, severity, size=X.shape)
        X = X + noise

    elif mode == "keypoint_dropout":
        mask = rng.random(size=(X.shape[0], 21, 1)) < severity
        X = np.where(mask, 0.0, X)

    elif mode == "translation":
        shift = rng.uniform(-severity, severity, size=(X.shape[0], 1, 3))
        X = X + shift

    elif mode == "scaling":
        scale = rng.uniform(1.0 - severity, 1.0 + severity, size=(X.shape[0], 1, 1))
        X = X * scale

    elif mode == "coordinate_perturbation":
        noise = rng.normal(0, severity, size=X.shape)
        shift = rng.uniform(-severity, severity, size=(X.shape[0], 1, 3))
        scale = rng.uniform(1.0 - severity, 1.0 + severity, size=(X.shape[0], 1, 1))
        X = X * scale + shift + noise

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    return X.reshape(-1, 63)


def evaluate_numpy_landmarks(model, X_np, y_np, batch_size=64, ci_seed=42):
    num_landmarks = getattr(model, "num_landmarks", 21)
    coord_dim = getattr(model, "coord_dim", 3)

    X_tensor = torch.tensor(
        X_np.reshape(-1, num_landmarks, coord_dim),
        dtype=torch.float32
    )
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    metrics, y_true, y_pred = evaluate_model(
        model=model,
        loader=loader,
        ci_seed=ci_seed,
        measure_speed=False,
    )

    return metrics, y_true, y_pred


def run_landmark_robustness(paths: PathConfig):
    print("\n>>> Running Landmark-level Robustness Test")

    seed = 42
    cfg = ABLATION_CONFIGS[-1]  # full_model

    _, artifacts = run_experiment(cfg, seed, paths)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    processed_data = artifacts["processed_data"]

    X_test_raw = processed_data["X_test_raw"]
    y_test = processed_data["y_test"]

    rows = []

    modes = [
        "none",
        "gaussian",
        "keypoint_dropout",
        "translation",
        "scaling",
        "coordinate_perturbation",
    ]

    # 单手 MediaPipe 坐标范围较小，先用较温和扰动，避免把协议设计成过强攻击。
    severity_map = {
        "none": [0.0],
        "gaussian": [0.0005, 0.001, 0.002, 0.005],
        "keypoint_dropout": [0.02, 0.05, 0.10, 0.20],
        "translation": [0.001, 0.002, 0.005, 0.01],
        "scaling": [0.005, 0.01, 0.02, 0.05],
        "coordinate_perturbation": [0.0005, 0.001, 0.002, 0.005],
    }

    for mode in modes:
        for severity in severity_map[mode]:
            X_perturbed_raw = perturb_landmarks(
                X_test_raw,
                mode=mode,
                severity=severity,
                seed=seed,
            )

            if scaler is not None:
                X_perturbed = scaler.transform(X_perturbed_raw)
            else:
                X_perturbed = X_perturbed_raw

            metrics, _, _ = evaluate_numpy_landmarks(
                model=model,
                X_np=X_perturbed,
                y_np=y_test,
                batch_size=64,
                ci_seed=seed,
            )

            rows.append({
                "mode": mode,
                "severity": severity,
                **metrics,
            })

            print(f"{mode} | severity={severity} | acc={metrics['accuracy']:.6f}")

    out_path = os.path.join(paths.result_dir, "robustness_landmark.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Saved landmark robustness results to: {out_path}")


# ============================================================
# 9. Statistical Test: McNemar
# ============================================================

def exact_mcnemar_pvalue(b, c):
    """
    Exact two-sided McNemar test.
    b = baseline correct, full wrong
    c = baseline wrong, full correct
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    prob = 0.0

    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)

    return min(1.0, 2.0 * prob)


def run_mcnemar_test(paths: PathConfig, seeds=None, baseline_variants=None):
    print("\n>>> Running McNemar Test")

    if seeds is None:
        seeds = SEEDS
    if baseline_variants is None:
        baseline_variants = BASELINE_MCNEMAR_VARIANTS

    rows = []

    for baseline_variant in baseline_variants:
        for seed in seeds:
            baseline_path = os.path.join(
                paths.result_dir,
                f"predictions_{baseline_variant}_seed{seed}.csv",
            )
            full_path = os.path.join(paths.result_dir, f"predictions_full_model_seed{seed}.csv")

            if not os.path.exists(baseline_path) or not os.path.exists(full_path):
                print(f"Skipping {baseline_variant} seed {seed}: prediction files not found.")
                continue

            base_df = pd.read_csv(baseline_path)
            full_df = pd.read_csv(full_path)

            y_true = base_df["y_true_id"].values
            baseline_pred = base_df["y_pred_id"].values
            full_pred = full_df["y_pred_id"].values

            baseline_correct = baseline_pred == y_true
            full_correct = full_pred == y_true

            both_correct = int(np.sum(baseline_correct & full_correct))
            baseline_only_correct = int(np.sum(baseline_correct & ~full_correct))
            full_only_correct = int(np.sum(~baseline_correct & full_correct))
            both_wrong = int(np.sum(~baseline_correct & ~full_correct))

            p_value = exact_mcnemar_pvalue(
                b=baseline_only_correct,
                c=full_only_correct,
            )

            baseline_accuracy = float(np.mean(baseline_correct))
            full_accuracy = float(np.mean(full_correct))
            delta_accuracy = full_accuracy - baseline_accuracy
            delta_correct = full_only_correct - baseline_only_correct

            if delta_correct > 0:
                direction = "full_model_better"
            elif delta_correct < 0:
                direction = "baseline_better"
            else:
                direction = "tie"

            rows.append({
                "baseline_variant": baseline_variant,
                "seed": seed,
                "baseline_accuracy": baseline_accuracy,
                "full_accuracy": full_accuracy,
                "delta_accuracy_full_minus_baseline": delta_accuracy,
                "both_correct": both_correct,
                "baseline_only_correct": baseline_only_correct,
                "full_only_correct": full_only_correct,
                "delta_correct_full_minus_baseline": delta_correct,
                "direction": direction,
                "both_wrong": both_wrong,
                "mcnemar_statistic_min_discordant": min(baseline_only_correct, full_only_correct),
                "discordant_total": baseline_only_correct + full_only_correct,
                "p_value": p_value,
            })

            print(
                f"{baseline_variant} seed {seed}: p={p_value:.6g} | "
                f"direction={direction} | "
                f"full_only={full_only_correct}, baseline_only={baseline_only_correct}, "
                f"delta_acc={delta_accuracy:+.6f}"
            )

    out_path = os.path.join(paths.result_dir, "mcnemar_baselines_vs_full_model.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Saved McNemar results to: {out_path}")


# ============================================================
# 10. Dataset Extraction Report
# ============================================================

def create_dataset_extraction_report(paths: PathConfig):
    """
    Create a simple extraction / class-count report for both supported layouts:
      A) data_dir/X_data.npy + y_labels.npy + class_mapping.npy
      B) data_dir/{train,val,test}/X_data.npy + y_labels.npy + class_mapping.npy

    The old version only supported layout A, so NUS npy_seed42 crashed because
    its y_labels.npy files are inside train/val/test subfolders.
    """
    print("\n>>> Creating Dataset Extraction Report")

    os.makedirs(paths.result_dir, exist_ok=True)

    if os.path.exists(paths.metadata_path):
        meta = pd.read_csv(paths.metadata_path)

        if "label" in meta.columns:
            meta["original_label"] = meta["label"].astype(str)
            meta["canonical_label"] = meta["label"].map(canonicalize_label)
            canon_report_path = os.path.join(paths.result_dir, "label_canonicalization_report.csv")
            save_label_canonicalization_report(
                meta["original_label"].values,
                meta["canonical_label"].values,
                canon_report_path,
            )
            # Use canonical labels for the dataset report so ASL 'a' and 'A'
            # are not counted as different classes.
            label_col = "canonical_label"
        else:
            label_col = "label"

        overall = (
            meta.groupby("status")
            .size()
            .reset_index(name="count")
        )

        total = len(meta)
        overall["percentage"] = overall["count"] / max(total, 1)

        overall_path = os.path.join(paths.result_dir, "dataset_extraction_report.csv")
        overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

        # One-row dataset accounting summary for the paper appendix.
        accounting = {
            "dataset": paths.dataset_name,
            "metadata_rows": int(len(meta)),
        }
        for _, row in overall.iterrows():
            accounting[f"status_{row['status']}_count"] = int(row["count"])
            accounting[f"status_{row['status']}_percentage"] = float(row["percentage"])
        if label_col in meta.columns:
            accounting["num_canonical_classes_in_metadata"] = int(meta[label_col].nunique())
        accounting_path = os.path.join(paths.result_dir, "dataset_accounting_summary.csv")
        pd.DataFrame([accounting]).to_csv(accounting_path, index=False, encoding="utf-8-sig")

        if label_col in meta.columns:
            class_summary = (
                meta.groupby([label_col, "status"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
                .rename(columns={label_col: "label"})
            )
            class_count_summary = (
                meta.groupby(label_col)
                .size()
                .reset_index(name="count")
                .rename(columns={label_col: "label"})
                .sort_values("label")
            )
            class_count_summary["percentage"] = (
                class_count_summary["count"] / max(int(class_count_summary["count"].sum()), 1)
            )
        else:
            class_summary = pd.DataFrame()
            class_count_summary = pd.DataFrame()

        class_summary_path = os.path.join(paths.result_dir, "dataset_class_summary.csv")
        class_summary.to_csv(class_summary_path, index=False, encoding="utf-8-sig")

        class_count_path = os.path.join(paths.result_dir, "dataset_class_count_summary.csv")
        class_count_summary.to_csv(class_count_path, index=False, encoding="utf-8-sig")

        print(f"Saved: {overall_path}")
        print(f"Saved: {class_summary_path}")
        print(f"Saved: {class_count_path}")
        print(f"Saved: {accounting_path}")
        return

    # Fallback for datasets that only have .npy files.
    # This supports both root-level .npy and split-subfolder .npy exports.
    try:
        features, raw_labels, class_mapping = load_dataset_arrays(paths)
    except Exception as exc:
        print(f"[{paths.dataset_name}] Could not create dataset report: {exc}")
        return

    original_names = np.asarray(labels_to_names(raw_labels, class_mapping))
    names = canonicalize_labels(original_names)
    canon_report_path = os.path.join(paths.result_dir, "label_canonicalization_report.csv")
    save_label_canonicalization_report(original_names, names, canon_report_path)

    counts = (
        pd.Series(names, name="label")
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["label", "count"]
    counts["percentage"] = counts["count"] / max(int(counts["count"].sum()), 1)

    out_path = os.path.join(paths.result_dir, "dataset_extraction_report.csv")
    counts.to_csv(out_path, index=False)

    shape_report = pd.DataFrame([{
        "dataset": paths.dataset_name,
        "x_shape": str(tuple(features.shape)),
        "y_shape": str(tuple(raw_labels.shape)),
        "num_classes": int(len(counts)),
        "num_classes_after_canonicalization": int(len(counts)),
        "num_samples": int(len(raw_labels)),
    }])
    shape_path = os.path.join(paths.result_dir, "dataset_shape_report.csv")
    shape_report.to_csv(shape_path, index=False)

    accounting_path = os.path.join(paths.result_dir, "dataset_accounting_summary.csv")
    pd.DataFrame([{
        "dataset": paths.dataset_name,
        "metadata_rows": 0,
        "npy_samples": int(len(raw_labels)),
        "npy_feature_shape": str(tuple(features.shape)),
        "num_canonical_classes_in_npy": int(len(counts)),
    }]).to_csv(accounting_path, index=False, encoding="utf-8-sig")

    print(f"metadata.csv not found. Saved fallback report: {out_path}")
    print(f"Saved: {shape_path}")
    print(f"Saved: {accounting_path}")


# ============================================================
# 11. Image-level Robustness
# ============================================================

def degrade_image(image, mode="none", severity=0):
    img = image.copy()

    if mode == "none":
        return img

    if mode == "blur":
        k = 2 * int(severity) + 1
        k = max(3, k)
        return cv2.GaussianBlur(img, (k, k), 0)

    if mode == "brightness":
        factor = max(0.2, 1.0 - 0.20 * severity)
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    if mode == "occlusion":
        h, w = img.shape[:2]
        occ_w = int(w * 0.12 * severity)
        occ_h = int(h * 0.12 * severity)

        x1 = max(0, w // 2 - occ_w // 2)
        y1 = max(0, h // 2 - occ_h // 2)
        x2 = min(w, x1 + occ_w)
        y2 = min(h, y1 + occ_h)

        img[y1:y2, x1:x2] = 0
        return img

    if mode == "background_clutter":
        noise = np.random.normal(0, 15 * severity, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown image degradation mode: {mode}")


def extract_label_from_image_path(img_path, label_encoder):
    """
    Return the canonical class name used by label_encoder.

    This fixes ASL path-name drift such as 'a' vs 'A', 'space' vs 'SPACE',
    and filenames like 'A_001.jpg'. It tries the parent folder first, then
    common filename tokens.
    """
    canonical_lookup = {
        canonicalize_label(cls): str(cls)
        for cls in label_encoder.classes_
    }

    parent = os.path.basename(os.path.dirname(img_path))
    stem = os.path.splitext(os.path.basename(img_path))[0]

    candidates = [parent, stem]
    for sep in ["_", "-", " "]:
        if sep in stem:
            candidates.append(stem.split(sep)[0])

    for cand in candidates:
        cand_key = canonicalize_label(cand)
        if cand_key in canonical_lookup:
            return canonical_lookup[cand_key]

    return None


def mediapipe_features_from_image(image, hands):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    # If MediaPipe returns multiple hands, use the hand with the largest 2D
    # landmark bounding box. This keeps the single-hand protocol while avoiding
    # arbitrary first-hand selection.
    def bbox_area(hand_landmarks):
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        return max(max(xs) - min(xs), 0.0) * max(max(ys) - min(ys), 0.0)

    hand_landmarks = max(results.multi_hand_landmarks, key=bbox_area)

    row_features = []
    for lm in hand_landmarks.landmark:
        row_features.extend([lm.x, lm.y, lm.z])

    if len(row_features) != 63:
        return None

    return np.asarray(row_features, dtype=np.float32).reshape(1, -1)


def run_image_robustness(paths: PathConfig):
    """
    Optional image-level robustness test.

    This requires:
    - OpenCV
    - MediaPipe
    - paths.image_test_dir containing image folders by class
    """
    print("\n>>> Running Image-level Robustness Test")

    if not os.path.exists(paths.image_test_dir):
        print(f"Image test directory not found: {paths.image_test_dir}")
        print("Skipping image-level robustness.")
        return

    try:
        import mediapipe as mp
    except Exception as e:
        print(f"MediaPipe unavailable: {e}")
        print("Skipping image-level robustness.")
        return

    seed = 42
    cfg = ABLATION_CONFIGS[-1]

    _, artifacts = run_experiment(cfg, seed, paths)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]

    print("Model classes:", list(label_encoder.classes_))

    device = next(model.parameters()).device

    image_files = []
    for root, _, files in os.walk(paths.image_test_dir):
        for f in files:
            if (not f.startswith("._")) and f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, f))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )

    modes = ["none", "blur", "brightness", "occlusion", "background_clutter"]
    severity_map = {
        "none": [0],
        "blur": [1, 2, 3],
        "brightness": [1, 2, 3],
        "occlusion": [1, 2, 3],
        "background_clutter": [1, 2, 3],
    }

    rows = []

    for mode in modes:
        for severity in severity_map[mode]:
            y_true = []
            y_pred = []

            total_images = 0
            skipped_unknown_label = 0
            unreadable = 0
            detection_failures = 0

            for img_path in image_files:
                true_label_name = extract_label_from_image_path(img_path, label_encoder)

                if true_label_name is None:
                    skipped_unknown_label += 1
                    continue

                try:
                    true_id = label_encoder.transform([true_label_name])[0]
                except ValueError:
                    skipped_unknown_label += 1
                    continue

                img = cv2.imread(img_path)

                if img is None:
                    unreadable += 1
                    continue

                total_images += 1

                img = degrade_image(img, mode=mode, severity=severity)
                features = mediapipe_features_from_image(img, hands)

                if features is None:
                    detection_failures += 1
                    continue

                # Keep image-level inference consistent with training:
                # raw MediaPipe landmarks -> wrist/palm normalization -> StandardScaler.
                features = normalize_single_hand_landmarks(to_single_hand_features(features))

                if scaler is not None:
                    features = scaler.transform(features)

                x_tensor = torch.tensor(
                    features.reshape(1, cfg.num_landmarks, cfg.coord_dim),
                    dtype=torch.float32
                ).to(device)

                with torch.no_grad():
                    logits, _, _ = model(x_tensor)
                    pred_id = logits.argmax(dim=1).cpu().item()

                y_true.append(true_id)
                y_pred.append(pred_id)

            if len(y_true) > 0:
                acc = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            else:
                acc = 0.0
                macro_f1 = 0.0
                weighted_f1 = 0.0

            failure_rate = detection_failures / max(total_images, 1)

            row = {
                "mode": mode,
                "severity": severity,
                "total_images": total_images,
                "valid_predictions": len(y_true),
                "unreadable": unreadable,
                "skipped_unknown_label": skipped_unknown_label,
                "mediapipe_detection_failures": detection_failures,
                "mediapipe_failure_rate": failure_rate,
                "accuracy_on_detected": acc,
                "macro_f1_on_detected": macro_f1,
                "weighted_f1_on_detected": weighted_f1,
            }

            rows.append(row)

            print(
                f"{mode} | severity={severity} | "
                f"acc={acc:.6f} | MediaPipe fail={failure_rate:.4f}"
            )

    hands.close()

    out_path = os.path.join(paths.result_dir, "robustness_image.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Saved image-level robustness results to: {out_path}")



# ============================================================
# 11.5 Independent final image prediction set
# ============================================================

def run_final_holdout_image_test(paths: PathConfig, seed: int = 42):
    """
    Evaluate the independent final_test image set: one image per class moved out
    before MediaPipe extraction. This set is never included in X_data.npy.

    It loads the trained full_model checkpoint produced by run_ablation().
    """
    print(f"\n>>> Running Final Holdout Image Test: {paths.dataset_name}")

    if not paths.image_test_dir or not os.path.exists(paths.image_test_dir):
        print(f"[{paths.dataset_name}] final image test dir not found. Skipping: {paths.image_test_dir}")
        return

    try:
        import mediapipe as mp
    except Exception as e:
        print(f"MediaPipe unavailable: {e}")
        print("Skipping final holdout image test.")
        return

    cfg = ABLATION_CONFIGS[-1]  # full_model

    scaler_path = os.path.join(paths.result_dir, f"scaler_full_model_seed{seed}.pkl")
    encoder_path = os.path.join(paths.result_dir, f"label_encoder_full_model_seed{seed}.pkl")
    model_path = os.path.join(paths.model_dir, f"full_model_seed{seed}.pth")

    missing = [p for p in [scaler_path, encoder_path, model_path] if not os.path.exists(p)]
    if missing:
        print(
            f"[{paths.dataset_name}] Missing trained full_model artifacts. "
            f"Run ablation/full_model seed {seed} first. Missing: {missing}"
        )
        return

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    model = build_model(cfg, num_classes=len(label_encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(load_state_dict_safely(model_path, device))
    model = model.to(device)
    model.eval()

    image_files = []
    for root, _, files in os.walk(paths.image_test_dir):
        for f in files:
            # Skip macOS AppleDouble files such as ._xxx.png.
            if f.startswith("._"):
                continue
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, f))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )

    rows = []
    y_true = []
    y_pred = []

    for img_path in image_files:
        true_label_name = extract_label_from_image_path(img_path, label_encoder)

        if true_label_name is None:
            rows.append({
                "image_path": img_path,
                "true_label": "",
                "pred_label": "",
                "confidence": 0.0,
                "is_correct": False,
                "status": "unknown_label",
            })
            continue

        img = cv2.imread(img_path)
        if img is None:
            rows.append({
                "image_path": img_path,
                "true_label": true_label_name,
                "pred_label": "",
                "confidence": 0.0,
                "is_correct": False,
                "status": "unreadable_image",
            })
            continue

        features = mediapipe_features_from_image(img, hands)
        if features is None:
            rows.append({
                "image_path": img_path,
                "true_label": true_label_name,
                "pred_label": "",
                "confidence": 0.0,
                "is_correct": False,
                "status": "mediapipe_detection_failed",
            })
            continue

        features = normalize_single_hand_landmarks(to_single_hand_features(features))
        if scaler is not None:
            features = scaler.transform(features)

        x_tensor = torch.tensor(
            features.reshape(1, cfg.num_landmarks, cfg.coord_dim),
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            logits, _, _ = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])

        true_id = int(label_encoder.transform([true_label_name])[0])
        pred_label = str(label_encoder.inverse_transform([pred_id])[0])
        correct = pred_id == true_id

        y_true.append(true_id)
        y_pred.append(pred_id)

        rows.append({
            "image_path": img_path,
            "true_label": true_label_name,
            "pred_label": pred_label,
            "confidence": confidence,
            "is_correct": bool(correct),
            "status": "ok",
        })

    hands.close()

    pred_out = os.path.join(paths.result_dir, f"final_holdout_predictions_seed{seed}.csv")
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_out, index=False, encoding="utf-8-sig")

    status_counts = Counter(pred_df["status"].tolist()) if len(pred_df) else Counter()
    total_files = len(image_files)
    valid_predictions = len(y_true)
    non_ok = int(sum(v for k, v in status_counts.items() if k != "ok"))
    correct_ok = int(sum(1 for r in rows if r.get("status") == "ok" and r.get("is_correct")))

    if valid_predictions > 0:
        accuracy_on_detected = accuracy_score(y_true, y_pred)
        macro_f1_on_detected = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1_on_detected = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    else:
        accuracy_on_detected = 0.0
        macro_f1_on_detected = 0.0
        weighted_f1_on_detected = 0.0

    summary = {
        "dataset": paths.dataset_name,
        "seed": seed,
        "num_trained_classes": int(len(label_encoder.classes_)),
        "trained_classes": "|".join([str(x) for x in label_encoder.classes_]),
        "total_files": total_files,
        "valid_predictions": valid_predictions,
        "valid_prediction_rate": valid_predictions / max(total_files, 1),
        # Legacy column names are kept as detected-only metrics for backward
        # compatibility with previous result aggregation scripts. Prefer the
        # explicit *_on_detected and *_counting_failures_as_wrong columns in
        # the paper.
        "accuracy": accuracy_on_detected,
        "macro_f1": macro_f1_on_detected,
        "weighted_f1": weighted_f1_on_detected,
        "accuracy_on_detected": accuracy_on_detected,
        "macro_f1_on_detected": macro_f1_on_detected,
        "weighted_f1_on_detected": weighted_f1_on_detected,
        # Conservative metric for diagnostic holdout: unreadable / unknown / no-hand
        # cases count as incorrect rather than being silently removed.
        "accuracy_counting_failures_as_wrong": correct_ok / max(total_files, 1),
        "non_ok_files": non_ok,
        "mediapipe_or_read_failures": non_ok,
        "unknown_label": int(status_counts.get("unknown_label", 0)),
        "unreadable_image": int(status_counts.get("unreadable_image", 0)),
        "mediapipe_detection_failed": int(status_counts.get("mediapipe_detection_failed", 0)),
    }

    status_out = os.path.join(paths.result_dir, f"final_holdout_status_counts_seed{seed}.csv")
    (
        pd.DataFrame([{"status": k, "count": v} for k, v in sorted(status_counts.items())])
        .to_csv(status_out, index=False, encoding="utf-8-sig")
    )

    summary_out = os.path.join(paths.result_dir, f"final_holdout_summary_seed{seed}.csv")
    pd.DataFrame([summary]).to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"Saved final holdout predictions: {pred_out}")
    print(f"Saved final holdout summary: {summary_out}")
    print(f"Saved final holdout status counts: {status_out}")
    print(
        f"[{paths.dataset_name}] final holdout: "
        f"valid={summary['valid_predictions']}/{summary['total_files']} "
        f"({summary['valid_prediction_rate']:.2%}), "
        f"acc_on_detected={summary['accuracy_on_detected']:.6f}, "
        f"acc_failures_wrong={summary['accuracy_counting_failures_as_wrong']:.6f}, "
        f"macro_f1_on_detected={summary['macro_f1_on_detected']:.6f}, "
        f"status_counts={dict(status_counts)}"
    )


# ============================================================
# 12. Main Runners
# ============================================================

def run_stability(paths: PathConfig):
    print("\n>>> Running Stability Verification")

    seeds = SEEDS
    cfg = ABLATION_CONFIGS[-1]  # full_model

    all_results = []

    for seed in seeds:
        result, _ = run_experiment(cfg, seed, paths)
        all_results.append(result)

    df = pd.DataFrame(all_results)

    out_path = os.path.join(paths.result_dir, "stability_full_model.csv")
    df.to_csv(out_path, index=False)

    summary = df.groupby("variant").agg({
        "accuracy": ["mean", "std"],
        "accuracy_ci_low": ["mean"],
        "accuracy_ci_high": ["mean"],
        "macro_f1": ["mean", "std"],
        "macro_f1_ci_low": ["mean"],
        "macro_f1_ci_high": ["mean"],
        "weighted_f1": ["mean", "std"],
        "latency_ms": ["mean", "std"],
        "fps": ["mean", "std"],
        "params": ["mean"],
    })

    summary = flatten_summary_columns(summary)

    summary_path = os.path.join(paths.result_dir, "stability_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")
