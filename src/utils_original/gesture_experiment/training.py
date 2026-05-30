from .common import *
from .config import *
from .data import *
from .models import *


def load_raw_data_with_names(paths: PathConfig, min_samples_per_class: int | None = None):
    """
    Load one dataset under the same single-hand 21x3 protocol.

    Important anti-leakage rule:
    This function only reads X_data.npy/y_labels.npy from paths.data_dir.
    The images moved to paths.image_test_dir are never used here.
    """
    if min_samples_per_class is None:
        min_samples_per_class = paths.min_samples_per_class

    features, raw_labels, class_mapping = load_dataset_arrays(paths)

    if len(features) != len(raw_labels):
        raise ValueError(
            f"[{paths.dataset_name}] X/y length mismatch: "
            f"X has {len(features)} rows, y has {len(raw_labels)} labels."
        )

    # Convert 126-dim two-hand MediaPipe exports to the common 63-dim single-hand input.
    features = to_single_hand_features(features)

    # Remove all-zero failed rows if present.
    if paths.filter_zero_rows:
        nonzero_mask = np.abs(features).sum(axis=1) > 1e-8
        removed = int((~nonzero_mask).sum())
        if removed > 0:
            print(f"[{paths.dataset_name}] Removed {removed} all-zero feature rows.")
        features = features[nonzero_mask]
        raw_labels = raw_labels[nonzero_mask]

    # Coordinate normalization must be done before train/val/test split StandardScaler.
    features = normalize_single_hand_landmarks(features)

    original_names = np.asarray(labels_to_names(raw_labels, class_mapping))
    raw_names = canonicalize_labels(original_names)

    if not np.array_equal(original_names.astype(str), raw_names.astype(str)):
        report_path = os.path.join(paths.result_dir, "label_canonicalization_report.csv")
        os.makedirs(paths.result_dir, exist_ok=True)
        save_label_canonicalization_report(original_names, raw_names, report_path)
        changed = int(np.sum(original_names.astype(str) != raw_names.astype(str)))
        print(
            f"[{paths.dataset_name}] Canonicalized {changed} label entries "
            f"before class filtering. Report: {report_path}"
        )

    # Drop tiny classes to keep stratified train/val/test reliable.
    unique_names, counts = np.unique(raw_names, return_counts=True)
    valid_names = unique_names[counts >= min_samples_per_class]
    valid_mask = np.isin(raw_names, valid_names)

    dropped = int((~valid_mask).sum())
    if dropped > 0:
        dropped_classes = sorted(set(raw_names[~valid_mask]))
        print(
            f"[{paths.dataset_name}] Dropped {dropped} samples from small classes "
            f"(<{min_samples_per_class} samples): {dropped_classes}"
        )

    features = features[valid_mask]
    filtered_names = raw_names[valid_mask]

    print(
        f"[{paths.dataset_name}] Loaded X={features.shape}, "
        f"classes={len(np.unique(filtered_names))}, samples={len(filtered_names)}"
    )

    return features.astype(np.float32), filtered_names.astype(str)


def load_raw_data(paths: PathConfig, min_samples_per_class: int | None = None):
    features, filtered_names = load_raw_data_with_names(paths, min_samples_per_class)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(filtered_names)

    return features.astype(np.float32), labels.astype(np.int64), label_encoder


def make_split(features, labels, seed):
    """
    70% train, 10% validation, 20% test.
    The split happens before normalization and SMOTE to avoid data leakage.
    """
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=2 / 3,
        random_state=seed,
        stratify=y_temp,
    )

    return {
        "train": (X_train_raw, y_train),
        "val": (X_val_raw, y_val),
        "test": (X_test_raw, y_test),
    }


def preprocess_split(split_data, cfg: ExperimentConfig, seed: int):
    X_train_raw, y_train = split_data["train"]
    X_val_raw, y_val = split_data["val"]
    X_test_raw, y_test = split_data["test"]

    scaler = StandardScaler() if cfg.use_normalization else None

    if cfg.use_normalization:
        X_train_proc = scaler.fit_transform(X_train_raw)
        X_val_proc = scaler.transform(X_val_raw)
        X_test_proc = scaler.transform(X_test_raw)
    else:
        X_train_proc = X_train_raw.copy()
        X_val_proc = X_val_raw.copy()
        X_test_proc = X_test_raw.copy()

    if cfg.use_smote:
        smote = SMOTE(random_state=seed)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)

    processed_data = {
        "X_train": X_train_proc,
        "y_train": y_train,
        "X_val": X_val_proc,
        "y_val": y_val,
        "X_test": X_test_proc,
        "y_test": y_test,

        # Raw data for robustness.
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
    }

    return processed_data, scaler


def make_loaders(processed_data, cfg: ExperimentConfig):
    def to_tensor(X, y):
        x_t = torch.tensor(
            X.reshape(-1, cfg.num_landmarks, cfg.coord_dim),
            dtype=torch.float32
        )
        y_t = torch.tensor(y, dtype=torch.long)
        return x_t, y_t

    X_train_t, y_train_t = to_tensor(processed_data["X_train"], processed_data["y_train"])
    X_val_t, y_val_t = to_tensor(processed_data["X_val"], processed_data["y_val"])
    X_test_t, y_test_t = to_tensor(processed_data["X_test"], processed_data["y_test"])

    train_ds = AugmentedGestureDataset(
        X_train_t,
        y_train_t,
        is_training=True,
        use_aug=cfg.use_data_augmentation,
    )
    val_ds = AugmentedGestureDataset(X_val_t, y_val_t, is_training=False, use_aug=False)
    test_ds = AugmentedGestureDataset(X_test_t, y_test_t, is_training=False, use_aug=False)

    return {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False),
    }


# ============================================================
# 6. Metrics, CI, Reports
# ============================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    low = np.percentile(scores, 100 * alpha / 2)
    high = np.percentile(scores, 100 * (1 - alpha / 2))

    return float(low), float(high)


def measure_latency(model, warmup=30, runs=100):
    device = next(model.parameters()).device
    model.eval()

    num_landmarks = getattr(model, "num_landmarks", 21)
    coord_dim = getattr(model, "coord_dim", 3)

    dummy = torch.randn(1, num_landmarks, coord_dim, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(runs):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

    latency_ms = (end - start) * 1000 / runs
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    return float(latency_ms), float(fps)


def evaluate_model(model, loader, ci_seed=42, measure_speed=True):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits, _, _ = model(inputs)
            preds = logits.argmax(dim=1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    latency_ms, fps = (0.0, 0.0)
    if measure_speed:
        latency_ms, fps = measure_latency(model)

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ci_seed=ci_seed,
        params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        latency_ms=latency_ms,
        fps=fps,
    )

    return metrics, y_true, y_pred


def classification_metrics(y_true, y_pred, ci_seed, params, latency_ms=0.0, fps=0.0):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    acc_low, acc_high = bootstrap_ci(
        y_true,
        y_pred,
        lambda yt, yp: accuracy_score(yt, yp),
        seed=ci_seed,
    )
    f1_low, f1_high = bootstrap_ci(
        y_true,
        y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        seed=ci_seed,
    )

    return {
        "accuracy": acc,
        "accuracy_ci_low": acc_low,
        "accuracy_ci_high": acc_high,
        "macro_f1": macro_f1,
        "macro_f1_ci_low": f1_low,
        "macro_f1_ci_high": f1_high,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "params": params,
        "latency_ms": latency_ms,
        "fps": fps,
    }


def save_predictions(y_true, y_pred, label_encoder, out_path):
    df = pd.DataFrame({
        "y_true_id": y_true,
        "y_pred_id": y_pred,
        "y_true_label": label_encoder.inverse_transform(y_true),
        "y_pred_label": label_encoder.inverse_transform(y_pred),
    })
    df.to_csv(out_path, index=False)


def pdf_output_path(out_path: str) -> str:
    """Normalize figure artifacts to PDF regardless of the requested suffix."""
    root, _ = os.path.splitext(out_path)
    return f"{root}.pdf"


def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    out_path = pdf_output_path(out_path)
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    })
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def stratified_tsne_sample(X, y, max_samples=2000, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    if len(y) <= max_samples:
        return X, y, np.arange(len(y))

    rng = np.random.default_rng(seed)
    selected = []
    classes, counts = np.unique(y, return_counts=True)
    quotas = np.maximum(1, np.floor(counts / counts.sum() * max_samples).astype(int))

    while quotas.sum() > max_samples:
        idx = int(np.argmax(quotas))
        if quotas[idx] > 1:
            quotas[idx] -= 1
        else:
            break

    for class_id, quota in zip(classes, quotas):
        class_idx = np.flatnonzero(y == class_id)
        take = min(int(quota), len(class_idx))
        selected.extend(rng.choice(class_idx, size=take, replace=False).tolist())

    selected = np.asarray(sorted(selected), dtype=int)
    return X[selected], y[selected], selected


def compute_tsne_embedding(X, seed=42):
    n_samples = len(X)
    if n_samples < 3:
        raise ValueError("t-SNE requires at least 3 samples.")

    perplexity = min(30, max(2, (n_samples - 1) // 3))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(X)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(X)


def save_tsne_plot(X, y, label_encoder, out_path, title="t-SNE of test features", seed=42):
    out_path = pdf_output_path(out_path)
    X_plot, y_plot, source_index = stratified_tsne_sample(X, y, seed=seed)
    embedding = compute_tsne_embedding(X_plot, seed=seed)
    class_names = label_encoder.inverse_transform(y_plot.astype(int))

    coords_path = os.path.splitext(out_path)[0] + "_coordinates.csv"
    pd.DataFrame({
        "source_index": source_index,
        "tsne_1": embedding[:, 0],
        "tsne_2": embedding[:, 1],
        "class_id": y_plot.astype(int),
        "class_label": class_names,
    }).to_csv(coords_path, index=False, encoding="utf-8-sig")

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    palette = sns.color_palette("tab20", n_colors=len(label_encoder.classes_))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=class_names,
        palette=palette,
        s=18,
        linewidth=0,
        alpha=0.82,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path, coords_path


def save_classification_report(y_true, y_pred, class_names, out_path):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=6,
        zero_division=0,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)


def flatten_summary_columns(df):
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in df.columns.to_flat_index()
    ]
    return df.reset_index()


def safe_file_size_mb(path):
    """Return file size in MB; NaN if the file does not exist."""
    if path and os.path.exists(path):
        return float(os.path.getsize(path) / (1024 * 1024))
    return float("nan")


def optional_package_version(package_name):
    try:
        module = __import__(package_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"not_available: {type(exc).__name__}: {exc}"




def get_torch_mha_fastpath_enabled():
    """Return current PyTorch MultiheadAttention fastpath state if available."""
    try:
        return torch.backends.mha.get_fastpath_enabled()
    except Exception:
        return None


def set_torch_mha_fastpath_enabled(enabled: bool):
    """
    Disable PyTorch fused MultiheadAttention export path during ONNX export.

    PyTorch may use aten::_native_multi_head_attention in eval mode. That fused
    operator is not supported by standard ONNX export in many PyTorch versions.
    Disabling the fastpath makes PyTorch trace decomposed attention ops.
    """
    try:
        torch.backends.mha.set_fastpath_enabled(enabled)
        return True
    except Exception:
        return False

class LogitsOnlyWrapper(nn.Module):
    """Wrap models that return (logits, attn_weights, reconstructed)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def try_compute_model_macs(model, cfg: ExperimentConfig, device):
    """
    Compute MACs/FLOPs with thop when available.

    Notes:
      - Some operations inside nn.MultiheadAttention may be partially counted
        by thop depending on installed versions.
      - FLOPs are reported as 2 * MACs, a common approximation.
    """
    stats = {
        "macs": float("nan"),
        "flops_approx": float("nan"),
        "thop_params": float("nan"),
        "flops_status": "not_run",
    }

    try:
        from thop import profile
    except Exception as exc:
        stats["flops_status"] = f"thop_not_available: {type(exc).__name__}: {exc}"
        return stats

    try:
        wrapper = LogitsOnlyWrapper(model).to(device).eval()
        dummy = torch.randn(1, cfg.num_landmarks, cfg.coord_dim, device=device)
        macs, thop_params = profile(wrapper, inputs=(dummy,), verbose=False)
        stats.update({
            "macs": float(macs),
            "flops_approx": float(2 * macs),
            "thop_params": float(thop_params),
            "flops_status": "ok",
        })
    except Exception as exc:
        stats["flops_status"] = f"failed: {type(exc).__name__}: {exc}"

    return stats


def try_export_onnx_and_check_parity(model, cfg: ExperimentConfig, onnx_path: str, device):
    """
    Export a logits-only ONNX model and compare ONNXRuntime logits with PyTorch.

    This gives a lightweight conversion-parity check that can be reported even
    before TensorRT is available. TensorRT parity can be run later from the saved
    ONNX/engine artifacts.
    """
    stats = {
        "onnx_model_size_mb": float("nan"),
        "onnx_export_status": "not_run",
        "onnx_parity_status": "not_run",
        "onnx_max_abs_diff": float("nan"),
        "onnx_argmax_match": float("nan"),
        "onnx_path": onnx_path,
    }

    try:
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        wrapper = LogitsOnlyWrapper(model).to(device).eval()
        dummy = torch.randn(2, cfg.num_landmarks, cfg.coord_dim, device=device)

        with torch.no_grad():
            torch_logits = wrapper(dummy).detach().cpu().numpy()

        # Disable PyTorch fused MultiheadAttention fastpath for ONNX export.
        # Without this, export may fail with aten::_native_multi_head_attention.
        old_mha_fastpath = get_torch_mha_fastpath_enabled()
        set_torch_mha_fastpath_enabled(False)
        try:
            torch.onnx.export(
                wrapper,
                dummy,
                onnx_path,
                input_names=["landmarks"],
                output_names=["logits"],
                dynamic_axes={
                    "landmarks": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        finally:
            if old_mha_fastpath is not None:
                set_torch_mha_fastpath_enabled(bool(old_mha_fastpath))

        stats["onnx_model_size_mb"] = safe_file_size_mb(onnx_path)
        stats["onnx_export_status"] = "ok"
    except Exception as exc:
        stats["onnx_export_status"] = f"failed: {type(exc).__name__}: {exc}"
        return stats

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_logits = sess.run(["logits"], {"landmarks": dummy.detach().cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
        argmax_match = float(np.mean(np.argmax(torch_logits, axis=1) == np.argmax(onnx_logits, axis=1)))
        stats.update({
            "onnx_parity_status": "ok",
            "onnx_max_abs_diff": max_abs_diff,
            "onnx_argmax_match": argmax_match,
        })
    except Exception as exc:
        stats["onnx_parity_status"] = f"onnxruntime_not_available_or_failed: {type(exc).__name__}: {exc}"

    return stats


def build_trtexec_command(onnx_path: str, engine_path: str):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--explicitBatch",
    ]
    if TENSORRT_FP16:
        cmd.append("--fp16")
    return cmd


def collect_model_deployment_stats(cfg: ExperimentConfig, paths: PathConfig, seed: int, model, model_path: str):
    """Collect FLOPs, file sizes, ONNX export status, and TensorRT command/size."""
    device = next(model.parameters()).device
    deploy_dir = os.path.join(paths.result_dir, "deployment")
    os.makedirs(deploy_dir, exist_ok=True)

    onnx_path = os.path.join(deploy_dir, f"{cfg.name}_seed{seed}.onnx")
    engine_path = os.path.join(deploy_dir, f"{cfg.name}_seed{seed}.engine")
    trtexec_cmd = build_trtexec_command(onnx_path, engine_path)
    trtexec_cmd_text = " ".join([f'"{x}"' if " " in x else x for x in trtexec_cmd])

    stats = {
        "model_file_size_mb": safe_file_size_mb(model_path),
        "pth_model_size_mb": safe_file_size_mb(model_path),
        "onnx_model_size_mb": float("nan"),
        "tensorrt_engine_size_mb": safe_file_size_mb(engine_path),
        "tensorrt_version": optional_package_version("tensorrt"),
        "trtexec_available": bool(shutil.which("trtexec")),
        "tensorrt_conversion_status": "not_run_RUN_TENSORRT_CONVERSION_false",
        "tensorrt_parity_status": "not_run_no_engine_or_runtime",
        "tensorrt_command": trtexec_cmd_text,
        "onnx_path": onnx_path,
        "tensorrt_engine_path": engine_path,
    }

    stats.update(try_compute_model_macs(model, cfg, device))

    if RUN_ONNX_EXPORT and cfg.name in DEPLOYMENT_EXPORT_VARIANTS:
        stats.update(try_export_onnx_and_check_parity(model, cfg, onnx_path, device))
    else:
        stats["onnx_export_status"] = "skipped_by_DEPLOYMENT_EXPORT_VARIANTS_or_RUN_ONNX_EXPORT_false"
        stats["onnx_parity_status"] = "skipped"

    # Write the TensorRT conversion command even if TensorRT is not installed.
    command_path = os.path.join(deploy_dir, f"trtexec_command_{cfg.name}_seed{seed}.txt")
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(trtexec_cmd_text + "\n")
    stats["tensorrt_command_file"] = command_path

    if RUN_TENSORRT_CONVERSION and stats.get("onnx_export_status") == "ok":
        if shutil.which("trtexec"):
            try:
                completed = subprocess.run(
                    trtexec_cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                log_path = os.path.join(deploy_dir, f"trtexec_{cfg.name}_seed{seed}.log")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(completed.stdout)
                stats["tensorrt_conversion_status"] = "ok" if completed.returncode == 0 else f"failed_returncode_{completed.returncode}"
                stats["tensorrt_engine_size_mb"] = safe_file_size_mb(engine_path)
                stats["tensorrt_log_path"] = log_path
            except Exception as exc:
                stats["tensorrt_conversion_status"] = f"failed: {type(exc).__name__}: {exc}"
        else:
            stats["tensorrt_conversion_status"] = "trtexec_not_available"

    row_path = os.path.join(deploy_dir, f"deployment_stats_{cfg.name}_seed{seed}.csv")
    pd.DataFrame([{"dataset": paths.dataset_name, "variant": cfg.name, "seed": seed, **stats}]).to_csv(
        row_path,
        index=False,
        encoding="utf-8-sig",
    )
    stats["deployment_stats_path"] = row_path
    return stats


def collect_svm_deployment_stats(cfg: ExperimentConfig, paths: PathConfig, seed: int, model_path: str):
    deploy_dir = os.path.join(paths.result_dir, "deployment")
    os.makedirs(deploy_dir, exist_ok=True)
    stats = {
        "model_file_size_mb": safe_file_size_mb(model_path),
        "pth_model_size_mb": float("nan"),
        "onnx_model_size_mb": float("nan"),
        "tensorrt_engine_size_mb": float("nan"),
        "macs": float("nan"),
        "flops_approx": float("nan"),
        "thop_params": float("nan"),
        "flops_status": "not_applicable_svm",
        "onnx_export_status": "not_applicable_svm",
        "onnx_parity_status": "not_applicable_svm",
        "onnx_max_abs_diff": float("nan"),
        "onnx_argmax_match": float("nan"),
        "tensorrt_version": optional_package_version("tensorrt"),
        "trtexec_available": bool(shutil.which("trtexec")),
        "tensorrt_conversion_status": "not_applicable_svm",
        "tensorrt_parity_status": "not_applicable_svm",
    }
    row_path = os.path.join(deploy_dir, f"deployment_stats_{cfg.name}_seed{seed}.csv")
    pd.DataFrame([{"dataset": paths.dataset_name, "variant": cfg.name, "seed": seed, **stats}]).to_csv(
        row_path,
        index=False,
        encoding="utf-8-sig",
    )
    stats["deployment_stats_path"] = row_path
    return stats


def save_split_accounting(paths: PathConfig, cfg: ExperimentConfig, seed: int, split_data, processed_data, label_encoder):
    """
    Save exact train/val/test counts and per-class support.

    Raw split counts are the true 70/10/20 split counts. processed_train_count
    includes SMOTE-resampled training rows when cfg.use_smote=True.
    """
    out_dir = os.path.join(paths.result_dir, "split_accounting")
    os.makedirs(out_dir, exist_ok=True)

    X_train_raw, y_train_raw = split_data["train"]
    X_val_raw, y_val = split_data["val"]
    X_test_raw, y_test = split_data["test"]
    y_train_proc = np.asarray(processed_data["y_train"])

    summary = {
        "dataset": paths.dataset_name,
        "variant": cfg.name,
        "seed": seed,
        "num_classes": int(len(label_encoder.classes_)),
        "total_after_filtering": int(len(y_train_raw) + len(y_val) + len(y_test)),
        "train_raw_count": int(len(y_train_raw)),
        "val_count": int(len(y_val)),
        "test_count": int(len(y_test)),
        "train_processed_count": int(len(y_train_proc)),
        "smote_enabled": bool(cfg.use_smote),
        "smote_added_count": int(len(y_train_proc) - len(y_train_raw)),
        "augmentation_enabled": bool(cfg.use_data_augmentation),
        "normalization_enabled": bool(cfg.use_normalization),
    }

    summary_path = os.path.join(out_dir, f"split_accounting_{cfg.name}_seed{seed}.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    rows = []
    for class_id, class_name in enumerate(label_encoder.classes_):
        rows.append({
            "dataset": paths.dataset_name,
            "variant": cfg.name,
            "seed": seed,
            "class_id": int(class_id),
            "class_label": str(class_name),
            "train_raw_count": int(np.sum(np.asarray(y_train_raw) == class_id)),
            "train_processed_count": int(np.sum(y_train_proc == class_id)),
            "val_count": int(np.sum(np.asarray(y_val) == class_id)),
            "test_count": int(np.sum(np.asarray(y_test) == class_id)),
        })

    class_path = os.path.join(out_dir, f"split_class_support_{cfg.name}_seed{seed}.csv")
    pd.DataFrame(rows).to_csv(class_path, index=False, encoding="utf-8-sig")

    return summary_path, class_path


def collect_split_accounting(paths: PathConfig):
    out_dir = os.path.join(paths.result_dir, "split_accounting")
    if not os.path.exists(out_dir):
        return None, None

    summary_frames = []
    class_frames = []
    for filename in os.listdir(out_dir):
        path = os.path.join(out_dir, filename)
        if filename.startswith("split_accounting_") and filename.endswith(".csv"):
            summary_frames.append(pd.read_csv(path))
        elif filename.startswith("split_class_support_") and filename.endswith(".csv"):
            class_frames.append(pd.read_csv(path))

    summary_out = None
    class_out = None
    if summary_frames:
        summary_df = pd.concat(summary_frames, ignore_index=True)
        summary_out = os.path.join(paths.result_dir, "split_accounting_all_variants.csv")
        summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    if class_frames:
        class_df = pd.concat(class_frames, ignore_index=True)
        class_out = os.path.join(paths.result_dir, "split_class_support_all_variants.csv")
        class_df.to_csv(class_out, index=False, encoding="utf-8-sig")

    return summary_out, class_out


def exact_mcnemar_pvalue(b, c):
    """
    Exact two-sided McNemar test.

    b: samples correct only for the baseline variant.
    c: samples correct only for the comparison variant.
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)

    return min(1.0, 2.0 * prob)


def save_clean_anatomical_ablation(paths: PathConfig):
    """Create the reviewer-facing clean anatomical ablation table."""
    ablation_path = os.path.join(paths.result_dir, "ablation_all_seeds.csv")
    if not os.path.exists(ablation_path):
        print(f"Clean anatomical ablation skipped; missing: {ablation_path}")
        return None, None

    df = pd.read_csv(ablation_path)
    needed = {CLEAN_ANATOMICAL_BASELINE, CLEAN_ANATOMICAL_FULL}
    available = set(df["variant"].unique()) if "variant" in df.columns else set()
    if not needed.issubset(available):
        print(
            "Clean anatomical ablation skipped; missing variants: "
            f"{sorted(needed - available)}"
        )
        return None, None

    rows = []
    for seed in sorted(df["seed"].unique()):
        wo = df[(df["variant"] == CLEAN_ANATOMICAL_BASELINE) & (df["seed"] == seed)]
        full = df[(df["variant"] == CLEAN_ANATOMICAL_FULL) & (df["seed"] == seed)]
        if wo.empty or full.empty:
            continue
        wo = wo.iloc[0]
        full = full.iloc[0]

        row = {
            "dataset": paths.dataset_name,
            "seed": int(seed),
            "without_anatomical_variant": CLEAN_ANATOMICAL_BASELINE,
            "with_anatomical_variant": CLEAN_ANATOMICAL_FULL,
            "accuracy_without_anatomical": float(wo["accuracy"]),
            "accuracy_with_anatomical": float(full["accuracy"]),
            "delta_accuracy_with_minus_without": float(full["accuracy"] - wo["accuracy"]),
            "macro_f1_without_anatomical": float(wo["macro_f1"]),
            "macro_f1_with_anatomical": float(full["macro_f1"]),
            "delta_macro_f1_with_minus_without": float(full["macro_f1"] - wo["macro_f1"]),
        }

        # Optional McNemar test for the clean pair if prediction files exist.
        wo_pred_path = os.path.join(paths.result_dir, f"predictions_{CLEAN_ANATOMICAL_BASELINE}_seed{seed}.csv")
        full_pred_path = os.path.join(paths.result_dir, f"predictions_{CLEAN_ANATOMICAL_FULL}_seed{seed}.csv")
        if os.path.exists(wo_pred_path) and os.path.exists(full_pred_path):
            wo_pred = pd.read_csv(wo_pred_path)
            full_pred = pd.read_csv(full_pred_path)
            y_true = wo_pred["y_true_id"].values
            wo_correct = wo_pred["y_pred_id"].values == y_true
            full_correct = full_pred["y_pred_id"].values == y_true
            without_only = int(np.sum(wo_correct & ~full_correct))
            with_only = int(np.sum(~wo_correct & full_correct))
            row.update({
                "without_only_correct": without_only,
                "with_only_correct": with_only,
                "mcnemar_p_value": exact_mcnemar_pvalue(without_only, with_only),
            })

        rows.append(row)

    out_path = os.path.join(paths.result_dir, "clean_anatomical_ablation.csv")
    clean_df = pd.DataFrame(rows)
    clean_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = clean_df.agg({
        "accuracy_without_anatomical": ["mean", "std"],
        "accuracy_with_anatomical": ["mean", "std"],
        "delta_accuracy_with_minus_without": ["mean", "std"],
        "macro_f1_without_anatomical": ["mean", "std"],
        "macro_f1_with_anatomical": ["mean", "std"],
        "delta_macro_f1_with_minus_without": ["mean", "std"],
    }).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    summary["dataset"] = paths.dataset_name
    summary_path = os.path.join(paths.result_dir, "clean_anatomical_ablation_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved clean anatomical ablation: {out_path}")
    print(f"Saved clean anatomical ablation summary: {summary_path}")
    return out_path, summary_path
