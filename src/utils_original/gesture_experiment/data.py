from .common import *
from .config import *


def normalize_single_hand_landmarks(X_raw, eps=1e-6):
    """
    X_raw: [N, 63], MediaPipe single-hand landmarks.
    Return: [N, 63], wrist-centered and palm-scale normalized landmarks.
    """
    X = X_raw.reshape(-1, 21, 3).copy()

    wrist = X[:, 0:1, :]      # landmark 0
    middle_mcp = X[:, 9:10, :]  # landmark 9

    # 平移归一化：以 wrist 为原点
    X = X - wrist

    # 尺度归一化：除以 wrist 到 middle MCP 的距离
    palm_size = np.linalg.norm(middle_mcp - wrist, axis=-1, keepdims=True)
    palm_size = np.maximum(palm_size, eps)

    X = X / palm_size

    return X.reshape(-1, 63).astype(np.float32)



def to_single_hand_features(features: np.ndarray) -> np.ndarray:
    """
    Convert extracted MediaPipe features to the common 21x3 single-hand protocol.

    Supported inputs:
      [N, 63]  -> already single hand
      [N, 126] -> two-hand export; choose the non-zero / stronger detected hand
    """
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got shape {features.shape}")

    if features.shape[1] == 63:
        return features

    if features.shape[1] == 126:
        hands = features.reshape(-1, 2, 21, 3)
        hand_energy = np.linalg.norm(hands.reshape(hands.shape[0], 2, -1), axis=2)
        best_hand = np.argmax(hand_energy, axis=1)
        single = hands[np.arange(hands.shape[0]), best_hand]
        return single.reshape(-1, 63).astype(np.float32)

    raise ValueError(
        f"Unsupported feature dimension {features.shape[1]}. "
        "This training script expects MediaPipe landmarks with 63 dims "
        "or two-hand landmarks with 126 dims. If NUS uses a different compact "
        "spatial-token dimension, first export/convert it to 21x3 landmarks."
    )


def labels_to_names(raw_labels: np.ndarray, class_mapping: dict) -> list[str]:
    """
    Convert y_labels.npy to class names robustly.

    Works for:
      - integer labels + class_mapping {'A': 0, 'B': 1}
      - string labels directly in y_labels.npy
    """
    raw_labels = np.asarray(raw_labels)

    if raw_labels.dtype.kind in {"U", "S", "O"}:
        return [str(x) for x in raw_labels]

    idx_to_name = {int(v): str(k) for k, v in class_mapping.items()}
    names = []
    for item in raw_labels:
        label_id = int(item)
        if label_id not in idx_to_name:
            raise ValueError(
                f"Label id {label_id} not found in class_mapping.npy. "
                f"Available ids: {sorted(idx_to_name.keys())[:10]}..."
            )
        names.append(idx_to_name[label_id])
    return names


SPECIAL_LABEL_ALIASES = {
    "space": "SPACE",
    "blank": "SPACE",
    "del": "DEL",
    "delete": "DEL",
    "deletion": "DEL",
    "nothing": "NOTHING",
    "none": "NOTHING",
    "no_hand": "NOTHING",
    "nohand": "NOTHING",
}


def canonicalize_label(label) -> str:
    """
    Canonicalize dataset labels before filtering, splitting, reporting, and
    final-holdout matching. This fixes common ASL folder/name drift such as
    'a' vs 'A', 'space' vs 'SPACE', and 'del' vs 'DEL'.

    The function intentionally keeps multi-character non-alias labels unchanged
    except for trimming whitespace, so numeric or dataset-specific classes are
    not accidentally collapsed.
    """
    s = str(label).strip()
    if not s:
        return s

    compact = s.replace("-", "_").replace(" ", "_")
    compact_lower = compact.lower()

    if compact_lower in SPECIAL_LABEL_ALIASES:
        return SPECIAL_LABEL_ALIASES[compact_lower]

    # ASL alphabet folders are sometimes mixed case. A single letter should be
    # treated as the same class regardless of case.
    if len(compact) == 1 and compact.isalpha():
        return compact.upper()

    return compact


def canonicalize_labels(labels) -> np.ndarray:
    return np.asarray([canonicalize_label(x) for x in labels])


def save_label_canonicalization_report(raw_names, canonical_names, out_path):
    """Save a transparent audit trail of label merges caused by canonicalization."""
    raw_names = np.asarray(raw_names).astype(str)
    canonical_names = np.asarray(canonical_names).astype(str)

    df = pd.DataFrame({
        "original_label": raw_names,
        "canonical_label": canonical_names,
    })
    report = (
        df.groupby(["original_label", "canonical_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["canonical_label", "original_label"])
    )
    report["changed"] = report["original_label"] != report["canonical_label"]
    report.to_csv(out_path, index=False, encoding="utf-8-sig")


def load_dataset_arrays(paths: PathConfig):
    """
    Load features/labels/class_mapping from either:
      A) data_dir/X_data.npy + y_labels.npy + class_mapping.npy
      B) data_dir/{train,val,test}/X_data.npy + y_labels.npy + class_mapping.npy

    Case B is useful for NUS exports. We concatenate all available split folders,
    then this script performs its own same-seed stratified split.
    """
    direct_required = [paths.x_path, paths.y_path, paths.class_mapping_path]
    if all(os.path.exists(p) for p in direct_required):
        features = np.load(paths.x_path)
        raw_labels = np.load(paths.y_path, allow_pickle=True)
        class_mapping = np.load(paths.class_mapping_path, allow_pickle=True).item()
        return features, raw_labels, class_mapping

    split_names = ["train", "val", "test"]
    feature_parts = []
    label_parts = []
    class_mapping = None
    found_splits = []

    for split_name in split_names:
        split_dir = os.path.join(paths.data_dir, split_name)
        x_path = os.path.join(split_dir, "X_data.npy")
        y_path = os.path.join(split_dir, "y_labels.npy")
        mapping_path = os.path.join(split_dir, "class_mapping.npy")

        if os.path.exists(x_path) and os.path.exists(y_path):
            feature_parts.append(np.load(x_path))
            label_parts.append(np.load(y_path, allow_pickle=True))
            found_splits.append(split_name)

            if class_mapping is None:
                if os.path.exists(paths.class_mapping_path):
                    class_mapping = np.load(paths.class_mapping_path, allow_pickle=True).item()
                elif os.path.exists(mapping_path):
                    class_mapping = np.load(mapping_path, allow_pickle=True).item()

    if feature_parts and class_mapping is not None:
        features = np.concatenate(feature_parts, axis=0)
        raw_labels = np.concatenate(label_parts, axis=0)
        print(
            f"[{paths.dataset_name}] Concatenated split folders {found_splits} "
            f"from {paths.data_dir}; internal train/val/test split will be regenerated."
        )
        return features, raw_labels, class_mapping

    missing = "\n  ".join(direct_required)
    raise FileNotFoundError(
        f"[{paths.dataset_name}] Could not find dataset npy files. Expected either:\n"
        f"  {missing}\n"
        f"or split folders under: {paths.data_dir}/train, val, test"
    )


def validate_landmark_dataset(paths: PathConfig):
    """
    Return whether one dataset can run under this script's 21x3 protocol.

    Some external exports store CNN visual tokens instead of MediaPipe
    landmarks. Those cannot be passed to the MLP/GCN/attention models here.
    """
    try:
        features, raw_labels, _ = load_dataset_arrays(paths)
    except Exception as exc:
        return False, str(exc)

    if len(features) != len(raw_labels):
        return False, (
            f"[{paths.dataset_name}] X/y length mismatch before training: "
            f"X has {len(features)} rows, y has {len(raw_labels)} labels."
        )

    if features.ndim != 2 or features.shape[1] not in {63, 126}:
        return False, (
            f"[{paths.dataset_name}] Incompatible feature shape {features.shape}. "
            "Expected MediaPipe landmarks shaped [N, 63] or [N, 126]. "
            "Visual-token exports must be converted to 21x3 landmarks before "
            "running this comparison."
        )

    return True, ""
