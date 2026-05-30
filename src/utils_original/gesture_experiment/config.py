from .common import *
import json


@dataclass
class PathConfig:
    """
    A single dataset configuration.

    Required directory structure for data_dir:
        data_dir/
          X_data.npy
          y_labels.npy
          class_mapping.npy

    image_test_dir is optional and is only used for the independent final
    image-level prediction set that you moved out before MediaPipe extraction.
    """
    dataset_name: str = "asl_single_hand"
    data_dir: str = "./asl_npy_dataset_single_hand"
    result_dir: str = "./multi_dataset_results/asl_single_hand"
    model_dir: str = "./multi_dataset_results/asl_single_hand/models"

    x_path: str = "./asl_npy_dataset_single_hand/X_data.npy"
    y_path: str = "./asl_npy_dataset_single_hand/y_labels.npy"
    class_mapping_path: str = "./asl_npy_dataset_single_hand/class_mapping.npy"
    metadata_path: str = "./asl_npy_dataset_single_hand/metadata.csv"

    image_test_dir: str = ""

    # keep classes with enough samples for stratified train/val/test split.
    # Your datasets have many samples per class, so 20 is safer than the old 100
    # when MediaPipe drops some difficult images.
    min_samples_per_class: int = 20

    # zero-fill spatial-token exports may contain all-zero failed rows.
    filter_zero_rows: bool = True


@dataclass
class DatasetSpec:
    dataset_name: str
    data_dir: str
    image_test_dir: str = ""
    result_dir: str = ""
    model_dir: str = ""
    min_samples_per_class: int = 20
    filter_zero_rows: bool = True


RESULT_ROOT = "./multi_dataset_results_fixed_labels"

# Five random seeds used for baseline comparison, ablation, and McNemar tests.
# The first three are the original seeds; 3407 and 1234 are the two added seeds.
SEEDS = [42, 2024, 2025, 3407, 1234]

# ============================================================
# Optional deployment / efficiency reporting
# ============================================================
# These flags are safe defaults for a normal training machine. ONNX export and
# FLOP counting are attempted, but failures are recorded in CSV instead of
# crashing the whole experiment. TensorRT conversion is disabled by default
# because it requires NVIDIA TensorRT/trtexec on the host machine.
RUN_ONNX_EXPORT = True
RUN_TENSORRT_CONVERSION = False
TENSORRT_FP16 = True

# Export deployment artifacts for these variants. Add/remove names as needed.
DEPLOYMENT_EXPORT_VARIANTS = {
    "baseline_mlp",
    "baseline_pointnet",
    "baseline_gcn",
    "baseline_transformer_lite",
    "full_model_wo_anatomical_loss",
    "full_model",
}

CLEAN_ANATOMICAL_BASELINE = "full_model_wo_anatomical_loss"
CLEAN_ANATOMICAL_FULL = "full_model"


# 修改这里即可：data_dir 必须指向已经提取好的 npy 目录。
# 对 ASL / Indian：先把 final_test 每类 1 张移出去，再对剩余图片跑 MediaPipe 提取，
# 输出到下面的 npy_for_training 目录。
# 对 NUS：如果你用 manifest 导出脚本，data_dir 可以指向 npy_seed42/train；
# 如果你已经合并成一个 X_data.npy/y_labels.npy，则指向合并后的目录。
DATASET_SPECS = [
    # 1. 小 ASL 数据集
    DatasetSpec(
        dataset_name="asl_dataset",
        data_dir="./data/asl_dataset/processed_landmarks",
        image_test_dir="./data/asl_dataset/final_holdout",
    ),

    # 2. Indian Sign Language
    DatasetSpec(
        dataset_name="indian_sign_language",
        data_dir="./data/indian_sign_language/processed_landmarks",
        image_test_dir="./data/indian_sign_language/final_holdout",
    ),

    # 3. NUS Hand Posture，使用你刚刚提取出来的 MediaPipe 坐标
    DatasetSpec(
        dataset_name="nus_hand_posture",
        data_dir="./data/nus_hand_posture/processed_landmarks",
        image_test_dir="./data/nus_hand_posture/final_holdout",
    ),

    # 4. 大 ASL 数据集
    DatasetSpec(
        dataset_name="asl_large_dataset",
        data_dir="./data/asl_large_dataset/processed_landmarks",
        image_test_dir="./data/asl_large_dataset/final_holdout",
    ),
]


def build_paths_from_spec(spec: DatasetSpec, result_root: str = RESULT_ROOT) -> PathConfig:
    result_dir = spec.result_dir or os.path.join(result_root, spec.dataset_name)
    return PathConfig(
        dataset_name=spec.dataset_name,
        data_dir=spec.data_dir,
        result_dir=result_dir,
        model_dir=spec.model_dir or os.path.join(result_dir, "models"),
        x_path=os.path.join(spec.data_dir, "X_data.npy"),
        y_path=os.path.join(spec.data_dir, "y_labels.npy"),
        class_mapping_path=os.path.join(spec.data_dir, "class_mapping.npy"),
        metadata_path=os.path.join(spec.data_dir, "metadata.csv"),
        image_test_dir=spec.image_test_dir,
        min_samples_per_class=spec.min_samples_per_class,
        filter_zero_rows=spec.filter_zero_rows,
    )


DEFAULT_RUN_FLAGS = {
    "baseline_comparison": True,
    "ablation": True,
    "stability": False,
    "landmark_robustness": False,
    "image_robustness": False,
    "final_holdout": True,
    "final_holdout_all_seeds": False,
}


def default_runtime_config() -> dict:
    return {
        "result_root": RESULT_ROOT,
        "seeds": SEEDS,
        "datasets": [
            {
                "dataset_name": spec.dataset_name,
                "data_dir": spec.data_dir,
                "image_test_dir": spec.image_test_dir,
                "result_dir": spec.result_dir,
                "model_dir": spec.model_dir,
                "min_samples_per_class": spec.min_samples_per_class,
                "filter_zero_rows": spec.filter_zero_rows,
            }
            for spec in DATASET_SPECS
        ],
        "run": dict(DEFAULT_RUN_FLAGS),
    }


def load_runtime_config(config_path: str = "configs/experiment_config.json") -> dict:
    cfg = default_runtime_config()
    if not os.path.exists(config_path) and config_path == "configs/experiment_config.json":
        legacy_path = "experiment_config.json"
        if os.path.exists(legacy_path):
            config_path = legacy_path

    if not os.path.exists(config_path):
        return cfg

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    cfg["result_root"] = user_cfg.get("result_root", cfg["result_root"])
    cfg["seeds"] = user_cfg.get("seeds", cfg["seeds"])

    if "datasets" in user_cfg:
        cfg["datasets"] = user_cfg["datasets"]

    cfg["run"].update(user_cfg.get("run", {}))
    return cfg


def dataset_specs_from_config(cfg: dict) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            dataset_name=item["dataset_name"],
            data_dir=item["data_dir"],
            image_test_dir=item.get("image_test_dir", ""),
            result_dir=item.get("result_dir", ""),
            model_dir=item.get("model_dir", ""),
            min_samples_per_class=item.get("min_samples_per_class", 20),
            filter_zero_rows=item.get("filter_zero_rows", True),
        )
        for item in cfg.get("datasets", [])
    ]


@dataclass
class ExperimentConfig:
    name: str
    model_type: str = "attn"  # "mlp", "pointnet", "gcn", "transformer", "svm", or "attn"

    use_normalization: bool = True
    use_positional_encoding: bool = True
    use_attention: bool = True
    use_residual: bool = True
    use_anatomical_loss: bool = False
    use_smote: bool = True
    use_data_augmentation: bool = True

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 7

    embed_dim: int = 64
    num_heads: int = 8

    # 单手 ASL：21 landmarks × 3 coordinates = 63 features
    num_landmarks: int = 21
    coord_dim: int = 3

    # Anatomical regularization weights.
    reconstruction_weight: float = 0.10
    anatomical_weight: float = 0.01


# Same-protocol baseline comparison configs.
# These are used for the paper table comparing common landmark classifiers under
# the identical 70/10/20 split, train-only scaler/SMOTE, and the same seeds.
BASELINE_CONFIGS = [
    ExperimentConfig(
        name="baseline_mlp",
        model_type="mlp",
        use_normalization=True,
        use_positional_encoding=False,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=True,
        use_data_augmentation=True,
    ),
    ExperimentConfig(
        name="baseline_svm",
        model_type="svm",
        use_normalization=True,
        use_positional_encoding=False,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=True,
        # SVM does not use online augmentation; SMOTE is the train-only balancer.
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="baseline_pointnet",
        model_type="pointnet",
        use_normalization=True,
        use_positional_encoding=False,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=True,
        use_data_augmentation=True,
    ),
    ExperimentConfig(
        name="baseline_gcn",
        model_type="gcn",
        use_normalization=True,
        use_positional_encoding=False,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=True,
        use_data_augmentation=True,
    ),
    ExperimentConfig(
        name="baseline_transformer_lite",
        model_type="transformer",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=True,
        use_data_augmentation=True,
    ),
    ExperimentConfig(
        name="full_model",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=True,
        use_anatomical_loss=True,
        use_smote=True,
        use_data_augmentation=True,
    ),
]


# Stepwise ablation configs.
# These answer a different question from BASELINE_CONFIGS: which component of the
# proposed pipeline contributes to the final model.
ABLATION_CONFIGS = [
    ExperimentConfig(
        name="ablation_plain_mlp",
        model_type="mlp",
        use_normalization=False,
        use_positional_encoding=False,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=False,
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="plus_normalization",
        model_type="mlp",
        use_normalization=True,
        use_positional_encoding=False,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=False,
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="plus_positional_encoding",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=False,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=False,
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="plus_spatial_attention",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=False,
        use_anatomical_loss=False,
        use_smote=False,
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="plus_residual",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=True,
        use_anatomical_loss=False,
        use_smote=False,
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="plus_anatomical_loss",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=True,
        use_anatomical_loss=True,
        use_smote=False,
        use_data_augmentation=False,
    ),
    ExperimentConfig(
        name="plus_smote",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=True,
        use_anatomical_loss=True,
        use_smote=True,
        use_data_augmentation=False,
    ),
    # Clean anatomical-loss ablation required by reviewers:
    # identical to full_model except use_anatomical_loss=False.
    ExperimentConfig(
        name="full_model_wo_anatomical_loss",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=True,
        use_anatomical_loss=False,
        use_smote=True,
        use_data_augmentation=True,
    ),
    ExperimentConfig(
        name="full_model",
        model_type="attn",
        use_normalization=True,
        use_positional_encoding=True,
        use_attention=True,
        use_residual=True,
        use_anatomical_loss=True,
        use_smote=True,
        use_data_augmentation=True,
    ),
]


BASELINE_MCNEMAR_VARIANTS = [
    cfg.name for cfg in BASELINE_CONFIGS if cfg.name != "full_model"
]
