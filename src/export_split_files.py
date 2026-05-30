import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _path_setup import add_original_utils_to_path

add_original_utils_to_path()

from gesture_experiment.config import (  # noqa: E402
    build_paths_from_spec,
    dataset_specs_from_config,
    load_runtime_config,
)
from gesture_experiment.training import load_raw_data  # noqa: E402


EXPORT_SEEDS = [42, 2024, 2025, 3407, 1234]
EXPECTED_DATASETS = {
    "asl_dataset",
    "indian_sign_language",
    "nus_hand_posture",
    "asl_large_dataset",
}


def make_split_indices(labels: np.ndarray, seed: int):
    sample_ids = np.arange(len(labels), dtype=int)
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        sample_ids,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids,
        y_temp,
        test_size=2 / 3,
        random_state=seed,
        stratify=y_temp,
    )
    return {
        "train": (train_ids, y_train),
        "val": (val_ids, y_val),
        "test": (test_ids, y_test),
    }


def split_output_dir(data_dir: str) -> Path:
    data_path = Path(data_dir)
    if data_path.name == "processed_landmarks":
        return data_path.parent / "split_files"
    return data_path / "split_files"


def split_frame(dataset: str, seed: int, split_name: str, sample_ids, labels, label_encoder):
    label_ids = np.asarray(labels, dtype=int)
    return pd.DataFrame(
        {
            "sample_id": np.asarray(sample_ids, dtype=int),
            "label_id": label_ids,
            "label_name": label_encoder.inverse_transform(label_ids),
            "split": split_name,
            "seed": int(seed),
            "dataset": dataset,
        }
    ).sort_values("sample_id")


def assert_no_overlap(split_indices, total_count: int):
    id_sets = {
        name: set(np.asarray(ids, dtype=int).tolist())
        for name, (ids, _) in split_indices.items()
    }
    checks = [
        id_sets["train"].isdisjoint(id_sets["val"]),
        id_sets["train"].isdisjoint(id_sets["test"]),
        id_sets["val"].isdisjoint(id_sets["test"]),
        sum(len(values) for values in id_sets.values()) == total_count,
    ]
    if not all(checks):
        raise AssertionError("Split overlap or count check failed.")
    return True


def export_dataset_splits(paths, seeds):
    features, labels, label_encoder = load_raw_data(paths)
    total_count = len(features)
    out_dir = split_output_dir(paths.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for seed in seeds:
        split_indices = make_split_indices(labels, seed)
        overlap_ok = assert_no_overlap(split_indices, total_count)

        for split_name, (sample_ids, split_labels) in split_indices.items():
            out_path = out_dir / f"seed{seed}_{split_name}.csv"
            df = split_frame(
                dataset=paths.dataset_name,
                seed=seed,
                split_name=split_name,
                sample_ids=sample_ids,
                labels=split_labels,
                label_encoder=label_encoder,
            )
            df.to_csv(out_path, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "dataset": paths.dataset_name,
                "seed": int(seed),
                "train_count": int(len(split_indices["train"][0])),
                "val_count": int(len(split_indices["val"][0])),
                "test_count": int(len(split_indices["test"][0])),
                "total_count": int(total_count),
                "num_classes": int(len(label_encoder.classes_)),
                "overlap_check_passed": bool(overlap_ok),
            }
        )

    summary_path = out_dir / "split_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def selected_dataset_specs(config, dataset_names):
    specs = dataset_specs_from_config(config)
    if dataset_names:
        wanted = set(dataset_names)
        specs = [spec for spec in specs if spec.dataset_name in wanted]
        missing = wanted - {spec.dataset_name for spec in specs}
        if missing:
            raise ValueError(f"Requested datasets are not configured: {sorted(missing)}")
    return [spec for spec in specs if spec.dataset_name in EXPECTED_DATASETS]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic sample-level train/val/test split CSV files "
            "using the same 70/10/20 stratified protocol as training."
        )
    )
    parser.add_argument("--config", default="configs/experiment_config.json", help="Runtime config JSON.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name to export; repeatable.")
    parser.add_argument("--seed", action="append", type=int, default=[], help="Seed to export; repeatable.")
    parser.add_argument("--continue-on-missing", action="store_true", help="Skip datasets with missing arrays.")
    args = parser.parse_args()

    runtime_cfg = load_runtime_config(args.config)
    seeds = args.seed or EXPORT_SEEDS
    specs = selected_dataset_specs(runtime_cfg, args.dataset)
    if not specs:
        raise ValueError("No configured datasets selected for split export.")

    exported = []
    for spec in specs:
        paths = build_paths_from_spec(spec, result_root=runtime_cfg["result_root"])
        try:
            summary_path = export_dataset_splits(paths, seeds)
        except FileNotFoundError:
            if args.continue_on_missing:
                print(f"[{paths.dataset_name}] skipped: missing processed landmark arrays.")
                continue
            raise
        exported.append(summary_path)
        print(f"[{paths.dataset_name}] saved split summary: {summary_path}")

    if not exported:
        raise RuntimeError("No split files were exported.")


if __name__ == "__main__":
    main()
