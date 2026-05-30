import numpy as np
import pytest
from pathlib import Path
import pandas as pd

from gesture_experiment.training import make_split


def test_train_val_test_split_has_no_sample_overlap():
    features = np.arange(120 * 63, dtype=np.float32).reshape(120, 63)
    labels = np.repeat(np.arange(6), 20)

    split = make_split(features, labels, seed=42)

    row_sets = {
        name: {tuple(row) for row in data[0]}
        for name, data in split.items()
    }
    assert row_sets["train"].isdisjoint(row_sets["val"])
    assert row_sets["train"].isdisjoint(row_sets["test"])
    assert row_sets["val"].isdisjoint(row_sets["test"])
    assert sum(len(rows) for rows in row_sets.values()) == len(features)


def test_split_preserves_expected_70_10_20_sizes():
    features = np.arange(100 * 63, dtype=np.float32).reshape(100, 63)
    labels = np.repeat(np.arange(5), 20)

    split = make_split(features, labels, seed=2024)

    assert len(split["train"][0]) == 70
    assert len(split["val"][0]) == 10
    assert len(split["test"][0]) == 20


def test_generated_split_csvs_have_no_overlap_if_present():
    split_dirs = list(Path("data").glob("*/split_files"))
    csv_sets = []
    for split_dir in split_dirs:
        for train_path in split_dir.glob("seed*_train.csv"):
            prefix = train_path.name.replace("_train.csv", "")
            val_path = split_dir / f"{prefix}_val.csv"
            test_path = split_dir / f"{prefix}_test.csv"
            if val_path.exists() and test_path.exists():
                csv_sets.append((train_path, val_path, test_path))

    if not csv_sets:
        pytest.skip("No generated split CSV triplets found.")

    for train_path, val_path, test_path in csv_sets:
        train_ids = set(pd.read_csv(train_path)["sample_id"].astype(int))
        val_ids = set(pd.read_csv(val_path)["sample_id"].astype(int))
        test_ids = set(pd.read_csv(test_path)["sample_id"].astype(int))

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
