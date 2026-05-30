import numpy as np

from gesture_experiment.config import ExperimentConfig
from gesture_experiment.data import normalize_single_hand_landmarks, to_single_hand_features
from gesture_experiment.training import preprocess_split


def _hand_sample(scale=1.0, offset=0.0):
    sample = np.zeros((21, 3), dtype=np.float32)
    sample[:, 0] = np.arange(21, dtype=np.float32) * scale + offset
    sample[9, 1] = scale
    return sample.reshape(-1)


def test_single_hand_normalization_is_wrist_centered_and_scale_stable():
    raw = np.stack([_hand_sample(scale=2.0, offset=5.0)])
    normalized = normalize_single_hand_landmarks(raw).reshape(1, 21, 3)

    assert normalized.shape == (1, 21, 3)
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized[:, 0, :], 0.0, atol=1e-6)
    palm_distance = np.linalg.norm(normalized[:, 9, :] - normalized[:, 0, :], axis=1)
    np.testing.assert_allclose(palm_distance, 1.0, atol=1e-6)


def test_two_hand_features_choose_nonzero_hand():
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.ones((21, 3), dtype=np.float32)
    features = np.concatenate([left.reshape(-1), right.reshape(-1)]).reshape(1, -1)

    selected = to_single_hand_features(features)

    assert selected.shape == (1, 63)
    assert np.isfinite(selected).all()
    np.testing.assert_allclose(selected, right.reshape(1, -1))


def test_scaler_is_fit_on_training_split_only():
    train = np.vstack([np.zeros(63), np.ones(63)])
    val = np.full((1, 63), 10.0)
    test = np.full((1, 63), 20.0)
    split_data = {
        "train": (train, np.array([0, 1])),
        "val": (val, np.array([0])),
        "test": (test, np.array([1])),
    }
    cfg = ExperimentConfig(
        name="no_leakage",
        use_normalization=True,
        use_smote=False,
    )

    processed, scaler = preprocess_split(split_data, cfg, seed=42)

    np.testing.assert_allclose(scaler.mean_, np.full(63, 0.5))
    np.testing.assert_allclose(processed["X_train"].mean(axis=0), np.zeros(63))
    assert processed["X_val"].mean() > 1.0
    assert processed["X_test"].mean() > processed["X_val"].mean()
