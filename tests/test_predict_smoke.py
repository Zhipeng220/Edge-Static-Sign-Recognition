from types import SimpleNamespace
import subprocess
import sys

import numpy as np

from predict import add_predictions_to_rows, predict_features, preprocess_for_model


class IdentityScaler:
    def transform(self, x):
        return x


class DummyClassifier:
    def predict(self, x):
        return (x[:, 0] > 0).astype(int)


class DummyLabelEncoder:
    classes_ = np.asarray(["NEG", "POS"])

    def inverse_transform(self, ids):
        return self.classes_[np.asarray(ids, dtype=int)]

    def transform(self, labels):
        mapping = {name: idx for idx, name in enumerate(self.classes_)}
        return np.asarray([mapping[label] for label in labels], dtype=int)


def test_predict_features_svm_smoke():
    bundle = {
        "cfg": SimpleNamespace(model_type="svm"),
        "model": DummyClassifier(),
        "label_encoder": DummyLabelEncoder(),
    }
    x = np.asarray([[-1.0] + [0.0] * 62, [2.0] + [0.0] * 62], dtype=np.float32)

    pred_ids, pred_labels, confidences = predict_features(bundle, x)

    np.testing.assert_array_equal(pred_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(pred_labels, np.asarray(["NEG", "POS"]))
    assert np.isnan(confidences).all()


def test_preprocess_for_model_smoke():
    x = np.zeros((1, 63), dtype=np.float32)
    x[0, 27] = 1.0

    processed = preprocess_for_model(x, IdentityScaler())

    assert processed.shape == (1, 63)
    assert processed.dtype == np.float32


def test_add_predictions_to_rows_smoke():
    rows = [{"status": "ok", "true_label": "POS"}]

    updated = add_predictions_to_rows(
        rows=rows,
        valid_indices=[0],
        true_labels=["POS"],
        pred_ids=np.asarray([1]),
        pred_labels=np.asarray(["POS"]),
        confidences=np.asarray([0.9]),
        label_encoder=DummyLabelEncoder(),
    )

    assert updated[0]["pred_label"] == "POS"
    assert updated[0]["is_correct"] is True


def test_cli_help_smoke():
    commands = [
        [sys.executable, "src/predict.py", "--help"],
        [sys.executable, "src/evaluate.py", "--help"],
        [sys.executable, "src/export_split_files.py", "--help"],
        [sys.executable, "src/export_scaler_params.py", "--help"],
    ]
    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
