import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ID_COLUMN_CANDIDATES = [
    ("y_true_id", "y_pred_id"),
    ("true_id", "pred_id"),
]
LABEL_COLUMN_CANDIDATES = [
    ("y_true_label", "y_pred_label"),
    ("true_label", "pred_label"),
]
OUTPUT_FILES = [
    "metrics_summary.csv",
    "classification_report.txt",
    "confusion_matrix.csv",
    "confusion_matrix.png",
]


def resolve_columns(df: pd.DataFrame):
    for true_col, pred_col in ID_COLUMN_CANDIDATES:
        if true_col in df.columns and pred_col in df.columns:
            return true_col, pred_col, "id"
    for true_col, pred_col in LABEL_COLUMN_CANDIDATES:
        if true_col in df.columns and pred_col in df.columns:
            return true_col, pred_col, "label"
    expected = " or ".join(
        [f"{a}/{b}" for a, b in ID_COLUMN_CANDIDATES + LABEL_COLUMN_CANDIDATES]
    )
    raise ValueError(f"Prediction CSV must contain one column pair: {expected}")


def valid_prediction_rows(df: pd.DataFrame, true_col: str, pred_col: str):
    true_values = df[true_col]
    pred_values = df[pred_col]
    valid = true_values.notna() & pred_values.notna()
    valid &= true_values.astype(str).str.strip() != ""
    valid &= pred_values.astype(str).str.strip() != ""
    return valid


def prepare_vectors(df: pd.DataFrame, true_col: str, pred_col: str, column_type: str):
    valid = valid_prediction_rows(df, true_col, pred_col)
    skipped = int((~valid).sum())
    labeled = df.loc[valid, [true_col, pred_col]].copy()
    if labeled.empty:
        raise ValueError(
            f"No valid labeled predictions remain after skipping {skipped} rows."
        )

    if column_type == "id":
        try:
            y_true = labeled[true_col].astype(int).to_numpy()
            y_pred = labeled[pred_col].astype(int).to_numpy()
        except ValueError as exc:
            raise ValueError("ID prediction columns must contain integer values.") from exc
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        label_names = [str(label) for label in labels]
    else:
        y_true = labeled[true_col].astype(str).to_numpy()
        y_pred = labeled[pred_col].astype(str).to_numpy()
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        label_names = labels

    return y_true, y_pred, labels, label_names, skipped


def ensure_outputs_available(output_dir: Path, overwrite: bool):
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(existing)
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation files: {names}. "
            "Pass --overwrite to replace them."
        )


def save_confusion_matrix_png(cm: np.ndarray, label_names: list[str], out_path: Path):
    fig_width = max(6.0, min(18.0, 0.35 * len(label_names) + 4.0))
    fig_height = max(5.0, min(18.0, 0.35 * len(label_names) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_predictions(predictions_path: Path, output_dir: Path, overwrite: bool):
    df = pd.read_csv(predictions_path)
    true_col, pred_col, column_type = resolve_columns(df)
    y_true, y_pred, labels, label_names, skipped = prepare_vectors(
        df, true_col, pred_col, column_type
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_available(output_dir, overwrite)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "sample_count": int(len(y_true)),
        "skipped_rows": int(skipped),
        "source_predictions": str(predictions_path),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_summary.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    save_confusion_matrix_png(cm, label_names, output_dir / "confusion_matrix.png")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a prediction CSV and export metrics, report, and confusion matrix."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV.")
    parser.add_argument("--output-dir", default="results/evaluation", help="Directory for evaluation outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing evaluation outputs.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {predictions_path}")
    if not predictions_path.is_file():
        raise ValueError(f"--predictions must be a file: {predictions_path}")

    metrics = evaluate_predictions(
        predictions_path=predictions_path,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
    )
    print(f"Saved evaluation outputs to: {os.path.abspath(args.output_dir)}")
    print(f"Skipped rows without valid labels: {metrics['skipped_rows']}")
    print(f"Evaluated samples: {metrics['sample_count']}")


if __name__ == "__main__":
    main()
