import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def export_scaler_params(scaler_path: Path, output_dir: Path):
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not scaler_path.is_file():
        raise ValueError(f"--scaler must be a file: {scaler_path}")

    scaler = joblib.load(scaler_path)
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("Scaler must expose mean_ and scale_ attributes.")

    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if mean.shape != scale.shape:
        raise ValueError(f"Scaler mean_ and scale_ shapes differ: {mean.shape} vs {scale.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = scaler_path.stem
    mean_path = output_dir / f"{stem}_mean.npy"
    scale_path = output_dir / f"{stem}_scale.npy"
    metadata_path = output_dir / f"{stem}_metadata.json"

    np.save(mean_path, mean)
    np.save(scale_path, scale)

    metadata = {
        "feature_count": int(mean.size),
        "mean_shape": list(mean.shape),
        "scale_shape": list(scale.shape),
        "source_scaler": str(scaler_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return mean_path, scale_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Export a fitted StandardScaler mean_ and scale_ arrays for deployment."
    )
    parser.add_argument("--scaler", required=True, help="Path to scaler .pkl file.")
    parser.add_argument("--output-dir", default="deployment/jetson", help="Output directory.")
    args = parser.parse_args()

    mean_path, scale_path, metadata_path = export_scaler_params(
        scaler_path=Path(args.scaler),
        output_dir=Path(args.output_dir),
    )
    print(f"Saved scaler mean: {mean_path}")
    print(f"Saved scaler scale: {scale_path}")
    print(f"Saved scaler metadata: {metadata_path}")


if __name__ == "__main__":
    main()
