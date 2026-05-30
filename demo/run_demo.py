import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "demo" / "sample_images"
    default_model = repo_root / "models" / "full_model_seed42.pth"
    default_scaler = repo_root / "models" / "scaler_full_model_seed42.pkl"
    default_encoder = repo_root / "models" / "label_encoder_full_model_seed42.pkl"

    parser = argparse.ArgumentParser(description="Run the packaged prediction demo.")
    parser.add_argument("--input", default=str(default_input), help="Image file/dir or .npy input.")
    parser.add_argument("--dataset", default="asl_large_dataset", help="Dataset name from configs/experiment_config.json.")
    parser.add_argument("--variant", default="full_model", help="Model variant.")
    parser.add_argument("--seed", type=int, default=42, help="Artifact seed.")
    parser.add_argument("--output-dir", default=str(repo_root / "demo" / "outputs"), help="Prediction output directory.")
    parser.add_argument("--model-path", default=str(default_model), help="Path to model checkpoint.")
    parser.add_argument("--scaler-path", default=str(default_scaler), help="Path to fitted scaler artifact.")
    parser.add_argument("--encoder-path", default=str(default_encoder), help="Path to label encoder artifact.")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE generation.")
    args = parser.parse_args()

    required_artifacts = [
        Path(args.model_path),
        Path(args.scaler_path),
        Path(args.encoder_path),
    ]
    missing = [str(path) for path in required_artifacts if not path.exists()]
    if missing:
        print("Missing required demo artifact(s):", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        str(repo_root / "src" / "predict.py"),
        "--input",
        args.input,
        "--dataset",
        args.dataset,
        "--variant",
        args.variant,
        "--seed",
        str(args.seed),
        "--output-dir",
        args.output_dir,
        "--model-path",
        args.model_path,
        "--scaler-path",
        args.scaler_path,
        "--encoder-path",
        args.encoder_path,
    ]
    if args.no_tsne:
        command.append("--no-tsne")

    return subprocess.call(command, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
