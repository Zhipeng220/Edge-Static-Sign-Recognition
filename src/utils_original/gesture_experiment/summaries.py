from .common import *
from .config import *
from .training import *
from .experiments import *


def summarize_experiment_results(df: pd.DataFrame, summary_path: str):
    """Save the common mean/std summary used by baseline and ablation tables."""
    agg_spec = {
        "accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "weighted_f1": ["mean", "std"],
        "precision_macro": ["mean", "std"],
        "recall_macro": ["mean", "std"],
        "latency_ms": ["mean", "std"],
        "fps": ["mean", "std"],
        "params": ["mean"],
    }

    optional_numeric_cols = [
        "macs",
        "flops_approx",
        "thop_params",
        "model_file_size_mb",
        "pth_model_size_mb",
        "onnx_model_size_mb",
        "tensorrt_engine_size_mb",
        "onnx_max_abs_diff",
        "onnx_argmax_match",
    ]
    for col in optional_numeric_cols:
        if col in df.columns:
            agg_spec[col] = ["mean", "std"]

    summary = df.groupby("variant").agg(agg_spec)

    summary = flatten_summary_columns(summary)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def run_baseline_comparison(paths: PathConfig, seeds=None):
    """
    Same-protocol strong baseline comparison for the paper table.

    All variants use the same dataset loader, 70/10/20 stratified split seeds,
    train-only preprocessing, and comparable train-only balancing. SVM receives
    SMOTE-resampled normalized features but not online neural augmentation.
    """
    print("\n>>> Running Same-Protocol Baseline Comparison")

    seeds = seeds or SEEDS
    all_results = []

    for cfg in BASELINE_CONFIGS:
        for seed in seeds:
            result, _ = run_experiment(cfg, seed, paths)
            all_results.append(result)

    df = pd.DataFrame(all_results)

    out_path = os.path.join(paths.result_dir, "baseline_comparison_all_seeds.csv")
    df.to_csv(out_path, index=False)

    summary_path = os.path.join(paths.result_dir, "baseline_comparison_summary.csv")
    summarize_experiment_results(df, summary_path)

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")


def run_ablation(paths: PathConfig, seeds=None):
    """Stepwise ablation of the proposed pipeline, separated from baselines."""
    print("\n>>> Running Ablation Study")

    seeds = seeds or SEEDS
    all_results = []

    for cfg in ABLATION_CONFIGS:
        for seed in seeds:
            result, _ = run_experiment(cfg, seed, paths)
            all_results.append(result)

    df = pd.DataFrame(all_results)

    out_path = os.path.join(paths.result_dir, "ablation_all_seeds.csv")
    df.to_csv(out_path, index=False)

    summary_path = os.path.join(paths.result_dir, "ablation_summary.csv")
    summarize_experiment_results(df, summary_path)

    print(f"Saved: {out_path}")
    print(f"Saved: {summary_path}")
