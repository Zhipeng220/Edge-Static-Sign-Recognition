from _path_setup import add_original_utils_to_path

add_original_utils_to_path()

from gesture_experiment.common import *
from gesture_experiment.config import *
from gesture_experiment.data import *
from gesture_experiment.training import *
from gesture_experiment.experiments import *
from gesture_experiment.robustness import *
from gesture_experiment.summaries import *


def main():
    """
    Multi-dataset comparison entry.

    Main protocol:
      - same BASELINE_CONFIGS for same-protocol strong baseline comparison
      - same ABLATION_CONFIGS for stepwise component ablation
      - same seeds: SEEDS = [42, 2024, 2025, 3407, 1234]
      - same 70/10/20 stratified split generated inside this training script
      - different data_dir only

    Heavy optional robustness experiments are disabled by default.
    """
    runtime_cfg = load_runtime_config()
    result_root = runtime_cfg["result_root"]
    run_flags = runtime_cfg["run"]
    seeds = runtime_cfg["seeds"]

    run_baseline_comparison_enabled = run_flags["baseline_comparison"]
    run_ablation_enabled = run_flags["ablation"]
    run_stability_enabled = run_flags["stability"]
    run_landmark_robustness_enabled = run_flags["landmark_robustness"]
    run_image_robustness_enabled = run_flags["image_robustness"]
    run_final_holdout_enabled = run_flags["final_holdout"]
    # Keep false unless you explicitly want image-level final holdout for all seeds.
    # The main classifier tables already use all configured seeds.
    run_final_holdout_all_seeds_enabled = run_flags["final_holdout_all_seeds"]

    os.makedirs(result_root, exist_ok=True)

    all_baseline_seeds = []
    all_baseline_summaries = []
    all_ablation_seeds = []
    all_ablation_summaries = []
    all_mcnemar = []
    all_final_holdout = []
    all_clean_anatomical = []
    all_clean_anatomical_summaries = []
    all_split_accounting = []
    all_split_class_support = []
    all_dataset_accounting = []

    dataset_specs = dataset_specs_from_config(runtime_cfg)
    dataset_paths = [build_paths_from_spec(spec, result_root=result_root) for spec in dataset_specs]

    for paths in dataset_paths:
        print("\n" + "=" * 80)
        print(f"Running dataset: {paths.dataset_name}")
        print(f"Data dir: {paths.data_dir}")
        print("=" * 80)

        dataset_ok, reason = validate_landmark_dataset(paths)
        if not dataset_ok:
            print(f"Skipping dataset: {reason}")
            continue

        os.makedirs(paths.result_dir, exist_ok=True)
        os.makedirs(paths.model_dir, exist_ok=True)

        create_dataset_extraction_report(paths)

        if run_stability_enabled:
            run_stability(paths)

        # Paper Table 1: same-protocol strong landmark baselines.
        if run_baseline_comparison_enabled:
            run_baseline_comparison(paths, seeds=seeds)

        # Paper Table 2: stepwise ablation of the proposed pipeline.
        if run_ablation_enabled:
            run_ablation(paths, seeds=seeds)
            save_clean_anatomical_ablation(paths)

        # Statistical comparison: strong baselines vs full_model using the same seeds.
        if run_baseline_comparison_enabled:
            run_mcnemar_test(
                paths,
                seeds=seeds,
                baseline_variants=BASELINE_MCNEMAR_VARIANTS,
            )

        if run_landmark_robustness_enabled:
            run_landmark_robustness(paths)

        if run_image_robustness_enabled:
            run_image_robustness(paths)

        # Independent one-image-per-class final prediction set.
        if run_final_holdout_enabled:
            final_holdout_seeds = seeds if run_final_holdout_all_seeds_enabled else [seeds[0]]
            for holdout_seed in final_holdout_seeds:
                run_final_holdout_image_test(paths, seed=holdout_seed)

        baseline_all_path = os.path.join(paths.result_dir, "baseline_comparison_all_seeds.csv")
        baseline_summary_path = os.path.join(paths.result_dir, "baseline_comparison_summary.csv")
        ablation_all_path = os.path.join(paths.result_dir, "ablation_all_seeds.csv")
        ablation_summary_path = os.path.join(paths.result_dir, "ablation_summary.csv")
        mcnemar_path = os.path.join(paths.result_dir, "mcnemar_baselines_vs_full_model.csv")
        final_holdout_path = os.path.join(paths.result_dir, f"final_holdout_summary_seed{seeds[0]}.csv")
        clean_anatomical_path = os.path.join(paths.result_dir, "clean_anatomical_ablation.csv")
        clean_anatomical_summary_path = os.path.join(paths.result_dir, "clean_anatomical_ablation_summary.csv")
        dataset_accounting_path = os.path.join(paths.result_dir, "dataset_accounting_summary.csv")

        split_summary_path, split_class_path = collect_split_accounting(paths)

        if os.path.exists(baseline_all_path):
            df = pd.read_csv(baseline_all_path)
            df["dataset"] = paths.dataset_name
            all_baseline_seeds.append(df)

        if os.path.exists(baseline_summary_path):
            df = pd.read_csv(baseline_summary_path)
            df["dataset"] = paths.dataset_name
            all_baseline_summaries.append(df)

        if os.path.exists(ablation_all_path):
            df = pd.read_csv(ablation_all_path)
            df["dataset"] = paths.dataset_name
            all_ablation_seeds.append(df)

        if os.path.exists(ablation_summary_path):
            df = pd.read_csv(ablation_summary_path)
            df["dataset"] = paths.dataset_name
            all_ablation_summaries.append(df)

        if os.path.exists(mcnemar_path):
            df = pd.read_csv(mcnemar_path)
            df["dataset"] = paths.dataset_name
            all_mcnemar.append(df)

        # Collect final holdout summaries. By default this is seed42 only;
        # if final_holdout_all_seeds=true, this collects all available seeds.
        final_holdout_summary_paths = []
        if os.path.exists(final_holdout_path):
            final_holdout_summary_paths.append(final_holdout_path)
        if run_final_holdout_all_seeds_enabled:
            for holdout_seed in seeds:
                p = os.path.join(paths.result_dir, f"final_holdout_summary_seed{holdout_seed}.csv")
                if os.path.exists(p) and p not in final_holdout_summary_paths:
                    final_holdout_summary_paths.append(p)
        for p in final_holdout_summary_paths:
            df = pd.read_csv(p)
            df["dataset"] = paths.dataset_name
            all_final_holdout.append(df)

        if os.path.exists(clean_anatomical_path):
            df = pd.read_csv(clean_anatomical_path)
            df["dataset"] = paths.dataset_name
            all_clean_anatomical.append(df)

        if os.path.exists(clean_anatomical_summary_path):
            df = pd.read_csv(clean_anatomical_summary_path)
            df["dataset"] = paths.dataset_name
            all_clean_anatomical_summaries.append(df)

        if split_summary_path and os.path.exists(split_summary_path):
            df = pd.read_csv(split_summary_path)
            df["dataset"] = paths.dataset_name
            all_split_accounting.append(df)

        if split_class_path and os.path.exists(split_class_path):
            df = pd.read_csv(split_class_path)
            df["dataset"] = paths.dataset_name
            all_split_class_support.append(df)

        if os.path.exists(dataset_accounting_path):
            df = pd.read_csv(dataset_accounting_path)
            df["dataset"] = paths.dataset_name
            all_dataset_accounting.append(df)

    # Combined cross-dataset files for paper tables.
    if all_baseline_seeds:
        out = os.path.join(result_root, "ALL_DATASETS_baseline_comparison_all_seeds.csv")
        pd.concat(all_baseline_seeds, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined baseline seeds: {out}")

    if all_baseline_summaries:
        out = os.path.join(result_root, "ALL_DATASETS_baseline_comparison_summary.csv")
        pd.concat(all_baseline_summaries, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined baseline summary: {out}")

    if all_ablation_seeds:
        out = os.path.join(result_root, "ALL_DATASETS_ablation_all_seeds.csv")
        pd.concat(all_ablation_seeds, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined ablation seeds: {out}")

    if all_ablation_summaries:
        out = os.path.join(result_root, "ALL_DATASETS_ablation_summary.csv")
        pd.concat(all_ablation_summaries, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined ablation summary: {out}")

    if all_mcnemar:
        out = os.path.join(result_root, "ALL_DATASETS_mcnemar.csv")
        pd.concat(all_mcnemar, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined McNemar results: {out}")

    if all_final_holdout:
        out = os.path.join(result_root, "ALL_DATASETS_final_holdout_summary.csv")
        pd.concat(all_final_holdout, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined final holdout summary: {out}")

    if all_clean_anatomical:
        out = os.path.join(result_root, "ALL_DATASETS_clean_anatomical_ablation.csv")
        pd.concat(all_clean_anatomical, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined clean anatomical ablation: {out}")

    if all_clean_anatomical_summaries:
        out = os.path.join(result_root, "ALL_DATASETS_clean_anatomical_ablation_summary.csv")
        pd.concat(all_clean_anatomical_summaries, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined clean anatomical ablation summary: {out}")

    if all_split_accounting:
        out = os.path.join(result_root, "ALL_DATASETS_split_accounting.csv")
        pd.concat(all_split_accounting, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined split accounting: {out}")

    if all_split_class_support:
        out = os.path.join(result_root, "ALL_DATASETS_split_class_support.csv")
        pd.concat(all_split_class_support, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined split class support: {out}")

    if all_dataset_accounting:
        out = os.path.join(result_root, "ALL_DATASETS_dataset_accounting_summary.csv")
        pd.concat(all_dataset_accounting, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved combined dataset accounting summary: {out}")

if __name__ == "__main__":
    print("[INFO] Starting gesture experiment...")
    main()
    print("[INFO] Finished gesture experiment.")
