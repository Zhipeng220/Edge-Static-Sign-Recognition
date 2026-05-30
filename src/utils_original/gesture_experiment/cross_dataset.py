from .common import *
from .config import *
from .training import *
from .models import *
from .experiments import *


def experiment_configs_by_names(names):
    cfg_by_name = {}
    for cfg in [*BASELINE_CONFIGS, *ABLATION_CONFIGS]:
        cfg_by_name[cfg.name] = cfg

    missing = [name for name in names if name not in cfg_by_name]
    if missing:
        raise ValueError(f"Unknown cross-dataset variants: {missing}")

    return [cfg_by_name[name] for name in names]


def make_cross_dataset_split(source_paths: PathConfig, target_paths: PathConfig, seed: int):
    source_X, source_names = load_raw_data_with_names(source_paths)
    target_X, target_names = load_raw_data_with_names(target_paths)

    common_classes = sorted(set(source_names) & set(target_names))
    if len(common_classes) < 2:
        raise ValueError(
            f"Cross-dataset validation needs at least 2 shared classes; "
            f"{source_paths.dataset_name} and {target_paths.dataset_name} share {common_classes}."
        )

    source_mask = np.isin(source_names, common_classes)
    target_mask = np.isin(target_names, common_classes)
    source_X = source_X[source_mask]
    source_names = source_names[source_mask]
    target_X = target_X[target_mask]
    target_names = target_names[target_mask]

    label_encoder = LabelEncoder()
    label_encoder.fit(common_classes)
    source_y = label_encoder.transform(source_names)
    target_y = label_encoder.transform(target_names)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        source_X,
        source_y,
        test_size=0.20,
        random_state=seed,
        stratify=source_y,
    )

    return {
        "train": (X_train_raw, y_train),
        "val": (X_val_raw, y_val),
        "test": (target_X, target_y),
    }, label_encoder, common_classes


def run_cross_dataset_experiment(
    cfg: ExperimentConfig,
    seed: int,
    source_paths: PathConfig,
    target_paths: PathConfig,
    out_paths: PathConfig,
):
    print(
        f"\n--- Cross Dataset | Train: {source_paths.dataset_name} | "
        f"Test: {target_paths.dataset_name} | Experiment: {cfg.name} | Seed: {seed} ---"
    )

    set_seed(seed)
    os.makedirs(out_paths.result_dir, exist_ok=True)
    os.makedirs(out_paths.model_dir, exist_ok=True)

    split_data, label_encoder, common_classes = make_cross_dataset_split(
        source_paths=source_paths,
        target_paths=target_paths,
        seed=seed,
    )
    processed_data, scaler = preprocess_split(split_data, cfg, seed)

    if cfg.model_type == "svm":
        result, artifacts = run_svm_experiment(
            cfg=cfg,
            seed=seed,
            paths=out_paths,
            processed_data=processed_data,
            split_data=split_data,
            scaler=scaler,
            label_encoder=label_encoder,
        )
    else:
        loaders = make_loaders(processed_data, cfg)
        model = build_model(cfg, num_classes=len(label_encoder.classes_))
        best_model_path = os.path.join(out_paths.model_dir, f"{cfg.name}_seed{seed}.pth")
        model = train_one_model(
            model=model,
            loaders=loaders,
            cfg=cfg,
            save_path=best_model_path,
            scaler=scaler,
        )

        metrics, y_true, y_pred = evaluate_model(
            model=model,
            loader=loaders["test"],
            ci_seed=seed,
            measure_speed=True,
        )

        pred_path = os.path.join(out_paths.result_dir, f"predictions_{cfg.name}_seed{seed}.csv")
        cm_path = os.path.join(out_paths.result_dir, f"confusion_matrix_{cfg.name}_seed{seed}.pdf")
        report_path = os.path.join(out_paths.result_dir, f"classification_report_{cfg.name}_seed{seed}.txt")

        save_predictions(y_true, y_pred, label_encoder, pred_path)
        save_confusion_matrix(y_true, y_pred, label_encoder.classes_, cm_path)
        save_classification_report(y_true, y_pred, label_encoder.classes_, report_path)

        deployment_stats = collect_model_deployment_stats(
            cfg=cfg,
            paths=out_paths,
            seed=seed,
            model=model,
            model_path=best_model_path,
        )

        result = {
            "dataset": out_paths.dataset_name,
            "variant": cfg.name,
            "seed": seed,
            **metrics,
            **deployment_stats,
        }
        artifacts = {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "y_true": y_true,
            "y_pred": y_pred,
            "processed_data": processed_data,
            "split_data": split_data,
        }

    result.update({
        "train_dataset": source_paths.dataset_name,
        "test_dataset": target_paths.dataset_name,
        "shared_classes": len(common_classes),
        "train_samples": int(len(split_data["train"][1])),
        "val_samples": int(len(split_data["val"][1])),
        "test_samples": int(len(split_data["test"][1])),
    })
    artifacts["common_classes"] = common_classes
    return result, artifacts


def run_cross_dataset_validation(dataset_paths, result_root: str, seeds, cross_cfg: dict):
    if not cross_cfg.get("enabled", False):
        return []

    dataset_by_name = {paths.dataset_name: paths for paths in dataset_paths}
    variant_names = cross_cfg.get("variants", ["full_model"])
    variants = experiment_configs_by_names(variant_names)
    pairs = cross_cfg.get("pairs", [])
    all_results = []

    cross_root = os.path.join(result_root, "cross_dataset_validation")
    os.makedirs(cross_root, exist_ok=True)

    for pair in pairs:
        train_name = pair["train_dataset"]
        test_name = pair["test_dataset"]
        if train_name not in dataset_by_name or test_name not in dataset_by_name:
            raise ValueError(
                f"Unknown cross-dataset pair {train_name} -> {test_name}. "
                f"Available datasets: {sorted(dataset_by_name)}"
            )

        source_paths = dataset_by_name[train_name]
        target_paths = dataset_by_name[test_name]
        pair_name = f"train_{train_name}__test_{test_name}"
        out_dir = os.path.join(cross_root, pair_name)
        out_paths = PathConfig(
            dataset_name=pair_name,
            data_dir=source_paths.data_dir,
            result_dir=out_dir,
            model_dir=os.path.join(out_dir, "models"),
            x_path=source_paths.x_path,
            y_path=source_paths.y_path,
            class_mapping_path=source_paths.class_mapping_path,
            metadata_path=source_paths.metadata_path,
            image_test_dir=target_paths.image_test_dir,
            min_samples_per_class=source_paths.min_samples_per_class,
            filter_zero_rows=source_paths.filter_zero_rows,
        )

        pair_results = []
        for cfg in variants:
            for seed in seeds:
                result, _ = run_cross_dataset_experiment(
                    cfg=cfg,
                    seed=seed,
                    source_paths=source_paths,
                    target_paths=target_paths,
                    out_paths=out_paths,
                )
                pair_results.append(result)
                all_results.append(result)

        pair_df = pd.DataFrame(pair_results)
        pair_out = os.path.join(out_dir, "cross_dataset_all_seeds.csv")
        pair_df.to_csv(pair_out, index=False, encoding="utf-8-sig")
        summarize_cross_dataset_results(pair_df, os.path.join(out_dir, "cross_dataset_summary.csv"))
        print(f"Saved cross-dataset results: {pair_out}")

    if all_results:
        all_df = pd.DataFrame(all_results)
        all_out = os.path.join(cross_root, "ALL_cross_dataset_all_seeds.csv")
        all_df.to_csv(all_out, index=False, encoding="utf-8-sig")
        summarize_cross_dataset_results(
            all_df,
            os.path.join(cross_root, "ALL_cross_dataset_summary.csv"),
        )
        print(f"Saved combined cross-dataset results: {all_out}")

    return all_results


def summarize_cross_dataset_results(df: pd.DataFrame, summary_path: str):
    group_cols = ["train_dataset", "test_dataset", "variant"]
    numeric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_macro",
        "recall_macro",
        "latency_ms",
        "fps",
        "params",
        "shared_classes",
        "train_samples",
        "val_samples",
        "test_samples",
    ]
    agg_spec = {
        col: ["mean", "std"]
        for col in numeric_cols
        if col in df.columns and col != "params"
    }
    if "params" in df.columns:
        agg_spec["params"] = ["mean"]

    summary = flatten_summary_columns(df.groupby(group_cols).agg(agg_spec))
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary
