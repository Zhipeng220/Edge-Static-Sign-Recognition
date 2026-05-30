from .common import *
from .config import *
from .data import *
from .models import *
from .training import *


def load_state_dict_safely(save_path, device):
    try:
        return torch.load(save_path, map_location=device, weights_only=True)
    except TypeError:
        # For older PyTorch versions.
        return torch.load(save_path, map_location=device)


def train_one_model(model, loaders, cfg: ExperimentConfig, save_path: str, scaler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_acc = -1.0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()

        train_losses = []

        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _, reconstructed = model(inputs)
            loss, loss_dict = compute_loss(
                outputs=logits,
                labels=labels,
                reconstructed=reconstructed,
                inputs=inputs,
                cfg=cfg,
                scaler=scaler,
            )

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in loaders["val"]:
                inputs = inputs.to(device)
                logits, _, _ = model(inputs)
                preds = logits.argmax(dim=1).cpu()

                val_preds.append(preds)
                val_labels.append(labels)

        y_val = torch.cat(val_labels).numpy()
        p_val = torch.cat(val_preds).numpy()
        val_acc = accuracy_score(y_val, p_val)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {np.mean(train_losses):.6f} | "
            f"Val Acc: {val_acc:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(load_state_dict_safely(save_path, device))
    model = model.to(device)

    return model


def svm_parameter_count(model):
    """Approximate the stored learned SVM coefficients for table comparison."""
    return int(
        model.support_vectors_.size
        + model.dual_coef_.size
        + model.intercept_.size
    )


def measure_svm_latency(model, input_features, warmup=30, runs=100):
    dummy = np.asarray(input_features[:1], dtype=np.float32)

    for _ in range(warmup):
        model.predict(dummy)

    start = time.perf_counter()
    for _ in range(runs):
        model.predict(dummy)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000 / runs
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return float(latency_ms), float(fps)


def run_svm_experiment(
    cfg: ExperimentConfig,
    seed: int,
    paths: PathConfig,
    processed_data,
    split_data,
    scaler,
    label_encoder,
):
    model = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        decision_function_shape="ovr",
    )
    model.fit(processed_data["X_train"], processed_data["y_train"])

    y_true = np.asarray(processed_data["y_test"])
    y_pred = model.predict(processed_data["X_test"])
    latency_ms, fps = measure_svm_latency(model, processed_data["X_test"])

    metrics = classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ci_seed=seed,
        params=svm_parameter_count(model),
        latency_ms=latency_ms,
        fps=fps,
    )

    pred_path = os.path.join(paths.result_dir, f"predictions_{cfg.name}_seed{seed}.csv")
    cm_path = os.path.join(paths.result_dir, f"confusion_matrix_{cfg.name}_seed{seed}.pdf")
    report_path = os.path.join(paths.result_dir, f"classification_report_{cfg.name}_seed{seed}.txt")
    model_path = os.path.join(paths.model_dir, f"{cfg.name}_seed{seed}.pkl")

    save_predictions(y_true, y_pred, label_encoder, pred_path)
    save_confusion_matrix(y_true, y_pred, label_encoder.classes_, cm_path)
    save_classification_report(y_true, y_pred, label_encoder.classes_, report_path)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
        },
        model_path,
    )

    deployment_stats = collect_svm_deployment_stats(
        cfg=cfg,
        paths=paths,
        seed=seed,
        model_path=model_path,
    )

    result = {
        "dataset": paths.dataset_name,
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
    return result, artifacts


def run_experiment(cfg: ExperimentConfig, seed: int, paths: PathConfig):
    print(
        f"\n--- Dataset: {paths.dataset_name} | "
        f"Experiment: {cfg.name} | Seed: {seed} ---"
    )

    set_seed(seed)

    os.makedirs(paths.result_dir, exist_ok=True)
    os.makedirs(paths.model_dir, exist_ok=True)

    features, labels, label_encoder = load_raw_data(paths)
    split_data = make_split(features, labels, seed)
    processed_data, scaler = preprocess_split(split_data, cfg, seed)

    save_split_accounting(
        paths=paths,
        cfg=cfg,
        seed=seed,
        split_data=split_data,
        processed_data=processed_data,
        label_encoder=label_encoder,
    )

    if cfg.model_type == "svm":
        return run_svm_experiment(
            cfg=cfg,
            seed=seed,
            paths=paths,
            processed_data=processed_data,
            split_data=split_data,
            scaler=scaler,
            label_encoder=label_encoder,
        )

    loaders = make_loaders(processed_data, cfg)

    model = build_model(cfg, num_classes=len(label_encoder.classes_))

    best_model_path = os.path.join(paths.model_dir, f"{cfg.name}_seed{seed}.pth")

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

    pred_path = os.path.join(paths.result_dir, f"predictions_{cfg.name}_seed{seed}.csv")
    cm_path = os.path.join(paths.result_dir, f"confusion_matrix_{cfg.name}_seed{seed}.pdf")
    report_path = os.path.join(paths.result_dir, f"classification_report_{cfg.name}_seed{seed}.txt")

    save_predictions(y_true, y_pred, label_encoder, pred_path)
    save_confusion_matrix(y_true, y_pred, label_encoder.classes_, cm_path)
    save_classification_report(y_true, y_pred, label_encoder.classes_, report_path)

    if cfg.name == "full_model":
        tsne_path = os.path.join(paths.result_dir, f"tsne_{cfg.name}_seed{seed}.pdf")
        try:
            plot_path, coords_path = save_tsne_plot(
                X=processed_data["X_test"],
                y=processed_data["y_test"],
                label_encoder=label_encoder,
                out_path=tsne_path,
                title=f"{paths.dataset_name} full_model test-set t-SNE",
                seed=seed,
            )
            print(f"Saved t-SNE plot: {plot_path}")
            print(f"Saved t-SNE coordinates: {coords_path}")
        except Exception as exc:
            print(f"t-SNE plot skipped for {cfg.name} seed {seed}: {type(exc).__name__}: {exc}")

        joblib.dump(scaler, os.path.join(paths.result_dir, f"scaler_{cfg.name}_seed{seed}.pkl"))
        joblib.dump(label_encoder, os.path.join(paths.result_dir, f"label_encoder_{cfg.name}_seed{seed}.pkl"))

    deployment_stats = collect_model_deployment_stats(
        cfg=cfg,
        paths=paths,
        seed=seed,
        model=model,
        model_path=best_model_path,
    )

    result = {
        "dataset": paths.dataset_name,
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

    return result, artifacts
