from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..generalization.build_dataset import (
    DEFAULT_FEATURE_NAME,
    binarize_labels,
    canonicalize_dataset_name,
    canonicalize_model_name,
    compute_main_probe_split_indices,
    get_feature_slices,
    load_selected_feature_source,
)
from ..generalization.evaluate import save_aggregate_comparison_summary
from ..generalization.evaluate import save_comparison_summary
from ..generalization.paths import (
    DEFAULT_OUTPUT_ROOT,
    all_datasets_feature_root,
    all_datasets_shared_train_root,
    model_gen_feature_root,
    task_gen_feature_root,
)
from ..probe.models import BrierScoreLoss, LinearProbe, MLPProbe
from ..probe.train import (
    evaluate,
    make_feature_outdir,
    plot_training_history,
    print_predictions,
    train_epoch,
)


def _load_payload(data_path: str) -> Dict[str, Any]:
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    required = {"X", "y", "feature_names", "rows", "metadata", "dataset_names", "split_labels"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(
            f"Generalization dataset at {data_path} is missing required keys: {missing}."
        )
    return payload


def _feature_slice_from_payload(payload: Dict[str, Any], feature_name: str) -> slice:
    feature_names = payload.get("feature_names")
    if feature_names is None:
        raise ValueError(f"Generalization dataset at {payload!r} is missing feature_names.")
    name_to_slice = get_feature_slices(
        feature_names=list(feature_names),
        feature_dims=payload.get("feature_dims"),
        total_dim=int(payload["X"].shape[1]),
    )
    if feature_name not in name_to_slice:
        raise ValueError(
            f"Feature {feature_name!r} not present in generalization dataset. "
            f"Available features: {feature_names}"
        )
    return name_to_slice[feature_name]


def _tensor_stats(y: torch.Tensor) -> Dict[str, Any]:
    num_samples = int(y.numel())
    if num_samples == 0:
        return {
            "num_samples": 0,
            "num_correct": 0,
            "num_incorrect": 0,
            "accuracy": 0.0,
            "variance": 0.0,
        }
    return {
        "num_samples": num_samples,
        "num_correct": int((y == 1.0).sum().item()),
        "num_incorrect": int((y == 0.0).sum().item()),
        "accuracy": float(y.mean().item()),
        "variance": float(y.var(unbiased=False).item()),
    }


def _has_two_classes(y: torch.Tensor) -> bool:
    unique_vals = torch.unique(y.float().cpu())
    return bool(len(unique_vals) == 2 and torch.all((unique_vals == 0.0) | (unique_vals == 1.0)).item())


def _format_mean_std(mean_value: Optional[float], std_value: Optional[float]) -> str:
    if mean_value is None:
        return "N/A"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f}±{std_value:.4f}"


def _selection_summary_from_mask(
    dataset_names: Sequence[str],
    split_labels: Sequence[str],
    mask: torch.Tensor,
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for idx, include in enumerate(mask.tolist()):
        if not include:
            continue
        dataset_name = dataset_names[idx]
        split_name = split_labels[idx]
        summary.setdefault(dataset_name, {"train": 0, "val": 0, "test": 0})
        summary[dataset_name][split_name] += 1
    return summary


def _single_dataset_summary(dataset_name: str, split_name: str, count: int) -> Dict[str, Dict[str, int]]:
    summary = {dataset_name: {"train": 0, "val": 0, "test": 0}}
    summary[dataset_name][split_name] = int(count)
    return summary


def _normalize_split_tensors(
    *,
    train_X: torch.Tensor,
    val_X: torch.Tensor,
    test_X: torch.Tensor,
    normalize: bool,
) -> Dict[str, Any]:
    out = {
        "train_X": train_X.clone(),
        "val_X": val_X.clone(),
        "test_X": test_X.clone(),
        "mean": None,
        "std": None,
    }
    if not normalize:
        return out

    mean = out["train_X"].mean(dim=0)
    std = out["train_X"].std(dim=0, unbiased=False).clamp_min(1e-6)
    out["train_X"] = (out["train_X"] - mean) / std
    out["val_X"] = (out["val_X"] - mean) / std
    out["test_X"] = (out["test_X"] - mean) / std
    out["mean"] = mean
    out["std"] = std
    return out


def _build_model(
    *,
    model_type: str,
    input_dim: int,
    hidden_dims: Optional[List[int]],
    dropout: float,
    activation: str,
    device: torch.device,
):
    model_key = model_type.lower()
    if model_key == "linear":
        return LinearProbe(input_dim=input_dim).to(device)
    if model_key == "mlp":
        if hidden_dims is None:
            hidden_dims = [256, 128]
        return MLPProbe(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        ).to(device)
    raise ValueError(f"Unsupported model_type: {model_type!r}. Use 'linear' or 'mlp'.")


def _make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def _select_from_combined_payload(
    payload: Dict[str, Any],
    mask: torch.Tensor,
    feature_name: str,
) -> Dict[str, Any]:
    feature_slice = _feature_slice_from_payload(payload, feature_name)
    X = payload["X"][mask][:, feature_slice].float().clone()
    y = payload["y"][mask].float().clone()
    rows = [payload["rows"][idx] for idx, include in enumerate(mask.tolist()) if include]
    return {"X": X, "y": y, "rows": rows}


def _select_from_source_payload(
    source_payload: Dict[str, Any],
    indices: Sequence[int],
    binarize_threshold: float,
) -> Dict[str, Any]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    X = source_payload["X"][index_tensor].float().clone()
    y = binarize_labels(source_payload["y_raw"][index_tensor].float(), threshold=binarize_threshold)
    rows = [source_payload["rows"][idx] for idx in indices]
    return {"X": X, "y": y, "rows": rows}


def _sample_index_pool(
    pool_indices: Sequence[int],
    desired_count: int,
    seed: int,
) -> Dict[str, Any]:
    pool = list(pool_indices)
    if len(pool) == 0:
        raise ValueError("Cannot sample from an empty index pool.")

    generator = torch.Generator().manual_seed(seed)
    if desired_count <= len(pool):
        perm = torch.randperm(len(pool), generator=generator)[:desired_count].tolist()
        chosen = [pool[idx] for idx in perm]
        with_replacement = False
    else:
        sampled = torch.randint(low=0, high=len(pool), size=(desired_count,), generator=generator).tolist()
        chosen = [pool[idx] for idx in sampled]
        with_replacement = True

    return {
        "indices": chosen,
        "requested_count": int(desired_count),
        "available_count": int(len(pool)),
        "with_replacement": with_replacement,
    }


def _load_main_probe_test_split(
    *,
    dataset_name: str,
    per_dataset_metadata: Dict[str, Any],
    feature_name: str,
    binarize_threshold: float,
) -> Dict[str, Any]:
    dataset_meta = per_dataset_metadata[dataset_name]
    source = load_selected_feature_source(dataset_meta["source_path"], feature_name)
    split_indices = compute_main_probe_split_indices(
        num_samples=len(source["y_raw"]),
        train_fraction=float(dataset_meta["main_probe_train_fraction"]),
        val_fraction=float(dataset_meta["main_probe_val_fraction"]),
        seed=int(dataset_meta["main_probe_split_seed"]),
    )
    selected = _select_from_source_payload(
        source,
        indices=split_indices["test"],
        binarize_threshold=binarize_threshold,
    )
    return {
        "source_path": dataset_meta["source_path"],
        "selection": selected,
        "split_indices": split_indices,
    }


def _train_probe_run(
    *,
    feature_names: List[str],
    output_dir: str,
    run_variant: str,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    train_selection_summary: Dict[str, Dict[str, int]],
    val_selection_summary: Dict[str, Dict[str, int]],
    sampling_info: Optional[Dict[str, Any]],
    model_type: str,
    hidden_dims: Optional[List[int]],
    dropout: float,
    activation: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    normalize: bool,
    seed: int,
    device: torch.device,
    use_class_weights: bool,
    shuffle_train_labels: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    print("\n------------------------------------------------------------")
    print(f"Starting run variant: {run_variant}")
    print("------------------------------------------------------------")
    print(f"Train samples: {len(train_y)}")
    print(f"Val samples:   {len(val_y)}")

    train_y_local = train_y.clone()
    if shuffle_train_labels:
        print("[CONTROL] Shuffling train labels")
        perm = torch.randperm(train_y_local.shape[0])
        train_y_local = train_y_local[perm]

    normalized = _normalize_split_tensors(
        train_X=train_X,
        val_X=val_X,
        test_X=val_X[:0],
        normalize=normalize,
    )

    output_path = make_feature_outdir(model_type, feature_names, base_dir=output_dir)
    os.makedirs(output_path, exist_ok=True)

    use_pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        normalized["train_X"],
        train_y_local,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
    )
    val_loader = _make_loader(
        normalized["val_X"],
        val_y,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    train_stats = _tensor_stats(train_y_local)
    val_stats = _tensor_stats(val_y)
    dataset_stats = {
        "train": train_stats,
        "val": val_stats,
        "train_selection": train_selection_summary,
        "val_selection": val_selection_summary,
    }
    if sampling_info is not None:
        dataset_stats["sampling_info"] = sampling_info

    input_dim = normalized["train_X"].shape[1]
    model = _build_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        device=device,
    )

    if use_class_weights:
        num_pos = train_stats["num_correct"]
        num_neg = train_stats["num_incorrect"]
        neg_weight = num_pos / num_neg if num_neg > 0 else 1.0
    else:
        neg_weight = 1.0

    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Feature names: {feature_names}")
    print(f"Output dir: {output_path}")
    print(f"Negative-class weight: {neg_weight:.4f}")

    criterion = BrierScoreLoss(neg_weight=neg_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auroc": [],
        "val_auroc": [],
    }

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            train_metrics = evaluate(model, train_loader, criterion, device)
            train_auroc = train_metrics.get("auroc", None)
        else:
            train_auroc = history["train_auroc"][-1] if history["train_auroc"] else None

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["train_auroc"].append(float(train_auroc) if train_auroc is not None else None)
        history["val_auroc"].append(float(val_metrics["auroc"]) if "auroc" in val_metrics else None)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_ece": float(val_metrics["ece"]),
                "feature_names": feature_names,
                "input_dim": input_dim,
                "normalize": normalize,
                "mean": normalized["mean"],
                "std": normalized["std"],
                "val_auroc": float(val_metrics["auroc"]) if "auroc" in val_metrics else None,
            }
            torch.save(checkpoint, os.path.join(output_path, "best_model.pt"))

        if epoch == 0 or (epoch + 1) % 10 == 0:
            val_auroc = val_metrics.get("auroc")
            val_auroc_str = f"{val_auroc:.4f}" if val_auroc is not None else "N/A"
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"TrainLoss: {train_loss:.4f} | "
                f"ValLoss: {val_metrics['loss']:.4f} | "
                f"ValAUROC: {val_auroc_str} | "
                f"ValECE: {val_metrics['ece']:.4f}"
            )

    checkpoint = torch.load(os.path.join(output_path, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    with open(os.path.join(output_path, "training_history.json"), "w") as handle:
        json.dump(history, handle, indent=2)
    plot_training_history(history, output_path)

    return {
        "output_path": output_path,
        "model": model,
        "criterion": criterion,
        "normalize": normalize,
        "mean": normalized["mean"],
        "std": normalized["std"],
        "dataset_stats_base": dataset_stats,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
    }


def _evaluate_trained_probe(
    *,
    trained_run: Dict[str, Any],
    eval_output_dir: str,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    test_rows: List[Dict[str, Any]],
    test_selection_summary: Dict[str, Dict[str, int]],
    feature_names: List[str],
    model_type: str,
    hidden_dims: Optional[List[int]],
    dropout: float,
    activation: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    print_test_predictions: bool,
    use_class_weights: bool,
    shuffle_train_labels: bool,
    evaluation_name: str,
    data_path: str,
    held_out_source_path: str,
    held_out_dataset: str,
    sampling_info: Optional[Dict[str, Any]] = None,
) -> str:
    if trained_run["normalize"]:
        test_X_local = (test_X.clone() - trained_run["mean"]) / trained_run["std"]
    else:
        test_X_local = test_X.clone()

    test_loader = _make_loader(
        test_X_local,
        test_y,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = trained_run["model"]
    criterion = trained_run["criterion"]
    output_path = trained_run["output_path"] if eval_output_dir == trained_run["output_path"] else make_feature_outdir(
        model_type,
        feature_names,
        base_dir=eval_output_dir,
    )
    os.makedirs(output_path, exist_ok=True)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        return_preds=True,
        save_reliability_path=os.path.join(output_path, "reliability_diagram.png"),
        reliability_title=f"Reliability diagram ({evaluation_name} on {held_out_dataset})",
    )

    metrics_out = {
        "test_loss": test_metrics["loss"],
        "test_auroc": test_metrics.get("auroc", None),
        "test_ece": test_metrics["ece"],
        "best_epoch": trained_run["best_epoch"],
        "best_val_loss": trained_run["best_val_loss"],
    }
    with open(os.path.join(output_path, "metrics_test.json"), "w") as handle:
        json.dump(metrics_out, handle, indent=2)

    preds = test_metrics["predictions"].numpy()
    labels = test_metrics["labels"].numpy()
    df = pd.DataFrame({"prediction": preds, "label": labels})
    df.to_csv(os.path.join(output_path, "summary_predictions.csv"), index=False)

    df0 = df[df["label"] == 0].sort_values("prediction", ascending=False).head(50)
    df1 = df[df["label"] == 1].sort_values("prediction", ascending=True).head(50)
    pd.concat([df0, df1], axis=0).to_csv(
        os.path.join(output_path, "summary_predictions_sample.csv"),
        index=False,
    )

    if print_test_predictions:
        print_predictions(
            model,
            test_loader,
            device,
            num_samples=50,
            save_path=os.path.join(output_path, "test_predictions.json"),
        )

    with open(os.path.join(output_path, "test_rows.json"), "w") as handle:
        json.dump(test_rows, handle, indent=2)

    dataset_stats = dict(trained_run["dataset_stats_base"])
    dataset_stats["test"] = _tensor_stats(test_y)
    dataset_stats["test_selection"] = test_selection_summary
    if sampling_info is not None:
        dataset_stats["sampling_info"] = sampling_info

    config = {
        "data_path": data_path,
        "held_out_source_path": held_out_source_path,
        "feature_names": feature_names,
        "model_type": model_type,
        "hidden_dims": hidden_dims if model_type.lower() == "mlp" else None,
        "dropout": dropout if model_type.lower() == "mlp" else None,
        "activation": activation if model_type.lower() == "mlp" else None,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "normalize": trained_run["normalize"],
        "seed": seed,
        "run_variant": evaluation_name,
        "held_out_dataset": held_out_dataset,
        "shuffle_train_labels": shuffle_train_labels,
        "use_class_weights": use_class_weights,
        "best_epoch": trained_run["best_epoch"],
        "best_val_loss": trained_run["best_val_loss"],
        "test_loss": test_metrics["loss"],
        "test_auroc": test_metrics.get("auroc", None),
        "test_ece": test_metrics["ece"],
        "dataset_stats": dataset_stats,
    }
    with open(os.path.join(output_path, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)

    print(
        f"Completed {evaluation_name}: "
        f"Brier={test_metrics['loss']:.4f}, "
        f"ECE={test_metrics['ece']:.4f}, "
        f"AUROC={test_metrics.get('auroc', 'N/A')}"
    )
    return output_path


def _run_single_training(
    *,
    feature_names: List[str],
    output_dir: str,
    run_variant: str,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    test_rows: List[Dict[str, Any]],
    train_selection_summary: Dict[str, Dict[str, int]],
    val_selection_summary: Dict[str, Dict[str, int]],
    test_selection_summary: Dict[str, Dict[str, int]],
    sampling_info: Optional[Dict[str, Any]],
    model_type: str,
    hidden_dims: Optional[List[int]],
    dropout: float,
    activation: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    normalize: bool,
    seed: int,
    device: torch.device,
    print_test_predictions: bool,
    use_class_weights: bool,
    shuffle_train_labels: bool,
    held_out_dataset: str,
    data_path: str,
    held_out_source_path: str,
) -> str:
    trained_run = _train_probe_run(
        feature_names=feature_names,
        output_dir=output_dir,
        run_variant=run_variant,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        train_selection_summary=train_selection_summary,
        val_selection_summary=val_selection_summary,
        sampling_info=sampling_info,
        model_type=model_type,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        normalize=normalize,
        seed=seed,
        device=device,
        use_class_weights=use_class_weights,
        shuffle_train_labels=shuffle_train_labels,
    )
    return _evaluate_trained_probe(
        trained_run=trained_run,
        eval_output_dir=trained_run["output_path"],
        test_X=test_X,
        test_y=test_y,
        test_rows=test_rows,
        test_selection_summary=test_selection_summary,
        feature_names=feature_names,
        model_type=model_type,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        print_test_predictions=print_test_predictions,
        use_class_weights=use_class_weights,
        shuffle_train_labels=shuffle_train_labels,
        evaluation_name=run_variant,
        data_path=data_path,
        held_out_source_path=held_out_source_path,
        held_out_dataset=held_out_dataset,
        sampling_info=sampling_info,
    )


def train_leave_one_out(
    *,
    data_path: str,
    model_name: str,
    held_out_dataset: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    model_type: str = "mlp",
    feature_name: str = DEFAULT_FEATURE_NAME,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    activation: str = "relu",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    normalize: bool = True,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    print_test_predictions: bool = True,
    use_class_weights: bool = False,
) -> Dict[str, str]:
    canonical_model = canonicalize_model_name(model_name)
    canonical_held_out = canonicalize_dataset_name(held_out_dataset)

    print("============================================================")
    print("Running leave-one-out generalization experiment")
    print("============================================================")
    print(f"Model: {canonical_model}")
    print(f"Held-out dataset: {canonical_held_out}")
    print(f"Data path: {data_path}")

    payload = _load_payload(data_path)
    feature_names = payload.get("feature_names", [feature_name])
    if feature_name not in feature_names:
        raise ValueError(
            f"Feature {feature_name!r} not present in generalization dataset {data_path}. "
            f"Available features: {feature_names}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = payload["metadata"]
    binarize_threshold = float(metadata["binarize_threshold"])
    per_dataset_metadata = metadata["per_dataset"]
    dataset_names = [canonicalize_dataset_name(name) for name in payload["dataset_names"]]
    split_labels = list(payload["split_labels"])

    source_train_mask = torch.tensor(
        [dataset != canonical_held_out and split == "train" for dataset, split in zip(dataset_names, split_labels)],
        dtype=torch.bool,
    )
    source_val_mask = torch.tensor(
        [dataset != canonical_held_out and split == "val" for dataset, split in zip(dataset_names, split_labels)],
        dtype=torch.bool,
    )

    source_train = _select_from_combined_payload(payload, source_train_mask, feature_name)
    source_val = _select_from_combined_payload(payload, source_val_mask, feature_name)

    if len(source_train["y"]) == 0 or len(source_val["y"]) == 0:
        raise ValueError(
            f"Invalid source-domain selection for held-out dataset {canonical_held_out}: "
            f"train={len(source_train['y'])}, val={len(source_val['y'])}."
        )

    held_out_test_bundle = _load_main_probe_test_split(
        dataset_name=canonical_held_out,
        per_dataset_metadata=per_dataset_metadata,
        feature_name=feature_name,
        binarize_threshold=binarize_threshold,
    )
    held_out_source_path = held_out_test_bundle["source_path"]
    held_out_source = load_selected_feature_source(held_out_source_path, feature_name)
    main_probe_split_indices = held_out_test_bundle["split_indices"]
    held_out_test = held_out_test_bundle["selection"]

    in_domain_train_sampling = _sample_index_pool(
        main_probe_split_indices["train"],
        desired_count=len(source_train["y"]),
        seed=seed + 1001,
    )
    in_domain_val_sampling = _sample_index_pool(
        main_probe_split_indices["val"],
        desired_count=len(source_val["y"]),
        seed=seed + 2002,
    )

    in_domain_train = _select_from_source_payload(
        held_out_source,
        indices=in_domain_train_sampling["indices"],
        binarize_threshold=binarize_threshold,
    )
    in_domain_val = _select_from_source_payload(
        held_out_source,
        indices=in_domain_val_sampling["indices"],
        binarize_threshold=binarize_threshold,
    )

    print(
        f"Held-out full main-probe test set size: {len(held_out_test['y'])} | "
        f"variance={held_out_test['y'].var(unbiased=False).item():.4f}"
    )
    if in_domain_train_sampling["with_replacement"] or in_domain_val_sampling["with_replacement"]:
        print(
            "In-domain matched sampling required replacement because the held-out dataset "
            "does not have enough original train/val samples to match the source-domain counts."
        )

    held_out_root = task_gen_feature_root(output_root, canonical_model, canonical_held_out, feature_name)
    held_out_root.mkdir(parents=True, exist_ok=True)

    source_train_summary = _selection_summary_from_mask(dataset_names, split_labels, source_train_mask)
    source_val_summary = _selection_summary_from_mask(dataset_names, split_labels, source_val_mask)
    held_out_test_summary = _single_dataset_summary(canonical_held_out, "test", len(held_out_test["y"]))

    in_domain_sampling_info = {
        "train": in_domain_train_sampling,
        "val": in_domain_val_sampling,
    }

    seed_values = seeds if seeds is not None and len(seeds) > 0 else [seed]
    print(f"Running seeds: {seed_values}")

    seed_variant_runs: Dict[int, Dict[str, str]] = {}
    for current_seed in seed_values:
        print("\n============================================================")
        print(f"Seed {current_seed}")
        print("============================================================")
        seed_root = held_out_root / "seed_runs" / f"seed_{current_seed}"
        run_dirs: Dict[str, str] = {}
        run_dirs["leave_one_out"] = _run_single_training(
            feature_names=[feature_name],
            output_dir=str(seed_root / "leave_one_out"),
            run_variant="leave_one_out",
            train_X=source_train["X"],
            train_y=source_train["y"],
            val_X=source_val["X"],
            val_y=source_val["y"],
            test_X=held_out_test["X"],
            test_y=held_out_test["y"],
            test_rows=held_out_test["rows"],
            train_selection_summary=source_train_summary,
            val_selection_summary=source_val_summary,
            test_selection_summary=held_out_test_summary,
            sampling_info=None,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            normalize=normalize,
            seed=current_seed,
            device=device,
            print_test_predictions=print_test_predictions,
            use_class_weights=use_class_weights,
            shuffle_train_labels=False,
            held_out_dataset=canonical_held_out,
            data_path=data_path,
            held_out_source_path=held_out_source_path,
        )
        run_dirs["shuffled_label_control"] = _run_single_training(
            feature_names=[feature_name],
            output_dir=str(seed_root / "shuffled_label_control"),
            run_variant="shuffled_label_control",
            train_X=source_train["X"],
            train_y=source_train["y"],
            val_X=source_val["X"],
            val_y=source_val["y"],
            test_X=held_out_test["X"],
            test_y=held_out_test["y"],
            test_rows=held_out_test["rows"],
            train_selection_summary=source_train_summary,
            val_selection_summary=source_val_summary,
            test_selection_summary=held_out_test_summary,
            sampling_info=None,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            normalize=normalize,
            seed=current_seed,
            device=device,
            print_test_predictions=print_test_predictions,
            use_class_weights=use_class_weights,
            shuffle_train_labels=True,
            held_out_dataset=canonical_held_out,
            data_path=data_path,
            held_out_source_path=held_out_source_path,
        )
        run_dirs["in_domain"] = _run_single_training(
            feature_names=[feature_name],
            output_dir=str(seed_root / "in_domain"),
            run_variant="in_domain",
            train_X=in_domain_train["X"],
            train_y=in_domain_train["y"],
            val_X=in_domain_val["X"],
            val_y=in_domain_val["y"],
            test_X=held_out_test["X"],
            test_y=held_out_test["y"],
            test_rows=held_out_test["rows"],
            train_selection_summary=_single_dataset_summary(canonical_held_out, "train", len(in_domain_train["y"])),
            val_selection_summary=_single_dataset_summary(canonical_held_out, "val", len(in_domain_val["y"])),
            test_selection_summary=held_out_test_summary,
            sampling_info=in_domain_sampling_info,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            normalize=normalize,
            seed=current_seed,
            device=device,
            print_test_predictions=print_test_predictions,
            use_class_weights=use_class_weights,
            shuffle_train_labels=False,
            held_out_dataset=canonical_held_out,
            data_path=data_path,
            held_out_source_path=held_out_source_path,
        )
        seed_variant_runs[int(current_seed)] = run_dirs
        save_comparison_summary(
            output_path=str(held_out_root / f"comparison_summary_seed_{current_seed}.json"),
            model_name=canonical_model,
            held_out_dataset=canonical_held_out,
            data_path=data_path,
            variants=run_dirs,
        )

    summary_path = held_out_root / "comparison_summary.json"
    aggregate_summary = save_aggregate_comparison_summary(
        output_path=str(summary_path),
        model_name=canonical_model,
        held_out_dataset=canonical_held_out,
        data_path=data_path,
        seed_variant_runs=seed_variant_runs,
    )

    print("\nSaved generalization comparison summary:")
    print(f"  {summary_path}")
    for variant_name, variant_summary in aggregate_summary["variants"].items():
        means = variant_summary["mean_metrics"]
        stds = variant_summary["std_metrics"]
        print(
            f"  {variant_name}: "
            f"Brier={_format_mean_std(means['test_loss'], stds['test_loss'])}, "
            f"ECE={_format_mean_std(means['test_ece'], stds['test_ece'])}, "
            f"AUROC={_format_mean_std(means['test_auroc'], stds['test_auroc'])}"
        )

    return {
        "held_out_root": str(held_out_root),
        "summary_path": str(summary_path),
    }


def train_all_datasets(
    *,
    data_path: str,
    model_name: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    model_type: str = "mlp",
    feature_name: str = DEFAULT_FEATURE_NAME,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    activation: str = "relu",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    normalize: bool = True,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    print_test_predictions: bool = True,
    use_class_weights: bool = False,
    include_shuffled_label_control: bool = True,
) -> Dict[str, str]:
    canonical_model = canonicalize_model_name(model_name)

    print("============================================================")
    print("Running all-datasets generalization experiment")
    print("============================================================")
    print(f"Model: {canonical_model}")
    print(f"Data path: {data_path}")

    payload = _load_payload(data_path)
    feature_names = payload.get("feature_names", [feature_name])
    if feature_name not in feature_names:
        raise ValueError(
            f"Feature {feature_name!r} not present in generalization dataset {data_path}. "
            f"Available features: {feature_names}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = payload["metadata"]
    binarize_threshold = float(metadata["binarize_threshold"])
    per_dataset_metadata = metadata["per_dataset"]

    dataset_names = [canonicalize_dataset_name(name) for name in payload["dataset_names"]]
    split_labels = list(payload["split_labels"])

    all_train_mask = torch.tensor([split == "train" for split in split_labels], dtype=torch.bool)
    all_val_mask = torch.tensor([split == "val" for split in split_labels], dtype=torch.bool)

    all_train = _select_from_combined_payload(payload, all_train_mask, feature_name)
    all_val = _select_from_combined_payload(payload, all_val_mask, feature_name)
    if len(all_train["y"]) == 0 or len(all_val["y"]) == 0:
        raise ValueError(
            f"Invalid all-datasets selection: train={len(all_train['y'])}, val={len(all_val['y'])}."
        )

    train_summary = _selection_summary_from_mask(dataset_names, split_labels, all_train_mask)
    val_summary = _selection_summary_from_mask(dataset_names, split_labels, all_val_mask)

    test_splits: Dict[str, Dict[str, Any]] = {}
    for dataset_name in per_dataset_metadata.keys():
        test_bundle = _load_main_probe_test_split(
            dataset_name=dataset_name,
            per_dataset_metadata=per_dataset_metadata,
            feature_name=feature_name,
            binarize_threshold=binarize_threshold,
        )
        test_splits[dataset_name] = test_bundle
        selection = test_bundle["selection"]
        print(
            f"Test set {dataset_name}: {len(selection['y'])} samples | "
            f"variance={selection['y'].var(unbiased=False).item():.4f}"
        )

    dataset_roots = {
        dataset_name: all_datasets_feature_root(output_root, canonical_model, dataset_name, feature_name)
        for dataset_name in test_splits.keys()
    }
    for dataset_root in dataset_roots.values():
        dataset_root.mkdir(parents=True, exist_ok=True)

    shared_train_root = all_datasets_shared_train_root(output_root, canonical_model, feature_name)
    shared_train_root.mkdir(parents=True, exist_ok=True)

    seed_values = seeds if seeds is not None and len(seeds) > 0 else [seed]
    print(f"Running seeds: {seed_values}")

    dataset_seed_variant_runs: Dict[str, Dict[int, Dict[str, str]]] = {
        dataset_name: {} for dataset_name in test_splits.keys()
    }
    for current_seed in seed_values:
        print("\n============================================================")
        print(f"Seed {current_seed}")
        print("============================================================")

        shared_seed_root = shared_train_root / "seed_runs" / f"seed_{current_seed}"
        trained_main = _train_probe_run(
            feature_names=[feature_name],
            output_dir=str(shared_seed_root / "train_all_datasets" / "shared_train"),
            run_variant="train_all_datasets",
            train_X=all_train["X"],
            train_y=all_train["y"],
            val_X=all_val["X"],
            val_y=all_val["y"],
            train_selection_summary=train_summary,
            val_selection_summary=val_summary,
            sampling_info=None,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            normalize=normalize,
            seed=current_seed,
            device=device,
            use_class_weights=use_class_weights,
            shuffle_train_labels=False,
        )

        trained_shuffle = None
        if include_shuffled_label_control:
            trained_shuffle = _train_probe_run(
                feature_names=[feature_name],
                output_dir=str(shared_seed_root / "shuffled_label_control" / "shared_train"),
                run_variant="shuffled_label_control",
                train_X=all_train["X"],
                train_y=all_train["y"],
                val_X=all_val["X"],
                val_y=all_val["y"],
                train_selection_summary=train_summary,
                val_selection_summary=val_summary,
                sampling_info=None,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                normalize=normalize,
                seed=current_seed,
                device=device,
                use_class_weights=use_class_weights,
                shuffle_train_labels=True,
            )

        for dataset_name, test_bundle in test_splits.items():
            dataset_root = dataset_roots[dataset_name]
            dataset_seed_root = dataset_root / "seed_runs" / f"seed_{current_seed}"
            dataset_run_dirs: Dict[str, str] = {}
            test_selection = test_bundle["selection"]
            test_summary = _single_dataset_summary(dataset_name, "test", len(test_selection["y"]))
            main_probe_split_indices = test_bundle["split_indices"]
            in_domain_source = load_selected_feature_source(test_bundle["source_path"], feature_name)
            own_train_count = int(train_summary.get(dataset_name, {}).get("train", 0))
            own_val_count = int(val_summary.get(dataset_name, {}).get("val", 0))
            in_domain_train_sampling = _sample_index_pool(
                main_probe_split_indices["train"],
                desired_count=len(all_train["y"]),
                seed=current_seed + 1001,
            )
            in_domain_val_sampling = _sample_index_pool(
                main_probe_split_indices["val"],
                desired_count=len(all_val["y"]),
                seed=current_seed + 2002,
            )
            in_domain_same_count_train_sampling = _sample_index_pool(
                main_probe_split_indices["train"],
                desired_count=own_train_count,
                seed=current_seed + 3003,
            )
            in_domain_same_count_val_sampling = _sample_index_pool(
                main_probe_split_indices["val"],
                desired_count=own_val_count,
                seed=current_seed + 4004,
            )
            in_domain_train = _select_from_source_payload(
                in_domain_source,
                indices=in_domain_train_sampling["indices"],
                binarize_threshold=binarize_threshold,
            )
            in_domain_val = _select_from_source_payload(
                in_domain_source,
                indices=in_domain_val_sampling["indices"],
                binarize_threshold=binarize_threshold,
            )
            in_domain_same_count_train = _select_from_source_payload(
                in_domain_source,
                indices=in_domain_same_count_train_sampling["indices"],
                binarize_threshold=binarize_threshold,
            )
            in_domain_same_count_val = _select_from_source_payload(
                in_domain_source,
                indices=in_domain_same_count_val_sampling["indices"],
                binarize_threshold=binarize_threshold,
            )
            in_domain_sampling_info = {
                "train": in_domain_train_sampling,
                "val": in_domain_val_sampling,
            }
            in_domain_same_count_sampling_info = {
                "train": in_domain_same_count_train_sampling,
                "val": in_domain_same_count_val_sampling,
            }

            dataset_run_dirs["train_all_datasets"] = _evaluate_trained_probe(
                trained_run=trained_main,
                eval_output_dir=str(dataset_seed_root / "train_all_datasets"),
                test_X=test_selection["X"],
                test_y=test_selection["y"],
                test_rows=test_selection["rows"],
                test_selection_summary=test_summary,
                feature_names=[feature_name],
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=current_seed,
                device=device,
                print_test_predictions=print_test_predictions,
                use_class_weights=use_class_weights,
                shuffle_train_labels=False,
                evaluation_name="train_all_datasets",
                data_path=data_path,
                held_out_source_path=test_bundle["source_path"],
                held_out_dataset=dataset_name,
            )

            trained_in_domain = _train_probe_run(
                feature_names=[feature_name],
                output_dir=str(dataset_seed_root / "in_domain" / "shared_train"),
                run_variant="in_domain",
                train_X=in_domain_train["X"],
                train_y=in_domain_train["y"],
                val_X=in_domain_val["X"],
                val_y=in_domain_val["y"],
                train_selection_summary=_single_dataset_summary(dataset_name, "train", len(in_domain_train["y"])),
                val_selection_summary=_single_dataset_summary(dataset_name, "val", len(in_domain_val["y"])),
                sampling_info=in_domain_sampling_info,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                normalize=normalize,
                seed=current_seed,
                device=device,
                use_class_weights=use_class_weights,
                shuffle_train_labels=False,
            )
            dataset_run_dirs["in_domain"] = _evaluate_trained_probe(
                trained_run=trained_in_domain,
                eval_output_dir=str(dataset_seed_root / "in_domain"),
                test_X=test_selection["X"],
                test_y=test_selection["y"],
                test_rows=test_selection["rows"],
                test_selection_summary=test_summary,
                feature_names=[feature_name],
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=current_seed,
                device=device,
                print_test_predictions=print_test_predictions,
                use_class_weights=use_class_weights,
                shuffle_train_labels=False,
                evaluation_name="in_domain",
                data_path=data_path,
                held_out_source_path=test_bundle["source_path"],
                held_out_dataset=dataset_name,
                sampling_info=in_domain_sampling_info,
            )

            trained_in_domain_same_count = _train_probe_run(
                feature_names=[feature_name],
                output_dir=str(dataset_seed_root / "in_domain_same_dataset_count" / "shared_train"),
                run_variant="in_domain_same_dataset_count",
                train_X=in_domain_same_count_train["X"],
                train_y=in_domain_same_count_train["y"],
                val_X=in_domain_same_count_val["X"],
                val_y=in_domain_same_count_val["y"],
                train_selection_summary=_single_dataset_summary(dataset_name, "train", len(in_domain_same_count_train["y"])),
                val_selection_summary=_single_dataset_summary(dataset_name, "val", len(in_domain_same_count_val["y"])),
                sampling_info=in_domain_same_count_sampling_info,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                normalize=normalize,
                seed=current_seed,
                device=device,
                use_class_weights=use_class_weights,
                shuffle_train_labels=False,
            )
            dataset_run_dirs["in_domain_same_dataset_count"] = _evaluate_trained_probe(
                trained_run=trained_in_domain_same_count,
                eval_output_dir=str(dataset_seed_root / "in_domain_same_dataset_count"),
                test_X=test_selection["X"],
                test_y=test_selection["y"],
                test_rows=test_selection["rows"],
                test_selection_summary=test_summary,
                feature_names=[feature_name],
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=current_seed,
                device=device,
                print_test_predictions=print_test_predictions,
                use_class_weights=use_class_weights,
                shuffle_train_labels=False,
                evaluation_name="in_domain_same_dataset_count",
                data_path=data_path,
                held_out_source_path=test_bundle["source_path"],
                held_out_dataset=dataset_name,
                sampling_info=in_domain_same_count_sampling_info,
            )

            if trained_shuffle is not None:
                dataset_run_dirs["shuffled_label_control"] = _evaluate_trained_probe(
                    trained_run=trained_shuffle,
                    eval_output_dir=str(dataset_seed_root / "shuffled_label_control"),
                    test_X=test_selection["X"],
                    test_y=test_selection["y"],
                    test_rows=test_selection["rows"],
                    test_selection_summary=test_summary,
                    feature_names=[feature_name],
                    model_type=model_type,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    activation=activation,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=current_seed,
                    device=device,
                    print_test_predictions=print_test_predictions,
                    use_class_weights=use_class_weights,
                    shuffle_train_labels=True,
                    evaluation_name="shuffled_label_control",
                    data_path=data_path,
                    held_out_source_path=test_bundle["source_path"],
                    held_out_dataset=dataset_name,
                )

            dataset_seed_variant_runs[dataset_name][int(current_seed)] = dataset_run_dirs
            save_comparison_summary(
                output_path=str(dataset_root / f"comparison_summary_seed_{current_seed}.json"),
                model_name=canonical_model,
                held_out_dataset=dataset_name,
                data_path=data_path,
                variants=dataset_run_dirs,
            )

    summary_paths: Dict[str, str] = {}
    print("\nSaved all-datasets comparison summaries:")
    for dataset_name, dataset_root in dataset_roots.items():
        summary_path = dataset_root / "comparison_summary.json"
        summary_paths[dataset_name] = str(summary_path)
        aggregate_summary = save_aggregate_comparison_summary(
            output_path=str(summary_path),
            model_name=canonical_model,
            held_out_dataset=dataset_name,
            data_path=data_path,
            seed_variant_runs=dataset_seed_variant_runs[dataset_name],
        )
        print(f"  {dataset_name}: {summary_path}")
        for variant_name, variant_summary in aggregate_summary["variants"].items():
            means = variant_summary["mean_metrics"]
            stds = variant_summary["std_metrics"]
            print(
                f"    {variant_name}: "
                f"Brier={_format_mean_std(means['test_loss'], stds['test_loss'])}, "
                f"ECE={_format_mean_std(means['test_ece'], stds['test_ece'])}, "
                f"AUROC={_format_mean_std(means['test_auroc'], stds['test_auroc'])}"
            )

    model_results_root = next(iter(dataset_roots.values())).parent.parent if dataset_roots else Path(output_root)
    return {
        "all_datasets_root": str(model_results_root),
        "summary_paths": json.dumps(summary_paths),
    }


def train_model_generalization(
    *,
    source_data_path: str,
    target_data_path: str,
    source_model_name: str,
    target_model_name: str,
    dataset_name: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    model_type: str = "mlp",
    feature_name: str = DEFAULT_FEATURE_NAME,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    activation: str = "relu",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    normalize: bool = True,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    print_test_predictions: bool = True,
    use_class_weights: bool = False,
    include_shuffled_label_control: bool = True,
) -> Dict[str, str]:
    canonical_source_model = canonicalize_model_name(source_model_name)
    canonical_target_model = canonicalize_model_name(target_model_name)
    canonical_dataset = canonicalize_dataset_name(dataset_name)
    direction_name = f"{canonical_source_model}_to_{canonical_target_model}"

    print("============================================================")
    print("Running cross-model generalization experiment")
    print("============================================================")
    print(f"Source model: {canonical_source_model}")
    print(f"Target model: {canonical_target_model}")
    print(f"Dataset: {canonical_dataset}")
    print(f"Source data path: {source_data_path}")
    print(f"Target data path: {target_data_path}")

    source_payload = _load_payload(source_data_path)
    target_payload = _load_payload(target_data_path)

    source_feature_names = source_payload.get("feature_names", [feature_name])
    target_feature_names = target_payload.get("feature_names", [feature_name])
    if feature_name not in source_feature_names:
        reason = (
            f"Feature {feature_name!r} not present in source generalization dataset {source_data_path}. "
            f"Available features: {source_feature_names}"
        )
        print(f"[skip] {direction_name} / {canonical_dataset}: {reason}")
        return {"skipped": "true", "reason": reason}
    if feature_name not in target_feature_names:
        reason = (
            f"Feature {feature_name!r} not present in target generalization dataset {target_data_path}. "
            f"Available features: {target_feature_names}"
        )
        print(f"[skip] {direction_name} / {canonical_dataset}: {reason}")
        return {"skipped": "true", "reason": reason}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_dataset_names = [canonicalize_dataset_name(name) for name in source_payload["dataset_names"]]
    source_split_labels = list(source_payload["split_labels"])
    source_train_mask = torch.tensor(
        [
            dataset == canonical_dataset and split == "train"
            for dataset, split in zip(source_dataset_names, source_split_labels)
        ],
        dtype=torch.bool,
    )
    source_val_mask = torch.tensor(
        [
            dataset == canonical_dataset and split == "val"
            for dataset, split in zip(source_dataset_names, source_split_labels)
        ],
        dtype=torch.bool,
    )

    source_train = _select_from_combined_payload(source_payload, source_train_mask, feature_name)
    source_val = _select_from_combined_payload(source_payload, source_val_mask, feature_name)
    if len(source_train["y"]) == 0 or len(source_val["y"]) == 0:
        reason = (
            f"Invalid source selection for dataset {canonical_dataset} in {source_data_path}: "
            f"train={len(source_train['y'])}, val={len(source_val['y'])}."
        )
        print(f"[skip] {direction_name} / {canonical_dataset}: {reason}")
        return {"skipped": "true", "reason": reason}

    target_metadata = target_payload["metadata"]
    target_binarize_threshold = float(target_metadata["binarize_threshold"])
    target_test_bundle = _load_main_probe_test_split(
        dataset_name=canonical_dataset,
        per_dataset_metadata=target_metadata["per_dataset"],
        feature_name=feature_name,
        binarize_threshold=target_binarize_threshold,
    )
    target_test = target_test_bundle["selection"]
    if len(target_test["y"]) == 0:
        reason = "Target test split is empty."
        print(f"[skip] {direction_name} / {canonical_dataset}: {reason}")
        return {"skipped": "true", "reason": reason}
    if not _has_two_classes(target_test["y"]):
        reason = (
            "Target test split does not contain both binary classes, so AUROC is undefined "
            f"(n={len(target_test['y'])}, mean={target_test['y'].mean().item():.4f})."
        )
        print(f"[skip] {direction_name} / {canonical_dataset}: {reason}")
        return {"skipped": "true", "reason": reason}
    print(
        f"Target test set size: {len(target_test['y'])} | "
        f"variance={target_test['y'].var(unbiased=False).item():.4f}"
    )
    model_gen_root = model_gen_feature_root(output_root, direction_name, canonical_dataset, feature_name)
    model_gen_root.mkdir(parents=True, exist_ok=True)

    source_train_summary = _selection_summary_from_mask(source_dataset_names, source_split_labels, source_train_mask)
    source_val_summary = _selection_summary_from_mask(source_dataset_names, source_split_labels, source_val_mask)
    target_test_summary = _single_dataset_summary(canonical_dataset, "test", len(target_test["y"]))

    seed_values = seeds if seeds is not None and len(seeds) > 0 else [seed]
    print(f"Running seeds: {seed_values}")

    seed_variant_runs: Dict[int, Dict[str, str]] = {}
    for current_seed in seed_values:
        print("\n============================================================")
        print(f"Seed {current_seed}")
        print("============================================================")

        seed_root = model_gen_root / "seed_runs" / f"seed_{current_seed}"
        run_dirs: Dict[str, str] = {}
        run_dirs["model_gen"] = _run_single_training(
            feature_names=[feature_name],
            output_dir=str(seed_root / "model_gen"),
            run_variant="model_gen",
            train_X=source_train["X"],
            train_y=source_train["y"],
            val_X=source_val["X"],
            val_y=source_val["y"],
            test_X=target_test["X"],
            test_y=target_test["y"],
            test_rows=target_test["rows"],
            train_selection_summary=source_train_summary,
            val_selection_summary=source_val_summary,
            test_selection_summary=target_test_summary,
            sampling_info=None,
            model_type=model_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            normalize=normalize,
            seed=current_seed,
            device=device,
            print_test_predictions=print_test_predictions,
            use_class_weights=use_class_weights,
            shuffle_train_labels=False,
            held_out_dataset=canonical_dataset,
            data_path=source_data_path,
            held_out_source_path=target_test_bundle["source_path"],
        )
        if include_shuffled_label_control:
            run_dirs["shuffled_label_control"] = _run_single_training(
                feature_names=[feature_name],
                output_dir=str(seed_root / "shuffled_label_control"),
                run_variant="shuffled_label_control",
                train_X=source_train["X"],
                train_y=source_train["y"],
                val_X=source_val["X"],
                val_y=source_val["y"],
                test_X=target_test["X"],
                test_y=target_test["y"],
                test_rows=target_test["rows"],
                train_selection_summary=source_train_summary,
                val_selection_summary=source_val_summary,
                test_selection_summary=target_test_summary,
                sampling_info=None,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=activation,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                normalize=normalize,
                seed=current_seed,
                device=device,
                print_test_predictions=print_test_predictions,
                use_class_weights=use_class_weights,
                shuffle_train_labels=True,
                held_out_dataset=canonical_dataset,
                data_path=source_data_path,
                held_out_source_path=target_test_bundle["source_path"],
            )

        seed_variant_runs[int(current_seed)] = run_dirs
        save_comparison_summary(
            output_path=str(model_gen_root / f"comparison_summary_seed_{current_seed}.json"),
            model_name=direction_name,
            held_out_dataset=canonical_dataset,
            data_path=source_data_path,
            variants=run_dirs,
        )

    summary_path = model_gen_root / "comparison_summary.json"
    aggregate_summary = save_aggregate_comparison_summary(
        output_path=str(summary_path),
        model_name=direction_name,
        held_out_dataset=canonical_dataset,
        data_path=source_data_path,
        seed_variant_runs=seed_variant_runs,
    )

    print("\nSaved cross-model comparison summary:")
    print(f"  {summary_path}")
    for variant_name, variant_summary in aggregate_summary["variants"].items():
        means = variant_summary["mean_metrics"]
        stds = variant_summary["std_metrics"]
        print(
            f"  {variant_name}: "
            f"Brier={_format_mean_std(means['test_loss'], stds['test_loss'])}, "
            f"ECE={_format_mean_std(means['test_ece'], stds['test_ece'])}, "
            f"AUROC={_format_mean_std(means['test_auroc'], stds['test_auroc'])}"
        )

    return {
        "model_gen_root": str(model_gen_root),
        "summary_path": str(summary_path),
    }
