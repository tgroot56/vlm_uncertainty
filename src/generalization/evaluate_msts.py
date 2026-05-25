from __future__ import annotations

import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.probe.data import build_feature_slice_map
from src.probe.metrics import compute_auroc, compute_ece
from src.probe.models import LinearProbe, MLPProbe


DEFAULT_MSTS_RUN_DIR = (
    "/projects/prjs2014/msts/llava-hf/llava-1.5-7b-hf/run_14423ff29d"
)
DEFAULT_OUTPUT_ROOT = "/projects/prjs2014/tgroot/probe_results/msts_eval/llava"
DEFAULT_MODEL_TYPES = ("linear", "mlp")


FEATURE_GROUPS: Dict[str, Sequence[str]] = {
    "lm_visual_mean": (
        "lm_visual_mean_layer_16",
        "lm_visual_mean_layer_-1",
    ),
    "lm_answer_mean_lasttok": (
        "lm_answer_mean_layer_16",
        "lm_answer_mean_layer_-1",
        "lm_answer_lasttok_layer_16",
        "lm_answer_lasttok_layer_-1",
    ),
    "lm_question_mean_lasttok": (
        "lm_question_mean_layer_16",
        "lm_question_mean_layer_-1",
        "lm_question_lasttok_layer_16",
        "lm_question_lasttok_layer_-1",
    ),
    "lm_fullspan_lasttok": (
        "lm_fullspan_lasttok_layer_16",
        "lm_fullspan_lasttok_layer_-1",
    ),
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    checkpoint_root: str


SOURCE_SPECS: Sequence[SourceSpec] = (
    SourceSpec("vqa-v2", "probe_results/llava/vqa-v2"),
    SourceSpec("coco-qa-vi", "probe_results/llava/coco-qa-vi"),
    SourceSpec("imagenet-r", "probe_results/llava/imagenet-r"),
    SourceSpec("pope_random", "probe_results/llava/pope/random"),
    SourceSpec("pope_adversarial", "probe_results/llava/pope/adversarial"),
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _normalize_binary_label(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("Encountered missing manual_label in MSTS annotation CSV.")
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "1.0", "true", "yes", "safe", "correct"}:
            return 1.0
        if value in {"0", "0.0", "false", "no", "unsafe", "incorrect"}:
            return 0.0
    numeric = float(value)
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"Expected binary MSTS manual_label values, got {value!r}.")
    return numeric


def build_annotated_msts_dataset(
    *,
    supervision_path: str,
    annotation_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Create an annotated copy of the MSTS supervision dataset with y from CSV."""
    supervision = torch.load(supervision_path, map_location="cpu", weights_only=False)
    annotations = pd.read_csv(annotation_path)

    rows = supervision.get("rows")
    if rows is None:
        raise RuntimeError(f"Missing 'rows' in {supervision_path}.")
    if len(rows) != len(annotations):
        raise RuntimeError(
            f"MSTS supervision rows ({len(rows)}) and annotations ({len(annotations)}) differ."
        )

    if "manual_label" not in annotations.columns:
        raise RuntimeError(f"Missing manual_label column in {annotation_path}.")

    labels = torch.tensor(
        [_normalize_binary_label(value) for value in annotations["manual_label"].tolist()],
        dtype=torch.float32,
    )

    annotated = copy.deepcopy(supervision)
    annotated["y"] = labels
    for idx, row in enumerate(annotated["rows"]):
        row["manual_label"] = float(labels[idx].item())
        if "manual_notes" in annotations.columns and not pd.isna(annotations.loc[idx, "manual_notes"]):
            row["manual_notes"] = str(annotations.loc[idx, "manual_notes"])

    metadata = dict(annotated.get("metadata", {}))
    metadata.update(
        {
            "label_placeholder": False,
            "label_source": annotation_path,
            "label_mean": float(labels.mean().item()),
            "label_variance": float(labels.var(unbiased=False).item()),
            "num_annotated_labels": int(labels.numel()),
            "manual_label_counts": {
                "0": int((labels == 0).sum().item()),
                "1": int((labels == 1).sum().item()),
            },
        }
    )
    annotated["metadata"] = metadata

    target_path = output_path or str(Path(supervision_path).with_name("supervision_dataset_annotated.pt"))
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(annotated, target_path)
    return target_path


def _load_model_for_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _read_json(checkpoint_path.parent / "config.json")
    input_dim = int(checkpoint["input_dim"])
    model_type = str(config.get("model_type") or checkpoint_path.parent.parent.name).lower()

    if model_type == "linear":
        model = LinearProbe(input_dim=input_dim)
    elif model_type == "mlp":
        model = MLPProbe(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims") or [256, 128],
            dropout=float(config.get("dropout", 0.2)),
            activation=str(config.get("activation", "relu")),
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type!r} for {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, config


def _select_features(payload: Dict[str, Any], feature_names: Sequence[str]) -> torch.Tensor:
    x_all = payload["X"].float()
    all_feature_names = payload["feature_names"]
    feature_dims = payload.get("feature_dims")
    name_to_slice = build_feature_slice_map(all_feature_names, feature_dims, x_all.shape[1])

    missing = [name for name in feature_names if name not in name_to_slice]
    if missing:
        raise ValueError(
            f"Target dataset missing features {missing}. Available: {list(name_to_slice.keys())}"
        )

    return torch.cat([x_all[:, name_to_slice[name]] for name in feature_names], dim=1)


def _apply_checkpoint_normalization(
    x: torch.Tensor,
    checkpoint: Dict[str, Any],
) -> torch.Tensor:
    if not bool(checkpoint.get("normalize", False)):
        return x
    mean = checkpoint.get("mean")
    std = checkpoint.get("std")
    if mean is None or std is None:
        raise RuntimeError("Checkpoint requested normalization but has no mean/std.")
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
    return (x - mean_t) / std_t


@torch.no_grad()
def _predict_all(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    preds: List[torch.Tensor] = []
    for (features,) in loader:
        preds.append(model(features.to(device)).detach().cpu().float())
    return torch.cat(preds, dim=0).view(-1)


def _metrics(predictions: torch.Tensor, labels: torch.Tensor, num_bins: int) -> Dict[str, Any]:
    predictions = predictions.float().view(-1)
    labels = labels.float().view(-1)
    brier = torch.mean((predictions - labels) ** 2).item()
    ece = compute_ece(predictions, labels, num_bins=num_bins)
    auroc = compute_auroc(predictions, labels)
    return {
        "brier": float(brier),
        "ece": float(ece["ece"]),
        "auroc": float(auroc),
        "mean_prediction": float(predictions.mean().item()),
        "mean_label": float(labels.mean().item()),
        "num_samples": int(labels.numel()),
        "num_positive": int((labels == 1).sum().item()),
        "num_negative": int((labels == 0).sum().item()),
    }


def _checkpoint_path(
    *,
    source_root: Path,
    model_type: str,
    feature_name: str,
) -> Path:
    return source_root / model_type / feature_name / "best_model.pt"


def _iter_eval_specs(
    *,
    sources: Sequence[SourceSpec],
    model_types: Sequence[str],
    feature_groups: Dict[str, Sequence[str]],
) -> Iterable[Dict[str, str]]:
    for source in sources:
        source_root = Path(source.checkpoint_root)
        for model_type in model_types:
            for group_name, feature_names in feature_groups.items():
                for feature_name in feature_names:
                    yield {
                        "source_dataset": source.name,
                        "source_root": str(source_root),
                        "model_type": model_type,
                        "feature_group": group_name,
                        "feature_name": feature_name,
                        "checkpoint_path": str(
                            _checkpoint_path(
                                source_root=source_root,
                                model_type=model_type,
                                feature_name=feature_name,
                            )
                        ),
                    }


def evaluate_msts_probes(
    *,
    msts_run_dir: str = DEFAULT_MSTS_RUN_DIR,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    model_types: Sequence[str] = DEFAULT_MODEL_TYPES,
    batch_size: int = 64,
    device: Optional[str] = None,
    num_ece_bins: int = 10,
) -> Dict[str, Any]:
    run_dir = Path(msts_run_dir)
    supervision_path = run_dir / "supervision_dataset.pt"
    annotation_path = run_dir / "annotation_template.csv"
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "supervision_dataset_annotated.pt"

    annotated_dataset_path = build_annotated_msts_dataset(
        supervision_path=str(supervision_path),
        annotation_path=str(annotation_path),
        output_path=str(annotated_path),
    )
    payload = torch.load(annotated_dataset_path, map_location="cpu", weights_only=False)
    labels = payload["y"].float()

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    summary_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for spec in _iter_eval_specs(
        sources=SOURCE_SPECS,
        model_types=model_types,
        feature_groups=FEATURE_GROUPS,
    ):
        checkpoint_path = Path(spec["checkpoint_path"])
        if not checkpoint_path.exists():
            skipped_rows.append({**spec, "reason": "missing_checkpoint"})
            continue

        model, checkpoint, config = _load_model_for_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device_t,
        )
        checkpoint_features = checkpoint.get("feature_names") or [spec["feature_name"]]
        x = _select_features(payload, checkpoint_features)
        x = _apply_checkpoint_normalization(x, checkpoint)
        predictions = _predict_all(
            model=model,
            x=x,
            batch_size=batch_size,
            device=device_t,
        )
        metrics = _metrics(predictions, labels, num_bins=num_ece_bins)

        result_dir = (
            output_dir
            / spec["source_dataset"]
            / spec["model_type"]
            / spec["feature_group"]
            / spec["feature_name"]
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        row = {
            **spec,
            **metrics,
            "checkpoint_feature_names": checkpoint_features,
            "target_dataset_path": annotated_dataset_path,
            "target_annotation_path": str(annotation_path),
            "result_dir": str(result_dir),
            "normalize": bool(checkpoint.get("normalize", False)),
            "config_model_type": config.get("model_type"),
        }
        summary_rows.append(row)

        _write_json(result_dir / "metrics.json", row)
        pd.DataFrame(
            {
                "idx": list(range(len(labels))),
                "prediction": predictions.numpy(),
                "label": labels.numpy(),
            }
        ).to_csv(result_dir / "predictions.csv", index=False)

    summary_path = output_dir / "msts_probe_eval_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    skipped_path = output_dir / "msts_probe_eval_skipped.csv"
    pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)

    _write_json(
        output_dir / "msts_probe_eval_metadata.json",
        {
            "msts_run_dir": str(run_dir),
            "annotated_dataset_path": annotated_dataset_path,
            "output_root": str(output_dir),
            "target_num_samples": int(len(labels)),
            "target_num_positive": int(labels.sum().item()),
            "target_num_negative": int((labels == 0).sum().item()),
            "num_probe_evaluations": len(summary_rows),
            "num_skipped": len(skipped_rows),
            "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
            "source_datasets": [source.name for source in SOURCE_SPECS],
            "model_types": list(model_types),
            "batch_size": batch_size,
            "num_ece_bins": num_ece_bins,
        },
    )

    return {
        "annotated_dataset_path": annotated_dataset_path,
        "summary_path": str(summary_path),
        "skipped_path": str(skipped_path),
        "target_num_samples": int(len(labels)),
        "target_num_positive": int(labels.sum().item()),
        "target_num_negative": int((labels == 0).sum().item()),
        "num_probe_evaluations": len(summary_rows),
        "num_skipped": len(skipped_rows),
    }
