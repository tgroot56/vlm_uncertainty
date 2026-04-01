from __future__ import annotations

import copy
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from .paths import (
    DEFAULT_OUTPUT_ROOT,
    generalization_dataset_dir,
    generalization_dataset_metadata_path,
    generalization_dataset_path,
)
DEFAULT_FEATURE_NAME = "lm_answer_mean_layer_16"
DEFAULT_FEATURE_NAMES = (
    "lm_answer_mean_layer_16",
    "answer_gen_entropy_mean",
    "lm_fullspan_lasttok_layer_16",
)
DEFAULT_MAX_PER_DATASET = 3000
DEFAULT_DATASETS = ("vqa-v2", "coco-qa-vi", "imagenet-r", "pope/random")
DEFAULT_SOURCE_TRAIN_FRACTION = 0.85
DEFAULT_SOURCE_VAL_FRACTION = 0.15
DEFAULT_MAIN_PROBE_SPLIT_SEED = 42
DEFAULT_MAIN_PROBE_TRAIN_FRACTION = 0.7
DEFAULT_MAIN_PROBE_VAL_FRACTION = 0.15


@dataclass(frozen=True)
class ModelSourceSpec:
    model_name: str
    model_id: str
    source_patterns: Dict[str, List[str]]


MODEL_SPECS: Dict[str, ModelSourceSpec] = {
    "qwen": ModelSourceSpec(
        model_name="qwen",
        model_id="Qwen/Qwen3.5-9B",
        source_patterns={
            "vqa-v2": [
                "/projects/prjs2014/qwen/vqa-v2/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/qwen/vqa-v2/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
            ],
            "coco-qa-vi": [
                "/projects/prjs2014/qwen/coco-qa-vi/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/qwen/coco-qa-vi/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
            ],
            "imagenet-r": [
                "/projects/prjs2014/qwen/imagenet-r/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/qwen/imagenet-r/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
            ],
            "pope/random": [
                "/projects/prjs2014/qwen/pope/random/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/qwen/pope/random/Qwen/Qwen3.5-9B/*/supervision_dataset.pt",
            ],
        },
    ),
    "llava": ModelSourceSpec(
        model_name="llava",
        model_id="llava-hf/llava-1.5-7b-hf",
        source_patterns={
            "vqa-v2": [
                "/projects/prjs2014/vqa-v2/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/vqa-v2/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
            ],
            "coco-qa-vi": [
                "/projects/prjs2014/coco-qa-vi/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/coco-qa-vi/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
            ],
            "imagenet-r": [
                "/projects/prjs2014/imagenet_r_16mc/imagenet-r/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/imagenet_r_16mc/imagenet-r/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
            ],
            "pope/random": [
                "/projects/prjs2014/pope/random/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/pope/random/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
                "/scratch-shared/tgroot/supervised_datasets/pope/llava-hf/llava-1.5-7b-hf/*/supervision_dataset.pt",
            ],
        },
    ),
}


def canonicalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("_", "-")
    if key in {"qwen", "qwen-3.5-9b", "qwen3.5-9b", "qwen/qwen3.5-9b"}:
        return "qwen"
    if key in {"llava", "llava-1.5-7b", "llava1.5-7b", "llava-hf/llava-1.5-7b-hf"}:
        return "llava"
    raise ValueError(
        f"Unsupported model name: {model_name!r}. Use one of: qwen, qwen-3.5-9B, llava, llava-1.5-7B."
    )


def canonicalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower().replace("_", "-")
    if key in {"vqa-v2", "vqav2"}:
        return "vqa-v2"
    if key in {"coco-qa-vi", "cocoqavi"}:
        return "coco-qa-vi"
    if key in {"imagenet-r", "imagenetr"}:
        return "imagenet-r"
    if key in {"pope/random", "pope-random", "pope/random/", "pope"}:
        return "pope/random"
    raise ValueError(
        f"Unsupported dataset name: {dataset_name!r}. Use one of: {', '.join(DEFAULT_DATASETS)}."
    )


def _infer_feature_dim(feature_name: str) -> int:
    if feature_name.startswith("vision_"):
        return 1024
    if feature_name.startswith("visual_merged_"):
        return 4096
    if feature_name.startswith("lm_"):
        return 4096
    if feature_name.startswith("answer_gen_"):
        return 1
    raise ValueError(f"Could not infer feature dimension for {feature_name!r}.")


def _build_feature_slices(
    feature_names: List[str],
    feature_dims: Optional[List[int]],
    total_dim: int,
) -> Dict[str, slice]:
    if feature_dims is None:
        dims = [_infer_feature_dim(name) for name in feature_names]
    else:
        dims = feature_dims
        if len(dims) != len(feature_names):
            raise ValueError(
                f"feature_dims length mismatch: {len(dims)} dims for {len(feature_names)} features."
            )

    name_to_slice: Dict[str, slice] = {}
    offset = 0
    for name, dim in zip(feature_names, dims):
        name_to_slice[name] = slice(offset, offset + dim)
        offset += dim

    if offset != total_dim:
        raise ValueError(
            f"Feature dimension reconstruction mismatch: inferred {offset} columns but X has {total_dim}."
        )
    return name_to_slice


def get_feature_slices(
    feature_names: List[str],
    feature_dims: Optional[List[int]],
    total_dim: int,
) -> Dict[str, slice]:
    return _build_feature_slices(feature_names, feature_dims, total_dim)


def _find_latest_matching_file(patterns: Iterable[str]) -> str:
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    unique_candidates = sorted(set(candidates), key=lambda path: os.path.getmtime(path), reverse=True)
    if not unique_candidates:
        raise FileNotFoundError(
            "Could not find a matching supervision dataset for any of these patterns:\n"
            + "\n".join(f"  - {pattern}" for pattern in patterns)
        )
    return unique_candidates[0]


def resolve_source_paths(
    model_name: str,
    overrides: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, str]:
    canonical_model = canonicalize_model_name(model_name)
    spec = MODEL_SPECS[canonical_model]
    overrides = overrides or {}

    resolved: Dict[str, str] = {}
    for dataset_name in DEFAULT_DATASETS:
        override_path = overrides.get(dataset_name)
        if override_path:
            resolved[dataset_name] = override_path
            continue
        resolved[dataset_name] = _find_latest_matching_file(spec.source_patterns[dataset_name])
    return resolved


def binarize_labels(y: torch.Tensor, threshold: float) -> torch.Tensor:
    return (y >= threshold).float()


def normalize_feature_name_list(feature_names: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for feature_name in feature_names:
        name = str(feature_name).strip()
        if not name or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    if not ordered:
        raise ValueError("At least one non-empty feature name must be provided.")
    return ordered


def compute_main_probe_split_indices(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    if not (0.0 < train_fraction < 1.0 and 0.0 <= val_fraction < 1.0):
        raise ValueError("train_fraction and val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be strictly smaller than 1.")

    train_count = int(num_samples * train_fraction)
    val_count = int(num_samples * val_fraction)
    test_count = num_samples - train_count - val_count

    if test_count <= 0:
        raise ValueError("Split fractions leave no test samples.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    return {
        "train": perm[:train_count],
        "val": perm[train_count:train_count + val_count],
        "test": perm[train_count + val_count:],
    }


def _compute_source_train_val_labels(
    num_samples: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> List[str]:
    if not (0.0 < train_fraction < 1.0 and 0.0 < val_fraction < 1.0):
        raise ValueError("Source train_fraction and val_fraction must both be between 0 and 1.")
    total = train_fraction + val_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Source train_fraction + val_fraction must equal 1.0 when no source test split is used; got {total:.6f}."
        )

    train_count = int(num_samples * train_fraction)
    labels = [""] * num_samples
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator).tolist()

    for idx in perm[:train_count]:
        labels[idx] = "train"
    for idx in perm[train_count:]:
        labels[idx] = "val"

    if any(label == "" for label in labels):
        raise RuntimeError("Source split assignment left unlabeled samples.")
    return labels


def load_selected_feature_source(
    source_path: str,
    feature_name: str,
) -> Dict[str, Any]:
    return load_selected_features_source(source_path, [feature_name])


def load_selected_features_source(
    source_path: str,
    feature_names: List[str],
) -> Dict[str, Any]:
    payload = torch.load(source_path, map_location="cpu", weights_only=False)

    X_all = payload["X"].float()
    y_raw = payload["y"].float()
    available_feature_names = payload["feature_names"]
    feature_dims = payload.get("feature_dims")
    rows = payload.get("rows") or [None] * len(y_raw)
    metadata = payload.get("metadata") or {}

    requested_feature_names = normalize_feature_name_list(feature_names)
    name_to_slice = _build_feature_slices(available_feature_names, feature_dims, X_all.shape[1])
    missing = [feature_name for feature_name in requested_feature_names if feature_name not in name_to_slice]
    if missing:
        raise ValueError(
            f"Feature(s) {missing!r} not present in {source_path}. "
            f"Available features: {available_feature_names}"
        )

    X_parts = []
    selected_feature_dims = []
    for feature_name in requested_feature_names:
        feature_slice = name_to_slice[feature_name]
        X_parts.append(X_all[:, feature_slice].contiguous())
        selected_feature_dims.append(int(feature_slice.stop - feature_slice.start))
    X = torch.cat(X_parts, dim=1).contiguous()

    return {
        "X": X,
        "y_raw": y_raw,
        "rows": rows,
        "metadata": metadata,
        "feature_names": requested_feature_names,
        "feature_dims": selected_feature_dims,
        "feature_dim": int(X.shape[1]),
    }


def build_generalization_dataset(
    *,
    model_name: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    feature_name: str = DEFAULT_FEATURE_NAME,
    feature_names: Optional[List[str]] = None,
    max_per_dataset: int = DEFAULT_MAX_PER_DATASET,
    sampling_seed: int = 42,
    split_seed: int = 42,
    binarize_threshold: float = 1.0,
    train_fraction: float = DEFAULT_SOURCE_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_SOURCE_VAL_FRACTION,
    main_probe_split_seed: int = DEFAULT_MAIN_PROBE_SPLIT_SEED,
    main_probe_train_fraction: float = DEFAULT_MAIN_PROBE_TRAIN_FRACTION,
    main_probe_val_fraction: float = DEFAULT_MAIN_PROBE_VAL_FRACTION,
    source_overrides: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, str]:
    canonical_model = canonicalize_model_name(model_name)
    spec = MODEL_SPECS[canonical_model]
    resolved_paths = resolve_source_paths(canonical_model, source_overrides)
    selected_feature_names = normalize_feature_name_list(
        feature_names if feature_names is not None else [feature_name]
    )

    print("============================================================")
    print("Building generalization supervision dataset")
    print("============================================================")
    print(f"Model: {canonical_model} ({spec.model_id})")
    print(f"Features: {selected_feature_names}")
    print(f"Max samples per dataset: {max_per_dataset}")
    print(f"Sampling seed: {sampling_seed}")
    print(f"Source split seed: {split_seed}")
    print(f"Source train/val split: {train_fraction:.2f}/{val_fraction:.2f}")
    print(
        f"Main-probe held-out test split: "
        f"{main_probe_train_fraction:.2f}/{main_probe_val_fraction:.2f}/"
        f"{1.0 - main_probe_train_fraction - main_probe_val_fraction:.2f} "
        f"(seed={main_probe_split_seed})"
    )

    model_output_dir = generalization_dataset_dir(output_root, canonical_model)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = generalization_dataset_path(output_root, canonical_model)
    metadata_path = generalization_dataset_metadata_path(output_root, canonical_model)

    X_chunks: List[torch.Tensor] = []
    y_chunks: List[torch.Tensor] = []
    rows_out: List[Dict[str, Any]] = []
    dataset_names_out: List[str] = []
    split_labels_out: List[str] = []
    source_row_indices_out: List[int] = []

    total_feature_dim: Optional[int] = None
    selected_feature_dims: Optional[List[int]] = None
    per_dataset_metadata: Dict[str, Dict[str, Any]] = {}

    for dataset_rank, dataset_name in enumerate(DEFAULT_DATASETS):
        source_path = resolved_paths[dataset_name]
        print(f"\nLoading source dataset: {dataset_name}")
        print(f"  Source path: {source_path}")
        loaded = load_selected_features_source(source_path, selected_feature_names)
        X = loaded["X"]
        y_raw = loaded["y_raw"]
        rows = loaded["rows"]
        source_metadata = loaded["metadata"]

        if total_feature_dim is None:
            total_feature_dim = loaded["feature_dim"]
            selected_feature_dims = list(loaded["feature_dims"])
        elif total_feature_dim != loaded["feature_dim"]:
            raise ValueError(
                f"Feature dimension mismatch for {dataset_name}: expected {total_feature_dim}, got {loaded['feature_dim']}."
            )
        elif selected_feature_dims != list(loaded["feature_dims"]):
            raise ValueError(
                f"Feature dimension list mismatch for {dataset_name}: "
                f"expected {selected_feature_dims}, got {loaded['feature_dims']}."
            )

        num_available = X.shape[0]
        num_selected = min(max_per_dataset, num_available)
        sample_generator = torch.Generator().manual_seed(sampling_seed + dataset_rank)
        sampled_indices = torch.randperm(num_available, generator=sample_generator)[:num_selected].tolist()

        X_selected = X[sampled_indices].clone()
        y_raw_selected = y_raw[sampled_indices].clone()
        y_binary_selected = binarize_labels(y_raw_selected, threshold=binarize_threshold)

        split_labels = _compute_source_train_val_labels(
            num_samples=num_selected,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=split_seed + dataset_rank,
        )

        y_binary_full = binarize_labels(y_raw, threshold=binarize_threshold)
        main_probe_split_indices = compute_main_probe_split_indices(
            num_samples=len(y_binary_full),
            train_fraction=main_probe_train_fraction,
            val_fraction=main_probe_val_fraction,
            seed=main_probe_split_seed,
        )
        main_probe_test_y = y_binary_full[main_probe_split_indices["test"]]

        for local_idx, source_row_index in enumerate(sampled_indices):
            row_obj = rows[source_row_index] if source_row_index < len(rows) else None
            if isinstance(row_obj, dict):
                row = copy.deepcopy(row_obj)
            else:
                row = {}

            row["dataset_name"] = dataset_name
            row["generalization_split"] = split_labels[local_idx]
            row["source_row_index"] = int(source_row_index)
            row["source_path"] = source_path
            row["source_score_raw"] = float(y_raw_selected[local_idx].item())
            row["label_binary"] = float(y_binary_selected[local_idx].item())

            rows_out.append(row)
            dataset_names_out.append(dataset_name)
            split_labels_out.append(split_labels[local_idx])
            source_row_indices_out.append(int(source_row_index))

        X_chunks.append(X_selected)
        y_chunks.append(y_binary_selected)

        split_counts = {
            "train": split_labels.count("train"),
            "val": split_labels.count("val"),
            "test": split_labels.count("test"),
        }

        per_dataset_metadata[dataset_name] = {
            "source_path": source_path,
            "source_dataset_id": source_metadata.get("dataset_id"),
            "source_model_id": source_metadata.get("model_id"),
            "num_available": int(num_available),
            "num_selected": int(num_selected),
            "sampling_seed": int(sampling_seed + dataset_rank),
            "split_seed": int(split_seed + dataset_rank),
            "raw_label_mean": float(y_raw_selected.mean().item()),
            "raw_label_variance": float(y_raw_selected.var(unbiased=False).item()),
            "mean_accuracy": float(y_binary_selected.mean().item()),
            "variance": float(y_binary_selected.var(unbiased=False).item()),
            "main_probe_split_seed": int(main_probe_split_seed),
            "main_probe_train_fraction": float(main_probe_train_fraction),
            "main_probe_val_fraction": float(main_probe_val_fraction),
            "main_probe_test_count": int(len(main_probe_split_indices["test"])),
            "main_probe_test_mean_accuracy": float(main_probe_test_y.mean().item()),
            "main_probe_test_variance": float(main_probe_test_y.var(unbiased=False).item()),
            "split_counts": split_counts,
            "selected_row_indices": [int(idx) for idx in sampled_indices],
        }
        print(
            f"  Selected {num_selected}/{num_available} samples | "
            f"binary accuracy={per_dataset_metadata[dataset_name]['mean_accuracy']:.4f} | "
            f"variance={per_dataset_metadata[dataset_name]['variance']:.4f} | "
            f"main-probe test variance={per_dataset_metadata[dataset_name]['main_probe_test_variance']:.4f}"
        )

    if total_feature_dim is None or selected_feature_dims is None:
        raise RuntimeError("No datasets were loaded; could not build generalization dataset.")

    X_all = torch.cat(X_chunks, dim=0).float()
    y_all = torch.cat(y_chunks, dim=0).float()

    metadata = {
        "experiment": "generalizability",
        "model_name": canonical_model,
        "model_id": spec.model_id,
        "datasets": list(DEFAULT_DATASETS),
        "num_samples": int(X_all.shape[0]),
        "feature_dim": int(X_all.shape[1]),
        "feature_names": selected_feature_names,
        "feature_dims": selected_feature_dims,
        "sampling_seed": int(sampling_seed),
        "source_split_seed": int(split_seed),
        "main_probe_split_seed": int(main_probe_split_seed),
        "binarize_threshold": float(binarize_threshold),
        "source_train_fraction": float(train_fraction),
        "source_val_fraction": float(val_fraction),
        "main_probe_train_fraction": float(main_probe_train_fraction),
        "main_probe_val_fraction": float(main_probe_val_fraction),
        "main_probe_test_fraction": float(1.0 - main_probe_train_fraction - main_probe_val_fraction),
        "max_per_dataset": int(max_per_dataset),
        "label_mean": float(y_all.mean().item()),
        "label_variance": float(y_all.var(unbiased=False).item()),
        "per_dataset": per_dataset_metadata,
    }

    payload = {
        "X": X_all,
        "y": y_all,
        "feature_names": selected_feature_names,
        "feature_dims": selected_feature_dims,
        "rows": rows_out,
        "dataset_names": dataset_names_out,
        "split_labels": split_labels_out,
        "source_row_indices": source_row_indices_out,
        "metadata": metadata,
    }

    torch.save(payload, dataset_path)
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSaved combined dataset:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Metadata path: {metadata_path}")
    print(f"  Total samples: {X_all.shape[0]}")
    print(f"  Label mean: {metadata['label_mean']:.4f}")

    return {
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
    }
