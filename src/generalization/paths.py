from __future__ import annotations

from pathlib import Path


DEFAULT_OUTPUT_ROOT = "/projects/prjs2014/probe_results"
GENERALIZATION_DIRNAME = "generalization"
DATASETS_DIRNAME = "datasets"
TASK_GEN_DIRNAME = "task_gen"
MODEL_GEN_DIRNAME = "model_gen"
ALL_DATASETS_DIRNAME = "all_datasets"
PLOTS_DIRNAME = "plots"


def generalization_root(output_root: str) -> Path:
    return Path(output_root) / GENERALIZATION_DIRNAME


def datasets_root(output_root: str) -> Path:
    return generalization_root(output_root) / DATASETS_DIRNAME


def task_gen_root(output_root: str) -> Path:
    return generalization_root(output_root) / TASK_GEN_DIRNAME


def all_datasets_root(output_root: str) -> Path:
    return generalization_root(output_root) / ALL_DATASETS_DIRNAME


def model_gen_root(output_root: str) -> Path:
    return generalization_root(output_root) / MODEL_GEN_DIRNAME


def generalization_dataset_dir(output_root: str, canonical_model: str) -> Path:
    return datasets_root(output_root) / canonical_model


def generalization_dataset_path(output_root: str, canonical_model: str) -> Path:
    return generalization_dataset_dir(output_root, canonical_model) / "generalizability_supervision_dataset.pt"


def generalization_dataset_metadata_path(output_root: str, canonical_model: str) -> Path:
    return (
        generalization_dataset_dir(output_root, canonical_model)
        / "generalizability_supervision_dataset_metadata.json"
    )


def task_gen_feature_root(
    output_root: str,
    canonical_model: str,
    canonical_dataset: str,
    feature_name: str,
) -> Path:
    return task_gen_root(output_root) / canonical_model / canonical_dataset / feature_name


def all_datasets_feature_root(
    output_root: str,
    canonical_model: str,
    canonical_dataset: str,
    feature_name: str,
) -> Path:
    return all_datasets_root(output_root) / canonical_model / canonical_dataset / feature_name


def all_datasets_shared_train_root(
    output_root: str,
    canonical_model: str,
    feature_name: str,
) -> Path:
    return all_datasets_root(output_root) / canonical_model / "_shared_train" / feature_name


def model_gen_feature_root(
    output_root: str,
    model_key: str,
    canonical_dataset: str,
    feature_name: str,
) -> Path:
    return model_gen_root(output_root) / model_key / canonical_dataset / feature_name


def plot_root(output_root: str, mode: str) -> Path:
    mode_dir = {
        "leave_one_out": TASK_GEN_DIRNAME,
        "all_datasets": ALL_DATASETS_DIRNAME,
        "model_gen": MODEL_GEN_DIRNAME,
    }[mode]
    return generalization_root(output_root) / mode_dir / PLOTS_DIRNAME
