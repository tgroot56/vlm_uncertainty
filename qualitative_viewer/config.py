from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PATCHING_ROOT = Path(os.environ.get("PATCHING_ROOT", "/projects/prjs2014/patching"))
OUTPUT_ROOT = PATCHING_ROOT / "qualitative_viewer"
PAPER_EXAMPLES_DIR = OUTPUT_ROOT / "paper_examples"
SOFTMAX_CHANGES_ROOT = PATCHING_ROOT / "softmax_changes"
SOFTMAX_CHANGES_DATASETS = ["coco-qa-vi"]
REPO_ROOT = Path(os.environ.get("VLM_UQ_REPO_ROOT", "/home/tgroot/vlm_uncertainty"))

DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]
DATASET_LABELS = {
    "vqa-v2": "VQA-v2",
    "coco-qa-vi": "COCO-QA",
    "imagenet-r": "ImageNet-R",
    "pope": "POPE",
}

SPAN_DISPLAY = {
    "text_pre": "Template prefix",
    "visual": "Visual tokens",
    "question": "Question",
    "assistant_marker": "Template suffix",
}

SPAN_ORDER = ["text_pre", "visual", "question", "assistant_marker"]
METRICS_FOR_PLOT = ["top1_probability", "score"]
MIN_POS_SAMPLES = 35
POSITIONAL_PATCH_LAYER = 17
FINAL_PLOT_SAMPLE_LIMITS = {
    "LLaVA-1.5-7B": 100,
    "Qwen3.5-9B": 100,
}
FINAL_LAYER_PLOT = (
    PATCHING_ROOT
    / "layer_sweep_results"
    / "two_metrics_layer_sweep_clean_answer_confidence_clean_correct_corrupted_incorrect_unclipped.pdf"
)
FINAL_POSITIONAL_PLOT = (
    PATCHING_ROOT
    / "positional_patching_results_including_visual"
    / "two_metrics_positional_sweep_clean_answer_confidence_clean_correct_corrupted_incorrect_n35_unclipped.pdf"
)
GRADIENT_GUIDED_MODEL_LABEL = "LLaVA-1.5-7B"
GRADIENT_GUIDED_DATASET = "pope"
GRADIENT_GUIDED_OBJECTIVE = "obj_yes"
GRADIENT_GUIDED_OBJECTIVE_NAME = "maximize_p_yes"
GRADIENT_GUIDED_LAYERS = (1, 11, 15, 18)
GRADIENT_GUIDED_ROOT = (
    PATCHING_ROOT
    / "gradient_guided_token_patching"
    / "llava-hf_llava-1-5-7b-hf"
    / "pope"
    / "random"
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    slug: str
    model_id: str
    result_subdir: str
    qwen_prefill_empty_thinking: bool = False
    processor_ref: str | None = None

    @property
    def data_root(self) -> Path:
        return PATCHING_ROOT / self.slug

    @property
    def sweep_root(self) -> Path:
        return PATCHING_ROOT / "positional_patching_results_including_visual" / self.slug

    @property
    def layer_sweep_root(self) -> Path:
        return PATCHING_ROOT / "layer_sweep_results" / self.slug


QWEN_SNAPSHOT = (
    PATCHING_ROOT.parent
    / "tgroot/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/"
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)

MODEL_SPECS = {
    "LLaVA-1.5-7B": ModelSpec(
        label="LLaVA-1.5-7B",
        slug="llava-hf_llava-1-5-7b-hf",
        model_id="llava-hf/llava-1.5-7b-hf",
        result_subdir="extreme_clean_answer_confidence",
        qwen_prefill_empty_thinking=False,
        processor_ref="llava-hf/llava-1.5-7b-hf",
    ),
    "Qwen3.5-9B": ModelSpec(
        label="Qwen3.5-9B",
        slug="Qwen_Qwen3-5-9B",
        model_id="Qwen/Qwen3.5-9B",
        result_subdir="extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
        qwen_prefill_empty_thinking=True,
        processor_ref=str(QWEN_SNAPSHOT),
    ),
}


LAYER_SWEEP_RESULT_SUBDIR_OVERRIDES = {
    ("LLaVA-1.5-7B", "vqa-v2"): "extreme_clean_answer_confidence_stripped_softmax",
    ("LLaVA-1.5-7B", "coco-qa-vi"): "extreme_clean_answer_confidence_stripped_softmax",
    ("LLaVA-1.5-7B", "imagenet-r"): "extreme_clean_answer_confidence_stripped_softmax",
    ("LLaVA-1.5-7B", "pope"): "extreme_answer_confidence_clean_answer",
    ("Qwen3.5-9B", "vqa-v2"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
    ("Qwen3.5-9B", "coco-qa-vi"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
    ("Qwen3.5-9B", "imagenet-r"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
    ("Qwen3.5-9B", "pope"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
}

POSITIONAL_SWEEP_RESULT_SUBDIR_OVERRIDES = {
    ("LLaVA-1.5-7B", "vqa-v2"): "extreme_clean_answer_confidence",
    ("LLaVA-1.5-7B", "coco-qa-vi"): "extreme_clean_answer_confidence",
    ("LLaVA-1.5-7B", "imagenet-r"): "extreme_clean_answer_confidence",
    ("LLaVA-1.5-7B", "pope"): "extreme_clean_answer_confidence",
    ("Qwen3.5-9B", "vqa-v2"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
    ("Qwen3.5-9B", "coco-qa-vi"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
    ("Qwen3.5-9B", "imagenet-r"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
    ("Qwen3.5-9B", "pope"): "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking",
}


def dataset_path_component(dataset: str) -> Path:
    if dataset == "pope":
        return Path("pope/random")
    return Path(dataset)


def parquet_path(model_label: str, dataset: str) -> Path:
    spec = MODEL_SPECS[model_label]
    result_subdir = POSITIONAL_SWEEP_RESULT_SUBDIR_OVERRIDES.get(
        (model_label, dataset),
        spec.result_subdir,
    )
    return (
        spec.sweep_root
        / dataset_path_component(dataset)
        / result_subdir
        / "positional_patching_results.parquet"
    )


def layer_sweep_path(model_label: str, dataset: str) -> Path:
    spec = MODEL_SPECS[model_label]
    result_subdir = LAYER_SWEEP_RESULT_SUBDIR_OVERRIDES.get(
        (model_label, dataset),
        spec.result_subdir,
    )
    return (
        spec.layer_sweep_root
        / dataset_path_component(dataset)
        / result_subdir
        / "layer_sweep_results.parquet"
    )


def softmax_changes_path(model_label: str, dataset: str) -> Path:
    """JSON written by ``python -m src.cli.run_softmax_changes`` for this model/dataset."""
    spec = MODEL_SPECS[model_label]
    return (
        SOFTMAX_CHANGES_ROOT
        / spec.slug
        / dataset_path_component(dataset)
        / "softmax_changes.json"
    )


def patching_dataset_path(model_label: str, dataset: str) -> Path:
    spec = MODEL_SPECS[model_label]
    return spec.data_root / dataset_path_component(dataset) / f"patching_dataset_{dataset}.json"


def supervised_metadata_path(model_label: str, dataset: str, sample_id: str) -> Path | None:
    spec = MODEL_SPECS[model_label]
    if spec.slug != "llava-hf_llava-1-5-7b-hf":
        return None
    return (
        PATCHING_ROOT
        / "supervised_datasets"
        / spec.slug
        / dataset
        / spec.slug
        / str(sample_id)
        / "metadata.json"
    )


def all_parquet_paths() -> dict[tuple[str, str], Path]:
    return {
        (model_label, dataset): parquet_path(model_label, dataset)
        for model_label in MODEL_SPECS
        for dataset in DATASETS
    }


def all_layer_sweep_paths() -> dict[tuple[str, str], Path]:
    return {
        (model_label, dataset): layer_sweep_path(model_label, dataset)
        for model_label in MODEL_SPECS
        for dataset in DATASETS
    }
