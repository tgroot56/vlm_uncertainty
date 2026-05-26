from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PATCHING_ROOT = Path(os.environ.get("PATCHING_ROOT", "/projects/prjs2014/patching"))
OUTPUT_ROOT = PATCHING_ROOT / "qualitative_viewer"
PAPER_EXAMPLES_DIR = OUTPUT_ROOT / "paper_examples"
REPO_ROOT = Path(os.environ.get("VLM_UQ_REPO_ROOT", "/home/tgroot/vlm_uncertainty"))

DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]
DATASET_LABELS = {
    "vqa-v2": "VQA-v2",
    "coco-qa-vi": "COCO-QA-VI",
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
MIN_POS_SAMPLES = 100
QWEN_VISUAL_MIN_FRACTION = 0.10


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
        result_subdir="extreme_answer_confidence",
        qwen_prefill_empty_thinking=False,
        processor_ref="llava-hf/llava-1.5-7b-hf",
    ),
    "Qwen3.5-9B": ModelSpec(
        label="Qwen3.5-9B",
        slug="Qwen_Qwen3-5-9B",
        model_id="Qwen/Qwen3.5-9B",
        result_subdir="extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking",
        qwen_prefill_empty_thinking=True,
        processor_ref=str(QWEN_SNAPSHOT),
    ),
}


def dataset_path_component(dataset: str) -> Path:
    if dataset == "pope":
        return Path("pope/random")
    return Path(dataset)


def parquet_path(model_label: str, dataset: str) -> Path:
    spec = MODEL_SPECS[model_label]
    return (
        spec.sweep_root
        / dataset_path_component(dataset)
        / spec.result_subdir
        / "positional_patching_results.parquet"
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

