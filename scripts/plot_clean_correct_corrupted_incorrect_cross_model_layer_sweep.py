"""Plot clean-correct/corrupted-incorrect layer sweeps for LLaVA and Qwen.

The figure matches the existing LLaVA-only two-metric layer-sweep layout while
adding Qwen3.5-9B as a second row. Only scalar columns required for plotting are
read, so the saved stripped-token softmax distributions remain out of memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import (
    DATASET_LABELS,
    SEVERITY_ORDER,
    _attach_baselines,
    _draw_layer_ax_two_metrics,
    save,
)


PATCHING_ROOT = Path("/projects/prjs2014/patching")
LLAVA_ROOT = PATCHING_ROOT / "layer_sweep_results/llava-hf_llava-1-5-7b-hf"
QWEN_ROOT = PATCHING_ROOT / "layer_sweep_results/Qwen_Qwen3-5-9B"
OUTPUT_DIR = LLAVA_ROOT
OUTPUT_NAME = (
    "all_datasets_cross_model_clean_correct_corrupted_incorrect_"
    "clean_answer_two_metrics_layer_sweep_nofilter"
)

DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]
METRICS = ["clean_answer_probability", "score"]
READ_COLUMNS = [
    "sample_id",
    "severity",
    "condition",
    "layer",
    "position_offset",
    "span_label",
    "top1_probability",
    *METRICS,
]

LLAVA_PATHS = {
    "vqa-v2": LLAVA_ROOT
    / "vqa-v2/extreme_clean_answer_confidence_stripped_softmax/layer_sweep_results.parquet",
    "coco-qa-vi": LLAVA_ROOT
    / "coco-qa-vi/extreme_clean_answer_confidence_stripped_softmax/layer_sweep_results.parquet",
    "imagenet-r": LLAVA_ROOT
    / "imagenet-r/extreme_clean_answer_confidence_stripped_softmax/layer_sweep_results.parquet",
    "pope": LLAVA_ROOT
    / "pope/random/extreme_answer_confidence_clean_answer/layer_sweep_results.parquet",
}

QWEN_SUBDIR = "extreme_clean_answer_confidence_stripped_softmax_finaltoken_qwen_prefill_empty_thinking"
QWEN_PATHS = {
    "vqa-v2": QWEN_ROOT / "vqa-v2" / QWEN_SUBDIR / "layer_sweep_results.parquet",
    "coco-qa-vi": QWEN_ROOT / "coco-qa-vi" / QWEN_SUBDIR / "layer_sweep_results.parquet",
    "imagenet-r": QWEN_ROOT / "imagenet-r" / QWEN_SUBDIR / "layer_sweep_results.parquet",
    "pope": QWEN_ROOT / "pope/random" / QWEN_SUBDIR / "layer_sweep_results.parquet",
}


def prepare_filtered_layer_sweep(
    parquet_path: Path,
) -> tuple[pd.DataFrame, list[int], list[str], int]:
    """Select final-token patched rows where clean is correct and corruption is not."""
    df = pd.read_parquet(parquet_path, columns=READ_COLUMNS)
    clean = (
        df[df["condition"].eq("clean")][["sample_id", "severity", "score"]]
        .rename(columns={"score": "clean_filter_score"})
    )
    corrupted = (
        df[df["condition"].eq("degraded")][["sample_id", "severity", "score"]]
        .rename(columns={"score": "corrupted_filter_score"})
    )
    keys = clean.merge(corrupted, on=["sample_id", "severity"])
    keys = keys[
        keys["clean_filter_score"].astype(float).gt(0)
        & keys["corrupted_filter_score"].astype(float).eq(0)
    ][["sample_id", "severity"]].drop_duplicates()

    patched = df[
        df["condition"].eq("patched")
        & df["position_offset"].eq(-1)
        & df["span_label"].isna()
    ].copy()
    patched = patched.merge(keys, on=["sample_id", "severity"], how="inner")
    patched = _attach_baselines(
        patched,
        df,
        metrics=["top1_probability", *METRICS],
    )

    layers = sorted(patched["layer"].dropna().astype(int).unique())
    severities = [s for s in SEVERITY_ORDER if s in patched["severity"].values]
    return patched, layers, severities, len(keys)


def main() -> None:
    model_entries = [
        ("LLaVA-1.5-7B", LLAVA_PATHS),
        ("Qwen3.5-9B", QWEN_PATHS),
    ]
    fig, axes = plt.subplots(
        len(model_entries),
        len(DATASETS),
        figsize=(5.5 * len(DATASETS), 4.5 * len(model_entries)),
        sharey=False,
        squeeze=False,
    )
    legend_handles = None

    for row, (model_label, paths) in enumerate(model_entries):
        for col, dataset in enumerate(DATASETS):
            path = paths[dataset]
            if not path.exists():
                raise FileNotFoundError(path)
            frame, layers, severities, n_examples = prepare_filtered_layer_sweep(path)
            print(f"{model_label:13s} {dataset:10s}: {n_examples:4d} examples")

            ax = axes[row, col]
            handles = _draw_layer_ax_two_metrics(
                ax,
                frame,
                layers,
                severities,
                show_xlabel=row == len(model_entries) - 1,
                show_ylabel=col == 0,
                metrics=METRICS,
                clip_recovery=True,
            )
            if legend_handles is None:
                legend_handles = handles
            ax.set_title(
                DATASET_LABELS.get(dataset, dataset) if row == 0 else "",
                fontsize=18,
                fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel(f"{model_label}\nAggregate recovery", fontsize=14)

    fig.suptitle(
        "Patching Layer Sweep - Clean-Answer Confidence & Accuracy Recovery",
        fontsize=22,
        y=1.03,
    )
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, OUTPUT_DIR, OUTPUT_NAME)


if __name__ == "__main__":
    main()
