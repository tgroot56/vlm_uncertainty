"""Regenerate raw no-filter cross-model layer-sweep plots.

This script uses the same layer-sweep parquet inputs as
``scripts/plot_two_metrics_layer_sweep.py`` and overwrites:

- cross_model_layer_sweep_vqa_coco_raw_nofilter.pdf/.png
- cross_model_layer_sweep_imagenet_pope_raw_nofilter.pdf/.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import plot_cross_model_layer_sweep


PATCHING_ROOT = Path("/projects/prjs2014/patching")
CROSS_OUT = PATCHING_ROOT / "filtered_plots/cross_model/sample_cutoff"

LLAVA_LS = PATCHING_ROOT / "layer_sweep_results/llava-hf_llava-1-5-7b-hf"
QWEN_LS = PATCHING_ROOT / "layer_sweep_results/Qwen_Qwen3-5-9B"

DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]
PAIRS = [
    (["vqa-v2", "coco-qa-vi"], "vqa_coco"),
    (["imagenet-r", "pope"], "imagenet_pope"),
]
TITLES = {
    "vqa_coco": "Patching Layer Sweep - Raw Confidence & Accuracy Recovery - VQA-V2 & COCO-QA-VI",
    "imagenet_pope": "Patching Layer Sweep - Raw Confidence & Accuracy Recovery - ImageNet-R & POPE",
}

LLAVA_SUBDIR = "extreme_answer_confidence"
QWEN_SUBDIR_DEFAULT = "extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking"
QWEN_SUBDIRS = {
    "vqa-v2": QWEN_SUBDIR_DEFAULT,
    "coco-qa-vi": QWEN_SUBDIR_DEFAULT,
    "imagenet-r": QWEN_SUBDIR_DEFAULT,
    "pope": f"random/{QWEN_SUBDIR_DEFAULT}",
}

REQUIRED_CONFIDENCE_COLUMNS = {
    "top1_probability",
    "answer_geom_mean_probability",
}
CONFIDENCE_TOLERANCE = 1e-10


def llava_dataset_component(dataset: str) -> Path:
    if dataset == "pope":
        return Path("pope/random")
    return Path(dataset)


def assert_answer_confidence(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, columns=sorted(REQUIRED_CONFIDENCE_COLUMNS))
    diff = (df["top1_probability"] - df["answer_geom_mean_probability"]).abs()
    max_diff = float(np.nanmax(diff.to_numpy(dtype=float))) if len(diff) else 0.0
    if max_diff > CONFIDENCE_TOLERANCE:
        raise ValueError(
            f"{path}: top1_probability differs from answer_geom_mean_probability "
            f"(max abs diff={max_diff:.3g})"
        )


def main() -> None:
    llava_paths = {
        ds: str(LLAVA_LS / llava_dataset_component(ds) / LLAVA_SUBDIR / "layer_sweep_results.parquet")
        for ds in DATASETS
    }
    qwen_paths = {
        ds: str(QWEN_LS / ds / QWEN_SUBDIRS[ds] / "layer_sweep_results.parquet")
        for ds in DATASETS
    }
    model_entries = [
        ("LLaVA-1.5-7B", llava_paths),
        ("Qwen3.5-9B", qwen_paths),
    ]

    print("Using layer-sweep inputs from scripts/plot_two_metrics_layer_sweep.py:")
    for label, paths in model_entries:
        for dataset, path in paths.items():
            parquet_path = Path(path)
            assert_answer_confidence(parquet_path)
            print(f"  {label:15s} {dataset:12s}: {parquet_path}")

    for datasets_pair, pair_name in PAIRS:
        print(f"\nRegenerating raw no-filter layer sweep: {pair_name}")
        plot_cross_model_layer_sweep(
            model_entries,
            datasets_pair,
            output_dir=CROSS_OUT,
            output_name=f"cross_model_layer_sweep_{pair_name}",
            raw=True,
            min_delta=0.0,
            title=TITLES.get(pair_name),
        )

    print("\nVerifying outputs:")
    for _, pair_name in PAIRS:
        for ext in ("pdf", "png"):
            out = CROSS_OUT / f"cross_model_layer_sweep_{pair_name}_raw_nofilter.{ext}"
            print(f"  {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
