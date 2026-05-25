"""One-off script: generate cross-model extreme-severity combined figures.

Produces filtered_plots/cross_model/:
  cross_model_layer_sweep_vqa_coco.pdf/.png
  cross_model_positional_sweep_vqa_coco.pdf/.png
  cross_model_layer_sweep_imagenet_pope.pdf/.png
  cross_model_positional_sweep_imagenet_pope.pdf/.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import (
    plot_cross_model_layer_sweep,
    plot_cross_model_positional_sweep,
)

_ROOT = Path("/projects/prjs2014/patching")
CROSS_OUT = _ROOT / "filtered_plots/cross_model/sample_cutoff"
MIN_POS_SAMPLES = 100

LLAVA_LS = _ROOT / "layer_sweep_results/llava-hf_llava-1-5-7b-hf"
LLAVA_PS = _ROOT / "positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf"
QWEN_LS  = _ROOT / "layer_sweep_results/Qwen_Qwen3-5-9B"
QWEN_PS  = _ROOT / "positional_patching_results_including_visual/Qwen_Qwen3-5-9B"

ALL_DS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]

# These answer-confidence runs are the post-fix datasets. Refuse to plot older
# first-token-confidence outputs by checking the parquet schema and values below.
LLAVA_LAYER_SUBDIR = "extreme_answer_confidence"
LLAVA_POS_SUBDIR = "extreme_answer_confidence"
QWEN_LAYER_SUBDIR = "extreme_answer_confidence_finaltoken"
QWEN_POS_SUBDIR = "extreme_answer_confidence"

REQUIRED_CONFIDENCE_COLUMNS = {
    "top1_probability",
    "answer_geom_mean_probability",
    "first_generated_token_top1_probability",
}
CONFIDENCE_TOLERANCE = 1e-10


def _assert_answer_confidence(parquet_path: str) -> None:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(parquet_path)

    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema.names)
    missing = REQUIRED_CONFIDENCE_COLUMNS.difference(available_columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = pd.read_parquet(path, columns=sorted(REQUIRED_CONFIDENCE_COLUMNS))
    diff = (df["top1_probability"] - df["answer_geom_mean_probability"]).abs()
    max_diff = float(np.nanmax(diff.to_numpy(dtype=float))) if len(diff) else 0.0
    if max_diff > CONFIDENCE_TOLERANCE:
        raise ValueError(
            f"{path}: top1_probability differs from answer_geom_mean_probability "
            f"(max abs diff={max_diff:.3g})"
        )


llava_layer_paths = {ds: str(LLAVA_LS / ds / LLAVA_LAYER_SUBDIR / "layer_sweep_results.parquet") for ds in ALL_DS}
llava_pos_paths = {ds: str(LLAVA_PS / ds / LLAVA_POS_SUBDIR / "positional_patching_results.parquet") for ds in ALL_DS}

qwen_layer_paths = {ds: str(QWEN_LS / ds / QWEN_LAYER_SUBDIR / "layer_sweep_results.parquet") for ds in ALL_DS}
qwen_pos_paths = {ds: str(QWEN_PS / ds / QWEN_POS_SUBDIR / "positional_patching_results.parquet") for ds in ALL_DS}

model_entries_layer = [
    ("LLaVA-1.5-7B", llava_layer_paths),
    ("Qwen3.5-9B",   qwen_layer_paths),
]
model_entries_pos = [
    ("LLaVA-1.5-7B", llava_pos_paths),
    ("Qwen3.5-9B",   qwen_pos_paths),
]

PAIRS = [
    (["vqa-v2", "coco-qa-vi"], "vqa_coco"),
    (["imagenet-r", "pope"],   "imagenet_pope"),
]

for _model_label, dataset_paths in model_entries_layer + model_entries_pos:
    for _dataset, _path in dataset_paths.items():
        _assert_answer_confidence(_path)

PLOT_MODES = [
    {"raw": False, "min_delta": 0.05},
    {"raw": False, "min_delta": 0.0},
    {"raw": True, "min_delta": 0.0},
]

for mode in PLOT_MODES:
    if mode["raw"] and mode["min_delta"] <= 0:
        mode_label = " raw no-filter"
    elif mode["min_delta"] <= 0:
        mode_label = " aggregate recovery no-filter"
    else:
        mode_label = " aggregate recovery filtered"
    print(f"\n=== Plot mode{mode_label} ===")
    for pair, pair_name in PAIRS:
        print(f"\n=== Layer sweep: {pair_name} ===")
        plot_cross_model_layer_sweep(
            model_entries_layer, pair,
            output_dir=CROSS_OUT,
            output_name=f"cross_model_layer_sweep_{pair_name}",
            raw=mode["raw"],
            min_delta=mode["min_delta"],
        )
        print(f"=== Positional sweep: {pair_name} ===")
        plot_cross_model_positional_sweep(
            model_entries_pos, pair,
            output_dir=CROSS_OUT,
            output_name=f"cross_model_positional_sweep_{pair_name}",
            raw=mode["raw"],
            min_delta=mode["min_delta"],
            min_pos_samples=MIN_POS_SAMPLES,
        )

print("\nDone.")
