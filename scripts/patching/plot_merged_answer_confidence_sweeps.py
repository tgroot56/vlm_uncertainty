"""Regenerate merged sweep plots from parquet using answer-token confidence."""
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.plot_sweep_filtered import (  # noqa: E402
    plot_all_datasets_layer_sweep,
    plot_all_datasets_positional_sweep,
)


METRICS = ["top1_probability", "score"]
REQUIRED_CONFIDENCE_COLUMNS = {
    "top1_probability",
    "answer_geom_mean_probability",
    "first_generated_token_top1_probability",
}
CONFIDENCE_TOLERANCE = 1e-10
EXPECTED_POSITIONAL_SAMPLES = 1000

QWEN_POS_ROOT = Path(
    "/projects/prjs2014/patching/positional_patching_results_including_visual/"
    "Qwen_Qwen3-5-9B"
)
QWEN_LAYER_ROOT = Path(
    "/projects/prjs2014/patching/layer_sweep_results/Qwen_Qwen3-5-9B"
)
LLAVA_POS_ROOT = Path(
    "/projects/prjs2014/patching/positional_patching_results_including_visual/"
    "llava-hf_llava-1-5-7b-hf"
)
LLAVA_LAYER_ROOT = Path(
    "/projects/prjs2014/patching/layer_sweep_results/llava-hf_llava-1-5-7b-hf"
)


def _assert_answer_confidence(parquet_path: str) -> None:
    """Fail if top1_probability is not the answer-token confidence column."""
    output_dir = Path(parquet_path).parent
    checkpoint_path = output_dir / "positional_patching_checkpoint.json"
    if checkpoint_path.exists():
        with checkpoint_path.open() as f:
            checkpoint = json.load(f)
        completed = len(checkpoint.get("completed_ids", []))
        expected = int(
            checkpoint.get("total_samples")
            or checkpoint.get("num_samples")
            or EXPECTED_POSITIONAL_SAMPLES
        )
        if completed < expected:
            raise ValueError(
                f"{parquet_path} still has an active checkpoint "
                f"({completed}/{expected} samples completed); refusing to plot partial data."
            )

    parquet_file = pq.ParquetFile(parquet_path)
    available_columns = set(parquet_file.schema.names)
    missing = REQUIRED_CONFIDENCE_COLUMNS.difference(available_columns)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    df = pd.read_parquet(parquet_path, columns=sorted(REQUIRED_CONFIDENCE_COLUMNS))
    missing = REQUIRED_CONFIDENCE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    diff = (df["top1_probability"] - df["answer_geom_mean_probability"]).abs()
    max_diff = float(np.nanmax(diff.to_numpy(dtype=float))) if len(diff) else 0.0
    if max_diff > CONFIDENCE_TOLERANCE:
        raise ValueError(
            f"{parquet_path}: top1_probability differs from "
            f"answer_geom_mean_probability (max abs diff={max_diff:.3g}); "
            "refusing to plot first-token confidence by accident."
        )


def _partition_valid_inputs(
    dataset_parquets: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    valid = []
    skipped = []
    for dataset, parquet_path in dataset_parquets:
        try:
            _assert_answer_confidence(parquet_path)
        except ValueError as exc:
            skipped.append((dataset, str(exc)))
        else:
            valid.append((dataset, parquet_path))
    return valid, skipped


def _run_if_valid(
    label: str,
    dataset_parquets: list[tuple[str, str]],
    plot_fn: Callable[[list[tuple[str, str]]], None],
) -> bool:
    valid, skipped = _partition_valid_inputs(dataset_parquets)
    print(f"\nCHECK {label}")
    for dataset, reason in skipped:
        print(f"  skip {dataset}: {reason}")
    if skipped:
        print(f"  not plotting {label}; waiting for all requested datasets")
        return False
    if not valid:
        print(f"  no valid datasets remain for {label}")
        return False
    print(f"  plot {', '.join(dataset for dataset, _path in valid)}")
    plot_fn(valid)
    return True


def main() -> None:
    qwen_pos = [
        ("vqa-v2", str(QWEN_POS_ROOT / "vqa-v2/extreme_answer_confidence/positional_patching_results.parquet")),
        ("coco-qa-vi", str(QWEN_POS_ROOT / "coco-qa-vi/extreme_answer_confidence/positional_patching_results.parquet")),
        ("imagenet-r", str(QWEN_POS_ROOT / "imagenet-r/extreme_answer_confidence/positional_patching_results.parquet")),
        ("pope", str(QWEN_POS_ROOT / "pope/extreme_answer_confidence/positional_patching_results.parquet")),
    ]
    qwen_layer = [
        ("vqa-v2", str(QWEN_LAYER_ROOT / "vqa-v2/extreme_answer_confidence_finaltoken/layer_sweep_results.parquet")),
        ("coco-qa-vi", str(QWEN_LAYER_ROOT / "coco-qa-vi/extreme_answer_confidence_finaltoken/layer_sweep_results.parquet")),
        ("imagenet-r", str(QWEN_LAYER_ROOT / "imagenet-r/extreme_answer_confidence_finaltoken/layer_sweep_results.parquet")),
        ("pope", str(QWEN_LAYER_ROOT / "pope/extreme_answer_confidence_finaltoken/layer_sweep_results.parquet")),
    ]
    llava_pos = [
        ("vqa-v2", str(LLAVA_POS_ROOT / "vqa-v2/extreme_answer_confidence/positional_patching_results.parquet")),
        ("coco-qa-vi", str(LLAVA_POS_ROOT / "coco-qa-vi/extreme_answer_confidence/positional_patching_results.parquet")),
        ("imagenet-r", str(LLAVA_POS_ROOT / "imagenet-r/extreme_answer_confidence/positional_patching_results.parquet")),
        ("pope", str(LLAVA_POS_ROOT / "pope/extreme_answer_confidence/positional_patching_results.parquet")),
    ]
    llava_layer = [
        ("vqa-v2", str(LLAVA_LAYER_ROOT / "vqa-v2/extreme_answer_confidence/layer_sweep_results.parquet")),
        ("coco-qa-vi", str(LLAVA_LAYER_ROOT / "coco-qa-vi/extreme_answer_confidence/layer_sweep_results.parquet")),
        ("imagenet-r", str(LLAVA_LAYER_ROOT / "imagenet-r/extreme_answer_confidence/layer_sweep_results.parquet")),
        ("pope", str(LLAVA_LAYER_ROOT / "pope/extreme_answer_confidence/layer_sweep_results.parquet")),
    ]

    plotted = []
    plotted.append(_run_if_valid(
        "Qwen positional sweep",
        qwen_pos,
        lambda valid: plot_all_datasets_positional_sweep(
            valid,
            QWEN_POS_ROOT,
            raw=True,
            min_delta=0.0,
            metrics=METRICS,
            title="Qwen3-5-9B: Positional Sweep",
            output_name="merged_answer_confidence_score_positional_sweep",
        ),
    ))
    plotted.append(_run_if_valid(
        "Qwen layer sweep",
        qwen_layer,
        lambda valid: plot_all_datasets_layer_sweep(
            valid,
            QWEN_LAYER_ROOT,
            raw=True,
            min_delta=0.0,
            metrics=METRICS,
            title="Qwen3-5-9B: Layer Sweep",
            output_name="merged_answer_confidence_score_layer_sweep",
        ),
    ))
    plotted.append(_run_if_valid(
        "LLaVA positional sweep",
        llava_pos,
        lambda valid: plot_all_datasets_positional_sweep(
            valid,
            LLAVA_POS_ROOT,
            raw=True,
            min_delta=0.0,
            metrics=METRICS,
            title="LLaVA-1.5-7B: Positional Sweep",
            output_name="merged_answer_confidence_score_positional_sweep",
        ),
    ))
    plotted.append(_run_if_valid(
        "LLaVA layer sweep",
        llava_layer,
        lambda valid: plot_all_datasets_layer_sweep(
            valid,
            LLAVA_LAYER_ROOT,
            raw=True,
            min_delta=0.0,
            metrics=METRICS,
            title="LLaVA-1.5-7B: Layer Sweep",
            output_name="merged_answer_confidence_score_layer_sweep",
        ),
    ))

    if not all(plotted):
        raise SystemExit(
            "One or more requested plots were skipped because their parquet "
            "inputs do not prove top1_probability is answer-token confidence."
        )


if __name__ == "__main__":
    main()
