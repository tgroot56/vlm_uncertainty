"""Plot the updated 100-sample Qwen COCO-QA-VI layer sweep."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from src.cli.plot_sweep_filtered import plot_two_metrics_cross_model_layer_sweep


def main() -> None:
    storage_root = Path(os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014"))
    model_short = os.environ.get("VLM_SHORT", "Qwen_Qwen3-5-9B")
    output_tag = os.environ.get(
        "OUTPUT_TAG",
        "extreme_clean_answer_confidence_stripped_softmax_finaltoken",
    )
    output_dir = (
        storage_root
        / "patching"
        / "layer_sweep_results"
        / model_short
        / "coco-qa-vi"
        / f"{output_tag}_qwen_prefill_empty_thinking"
    )
    parquet_path = output_dir / "layer_sweep_results.parquet"

    # Avoid loading the multi-gigabyte stripped-softmax byte payloads into RAM.
    plot_columns = [
        "sample_id",
        "severity",
        "condition",
        "layer",
        "position_offset",
        "span_label",
        "top1_probability",
        "clean_answer_probability",
        "score",
    ]
    with TemporaryDirectory(prefix="qwen_coco_layer_plot_") as tmp_dir:
        compact_parquet = Path(tmp_dir) / "plot_metrics.parquet"
        pd.read_parquet(parquet_path, columns=plot_columns).to_parquet(
            compact_parquet, index=False
        )
        plot_two_metrics_cross_model_layer_sweep(
            model_entries=[("Qwen3.5-9B", {"coco-qa-vi": str(compact_parquet)})],
            datasets=["coco-qa-vi"],
            output_dir=output_dir,
            output_name=(
                "layer_sweep_expressed_clean_answer_confidence_accuracy_recovery"
            ),
            min_delta=0.0,
            metrics=["top1_probability", "clean_answer_probability", "score"],
            title=(
                "Qwen3.5-9B COCO-QA-VI Layer Sweep - Expressed Confidence, "
                "Clean-Answer Confidence & Accuracy Recovery (100 samples)"
            ),
        )


if __name__ == "__main__":
    main()
