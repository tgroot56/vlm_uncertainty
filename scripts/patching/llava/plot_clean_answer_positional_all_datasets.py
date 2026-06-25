"""Plot the 100-sample LLaVA positional sweeps with clean-answer confidence."""

from __future__ import annotations

import os
from pathlib import Path

from src.cli.plot_sweep_filtered import plot_two_metrics_cross_model_positional_sweep


def main() -> None:
    storage_root = Path(os.environ.get("VLM_UQ_STORAGE_ROOT", "/projects/prjs2014"))
    model_short = os.environ.get("VLM_SHORT", "llava-hf_llava-1-5-7b-hf")
    output_tag = os.environ.get("OUTPUT_TAG", "extreme_clean_answer_confidence")
    sample_count = os.environ.get("SAMPLE_COUNT", "100")
    root = (
        storage_root
        / "patching"
        / "positional_patching_results_including_visual"
        / model_short
    )
    dataset_paths = {
        "vqa-v2": str(root / "vqa-v2" / output_tag / "positional_patching_results.parquet"),
        "pope": str(
            root / "pope" / "random" / output_tag / "positional_patching_results.parquet"
        ),
        "coco-qa-vi": str(
            root / "coco-qa-vi" / output_tag / "positional_patching_results.parquet"
        ),
        "imagenet-r": str(
            root / "imagenet-r" / output_tag / "positional_patching_results.parquet"
        ),
    }
    output_dir = root / "all_datasets" / output_tag

    plot_two_metrics_cross_model_positional_sweep(
        model_entries=[("LLaVA-1.5-7B", dataset_paths)],
        datasets=["vqa-v2", "pope", "coco-qa-vi", "imagenet-r"],
        output_dir=output_dir,
        output_name=(
            "all_datasets_positional_sweep_"
            f"clean_answer_confidence_accuracy_recovery_{sample_count}"
        ),
        min_delta=0.0,
        metrics=["clean_answer_probability", "score"],
        title=(
            "LLaVA Positional Sweep - "
            f"Clean-Answer Confidence & Accuracy Recovery ({sample_count} samples)"
        ),
    )


if __name__ == "__main__":
    main()
