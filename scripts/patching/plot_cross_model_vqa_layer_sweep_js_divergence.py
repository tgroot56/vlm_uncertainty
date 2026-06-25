#!/usr/bin/env python
"""Plot VQA-v2 clean-vs-patched softmax JS divergence for LLaVA and Qwen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.patching.plot_vqa_layer_sweep_js_divergence import (
    compute_layer_js_streaming,
)


PATCHING_ROOT = Path("/projects/prjs2014/patching")
LLAVA_PARQUET = (
    PATCHING_ROOT
    / "layer_sweep_results/llava-hf_llava-1-5-7b-hf/vqa-v2"
    / "extreme_clean_answer_confidence_stripped_softmax/layer_sweep_results.parquet"
)
QWEN_PARQUET = (
    PATCHING_ROOT
    / "layer_sweep_results/Qwen_Qwen3-5-9B/vqa-v2"
    / "extreme_clean_answer_confidence_stripped_softmax_"
    "finaltoken_qwen_prefill_empty_thinking/layer_sweep_results.parquet"
)
DEFAULT_OUTPUT_DIR = PATCHING_ROOT / "layer_sweep_results/cross_model/vqa-v2"
DEFAULT_OUTPUT_NAME = "cross_model_vqa_layer_sweep_js_clean_patched_single_token"


def load_or_compute_summary(
    parquet_path: Path,
    summary_path: Path,
    *,
    position_offset: int,
    recompute: bool,
) -> pd.DataFrame:
    if (
        not recompute
        and summary_path.exists()
        and summary_path.stat().st_mtime >= parquet_path.stat().st_mtime
    ):
        print(f"Using current summary: {summary_path}")
        return pd.read_csv(summary_path)

    print(f"Computing summary from {parquet_path}")
    summary = compute_layer_js_streaming(
        parquet_path,
        position_offset=position_offset,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    return summary


def plot_cross_model(
    summaries: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    output_name: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(summaries),
        figsize=(13.5, 5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for ax, (model_label, summary) in zip(axes, summaries):
        ax.plot(
            summary["display_layer"],
            summary["mean_js_clean_patched"],
            "o-",
            color="#2563eb",
            linewidth=2,
            markersize=4.5,
        )
        ax.set_title(model_label, fontsize=16, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_xticks(summary["display_layer"].tolist())
        ax.set_xticklabels(
            [str(value) for value in summary["display_layer"]],
            fontsize=8,
            rotation=45,
        )
        ax.grid(alpha=0.3)
        n_min = int(summary["n_single_token_examples"].min())
        n_max = int(summary["n_single_token_examples"].max())
        ax.text(
            0.02,
            0.97,
            f"Single-token examples/layer: {n_min}-{n_max}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    axes[0].set_ylabel("Mean JS divergence (nats)", fontsize=13)
    fig.suptitle(
        "VQA-v2 Layer Sweep: JS(Clean Softmax, Patched Softmax)",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in [("pdf", {}), ("png", {"dpi": 200})]:
        output_path = output_dir / f"{output_name}.{extension}"
        fig.savefig(output_path, bbox_inches="tight", **kwargs)
        print(f"Saved {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava_parquet", type=Path, default=LLAVA_PARQUET)
    parser.add_argument("--qwen_parquet", type=Path, default=QWEN_PARQUET)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--position_offset", type=int, default=-1)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute summaries even when current cached CSVs exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = [
        ("LLaVA-1.5-7B", args.llava_parquet),
        ("Qwen3.5-9B", args.qwen_parquet),
    ]
    summaries = []
    for model_label, parquet_path in entries:
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        summary_path = args.output_dir / (
            f"{args.output_name}_{model_label.lower().replace('.', '_').replace('-', '_')}.csv"
        )
        summaries.append(
            (
                model_label,
                load_or_compute_summary(
                    parquet_path,
                    summary_path,
                    position_offset=args.position_offset,
                    recompute=args.recompute,
                ),
            )
        )
    plot_cross_model(summaries, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
