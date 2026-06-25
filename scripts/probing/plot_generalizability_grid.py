"""Plot the two-model leave-one-dataset-out generalizability grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tables_and_plots.plot_generalization_auroc import (
    collect_generalization_aurocs,
)


DATASETS = ("vqa-v2", "coco-qa-vi", "imagenet-r", "pope/random")
DATASET_TITLES = {
    "vqa-v2": "VQA-v2",
    "coco-qa-vi": "COCO-QA",
    "imagenet-r": "ImageNet-R",
    "pope/random": "POPE",
}
MODELS = (("llava", "LLaVA-1.5-7B"), ("qwen", "Qwen3.5-9B"))
VARIANTS = (
    ("leave_one_out", "Leave-one-out", "#3487b7"),
    ("in_domain", "In-domain", "#43ad5c"),
)


def plot_grid(root: Path, feature_name: str, output: Path) -> None:
    scores = {
        model: collect_generalization_aurocs(
            root=root,
            model_name=model,
            datasets=list(DATASETS),
            feature_name=feature_name,
        )
        for model, _ in MODELS
    }

    fig, axes = plt.subplots(
        len(MODELS),
        len(DATASETS),
        figsize=(30, 14),
        sharex=True,
        sharey=True,
    )

    x = np.arange(len(VARIANTS))
    for row, (model, model_title) in enumerate(MODELS):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            for index, (variant, _, color) in enumerate(VARIANTS):
                mean = scores[model][dataset][variant]["mean"]
                std = scores[model][dataset][variant]["std"]
                bar = ax.bar(
                    index,
                    mean,
                    yerr=std,
                    width=0.92,
                    capsize=5,
                    color=color,
                    edgecolor="black",
                    linewidth=1.4,
                    error_kw={"elinewidth": 1.8, "capthick": 1.8},
                    zorder=3,
                )
                ax.bar_label(
                    bar,
                    labels=[f"{mean:.3f}"],
                    padding=5,
                    fontsize=11,
                )

            ax.axhline(
                0.5,
                color="black",
                linestyle="--",
                linewidth=2.5,
                alpha=0.72,
                zorder=5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [label for _, label, _ in VARIANTS],
                rotation=24,
                ha="right",
                fontsize=28,
            )
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="y", labelsize=28)
            ax.set_ylim(0.0, 1.08)
            ax.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.35, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                ax.set_title(
                    DATASET_TITLES[dataset],
                    fontsize=36,
                    fontweight="bold",
                    pad=18,
                )
            if col == 0:
                ax.set_ylabel(
                    f"{model_title}\nTest AUROC",
                    fontsize=34,
                    labelpad=12,
                )

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for _, label, color in VARIANTS
    ]
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.5, label="Chance AUROC")
    )
    fig.suptitle(
        "Leave-One-Dataset-Out Generalizability",
        fontsize=38,
        y=0.985,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=True,
        fontsize=34,
        title_fontsize=12,
        handlelength=1.8,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.79, hspace=0.38, wspace=0.16)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output} (+ png)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/projects/prjs2014/probe_results"),
    )
    parser.add_argument(
        "--feature-name",
        default="lm_fullspan_lasttok_layer_16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/projects/prjs2014/probe_results/generalization/task_gen/plots/"
            "generalizability_grid_lm_final_token_middle_layer_auroc.pdf"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_grid(args.root, args.feature_name, args.output)
