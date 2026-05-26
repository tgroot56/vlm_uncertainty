"""Generate LLaVA VQA-v2 layer sweep for last visual/question LM tokens."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import (  # noqa: E402
    _draw_layer_ax_two_metrics,
    _prepare_layer_sweep,
    save,
)

PARQUET_PATH = Path(
    "/projects/prjs2014/patching/layer_sweep_results_question_visual/"
    "llava-hf_llava-1-5-7b-hf/vqa-v2/layer_sweep_results.parquet"
)
OUTPUT_DIR = Path("/projects/prjs2014/patching/filtered_plots/cross_model/sample_cutoff")
OUTPUT_NAME = "two_metrics_layer_sweep_llava_vqa_question_visual_nofilter"

PANELS = [
    ("last_visual", "LM Visual (lasttok)"),
    ("last_question", "LM Question (lasttok)"),
]


def main() -> None:
    print(f"Reading {PARQUET_PATH}")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=False, squeeze=False)
    legend_handles = None

    for col, (span_label, title) in enumerate(PANELS):
        ax = axes[0, col]
        df, layers, sevs, n_total, n_filtered = _prepare_layer_sweep(
            str(PARQUET_PATH),
            min_delta=0.0,
            span_label=span_label,
        )
        print(
            f"  {span_label}: samples={n_total}, after_filter={n_filtered}, "
            f"layers={min(layers) + 1}-{max(layers) + 1}, severities={sevs}"
        )
        handles = _draw_layer_ax_two_metrics(
            ax,
            df,
            layers,
            sevs,
            show_xlabel=True,
            show_ylabel=(col == 0),
        )
        if legend_handles is None:
            legend_handles = handles
        ax.set_title(title, fontsize=18, fontweight="bold")
        if col == 0:
            ax.set_ylabel("LLaVA-1.5-7B\nAggregate Recovery", fontsize=14)

    fig.suptitle("Patching Layer Sweep - Confidence & Accuracy Recovery", fontsize=22, y=1.03)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
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
