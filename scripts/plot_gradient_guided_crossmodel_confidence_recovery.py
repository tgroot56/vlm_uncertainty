"""Compose a two-panel cross-model figure of gradient-guided clean-answer
confidence recovery (diverging stacked token-span contributions).

Left panel = Qwen3.5-9B, right panel = LLaVA-1.5-7B. Each panel is rebuilt
faithfully from the per-model aggregates already written by
``scripts/plot_gradient_guided_objective_contributions.py`` (the
``objective_summary_10samples_crossmodel/maximize_p_yes`` directory), reusing
the exact diverging-stacked drawing logic.

Typography and figure dimensions are defined as editable constants below.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

# --- token-span ordering / colours (identical to the source plotting script) ---
SPAN_ORDER = [
    "Template Prefix",
    "Visual",
    "Question",
    "Template Suffix",
    "Final prompt token",
]
SPAN_COLORS = {
    "Template Prefix": "#64748b",
    "Visual": "#0284c7",
    "Question": "#dc2626",
    "Template Suffix": "#7c3aed",
    "Final prompt token": "#16a34a",
}
POS_COL = "mean_positive_clean_confidence_recovery_contribution"
NEG_COL = "mean_negative_clean_confidence_recovery_contribution"
TOTAL_COL = "mean_clean_confidence_recovery"

# Companion line-chart metrics (mean event-patch counts across layers).
# Colours match the orange/blue used in the positional/layer two-metric sweeps
# (``TWO_METRIC_COLORS`` in ``src/cli/plot_sweep_filtered.py``).
YES_COL = "mean_patches_to_yes_among_reached"
C80_COL = "mean_patches_to_clean_confidence_recovery_80_among_reached"
COMPANION_COLORS = {
    "yes": "#d97706",  # orange
    "c80": "#2563eb",  # blue
}
COMPANION_LABELS = {
    "yes": "Mean patches to 'Yes'",
    "c80": "Mean patches to C80",
}

GG_ROOT = Path("/projects/prjs2014/patching/gradient_guided_token_patching")
# (column label, aggregate directory) — left to right.
PANELS = [
    ("LLaVA-1.5-7B", GG_ROOT / "llava-hf_llava-1-5-7b-hf/pope/random"),
    ("Qwen3.5-9B", GG_ROOT / "Qwen_Qwen3-5-9B/pope/random"),
]
SUMMARY_SUBDIR = "objective_summary_10samples_crossmodel/maximize_p_yes"

FIG_TITLE = (
    "Gradient-Guided Patching Layer Sweep: "
    "Token-Span Contributions to Confidence Recovery"
)
OUTPUT_BASE = GG_ROOT / "gradient_guided_confidence_recovery_crossmodel"

# --- figure typography / dimensions ---
FIGURE_TITLE_SIZE = 38
SUBPLOT_TITLE_SIZE = 36
AXIS_LABEL_SIZE = 34
X_TICK_LABEL_SIZE = 24
Y_TICK_LABEL_SIZE = 28
LEGEND_FONT_SIZE = 34
LEGEND_TITLE_SIZE = 12
ANNOTATION_FONT_SIZE = 11  # Reserved for figures that enable annotations.

FIGURE_WIDTH = 30
FIGURE_HEIGHT = 14


def draw_panel(ax, aggregate: pd.DataFrame, layer_summary: pd.DataFrame,
               show_xlabel: bool = True) -> None:
    """Diverging stacked bars + total line, identical to plot_stacked(signed)."""
    layers = sorted(aggregate["requested_layer"].astype(int).unique().tolist())
    positive_bottom = [0.0] * len(layers)
    negative_bottom = [0.0] * len(layers)
    for span in SPAN_ORDER:
        rows = aggregate[aggregate["token_span"] == span].set_index("requested_layer")
        positive_values = [float(rows.loc[layer, POS_COL]) for layer in layers]
        negative_values = [float(rows.loc[layer, NEG_COL]) for layer in layers]
        ax.bar(layers, positive_values, bottom=positive_bottom, width=0.78,
               color=SPAN_COLORS[span])
        ax.bar(layers, negative_values, bottom=negative_bottom, width=0.78,
               color=SPAN_COLORS[span])
        positive_bottom = [b + v for b, v in zip(positive_bottom, positive_values)]
        negative_bottom = [b + v for b, v in zip(negative_bottom, negative_values)]

    totals = layer_summary.set_index("requested_layer")
    ax.plot(layers, [float(totals.loc[layer, TOTAL_COL]) for layer in layers],
            color="#111827", linestyle="--", linewidth=1.7, marker="o", markersize=3)
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    # Full-recovery reference line (RR=1), matching the layer/positional sweeps.
    ax.axhline(1.0, color="#6b7280", linewidth=1.5, linestyle=":", alpha=0.85, zorder=2)

    # (The per-bar event-patch annotations are replaced by the companion line panels.)

    ax.set_xticks(layers)
    ax.set_xticklabels(
        [str(layer) for layer in layers],
        fontsize=X_TICK_LABEL_SIZE,
        rotation=45,
    )
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=AXIS_LABEL_SIZE)
    # Fixed recovery y-axis (0.0–1.0, step 0.2), shared across both model panels.
    # Headroom keeps the ~1.1 stacked peak and the RR=1 reference line visible.
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
    ax.set_ylim(0.0, 1.12)
    ax.grid(axis="y", alpha=0.25)


def draw_companion(ax, layer_summary: pd.DataFrame, layers: list[int],
                   show_ylabel: bool) -> None:
    """Companion line chart: mean event-patch counts (Yes / C80) across layers."""
    totals = layer_summary.set_index("requested_layer")
    yes_values = [float(totals.loc[layer, YES_COL]) for layer in layers]
    c80_values = [float(totals.loc[layer, C80_COL]) for layer in layers]
    ax.plot(layers, yes_values, color=COMPANION_COLORS["yes"], linewidth=1.7,
            marker="o", markersize=3, label=COMPANION_LABELS["yes"])
    ax.plot(layers, c80_values, color=COMPANION_COLORS["c80"], linewidth=1.7,
            marker="o", markersize=3, label=COMPANION_LABELS["c80"])

    ax.set_xticks(layers)
    ax.set_xticklabels(
        [str(layer) for layer in layers],
        fontsize=X_TICK_LABEL_SIZE,
        rotation=45,
    )
    ax.set_xlabel("Layer", fontsize=AXIS_LABEL_SIZE)
    # Fixed event-patch y-axis (ticks 1–9, step 1), shared across both model panels.
    # A little headroom below the '1' tick keeps lines at value 1 off the baseline.
    ax.set_yticks(list(range(1, 10)))
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
    ax.set_ylim(0.7, 9)
    if show_ylabel:
        ax.set_ylabel("Mean patches", fontsize=AXIS_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)


def main(show_companion: bool = True) -> None:
    if show_companion:
        # Two stacked rows per model: a main GGP panel (height 3) and a companion
        # line panel (height 1) sharing the same x-axis and column width.
        fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        gs = fig.add_gridspec(
            2, len(PANELS), height_ratios=[3, 1],
            top=0.75, bottom=0.12, left=0.045, right=0.99,
            wspace=0.16, hspace=0.55,
        )
        main_axes = [fig.add_subplot(gs[0, col]) for col in range(len(PANELS))]
        companion_axes = [
            fig.add_subplot(gs[1, col], sharex=main_axes[col])
            for col in range(len(PANELS))
        ]
    else:
        fig, axes = plt.subplots(1, len(PANELS), figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
                                 squeeze=False)
        main_axes = list(axes[0])
        companion_axes = [None] * len(PANELS)

    for col, (model_label, model_root) in enumerate(PANELS):
        summary_dir = model_root / SUMMARY_SUBDIR
        aggregate = pd.read_parquet(summary_dir / "mean_layer_span_contributions.parquet")
        layer_summary = pd.read_parquet(summary_dir / "layer_summary.parquet")
        layers = sorted(aggregate["requested_layer"].astype(int).unique().tolist())

        ax = main_axes[col]
        # When a companion panel sits below, it carries the shared "Layer" x-label.
        draw_panel(ax, aggregate, layer_summary, show_xlabel=not show_companion)
        ax.set_title(model_label, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Mean confidence recovery", fontsize=AXIS_LABEL_SIZE)
            ax.yaxis.set_label_coords(-0.07, 0.5)

        if show_companion:
            draw_companion(companion_axes[col], layer_summary, layers,
                           show_ylabel=(col == 0))
            if col == 0:
                companion_axes[col].yaxis.set_label_coords(-0.07, 0.5)

    # Shared horizontal legend (spans + total) between the title and model headers.
    legend_handles = [Patch(facecolor=SPAN_COLORS[span], label=span) for span in SPAN_ORDER]
    legend_handles.append(
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.7,
               marker="o", markersize=3, label="Total")
    )
    legend_handles.append(
        Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",
               label="Full recovery (RR=1)")
    )
    # Companion (Yes / C80) legends are drawn inside each companion panel, not here.

    fig.suptitle(FIG_TITLE, fontsize=FIGURE_TITLE_SIZE, y=0.99)
    if not show_companion:
        fig.subplots_adjust(top=0.84, bottom=0.10, left=0.045, right=0.99, wspace=0.10)
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        bbox_to_anchor=(0.5, 0.95),
    )

    if show_companion:
        companion_handles = [
            Line2D(
                [0], [0], color=COMPANION_COLORS["yes"], linewidth=1.7,
                marker="o", markersize=3, label=COMPANION_LABELS["yes"],
            ),
            Line2D(
                [0], [0], color=COMPANION_COLORS["c80"], linewidth=1.7,
                marker="o", markersize=3, label=COMPANION_LABELS["c80"],
            ),
        ]
        fig.legend(
            handles=companion_handles,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            bbox_to_anchor=(0.5, 0.31),
        )

    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"))
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {OUTPUT_BASE.with_suffix('.pdf')}")
    print(f"Saved {OUTPUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
