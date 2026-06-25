"""Render filtered cross-model clean-answer confidence recovery sweeps.

Matches the existing two-metric 2x4 cross-model figures, retaining only
clean-correct/corrupted-incorrect examples and replacing generated-answer
confidence with clean-answer confidence. Environment variables can select
alternate result tags, sample limits, and output names.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import (
    DATASET_LABELS,
    SEVERITY_ORDER,
    SPAN_ORDER,
    TWO_METRIC_COLORS,
    TWO_METRIC_LEGEND_LABELS,
    _attach_baselines,
    _draw_layer_ax_two_metrics,
    _draw_positional_ax_two_metrics,
    save,
)


ROOT = Path("/projects/prjs2014/patching")
LLAVA_SAMPLE_LIMIT = int(os.environ.get("LLAVA_SAMPLE_LIMIT", "100"))
QWEN_SAMPLE_LIMIT = int(os.environ.get("QWEN_SAMPLE_LIMIT", "100"))
FILTERED_SAMPLE_LIMIT = int(os.environ.get("FILTERED_SAMPLE_LIMIT", "0"))
QWEN_REQUIRED_RETAINED = int(os.environ.get("QWEN_REQUIRED_RETAINED", "0"))
MIN_POSITION_SAMPLES = int(os.environ.get("MIN_POSITION_SAMPLES", "35"))
PLOT_KIND = os.environ.get("PLOT_KIND", "both")
DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]
DATASET_SPLITS = [
    ("vqa_coco", ["vqa-v2", "coco-qa-vi"]),
    ("imagenet_pope", ["imagenet-r", "pope"]),
]
METRICS = ["clean_answer_probability", "score"]
REFERENCE_DATASET_LABELS = dict(DATASET_LABELS)

LLAVA_SHORT = "llava-hf_llava-1-5-7b-hf"
QWEN_SHORT = "Qwen_Qwen3-5-9B"
LLAVA_POSITIONAL_TAG = "extreme_clean_answer_confidence"
LLAVA_LAYER_TAG = "extreme_clean_answer_confidence_stripped_softmax"
LLAVA_POPE_LAYER_TAG = "extreme_answer_confidence_clean_answer"
QWEN_TAG = os.environ.get(
    "QWEN_TAG",
    "extreme_clean_answer_confidence_stripped_softmax_"
    "finaltoken_qwen_prefill_empty_thinking",
)

POSITIONAL_OUTPUT = os.environ.get(
    "POSITIONAL_OUTPUT",
    "two_metrics_positional_sweep_clean_answer_confidence_"
    "clean_correct_corrupted_incorrect_n35_unclipped",
)
LAYER_OUTPUT = os.environ.get(
    "LAYER_OUTPUT",
    "two_metrics_layer_sweep_clean_answer_confidence_"
    "clean_correct_corrupted_incorrect_unclipped",
)


def dataset_component(dataset: str) -> Path:
    return Path("pope/random") if dataset == "pope" else Path(dataset)


def positional_paths(model_short: str, tag: str) -> dict[str, Path]:
    base = ROOT / "positional_patching_results_including_visual" / model_short
    return {
        dataset: base / dataset_component(dataset) / tag / "positional_patching_results.parquet"
        for dataset in DATASETS
    }


def layer_paths(model_short: str, tag: str) -> dict[str, Path]:
    base = ROOT / "layer_sweep_results" / model_short
    paths = {
        dataset: base / dataset_component(dataset) / tag / "layer_sweep_results.parquet"
        for dataset in DATASETS
    }
    if model_short == LLAVA_SHORT:
        paths["pope"] = (
            base
            / "pope/random"
            / LLAVA_POPE_LAYER_TAG
            / "layer_sweep_results.parquet"
        )
    return paths


def load_filtered_rows(
    path: Path,
    extra_columns: list[str],
    sample_limit: int,
) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    columns = list(
        dict.fromkeys(
            column
            for column in [
                "sample_id",
                "severity",
                "condition",
                "score",
                "clean_answer_probability",
                *extra_columns,
            ]
            if column in available
        )
    )
    frame = pd.read_parquet(path, columns=columns)
    first_ids = list(pd.unique(frame["sample_id"]))[:sample_limit]
    frame = frame[frame["sample_id"].isin(first_ids)].copy()

    clean = (
        frame[frame["condition"].eq("clean")][["sample_id", "severity", "score"]]
        .drop_duplicates(["sample_id", "severity"])
        .rename(columns={"score": "clean_filter_score"})
    )
    corrupted = (
        frame[frame["condition"].eq("degraded")][["sample_id", "severity", "score"]]
        .drop_duplicates(["sample_id", "severity"])
        .rename(columns={"score": "corrupted_filter_score"})
    )
    keys = clean.merge(corrupted, on=["sample_id", "severity"])
    keys = keys[
        keys["clean_filter_score"].astype(float).gt(0)
        & keys["corrupted_filter_score"].astype(float).eq(0)
    ][["sample_id", "severity"]]
    filtered = frame.merge(keys, on=["sample_id", "severity"], how="inner")
    if FILTERED_SAMPLE_LIMIT:
        retained_ids = list(pd.unique(filtered["sample_id"]))[:FILTERED_SAMPLE_LIMIT]
        filtered = filtered[filtered["sample_id"].isin(retained_ids)]
    return filtered


def prepare_layer(
    path: Path,
    sample_limit: int,
) -> tuple[pd.DataFrame, list[int], list[str], int]:
    frame = load_filtered_rows(
        path,
        ["layer", "position_offset", "span_label"],
        sample_limit,
    )
    patched = frame[
        frame["condition"].eq("patched")
        & frame["position_offset"].eq(-1)
        & frame["span_label"].isna()
    ].copy()
    patched = _attach_baselines(patched, frame, metrics=METRICS)
    layers = sorted(patched["layer"].dropna().astype(int).unique())
    severities = [s for s in SEVERITY_ORDER if s in patched["severity"].values]
    return patched, layers, severities, patched["sample_id"].nunique()


def prepare_positional(
    path: Path,
    sample_limit: int,
) -> tuple[pd.DataFrame, list[int], dict[str, list[int]], list[str], int]:
    frame = load_filtered_rows(
        path,
        ["token_position", "span_label"],
        sample_limit,
    )
    patched = frame[frame["condition"].str.startswith("patched", na=False)].copy()
    patched = _attach_baselines(patched, frame, metrics=METRICS)
    retained_examples = patched["sample_id"].nunique()

    span_starts = (
        patched.groupby(["sample_id", "span_label"])["token_position"]
        .min()
        .reset_index(name="span_start")
    )
    patched = patched.merge(span_starts, on=["sample_id", "span_label"])
    patched["span_rel_pos"] = (
        patched["token_position"] - patched["span_start"]
    ).astype(int)

    counts = (
        patched.groupby(["span_label", "span_rel_pos", "severity"])["sample_id"]
        .nunique()
        .reset_index(name="n")
    )
    min_counts = counts.groupby(["span_label", "span_rel_pos"])["n"].min()
    valid = {
        (span, rel_pos)
        for (span, rel_pos), count in min_counts.items()
        if count >= MIN_POSITION_SAMPLES
    }
    if not valid:
        severities = [s for s in SEVERITY_ORDER if s in patched["severity"].values]
        return patched.iloc[0:0], [], {}, severities, retained_examples
    patched = patched[
        patched.apply(
            lambda row: (row["span_label"], row["span_rel_pos"]) in valid,
            axis=1,
        )
    ]

    canonical_rows: list[dict[str, int | str]] = []
    span_regions: dict[str, list[int]] = {}
    x = 0
    for span in SPAN_ORDER:
        relative_positions = sorted(rel for label, rel in valid if label == span)
        if not relative_positions:
            continue
        xs = list(range(x, x + len(relative_positions)))
        span_regions[span] = xs
        canonical_rows.extend(
            {
                "span_label": span,
                "span_rel_pos": relative_position,
                "canonical_x": canonical_x,
            }
            for relative_position, canonical_x in zip(relative_positions, xs)
        )
        x += len(relative_positions) + 1

    patched = patched.merge(
        pd.DataFrame(canonical_rows),
        on=["span_label", "span_rel_pos"],
        how="inner",
    )
    canonical_xs = sorted(patched["canonical_x"].unique().astype(int))
    severities = [s for s in SEVERITY_ORDER if s in patched["severity"].values]
    return (
        patched,
        canonical_xs,
        span_regions,
        severities,
        retained_examples,
    )


def configure_reference_confidence_style() -> None:
    TWO_METRIC_COLORS["clean_answer_probability"] = TWO_METRIC_COLORS["top1_probability"]
    TWO_METRIC_LEGEND_LABELS["clean_answer_probability"] = "Confidence recovery"


def validate_retained(model_label: str, dataset: str, retained: int) -> None:
    if (
        model_label == "Qwen3.5-9B"
        and QWEN_REQUIRED_RETAINED
        and retained != QWEN_REQUIRED_RETAINED
    ):
        raise RuntimeError(
            f"{dataset}: retained {retained} Qwen samples after filtering; "
            f"expected exactly {QWEN_REQUIRED_RETAINED}"
        )


def render_layer(
    model_entries: list[tuple[str, dict[str, Path], int]],
    datasets: list[str] | None = None,
    output_name: str = LAYER_OUTPUT,
    *,
    style: dict[str, float] | None = None,
    text: dict[str, object] | None = None,
    layout: dict[str, float] | None = None,
) -> None:
    if datasets is None:
        datasets = DATASETS
    style = style or {}
    text = text or {}
    layout = layout or {}
    subplot_titles = text.get("subplot_titles", REFERENCE_DATASET_LABELS)
    model_labels = text.get("model_labels", {})
    y_label_template = text.get("y_label_template", "{model_label}\nMean recovery")
    n_models = len(model_entries)
    n_datasets = len(datasets)
    fig, axes = plt.subplots(
        n_models,
        n_datasets,
        figsize=(
            layout.get("figure_width", 5.5 * n_datasets),
            layout.get("figure_height", 4.5 * n_models),
        ),
        squeeze=False,
    )
    legend_handles = None
    for row, (model_label, paths, sample_limit) in enumerate(model_entries):
        for col, dataset in enumerate(datasets):
            frame, layers, severities, retained = prepare_layer(
                paths[dataset],
                sample_limit,
            )
            validate_retained(model_label, dataset, retained)
            print(f"layer      {model_label:13s} {dataset:10s}: {retained} retained")
            handles = _draw_layer_ax_two_metrics(
                axes[row, col],
                frame,
                layers,
                severities,
                show_xlabel=row == n_models - 1,
                show_ylabel=col == 0,
                metrics=METRICS,
                clip_recovery=False,
                style=style,
                text=text,
                layout=layout,
            )
            legend_handles = legend_handles or handles
            axes[row, col].set_title(
                subplot_titles.get(dataset, REFERENCE_DATASET_LABELS.get(dataset, dataset)) if row == 0 else "",
                fontsize=style.get("subplot_title_size", 18),
                fontweight="bold",
            )
            if col == 0:
                display_model_label = model_labels.get(model_label, model_label)
                axes[row, col].set_ylabel(
                    y_label_template.format(model_label=display_model_label),
                    fontsize=style.get("axis_label_size", 14),
                )

    fig.suptitle(
        text.get("figure_title", "Patching Layer Sweep - Confidence & Accuracy Recovery"),
        fontsize=style.get("figure_title_size", 22),
        y=1.03,
    )
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=style.get("legend_font_size", 12),
        title=text.get("legend_title", None),
        title_fontsize=style.get("legend_title_size", style.get("legend_font_size", 12)),
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if layout.get("wspace", None) is not None or layout.get("hspace", None) is not None:
        fig.subplots_adjust(wspace=layout.get("wspace", None), hspace=layout.get("hspace", None))
    save(fig, ROOT / "layer_sweep_results", output_name)


def _center_and_stagger_span_tick_labels(
    axes,
    *,
    stagger_overlaps: bool,
    stagger_step: float,
) -> int:
    """Center span labels and move horizontally colliding labels to lower rows."""
    bottom_axes = list(axes[-1, :])
    labels_by_axis = []
    for ax in bottom_axes:
        labels = [label for label in ax.get_xticklabels() if label.get_visible() and label.get_text()]
        for label in labels:
            label.set_ha("center")
        labels_by_axis.append(labels)

    axes[0, 0].figure.canvas.draw()
    renderer = axes[0, 0].figure.canvas.get_renderer()
    maximum_level = 0
    for labels in labels_by_axis:
        if not labels or not stagger_overlaps:
            continue
        base_y = labels[0].get_position()[1]
        right_edges_by_level: list[float] = []
        for label in labels:
            label.set_y(base_y)
        axes[0, 0].figure.canvas.draw()
        boxes = [label.get_window_extent(renderer) for label in labels]
        for label, box in sorted(zip(labels, boxes), key=lambda item: item[1].x0):
            level = next(
                (
                    candidate
                    for candidate, right_edge in enumerate(right_edges_by_level)
                    if box.x0 >= right_edge + 4.0
                ),
                len(right_edges_by_level),
            )
            if level == len(right_edges_by_level):
                right_edges_by_level.append(box.x1)
            else:
                right_edges_by_level[level] = box.x1
            label.set_y(base_y - level * stagger_step)
            maximum_level = max(maximum_level, level)
    return maximum_level


def render_positional(
    model_entries: list[tuple[str, dict[str, Path], int]],
    datasets: list[str] | None = None,
    output_name: str = POSITIONAL_OUTPUT,
    *,
    style: dict[str, float] | None = None,
    text: dict[str, object] | None = None,
    layout: dict[str, float] | None = None,
) -> None:
    if datasets is None:
        datasets = DATASETS
    style = style or {}
    text = text or {}
    layout = layout or {}
    subplot_titles = text.get("subplot_titles", REFERENCE_DATASET_LABELS)
    model_labels = text.get("model_labels", {})
    y_label_template = text.get("y_label_template", "{model_label}\nMean recovery")
    n_models = len(model_entries)
    n_datasets = len(datasets)
    fig, axes = plt.subplots(
        n_models,
        n_datasets,
        figsize=(
            layout.get("figure_width", 5.8 * n_datasets),
            layout.get("figure_height", 4.6 * n_models),
        ),
        squeeze=False,
    )
    legend_handles = None
    for row, (model_label, paths, sample_limit) in enumerate(model_entries):
        for col, dataset in enumerate(datasets):
            frame, xs, spans, severities, retained = prepare_positional(
                paths[dataset],
                sample_limit,
            )
            validate_retained(model_label, dataset, retained)
            print(f"positional {model_label:13s} {dataset:10s}: {retained} retained")
            ax = axes[row, col]
            if frame.empty or not xs:
                ax.text(
                    0.5,
                    0.5,
                    text.get("data_not_available", "Data not available"),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=style.get("annotation_font_size", 11),
                    color="#9ca3af",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                handles = _draw_positional_ax_two_metrics(
                    ax,
                    frame,
                    xs,
                    spans,
                    severities,
                    show_xlabel=row == n_models - 1,
                    show_ylabel=col == 0,
                    metrics=METRICS,
                    clip_recovery=False,
                    style=style,
                    text=text,
                    layout=layout,
                )
                legend_handles = legend_handles or handles
            axes[row, col].set_title(
                subplot_titles.get(dataset, REFERENCE_DATASET_LABELS.get(dataset, dataset)) if row == 0 else "",
                fontsize=style.get("subplot_title_size", 18),
                fontweight="bold",
            )
            if col == 0:
                display_model_label = model_labels.get(model_label, model_label)
                axes[row, col].set_ylabel(
                    y_label_template.format(model_label=display_model_label),
                    fontsize=style.get("axis_label_size", 14),
                )

    fig.suptitle(
        text.get("figure_title", "Patching Positional Sweep - Confidence & Accuracy Recovery"),
        fontsize=style.get("figure_title_size", 22),
        y=1.03,
    )
    legend_kwargs = {
        "handles": legend_handles,
        "labels": [handle.get_label() for handle in legend_handles],
        "loc": "upper center",
        "frameon": False,
        "fontsize": style.get("legend_font_size", 11),
        "title": text.get("legend_title", None),
        "title_fontsize": style.get("legend_title_size", style.get("legend_font_size", 11)),
        "bbox_to_anchor": (0.5, 0.99),
    }
    requested_legend_columns = layout.get("legend_columns", 7)
    legend_columns = min(len(legend_handles), max(1, int(requested_legend_columns)))
    legend = fig.legend(ncol=legend_columns, **legend_kwargs)
    layout_top = 0.96
    if layout.get("auto_wrap_legend", False):
        max_legend_width = fig.bbox.width * layout.get("legend_max_width_fraction", 0.94)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        while legend_columns > 1 and legend.get_window_extent(renderer).width > max_legend_width:
            legend.remove()
            legend_columns -= 1
            legend = fig.legend(ncol=legend_columns, **legend_kwargs)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        if legend_columns < len(legend_handles):
            legend_box = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            layout_top = min(layout_top, legend_box.y0 - 0.015)
    fig.tight_layout(rect=(0, 0, 1, layout_top))
    if layout.get("wspace", None) is not None or layout.get("hspace", None) is not None:
        fig.subplots_adjust(wspace=layout.get("wspace", None), hspace=layout.get("hspace", None))
    if layout.get("center_span_labels", False) or layout.get("stagger_overlapping_span_labels", False):
        _center_and_stagger_span_tick_labels(
            axes,
            stagger_overlaps=layout.get("stagger_overlapping_span_labels", False),
            stagger_step=layout.get("span_label_stagger_step", 0.08),
        )
    save(
        fig,
        ROOT / "positional_patching_results_including_visual",
        output_name,
    )


def split_output_name(base_name: str, split_name: str) -> str:
    return f"{base_name}_{split_name}"


def main() -> None:
    configure_reference_confidence_style()
    positional_entries = [
        (
            "LLaVA-1.5-7B",
            positional_paths(LLAVA_SHORT, LLAVA_POSITIONAL_TAG),
            LLAVA_SAMPLE_LIMIT,
        ),
        ("Qwen3.5-9B", positional_paths(QWEN_SHORT, QWEN_TAG), QWEN_SAMPLE_LIMIT),
    ]
    layer_entries = [
        (
            "LLaVA-1.5-7B",
            layer_paths(LLAVA_SHORT, LLAVA_LAYER_TAG),
            LLAVA_SAMPLE_LIMIT,
        ),
        ("Qwen3.5-9B", layer_paths(QWEN_SHORT, QWEN_TAG), QWEN_SAMPLE_LIMIT),
    ]
    if PLOT_KIND in {"both", "positional"}:
        render_positional(positional_entries)
        for split_name, datasets in DATASET_SPLITS:
            render_positional(
                positional_entries,
                datasets=datasets,
                output_name=split_output_name(POSITIONAL_OUTPUT, split_name),
            )
    if PLOT_KIND in {"both", "layer"}:
        render_layer(layer_entries)
        for split_name, datasets in DATASET_SPLITS:
            render_layer(
                layer_entries,
                datasets=datasets,
                output_name=split_output_name(LAYER_OUTPUT, split_name),
            )


if __name__ == "__main__":
    main()
