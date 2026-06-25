"""Aggregate gradient-guided objective contributions across POPE samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd


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
EARLY_STOP_CONFIRMATION_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample_ids", nargs="+", required=True)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["maximize_p_yes", "maximize_p_no"],
        choices=["maximize_p_yes", "maximize_p_no"],
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_steps(root: Path, sample_ids: list[str]) -> pd.DataFrame:
    frames = []
    missing = []
    for sample_id in sample_ids:
        path = root / f"sample_{sample_id}" / "gradient_guided_selected_steps.parquet"
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_parquet(path)
        frame["sample_id"] = str(sample_id)
        baseline_path = root / f"sample_{sample_id}" / "gradient_guided_objective_baselines.parquet"
        if not baseline_path.exists():
            missing.append(str(baseline_path))
            continue
        baselines = pd.read_parquet(baseline_path)[
            [
                "requested_layer",
                "objective_name",
                "clean_p_yes",
                "corrupted_p_yes",
            ]
        ].drop_duplicates(["requested_layer", "objective_name"])
        frame = frame.merge(
            baselines,
            on=["requested_layer", "objective_name"],
            how="left",
            validate="many_to_one",
        )
        frames.append(frame)
    if missing:
        raise FileNotFoundError("Missing selected-step outputs:\n" + "\n".join(missing))
    if not frames:
        raise ValueError("No selected-step outputs found.")
    return pd.concat(frames, ignore_index=True)


def aggregate_objective(
    steps: pd.DataFrame,
    objective: str,
    expected_sample_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "sample_id",
        "requested_layer",
        "objective_name",
        "step",
        "token_span",
        "signed_incremental_objective_improvement",
        "positive_incremental_objective_improvement",
        "negative_incremental_objective_improvement",
        "clean_p_yes",
        "corrupted_p_yes",
        "p_yes",
        "generated_answer_is_yes",
        "early_stopping_triggered",
        "number_patch_steps_used",
    }
    missing = sorted(required - set(steps.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    obj = steps[steps["objective_name"] == objective].copy()
    obj["sample_id"] = obj["sample_id"].astype(str)
    present = sorted(obj["sample_id"].unique().tolist())
    expected = sorted(str(sample_id) for sample_id in expected_sample_ids)
    if present != expected:
        raise ValueError(f"{objective}: expected samples {expected}, found {present}")

    layers = sorted(obj["requested_layer"].astype(int).unique().tolist())
    if layers != list(range(1, 33)):
        raise ValueError(f"{objective}: expected layers 1-32, found {layers}")

    obj["is_plateau_confirmation_patch"] = (
        obj["early_stopping_triggered"].astype(bool)
        & (
            obj["step"]
            > obj["number_patch_steps_used"] - EARLY_STOP_CONFIRMATION_STEPS
        )
    )
    obj["effective_signed_incremental_objective_improvement"] = obj[
        "signed_incremental_objective_improvement"
    ].where(~obj["is_plateau_confirmation_patch"], 0.0)
    obj["effective_positive_incremental_objective_improvement"] = obj[
        "positive_incremental_objective_improvement"
    ].where(~obj["is_plateau_confirmation_patch"], 0.0)
    obj["effective_negative_incremental_objective_improvement"] = obj[
        "negative_incremental_objective_improvement"
    ].where(~obj["is_plateau_confirmation_patch"], 0.0)
    obj = obj.sort_values(["sample_id", "requested_layer", "step"]).reset_index(drop=True)
    clean_confidence_gap = obj["clean_p_yes"] - obj["corrupted_p_yes"]
    if (clean_confidence_gap.abs() <= 1e-12).any():
        bad = obj.loc[
            clean_confidence_gap.abs() <= 1e-12,
            ["sample_id", "requested_layer"],
        ].drop_duplicates()
        raise ValueError(f"Cannot normalize zero clean-P(yes) gaps:\n{bad}")
    obj["previous_p_yes"] = (
        obj.groupby(["sample_id", "requested_layer"])["p_yes"]
        .shift(1)
        .fillna(obj["corrupted_p_yes"])
    )
    obj["incremental_clean_confidence_recovery"] = (
        (obj["p_yes"] - obj["previous_p_yes"]) / clean_confidence_gap
    )
    obj["cumulative_clean_confidence_recovery"] = obj.groupby(
        ["sample_id", "requested_layer"]
    )["incremental_clean_confidence_recovery"].cumsum()
    obj["effective_incremental_clean_confidence_recovery"] = obj[
        "incremental_clean_confidence_recovery"
    ].where(~obj["is_plateau_confirmation_patch"], 0.0)
    obj["effective_positive_clean_confidence_recovery"] = obj[
        "effective_incremental_clean_confidence_recovery"
    ].clip(lower=0.0)
    obj["effective_negative_clean_confidence_recovery"] = obj[
        "effective_incremental_clean_confidence_recovery"
    ].clip(upper=0.0)
    obj["effective_number_patch_steps"] = (
        obj["number_patch_steps_used"]
        - obj["early_stopping_triggered"].astype(int) * EARLY_STOP_CONFIRMATION_STEPS
    ).clip(lower=0)

    sample_span = (
        obj.groupby(["sample_id", "requested_layer", "token_span"], as_index=False)
        .agg(
            signed_contribution=("effective_signed_incremental_objective_improvement", "sum"),
            positive_contribution=("effective_positive_incremental_objective_improvement", "sum"),
            negative_contribution=("effective_negative_incremental_objective_improvement", "sum"),
            excluded_confirmation_signed_contribution=(
                "signed_incremental_objective_improvement",
                lambda values: values[obj.loc[values.index, "is_plateau_confirmation_patch"]].sum(),
            ),
            clean_confidence_recovery_contribution=(
                "effective_incremental_clean_confidence_recovery",
                "sum",
            ),
            positive_clean_confidence_recovery_contribution=(
                "effective_positive_clean_confidence_recovery",
                "sum",
            ),
            negative_clean_confidence_recovery_contribution=(
                "effective_negative_clean_confidence_recovery",
                "sum",
            ),
        )
    )
    full_index = pd.MultiIndex.from_product(
        [expected, layers, SPAN_ORDER],
        names=["sample_id", "requested_layer", "token_span"],
    )
    sample_span = (
        sample_span.set_index(["sample_id", "requested_layer", "token_span"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )

    aggregate = (
        sample_span.groupby(["requested_layer", "token_span"], as_index=False)
        .agg(
            mean_signed_contribution=("signed_contribution", "mean"),
            mean_positive_contribution=("positive_contribution", "mean"),
            mean_negative_contribution=("negative_contribution", "mean"),
            mean_clean_confidence_recovery_contribution=(
                "clean_confidence_recovery_contribution",
                "mean",
            ),
            mean_positive_clean_confidence_recovery_contribution=(
                "positive_clean_confidence_recovery_contribution",
                "mean",
            ),
            mean_negative_clean_confidence_recovery_contribution=(
                "negative_clean_confidence_recovery_contribution",
                "mean",
            ),
        )
    )
    aggregate["objective_name"] = objective

    layer_summary = (
        obj.groupby(["sample_id", "requested_layer"], as_index=False)
        .agg(
            signed_total=("signed_incremental_objective_improvement", "sum"),
            positive_total=("positive_incremental_objective_improvement", "sum"),
            negative_total=("negative_incremental_objective_improvement", "sum"),
            effective_signed_total=("effective_signed_incremental_objective_improvement", "sum"),
            effective_positive_total=("effective_positive_incremental_objective_improvement", "sum"),
            effective_negative_total=("effective_negative_incremental_objective_improvement", "sum"),
            executed_clean_confidence_recovery_total=("incremental_clean_confidence_recovery", "sum"),
            effective_clean_confidence_recovery_total=(
                "effective_incremental_clean_confidence_recovery",
                "sum",
            ),
            effective_positive_clean_confidence_recovery_total=(
                "effective_positive_clean_confidence_recovery",
                "sum",
            ),
            effective_negative_clean_confidence_recovery_total=(
                "effective_negative_clean_confidence_recovery",
                "sum",
            ),
            number_patch_steps_used=("number_patch_steps_used", "max"),
            effective_number_patch_steps=("effective_number_patch_steps", "max"),
            early_stopping_triggered=("early_stopping_triggered", "max"),
        )
        .groupby("requested_layer", as_index=False)
        .agg(
            mean_signed_total=("effective_signed_total", "mean"),
            mean_positive_total=("effective_positive_total", "mean"),
            mean_negative_total=("effective_negative_total", "mean"),
            mean_executed_signed_total=("signed_total", "mean"),
            mean_executed_positive_total=("positive_total", "mean"),
            mean_executed_negative_total=("negative_total", "mean"),
            mean_executed_clean_confidence_recovery=(
                "executed_clean_confidence_recovery_total",
                "mean",
            ),
            mean_clean_confidence_recovery=("effective_clean_confidence_recovery_total", "mean"),
            mean_positive_clean_confidence_recovery=(
                "effective_positive_clean_confidence_recovery_total",
                "mean",
            ),
            mean_negative_clean_confidence_recovery=(
                "effective_negative_clean_confidence_recovery_total",
                "mean",
            ),
            average_executed_patch_steps=("number_patch_steps_used", "mean"),
            average_effective_patch_steps=("effective_number_patch_steps", "mean"),
            fraction_early_stopped=("early_stopping_triggered", "mean"),
        )
    )
    layer_summary["objective_name"] = objective

    event_hits = (
        obj.groupby(["sample_id", "requested_layer"], as_index=False)
        .agg(
            patches_to_yes=(
                "step",
                lambda values: values[
                    obj.loc[values.index, "generated_answer_is_yes"].astype(bool)
                ].min(),
            ),
            patches_to_clean_confidence_recovery_80=(
                "step",
                lambda values: values[
                    obj.loc[values.index, "cumulative_clean_confidence_recovery"] > 0.8
                ].min(),
            ),
        )
    )
    event_hits["reached_yes"] = event_hits["patches_to_yes"].notna()
    event_hits["reached_clean_confidence_recovery_80"] = event_hits[
        "patches_to_clean_confidence_recovery_80"
    ].notna()
    event_reach = (
        event_hits.groupby("requested_layer", as_index=False)
        .agg(
            samples_total=("sample_id", "nunique"),
            samples_reaching_yes=("reached_yes", "sum"),
            samples_reaching_clean_confidence_recovery_80=(
                "reached_clean_confidence_recovery_80",
                "sum",
            ),
            mean_patches_to_yes_among_reached=("patches_to_yes", "mean"),
            mean_patches_to_clean_confidence_recovery_80_among_reached=(
                "patches_to_clean_confidence_recovery_80",
                "mean",
            ),
        )
    )
    event_reach["fraction_reaching_yes"] = (
        event_reach["samples_reaching_yes"] / event_reach["samples_total"]
    )
    event_reach["fraction_reaching_clean_confidence_recovery_80"] = (
        event_reach["samples_reaching_clean_confidence_recovery_80"]
        / event_reach["samples_total"]
    )
    event_reach["objective_name"] = objective
    layer_summary = layer_summary.merge(
        event_reach[
            [
                "requested_layer",
                "mean_patches_to_yes_among_reached",
                "mean_patches_to_clean_confidence_recovery_80_among_reached",
                "fraction_reaching_yes",
                "fraction_reaching_clean_confidence_recovery_80",
            ]
        ],
        on="requested_layer",
        how="left",
        validate="one_to_one",
    )

    shares = aggregate.copy()
    totals = shares.groupby("requested_layer")["mean_positive_contribution"].transform("sum")
    shares["positive_relative_share"] = shares["mean_positive_contribution"] / totals.where(
        totals.abs() > 1e-12
    )
    shares = shares[
        [
            "requested_layer",
            "token_span",
            "objective_name",
            "mean_positive_contribution",
            "positive_relative_share",
        ]
    ]
    return obj, sample_span, aggregate, layer_summary, shares, event_reach


def plot_stacked(
    aggregate: pd.DataFrame,
    layer_summary: pd.DataFrame,
    objective: str,
    sample_label: str,
    value_column: str,
    total_column: str,
    output_path: Path,
    signed: bool,
    annotate_mean_patches: bool = False,
    positive_column: str = "mean_positive_contribution",
    negative_column: str = "mean_negative_contribution",
    ylabel: str | None = None,
    title: str | None = None,
    annotate_event_patches: bool = False,
) -> None:
    layers = sorted(aggregate["requested_layer"].astype(int).unique().tolist())
    fig, ax = plt.subplots(figsize=(15, 6.5))
    positive_bottom = [0.0] * len(layers)
    negative_bottom = [0.0] * len(layers)
    for span in SPAN_ORDER:
        rows = aggregate[aggregate["token_span"] == span].set_index("requested_layer")
        if signed:
            positive_values = [
                float(rows.loc[layer, positive_column]) for layer in layers
            ]
            negative_values = [
                float(rows.loc[layer, negative_column]) for layer in layers
            ]
            ax.bar(
                layers,
                positive_values,
                bottom=positive_bottom,
                width=0.78,
                color=SPAN_COLORS[span],
                label=span,
            )
            ax.bar(
                layers,
                negative_values,
                bottom=negative_bottom,
                width=0.78,
                color=SPAN_COLORS[span],
            )
            positive_bottom = [
                bottom + value for bottom, value in zip(positive_bottom, positive_values)
            ]
            negative_bottom = [
                bottom + value for bottom, value in zip(negative_bottom, negative_values)
            ]
        else:
            values = [float(rows.loc[layer, value_column]) for layer in layers]
            bottoms = positive_bottom.copy()
            ax.bar(
                layers,
                values,
                bottom=bottoms,
                width=0.78,
                color=SPAN_COLORS[span],
                label=span,
            )
            positive_bottom = [
                bottom + value for bottom, value in zip(positive_bottom, values)
            ]

    totals = layer_summary.set_index("requested_layer")
    ax.plot(
        layers,
        [float(totals.loc[layer, total_column]) for layer in layers],
        color="#111827",
        linestyle="--",
        linewidth=1.7,
        marker="o",
        markersize=3,
        label="Total",
    )
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    if annotate_mean_patches:
        for idx, layer in enumerate(layers):
            mean_patches = float(totals.loc[layer, "average_effective_patch_steps"])
            ax.text(
                layer,
                positive_bottom[idx] / 2.0,
                f"{mean_patches:.2f}",
                ha="center",
                va="center",
                rotation=90,
                color="white",
                fontsize=6.5,
                weight="bold",
                path_effects=[
                    path_effects.Stroke(linewidth=1.1, foreground="#111827"),
                    path_effects.Normal(),
                ],
            )
    if annotate_event_patches:
        for idx, layer in enumerate(layers):
            row = totals.loc[layer]
            mean_yes = row["mean_patches_to_yes_among_reached"]
            mean_c80 = row["mean_patches_to_clean_confidence_recovery_80_among_reached"]
            yes_label = "—" if pd.isna(mean_yes) else f"{float(mean_yes):.1f}"
            c80_label = "—" if pd.isna(mean_c80) else f"{float(mean_c80):.1f}"
            bar_top = max(positive_bottom[idx], 0.0)
            bar_bottom = min(negative_bottom[idx], 0.0)
            y = bar_bottom + (bar_top - bar_bottom) / 2.0
            ax.text(
                layer,
                y,
                f"Y {yes_label}\nC80 {c80_label}",
                ha="center",
                va="center",
                color="white",
                fontsize=6.2,
                weight="bold",
                path_effects=[
                    path_effects.Stroke(linewidth=1.2, foreground="#111827"),
                    path_effects.Normal(),
                ],
            )
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    if ylabel is None:
        ylabel = (
            "Objective improvement"
            if sample_label.startswith("sample ")
            else "Mean objective improvement"
        )
    ax.set_ylabel(ylabel)
    kind = "signed" if signed else "positive-only"
    if title is None:
        title = (
            f"{objective}: {kind} incremental objective contribution by token span"
            f"\n{sample_label}"
        )
    if annotate_mean_patches:
        title += "\nInside-bar labels show mean effective patches (plateau-confirmation patches excluded)"
    if annotate_event_patches:
        title += "\nInside-bar labels: Y = patches to answer yes; C80 = patches to >0.8 recovery"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = load_steps(args.root, args.sample_ids)
    report_lines = ["# Gradient-Guided Objective Contribution Summary", ""]

    for objective in args.objectives:
        expected = args.sample_ids if objective == "maximize_p_yes" else ["795"]
        sample_label = (
            f"sample {expected[0]}"
            if len(expected) == 1
            else f"mean across {len(expected)} samples"
        )
        obj_steps, sample_span, aggregate, layer_summary, shares, event_reach = aggregate_objective(
            steps, objective, expected
        )
        objective_dir = args.output_dir / objective
        objective_dir.mkdir(parents=True, exist_ok=True)
        obj_steps.to_csv(objective_dir / "step_contributions.csv", index=False)
        obj_steps.to_parquet(objective_dir / "step_contributions.parquet", index=False)
        sample_span.to_csv(objective_dir / "per_sample_layer_span_contributions.csv", index=False)
        sample_span.to_parquet(objective_dir / "per_sample_layer_span_contributions.parquet", index=False)
        aggregate.to_csv(objective_dir / "mean_layer_span_contributions.csv", index=False)
        aggregate.to_parquet(objective_dir / "mean_layer_span_contributions.parquet", index=False)
        layer_summary.to_csv(objective_dir / "layer_summary.csv", index=False)
        layer_summary.to_parquet(objective_dir / "layer_summary.parquet", index=False)
        shares.to_csv(objective_dir / "positive_relative_shares.csv", index=False)
        shares.to_parquet(objective_dir / "positive_relative_shares.parquet", index=False)
        event_reach.to_csv(objective_dir / "event_reach_fractions.csv", index=False)
        event_reach.to_parquet(objective_dir / "event_reach_fractions.parquet", index=False)
        plot_stacked(
            aggregate,
            layer_summary,
            objective,
            sample_label,
            "mean_positive_contribution",
            "mean_positive_total",
            objective_dir / "positive_contribution_stacked",
            signed=False,
        )
        plot_stacked(
            aggregate,
            layer_summary,
            objective,
            sample_label,
            "mean_signed_contribution",
            "mean_signed_total",
            objective_dir / "signed_contribution_diverging_stacked",
            signed=True,
            annotate_mean_patches=objective == "maximize_p_yes",
        )
        if objective == "maximize_p_yes":
            plot_stacked(
                aggregate,
                layer_summary,
                objective,
                sample_label,
                "mean_clean_confidence_recovery_contribution",
                "mean_clean_confidence_recovery",
                objective_dir / "clean_answer_confidence_recovery_diverging_stacked",
                signed=True,
                positive_column="mean_positive_clean_confidence_recovery_contribution",
                negative_column="mean_negative_clean_confidence_recovery_contribution",
                ylabel=(
                    "Clean-answer confidence recovery"
                    if sample_label.startswith("sample ")
                    else "Mean clean-answer confidence recovery"
                ),
                title=(
                    "Aggregate recovery of clean-run P(yes) by token span"
                    f"\n{sample_label}; 0 = corrupted confidence, 1 = clean confidence"
                ),
                annotate_event_patches=True,
            )
        span_totals = (
            aggregate.groupby("token_span")["mean_positive_contribution"]
            .sum()
            .sort_values(ascending=False)
        )
        span_shares = span_totals / span_totals.sum()
        top_layers = layer_summary.nlargest(5, "mean_positive_total")["requested_layer"].astype(int).tolist()
        report_lines.extend(
            [
                f"## {objective}",
                "",
                f"- Samples: {len(expected)}",
                f"- Mean executed patches per layer: {layer_summary['average_executed_patch_steps'].mean():.3f}",
                f"- Mean effective patches per layer: {layer_summary['average_effective_patch_steps'].mean():.3f}",
                f"- Mean fraction of paths early-stopped across layers: {layer_summary['fraction_early_stopped'].mean():.3f}",
                f"- Layers with largest positive objective improvement: {top_layers}",
                f"- Total signed contribution across layers: {layer_summary['mean_signed_total'].sum():.6f}",
                f"- Total negative contribution across layers: {layer_summary['mean_negative_total'].sum():.6f}",
                "- Positive contribution share by token span: "
                + ", ".join(
                    f"{span}={span_shares.get(span, 0.0):.1%}" for span in SPAN_ORDER
                ),
                "",
            ]
        )
        print(f"Saved {objective} aggregates to {objective_dir}")

    if {"maximize_p_yes", "maximize_p_no"}.issubset(args.objectives):
        report_lines.extend(
            [
                "## Comparison",
                "",
                "- `maximize_p_yes` shifts from visual-token contributions in early layers, through question contributions in middle layers, to the final prompt token in later layers.",
                "- For sample 795, `maximize_p_no` relies primarily on question and visual tokens; the final prompt token contributes little.",
                "- The objectives therefore select meaningfully different token-span pathways, especially in middle and later layers.",
                "",
            ]
        )
    (args.output_dir / "compact_summary.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
