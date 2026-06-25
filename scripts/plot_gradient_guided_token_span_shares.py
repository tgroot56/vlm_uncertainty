"""Plot token-span shares of incremental clean-answer confidence recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


LAYERS = [7, 10, 13, 18]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to gradient_guided_selected_steps.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for the plot and computed share tables.",
    )
    return parser.parse_args()


def token_span(row: pd.Series) -> str:
    if str(row["token_type"]) == "final_prompt_token":
        return "Final prompt token"

    group = str(row["token_group"])
    if group == "text_pre":
        return "Template Prefix"
    if group == "visual":
        return "Visual"
    if group == "question":
        return "Question"
    if group == "assistant_marker":
        return "Template Suffix"
    raise ValueError(
        f"Cannot map token position {row['token_position']} with "
        f"token_group={group!r}, token_type={row['token_type']!r}."
    )


def compute_shares(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "requested_layer",
        "run_type",
        "step",
        "token_position",
        "token_group",
        "token_type",
        "p_clean_recovery",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    steps = selected[
        (selected["run_type"] == "confidence_recovery")
        & selected["requested_layer"].isin(LAYERS)
    ].copy()
    present_layers = sorted(steps["requested_layer"].astype(int).unique().tolist())
    if present_layers != LAYERS:
        raise ValueError(f"Expected layers {LAYERS}, found {present_layers}")

    steps = steps.sort_values(["requested_layer", "step"]).reset_index(drop=True)
    steps["previous_p_clean_recovery"] = (
        steps.groupby("requested_layer")["p_clean_recovery"].shift(1).fillna(0.0)
    )
    steps["incremental_p_clean_recovery"] = (
        steps["p_clean_recovery"] - steps["previous_p_clean_recovery"]
    )
    steps["token_span"] = steps.apply(token_span, axis=1)

    totals = (
        steps.groupby("requested_layer")["incremental_p_clean_recovery"]
        .sum()
        .rename("total_incremental_p_clean_recovery")
    )
    if (totals.abs() < 1e-12).any():
        bad_layers = totals[totals.abs() < 1e-12].index.astype(int).tolist()
        raise ValueError(f"Cannot compute shares because total incremental recovery is zero for {bad_layers}")

    shares = (
        steps.groupby(["requested_layer", "token_span"])["incremental_p_clean_recovery"]
        .sum()
        .rename("span_incremental_p_clean_recovery")
        .reset_index()
        .merge(totals.reset_index(), on="requested_layer", how="left")
    )
    shares["share"] = (
        shares["span_incremental_p_clean_recovery"]
        / shares["total_incremental_p_clean_recovery"]
    )

    full_index = pd.MultiIndex.from_product(
        [LAYERS, SPAN_ORDER],
        names=["requested_layer", "token_span"],
    )
    shares = (
        shares.set_index(["requested_layer", "token_span"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    for layer in LAYERS:
        total = float(totals.loc[layer])
        mask = shares["requested_layer"] == layer
        shares.loc[mask, "total_incremental_p_clean_recovery"] = total
    shares["share_percent"] = shares["share"] * 100.0
    return steps, shares


def plot_shares(shares: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for span in SPAN_ORDER:
        rows = shares[shares["token_span"] == span].sort_values("requested_layer")
        ax.plot(
            rows["requested_layer"],
            rows["share"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=SPAN_COLORS[span],
            label=span,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of incremental clean-answer confidence recovery")
    ax.set_xticks(LAYERS)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="both", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    ax.set_title("Gradient-guided patching: clean-answer confidence recovery by token span")
    fig.tight_layout()
    fig.savefig(output_dir / "token_span_incremental_recovery_share_lines.png", dpi=300)
    fig.savefig(output_dir / "token_span_incremental_recovery_share_lines.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_parquet(args.input)
    steps, shares = compute_shares(selected)

    steps.to_csv(args.output_dir / "token_span_incremental_recovery_steps.csv", index=False)
    steps.to_parquet(args.output_dir / "token_span_incremental_recovery_steps.parquet", index=False)
    shares.to_csv(args.output_dir / "token_span_incremental_recovery_shares.csv", index=False)
    shares.to_parquet(args.output_dir / "token_span_incremental_recovery_shares.parquet", index=False)
    plot_shares(shares, args.output_dir)

    print(shares.pivot(index="requested_layer", columns="token_span", values="share").to_string())


if __name__ == "__main__":
    main()
