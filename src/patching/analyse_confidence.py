"""Identify patching conditions that boost confidence without recovering accuracy.

"Confidently wrong" = patching clean activations into a degraded forward pass
raises top-1 probability (and lowers entropy) but the model still predicts the
wrong answer.  These conditions are pinpointing *where* clean information
enters and actively reinforces the incorrect degraded prediction rather than
correcting it.

Outcome taxonomy per (sample_id, severity, condition):
  correctly_recovered : score recovered  AND  top1_prob increased
  confidently_wrong   : score NOT recovered AND top1_prob increased AND entropy decreased
  neutral_wrong       : score NOT recovered AND no meaningful confidence change
  neutral_correct     : score recovered    AND no meaningful confidence change

Usage:
    python -m src.patching.analyse_confidence \\
        --parquet /path/to/patching_results.parquet \\
        [--score_threshold 1.0] \\
        [--top1_delta_min 0.01] \\
        [--entropy_delta_max -0.01]
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd


# Conditions that are baselines — skip when computing outcomes
_BASELINE_CONDITIONS = {"clean", "degraded"}

OUTCOME_ORDER = [
    "correctly_recovered",
    "confidently_wrong",
    "neutral_correct",
    "neutral_wrong",
]


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_outcomes(
    df: pd.DataFrame,
    score_threshold: float = 1.0,
    top1_delta_min: float = 0.01,
    entropy_delta_max: float = -0.01,
) -> pd.DataFrame:
    """Return a DataFrame with one row per (sample_id, severity, condition)
    labelled by outcome, plus delta columns relative to the degraded baseline.

    Args:
        df: patching_results.parquet loaded as a DataFrame.
        score_threshold: minimum score to count as 'correct'.
        top1_delta_min: minimum increase in top1_probability vs degraded to
            count as a meaningful confidence boost.
        entropy_delta_max: maximum (most negative) entropy change vs degraded
            to count as a meaningful entropy drop.  Default -0.01 means at
            least a 0.01 decrease in entropy is required alongside the top-1
            boost to classify as confidently_wrong.

    Returns:
        DataFrame with columns:
          sample_id, severity, condition,
          score, top1_probability, output_entropy,
          deg_score, deg_top1, deg_entropy,
          delta_score, delta_top1, delta_entropy,
          outcome
    """
    # Extract degraded baseline per (sample_id, severity)
    deg = (
        df[df["condition"] == "degraded"]
        [["sample_id", "severity", "score", "top1_probability", "output_entropy"]]
        .rename(columns={
            "score": "deg_score",
            "top1_probability": "deg_top1",
            "output_entropy": "deg_entropy",
        })
    )

    # Work only on patching conditions (not baselines)
    patched = df[~df["condition"].isin(_BASELINE_CONDITIONS)].copy()

    merged = patched.merge(deg, on=["sample_id", "severity"])

    merged["delta_score"]   = merged["score"]            - merged["deg_score"]
    merged["delta_top1"]    = merged["top1_probability"] - merged["deg_top1"]
    merged["delta_entropy"] = merged["output_entropy"]   - merged["deg_entropy"]

    score_recovered  = merged["score"] >= score_threshold
    confidence_boost = merged["delta_top1"] >= top1_delta_min
    entropy_drop     = merged["delta_entropy"] <= entropy_delta_max

    def _label(row) -> str:
        rec  = row["score"] >= score_threshold
        boost = row["delta_top1"] >= top1_delta_min
        edrop = row["delta_entropy"] <= entropy_delta_max
        if rec and boost:
            return "correctly_recovered"
        if not rec and boost and edrop:
            return "confidently_wrong"
        if rec:
            return "neutral_correct"
        return "neutral_wrong"

    merged["outcome"] = merged.apply(_label, axis=1)

    cols = [
        "sample_id", "severity", "condition",
        "score", "top1_probability", "output_entropy",
        "deg_score", "deg_top1", "deg_entropy",
        "delta_score", "delta_top1", "delta_entropy",
        "outcome",
    ]
    return merged[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def condition_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return per-condition outcome counts, percentages, and mean deltas.

    Sorted by fraction confidently_wrong descending so the most problematic
    conditions appear first.
    """
    total_per_cond = outcomes.groupby("condition").size().rename("N_total")

    outcome_counts = (
        outcomes.groupby(["condition", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0)
    )

    mean_deltas = outcomes.groupby("condition").agg(
        mean_delta_top1=("delta_top1", "mean"),
        mean_delta_entropy=("delta_entropy", "mean"),
        mean_delta_score=("delta_score", "mean"),
    )

    # Confidently-wrong subset means
    cw = outcomes[outcomes["outcome"] == "confidently_wrong"]
    cw_means = cw.groupby("condition").agg(
        cw_mean_delta_top1=("delta_top1", "mean"),
        cw_mean_delta_entropy=("delta_entropy", "mean"),
    )

    summary = (
        outcome_counts
        .join(total_per_cond)
        .join(mean_deltas)
        .join(cw_means)
    )

    for col in OUTCOME_ORDER:
        summary[f"pct_{col}"] = 100 * summary[col] / summary["N_total"].clip(lower=1)

    summary = summary.sort_values("pct_confidently_wrong", ascending=False)
    return summary.reset_index()


def severity_breakdown(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, severity) outcome counts — useful for checking if
    confidently-wrong is driven by a specific degradation level."""
    total = (
        outcomes.groupby(["condition", "severity"]).size().rename("N_total")
    )
    counts = (
        outcomes.groupby(["condition", "severity", "outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0)
    )
    result = counts.join(total)
    for col in OUTCOME_ORDER:
        result[f"pct_{col}"] = 100 * result[col] / result["N_total"].clip(lower=1)
    return result.reset_index().sort_values(
        ["condition", "severity", "pct_confidently_wrong"], ascending=[True, True, False]
    )


def get_confidently_wrong_samples(
    outcomes: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    """Return all (sample_id, severity) rows classified as confidently_wrong
    for a specific condition, sorted by delta_top1 descending (most egregious first).
    """
    mask = (outcomes["outcome"] == "confidently_wrong") & (outcomes["condition"] == condition)
    cols = [
        "sample_id", "severity",
        "score", "deg_score",
        "top1_probability", "deg_top1", "delta_top1",
        "output_entropy", "deg_entropy", "delta_entropy",
    ]
    return (
        outcomes.loc[mask, cols]
        .sort_values("delta_top1", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_condition_summary(summary: pd.DataFrame, top1_delta_min: float, entropy_delta_max: float) -> None:
    print(f"\nPatching outcome analysis")
    print(f"  confidence boost threshold : Δtop1 ≥ {top1_delta_min}")
    print(f"  entropy drop threshold     : Δentropy ≤ {entropy_delta_max}")
    print()

    # Header
    w = 38
    print(f"{'Condition':<{w}}  {'N':>5}  {'RecovPct':>8}  {'ConfWrong%':>10}  "
          f"{'ΔTop1(CW)':>10}  {'ΔEntr(CW)':>10}  {'ΔTop1(all)':>11}  {'ΔScore(all)':>12}")
    print("─" * 120)

    for _, row in summary.iterrows():
        cw_dt1 = row.get("cw_mean_delta_top1", float("nan"))
        cw_de  = row.get("cw_mean_delta_entropy", float("nan"))
        cw_dt1_str = f"{cw_dt1:+.4f}" if not np.isnan(cw_dt1) else "  n/a  "
        cw_de_str  = f"{cw_de:+.4f}"  if not np.isnan(cw_de)  else "  n/a  "

        print(
            f"{row['condition']:<{w}}  "
            f"{int(row['N_total']):>5}  "
            f"{row['pct_correctly_recovered']:>7.1f}%  "
            f"{row['pct_confidently_wrong']:>9.1f}%  "
            f"{cw_dt1_str:>10}  "
            f"{cw_de_str:>10}  "
            f"{row['mean_delta_top1']:>+10.4f}  "
            f"{row['mean_delta_score']:>+11.4f}"
        )


def print_severity_breakdown(sev_df: pd.DataFrame, condition: str) -> None:
    sub = sev_df[sev_df["condition"] == condition]
    if sub.empty:
        return
    print(f"\n  Severity breakdown for: {condition}")
    print(f"  {'Severity':<12}  {'N':>5}  {'RecovPct':>8}  {'ConfWrong%':>10}")
    print(f"  {'─'*48}")
    for _, row in sub.iterrows():
        print(
            f"  {row['severity']:<12}  "
            f"{int(row['N_total']):>5}  "
            f"{row['pct_correctly_recovered']:>7.1f}%  "
            f"{row['pct_confidently_wrong']:>9.1f}%"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find patching conditions that boost confidence without recovering accuracy"
    )
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to patching_results.parquet")
    parser.add_argument("--score_threshold", type=float, default=1.0,
                        help="Minimum score to count as correct (default 1.0)")
    parser.add_argument("--top1_delta_min", type=float, default=0.01,
                        help="Minimum Δtop1_probability to count as a confidence boost (default 0.01)")
    parser.add_argument("--entropy_delta_max", type=float, default=-0.01,
                        help="Maximum Δentropy (must be negative) for confidently_wrong (default -0.01)")
    parser.add_argument("--show_severity", action="store_true",
                        help="Print per-severity breakdown for each condition")
    parser.add_argument("--show_samples", type=str, default=None, metavar="CONDITION",
                        help="Print all confidently-wrong samples for a specific condition")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Severities: {sorted(df['severity'].unique())}")

    outcomes = classify_outcomes(
        df,
        score_threshold=args.score_threshold,
        top1_delta_min=args.top1_delta_min,
        entropy_delta_max=args.entropy_delta_max,
    )

    summary = condition_summary(outcomes)
    print_condition_summary(summary, args.top1_delta_min, args.entropy_delta_max)

    if args.show_severity:
        sev_df = severity_breakdown(outcomes)
        for cond in summary["condition"]:
            print_severity_breakdown(sev_df, cond)

    if args.show_samples:
        cw_samples = get_confidently_wrong_samples(outcomes, args.show_samples)
        print(f"\nConfidently-wrong samples for condition '{args.show_samples}' "
              f"(N={len(cw_samples)}, sorted by Δtop1 desc):")
        print(cw_samples.to_string(index=False))

    # Save labelled outcomes next to the parquet
    out_path = args.parquet.replace(".parquet", "_confidence_outcomes.parquet")
    outcomes.to_parquet(out_path, index=False)
    print(f"\nPer-row outcome labels saved to {out_path}")


if __name__ == "__main__":
    main()
