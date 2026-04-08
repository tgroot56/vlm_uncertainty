"""Stratify patching-results samples into correctness groups.

Groups (defined per sample_id × severity):
  - recoverable_failure : clean correct, degraded incorrect
  - no_effect           : clean correct, degraded correct
  - clean_incorrect     : clean incorrect (regardless of degraded)

Usage:
    python -m src.patching.sample_groups \
        --parquet /path/to/patching_results.parquet \
        [--threshold 1.0]
"""

from __future__ import annotations

import argparse
from typing import Dict

import pandas as pd


GROUPS = {
    "recoverable_failure": "clean correct  → degraded incorrect",
    "no_effect":           "clean correct  → degraded correct",
    "clean_incorrect":     "clean incorrect (regardless of degraded)",
}


def assign_groups(
    df: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame with one row per (sample_id, severity) labelled by group.

    Args:
        df: the full patching results parquet (all conditions).
        threshold: minimum score to count as 'correct' (default 1.0).

    Returns:
        DataFrame with columns [sample_id, severity, group,
        clean_score, degraded_score].
    """
    # Extract just the clean and degraded baseline rows
    clean = (
        df[df["condition"] == "clean"][["sample_id", "severity", "score"]]
        .rename(columns={"score": "clean_score"})
    )
    degraded = (
        df[df["condition"] == "degraded"][["sample_id", "severity", "score"]]
        .rename(columns={"score": "degraded_score"})
    )

    base = clean.merge(degraded, on=["sample_id", "severity"])
    base["clean_correct"] = base["clean_score"] >= threshold
    base["degraded_correct"] = base["degraded_score"] >= threshold

    def _label(row: pd.Series) -> str:
        if not row["clean_correct"]:
            return "clean_incorrect"
        if row["degraded_correct"]:
            return "no_effect"
        return "recoverable_failure"

    base["group"] = base.apply(_label, axis=1)
    return base[["sample_id", "severity", "group", "clean_score", "degraded_score"]]


def print_group_summary(groups: pd.DataFrame, threshold: float) -> None:
    """Print group counts and percentages, overall and per severity."""
    total_obs = len(groups)
    n_samples = groups["sample_id"].nunique()
    severities = sorted(groups["severity"].unique())

    print(f"\nCorrectness threshold : score >= {threshold}")
    print(f"Unique samples        : {n_samples}")
    print(f"Total observations    : {total_obs}  "
          f"(samples × severities = {n_samples} × {len(severities)})")

    # --- overall ---
    print(f"\n{'─'*60}")
    print(f"{'Overall (all severities combined)':}")
    print(f"{'─'*60}")
    overall = groups["group"].value_counts()
    for group, desc in GROUPS.items():
        n = int(overall.get(group, 0))
        pct = 100 * n / total_obs
        print(f"  {desc:<44s}  {n:>5d}  ({pct:5.1f}%)")
    print(f"  {'Total':<44s}  {total_obs:>5d}")

    # --- per severity ---
    print(f"\n{'─'*60}")
    print(f"{'Per severity':}")
    print(f"{'─'*60}")
    sev_counts = groups.groupby(["severity", "group"]).size().unstack(fill_value=0)
    for sev in severities:
        row = sev_counts.loc[sev] if sev in sev_counts.index else {}
        total_sev = sum(row.get(g, 0) for g in GROUPS)
        print(f"\n  Severity: {sev}  (N={total_sev})")
        for group, desc in GROUPS.items():
            n = int(row.get(group, 0))
            pct = 100 * n / total_sev if total_sev > 0 else 0
            print(f"    {desc:<44s}  {n:>5d}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify patching samples by correctness group")
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to patching_results.parquet")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Score threshold for 'correct' (default 1.0)")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    groups = assign_groups(df, threshold=args.threshold)
    print_group_summary(groups, threshold=args.threshold)


if __name__ == "__main__":
    main()
