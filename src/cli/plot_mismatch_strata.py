"""Stratified mismatch plot.

Reads the per-sample CSV from ``src.cli.run_mismatch_sanity_check`` and
produces a 1x2 figure that isolates the within-stratum effect of each signal,
removing the A-vs-D confounding that plagues the un-stratified signed-mismatch
plot.

Left panel  — stratified by output confidence (conf_pct above/below median):
    x = probe_percentile (binned, equal-count)
    curves: output-confident (top half) and output-uncertain (bottom half)
    Right→left along the top curve = transition A → B (Cell A to Cell B).
    Right→left along the bottom curve = transition C → D.

Right panel — stratified by probe (probe_pct above/below median):
    x = confidence_percentile (binned, equal-count)
    curves: probe-high (top half) and probe-low (bottom half)
    Right→left along the top curve = transition A → C.
    Right→left along the bottom curve = transition B → D.

Each curve has 95% Wilson-score CIs; the overall base error rate is shown
as a dashed grey reference line in both panels.

Usage:
    python -m src.cli.plot_mismatch_strata \\
        --csv /path/to/mismatch_sanity_check_<model>_<dataset>.csv \\
        [--bins 10] [--output /path/to/figure.png]
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cli.plot_signed_mismatch import wilson_interval


def bin_curve(
    x: np.ndarray,
    incorrect: np.ndarray,
    bins: int,
) -> pd.DataFrame:
    """Equal-count bin x, then compute err_rate + Wilson 95% CI per bin."""
    if len(x) == 0:
        return pd.DataFrame(columns=[
            "bin", "edge_lo", "edge_hi", "x_mean",
            "n", "n_incorrect", "error_rate", "ci_lo", "ci_hi",
        ])
    bin_idx, edges = pd.qcut(x, q=bins, retbins=True, labels=False, duplicates="drop")
    rows = []
    for k in range(int(bin_idx.max()) + 1):
        mask = bin_idx == k
        n = int(mask.sum())
        n_inc = int(incorrect[mask].sum())
        rows.append(dict(
            bin=k,
            edge_lo=float(edges[k]),
            edge_hi=float(edges[k + 1]),
            x_mean=float(x[mask].mean()) if n else float("nan"),
            n=n,
            n_incorrect=n_inc,
            error_rate=(n_inc / n) if n else float("nan"),
        ))
    out = pd.DataFrame(rows)
    lo, hi = wilson_interval(out["n_incorrect"].values, out["n"].values)
    out["ci_lo"] = lo
    out["ci_hi"] = hi
    return out


def plot_curve_with_ci(
    ax,
    bin_df: pd.DataFrame,
    color: str,
    label: str,
) -> None:
    if len(bin_df) == 0:
        return
    x = bin_df["x_mean"].values
    y = bin_df["error_rate"].values
    yerr_lo = np.clip(y - bin_df["ci_lo"].values, 0.0, None)
    yerr_hi = np.clip(bin_df["ci_hi"].values - y, 0.0, None)
    ax.errorbar(
        x, y,
        yerr=[yerr_lo, yerr_hi],
        fmt="o-", color=color, ecolor=color,
        elinewidth=1.0, capsize=3.0, markersize=5.0, linewidth=1.5,
        label=label,
    )


def annotate_cells(ax, mid: float, text_top_right: str, text_top_left: str,
                   text_bot_right: str, text_bot_left: str, ymax: float) -> None:
    """Drop little Cell-A/B/C/D corner labels so the reader sees the geometry."""
    kw = dict(fontsize=9, alpha=0.6, ha="center", va="center")
    ax.text(0.92, ymax * 0.96, text_top_right, **kw)
    ax.text(0.08, ymax * 0.96, text_top_left, **kw)
    ax.text(0.92, ymax * 0.04, text_bot_right, **kw)
    ax.text(0.08, ymax * 0.04, text_bot_left, **kw)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Per-sample mismatch CSV.")
    p.add_argument("--bins", type=int, default=10,
                   help="Equal-count bins per stratum (default: 10).")
    p.add_argument(
        "--output", default=None,
        help="Output PNG path. Defaults to <csv_basename>_strata.png alongside the CSV.",
    )
    p.add_argument(
        "--title", default=None,
        help="Figure suptitle. Defaults to a name derived from the CSV filename.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    df = pd.read_csv(args.csv)
    required = {"confidence_percentile", "probe_percentile", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {args.csv!r} is missing columns: {sorted(missing)}")

    conf_pct = df["confidence_percentile"].values
    probe_pct = df["probe_percentile"].values
    incorrect = (df["correct"].values == 0).astype(int)

    hi_conf = conf_pct >= 0.5
    hi_probe = probe_pct >= 0.5

    # Left panel: stratified by output confidence; x = probe_pct
    left_top = bin_curve(probe_pct[hi_conf], incorrect[hi_conf], args.bins)   # A↔B
    left_bot = bin_curve(probe_pct[~hi_conf], incorrect[~hi_conf], args.bins) # C↔D

    # Right panel: stratified by probe; x = conf_pct
    right_top = bin_curve(conf_pct[hi_probe], incorrect[hi_probe], args.bins) # A↔C
    right_bot = bin_curve(conf_pct[~hi_probe], incorrect[~hi_probe], args.bins) # B↔D

    base_err = float(incorrect.mean())

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharey=True)

    # Determine a common y-axis upper bound so the two panels are comparable
    ymax_candidates = []
    for d in (left_top, left_bot, right_top, right_bot):
        if len(d):
            ymax_candidates.append(np.nanmax(d["ci_hi"].values))
    ymax = float(max(ymax_candidates) * 1.10) if ymax_candidates else 1.0

    # --- Left panel ---
    ax = axes[0]
    plot_curve_with_ci(ax, left_top, "C0",
                       label="output confident (conf_pct ≥ 0.5)")
    plot_curve_with_ci(ax, left_bot, "C3",
                       label="output uncertain (conf_pct < 0.5)")
    ax.axhline(base_err, color="grey", linestyle="--", linewidth=1.0,
               label=f"base error rate = {base_err:.1%}")
    ax.set_xlabel("probe_percentile (within-stratum)")
    ax.set_ylabel("Error rate")
    ax.set_title("Stratified by output confidence\n(x = probe; B↔A on top, D↔C on bottom)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    annotate_cells(ax, 0.5,
                   text_top_right="A", text_top_left="B",
                   text_bot_right="C", text_bot_left="D",
                   ymax=ymax)

    # --- Right panel ---
    ax = axes[1]
    plot_curve_with_ci(ax, right_top, "C0",
                       label="probe high (probe_pct ≥ 0.5)")
    plot_curve_with_ci(ax, right_bot, "C3",
                       label="probe low (probe_pct < 0.5)")
    ax.axhline(base_err, color="grey", linestyle="--", linewidth=1.0,
               label=f"base error rate = {base_err:.1%}")
    ax.set_xlabel("confidence_percentile (within-stratum)")
    ax.set_title("Stratified by probe\n(x = confidence; C↔A on top, D↔B on bottom)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    annotate_cells(ax, 0.5,
                   text_top_right="A", text_top_left="C",
                   text_bot_right="B", text_bot_left="D",
                   ymax=ymax)

    title = args.title or (
        "Stratified mismatch — "
        + os.path.splitext(os.path.basename(args.csv))[0]
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = args.output or (
        os.path.splitext(args.csv)[0] + "_strata.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    bins_csv = os.path.splitext(output_path)[0] + "_bins.csv"
    bundle = pd.concat([
        left_top.assign(panel="left", stratum="output_confident"),
        left_bot.assign(panel="left", stratum="output_uncertain"),
        right_top.assign(panel="right", stratum="probe_high"),
        right_bot.assign(panel="right", stratum="probe_low"),
    ], ignore_index=True)
    bundle.to_csv(bins_csv, index=False)

    print(f"Total samples: {len(df)} | Base error rate: {base_err:.1%}")
    print(f"Saved figure: {output_path}")
    print(f"Saved per-bin stats: {bins_csv}")


if __name__ == "__main__":
    main()
