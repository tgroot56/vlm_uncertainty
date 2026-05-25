"""Signed-mismatch error-rate plot.

Reads a per-sample CSV produced by `src.cli.run_mismatch_sanity_check`, computes
signed mismatch `m = confidence_percentile - probe_percentile in [-1, +1]`,
bins into equal-count quantiles, and plots error rate per bin with binomial
95% confidence intervals (Wilson score). A horizontal reference line marks the
overall base error rate.

Sign convention:
  m > 0 : output more confident than probe (overconfident region; Cell B side)
  m < 0 : output less confident than probe (underconfident region; Cell C side)
  m = 0 : signals agree at the same percentile

Usage:
    python -m src.cli.plot_signed_mismatch \\
        --csv /path/to/mismatch_sanity_check_<model>_<dataset>.csv \\
        [--bins 20] [--output /path/to/figure.png]
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


WILSON_Z = 1.959963984540054  # 95% two-sided


def wilson_interval(n_success: np.ndarray, n_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score 95% interval for a binomial proportion. Vectorised."""
    n = np.maximum(n_total.astype(float), 1.0)
    p = n_success.astype(float) / n
    z2 = WILSON_Z ** 2
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = WILSON_Z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    # When n == 0, force NaN
    lo = np.where(n_total > 0, lo, np.nan)
    hi = np.where(n_total > 0, hi, np.nan)
    return lo, hi


def compute_bin_stats(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    m = df["confidence_percentile"].values - df["probe_percentile"].values
    incorrect = (df["correct"].values == 0).astype(int)

    # qcut produces equal-count bins; duplicates="drop" guards against
    # repeated bin edges when many ties exist in m.
    bin_idx, edges = pd.qcut(m, q=bins, retbins=True, labels=False, duplicates="drop")

    rows = []
    for k in range(int(bin_idx.max()) + 1):
        mask = bin_idx == k
        n = int(mask.sum())
        n_inc = int(incorrect[mask].sum())
        m_mean = float(m[mask].mean()) if n else float("nan")
        m_lo, m_hi = float(edges[k]), float(edges[k + 1])
        err = (n_inc / n) if n else float("nan")
        rows.append(dict(
            bin=k,
            edge_lo=m_lo,
            edge_hi=m_hi,
            mismatch_mean=m_mean,
            n=n,
            n_incorrect=n_inc,
            error_rate=err,
        ))
    out = pd.DataFrame(rows)
    lo, hi = wilson_interval(out["n_incorrect"].values, out["n"].values)
    out["ci_lo"] = lo
    out["ci_hi"] = hi
    return out


def plot_signed_mismatch(
    bin_stats: pd.DataFrame,
    base_error_rate: float,
    title: str,
    output_path: str,
) -> None:
    x = bin_stats["mismatch_mean"].values
    y = bin_stats["error_rate"].values
    lo = bin_stats["ci_lo"].values
    hi = bin_stats["ci_hi"].values
    yerr_lo = np.clip(y - lo, 0.0, None)
    yerr_hi = np.clip(hi - y, 0.0, None)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axhline(base_error_rate, color="grey", linestyle="--", linewidth=1.0,
               label=f"base error rate = {base_error_rate:.1%}")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.errorbar(
        x, y,
        yerr=[yerr_lo, yerr_hi],
        fmt="o-", color="C0", ecolor="C0", elinewidth=1.0, capsize=3.0,
        markersize=5.0, linewidth=1.5,
        label="error rate per bin (95% Wilson CI)",
    )

    ax.set_xlabel(
        "Signed mismatch  m = conf_pct − probe_pct\n"
        "(< 0: probe more confident than output  |  > 0: output more confident than probe)"
    )
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def derive_default_output(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return f"{base}_signed_mismatch.png"


def derive_title(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    base, _ = os.path.splitext(base)
    return f"Signed mismatch vs error rate — {base}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Per-sample mismatch CSV.")
    p.add_argument("--bins", type=int, default=20, help="Number of equal-count bins.")
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <csv_basename>_signed_mismatch.png alongside the CSV.",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Plot title. Defaults to a name derived from the CSV filename.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    df = pd.read_csv(args.csv)

    required = {"confidence_percentile", "probe_percentile", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {args.csv!r} is missing columns: {sorted(missing)}")

    bin_stats = compute_bin_stats(df, args.bins)
    base_error_rate = float((df["correct"] == 0).mean())

    output_path = args.output or derive_default_output(args.csv)
    title = args.title or derive_title(args.csv)
    plot_signed_mismatch(bin_stats, base_error_rate, title, output_path)

    bins_csv = os.path.splitext(output_path)[0] + "_bins.csv"
    bin_stats.to_csv(bins_csv, index=False)

    print(f"Total samples: {len(df)} | Bins: {len(bin_stats)} | Base error rate: {base_error_rate:.1%}")
    print(bin_stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved figure: {output_path}")
    print(f"Saved per-bin stats: {bins_csv}")


if __name__ == "__main__":
    main()
