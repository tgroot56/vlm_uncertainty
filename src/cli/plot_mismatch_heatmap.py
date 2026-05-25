"""2D error-rate heatmap over (probe_percentile, confidence_percentile).

Reads the per-sample CSV from ``src.cli.run_mismatch_sanity_check`` and bins
both axes into equal-count quantiles (default 10x10). Each cell is coloured
by mean error rate; cells are annotated with sample count and (optionally)
error rate. Use to read the full A/B/C/D landscape at a glance:

    high conf
       ┌──────────────┬──────────────┐
       │   B (top-L)  │   A (top-R)  │   ← "output confident" row
       │ overconfident│   aligned    │
       ├──────────────┼──────────────┤
       │   D (bot-L)  │   C (bot-R)  │   ← "output uncertain" row
       │   aligned    │under-comm.   │
       └──────────────┴──────────────┘
    low conf      low probe ↔ high probe

Usage:
    python -m src.cli.plot_mismatch_heatmap \\
        --csv /path/to/mismatch_sanity_check_<model>_<dataset>.csv \\
        [--bins 10] [--annotate err+n] [--output /path/to/figure.png]
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Per-sample mismatch CSV.")
    p.add_argument("--bins", type=int, default=10,
                   help="Equal-count bins per axis (default: 10).")
    p.add_argument(
        "--annotate", default="err+n",
        choices=("none", "err", "n", "err+n"),
        help="What to write inside each cell.",
    )
    p.add_argument("--cmap", default="RdYlGn_r",
                   help="Matplotlib colormap (default: RdYlGn_r — red = high error).")
    p.add_argument("--vmin", type=float, default=None,
                   help="Lower colour bound. Defaults to data min.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Upper colour bound. Defaults to data max.")
    p.add_argument(
        "--output", default=None,
        help="Output PNG path. Defaults to <csv_basename>_heatmap.png alongside the CSV.",
    )
    p.add_argument(
        "--title", default=None,
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

    conf_pct = df["confidence_percentile"].values
    probe_pct = df["probe_percentile"].values
    incorrect = (df["correct"].values == 0).astype(int)

    # Equal-count bins per axis
    probe_idx, probe_edges = pd.qcut(
        probe_pct, q=args.bins, retbins=True, labels=False, duplicates="drop"
    )
    conf_idx, conf_edges = pd.qcut(
        conf_pct, q=args.bins, retbins=True, labels=False, duplicates="drop"
    )
    n_pb = int(probe_idx.max()) + 1
    n_cf = int(conf_idx.max()) + 1

    err = np.full((n_cf, n_pb), np.nan)
    counts = np.zeros((n_cf, n_pb), dtype=int)
    for ci in range(n_cf):
        for pi in range(n_pb):
            mask = (conf_idx == ci) & (probe_idx == pi)
            n = int(mask.sum())
            counts[ci, pi] = n
            if n > 0:
                err[ci, pi] = float(incorrect[mask].mean())

    base_err = float(incorrect.mean())

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    vmin = args.vmin if args.vmin is not None else float(np.nanmin(err))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(err))
    im = ax.imshow(
        err,
        origin="lower",
        cmap=args.cmap,
        vmin=vmin, vmax=vmax,
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="auto",
        interpolation="nearest",
    )

    # Annotations
    if args.annotate != "none":
        x_centers = (probe_edges[:-1] + probe_edges[1:]) / 2.0
        y_centers = (conf_edges[:-1] + conf_edges[1:]) / 2.0
        # Map cell mid-rank to a pos in [0,1] along axis (the imshow extent)
        # We use evenly-spaced centers since extent is [0, 1] regardless of bin widths
        x_pos = (np.arange(n_pb) + 0.5) / n_pb
        y_pos = (np.arange(n_cf) + 0.5) / n_cf
        for ci in range(n_cf):
            for pi in range(n_pb):
                if counts[ci, pi] == 0:
                    continue
                e = err[ci, pi]
                # Pick text colour by background luminance
                rgba = im.cmap(im.norm(e))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                lines = []
                if "err" in args.annotate:
                    lines.append(f"{e:.0%}")
                if "n" in args.annotate:
                    lines.append(f"n={counts[ci, pi]}")
                ax.text(
                    x_pos[pi], y_pos[ci], "\n".join(lines),
                    ha="center", va="center",
                    color=txt_color, fontsize=8,
                )

    # Cell-region annotations (A/B/C/D in corners of the axes themselves)
    corner_kw = dict(fontsize=14, fontweight="bold", alpha=0.85,
                     ha="center", va="center")
    ax.text(0.07, 0.93, "B",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7,
                      edgecolor="none"),
            **corner_kw)
    ax.text(0.93, 0.93, "A",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7,
                      edgecolor="none"),
            **corner_kw)
    ax.text(0.07, 0.07, "D",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7,
                      edgecolor="none"),
            **corner_kw)
    ax.text(0.93, 0.07, "C",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7,
                      edgecolor="none"),
            **corner_kw)

    # Median-split reference lines
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("probe_percentile  (low ←  → high)")
    ax.set_ylabel("confidence_percentile  (low ←  → high)")
    title = args.title or (
        "Error-rate heatmap — "
        + os.path.splitext(os.path.basename(args.csv))[0]
    )
    ax.set_title(f"{title}\n(base error rate = {base_err:.1%}, n = {len(df)})")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Error rate")

    fig.tight_layout()

    output_path = args.output or (
        os.path.splitext(args.csv)[0] + "_heatmap.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    # Long-form per-cell stats CSV
    rows = []
    for ci in range(n_cf):
        for pi in range(n_pb):
            rows.append(dict(
                conf_bin=ci,
                probe_bin=pi,
                conf_lo=float(conf_edges[ci]),
                conf_hi=float(conf_edges[ci + 1]),
                probe_lo=float(probe_edges[pi]),
                probe_hi=float(probe_edges[pi + 1]),
                n=int(counts[ci, pi]),
                error_rate=float(err[ci, pi]) if counts[ci, pi] else float("nan"),
            ))
    cells_csv = os.path.splitext(output_path)[0] + "_cells.csv"
    pd.DataFrame(rows).to_csv(cells_csv, index=False)

    print(f"Total samples: {len(df)} | Bins: {n_pb} x {n_cf} | Base error rate: {base_err:.1%}")
    print(f"Error rate range: {np.nanmin(err):.1%} – {np.nanmax(err):.1%}")
    print(f"Saved figure: {output_path}")
    print(f"Saved per-cell stats: {cells_csv}")


if __name__ == "__main__":
    main()
