"""Plot POPE layer-sweep mean answer confidence for LLaVA and Qwen.

This uses the same POPE layer-sweep parquets that feed
``two_metrics_layer_sweep_nofilter.pdf`` but plots raw patched
``top1_probability`` instead of aggregate recovery.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cli.plot_sweep_filtered import (
    SEVERITY_COLORS,
    bootstrap_ci,
    save,
    severity_label,
    _prepare_layer_sweep,
)


PATCHING_ROOT = Path("/projects/prjs2014/patching")
OUTPUT_DIR = PATCHING_ROOT / "filtered_plots/cross_model/sample_cutoff"
OUTPUT_NAME = "pope_layer_sweep_mean_answer_confidence_nofilter"

POPE_PATHS = [
    (
        "LLaVA-1.5-7B",
        PATCHING_ROOT
        / "layer_sweep_results/llava-hf_llava-1-5-7b-hf/pope/random"
        / "extreme_answer_confidence/layer_sweep_results.parquet",
    ),
    (
        "Qwen3.5-9B",
        PATCHING_ROOT
        / "layer_sweep_results/Qwen_Qwen3-5-9B/pope/random"
        / "extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking"
        / "layer_sweep_results.parquet",
    ),
]


def draw_model_panel(ax: plt.Axes, model_label: str, parquet_path: Path) -> None:
    df, layers, sevs, n_total, n_filtered = _prepare_layer_sweep(
        str(parquet_path),
        min_delta=0.0,
    )

    print(f"{model_label}:")
    print(f"  parquet: {parquet_path}")
    print(f"  samples before/after no-filter: {n_total}/{n_filtered}")
    print(f"  patched final-token rows: {len(df)}")
    print(f"  layers: {len(layers)} ({min(layers) + 1}..{max(layers) + 1})")

    for sev in sevs:
        sub = df[df["severity"] == sev]
        clean_mean = float(sub["clean_top1_probability"].mean())
        degraded_mean = float(sub["degraded_top1_probability"].mean())
        print(
            f"  {sev}: clean mean={clean_mean:.6f}, "
            f"degraded mean={degraded_mean:.6f}, rows={len(sub)}"
        )

        color = SEVERITY_COLORS.get(sev, "#7c3aed")
        display_xs = [layer + 1 for layer in layers]
        ys, los, his = [], [], []
        for layer in layers:
            vals = sub[sub["layer"] == layer]["top1_probability"].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(vals)
            ys.append(mean)
            los.append(lo)
            his.append(hi)

        ys = np.asarray(ys, dtype=float)
        los = np.asarray(los, dtype=float)
        his = np.asarray(his, dtype=float)

        ax.plot(
            display_xs,
            ys,
            marker="o",
            linestyle="-",
            color=color,
            linewidth=2,
            markersize=5,
            label=f"Patched ({severity_label(sev)})",
            zorder=4,
        )
        ax.fill_between(display_xs, los, his, color=color, alpha=0.15)
        ax.axhline(
            clean_mean,
            color="#374151",
            linewidth=1.5,
            linestyle=":",
            alpha=0.85,
            label="Clean baseline",
        )
        ax.axhline(
            degraded_mean,
            color="#6b7280",
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
            label="Degraded baseline",
        )

    display_xs_all = [layer + 1 for layer in layers]
    ax.set_xticks(display_xs_all)
    ax.set_xticklabels([str(x) for x in display_xs_all], fontsize=8, rotation=45)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_title(model_label, fontsize=17, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.65, 0.95)


def main() -> None:
    missing = [path for _, path in POPE_PATHS if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing input parquet: {path}", file=sys.stderr)
        raise SystemExit(1)

    fig, axes = plt.subplots(
        1,
        len(POPE_PATHS),
        figsize=(11.5, 4.7),
        sharey=True,
        squeeze=False,
    )

    for ax, (model_label, parquet_path) in zip(axes[0], POPE_PATHS):
        draw_model_panel(ax, model_label, parquet_path)

    axes[0, 0].set_ylabel("Mean answer confidence", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:3],
        labels[:3],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle("POPE Layer Sweep - Mean Answer Confidence", fontsize=20, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, OUTPUT_DIR, OUTPUT_NAME)

    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"{OUTPUT_NAME}.{ext}"
        print(f"  verified: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
