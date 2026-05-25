"""Generate layer-sweep and positional-sweep recovery plots with |delta| >= 0.05 filter.

Produces one combined figure per sweep type, with subplots for answer confidence
(stored as top1_probability), output entropy, and score recovery. Delta filter is always based on
|clean_top1 - degraded_top1| >= DELTA_THRESHOLD to keep sample selection consistent.

Usage:
    python -m src.cli.plot_sweep_filtered --output_dir /path/to/output/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DELTA_THRESHOLD = 0.05
EPSILON = 1e-8
MIN_POS_SAMPLES = 10   # drop tail positions with fewer samples than this
METRICS = ["top1_probability", "output_entropy", "score"]

METRIC_LABELS = {
    "top1_probability": "Answer confidence",
    "output_entropy":   "Output entropy",
    "score":            "Score",
}

SPAN_COLORS = {
    "text_pre":         "#94a3b8",
    "visual":           "#0369a1",
    "question":         "#c2410c",
    "assistant_marker": "#4338ca",
}

SPAN_DISPLAY = {
    "text_pre":         "Template Prefix",
    "visual":           "Visual tokens",
    "question":         "Question",
    "assistant_marker": "Template Suffix",
}

SPAN_AXIS_LABELS = {
    "text_pre":         "Template\nPrefix",
    "visual":           "Visual tokens",
    "question":         "Question",
    "assistant_marker": "Template\nSuffix",
}

SEVERITY_COLORS = {
    "mild":     "#0ea5e9",
    "moderate": "#f59e0b",
    "severe":   "#ef4444",
    "extreme":  "#7c3aed",
}
SEVERITY_ORDER = ["mild", "moderate", "severe", "extreme"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def severity_label(severity: str) -> str:
    return "Recovery" if severity == "extreme" else severity.capitalize()


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compute_recovery(patched_vals, clean_vals, degraded_vals):
    delta = clean_vals - degraded_vals
    raw = (patched_vals - degraded_vals) / (delta + np.sign(delta + EPSILON) * EPSILON)
    return np.clip(raw, -5, 5)


def compute_aggregate_recovery(patched_vals, clean_vals, degraded_vals):
    patched_vals = np.asarray(patched_vals, dtype=float)
    clean_vals = np.asarray(clean_vals, dtype=float)
    degraded_vals = np.asarray(degraded_vals, dtype=float)
    valid = ~(np.isnan(patched_vals) | np.isnan(clean_vals) | np.isnan(degraded_vals))
    if not valid.any():
        return float("nan")

    patched_mean = patched_vals[valid].mean()
    clean_mean = clean_vals[valid].mean()
    degraded_mean = degraded_vals[valid].mean()
    delta = clean_mean - degraded_mean
    eps = EPSILON if delta >= 0 else -EPSILON
    raw = (patched_mean - degraded_mean) / (delta + eps)
    return float(np.clip(raw, -5, 5))


def aggregate_recovery_ci(frame: pd.DataFrame, metric: str, n: int = 1000, seed: int = 42):
    cols = [metric, f"clean_{metric}", f"degraded_{metric}"]
    values = frame[cols].dropna()
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    patched = values[metric].to_numpy(dtype=float)
    clean = values[f"clean_{metric}"].to_numpy(dtype=float)
    degraded = values[f"degraded_{metric}"].to_numpy(dtype=float)
    mean = compute_aggregate_recovery(patched, clean, degraded)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), (n, len(values)), replace=True)
    patched_mean = patched[idx].mean(axis=1)
    clean_mean = clean[idx].mean(axis=1)
    degraded_mean = degraded[idx].mean(axis=1)
    delta = clean_mean - degraded_mean
    eps = np.where(delta >= 0, EPSILON, -EPSILON)
    boot = np.clip((patched_mean - degraded_mean) / (delta + eps), -5, 5)
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return mean, float("nan"), float("nan")
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def save(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 200})]:
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  Saved {output_dir}/{name}.pdf/.png")


def _apply_delta_filter(df: pd.DataFrame, min_delta: float = DELTA_THRESHOLD) -> pd.DataFrame:
    if min_delta <= 0:
        return df
    return df[(df["clean_top1_probability"] - df["degraded_top1_probability"]).abs() >= min_delta]


def _attach_baselines(patched: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for cond in ("clean", "degraded"):
        bl = (
            df[df["condition"] == cond]
            [["sample_id", "severity"] + METRICS]
            .rename(columns={m: f"{cond}_{m}" for m in METRICS})
        )
        patched = patched.merge(bl, on=["sample_id", "severity"])
    return patched


def _add_recovery_columns(patched: pd.DataFrame) -> pd.DataFrame:
    for m in METRICS:
        patched[f"recovery_{m}"] = compute_recovery(
            patched[m].values,
            patched[f"clean_{m}"].values,
            patched[f"degraded_{m}"].values,
        )
    return patched


def _prepare_layer_sweep(parquet_path: str, min_delta: float = DELTA_THRESHOLD,
                         span_label: str | None = None) -> tuple[pd.DataFrame, list[int], list[str], int, int]:
    """Load and filter layer sweep data. Returns (df, layers, severities, n_total, n_filtered).

    When ``span_label`` is None, selects the "final" variant (position_offset == -1
    rows with no span_label). When given (e.g. "last_question", "last_visual"),
    selects all patched rows tagged with that span label — used by the
    ``question_visual`` position set, where each span has varying offsets per sample.
    """
    df = pd.read_parquet(parquet_path)
    patched = df[df["condition"] == "patched"].copy()
    if span_label is None:
        patched = patched[patched["position_offset"] == -1]
    else:
        patched = patched[patched["span_label"] == span_label]
    patched = _attach_baselines(patched, df)

    n_total = patched["sample_id"].nunique()
    patched = _apply_delta_filter(patched, min_delta=min_delta)
    n_filtered = patched["sample_id"].nunique()
    patched = _add_recovery_columns(patched)

    layers = sorted(patched["layer"].dropna().unique().astype(int))
    sevs = [s for s in SEVERITY_ORDER if s in patched["severity"].values]
    return patched, layers, sevs, n_total, n_filtered


SPAN_ORDER = ["text_pre", "visual", "question", "assistant_marker"]


def _prepare_positional_sweep(
    parquet_path: str,
    min_delta: float = DELTA_THRESHOLD,
    min_pos_samples: int = MIN_POS_SAMPLES,
) -> tuple[pd.DataFrame, list[int], dict, list[str], int, int]:
    """Load and filter positional sweep data.

    Aggregation uses span-relative positions (token_position - per-sample span start)
    so that e.g. assistant_marker token 0 is always token 0 regardless of question length.
    Returns (df, canonical_xs, span_regions, severities, n_total, n_filtered).
    """
    df = pd.read_parquet(parquet_path)
    patched = df[df["condition"].str.startswith("patched")].copy()
    patched = _attach_baselines(patched, df)

    n_total = df[df["condition"] == "clean"]["sample_id"].nunique()
    patched = _apply_delta_filter(patched, min_delta=min_delta)
    n_filtered = patched["sample_id"].nunique()
    patched = _add_recovery_columns(patched)

    # Compute per-sample span-start token position, then span-relative position
    span_starts = (
        patched.groupby(["sample_id", "span_label"])["token_position"]
        .min()
        .reset_index(name="span_start")
    )
    patched = patched.merge(span_starts, on=["sample_id", "span_label"])
    patched["span_rel_pos"] = (patched["token_position"] - patched["span_start"]).astype(int)

    sevs_present = [s for s in SEVERITY_ORDER if s in patched["severity"].values]

    # Drop (span, rel_pos) slots where any severity has too few samples.
    pos_counts = (
        patched.groupby(["span_label", "span_rel_pos", "severity"])["sample_id"]
        .nunique()
        .reset_index(name="n")
    )
    min_n = pos_counts.groupby(["span_label", "span_rel_pos"])["n"].min()
    valid_idx = set(min_n[min_n >= min_pos_samples].index.tolist())  # set of (span, rel_pos)
    patched = patched[patched.apply(
        lambda r: (r["span_label"], r["span_rel_pos"]) in valid_idx, axis=1
    )]

    # Build canonical x-axis: spans in fixed order, rel_pos within each span, 1-unit gap between spans
    canonical_rows = []
    span_regions: dict[str, list[int]] = {}
    x = 0
    for span in SPAN_ORDER:
        valid_rel = sorted({rp for (sl, rp) in valid_idx if sl == span})
        if not valid_rel:
            continue
        xs = list(range(x, x + len(valid_rel)))
        span_regions[span] = xs
        for rel_pos, cx in zip(valid_rel, xs):
            canonical_rows.append({"span_label": span, "span_rel_pos": rel_pos, "canonical_x": cx})
        x += len(valid_rel) + 1  # 1-unit visual gap between spans

    canonical_df = pd.DataFrame(canonical_rows)
    patched = patched.merge(canonical_df, on=["span_label", "span_rel_pos"], how="inner")

    canonical_xs = sorted(patched["canonical_x"].unique().astype(int))
    return patched, canonical_xs, span_regions, sevs_present, n_total, n_filtered


# ---------------------------------------------------------------------------
# Axes-level drawing
# ---------------------------------------------------------------------------

def _draw_layer_ax(ax: plt.Axes, df: pd.DataFrame, metric: str,
                   layers: list[int], sevs: list[str],
                   show_legend: bool, show_xlabel: bool,
                   raw: bool = False, show_ylabel: bool = True) -> None:
    for sev in sevs:
        color = SEVERITY_COLORS[sev]
        sub = df[df["severity"] == sev]

        if raw:
            clean_val    = sub["clean_score" if metric == "score" else f"clean_{metric}"].mean()
            degraded_val = sub["degraded_score" if metric == "score" else f"degraded_{metric}"].mean()
            col = "score" if metric == "score" else metric
            ys, los, his = [], [], []
            for layer in layers:
                vals = sub[sub["layer"] == layer][col].values
                m, lo, hi = bootstrap_ci(vals)
                ys.append(m); los.append(lo); his.append(hi)
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            ax.axhline(clean_val,    color=color, linewidth=1.5, linestyle=":",  alpha=0.7, zorder=2)
            ax.axhline(degraded_val, color=color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
            ax.plot(layers, ys, "o-", color=color, linewidth=2, markersize=5,
                    label=severity_label(sev), zorder=4)
            ax.fill_between(layers, los, his, color=color, alpha=0.15)
        else:
            ys, los, his = [], [], []
            for layer in layers:
                m, lo, hi = aggregate_recovery_ci(sub[sub["layer"] == layer], metric)
                ys.append(m); los.append(lo); his.append(hi)
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            ax.plot(layers, ys, "o-", color=color, linewidth=2, markersize=5,
                    label=severity_label(sev), zorder=4)
            ax.fill_between(layers, los, his, color=color, alpha=0.15)

    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8, rotation=45)
    if show_xlabel:
        ax.set_xlabel("Layer index", fontsize=16)
    ax.grid(alpha=0.3)

    if raw:
        ylabel = {"top1_probability": "Mean answer confidence",
                  "output_entropy":   "Mean output entropy",
                  "score":            "Mean accuracy"}.get(metric, metric)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(METRIC_LABELS[metric], fontsize=18, fontweight="bold")
        if show_legend:
            sev_handles = [
                plt.Line2D([0], [0], color=SEVERITY_COLORS[s], linewidth=2, label=severity_label(s))
                for s in sevs
            ]
            ref_handles = [
                plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",  label="Clean baseline"),
                plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle="--", label="Degraded baseline"),
            ]
            ax.legend(handles=sev_handles + ref_handles, fontsize=8,
                      title="Severity", loc="upper left")
    else:
        ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="#6b7280", linewidth=1.5, linestyle=":", alpha=0.85, zorder=2)
        if show_ylabel:
            ax.set_ylabel("Aggregate recovery", fontsize=16)
        else:
            ax.set_ylabel("")
        ax.set_title(METRIC_LABELS[metric], fontsize=18, fontweight="bold")
        if show_legend:
            handles = [
                plt.Line2D([0], [0], color=SEVERITY_COLORS[s], linewidth=2, label=severity_label(s))
                for s in sevs
            ] + [
                plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",
                           label="Full recovery (RR=1)"),
            ]
            ax.legend(handles=handles, fontsize=8, title="Severity", loc="upper left")


def _draw_layer_ax_score_UNUSED(ax: plt.Axes, df: pd.DataFrame,
                         layers: list[int], sevs: list[str],
                         show_legend: bool, show_xlabel: bool) -> None:
    """Score panel: raw mean accuracy (clean / degraded / patched) instead of RR."""
    for sev in sevs:
        color = SEVERITY_COLORS[sev]
        sub = df[df["severity"] == sev]

        # Clean and degraded are constant across layers — use single mean
        clean_acc    = sub["clean_score"].mean()
        degraded_acc = sub["degraded_score"].mean()

        pat_ys, pat_los, pat_his = [], [], []
        for layer in layers:
            vals = sub[sub["layer"] == layer]["score"].values
            m, lo, hi = bootstrap_ci(vals)
            pat_ys.append(m); pat_los.append(lo); pat_his.append(hi)
        pat_ys = np.array(pat_ys)
        pat_los = np.array(pat_los)
        pat_his = np.array(pat_his)

        ax.axhline(clean_acc,    color=color, linewidth=1.5, linestyle=":",  alpha=0.7, zorder=2)
        ax.axhline(degraded_acc, color=color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
        ax.plot(layers, pat_ys, "o-", color=color, linewidth=2, markersize=5,
                label=severity_label(sev), zorder=4)
        ax.fill_between(layers, pat_los, pat_his, color=color, alpha=0.15)

    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8, rotation=45)
    ax.set_ylabel("Mean accuracy", fontsize=16)
    ax.set_title("Score (raw accuracy)", fontsize=18, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Layer index", fontsize=16)
    ax.grid(alpha=0.3)
    if show_legend:
        sev_handles = [
            plt.Line2D([0], [0], color=SEVERITY_COLORS[s], linewidth=2, label=severity_label(s))
            for s in sevs
        ]
        ref_handles = [
            plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",  label="Clean baseline"),
            plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle="--", label="Degraded baseline"),
        ]
        ax.legend(handles=sev_handles + ref_handles, fontsize=8,
                  title="Severity", loc="lower right")


def _draw_positional_ax(ax: plt.Axes, df: pd.DataFrame, metric: str,
                        canonical_xs: list[int], span_regions: dict,
                        sevs: list[str], show_legend: bool, show_xlabel: bool,
                        raw: bool = False, show_ylabel: bool = True) -> None:
    for sev in sevs:
        color = SEVERITY_COLORS[sev]
        sub = df[df["severity"] == sev]

        if raw:
            col = "score" if metric == "score" else metric
            clean_col    = "clean_score" if metric == "score" else f"clean_{metric}"
            degraded_col = "degraded_score" if metric == "score" else f"degraded_{metric}"
            clean_val    = sub[clean_col].mean()
            degraded_val = sub[degraded_col].mean()
            ys, los, his = [], [], []
            for cx in canonical_xs:
                m, lo, hi = bootstrap_ci(sub[sub["canonical_x"] == cx][col].values)
                ys.append(m); los.append(lo); his.append(hi)
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            ax.axhline(clean_val,    color=color, linewidth=1.5, linestyle=":",  alpha=0.7, zorder=2)
            ax.axhline(degraded_val, color=color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
            ax.plot(canonical_xs, ys, "o-", color=color, linewidth=2, markersize=3,
                    label=severity_label(sev), zorder=4)
            ax.fill_between(canonical_xs, los, his, color=color, alpha=0.15)
        else:
            ys, los, his = [], [], []
            for cx in canonical_xs:
                m, lo, hi = aggregate_recovery_ci(sub[sub["canonical_x"] == cx], metric)
                ys.append(m); los.append(lo); his.append(hi)
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            ax.plot(canonical_xs, ys, "o-", color=color, linewidth=2, markersize=3,
                    label=severity_label(sev), zorder=4)
            ax.fill_between(canonical_xs, los, his, color=color, alpha=0.15)

    # Span background shading + centered span-name x tick labels + boundary dividers
    span_patches = []
    all_cx = sorted(cx for cxs in span_regions.values() for cx in cxs)
    tick_positions = []
    tick_labels = []
    tick_spans = []
    for span, cx_list in span_regions.items():
        color = SPAN_COLORS.get(span, "#e5e7eb")
        lo_x, hi_x = min(cx_list) - 0.5, max(cx_list) + 0.5
        ax.axvspan(lo_x, hi_x, color=color, alpha=0.12, zorder=0)
        # Vertical divider at left edge (skip if it's the very first span)
        if lo_x > min(all_cx) - 0.5:
            ax.axvline(lo_x, color="#9ca3af", linewidth=0.8, linestyle="-", zorder=1)
        tick_positions.append((lo_x + hi_x) / 2)
        tick_labels.append(SPAN_AXIS_LABELS.get(span, SPAN_DISPLAY.get(span, span)))
        tick_spans.append(span)
        span_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.5, label=SPAN_DISPLAY.get(span, span))
        )
    ax.set_xticks(tick_positions)
    if show_xlabel:
        ax.set_xticklabels(tick_labels, fontsize=14, color="#374151")
        for label, span in zip(ax.get_xticklabels(), tick_spans):
            if span == "question":
                label.set_ha("right")
            elif span == "assistant_marker":
                label.set_ha("left")
            else:
                label.set_ha("center")
        ax.tick_params(axis="x", length=0, pad=10)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    ax.grid(alpha=0.3)
    ax.set_title(METRIC_LABELS[metric], fontsize=18, fontweight="bold")

    if raw:
        ylabel = {"top1_probability": "Mean answer confidence",
                  "output_entropy":   "Mean output entropy",
                  "score":            "Mean accuracy"}.get(metric, metric)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=16)
        if show_legend:
            sev_handles = [
                plt.Line2D([0], [0], color=SEVERITY_COLORS[s], linewidth=2, label=severity_label(s))
                for s in sevs
            ]
            ref_handles = [
                plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",  label="Clean baseline"),
                plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle="--", label="Degraded baseline"),
            ]
            ax.legend(handles=sev_handles + ref_handles + span_patches,
                      fontsize=8, title="Severity / Region", loc="upper left")
    else:
        ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="#6b7280", linewidth=1.5, linestyle=":", alpha=0.85, zorder=2)
        if show_ylabel:
            ax.set_ylabel("Aggregate recovery", fontsize=16)
        else:
            ax.set_ylabel("")
        if show_legend:
            sev_handles = [
                plt.Line2D([0], [0], color=SEVERITY_COLORS[s], linewidth=2, label=severity_label(s))
                for s in sevs
            ]
            rr1_handle = plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",
                                    label="Full recovery (RR=1)")
            ax.legend(handles=sev_handles + [rr1_handle] + span_patches,
                      fontsize=8, title="Severity / Region", loc="upper left")


def _draw_positional_ax_score_UNUSED(ax: plt.Axes, df: pd.DataFrame,
                               positions: list[int], span_regions: dict,
                               sevs: list[str], show_legend: bool, show_xlabel: bool) -> None:
    """Score panel: raw mean accuracy (clean / degraded / patched) instead of RR."""
    for sev in sevs:
        color = SEVERITY_COLORS[sev]
        sub = df[df["severity"] == sev]

        clean_acc    = sub["clean_score"].mean()
        degraded_acc = sub["degraded_score"].mean()

        pat_ys, pat_los, pat_his = [], [], []
        for pos in positions:
            vals = sub[sub["position_index"] == pos]["score"].values
            m, lo, hi = bootstrap_ci(vals)
            pat_ys.append(m); pat_los.append(lo); pat_his.append(hi)
        pat_ys = np.array(pat_ys)
        pat_los = np.array(pat_los)
        pat_his = np.array(pat_his)

        ax.axhline(clean_acc,    color=color, linewidth=1.5, linestyle=":",  alpha=0.7, zorder=2)
        ax.axhline(degraded_acc, color=color, linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
        ax.plot(positions, pat_ys, "o-", color=color, linewidth=2, markersize=3,
                label=severity_label(sev), zorder=4)
        ax.fill_between(positions, pat_los, pat_his, color=color, alpha=0.15)

    span_patches = []
    for span, pos_list in span_regions.items():
        color = SPAN_COLORS.get(span, "#e5e7eb")
        ax.axvspan(min(pos_list) - 0.5, max(pos_list) + 0.5, color=color, alpha=0.12, zorder=0)
        span_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.5, label=SPAN_DISPLAY.get(span, span))
        )

    ax.set_ylabel("Mean accuracy", fontsize=16)
    ax.set_title("Score (raw accuracy)", fontsize=18, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("Token position index", fontsize=16)
    ax.grid(alpha=0.3)
    if show_legend:
        sev_handles = [
            plt.Line2D([0], [0], color=SEVERITY_COLORS[s], linewidth=2, label=severity_label(s))
            for s in sevs
        ]
        ref_handles = [
            plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",  label="Clean baseline"),
            plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle="--", label="Degraded baseline"),
        ]
        ax.legend(
            handles=sev_handles + ref_handles + span_patches,
            fontsize=8, title="Severity / Region", loc="upper right",
        )


# ---------------------------------------------------------------------------
# Combined figures
# ---------------------------------------------------------------------------

def plot_layer_sweep_combined(parquet_path: str, output_dir: Path,
                              raw: bool = False, min_delta: float = DELTA_THRESHOLD) -> None:
    # Detect mode: "final" (has rows at position_offset == -1 with no span_label)
    # vs "question_visual" (patched rows tagged with span_label).
    _df_peek = pd.read_parquet(parquet_path)
    _df_peek = _df_peek[["condition", "position_offset"] + (["span_label"] if "span_label" in _df_peek.columns else [])]
    patched_peek = _df_peek[_df_peek["condition"] == "patched"]
    if "span_label" in patched_peek.columns:
        final_rows = ((patched_peek["position_offset"] == -1) & patched_peek["span_label"].isna()).sum()
        span_labels_present = [s for s in patched_peek["span_label"].dropna().unique().tolist() if s]
    else:
        final_rows = (patched_peek["position_offset"] == -1).sum()
        span_labels_present = []

    if final_rows > 0:
        variants: list[tuple[str | None, str, str]] = [(None, "Final prompt token (position offset −1)", "")]
    else:
        variants = [(sl, f"Span: {sl}", f"_{sl}") for sl in span_labels_present]

    suffix = "_raw" if raw else ""
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    for span_filter, title_span, name_span in variants:
        df, layers, sevs, n_total, n_filtered = _prepare_layer_sweep(
            parquet_path, min_delta=min_delta, span_label=span_filter,
        )
        filter_label = (f"|Δ top-1| ≥ {min_delta} filter  ·  {n_filtered}/{n_total} samples  ·  "
                        if min_delta > 0 else f"no filter  ·  {n_total} samples  ·  ")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
        for i, (ax, metric) in enumerate(zip(axes, METRICS)):
            _draw_layer_ax(ax, df, metric, layers, sevs,
                           show_legend=(i == 0), show_xlabel=True, raw=raw,
                           show_ylabel=(i == 0))

        fig.suptitle(
            f"Layer Sweep — {title_span}  ·  "
            f"{filter_label}95% bootstrap CI",
            fontsize=20, y=1.04,
        )
        fig.tight_layout()
        save(fig, output_dir, f"layer_sweep_combined_filtered{suffix}{nofilter_suffix}{name_span}")


def plot_positional_sweep_combined(parquet_path: str, output_dir: Path,
                                   raw: bool = False, min_delta: float = DELTA_THRESHOLD,
                                   min_pos_samples: int = MIN_POS_SAMPLES) -> None:
    df, canonical_xs, span_regions, sevs, n_total, n_filtered = _prepare_positional_sweep(
        parquet_path,
        min_delta=min_delta,
        min_pos_samples=min_pos_samples,
    )
    suffix = "_raw" if raw else ""
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""
    filter_label = (f"|Δ top-1| ≥ {min_delta} filter  ·  {n_filtered}/{n_total} samples  ·  "
                    if min_delta > 0 else f"no filter  ·  {n_total} samples  ·  ")

    fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=False)
    for i, (ax, metric) in enumerate(zip(axes, METRICS)):
        _draw_positional_ax(ax, df, metric, canonical_xs, span_regions, sevs,
                            show_legend=(i == 0), show_xlabel=True, raw=raw,
                            show_ylabel=(i == 0))

    fig.suptitle(
        f"Positional Sweep — Recovery by patched token position  ·  "
        f"{filter_label}n_pos ≥ {min_pos_samples}  ·  95% bootstrap CI",
        fontsize=20, y=1.04,
    )
    fig.tight_layout()
    save(fig, output_dir, f"positional_sweep_combined_filtered{suffix}{nofilter_suffix}")


# ---------------------------------------------------------------------------
# All-dataset figures (4 rows × 3 cols)
# ---------------------------------------------------------------------------

DATASET_LABELS = {
    "vqa-v2":    "VQA-v2",
    "pope":      "POPE",
    "coco-qa-vi": "COCO-QA-VI",
    "imagenet-r": "ImageNet-R",
}

MODEL_LABELS = {
    "llava-hf_llava-1-5-7b-hf": "LLaVA-1.5-7B",
    "Qwen_Qwen3-5-9B":          "Qwen3.5-9B",
}

# Metrics shown in cross-model combined figures (no entropy)
CROSS_MODEL_METRICS = ["top1_probability", "score"]


def plot_all_datasets_layer_sweep(
    dataset_parquets: list[tuple[str, str]],   # [(dataset_name, parquet_path), ...]
    output_dir: Path,
    raw: bool = False,
    min_delta: float = DELTA_THRESHOLD,
    metrics: list[str] | None = None,
    title: str | None = None,
    output_name: str | None = None,
) -> None:
    """4-row × n_metrics-col figure: rows = datasets, cols = metrics."""
    if metrics is None:
        metrics = METRICS
    n_datasets = len(dataset_parquets)
    n_metrics = len(metrics)
    suffix = "_raw" if raw else ""
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    fig, axes = plt.subplots(
        n_datasets, n_metrics,
        figsize=(6 * n_metrics, 4.5 * n_datasets),
        sharey=False, squeeze=False,
    )

    for row, (dataset, parquet_path) in enumerate(dataset_parquets):
        df, layers, sevs, n_total, n_filtered = _prepare_layer_sweep(parquet_path, min_delta=min_delta)

        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            _draw_layer_ax(
                ax, df, metric, layers, sevs,
                show_legend=(col == 0 and row == 0),
                show_xlabel=(row == n_datasets - 1),
                raw=raw,
                show_ylabel=(col == 0),
            )
            if col == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS.get(dataset, dataset)}\n"
                    + ({"top1_probability": "Mean answer confidence",
                        "output_entropy":   "Mean entropy",
                        "score":            "Mean accuracy"}.get(metric, metric)
                       if raw else "Aggregate recovery"),
                    fontsize=16,
                )
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=18, fontweight="bold")
            else:
                ax.set_title("")

    filter_label = (f"|Δ top-1| ≥ {min_delta} filter  ·  " if min_delta > 0 else "no filter  ·  ")
    default_title = (
        f"Layer Sweep — Final prompt token (position offset −1)  ·  "
        f"{filter_label}95% bootstrap CI"
    )
    fig.suptitle(title if title is not None else default_title, fontsize=24, y=1.03)
    fig.tight_layout()
    save(fig, output_dir, output_name or f"all_datasets_layer_sweep{suffix}{nofilter_suffix}")


def plot_all_datasets_positional_sweep(
    dataset_parquets: list[tuple[str, str]],
    output_dir: Path,
    raw: bool = False,
    min_delta: float = DELTA_THRESHOLD,
    min_pos_samples: int = MIN_POS_SAMPLES,
    metrics: list[str] | None = None,
    title: str | None = None,
    output_name: str | None = None,
) -> None:
    """4-row × n_metrics-col figure: rows = datasets, cols = metrics."""
    if metrics is None:
        metrics = METRICS
    n_datasets = len(dataset_parquets)
    n_metrics = len(metrics)
    suffix = "_raw" if raw else ""
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    fig, axes = plt.subplots(
        n_datasets, n_metrics,
        figsize=(7.5 * n_metrics, 4.5 * n_datasets),
        sharey=False, squeeze=False,
    )

    for row, (dataset, parquet_path) in enumerate(dataset_parquets):
        df, canonical_xs, span_regions, sevs, n_total, n_filtered = _prepare_positional_sweep(
            parquet_path,
            min_delta=min_delta,
            min_pos_samples=min_pos_samples,
        )

        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            _draw_positional_ax(
                ax, df, metric, canonical_xs, span_regions, sevs,
                show_legend=(col == 0 and row == 0),
                show_xlabel=(row == n_datasets - 1),
                raw=raw,
                show_ylabel=(col == 0),
            )
            if col == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS.get(dataset, dataset)}\n"
                    + ({"top1_probability": "Mean answer confidence",
                        "output_entropy":   "Mean entropy",
                        "score":            "Mean accuracy"}.get(metric, metric)
                       if raw else "Aggregate recovery"),
                    fontsize=16,
                )
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=18, fontweight="bold")
            else:
                ax.set_title("")

    filter_label = (f"|Δ top-1| ≥ {min_delta} filter  ·  " if min_delta > 0 else "no filter  ·  ")
    default_title = (
        f"Positional Sweep — Recovery by patched token position  ·  "
        f"{filter_label}n_pos ≥ {min_pos_samples}  ·  95% bootstrap CI"
    )
    fig.suptitle(title if title is not None else default_title, fontsize=24, y=1.03)
    fig.tight_layout()
    save(fig, output_dir, output_name or f"all_datasets_positional_sweep{suffix}{nofilter_suffix}")


# ---------------------------------------------------------------------------
# Two-metric overlay figure: 2 rows (models) × 4 cols (datasets)
# each cell shows answer-confidence recovery + accuracy recovery as two lines
# ---------------------------------------------------------------------------

TWO_METRIC_COLORS = {
    "top1_probability": "#2563eb",  # blue
    "score":            "#d97706",  # amber
}
TWO_METRIC_LEGEND_LABELS = {
    "top1_probability": "Confidence recovery",
    "score":            "Accuracy recovery",
}


def _draw_layer_ax_two_metrics(
    ax: plt.Axes,
    df: pd.DataFrame,
    layers: list[int],
    sevs: list[str],
    show_xlabel: bool,
    show_ylabel: bool = True,
    metrics: list[str] | None = None,
) -> list:
    """Draw recovery curves for two metrics overlaid in one axes.

    Layers in ``df`` are 0-based decoder-block indices (block 0 = first
    transformer block).  The x-axis is displayed as 1-based (block_idx + 1)
    so that tick 1 corresponds to the first transformer block output.

    Returns a list of Line2D handles for use in a figure-level legend.
    """
    if metrics is None:
        metrics = ["top1_probability", "score"]
    linestyles = ["-", "--"]
    legend_handles = []
    for i, metric in enumerate(metrics):
        color = TWO_METRIC_COLORS.get(metric, "#374151")
        ls = linestyles[i % len(linestyles)]
        for sev in sevs:
            sub = df[df["severity"] == sev]
            ys, los, his = [], [], []
            for layer in layers:
                m, lo, hi = aggregate_recovery_ci(sub[sub["layer"] == layer], metric)
                ys.append(m)
                los.append(lo)
                his.append(hi)
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            # x-axis: display as 1-based (layer 0 → tick 1)
            display_xs = [l + 1 for l in layers]
            ax.plot(display_xs, ys, marker="o", linestyle=ls, color=color,
                    linewidth=2, markersize=5, zorder=4)
            ax.fill_between(display_xs, los, his, color=color, alpha=0.12)
        label = TWO_METRIC_LEGEND_LABELS.get(metric, metric)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=2, linestyle=ls, label=label)
        )

    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#6b7280", linewidth=1.5, linestyle=":", alpha=0.85, zorder=2)
    display_xs_all = [l + 1 for l in layers]
    ax.set_xticks(display_xs_all)
    ax.set_xticklabels([str(x) for x in display_xs_all], fontsize=8, rotation=45)
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=14)
    ax.grid(alpha=0.3)
    if show_ylabel:
        ax.set_ylabel("Aggregate recovery", fontsize=14)
    else:
        ax.set_ylabel("")

    legend_handles.append(
        plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",
                   label="Full recovery (RR=1)")
    )
    return legend_handles


def plot_two_metrics_cross_model_layer_sweep(
    model_entries: list[tuple[str, dict[str, str]]],  # [(model_label, {dataset: parquet_path}), ...]
    datasets: list[str],                               # all 4 datasets, in column order
    output_dir: Path,
    output_name: str,
    min_delta: float = 0.0,
    title: str | None = None,
    metrics: list[str] | None = None,
) -> None:
    """2-row × N-col layer-sweep figure overlaying two metrics per cell.

    Rows = models, cols = datasets.
    Each cell shows answer-confidence recovery and accuracy recovery as two lines.
    """
    if metrics is None:
        metrics = ["top1_probability", "score"]

    n_models = len(model_entries)
    n_cols = len(datasets)
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_models),
        sharey=False, squeeze=False,
    )

    fig_legend_handles: list | None = None

    for row, (model_label, dataset_paths) in enumerate(model_entries):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            parquet_path = dataset_paths.get(dataset, "")
            missing = not Path(parquet_path).exists() if parquet_path else True
            if not missing:
                try:
                    df, layers, sevs, n_total, n_filtered = _prepare_layer_sweep(
                        parquet_path, min_delta=min_delta
                    )
                    handles = _draw_layer_ax_two_metrics(
                        ax, df, layers, sevs,
                        show_xlabel=(row == n_models - 1),
                        show_ylabel=(col == 0),
                        metrics=metrics,
                    )
                    if fig_legend_handles is None:
                        fig_legend_handles = handles
                except Exception as exc:
                    missing = True
                    print(f"  WARNING: {model_label}/{dataset}: {exc}")
            if missing:
                ax.text(0.5, 0.5, "Data not available",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=11, color="#9ca3af")
                ax.set_xticks([])
                ax.set_yticks([])

            # Column header: dataset name (top row only)
            if row == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=18, fontweight="bold")
            else:
                ax.set_title("")

            # Row label: model name (leftmost col only)
            if col == 0:
                ax.set_ylabel(f"{model_label}\nAggregate recovery", fontsize=14)

    default_title = "Patching Layer Sweep - Confidence & Accuracy Recovery"
    fig.suptitle(title if title is not None else default_title, fontsize=22, y=1.03)

    # Figure-level horizontal legend between title and subplots
    if fig_legend_handles:
        fig.legend(
            handles=fig_legend_handles,
            labels=[h.get_label() for h in fig_legend_handles],
            loc="upper center",
            ncol=len(fig_legend_handles),
            frameon=False,
            fontsize=12,
            bbox_to_anchor=(0.5, 0.99),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, output_dir, output_name + nofilter_suffix)


def _draw_positional_ax_two_metrics(
    ax: plt.Axes,
    df: pd.DataFrame,
    canonical_xs: list[int],
    span_regions: dict,
    sevs: list[str],
    show_xlabel: bool,
    show_ylabel: bool = True,
    metrics: list[str] | None = None,
) -> list:
    """Draw confidence and accuracy recovery overlaid by patched position."""
    if metrics is None:
        metrics = ["top1_probability", "score"]
    linestyles = ["-", "--"]
    legend_handles = []
    for i, metric in enumerate(metrics):
        color = TWO_METRIC_COLORS.get(metric, "#374151")
        ls = linestyles[i % len(linestyles)]
        for sev in sevs:
            sub = df[df["severity"] == sev]
            ys, los, his = [], [], []
            for cx in canonical_xs:
                m, lo, hi = aggregate_recovery_ci(sub[sub["canonical_x"] == cx], metric)
                ys.append(m)
                los.append(lo)
                his.append(hi)
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            ax.plot(canonical_xs, ys, marker="o", linestyle=ls, color=color,
                    linewidth=2, markersize=3, zorder=4)
            ax.fill_between(canonical_xs, los, his, color=color, alpha=0.12)
        label = TWO_METRIC_LEGEND_LABELS.get(metric, metric)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=2, linestyle=ls, label=label)
        )

    span_patches = []
    all_cx = sorted(cx for cxs in span_regions.values() for cx in cxs)
    tick_positions = []
    tick_labels = []
    tick_spans = []
    for span, cx_list in span_regions.items():
        color = SPAN_COLORS.get(span, "#e5e7eb")
        lo_x, hi_x = min(cx_list) - 0.5, max(cx_list) + 0.5
        ax.axvspan(lo_x, hi_x, color=color, alpha=0.12, zorder=0)
        if all_cx and lo_x > min(all_cx) - 0.5:
            ax.axvline(lo_x, color="#9ca3af", linewidth=0.8, linestyle="-", zorder=1)
        tick_positions.append((lo_x + hi_x) / 2)
        tick_labels.append(SPAN_AXIS_LABELS.get(span, SPAN_DISPLAY.get(span, span)))
        tick_spans.append(span)
        span_patches.append(
            mpatches.Patch(facecolor=color, alpha=0.5, label=SPAN_DISPLAY.get(span, span))
        )

    ax.set_xticks(tick_positions)
    if show_xlabel:
        ax.set_xticklabels(tick_labels, fontsize=11, color="#374151")
        for label, span in zip(ax.get_xticklabels(), tick_spans):
            if span == "question":
                label.set_ha("right")
            elif span == "assistant_marker":
                label.set_ha("left")
            else:
                label.set_ha("center")
        ax.tick_params(axis="x", length=0, pad=8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#6b7280", linewidth=1.5, linestyle=":", alpha=0.85, zorder=2)
    ax.grid(alpha=0.3)
    if show_ylabel:
        ax.set_ylabel("Aggregate recovery", fontsize=14)
    else:
        ax.set_ylabel("")

    legend_handles.append(
        plt.Line2D([0], [0], color="#6b7280", linewidth=1.5, linestyle=":",
                   label="Full recovery (RR=1)")
    )
    return legend_handles + span_patches


def plot_two_metrics_cross_model_positional_sweep(
    model_entries: list[tuple[str, dict[str, str]]],
    datasets: list[str],
    output_dir: Path,
    output_name: str,
    min_delta: float = 0.0,
    min_pos_samples: int = MIN_POS_SAMPLES,
    title: str | None = None,
    metrics: list[str] | None = None,
) -> None:
    """2-row x N-col positional-sweep figure overlaying confidence and accuracy."""
    if metrics is None:
        metrics = ["top1_probability", "score"]

    n_models = len(model_entries)
    n_cols = len(datasets)
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(5.8 * n_cols, 4.6 * n_models),
        sharey=False, squeeze=False,
    )

    fig_legend_handles: list | None = None

    for row, (model_label, dataset_paths) in enumerate(model_entries):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            parquet_path = dataset_paths.get(dataset, "")
            missing = not Path(parquet_path).exists() if parquet_path else True
            if not missing:
                try:
                    df, canonical_xs, span_regions, sevs, n_total, n_filtered = _prepare_positional_sweep(
                        parquet_path,
                        min_delta=min_delta,
                        min_pos_samples=min_pos_samples,
                    )
                    handles = _draw_positional_ax_two_metrics(
                        ax, df, canonical_xs, span_regions, sevs,
                        show_xlabel=(row == n_models - 1),
                        show_ylabel=(col == 0),
                        metrics=metrics,
                    )
                    if fig_legend_handles is None:
                        fig_legend_handles = handles
                except Exception as exc:
                    missing = True
                    print(f"  WARNING: {model_label}/{dataset}: {exc}")
            if missing:
                ax.text(0.5, 0.5, "Data not available",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=11, color="#9ca3af")
                ax.set_xticks([])
                ax.set_yticks([])

            if row == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=18, fontweight="bold")
            else:
                ax.set_title("")

            if col == 0:
                ax.set_ylabel(f"{model_label}\nAggregate recovery", fontsize=14)

    default_title = "Patching Positional Sweep - Confidence & Accuracy Recovery"
    fig.suptitle(title if title is not None else default_title, fontsize=22, y=1.03)

    if fig_legend_handles:
        fig.legend(
            handles=fig_legend_handles,
            labels=[h.get_label() for h in fig_legend_handles],
            loc="upper center",
            ncol=min(len(fig_legend_handles), 7),
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, 0.99),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, output_dir, output_name + nofilter_suffix)


# ---------------------------------------------------------------------------
# Cross-model combined figures (2 rows × 4 cols)
# ---------------------------------------------------------------------------

def plot_cross_model_layer_sweep(
    model_entries: list[tuple[str, dict[str, str]]],  # [(model_label, {dataset: parquet_path}), ...]
    datasets_pair: list[str],                          # exactly 2 dataset keys, in order
    output_dir: Path,
    output_name: str,
    raw: bool = False,
    min_delta: float = DELTA_THRESHOLD,
    title: str | None = None,
) -> None:
    """2-row × 4-col combined layer-sweep figure.

    Rows = models, cols = (ds1 conf, ds1 score, ds2 conf, ds2 score).
    """
    metrics = CROSS_MODEL_METRICS
    n_models  = len(model_entries)
    n_metrics = len(metrics)
    n_cols    = len(datasets_pair) * n_metrics  # 4

    suffix         = "_raw"      if raw           else ""
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(6 * n_cols, 4.5 * n_models),
        sharey=False, squeeze=False,
    )

    for row, (model_label, dataset_paths) in enumerate(model_entries):
        for ds_idx, dataset in enumerate(datasets_pair):
            parquet_path = dataset_paths[dataset]
            missing = not Path(parquet_path).exists()
            if not missing:
                df, layers, sevs, n_total, n_filtered = _prepare_layer_sweep(
                    parquet_path, min_delta=min_delta
                )
            for m_idx, metric in enumerate(metrics):
                col = ds_idx * n_metrics + m_idx
                ax  = axes[row, col]
                if missing:
                    ax.text(0.5, 0.5, "Data not available",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=11, color="#9ca3af")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    _draw_layer_ax(
                        ax, df, metric, layers, sevs,
                        show_legend=(row == 0 and col == 0),
                        show_xlabel=(row == n_models - 1),
                        raw=raw,
                        show_ylabel=(col == 0),
                    )
                # Column header (top row only)
                if row == 0:
                    ax.set_title(
                        f"{DATASET_LABELS.get(dataset, dataset)}\n{METRIC_LABELS[metric]}",
                        fontsize=18, fontweight="bold",
                    )
                else:
                    ax.set_title("")
                # Row label (leftmost col only)
                if col == 0:
                    ylabel_metric = (
                        {"top1_probability": "Mean answer confidence",
                         "output_entropy":   "Mean entropy",
                         "score":            "Mean accuracy"}.get(metric, metric)
                        if raw else "Aggregate recovery"
                    )
                    ax.set_ylabel(f"{model_label}\n{ylabel_metric}", fontsize=16)

    filter_label = (f"|Δ top-1| ≥ {min_delta} filter  ·  " if min_delta > 0 else "no filter  ·  ")
    ds_label = " & ".join(DATASET_LABELS.get(d, d) for d in datasets_pair)
    default_title = f"Layer Sweep — {ds_label}  ·  {filter_label}95% bootstrap CI"
    fig.suptitle(title if title is not None else default_title, fontsize=24, y=1.03)
    fig.tight_layout()
    save(fig, output_dir, output_name + suffix + nofilter_suffix)


def plot_cross_model_positional_sweep(
    model_entries: list[tuple[str, dict[str, str]]],  # [(model_label, {dataset: parquet_path}), ...]
    datasets_pair: list[str],
    output_dir: Path,
    output_name: str,
    raw: bool = False,
    min_delta: float = DELTA_THRESHOLD,
    min_pos_samples: int = MIN_POS_SAMPLES,
    title: str | None = None,
) -> None:
    """2-row × 4-col combined positional-sweep figure.

    Rows = models, cols = (ds1 conf, ds1 score, ds2 conf, ds2 score).
    """
    metrics = CROSS_MODEL_METRICS
    n_models  = len(model_entries)
    n_metrics = len(metrics)
    n_cols    = len(datasets_pair) * n_metrics  # 4

    suffix         = "_raw"      if raw           else ""
    nofilter_suffix = "_nofilter" if min_delta <= 0 else ""

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(7.5 * n_cols, 4.5 * n_models),
        sharey=False, squeeze=False,
    )

    for row, (model_label, dataset_paths) in enumerate(model_entries):
        for ds_idx, dataset in enumerate(datasets_pair):
            parquet_path = dataset_paths[dataset]
            data_ok = False
            if Path(parquet_path).exists():
                try:
                    df, canonical_xs, span_regions, sevs, n_total, n_filtered = _prepare_positional_sweep(
                        parquet_path,
                        min_delta=min_delta,
                        min_pos_samples=min_pos_samples,
                    )
                    data_ok = True
                except Exception as exc:
                    print(f"  WARNING: could not prepare positional sweep for {model_label}/{dataset}: {exc}")
            for m_idx, metric in enumerate(metrics):
                col = ds_idx * n_metrics + m_idx
                ax  = axes[row, col]
                if not data_ok:
                    ax.text(0.5, 0.5, "Data not available",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=11, color="#9ca3af")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    _draw_positional_ax(
                        ax, df, metric, canonical_xs, span_regions, sevs,
                        show_legend=(row == 0 and col == 0),
                        show_xlabel=(row == n_models - 1),
                        raw=raw,
                        show_ylabel=(col == 0),
                    )
                if row == 0:
                    ax.set_title(
                        f"{DATASET_LABELS.get(dataset, dataset)}\n{METRIC_LABELS[metric]}",
                        fontsize=18, fontweight="bold",
                    )
                else:
                    ax.set_title("")
                if col == 0:
                    ylabel_metric = (
                        {"top1_probability": "Mean answer confidence",
                         "output_entropy":   "Mean entropy",
                         "score":            "Mean accuracy"}.get(metric, metric)
                        if raw else "Aggregate recovery"
                    )
                    ax.set_ylabel(f"{model_label}\n{ylabel_metric}", fontsize=16)

    filter_label = (f"|Δ top-1| ≥ {min_delta} filter  ·  " if min_delta > 0 else "no filter  ·  ")
    ds_label = " & ".join(DATASET_LABELS.get(d, d) for d in datasets_pair)
    default_title = f"Positional Sweep — {ds_label}  ·  {filter_label}n_pos ≥ {min_pos_samples}  ·  95% bootstrap CI"
    fig.suptitle(title if title is not None else default_title, fontsize=24, y=1.03)
    fig.tight_layout()
    save(fig, output_dir, output_name + suffix + nofilter_suffix)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PATCHING_ROOT        = Path("/projects/prjs2014/patching")
LAYER_SWEEP_ROOT      = _PATCHING_ROOT / "layer_sweep_results/llava-hf_llava-1-5-7b-hf"
POSITIONAL_SWEEP_ROOT = _PATCHING_ROOT / "positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf"
OUTPUT_ROOT           = _PATCHING_ROOT / "filtered_plots/llava-hf_llava-1-5-7b-hf"

QWEN_LAYER_SWEEP_ROOT      = _PATCHING_ROOT / "layer_sweep_results/Qwen_Qwen3-5-9B"
QWEN_POSITIONAL_SWEEP_ROOT = _PATCHING_ROOT / "positional_patching_results_including_visual/Qwen_Qwen3-5-9B"

CROSS_MODEL_OUTPUT_ROOT = _PATCHING_ROOT / "filtered_plots/cross_model"

DATASETS = ["vqa-v2", "pope", "coco-qa-vi", "imagenet-r"]

# Dataset pairs for cross-model combined figures
_PAIR_VQA_COCO        = ["vqa-v2", "coco-qa-vi"]
_PAIR_IMAGENET_POPE   = ["imagenet-r", "pope"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--layer_sweep_root",      default=str(LAYER_SWEEP_ROOT))
    parser.add_argument("--positional_sweep_root", default=str(POSITIONAL_SWEEP_ROOT))
    parser.add_argument("--output_root",           default=str(OUTPUT_ROOT))
    # Second model (Qwen by default)
    parser.add_argument("--second_layer_sweep_root",      default=str(QWEN_LAYER_SWEEP_ROOT),
                        help="Layer-sweep root for the second model (default: Qwen3.5-9B)")
    parser.add_argument("--second_positional_sweep_root", default=str(QWEN_POSITIONAL_SWEEP_ROOT),
                        help="Positional-sweep root for the second model (default: Qwen3.5-9B)")
    parser.add_argument("--first_model_label",  default="LLaVA-1.5-7B",
                        help="Display name for the first model")
    parser.add_argument("--second_model_label", default="Qwen3.5-9B",
                        help="Display name for the second model")
    parser.add_argument("--cross_model_output_root", default=str(CROSS_MODEL_OUTPUT_ROOT),
                        help="Output directory for cross-model combined figures")
    parser.add_argument("--second_model_sweep_subdir", default="extreme",
                        help="Subdirectory inserted between <dataset>/ and the parquet file for "
                             "the second model (e.g. 'extreme' for Qwen which stores data per severity)")
    parser.add_argument("--raw", action="store_true",
                        help="Plot raw metric values instead of recovery ratio")
    parser.add_argument("--no_filter", action="store_true",
                        help="Disable |delta| >= threshold filter (use all samples)")
    parser.add_argument("--exclude_entropy", action="store_true",
                        help="Omit output_entropy column from the combined figures")
    parser.add_argument("--skip_per_dataset", action="store_true",
                        help="Skip per-dataset and all-datasets figures; only produce cross-model figures")
    args = parser.parse_args()

    min_delta = 0.0 if args.no_filter else DELTA_THRESHOLD
    active_metrics = [m for m in METRICS if not (args.exclude_entropy and m == "output_entropy")]

    layer_parquets = [
        (ds, Path(args.layer_sweep_root) / ds / "layer_sweep_results.parquet")
        for ds in args.datasets
    ]
    pos_parquets = [
        (ds, Path(args.positional_sweep_root) / ds / "positional_patching_results.parquet")
        for ds in args.datasets
    ]
    shared_out = Path(args.output_root)

    if not args.skip_per_dataset:
        # Per-dataset figures
        for dataset, layer_parquet in layer_parquets:
            pos_parquet = Path(args.positional_sweep_root) / dataset / "positional_patching_results.parquet"
            out = shared_out / dataset
            print(f"\n=== {dataset} ===")
            print("  Layer sweep ...")
            plot_layer_sweep_combined(str(layer_parquet), out, raw=args.raw, min_delta=min_delta)
            print("  Positional sweep ...")
            plot_positional_sweep_combined(str(pos_parquet), out, raw=args.raw, min_delta=min_delta)

        # All-datasets combined figures
        suffix = "_raw" if args.raw else ""
        nofilter_suffix = "_nofilter" if args.no_filter else ""
        print(f"\n=== All-datasets layer sweep{suffix}{nofilter_suffix} ===")
        plot_all_datasets_layer_sweep(
            [(ds, str(p)) for ds, p in layer_parquets],
            shared_out, raw=args.raw, min_delta=min_delta,
            metrics=active_metrics,
            title="Final Prompt Token Activation Patching: Layer Sweep",
        )
        print(f"=== All-datasets positional sweep{suffix}{nofilter_suffix} ===")
        plot_all_datasets_positional_sweep(
            [(ds, str(p)) for ds, p in pos_parquets],
            shared_out, raw=args.raw, min_delta=min_delta,
            metrics=active_metrics,
            title="Final Prompt Token Activation Patching: Positional Sweep",
        )

    # Cross-model combined figures (2 rows × 4 cols)
    cross_out = Path(args.cross_model_output_root)

    subdir = args.second_model_sweep_subdir

    def _layer_path(root: str, ds: str, sub: str = "") -> str:
        base = Path(root) / ds
        return str((base / sub / "layer_sweep_results.parquet") if sub else (base / "layer_sweep_results.parquet"))

    def _pos_path(root: str, ds: str, sub: str = "") -> str:
        base = Path(root) / ds
        return str((base / sub / "positional_patching_results.parquet") if sub else (base / "positional_patching_results.parquet"))

    model_entries_layer = [
        (args.first_model_label,  {ds: _layer_path(args.layer_sweep_root, ds)             for ds in DATASETS}),
        (args.second_model_label, {ds: _layer_path(args.second_layer_sweep_root, ds, subdir) for ds in DATASETS}),
    ]
    model_entries_pos = [
        (args.first_model_label,  {ds: _pos_path(args.positional_sweep_root, ds)             for ds in DATASETS}),
        (args.second_model_label, {ds: _pos_path(args.second_positional_sweep_root, ds, subdir) for ds in DATASETS}),
    ]

    for pair, pair_name in [(_PAIR_VQA_COCO, "vqa_coco"), (_PAIR_IMAGENET_POPE, "imagenet_pope")]:
        print(f"\n=== Cross-model layer sweep: {pair} ===")
        plot_cross_model_layer_sweep(
            model_entries_layer, pair,
            output_dir=cross_out,
            output_name=f"cross_model_layer_sweep_{pair_name}",
            raw=args.raw, min_delta=min_delta,
        )
        print(f"=== Cross-model positional sweep: {pair} ===")
        plot_cross_model_positional_sweep(
            model_entries_pos, pair,
            output_dir=cross_out,
            output_name=f"cross_model_positional_sweep_{pair_name}",
            raw=args.raw, min_delta=min_delta,
        )


if __name__ == "__main__":
    main()
