"""Analysis plots for activation patching results.

Reads ``patching_results.parquet`` produced by ``src/patching/run_patching.py``
and generates six publication-quality figures:

  1  Patching Effect Matrix     – heatmap, span × severity, recovery_entropy
  2  Dose-Response Curves       – entropy vs degradation level per span
  3  Causal Specificity         – bars, primary vs control conditions
  4  Accuracy-Confidence Dissociation – scatter Δscore vs Δentropy
  5  Layer Sweep                – recovery_entropy vs layer index
  6  Cross-Span Probe Propagation – placeholder (requires separate probe data)

Usage::

    python -m src.cli.plot_patching_results \\
        --results_parquet /path/to/patching_results.parquet \\
        --output_dir /path/to/figures/
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not installed — Wilcoxon tests will be skipped in Figure 3.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPAN_RE: Dict[str, re.Pattern] = {
    "lm_visual_lasttok":   re.compile(r"^lm_visual_lasttok_layer_(\d+)$"),
    "lm_question_lasttok": re.compile(r"^lm_question_lasttok_layer_(\d+)$"),
    "lm_fullspan_lasttok": re.compile(r"^lm_fullspan_lasttok_layer_(\d+)$"),
    "lm_visual_span":      re.compile(r"^lm_visual_span_layer_(\d+)$"),
    "control_visual":      re.compile(r"^control_visual_lasttok_layer_(\d+)$"),
    "control_question":    re.compile(r"^control_question_lasttok_layer_(\d+)$"),
    "control_fullspan":    re.compile(r"^control_fullspan_lasttok_layer_(\d+)$"),
}

SPAN_LABELS: Dict[str, str] = {
    "lm_visual_lasttok":   "Visual\n(last tok)",
    "lm_question_lasttok": "Question\n(last tok)",
    "lm_fullspan_lasttok": "Last prompt\ntoken",
    "lm_visual_span":      "Full visual\nspan",
    "control_visual":      "Control\n(visual-size random pos)",
    "control_question":    "Control\n(question-size random pos)",
    "control_fullspan":    "Control\n(full-span random pos)",
}

SPAN_COLORS: Dict[str, str] = {
    "lm_visual_lasttok":   "#0369a1",
    "lm_question_lasttok": "#c2410c",
    "lm_fullspan_lasttok": "#4338ca",
    "lm_visual_span":      "#0f766e",
    "control_visual":      "#9ca3af",
    "control_question":    "#d1d5db",
    "control_fullspan":    "#6b7280",
    "random_sample":       "#d97706",
    "random_layer":        "#b45309",
}

SEVERITY_COLORS: Dict[str, str] = {
    "mild":     "#0ea5e9",
    "moderate": "#f59e0b",
    "severe":   "#ef4444",
}
SEVERITY_ORDER = ["mild", "moderate", "severe"]

# Spans that appear in Figure 2 panels, in order
FIG2_SPANS = ["lm_visual_lasttok", "lm_question_lasttok", "lm_fullspan_lasttok"]

EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def _classify_condition(condition: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (span_type, layer_idx) or (None, None) for baseline conditions."""
    for span_type, pat in _SPAN_RE.items():
        m = pat.match(condition)
        if m:
            return span_type, int(m.group(1))
    return None, None


def load_data(results_parquet: str) -> pd.DataFrame:
    """Load parquet and add ``span_type`` / ``layer_idx`` columns."""
    df = pd.read_parquet(results_parquet)
    span_types, layer_idxs = [], []
    for cond in df["condition"]:
        st, li = _classify_condition(cond)
        span_types.append(st)
        layer_idxs.append(li)
    df = df.copy()
    df["span_type"] = span_types
    df["layer_idx"] = pd.array(layer_idxs, dtype="Int64")
    return df


def _compute_recovery(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return patched rows with a ``recovery`` column for *metric*.

    recovery_i = clip((patched_i - deg_i) / (clean_i - deg_i + eps), -5, 5)
    """
    clean = (
        df[df["condition"] == "clean"]
        .drop_duplicates(["sample_id", "severity"])
        [["sample_id", "severity", metric]]
        .rename(columns={metric: f"{metric}_clean"})
    )
    deg = (
        df[df["condition"] == "degraded"]
        .drop_duplicates(["sample_id", "severity"])
        [["sample_id", "severity", metric]]
        .rename(columns={metric: f"{metric}_deg"})
    )
    patched = df[df["span_type"].notna()].copy()
    patched = patched.merge(clean, on=["sample_id", "severity"], how="left")
    patched = patched.merge(deg, on=["sample_id", "severity"], how="left")

    m_patch = patched[metric].values
    m_clean = patched[f"{metric}_clean"].values
    m_deg = patched[f"{metric}_deg"].values

    raw = (m_patch - m_deg) / (m_clean - m_deg + EPSILON)
    patched["recovery"] = np.clip(raw, -5, 5)
    return patched


def _peak_layer(rec_df: pd.DataFrame) -> int:
    """Layer index with the best mean recovery across all severities."""
    if rec_df.empty or rec_df["layer_idx"].isna().all():
        return 16
    return int(rec_df.groupby("layer_idx")["recovery"].mean().idxmax())


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) via bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return (
        float(values.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


def _wilcoxon_bonferroni(
    primary: np.ndarray,
    controls: List[Tuple[str, np.ndarray]],
    alpha: float = 0.05,
) -> List[Tuple[str, float, bool]]:
    """Paired Wilcoxon signed-rank tests with Bonferroni correction.

    Returns list of (label, p_corrected, significant).
    """
    if not _SCIPY or len(controls) == 0:
        return [(label, float("nan"), False) for label, _ in controls]

    n = len(controls)
    results = []
    for label, ctrl in controls:
        # Align lengths — use shared sample indices
        min_len = min(len(primary), len(ctrl))
        if min_len < 2:
            results.append((label, float("nan"), False))
            continue
        diff = primary[:min_len] - ctrl[:min_len]
        if np.all(diff == 0):
            results.append((label, 1.0, False))
            continue
        try:
            _, p_raw = _scipy_wilcoxon(diff, alternative="two-sided")
            p_corr = min(p_raw * n, 1.0)
            results.append((label, p_corr, p_corr < alpha))
        except Exception:
            results.append((label, float("nan"), False))
    return results


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Saving utility
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("pdf", {}), ("png", {"dpi": 200})]:
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"  Saved {output_dir}/{name}.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 1: Patching Effect Matrix (heatmap)
# ---------------------------------------------------------------------------

def figure_1_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str = "output_entropy",
) -> None:
    """Heatmap of mean recovery (metric) per span × severity.

    Uses the peak layer for each span type.  Cells annotated with mean ± std
    of the per-sample recovery distribution.
    """
    rec = _compute_recovery(df, metric)

    # Determine which span types are present and their peak layers
    available_spans = [
        s for s in [
            "lm_visual_span",
            "lm_visual_lasttok",
            "lm_question_lasttok",
            "lm_fullspan_lasttok",
        ]
        if s in rec["span_type"].values
    ]
    if not available_spans:
        warnings.warn("Figure 1: no patching conditions found.")
        return

    peak_layers: Dict[str, int] = {}
    for span in available_spans:
        sub = rec[rec["span_type"] == span]
        peak_layers[span] = _peak_layer(sub)

    available_sevs = [s for s in SEVERITY_ORDER if s in rec["severity"].values]
    n_rows, n_cols = len(available_spans), len(available_sevs)

    matrix_mean = np.full((n_rows, n_cols), np.nan)
    matrix_std = np.full((n_rows, n_cols), np.nan)

    for r, span in enumerate(available_spans):
        layer = peak_layers[span]
        sub = rec[(rec["span_type"] == span) & (rec["layer_idx"] == layer)]
        for c, sev in enumerate(available_sevs):
            vals = sub[sub["severity"] == sev]["recovery"].values
            if len(vals) > 0:
                matrix_mean[r, c] = float(np.nanmean(vals))
                matrix_std[r, c] = float(np.nanstd(vals))

    # Diverging colormap: blue = positive recovery, red = negative
    v_abs = np.nanmax(np.abs(matrix_mean))
    v_abs = max(v_abs, 0.05)  # avoid zero range
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-v_abs, vmax=v_abs)

    fig, ax = plt.subplots(figsize=(3 + n_cols * 1.5, 1.5 + n_rows * 1.2))
    im = ax.imshow(matrix_mean, cmap="RdBu", norm=norm, aspect="auto")

    # Cell annotations
    for r in range(n_rows):
        for c in range(n_cols):
            mn = matrix_mean[r, c]
            sd = matrix_std[r, c]
            if not np.isnan(mn):
                text = f"{mn:+.2f}\n±{sd:.2f}"
                brightness = abs(mn) / v_abs
                color = "white" if brightness > 0.5 else "black"
                ax.text(c, r, text, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    # Axes labels
    row_labels = [
        f"{SPAN_LABELS.get(s, s).replace(chr(10), ' ')} (L{peak_layers[s]})"
        for s in available_spans
    ]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([s.capitalize() for s in available_sevs], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Degradation severity", fontsize=12)
    ax.set_ylabel("Patching span", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Recovery ({metric.replace('_', ' ')})", fontsize=10)
    cbar.ax.axhline(0, color="black", linewidth=1.5)

    metric_label = metric.replace("output_", "").replace("_", " ")
    ax.set_title(
        f"Figure 1 — Patching Effect Matrix\n"
        f"Recovery of {metric_label} · blue = recovery · red = harm",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, output_dir, "figure_1_patching_effect_matrix")


# ---------------------------------------------------------------------------
# Figure 2: Dose-Response Curves
# ---------------------------------------------------------------------------

def figure_2_dose_response(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str = "output_entropy",
    probe_df: Optional[pd.DataFrame] = None,
) -> None:
    """One panel per primary span: metric vs degradation level.

    Lines: unpatched degraded, patched at peak layer, clean baseline.
    Error bands: ±1 std across samples.
    ``probe_df``: optional DataFrame with columns [severity, condition, probe_score]
    for a right Y-axis overlay.
    """
    rec = _compute_recovery(df, metric)
    available_sevs = [s for s in SEVERITY_ORDER if s in df["severity"].values]
    x_labels = available_sevs
    x_pos = np.arange(len(x_labels))

    fig, axes = plt.subplots(
        1, len(FIG2_SPANS), figsize=(5 * len(FIG2_SPANS), 5), sharey=True,
    )
    if len(FIG2_SPANS) == 1:
        axes = [axes]

    for ax, span in zip(axes, FIG2_SPANS):
        # Peak layer for this span
        sub_rec = rec[rec["span_type"] == span]
        if sub_rec.empty:
            ax.set_title(SPAN_LABELS.get(span, span).replace("\n", " ") + "\n(no data)")
            continue
        peak = _peak_layer(sub_rec)

        # Degraded baseline: mean ± std per severity
        deg_rows = df[df["condition"] == "degraded"]
        deg_mean, deg_std = [], []
        for sev in available_sevs:
            v = deg_rows[deg_rows["severity"] == sev][metric].values
            deg_mean.append(np.nanmean(v))
            deg_std.append(np.nanstd(v))
        deg_mean, deg_std = np.array(deg_mean), np.array(deg_std)

        # Patched: mean ± std per severity at peak layer
        patch_rows = rec[(rec["span_type"] == span) & (rec["layer_idx"] == peak)]
        pat_mean, pat_std = [], []
        for sev in available_sevs:
            v = patch_rows[patch_rows["severity"] == sev][metric].values
            pat_mean.append(np.nanmean(v))
            pat_std.append(np.nanstd(v))
        pat_mean, pat_std = np.array(pat_mean), np.array(pat_std)

        # Clean baseline (horizontal reference)
        clean_mean = df[df["condition"] == "clean"][metric].mean()
        clean_std = df[df["condition"] == "clean"][metric].std()

        # Plot degraded
        ax.plot(x_pos, deg_mean, "o-", color="#ef4444", linewidth=2,
                label="Degraded (unpatched)", zorder=3)
        ax.fill_between(x_pos, deg_mean - deg_std, deg_mean + deg_std,
                        color="#ef4444", alpha=0.15)

        # Plot patched
        ax.plot(x_pos, pat_mean, "s--", color=SPAN_COLORS.get(span, "#4338ca"),
                linewidth=2, label=f"Patched (L{peak})", zorder=4)
        ax.fill_between(x_pos, pat_mean - pat_std, pat_mean + pat_std,
                        color=SPAN_COLORS.get(span, "#4338ca"), alpha=0.15)

        # Clean reference
        ax.axhline(clean_mean, color="#22c55e", linewidth=1.5,
                   linestyle=":", label="Clean baseline", zorder=2)
        ax.axhspan(clean_mean - clean_std, clean_mean + clean_std,
                   color="#22c55e", alpha=0.08)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.capitalize() for s in x_labels])
        ax.set_xlabel("Degradation severity", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize=10)
        ax.set_title(SPAN_LABELS.get(span, span).replace("\n", " "), fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        # Optional right Y-axis for probe scores
        if probe_df is not None and span in probe_df.get("span_type", pd.Series()).values:
            ax2 = ax.twinx()
            for sev in available_sevs:
                pv = probe_df[
                    (probe_df["span_type"] == span) & (probe_df["severity"] == sev)
                ]["probe_score"].values
                # (simple scatter; extend as needed for your probe format)
            ax2.set_ylabel("Probe confidence", fontsize=9, color="#7c3aed")
            ax2.tick_params(axis="y", labelcolor="#7c3aed")

    fig.suptitle("Figure 2 — Dose-Response Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "figure_2_dose_response")


# ---------------------------------------------------------------------------
# Figure 3: Causal Specificity
# ---------------------------------------------------------------------------

def figure_3_causal_specificity(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    n_bootstrap: int = 1000,
    metric: str = "output_entropy",
) -> None:
    """Bar chart comparing primary conditions vs controls.

    Controls:
     - Same span at a non-peak layer  ("random layer" control)
     - Permuted sample assignment     ("random sample" control)

    Bootstrap 95 % CI on error bars.  Wilcoxon signed-rank tests (Bonferroni)
    between primary and each control, marked with *, **, ***.
    """
    rec = _compute_recovery(df, metric)
    available_sevs = [s for s in SEVERITY_ORDER if s in rec["severity"].values]

    # Identify available spans and their peak / non-peak layers
    all_spans = [s for s in [
        "lm_visual_span", "lm_visual_lasttok", "lm_question_lasttok", "lm_fullspan_lasttok"
    ] if s in rec["span_type"].values]

    peak_layers: Dict[str, int] = {}
    alt_layers: Dict[str, Optional[int]] = {}   # for "random layer" control
    for span in all_spans:
        sub = rec[rec["span_type"] == span]
        pk = _peak_layer(sub)
        peak_layers[span] = pk
        # Alternative layer: any available layer that isn't the peak
        available = [int(l) for l in sub["layer_idx"].dropna().unique() if int(l) != pk]
        alt_layers[span] = int(available[0]) if available else None

    # Build condition list: primary spans + random layer + random sample
    conditions: List[Dict] = []
    for span in all_spans:
        conditions.append({
            "key": f"{span}_peak",
            "label": SPAN_LABELS.get(span, span).replace("\n", " ") + f"\n(L{peak_layers[span]})",
            "span": span,
            "layer": peak_layers[span],
            "group": "primary",
            "color": SPAN_COLORS.get(span, "#6b7280"),
        })

    # Random layer control: pick one representative span (lm_fullspan_lasttok) at alt layer
    rl_span = "lm_fullspan_lasttok"
    if rl_span in all_spans and alt_layers.get(rl_span) is not None:
        conditions.append({
            "key": f"{rl_span}_alt",
            "label": f"Random layer\n(L{alt_layers[rl_span]})",
            "span": rl_span,
            "layer": alt_layers[rl_span],
            "group": "control",
            "color": SPAN_COLORS["random_layer"],
        })

    # Random sample control: computed synthetically (see below)
    conditions.append({
        "key": "random_sample",
        "label": "Random\nsample",
        "span": None,
        "layer": None,
        "group": "control",
        "color": SPAN_COLORS["random_sample"],
    })

    rng = np.random.default_rng(42)
    n_conds = len(conditions)
    x = np.arange(n_conds)
    bar_w = 0.22
    n_sevs = len(available_sevs)

    fig, ax = plt.subplots(figsize=(max(10, n_conds * 2.2), 5))

    # For Wilcoxon: gather primary vs control recovery arrays per severity
    # (key = severity, value = dict of condition_key -> per-sample recovery)
    per_sample: Dict[str, Dict[str, np.ndarray]] = {sev: {} for sev in available_sevs}

    for cond in conditions:
        for sev in available_sevs:
            if cond["key"] == "random_sample":
                # Use lm_fullspan_lasttok at peak layer, but permute the patched metric
                base_span = "lm_fullspan_lasttok"
                if base_span not in all_spans:
                    per_sample[sev]["random_sample"] = np.array([])
                    continue
                sub = rec[
                    (rec["span_type"] == base_span) &
                    (rec["layer_idx"] == peak_layers[base_span]) &
                    (rec["severity"] == sev)
                ].copy()
                if len(sub) < 2:
                    per_sample[sev]["random_sample"] = np.array([])
                    continue
                # Permute the patched metric, keep clean/deg baselines fixed
                perm_idx = rng.permutation(len(sub))
                m_perm = sub[metric].values[perm_idx]
                m_clean = sub[f"{metric}_clean"].values
                m_deg = sub[f"{metric}_deg"].values
                ctrl_rec = np.clip(
                    (m_perm - m_deg) / (m_clean - m_deg + EPSILON), -5, 5
                )
                per_sample[sev]["random_sample"] = ctrl_rec
            else:
                sub = rec[
                    (rec["span_type"] == cond["span"]) &
                    (rec["layer_idx"] == cond["layer"]) &
                    (rec["severity"] == sev)
                ]
                per_sample[sev][cond["key"]] = sub["recovery"].values

    # Draw bars
    for si, (sev, sev_color) in enumerate(SEVERITY_COLORS.items()):
        if sev not in available_sevs:
            continue
        means, lows, highs = [], [], []
        for cond in conditions:
            vals = per_sample[sev].get(cond["key"], np.array([]))
            m, lo, hi = _bootstrap_ci(vals, n_bootstrap=n_bootstrap)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        means = np.array(means, dtype=float)
        lows = np.array(lows, dtype=float)
        highs = np.array(highs, dtype=float)
        offset = (si - n_sevs / 2 + 0.5) * bar_w

        ax.bar(x + offset, means, bar_w, color=sev_color, alpha=0.75,
               label=sev.capitalize(), zorder=3)

        ye_lo = np.where(np.isnan(means - lows), 0, means - lows)
        ye_hi = np.where(np.isnan(highs - means), 0, highs - means)
        ax.errorbar(
            x + offset, means,
            yerr=[np.abs(ye_lo), np.abs(ye_hi)],
            fmt="none", color="black", capsize=3, linewidth=1, zorder=4,
        )

    # Wilcoxon tests: each primary vs random_sample control (using moderate severity)
    test_sev = "moderate" if "moderate" in available_sevs else available_sevs[-1]
    rs_vals = per_sample[test_sev].get("random_sample", np.array([]))
    rl_key = (f"{rl_span}_alt" if rl_span in all_spans and alt_layers.get(rl_span) is not None
               else None)

    y_top = ax.get_ylim()[1]
    for ci, cond in enumerate(conditions):
        if cond["group"] != "primary":
            continue
        primary_vals = per_sample[test_sev].get(cond["key"], np.array([]))
        controls_for_test = [("vs random sample", rs_vals)]
        if rl_key:
            rl_vals = per_sample[test_sev].get(rl_key, np.array([]))
            controls_for_test.append(("vs random layer", rl_vals))

        results = _wilcoxon_bonferroni(primary_vals, controls_for_test)
        stars_text = " / ".join(
            [_sig_stars(p) for _, p, _ in results if _sig_stars(p)]
        ) or ""
        if stars_text:
            y_label = ax.get_ylim()[1] * 1.02
            ax.text(ci, y_label, stars_text, ha="center", va="bottom", fontsize=11, color="#1e293b")

    # Separator line between primary and controls
    primary_count = sum(1 for c in conditions if c["group"] == "primary")
    if primary_count < n_conds:
        ax.axvline(primary_count - 0.5, color="#94a3b8", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.text(primary_count - 0.5, ax.get_ylim()[0],
                " controls →", fontsize=8, color="#94a3b8", va="bottom")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c["label"] for c in conditions], rotation=30, ha="right", fontsize=9,
    )
    ax.set_xlabel("Patching condition")
    ax.set_ylabel(f"Recovery ({metric.replace('_', ' ')})", fontsize=11)
    ax.legend(title="Severity", fontsize=9)
    ax.set_title(
        f"Figure 3 — Causal Specificity (n={n_bootstrap} bootstrap, "
        f"Wilcoxon Bonferroni-corrected at {test_sev})",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "figure_3_causal_specificity")


# ---------------------------------------------------------------------------
# Figure 4: Accuracy-Confidence Dissociation
# ---------------------------------------------------------------------------

def figure_4_dissociation(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    severity: str = "moderate",
) -> None:
    """Scatter: Δaccuracy (x) vs Δconfidence (y) per patching condition.

    Δaccuracy  = mean_patched_score - mean_degraded_score
    Δconfidence = mean_degraded_entropy - mean_patched_entropy  (positive = more confident)

    Each point is one patching condition (aggregated across samples).
    Colour and marker encode span type.
    """
    rec_score   = _compute_recovery(df, "score")
    rec_entropy = _compute_recovery(df, "output_entropy")

    sev_df_s = rec_score[rec_score["severity"] == severity]
    sev_df_e = rec_entropy[rec_entropy["severity"] == severity]

    if sev_df_s.empty:
        warnings.warn(f"Figure 4: no data for severity='{severity}'.")
        return

    # Reference baselines for the chosen severity
    deg_rows = df[(df["condition"] == "degraded") & (df["severity"] == severity)]
    clean_rows = df[(df["condition"] == "clean") & (df["severity"] == severity)]
    deg_score_mean = deg_rows["score"].mean()
    deg_entropy_mean = deg_rows["output_entropy"].mean()

    # Aggregate per condition
    points = []
    all_conditions = sev_df_s[["span_type", "layer_idx", "condition"]].drop_duplicates()
    for _, row in all_conditions.iterrows():
        span, layer, cond = row["span_type"], row["layer_idx"], row["condition"]
        score_vals = sev_df_s[sev_df_s["condition"] == cond]["score"].values
        entropy_vals = sev_df_e[sev_df_e["condition"] == cond]["output_entropy"].values
        if len(score_vals) == 0 or len(entropy_vals) == 0:
            continue
        delta_score = float(np.nanmean(score_vals)) - deg_score_mean
        delta_conf = deg_entropy_mean - float(np.nanmean(entropy_vals))
        short_label = f"{span.split('_')[1][:3].upper()} L{layer}" if layer is not None else str(span)
        points.append({
            "span": span,
            "layer": int(layer) if layer is not None else -1,
            "condition": cond,
            "delta_score": delta_score,
            "delta_conf": delta_conf,
            "label": short_label,
        })

    if not points:
        warnings.warn("Figure 4: no points to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    span_list = sorted({p["span"] for p in points})
    span_marker = {s: markers[i % len(markers)] for i, s in enumerate(span_list)}

    for p in points:
        color = SPAN_COLORS.get(p["span"], "#6b7280")
        ax.scatter(
            p["delta_score"], p["delta_conf"],
            color=color, marker=span_marker[p["span"]],
            s=90, zorder=4, edgecolors="white", linewidths=0.5,
        )
        ax.annotate(
            p["label"],
            xy=(p["delta_score"], p["delta_conf"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7.5, color="#374151",
        )

    # Reference lines
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#9ca3af", linewidth=0.8, linestyle="--")

    # Legend for span types
    legend_handles = [
        plt.scatter([], [], color=SPAN_COLORS.get(s, "#6b7280"),
                    marker=span_marker[s], s=70,
                    label=SPAN_LABELS.get(s, s).replace("\n", " "))
        for s in span_list
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best", title="Span type")

    ax.set_xlabel("Δ Accuracy  (patched − degraded)", fontsize=11)
    ax.set_ylabel("Δ Confidence  (degraded − patched entropy, ↑ = more confident)", fontsize=11)
    ax.set_title(
        f"Figure 4 — Accuracy vs Confidence Dissociation ({severity} degradation)\n"
        "Entangled: both axes move together · Dissociated: one moves, other stays flat",
        fontsize=11,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, output_dir, f"figure_4_dissociation_{severity}")


# ---------------------------------------------------------------------------
# Figure 5: Layer Sweep for Last Prompt Token
# ---------------------------------------------------------------------------

def figure_5_layer_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    span: str = "lm_fullspan_lasttok",
    metric: str = "output_entropy",
    probe_peak_layer: Optional[int] = None,
) -> None:
    """Recovery metric vs layer index for the specified span (default: last prompt token).

    One line per severity.  Vertical dashed line marks the probe peak layer
    if provided.
    """
    rec = _compute_recovery(df, metric)
    sub = rec[rec["span_type"] == span]

    if sub.empty:
        warnings.warn(f"Figure 5: span '{span}' not found in data.")
        return

    available_layers = sorted(sub["layer_idx"].dropna().unique().astype(int))
    available_sevs = [s for s in SEVERITY_ORDER if s in sub["severity"].values]

    fig, ax = plt.subplots(figsize=(max(6, len(available_layers) * 1.4), 4.5))

    for sev in available_sevs:
        color = SEVERITY_COLORS.get(sev, "#6b7280")
        ys, yerr_lo, yerr_hi = [], [], []
        for layer in available_layers:
            vals = sub[(sub["layer_idx"] == layer) & (sub["severity"] == sev)]["recovery"].values
            m, lo, hi = _bootstrap_ci(vals)
            ys.append(m)
            yerr_lo.append(m - lo)
            yerr_hi.append(hi - m)

        ys = np.array(ys, dtype=float)
        ax.plot(available_layers, ys, "o-", color=color, linewidth=2,
                markersize=7, label=sev.capitalize(), zorder=4)
        ax.fill_between(
            available_layers,
            ys - np.array(yerr_lo, dtype=float),
            ys + np.array(yerr_hi, dtype=float),
            color=color, alpha=0.15,
        )

    if probe_peak_layer is not None:
        ax.axvline(probe_peak_layer, color="#7c3aed", linewidth=1.5,
                   linestyle="--", alpha=0.8,
                   label=f"Probe peak (L{probe_peak_layer})")

    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xticks(available_layers)
    ax.set_xticklabels([str(l) for l in available_layers])
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(f"Recovery ({metric.replace('_', ' ')})", fontsize=11)
    ax.set_title(
        f"Figure 5 — Layer Sweep: {SPAN_LABELS.get(span, span).replace(chr(10), ' ')}\n"
        f"(recovery of {metric.replace('output_', '').replace('_', ' ')} "
        f"vs layer index, {len(available_layers)} layer(s) available)",
        fontsize=11,
    )
    ax.legend(title="Severity", fontsize=9)
    ax.grid(alpha=0.3)
    if len(available_layers) <= 5:
        ax.text(
            0.99, 0.02,
            "Note: only partial layer sweep available.\nRe-run with different --middle_layer values for a full sweep.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#9ca3af",
        )
    fig.tight_layout()
    _save(fig, output_dir, f"figure_5_layer_sweep_{span.replace('lm_', '')}")


# ---------------------------------------------------------------------------
# Figure 6: Cross-Span Probe Propagation (placeholder / optional)
# ---------------------------------------------------------------------------

def figure_6_probe_propagation(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    probe_propagation_df: Optional[pd.DataFrame] = None,
) -> None:
    """Grouped bar chart: Δprobe_confidence at last_prompt_token after patching at other spans.

    Requires a ``probe_propagation_df`` with columns:
      patching_location, severity, delta_probe_score, sample_id

    This data is NOT produced by the current pipeline. It requires running
    patched forward passes *and* extracting the hidden state at the
    last_prompt_token position, then applying a trained probe to that
    hidden state.

    When ``probe_propagation_df`` is None a placeholder figure is emitted.
    """
    if probe_propagation_df is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_axis_off()
        ax.text(
            0.5, 0.55,
            "Figure 6 — Cross-Span Probe Propagation",
            ha="center", va="center", fontsize=14, fontweight="bold",
            transform=ax.transAxes,
        )
        msg = (
            "This figure requires additional instrumentation not yet run:\n\n"
            "  1. Re-run patched forward passes with output_hidden_states=True\n"
            "     to capture hidden states at the last_prompt_token position.\n\n"
            "  2. Apply a trained probe (e.g., probe for lm_fullspan_lasttok)\n"
            "     to those hidden states to obtain probe_confidence scores.\n\n"
            "  3. Pass the resulting DataFrame as probe_propagation_df to this function.\n\n"
            "Expected schema:\n"
            "  patching_location | severity | delta_probe_score | sample_id"
        )
        ax.text(
            0.5, 0.45, msg,
            ha="center", va="top", fontsize=9.5, color="#4b5563",
            transform=ax.transAxes,
            family="monospace",
        )
        fig.tight_layout()
        _save(fig, output_dir, "figure_6_probe_propagation_placeholder")
        return

    # -- Actual implementation when probe data is available --
    available_sevs = [s for s in SEVERITY_ORDER
                      if s in probe_propagation_df["severity"].values]
    locations = sorted(probe_propagation_df["patching_location"].unique())
    n_locs = len(locations)
    n_sevs = len(available_sevs)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(max(7, n_locs * 2), 4.5))
    x = np.arange(n_locs)

    for si, sev in enumerate(available_sevs):
        color = SEVERITY_COLORS.get(sev, "#6b7280")
        means, lows, highs = [], [], []
        for loc in locations:
            vals = probe_propagation_df[
                (probe_propagation_df["patching_location"] == loc) &
                (probe_propagation_df["severity"] == sev)
            ]["delta_probe_score"].values
            m, lo, hi = _bootstrap_ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        means = np.array(means, dtype=float)
        offset = (si - n_sevs / 2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, color=color, alpha=0.8, label=sev.capitalize())
        ax.errorbar(
            x + offset, means,
            yerr=[means - np.array(lows), np.array(highs) - means],
            fmt="none", color="black", capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SPAN_LABELS.get(l, l).replace("\n", " ") for l in locations],
        rotation=20, ha="right",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Patching location")
    ax.set_ylabel("Δ Probe confidence at last_prompt_token ↑")
    ax.set_title(
        "Figure 6 — Cross-Span Probe Propagation\n"
        "Does restoring upstream activations propagate to the downstream uncertainty summary?",
        fontsize=11,
    )
    ax.legend(title="Severity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "figure_6_probe_propagation")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_parquet: str,
    output_dir: str,
    *,
    n_bootstrap: int = 1000,
    severity_for_fig4: str = "moderate",
    probe_peak_layer: Optional[int] = None,
    probe_propagation_df: Optional[pd.DataFrame] = None,
) -> None:
    """Generate all six figures from a patching results parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {results_parquet}")
    df = load_data(results_parquet)
    print(f"  {len(df)} rows, {df['sample_id'].nunique()} samples, "
          f"conditions: {sorted(df['condition'].unique())}")

    print("\n[1/6] Patching Effect Matrix ...")
    figure_1_heatmap(df, out)

    print("[2/6] Dose-Response Curves ...")
    figure_2_dose_response(df, out)

    print("[3/6] Causal Specificity ...")
    figure_3_causal_specificity(df, out, n_bootstrap=n_bootstrap)

    print(f"[4/6] Accuracy-Confidence Dissociation (severity={severity_for_fig4}) ...")
    figure_4_dissociation(df, out, severity=severity_for_fig4)

    print("[5/6] Layer Sweep ...")
    figure_5_layer_sweep(df, out, probe_peak_layer=probe_peak_layer)

    print("[6/6] Cross-Span Probe Propagation ...")
    figure_6_probe_propagation(df, out, probe_propagation_df=probe_propagation_df)

    print(f"\nAll figures saved to {out}/")
