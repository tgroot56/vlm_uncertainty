from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# CONFIG
# -----------------------
NO_SKILL_BRIER = {
    "imagenet-r": 0.18988,
    "vqa-v2": 0.153334,
}

PATHS = {
    ("imagenet-r", "true"): Path("probe_results/imagenet-r/16mc-clip"),
    ("imagenet-r", "shuffle"): Path("probe_results/imagenet-r/16mc-clip/baselines"),
    ("vqa-v2", "true"): Path("probe_results/vqa-v2"),
    ("vqa-v2", "shuffle"): Path("probe_results/vqa-v2/baselines"),
}

OUTDIR = Path("probe_results/_plots/brier_grouped_clean")

# New desired group order
GROUP_ORDER = ["prob_stats", "lm_answer", "lm_question", "lm_visual", "lm_full", "vision"]
GROUP_DISPLAY = {
    "prob_stats": "Prob stats",
    "lm_answer": "LM answer",
    "lm_question": "LM question",
    "lm_visual": "LM visual",
    "lm_full": "LM full",
    "vision": "Vision",
}

# Desired within-group ordering for LM_* families (and vision)
LM_SUFFIX_ORDER = ["mean_middle", "mean_final", "lasttok_middle", "lasttok_final"]


# -----------------------
# I/O
# -----------------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    if "feature_label" not in df.columns:
        return None
    for c in ["linear", "mlp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_brier(dataset: str, regime: str) -> Optional[pd.DataFrame]:
    return safe_read_csv(PATHS[(dataset, regime)] / "results_brier.csv")


# -----------------------
# Label parsing / grouping
# -----------------------
def feature_group(feature_label: str) -> str:
    s = feature_label.lower()

    # Prob stats (answer generation probability/entropy stats)
    if s.startswith("answer_gen_"):
        return "prob_stats"

    # LM families
    if s.startswith("lm_answer_"):
        return "lm_answer"
    if s.startswith("lm_question_including_") or s.startswith("lm_question_"):
        return "lm_question"
    if s.startswith("lm_visual_"):
        return "lm_visual"

    # LM full span family
    if s.startswith("lm_fullspan_"):
        return "lm_full"

    # Vision
    if s.startswith("vision_"):
        return "vision"

    return "vision" if s.startswith("vision") else "prob_stats" if s.startswith("answer_gen") else "vision" if s.startswith("image") else "other"


# Accept all the new prefixes that follow _{mean|lasttok}_layer_{k}
_LM_RE = re.compile(
    r"^(lm_answer|lm_question|lm_question_including|lm_visual|lm_fullspan|vision)_(mean|lasttok)_layer_(-?\d+)$",
    re.IGNORECASE,
)


def normalize_feature_name(feature_label: str) -> Tuple[str, str, str]:
    """
    Returns:
      display_name: x-axis label (without group prefix)
      group: one of GROUP_ORDER
      order_key: sorting key within group

    Rules:
      - answer_gen_*: display as the suffix after answer_gen_
      - lm_* / vision_*:
          mean_layer_-1     -> mean_final
          mean_layer_<k>    -> mean_middle
          lasttok_layer_-1  -> lasttok_final
          lasttok_layer_<k> -> lasttok_middle
        display name becomes that suffix (so LM groups share same x tick set),
        group is inferred from the prefix (with mapping for lm_fullspan + lm_question_including).
    """
    g = feature_group(feature_label)
    s = feature_label

    if g == "prob_stats":
        disp = s.replace("answer_gen_", "")
        return disp, g, disp  # order prob_stats alphabetically by name

    m = _LM_RE.match(s)
    if m:
        prefix, kind, layer_str = m.group(1), m.group(2), m.group(3)
        layer = int(layer_str)

        # Map some prefixes to your desired groups
        prefix_l = prefix.lower()
        if prefix_l == "lm_fullspan":
            g = "lm_full"
        elif prefix_l in {"lm_question_including", "lm_question"}:
            g = "lm_question"
        elif prefix_l == "lm_answer":
            g = "lm_answer"
        elif prefix_l == "lm_visual":
            g = "lm_visual"
        elif prefix_l == "vision":
            g = "vision"

        if kind.lower() == "mean":
            suffix = "mean_final" if layer == -1 else "mean_middle"
        else:
            suffix = "lasttok_final" if layer == -1 else "lasttok_middle"

        disp = suffix
        order_key = suffix
        return disp, g, order_key

    # fallback: strip known prefixes if present
    disp = s
    for pref in [
        "lm_answer_",
        "lm_question_including_",
        "lm_question_",
        "lm_visual_",
        "lm_fullspan_",
        "vision_",
        "answer_gen_",
    ]:
        disp = disp.replace(pref, "")
    return disp, g, disp


def sort_within_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by:
      group (GROUP_ORDER),
      then for lm_* and vision: LM_SUFFIX_ORDER,
      else alphabetical by display_name.
    """
    df = df.copy()

    df["group_rank"] = (
        df["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
        .fillna(len(GROUP_ORDER))
        .astype(int)
    )

    def within_rank(row):
        if row["group"] in {"lm_answer", "lm_question", "lm_visual", "lm_full", "vision"}:
            try:
                return LM_SUFFIX_ORDER.index(row["order_key"])
            except ValueError:
                return 999
        return 0

    df["within_rank"] = df.apply(within_rank, axis=1)
    df["alpha_rank"] = df["display_name"].astype(str)

    return (
        df.sort_values(["group_rank", "within_rank", "alpha_rank"], ascending=[True, True, True])
        .reset_index(drop=True)
    )


def compute_group_spans(groups: List[str]) -> Tuple[List[float], List[Tuple[int, int, str]]]:
    spans = []
    seps = []
    if not groups:
        return seps, spans

    start = 0
    current = groups[0]
    for i, g in enumerate(groups):
        if g != current:
            spans.append((start, i - 1, current))
            seps.append(i - 0.5)
            start = i
            current = g
    spans.append((start, len(groups) - 1, current))
    return seps, spans


def annotate_groups(ax, spans, y_top, fontsize=9):
    for (s, e, g) in spans:
        mid = (s + e) / 2
        ax.text(mid, y_top, GROUP_DISPLAY.get(g, g), ha="center", va="bottom", fontsize=fontsize)


# -----------------------
# Plotting
# -----------------------
def compute_global_ylim_brier() -> Tuple[float, float]:
    vals = []
    for ds in ["imagenet-r", "vqa-v2"]:
        for regime in ["true", "shuffle"]:
            df = load_brier(ds, regime)
            if df is None:
                continue
            for p in ["linear", "mlp"]:
                if p in df.columns:
                    vals.extend(df[p].dropna().tolist())
        vals.append(NO_SKILL_BRIER[ds])
    if not vals:
        return (0.0, 0.5)
    lo, hi = min(vals), max(vals)
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    return max(0.0, lo - pad), hi + pad


def plot_dataset_probe(dataset: str, probe_type: str, ylim: Tuple[float, float]):
    df_true = load_brier(dataset, "true")
    df_shuffle = load_brier(dataset, "shuffle")
    if df_true is None or df_shuffle is None:
        print(f"[skip] missing CSVs for {dataset}")
        return
    if probe_type not in df_true.columns or probe_type not in df_shuffle.columns:
        print(f"[skip] missing probe column {probe_type} for {dataset}")
        return

    dft = df_true[["feature_label", probe_type]].rename(columns={probe_type: "true"}).copy()
    dfs = df_shuffle[["feature_label", probe_type]].rename(columns={probe_type: "shuffle"}).copy()
    df = dft.merge(dfs, on="feature_label", how="inner")

    normalized = df["feature_label"].apply(normalize_feature_name)
    df["display_name"] = [t[0] for t in normalized]
    df["group"] = [t[1] for t in normalized]
    df["order_key"] = [t[2] for t in normalized]

    df = sort_within_group(df)

    x = np.arange(len(df))
    true_vals = df["true"].to_numpy()
    shuffle_vals = df["shuffle"].to_numpy()
    extra = shuffle_vals - true_vals

    seps, spans = compute_group_spans(df["group"].tolist())

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.axhspan(0, NO_SKILL_BRIER[dataset], alpha=0.06)
    ax.bar(x, true_vals, label="Probe (true labels)")
    ax.bar(x, extra, bottom=true_vals, label="Shuffled-label control", hatch="//", alpha=0.6)

    ax.axhline(NO_SKILL_BRIER[dataset], linestyle="--", linewidth=2, label="No-skill baseline")

    for sep in seps:
        ax.axvline(sep, linewidth=1, alpha=0.25)

    ax.set_ylabel("Brier score", fontsize=18)
    ax.set_ylim(ylim)
    ax.grid(True, axis="y", alpha=0.25)

    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"].tolist(), rotation=90, fontsize=18)
    ax.tick_params(axis="y", labelsize=12)

    y_top = ax.get_ylim()[1]
    annotate_groups(ax, spans, y_top=y_top * 1.00, fontsize=18)

    ax.legend(loc="upper right", fontsize=14)
    plt.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTDIR / f"{dataset}_{probe_type}.pdf"
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


def main():
    ylim = compute_global_ylim_brier()
    for dataset in ["imagenet-r", "vqa-v2"]:
        for probe_type in ["linear", "mlp"]:
            plot_dataset_probe(dataset, probe_type, ylim)


if __name__ == "__main__":
    main()