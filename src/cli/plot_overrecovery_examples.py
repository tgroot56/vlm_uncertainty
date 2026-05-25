"""Qualitative examples where patched top-1 probability at layer 31 exceeds the clean baseline.

These are the 'commitment overrecovery' cases: patching the final prompt token at the
commitment layer restores — and even surpasses — the clean confidence level.

Usage:
    python -m src.cli.plot_overrecovery_examples \
        --dataset vqa-v2 \
        --severity moderate \
        --n 12 \
        --layer 31 \
        --output_dir /projects/prjs2014/patching/filtered_plots/llava-hf_llava-1-5-7b-hf/qualitative
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_SLUG   = "llava-hf_llava-1-5-7b-hf"
PATCHING_ROOT = Path("/projects/prjs2014/patching")
LAYER_SWEEP_ROOT = PATCHING_ROOT / "layer_sweep_results" / MODEL_SLUG
META_ROOT        = PATCHING_ROOT / MODEL_SLUG
OUTPUT_ROOT      = PATCHING_ROOT / "filtered_plots" / MODEL_SLUG / "qualitative"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset: str, severity: str, layer: int):
    parquet = LAYER_SWEEP_ROOT / dataset / "layer_sweep_results.parquet"
    meta_file = META_ROOT / dataset / f"patching_dataset_{dataset}.json"

    df = pd.read_parquet(parquet)

    # Patched rows at the target layer and final prompt token
    patched = df[
        (df["condition"] == "patched") &
        (df["layer"] == layer) &
        (df["position_offset"] == -1) &
        (df["severity"] == severity)
    ][["sample_id", "severity", "predicted_answer", "top1_probability", "score"]].copy()
    patched = patched.rename(columns={
        "predicted_answer": "patched_answer",
        "top1_probability": "patched_top1",
        "score": "patched_score",
    })

    # Clean rows
    clean = df[
        (df["condition"] == "clean") &
        (df["severity"] == severity)
    ][["sample_id", "severity", "predicted_answer", "top1_probability", "score"]].copy()
    clean = clean.rename(columns={
        "predicted_answer": "clean_answer",
        "top1_probability": "clean_top1",
        "score": "clean_score",
    })

    # Degraded rows
    degraded = df[
        (df["condition"] == "degraded") &
        (df["severity"] == severity)
    ][["sample_id", "severity", "predicted_answer", "top1_probability", "score"]].copy()
    degraded = degraded.rename(columns={
        "predicted_answer": "degraded_answer",
        "top1_probability": "degraded_top1",
        "score": "degraded_score",
    })

    merged = patched.merge(clean, on=["sample_id", "severity"])
    merged = merged.merge(degraded, on=["sample_id", "severity"])

    # Overrecovery: patched top-1 strictly exceeds clean top-1
    merged = merged[merged["patched_top1"] > merged["clean_top1"]].copy()
    merged["delta"] = merged["patched_top1"] - merged["clean_top1"]
    merged = merged.sort_values("delta", ascending=False).reset_index(drop=True)

    # Load JSON metadata (question, image paths, gt answers)
    with open(meta_file) as f:
        raw = json.load(f)
    samples_list = raw["samples"]
    meta = {str(s["sample_id"]): s for s in samples_list}

    return merged, meta, severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gt_answer(meta_entry: dict) -> str:
    gt = meta_entry.get("correct_answer") or ""
    if not gt:
        answers = meta_entry.get("gt_answers", [])
        gt = answers[0] if answers else ""
    return str(gt)


def load_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def score_label(score: float) -> str:
    return "✓" if score >= 0.5 else "✗"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_examples(
    merged: pd.DataFrame,
    meta: dict,
    dataset: str,
    severity: str,
    layer: int,
    n: int,
    output_dir: Path,
    correct_only: bool = False,
    incorrect_only: bool = False,
):
    subset = merged.copy()
    if correct_only:
        subset = subset[subset["patched_score"] >= 0.5]
    if incorrect_only:
        subset = subset[subset["patched_score"] < 0.5]
    subset = subset.head(n)

    if subset.empty:
        print("  No examples matched the filter.")
        return

    n_rows = len(subset)
    fig = plt.figure(figsize=(18, 4.2 * n_rows))
    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55)

    for row_idx, (_, row) in enumerate(subset.iterrows()):
        sid = str(row["sample_id"])
        m = meta.get(sid, {})

        clean_img_path    = m.get("clean_image_path", "")
        pert = next(
            (p for p in m.get("visual_perturbations", []) if p["severity"] == severity),
            {}
        )
        deg_img_path = pert.get("blurred_image_path", "")

        clean_img   = load_image(clean_img_path)
        deg_img     = load_image(deg_img_path)
        question    = m.get("clean_question", "—")
        gt_answer   = get_gt_answer(m)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[row_idx],
            width_ratios=[1, 1, 1.8], wspace=0.12
        )

        # --- Clean image ---
        ax_clean = fig.add_subplot(inner[0])
        if clean_img:
            ax_clean.imshow(clean_img)
        else:
            ax_clean.text(0.5, 0.5, "image\nnot found", ha="center", va="center",
                          transform=ax_clean.transAxes, fontsize=9, color="gray")
        ax_clean.set_title("Clean", fontsize=9, color="#1e3a5f", fontweight="bold")
        ax_clean.axis("off")

        # --- Degraded image ---
        ax_deg = fig.add_subplot(inner[1])
        if deg_img:
            ax_deg.imshow(deg_img)
        else:
            ax_deg.text(0.5, 0.5, "image\nnot found", ha="center", va="center",
                        transform=ax_deg.transAxes, fontsize=9, color="gray")
        ax_deg.set_title(f"Degraded ({severity})", fontsize=9, color="#7c1d1d", fontweight="bold")
        ax_deg.axis("off")

        # --- Text panel ---
        ax_txt = fig.add_subplot(inner[2])
        ax_txt.axis("off")

        # (label, text, color, fontsize, fontweight, fontstyle)
        lines = [
            ("Q:",    question,  "#1e293b", 10, "bold",   "normal"),
            ("GT:",   gt_answer, "#374151",  9, "normal", "normal"),
            ("",      "",        "#374151",  5, "normal", "normal"),
        ]

        def add_row(label, answer, prob, sc, color):
            lines.append((
                label,
                f"{answer}   (p = {prob:.3f})   {score_label(sc)}",
                color, 9, "normal", "normal",
            ))

        add_row("Clean:",    row["clean_answer"],    row["clean_top1"],    row["clean_score"],    "#1e3a5f")
        add_row("Degraded:", row["degraded_answer"], row["degraded_top1"], row["degraded_score"], "#7c1d1d")
        add_row("Patched:",  row["patched_answer"],  row["patched_top1"],  row["patched_score"],  "#065f46")
        lines.append(("",       "",                                    "#374151",  5, "normal", "normal"))
        lines.append(("Δ top-1:", f"+{row['delta']:.3f}  (patched − clean)", "#7c3aed", 8.5, "normal", "italic"))

        y = 0.96
        for label, text, color, fsize, fweight, fstyle in lines:
            if label:
                ax_txt.text(0.0, y, label, transform=ax_txt.transAxes,
                            fontsize=fsize, fontweight="bold", color="#374151",
                            va="top", ha="left")
                ax_txt.text(0.22, y, text, transform=ax_txt.transAxes,
                            fontsize=fsize, fontweight=fweight, fontstyle=fstyle,
                            color=color, va="top", ha="left")
            dy = 0.12 if fsize >= 10 else (0.04 if fsize <= 5 else 0.09)
            y -= dy

    fig.suptitle(
        f"Overrecovery examples — Layer {layer}, position offset −1  ·  "
        f"Dataset: {dataset}  ·  Severity: {severity}  ·  "
        f"patched top-1 > clean top-1",
        fontsize=12, y=1.005,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "correct" if correct_only else ("incorrect" if incorrect_only else "all")
    name = f"overrecovery_layer{layer}_{dataset}_{severity}_{tag}"
    for ext, kw in [("pdf", {}), ("png", {"dpi": 150})]:
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  Saved {output_dir}/{name}.pdf/.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="vqa-v2")
    parser.add_argument("--severity",   default="moderate")
    parser.add_argument("--layer",      type=int, default=31)
    parser.add_argument("--n",          type=int, default=12,
                        help="Number of examples to show")
    parser.add_argument("--output_dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--correct_only",   action="store_true")
    parser.add_argument("--incorrect_only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Loading data for {args.dataset} / {args.severity} / layer {args.layer}...")
    merged, meta, severity = load_data(args.dataset, args.severity, args.layer)
    print(f"  {len(merged)} overrecovery samples found.")

    plot_examples(merged, meta, args.dataset, severity, args.layer,
                  args.n, out, args.correct_only, args.incorrect_only)


if __name__ == "__main__":
    main()
