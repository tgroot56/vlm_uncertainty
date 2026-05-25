"""Qualitative figures for LLaVA coco-qa-vi extreme-blur patching experiment.

Produces two figures:
  (1) correct_under_extreme_blur: clean + degraded image + Q/A for samples
      where the model was *still correct* under σ=50 blur.
  (2) top1_score_decoupling: per-sample final-prompt-token patching sweep
      alongside clean + degraded images, highlighting samples where top-1
      probability peaks at the final layer while accuracy already drops.

Run:
    python -m src.cli.plot_qualitative_extreme
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_PARQUET = "/projects/prjs2014/patching/layer_sweep_results/llava-hf_llava-1-5-7b-hf/coco-qa-vi/extreme/layer_sweep_results.parquet"
DEFAULT_DATASET = "/projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/coco-qa-vi/patching_dataset_coco-qa-vi.json"
DEFAULT_OUTPUT  = "/projects/prjs2014/patching/layer_sweep_results/llava-hf_llava-1-5-7b-hf/coco-qa-vi/extreme/qualitative"


def _load_images(sample: dict) -> tuple[Image.Image, Image.Image]:
    clean = Image.open(sample["clean_image_path"]).convert("RGB")
    extreme = next(vp for vp in sample["visual_perturbations"] if vp["severity"] == "extreme")
    degraded = Image.open(extreme["blurred_image_path"]).convert("RGB")
    return clean, degraded


def _sample_index(dataset_json: str) -> dict:
    with open(dataset_json) as f:
        data = json.load(f)
    return {int(s["sample_id"]): s for s in data["samples"]}


def plot_correct_under_blur(parquet: str, dataset_json: str, out_dir: Path, n: int = 8) -> None:
    df = pd.read_parquet(parquet)
    deg = df[df["condition"] == "degraded"]
    hits = deg[deg["score"] >= 1.0].sort_values("top1_probability", ascending=False)

    samples = _sample_index(dataset_json)
    rng = np.random.default_rng(0)
    # Pick a diverse set: top-N confident correct-under-blur samples, skipping duplicates of predictions
    seen_preds: set[str] = set()
    chosen: list[tuple[int, dict, float]] = []  # (sample_id, sample_dict, top1_prob)
    for _, row in hits.iterrows():
        sid = int(row["sample_id"])
        pred = str(row["predicted_answer"]).strip().lower()
        if sid not in samples or pred in seen_preds:
            continue
        chosen.append((sid, samples[sid], float(row["top1_probability"])))
        seen_preds.add(pred)
        if len(chosen) >= n:
            break

    # Grid: n rows, 2 cols (clean | degraded), text under each pair
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.2 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (sid, s, top1) in enumerate(chosen):
        clean, deg_img = _load_images(s)
        ax_c, ax_d = axes[i, 0], axes[i, 1]
        ax_c.imshow(clean); ax_c.axis("off")
        ax_d.imshow(deg_img); ax_d.axis("off")
        extreme_vp = next(vp for vp in s["visual_perturbations"] if vp["severity"] == "extreme")
        model_pred = extreme_vp["model_prediction"]
        gt = s["correct_answer"]
        q = s["clean_question"]
        ax_c.set_title(f"Clean  (sample {sid})", fontsize=10)
        ax_d.set_title(f"Extreme blur σ=50  ·  top-1={top1:.2f}", fontsize=10)
        caption = (
            f"Q: {q}\n"
            f"Model (blurred): “{model_pred}”  ·  GT: “{gt}”"
        )
        # Put a caption below the row, centered across the two axes
        fig.text(
            0.5,
            ax_c.get_position().y0 - 0.012,
            caption,
            ha="center", va="top", fontsize=9, wrap=True,
        )

    fig.suptitle(
        "LLaVA-1.5-7B on COCO-QA-VI — correct answers under extreme (σ=50) blur\n"
        f"(degraded baseline accuracy ≈ {deg['score'].mean():.2f}; these {n}/{len(hits)} are examples where the "
        "model was right despite near-uniform visual input)",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"correct_under_extreme_blur.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  Saved correct_under_extreme_blur.pdf/.png  ({len(chosen)} samples)")


def plot_decoupling(parquet: str, dataset_json: str, out_dir: Path, n: int = 4) -> None:
    df = pd.read_parquet(parquet)
    pat = df[(df["condition"] == "patched") & (df["position_offset"] == -1)].copy()
    clean = df[df["condition"] == "clean"][["sample_id", "score", "top1_probability"]].rename(
        columns={"score": "clean_score", "top1_probability": "clean_top1"}
    )
    deg = df[df["condition"] == "degraded"][["sample_id", "score", "top1_probability", "predicted_answer"]].rename(
        columns={"score": "deg_score", "top1_probability": "deg_top1", "predicted_answer": "deg_pred"}
    )
    pat = pat.merge(clean, on="sample_id").merge(deg, on="sample_id")

    deco = pat[(pat["clean_score"] >= 1.0) & (pat["deg_score"] < 1.0)]

    pivot_score = deco.pivot_table(index="sample_id", columns="layer", values="score")
    pivot_top1  = deco.pivot_table(index="sample_id", columns="layer", values="top1_probability")
    pivot_pred  = deco.pivot_table(index="sample_id", columns="layer", values="predicted_answer", aggfunc="first")

    final_layer = int(pivot_score.columns.max())

    # Candidates: score peaks at some mid-late layer (>=20) with value 1.0; score at final layer is < 1.0;
    # top1 at final layer > top1 at that mid layer.
    rows = []
    mid_range = list(range(20, final_layer))
    for sid in pivot_score.index:
        srow = pivot_score.loc[sid]
        trow = pivot_top1.loc[sid]
        if not np.any(np.isfinite(srow.values)):
            continue
        # Find the earliest mid-layer with score == 1.0
        hit_layer = None
        for L in mid_range:
            if L in srow.index and srow[L] >= 1.0:
                hit_layer = L
                break
        if hit_layer is None:
            continue
        if srow.get(final_layer, 1.0) >= 1.0:
            continue
        if trow.get(final_layer, 0.0) <= trow.get(hit_layer, 0.0):
            continue
        rows.append({
            "sample_id": sid,
            "hit_layer": hit_layer,
            "top1_gain": trow[final_layer] - trow[hit_layer],
            "pred_mid": pivot_pred.loc[sid, hit_layer],
            "pred_final": pivot_pred.loc[sid, final_layer],
        })
    cand = pd.DataFrame(rows).sort_values("top1_gain", ascending=False)

    samples = _sample_index(dataset_json)
    # Pick top-N by top1_gain, preferring diverse predictions
    chosen: list[tuple[int, int]] = []   # (sample_id, hit_layer)
    seen_preds = set()
    for _, r in cand.iterrows():
        sid = int(r["sample_id"])
        if sid not in samples:
            continue
        key = (str(r["pred_mid"]).lower(), str(r["pred_final"]).lower())
        if key in seen_preds:
            continue
        chosen.append((sid, int(r["hit_layer"])))
        seen_preds.add(key)
        if len(chosen) >= n:
            break

    # Layout: one row per sample, 3 cols: clean image | degraded image | line plot (score + top1 per layer)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.6 * n), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    if n == 1:
        axes = np.array([axes])
    for i, (sid, hit_layer) in enumerate(chosen):
        s = samples[sid]
        clean_img, deg_img = _load_images(s)
        ax_c, ax_d, ax_p = axes[i, 0], axes[i, 1], axes[i, 2]
        ax_c.imshow(clean_img); ax_c.axis("off"); ax_c.set_title(f"Clean  (sample {sid})", fontsize=10)
        ax_d.imshow(deg_img);   ax_d.axis("off"); ax_d.set_title("Extreme blur σ=50",  fontsize=10)

        sub = deco[deco["sample_id"] == sid].sort_values("layer")
        layers      = sub["layer"].astype(int).values
        score_vals  = sub["score"].astype(float).values
        top1_vals   = sub["top1_probability"].astype(float).values
        pred_by_layer = dict(zip(layers, sub["predicted_answer"].values))

        # Two y-axes on the same plot
        ax_p.plot(layers, score_vals, "o-", color="#dc2626", linewidth=2, markersize=4, label="Score (accuracy)")
        ax_p.set_ylabel("Score (accuracy)", color="#dc2626", fontsize=9)
        ax_p.tick_params(axis="y", labelcolor="#dc2626")
        ax_p.set_ylim(-0.05, 1.10)
        ax_p.set_xlabel("LM layer (patched @ offset −1)", fontsize=9)
        ax_p.grid(alpha=0.3)

        ax_t = ax_p.twinx()
        ax_t.plot(layers, top1_vals, "s-", color="#2563eb", linewidth=2, markersize=3, label="Top-1 probability")
        ax_t.set_ylabel("Top-1 probability", color="#2563eb", fontsize=9)
        ax_t.tick_params(axis="y", labelcolor="#2563eb")
        ax_t.set_ylim(0, 1.05)

        # Mark the mid-hit layer and final layer
        ax_p.axvline(hit_layer, color="#dc2626", linestyle="--", alpha=0.4)
        ax_p.axvline(final_layer, color="#2563eb", linestyle="--", alpha=0.4)
        ax_p.text(hit_layer, 1.06, f"L{hit_layer}", ha="center", fontsize=8, color="#dc2626")
        ax_p.text(final_layer, 1.06, f"L{final_layer}", ha="center", fontsize=8, color="#2563eb")

        q = s["clean_question"]
        gt = s["correct_answer"]
        pred_mid = pred_by_layer.get(hit_layer, "?")
        pred_final = pred_by_layer.get(final_layer, "?")
        extreme_vp = next(vp for vp in s["visual_perturbations"] if vp["severity"] == "extreme")
        pred_blur = extreme_vp["model_prediction"]
        caption = (
            f"Q: {q}   ·   GT: “{gt}”\n"
            f"Blurred: “{pred_blur}”   ·   Patched @ L{hit_layer}: “{pred_mid}” (✓)   ·   "
            f"Patched @ L{final_layer}: “{pred_final}” (✗)"
        )
        fig.text(
            0.5,
            ax_c.get_position().y0 - 0.010,
            caption,
            ha="center", va="top", fontsize=9,
        )

    fig.suptitle(
        "Decoupling of top-1 probability and accuracy under patching (coco-qa-vi, σ=50)\n"
        "Red = accuracy of patched prediction vs. GT   ·   Blue = top-1 probability of patched prediction\n"
        "At a mid-late layer the patched prediction matches GT; at the final layer it becomes more "
        "confident but shifts to a different surface form.",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"top1_score_decoupling.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  Saved top1_score_decoupling.pdf/.png  ({len(chosen)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n_blur", type=int, default=8)
    parser.add_argument("--n_decouple", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Output dir: {out}")
    plot_correct_under_blur(args.parquet, args.dataset, out, n=args.n_blur)
    plot_decoupling(args.parquet, args.dataset, out, n=args.n_decouple)


if __name__ == "__main__":
    main()
