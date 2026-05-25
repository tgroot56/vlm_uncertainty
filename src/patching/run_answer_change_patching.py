"""Answer-change analysis for activation patching.

For each sample, patches the final prompt token activation from the clean run
into the degraded run at one or more specified layers, then classifies the
outcome into transition categories (wrong->same wrong, wrong->different wrong,
wrong->correct, correct->same correct, correct->wrong).

The critical question: does patching increase confidence *within* the
wrong->same wrong subset (pure confidence modulation), or does confidence
rise only when the answer changes (answer reranking)?

Outputs
-------
- ``answer_change_results.parquet`` — per-sample wide-format rows
- ``answer_change_summary.csv``     — aggregated stats per (severity, layer, transition)
- ``answer_change_layer_summary.csv`` — per-layer overview stats
- Plots: transition frequencies, confidence deltas by category, layer sweep curves
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.patching.collect_activations import (
    _format_prompt,
    _MAX_NEW_TOKENS,
    _score_prediction,
)
from src.patching.run_patching import (
    _get_decoder_layers,
    _patched_generate,
    _run_baseline_forward,
)
from src.patching.run_layer_sweep_patching import (
    _extract_all_layer_activations,
    _get_final_prompt_positions,
    _get_layer_index_pairs,
)
from src.vlm import VLMAdapter, prepare_qwen_fused_inputs_embeds


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORRECTNESS_THRESHOLD = 0.5


@dataclass
class AnswerChangeConfig:
    """Configuration for the answer-change analysis experiment."""
    dataset_id: str = "vqa-v2"
    layers: List[int] = field(default_factory=lambda: [31])
    severity: str = "moderate"
    final_prompt_offset: int = -1   # -1 = last prompt token, -2 = second-to-last
    max_new_tokens: int = 4
    checkpoint_every: int = 25
    verbose: bool = False


# ---------------------------------------------------------------------------
# Extended output statistics
# ---------------------------------------------------------------------------

def _compute_extended_output_stats(gen_step_logits: torch.Tensor) -> Dict[str, Any]:
    """Compute output statistics from generation logits.

    Returns top-1 probability, entropy, logit margin, sequence log-probability,
    and the first generated token id.
    """
    if gen_step_logits.shape[1] == 0:
        return {
            "top1_probability": 0.0,
            "output_entropy": 0.0,
            "margin": 0.0,
            "sequence_log_probability": 0.0,
            "first_generated_token_id": -1,
        }

    # First-token statistics
    first_logits = gen_step_logits[0, 0, :].float()
    probs = F.softmax(first_logits, dim=0)
    top2_vals, _ = first_logits.topk(2)

    top1_prob = float(probs.max().item())
    entropy = float(-(probs * torch.log(probs + 1e-10)).sum().item())
    margin = float((top2_vals[0] - top2_vals[1]).item())
    first_token_id = int(first_logits.argmax().item())

    # Sequence log-probability (greedy path)
    all_logits = gen_step_logits[0].float()          # (T, V)
    gen_ids = all_logits.argmax(dim=-1)               # (T,)
    log_probs = F.log_softmax(all_logits, dim=-1)     # (T, V)
    seq_log_prob = float(
        sum(log_probs[t, gen_ids[t]].item() for t in range(len(gen_ids)))
    )

    return {
        "top1_probability": top1_prob,
        "output_entropy": entropy,
        "margin": margin,
        "sequence_log_probability": seq_log_prob,
        "first_generated_token_id": first_token_id,
    }


def _prob_of_token(gen_step_logits: torch.Tensor, token_id: int) -> float:
    """Return the softmax probability of ``token_id`` at the first generation step."""
    if gen_step_logits.shape[1] == 0 or token_id < 0:
        return 0.0
    first_logits = gen_step_logits[0, 0, :].float()
    probs = F.softmax(first_logits, dim=0)
    if token_id >= probs.shape[0]:
        return 0.0
    return float(probs[token_id].item())


# ---------------------------------------------------------------------------
# Transition classification
# ---------------------------------------------------------------------------

def _classify_transition(
    deg_score: float,
    patched_score: float,
    deg_answer: str,
    patched_answer: str,
) -> str:
    """Classify the answer transition after patching.

    Categories:
        wrong_to_same_wrong   — degraded was wrong, patched answer identical, still wrong
        wrong_to_diff_wrong   — degraded was wrong, answer changed, still wrong
        wrong_to_correct      — degraded was wrong, patched becomes correct
        correct_to_same_correct — degraded was correct, patched identical, still correct
        correct_to_correct_changed — degraded was correct, answer changed, still correct
        correct_to_wrong      — degraded was correct, patched becomes wrong
    """
    deg_correct = deg_score >= CORRECTNESS_THRESHOLD
    patched_correct = patched_score >= CORRECTNESS_THRESHOLD
    answer_same = deg_answer.strip().lower() == patched_answer.strip().lower()

    if not deg_correct:
        if answer_same:
            return "wrong_to_same_wrong"
        elif patched_correct:
            return "wrong_to_correct"
        else:
            return "wrong_to_diff_wrong"
    else:
        if answer_same:
            return "correct_to_same_correct"
        elif patched_correct:
            return "correct_to_correct_changed"
        else:
            return "correct_to_wrong"


# ---------------------------------------------------------------------------
# Checkpoint / save helpers
# ---------------------------------------------------------------------------

def _save_results(
    rows: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_path: Optional[str],
    completed_ids: set,
) -> None:
    import pandas as pd

    parquet_path = os.path.join(output_dir, "answer_change_results.parquet")
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    if checkpoint_path is not None:
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"completed_ids": sorted(completed_ids)}, f)
        os.replace(tmp, checkpoint_path)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_answer_change_patching(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_records: List[Dict[str, Any]],
    output_dir: str,
    cfg: AnswerChangeConfig,
) -> None:
    """Execute answer-change analysis patching and save results.

    For each sample: clean baseline, degraded baseline, then for each target
    layer patch the final prompt token and classify the answer transition.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_new_tokens = _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)

    severities = (
        ["mild", "moderate", "severe"] if cfg.severity == "all"
        else [cfg.severity]
    )

    # Checkpoint
    checkpoint_path = os.path.join(output_dir, "answer_change_checkpoint.json")
    completed_ids: set = set()
    rows: List[Dict[str, Any]] = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_ids = set(ckpt.get("completed_ids", []))
        parquet_path = os.path.join(output_dir, "answer_change_results.parquet")
        if completed_ids and os.path.exists(parquet_path):
            import pandas as pd
            rows = pd.read_parquet(parquet_path).to_dict("records")
        print(f"[answer-change] Resuming: {len(completed_ids)} samples done.")

    total = len(patching_records)
    print(f"[answer-change] Starting answer-change patching")
    print(f"  Samples:       {total}")
    print(f"  Layers:        {cfg.layers}")
    print(f"  Severities:    {severities}")
    print(f"  Prompt offset: {cfg.final_prompt_offset}")
    print(f"  Output:        {output_dir}/")

    t0 = time.time()

    for record in tqdm(patching_records, desc="answer-change patching"):
        sample_id = str(record["sample_id"])
        if sample_id in completed_ids:
            continue

        prompt = _format_prompt(cfg.dataset_id, record["clean_question"])
        clean_image = Image.open(record["clean_image_path"]).convert("RGB")

        # ---- clean activations at all layers --------------------------------
        all_layer_h, token_spans = _extract_all_layer_activations(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
        )

        # Map hidden-state indices to patchable decoder-layer indices
        decoder_layers = _get_decoder_layers(model, vlm_adapter)
        layer_index_pairs = _get_layer_index_pairs(
            num_hidden_state_layers=all_layer_h.shape[0],
            num_decoder_layers=len(decoder_layers),
        )
        # Build lookup: decoder_layer_idx -> hidden_state_idx
        decoder_to_hidden = {
            dec_idx: hs_idx for hs_idx, dec_idx in layer_index_pairs
        }

        # Identify the final prompt token position
        n_offset = abs(cfg.final_prompt_offset)
        final_positions = _get_final_prompt_positions(token_spans, n_offset)
        if not final_positions:
            if cfg.verbose:
                print(f"  [skip] {sample_id}: no non-visual positions found")
            clean_image.close()
            continue
        # Select the one at the requested offset
        # final_positions is sorted ascending; offset -1 = last = index -1
        pos = final_positions[-1]

        # ---- Qwen fast path: cache fused inputs_embeds per image (vision
        # tower is ~30x slower than the rest of a patching call on Qwen).
        use_fused = vlm_adapter.family == "qwen"
        clean_fused = (
            prepare_qwen_fused_inputs_embeds(model, vlm_adapter, processor, clean_image, prompt, device)
            if use_fused else None
        )

        # ---- clean baseline -------------------------------------------------
        clean_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            max_new_tokens=max_new_tokens, fused_bundle=clean_fused,
        )
        clean_image.close()
        clean_stats = _compute_extended_output_stats(clean_fwd["gen_step_logits"])
        clean_score = _score_prediction(cfg.dataset_id, clean_fwd["raw_text"], record)
        clean_answer = clean_fwd["raw_text"]

        # ---- per-severity loop ----------------------------------------------
        for pert in record.get("visual_perturbations", []):
            severity = pert["severity"]
            if severity not in severities:
                continue

            img_path = pert.get("blurred_image_path", "")
            if not img_path or not os.path.exists(img_path):
                continue

            blurred_image = Image.open(img_path).convert("RGB")
            blurred_fused = (
                prepare_qwen_fused_inputs_embeds(model, vlm_adapter, processor, blurred_image, prompt, device)
                if use_fused else None
            )

            # Degraded baseline
            deg_fwd = _run_baseline_forward(
                model=model, processor=processor, device=device,
                vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
                max_new_tokens=max_new_tokens, fused_bundle=blurred_fused,
            )
            deg_stats = _compute_extended_output_stats(deg_fwd["gen_step_logits"])
            deg_score = _score_prediction(cfg.dataset_id, deg_fwd["raw_text"], record)
            deg_answer = deg_fwd["raw_text"]

            # ---- patch at each requested layer ------------------------------
            for layer_idx in cfg.layers:
                if layer_idx not in decoder_to_hidden:
                    if cfg.verbose:
                        print(
                            f"  [skip] layer {layer_idx} not in decoder range "
                            f"(max {max(decoder_to_hidden.keys())})"
                        )
                    continue

                hs_idx = decoder_to_hidden[layer_idx]
                clean_act = all_layer_h[hs_idx, pos, :].unsqueeze(0)  # (1, H)

                patched_fwd = _patched_generate(
                    model=model, processor=processor, device=device,
                    vlm_adapter=vlm_adapter, image=blurred_image,
                    prompt=prompt, max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    positions=[pos],
                    clean_activations=clean_act,
                    fused_bundle=blurred_fused,
                )
                patched_stats = _compute_extended_output_stats(
                    patched_fwd["gen_step_logits"]
                )
                patched_score = _score_prediction(
                    cfg.dataset_id, patched_fwd["raw_text"], record,
                )
                patched_answer = patched_fwd["raw_text"]

                # Probability of specific answer tokens after patching
                prob_deg_tok = _prob_of_token(
                    patched_fwd["gen_step_logits"],
                    deg_stats["first_generated_token_id"],
                )
                prob_clean_tok = _prob_of_token(
                    patched_fwd["gen_step_logits"],
                    clean_stats["first_generated_token_id"],
                )

                answer_changed = (
                    deg_answer.strip().lower() != patched_answer.strip().lower()
                )
                transition = _classify_transition(
                    deg_score, patched_score, deg_answer, patched_answer,
                )

                rows.append({
                    "sample_id": sample_id,
                    "severity": severity,
                    "layer": layer_idx,
                    # Clean baseline
                    "clean_answer": clean_answer,
                    "clean_score": clean_score,
                    "clean_top1": clean_stats["top1_probability"],
                    "clean_entropy": clean_stats["output_entropy"],
                    # Degraded baseline
                    "deg_answer": deg_answer,
                    "deg_score": deg_score,
                    "deg_top1": deg_stats["top1_probability"],
                    "deg_entropy": deg_stats["output_entropy"],
                    "deg_margin": deg_stats["margin"],
                    "deg_seq_log_prob": deg_stats["sequence_log_probability"],
                    # Patched
                    "patched_answer": patched_answer,
                    "patched_score": patched_score,
                    "patched_top1": patched_stats["top1_probability"],
                    "patched_entropy": patched_stats["output_entropy"],
                    "patched_margin": patched_stats["margin"],
                    "patched_seq_log_prob": patched_stats["sequence_log_probability"],
                    # Transition
                    "answer_changed": answer_changed,
                    "transition_category": transition,
                    # Deltas (patched - degraded)
                    "delta_top1": patched_stats["top1_probability"] - deg_stats["top1_probability"],
                    "delta_entropy": patched_stats["output_entropy"] - deg_stats["output_entropy"],
                    "delta_margin": patched_stats["margin"] - deg_stats["margin"],
                    "delta_seq_log_prob": patched_stats["sequence_log_probability"] - deg_stats["sequence_log_probability"],
                    # Cross-condition token probabilities
                    "prob_deg_answer_first_token": prob_deg_tok,
                    "prob_clean_answer_first_token": prob_clean_tok,
                })

                if cfg.verbose:
                    print(
                        f"  [{severity}/layer {layer_idx}] "
                        f"deg=\"{deg_answer}\" → patched=\"{patched_answer}\" "
                        f"({transition}) "
                        f"delta_top1={patched_stats['top1_probability'] - deg_stats['top1_probability']:+.4f}"
                    )

            blurred_image.close()

        completed_ids.add(sample_id)

        if len(completed_ids) % cfg.checkpoint_every == 0:
            _save_results(rows, output_dir, checkpoint_path, completed_ids)
            elapsed = time.time() - t0
            rate = len(completed_ids) / elapsed if elapsed > 0 else 0
            print(
                f"[answer-change] {len(completed_ids)}/{total} samples "
                f"({elapsed:.0f}s, {rate:.2f} samples/s)"
            )

    # Final save
    _save_results(rows, output_dir, checkpoint_path=None, completed_ids=completed_ids)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    print(f"\n[answer-change] Finished {len(completed_ids)}/{total} samples in {elapsed:.1f}s")

    _print_answer_change_summary(rows, output_dir)
    plot_answer_change_results(
        os.path.join(output_dir, "answer_change_results.parquet"),
        output_dir,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_answer_change_summary(
    rows: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Compute and print aggregated answer-change statistics."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print("[answer-change] No results to summarize.")
        return

    # --- Per (severity, layer, transition_category) aggregation ---
    cat_agg = df.groupby(["severity", "layer", "transition_category"]).agg(
        count=("sample_id", "count"),
        mean_delta_top1=("delta_top1", "mean"),
        mean_delta_entropy=("delta_entropy", "mean"),
        mean_delta_margin=("delta_margin", "mean"),
        mean_delta_seq_log_prob=("delta_seq_log_prob", "mean"),
        mean_prob_deg_tok=("prob_deg_answer_first_token", "mean"),
        mean_prob_clean_tok=("prob_clean_answer_first_token", "mean"),
    ).reset_index()

    # Add fraction within each (severity, layer) group
    totals = df.groupby(["severity", "layer"]).size().reset_index(name="total")
    cat_agg = cat_agg.merge(totals, on=["severity", "layer"])
    cat_agg["fraction"] = cat_agg["count"] / cat_agg["total"]

    summary_path = os.path.join(output_dir, "answer_change_summary.csv")
    cat_agg.to_csv(summary_path, index=False)
    print(f"\n[answer-change] Category summary saved to {summary_path}")

    # Print category breakdown
    print(f"\n{'Sev':<10s} {'Layer':>5s} {'Category':<28s} {'N':>5s} "
          f"{'Frac':>6s} {'dTop1':>7s} {'dEntropy':>9s} {'dMargin':>8s}")
    print("-" * 90)
    for _, row in cat_agg.sort_values(["severity", "layer", "transition_category"]).iterrows():
        print(
            f"{row['severity']:<10s} {int(row['layer']):>5d} "
            f"{row['transition_category']:<28s} {int(row['count']):>5d} "
            f"{row['fraction']:>6.3f} {row['mean_delta_top1']:>+7.4f} "
            f"{row['mean_delta_entropy']:>+9.4f} {row['mean_delta_margin']:>+8.3f}"
        )

    # --- Per-layer overview ---
    layer_agg_rows: List[Dict[str, Any]] = []
    for (sev, layer), grp in df.groupby(["severity", "layer"]):
        n_total = len(grp)
        answer_change_rate = grp["answer_changed"].mean()

        wrong_mask = grp["deg_score"] < CORRECTNESS_THRESHOLD
        wrong_grp = grp[wrong_mask]
        n_wrong = len(wrong_grp)

        w2c = grp[grp["transition_category"] == "wrong_to_correct"]
        w2sw = grp[grp["transition_category"] == "wrong_to_same_wrong"]

        layer_agg_rows.append({
            "severity": sev,
            "layer": layer,
            "N_total": n_total,
            "N_deg_wrong": n_wrong,
            "answer_change_rate": answer_change_rate,
            "wrong_to_correct_rate": len(w2c) / n_wrong if n_wrong > 0 else float("nan"),
            "wrong_to_correct_count": len(w2c),
            "wrong_to_same_wrong_count": len(w2sw),
            "wrong_to_same_wrong_mean_delta_top1": w2sw["delta_top1"].mean() if len(w2sw) > 0 else float("nan"),
            "wrong_to_same_wrong_mean_delta_entropy": w2sw["delta_entropy"].mean() if len(w2sw) > 0 else float("nan"),
            "wrong_to_same_wrong_mean_delta_margin": w2sw["delta_margin"].mean() if len(w2sw) > 0 else float("nan"),
        })

    layer_df = pd.DataFrame(layer_agg_rows)
    layer_path = os.path.join(output_dir, "answer_change_layer_summary.csv")
    layer_df.to_csv(layer_path, index=False)
    print(f"\n[answer-change] Layer summary saved to {layer_path}")

    # Print the key statistic
    print(f"\n{'Sev':<10s} {'Layer':>5s} {'N':>5s} {'AnswChg':>8s} {'W2C rate':>9s} "
          f"{'W2SW dTop1':>11s} {'W2SW dEnt':>10s}")
    print("-" * 70)
    for _, row in layer_df.iterrows():
        w2sw_dt = row["wrong_to_same_wrong_mean_delta_top1"]
        w2sw_de = row["wrong_to_same_wrong_mean_delta_entropy"]
        print(
            f"{row['severity']:<10s} {int(row['layer']):>5d} "
            f"{int(row['N_total']):>5d} {row['answer_change_rate']:>8.3f} "
            f"{row['wrong_to_correct_rate']:>9.3f} "
            f"{w2sw_dt:>+11.4f} {w2sw_de:>+10.4f}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_answer_change_results(
    results_parquet: str,
    output_dir: str,
) -> None:
    """Generate answer-change analysis plot.

    Single figure: x-axis = layer number, y-axis = percentage of all answers,
    one line per transition category showing how each evolves across layers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(results_parquet):
        print(f"[answer-change] No results at {results_parquet}, skipping plots.")
        return

    df = pd.read_parquet(results_parquet)
    if df.empty:
        print("[answer-change] Empty results, skipping plots.")
        return

    severities = sorted(df["severity"].unique())

    for severity in severities:
        _plot_transition_lines(df[df["severity"] == severity], output_dir, severity)


# -- Transition category line plot across layers ------------------------------

CATEGORY_ORDER = [
    "wrong_to_same_wrong",
    "wrong_to_diff_wrong",
    "wrong_to_correct",
    "correct_to_same_correct",
    "correct_to_correct_changed",
    "correct_to_wrong",
]

CATEGORY_LABELS = {
    "wrong_to_same_wrong":        "Wrong -> Same Wrong",
    "wrong_to_diff_wrong":        "Wrong -> Diff Wrong",
    "wrong_to_correct":           "Wrong -> Correct",
    "correct_to_same_correct":    "Correct -> Same Correct",
    "correct_to_correct_changed": "Correct -> Changed Correct",
    "correct_to_wrong":           "Correct -> Wrong",
}

CATEGORY_COLORS = {
    "wrong_to_same_wrong":        "#ef4444",
    "wrong_to_diff_wrong":        "#f97316",
    "wrong_to_correct":           "#22c55e",
    "correct_to_same_correct":    "#3b82f6",
    "correct_to_correct_changed": "#8b5cf6",
    "correct_to_wrong":           "#6b7280",
}

CATEGORY_MARKERS = {
    "wrong_to_same_wrong":        "o",
    "wrong_to_diff_wrong":        "s",
    "wrong_to_correct":           "^",
    "correct_to_same_correct":    "D",
    "correct_to_correct_changed": "v",
    "correct_to_wrong":           "X",
}


def _plot_transition_lines(df, output_dir, severity):
    """Line plot: x = layer, y = % of all answers, one line per transition category."""
    import matplotlib.pyplot as plt

    layers_sorted = sorted(df["layer"].unique())
    if not layers_sorted:
        return

    # Compute per-layer fractions for each category
    present_cats = [
        c for c in CATEGORY_ORDER
        if c in df["transition_category"].values
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for cat in present_cats:
        fracs = []
        for layer in layers_sorted:
            ldf = df[df["layer"] == layer]
            total = len(ldf)
            count = (ldf["transition_category"] == cat).sum()
            fracs.append(100.0 * count / total if total > 0 else 0.0)

        ax.plot(
            layers_sorted, fracs,
            color=CATEGORY_COLORS.get(cat, "#9ca3af"),
            marker=CATEGORY_MARKERS.get(cat, "o"),
            markersize=4, linewidth=1.8, alpha=0.85,
            label=CATEGORY_LABELS.get(cat, cat),
        )

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("% of all answers", fontsize=11)
    ax.set_title(f"Answer Transitions across Layers — {severity}", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="both", alpha=0.25)
    ax.set_xticks(layers_sorted)
    ax.set_xticklabels(
        [str(l) for l in layers_sorted],
        fontsize=7 if len(layers_sorted) > 16 else 9,
    )

    fig.tight_layout()

    for ext in ("pdf", "png"):
        kw = {"dpi": 150} if ext == "png" else {}
        fig.savefig(
            os.path.join(output_dir, f"answer_transitions_{severity}.{ext}"),
            bbox_inches="tight", **kw,
        )
    plt.close(fig)
    print(f"[answer-change] Transition line plot saved ({severity}).")
