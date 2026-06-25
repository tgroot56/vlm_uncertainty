"""Single-token positional activation patching experiment.

For each degraded sample, iterates over prompt token positions and patches
exactly one position at a single fixed transformer layer using the corresponding
clean activation. The default sweep includes every non-visual prompt token plus
a sparse sample of visual tokens. Measures how recovery of confidence and
accuracy varies with token position.

The key question this answers: does recovery increase gradually for later prompt
tokens, or is there a sharp jump at the final prompt token?

Outputs:
  - Per-sample Parquet with one row per (sample, severity, token_position)
  - Summary CSV with mean recovery per position_index
  - Recovery plots (PDF + PNG) with x=position, y=recovery
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from src.patching.collect_activations import (
    VERSION_CLEAN,
    VISUAL_SEVERITY_VERSIONS,
    _format_prompt,
    _MAX_NEW_TOKENS,
    _score_prediction,
)
from src.patching.run_patching import (
    _compute_clean_answer_probability_stats,
    _compute_output_stats,
    _compute_pope_answer_probability_stats,
    _compute_stripped_answer_softmax_stats,
    _get_decoder_layers,
    _patched_generate,
    _run_baseline_forward,
    _stripped_answer_token_ids,
)
from src.uq_dataset_generation.token_spans import build_token_spans
from src.vlm import VLMAdapter, prepare_inputs, prepare_qwen_fused_inputs_embeds
from src.vlm.adapters import append_qwen_empty_thinking_preamble

RECOVERY_EPSILON = 1e-8
RECOVERY_CLIP = 5.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PositionalPatchingConfig:
    """Configuration for a positional patching run."""
    dataset_id: str = "vqa-v2"
    layer: int = 17
    severity: str = "moderate"  # "mild", "moderate", "severe", or "all"
    sample_visual_every: int = 10
    include_visual_last: int = 10
    max_new_tokens: int = 4
    max_new_tokens_override: Optional[int] = None
    checkpoint_every: int = 10
    verbose: bool = False
    qwen_prefill_empty_thinking: bool = False
    save_clean_answer_probability: bool = False
    save_stripped_answer_softmax: bool = False
    stripped_answer_softmax_dtype: str = "float32"
    retain_checkpoint: bool = False


# ---------------------------------------------------------------------------
# Clean activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_clean_layer_activations(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    image: Image.Image,
    prompt: str,
    layer_idx: int,
    qwen_prefill_empty_thinking: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], int]:
    """Forward pass on the clean image to extract all token activations at one layer.

    Returns:
        clean_layer_h: tensor of shape (prompt_len, H) on CPU — the clean
            activation at ``hidden_states[layer_idx]``.
        token_spans: dict from build_token_spans().
        patch_decoder_idx: the index into the decoder ModuleList whose
            *output* corresponds to ``hidden_states[layer_idx]``. On HF-style
            models where ``hidden_states[0]`` is the embedding output, this is
            ``layer_idx - 1``; if ``len(hidden_states) == num_decoder_layers``
            it is ``layer_idx``. Use this when registering the patching
            forward-hook — otherwise the clean activation is injected one
            block later than where it was captured (off-by-one layer).
    """
    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)
    if qwen_prefill_empty_thinking and vlm_adapter.family == "qwen":
        inputs = append_qwen_empty_thinking_preamble(inputs, device)
    input_ids = inputs["input_ids"][0]
    prompt_len = int(inputs["input_ids"].shape[1])

    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    lm_hidden_states = outputs.hidden_states

    # Keep only the target layer, move to CPU immediately
    layer_h = lm_hidden_states[layer_idx][0].detach().cpu()  # (prompt_len, H)

    lm_hidden_len = int(lm_hidden_states[0].shape[1])
    token_spans = build_token_spans(
        model=model,
        processor=processor,
        vlm_adapter=vlm_adapter,
        input_ids=input_ids,
        prompt_len=prompt_len,
        lm_hidden_len=lm_hidden_len,
    )

    num_hidden_state_layers = len(lm_hidden_states)
    num_decoder_layers = len(_get_decoder_layers(model, vlm_adapter))
    if num_hidden_state_layers == num_decoder_layers + 1:
        patch_decoder_idx = layer_idx - 1
    elif num_hidden_state_layers == num_decoder_layers:
        patch_decoder_idx = layer_idx
    else:
        raise RuntimeError(
            "Could not align LM hidden states with decoder layers: "
            f"hidden_state_layers={num_hidden_state_layers}, "
            f"decoder_layers={num_decoder_layers}."
        )
    if not 0 <= patch_decoder_idx < num_decoder_layers:
        raise ValueError(
            f"Resolved patch_decoder_idx={patch_decoder_idx} out of range "
            f"[0, {num_decoder_layers}) for layer_idx={layer_idx}."
        )

    return layer_h, token_spans, patch_decoder_idx


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _get_non_visual_positions(token_spans: Dict[str, Any]) -> List[int]:
    """Return all prompt token positions outside the visual span."""
    v_start = token_spans["visual_start"]
    v_end = token_spans["visual_end"]
    prompt_len = token_spans["prompt_len"]
    return list(range(0, v_start)) + list(range(v_end, prompt_len))


def _get_visual_positions(
    token_spans: Dict[str, Any],
    *,
    sample_every: int,
    include_last: int,
) -> List[int]:
    """Return a sparse subset of visual token positions to test."""
    if sample_every <= 0 and include_last <= 0:
        return []

    v_start = token_spans["visual_start"]
    v_end = token_spans["visual_end"]
    visual_positions = list(range(v_start, v_end))
    chosen: List[int] = []

    if sample_every > 0:
        chosen.extend(visual_positions[::sample_every])

    if include_last > 0:
        chosen.extend(visual_positions[-include_last:])

    return sorted(set(chosen))


def _get_test_positions(token_spans: Dict[str, Any], cfg: PositionalPatchingConfig) -> List[int]:
    """Return all token positions to test for positional patching."""
    positions = _get_non_visual_positions(token_spans)
    positions.extend(
        _get_visual_positions(
            token_spans,
            sample_every=cfg.sample_visual_every,
            include_last=cfg.include_visual_last,
        )
    )
    return sorted(set(positions))


def _classify_position(pos: int, token_spans: Dict[str, Any]) -> str:
    """Assign a semantic label to a tested token position."""
    if token_spans["visual_start"] <= pos < token_spans["visual_end"]:
        return "visual"
    if pos < token_spans["visual_start"]:
        return "text_pre"
    if pos < token_spans.get("question_start", token_spans["visual_end"]):
        return "text_post"
    if pos < token_spans.get("question_end", token_spans["prompt_len"]):
        return "question"
    return "assistant_marker"


# ---------------------------------------------------------------------------
# Checkpoint / save helpers
# ---------------------------------------------------------------------------

def _save_results(
    rows: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_path: Optional[str],
    completed_ids: set,
) -> None:
    """Save per-sample Parquet and optionally update checkpoint."""
    import pandas as pd

    parquet_path = os.path.join(output_dir, "positional_patching_results.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    if checkpoint_path is not None:
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"completed_ids": sorted(completed_ids)}, f)
        os.replace(tmp, checkpoint_path)


def _build_positional_recovery_table(df):
    """Attach per-sample recovery columns to positional patched rows."""
    import pandas as pd
    import numpy as np

    clean = (
        df[df["condition"] == "clean"]
        .drop_duplicates(["sample_id", "severity"])
        [[
            "sample_id",
            "severity",
            "score",
            "top1_probability",
            "output_entropy",
        ]]
        .rename(
            columns={
                "score": "score_clean",
                "top1_probability": "top1_probability_clean",
                "output_entropy": "output_entropy_clean",
            }
        )
    )
    degraded = (
        df[df["condition"] == "degraded"]
        .drop_duplicates(["sample_id", "severity"])
        [[
            "sample_id",
            "severity",
            "score",
            "top1_probability",
            "output_entropy",
        ]]
        .rename(
            columns={
                "score": "score_degraded",
                "top1_probability": "top1_probability_degraded",
                "output_entropy": "output_entropy_degraded",
            }
        )
    )

    patched = df[df["condition"].str.startswith("patched_pos_", na=False)].copy()
    patched = patched.merge(clean, on=["sample_id", "severity"], how="left")
    patched = patched.merge(degraded, on=["sample_id", "severity"], how="left")

    metric_specs = [
        ("score", "recovery_score"),
        ("top1_probability", "recovery_top1"),
        ("output_entropy", "recovery_entropy"),
    ]
    for metric, recovery_col in metric_specs:
        num = patched[metric] - patched[f"{metric}_degraded"]
        den = patched[f"{metric}_clean"] - patched[f"{metric}_degraded"] + RECOVERY_EPSILON
        patched[recovery_col] = np.clip(num / den, -RECOVERY_CLIP, RECOVERY_CLIP)

    return patched


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_positional_patching(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_records: List[Dict[str, Any]],
    output_dir: str,
    cfg: PositionalPatchingConfig,
) -> None:
    """Execute single-token positional patching for all samples.

    For each sample, extracts clean activations at cfg.layer, then patches one
    selected prompt token position at a time into the degraded forward pass.

    Args:
        model, processor, device, vlm_adapter: loaded VLM.
        patching_records: sample dicts from the patching dataset JSON.
        output_dir: where to write results.
        cfg: experiment configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_new_tokens = (
        cfg.max_new_tokens_override
        if cfg.max_new_tokens_override is not None
        else _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)
    )

    # Determine which severities to run
    if cfg.severity == "all":
        severities = ["mild", "moderate", "severe", "extreme"]
    else:
        severities = [cfg.severity]

    # Checkpoint support
    checkpoint_path = os.path.join(output_dir, "positional_patching_checkpoint.json")
    completed_ids: set = set()
    rows: List[Dict[str, Any]] = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_ids = {str(sample_id) for sample_id in ckpt.get("completed_ids", [])}
        parquet_path = os.path.join(output_dir, "positional_patching_results.parquet")
        if completed_ids and os.path.exists(parquet_path):
            import pandas as pd
            rows = pd.read_parquet(parquet_path).to_dict("records")
        print(f"[pos-patch] Resuming: {len(completed_ids)} samples done.")
    else:
        parquet_path = os.path.join(output_dir, "positional_patching_results.parquet")
        if os.path.exists(parquet_path):
            import pandas as pd
            existing = pd.read_parquet(parquet_path)
            if not existing.empty and "sample_id" in existing.columns:
                rows = existing.to_dict("records")
                completed_ids = {
                    str(sample_id)
                    for sample_id in existing["sample_id"].dropna().unique().tolist()
                }
                print(
                    f"[pos-patch] Resuming from existing parquet: "
                    f"{len(completed_ids)} samples done."
                )

    total = len(patching_records)

    print(f"[pos-patch] Starting positional patching")
    print(f"  Samples:    {total}")
    print(f"  Layer:      {cfg.layer}")
    print(f"  Severities: {severities}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(
        f"  Visual sampling: every={cfg.sample_visual_every or 'off'}, "
        f"last={cfg.include_visual_last}"
    )
    print(f"  Qwen prefill: {cfg.qwen_prefill_empty_thinking}")
    print(f"  Clean answer P: {cfg.save_clean_answer_probability}")
    print(
        f"  Stripped softmax: {cfg.save_stripped_answer_softmax}"
        + (
            f" ({cfg.stripped_answer_softmax_dtype})"
            if cfg.save_stripped_answer_softmax else ""
        )
    )
    print(f"  Retain checkpoint: {cfg.retain_checkpoint}")
    print(f"  Output:     {output_dir}/")

    t0 = time.time()
    for record in tqdm(patching_records, desc="positional patching"):
        sample_id = str(record["sample_id"])
        if sample_id in completed_ids:
            continue

        prompt = _format_prompt(cfg.dataset_id, record["clean_question"])

        # --- Extract clean activations at target layer ---
        clean_image = Image.open(record["clean_image_path"]).convert("RGB")

        clean_layer_h, token_spans, patch_decoder_idx = _extract_clean_layer_activations(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            layer_idx=cfg.layer,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        test_positions = _get_test_positions(token_spans, cfg)
        visual_positions = [
            pos for pos in test_positions
            if token_spans["visual_start"] <= pos < token_spans["visual_end"]
        ]

        if cfg.verbose:
            print(
                f"\n  [sample {sample_id}] prompt_len={token_spans['prompt_len']}, "
                f"visual=[{token_spans['visual_start']},{token_spans['visual_end']}), "
                f"tested positions={len(test_positions)} "
                f"(visual sampled={len(visual_positions)})"
            )

        # --- Qwen fast path: pre-compute fused embeds per image (see note in
        # run_layer_sweep_patching.py). 30-40x per-call speedup on Qwen.
        # The fused bundle must include the same empty-thinking preamble used
        # for activation extraction so prompt positions remain aligned.
        use_fused = vlm_adapter.family == "qwen"
        clean_fused = (
            prepare_qwen_fused_inputs_embeds(
                model,
                vlm_adapter,
                processor,
                clean_image,
                prompt,
                device,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            )
            if use_fused else None
        )

        # --- Run clean baseline ---
        clean_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            max_new_tokens=max_new_tokens,
            fused_bundle=clean_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        clean_image.close()
        clean_stats = _compute_output_stats(
            clean_fwd["gen_step_logits"], clean_fwd.get("gen_ids"), processor
        )
        clean_score = _score_prediction(cfg.dataset_id, clean_fwd["raw_text"], record)
        clean_answer_text = clean_fwd["raw_text"]
        clean_answer_token_ids = _stripped_answer_token_ids(
            clean_fwd.get("gen_ids"), processor
        )

        def _extra_answer_stats(fwd: Dict[str, Any]) -> Dict[str, Any]:
            extra: Dict[str, Any] = {}
            if cfg.dataset_id == "pope":
                extra.update(
                    _compute_pope_answer_probability_stats(
                        fwd["gen_step_logits"], processor, clean_answer_text
                    )
                )
            if cfg.save_clean_answer_probability:
                extra.update(
                    _compute_clean_answer_probability_stats(
                        fwd["gen_step_logits"],
                        fwd.get("gen_ids"),
                        processor,
                        clean_answer_text=clean_answer_text,
                        clean_answer_token_ids=clean_answer_token_ids,
                    )
                )
            if cfg.save_stripped_answer_softmax:
                extra.update(
                    _compute_stripped_answer_softmax_stats(
                        fwd["gen_step_logits"],
                        fwd.get("gen_ids"),
                        processor,
                        dtype=cfg.stripped_answer_softmax_dtype,
                    )
                )
            return extra

        # --- For each severity ---
        for pert in record.get("visual_perturbations", []):
            severity = pert["severity"]
            if severity not in severities:
                continue

            img_path = pert.get("blurred_image_path", "")
            if not img_path or not os.path.exists(img_path):
                continue

            blurred_image = Image.open(img_path).convert("RGB")
            blurred_fused = (
                prepare_qwen_fused_inputs_embeds(
                    model,
                    vlm_adapter,
                    processor,
                    blurred_image,
                    prompt,
                    device,
                    qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
                )
                if use_fused else None
            )

            # Degraded baseline (no patching)
            deg_fwd = _run_baseline_forward(
                model=model, processor=processor, device=device,
                vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
                max_new_tokens=max_new_tokens,
                fused_bundle=blurred_fused,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            )
            deg_stats = _compute_output_stats(
                deg_fwd["gen_step_logits"], deg_fwd.get("gen_ids"), processor
            )
            deg_score = _score_prediction(cfg.dataset_id, deg_fwd["raw_text"], record)

            # Record baselines
            rows.append({
                "sample_id": sample_id,
                "severity": severity,
                "condition": "clean",
                "token_position": None,
                "position_index": None,
                "span_label": None,
                "predicted_answer": clean_fwd["raw_text"],
                "score": clean_score,
                **clean_stats,
                **_extra_answer_stats(clean_fwd),
            })
            rows.append({
                "sample_id": sample_id,
                "severity": severity,
                "condition": "degraded",
                "token_position": None,
                "position_index": None,
                "span_label": None,
                "predicted_answer": deg_fwd["raw_text"],
                "score": deg_score,
                **deg_stats,
                **_extra_answer_stats(deg_fwd),
            })

            # --- Per-position patching ---
            for idx, pos in enumerate(test_positions):
                clean_act = clean_layer_h[pos].unsqueeze(0)  # (1, H)

                patched_fwd = _patched_generate(
                    model=model, processor=processor, device=device,
                    vlm_adapter=vlm_adapter, image=blurred_image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    layer_idx=patch_decoder_idx,
                    positions=[pos],
                    clean_activations=clean_act,
                    fused_bundle=blurred_fused,
                    qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
                )
                patched_stats = _compute_output_stats(
                    patched_fwd["gen_step_logits"], patched_fwd.get("gen_ids"), processor
                )
                patched_score = _score_prediction(
                    cfg.dataset_id, patched_fwd["raw_text"], record,
                )

                span_label = _classify_position(pos, token_spans)

                rows.append({
                    "sample_id": sample_id,
                    "severity": severity,
                    "condition": f"patched_pos_{pos}",
                    "token_position": pos,
                    "position_index": idx,
                    "span_label": span_label,
                    "predicted_answer": patched_fwd["raw_text"],
                    "score": patched_score,
                    **patched_stats,
                    **_extra_answer_stats(patched_fwd),
                })

                if cfg.verbose:
                    print(
                        f"    pos={pos} ({span_label}) "
                        f"deg={deg_stats['top1_probability']:.3f} "
                        f"→ patched={patched_stats['top1_probability']:.3f} "
                        f"(clean={clean_stats['top1_probability']:.3f})"
                    )

            blurred_image.close()

        completed_ids.add(sample_id)

        # Periodic checkpoint
        if len(completed_ids) % cfg.checkpoint_every == 0:
            _save_results(rows, output_dir, checkpoint_path, completed_ids)
            elapsed = time.time() - t0
            rate = len(completed_ids) / elapsed if elapsed > 0 else 0
            print(f"[pos-patch] {len(completed_ids)}/{total} samples "
                  f"({elapsed:.0f}s, {rate:.1f} samples/s)")

    # Final save
    _save_results(
        rows,
        output_dir,
        checkpoint_path=checkpoint_path if cfg.retain_checkpoint else None,
        completed_ids=completed_ids,
    )

    # Clean up checkpoint
    if not cfg.retain_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    print(f"\n[pos-patch] Finished {len(completed_ids)}/{total} samples in {elapsed:.1f}s")

    # Print summary and plot
    _print_positional_summary(rows, output_dir)

    from pathlib import Path as _Path
    from src.cli.plot_sweep_filtered import (
        plot_positional_sweep_combined,
        plot_two_metrics_cross_model_positional_sweep,
    )

    parquet_path = os.path.join(output_dir, "positional_patching_results.parquet")
    plot_positional_sweep_combined(parquet_path, _Path(output_dir), raw=False)
    plot_positional_sweep_combined(parquet_path, _Path(output_dir), raw=True)
    plot_positional_sweep_combined(parquet_path, _Path(output_dir), raw=True, min_delta=0.0)
    if cfg.save_clean_answer_probability:
        model_label = (
            "Qwen3.5-9B" if vlm_adapter.family == "qwen" else "LLaVA-1.5-7B"
        )
        plot_two_metrics_cross_model_positional_sweep(
            model_entries=[(model_label, {cfg.dataset_id: parquet_path})],
            datasets=[cfg.dataset_id],
            output_dir=_Path(output_dir),
            output_name="positional_sweep_clean_answer_confidence_accuracy_recovery",
            min_delta=0.0,
            metrics=["clean_answer_probability", "score"],
            title="Positional Sweep - Clean-Answer Confidence & Accuracy Recovery",
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_positional_summary(rows: List[Dict[str, Any]], output_dir: str) -> None:
    """Compute per-position recovery and save summary CSV."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print("[pos-patch] No results to summarize.")
        return

    patched_with_recovery = _build_positional_recovery_table(df)
    severities = df[df["condition"] == "degraded"]["severity"].unique()
    summary_rows: List[Dict[str, Any]] = []

    for severity in severities:
        sev_df = df[df["severity"] == severity]
        sev_patched = patched_with_recovery[patched_with_recovery["severity"] == severity]

        # Aggregate baselines across all samples for this severity
        clean_df = sev_df[sev_df["condition"] == "clean"]
        deg_df = sev_df[sev_df["condition"] == "degraded"]

        mean_clean_score = clean_df["score"].mean()
        mean_clean_top1 = clean_df["top1_probability"].mean()
        mean_clean_entropy = clean_df["output_entropy"].mean()

        mean_deg_score = deg_df["score"].mean()
        mean_deg_top1 = deg_df["top1_probability"].mean()
        mean_deg_entropy = deg_df["output_entropy"].mean()

        # Per-position aggregation with recovery computed per sample first.
        pos_agg = sev_patched.groupby("position_index").agg(
            token_position_mode=("token_position", lambda x: x.mode().iloc[0] if len(x) > 0 else None),
            span_label=("span_label", lambda x: x.mode().iloc[0] if len(x) > 0 else None),
            mean_score=("score", "mean"),
            mean_top1_prob=("top1_probability", "mean"),
            mean_entropy=("output_entropy", "mean"),
            recovery_score=("recovery_score", "mean"),
            recovery_top1=("recovery_top1", "mean"),
            recovery_entropy=("recovery_entropy", "mean"),
            N=("sample_id", "count"),
        ).reset_index()

        for _, row in pos_agg.iterrows():
            summary_rows.append({
                "severity": severity,
                "position_index": int(row["position_index"]),
                "token_position_mode": int(row["token_position_mode"]),
                "span_label": row["span_label"],
                "mean_score": row["mean_score"],
                "mean_top1_prob": row["mean_top1_prob"],
                "mean_entropy": row["mean_entropy"],
                "recovery_score": row["recovery_score"],
                "recovery_top1": row["recovery_top1"],
                "recovery_entropy": row["recovery_entropy"],
                "N": int(row["N"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "positional_patching_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[pos-patch] Summary saved to {summary_path}")

    # Print to console (abbreviated)
    print(f"\n{'Sev':<10s} {'Idx':>4s} {'Span':<18s} {'Score':>7s} {'Top-1':>7s} "
          f"{'Entropy':>8s} {'Rec(S)':>7s} {'Rec(T1)':>8s} {'Rec(E)':>7s} {'N':>5s}")
    print("-" * 95)
    for _, row in summary_df.iterrows():
        print(
            f"{row['severity']:<10s} {int(row['position_index']):>4d} "
            f"{row['span_label']:<18s} "
            f"{row['mean_score']:>7.4f} {row['mean_top1_prob']:>7.4f} "
            f"{row['mean_entropy']:>8.4f} {row['recovery_score']:>7.3f} "
            f"{row['recovery_top1']:>8.3f} {row['recovery_entropy']:>7.3f} "
            f"{int(row['N']):>5d}"
        )

    # Print baselines for context
    for severity in severities:
        sev_df = df[df["severity"] == severity]
        clean_df = sev_df[sev_df["condition"] == "clean"]
        deg_df = sev_df[sev_df["condition"] == "degraded"]
        print(f"\n  [{severity}] clean: score={clean_df['score'].mean():.4f} "
              f"top1={clean_df['top1_probability'].mean():.4f} "
              f"entropy={clean_df['output_entropy'].mean():.4f}")
        print(f"  [{severity}] degraded: score={deg_df['score'].mean():.4f} "
              f"top1={deg_df['top1_probability'].mean():.4f} "
              f"entropy={deg_df['output_entropy'].mean():.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_positional_recovery(
    results_parquet: str,
    output_dir: str,
    *,
    layer: int,
    title_prefix: str = "",
) -> None:
    """Plot per-position recovery curves from a positional patching results file.

    Creates a figure with 3 vertically stacked subplots:
      - Score recovery
      - Top-1 probability recovery
      - Entropy recovery

    x-axis = position_index (0-based index in the tested-token sweep)
    y-axis = recovery metric

    If multiple severities are present, each is plotted as a separate curve.
    Span boundaries (text_pre / question / assistant_marker) are shown as
    background shading.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(results_parquet):
        print(f"[pos-patch] No results file found at {results_parquet}, skipping plot.")
        return

    df = pd.read_parquet(results_parquet)
    if df.empty:
        print("[pos-patch] Empty results, skipping plot.")
        return

    severities = sorted(df[df["condition"] == "degraded"]["severity"].unique())
    patched_with_recovery = _build_positional_recovery_table(df)

    SEVERITY_COLORS = {
        "mild": "#0ea5e9",
        "moderate": "#f59e0b",
        "severe": "#ef4444",
        "extreme": "#7c3aed",
    }

    SPAN_COLORS = {
        "visual": "#dcfce7",
        "text_pre": "#e0f2fe",
        "text_post": "#e0f2fe",
        "question": "#fef3c7",
        "assistant_marker": "#fce7f3",
    }

    metrics = [
        ("recovery_score", "Score Recovery"),
        ("recovery_top1", "Top-1 Probability Recovery"),
        ("recovery_entropy", "Entropy Recovery"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for severity in severities:
        sev_df = df[df["severity"] == severity]
        patched_df = patched_with_recovery[patched_with_recovery["severity"] == severity]

        pos_agg = patched_df.groupby("position_index").agg(
            span_label=("span_label", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
            recovery_score=("recovery_score", "mean"),
            recovery_top1=("recovery_top1", "mean"),
            recovery_entropy=("recovery_entropy", "mean"),
        ).sort_index()

        positions = pos_agg.index.values
        span_labels = pos_agg["span_label"].values

        for ax, (metric_key, metric_title) in zip(axes, metrics):
            recovery = pos_agg[metric_key].values

            color = SEVERITY_COLORS.get(severity, "#6b7280")
            ax.plot(positions, recovery, marker="o", markersize=3,
                    linewidth=1.5, color=color, label=severity, alpha=0.85)

    # Add span shading and reference lines (using first severity's span labels)
    first_sev = severities[0]
    sev_df = df[(df["severity"] == first_sev) & df["condition"].str.startswith("patched_pos_")]
    pos_agg_first = sev_df.groupby("position_index").agg(
        span_label=("span_label", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
    ).sort_index()

    if len(pos_agg_first) > 0:
        positions_all = pos_agg_first.index.values
        labels_all = pos_agg_first["span_label"].values

        for ax, (_, metric_title) in zip(axes, metrics):
            # Span shading
            prev_label = labels_all[0] if len(labels_all) > 0 else ""
            span_start = positions_all[0] if len(positions_all) > 0 else 0
            for i in range(1, len(positions_all)):
                if labels_all[i] != prev_label:
                    bg_color = SPAN_COLORS.get(prev_label, "#f3f4f6")
                    ax.axvspan(span_start - 0.5, positions_all[i - 1] + 0.5,
                               alpha=0.3, color=bg_color, zorder=0)
                    span_start = positions_all[i]
                    prev_label = labels_all[i]
            # Final span
            if len(positions_all) > 0:
                bg_color = SPAN_COLORS.get(prev_label, "#f3f4f6")
                ax.axvspan(span_start - 0.5, positions_all[-1] + 0.5,
                           alpha=0.3, color=bg_color, zorder=0)

            # Reference lines
            ax.axhline(y=0.0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(y=1.0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.7)

            ax.set_ylabel(metric_title)
            ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Tested token index")

    if len(severities) > 1:
        axes[0].legend(loc="upper left", fontsize=9)

    title = f"{title_prefix}Positional Patching Recovery (layer {layer})"
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    # Add span label annotations below x-axis
    if len(pos_agg_first) > 0:
        unique_labels = []
        for lbl in labels_all:
            if not unique_labels or unique_labels[-1] != lbl:
                unique_labels.append(lbl)
        label_text = " → ".join(unique_labels)
        fig.text(0.5, -0.02, f"Span regions: {label_text}",
                 ha="center", fontsize=9, color="#6b7280")

    # Save
    for ext in ("pdf", "png"):
        sev_tag = "all" if len(severities) > 1 else severities[0]
        fname = f"positional_recovery_layer_{layer}_{sev_tag}.{ext}"
        save_kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(os.path.join(output_dir, fname),
                    bbox_inches="tight", **save_kwargs)

    plt.close(fig)
    sev_tag = "all" if len(severities) > 1 else severities[0]
    print(f"[pos-patch] Plot saved to {output_dir}/positional_recovery_layer_{layer}_{sev_tag}.pdf")
