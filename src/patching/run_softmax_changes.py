"""Softmax-changes proof-of-concept for the layer sweep.

For a small set of ``coco-qa-vi`` samples that satisfy the main layer-sweep
cohort filter (clean answer correct, corrupted answer incorrect), this records
how the model's *output* softmax distribution changes across the layer sweep.

Experimental setup (identical to the main layer-sweep plot):
  - corrupt with the requested ``severity`` (``extreme`` for coco-qa-vi),
  - patch the **final prompt token** (``position_offset == -1``) at **every LM
    layer**, one layer at a time,
  - read the model's output distribution after generation.

For every retained sample we store, at the first stripped answer-token step:
  - the clean baseline top-5 tokens / probabilities (unpatched, constant),
  - the corrupted baseline top-5 (unpatched, constant),
  - the patched top-5 for each patch layer L.

Output
------
A single self-contained JSON per (model, dataset) — small enough to load
directly in the qualitative viewer. See ``_write_json`` for the schema.

This module deliberately reuses the existing layer-sweep / patching pipeline
(``src.patching.run_layer_sweep_patching`` and ``src.patching.run_patching``)
so the numbers match the production layer sweep exactly. It does not introduce
any new model-loading, patching, or correctness-scoring logic.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.patching.collect_activations import (
    _format_prompt,
    _MAX_NEW_TOKENS,
    _score_prediction,
)
from src.patching.run_layer_sweep_patching import (
    _extract_all_layer_activations,
    _get_final_prompt_positions,
    _get_layer_index_pairs,
    _position_offset_label,
)
from src.patching.run_patching import (
    _answer_token_indices,
    _compute_clean_answer_probability_stats,
    _compute_output_stats,
    _get_decoder_layers,
    _patched_generate,
    _run_baseline_forward,
    _stripped_answer_token_ids,
    _token_piece,
)
from src.vlm import VLMAdapter, prepare_qwen_fused_inputs_embeds

# v2 adds per-run ``clean_answer_probability`` (P assigned by each run to the
# clean run's answer tokens) for the viewer's "Clean answer confidence" line.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SoftmaxChangesConfig:
    """Configuration for a softmax-changes proof-of-concept run."""
    dataset_id: str = "coco-qa-vi"
    severity: str = "extreme"
    n_samples: int = 10          # number of *retained* (filter-passing) samples
    max_candidates: int = 100    # bound on samples scanned looking for retained ones
    seed: int = 0
    top_k: int = 5
    max_new_tokens: int = 4
    max_new_tokens_override: Optional[int] = None
    qwen_prefill_empty_thinking: bool = False
    verbose: bool = False


# ---------------------------------------------------------------------------
# Top-5 softmax helper
# ---------------------------------------------------------------------------

def _compute_top5_softmax(
    gen_step_logits: torch.Tensor,
    gen_ids: Optional[torch.Tensor],
    processor: Any,
    *,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Top-k tokens of the output softmax at the first stripped answer-token step.

    Uses the same answer-token logic as ``_compute_output_stats`` so the decision
    step matches the plotted ``top1_probability``. Falls back to generation step 0
    when no stripped answer token can be located.
    """
    if (
        gen_step_logits is None
        or not isinstance(gen_step_logits, torch.Tensor)
        or gen_step_logits.ndim < 3
        or gen_step_logits.shape[1] == 0
    ):
        return []

    step_idx = 0
    if gen_ids is not None and processor is not None:
        answer_indices = [
            idx for idx in _answer_token_indices(gen_ids, processor)
            if idx < gen_step_logits.shape[1]
        ]
        if answer_indices:
            step_idx = answer_indices[0]

    probs = F.softmax(gen_step_logits[0, step_idx, :].float(), dim=0)
    k = int(min(top_k, probs.shape[0]))
    top_probs, top_ids = torch.topk(probs, k)
    entries: List[Dict[str, Any]] = []
    for rank, (prob, token_id) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
        token_id = int(token_id)
        entries.append({
            "rank": rank,
            "token_id": token_id,
            "token_text": _token_piece(processor, token_id) if processor is not None else str(token_id),
            "prob": float(prob),
        })
    return entries


def _finite_or_none(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` for non-numeric / NaN."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if numeric != numeric else numeric  # NaN != NaN


def _run_summary(
    fwd: Dict[str, Any],
    stats: Dict[str, Any],
    score: float,
    top5: List[Dict[str, Any]],
    clean_answer_probability: float,
) -> Dict[str, Any]:
    """Compact per-run record for the JSON payload.

    ``clean_answer_probability`` is the probability this run assigned to the
    clean run's stripped answer tokens (NaN-safe ``None`` when unavailable).
    """
    return {
        "answer": fwd["raw_text"],
        "score": float(score),
        "top1_probability": float(stats.get("top1_probability", float("nan"))),
        "output_entropy": float(stats.get("output_entropy", float("nan"))),
        "clean_answer_probability": clean_answer_probability,
        "top5": top5,
    }


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_softmax_changes(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_records: List[Dict[str, Any]],
    output_path: str,
    model_label: str,
    model_id: str,
    cfg: SoftmaxChangesConfig,
) -> Dict[str, Any]:
    """Scan a shuffled candidate pool, retain filter-passing samples, and for each
    record clean / corrupted / per-layer-patched top-5 output softmax. Writes JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    max_new_tokens = (
        cfg.max_new_tokens_override
        if cfg.max_new_tokens_override is not None
        else _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)
    )
    use_fused = vlm_adapter.family == "qwen"

    candidates = list(patching_records)
    random.Random(cfg.seed).shuffle(candidates)
    candidates = candidates[: cfg.max_candidates]

    print("[softmax-changes] Starting proof-of-concept run")
    print(f"  Model:            {model_label} ({model_id})")
    print(f"  Dataset:          {cfg.dataset_id}")
    print(f"  Severity:         {cfg.severity}")
    print(f"  Retain target:    {cfg.n_samples}")
    print(f"  Candidate pool:   {len(candidates)} (seed={cfg.seed})")
    print(f"  Top-k:            {cfg.top_k}")
    print(f"  Max new tokens:   {max_new_tokens}")
    print(f"  Qwen prefill:     {cfg.qwen_prefill_empty_thinking}")
    print(f"  Output:           {output_path}")

    samples_out: List[Dict[str, Any]] = []
    n_scanned = 0
    num_layers: Optional[int] = None
    t0 = time.time()

    for record in tqdm(candidates, desc="softmax-changes"):
        if len(samples_out) >= cfg.n_samples:
            break
        n_scanned += 1
        sample_id = str(record["sample_id"])

        # Locate the requested-severity perturbation up front; skip if absent.
        pert = next(
            (
                p for p in record.get("visual_perturbations", [])
                if p.get("severity") == cfg.severity
                and p.get("blurred_image_path")
                and os.path.exists(p["blurred_image_path"])
            ),
            None,
        )
        if pert is None:
            continue

        prompt = _format_prompt(cfg.dataset_id, record["clean_question"])
        clean_image = Image.open(record["clean_image_path"]).convert("RGB")
        blurred_image = Image.open(pert["blurred_image_path"]).convert("RGB")

        clean_fused = (
            prepare_qwen_fused_inputs_embeds(
                model, vlm_adapter, processor, clean_image, prompt, device,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            ) if use_fused else None
        )
        blurred_fused = (
            prepare_qwen_fused_inputs_embeds(
                model, vlm_adapter, processor, blurred_image, prompt, device,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            ) if use_fused else None
        )

        # ---- baselines (unpatched) ----
        clean_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            max_new_tokens=max_new_tokens, fused_bundle=clean_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        clean_score = _score_prediction(cfg.dataset_id, clean_fwd["raw_text"], record)

        deg_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
            max_new_tokens=max_new_tokens, fused_bundle=blurred_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        deg_score = _score_prediction(cfg.dataset_id, deg_fwd["raw_text"], record)

        # ---- cohort filter: clean correct, corrupted incorrect ----
        if not (clean_score > 0 and deg_score == 0):
            clean_image.close()
            blurred_image.close()
            if cfg.verbose:
                print(f"  [skip] {sample_id}: clean={clean_score}, corrupted={deg_score}")
            continue

        clean_stats = _compute_output_stats(clean_fwd["gen_step_logits"], clean_fwd.get("gen_ids"), processor)
        deg_stats = _compute_output_stats(deg_fwd["gen_step_logits"], deg_fwd.get("gen_ids"), processor)
        clean_top5 = _compute_top5_softmax(clean_fwd["gen_step_logits"], clean_fwd.get("gen_ids"), processor, top_k=cfg.top_k)
        deg_top5 = _compute_top5_softmax(deg_fwd["gen_step_logits"], deg_fwd.get("gen_ids"), processor, top_k=cfg.top_k)

        # P(clean answer) assigned by each run to the clean run's stripped answer
        # tokens (the viewer's "Clean answer confidence" line). Reuses the exact
        # layer-sweep helper so values match the production parquets.
        clean_answer_text = clean_fwd["raw_text"]
        clean_answer_token_ids = _stripped_answer_token_ids(clean_fwd.get("gen_ids"), processor)

        def _clean_answer_probability(fwd: Dict[str, Any]) -> Optional[float]:
            stats = _compute_clean_answer_probability_stats(
                fwd["gen_step_logits"], fwd.get("gen_ids"), processor,
                clean_answer_text=clean_answer_text,
                clean_answer_token_ids=clean_answer_token_ids,
            )
            return _finite_or_none(stats.get("clean_answer_probability"))

        clean_ap = _clean_answer_probability(clean_fwd)
        deg_ap = _clean_answer_probability(deg_fwd)

        # ---- clean activations at all layers + final prompt token position ----
        all_layer_h, token_spans = _extract_all_layer_activations(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        clean_image.close()

        decoder_layers = _get_decoder_layers(model, vlm_adapter)
        layer_index_pairs = _get_layer_index_pairs(
            num_hidden_state_layers=all_layer_h.shape[0],
            num_decoder_layers=len(decoder_layers),
        )
        if num_layers is None:
            num_layers = len(layer_index_pairs)

        positions = _get_final_prompt_positions(token_spans, 1)  # final prompt token (offset -1)
        if not positions:
            blurred_image.close()
            if cfg.verbose:
                print(f"  [skip] {sample_id}: no final prompt position")
            continue
        final_pos = positions[0]
        final_offset = _position_offset_label(final_pos, token_spans)

        # ---- per-layer patched runs ----
        layers_out: List[Dict[str, Any]] = []
        for hidden_state_idx, patch_layer_idx in layer_index_pairs:
            clean_act = all_layer_h[hidden_state_idx, final_pos, :].unsqueeze(0)  # (1, H)
            patched_fwd = _patched_generate(
                model=model, processor=processor, device=device,
                vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
                max_new_tokens=max_new_tokens, layer_idx=patch_layer_idx,
                positions=[final_pos], clean_activations=clean_act,
                fused_bundle=blurred_fused,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            )
            patched_stats = _compute_output_stats(
                patched_fwd["gen_step_logits"], patched_fwd.get("gen_ids"), processor
            )
            patched_score = _score_prediction(cfg.dataset_id, patched_fwd["raw_text"], record)
            patched_top5 = _compute_top5_softmax(
                patched_fwd["gen_step_logits"], patched_fwd.get("gen_ids"), processor, top_k=cfg.top_k
            )
            layers_out.append({
                "layer": int(patch_layer_idx),
                "patched": _run_summary(
                    patched_fwd, patched_stats, patched_score, patched_top5,
                    _clean_answer_probability(patched_fwd),
                ),
            })

        blurred_image.close()

        samples_out.append({
            "sample_id": sample_id,
            "question": str(record.get("clean_question", "")),
            "ground_truth": record.get("gt_answers") or record.get("correct_answer"),
            "clean_image_path": str(record.get("clean_image_path", "")),
            "degraded_image_path": str(pert.get("blurred_image_path", "")),
            "severity": cfg.severity,
            "final_prompt_token_position": int(final_pos),
            "position_offset": int(final_offset),
            "clean": _run_summary(clean_fwd, clean_stats, clean_score, clean_top5, clean_ap),
            "corrupted": _run_summary(deg_fwd, deg_stats, deg_score, deg_top5, deg_ap),
            "layers": layers_out,
        })
        if cfg.verbose:
            print(f"  [keep] {sample_id}: {len(samples_out)}/{cfg.n_samples} retained")

    elapsed = time.time() - t0
    payload = _write_json(
        output_path=output_path,
        model_label=model_label,
        model_id=model_id,
        cfg=cfg,
        max_new_tokens=max_new_tokens,
        n_scanned=n_scanned,
        num_layers=num_layers or 0,
        samples_out=samples_out,
    )
    print(
        f"\n[softmax-changes] Retained {len(samples_out)}/{cfg.n_samples} "
        f"(scanned {n_scanned}) in {elapsed:.1f}s → {output_path}"
    )
    if len(samples_out) < cfg.n_samples:
        print(
            f"[softmax-changes] WARNING: only {len(samples_out)} samples passed the filter "
            f"within max_candidates={cfg.max_candidates}. Raise --max_candidates to retain more."
        )
    return payload


def _write_json(
    *,
    output_path: str,
    model_label: str,
    model_id: str,
    cfg: SoftmaxChangesConfig,
    max_new_tokens: int,
    n_scanned: int,
    num_layers: int,
    samples_out: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "model_label": model_label,
            "model_id": model_id,
            "dataset": cfg.dataset_id,
            "severity": cfg.severity,
            "position_offset": -1,
            "answer_step": "first_stripped_answer_token",
            "top_k": cfg.top_k,
            "filter": "clean_score>0 and corrupted_score==0",
            "n_requested": cfg.n_samples,
            "n_candidates_scanned": n_scanned,
            "n_retained": len(samples_out),
            "seed": cfg.seed,
            "num_layers": num_layers,
            "max_new_tokens": max_new_tokens,
            "qwen_prefill_empty_thinking": cfg.qwen_prefill_empty_thinking,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "samples": samples_out,
    }
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, output_path)
    return payload
