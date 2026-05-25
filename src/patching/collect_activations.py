"""Collect per-span, per-layer activations for the activation-patching experiment.

For each sample in a previously built patching dataset, runs forward passes on
the clean image and each blurred severity, then stores:

  1.  Three last-token feature vectors per (version, layer):

        lm_visual_lasttok_layer_X   – last token of the visual span
        lm_question_lasttok_layer_X – last token of the question span
        lm_fullspan_lasttok_layer_X – last token of the full prompt span

  2.  Output-distribution statistics (logits, top-1, entropy, correctness) in a
      single Parquet file.

  3.  Per-sample metadata (token-span position indices) as JSON.

Sigma values are read from the patching dataset JSON (produced by the
calibration step) — no separate sigma argument needed.

Reuses ``build_token_spans`` from ``src.uq_dataset_generation.token_spans`` and
``extract_lm_last_k_tokens`` from the main UQ generation pipeline to guarantee
identical span definitions.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.labeling.vqa import score_vqa
from src.labeling.multiple_choice import score_multiple_choice
from src.uq_dataset_generation.token_spans import build_token_spans
from src.uq_dataset_generation.tasks.pope import _extract_yes_no_answer
from src.vlm import VLMAdapter, prepare_inputs
from utils.extract_features import (
    extract_lm_last_k_tokens,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERSION_CLEAN = "clean"
VISUAL_SEVERITY_VERSIONS = ["visual_mild", "visual_moderate", "visual_severe"]


@dataclass
class PatchingActivationConfig:
    """All knobs for an activation-collection run."""
    dataset_id: str = "vqa-v2"
    middle_layer: int = 16
    max_new_tokens: int = 4
    checkpoint_every: int = 50
    verbose: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset-specific prompt formatting and scoring
# ---------------------------------------------------------------------------

# Max new tokens per dataset — matches the probing experiments
_MAX_NEW_TOKENS: Dict[str, int] = {
    "vqa-v2": 4,
    "coco-qa-vi": 4,
    "pope": 15,
    "imagenet-r": 4,
}


def _format_prompt(dataset_id: str, question: str) -> str:
    """Reconstruct the instruction-formatted prompt from raw question text.

    Mirrors the prompt templates in src/data/common.py used by the probing
    experiments.
    """
    if dataset_id == "vqa-v2":
        return f"Question: {question}\nProvide a short answer."
    if dataset_id == "coco-qa-vi":
        return f"Question: {question}\nProvide a short answer."
    if dataset_id == "pope":
        return f"Question: {question}\nAnswer with exactly one word: yes or no."
    if dataset_id == "imagenet-r":
        # ImageNet-R prompts include the full option list and are stored as-is
        # in the patching dataset's clean_question field.
        return question
    raise ValueError(f"Unknown dataset_id '{dataset_id}' for prompt formatting")


def _score_prediction(
    dataset_id: str,
    pred_answer: str,
    sample_record: Dict[str, Any],
) -> float:
    """Score a prediction using the same logic as the probing experiments.

    Returns the soft/binary score in [0, 1].
    """
    if dataset_id == "vqa-v2":
        # Soft VQA accuracy using all GT answers
        result = score_vqa(
            pred_answer=pred_answer,
            gt_answers=sample_record.get("gt_answers"),
        )
        return float(result.score)

    if dataset_id == "coco-qa-vi":
        # Normalised exact match against single GT answer
        result = score_vqa(
            pred_answer=pred_answer,
            gt_answer_single=sample_record.get("correct_answer"),
        )
        return float(result.score)

    if dataset_id == "pope":
        # Extract yes/no decision, then normalised exact match
        parsed = _extract_yes_no_answer(pred_answer)
        result = score_vqa(
            pred_answer=parsed,
            gt_answer_single=sample_record.get("correct_answer"),
        )
        return float(result.score)

    if dataset_id == "imagenet-r":
        # Multiple-choice: compare predicted key vs gt_key
        result = score_multiple_choice(
            pred_letter=pred_answer,
            gt_letter=sample_record.get("gt_key"),
            normalize=True,
        )
        return float(result.score)

    raise ValueError(f"Unknown dataset_id '{dataset_id}' for scoring")


# ---------------------------------------------------------------------------
# Forward-pass helper
# ---------------------------------------------------------------------------

def _run_prompt_forward(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Run a prompt forward pass + short generation for a single image.

    Returns dict with:
        lm_hidden_states – tuple of (1, prompt_len, H) per layer
        token_spans      – dict from build_token_spans
        gen_step_logits  – (1, T, V) logits per generated step (post-thinking
                           for Qwen, so step 0 is the first real answer token)
        gen_ids          – (1, T)       (same trimming as gen_step_logits)
        raw_text         – decoded answer string (thinking block stripped)
    """
    from src.vlm import qwen_patching_max_new_tokens, strip_qwen_thinking_prefix

    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)
    input_ids = inputs["input_ids"][0]
    prompt_len = int(inputs["input_ids"].shape[1])

    # Prompt forward — gives us hidden states for all layers
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    lm_hidden_states = outputs.hidden_states

    # Short generation — gives us logits / predicted answer. Qwen3.5 emits
    # an empty '<think></think>' preamble before the real answer, so give it
    # extra budget for non-multichoice tasks.
    effective_max_new_tokens = qwen_patching_max_new_tokens(vlm_adapter, max_new_tokens)
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=effective_max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if vlm_adapter.family == "molmo":
        generate_kwargs["use_cache"] = False

    out = model.generate(**generate_kwargs)
    gen_ids_full = out.sequences[:, prompt_len:]
    gen_step_logits_full = (
        torch.stack(out.scores, dim=1) if len(out.scores) > 0
        else torch.empty((1, 0, model.config.vocab_size), device=device)
    )

    # For Qwen, trim the leading '<think>…</think>' preamble from gen_ids and
    # gen_step_logits so downstream consumers see step-0 = first real answer
    # token. This keeps `first_prob`/entropy metrics comparable across models.
    offset = 0
    if vlm_adapter.family == "qwen" and gen_ids_full.shape[1] > 0:
        token_list = gen_ids_full[0].tolist()
        THINK_END_ID = 248069
        if THINK_END_ID in token_list:
            end_idx = token_list.index(THINK_END_ID)
            offset = end_idx + 1
            while offset < len(token_list) and token_list[offset] in (271, 198):
                offset += 1

    gen_ids = gen_ids_full[:, offset:]
    gen_step_logits = gen_step_logits_full[:, offset:, :]

    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if vlm_adapter.family == "qwen":
        # Guard against pathological cases where the regex catches residual
        # think markup that didn't token-match (e.g. decoded whitespace).
        raw_text = strip_qwen_thinking_prefix(raw_text)

    # Token spans
    lm_hidden_len = int(lm_hidden_states[0].shape[1])
    token_spans = build_token_spans(
        model=model,
        processor=processor,
        vlm_adapter=vlm_adapter,
        input_ids=input_ids,
        prompt_len=prompt_len,
        lm_hidden_len=lm_hidden_len,
    )

    return {
        "lm_hidden_states": lm_hidden_states,
        "token_spans": token_spans,
        "gen_ids": gen_ids,
        "gen_step_logits": gen_step_logits,
        "raw_text": raw_text,
    }


# ---------------------------------------------------------------------------
# Feature extraction — mirrors src.uq_dataset_generation.core exactly
# ---------------------------------------------------------------------------

def _extract_features(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    token_spans: Dict[str, Any],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Extract last-token patching features at a single layer.

    Returns dict mapping feature name -> (1, H) tensor on CPU.
    """
    v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
    q0, q1 = token_spans["question_start"], token_spans["question_end"]
    fs0, fs1 = token_spans["fullspan_start"], token_spans["fullspan_end"]

    visual_last = extract_lm_last_k_tokens(lm_hidden_states, layer_idx, v0, v1, k=1)[:, 0, :]
    question_last = extract_lm_last_k_tokens(lm_hidden_states, layer_idx, q0, q1, k=1)[:, 0, :]
    fullspan_last = extract_lm_last_k_tokens(lm_hidden_states, layer_idx, fs0, fs1, k=1)[:, 0, :]

    return {
        f"lm_visual_lasttok_layer_{layer_idx}": visual_last.detach().cpu(),
        f"lm_question_lasttok_layer_{layer_idx}": question_last.detach().cpu(),
        f"lm_fullspan_lasttok_layer_{layer_idx}": fullspan_last.detach().cpu(),
    }


def _extract_visual_span(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    token_spans: Dict[str, Any],
    layer_idx: int,
) -> torch.Tensor:
    """Extract all visual token activations at a single layer.

    Returns a (V, H) tensor on CPU, where V is the number of visual tokens.
    Stored as lm_visual_span_layer_X.pt alongside the last-token features.
    """
    v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
    return lm_hidden_states[layer_idx][0, v0:v1, :].detach().cpu()


# ---------------------------------------------------------------------------
# Control spans
# ---------------------------------------------------------------------------

def _sample_control_span(
    available: List[int],
    length: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """Pick a contiguous control span of *length* from *available* positions.

    Falls back to a random contiguous slice of the prompt if the available pool
    is too small.
    """
    if length == 0:
        return (0, 0)

    # Try contiguous runs within available
    contiguous_starts: List[int] = []
    run_start = available[0] if available else 0
    run_len = 1
    for i in range(1, len(available)):
        if available[i] == available[i - 1] + 1:
            run_len += 1
        else:
            if run_len >= length:
                # Any start within [run_start, run_start + run_len - length] works
                for s in range(run_start, run_start + run_len - length + 1):
                    contiguous_starts.append(s)
            run_start = available[i]
            run_len = 1
    # Check last run
    if run_len >= length:
        for s in range(run_start, run_start + run_len - length + 1):
            contiguous_starts.append(s)

    if contiguous_starts:
        start = rng.choice(contiguous_starts)
        return (start, start + length)

    # Fallback: not enough contiguous positions — pick a random slice from
    # whatever is available (may be shorter than requested)
    if available:
        n = min(length, len(available))
        start_idx = rng.randint(0, len(available) - n)
        return (available[start_idx], available[start_idx + n - 1] + 1)

    return (0, 0)


def _build_control_positions(
    token_spans: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Tuple[int, int]]:
    """Build one control span per real span, each avoiding only its own span.

    Each control span is sampled from prompt positions that lie *outside* that
    span's own token range but are otherwise unconstrained:

      - ``control_visual``:  sampled from non-visual prompt positions
      - ``control_question``: sampled from non-question prompt positions
      - ``control_fullspan``: sampled from positions outside both visual and
        question spans (since fullspan already covers the full prompt, its
        control avoids the two semantically meaningful sub-spans)

    Control positions are fixed per sample (same RNG seed → same draw) and
    reused across all blur versions so that any difference in activations
    between versions at the control span is due to the image change, not
    different random draws.

    Returns a dict mapping control name → ``(start, end)`` tuple.
    """
    v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
    q0, q1 = token_spans["question_start"], token_spans["question_end"]
    prompt_len = token_spans["prompt_len"]
    all_positions = set(range(prompt_len))

    visual_set = set(range(v0, v1))
    question_set = set(range(q0, q1))

    avail_visual = sorted(all_positions - visual_set)
    avail_question = sorted(all_positions - question_set)
    avail_fullspan = sorted(all_positions - visual_set - question_set)

    return {
        "control_visual": _sample_control_span(avail_visual, v1 - v0, rng),
        "control_question": _sample_control_span(avail_question, q1 - q0, rng),
        "control_fullspan": _sample_control_span(avail_fullspan, len(avail_fullspan), rng),
    }


def _extract_control_features(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    control_positions: Dict[str, Tuple[int, int]],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Extract last-token control features (non-semantic positions) at a single layer."""
    cv0, cv1 = control_positions["control_visual"]
    cq0, cq1 = control_positions["control_question"]
    cf0, cf1 = control_positions["control_fullspan"]

    features: Dict[str, torch.Tensor] = {}

    if cv1 > cv0:
        features[f"control_visual_lasttok_layer_{layer_idx}"] = (
            extract_lm_last_k_tokens(lm_hidden_states, layer_idx, cv0, cv1, k=1)[:, 0, :].detach().cpu()
        )

    if cq1 > cq0:
        features[f"control_question_lasttok_layer_{layer_idx}"] = (
            extract_lm_last_k_tokens(lm_hidden_states, layer_idx, cq0, cq1, k=1)[:, 0, :].detach().cpu()
        )

    if cf1 > cf0:
        features[f"control_fullspan_lasttok_layer_{layer_idx}"] = (
            extract_lm_last_k_tokens(lm_hidden_states, layer_idx, cf0, cf1, k=1)[:, 0, :].detach().cpu()
        )

    return features


# ---------------------------------------------------------------------------
# Output distribution
# ---------------------------------------------------------------------------

def _compute_output_distribution(
    gen_step_logits: torch.Tensor,
    processor: Any,
) -> Dict[str, Any]:
    """Compute output distribution statistics at the first generated token.

    Scoring is handled separately by ``_score_prediction`` so this function
    only returns logit / entropy statistics.
    """
    if gen_step_logits.shape[1] == 0:
        return {
            "output_logits": torch.empty(0),
            "top1_token": "",
            "top1_probability": 0.0,
            "output_entropy": 0.0,
        }

    first_logits = gen_step_logits[0, 0, :]  # (V,)
    probs = F.softmax(first_logits.float(), dim=0)
    top1_prob, top1_idx = probs.max(dim=0)
    top1_token = processor.tokenizer.decode([int(top1_idx)])
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    return {
        "output_logits": first_logits.detach().cpu(),
        "top1_token": top1_token.strip(),
        "top1_probability": float(top1_prob.item()),
        "output_entropy": float(entropy),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _save_version_activations(
    version_dir: str,
    features: Dict[str, torch.Tensor],
    sample_id: str,
    version: str,
) -> None:
    """Save one .pt per feature containing the record dict from the spec."""
    os.makedirs(version_dir, exist_ok=True)
    for feat_name, tensor in features.items():
        record = {
            "sample_id": sample_id,
            "version": version,
            "feature_name": feat_name,
            "activations": tensor,  # (1, H)
        }
        torch.save(record, os.path.join(version_dir, f"{feat_name}.pt"))


def _save_sample_metadata(
    sample_dir: str,
    sample_id: str,
    token_spans: Dict[str, Any],
    control_positions: Dict[str, Tuple[int, int]],
) -> None:
    os.makedirs(sample_dir, exist_ok=True)
    serialisable = {
        k: v for k, v in token_spans.items()
        if isinstance(v, (int, float, str, list, tuple, bool))
    }
    metadata = {
        "sample_id": sample_id,
        "token_spans": serialisable,
        "control_positions": {k: list(v) for k, v in control_positions.items()},
    }
    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_checkpoint(path: str, completed_ids: set, num_rows: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"completed_ids": sorted(completed_ids), "num_output_rows": num_rows}, f)
    os.replace(tmp, path)


def _save_output_distributions(parquet_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    import pandas as pd
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_patching_activations(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_records: List[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    cfg: PatchingActivationConfig,
) -> None:
    """Collect and save per-span activations for all samples and versions.

    For each sample the sigma values / blurred images are read directly from
    the patching dataset records (produced by the calibration + build steps),
    so no sigma arguments are needed.

    Args:
        model, processor, device, vlm_adapter: loaded VLM.
        patching_records: sample dicts from the patching dataset JSON.
        output_dir: root output directory.
        model_name: directory-safe model name for output hierarchy.
        cfg: collection configuration.
    """
    layers = [cfg.middle_layer, cfg.middle_layer + 1, cfg.middle_layer + 2]
    versions = [VERSION_CLEAN] + VISUAL_SEVERITY_VERSIONS
    max_new_tokens = _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)
    rng = random.Random(cfg.seed)

    model_dir = os.path.join(output_dir, model_name)
    checkpoint_path = os.path.join(model_dir, "checkpoint.json")
    parquet_path = os.path.join(model_dir, "output_distributions.parquet")
    os.makedirs(model_dir, exist_ok=True)

    # Resume support
    ckpt = _load_checkpoint(checkpoint_path)
    completed_ids: set = set(ckpt["completed_ids"]) if ckpt else set()
    output_rows: List[Dict[str, Any]] = []
    if completed_ids and os.path.exists(parquet_path):
        import pandas as pd
        output_rows = pd.read_parquet(parquet_path).to_dict("records")
        print(f"[collect] Resuming: {len(completed_ids)} samples done, "
              f"{len(output_rows)} output-distribution rows loaded.")

    total = len(patching_records)
    first_sample_validated = len(completed_ids) > 0

    print(f"[collect] Starting activation collection")
    print(f"  Samples:  {total}")
    print(f"  Versions: {versions}")
    print(f"  Layers:   {layers}")
    print(f"  Features: lm_visual_lasttok, lm_question_lasttok, lm_fullspan_lasttok "
          f"(layers {layers}); lm_visual_span (layer {layers[0]} only)")
    print(f"  Output:   {model_dir}/")

    # Accumulators for summary stats
    version_entropy: Dict[str, List[float]] = {v: [] for v in versions}
    version_scores: Dict[str, List[float]] = {v: [] for v in versions}
    version_top1_prob: Dict[str, List[float]] = {v: [] for v in versions}

    t0 = time.time()
    for record in tqdm(patching_records, desc="collecting activations"):
        sample_id = str(record["sample_id"])
        if sample_id in completed_ids:
            continue

        prompt = _format_prompt(cfg.dataset_id, record["clean_question"])
        sample_dir = os.path.join(model_dir, sample_id)

        # Build version -> image path mapping
        version_image_paths: Dict[str, str] = {
            VERSION_CLEAN: record["clean_image_path"],
        }
        for pert in record.get("visual_perturbations", []):
            version_image_paths[f"visual_{pert['severity']}"] = pert["blurred_image_path"]

        # Control positions are computed once per sample (from the clean
        # version's token spans) and reused across all blur versions.
        control_positions: Optional[Dict[str, Tuple[int, int]]] = None
        metadata_saved = False

        for version in versions:
            img_path = version_image_paths.get(version)
            if img_path is None:
                continue

            image = Image.open(img_path).convert("RGB")

            fwd = _run_prompt_forward(
                model=model,
                processor=processor,
                device=device,
                vlm_adapter=vlm_adapter,
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            image.close()

            token_spans = fwd["token_spans"]

            # Validate shapes on first sample ever processed
            if not first_sample_validated:
                n_layers = len(fwd["lm_hidden_states"])
                hidden_dim = fwd["lm_hidden_states"][layers[0]].shape[-1]
                print(f"[collect] Model: {n_layers} LM layers, hidden_dim={hidden_dim}")
                for l in layers:
                    assert l < n_layers, (
                        f"Requested layer {l} but model only has {n_layers} layers "
                        f"(0..{n_layers-1}). Adjust --middle_layer."
                    )
                first_sample_validated = True

            # Compute control positions once per sample (from clean version)
            if control_positions is None:
                control_positions = _build_control_positions(token_spans, rng)

            # Save metadata once per sample
            if not metadata_saved:
                _save_sample_metadata(sample_dir, sample_id, token_spans, control_positions)
                metadata_saved = True

            # Extract last-token features + control features at each layer
            version_dir = os.path.join(sample_dir, version)
            all_features: Dict[str, torch.Tensor] = {}
            for layer_idx in layers:
                all_features.update(
                    _extract_features(fwd["lm_hidden_states"], token_spans, layer_idx)
                )
                all_features.update(
                    _extract_control_features(fwd["lm_hidden_states"], control_positions, layer_idx)
                )
            _save_version_activations(version_dir, all_features, sample_id, version)

            # Full visual span at middle layer only — (V, H) tensor
            visual_span = _extract_visual_span(
                fwd["lm_hidden_states"], token_spans, layers[0]
            )
            torch.save(
                {"sample_id": sample_id, "version": version,
                 "feature_name": f"lm_visual_span_layer_{layers[0]}",
                 "activations": visual_span},
                os.path.join(version_dir, f"lm_visual_span_layer_{layers[0]}.pt"),
            )

            # Output distribution (logit stats only — scoring is separate)
            out_dist = _compute_output_distribution(
                fwd["gen_step_logits"],
                processor,
            )
            torch.save(out_dist["output_logits"], os.path.join(version_dir, "output_logits.pt"))

            # Dataset-specific scoring
            score = _score_prediction(cfg.dataset_id, fwd["raw_text"], record)

            output_rows.append({
                "sample_id": sample_id,
                "version": version,
                "top1_token": out_dist["top1_token"],
                "top1_probability": out_dist["top1_probability"],
                "output_entropy": out_dist["output_entropy"],
                "score": score,
                "predicted_answer": fwd["raw_text"],
            })
            version_entropy[version].append(out_dist["output_entropy"])
            version_scores[version].append(score)
            version_top1_prob[version].append(out_dist["top1_probability"])

            if cfg.verbose:
                gt_display = record.get("correct_answer") or record.get("gt_key") or ""
                print(
                    f"  [{version}] Q: {prompt!r}  "
                    f"A: {fwd['raw_text']!r}  "
                    f"GT: {gt_display!r}  "
                    f"Score: {score:.2f}"
                )

            del fwd  # free GPU memory

        completed_ids.add(sample_id)

        # Periodic checkpoint + progress
        n_done = len(completed_ids)
        if n_done % cfg.checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, completed_ids, len(output_rows))
            _save_output_distributions(parquet_path, output_rows)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"[collect] {n_done}/{total} samples "
                  f"({elapsed:.0f}s, {rate:.1f} samples/s)")

    # Final save
    _save_output_distributions(parquet_path, output_rows)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    print(f"\n[collect] Finished {len(completed_ids)}/{total} samples in {elapsed:.1f}s")

    # Summary statistics
    print(f"\n{'Version':<20s} {'Mean Score':>10s} {'Mean Top-1 Prob':>16s} {'Mean Entropy':>14s} {'N':>6s}")
    print("-" * 70)
    for v in versions:
        if version_scores[v]:
            avg_score = sum(version_scores[v]) / len(version_scores[v])
            avg_top1 = sum(version_top1_prob[v]) / len(version_top1_prob[v])
            avg_ent = sum(version_entropy[v]) / len(version_entropy[v])
            print(f"{v:<20s} {avg_score:>10.4f} {avg_top1:>16.4f} {avg_ent:>14.4f} {len(version_scores[v]):>6d}")

    print(f"\nOutput saved to: {model_dir}/")
