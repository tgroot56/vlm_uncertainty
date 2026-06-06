"""Activation patching execution pipeline.

For each degraded sample, reruns the VLM forward pass while replacing (patching)
hidden-state activations at targeted token spans and layers with pre-cached
clean activations.  Measures how this changes confidence (top-1 probability),
entropy, and accuracy.

Patching conditions (each = a separate forward pass):
  - *_lasttok_layer_X : replace a single token's activation at layer X
  - lm_visual_span_layer_16 : replace all V visual token activations at layer 16

Outputs:
  - Per-sample Parquet with one row per (sample, severity, condition)
  - Summary CSV with mean / std / N per (severity, condition)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.patching.collect_activations import (
    VERSION_CLEAN,
    VISUAL_SEVERITY_VERSIONS,
    _format_prompt,
    _score_prediction,
    _MAX_NEW_TOKENS,
)
from src.uq_dataset_generation.tasks.pope import (
    _NO_SURFACE_FORMS,
    _YES_SURFACE_FORMS,
    _collect_unique_token_ids,
    _extract_yes_no_answer,
)
from src.vlm import (
    VLMAdapter,
    prepare_inputs,
    prepare_qwen_fused_inputs_embeds,
    qwen_patching_max_new_tokens,
    qwen_vision_cache_context,
    strip_qwen_thinking_prefix,
)


_QWEN_THINK_END_ID = 248069   # '</think>'
_QWEN_DOUBLE_NEWLINE = 271    # '\n\n' token commonly emitted after </think>
_QWEN_SINGLE_NEWLINE = 198    # '\n'


def _qwen_post_generate(
    vlm_adapter: VLMAdapter,
    gen_ids: torch.Tensor,
    gen_step_logits: torch.Tensor,
    processor: Any,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """For Qwen, strip the leading '<think>…</think>\\n\\n' preamble from
    gen_ids / gen_step_logits / raw_text so downstream consumers see the
    real first answer token at position 0. No-op for other families."""
    if vlm_adapter.family == "qwen" and gen_ids.shape[1] > 0:
        token_list = gen_ids[0].tolist()
        if _QWEN_THINK_END_ID in token_list:
            offset = token_list.index(_QWEN_THINK_END_ID) + 1
            while offset < len(token_list) and token_list[offset] in (
                _QWEN_DOUBLE_NEWLINE,
                _QWEN_SINGLE_NEWLINE,
            ):
                offset += 1
            gen_ids = gen_ids[:, offset:]
            gen_step_logits = gen_step_logits[:, offset:, :]

    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if vlm_adapter.family == "qwen":
        raw_text = strip_qwen_thinking_prefix(raw_text)
    return gen_ids, gen_step_logits, raw_text


def _token_piece(processor: Any, token_id: int) -> str:
    return processor.tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _answer_token_indices(gen_ids: torch.Tensor, processor: Any) -> List[int]:
    """Indices of generated tokens that form the visible evaluated answer.

    Generation can begin with formatting tokens that disappear under decoding
    and ``strip()`` (e.g. LLaVA's leading SentencePiece whitespace token).
    We trim those from the confidence calculation, along with special tokens.
    Internal tokens are retained so multi-token answers keep their full
    length-normalized likelihood.
    """
    if gen_ids.ndim != 2 or gen_ids.shape[0] == 0:
        return []

    token_ids = [int(t) for t in gen_ids[0].tolist()]
    special_ids = set(getattr(processor.tokenizer, "all_special_ids", []) or [])
    keep = [idx for idx, token_id in enumerate(token_ids) if token_id not in special_ids]

    while keep and _token_piece(processor, token_ids[keep[0]]).strip() == "":
        keep.pop(0)
    while keep and _token_piece(processor, token_ids[keep[-1]]).strip() == "":
        keep.pop()

    return keep


def _stripped_answer_token_ids(gen_ids: torch.Tensor, processor: Any) -> List[int]:
    """Generated token IDs that make up the stripped evaluated answer."""
    if gen_ids is None or not isinstance(gen_ids, torch.Tensor):
        return []
    token_ids = [int(t) for t in gen_ids[0].tolist()] if gen_ids.ndim == 2 and gen_ids.shape[0] else []
    return [
        token_ids[idx]
        for idx in _answer_token_indices(gen_ids, processor)
        if 0 <= idx < len(token_ids)
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PatchingExecutionConfig:
    """Configuration for an activation-patching run."""
    dataset_id: str = "vqa-v2"
    middle_layer: int = 16
    max_new_tokens: int = 4
    checkpoint_every: int = 50
    verbose: bool = False


# ---------------------------------------------------------------------------
# Resolving decoder layer modules
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: Any, vlm_adapter: VLMAdapter) -> torch.nn.ModuleList:
    """Return the ModuleList of decoder layers for the given VLM.

    Tries a set of known attribute paths first, then falls back to scanning
    named_modules() for a ModuleList large enough to be the decoder stack.
    """
    # Known paths: (attr, ...) chains to walk from model root
    candidates = [
        ("language_model", "layers"),            # LLaVA (llava-1.5-7b)
        ("language_model", "model", "layers"),   # LLaVA variants
        ("model", "layers"),                     # Qwen / generic CausalLM
        ("model", "transformer", "blocks"),      # Molmo
        ("transformer", "h"),                    # GPT-2 style
    ]
    for path in candidates:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and isinstance(node, torch.nn.ModuleList) and len(node) > 0:
            return node

    # Fallback: scan named_modules for the largest ModuleList (= decoder stack)
    best: torch.nn.ModuleList | None = None
    best_name = ""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > (len(best) if best is not None else 0):
            best = mod
            best_name = name
    if best is not None and len(best) >= 2:
        return best

    # Give a useful diagnostic before raising
    top_attrs = [k for k, v in model.__dict__.items() if not k.startswith("_")]
    raise RuntimeError(
        f"Cannot resolve decoder layers for VLM family '{vlm_adapter.family}'. "
        f"Top-level model attributes: {top_attrs}"
    )


# ---------------------------------------------------------------------------
# Hook-based patched generation
# ---------------------------------------------------------------------------

def _patched_generate(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    layer_idx: int,
    positions: List[int],
    clean_activations: torch.Tensor,
    fused_bundle: Optional[Dict[str, torch.Tensor]] = None,
    qwen_prefill_empty_thinking: bool = False,
) -> Dict[str, Any]:
    """Run model.generate() while patching hidden states at one layer.

    Args:
        layer_idx: which decoder layer to intercept.
        positions: token positions to replace (list of ints).
        clean_activations: tensor of shape (len(positions), H) with the
            clean hidden states to inject.
        fused_bundle: optional pre-computed fused inputs (from
            ``prepare_qwen_fused_inputs_embeds``). When provided, we call
            ``generate(inputs_embeds=...)`` directly and skip the vision
            tower. Only supported for Qwen — the caller is responsible for
            supplying this only when ``vlm_adapter.family == "qwen"``.
            Gives ~30-40x speedup on Qwen patching by avoiding the vision
            tower rerun on every condition.

    Returns dict with raw_text, gen_ids, gen_step_logits.
    """
    if fused_bundle is not None:
        inputs = fused_bundle["inputs"]
        prompt_len = int(fused_bundle["prompt_len"])
    else:
        inputs = prepare_inputs(
            vlm_adapter,
            processor,
            image,
            prompt,
            device,
            qwen_prefill_empty_thinking=qwen_prefill_empty_thinking,
        )
        prompt_len = int(inputs["input_ids"].shape[1])

    decoder_layers = _get_decoder_layers(model, vlm_adapter)
    target_layer = decoder_layers[layer_idx]

    clean_act = clean_activations.to(device=device, dtype=next(model.parameters()).dtype)
    patched = False

    def _hook(module, args, output):
        nonlocal patched
        # Only patch once (during prefill, not during cached decoding steps)
        if patched:
            return output

        # output is typically a tuple: (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        seq_len = hidden_states.shape[1]
        # Only patch when seq_len matches the full prompt (prefill step).
        # During cached generation steps seq_len == 1.
        if seq_len < prompt_len:
            return output

        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()
        for i, pos in enumerate(positions):
            if pos < seq_len:
                hidden_states[0, pos, :] = clean_act[i]

        patched = True

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    handle = target_layer.register_forward_hook(_hook)
    try:
        effective_max_new_tokens = (
            max_new_tokens
            if vlm_adapter.family == "qwen" and qwen_prefill_empty_thinking
            else qwen_patching_max_new_tokens(vlm_adapter, max_new_tokens)
        )
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if vlm_adapter.family == "molmo":
            generate_kwargs["use_cache"] = False

        if fused_bundle is not None:
            with qwen_vision_cache_context(model, fused_bundle):
                out = model.generate(**generate_kwargs)
        else:
            out = model.generate(**generate_kwargs)
    finally:
        handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    gen_step_logits = (
        torch.stack(out.scores, dim=1) if len(out.scores) > 0
        else torch.empty((1, 0, model.config.vocab_size), device=device)
    )
    gen_ids, gen_step_logits, raw_text = _qwen_post_generate(
        vlm_adapter, gen_ids, gen_step_logits, processor
    )

    return {
        "raw_text": raw_text,
        "gen_ids": gen_ids,
        "gen_step_logits": gen_step_logits,
    }


# ---------------------------------------------------------------------------
# Output distribution helpers (reused from collect_activations pattern)
# ---------------------------------------------------------------------------

def _compute_output_stats(
    gen_step_logits: torch.Tensor,
    gen_ids: Optional[torch.Tensor] = None,
    processor: Any = None,
) -> Dict[str, Any]:
    """Compute output confidence from the visible answer tokens.

    ``top1_probability`` is intentionally the answer-level confidence used by
    plots: the geometric mean of probabilities assigned to generated tokens
    that survive decoding/postprocessing. The older first-generated-token
    quantity is retained as ``first_generated_token_top1_probability``.
    """
    if gen_step_logits.shape[1] == 0:
        return {
            "top1_probability": 0.0,
            "output_entropy": 0.0,
            "answer_geom_mean_probability": 0.0,
            "answer_mean_logprob": 0.0,
            "answer_min_token_probability": 0.0,
            "answer_first_token_probability": 0.0,
            "answer_token_count": 0,
            "answer_token_ids": "[]",
            "answer_token_text": "",
            "first_generated_token_top1_probability": 0.0,
            "first_generated_token_entropy": 0.0,
        }

    first_logits = gen_step_logits[0, 0, :].float()
    first_probs = F.softmax(first_logits, dim=0)
    first_top1_prob = float(first_probs.max().item())
    first_entropy = float(-(first_probs * torch.log(first_probs + 1e-10)).sum().item())

    if gen_ids is None or processor is None:
        return {
            "top1_probability": first_top1_prob,
            "output_entropy": first_entropy,
            "answer_geom_mean_probability": first_top1_prob,
            "answer_mean_logprob": float(torch.log(first_probs.max() + 1e-10).item()),
            "answer_min_token_probability": first_top1_prob,
            "answer_first_token_probability": first_top1_prob,
            "answer_token_count": 1,
            "answer_token_ids": "[]",
            "answer_token_text": "",
            "first_generated_token_top1_probability": first_top1_prob,
            "first_generated_token_entropy": first_entropy,
        }

    answer_indices = [
        idx for idx in _answer_token_indices(gen_ids, processor)
        if idx < gen_step_logits.shape[1]
    ]
    if not answer_indices:
        answer_indices = [0]

    token_probs: List[float] = []
    token_logprobs: List[float] = []
    token_entropies: List[float] = []
    token_ids: List[int] = []
    token_texts: List[str] = []
    for idx in answer_indices:
        token_id = int(gen_ids[0, idx].item())
        probs = F.softmax(gen_step_logits[0, idx, :].float(), dim=0)
        prob = float(probs[token_id].item())
        token_probs.append(prob)
        token_logprobs.append(float(torch.log(probs[token_id] + 1e-10).item()))
        token_entropies.append(float(-(probs * torch.log(probs + 1e-10)).sum().item()))
        token_ids.append(token_id)
        token_texts.append(_token_piece(processor, token_id))

    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    geom_mean_prob = float(torch.exp(torch.tensor(mean_logprob)).item())

    return {
        "top1_probability": geom_mean_prob,
        "output_entropy": token_entropies[0],
        "answer_geom_mean_probability": geom_mean_prob,
        "answer_mean_logprob": mean_logprob,
        "answer_min_token_probability": min(token_probs),
        "answer_first_token_probability": token_probs[0],
        "answer_token_count": len(token_ids),
        "answer_token_ids": json.dumps(token_ids),
        "answer_token_text": json.dumps(token_texts),
        "first_generated_token_top1_probability": first_top1_prob,
        "first_generated_token_entropy": first_entropy,
    }


def _token_prob_entry(
    *,
    probs: torch.Tensor,
    log_probs: torch.Tensor,
    token_ids: List[int],
    processor: Any,
    prefix: str,
) -> Dict[str, Any]:
    """Return max-probability stats over tokenized answer surface variants."""
    if not token_ids:
        return {
            f"{prefix}_probability": float("nan"),
            f"{prefix}_logprob": float("nan"),
            f"{prefix}_token_id": None,
            f"{prefix}_token_text": "",
            f"{prefix}_candidate_token_ids": "[]",
        }

    device = probs.device
    idx = torch.tensor(token_ids, device=device, dtype=torch.long)
    candidate_probs = probs[idx]
    best_pos = int(torch.argmax(candidate_probs).item())
    token_id = int(token_ids[best_pos])
    return {
        f"{prefix}_probability": float(candidate_probs[best_pos].item()),
        f"{prefix}_logprob": float(log_probs[token_id].item()),
        f"{prefix}_token_id": token_id,
        f"{prefix}_token_text": _token_piece(processor, token_id),
        f"{prefix}_candidate_token_ids": json.dumps(token_ids),
    }


def _compute_pope_answer_probability_stats(
    gen_step_logits: torch.Tensor,
    processor: Any,
    clean_answer: Optional[str],
) -> Dict[str, Any]:
    """POPE yes/no probabilities from the first real answer-token distribution.

    Uses the same model-tokenizer surface variants as POPE logit scoring
    (e.g. "yes", " yes", casing variants). For each answer bucket we retain
    the highest-probability token variant, matching the existing max-logit
    yes/no convention while saving probabilities/logprobs for later log-odds.
    """
    clean_bucket = _extract_yes_no_answer(clean_answer or "").lower()
    base = {
        "clean_answer_text": clean_answer or "",
        "clean_answer_bucket": clean_bucket if clean_bucket in {"yes", "no"} else "",
        "clean_answer_probability": float("nan"),
        "clean_answer_logprob": float("nan"),
        "clean_answer_token_id": None,
        "clean_answer_token_text": "",
    }
    if (
        gen_step_logits is None
        or processor is None
        or getattr(processor, "tokenizer", None) is None
        or not isinstance(gen_step_logits, torch.Tensor)
        or gen_step_logits.ndim < 3
        or gen_step_logits.shape[1] == 0
    ):
        return {
            **base,
            "pope_yes_probability": float("nan"),
            "pope_yes_logprob": float("nan"),
            "pope_yes_token_id": None,
            "pope_yes_token_text": "",
            "pope_yes_candidate_token_ids": "[]",
            "pope_no_probability": float("nan"),
            "pope_no_logprob": float("nan"),
            "pope_no_token_id": None,
            "pope_no_token_text": "",
            "pope_no_candidate_token_ids": "[]",
        }

    tokenizer = processor.tokenizer
    yes_ids = _collect_unique_token_ids(tokenizer, _YES_SURFACE_FORMS)
    no_ids = _collect_unique_token_ids(tokenizer, _NO_SURFACE_FORMS)
    first_logits = gen_step_logits[0, 0, :].float()
    probs = F.softmax(first_logits, dim=0)
    log_probs = F.log_softmax(first_logits, dim=0)

    yes_stats = _token_prob_entry(
        probs=probs,
        log_probs=log_probs,
        token_ids=yes_ids,
        processor=processor,
        prefix="pope_yes",
    )
    no_stats = _token_prob_entry(
        probs=probs,
        log_probs=log_probs,
        token_ids=no_ids,
        processor=processor,
        prefix="pope_no",
    )

    if clean_bucket == "yes":
        base.update({
            "clean_answer_probability": yes_stats["pope_yes_probability"],
            "clean_answer_logprob": yes_stats["pope_yes_logprob"],
            "clean_answer_token_id": yes_stats["pope_yes_token_id"],
            "clean_answer_token_text": yes_stats["pope_yes_token_text"],
        })
    elif clean_bucket == "no":
        base.update({
            "clean_answer_probability": no_stats["pope_no_probability"],
            "clean_answer_logprob": no_stats["pope_no_logprob"],
            "clean_answer_token_id": no_stats["pope_no_token_id"],
            "clean_answer_token_text": no_stats["pope_no_token_text"],
        })

    return {**base, **yes_stats, **no_stats}


def _compute_clean_answer_probability_stats(
    gen_step_logits: torch.Tensor,
    gen_ids: Optional[torch.Tensor],
    processor: Any,
    *,
    clean_answer_text: str,
    clean_answer_token_ids: List[int],
) -> Dict[str, Any]:
    """Probability assigned to the clean-run stripped answer tokens.

    The clean answer tokens are taken from the clean run after the same
    generated-answer stripping used for scoring. For another run, we align
    those clean tokens to that run's stripped generated-answer positions and
    report the geometric mean probability when all clean tokens fit.
    """
    token_texts = [_token_piece(processor, token_id) for token_id in clean_answer_token_ids]
    base = {
        "clean_answer_text": clean_answer_text,
        "clean_answer_token_ids": json.dumps([int(t) for t in clean_answer_token_ids]),
        "clean_answer_token_text": json.dumps(token_texts),
        "clean_answer_token_count": len(clean_answer_token_ids),
        "clean_answer_probability": float("nan"),
        "clean_answer_logprob": float("nan"),
        "clean_answer_probability_token_count": 0,
        "clean_answer_probability_full_length": False,
    }
    if (
        gen_step_logits is None
        or gen_ids is None
        or processor is None
        or not clean_answer_token_ids
        or not isinstance(gen_step_logits, torch.Tensor)
        or not isinstance(gen_ids, torch.Tensor)
        or gen_step_logits.ndim < 3
        or gen_step_logits.shape[1] == 0
    ):
        return base

    answer_indices = [
        idx for idx in _answer_token_indices(gen_ids, processor)
        if idx < gen_step_logits.shape[1]
    ]
    if len(answer_indices) < len(clean_answer_token_ids):
        base["clean_answer_probability_token_count"] = len(answer_indices)
        return base

    logprobs: List[float] = []
    vocab_size = int(gen_step_logits.shape[-1])
    for step_idx, token_id in zip(answer_indices, clean_answer_token_ids):
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            return base
        log_probs = F.log_softmax(gen_step_logits[0, step_idx, :].float(), dim=0)
        logprobs.append(float(log_probs[token_id].item()))

    mean_logprob = float(sum(logprobs) / len(logprobs))
    base.update({
        "clean_answer_probability": float(torch.exp(torch.tensor(mean_logprob)).item()),
        "clean_answer_logprob": mean_logprob,
        "clean_answer_probability_token_count": len(logprobs),
        "clean_answer_probability_full_length": True,
    })
    return base


def _compute_stripped_answer_softmax_stats(
    gen_step_logits: torch.Tensor,
    gen_ids: Optional[torch.Tensor],
    processor: Any,
    *,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Serialize softmax distributions for stripped generated-answer tokens.

    The binary column is row-major bytes with shape recorded separately as
    ``[answer_token_count, vocab_size]``. This keeps the parquet self-contained
    while avoiding enormous nested Python float lists.
    """
    dtype = dtype.lower()
    if dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported stripped-answer softmax dtype: {dtype}")

    vocab_size = (
        int(gen_step_logits.shape[-1])
        if isinstance(gen_step_logits, torch.Tensor) and gen_step_logits.ndim >= 3
        else 0
    )
    base = {
        "stripped_answer_token_indices": "[]",
        "stripped_answer_token_ids": "[]",
        "stripped_answer_token_text": "[]",
        "stripped_answer_token_count": 0,
        "stripped_answer_softmax_shape": json.dumps([0, vocab_size]),
        "stripped_answer_softmax_dtype": dtype,
        "stripped_answer_softmax_encoding": "row_major_bytes",
        "stripped_answer_softmax_num_bytes": 0,
        "stripped_answer_softmax_bytes": b"",
    }
    if (
        gen_step_logits is None
        or gen_ids is None
        or processor is None
        or not isinstance(gen_step_logits, torch.Tensor)
        or not isinstance(gen_ids, torch.Tensor)
        or gen_step_logits.ndim < 3
        or gen_step_logits.shape[1] == 0
    ):
        return base

    answer_indices = [
        idx for idx in _answer_token_indices(gen_ids, processor)
        if idx < gen_step_logits.shape[1]
    ]
    token_ids = [int(gen_ids[0, idx].item()) for idx in answer_indices]
    token_texts = [_token_piece(processor, token_id) for token_id in token_ids]
    if not answer_indices:
        base.update({
            "stripped_answer_token_indices": "[]",
            "stripped_answer_token_ids": "[]",
            "stripped_answer_token_text": "[]",
        })
        return base

    logits = gen_step_logits[0, answer_indices, :].float()
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    probs = probs.astype(np.float16 if dtype == "float16" else np.float32, copy=False)
    payload = probs.tobytes(order="C")
    return {
        "stripped_answer_token_indices": json.dumps([int(i) for i in answer_indices]),
        "stripped_answer_token_ids": json.dumps(token_ids),
        "stripped_answer_token_text": json.dumps(token_texts),
        "stripped_answer_token_count": len(token_ids),
        "stripped_answer_softmax_shape": json.dumps([len(token_ids), vocab_size]),
        "stripped_answer_softmax_dtype": dtype,
        "stripped_answer_softmax_encoding": "row_major_bytes",
        "stripped_answer_softmax_num_bytes": len(payload),
        "stripped_answer_softmax_bytes": payload,
    }



# ---------------------------------------------------------------------------
# Loading cached activations for a sample
# ---------------------------------------------------------------------------

def _load_cached_activations(
    activations_dir: str,
    sample_id: str,
    version: str,
) -> Dict[str, torch.Tensor]:
    """Load all .pt activation files for a (sample, version) pair.

    Returns a dict mapping feature_name -> activations tensor.
    """
    version_dir = os.path.join(activations_dir, sample_id, version)
    result: Dict[str, torch.Tensor] = {}
    if not os.path.isdir(version_dir):
        return result
    for fname in os.listdir(version_dir):
        if not fname.endswith(".pt") or fname == "output_logits.pt":
            continue
        data = torch.load(os.path.join(version_dir, fname), map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "feature_name" in data:
            result[data["feature_name"]] = data["activations"]
        else:
            # Bare tensor (e.g. output_logits)
            result[fname.replace(".pt", "")] = data
    return result


def _load_sample_metadata(
    activations_dir: str,
    sample_id: str,
) -> Dict[str, Any]:
    """Load the token-span metadata for a sample."""
    path = os.path.join(activations_dir, sample_id, "metadata.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build patching conditions from cached features
# ---------------------------------------------------------------------------

@dataclass
class PatchingCondition:
    """One patching condition to evaluate."""
    name: str            # e.g. "lm_visual_lasttok_layer_16"
    layer_idx: int       # decoder layer to hook
    positions: List[int] # token positions to replace
    activations: torch.Tensor  # (len(positions), H)


def _build_patching_conditions(
    clean_features: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    middle_layer: int,
) -> List[PatchingCondition]:
    """Build the list of patching conditions from cached clean activations.

    Returns one condition per available feature. Last-token features yield
    single-position patches; the visual span yields a full-span patch.
    """
    spans = metadata["token_spans"]
    layers = [middle_layer, middle_layer + 1, middle_layer + 2]
    conditions: List[PatchingCondition] = []

    for layer_idx in layers:
        # --- last-token conditions ---
        # Visual last token
        key = f"lm_visual_lasttok_layer_{layer_idx}"
        if key in clean_features:
            v_end = spans["visual_end"]
            pos = v_end - 1  # last token of the visual span
            act = clean_features[key]
            if act.ndim == 2:
                act = act[0]  # (1, H) -> (H,)
            conditions.append(PatchingCondition(
                name=key, layer_idx=layer_idx,
                positions=[pos], activations=act.unsqueeze(0),
            ))

        # Question last token
        key = f"lm_question_lasttok_layer_{layer_idx}"
        if key in clean_features:
            q_end = spans["question_end"]
            pos = q_end - 1
            act = clean_features[key]
            if act.ndim == 2:
                act = act[0]
            conditions.append(PatchingCondition(
                name=key, layer_idx=layer_idx,
                positions=[pos], activations=act.unsqueeze(0),
            ))

        # Fullspan last token
        key = f"lm_fullspan_lasttok_layer_{layer_idx}"
        if key in clean_features:
            fs_end = spans["fullspan_end"]
            pos = fs_end - 1
            act = clean_features[key]
            if act.ndim == 2:
                act = act[0]
            conditions.append(PatchingCondition(
                name=key, layer_idx=layer_idx,
                positions=[pos], activations=act.unsqueeze(0),
            ))

    # --- full visual span (middle layer only) ---
    span_key = f"lm_visual_span_layer_{middle_layer}"
    if span_key in clean_features:
        v_start = spans["visual_start"]
        v_end = spans["visual_end"]
        positions = list(range(v_start, v_end))
        act = clean_features[span_key]  # (V, H)
        conditions.append(PatchingCondition(
            name=span_key, layer_idx=middle_layer,
            positions=positions, activations=act,
        ))

    return conditions


# ---------------------------------------------------------------------------
# Baseline (unpatched) forward pass
# ---------------------------------------------------------------------------

def _run_baseline_forward(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    fused_bundle: Optional[Dict[str, torch.Tensor]] = None,
    qwen_prefill_empty_thinking: bool = False,
) -> Dict[str, Any]:
    """Run a normal (unpatched) forward pass and return text + logits.

    If ``fused_bundle`` is supplied (Qwen only), we skip the vision tower by
    reusing pre-computed fused ``inputs_embeds``.
    """
    if fused_bundle is not None:
        inputs = fused_bundle["inputs"]
        prompt_len = int(fused_bundle["prompt_len"])
    else:
        inputs = prepare_inputs(
            vlm_adapter,
            processor,
            image,
            prompt,
            device,
            qwen_prefill_empty_thinking=qwen_prefill_empty_thinking,
        )
        prompt_len = int(inputs["input_ids"].shape[1])

    effective_max_new_tokens = (
        max_new_tokens
        if vlm_adapter.family == "qwen" and qwen_prefill_empty_thinking
        else qwen_patching_max_new_tokens(vlm_adapter, max_new_tokens)
    )
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=effective_max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if vlm_adapter.family == "molmo":
        generate_kwargs["use_cache"] = False

    if fused_bundle is not None:
        with qwen_vision_cache_context(model, fused_bundle):
            out = model.generate(**generate_kwargs)
    else:
        out = model.generate(**generate_kwargs)

    gen_ids = out.sequences[:, prompt_len:]
    gen_step_logits = (
        torch.stack(out.scores, dim=1) if len(out.scores) > 0
        else torch.empty((1, 0, model.config.vocab_size), device=device)
    )
    gen_ids, gen_step_logits, raw_text = _qwen_post_generate(
        vlm_adapter, gen_ids, gen_step_logits, processor
    )

    return {
        "raw_text": raw_text,
        "gen_ids": gen_ids,
        "gen_step_logits": gen_step_logits,
    }


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_activation_patching(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_records: List[Dict[str, Any]],
    activations_dir: str,
    output_dir: str,
    cfg: PatchingExecutionConfig,
) -> None:
    """Execute activation patching for all samples and save results.

    Args:
        model, processor, device, vlm_adapter: loaded VLM.
        patching_records: sample dicts from the patching dataset JSON.
        activations_dir: directory containing cached activations
            (the model_name/ subdirectory from collect_patching_activations).
        output_dir: where to write result Parquet + summary CSV.
        cfg: execution configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_new_tokens = _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)

    # Checkpoint support
    checkpoint_path = os.path.join(output_dir, "patching_checkpoint.json")
    completed_ids: set = set()
    rows: List[Dict[str, Any]] = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_ids = set(ckpt.get("completed_ids", []))
        parquet_path = os.path.join(output_dir, "patching_results.parquet")
        if completed_ids and os.path.exists(parquet_path):
            import pandas as pd
            rows = pd.read_parquet(parquet_path).to_dict("records")
        print(f"[patch] Resuming: {len(completed_ids)} samples done.")

    total = len(patching_records)
    severities = [v for v in VISUAL_SEVERITY_VERSIONS]

    print(f"[patch] Starting activation patching")
    print(f"  Samples:    {total}")
    print(f"  Severities: {severities}")
    print(f"  Output:     {output_dir}/")

    t0 = time.time()
    for record in tqdm(patching_records, desc="patching"):
        sample_id = str(record["sample_id"])
        if sample_id in completed_ids:
            continue

        prompt = _format_prompt(cfg.dataset_id, record["clean_question"])

        # Load cached clean activations + metadata
        clean_features = _load_cached_activations(activations_dir, sample_id, VERSION_CLEAN)
        if not clean_features:
            if cfg.verbose:
                print(f"  [skip] {sample_id}: no clean activations found")
            continue

        metadata = _load_sample_metadata(activations_dir, sample_id)
        conditions = _build_patching_conditions(clean_features, metadata, cfg.middle_layer)

        if not conditions:
            if cfg.verbose:
                print(f"  [skip] {sample_id}: no patching conditions built")
            continue

        # --- Qwen fast path: pre-compute fused inputs_embeds per image so the
        # vision tower runs once instead of per-condition (30-40x speedup).
        # For other VLMs, keep the original path — they have cheap vision towers.
        use_fused = vlm_adapter.family == "qwen"

        # --- Run clean baseline (from clean image) ---
        clean_image = Image.open(record["clean_image_path"]).convert("RGB")
        clean_fused = (
            prepare_qwen_fused_inputs_embeds(model, vlm_adapter, processor, clean_image, prompt, device)
            if use_fused else None
        )
        clean_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            max_new_tokens=max_new_tokens, fused_bundle=clean_fused,
        )
        clean_image.close()
        clean_stats = _compute_output_stats(
            clean_fwd["gen_step_logits"], clean_fwd.get("gen_ids"), processor
        )
        clean_score = _score_prediction(cfg.dataset_id, clean_fwd["raw_text"], record)

        # --- For each severity level ---
        for pert in record.get("visual_perturbations", []):
            severity = pert["severity"]
            version = f"visual_{severity}"
            img_path = pert.get("blurred_image_path", "")
            if not img_path or not os.path.exists(img_path):
                continue

            blurred_image = Image.open(img_path).convert("RGB")
            blurred_fused = (
                prepare_qwen_fused_inputs_embeds(model, vlm_adapter, processor, blurred_image, prompt, device)
                if use_fused else None
            )

            # Degraded baseline (no patching)
            deg_fwd = _run_baseline_forward(
                model=model, processor=processor, device=device,
                vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
                max_new_tokens=max_new_tokens, fused_bundle=blurred_fused,
            )
            deg_stats = _compute_output_stats(
                deg_fwd["gen_step_logits"], deg_fwd.get("gen_ids"), processor
            )
            deg_score = _score_prediction(cfg.dataset_id, deg_fwd["raw_text"], record)

            # Record baselines
            rows.append({
                "sample_id": sample_id,
                "severity": severity,
                "condition": "degraded",
                "predicted_answer": deg_fwd["raw_text"],
                "score": deg_score,
                **deg_stats,
            })
            rows.append({
                "sample_id": sample_id,
                "severity": severity,
                "condition": "clean",
                "predicted_answer": clean_fwd["raw_text"],
                "score": clean_score,
                **clean_stats,
            })

            # --- Run each patching condition ---
            for cond in conditions:
                patched_fwd = _patched_generate(
                    model=model, processor=processor, device=device,
                    vlm_adapter=vlm_adapter, image=blurred_image,
                    prompt=prompt, max_new_tokens=max_new_tokens,
                    layer_idx=cond.layer_idx,
                    positions=cond.positions,
                    clean_activations=cond.activations,
                    fused_bundle=blurred_fused,
                )
                patched_stats = _compute_output_stats(
                    patched_fwd["gen_step_logits"], patched_fwd.get("gen_ids"), processor
                )
                patched_score = _score_prediction(
                    cfg.dataset_id, patched_fwd["raw_text"], record,
                )

                rows.append({
                    "sample_id": sample_id,
                    "severity": severity,
                    "condition": cond.name,
                    "predicted_answer": patched_fwd["raw_text"],
                    "score": patched_score,
                    **patched_stats,
                })

                if cfg.verbose:
                    print(
                        f"  [{severity}/{cond.name}] "
                        f"deg={deg_stats['top1_probability']:.3f} "
                        f"→ patched={patched_stats['top1_probability']:.3f} "
                        f"(clean={clean_stats['top1_probability']:.3f}, "
                        f"recovery={recovery_top1:.3f})"
                    )

            blurred_image.close()

        completed_ids.add(sample_id)

        # Periodic checkpoint
        if len(completed_ids) % cfg.checkpoint_every == 0:
            _save_results(rows, output_dir, checkpoint_path, completed_ids)
            elapsed = time.time() - t0
            rate = len(completed_ids) / elapsed if elapsed > 0 else 0
            print(f"[patch] {len(completed_ids)}/{total} samples "
                  f"({elapsed:.0f}s, {rate:.1f} samples/s)")

    # Final save
    _save_results(rows, output_dir, checkpoint_path=None, completed_ids=completed_ids)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    print(f"\n[patch] Finished {len(completed_ids)}/{total} samples in {elapsed:.1f}s")

    # Print summary
    _print_summary(rows, output_dir)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def _save_results(
    rows: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_path: Optional[str],
    completed_ids: set,
) -> None:
    """Save per-sample Parquet and optionally update checkpoint."""
    import pandas as pd

    parquet_path = os.path.join(output_dir, "patching_results.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    if checkpoint_path is not None:
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"completed_ids": sorted(completed_ids)}, f)
        os.replace(tmp, checkpoint_path)


def _print_summary(rows: List[Dict[str, Any]], output_dir: str) -> None:
    """Compute and save aggregated summary statistics.

    Recovery is computed at the aggregate level to avoid per-sample
    denominator instability:
        recovery = (mean_patched - mean_degraded) / (mean_clean - mean_degraded + epsilon)
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print("[patch] No results to summarize.")
        return

    epsilon = 1e-8

    # Aggregate raw metrics per (severity, condition)
    agg = df.groupby(["severity", "condition"]).agg(
        mean_score=("score", "mean"),
        mean_top1_prob=("top1_probability", "mean"),
        mean_entropy=("output_entropy", "mean"),
        N=("sample_id", "count"),
    ).reset_index()

    # Look up the aggregate degraded and clean baselines per severity
    deg = agg[agg["condition"] == "degraded"].set_index("severity")
    cln = agg[agg["condition"] == "clean"].set_index("severity")

    def _agg_recovery(row: pd.Series, metric: str) -> float:
        sev = row["severity"]
        if sev not in deg.index or sev not in cln.index:
            return float("nan")
        mean_degraded = deg.loc[sev, metric]
        mean_clean = cln.loc[sev, metric]
        return (row[metric] - mean_degraded) / (mean_clean - mean_degraded + epsilon)

    agg["recovery_top1"] = agg.apply(lambda r: _agg_recovery(r, "mean_top1_prob"), axis=1)
    agg["recovery_entropy"] = agg.apply(lambda r: _agg_recovery(r, "mean_entropy"), axis=1)
    agg["recovery_score"] = agg.apply(lambda r: _agg_recovery(r, "mean_score"), axis=1)

    summary_path = os.path.join(output_dir, "patching_summary.csv")
    agg.to_csv(summary_path, index=False)
    print(f"\n[patch] Summary saved to {summary_path}")

    # Print to console
    print(f"\n{'Severity':<12s} {'Condition':<35s} {'Score':>7s} {'Top-1':>7s} "
          f"{'Entropy':>8s} {'Rec(Score)':>10s} {'Rec(T1)':>8s} {'Rec(Ent)':>9s} {'N':>5s}")
    print("-" * 110)
    for _, row in agg.iterrows():
        print(
            f"{row['severity']:<12s} {row['condition']:<35s} "
            f"{row['mean_score']:>7.4f} {row['mean_top1_prob']:>7.4f} "
            f"{row['mean_entropy']:>8.4f} {row['recovery_score']:>10.3f} "
            f"{row['recovery_top1']:>8.3f} {row['recovery_entropy']:>9.3f} "
            f"{int(row['N']):>5d}"
        )

    print(f"\nPer-sample results: {os.path.join(output_dir, 'patching_results.parquet')}")
