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
from src.vlm import VLMAdapter, prepare_inputs


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
) -> Dict[str, Any]:
    """Run model.generate() while patching hidden states at one layer.

    Args:
        layer_idx: which decoder layer to intercept.
        positions: token positions to replace (list of ints).
        clean_activations: tensor of shape (len(positions), H) with the
            clean hidden states to inject.

    Returns dict with raw_text, gen_step_logits (for entropy / top-1 prob).
    """
    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)
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
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if vlm_adapter.family == "molmo":
            generate_kwargs["use_cache"] = False

        out = model.generate(**generate_kwargs)
    finally:
        handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    gen_step_logits = (
        torch.stack(out.scores, dim=1) if len(out.scores) > 0
        else torch.empty((1, 0, model.config.vocab_size), device=device)
    )
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return {
        "raw_text": raw_text,
        "gen_step_logits": gen_step_logits,
    }


# ---------------------------------------------------------------------------
# Output distribution helpers (reused from collect_activations pattern)
# ---------------------------------------------------------------------------

def _compute_output_stats(gen_step_logits: torch.Tensor) -> Dict[str, float]:
    """Compute top-1 probability and entropy from the first generated token."""
    if gen_step_logits.shape[1] == 0:
        return {"top1_probability": 0.0, "output_entropy": 0.0}

    first_logits = gen_step_logits[0, 0, :].float()
    probs = F.softmax(first_logits, dim=0)
    top1_prob = float(probs.max().item())
    entropy = float(-(probs * torch.log(probs + 1e-10)).sum().item())

    return {"top1_probability": top1_prob, "output_entropy": entropy}



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
) -> Dict[str, Any]:
    """Run a normal (unpatched) forward pass and return text + logits."""
    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[1])

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if vlm_adapter.family == "molmo":
        generate_kwargs["use_cache"] = False

    out = model.generate(**generate_kwargs)
    gen_ids = out.sequences[:, prompt_len:]
    gen_step_logits = (
        torch.stack(out.scores, dim=1) if len(out.scores) > 0
        else torch.empty((1, 0, model.config.vocab_size), device=device)
    )
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return {"raw_text": raw_text, "gen_step_logits": gen_step_logits}


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

        # --- Run clean baseline (from clean image) ---
        clean_image = Image.open(record["clean_image_path"]).convert("RGB")
        clean_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        clean_image.close()
        clean_stats = _compute_output_stats(clean_fwd["gen_step_logits"])
        clean_score = _score_prediction(cfg.dataset_id, clean_fwd["raw_text"], record)

        # --- For each severity level ---
        for pert in record.get("visual_perturbations", []):
            severity = pert["severity"]
            version = f"visual_{severity}"
            img_path = pert.get("blurred_image_path", "")
            if not img_path or not os.path.exists(img_path):
                continue

            blurred_image = Image.open(img_path).convert("RGB")

            # Degraded baseline (no patching)
            deg_fwd = _run_baseline_forward(
                model=model, processor=processor, device=device,
                vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            deg_stats = _compute_output_stats(deg_fwd["gen_step_logits"])
            deg_score = _score_prediction(cfg.dataset_id, deg_fwd["raw_text"], record)

            # Record baselines
            rows.append({
                "sample_id": sample_id,
                "severity": severity,
                "condition": "degraded",
                "predicted_answer": deg_fwd["raw_text"],
                "score": deg_score,
                "top1_probability": deg_stats["top1_probability"],
                "output_entropy": deg_stats["output_entropy"],
            })
            rows.append({
                "sample_id": sample_id,
                "severity": severity,
                "condition": "clean",
                "predicted_answer": clean_fwd["raw_text"],
                "score": clean_score,
                "top1_probability": clean_stats["top1_probability"],
                "output_entropy": clean_stats["output_entropy"],
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
                )
                patched_stats = _compute_output_stats(patched_fwd["gen_step_logits"])
                patched_score = _score_prediction(
                    cfg.dataset_id, patched_fwd["raw_text"], record,
                )

                rows.append({
                    "sample_id": sample_id,
                    "severity": severity,
                    "condition": cond.name,
                    "predicted_answer": patched_fwd["raw_text"],
                    "score": patched_score,
                    "top1_probability": patched_stats["top1_probability"],
                    "output_entropy": patched_stats["output_entropy"],
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
