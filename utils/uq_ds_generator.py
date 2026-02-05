"""
Supervised-UQ dataset generation.

Produces:
- X: [N, D] float32 tensor (concatenated features)
- y: [N] float32 tensor (0/1 correctness label)
- meta: python dict with identifiers, feature names, etc.

Saved as:
  <output_root>/<dataset_id>/<model_id>/<run_id>/
      supervision_dataset.pt
      metadata.json
"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from utils.extract_features import (
    extract_vision_features_mean_pool,
    extract_lm_features_mean_pool,
    extract_lm_last_k_tokens
)
from src.labeling.multiple_choice import score_multiple_choice
from src.labeling.vqa import score_vqa
from utils.image_resolver import resolve_image

# You should import your own function from wherever it lives:
# from uq.models.vlm.infer import predict_letter_and_logits_with_features


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class SupervisionGenConfig:
    dataset_id: str
    model_id: str
    output_root: str = "outputs/supervised_datasets"

    seed_offset: int = 42
    max_samples: Optional[int] = None          # None = all
    verbose: bool = False

    # Feature switches (kept explicit so you can ablate easily)
    use_vision_middle: bool = True
    use_vision_final: bool = True
    use_lm_visual_middle: bool = True
    use_lm_visual_final: bool = True
    use_lm_prompt_middle: bool = True
    use_lm_prompt_final: bool = True
    use_lm_answer_middle: bool = True
    use_lm_answer_final: bool = True
    use_lm_visual_middle_lasttoken: bool = True
    use_lm_visual_final_lasttoken: bool = True
    use_lm_prompt_middle_lasttoken: bool = True
    use_lm_prompt_final_lasttoken: bool = True
    use_lm_answer_middle_lasttoken: bool = True
    use_lm_answer_final_lasttoken: bool = True
    use_answer_prob_entropy_stats: bool = True


    # Layers: if None, "middle" uses n//2, "final" uses -1
    # (kept here for future flexibility)
    # You can ignore for now.
    force_middle_layer: Optional[int] = None


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _safe_float32_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu().float()


def _assert_valid_2d(features: torch.Tensor, name: str) -> None:
    assert features.dim() == 2, f"{name}: expected 2D, got {tuple(features.shape)}"
    assert not torch.isnan(features).any(), f"{name}: contains NaNs"
    assert not torch.isinf(features).any(), f"{name}: contains Infs"


def _run_id_from_config(cfg: SupervisionGenConfig) -> str:
    key = {
        "dataset_id": cfg.dataset_id,
        "model_id": cfg.model_id,
        "seed_offset": cfg.seed_offset,
        "max_samples": cfg.max_samples,
        "features": {
            "vision_middle": cfg.use_vision_middle,
            "vision_final": cfg.use_vision_final,
            "lm_visual_middle": cfg.use_lm_visual_middle,
            "lm_visual_final": cfg.use_lm_visual_final,
            "lm_prompt_middle": cfg.use_lm_prompt_middle,
            "lm_prompt_final": cfg.use_lm_prompt_final,
            "lm_answer_middle": cfg.use_lm_answer_middle,
            "lm_answer_final": cfg.use_lm_answer_final,
            "lm_visual_middle_lasttoken": cfg.use_lm_visual_middle_lasttoken,
            "lm_visual_final_lasttoken": cfg.use_lm_visual_final_lasttoken,
            "lm_prompt_middle_lasttoken": cfg.use_lm_prompt_middle_lasttoken,
            "lm_prompt_final_lasttoken": cfg.use_lm_prompt_final_lasttoken,
            "lm_answer_middle_lasttoken": cfg.use_lm_answer_middle_lasttoken,
            "lm_answer_final_lasttoken": cfg.use_lm_answer_final_lasttoken,
            "answer_prob_entropy_stats": cfg.use_answer_prob_entropy_stats,
        },
    }
    s = json.dumps(key, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:10]


def _make_output_dir(cfg: SupervisionGenConfig) -> str:
    run_id = _run_id_from_config(cfg)
    out_dir = os.path.join(cfg.output_root, cfg.dataset_id, cfg.model_id, f"run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"\nLoaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from sample {len(checkpoint['y_list'])}")
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def _save_checkpoint(
    checkpoint_path: str,
    X_parts: List[torch.Tensor],
    y_list: List[float],
    rows: List[Dict[str, Any]],
    feature_names: List[str],
    processed_indices: List[int],
) -> None:
    """
    Save intermediate checkpoint ATOMICALLY:
    - write to checkpoint_path + ".tmp"
    - fsync
    - os.replace to checkpoint_path (atomic on most filesystems)

    Guarantee: if saving fails, the previous checkpoint_path is unchanged.
    """
    tmp_path = checkpoint_path + ".tmp"

    payload = {
        "X_parts": X_parts,
        "y_list": y_list,
        "rows": rows,
        "feature_names": feature_names,
        "processed_indices": processed_indices,
    }

    # Write to temp file first
    torch.save(payload, tmp_path)

    # Ensure bytes hit disk before rename (extra safety on network FS)
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        # fsync might not be supported in some environments; rename still helps
        pass

    # Atomic replace: either old checkpoint stays, or new one fully replaces it
    os.replace(tmp_path, checkpoint_path)



# ---------------------------------------------------------------------
# Feature extraction wrappers (thin)
# ---------------------------------------------------------------------

def _extract_vision_mean_pool(vision_hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
    feats = extract_vision_features_mean_pool(vision_hidden_states, layer_idx)
    _assert_valid_2d(feats, f"vision_layer_{layer_idx}")
    return feats


def _extract_lm_mean_pool(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
    token_start: int,
    token_end: int,
    name: str,
) -> torch.Tensor:
    feats = extract_lm_features_mean_pool(lm_hidden_states, layer_idx, token_start, token_end)
    _assert_valid_2d(feats, name)
    return feats

def _extract_lm_last_token(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
    token_start: int,
    token_end: int,
    name: str,
) -> torch.Tensor:
    """
    Returns (B, H) last-token vector from span [token_start, token_end).
    """
    feats_3d = extract_lm_last_k_tokens(
        hidden_states=lm_hidden_states,
        layer_idx=layer_idx,
        token_start=token_start,
        token_end=token_end,
        k=1,
    )  # (B, 1, H)

    assert feats_3d.dim() == 3 and feats_3d.shape[1] == 1, f"{name}: expected (B,1,H), got {tuple(feats_3d.shape)}"
    feats = feats_3d[:, 0, :]  # (B, H)

    _assert_valid_2d(feats, name)
    return feats

def _extract_gen_prob_entropy_stats(
    gen_step_logits: torch.Tensor,  # (B, T, V)
    gen_ids: torch.Tensor,          # (B, T)
    name_prefix: str,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute probability + entropy summary stats over the generated answer tokens.

    Returns:
      feats: (B, 10) tensor
      names: 10 feature names (matching order)
    """
    assert gen_step_logits.dim() == 3, f"{name_prefix}: expected (B,T,V)"
    assert gen_ids.dim() == 2, f"{name_prefix}: expected (B,T)"
    B, T, V = gen_step_logits.shape
    assert gen_ids.shape[0] == B and gen_ids.shape[1] == T, f"{name_prefix}: shape mismatch"

    probs = F.softmax(gen_step_logits, dim=-1)                   # (B, T, V)
    token_probs = probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # Negative probs/log-probs like the paper (monotonic transforms; consistent with their signs)
    neg_p = -token_probs
    neg_logp = -torch.log(token_probs + 1e-10)

    # probability stats (6)
    p_max = neg_p.max(dim=1).values
    p_min = neg_p.min(dim=1).values
    p_mean = neg_p.mean(dim=1)
    p_std = neg_p.std(dim=1) if T > 1 else torch.zeros_like(p_mean)

    logp_mean = neg_logp.mean(dim=1)
    logp_std = neg_logp.std(dim=1) if T > 1 else torch.zeros_like(logp_mean)

    # entropy stats (4)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)   # (B, T)
    e_max = entropy.max(dim=1).values
    e_min = entropy.min(dim=1).values
    e_mean = entropy.mean(dim=1)
    e_std = entropy.std(dim=1) if T > 1 else torch.zeros_like(e_mean)

    feats = torch.stack(
        [p_max, p_min, p_mean, p_std, logp_mean, logp_std, e_max, e_min, e_mean, e_std],
        dim=1,
    )  # (B, 10)

    names = [
        f"{name_prefix}_negp_max",
        f"{name_prefix}_negp_min",
        f"{name_prefix}_negp_mean",
        f"{name_prefix}_negp_std",
        f"{name_prefix}_neglogp_mean",
        f"{name_prefix}_neglogp_std",
        f"{name_prefix}_entropy_max",
        f"{name_prefix}_entropy_min",
        f"{name_prefix}_entropy_mean",
        f"{name_prefix}_entropy_std",
    ]
    return feats, names



def _get_middle_idx(n_layers: int, forced: Optional[int] = None) -> int:
    if forced is not None:
        return forced
    return n_layers // 2


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

@torch.no_grad()
def generate_supervised_uq_dataset(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    samples: List[Dict[str, Any]],
    cfg: SupervisionGenConfig,
    predict_fn,
    checkpoint_every: int = 50,  # Save every N samples
) -> Dict[str, Any]:
    """
    Generate supervised-UQ dataset for correctness prediction with checkpointing.

    Returns a dict:
      {
        "X": FloatTensor [N, D],
        "y": FloatTensor [N],
        "feature_names": List[str],
        "rows": List[Dict]   # optional per-sample metadata (small)
        "metadata": Dict
      }

    Also saves to disk under cfg.output_root with intermediate checkpoints.
    """

    out_dir = _make_output_dir(cfg)
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")

    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]

    print(f"\nGenerating supervised UQ dataset")
    print(f"  dataset_id: {cfg.dataset_id}")
    print(f"  model_id:   {cfg.model_id}")
    print(f"  samples:    {len(samples)}")
    print(f"  output:     {out_dir}")
    print(f"  checkpoint: every {checkpoint_every} samples")
    print(f"  verbose:    {cfg.verbose}")

    # Try to load checkpoint
    checkpoint = _load_checkpoint(checkpoint_path)
    
    if checkpoint is not None:
        X_parts = checkpoint["X_parts"]
        y_list = checkpoint["y_list"]
        rows = checkpoint["rows"]
        feature_names = checkpoint["feature_names"]
        processed_indices = set(checkpoint["processed_indices"])
        feature_dim_known = len(feature_names) > 0
        start_idx = len(y_list)
        print(f"  Resuming from sample {start_idx}/{len(samples)}")
        for r in rows:
            if "gt_answers" not in r:
                r["gt_answers"] = None
    else:
        X_parts = []
        y_list = []
        rows = []
        feature_names = []
        processed_indices = set()
        feature_dim_known = False
        start_idx = 0
        print(f"  Starting fresh")

    # Update progress bar only every 1% to reduce I/O overhead
    total_samples = len(samples)

    processed_count = len(processed_indices)
    last_printed_pct = int(100 * processed_count / total_samples) if total_samples > 0 else 100

    if checkpoint is not None:
        print(f"[{last_printed_pct:3d}%] Resuming at {processed_count}/{total_samples}")
    else:
        print(f"[{last_printed_pct:3d}%] Starting at {processed_count}/{total_samples}")

    # Update at most every ~1% OR every 30s (whichever is less frequent)
    miniters = max(1, total_samples // 100)   # ~1% steps

    iterator = tqdm(
        range(total_samples),
        desc="Generating",
        total=total_samples,
        miniters=miniters,
        mininterval=30.0,
        smoothing=0.1,
        dynamic_ncols=True,
        disable=cfg.verbose,
        leave=True,
    )


    for sample_idx in iterator:
        if sample_idx in processed_indices:
            continue

        sample = samples[sample_idx]
        image = resolve_image(sample)

        prediction_output = predict_fn(
            model=model,
            processor=processor,
            device=device,
            image=image,
            prompt=sample["prompt"],
        )
        
        if len(prediction_output) == 10:  # Classification (A/B/C/D)
            (
                pred_letter,
                option_probs,
                option_logits,
                raw_text,
                vision_hidden_states,
                lm_hidden_states,
                token_spans,
                answer_hidden_states,
                gen_ids,
                gen_step_logits,
            ) = prediction_output

            corr = score_multiple_choice(
                pred_letter=pred_letter,
                gt_letter=sample.get("gt_letter"),
                normalize=True,
            )

        elif len(prediction_output) == 7:  # Open-ended VQA
            pred_answer, vision_hidden_states, lm_hidden_states, token_spans, answer_hidden_states, gen_ids, gen_step_logits = prediction_output

            corr = score_vqa(
                pred_answer=pred_answer,
                gt_answers=sample.get("gt_answers"),
                gt_answer_single=sample.get("gt_answer")
            )

            option_probs = None
            option_logits = None
            raw_text = pred_answer
            pred_letter = None

        else:
            raise ValueError(f"Unexpected predict_fn output length: {len(prediction_output)}")

        y_list.append(float(corr.score))

        # --- collect feature tensors for THIS sample (then concat) ---
        per_sample_feats: List[torch.Tensor] = []
        per_sample_names: List[str] = []

        # Vision features
        if vision_hidden_states is not None and (cfg.use_vision_middle or cfg.use_vision_final):
            n_v = len(vision_hidden_states)
            mid_v = _get_middle_idx(n_v, cfg.force_middle_layer)

            if cfg.use_vision_middle:
                f = _extract_vision_mean_pool(vision_hidden_states, mid_v)
                per_sample_feats.append(f)
                per_sample_names.append(f"vision_mean_layer_{mid_v}")

            if cfg.use_vision_final:
                f = _extract_vision_mean_pool(vision_hidden_states, -1)
                per_sample_feats.append(f)
                per_sample_names.append("vision_mean_layer_-1")

        # LM features for visual/prompt spans
        if lm_hidden_states is not None and token_spans:
            n_lm = len(lm_hidden_states)
            mid_lm = _get_middle_idx(n_lm, cfg.force_middle_layer)

            v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
            p0, p1 = token_spans["text_post_start"], token_spans["text_post_end"]

            if cfg.use_lm_visual_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, v0, v1, name=f"lm_mid_visual_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_visual_mean_layer_{mid_lm}")

            if cfg.use_lm_visual_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, v0, v1, name="lm_final_visual_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_visual_mean_layer_-1")

            if cfg.use_lm_prompt_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, p0, p1, name=f"lm_mid_prompt_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_prompt_mean_layer_{mid_lm}")

            if cfg.use_lm_prompt_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, p0, p1, name="lm_final_prompt_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_prompt_mean_layer_-1")

            if cfg.use_lm_visual_middle_lasttoken:
                f = _extract_lm_last_token(
                    lm_hidden_states, mid_lm, v0, v1, name=f"lm_mid_visual_lasttok_layer_{mid_lm}"
                )
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_visual_lasttok_layer_{mid_lm}")

            if cfg.use_lm_visual_final_lasttoken:
                f = _extract_lm_last_token(
                    lm_hidden_states, -1, v0, v1, name="lm_final_visual_lasttok_-1"
                )
                per_sample_feats.append(f)
                per_sample_names.append("lm_visual_lasttok_layer_-1")

            if cfg.use_lm_prompt_middle_lasttoken:
                f = _extract_lm_last_token(
                    lm_hidden_states, mid_lm, p0, p1, name=f"lm_mid_prompt_lasttok_layer_{mid_lm}"
                )
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_prompt_lasttok_layer_{mid_lm}")

            if cfg.use_lm_prompt_final_lasttoken:
                f = _extract_lm_last_token(
                    lm_hidden_states, -1, p0, p1, name="lm_final_prompt_lasttok_-1"
                )
                per_sample_feats.append(f)
                per_sample_names.append("lm_prompt_lasttok_layer_-1")


        # Answer features (using answer_hidden_states + answer span)
        if answer_hidden_states is not None and token_spans and (
            cfg.use_lm_answer_middle
            or cfg.use_lm_answer_final
            or cfg.use_lm_answer_middle_lasttoken
            or cfg.use_lm_answer_final_lasttoken
        ):
            n_ans = len(answer_hidden_states)
            mid_ans = _get_middle_idx(n_ans, cfg.force_middle_layer)

            a0, a1 = token_spans["answer_start"], token_spans["answer_end"]

            if cfg.use_lm_answer_middle:
                f = _extract_lm_mean_pool(answer_hidden_states, mid_ans, a0, a1, name=f"lm_mid_answer_{mid_ans}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_answer_mean_layer_{mid_ans}")

            if cfg.use_lm_answer_final:
                f = _extract_lm_mean_pool(answer_hidden_states, -1, a0, a1, name="lm_final_answer_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_answer_mean_layer_-1")

            if cfg.use_lm_answer_middle_lasttoken:
                f = _extract_lm_last_token(
                    answer_hidden_states, mid_ans, a0, a1, name=f"lm_mid_answer_lasttok_layer_{mid_ans}"
                )
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_answer_lasttok_layer_{mid_ans}")

            if cfg.use_lm_answer_final_lasttoken:
                f = _extract_lm_last_token(
                    answer_hidden_states, -1, a0, a1, name="lm_final_answer_lasttok_-1"
                )
                per_sample_feats.append(f)
                per_sample_names.append("lm_answer_lasttok_layer_-1")
            
        if cfg.use_answer_prob_entropy_stats:
            if gen_step_logits is None or gen_ids is None or gen_ids.shape[1] == 0:
                # optionally skip or fill zeros
                pass
            else:
                stats, stat_names = _extract_gen_prob_entropy_stats(
                    gen_step_logits=gen_step_logits,
                    gen_ids=gen_ids,
                    name_prefix="answer_gen",
                )
            _assert_valid_2d(stats, "answer_gen_prob_entropy_stats")
            per_sample_feats.append(stats)
            per_sample_names.extend(stat_names)


            

        if len(per_sample_feats) == 0:
            raise RuntimeError("No features were extracted for this sample. Check cfg feature flags.")

        # Each feature extractor returns [B, D]; here B should be 1.
        # Concatenate along feature dimension -> [1, D_total]
        per_x = torch.cat(per_sample_feats, dim=1).cpu().float()

        if not feature_dim_known:
            feature_names = per_sample_names[:]  # fixed ordering
            feature_dim_known = True
        else:
            # Sanity: ensure consistent feature order across samples
            if per_sample_names != feature_names:
                raise RuntimeError(
                    "Feature name/order mismatch. This suggests different feature config.\n"
                    f"Expected: {feature_names}\nGot:      {per_sample_names}\n"
                    "Delete checkpoint to start fresh with new features."
                )

        X_parts.append(per_x)
        processed_indices.add(sample_idx)

        processed_count += 1
        pct = int(100 * processed_count / total_samples)

        if pct > last_printed_pct:
            last_printed_pct = pct
            tqdm.write(
                f"[{pct:3d}%] processed={processed_count}/{total_samples} | "
                f"elapsed={iterator.format_dict['elapsed']:.0f}s | "
                f"eta={iterator.format_dict.get('remaining', float('nan')):.0f}s"
            )


        # keep small metadata row (don't store huge tensors here)
        rows.append(
            {
                "idx": sample.get("idx"),
                "gt_letter": sample.get("gt_letter"),
                "gt_answers": sample.get("gt_answers"),
                "pred_letter": pred_letter,
                "gt_class": sample.get("gt_class"),
                "raw_text": raw_text,
                "option_probs": option_probs,
                "option_logits": option_logits,
                "token_spans": token_spans,
            }
        )

        # Save checkpoint periodically
        if len(y_list) % checkpoint_every == 0:
            try:
                _save_checkpoint(
                    checkpoint_path,
                    X_parts,
                    y_list,
                    rows,
                    feature_names,
                    list(processed_indices),
                )
            except Exception as e:
                # Don't crash the whole run; keep going.
                # You can also print free disk space here if you want.
                print(f"\nWarning: checkpoint save failed at {len(y_list)} samples: {e}")

        if cfg.verbose:
            print("\n--- sample ---")
            print(f"idx: {sample.get('idx')}, gt: {sample.get('gt_letter')} pred: {pred_letter}")
            print(raw_text)

    # Stack to [N, D]
    X = torch.cat(X_parts, dim=0)
    y = torch.tensor(y_list, dtype=torch.float32)

    metadata = {
        "dataset_id": cfg.dataset_id,
        "model_id": cfg.model_id,
        "num_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "feature_names": feature_names,
        "seed_offset": cfg.seed_offset,
        "max_samples": cfg.max_samples,
        "label_mean": float(y.mean().item()),
    }

    # Save final artifacts
    pt_path = os.path.join(out_dir, "supervision_dataset.pt")
    torch.save(
        {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "rows": rows,
            "metadata": metadata,
        },
        pt_path,
    )

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\nRemoved checkpoint file (processing complete)")

    print(f"\nSaved supervised dataset:")
    print(f"  {pt_path}")
    print(f"  N={X.shape[0]}  D={X.shape[1]}  label_mean={metadata['label_mean']:.3f}")

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "rows": rows,
        "metadata": metadata,
        "output_dir": out_dir,
        "pt_path": pt_path,
    }
