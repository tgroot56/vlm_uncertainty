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

from utils.extract_features import (
    extract_vision_features_mean_pool,
    extract_lm_features_mean_pool,
)

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
        },
    }
    s = json.dumps(key, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:10]


def _make_output_dir(cfg: SupervisionGenConfig) -> str:
    run_id = _run_id_from_config(cfg)
    out_dir = os.path.join(cfg.output_root, cfg.dataset_id, cfg.model_id, f"run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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
    predict_fn,  # pass predict_letter_and_logits_with_features to avoid circular imports
) -> Dict[str, Any]:
    """
    Generate supervised-UQ dataset for correctness prediction.

    Returns a dict:
      {
        "X": FloatTensor [N, D],
        "y": FloatTensor [N],
        "feature_names": List[str],
        "rows": List[Dict]   # optional per-sample metadata (small)
        "metadata": Dict
      }

    Also saves to disk under cfg.output_root.
    """

    out_dir = _make_output_dir(cfg)

    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]

    print(f"\nGenerating supervised UQ dataset")
    print(f"  dataset_id: {cfg.dataset_id}")
    print(f"  model_id:   {cfg.model_id}")
    print(f"  samples:    {len(samples)}")
    print(f"  output:     {out_dir}")
    print(f"  verbose:    {cfg.verbose}")

    X_parts: List[torch.Tensor] = []
    y_list: List[float] = []
    rows: List[Dict[str, Any]] = []

    # We'll determine feature_names dynamically in the same order as concatenation.
    feature_names: List[str] = []
    feature_dim_known = False

    iterator = tqdm(samples, desc="Generating", disable=cfg.verbose)
    for sample in iterator:
        # --- run model inference + hidden state capture ---
        pred_letter, option_probs, option_logits, raw_text, vision_hidden_states, lm_hidden_states, token_spans, answer_hidden_states = predict_fn(
            model=model,
            processor=processor,
            device=device,
            image=sample["image"],
            prompt=sample["prompt"],
        )

        is_correct = (pred_letter == sample["gt_letter"])
        y_list.append(1.0 if is_correct else 0.0)

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
            p0, p1 = token_spans["prompt_start"], token_spans["prompt_end"]

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

        # Answer features (using answer_hidden_states + answer span)
        if answer_hidden_states is not None and token_spans and (cfg.use_lm_answer_middle or cfg.use_lm_answer_final):
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
                    "Feature name/order mismatch across samples.\n"
                    f"Expected: {feature_names}\nGot:      {per_sample_names}"
                )

        X_parts.append(per_x)

        # keep small metadata row (don’t store huge tensors here)
        rows.append(
            {
                "idx": sample.get("idx"),
                "gt_letter": sample.get("gt_letter"),
                "pred_letter": pred_letter,
                "is_correct": is_correct,
                "gt_class": sample.get("gt_class"),
                "raw_text": raw_text,
                "option_probs": option_probs,
                "option_logits": option_logits,
                "token_spans": token_spans,
            }
        )

        if cfg.verbose:
            print("\n--- sample ---")
            print(f"idx: {sample.get('idx')}, gt: {sample.get('gt_letter')} pred: {pred_letter} correct: {is_correct}")
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

    # Save artifacts
    pt_path = os.path.join(out_dir, "supervision_dataset.pt")
    torch.save(
        {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "rows": rows,          # can omit later if too large
            "metadata": metadata,
        },
        pt_path,
    )

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

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
