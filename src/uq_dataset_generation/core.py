"""shared generation loop + feature extraction + checkpointing + saving"""

from __future__ import annotations

import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.extract_features import (
    extract_vision_features_mean_pool,
    extract_lm_features_mean_pool,
    extract_lm_last_k_tokens,
    extract_lm_features_mean_pool_multi,
    extract_lm_last_token_multi,
)
from utils.image_resolver import resolve_image

from .config import SupervisionGenConfig
from .registry import get_task


# ----------------------------
# Small utilities
# ----------------------------

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
            "lm_question_middle": cfg.use_lm_question_middle,
            "lm_question_final": cfg.use_lm_question_final,
            "lm_answer_middle": cfg.use_lm_answer_middle,
            "lm_answer_final": cfg.use_lm_answer_final,
            "lm_visual_middle_lasttoken": cfg.use_lm_visual_middle_lasttoken,
            "lm_visual_final_lasttoken": cfg.use_lm_visual_final_lasttoken,
            "lm_question_middle_lasttoken": cfg.use_lm_question_middle_lasttoken,
            "lm_question_final_lasttoken": cfg.use_lm_question_final_lasttoken,
            "lm_answer_middle_lasttoken": cfg.use_lm_answer_middle_lasttoken,
            "lm_answer_final_lasttoken": cfg.use_lm_answer_final_lasttoken,
            "answer_prob_entropy_stats": cfg.use_answer_prob_entropy_stats,
            "lm_fullspan_middle": cfg.use_lm_fullspan_middle,
            "lm_fullspan_final": cfg.use_lm_fullspan_final,
            "lm_fullspan_middle_lasttoken": cfg.use_lm_fullspan_middle_lasttoken,
            "lm_fullspan_final_lasttoken": cfg.use_lm_fullspan_final_lasttoken,
            "lm_question_including_middle": cfg.use_lm_question_including_middle,
            "lm_question_including_final": cfg.use_lm_question_including_final,
            "lm_question_including_middle_lasttoken": cfg.use_lm_question_including_middle_lasttoken,
            "lm_question_including_final_lasttoken": cfg.use_lm_question_including_final_lasttoken,
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
    tmp_path = checkpoint_path + ".tmp"
    payload = {
        "X_parts": X_parts,
        "y_list": y_list,
        "rows": rows,
        "feature_names": feature_names,
        "processed_indices": processed_indices,
    }
    torch.save(payload, tmp_path)
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp_path, checkpoint_path)


def _get_middle_idx(n_layers: int, forced: Optional[int] = None) -> int:
    return forced if forced is not None else (n_layers // 2)


# ----------------------------
# Feature wrappers
# ----------------------------

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
    feats_3d = extract_lm_last_k_tokens(
        hidden_states=lm_hidden_states,
        layer_idx=layer_idx,
        token_start=token_start,
        token_end=token_end,
        k=1,
    )
    assert feats_3d.dim() == 3 and feats_3d.shape[1] == 1, f"{name}: expected (B,1,H), got {tuple(feats_3d.shape)}"
    feats = feats_3d[:, 0, :]
    _assert_valid_2d(feats, name)
    return feats

def _extract_lm_mean_pool_multi(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
    spans: List[Tuple[int, int]],
    name: str,
) -> torch.Tensor:
    feats = extract_lm_features_mean_pool_multi(lm_hidden_states, layer_idx, spans)
    _assert_valid_2d(feats, name)
    return feats

def _extract_lm_last_token_multi(
    lm_hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
    spans: List[Tuple[int, int]],
    name: str,
) -> torch.Tensor:
    feats = extract_lm_last_token_multi(lm_hidden_states, layer_idx, spans)
    _assert_valid_2d(feats, name)
    return feats


def _extract_gen_prob_entropy_stats(
    gen_step_logits: torch.Tensor,  # (B, T, V)
    gen_ids: torch.Tensor,          # (B, T)
    name_prefix: str,
) -> Tuple[torch.Tensor, List[str]]:
    probs = F.softmax(gen_step_logits, dim=-1)
    token_probs = probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)

    neg_p = -token_probs
    neg_logp = -torch.log(token_probs + 1e-10)

    p_max = neg_p.max(dim=1).values
    p_min = neg_p.min(dim=1).values
    p_mean = neg_p.mean(dim=1)
    p_std = neg_p.std(dim=1) if gen_ids.shape[1] > 1 else torch.zeros_like(p_mean)

    logp_mean = neg_logp.mean(dim=1)
    logp_std = neg_logp.std(dim=1) if gen_ids.shape[1] > 1 else torch.zeros_like(logp_mean)

    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    e_max = entropy.max(dim=1).values
    e_min = entropy.min(dim=1).values
    e_mean = entropy.mean(dim=1)
    e_std = entropy.std(dim=1) if gen_ids.shape[1] > 1 else torch.zeros_like(e_mean)

    feats = torch.stack([p_max, p_min, p_mean, p_std, logp_mean, logp_std, e_max, e_min, e_mean, e_std], dim=1)
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


# ----------------------------
# Main generator
# ----------------------------

@torch.no_grad()
def generate_supervised_uq_dataset(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    samples: List[Dict[str, Any]],
    cfg: SupervisionGenConfig,
    predict_fn,
    checkpoint_every: int = 50,
) -> Dict[str, Any]:
    task = get_task(cfg.dataset_id)

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

    checkpoint = _load_checkpoint(checkpoint_path)

    if checkpoint is not None:
        X_parts = checkpoint["X_parts"]
        y_list = checkpoint["y_list"]
        rows = checkpoint["rows"]
        feature_names = checkpoint["feature_names"]
        processed_indices = set(checkpoint["processed_indices"])
        feature_dim_known = len(feature_names) > 0
        print(f"  Resuming from sample {len(y_list)}/{len(samples)}")
    else:
        X_parts: List[torch.Tensor] = []
        y_list: List[float] = []
        rows: List[Dict[str, Any]] = []
        feature_names: List[str] = []
        processed_indices = set()
        feature_dim_known = False
        print("  Starting fresh")

    total_samples = len(samples)
    miniters = max(1, total_samples // 100) if total_samples > 0 else 1

    iterator = tqdm(
        range(total_samples),
        desc="Generating",
        total=total_samples,
        miniters=miniters,
        mininterval=30.0,
        smoothing=0.1,
        dynamic_ncols=True,
        disable=False,
        leave=True,
    )

    for sample_idx in iterator:
        if sample_idx in processed_indices:
            continue

        sample = samples[sample_idx]
        image = resolve_image(sample)

        # dataset-specific prediction kwargs (e.g., option_keys later)
        pred_kwargs = task.predict_kwargs(sample)

        if "verbose" not in pred_kwargs:
            pred_kwargs["verbose"] = cfg.verbose    

        prediction_output = predict_fn(
            model=model,
            processor=processor,
            device=device,
            image=image,
            **pred_kwargs,
        )

        parsed = task.interpret_output(sample=sample, prediction_output=prediction_output)
        y_list.append(float(parsed["score"]))

        vision_hidden_states = parsed["vision_hidden_states"]
        lm_hidden_states = parsed["lm_hidden_states"]
        token_spans = parsed["token_spans"]
        answer_hidden_states = parsed["answer_hidden_states"]
        gen_ids = parsed["gen_ids"]
        gen_step_logits = parsed["gen_step_logits"]

        # -------- features --------
        per_sample_feats: List[torch.Tensor] = []
        per_sample_names: List[str] = []

        # Vision
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

        # LM question/visual spans
        if lm_hidden_states is not None and token_spans:
            n_lm = len(lm_hidden_states)
            mid_lm = _get_middle_idx(n_lm, cfg.force_middle_layer)

            v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
            q0, q1 = token_spans["question_start"], token_spans["question_end"]
            fs0, fs1 = token_spans["fullspan_start"], token_spans["fullspan_end"]
            qin_spans = token_spans["question_including_spans"]

            if cfg.use_lm_visual_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, v0, v1, name=f"lm_mid_visual_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_visual_mean_layer_{mid_lm}")

            if cfg.use_lm_visual_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, v0, v1, name="lm_final_visual_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_visual_mean_layer_-1")

            if cfg.use_lm_question_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, q0, q1, name=f"lm_mid_question_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_question_mean_layer_{mid_lm}")

            if cfg.use_lm_question_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, q0, q1, name="lm_final_question_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_question_mean_layer_-1")

            if cfg.use_lm_visual_middle_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, mid_lm, v0, v1, name=f"lm_mid_visual_lasttok_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_visual_lasttok_layer_{mid_lm}")

            if cfg.use_lm_visual_final_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, -1, v0, v1, name="lm_final_visual_lasttok_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_visual_lasttok_layer_-1")

            if cfg.use_lm_question_middle_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, mid_lm, q0, q1, name=f"lm_mid_question_lasttok_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_question_lasttok_layer_{mid_lm}")

            if cfg.use_lm_question_final_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, -1, q0, q1, name="lm_final_question_lasttok_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_question_lasttok_layer_-1")
            # full span features
            if cfg.use_lm_fullspan_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, fs0, fs1, name=f"lm_mid_fullspan_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_fullspan_mean_layer_{mid_lm}")

            if cfg.use_lm_fullspan_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, fs0, fs1, name="lm_final_fullspan_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_fullspan_mean_layer_-1")

            if cfg.use_lm_fullspan_middle_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, mid_lm, fs0, fs1, name=f"lm_mid_fullspan_lasttok_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_fullspan_lasttok_layer_{mid_lm}")

            if cfg.use_lm_fullspan_final_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, -1, fs0, fs1, name="lm_final_fullspan_lasttok_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_fullspan_lasttok_layer_-1")

            if cfg.use_lm_question_including_middle:
                f = _extract_lm_mean_pool_multi(lm_hidden_states, mid_lm, qin_spans, name=f"lm_mid_question_including_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_question_including_mean_layer_{mid_lm}")

            if cfg.use_lm_question_including_final:
                f = _extract_lm_mean_pool_multi(lm_hidden_states, -1, qin_spans, name="lm_final_question_including_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_question_including_mean_layer_-1")

            if cfg.use_lm_question_including_middle_lasttoken:
                f = _extract_lm_last_token_multi(lm_hidden_states, mid_lm, qin_spans, name=f"lm_mid_question_including_lasttok_{mid_lm}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_question_including_lasttok_layer_{mid_lm}")

            if cfg.use_lm_question_including_final_lasttoken:
                f = _extract_lm_last_token_multi(lm_hidden_states, -1, qin_spans, name="lm_final_question_including_lasttok_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_question_including_lasttok_layer_-1")

        # Answer span features
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
                f = _extract_lm_last_token(answer_hidden_states, mid_ans, a0, a1, name=f"lm_mid_answer_lasttok_{mid_ans}")
                per_sample_feats.append(f)
                per_sample_names.append(f"lm_answer_lasttok_layer_{mid_ans}")

            if cfg.use_lm_answer_final_lasttoken:
                f = _extract_lm_last_token(answer_hidden_states, -1, a0, a1, name="lm_final_answer_lasttok_-1")
                per_sample_feats.append(f)
                per_sample_names.append("lm_answer_lasttok_layer_-1")

        # Answer prob/entropy stats
        if cfg.use_answer_prob_entropy_stats and gen_step_logits is not None and gen_ids is not None and gen_ids.shape[1] > 0:
            stats, stat_names = _extract_gen_prob_entropy_stats(gen_step_logits=gen_step_logits, gen_ids=gen_ids, name_prefix="answer_gen")
            _assert_valid_2d(stats, "answer_gen_prob_entropy_stats")
            per_sample_feats.append(stats)
            per_sample_names.extend(stat_names)

        if len(per_sample_feats) == 0:
            raise RuntimeError("No features were extracted for this sample. Check cfg feature flags.")

        per_x = torch.cat(per_sample_feats, dim=1).cpu().float()

        if not feature_dim_known:
            feature_names = per_sample_names[:]
            feature_dim_known = True
        else:
            if per_sample_names != feature_names:
                raise RuntimeError(
                    "Feature name/order mismatch.\n"
                    f"Expected: {feature_names}\nGot:      {per_sample_names}\n"
                    "Delete checkpoint to start fresh."
                )

        X_parts.append(per_x)
        processed_indices.add(sample_idx)
        rows.append(parsed["row"])

        if len(y_list) % checkpoint_every == 0:
            try:
                _save_checkpoint(checkpoint_path, X_parts, y_list, rows, feature_names, list(processed_indices))
            except Exception as e:
                print(f"\nWarning: checkpoint save failed at {len(y_list)} samples: {e}")

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

    pt_path = os.path.join(out_dir, "supervision_dataset.pt")
    torch.save({"X": X, "y": y, "feature_names": feature_names, "rows": rows, "metadata": metadata}, pt_path)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nRemoved checkpoint file (processing complete)")

    print(f"\nSaved supervised dataset:\n  {pt_path}\n  N={X.shape[0]}  D={X.shape[1]}  label_mean={metadata['label_mean']:.3f}")

    return {"X": X, "y": y, "feature_names": feature_names, "rows": rows, "metadata": metadata, "output_dir": out_dir, "pt_path": pt_path}
