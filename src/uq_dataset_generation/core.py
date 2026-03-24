"""shared generation loop + feature extraction + checkpointing + saving"""

from __future__ import annotations

import os
import json
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.vlm import (
    extract_qwen_selected_vision_tensors_batch,
    extract_vision_features_bundle,
    prepare_inputs,
    prepare_qwen_visual_only_inputs_batch,
)
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
        "dataset_split": cfg.dataset_split,
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
            "lm_visual_all_layers_mean": cfg.use_lm_visual_all_layers_mean,
            "lm_question_all_layers_mean": cfg.use_lm_question_all_layers_mean,
            "lm_answer_all_layers_mean": cfg.use_lm_answer_all_layers_mean,
        },
    }
    s = json.dumps(key, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:10]


def _make_output_dir(cfg: SupervisionGenConfig) -> str:
    run_id = _run_id_from_config(cfg)
    path_parts = [cfg.output_root, cfg.dataset_id]
    if cfg.dataset_split:
        path_parts.append(cfg.dataset_split)
    path_parts.extend([cfg.model_id, f"run_{run_id}"])
    out_dir = os.path.join(*path_parts)
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
    feature_dims: List[int],
    processed_indices: List[int],
) -> None:
    tmp_path = checkpoint_path + ".tmp"
    payload = {
        "X_parts": X_parts,
        "y_list": y_list,
        "rows": rows,
        "feature_names": feature_names,
        "feature_dims": feature_dims,
        "processed_indices": processed_indices,
    }
    torch.save(payload, tmp_path)
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp_path, checkpoint_path)


def _load_visual_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"\nLoaded visual checkpoint from {checkpoint_path}")
        print(f"  Resuming from sample {len(checkpoint['processed_indices'])}")
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load visual checkpoint: {e}")
        return None


def _save_visual_checkpoint(
    checkpoint_path: str,
    X_parts: List[torch.Tensor],
    feature_names: List[str],
    feature_dims: List[int],
    processed_indices: List[int],
) -> None:
    tmp_path = checkpoint_path + ".tmp"
    payload = {
        "X_parts": X_parts,
        "feature_names": feature_names,
        "feature_dims": feature_dims,
        "processed_indices": processed_indices,
    }
    torch.save(payload, tmp_path)
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp_path, checkpoint_path)


def _load_source_supervision_dataset(
    *,
    cfg: SupervisionGenConfig,
    source_supervision_path: Optional[str],
    expected_num_samples: int,
) -> Dict[str, Any]:
    out_dir = _make_output_dir(cfg)
    source_path = source_supervision_path or os.path.join(out_dir, "supervision_dataset.pt")

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            "Vision-only extraction requires an existing supervision_dataset.pt "
            f"to reuse labels/rows, but none was found at: {source_path}"
        )

    source = torch.load(source_path, weights_only=False, map_location="cpu")

    y = source.get("y")
    rows = source.get("rows")
    if y is None or rows is None:
        raise RuntimeError(
            f"Expected 'y' and 'rows' in source supervision dataset at {source_path}, "
            f"found keys: {sorted(source.keys())}"
        )

    if len(y) < expected_num_samples or len(rows) < expected_num_samples:
        raise RuntimeError(
            "Source supervision dataset is smaller than the requested visual-only run. "
            f"source_y={len(y)} source_rows={len(rows)} expected={expected_num_samples}"
        )

    if len(y) != expected_num_samples or len(rows) != expected_num_samples:
        print(
            "Warning: source supervision dataset size does not exactly match the "
            f"requested run (source_y={len(y)}, source_rows={len(rows)}, expected={expected_num_samples}). "
            "Truncating to the requested number of samples."
        )

    return {
        "path": source_path,
        "y": y[:expected_num_samples].cpu().float(),
        "rows": rows[:expected_num_samples],
    }


def _get_middle_idx(n_layers: int, forced: Optional[int] = None) -> int:
    return forced if forced is not None else (n_layers // 2)


def _get_all_lm_layer_indices(hidden_states: Tuple[torch.Tensor, ...]) -> List[int]:
    # LM hidden_states usually include the embedding output at index 0, followed by the
    # actual transformer layers. For layer sweeps we skip the embedding slot when present.
    num_layers = len(hidden_states)
    if num_layers <= 1:
        return [0]
    return list(range(1, num_layers))


# ----------------------------
# Feature wrappers
# ----------------------------

def _extract_vision_mean_pool(vision_hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
    feats = extract_vision_features_mean_pool(vision_hidden_states, layer_idx)
    _assert_valid_2d(feats, f"vision_layer_{layer_idx}")
    return feats


def _extract_qwen_vision_mean_pool(
    vision_hidden_states: Tuple[torch.Tensor, ...],
    layer_idx: int,
) -> torch.Tensor:
    layer_hidden_states = vision_hidden_states[layer_idx]
    if layer_hidden_states.dim() == 2:
        feats = layer_hidden_states.mean(dim=0, keepdim=True)
    elif layer_hidden_states.dim() == 3:
        feats = layer_hidden_states.mean(dim=1)
    else:
        raise AssertionError(
            f"qwen_vision_layer_{layer_idx}: expected 2D or 3D tensor, "
            f"got {tuple(layer_hidden_states.shape)}"
        )
    _assert_valid_2d(feats, f"qwen_vision_layer_{layer_idx}")
    return feats


def _extract_qwen_token_sequence_mean(tokens: torch.Tensor, name: str) -> torch.Tensor:
    if tokens.dim() == 2:
        feats = tokens.mean(dim=0, keepdim=True)
    elif tokens.dim() == 3:
        feats = tokens.mean(dim=1)
    else:
        raise AssertionError(f"{name}: expected 2D or 3D token tensor, got {tuple(tokens.shape)}")
    _assert_valid_2d(feats, name)
    return feats


def _extract_qwen_token_sequence_last(tokens: torch.Tensor, name: str) -> torch.Tensor:
    if tokens.dim() == 2:
        feats = tokens[-1:, :]
    elif tokens.dim() == 3:
        feats = tokens[:, -1, :]
    else:
        raise AssertionError(f"{name}: expected 2D or 3D token tensor, got {tuple(tokens.shape)}")
    _assert_valid_2d(feats, name)
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


def _extract_token_sequence_mean(tokens: torch.Tensor, name: str) -> torch.Tensor:
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(1)
    assert tokens.dim() == 3, f"{name}: expected 3D token tensor, got {tuple(tokens.shape)}"
    feats = tokens.mean(dim=1)
    _assert_valid_2d(feats, name)
    return feats


def _extract_token_sequence_last(tokens: torch.Tensor, name: str) -> torch.Tensor:
    if tokens.dim() == 2:
        feats = tokens
    else:
        assert tokens.dim() == 3, f"{name}: expected 3D token tensor, got {tuple(tokens.shape)}"
        feats = tokens[:, -1, :]
    _assert_valid_2d(feats, name)
    return feats


def _append_feature_block(
    feats_list: List[torch.Tensor],
    names_list: List[str],
    dims_list: List[int],
    feats: torch.Tensor,
    name: str,
) -> None:
    feats_list.append(feats)
    names_list.append(name)
    dims_list.append(int(feats.shape[1]))


def _extract_qwen_visual_feature_blocks(
    *,
    vision_hidden_states: Optional[Tuple[torch.Tensor, ...]],
    vision_merged_states: Optional[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    if vision_hidden_states is None or vision_merged_states is None:
        raise RuntimeError(
            "Qwen vision-only extraction expected both vision_hidden_states and "
            "vision_merged_states, but one or both were missing."
        )

    per_sample_feats: List[torch.Tensor] = []
    per_sample_names: List[str] = []
    per_sample_dims: List[int] = []

    _append_feature_block(
        per_sample_feats,
        per_sample_names,
        per_sample_dims,
        _extract_qwen_vision_mean_pool(vision_hidden_states, 13),
        "vision_mean_layer_13",
    )
    _append_feature_block(
        per_sample_feats,
        per_sample_names,
        per_sample_dims,
        _extract_qwen_vision_mean_pool(vision_hidden_states, -1),
        "vision_mean_layer_-1",
    )
    _append_feature_block(
        per_sample_feats,
        per_sample_names,
        per_sample_dims,
        _extract_qwen_token_sequence_mean(vision_merged_states, "visual_merged_mean"),
        "visual_merged_mean",
    )
    _append_feature_block(
        per_sample_feats,
        per_sample_names,
        per_sample_dims,
        _extract_qwen_token_sequence_last(vision_merged_states, "visual_merged_lasttok"),
        "visual_merged_lasttok",
    )

    return per_sample_feats, per_sample_names, per_sample_dims


def _extract_qwen_visual_feature_matrix_from_splits(
    *,
    vision_hidden_state_splits: Tuple[Tuple[torch.Tensor, ...], ...],
    vision_merged_state_splits: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, List[str], List[int]]:
    if len(vision_hidden_state_splits) != 2:
        raise RuntimeError(
            "Expected exactly two captured Qwen vision layers (layer 13 and final layer), "
            f"got {len(vision_hidden_state_splits)}."
        )

    layer_13_splits, layer_final_splits = vision_hidden_state_splits
    num_samples = len(vision_merged_state_splits)
    if len(layer_13_splits) != num_samples or len(layer_final_splits) != num_samples:
        raise RuntimeError(
            "Qwen batched vision extraction returned inconsistent per-image split counts "
            f"(layer13={len(layer_13_splits)}, final={len(layer_final_splits)}, merged={num_samples})."
        )

    feature_names = [
        "vision_mean_layer_13",
        "vision_mean_layer_-1",
        "visual_merged_mean",
        "visual_merged_lasttok",
    ]
    feature_dims: List[int] = []
    rows: List[torch.Tensor] = []

    for sample_idx in range(num_samples):
        sample_blocks = [
            _extract_qwen_token_sequence_mean(layer_13_splits[sample_idx], "vision_mean_layer_13"),
            _extract_qwen_token_sequence_mean(layer_final_splits[sample_idx], "vision_mean_layer_-1"),
            _extract_qwen_token_sequence_mean(vision_merged_state_splits[sample_idx], "visual_merged_mean"),
            _extract_qwen_token_sequence_last(vision_merged_state_splits[sample_idx], "visual_merged_lasttok"),
        ]
        if not feature_dims:
            feature_dims = [int(block.shape[1]) for block in sample_blocks]
        rows.append(torch.cat(sample_blocks, dim=1))

    return torch.cat(rows, dim=0), feature_names, feature_dims


# ----------------------------
# Main generator
# ----------------------------

@torch.no_grad()
def generate_qwen_visual_supervised_dataset(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: Any,
    samples: List[Dict[str, Any]],
    cfg: SupervisionGenConfig,
    source_supervision_path: Optional[str] = None,
    checkpoint_every: int = 50,
    vision_batch_size: int = 4,
) -> Dict[str, Any]:
    if "qwen" not in cfg.model_id.lower():
        raise ValueError("generate_qwen_visual_supervised_dataset only supports Qwen models.")

    out_dir = _make_output_dir(cfg)
    visual_pt_path = os.path.join(out_dir, "supervision_dataset_visual.pt")
    metadata_path = os.path.join(out_dir, "metadata_visual.json")
    checkpoint_path = os.path.join(out_dir, "checkpoint_visual.pt")

    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]

    total_samples = len(samples)
    source = _load_source_supervision_dataset(
        cfg=cfg,
        source_supervision_path=source_supervision_path,
        expected_num_samples=total_samples,
    )
    y = source["y"]
    rows = source["rows"]
    vision_batch_size = max(1, int(vision_batch_size))

    print(f"\nGenerating Qwen vision-only supervised dataset")
    print(f"  dataset_id: {cfg.dataset_id}")
    print(f"  model_id:   {cfg.model_id}")
    print(f"  samples:    {total_samples}")
    print(f"  output:     {out_dir}")
    print(f"  source:     {source['path']}")
    print(f"  vision batch size: {vision_batch_size}")
    print(f"  checkpoint: every {checkpoint_every} samples")
    print(f"  verbose:    {cfg.verbose}")

    checkpoint = _load_visual_checkpoint(checkpoint_path)
    if checkpoint is not None:
        X_parts = checkpoint["X_parts"]
        feature_names = checkpoint.get("feature_names", [])
        feature_dims = checkpoint.get("feature_dims", [])
        processed_indices = set(checkpoint["processed_indices"])
        feature_dim_known = len(feature_names) > 0
        print(f"  Resuming from sample {len(processed_indices)}/{total_samples}")
    else:
        X_parts: List[torch.Tensor] = []
        feature_names = []
        feature_dims: List[int] = []
        processed_indices = set()
        feature_dim_known = False
        print("  Starting fresh")

    total_batches = max(1, math.ceil(total_samples / vision_batch_size)) if total_samples > 0 else 1
    iterator = tqdm(
        total=total_samples,
        desc="Generating visual-only",
        initial=len(processed_indices),
        miniters=max(1, total_batches // 100),
        mininterval=30.0,
        smoothing=0.1,
        dynamic_ncols=True,
        disable=False,
        leave=True,
    )

    for batch_start in range(0, total_samples, vision_batch_size):
        batch_end = min(total_samples, batch_start + vision_batch_size)
        batch_indices = [sample_idx for sample_idx in range(batch_start, batch_end) if sample_idx not in processed_indices]
        if not batch_indices:
            continue

        prev_processed_count = len(processed_indices)
        batch_images = [resolve_image(samples[sample_idx]) for sample_idx in batch_indices]
        inputs = prepare_qwen_visual_only_inputs_batch(processor, batch_images, device)
        vision_batch = extract_qwen_selected_vision_tensors_batch(
            vlm_adapter,
            model,
            inputs,
            capture_layer_indices=(13, -1),
        )
        batch_x, per_sample_names, per_sample_dims = _extract_qwen_visual_feature_matrix_from_splits(
            vision_hidden_state_splits=vision_batch["vision_hidden_state_splits"],
            vision_merged_state_splits=vision_batch["vision_merged_state_splits"],
        )

        if batch_x.shape[0] != len(batch_indices):
            raise RuntimeError(
                f"Batched Qwen visual extraction returned {batch_x.shape[0]} rows for "
                f"{len(batch_indices)} requested samples."
            )

        if not feature_dim_known:
            feature_names = per_sample_names[:]
            feature_dims = per_sample_dims[:]
            feature_dim_known = True
        else:
            if per_sample_names != feature_names or per_sample_dims != feature_dims:
                raise RuntimeError(
                    "Visual feature name/order/dim mismatch.\n"
                    f"Expected: {feature_names}\nGot:      {per_sample_names}\n"
                    f"Expected dims: {feature_dims}\nGot dims:      {per_sample_dims}\n"
                    "Delete checkpoint_visual.pt to start fresh."
                )

        batch_rows = batch_x.cpu().float().split(1, dim=0)
        for sample_idx, row in zip(batch_indices, batch_rows):
            X_parts.append(row)
            processed_indices.add(sample_idx)

        iterator.update(len(batch_indices))

        if (len(processed_indices) // checkpoint_every) > (prev_processed_count // checkpoint_every):
            try:
                _save_visual_checkpoint(
                    checkpoint_path,
                    X_parts,
                    feature_names,
                    feature_dims,
                    list(processed_indices),
                )
            except Exception as e:
                print(f"\nWarning: visual checkpoint save failed at {len(processed_indices)} samples: {e}")

    iterator.close()

    X = torch.cat(X_parts, dim=0)
    if X.shape[0] != len(y):
        raise RuntimeError(
            f"Visual feature matrix has {X.shape[0]} rows, but source labels have {len(y)} rows."
        )

    metadata = {
        "dataset_id": cfg.dataset_id,
        "model_id": cfg.model_id,
        "num_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "feature_names": feature_names,
        "feature_dims": feature_dims,
        "seed_offset": cfg.seed_offset,
        "max_samples": cfg.max_samples,
        "label_mean": float(y.mean().item()),
        "label_variance": float(y.var(unbiased=False).item()),
        "mode": "qwen_vision_only",
        "source_supervision_path": source["path"],
    }

    torch.save(
        {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "feature_dims": feature_dims,
            "rows": rows,
            "metadata": metadata,
        },
        visual_pt_path,
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nRemoved visual checkpoint file (processing complete)")

    print(
        f"\nSaved vision-only supervised dataset:\n"
        f"  {visual_pt_path}\n"
        f"  N={X.shape[0]}  D={X.shape[1]}  label_mean={metadata['label_mean']:.3f}"
    )

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "rows": rows,
        "metadata": metadata,
        "output_dir": out_dir,
        "pt_path": visual_pt_path,
    }


@torch.no_grad()
def generate_supervised_uq_dataset(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: Any,
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
        feature_dims = checkpoint.get("feature_dims", [])
        processed_indices = set(checkpoint["processed_indices"])
        feature_dim_known = len(feature_names) > 0
        print(f"  Resuming from sample {len(y_list)}/{len(samples)}")
    else:
        X_parts: List[torch.Tensor] = []
        y_list: List[float] = []
        rows: List[Dict[str, Any]] = []
        feature_names: List[str] = []
        feature_dims: List[int] = []
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
            vlm_adapter=vlm_adapter,
            **pred_kwargs,
        )

        parsed = task.interpret_output(sample=sample, prediction_output=prediction_output)
        y_list.append(float(parsed["score"]))

        vision_hidden_states = parsed["vision_hidden_states"]
        vision_merged_states = parsed.get("vision_merged_states")
        lm_hidden_states = parsed["lm_hidden_states"]
        token_spans = parsed["token_spans"]
        answer_hidden_states = parsed["answer_hidden_states"]
        gen_ids = parsed["gen_ids"]
        gen_step_logits = parsed["gen_step_logits"]

        # -------- features --------
        per_sample_feats: List[torch.Tensor] = []
        per_sample_names: List[str] = []
        per_sample_dims: List[int] = []

        model_id_lower = cfg.model_id.lower()
        is_qwen = "qwen" in model_id_lower

        if is_qwen:
            if vision_hidden_states is not None:
                _append_feature_block(
                    per_sample_feats,
                    per_sample_names,
                    per_sample_dims,
                    _extract_qwen_vision_mean_pool(vision_hidden_states, 13),
                    "vision_mean_layer_13",
                )
                _append_feature_block(
                    per_sample_feats,
                    per_sample_names,
                    per_sample_dims,
                    _extract_qwen_vision_mean_pool(vision_hidden_states, -1),
                    "vision_mean_layer_-1",
                )

            if vision_merged_states is not None:
                _append_feature_block(
                    per_sample_feats,
                    per_sample_names,
                    per_sample_dims,
                    _extract_qwen_token_sequence_mean(vision_merged_states, "visual_merged_mean"),
                    "visual_merged_mean",
                )
                _append_feature_block(
                    per_sample_feats,
                    per_sample_names,
                    per_sample_dims,
                    _extract_qwen_token_sequence_last(vision_merged_states, "visual_merged_lasttok"),
                    "visual_merged_lasttok",
                )

            if lm_hidden_states is not None and token_spans:
                v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
                q0, q1 = token_spans["question_start"], token_spans["question_end"]
                fs0, fs1 = token_spans["fullspan_start"], token_spans["fullspan_end"]

                qwen_prompt_specs = [
                    ("lm_visual_mean_layer_15", lambda: _extract_lm_mean_pool(lm_hidden_states, 15, v0, v1, name="lm_visual_mean_layer_15")),
                    ("lm_visual_mean_layer_-1", lambda: _extract_lm_mean_pool(lm_hidden_states, -1, v0, v1, name="lm_visual_mean_layer_-1")),
                    ("lm_visual_lasttok_layer_16", lambda: _extract_lm_last_token(lm_hidden_states, 16, v0, v1, name="lm_visual_lasttok_layer_16")),
                    ("lm_visual_lasttok_layer_-1", lambda: _extract_lm_last_token(lm_hidden_states, -1, v0, v1, name="lm_visual_lasttok_layer_-1")),
                    ("lm_question_mean_layer_16", lambda: _extract_lm_mean_pool(lm_hidden_states, 16, q0, q1, name="lm_question_mean_layer_16")),
                    ("lm_question_mean_layer_-1", lambda: _extract_lm_mean_pool(lm_hidden_states, -1, q0, q1, name="lm_question_mean_layer_-1")),
                    ("lm_question_lasttok_layer_16", lambda: _extract_lm_last_token(lm_hidden_states, 16, q0, q1, name="lm_question_lasttok_layer_16")),
                    ("lm_question_lasttok_layer_-1", lambda: _extract_lm_last_token(lm_hidden_states, -1, q0, q1, name="lm_question_lasttok_layer_-1")),
                    ("lm_fullspan_mean_layer_16", lambda: _extract_lm_mean_pool(lm_hidden_states, 16, fs0, fs1, name="lm_fullspan_mean_layer_16")),
                    ("lm_fullspan_mean_layer_-1", lambda: _extract_lm_mean_pool(lm_hidden_states, -1, fs0, fs1, name="lm_fullspan_mean_layer_-1")),
                    ("lm_fullspan_lasttok_layer_16", lambda: _extract_lm_last_token(lm_hidden_states, 16, fs0, fs1, name="lm_fullspan_lasttok_layer_16")),
                    ("lm_fullspan_lasttok_layer_-1", lambda: _extract_lm_last_token(lm_hidden_states, -1, fs0, fs1, name="lm_fullspan_lasttok_layer_-1")),
                    ("lm_visual_lasttok_selfattn_layer_15", lambda: _extract_lm_last_token(lm_hidden_states, 15, v0, v1, name="lm_visual_lasttok_selfattn_layer_15")),
                    ("lm_question_mean_selfattn_layer_15", lambda: _extract_lm_mean_pool(lm_hidden_states, 15, q0, q1, name="lm_question_mean_selfattn_layer_15")),
                    ("lm_question_lasttok_selfattn_layer_15", lambda: _extract_lm_last_token(lm_hidden_states, 15, q0, q1, name="lm_question_lasttok_selfattn_layer_15")),
                    ("lm_fullspan_mean_selfattn_layer_15", lambda: _extract_lm_mean_pool(lm_hidden_states, 15, fs0, fs1, name="lm_fullspan_mean_selfattn_layer_15")),
                    ("lm_fullspan_lasttok_selfattn_layer_15", lambda: _extract_lm_last_token(lm_hidden_states, 15, fs0, fs1, name="lm_fullspan_lasttok_selfattn_layer_15")),
                ]

                for feature_name, feature_fn in qwen_prompt_specs:
                    _append_feature_block(
                        per_sample_feats,
                        per_sample_names,
                        per_sample_dims,
                        feature_fn(),
                        feature_name,
                    )

            if answer_hidden_states is not None and token_spans and token_spans.get("answer_length", 0) > 0:
                a0, a1 = token_spans["answer_start"], token_spans["answer_end"]
                qwen_answer_specs = [
                    ("lm_answer_mean_layer_16", lambda: _extract_lm_mean_pool(answer_hidden_states, 16, a0, a1, name="lm_answer_mean_layer_16")),
                    ("lm_answer_mean_layer_-1", lambda: _extract_lm_mean_pool(answer_hidden_states, -1, a0, a1, name="lm_answer_mean_layer_-1")),
                    ("lm_answer_lasttok_layer_16", lambda: _extract_lm_last_token(answer_hidden_states, 16, a0, a1, name="lm_answer_lasttok_layer_16")),
                    ("lm_answer_lasttok_layer_-1", lambda: _extract_lm_last_token(answer_hidden_states, -1, a0, a1, name="lm_answer_lasttok_layer_-1")),
                    ("lm_answer_mean_selfattn_layer_15", lambda: _extract_lm_mean_pool(answer_hidden_states, 15, a0, a1, name="lm_answer_mean_selfattn_layer_15")),
                    ("lm_answer_lasttok_selfattn_layer_15", lambda: _extract_lm_last_token(answer_hidden_states, 15, a0, a1, name="lm_answer_lasttok_selfattn_layer_15")),
                ]

                for feature_name, feature_fn in qwen_answer_specs:
                    _append_feature_block(
                        per_sample_feats,
                        per_sample_names,
                        per_sample_dims,
                        feature_fn(),
                        feature_name,
                    )

        if (not is_qwen) and (cfg.use_vision_middle or cfg.use_vision_final) and vision_hidden_states is None:
            raise RuntimeError(
                "Vision features were requested, but no vision hidden states were returned "
                f"for model_id={cfg.model_id!r}. Check the VLM adapter vision_attr_paths "
                "and the current transformers model structure."
            )

        # Vision
        if (not is_qwen) and vision_hidden_states is not None and (cfg.use_vision_middle or cfg.use_vision_final):
            n_v = len(vision_hidden_states)
            mid_v = _get_middle_idx(n_v, cfg.force_middle_layer)

            if cfg.use_vision_middle:
                f = _extract_vision_mean_pool(vision_hidden_states, mid_v)
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"vision_mean_layer_{mid_v}")

            if cfg.use_vision_final:
                f = _extract_vision_mean_pool(vision_hidden_states, -1)
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "vision_mean_layer_-1")

        # LM question/visual spans
        if (not is_qwen) and lm_hidden_states is not None and token_spans:
            n_lm = len(lm_hidden_states)
            mid_lm = _get_middle_idx(n_lm, cfg.force_middle_layer)
            all_lm_indices = _get_all_lm_layer_indices(lm_hidden_states)

            v0, v1 = token_spans["visual_start"], token_spans["visual_end"]
            q0, q1 = token_spans["question_start"], token_spans["question_end"]
            fs0, fs1 = token_spans["fullspan_start"], token_spans["fullspan_end"]
            qin_spans = token_spans["question_including_spans"]

            if cfg.use_lm_visual_all_layers_mean:
                for layer_idx in all_lm_indices:
                    f = _extract_lm_mean_pool(
                        lm_hidden_states, layer_idx, v0, v1, name=f"lm_visual_all_layers_{layer_idx}"
                    )
                    _append_feature_block(
                        per_sample_feats,
                        per_sample_names,
                        per_sample_dims,
                        f,
                        f"lm_visual_mean_layer_{layer_idx}",
                    )

            if cfg.use_lm_visual_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, v0, v1, name=f"lm_mid_visual_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_visual_mean_layer_{mid_lm}")

            if cfg.use_lm_visual_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, v0, v1, name="lm_final_visual_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_visual_mean_layer_-1")

            if cfg.use_lm_question_all_layers_mean:
                for layer_idx in all_lm_indices:
                    f = _extract_lm_mean_pool(
                        lm_hidden_states, layer_idx, q0, q1, name=f"lm_question_all_layers_{layer_idx}"
                    )
                    _append_feature_block(
                        per_sample_feats,
                        per_sample_names,
                        per_sample_dims,
                        f,
                        f"lm_question_mean_layer_{layer_idx}",
                    )

            if cfg.use_lm_question_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, q0, q1, name=f"lm_mid_question_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_question_mean_layer_{mid_lm}")

            if cfg.use_lm_question_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, q0, q1, name="lm_final_question_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_question_mean_layer_-1")

            if cfg.use_lm_visual_middle_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, mid_lm, v0, v1, name=f"lm_mid_visual_lasttok_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_visual_lasttok_layer_{mid_lm}")

            if cfg.use_lm_visual_final_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, -1, v0, v1, name="lm_final_visual_lasttok_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_visual_lasttok_layer_-1")

            if cfg.use_lm_question_middle_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, mid_lm, q0, q1, name=f"lm_mid_question_lasttok_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_question_lasttok_layer_{mid_lm}")

            if cfg.use_lm_question_final_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, -1, q0, q1, name="lm_final_question_lasttok_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_question_lasttok_layer_-1")
            # full span features
            if cfg.use_lm_fullspan_middle:
                f = _extract_lm_mean_pool(lm_hidden_states, mid_lm, fs0, fs1, name=f"lm_mid_fullspan_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_fullspan_mean_layer_{mid_lm}")

            if cfg.use_lm_fullspan_final:
                f = _extract_lm_mean_pool(lm_hidden_states, -1, fs0, fs1, name="lm_final_fullspan_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_fullspan_mean_layer_-1")

            if cfg.use_lm_fullspan_middle_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, mid_lm, fs0, fs1, name=f"lm_mid_fullspan_lasttok_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_fullspan_lasttok_layer_{mid_lm}")

            if cfg.use_lm_fullspan_final_lasttoken:
                f = _extract_lm_last_token(lm_hidden_states, -1, fs0, fs1, name="lm_final_fullspan_lasttok_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_fullspan_lasttok_layer_-1")

            if cfg.use_lm_question_including_middle:
                f = _extract_lm_mean_pool_multi(lm_hidden_states, mid_lm, qin_spans, name=f"lm_mid_question_including_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_question_including_mean_layer_{mid_lm}")

            if cfg.use_lm_question_including_final:
                f = _extract_lm_mean_pool_multi(lm_hidden_states, -1, qin_spans, name="lm_final_question_including_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_question_including_mean_layer_-1")

            if cfg.use_lm_question_including_middle_lasttoken:
                f = _extract_lm_last_token_multi(lm_hidden_states, mid_lm, qin_spans, name=f"lm_mid_question_including_lasttok_{mid_lm}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_question_including_lasttok_layer_{mid_lm}")

            if cfg.use_lm_question_including_final_lasttoken:
                f = _extract_lm_last_token_multi(lm_hidden_states, -1, qin_spans, name="lm_final_question_including_lasttok_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_question_including_lasttok_layer_-1")

        # Answer span features
        if (not is_qwen) and answer_hidden_states is not None and token_spans and token_spans.get("answer_length", 0) > 0 and (
            cfg.use_lm_answer_all_layers_mean
            or
            cfg.use_lm_answer_middle
            or cfg.use_lm_answer_final
            or cfg.use_lm_answer_middle_lasttoken
            or cfg.use_lm_answer_final_lasttoken
        ):
            n_ans = len(answer_hidden_states)
            mid_ans = _get_middle_idx(n_ans, cfg.force_middle_layer)
            all_answer_indices = _get_all_lm_layer_indices(answer_hidden_states)

            a0, a1 = token_spans["answer_start"], token_spans["answer_end"]

            if cfg.use_lm_answer_all_layers_mean:
                for layer_idx in all_answer_indices:
                    f = _extract_lm_mean_pool(
                        answer_hidden_states, layer_idx, a0, a1, name=f"lm_answer_all_layers_{layer_idx}"
                    )
                    _append_feature_block(
                        per_sample_feats,
                        per_sample_names,
                        per_sample_dims,
                        f,
                        f"lm_answer_mean_layer_{layer_idx}",
                    )

            if cfg.use_lm_answer_middle:
                f = _extract_lm_mean_pool(answer_hidden_states, mid_ans, a0, a1, name=f"lm_mid_answer_{mid_ans}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_answer_mean_layer_{mid_ans}")

            if cfg.use_lm_answer_final:
                f = _extract_lm_mean_pool(answer_hidden_states, -1, a0, a1, name="lm_final_answer_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_answer_mean_layer_-1")

            if cfg.use_lm_answer_middle_lasttoken:
                f = _extract_lm_last_token(answer_hidden_states, mid_ans, a0, a1, name=f"lm_mid_answer_lasttok_{mid_ans}")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, f"lm_answer_lasttok_layer_{mid_ans}")

            if cfg.use_lm_answer_final_lasttoken:
                f = _extract_lm_last_token(answer_hidden_states, -1, a0, a1, name="lm_final_answer_lasttok_-1")
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, f, "lm_answer_lasttok_layer_-1")

        # Answer prob/entropy stats
        if cfg.use_answer_prob_entropy_stats and gen_step_logits is not None and gen_ids is not None and gen_ids.shape[1] > 0:
            stats, stat_names = _extract_gen_prob_entropy_stats(gen_step_logits=gen_step_logits, gen_ids=gen_ids, name_prefix="answer_gen")
            _assert_valid_2d(stats, "answer_gen_prob_entropy_stats")
            stat_blocks = [stats[:, i:i+1] for i in range(stats.shape[1])]
            for feature_name, feature_block in zip(stat_names, stat_blocks):
                _append_feature_block(per_sample_feats, per_sample_names, per_sample_dims, feature_block, feature_name)

        if len(per_sample_feats) == 0:
            raise RuntimeError("No features were extracted for this sample. Check cfg feature flags.")

        per_x = torch.cat(per_sample_feats, dim=1).cpu().float()

        if not feature_dim_known:
            feature_names = per_sample_names[:]
            feature_dims = per_sample_dims[:]
            feature_dim_known = True
        else:
            if per_sample_names != feature_names or per_sample_dims != feature_dims:
                raise RuntimeError(
                    "Feature name/order/dim mismatch.\n"
                    f"Expected: {feature_names}\nGot:      {per_sample_names}\n"
                    f"Expected dims: {feature_dims}\nGot dims:      {per_sample_dims}\n"
                    "Delete checkpoint to start fresh."
                )

        X_parts.append(per_x)
        processed_indices.add(sample_idx)
        rows.append(parsed["row"])

        if len(y_list) % checkpoint_every == 0:
            try:
                _save_checkpoint(checkpoint_path, X_parts, y_list, rows, feature_names, feature_dims, list(processed_indices))
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
        "feature_dims": feature_dims,
        "seed_offset": cfg.seed_offset,
        "max_samples": cfg.max_samples,
        "label_mean": float(y.mean().item()),
        "label_variance": float(y.var(unbiased=False).item()),
    }

    pt_path = os.path.join(out_dir, "supervision_dataset.pt")
    torch.save(
        {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "feature_dims": feature_dims,
            "rows": rows,
            "metadata": metadata,
        },
        pt_path,
    )

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nRemoved checkpoint file (processing complete)")

    print(f"\nSaved supervised dataset:\n  {pt_path}\n  N={X.shape[0]}  D={X.shape[1]}  label_mean={metadata['label_mean']:.3f}")

    return {"X": X, "y": y, "feature_names": feature_names, "rows": rows, "metadata": metadata, "output_dir": out_dir, "pt_path": pt_path}
