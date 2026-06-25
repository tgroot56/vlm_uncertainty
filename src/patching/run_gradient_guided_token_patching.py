"""Gradient-guided greedy token activation patching for qualitative POPE pilots."""

from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.patching.collect_activations import _MAX_NEW_TOKENS, _format_prompt, _score_prediction
from src.patching.run_layer_sweep_patching import _extract_all_layer_activations
from src.patching.run_patching import (
    _compute_output_stats,
    _compute_pope_answer_probability_stats,
    _get_decoder_layers,
    _patched_generate,
    _run_baseline_forward,
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
    qwen_vision_cache_context,
)


RECOVERY_EPSILON = 1e-8
YELLOW = "#fff1a8"
YELLOW_DARK = "#b45309"
ORANGE = (245, 158, 11, 255)
WHITE = (255, 255, 255, 235)


@dataclass
class GradientGuidedTokenPatchingConfig:
    dataset_id: str = "pope"
    sample_id: str = "795"
    severity: str = "extreme"
    human_layers: List[int] = field(default_factory=lambda: [7, 10, 13, 18])
    max_patches: int = 8
    visual_top_k: int = 64
    require_clean_yes: bool = True
    max_new_tokens: int = 15
    verbose: bool = False
    make_figures: bool = True
    run_types: List[str] = field(
        default_factory=lambda: ["answer_recovery", "confidence_recovery"]
    )
    early_stop_window: int = 0
    early_stop_abs_sum_threshold: float = 0.0
    require_corrupted_no: bool = False
    qwen_prefill_empty_thinking: bool = False

    @property
    def code_layers(self) -> List[int]:
        return [layer - 1 for layer in self.human_layers]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _json_dumpable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _decode_token_piece(processor: Any, token_id: int, token: str) -> str:
    try:
        piece = processor.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        piece = str(token).replace("▁", " ")
    if token == "▁":
        return " "
    if token.startswith("▁") and piece and not piece.startswith(" "):
        return " " + piece
    if token.startswith("Ġ") and piece and not piece.startswith(" "):
        return " " + piece
    return piece or str(token).replace("▁", " ")


def _visible_token_text(piece: str, fallback: str) -> str:
    if piece and all(ch in "\n\r\t" for ch in piece):
        return "".join({"\n": r"\n", "\r": r"\r", "\t": r"\t"}[ch] for ch in piece)
    if piece.strip():
        return piece.strip()
    if fallback.strip():
        return fallback.strip()
    return " "


def _classify_position(pos: int, spans: Dict[str, Any]) -> str:
    if int(spans["visual_start"]) <= pos < int(spans["visual_end"]):
        return "visual"
    if pos < int(spans["visual_start"]):
        return "text_pre"
    if pos < int(spans.get("question_start", spans["visual_end"])):
        return "text_post"
    if pos < int(spans.get("question_end", spans["prompt_len"])):
        return "question"
    return "assistant_marker"


def _token_type(pos: int, token: Dict[str, Any], spans: Dict[str, Any]) -> str:
    if pos == int(spans["prompt_len"]) - 1:
        return "final_prompt_token"
    span = str(token["span_label"])
    text = str(token.get("decoded", token.get("display", ""))).strip().lower()
    raw = str(token.get("token", ""))
    if span == "visual":
        if raw == "<|vision_start|>":
            return "visual_boundary_start"
        if raw == "<|vision_end|>":
            return "visual_boundary_end"
        return "visual_patch"
    if raw == "<s>":
        return "bos_token"
    if span == "text_pre":
        return "template_prefix"
    if span == "assistant_marker":
        return "assistant_marker"
    if span == "question":
        if pos < int(spans["question_start"]) + 4:
            return "question_prefix"
        if text in {"answer", "with", "exactly", "one", "word", "yes", "or", "no"} or pos >= int(spans["question_end"]) - 10:
            return "answer_instruction"
        return "question_word"
    return span


def _token_span(token: Dict[str, Any]) -> str:
    if str(token["token_type"]) == "final_prompt_token":
        return "Final prompt token"
    group = str(token["token_group"])
    if group in {"text_pre", "text_post"}:
        return "Template Prefix"
    if group == "visual":
        return "Visual"
    if group == "question":
        return "Question"
    if group == "assistant_marker":
        return "Template Suffix"
    raise ValueError(
        f"Cannot map token position {token['token_position']} with "
        f"token_group={group!r}, token_type={token['token_type']!r}."
    )


def _llava_visual_geometry(processor: Any, spans: Dict[str, Any]) -> Dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    crop_size = getattr(image_processor, "crop_size", {}) if image_processor is not None else {}
    size = getattr(image_processor, "size", {}) if image_processor is not None else {}
    patch_size = int(getattr(processor, "patch_size", 0) or getattr(image_processor, "patch_size", 0) or 14)

    def dim(obj: Any, key: str, default: int) -> int:
        if isinstance(obj, dict):
            return int(obj.get(key, default) or default)
        return int(getattr(obj, key, default) or default)

    crop_h = dim(crop_size, "height", 336)
    crop_w = dim(crop_size, "width", 336)
    shortest_edge = dim(size, "shortest_edge", min(crop_h, crop_w))
    rows = crop_h // patch_size
    cols = crop_w // patch_size
    return {
        "family": "llava",
        "source": f"LLaVA CLIP {rows}x{cols} crop geometry",
        "grid_rows": rows,
        "grid_cols": cols,
        "mappable_visual_tokens": rows * cols,
        "visual_token_offset": 0,
        "patch_size": patch_size,
        "crop_height": crop_h,
        "crop_width": crop_w,
        "shortest_edge": shortest_edge,
    }


def _qwen_visual_geometry(
    processor: Any,
    inputs: Dict[str, Any],
    spans: Dict[str, Any],
) -> Dict[str, Any]:
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("Qwen inputs do not contain image_grid_thw.")
    grid = [int(x) for x in image_grid_thw.detach().cpu().reshape(-1, 3)[0].tolist()]
    image_processor = getattr(processor, "image_processor", None)
    merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
    t, grid_h, grid_w = grid
    rows = grid_h // merge_size
    cols = grid_w // merge_size
    mappable = t * rows * cols
    expected = max(0, int(spans.get("num_visual_tokens", 0)) - 2)
    if t != 1 or rows <= 0 or cols <= 0 or mappable != expected:
        raise RuntimeError(
            "Qwen visual geometry does not match the recovered visual span: "
            f"grid={grid}, merge_size={merge_size}, mappable={mappable}, expected={expected}."
        )
    return {
        "family": "qwen",
        "source": "Qwen image_grid_thw-derived merged grid",
        "image_grid_thw": grid,
        "merge_size": merge_size,
        "grid_rows": rows,
        "grid_cols": cols,
        "mappable_visual_tokens": mappable,
        "visual_token_offset": 1,
    }


def _resize_shortest_edge_size(width: int, height: int, shortest_edge: int) -> Tuple[int, int, float, float]:
    if width <= 0 or height <= 0 or shortest_edge <= 0:
        return width, height, 1.0, 1.0
    if width <= height:
        resized_w = shortest_edge
        resized_h = int(shortest_edge * height / width)
    else:
        resized_h = shortest_edge
        resized_w = int(shortest_edge * width / height)
    resized_w = max(1, resized_w)
    resized_h = max(1, resized_h)
    return resized_w, resized_h, resized_w / width, resized_h / height


def _llava_clip_rect(
    image_size: Tuple[int, int],
    row: int,
    col: int,
    rows: int,
    cols: int,
    geometry: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    width, height = image_size
    crop_w = int(geometry.get("crop_width", 336))
    crop_h = int(geometry.get("crop_height", 336))
    shortest = int(geometry.get("shortest_edge", min(crop_w, crop_h)))
    resized_w, resized_h, scale_x, scale_y = _resize_shortest_edge_size(width, height, shortest)
    crop_left = max(0.0, (resized_w - crop_w) / 2.0)
    crop_top = max(0.0, (resized_h - crop_h) / 2.0)
    cell_x0 = col * crop_w / cols
    cell_y0 = row * crop_h / rows
    cell_x1 = (col + 1) * crop_w / cols
    cell_y1 = (row + 1) * crop_h / rows
    x0 = math.floor((crop_left + cell_x0) / scale_x)
    y0 = math.floor((crop_top + cell_y0) / scale_y)
    x1 = math.ceil((crop_left + cell_x1) / scale_x)
    y1 = math.ceil((crop_top + cell_y1) / scale_y)
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _visual_rect_for_position(
    image_size: Tuple[int, int],
    pos: int,
    spans: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, int]]:
    visual_rel = pos - int(spans["visual_start"])
    rows = int(geometry["grid_rows"])
    cols = int(geometry["grid_cols"])
    image_rel = visual_rel - int(geometry.get("visual_token_offset", 0))
    mappable = int(geometry.get("mappable_visual_tokens", rows * cols))
    if image_rel < 0 or image_rel >= mappable:
        return None, {"visual_rel_pos": visual_rel, "image_rel_pos": image_rel}
    row = min(rows - 1, max(0, image_rel // cols))
    col = min(cols - 1, max(0, image_rel % cols))
    if geometry.get("family") == "qwen":
        width, height = image_size
        rect = (
            math.floor(col * width / cols),
            math.floor(row * height / rows),
            math.ceil((col + 1) * width / cols),
            math.ceil((row + 1) * height / rows),
        )
    else:
        rect = _llava_clip_rect(image_size, row, col, rows, cols, geometry)
    return rect, {
        "visual_rel_pos": visual_rel,
        "image_rel_pos": image_rel,
        "grid_row": row,
        "grid_col": col,
    }


def _token_records(processor: Any, input_ids: torch.Tensor, spans: Dict[str, Any]) -> List[Dict[str, Any]]:
    token_ids = [int(x) for x in input_ids.tolist()]
    token_strings = processor.tokenizer.convert_ids_to_tokens(token_ids)
    records: List[Dict[str, Any]] = []
    visual_start = int(spans["visual_start"])
    qwen_visual_span = (
        visual_start < len(token_strings)
        and str(token_strings[visual_start]) == "<|vision_start|>"
    )
    for pos, (token_id, token) in enumerate(zip(token_ids, token_strings)):
        span = _classify_position(pos, spans)
        decoded = _decode_token_piece(processor, token_id, str(token))
        if span == "visual":
            if str(token) == "<|vision_start|>":
                display = "[VIS_START]"
            elif str(token) == "<|vision_end|>":
                display = "[VIS_END]"
            else:
                visual_offset = 1 if qwen_visual_span else 0
                display = f"V{pos - visual_start - visual_offset:03d}"
        else:
            display = _visible_token_text(decoded, str(token))
        item = {
            "token_position": pos,
            "token_index": pos,
            "token_id": token_id,
            "token": str(token),
            "decoded": decoded,
            "token_text": display,
            "span_label": span,
            "token_group": span,
        }
        item["token_type"] = _token_type(pos, item, spans)
        item["token_span"] = _token_span(item)
        records.append(item)
    return records


def _positions_for_pool(spans: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
    prompt_len = int(spans["prompt_len"])
    visual_start = int(spans["visual_start"])
    visual_end = int(spans["visual_end"])
    all_positions = list(range(prompt_len))
    visual_positions = list(range(visual_start, visual_end))
    non_visual_positions = [p for p in all_positions if p < visual_start or p >= visual_end]
    return all_positions, visual_positions, non_visual_positions


def _pope_token_ids(processor: Any) -> Tuple[List[int], List[int]]:
    yes_ids = _collect_unique_token_ids(processor.tokenizer, _YES_SURFACE_FORMS)
    no_ids = _collect_unique_token_ids(processor.tokenizer, _NO_SURFACE_FORMS)
    if not yes_ids or not no_ids:
        raise RuntimeError("Could not tokenize POPE yes/no surface forms.")
    return yes_ids, no_ids


def _yes_no_probs_from_logits(
    logits: torch.Tensor,
    yes_ids: List[int],
    no_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits.float(), dim=-1)
    yes_idx = torch.tensor(yes_ids, device=logits.device, dtype=torch.long)
    no_idx = torch.tensor(no_ids, device=logits.device, dtype=torch.long)
    return probs[yes_idx].max(), probs[no_idx].max()


def _compute_recovery(value: float, clean_value: float, degraded_value: float) -> float:
    denom = clean_value - degraded_value
    eps = RECOVERY_EPSILON if denom >= 0 else -RECOVERY_EPSILON
    return float((value - degraded_value) / (denom + eps))


def _stats_from_forward(
    fwd: Dict[str, Any],
    processor: Any,
    record: Dict[str, Any],
    cfg: GradientGuidedTokenPatchingConfig,
    clean_answer_text: str,
) -> Dict[str, Any]:
    output_stats = _compute_output_stats(fwd["gen_step_logits"], fwd.get("gen_ids"), processor)
    pope_stats = _compute_pope_answer_probability_stats(
        fwd["gen_step_logits"],
        processor,
        clean_answer_text,
    )
    raw_text = str(fwd["raw_text"])
    bucket = _extract_yes_no_answer(raw_text).lower()
    if bucket not in {"yes", "no"}:
        bucket = "other"
    score = _score_prediction(cfg.dataset_id, raw_text, record)
    p_yes = _safe_float(pope_stats.get("pope_yes_probability"))
    p_no = _safe_float(pope_stats.get("pope_no_probability"))
    p_clean = _safe_float(pope_stats.get("clean_answer_probability"))
    return {
        "predicted_answer": raw_text,
        "answer_bucket": bucket,
        "score": float(score),
        "p_yes": p_yes,
        "p_no": p_no,
        "p_clean": p_clean,
        "yes_no_margin": p_yes - p_no,
        **output_stats,
        **pope_stats,
    }


def _capture_gradient_scores(
    *,
    model: Any,
    device: torch.device,
    degraded_inputs: Dict[str, torch.Tensor],
    vlm_adapter: VLMAdapter,
    code_layer_idx: int,
    clean_layer_h: torch.Tensor,
    selected_positions: List[int],
    objective: str,
    yes_ids: List[int],
    no_ids: List[int],
    degraded_fused: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    decoder_layers = _get_decoder_layers(model, vlm_adapter)
    target_layer = decoder_layers[code_layer_idx]
    prompt_len = int(degraded_inputs["input_ids"].shape[1])
    clean_layer_device = clean_layer_h.to(device=device, dtype=next(model.parameters()).dtype)
    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] < prompt_len:
            return output
        patched_hidden = hidden.detach().clone()
        for pos in selected_positions:
            patched_hidden[0, pos, :] = clean_layer_device[pos]
        patched_hidden.requires_grad_(True)
        captured["hidden"] = patched_hidden
        if isinstance(output, tuple):
            return (patched_hidden,) + output[1:]
        return patched_hidden

    model.zero_grad(set_to_none=True)
    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            cache_context = (
                qwen_vision_cache_context(model, degraded_fused)
                if degraded_fused is not None
                else nullcontext()
            )
            with cache_context:
                outputs = model(**degraded_inputs, return_dict=True)
            logits = outputs.logits[0, prompt_len - 1, :]
            yes_prob, no_prob = _yes_no_probs_from_logits(logits, yes_ids, no_ids)
            if objective == "answer":
                scalar = yes_prob - no_prob
            elif objective in {"confidence", "maximize_p_yes"}:
                scalar = yes_prob
            elif objective == "maximize_p_no":
                scalar = no_prob
            else:
                raise ValueError(f"Unsupported gradient objective: {objective}")
            scalar.backward()
    finally:
        handle.remove()

    if "hidden" not in captured or captured["hidden"].grad is None:
        raise RuntimeError("Failed to capture gradient for target layer output.")

    hidden = captured["hidden"].detach()[0, :prompt_len, :].float().cpu()
    grad = captured["hidden"].grad.detach()[0, :prompt_len, :].float().cpu()
    clean = clean_layer_h[:prompt_len, :].float().cpu()
    scores: Dict[int, float] = {}
    selected = set(selected_positions)
    for pos in range(prompt_len):
        if pos in selected:
            continue
        delta = clean[pos] - hidden[pos]
        scores[pos] = float(torch.dot(grad[pos], delta).item())

    return scores, {
        "gradient_objective_value": float(scalar.detach().float().item()),
        "gradient_p_yes": float(yes_prob.detach().float().item()),
        "gradient_p_no": float(no_prob.detach().float().item()),
    }


def _candidate_pool(
    gradient_scores: Dict[int, float],
    visual_positions: List[int],
    non_visual_positions: List[int],
    selected_positions: List[int],
    visual_top_k: int,
) -> List[int]:
    selected = set(selected_positions)
    non_visual = [p for p in non_visual_positions if p not in selected]
    visual_ranked = sorted(
        [p for p in visual_positions if p not in selected],
        key=lambda p: (gradient_scores.get(p, float("-inf")), -p),
        reverse=True,
    )
    visual = visual_ranked[: max(0, visual_top_k)]
    return sorted(set(non_visual + visual))


def _exact_patch_eval(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    degraded_image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    code_layer_idx: int,
    clean_layer_h: torch.Tensor,
    selected_positions: List[int],
    candidate_pos: int,
    record: Dict[str, Any],
    cfg: GradientGuidedTokenPatchingConfig,
    clean_answer_text: str,
    baseline_clean: Dict[str, Any],
    baseline_degraded: Dict[str, Any],
    degraded_fused: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trial_positions = selected_positions + [candidate_pos]
    clean_activations = clean_layer_h[trial_positions, :]
    with torch.no_grad():
        patched_fwd = _patched_generate(
            model=model,
            processor=processor,
            device=device,
            vlm_adapter=vlm_adapter,
            image=degraded_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            layer_idx=code_layer_idx,
            positions=trial_positions,
            clean_activations=clean_activations,
            fused_bundle=degraded_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
    stats = _stats_from_forward(patched_fwd, processor, record, cfg, clean_answer_text)
    stats["answer_recovered"] = stats["answer_bucket"] == "yes"
    stats["answer_recovery"] = _compute_recovery(
        stats["score"],
        baseline_clean["score"],
        baseline_degraded["score"],
    )
    stats["p_clean_recovery"] = _compute_recovery(
        stats["p_clean"],
        baseline_clean["p_clean"],
        baseline_degraded["p_clean"],
    )
    return stats


def _candidate_sort_key(
    row: Dict[str, Any],
    run_type: str,
    current_p_clean: float,
    current_margin: float,
) -> Tuple[Any, ...]:
    p_clean = _safe_float(row["p_clean"])
    margin = _safe_float(row["yes_no_margin"])
    if run_type == "maximize_p_yes":
        return (
            _safe_float(row["p_yes"]),
            _safe_float(row.get("gradient_score")),
            -int(row["candidate_position"]),
        )
    if run_type == "maximize_p_no":
        return (
            _safe_float(row["p_no"]),
            _safe_float(row.get("gradient_score")),
            -int(row["candidate_position"]),
        )
    if run_type == "answer_recovery":
        return (
            bool(row["answer_recovered"]),
            p_clean if row["answer_recovered"] else margin - current_margin,
            margin - current_margin,
            p_clean - current_p_clean,
            _safe_float(row.get("gradient_score")),
            -int(row["candidate_position"]),
        )
    return (
        p_clean - current_p_clean,
        p_clean,
        margin,
        bool(row["answer_recovered"]),
        _safe_float(row.get("gradient_score")),
        -int(row["candidate_position"]),
    )


def _objective_value(stats: Dict[str, Any], run_type: str) -> float:
    if run_type == "maximize_p_yes":
        return _safe_float(stats["p_yes"])
    if run_type == "maximize_p_no":
        return _safe_float(stats["p_no"])
    if run_type == "confidence_recovery":
        return _safe_float(stats["p_clean"])
    if run_type == "answer_recovery":
        return _safe_float(stats["yes_no_margin"])
    raise ValueError(f"Unsupported run type: {run_type}")


def _run_one_greedy_path(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    degraded_inputs: Dict[str, torch.Tensor],
    degraded_image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    clean_layer_h: torch.Tensor,
    code_layer_idx: int,
    human_layer: int,
    run_type: str,
    record: Dict[str, Any],
    cfg: GradientGuidedTokenPatchingConfig,
    token_records: List[Dict[str, Any]],
    visual_positions: List[int],
    non_visual_positions: List[int],
    yes_ids: List[int],
    no_ids: List[int],
    clean_answer_text: str,
    baseline_clean: Dict[str, Any],
    baseline_degraded: Dict[str, Any],
    degraded_fused: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_positions: List[int] = []
    selected_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    current_p_clean = baseline_degraded["p_clean"]
    current_margin = baseline_degraded["yes_no_margin"]
    current_objective = _objective_value(baseline_degraded, run_type)
    objective_deltas: List[float] = []
    stopped_reason = "max_patches"
    objective = "answer" if run_type == "answer_recovery" else run_type
    if run_type == "confidence_recovery":
        objective = "confidence"
    generalized_objective = run_type in {"maximize_p_yes", "maximize_p_no"}

    for step_idx in range(1, cfg.max_patches + 1):
        gradient_scores, gradient_meta = _capture_gradient_scores(
            model=model,
            device=device,
            degraded_inputs=degraded_inputs,
            vlm_adapter=vlm_adapter,
            code_layer_idx=code_layer_idx,
            clean_layer_h=clean_layer_h,
            selected_positions=selected_positions,
            objective=objective,
            yes_ids=yes_ids,
            no_ids=no_ids,
            degraded_fused=degraded_fused,
        )
        pool = _candidate_pool(
            gradient_scores,
            visual_positions,
            non_visual_positions,
            selected_positions,
            cfg.visual_top_k,
        )
        gradient_rank = {
            pos: rank
            for rank, pos in enumerate(
                sorted(gradient_scores, key=lambda p: (gradient_scores[p], -p), reverse=True),
                start=1,
            )
        }
        exact_rows: List[Dict[str, Any]] = []
        for candidate_pos in pool:
            stats = _exact_patch_eval(
                model=model,
                processor=processor,
                device=device,
                vlm_adapter=vlm_adapter,
                degraded_image=degraded_image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                code_layer_idx=code_layer_idx,
                clean_layer_h=clean_layer_h,
                selected_positions=selected_positions,
                candidate_pos=candidate_pos,
                record=record,
                cfg=cfg,
                clean_answer_text=clean_answer_text,
                baseline_clean=baseline_clean,
                baseline_degraded=baseline_degraded,
                degraded_fused=degraded_fused,
            )
            token = token_records[candidate_pos]
            row = {
                "sample_id": cfg.sample_id,
                "severity": cfg.severity,
                "run_type": run_type,
                "requested_layer": human_layer,
                "code_layer_idx": code_layer_idx,
                "step": step_idx,
                "candidate_position": candidate_pos,
                "candidate_token_text": token["token_text"],
                "candidate_token_id": token["token_id"],
                "candidate_span_label": token["span_label"],
                "candidate_token_type": token["token_type"],
                "candidate_token_span": token["token_span"],
                "gradient_score": gradient_scores.get(candidate_pos, float("nan")),
                "gradient_rank": gradient_rank.get(candidate_pos),
                "pool_size": len(pool),
                "selected_positions_before": json.dumps(selected_positions),
                **gradient_meta,
                **stats,
            }
            row["objective_name"] = run_type
            row["objective_value_before"] = current_objective
            row["objective_value_after"] = _objective_value(stats, run_type)
            row["signed_incremental_objective_improvement"] = (
                row["objective_value_after"] - current_objective
            )
            exact_rows.append(row)
            candidate_rows.append(row.copy())

        if not exact_rows:
            stopped_reason = "no_candidates"
            break

        best = max(
            exact_rows,
            key=lambda row: _candidate_sort_key(row, run_type, current_p_clean, current_margin),
        )
        if run_type == "confidence_recovery" and best["p_clean"] <= current_p_clean + RECOVERY_EPSILON:
            stopped_reason = "no_confidence_improvement"
            break

        objective_before = current_objective
        objective_after = _objective_value(best, run_type)
        signed_delta = objective_after - objective_before
        selected_positions.append(int(best["candidate_position"]))
        token = token_records[int(best["candidate_position"])]
        selected_row = {
            "sample_id": cfg.sample_id,
            "severity": cfg.severity,
            "run_type": run_type,
            "requested_layer": human_layer,
            "code_layer_idx": code_layer_idx,
            "step": step_idx,
            "patch_order": step_idx,
            "token_position": int(best["candidate_position"]),
            "token_index": int(best["candidate_position"]),
            "token_id": int(token["token_id"]),
            "token_text": token["token_text"],
            "token_raw": token["token"],
            "token_decoded": token["decoded"],
            "span_label": token["span_label"],
            "token_group": token["token_group"],
            "token_type": token["token_type"],
            "token_span": token["token_span"],
            "selected_positions": json.dumps(selected_positions),
            "selected_token_texts": json.dumps([token_records[p]["token_text"] for p in selected_positions]),
            "selected_token_groups": json.dumps([token_records[p]["token_group"] for p in selected_positions]),
            "gradient_score": best.get("gradient_score"),
            "gradient_rank": best.get("gradient_rank"),
            "pool_size": best.get("pool_size"),
            "predicted_answer": best["predicted_answer"],
            "answer_bucket": best["answer_bucket"],
            "clean_answer_recovered": bool(best["answer_recovered"]),
            "answer_recovery": best["answer_recovery"],
            "p_clean": best["p_clean"],
            "p_yes": best["p_yes"],
            "p_no": best["p_no"],
            "yes_no_margin": best["yes_no_margin"],
            "p_clean_recovery": best["p_clean_recovery"],
            "score": best["score"],
            "top1_probability": best["top1_probability"],
            "answer_geom_mean_probability": best.get("answer_geom_mean_probability"),
            "given_generated_answer_confidence": best.get("answer_geom_mean_probability"),
            "objective_name": run_type,
            "objective_value_before": objective_before,
            "objective_value_after": objective_after,
            "signed_incremental_objective_improvement": signed_delta,
            "positive_incremental_objective_improvement": max(0.0, signed_delta),
            "negative_incremental_objective_improvement": min(0.0, signed_delta),
            "generated_answer_is_yes": best["answer_bucket"] == "yes",
            "generated_answer_is_no": best["answer_bucket"] == "no",
            "generated_answer_matches_clean": best["answer_bucket"] == "yes",
            "generated_answer_matches_corrupted": best["answer_bucket"] == "no",
            "early_stopping_triggered": False,
            "number_patch_steps_used": None,
            "stopped_reason": "",
        }
        selected_rows.append(selected_row)
        current_p_clean = _safe_float(best["p_clean"])
        current_margin = _safe_float(best["yes_no_margin"])
        current_objective = objective_after
        objective_deltas.append(signed_delta)

        if (
            generalized_objective
            and cfg.early_stop_window > 0
            and len(objective_deltas) >= cfg.early_stop_window
            and sum(abs(delta) for delta in objective_deltas[-cfg.early_stop_window :])
            < cfg.early_stop_abs_sum_threshold
        ):
            stopped_reason = "rolling_abs_objective_change_below_threshold"
            break

        if run_type == "answer_recovery" and bool(best["answer_recovered"]):
            stopped_reason = "answer_recovered"
            break
        if run_type == "confidence_recovery" and current_p_clean >= baseline_clean["p_clean"] - RECOVERY_EPSILON:
            stopped_reason = "clean_p_reached"
            break

    if selected_rows:
        early_stopped = stopped_reason == "rolling_abs_objective_change_below_threshold"
        for row in selected_rows:
            row["early_stopping_triggered"] = early_stopped
            row["number_patch_steps_used"] = len(selected_rows)
            row["stopped_reason"] = stopped_reason
    return selected_rows, candidate_rows


def _load_selected_record(patching_dataset: str, cfg: GradientGuidedTokenPatchingConfig) -> Dict[str, Any]:
    with open(patching_dataset) as f:
        data = json.load(f)
    records = data.get("samples", data if isinstance(data, list) else [])
    matches = [record for record in records if str(record.get("sample_id")) == str(cfg.sample_id)]
    if not matches:
        raise ValueError(f"Sample id {cfg.sample_id!r} not found in {patching_dataset}")
    record = matches[0]
    if cfg.require_clean_yes:
        answer = str(record.get("correct_answer", "")).strip().lower()
        clean_pred = _extract_yes_no_answer(str(record.get("clean_model_prediction", ""))).lower()
        if answer != "yes" or clean_pred != "yes" or not bool(record.get("clean_model_correct", False)):
            raise ValueError(
                "This pilot is configured for clean/correct yes POPE samples only; "
                f"got correct_answer={answer!r}, clean_prediction={clean_pred!r}."
            )
    if cfg.require_corrupted_no:
        perturbation = next(
            (
                item
                for item in record.get("visual_perturbations", [])
                if str(item.get("severity")) == cfg.severity
            ),
            None,
        )
        if perturbation is None:
            raise ValueError(f"Sample {cfg.sample_id} has no {cfg.severity!r} perturbation.")
        corrupted_pred = _extract_yes_no_answer(
            str(perturbation.get("model_prediction", ""))
        ).lower()
        if corrupted_pred != "no" or bool(perturbation.get("model_correct", True)):
            raise ValueError(
                "This run requires an incorrect corrupted no answer; "
                f"got prediction={corrupted_pred!r}, "
                f"model_correct={perturbation.get('model_correct')!r}."
            )
    return record


def _severity_image(record: Dict[str, Any], severity: str) -> str:
    for pert in record.get("visual_perturbations", []):
        if pert.get("severity") == severity:
            return str(pert.get("blurred_image_path", ""))
    raise ValueError(f"Sample {record.get('sample_id')} has no {severity!r} perturbation.")


def _save_table(rows: List[Dict[str, Any]], path_base: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    df.to_csv(path_base.with_suffix(".csv"), index=False)


def _format_prob(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "nan"


def _image_with_numbered_visual_patches(
    clean_path: str,
    degraded_path: str,
    selected_steps: List[Dict[str, Any]],
    spans: Dict[str, Any],
    geometry: Dict[str, Any],
    max_side: int = 1100,
) -> Image.Image:
    clean = Image.open(clean_path).convert("RGB")
    scale = min(max_side / max(clean.size), 1.0)
    scaled_rects: List[Tuple[int, int, int, int, int]] = []
    highlighted = clean.copy()
    for row in selected_steps:
        if row["span_label"] != "visual":
            continue
        rect, _ = _visual_rect_for_position(
            clean.size,
            int(row["token_position"]),
            spans,
            geometry,
        )
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        scaled_rects.append((x0, y0, x1, y1, int(row["patch_order"])))

    if scale < 1.0:
        size = (round(highlighted.width * scale), round(highlighted.height * scale))
        highlighted = highlighted.resize(size, Image.Resampling.LANCZOS)
        scaled_rects = [
            (round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale), order)
            for x0, y0, x1, y1, order in scaled_rects
        ]

    draw = ImageDraw.Draw(highlighted, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for x0, y0, x1, y1, order in scaled_rects:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=ORANGE, width=4)
        draw.rectangle((x0 + 4, y0 + 4, x1 - 5, y1 - 5), outline=WHITE, width=1)
        label = str(order)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0] + 8
        label_h = bbox[3] - bbox[1] + 6
        lx0, ly0 = x0 + 4, y0 + 4
        draw.rounded_rectangle(
            (lx0, ly0, lx0 + label_w, ly0 + label_h),
            radius=3,
            fill=(255, 241, 168, 245),
            outline=(180, 83, 9, 255),
            width=1,
        )
        draw.text((lx0 + 4, ly0 + 2), label, fill=(17, 24, 39, 255), font=font)
    clean.close()
    return highlighted


def _token_piece_for_figure(token: Dict[str, Any]) -> str:
    if token["span_label"] == "visual":
        return token["token_text"]
    piece = str(token.get("decoded", ""))
    if piece and all(ch in "\n\r\t" for ch in piece):
        return "".join({"\n": r"\n", "\r": r"\r", "\t": r"\t"}[ch] for ch in piece)
    if piece:
        return piece
    return str(token.get("token_text", token.get("token", "")))


def _prompt_segments(
    token_records: List[Dict[str, Any]],
    selected_steps: List[Dict[str, Any]],
) -> List[Tuple[str, bool]]:
    order_by_pos = {int(row["token_position"]): int(row["patch_order"]) for row in selected_steps}
    visual_selected = [row for row in selected_steps if row["span_label"] == "visual"]
    visual_selected = sorted(visual_selected, key=lambda row: int(row["patch_order"]))
    segments: List[Tuple[str, bool]] = []
    in_visual = False
    for token in token_records:
        pos = int(token["token_position"])
        if token["span_label"] == "visual":
            if not in_visual:
                if visual_selected:
                    segments.append(("[image tokens: ", False))
                    for idx, row in enumerate(visual_selected):
                        if idx:
                            segments.append((", ", False))
                        segments.append((f" [{int(row['patch_order'])}: {row['token_text']}] ", True))
                    segments.append(("]", False))
                else:
                    segments.append(("[image tokens]", False))
                in_visual = True
            continue
        in_visual = False
        piece = _token_piece_for_figure(token)
        if pos in order_by_pos:
            shown = _visible_token_text(piece, str(token.get("token_text", "")))
            segments.append((f" [{order_by_pos[pos]}: {shown}] ", True))
        else:
            segments.append((piece, False))
    return segments


def _wrap_segments(segments: List[Tuple[str, bool]], width: int = 96) -> List[List[Tuple[str, bool]]]:
    lines: List[List[Tuple[str, bool]]] = [[]]
    current_len = 0
    for text, highlighted in segments:
        for part in re.findall(r"\n|[^\S\n]+|[^\s]+", text):
            if part == "\n":
                lines.append([])
                current_len = 0
                continue
            if part.isspace():
                if current_len > 0:
                    lines[-1].append((part, highlighted))
                    current_len += len(part)
                continue
            if current_len > 0 and current_len + len(part) > width:
                lines.append([])
                current_len = 0
            lines[-1].append((part, highlighted))
            current_len += len(part)
    return lines or [[("Prompt unavailable.", False)]]


def _draw_segments(ax: Any, segments: List[Tuple[str, bool]], y: float, fontsize: int = 10) -> float:
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    line_step = 0.07
    for line in _wrap_segments(segments):
        x = 0.02
        for part, highlighted in line:
            bbox = None
            if highlighted:
                bbox = {
                    "boxstyle": "round,pad=0.18",
                    "facecolor": YELLOW,
                    "edgecolor": YELLOW_DARK,
                    "linewidth": 0.8,
                }
            text = ax.text(
                x,
                y,
                part,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=fontsize,
                family="DejaVu Sans Mono",
                bbox=bbox,
            )
            extent = text.get_window_extent(renderer=renderer)
            x += extent.width / ax.bbox.width + 0.003
        y -= line_step
    return y


def _save_figure(
    *,
    output_dir: Path,
    record: Dict[str, Any],
    token_records: List[Dict[str, Any]],
    spans: Dict[str, Any],
    geometry: Dict[str, Any],
    baseline_clean: Dict[str, Any],
    baseline_degraded: Dict[str, Any],
    selected_steps: List[Dict[str, Any]],
    human_layer: int,
    run_type: str,
    cfg: GradientGuidedTokenPatchingConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objective_dir = {
        "maximize_p_yes": "obj_yes",
        "maximize_p_no": "obj_no",
    }.get(run_type)
    fig_dir = output_dir / objective_dir if objective_dir else output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    clean_path = str(record["clean_image_path"])
    degraded_path = _severity_image(record, cfg.severity)
    patched_image = _image_with_numbered_visual_patches(
        clean_path,
        degraded_path,
        selected_steps,
        spans,
        geometry,
    )
    final = selected_steps[-1] if selected_steps else {}
    prompt_segments = _prompt_segments(token_records, selected_steps)

    fig = plt.figure(figsize=(16, 10.5), facecolor="white")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.95, 0.9], hspace=0.24, wspace=0.12)
    image_axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    for ax, title, source, stats in [
        (image_axes[0], "Clean", clean_path, baseline_clean),
        (image_axes[1], "Corrupted", degraded_path, baseline_degraded),
        (image_axes[2], "Final Patched", patched_image, final),
    ]:
        ax.axis("off")
        if isinstance(source, Image.Image):
            image = source
        else:
            image = Image.open(str(source)).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"{title}\nanswer={stats.get('predicted_answer', '')} | "
            f"P(Yes)={_format_prob(stats.get('p_yes'))} | "
            f"P(No)={_format_prob(stats.get('p_no'))}",
            fontsize=12,
            weight="bold",
        )
        if not isinstance(source, Image.Image):
            image.close()

    ax_prompt = fig.add_subplot(gs[1, :])
    ax_prompt.axis("off")
    ax_prompt.text(0.02, 0.98, "Patched prompt tokens", transform=ax_prompt.transAxes, fontsize=12, weight="bold", va="top")
    _draw_segments(ax_prompt, prompt_segments, 0.82, fontsize=9)

    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    generalized_objective = run_type in {"maximize_p_yes", "maximize_p_no"}
    table_rows = []
    for row in selected_steps:
        if generalized_objective:
            table_rows.append(
                [
                    int(row["patch_order"]),
                    f"{row['token_position']} / {row['token_text']}",
                    row["token_span"],
                    row["token_type"],
                    row["predicted_answer"],
                    _format_prob(row["objective_value_before"]),
                    _format_prob(row["objective_value_after"]),
                    _format_prob(row["signed_incremental_objective_improvement"]),
                    _format_prob(row["p_yes"]),
                    _format_prob(row["p_no"]),
                ]
            )
        else:
            table_rows.append(
                [
                    int(row["patch_order"]),
                    f"{row['token_position']} / {row['token_text']}",
                    row["token_group"],
                    row["token_type"],
                    row["predicted_answer"],
                    "yes" if row["clean_answer_recovered"] else "no",
                    _format_prob(row["answer_recovery"]),
                    _format_prob(row["p_clean"]),
                    _format_prob(row["p_clean_recovery"]),
                ]
            )
    if generalized_objective:
        col_labels = ["#", "token", "span", "type", "answer", "obj. before", "obj. after", "signed delta", "P(Yes)", "P(No)"]
        col_widths = [0.035, 0.16, 0.13, 0.16, 0.11, 0.09, 0.09, 0.08, 0.07, 0.07]
    else:
        col_labels = ["#", "token", "group", "type", "answer", "recovered", "ans rec.", "P(Yes)", "P rec."]
        col_widths = [0.04, 0.18, 0.11, 0.18, 0.12, 0.09, 0.08, 0.08, 0.08]
    if not table_rows:
        table_rows = [["-"] * len(col_labels)]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    title_run = run_type.replace("_", " ")
    model_label = "Qwen3.5-9B" if geometry.get("family") == "qwen" else "LLaVA"
    fig.suptitle(
        f"{model_label} POPE sample {cfg.sample_id} | requested layer {human_layer} | {title_run}",
        fontsize=16,
        weight="bold",
    )
    stem = (
        f"layer{human_layer}"
        if generalized_objective
        else f"llava_pope_sample_{cfg.sample_id}_layer_{human_layer}_{run_type}"
    )
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_summary(output_dir: Path, selected_rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    df = pd.DataFrame(selected_rows)
    if df.empty:
        return
    rows: List[Dict[str, Any]] = []
    for layer, layer_df in df.groupby("requested_layer", sort=True):
        by_run = {run_type: grp.sort_values("patch_order") for run_type, grp in layer_df.groupby("run_type")}
        answer = by_run.get("answer_recovery", pd.DataFrame())
        conf = by_run.get("confidence_recovery", pd.DataFrame())
        answer_positions = set(answer["token_position"].astype(int).tolist()) if not answer.empty else set()
        conf_positions = set(conf["token_position"].astype(int).tolist()) if not conf.empty else set()
        union = answer_positions | conf_positions
        intersection = answer_positions & conf_positions
        rows.append(
            {
                "requested_layer": int(layer),
                "answer_groups_order": " -> ".join(answer["token_group"].astype(str).tolist()) if not answer.empty else "",
                "confidence_groups_order": " -> ".join(conf["token_group"].astype(str).tolist()) if not conf.empty else "",
                "answer_positions": json.dumps(answer["token_position"].astype(int).tolist()) if not answer.empty else "[]",
                "confidence_positions": json.dumps(conf["token_position"].astype(int).tolist()) if not conf.empty else "[]",
                "overlap_positions": json.dumps(sorted(intersection)),
                "jaccard": len(intersection) / len(union) if union else float("nan"),
                "answer_success_step": int(answer[answer["clean_answer_recovered"]]["patch_order"].min())
                if not answer.empty and answer["clean_answer_recovered"].any() else None,
                "confidence_success_step": int(conf[conf["p_clean_recovery"] >= 1.0]["patch_order"].min())
                if not conf.empty and (conf["p_clean_recovery"] >= 1.0).any() else None,
                "same_first_patch": (
                    int(answer.iloc[0]["token_position"]) == int(conf.iloc[0]["token_position"])
                    if not answer.empty and not conf.empty else False
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "token_group_comparison_summary.csv", index=False)
    with (output_dir / "token_group_comparison_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Token Group Comparison Summary\n\n")
        for _, row in summary.iterrows():
            f.write(f"## Layer {int(row['requested_layer'])}\n\n")
            f.write(f"- Answer-recovery groups: {row['answer_groups_order'] or 'none'}\n")
            f.write(f"- Confidence-recovery groups: {row['confidence_groups_order'] or 'none'}\n")
            f.write(f"- Exact-position overlap: {row['overlap_positions']} (Jaccard {row['jaccard']:.3f})\n")
            f.write(f"- Same first patch: {bool(row['same_first_patch'])}\n\n")


def _baseline_rows(
    cfg: GradientGuidedTokenPatchingConfig,
    baseline_clean: Dict[str, Any],
    baseline_degraded: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for human_layer, code_layer_idx in zip(cfg.human_layers, cfg.code_layers):
        for run_type in cfg.run_types:
            rows.append(
                {
                    "sample_id": cfg.sample_id,
                    "severity": cfg.severity,
                    "requested_layer": human_layer,
                    "code_layer_idx": code_layer_idx,
                    "objective_name": run_type,
                    "clean_generated_answer": baseline_clean["predicted_answer"],
                    "corrupted_generated_answer": baseline_degraded["predicted_answer"],
                    "clean_answer_bucket": baseline_clean["answer_bucket"],
                    "corrupted_answer_bucket": baseline_degraded["answer_bucket"],
                    "clean_p_yes": baseline_clean["p_yes"],
                    "corrupted_p_yes": baseline_degraded["p_yes"],
                    "clean_p_no": baseline_clean["p_no"],
                    "corrupted_p_no": baseline_degraded["p_no"],
                    "clean_given_generated_answer_confidence": baseline_clean.get(
                        "answer_geom_mean_probability"
                    ),
                    "corrupted_given_generated_answer_confidence": baseline_degraded.get(
                        "answer_geom_mean_probability"
                    ),
                    "clean_top1_probability": baseline_clean.get("top1_probability"),
                    "corrupted_top1_probability": baseline_degraded.get("top1_probability"),
                    "clean_objective_value": _objective_value(baseline_clean, run_type),
                    "corrupted_objective_value": _objective_value(baseline_degraded, run_type),
                }
            )
    return rows


def run_gradient_guided_token_patching(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_dataset: str,
    output_dir: str,
    cfg: GradientGuidedTokenPatchingConfig,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    record = _load_selected_record(patching_dataset, cfg)
    prompt = _format_prompt(cfg.dataset_id, str(record["clean_question"]))
    max_new_tokens = _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)
    clean_answer_text = str(record.get("clean_model_prediction") or record.get("correct_answer") or "yes")

    print("[grad-token] Starting gradient-guided token patching")
    print(f"  sample_id:       {cfg.sample_id}")
    print(f"  severity:        {cfg.severity}")
    print(f"  requested layers:{cfg.human_layers}")
    print(f"  code layers:     {cfg.code_layers}")
    print(f"  visual top-k:    {cfg.visual_top_k}")
    print(f"  max patches:     {cfg.max_patches}")
    print(f"  objectives:      {cfg.run_types}")
    print(
        f"  early stopping:  window={cfg.early_stop_window}, "
        f"abs-sum threshold={cfg.early_stop_abs_sum_threshold}"
    )
    print(f"  output:          {output_path}")

    clean_image = Image.open(record["clean_image_path"]).convert("RGB")
    degraded_path = _severity_image(record, cfg.severity)
    degraded_image = Image.open(degraded_path).convert("RGB")

    all_layer_h, spans = _extract_all_layer_activations(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=vlm_adapter,
        image=clean_image,
        prompt=prompt,
        qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
    )
    clean_inputs = prepare_inputs(
        vlm_adapter,
        processor,
        clean_image,
        prompt,
        device,
        qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
    )
    degraded_inputs = prepare_inputs(
        vlm_adapter,
        processor,
        degraded_image,
        prompt,
        device,
        qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
    )
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
        if use_fused
        else None
    )
    degraded_fused = (
        prepare_qwen_fused_inputs_embeds(
            model,
            vlm_adapter,
            processor,
            degraded_image,
            prompt,
            device,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        if use_fused
        else None
    )
    if degraded_fused is not None:
        degraded_inputs = degraded_fused["inputs"]
    input_ids = clean_inputs["input_ids"][0].detach().cpu()
    token_records = _token_records(processor, input_ids, spans)
    geometry = (
        _qwen_visual_geometry(processor, clean_inputs, spans)
        if vlm_adapter.family == "qwen"
        else _llava_visual_geometry(processor, spans)
    )
    spans["visual_geometry"] = geometry
    all_positions, visual_positions, non_visual_positions = _positions_for_pool(spans)
    yes_ids, no_ids = _pope_token_ids(processor)

    with torch.no_grad():
        clean_fwd = _run_baseline_forward(
            model=model,
            processor=processor,
            device=device,
            vlm_adapter=vlm_adapter,
            image=clean_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            fused_bundle=clean_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        degraded_fwd = _run_baseline_forward(
            model=model,
            processor=processor,
            device=device,
            vlm_adapter=vlm_adapter,
            image=degraded_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            fused_bundle=degraded_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
    baseline_clean = _stats_from_forward(clean_fwd, processor, record, cfg, clean_answer_text)
    baseline_degraded = _stats_from_forward(degraded_fwd, processor, record, cfg, clean_answer_text)

    meta = {
        "sample_id": cfg.sample_id,
        "question": record.get("clean_question", ""),
        "correct_answer": record.get("correct_answer", ""),
        "clean_image_path": record.get("clean_image_path", ""),
        "degraded_image_path": degraded_path,
        "severity": cfg.severity,
        "requested_layers": cfg.human_layers,
        "code_layers": cfg.code_layers,
        "objectives": cfg.run_types,
        "early_stop_window": cfg.early_stop_window,
        "early_stop_abs_sum_threshold": cfg.early_stop_abs_sum_threshold,
        "model_family": vlm_adapter.family,
        "qwen_prefill_empty_thinking": cfg.qwen_prefill_empty_thinking,
        "prompt_len": spans["prompt_len"],
        "visual_start": spans["visual_start"],
        "visual_end": spans["visual_end"],
        "num_positions": len(all_positions),
        "num_visual_positions": len(visual_positions),
        "num_non_visual_positions": len(non_visual_positions),
        "baseline_clean": {k: _json_dumpable(v) for k, v in baseline_clean.items()},
        "baseline_degraded": {k: _json_dumpable(v) for k, v in baseline_degraded.items()},
        "token_spans": {k: _json_dumpable(v) for k, v in spans.items()},
    }
    (output_path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _save_table(token_records, output_path / "prompt_tokens")

    selected_rows_all: List[Dict[str, Any]] = []
    candidate_rows_all: List[Dict[str, Any]] = []
    baseline_rows = _baseline_rows(cfg, baseline_clean, baseline_degraded)
    _save_table(baseline_rows, output_path / "gradient_guided_objective_baselines")
    total_paths = len(cfg.human_layers) * len(cfg.run_types)
    completed_paths = 0
    for human_layer, code_layer_idx in zip(cfg.human_layers, cfg.code_layers):
        clean_layer_h = all_layer_h[code_layer_idx]
        for run_type in cfg.run_types:
            print(f"[grad-token] layer {human_layer} ({code_layer_idx}) / {run_type}", flush=True)
            selected_rows, candidate_rows = _run_one_greedy_path(
                model=model,
                processor=processor,
                device=device,
                vlm_adapter=vlm_adapter,
                degraded_inputs=degraded_inputs,
                degraded_image=degraded_image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                clean_layer_h=clean_layer_h,
                code_layer_idx=code_layer_idx,
                human_layer=human_layer,
                run_type=run_type,
                record=record,
                cfg=cfg,
                token_records=token_records,
                visual_positions=visual_positions,
                non_visual_positions=non_visual_positions,
                yes_ids=yes_ids,
                no_ids=no_ids,
                clean_answer_text=clean_answer_text,
                baseline_clean=baseline_clean,
                baseline_degraded=baseline_degraded,
                degraded_fused=degraded_fused,
            )
            selected_rows_all.extend(selected_rows)
            candidate_rows_all.extend(candidate_rows)
            _save_table(selected_rows_all, output_path / "gradient_guided_selected_steps")
            completed_paths += 1
            if cfg.make_figures:
                _save_figure(
                    output_dir=output_path,
                    record=record,
                    token_records=token_records,
                    spans=spans,
                    geometry=geometry,
                    baseline_clean=baseline_clean,
                    baseline_degraded=baseline_degraded,
                    selected_steps=selected_rows,
                    human_layer=human_layer,
                    run_type=run_type,
                    cfg=cfg,
                )
            if selected_rows:
                last = selected_rows[-1]
                print(
                    "  selected "
                    f"{[row['token_position'] for row in selected_rows]} "
                    f"spans={[row['token_span'] for row in selected_rows]} "
                    f"stop={last['stopped_reason']} "
                    f"p_yes={last['p_yes']:.4f} p_no={last['p_no']:.4f}",
                    flush=True,
                )
            elapsed = time.time() - t0
            eta_seconds = (elapsed / completed_paths) * (total_paths - completed_paths)
            print(
                f"  progress={completed_paths}/{total_paths} "
                f"elapsed={elapsed / 60:.1f}m eta={eta_seconds / 60:.1f}m",
                flush=True,
            )

    _save_table(selected_rows_all, output_path / "gradient_guided_selected_steps")
    _save_table(candidate_rows_all, output_path / "gradient_guided_candidate_audit")
    if {"answer_recovery", "confidence_recovery"}.issubset(set(cfg.run_types)):
        _write_summary(output_path, selected_rows_all)

    clean_image.close()
    degraded_image.close()
    elapsed = time.time() - t0
    print(f"[grad-token] Finished in {elapsed:.1f}s")
