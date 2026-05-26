from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm_uncertainty_matplotlib")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from .config import (
        DATASET_LABELS,
        DATASETS,
        METRICS_FOR_PLOT,
        MIN_POS_SAMPLES,
        MODEL_SPECS,
        OUTPUT_ROOT,
        QWEN_VISUAL_MIN_FRACTION,
        REPO_ROOT,
        SPAN_DISPLAY,
        all_parquet_paths,
        parquet_path,
        patching_dataset_path,
        supervised_metadata_path,
    )
except ImportError:  # pragma: no cover - useful when running scripts directly.
    from config import (  # type: ignore
        DATASET_LABELS,
        DATASETS,
        METRICS_FOR_PLOT,
        MIN_POS_SAMPLES,
        MODEL_SPECS,
        OUTPUT_ROOT,
        QWEN_VISUAL_MIN_FRACTION,
        REPO_ROOT,
        SPAN_DISPLAY,
        all_parquet_paths,
        parquet_path,
        patching_dataset_path,
        supervised_metadata_path,
    )


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    from src.cli.plot_sweep_filtered import (  # type: ignore
        METRICS,
        aggregate_recovery_ci,
        compute_recovery,
        _prepare_positional_sweep,
    )
except Exception:  # pragma: no cover - kept as a transparent fallback.
    METRICS = ["top1_probability", "output_entropy", "score"]
    EPSILON = 1e-8

    def compute_recovery(patched_vals, clean_vals, degraded_vals):
        delta = clean_vals - degraded_vals
        raw = (patched_vals - degraded_vals) / (delta + np.sign(delta + EPSILON) * EPSILON)
        return np.clip(raw, -5, 5)

    def aggregate_recovery_ci(frame: pd.DataFrame, metric: str, n: int = 1000, seed: int = 42):
        cols = [metric, f"clean_{metric}", f"degraded_{metric}"]
        values = frame[cols].dropna()
        if len(values) == 0:
            return float("nan"), float("nan"), float("nan")
        patched = values[metric].to_numpy(dtype=float)
        clean = values[f"clean_{metric}"].to_numpy(dtype=float)
        degraded = values[f"degraded_{metric}"].to_numpy(dtype=float)
        valid = ~(np.isnan(patched) | np.isnan(clean) | np.isnan(degraded))
        patched_mean = patched[valid].mean()
        clean_mean = clean[valid].mean()
        degraded_mean = degraded[valid].mean()
        delta = clean_mean - degraded_mean
        eps = EPSILON if delta >= 0 else -EPSILON
        mean = float(np.clip((patched_mean - degraded_mean) / (delta + eps), -5, 5))
        return mean, float("nan"), float("nan")

    def _prepare_positional_sweep(
        parquet_path: str,
        min_delta: float = 0.0,
        min_pos_samples: int = MIN_POS_SAMPLES,
        span_min_pos_samples: dict[str, int] | None = None,
    ):
        df = pd.read_parquet(parquet_path)
        patched = df[df["condition"].str.startswith("patched")].copy()
        for cond in ("clean", "degraded"):
            bl = (
                df[df["condition"] == cond][["sample_id", "severity"] + METRICS]
                .rename(columns={m: f"{cond}_{m}" for m in METRICS})
            )
            patched = patched.merge(bl, on=["sample_id", "severity"])
        for metric in METRICS:
            patched[f"recovery_{metric}"] = compute_recovery(
                patched[metric].values,
                patched[f"clean_{metric}"].values,
                patched[f"degraded_{metric}"].values,
            )
        span_starts = (
            patched.groupby(["sample_id", "span_label"])["token_position"]
            .min()
            .reset_index(name="span_start")
        )
        patched = patched.merge(span_starts, on=["sample_id", "span_label"])
        patched["span_rel_pos"] = (patched["token_position"] - patched["span_start"]).astype(int)
        pos_counts = (
            patched.groupby(["span_label", "span_rel_pos", "severity"])["sample_id"]
            .nunique()
            .reset_index(name="n")
        )
        span_min_pos_samples = span_min_pos_samples or {}
        valid_idx = {
            (span, rel_pos)
            for (span, rel_pos), n in pos_counts.groupby(["span_label", "span_rel_pos"])["n"].min().items()
            if n >= span_min_pos_samples.get(span, min_pos_samples)
        }
        patched = patched[patched.apply(lambda r: (r["span_label"], r["span_rel_pos"]) in valid_idx, axis=1)]
        canonical_rows = []
        span_regions = {}
        x_pos = 0
        for span in ["text_pre", "visual", "question", "assistant_marker"]:
            valid_rel = sorted({rp for sl, rp in valid_idx if sl == span})
            if not valid_rel:
                continue
            xs = list(range(x_pos, x_pos + len(valid_rel)))
            span_regions[span] = xs
            for rel_pos, cx in zip(valid_rel, xs):
                canonical_rows.append({"span_label": span, "span_rel_pos": rel_pos, "canonical_x": cx})
            x_pos += len(valid_rel) + 1
        canonical_df = pd.DataFrame(canonical_rows)
        patched = patched.merge(canonical_df, on=["span_label", "span_rel_pos"], how="inner")
        canonical_xs = sorted(patched["canonical_x"].unique().astype(int))
        sevs = sorted(patched["severity"].dropna().unique())
        n_total = df[df["condition"] == "clean"]["sample_id"].nunique()
        return patched, canonical_xs, span_regions, sevs, n_total, patched["sample_id"].nunique()


ANSWER_STRIP_RE = re.compile(r"^\s*<think>.*?</think>\s*", flags=re.DOTALL)


def stripped_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    text = ANSWER_STRIP_RE.sub("", text, count=1).strip()
    return text


def norm_answer(value: Any) -> str:
    return " ".join(stripped_answer(value).lower().split())


def confidence_column(df: pd.DataFrame) -> str:
    for col in ("answer_geom_prob", "answer_geom_mean_probability", "top1_probability"):
        if col in df.columns:
            return col
    raise KeyError("No confidence column found. Expected answer_geom_prob, answer_geom_mean_probability, or top1_probability.")


def _span_minimums(model_label: str, parquet: Path) -> dict[str, int] | None:
    if "qwen" not in model_label.lower():
        return None
    n_clean = pd.read_parquet(parquet, columns=["condition", "sample_id"])
    n_clean = n_clean[n_clean["condition"] == "clean"]["sample_id"].nunique()
    return {"visual": max(MIN_POS_SAMPLES, math.ceil(QWEN_VISUAL_MIN_FRACTION * n_clean))}


@lru_cache(maxsize=16)
def raw_parquet(model_label: str, dataset: str) -> pd.DataFrame:
    path = parquet_path(model_label, dataset)
    df = pd.read_parquet(path)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


@lru_cache(maxsize=16)
def plot_rows(model_label: str, dataset: str) -> pd.DataFrame:
    path = parquet_path(model_label, dataset)
    span_min_pos_samples = _span_minimums(model_label, path)
    patched, _, _, _, _, _ = _prepare_positional_sweep(
        str(path),
        min_delta=0.0,
        min_pos_samples=MIN_POS_SAMPLES,
        span_min_pos_samples=span_min_pos_samples,
    )
    patched = patched.copy()
    patched["sample_id"] = patched["sample_id"].astype(str)

    raw = raw_parquet(model_label, dataset)
    conf_col = confidence_column(raw)
    baseline_cols = [
        "sample_id",
        "severity",
        "predicted_answer",
        "score",
        "top1_probability",
        conf_col,
    ]
    for cond in ("clean", "degraded"):
        base = (
            raw[raw["condition"] == cond][baseline_cols]
            .drop_duplicates(["sample_id", "severity"])
            .rename(
                columns={
                    "predicted_answer": f"{cond}_answer",
                    "score": f"{cond}_correctness",
                    "top1_probability": f"{cond}_plot_confidence",
                    conf_col: f"{cond}_confidence",
                }
            )
        )
        patched = patched.merge(base, on=["sample_id", "severity"], how="left")

    patched = patched.rename(
        columns={
            "predicted_answer": "patched_answer",
            "score": "patched_correctness",
            "top1_probability": "patched_plot_confidence",
            conf_col: "patched_confidence",
            "recovery_score": "accuracy_recovery",
            "recovery_top1_probability": "confidence_recovery",
        }
    )
    if "patched_confidence" not in patched.columns:
        patched["patched_confidence"] = patched["patched_plot_confidence"]
    for cond in ("clean", "degraded"):
        if f"{cond}_confidence" not in patched.columns:
            patched[f"{cond}_confidence"] = patched[f"{cond}_plot_confidence"]

    patched["model"] = model_label
    patched["dataset"] = dataset
    patched["dataset_label"] = DATASET_LABELS.get(dataset, dataset)
    patched["span_display"] = patched["span_label"].map(lambda x: SPAN_DISPLAY.get(str(x), str(x)))
    patched["patched_token_position"] = patched["token_position"].astype(int)
    patched["patched_position_index"] = patched["position_index"].astype(int)
    patched["clean_answer_stripped"] = patched["clean_answer"].map(stripped_answer)
    patched["degraded_answer_stripped"] = patched["degraded_answer"].map(stripped_answer)
    patched["patched_answer_stripped"] = patched["patched_answer"].map(stripped_answer)
    patched["answer_changed"] = patched["patched_answer_stripped"] != patched["degraded_answer_stripped"]
    patched["answer_changed_normalized"] = (
        patched["patched_answer"].map(norm_answer) != patched["degraded_answer"].map(norm_answer)
    )
    patched["correctness_improved"] = patched["patched_correctness"] > patched["degraded_correctness"]
    patched["confidence_improved"] = patched["patched_confidence"] > patched["degraded_confidence"]
    patched["confidence_exceeds_clean"] = patched["patched_confidence"] > patched["clean_confidence"]

    final_pos = patched.groupby(["sample_id", "severity"])["patched_token_position"].transform("max")
    patched["is_final_prompt_token"] = patched["patched_token_position"] == final_pos
    patched["row_id"] = (
        patched["model"].astype(str)
        + " | "
        + patched["dataset"].astype(str)
        + " | sample "
        + patched["sample_id"].astype(str)
        + " | "
        + patched["span_label"].astype(str)
        + " pos "
        + patched["patched_token_position"].astype(str)
    )
    return patched


@lru_cache(maxsize=1)
def all_plot_rows() -> pd.DataFrame:
    frames = []
    for (model_label, dataset), path in all_parquet_paths().items():
        if path.exists():
            frames.append(plot_rows(model_label, dataset))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=16)
def records_by_id(model_label: str, dataset: str) -> dict[str, dict[str, Any]]:
    path = patching_dataset_path(model_label, dataset)
    with path.open() as f:
        data = json.load(f)
    samples = data.get("samples", data if isinstance(data, list) else [])
    return {str(sample["sample_id"]): sample for sample in samples}


def sample_record(model_label: str, dataset: str, sample_id: str) -> dict[str, Any]:
    return records_by_id(model_label, dataset).get(str(sample_id), {})


def blurred_image_path(record: dict[str, Any], severity: str) -> str:
    for perturbation in record.get("visual_perturbations", []) or []:
        if perturbation.get("severity") == severity:
            return str(perturbation.get("blurred_image_path", ""))
    return ""


def ground_truth_text(record: dict[str, Any], dataset: str) -> str:
    accepted = record.get("gt_answers") or []
    correct = record.get("correct_answer", "")
    if accepted:
        counts = pd.Series([str(x) for x in accepted]).value_counts()
        answers = ", ".join(f"{answer} ({count})" for answer, count in counts.items())
        return f"{answers}"
    return str(correct)


def sample_token_span_metadata(model_label: str, dataset: str, sample_id: str) -> dict[str, Any] | None:
    path = supervised_metadata_path(model_label, dataset, str(sample_id))
    if path is None or not path.exists():
        return None
    with path.open() as f:
        metadata = json.load(f)
    return metadata.get("token_spans")


def inferred_spans_from_rows(rows: pd.DataFrame, sample_id: str) -> dict[str, Any]:
    sub = rows[rows["sample_id"].astype(str) == str(sample_id)]
    spans: dict[str, Any] = {}
    for span, grp in sub.dropna(subset=["span_label"]).groupby("span_label"):
        positions = grp["patched_token_position"].astype(int)
        spans[f"{span}_start"] = int(positions.min())
        spans[f"{span}_end"] = int(positions.max() + 1)
        if span == "visual":
            spans["num_visual_tokens"] = int(positions.max() - positions.min() + 1)
            spans["visual_start"] = int(positions.min())
            spans["visual_end"] = int(positions.max() + 1)
    if not spans and not sub.empty:
        max_pos = int(sub["patched_token_position"].max())
        spans = {"fullspan_start": 0, "fullspan_end": max_pos + 1, "prompt_len": max_pos + 1}
    return spans


def _classify_position(pos: int, spans: dict[str, Any]) -> str:
    if spans.get("visual_start", -1) <= pos < spans.get("visual_end", -1):
        return "visual"
    if pos < spans.get("visual_start", -1):
        return "text_pre"
    if pos < spans.get("question_start", spans.get("visual_end", 0)):
        return "text_post"
    if pos < spans.get("question_end", spans.get("prompt_len", 10**9)):
        return "question"
    return "assistant_marker"


@lru_cache(maxsize=4)
def load_processor(model_label: str):
    from transformers import AutoProcessor

    spec = MODEL_SPECS[model_label]
    refs = []
    if spec.processor_ref:
        refs.append(spec.processor_ref)
    refs.append(spec.model_id)
    last_error: Exception | None = None
    for ref in refs:
        try:
            return AutoProcessor.from_pretrained(ref, trust_remote_code=True, local_files_only=True)
        except Exception as exc:  # pragma: no cover - depends on cluster cache.
            last_error = exc
            continue
    raise RuntimeError(f"Could not load cached processor for {model_label}: {last_error}")


def _dict_dimension(value: Any, key: str, default: int) -> int:
    if isinstance(value, dict):
        raw = value.get(key, default)
    else:
        raw = getattr(value, key, default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _llava_visual_geometry(processor: Any, spans: dict[str, Any]) -> dict[str, Any] | None:
    image_processor = getattr(processor, "image_processor", None)
    crop_size = getattr(image_processor, "crop_size", {}) if image_processor is not None else {}
    size = getattr(image_processor, "size", {}) if image_processor is not None else {}
    patch_size = int(getattr(processor, "patch_size", 0) or getattr(image_processor, "patch_size", 0) or 14)
    crop_h = _dict_dimension(crop_size, "height", 336)
    crop_w = _dict_dimension(crop_size, "width", 336)
    shortest_edge = _dict_dimension(size, "shortest_edge", min(crop_h, crop_w))
    if patch_size <= 0 or crop_h <= 0 or crop_w <= 0:
        return None
    rows = crop_h // patch_size
    cols = crop_w // patch_size
    mappable = rows * cols
    if mappable <= 0 or mappable != int(spans.get("num_visual_tokens", 0)):
        return None
    return {
        "family": "llava",
        "source": f"LLaVA CLIP {rows}x{cols} crop geometry",
        "grid_rows": rows,
        "grid_cols": cols,
        "mappable_visual_tokens": mappable,
        "visual_token_offset": 0,
        "patch_size": patch_size,
        "crop_height": crop_h,
        "crop_width": crop_w,
        "shortest_edge": shortest_edge,
    }


def _qwen_visual_geometry(processor: Any, inputs: dict[str, Any], spans: dict[str, Any]) -> dict[str, Any] | None:
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return None
    try:
        grid = [int(x) for x in image_grid_thw.detach().cpu().reshape(-1, 3)[0].tolist()]
    except Exception:
        return None
    image_processor = getattr(processor, "image_processor", None)
    merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
    t, grid_h, grid_w = grid
    rows = grid_h // merge_size
    cols = grid_w // merge_size
    mappable = t * rows * cols
    expected = max(0, int(spans.get("num_visual_tokens", 0)) - 2)
    if merge_size <= 0 or rows <= 0 or cols <= 0 or t != 1 or mappable != expected:
        return None
    return {
        "family": "qwen",
        "source": "Qwen image_grid_thw-derived merged grid",
        "image_grid_thw": grid,
        "merge_size": merge_size,
        "grid_rows": rows,
        "grid_cols": cols,
        "mappable_visual_tokens": mappable,
        "visual_token_offset": 1,
        "vision_start_rel_pos": 0,
        "vision_end_rel_pos": int(spans.get("num_visual_tokens", 0)) - 1,
    }


def _visual_display_for_token(
    model_label: str,
    token: str,
    position: int,
    spans: dict[str, Any],
) -> tuple[str, str | None]:
    visual_start = int(spans["visual_start"])
    visual_rel = position - visual_start
    if "qwen" in model_label.lower():
        if token == "<|vision_start|>":
            return "[VIS_START]", "boundary_start"
        if token == "<|vision_end|>":
            return "[VIS_END]", "boundary_end"
        image_rel = visual_rel - 1
        return f"[V{image_rel:03d}]", "image"
    return f"[V{visual_rel:03d}]", "image"


def exact_prompt_tokens(
    model_label: str,
    dataset: str,
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    notes: list[str] = []
    try:
        import torch
        from src.patching.collect_activations import _format_prompt
        from src.uq_dataset_generation.token_spans import build_token_spans
        from src.vlm.adapters import get_vlm_adapter, prepare_inputs

        spec = MODEL_SPECS[model_label]
        adapter = get_vlm_adapter(spec.model_id)
        processor = load_processor(model_label)
        image_path = record.get("clean_image_path", "")
        if not image_path:
            raise FileNotFoundError("clean_image_path missing from patching dataset JSON")
        image = Image.open(image_path).convert("RGB")
        prompt = _format_prompt(dataset, str(record.get("clean_question", "")))
        inputs = prepare_inputs(
            adapter,
            processor,
            image,
            prompt,
            torch.device("cpu"),
            qwen_prefill_empty_thinking=spec.qwen_prefill_empty_thinking,
        )
        image.close()
        input_ids = inputs["input_ids"][0]
        prompt_len = int(inputs["input_ids"].shape[1])
        spans = build_token_spans(
            model=None,
            processor=processor,
            vlm_adapter=adapter,
            input_ids=input_ids,
            prompt_len=prompt_len,
            lm_hidden_len=prompt_len,
        )
        if adapter.family == "llava":
            geometry = _llava_visual_geometry(processor, spans)
        elif adapter.family == "qwen":
            geometry = _qwen_visual_geometry(processor, inputs, spans)
        else:
            geometry = None
        if geometry is not None:
            spans["visual_geometry"] = geometry
        token_strings = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        tokens = []
        for idx, tok in enumerate(token_strings):
            span = _classify_position(idx, spans)
            visual_role = None
            if span == "visual":
                display, visual_role = _visual_display_for_token(model_label, str(tok), idx, spans)
            else:
                display = str(tok)
            tokens.append(
                {
                    "position": idx,
                    "token": str(tok),
                    "display": display,
                    "span_label": span,
                    "visual_role": visual_role,
                    "exact": True,
                }
            )
        return tokens, spans, notes
    except Exception as exc:
        notes.append(
            "Exact tokenizer reconstruction unavailable: "
            f"{type(exc).__name__}: {str(exc)[:220]}"
        )
        return [], {}, notes


def fallback_prompt_tokens(
    rows: pd.DataFrame,
    row: pd.Series,
    record: dict[str, Any],
    spans: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    notes = [
        "Prompt tokens are approximate placeholders because exact tokenizer reconstruction was unavailable.",
    ]
    sample_id = str(row["sample_id"])
    spans = dict(spans or inferred_spans_from_rows(rows, sample_id))
    max_pos = int(max(row["patched_token_position"], spans.get("prompt_len", spans.get("fullspan_end", 0)) - 1))
    if "fullspan_end" not in spans:
        spans["fullspan_end"] = max_pos + 1
    if "prompt_len" not in spans:
        spans["prompt_len"] = spans["fullspan_end"]
    tokens = []
    for pos in range(int(spans.get("fullspan_start", 0)), int(spans["fullspan_end"])):
        span = _classify_position(pos, spans)
        if span == "visual":
            display = f"[V{pos - int(spans.get('visual_start', 0)):03d}]"
        elif span == "question":
            display = f"[question_{pos - int(spans.get('question_start', 0))}]"
        elif span == "assistant_marker":
            display = f"[suffix_{pos - int(spans.get('question_end', 0))}]"
        else:
            display = f"[prefix_{pos}]"
        tokens.append(
            {
                "position": pos,
                "token": display,
                "display": display,
                "span_label": span,
                "exact": False,
            }
        )
    return tokens, spans, notes


def prompt_tokens_for_row(rows: pd.DataFrame, row: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    model_label = str(row["model"])
    dataset = str(row["dataset"])
    sample_id = str(row["sample_id"])
    record = sample_record(model_label, dataset, sample_id)
    tokens, spans, notes = exact_prompt_tokens(model_label, dataset, record)
    if tokens:
        return tokens, spans, notes
    saved_spans = sample_token_span_metadata(model_label, dataset, sample_id)
    fallback_tokens, fallback_spans, fallback_notes = fallback_prompt_tokens(rows, row, record, saved_spans)
    return fallback_tokens, fallback_spans, notes + fallback_notes


def token_window(tokens: list[dict[str, Any]], patched_pos: int, radius: int = 8) -> list[dict[str, Any]]:
    if not tokens:
        return []
    positions = [int(t["position"]) for t in tokens]
    if patched_pos not in positions:
        return tokens[: min(len(tokens), radius * 2 + 1)]
    idx = positions.index(patched_pos)
    lo = max(0, idx - radius)
    hi = min(len(tokens), idx + radius + 1)
    return tokens[lo:hi]


def token_html(tokens: list[dict[str, Any]], patched_pos: int) -> str:
    pieces = []
    for tok in tokens:
        span = html.escape(str(tok.get("span_label", "")))
        display = html.escape(str(tok.get("display", tok.get("token", ""))))
        pos = int(tok["position"])
        cls = f"token span-{span}"
        if pos == patched_pos:
            cls += " patched"
        pieces.append(f'<span class="{cls}" title="position {pos}, span {span}">{display}</span>')
    return (
        "<style>"
        ".prompt-tokens{max-height:260px;overflow:auto;line-height:2.15;padding:10px;border:1px solid #d1d5db;border-radius:6px;background:#ffffff;}"
        ".token{display:inline-block;padding:1px 5px;margin:2px;border-radius:4px;border:1px solid #e5e7eb;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;}"
        ".span-text_pre{background:#f8fafc;}"
        ".span-visual{background:#e0f2fe;}"
        ".span-question{background:#ffedd5;}"
        ".span-assistant_marker{background:#e0e7ff;}"
        ".span-text_post{background:#f1f5f9;}"
        ".patched{background:#fde68a !important;border:2px solid #b45309;color:#111827;font-weight:700;}"
        "</style>"
        f"<div class='prompt-tokens'>{''.join(pieces)}</div>"
    )


def _regular_grid_rect(
    image_size: tuple[int, int],
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x0 = math.floor(col * width / cols)
    y0 = math.floor(row * height / rows)
    x1 = math.ceil((col + 1) * width / cols)
    y1 = math.ceil((row + 1) * height / rows)
    return x0, y0, x1, y1


def _resize_shortest_edge_size(width: int, height: int, shortest_edge: int) -> tuple[int, int, float]:
    if width <= 0 or height <= 0 or shortest_edge <= 0:
        return width, height, 1.0
    scale = shortest_edge / min(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale)), scale


def _llava_clip_rect(
    image_size: tuple[int, int],
    row: int,
    col: int,
    rows: int,
    cols: int,
    geometry: dict[str, Any],
) -> tuple[int, int, int, int]:
    width, height = image_size
    crop_w = int(geometry.get("crop_width", 336))
    crop_h = int(geometry.get("crop_height", 336))
    shortest = int(geometry.get("shortest_edge", min(crop_w, crop_h)))
    resized_w, resized_h, scale = _resize_shortest_edge_size(width, height, shortest)
    crop_left = max(0.0, (resized_w - crop_w) / 2.0)
    crop_top = max(0.0, (resized_h - crop_h) / 2.0)
    cell_x0 = col * crop_w / cols
    cell_y0 = row * crop_h / rows
    cell_x1 = (col + 1) * crop_w / cols
    cell_y1 = (row + 1) * crop_h / rows
    x0 = math.floor((crop_left + cell_x0) / scale)
    y0 = math.floor((crop_top + cell_y0) / scale)
    x1 = math.ceil((crop_left + cell_x1) / scale)
    y1 = math.ceil((crop_top + cell_y1) / scale)
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def visual_token_rectangle(
    image_size: tuple[int, int],
    visual_rel_pos: int,
    visual_token_count: int,
    visual_geometry: dict[str, Any] | None,
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any], str]:
    if not visual_geometry:
        return (
            None,
            {"reason": "missing_geometry"},
            "No model-specific visual-token geometry was recovered, so no defensible pixel highlight is drawn.",
        )

    rows = int(visual_geometry.get("grid_rows", 0) or 0)
    cols = int(visual_geometry.get("grid_cols", 0) or 0)
    offset = int(visual_geometry.get("visual_token_offset", 0) or 0)
    mappable = int(visual_geometry.get("mappable_visual_tokens", 0) or 0)
    image_rel = int(visual_rel_pos) - offset
    meta = {
        "grid_rows": rows,
        "grid_cols": cols,
        "visual_rel_pos": int(visual_rel_pos),
        "image_rel_pos": image_rel,
        "source": visual_geometry.get("source", "model-specific visual-token geometry"),
        "family": visual_geometry.get("family", ""),
    }

    if rows <= 0 or cols <= 0 or mappable <= 0:
        meta["reason"] = "invalid_geometry"
        return None, meta, "Recovered visual-token geometry was incomplete, so no pixel highlight is drawn."
    if image_rel < 0:
        meta["reason"] = "boundary"
        return None, meta, "This Qwen visual prompt token is a <|vision_start|> boundary marker, not an image patch."
    if image_rel >= mappable:
        meta["reason"] = "boundary"
        return None, meta, "This Qwen visual prompt token is a <|vision_end|> boundary marker, not an image patch."
    if visual_token_count and image_rel >= visual_token_count:
        meta["reason"] = "out_of_range"
        return None, meta, "The patched visual-token index is outside the recovered image-token grid."

    row = min(rows - 1, max(0, image_rel // cols))
    col = min(cols - 1, max(0, image_rel % cols))
    meta.update({"grid_row": row, "grid_col": col})

    family = str(visual_geometry.get("family", ""))
    if family == "llava":
        rect = _llava_clip_rect(image_size, row, col, rows, cols, visual_geometry)
        note = (
            f"Highlight uses {visual_geometry.get('source', 'LLaVA CLIP crop geometry')} "
            "projected back to the original image."
        )
    elif family == "qwen":
        rect = _regular_grid_rect(image_size, row, col, rows, cols)
        note = (
            f"Highlight uses {visual_geometry.get('source', 'Qwen image_grid_thw-derived merged grid')} "
            "over the image aspect ratio."
        )
    else:
        meta["reason"] = "unsupported_family"
        return None, meta, "Unsupported model family for visual-token pixel highlighting."

    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        meta["reason"] = "empty_rect"
        return None, meta, "The recovered visual-token rectangle was empty after projection."
    return rect, meta, note


def highlighted_degraded_image(
    model_label: str,
    clean_path: str,
    degraded_path: str,
    visual_rel_pos: int,
    visual_token_count: int,
    visual_geometry: dict[str, Any] | None = None,
    max_side: int = 900,
) -> tuple[Image.Image | None, dict[str, Any], str]:
    if not clean_path or not degraded_path:
        return None, {}, "Clean or degraded image path is missing."
    clean = Image.open(clean_path).convert("RGB")
    degraded = Image.open(degraded_path).convert("RGB")
    if degraded.size != clean.size:
        degraded = degraded.resize(clean.size, Image.Resampling.BICUBIC)
    rect, meta, note = visual_token_rectangle(
        clean.size,
        visual_rel_pos,
        visual_token_count,
        visual_geometry,
    )
    meta["model"] = model_label
    if rect is None:
        clean.close()
        degraded.close()
        return None, meta, note
    x0, y0, x1, y1 = rect
    scale = min(max_side / max(clean.size), 1.0)
    if scale < 1.0:
        size = (round(clean.width * scale), round(clean.height * scale))
        clean = clean.resize(size, Image.Resampling.LANCZOS)
        degraded = degraded.resize(size, Image.Resampling.LANCZOS)
        x0 = round(x0 * scale)
        y0 = round(y0 * scale)
        x1 = round(x1 * scale)
        y1 = round(y1 * scale)
    highlighted = degraded.copy()
    highlighted.paste(clean.crop((x0, y0, x1, y1)), (x0, y0))
    draw = ImageDraw.Draw(highlighted, "RGBA")
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(245, 158, 11, 255), width=5)
    draw.rectangle((x0 + 5, y0 + 5, x1 - 6, y1 - 6), outline=(255, 255, 255, 230), width=2)
    meta["rect"] = [x0, y0, x1, y1]
    return highlighted, meta, note


def format_answer_with_conf(answer: Any, confidence: Any) -> str:
    try:
        conf = float(confidence)
        return f"{stripped_answer(answer)} ({conf:.2f})"
    except Exception:
        return f"{stripped_answer(answer)} (NA)"


def latex_escape(text: Any) -> str:
    s = stripped_answer(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def compact_row_dict(row: pd.Series, record: dict[str, Any] | None = None, category: str | None = None) -> dict[str, Any]:
    record = record or {}
    return {
        "category": category or "",
        "model": row["model"],
        "dataset": row["dataset"],
        "sample_id": row["sample_id"],
        "severity": row["severity"],
        "patched_span": row["span_label"],
        "patched_span_display": row.get("span_display", row["span_label"]),
        "patched_token_position": int(row["patched_token_position"]),
        "span_relative_position": int(row["span_rel_pos"]),
        "position_index": int(row["patched_position_index"]),
        "clean_answer": row["clean_answer_stripped"],
        "degraded_answer": row["degraded_answer_stripped"],
        "patched_answer": row["patched_answer_stripped"],
        "ground_truth": ground_truth_text(record, str(row["dataset"])) if record else "",
        "clean_correctness": float(row["clean_correctness"]),
        "degraded_correctness": float(row["degraded_correctness"]),
        "patched_correctness": float(row["patched_correctness"]),
        "clean_confidence": float(row["clean_confidence"]),
        "degraded_confidence": float(row["degraded_confidence"]),
        "patched_confidence": float(row["patched_confidence"]),
        "accuracy_recovery": float(row["accuracy_recovery"]),
        "confidence_recovery": float(row["confidence_recovery"]),
        "answer_changed": bool(row["answer_changed"]),
        "correctness_improved": bool(row["correctness_improved"]),
        "confidence_improved": bool(row["confidence_improved"]),
        "confidence_exceeds_clean": bool(row["confidence_exceeds_clean"]),
    }


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "validation").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "paper_examples").mkdir(parents=True, exist_ok=True)
