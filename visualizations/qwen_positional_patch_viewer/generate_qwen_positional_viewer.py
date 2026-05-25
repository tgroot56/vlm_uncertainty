#!/usr/bin/env python3
"""Build a static qualitative viewer for Qwen positional patching examples."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw


MODEL_ROOT = Path("/projects/prjs2014/patching/Qwen_Qwen3-5-9B")
SWEEP_ROOT = Path(
    "/projects/prjs2014/patching/positional_patching_results_including_visual/"
    "Qwen_Qwen3-5-9B"
)
DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]
DATASET_LABELS = {
    "vqa-v2": "VQA-v2",
    "coco-qa-vi": "COCO-QA-VI",
    "imagenet-r": "ImageNet-R",
    "pope": "POPE",
}
OUTPUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUTPUT_DIR / "assets"
EXAMPLES_PER_DATASET = 10
LAYER = 17
MAX_IMAGE_SIDE = 760


def _load_records(dataset: str) -> dict[str, dict[str, Any]]:
    path = MODEL_ROOT / dataset / f"patching_dataset_{dataset}.json"
    with path.open() as f:
        data = json.load(f)
    return {str(sample["sample_id"]): sample for sample in data["samples"]}


def _extreme_blur_path(sample: dict[str, Any]) -> str:
    for pert in sample.get("visual_perturbations", []):
        if pert.get("severity") == "extreme":
            return pert["blurred_image_path"]
    raise KeyError(f"sample {sample.get('sample_id')} has no extreme perturbation")


def _single_baseline(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    cols = ["sample_id", "predicted_answer", "score", "top1_probability"]
    return (
        df[df["condition"] == condition][cols]
        .drop_duplicates("sample_id")
        .rename(
            columns={
                "predicted_answer": f"{condition}_answer",
                "score": f"{condition}_score",
                "top1_probability": f"{condition}_confidence",
            }
        )
    )


def _select_examples(dataset: str, records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    parquet_path = (
        SWEEP_ROOT
        / dataset
        / "extreme_answer_confidence"
        / "positional_patching_results.parquet"
    )
    df = pd.read_parquet(parquet_path)
    df["sample_id"] = df["sample_id"].astype(str)

    patched = df[
        (df["severity"] == "extreme")
        & (df["span_label"] == "visual")
        & (df["condition"].str.startswith("patched"))
    ].copy()

    clean = _single_baseline(df, "clean")
    degraded = _single_baseline(df, "degraded")
    patched = patched.merge(clean, on="sample_id").merge(degraded, on="sample_id")

    visual_bounds = (
        patched.groupby("sample_id")["token_position"]
        .agg(visual_start="min", visual_end_inclusive="max")
        .reset_index()
    )
    patched = patched.merge(visual_bounds, on="sample_id")
    patched["visual_rel_pos"] = (
        patched["token_position"] - patched["visual_start"]
    ).astype(int)
    patched["visual_token_count_estimate"] = (
        patched["visual_end_inclusive"] - patched["visual_start"] + 1
    ).astype(int)

    patched["confidence_delta"] = (
        patched["top1_probability"] - patched["degraded_confidence"]
    )
    patched["score_delta"] = patched["score"] - patched["degraded_score"]
    patched["confidence_recovery"] = (
        patched["top1_probability"] - patched["degraded_confidence"]
    ) / (
        patched["clean_confidence"] - patched["degraded_confidence"]
    ).replace(0, pd.NA)
    patched["answer_changed"] = (
        patched["predicted_answer"].astype(str)
        != patched["degraded_answer"].astype(str)
    )
    patched["clean_target_available"] = patched["clean_score"] > patched["degraded_score"]
    patched["sample_has_image"] = patched["sample_id"].map(
        lambda sid: sid in records
        and Path(records[sid]["clean_image_path"]).exists()
        and Path(_extreme_blur_path(records[sid])).exists()
    )
    patched = patched[patched["sample_has_image"]]

    best_by_sample = (
        patched.sort_values(
            [
                "clean_target_available",
                "score_delta",
                "answer_changed",
                "confidence_delta",
                "top1_probability",
            ],
            ascending=[False, False, False, False, False],
        )
        .groupby("sample_id", as_index=False)
        .head(1)
    )

    answer_change = best_by_sample[best_by_sample["answer_changed"]]
    improved = answer_change[
        (answer_change["score_delta"] > 0) | (answer_change["confidence_delta"] > 0.05)
    ]
    remainder = best_by_sample.drop(improved.index, errors="ignore")
    selected = pd.concat(
        [
            improved.sort_values(
                ["score_delta", "confidence_delta"], ascending=[False, False]
            ),
            remainder.sort_values(
                ["score_delta", "confidence_delta"], ascending=[False, False]
            ),
        ]
    )
    return selected.head(EXAMPLES_PER_DATASET).copy()


def _grid_for_token_count(token_count: int, width: int, height: int) -> tuple[int, int]:
    aspect = max(width / max(height, 1), 0.2)
    cols = max(1, round(math.sqrt(max(token_count, 1) * aspect)))
    rows = max(1, math.ceil(max(token_count, 1) / cols))
    return cols, rows


def _resize_pair(clean_path: str, blurred_path: str) -> tuple[Image.Image, Image.Image]:
    clean = Image.open(clean_path).convert("RGB")
    blurred = Image.open(blurred_path).convert("RGB")
    if blurred.size != clean.size:
        blurred = blurred.resize(clean.size, Image.Resampling.BICUBIC)

    scale = min(MAX_IMAGE_SIDE / max(clean.size), 1.0)
    if scale < 1.0:
        new_size = (round(clean.width * scale), round(clean.height * scale))
        clean = clean.resize(new_size, Image.Resampling.LANCZOS)
        blurred = blurred.resize(new_size, Image.Resampling.LANCZOS)
    return clean, blurred


def _make_patch_visual(
    clean: Image.Image,
    blurred: Image.Image,
    visual_rel_pos: int,
    visual_token_count: int,
) -> tuple[Image.Image, dict[str, Any]]:
    cols, rows = _grid_for_token_count(visual_token_count, clean.width, clean.height)
    row = min(rows - 1, max(0, visual_rel_pos // cols))
    col = min(cols - 1, max(0, visual_rel_pos % cols))

    x0 = math.floor(col * clean.width / cols)
    y0 = math.floor(row * clean.height / rows)
    x1 = math.ceil((col + 1) * clean.width / cols)
    y1 = math.ceil((row + 1) * clean.height / rows)

    patched = blurred.copy()
    patched.paste(clean.crop((x0, y0, x1, y1)), (x0, y0))
    draw = ImageDraw.Draw(patched, "RGBA")
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(124, 58, 237, 255), width=4)
    draw.rectangle((x0 + 4, y0 + 4, x1 - 5, y1 - 5), outline=(255, 255, 255, 210), width=2)

    return patched, {
        "grid_cols": cols,
        "grid_rows": rows,
        "grid_col": col,
        "grid_row": row,
        "rect": [x0, y0, x1, y1],
    }


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=88, optimize=True)


def _format_answer(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _build_example(dataset: str, row: pd.Series, sample: dict[str, Any]) -> dict[str, Any]:
    uid = f"{dataset}_{row.sample_id}_pos{int(row.token_position)}"
    rel_asset_dir = Path("assets") / dataset
    clean_asset = rel_asset_dir / f"{uid}_clean.jpg"
    blurred_asset = rel_asset_dir / f"{uid}_blurred.jpg"
    restored_asset = rel_asset_dir / f"{uid}_restored.jpg"

    clean_img, blurred_img = _resize_pair(
        sample["clean_image_path"],
        _extreme_blur_path(sample),
    )
    restored_img, patch_meta = _make_patch_visual(
        clean_img,
        blurred_img,
        int(row.visual_rel_pos),
        int(row.visual_token_count_estimate),
    )

    _save_image(clean_img, OUTPUT_DIR / clean_asset)
    _save_image(blurred_img, OUTPUT_DIR / blurred_asset)
    _save_image(restored_img, OUTPUT_DIR / restored_asset)

    clean_answer = _format_answer(row.clean_answer)
    degraded_answer = _format_answer(row.degraded_answer)
    patched_answer = _format_answer(row.predicted_answer)

    return {
        "uid": uid,
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "sample_id": str(row.sample_id),
        "question": str(sample.get("clean_question", "")).strip(),
        "correct_answer": str(sample.get("correct_answer", "")).strip(),
        "gt_key": str(sample.get("gt_key", "")).strip(),
        "clean_image": clean_asset.as_posix(),
        "blurred_image": blurred_asset.as_posix(),
        "restored_image": restored_asset.as_posix(),
        "clean_answer": clean_answer,
        "degraded_answer": degraded_answer,
        "patched_answer": patched_answer,
        "answer_changed": patched_answer != degraded_answer,
        "clean_score": float(row.clean_score),
        "degraded_score": float(row.degraded_score),
        "patched_score": float(row.score),
        "score_delta": float(row.score_delta),
        "clean_confidence": float(row.clean_confidence),
        "degraded_confidence": float(row.degraded_confidence),
        "patched_confidence": float(row.top1_probability),
        "confidence_delta": float(row.confidence_delta),
        "confidence_recovery": (
            None
            if pd.isna(row.confidence_recovery)
            else float(row.confidence_recovery)
        ),
        "layer": LAYER,
        "severity": "extreme",
        "token_position": int(row.token_position),
        "visual_rel_pos": int(row.visual_rel_pos),
        "visual_token_count_estimate": int(row.visual_token_count_estimate),
        "patch_meta": patch_meta,
    }


def main() -> None:
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for dataset in DATASETS:
        records = _load_records(dataset)
        selected = _select_examples(dataset, records)
        counts[dataset] = len(selected)
        for row in selected.itertuples(index=False):
            sample = records[str(row.sample_id)]
            examples.append(_build_example(dataset, row, sample))

    payload = {
        "title": "Qwen3.5-9B Positional Patching Viewer",
        "model": "Qwen/Qwen3.5-9B",
        "layer": LAYER,
        "severity": "extreme",
        "datasets": [{"id": ds, "label": DATASET_LABELS[ds], "count": counts[ds]} for ds in DATASETS],
        "examples": examples,
        "notes": [
            "The restored image is a visualization of the patched visual-token region.",
            "The actual intervention patches a hidden activation at one prompt token position, not pixels.",
        ],
    }
    with (OUTPUT_DIR / "samples.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(examples)} examples to {OUTPUT_DIR}")
    for dataset in DATASETS:
        print(f"  {dataset}: {counts[dataset]}")


if __name__ == "__main__":
    main()
