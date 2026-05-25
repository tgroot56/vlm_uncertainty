"""
Extend an existing patching dataset with an additional ``"extreme"`` blur
severity. The extreme level is not calibrated — it uses a fixed large sigma
(default: :data:`src.patching.calibrate_blur.EXTREME_BLUR_SIGMA`, = 50.0) that
collapses the input image to an effectively uniform colour field.

For every sample already in ``patching_dataset_<dataset>.json`` this CLI:
  1. Skips the sample if an ``"extreme"`` entry is already present (idempotent).
  2. Loads the clean image from disk.
  3. Applies the fixed-sigma Gaussian blur.
  4. Runs a single forward pass for the blurred prediction + top-1 probability.
  5. Appends a new ``{"severity": "extreme", ...}`` entry to
     ``visual_perturbations`` and optionally saves the blurred PNG.

It also records the extreme sigma in ``metadata.calibration.selected_sigmas``
and, if a sidecar ``calibration.json`` exists, patches it in place.

Usage::

    python -m src.cli.extend_patching_dataset_extreme \
        --patching_dataset /path/to/patching_dataset_coco-qa-vi.json \
        --vlm llava-hf/llava-1.5-7b-hf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from ..storage_paths import configure_hf_cache_env

configure_hf_cache_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append an 'extreme' blur severity to an existing patching dataset."
    )
    parser.add_argument(
        "--patching_dataset", type=str, required=True,
        help="Path to an existing patching_dataset_<dataset>.json.",
    )
    parser.add_argument(
        "--vlm", type=str, required=True,
        help="HuggingFace model id used to build the original dataset.",
    )
    parser.add_argument(
        "--extreme_sigma", type=float, default=None,
        help="Sigma for the extreme Gaussian blur. Default: EXTREME_BLUR_SIGMA (50.0).",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Where to write the extended JSON. Default: overwrite --patching_dataset.",
    )
    parser.add_argument(
        "--no_save_images", action="store_true",
        help="Skip saving the extreme-blur PNG to disk (metadata only).",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=50,
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4,
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _atomic_write_json(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".extend_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _patch_sidecar_calibration(dataset_path: str, extreme_sigma: float) -> None:
    """Update the neighbouring calibration.json (if present) with extreme sigma."""
    cal_path = os.path.join(os.path.dirname(dataset_path), "calibration.json")
    if not os.path.exists(cal_path):
        return
    with open(cal_path) as f:
        cal = json.load(f)
    cal.setdefault("selected_sigmas", {})["extreme"] = float(extreme_sigma)
    cal.setdefault("selected_accuracies", {})["extreme"] = None
    cal.setdefault("selected_drops", {})["extreme"] = None
    _atomic_write_json(cal, cal_path)
    print(f"[extend] Patched sidecar calibration: {cal_path}")


def main() -> None:
    args = parse_args()

    # Lazy imports to keep --help fast
    from src.patching.calibrate_blur import EXTREME_BLUR_SIGMA
    from src.patching.build_dataset import (
        _predict, _save_blurred_image, _score_sample,
    )
    from src.patching.perturbations import apply_gaussian_blur
    from src.vlm.adapters import load_model

    extreme_sigma = float(args.extreme_sigma) if args.extreme_sigma is not None else float(EXTREME_BLUR_SIGMA)
    print(f"[extend] Extreme sigma: {extreme_sigma}")

    # ---- load existing dataset ----
    print(f"[extend] Loading patching dataset: {args.patching_dataset}")
    with open(args.patching_dataset) as f:
        data = json.load(f)
    metadata = data["metadata"]
    records: List[Dict[str, Any]] = data["samples"]
    dataset_id = metadata.get("dataset_id", "vqa-v2")
    print(f"[extend] dataset={dataset_id}  samples={len(records)}")

    output_path = args.output_path or args.patching_dataset
    output_dir = os.path.dirname(output_path)

    # ---- how many samples still need extreme? ----
    def _has_extreme(rec: Dict[str, Any]) -> bool:
        return any(
            str(vp.get("severity", "")).lower() == "extreme"
            for vp in rec.get("visual_perturbations", [])
        )

    pending = [r for r in records if not _has_extreme(r)]
    print(f"[extend] {len(pending)} samples need an 'extreme' entry "
          f"({len(records) - len(pending)} already done).")
    if not pending:
        print("[extend] Nothing to do.")
        # Still ensure metadata/sidecar reflect extreme.
        metadata.setdefault("calibration", {}).setdefault("selected_sigmas", {})["extreme"] = extreme_sigma
        _atomic_write_json(data, output_path)
        _patch_sidecar_calibration(output_path, extreme_sigma)
        return

    # ---- load model ----
    print(f"[extend] Loading model: {args.vlm}")
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)

    # ---- iterate ----
    t0 = time.time()
    processed = 0
    for rec in tqdm(records, desc="extend extreme"):
        if _has_extreme(rec):
            continue

        clean_path = rec.get("clean_image_path", "")
        if not clean_path or not os.path.exists(clean_path):
            print(f"[extend] WARNING: missing clean image for sample "
                  f"{rec.get('sample_id')}; skipping.")
            continue

        image = Image.open(clean_path).convert("RGB")
        prompt_question = rec.get("clean_question", "")

        # The clean prompt stored in the record is just the question text; the
        # original build step applied dataset-specific prompt formatting via
        # the sample's "prompt" field. We re-use the same _predict helper that
        # the original build used, passing the already-formatted prompt that
        # was persisted implicitly: because the JSON only stores the question,
        # we rely on the fact that _predict accepts whatever prompt string the
        # caller provides. The original build used ``sample["prompt"]``; the
        # JSON schema does not store it, so we reconstruct via dataset_id-
        # aware formatting below.
        prompt = _format_prompt_for_dataset(dataset_id, prompt_question)

        blurred = apply_gaussian_blur(image, sigma=extreme_sigma)
        pred, top1_prob = _predict(
            model, processor, device, vlm_adapter,
            image=blurred, prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        score = _score_sample(pred, _record_to_sample(rec), dataset_id)
        correct = score >= 1.0

        blurred_path = ""
        if not args.no_save_images:
            sample_id = str(rec.get("sample_id", ""))
            blurred_path = _save_blurred_image(
                blurred, extreme_sigma, sample_id, output_dir,
            )

        rec.setdefault("visual_perturbations", []).append({
            "severity": "extreme",
            "blurred_image_path": blurred_path,
            "sigma": extreme_sigma,
            "model_prediction": pred,
            "model_correct": correct,
            "top1_probability": top1_prob,
        })
        processed += 1

        if processed % args.checkpoint_every == 0:
            metadata.setdefault("calibration", {}).setdefault("selected_sigmas", {})["extreme"] = extreme_sigma
            _atomic_write_json(data, output_path)

    # Final write
    metadata.setdefault("calibration", {}).setdefault("selected_sigmas", {})["extreme"] = extreme_sigma
    _atomic_write_json(data, output_path)
    _patch_sidecar_calibration(output_path, extreme_sigma)

    # Summary
    extreme_top1 = [
        vp["top1_probability"] for r in records
        for vp in r.get("visual_perturbations", [])
        if vp.get("severity") == "extreme"
    ]
    extreme_acc = sum(
        1 for r in records
        for vp in r.get("visual_perturbations", [])
        if vp.get("severity") == "extreme" and vp.get("model_correct")
    ) / max(len(extreme_top1), 1)
    elapsed = time.time() - t0
    print(
        f"[extend] Done. +{processed} extreme entries in {elapsed:.1f}s | "
        f"extreme acc={extreme_acc:.3f} | "
        f"extreme top-1 prob mean={sum(extreme_top1)/max(len(extreme_top1),1):.3f}"
    )


def _record_to_sample(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a minimal ``sample`` dict for ``_score_sample``."""
    return {
        "gt_answers": rec.get("gt_answers", []),
        "gt_answer": rec.get("correct_answer", ""),
        "gt_key": rec.get("gt_key", ""),
    }


def _format_prompt_for_dataset(dataset_id: str, question: str) -> str:
    """Rebuild the prompt string the original build used, per dataset.

    The JSON schema persists only the question, not the full prompt. The
    prompt formatting used at build time is dataset-specific; we re-apply it
    here so the extreme-blur forward pass uses the same template.
    """
    from src.patching.collect_activations import _format_prompt
    return _format_prompt(dataset_id, question)


if __name__ == "__main__":
    main()
