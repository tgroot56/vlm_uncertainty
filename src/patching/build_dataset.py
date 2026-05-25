"""Build paired clean-degraded dataset for the activation-patching experiment.

Pipeline:
  1. Load a VQA-style dataset via the shared data registry.
  2. (Optional) Run or load blur-sigma calibration on a held-out subset.
  3. For every sample:
       a. Run inference on the clean image  → store prediction + correctness.
       b. For each severity level, blur the image and run inference → store.
  4. Save the full paired dataset as a single JSON file.

The output JSON follows the schema specified in the experiment design document.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.labeling.vqa import score_vqa
from src.labeling.multiple_choice import score_multiple_choice
from src.uq_dataset_generation.tasks.pope import _extract_yes_no_answer
from src.patching.calibrate_blur import (
    CalibrationResult,
    calibrate_blur_sigmas,
    load_calibration,
    save_calibration,
)
from src.patching.perturbations import apply_gaussian_blur


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_image_for_png(image: Image.Image) -> Image.Image:
    """Convert non-PNG-friendly PIL modes to RGB before saving."""
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")

def _score_sample(pred_answer: str, sample: Dict[str, Any], dataset_id: str) -> float:
    """Score a prediction using dataset-specific logic matching the probing experiments."""
    if dataset_id == "imagenet-r":
        result = score_multiple_choice(
            pred_letter=pred_answer,
            gt_letter=sample.get("gt_key", ""),
            normalize=True,
        )
        return result.score

    if dataset_id == "pope":
        parsed = _extract_yes_no_answer(pred_answer)
        result = score_vqa(
            pred_answer=parsed,
            gt_answer_single=sample.get("gt_answer"),
        )
        return result.score

    # VQA-v2 (soft accuracy) and COCO-QA-VI (exact match via single GT)
    result = score_vqa(
        pred_answer=pred_answer,
        gt_answers=sample.get("gt_answers"),
        gt_answer_single=sample.get("gt_answer"),
    )
    return result.score


def _predict(
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int = 4,
) -> tuple[str, float]:
    from src.vlm import (
        prepare_inputs,
        qwen_patching_max_new_tokens,
        strip_qwen_thinking_prefix,
    )

    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)
    effective_max_new_tokens = qwen_patching_max_new_tokens(vlm_adapter, max_new_tokens)
    out = model.generate(
        **inputs,
        max_new_tokens=effective_max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if vlm_adapter.family == "qwen":
        pred = strip_qwen_thinking_prefix(pred)

    # Top-1 probability from the first post-thinking generated token's logits
    top1_prob = 0.0
    if out.scores:
        think_end_id = 248069  # '</think>' token for Qwen3.5
        logits_idx = 0
        if vlm_adapter.family == "qwen":
            gen_token_ids = gen_ids[0].tolist()
            for i, tid in enumerate(gen_token_ids):
                if tid == think_end_id and i + 1 < len(gen_token_ids):
                    logits_idx = i + 1
                    # skip any newline tokens that commonly follow </think>
                    while (
                        logits_idx + 1 < len(gen_token_ids)
                        and gen_token_ids[logits_idx] == 271  # '\n\n'
                    ):
                        logits_idx += 1
                    break
            logits_idx = min(logits_idx, len(out.scores) - 1)
        first_logits = out.scores[logits_idx][0].float()
        top1_prob = float(F.softmax(first_logits, dim=0).max().item())

    return pred, top1_prob


def _save_blurred_image(
    image: Image.Image,
    sigma: float,
    sample_id: str,
    output_dir: str,
) -> str:
    """Save a blurred image to disk and return its path."""
    img_dir = os.path.join(output_dir, "blurred_images")
    os.makedirs(img_dir, exist_ok=True)
    fname = f"{sample_id}_sigma{sigma:.1f}.png"
    path = os.path.join(img_dir, fname)
    _prepare_image_for_png(image).save(path)
    return path


def _save_clean_image(
    image: Image.Image,
    sample_id: str,
    output_dir: str,
) -> str:
    """Save the clean image to disk and return its path."""
    img_dir = os.path.join(output_dir, "clean_images")
    os.makedirs(img_dir, exist_ok=True)
    fname = f"{sample_id}.png"
    path = os.path.join(img_dir, fname)
    _prepare_image_for_png(image).save(path)
    return path


def _make_sample_id(sample: Dict[str, Any]) -> str:
    """Derive a stable string id for a sample."""
    qid = sample.get("question_id")
    if qid is not None:
        return str(qid)
    sid = sample.get("sample_id")
    if sid is not None:
        return str(sid)
    return str(sample["idx"])


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

@torch.inference_mode()
def build_patching_dataset(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: Any,
    samples: List[Dict[str, Any]],
    calibration: CalibrationResult,
    output_dir: str,
    model_id: str,
    dataset_id: str,
    save_images: bool = True,
    max_new_tokens: int = 4,
    checkpoint_every: int = 50,
) -> List[Dict[str, Any]]:
    """Build the paired dataset for visual perturbations.

    Args:
        model, processor, device, vlm_adapter: loaded VLM.
        samples: prepared dataset samples.
        calibration: calibrated sigma values per severity level.
        output_dir: root directory for outputs (images + JSON).
        model_id: HuggingFace model identifier (for metadata).
        dataset_id: dataset identifier (for metadata).
        save_images: whether to persist blurred/clean images to disk.
        max_new_tokens: generation budget per sample.
        checkpoint_every: save intermediate checkpoint every N samples.

    Returns:
        List of paired-sample dicts matching the experiment schema.
    """
    os.makedirs(output_dir, exist_ok=True)
    severities = sorted(calibration.selected_sigmas.keys())
    checkpoint_path = os.path.join(output_dir, f"patching_dataset_{dataset_id}_checkpoint.json")

    # Resume from checkpoint if available
    completed_ids: set = set()
    records: List[Dict[str, Any]] = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        records = checkpoint_data.get("samples", [])
        completed_ids = {r["sample_id"] for r in records}
        print(f"[build] Resuming from checkpoint: {len(completed_ids)} samples already done.")

    print(f"[build] Building patching dataset: {len(samples)} samples, "
          f"severities={severities}")
    print(f"[build] Calibrated sigmas: {calibration.selected_sigmas}")

    t0 = time.time()
    for i, sample in enumerate(tqdm(samples, desc="building paired dataset")):
        sample_id = _make_sample_id(sample)
        if sample_id in completed_ids:
            continue

        image: Image.Image = sample["image"]
        prompt: str = sample["prompt"]
        question: str = sample.get("question", prompt)

        # --- clean prediction ---
        clean_pred, clean_top1_prob = _predict(
            model, processor, device, vlm_adapter,
            image=image, prompt=prompt, max_new_tokens=max_new_tokens,
        )
        clean_score = _score_sample(clean_pred, sample, dataset_id)
        clean_correct = clean_score >= 1.0

        clean_image_path = ""
        if save_images:
            clean_image_path = _save_clean_image(image, sample_id, output_dir)

        # --- visual perturbations ---
        visual_perturbations: List[Dict[str, Any]] = []
        for severity in severities:
            sigma = calibration.selected_sigmas[severity]
            blurred = apply_gaussian_blur(image, sigma=sigma)

            blur_pred, blur_top1_prob = _predict(
                model, processor, device, vlm_adapter,
                image=blurred, prompt=prompt, max_new_tokens=max_new_tokens,
            )
            blur_score = _score_sample(blur_pred, sample, dataset_id)
            blur_correct = blur_score >= 1.0

            blurred_image_path = ""
            if save_images:
                blurred_image_path = _save_blurred_image(
                    blurred, sigma, sample_id, output_dir,
                )

            visual_perturbations.append({
                "severity": severity,
                "blurred_image_path": blurred_image_path,
                "sigma": sigma,
                "model_prediction": blur_pred,
                "model_correct": blur_correct,
                "top1_probability": blur_top1_prob,
            })

        # --- build record ---
        gt_answers = sample.get("gt_answers", [])
        correct_answer = sample.get("gt_answer", "")
        if not correct_answer and gt_answers:
            correct_answer = gt_answers[0]

        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "clean_image_path": clean_image_path,
            "clean_question": question,
            "correct_answer": correct_answer,
            "gt_answers": gt_answers,
            "clean_model_prediction": clean_pred,
            "clean_model_correct": clean_correct,
            "clean_top1_probability": clean_top1_prob,
            "visual_perturbations": visual_perturbations,
            "textual_perturbations": [],  # placeholder for step 2
        }
        # Preserve dataset-specific GT fields (e.g. gt_key for ImageNet-R)
        if "gt_key" in sample:
            record["gt_key"] = sample["gt_key"]
        records.append(record)
        completed_ids.add(sample_id)

        # Periodic checkpoint
        if (len(records) % checkpoint_every == 0):
            _save_checkpoint(records, calibration, model_id, dataset_id, checkpoint_path)

    elapsed = time.time() - t0

    # Summary stats
    clean_acc = sum(r["clean_model_correct"] for r in records) / max(len(records), 1)
    clean_top1 = sum(r["clean_top1_probability"] for r in records) / max(len(records), 1)
    blur_top1_by_sev: Dict[str, List[float]] = {}
    for r in records:
        for vp in r["visual_perturbations"]:
            sev = str(vp["severity"])
            blur_top1_by_sev.setdefault(sev, []).append(vp["top1_probability"])
    blur_top1_str = ", ".join(
        f"{sev}: {sum(vals)/len(vals):.3f}" for sev, vals in sorted(blur_top1_by_sev.items())
    )
    print(
        f"[build] Finished {len(records)} samples in {elapsed:.1f}s | "
        f"clean acc={clean_acc:.3f} | clean top-1 prob={clean_top1:.3f} | "
        f"blurred top-1 prob [{blur_top1_str}]"
    )

    # --- save final dataset ---
    final_path = os.path.join(output_dir, f"patching_dataset_{dataset_id}.json")
    _save_final(records, calibration, model_id, dataset_id, final_path)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return records


def _save_checkpoint(
    records: List[Dict[str, Any]],
    calibration: CalibrationResult,
    model_id: str,
    dataset_id: str,
    path: str,
) -> None:
    payload = _build_payload(records, calibration, model_id, dataset_id)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_final(
    records: List[Dict[str, Any]],
    calibration: CalibrationResult,
    model_id: str,
    dataset_id: str,
    path: str,
) -> None:
    payload = _build_payload(records, calibration, model_id, dataset_id)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[build] Saved final dataset to {path}")


def _build_payload(
    records: List[Dict[str, Any]],
    calibration: CalibrationResult,
    model_id: str,
    dataset_id: str,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "num_samples": len(records),
            "calibration": {
                "baseline_accuracy": calibration.baseline_accuracy,
                "selected_sigmas": calibration.selected_sigmas,
                "selected_accuracies": calibration.selected_accuracies,
                "selected_drops": calibration.selected_drops,
                "calibration_num_samples": calibration.num_samples,
            },
        },
        "samples": records,
    }
