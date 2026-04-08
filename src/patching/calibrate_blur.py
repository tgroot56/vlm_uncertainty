"""Calibrate Gaussian-blur sigma values against a VLM.

Sweep sigma ∈ {1, 2, …, 40} on a held-out subset, measure accuracy at each
level, and select the sigma values that best match three target accuracy-drop
bands (mild, moderate, severe).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.patching.perturbations import apply_gaussian_blur
from src.labeling.vqa import score_vqa
from src.labeling.multiple_choice import score_multiple_choice
from src.uq_dataset_generation.tasks.pope import _extract_yes_no_answer


@dataclass
class CalibrationResult:
    """Stores the outcome of a blur-sigma calibration run."""
    baseline_accuracy: float
    sigma_accuracy_curve: Dict[int, float]  # sigma -> accuracy
    selected_sigmas: Dict[str, float]       # severity -> sigma
    selected_accuracies: Dict[str, float]   # severity -> accuracy
    selected_drops: Dict[str, float]        # severity -> relative accuracy drop (fraction of baseline)
    num_samples: int


# Target relative accuracy drops (fraction of baseline accuracy).
_DROP_TARGETS: Dict[str, float] = {
    "mild":     0.05,
    "moderate": 0.15,
    "severe":   0.30,
}


def _score_sample(pred_answer: str, sample: Dict[str, Any], dataset_id: str = "") -> float:
    """Score a single prediction using dataset-specific logic."""
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

    result = score_vqa(
        pred_answer=pred_answer,
        gt_answers=sample.get("gt_answers"),
        gt_answer_single=sample.get("gt_answer"),
    )
    return result.score


def _predict_open_ended(
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int = 4,
) -> str:
    """Run a lightweight greedy decode (no feature extraction needed)."""
    from src.vlm import prepare_inputs

    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def _select_best_sigma(
    curve: Dict[int, float],
    baseline: float,
    target: float,
) -> Tuple[int, float, float]:
    """Pick the sigma whose relative accuracy drop is closest to *target*.

    Relative drop is defined as ``(baseline - acc) / baseline``.

    Returns (sigma, accuracy, relative_drop).
    """
    best_sigma = None
    best_dist = float("inf")
    best_acc = baseline
    for sigma, acc in sorted(curve.items()):
        rel_drop = (baseline - acc) / baseline if baseline > 0 else 0.0
        dist = abs(rel_drop - target)
        if dist < best_dist:
            best_dist = dist
            best_sigma = sigma
            best_acc = acc
    rel_drop = (baseline - best_acc) / baseline if baseline > 0 else 0.0
    return best_sigma, best_acc, rel_drop


@torch.inference_mode()
def calibrate_blur_sigmas(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: Any,
    samples: List[Dict[str, Any]],
    sigma_range: range = range(1, 41),
    max_new_tokens: int = 4,
    dataset_id: str = "",
) -> CalibrationResult:
    """Run the full calibration sweep and return the selected sigmas.

    Args:
        model, processor, device, vlm_adapter: loaded VLM.
        samples: list of prepared samples (must contain image, prompt, gt_answers/gt_answer).
        sigma_range: integer sigma values to sweep (default 1..40).
        max_new_tokens: generation budget per sample.
        dataset_id: dataset identifier for dataset-specific scoring.

    Returns:
        CalibrationResult with selected sigma per severity level.
    """
    num_samples = len(samples)

    # --- baseline accuracy (clean images) ---
    print(f"[calibrate] Computing baseline accuracy on {num_samples} samples …")
    baseline_scores: List[float] = []
    for sample in tqdm(samples, desc="baseline"):
        pred = _predict_open_ended(
            model, processor, device, vlm_adapter,
            image=sample["image"],
            prompt=sample["prompt"],
            max_new_tokens=max_new_tokens,
        )
        baseline_scores.append(_score_sample(pred, sample, dataset_id))
    baseline_accuracy = sum(baseline_scores) / len(baseline_scores)
    print(f"[calibrate] Baseline accuracy: {baseline_accuracy:.4f}")

    # --- sweep sigma values ---
    sigma_accuracy: Dict[int, float] = {}
    for sigma in sigma_range:
        scores: List[float] = []
        for sample in tqdm(samples, desc=f"sigma={sigma:2d}"):
            blurred = apply_gaussian_blur(sample["image"], sigma=float(sigma))
            pred = _predict_open_ended(
                model, processor, device, vlm_adapter,
                image=blurred,
                prompt=sample["prompt"],
                max_new_tokens=max_new_tokens,
            )
            scores.append(_score_sample(pred, sample, dataset_id))
        acc = sum(scores) / len(scores)
        rel_drop = (baseline_accuracy - acc) / baseline_accuracy if baseline_accuracy > 0 else 0.0
        sigma_accuracy[sigma] = acc
        print(f"[calibrate] sigma={sigma:2d}  acc={acc:.4f}  rel_drop={rel_drop:+.4f} ({rel_drop*100:+.1f}%)")
        if rel_drop > max(_DROP_TARGETS.values()):
            print(f"[calibrate] rel_drop exceeded {max(_DROP_TARGETS.values())*100:.0f}% — stopping sweep early.")
            break

    # --- select sigmas for each severity band ---
    selected_sigmas: Dict[str, float] = {}
    selected_accuracies: Dict[str, float] = {}
    selected_drops: Dict[str, float] = {}

    for severity, target in _DROP_TARGETS.items():
        sigma, acc, rel_drop = _select_best_sigma(sigma_accuracy, baseline_accuracy, target)
        selected_sigmas[severity] = float(sigma)
        selected_accuracies[severity] = acc
        selected_drops[severity] = rel_drop
        print(
            f"[calibrate] {severity:>8s}: sigma={sigma}, acc={acc:.4f}, "
            f"rel_drop={rel_drop:+.4f} ({rel_drop*100:+.1f}%, target={target*100:.0f}%)"
        )

    return CalibrationResult(
        baseline_accuracy=baseline_accuracy,
        sigma_accuracy_curve=sigma_accuracy,
        selected_sigmas=selected_sigmas,
        selected_accuracies=selected_accuracies,
        selected_drops=selected_drops,
        num_samples=num_samples,
    )


def save_calibration(result: CalibrationResult, path: str) -> None:
    """Persist calibration result as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "baseline_accuracy": result.baseline_accuracy,
        "sigma_accuracy_curve": {str(k): v for k, v in result.sigma_accuracy_curve.items()},
        "selected_sigmas": result.selected_sigmas,
        "selected_accuracies": result.selected_accuracies,
        "selected_drops": result.selected_drops,
        "num_samples": result.num_samples,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[calibrate] Saved calibration to {path}")


def load_calibration(path: str) -> CalibrationResult:
    """Load a previously saved calibration result."""
    with open(path) as f:
        data = json.load(f)
    return CalibrationResult(
        baseline_accuracy=data["baseline_accuracy"],
        sigma_accuracy_curve={int(k): v for k, v in data["sigma_accuracy_curve"].items()},
        selected_sigmas=data["selected_sigmas"],
        selected_accuracies=data["selected_accuracies"],
        selected_drops=data["selected_drops"],
        num_samples=data["num_samples"],
    )


def save_blur_examples(
    result: CalibrationResult,
    samples: List[Dict[str, Any]],
    output_dir: str,
    num_examples: int = 2,
) -> None:
    """Save side-by-side blur examples for visual inspection.

    For each of the first ``num_examples`` samples, saves the clean image and
    one blurred version per severity level under::

        <output_dir>/blur_examples/
            sample_<id>_clean.png
            sample_<id>_mild_sigma<s>.png
            sample_<id>_moderate_sigma<s>.png
            sample_<id>_severe_sigma<s>.png

    Args:
        result: calibration result containing the selected sigmas.
        samples: the calibration sample list (images must be PIL Images).
        output_dir: root output directory (blur_examples/ is created inside it).
        num_examples: how many samples to visualise (default 2).
    """
    from PIL import Image as PILImage

    examples_dir = os.path.join(output_dir, "blur_examples")
    os.makedirs(examples_dir, exist_ok=True)

    chosen = samples[:num_examples]
    severities = ["mild", "moderate", "severe"]

    for i, sample in enumerate(chosen):
        image: PILImage.Image = sample["image"]
        sample_tag = f"sample_{i:02d}"

        # Clean
        clean_path = os.path.join(examples_dir, f"{sample_tag}_clean.png")
        image.save(clean_path)

        # One blurred version per severity
        for severity in severities:
            sigma = result.selected_sigmas.get(severity)
            if sigma is None:
                continue
            blurred = apply_gaussian_blur(image, sigma=float(sigma))
            fname = f"{sample_tag}_{severity}_sigma{int(sigma)}.png"
            blurred.save(os.path.join(examples_dir, fname))

        print(
            f"[calibrate] Saved blur examples for {sample_tag} "
            f"(sigmas: {result.selected_sigmas}) → {examples_dir}"
        )
