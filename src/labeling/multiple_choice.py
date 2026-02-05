# src/labeling/multiple_choice.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CorrectnessResult:
    """Result of correctness scoring for a single sample."""
    is_correct: bool
    score: float  # 0.0 or 1.0
    details: Dict[str, Any]


def score_multiple_choice(
    *,
    pred_letter: Optional[str],
    gt_letter: Optional[str],
    normalize: bool = True,
) -> CorrectnessResult:
    """
    Compute correctness for multiple-choice VQA/classification.

    Args:
        pred_letter: model prediction (e.g. "A"/"B"/"C"/"D")
        gt_letter: ground-truth option letter
        normalize: if True, strips whitespace and uppercases letters

    Returns:
        CorrectnessResult with:
          - is_correct: bool
          - score: float (0.0/1.0)
          - details: dict with debug fields
    """
    if pred_letter is None or gt_letter is None:
        # Treat missing labels/preds as incorrect (safe default)
        return CorrectnessResult(
            is_correct=False,
            score=0.0,
            details={"reason": "missing_pred_or_gt", "pred_letter": pred_letter, "gt_letter": gt_letter},
        )

    p = pred_letter
    g = gt_letter
    if normalize:
        p = str(p).strip().upper()
        g = str(g).strip().upper()

    is_correct = (p == g)
    return CorrectnessResult(
        is_correct=is_correct,
        score=1.0 if is_correct else 0.0,
        details={"pred_letter": p, "gt_letter": g},
    )
