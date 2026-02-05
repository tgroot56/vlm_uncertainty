# src/labeling/vqa.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
from collections import Counter


@dataclass
class CorrectnessResult:
    """Result of correctness scoring for a single VQA sample."""
    score: float              # VQA soft accuracy in [0, 1]
    details: Dict[str, Any]   # debug info


# -----------------------------
# Normalization (VQAEval-like)
# -----------------------------

_ARTICLES = {"a", "an", "the"}

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id": "i'd", "ill": "i'll", "im": "i'm", "ive": "i've", "isnt": "isn't",
    "itd": "it'd", "itll": "it'll", "its": "it's", "mightnt": "mightn't",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "oclock": "o'clock", "shant": "shan't",
    "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "wasnt": "wasn't", "wed": "we'd", "well": "we'll", "were": "we're", "weve": "we've",
    "werent": "weren't", "whatd": "what'd", "whatll": "what'll", "whats": "what's",
    "whens": "when's", "whered": "where'd", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}

_PUNCT = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")


def _normalize_vqa_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    a = ans.strip().lower()
    a = _PUNCT.sub(" ", a)
    a = _MULTI_SPACE.sub(" ", a).strip()

    tokens = []
    for w in a.split(" "):
        tokens.append(_CONTRACTIONS.get(w, w))
    a = " ".join(tokens)

    tokens = [w for w in a.split(" ") if w not in _ARTICLES]
    a = " ".join(tokens)

    return _MULTI_SPACE.sub(" ", a).strip()


# -----------------------------
# VQA soft accuracy (VQAEval)
# -----------------------------

def vqa_soft_accuracy(pred_answer: str, gt_answers: List[str]) -> float:
    pred_norm = _normalize_vqa_answer(pred_answer)
    gt_norm = [_normalize_vqa_answer(a) for a in gt_answers if a is not None]

    if len(gt_norm) == 0:
        return 0.0

    counts = Counter(gt_norm)
    match_count = counts.get(pred_norm, 0)

    # Official VQA rule
    return float(min(match_count / 3.0, 1.0))


def score_vqa(
    *,
    pred_answer: str,
    gt_answers: Optional[List[str]] = None,
    gt_answer_single: Optional[str] = None,
) -> CorrectnessResult:
    """
    If gt_answers is provided (typical VQAv2/AdVQA): return soft VQA score in [0,1].
    Else fall back to normalized exact match against gt_answer_single.
    """
    pred_norm = _normalize_vqa_answer(pred_answer)

    if gt_answers is not None:
        score = vqa_soft_accuracy(pred_answer, gt_answers)
        return CorrectnessResult(
            score=score,
            details={
                "mode": "vqa_soft",
                "pred_norm": pred_norm,
                "gt_norm_counts": dict(Counter(_normalize_vqa_answer(a) for a in gt_answers if a is not None)),
            },
        )

    if gt_answer_single is None:
        return CorrectnessResult(
            score=0.0,
            details={"reason": "missing_gt", "mode": "exact_fallback", "pred_norm": pred_norm},
        )

    gt_norm = _normalize_vqa_answer(gt_answer_single)
    score = 1.0 if pred_norm == gt_norm else 0.0

    return CorrectnessResult(
        score=score,
        details={
            "mode": "exact_fallback",
            "pred_norm": pred_norm,
            "gt_norm": gt_norm,
        },
    )
