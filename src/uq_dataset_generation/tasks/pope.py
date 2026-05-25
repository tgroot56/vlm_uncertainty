from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

import torch

from src.labeling.vqa import score_vqa


_NO_PATTERNS = (
    # direct
    r"\bno\b",
    r"\bnot\b",
    r"\bnone\b",
    r"\bnothing\b",
    r"\bnobody\b",
    r"\bneither\b",
    r"\bnever\b",
    # contractions (both with and without apostrophes, and curly apostrophes)
    r"\bisn['’]?t\b",
    r"\baren['’]?t\b",
    r"\bwasn['’]?t\b",
    r"\bweren['’]?t\b",
    r"\bdoesn['’]?t\b",
    r"\bdon['’]?t\b",
    r"\bdidn['’]?t\b",
    r"\bhasn['’]?t\b",
    r"\bhaven['’]?t\b",
    r"\bcan['’]?t\b",
    r"\bcannot\b",
    r"\bwon['’]?t\b",
    # negated descriptions
    r"\bthere (?:is|are|appears to be|seems to be) no\b",
    r"\bi (?:do not|don['’]?t|cannot|can['’]?t) see\b",
    r"\bthe image does not (?:contain|show|have|include|feature)\b",
    r"\bthe image doesn['’]?t (?:contain|show|have|include|feature)\b",
)

# Strong yes tokens that by themselves indicate an affirmative answer.
_STRONG_YES_PATTERNS = (
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\byup\b",
    r"\baffirmative\b",
    r"\bindeed\b",
    r"\bcorrect\b",
    r"\btrue\b",
    r"\bcertainly\b",
)

# Weak yes signals — only count if no explicit negation appears anywhere in the text.
_WEAK_YES_PATTERNS = (
    r"\bthere (?:is|are|appears to be|seems to be)\b",
    r"\bi (?:can|do) see\b",
    r"\bthe image (?:contains|shows|has|depicts|features|includes)\b",
)


def _extract_yes_no_answer(text: str) -> str:
    """
    Map a free-form POPE answer onto a canonical "yes"/"no" token.

    Strategy: look for explicit negation anywhere in the text first (because POPE
    answers like "There are none" or "The image does not contain a skis" must be
    read as "no" even though their prefix can resemble a positive description).
    If no negation appears, look for an explicit yes word; finally fall back to
    positive descriptions ("there is a ...", "I can see ...", ...).
    """
    normalized = (text or "").strip()
    lowered = normalized.lower()
    if not lowered:
        return normalized

    for pattern in _NO_PATTERNS:
        if re.search(pattern, lowered):
            return "no"

    for pattern in _STRONG_YES_PATTERNS:
        if re.search(pattern, lowered):
            return "yes"

    for pattern in _WEAK_YES_PATTERNS:
        if re.search(pattern, lowered):
            return "yes"

    return normalized


_YES_SURFACE_FORMS = ("yes", " yes", "Yes", " Yes", "YES", " YES")
_NO_SURFACE_FORMS = ("no", " no", "No", " No", "NO", " NO")


def _first_token_id(tokenizer: Any, surface: str) -> Optional[int]:
    try:
        ids = tokenizer.encode(surface, add_special_tokens=False)
    except Exception:
        return None
    if not ids:
        return None
    return int(ids[0])


def _collect_unique_token_ids(tokenizer: Any, surfaces: Tuple[str, ...]) -> List[int]:
    collected: List[int] = []
    seen = set()
    for surface in surfaces:
        tid = _first_token_id(tokenizer, surface)
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        collected.append(tid)
    return collected


def _logit_scored_yes_no(
    gen_step_logits: Optional[torch.Tensor],
    tokenizer: Any,
) -> Optional[str]:
    """Pick "yes" vs "no" from the first-step logits.

    Mirrors the original POPE evaluation protocol: compare the probability of
    the yes-tokens vs the no-tokens under the model's distribution for the very
    first generated token. Returns None if logits are unavailable or neither
    vocabulary surface is tokenizable.
    """
    if gen_step_logits is None or tokenizer is None:
        return None
    if not isinstance(gen_step_logits, torch.Tensor):
        return None
    if gen_step_logits.ndim < 3 or gen_step_logits.shape[1] == 0:
        return None

    yes_ids = _collect_unique_token_ids(tokenizer, _YES_SURFACE_FORMS)
    no_ids = _collect_unique_token_ids(tokenizer, _NO_SURFACE_FORMS)
    if not yes_ids or not no_ids:
        return None

    first_logits = gen_step_logits[0, 0, :].float()
    yes_logit = first_logits[torch.tensor(yes_ids, device=first_logits.device)].max().item()
    no_logit = first_logits[torch.tensor(no_ids, device=first_logits.device)].max().item()

    return "yes" if yes_logit >= no_logit else "no"


def _build_prompt(question: str, vlm_adapter: Any) -> str:
    """Per-family POPE prompt.

    Molmo2 ignores short imperatives like "Answer with exactly one word: yes or no"
    and instead produces descriptive captions or pointing tokens. A firmer, more
    explicit instruction is needed *only* for Molmo — other models already comply
    with the existing prompt and we must not alter their behavior.
    """
    family = getattr(vlm_adapter, "family", None) if vlm_adapter is not None else None
    if family == "molmo":
        # Molmo2-O is trained heavily on pointing and captioning styles and
        # tends to ignore short "one word" instructions. An "Answer:" prefix
        # paired with a constrained option list nudges it to complete with
        # just "yes" or "no" instead of a free-form caption.
        return (
            f"Question: {question}\n"
            "Choose one: yes or no.\n"
            "Answer:"
        )
    return f"Question: {question}\nAnswer with exactly one word: yes or no."


@dataclass(frozen=True)
class PopeTask:
    """
    Handles open-ended yes/no VQA for POPE.
    """
    dataset_id: str = "pope"
    prediction_mode: str = "open_ended_vqa"

    def dataset_load_kwargs(self, args: Any, device: Any) -> Dict[str, Any]:
        return {}

    def predict_kwargs(self, sample: Dict[str, Any], vlm_adapter: Any = None) -> Dict[str, Any]:
        question = sample.get("question") or sample.get("statement") or ""
        prompt = _build_prompt(question, vlm_adapter) if question else sample["prompt"]
        return {
            "prompt": prompt,
            "max_new_tokens": 15,
        }

    def interpret_output(
        self,
        *,
        sample: Dict[str, Any],
        prediction_output: Tuple[Any, ...],
        vlm_adapter: Any = None,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        if len(prediction_output) != 8:
            raise ValueError(f"POPE expects predict_fn output length 8, got {len(prediction_output)}")

        (
            pred_answer,
            vision_hidden_states,
            vision_merged_states,
            lm_hidden_states,
            token_spans,
            answer_hidden_states,
            gen_ids,
            gen_step_logits,
        ) = prediction_output

        parsed_answer = _extract_yes_no_answer(pred_answer)
        pred_source = "text"

        # For Molmo (and as a safety net generally), if text-parsing could not
        # resolve to an explicit yes/no, fall back to first-token logit scoring.
        # This matches the original POPE evaluation protocol and gives a fair
        # correctness signal even when the model emits a free-form caption.
        family = getattr(vlm_adapter, "family", None) if vlm_adapter is not None else None
        needs_logit_fallback = family == "molmo" and parsed_answer.lower() not in {"yes", "no"}
        if needs_logit_fallback:
            logit_answer = _logit_scored_yes_no(gen_step_logits, tokenizer)
            if logit_answer is not None:
                parsed_answer = logit_answer
                pred_source = "logits"

        corr = score_vqa(
            pred_answer=parsed_answer,
            gt_answer_single=sample.get("gt_answer"),
        )

        row = {
            "idx": sample.get("idx"),
            "question_id": sample.get("question_id"),
            "sample_id": sample.get("sample_id"),
            "pred_answer": parsed_answer,
            "pred_source": pred_source,
            "raw_text": pred_answer,
            "gt_answer": sample.get("gt_answer"),
            "category": sample.get("category"),
            "token_spans": token_spans,
        }

        return {
            "score": float(corr.score),
            "row": row,
            "vision_hidden_states": vision_hidden_states,
            "vision_merged_states": vision_merged_states,
            "lm_hidden_states": lm_hidden_states,
            "token_spans": token_spans,
            "answer_hidden_states": answer_hidden_states,
            "gen_ids": gen_ids,
            "gen_step_logits": gen_step_logits,
        }
