from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import re

from src.labeling.vqa import score_vqa


def _extract_yes_no_answer(text: str) -> str:
    """Extract the first explicit yes/no decision from a free-form POPE answer."""
    normalized = (text or "").strip()
    lowered = normalized.lower()
    if not lowered:
        return normalized

    no_patterns = (
        r"\bno\b",
        r"\bnot\b",
        r"\bthere (?:is|are|appears to be|seems to be) no\b",
        r"\bthere (?:is|are|appears to be|seems to be) not\b",
        r"\bthere (?:isn['’]?t|aren['’]?t)\b",
        r"\bi (?:do not|don['’]?t|cannot|can['’]?t) see\b",
        r"\bthe image does not (?:contain|show|have)\b",
        r"\bthe image doesn['’]?t (?:contain|show|have)\b",
        r"\bdoes not\b",
        r"\bdoesn['’]?t\b",
        r"\bcannot\b",
        r"\bcan['’]?t\b",
    )
    yes_patterns = (
        r"\byes\b",
        r"\bthere (?:is|are|appears to be|seems to be)\b",
        r"\bi (?:can|do) see\b",
        r"\bthe image (?:contains|shows|has)\b",
    )

    no_match = min((m.start() for pattern in no_patterns if (m := re.search(pattern, lowered))), default=None)
    yes_match = min((m.start() for pattern in yes_patterns if (m := re.search(pattern, lowered))), default=None)

    if no_match is not None and (yes_match is None or no_match <= yes_match):
        return "no"
    if yes_match is not None:
        return "yes"

    return normalized


@dataclass(frozen=True)
class PopeTask:
    """
    Handles open-ended yes/no VQA for POPE.
    """
    dataset_id: str = "pope"
    prediction_mode: str = "open_ended_vqa"

    def dataset_load_kwargs(self, args: Any, device: Any) -> Dict[str, Any]:
        return {}

    def predict_kwargs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Allow enough tokens for chatty VLMs to reach a yes/no decision; the
        # scorer extracts the task answer from the generated text below.
        return {
            "prompt": sample["prompt"],
            "max_new_tokens": 15,
        }

    def interpret_output(
        self,
        *,
        sample: Dict[str, Any],
        prediction_output: Tuple[Any, ...],
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

        corr = score_vqa(
            pred_answer=parsed_answer,
            gt_answer_single=sample.get("gt_answer"),
        )

        row = {
            "idx": sample.get("idx"),
            "question_id": sample.get("question_id"),
            "sample_id": sample.get("sample_id"),
            "pred_answer": parsed_answer,
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
