from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.labeling.vqa import score_vqa


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
        # POPE is a yes/no task, so we keep generation tight to avoid extra
        # explanatory tokens that do not carry useful supervision signal.
        return {
            "prompt": sample["prompt"],
            "max_new_tokens": 2,
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

        corr = score_vqa(
            pred_answer=pred_answer,
            gt_answer_single=sample.get("gt_answer"),
        )

        row = {
            "idx": sample.get("idx"),
            "question_id": sample.get("question_id"),
            "sample_id": sample.get("sample_id"),
            "pred_answer": pred_answer,
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
