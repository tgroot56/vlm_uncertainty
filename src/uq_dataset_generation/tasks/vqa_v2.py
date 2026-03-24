from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.labeling.vqa import score_vqa


@dataclass(frozen=True)
class VQAv2Task:
    """
    Handles:
      - interpreting predict_fn output (open-ended VQA)
      - scoring correctness
      - building per-row metadata

    Assumes sample has: prompt, gt_answers (list), maybe gt_answer, question_id, image_id, dataset_id
    """
    dataset_id: str = "vqa-v2"
    prediction_mode: str = "open_ended_vqa"

    def dataset_load_kwargs(self, args: Any, device: Any) -> Dict[str, Any]:
        return {}

    def predict_kwargs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": sample["prompt"]}

    def interpret_output(
        self,
        *,
        sample: Dict[str, Any],
        prediction_output: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        """
        Expects VQA output length 8:
          (pred_answer, vision_hidden_states, vision_merged_states, lm_hidden_states, token_spans,
           answer_hidden_states, gen_ids, gen_step_logits)
        """
        if len(prediction_output) != 8:
            raise ValueError(f"VQA-v2 expects predict_fn output length 8, got {len(prediction_output)}")

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
            gt_answers=sample.get("gt_answers"),
            gt_answer_single=sample.get("gt_answer"),
        )

        row = {
            "idx": sample.get("idx"),
            "question_id": sample.get("question_id"),
            "image_id": sample.get("image_id"),
            "pred_answer": pred_answer,
            "raw_text": pred_answer,
            "gt_answers": sample.get("gt_answers"),
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
