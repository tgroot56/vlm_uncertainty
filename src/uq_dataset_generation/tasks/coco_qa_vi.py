from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.labeling.vqa import score_vqa
from src.vlm import strip_qwen_thinking_prefix


@dataclass(frozen=True)
class CocoQaViTask:
    """
    Handles open-ended VQA for COCO-QA

    This dataset has a single ground-truth answer per sample, so correctness falls
    back to normalized exact match via score_vqa(..., gt_answer_single=...).
    """
    dataset_id: str = "coco-qa-vi"
    prediction_mode: str = "open_ended_vqa"

    def dataset_load_kwargs(self, args: Any, device: Any) -> Dict[str, Any]:
        return {}

    def predict_kwargs(self, sample: Dict[str, Any], vlm_adapter: Any = None) -> Dict[str, Any]:
        kwargs = {"prompt": sample["prompt"]}
        if getattr(vlm_adapter, "family", None) == "qwen":
            kwargs["max_new_tokens"] = 15
        return kwargs

    def interpret_output(
        self,
        *,
        sample: Dict[str, Any],
        prediction_output: Tuple[Any, ...],
        vlm_adapter: Any = None,
        tokenizer: Any = None,
    ) -> Dict[str, Any]:
        if len(prediction_output) != 8:
            raise ValueError(f"COCO-QA-VI expects predict_fn output length 8, got {len(prediction_output)}")

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

        scored_answer = (
            strip_qwen_thinking_prefix(pred_answer)
            if getattr(vlm_adapter, "family", None) == "qwen"
            else pred_answer
        )

        corr = score_vqa(
            pred_answer=scored_answer,
            gt_answer_single=sample.get("gt_answer"),
        )

        row = {
            "idx": sample.get("idx"),
            "question_id": sample.get("question_id"),
            "image_id": sample.get("image_id"),
            "pred_answer": scored_answer,
            "raw_text": pred_answer,
            "gt_answer": sample.get("gt_answer"),
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
