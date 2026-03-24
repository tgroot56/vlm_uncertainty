from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.data.datasets.imagenet_r import ClipNegSamplerConfig
from src.labeling.multiple_choice import score_multiple_choice


@dataclass(frozen=True)
class ImageNetRTask:
    """
    Handles:
      - interpreting predict_fn output (MC classification)
      - scoring correctness
      - building per-row metadata

    Assumes sample has: prompt, gt_key (and optionally gt_class, option_map)
    """

    dataset_id: str = "imagenet-r"
    prediction_mode: str = "multiple_choice"

    def dataset_load_kwargs(self, args: Any, device: Any) -> Dict[str, Any]:
        dataset_kwargs: Dict[str, Any] = {
            "k_options": args.k_options,
            "use_clip_hard_negatives": args.clip_negatives,
        }

        if dataset_kwargs["use_clip_hard_negatives"]:
            dataset_kwargs["clip_cfg"] = ClipNegSamplerConfig(
                mode=args.clip_mode,
                top_pool=args.clip_top_pool,
                device=str(device),
            )

        return dataset_kwargs

    def predict_kwargs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        option_keys = list(sample["option_map"].keys())
        return {"prompt": sample["prompt"], "option_letters": option_keys}

    def interpret_output(
        self,
        *,
        sample: Dict[str, Any],
        prediction_output: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        """
        Expects the "with_features" MC output of length 11:
          (pred_letter, option_probs, option_logits, raw_text,
           vision_hidden_states, vision_merged_states, lm_hidden_states, token_spans,
           answer_hidden_states, gen_ids, gen_step_logits)
        """
        if len(prediction_output) != 11:
            raise ValueError(f"ImageNet-R expects predict_fn output length 11, got {len(prediction_output)}")

        (
            pred_key,
            option_probs,
            option_logits,
            raw_text,
            vision_hidden_states,
            vision_merged_states,
            lm_hidden_states,
            token_spans,
            answer_hidden_states,
            gen_ids,
            gen_step_logits,
        ) = prediction_output

        # IMPORTANT: use gt_key (works for any K)
        gt_key = sample.get("gt_key") or sample.get("gt_letter")
        corr = score_multiple_choice(pred_letter=pred_key, gt_letter=gt_key, normalize=True)

        row = {
            "idx": sample.get("idx"),
            "gt_key": gt_key,
            "gt_class": sample.get("gt_class"),
            "pred_key": pred_key,
            "raw_text": raw_text,
            "option_probs": option_probs,
            "option_logits": option_logits,
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
