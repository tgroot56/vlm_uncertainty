from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class MSTSTask:
    """Handles open-ended MSTS safety prompts with pending manual labels."""

    dataset_id: str = "msts"
    prediction_mode: str = "open_ended_vqa"
    label_placeholder: bool = True

    def dataset_load_kwargs(self, args: Any, device: Any) -> Dict[str, Any]:
        return {}

    def predict_kwargs(self, sample: Dict[str, Any], vlm_adapter: Any = None) -> Dict[str, Any]:
        return {
            "prompt": sample["prompt"],
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 1.0,
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
            raise ValueError(f"MSTS expects predict_fn output length 8, got {len(prediction_output)}")

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

        row = {
            "idx": sample.get("idx"),
            "prompt_id": sample.get("prompt_id"),
            "prompt_type": sample.get("prompt_type"),
            "prompt_text": sample.get("prompt_text") or sample.get("prompt"),
            "case_id": sample.get("case_id"),
            "case_text": sample.get("case_text"),
            "hazard_category": sample.get("hazard_category"),
            "hazard_subcategory": sample.get("hazard_subcategory"),
            "hazard_subsubcategory": sample.get("hazard_subsubcategory"),
            "unsafe_image_id": sample.get("unsafe_image_id"),
            "unsafe_image_description": sample.get("unsafe_image_description"),
            "unsafe_image_url": sample.get("unsafe_image_url"),
            "unsafe_image_license": sample.get("unsafe_image_license"),
            "unsafe_image_cw": sample.get("unsafe_image_cw"),
            "image_path": sample.get("image_path"),
            "pred_answer": pred_answer,
            "raw_text": pred_answer,
            "manual_label": sample.get("manual_label"),
            "token_spans": token_spans,
        }

        return {
            "score": float("nan"),
            "row": row,
            "vision_hidden_states": vision_hidden_states,
            "vision_merged_states": vision_merged_states,
            "lm_hidden_states": lm_hidden_states,
            "token_spans": token_spans,
            "answer_hidden_states": answer_hidden_states,
            "gen_ids": gen_ids,
            "gen_step_logits": gen_step_logits,
        }

    def metadata_extras(self) -> Dict[str, Any]:
        return {
            "label_placeholder": True,
            "label_description": "NaN placeholders; fill manual labels after reviewing MSTS responses.",
            "msts_prompt_filter": "prompt_text.startswith('Should I')",
            "msts_expected_num_samples": 200,
            "msts_max_new_tokens": 512,
        }

    def write_auxiliary_outputs(
        self,
        *,
        output_dir: str,
        rows: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        annotation_path = os.path.join(output_dir, "annotation_template.csv")
        fieldnames = [
            "idx",
            "prompt_id",
            "case_id",
            "hazard_category",
            "hazard_subcategory",
            "hazard_subsubcategory",
            "unsafe_image_id",
            "unsafe_image_description",
            "unsafe_image_url",
            "unsafe_image_license",
            "unsafe_image_cw",
            "image_path",
            "prompt_text",
            "response",
            "manual_label",
            "manual_notes",
        ]

        with open(annotation_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "idx": row.get("idx"),
                        "prompt_id": row.get("prompt_id"),
                        "case_id": row.get("case_id"),
                        "hazard_category": row.get("hazard_category"),
                        "hazard_subcategory": row.get("hazard_subcategory"),
                        "hazard_subsubcategory": row.get("hazard_subsubcategory"),
                        "unsafe_image_id": row.get("unsafe_image_id"),
                        "unsafe_image_description": row.get("unsafe_image_description"),
                        "unsafe_image_url": row.get("unsafe_image_url"),
                        "unsafe_image_license": row.get("unsafe_image_license"),
                        "unsafe_image_cw": row.get("unsafe_image_cw"),
                        "image_path": row.get("image_path"),
                        "prompt_text": row.get("prompt_text"),
                        "response": row.get("raw_text"),
                        "manual_label": "",
                        "manual_notes": "",
                    }
                )

        metadata["annotation_template_path"] = annotation_path
