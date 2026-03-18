from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class SupervisionGenConfig:
    dataset_id: str
    model_id: str
    output_root: str = "/scratch-shared/tgroot/supervised_datasets"

    seed_offset: int = 42
    max_samples: Optional[int] = None
    verbose: bool = False

    # Feature switches
    use_vision_middle: bool = True
    use_vision_final: bool = True

    use_lm_visual_middle: bool = True
    use_lm_visual_final: bool = True

    # Rename prompt -> question
    use_lm_question_middle: bool = True
    use_lm_question_final: bool = True

    use_lm_answer_middle: bool = True
    use_lm_answer_final: bool = True

    use_lm_visual_middle_lasttoken: bool = True
    use_lm_visual_final_lasttoken: bool = True

    # Rename prompt -> question
    use_lm_question_middle_lasttoken: bool = True
    use_lm_question_final_lasttoken: bool = True

    use_lm_answer_middle_lasttoken: bool = True
    use_lm_answer_final_lasttoken: bool = True

    use_answer_prob_entropy_stats: bool = True

    # New: fullspan (prompt incl. image tokens)
    use_lm_fullspan_middle: bool = True
    use_lm_fullspan_final: bool = True
    use_lm_fullspan_middle_lasttoken: bool = True
    use_lm_fullspan_final_lasttoken: bool = True

    # New: question_including (all text excluding image tokens)
    use_lm_question_including_middle: bool = True
    use_lm_question_including_final: bool = True
    use_lm_question_including_middle_lasttoken: bool = True
    use_lm_question_including_final_lasttoken: bool = True

    # Layers
    force_middle_layer: Optional[int] = None