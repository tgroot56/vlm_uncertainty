from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from src.storage_paths import default_supervision_output_root


@dataclass
class SupervisionGenConfig:
    dataset_id: str
    model_id: str
    output_root: str = field(default_factory=default_supervision_output_root)
    dataset_split: Optional[str] = None

    seed_offset: int = 42
    max_samples: Optional[int] = None
    run_name: Optional[str] = None
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

    # Layer-sweep options for dedicated experiments
    use_vision_all_layers_mean: bool = False
    use_lm_visual_all_layers_mean: bool = False
    use_lm_visual_all_layers_lasttoken: bool = False
    use_lm_question_all_layers_mean: bool = False
    use_lm_question_all_layers_lasttoken: bool = False
    use_lm_answer_all_layers_mean: bool = False
    use_lm_answer_all_layers_lasttoken: bool = False
    use_lm_fullspan_all_layers_lasttoken: bool = False
    use_answer_geom_mean_probability: bool = False

    # Layers
    force_middle_layer: Optional[int] = None
