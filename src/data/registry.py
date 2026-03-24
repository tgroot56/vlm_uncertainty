from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .common import prepare_coco_qa_vi_split, prepare_pope_split, prepare_vqa_v2_split
from .datasets.imagenet_r import prepare_imagenet_r_split


@dataclass(frozen=True)
class DatasetSpec:
    """
    Defines how to load and prepare a dataset split into a standardized list of dict samples.

    prepare_split MUST return List[Dict] where each dict is a "prepared sample".
    """
    hf_name: str
    hf_config: Optional[str] = None
    default_split: str = "test"
    prepare_split: Callable[..., List[Dict]] = None
    # prepare_split(split_obj, seed_offset, max_samples, **dataset_kwargs) -> prepared_samples


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "imagenet-r": DatasetSpec(
        hf_name="axiong/imagenet-r",
        default_split="test", # only exists a test split
        prepare_split=prepare_imagenet_r_split,
    ),
    "vqa-v2": DatasetSpec(
        hf_name="HuggingFaceM4/VQAv2",
        default_split="validation", # test split does not include the ground-truth answers. Validation big enough for our purposes.
        prepare_split=prepare_vqa_v2_split,
    ),
    "coco-qa-vi": DatasetSpec(
        hf_name="ThucPD/coco-qa-vi",
        default_split="test", # start with just using test, maybe use train/test later if we want more samples
        prepare_split=prepare_coco_qa_vi_split,
    ),
    "pope": DatasetSpec(
        hf_name="lmms-lab/POPE",
        hf_config="Full",
        default_split="adversarial",
        prepare_split=prepare_pope_split,
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS.keys()))
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Known: {known}")
    return DATASET_SPECS[dataset_id]
