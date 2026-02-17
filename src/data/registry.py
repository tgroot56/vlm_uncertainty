from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .common import prepare_vqa_v2_split
from .datasets.imagenet_r import prepare_imagenet_r_split


@dataclass(frozen=True)
class DatasetSpec:
    """
    Defines how to load and prepare a dataset split into a standardized list of dict samples.

    prepare_split MUST return List[Dict] where each dict is a "prepared sample".
    """
    hf_name: str
    default_split: str = "test"
    prepare_split: Callable[..., List[Dict]] = None
    # prepare_split(split_obj, seed_offset, max_samples, **dataset_kwargs) -> prepared_samples


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "imagenet-r": DatasetSpec(
        hf_name="axiong/imagenet-r",
        default_split="test",
        prepare_split=prepare_imagenet_r_split,
    ),
    "vqa-v2": DatasetSpec(
        hf_name="HuggingFaceM4/VQAv2",
        default_split="validation",
        prepare_split=prepare_vqa_v2_split,
    ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS.keys()))
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Known: {known}")
    return DATASET_SPECS[dataset_id]
