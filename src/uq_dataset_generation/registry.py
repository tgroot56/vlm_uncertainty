from __future__ import annotations
from typing import Any

from .tasks import CocoQaViTask, ImageNetRTask, MSTSTask, PopeTask, VQAv2Task


def get_task(dataset_id: str) -> Any:
    if dataset_id == "imagenet-r":
        return ImageNetRTask()
    if dataset_id == "vqa-v2":
        return VQAv2Task()
    if dataset_id == "coco-qa-vi":
        return CocoQaViTask()
    if dataset_id == "pope":
        return PopeTask()
    if dataset_id == "msts":
        return MSTSTask()
    raise ValueError(f"Unknown dataset_id '{dataset_id}' for UQ generation task registry")
