from __future__ import annotations
from typing import Any

from .tasks import ImageNetRTask, VQAv2Task


def get_task(dataset_id: str) -> Any:
    if dataset_id == "imagenet-r":
        return ImageNetRTask()
    if dataset_id == "vqa-v2":
        return VQAv2Task()
    raise ValueError(f"Unknown dataset_id '{dataset_id}' for UQ generation task registry")
