"""dataset-specific how to interpret predict_fn output, how to score, what to store in rows"""

from .imagenet_r import ImageNetRTask
from .vqa_v2 import VQAv2Task

__all__ = ["ImageNetRTask", "VQAv2Task"]
