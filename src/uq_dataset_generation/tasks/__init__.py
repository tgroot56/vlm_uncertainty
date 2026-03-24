"""dataset-specific how to interpret predict_fn output, how to score, what to store in rows"""

from .coco_qa_vi import CocoQaViTask
from .imagenet_r import ImageNetRTask
from .pope import PopeTask
from .vqa_v2 import VQAv2Task

__all__ = ["CocoQaViTask", "ImageNetRTask", "PopeTask", "VQAv2Task"]
