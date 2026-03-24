# utils/image_resolver.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from datasets import load_dataset
from src.storage_paths import (
    configure_hf_cache_env,
    hf_datasets_cache,
    image_resolver_cache_dir,
)

configure_hf_cache_env()

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore


# ----------------------------
# VQA-v2 lazy image index
# ----------------------------

_VQA_DATASET = None
_VQA_IMAGE_INDEX = None


def _default_vqa_cache_dir() -> str:
    return os.path.join(image_resolver_cache_dir(), "vqa_v2")


def _init_vqa(cache_dir: Optional[str] = None, split: str = "validation") -> None:
    global _VQA_DATASET, _VQA_IMAGE_INDEX
    if _VQA_DATASET is not None:
        return

    cache_dir = cache_dir or _default_vqa_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, f"image_id_to_idx_{split}.json")

    _VQA_DATASET = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=split,
        trust_remote_code=True,
        cache_dir=hf_datasets_cache(),
    )

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            idx_map = json.load(f)
        _VQA_IMAGE_INDEX = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in idx_map.items()}
    else:
        image_ids = _VQA_DATASET["image_id"]
        idx_map = {}
        for i, img_id in enumerate(image_ids):
            idx_map.setdefault(img_id, i)
        _VQA_IMAGE_INDEX = idx_map
        with open(index_path, "w") as f:
            json.dump({str(k): int(v) for k, v in idx_map.items()}, f)


def _resolve_vqa_image(sample: Dict[str, Any]) -> Any:
    global _VQA_DATASET, _VQA_IMAGE_INDEX
    _init_vqa()

    image_id = sample["image_id"]
    idx = _VQA_IMAGE_INDEX.get(image_id)
    if idx is None:
        # common pitfall: type mismatch (str vs int)
        alt = None
        if isinstance(image_id, str) and image_id.isdigit():
            alt = _VQA_IMAGE_INDEX.get(int(image_id))
        elif isinstance(image_id, int):
            alt = _VQA_IMAGE_INDEX.get(str(image_id))

        if alt is None:
            raise ValueError(f"VQA image_id {image_id} not found in index")
        idx = alt

    return _VQA_DATASET[idx]["image"]


# ----------------------------
# Generic resolver
# ----------------------------

def resolve_image(sample: Dict[str, Any]) -> Any:
    """
    Resolve image for a prepared sample.

    Priority:
    1) If sample already contains an image object under "image": return it (ImageNet-R, many HF datasets)
    2) Else if dataset_id == "vqa-v2": fetch image from VQA dataset via image_id index
    3) Else error
    """
    # 1) Generic fast path: image object already present
    if "image" in sample and sample["image"] is not None:
        img = sample["image"]
        # Usually PIL.Image.Image. If PIL isn't installed, just return it anyway.
        if Image is None:
            return img
        # If it is PIL, return; if not PIL (rare), still return and let downstream handle.
        if isinstance(img, Image.Image):
            return img
        return img

    # 2) VQA special-case
    if sample.get("dataset_id") == "vqa-v2":
        return _resolve_vqa_image(sample)

    raise ValueError(f"Cannot resolve image for sample: {sample.keys()}")
