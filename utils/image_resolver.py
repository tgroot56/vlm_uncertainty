import os, json
from datasets import load_dataset

_VQA_DATASET = None
_VQA_IMAGE_INDEX = None

def _init_vqa(cache_dir="cached_datasets/vqa_v2"):
    global _VQA_DATASET, _VQA_IMAGE_INDEX
    if _VQA_DATASET is not None:
        return

    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "image_id_to_idx.json")

    _VQA_DATASET = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="validation",
        trust_remote_code=True,
    )

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            # keys may load as strings; normalize later
            _VQA_IMAGE_INDEX = json.load(f)
            # convert keys back to int when possible
            _VQA_IMAGE_INDEX = {int(k) if k.isdigit() else k: v for k, v in _VQA_IMAGE_INDEX.items()}
    else:
        image_ids = _VQA_DATASET["image_id"]
        idx_map = {}
        for i, img_id in enumerate(image_ids):
            idx_map.setdefault(img_id, i)
        _VQA_IMAGE_INDEX = idx_map
        with open(index_path, "w") as f:
            json.dump({str(k): int(v) for k, v in idx_map.items()}, f)

def resolve_image(sample):
    global _VQA_DATASET, _VQA_IMAGE_INDEX

    if sample.get("dataset_id") != "vqa-v2":
        raise ValueError(f"Cannot resolve image for sample: {sample.keys()}")

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
            raise ValueError(f"Image ID {image_id} not found in dataset")
        idx = alt

    return _VQA_DATASET[idx]["image"]
