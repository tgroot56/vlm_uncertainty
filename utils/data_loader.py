"""
Dataset loading + optional sample preparation + caching.

Some datasets may require additional preparation (e.g. ImageNet-R as 4-way MC).
Most datasets will just return normalized sample dicts (image, question, gt, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import hashlib
import os
import pickle
import random

from datasets import load_dataset

CACHE_DIR = "cached_datasets"
LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------
# Dataset spec
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """
    Defines how to load and prepare a dataset split into a standardized list of dict samples.

    prepare_split MUST return List[Dict] where each dict is a "prepared sample"
    (fields depend on dataset, but should at least include 'idx' and any inputs needed later).
    """
    hf_name: str
    default_split: str = "test"
    prepare_split: Callable[[Any, int, Optional[int]], List[Dict]] = None
    # prepare_split(split_obj, seed_offset, max_samples) -> prepared_samples


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_hf_dataset(hf_name: str):
    return load_dataset(hf_name)


def _cache_key(dataset_id: str, split: str, seed_offset: int, num_samples: int) -> str:
    s = f"{dataset_id}|{split}|{seed_offset}|{num_samples}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


def _cache_path(dataset_id: str, split: str, seed_offset: int, num_samples: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"prepared_{_cache_key(dataset_id, split, seed_offset, num_samples)}.pkl")


def _load_cache(dataset_id: str, split: str, seed_offset: int, num_samples: int) -> Optional[List[Dict]]:
    path = _cache_path(dataset_id, split, seed_offset, num_samples)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)

        meta = payload.get("meta", {})
        if (
            meta.get("dataset_id") == dataset_id
            and meta.get("split") == split
            and meta.get("seed_offset") == seed_offset
            and meta.get("num_samples") == num_samples
        ):
            print(f"Loaded prepared dataset from cache: {path}")
            return payload["samples"]

        print("Cache metadata mismatch; ignoring cache.")
        return None
    except Exception as e:
        print(f"Failed to load cache ({path}): {e}")
        return None


def _save_cache(samples: List[Dict], dataset_id: str, split: str, seed_offset: int) -> str:
    path = _cache_path(dataset_id, split, seed_offset, len(samples))
    payload = {
        "meta": {
            "dataset_id": dataset_id,
            "split": split,
            "seed_offset": seed_offset,
            "num_samples": len(samples),
        },
        "samples": samples,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved prepared dataset cache: {path}")
    return path


def _find_matching_cache(dataset_id: str, split: str, seed_offset: int) -> Optional[List[Dict]]:
    """
    Optimized cache search: returns the first cache matching dataset_id/split/seed_offset,
    regardless of num_samples. Useful when you don't want to load HF dataset just to
    learn split size.
    """
    if not os.path.exists(CACHE_DIR):
        return None

    for fname in os.listdir(CACHE_DIR):
        if not fname.startswith("prepared_") or not fname.endswith(".pkl"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("meta", {})
            if (
                meta.get("dataset_id") == dataset_id
                and meta.get("split") == split
                and meta.get("seed_offset") == seed_offset
            ):
                print(f"Found matching cache: {path}")
                return payload["samples"]
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------
# ImageNet-R: MC preparation (only dataset that needs it, for now)
# ---------------------------------------------------------------------

def _build_mc_prompt_4way(
    question: str,
    gt_label: str,
    all_class_names: List[str],
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    distractors = rng.sample([c for c in all_class_names if c != gt_label], 3)
    options = [gt_label] + distractors
    rng.shuffle(options)

    option_map = {LETTERS[i]: options[i] for i in range(4)}
    gt_letter = next(k for k, v in option_map.items() if v == gt_label)

    prompt = (
        f"{question}\n\n"
        "Choices:\n"
        + "\n".join(f"{k}. {v}" for k, v in option_map.items())
        + "\n\nAnswer with only A, B, C, or D."
    )

    return {
        "prompt": prompt,
        "option_map": option_map,
        "gt_letter": gt_letter,
    }


def prepare_imagenet_r_split(split_obj, seed_offset: int, max_samples: Optional[int]) -> List[Dict]:
    """
    Prepare ImageNet-R as a multiple-choice dataset (A/B/C/D).
    Keeps logic equivalent to your original code.
    """
    if max_samples is not None:
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    all_class_names = sorted(set(split_obj["class_name"]))

    samples: List[Dict] = []
    for idx in range(len(split_obj)):
        row = split_obj[idx]
        gt_class = row["class_name"]

        mc = _build_mc_prompt_4way(
            question="Which object is shown in the image?",
            gt_label=gt_class,
            all_class_names=all_class_names,
            seed=seed_offset + idx,
        )

        samples.append(
            {
                "idx": idx,
                "image": row["image"],
                "gt_class": gt_class,
                # MC-specific fields (only for ImageNet-R)
                **mc,
            }
        )

    return samples


# ---------------------------------------------------------------------
# Default preparation (for "normal" datasets)
# ---------------------------------------------------------------------

def prepare_default_split(split_obj, seed_offset: int, max_samples: Optional[int]) -> List[Dict]:
    """
    Minimal preparation: just wrap raw rows with an idx.
    Use this for datasets where you don't need custom prompting.
    """
    if max_samples is not None:
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    return [{"idx": i, **split_obj[i]} for i in range(len(split_obj))]


# ---------------------------------------------------------------------
# Dataset registry (add datasets here)
# ---------------------------------------------------------------------

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "imagenet-r": DatasetSpec(
        hf_name="axiong/imagenet-r",
        default_split="test",
        prepare_split=prepare_imagenet_r_split,
    ),

    # Example generic dataset (no MC):
    # "vqa2": DatasetSpec(
    #     hf_name="some/vqa2",
    #     default_split="validation",
    #     prepare_split=prepare_default_split,
    # ),
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS.keys()))
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Known: {known}")
    return DATASET_SPECS[dataset_id]


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------

def load_dataset_prepared(
    dataset_id: str,
    split: Optional[str] = None,
    seed_offset: int = 42,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
) -> List[Dict]:
    """
    Load and prepare a dataset split into a list of standardized dict samples.
    - ImageNet-R becomes MC (prompt/options/gt_letter).
    - Other datasets can use prepare_default_split or their own prepare_*.

    Returns:
        List[Dict] prepared samples.
    """
    spec = get_dataset_spec(dataset_id)
    split = split or spec.default_split

    # Optimized cache check (optional)
    if use_cache:
        cached_any = _find_matching_cache(dataset_id, split, seed_offset)
        if cached_any is not None and (max_samples is None or len(cached_any) >= max_samples):
            return cached_any if max_samples is None else cached_any[:max_samples]

    ds = load_hf_dataset(spec.hf_name)
    split_obj = ds[split]

    # Exact cache check (now we know num_samples after optional truncation)
    # We do truncation inside prepare_split, but need num_samples for exact cache.
    # Easiest: load exact cache only when max_samples is None.
    if use_cache and max_samples is None:
        cached_exact = _load_cache(dataset_id, split, seed_offset, len(split_obj))
        if cached_exact is not None:
            return cached_exact

    print(f"Preparing dataset: dataset_id={dataset_id}, split={split}")
    samples = spec.prepare_split(split_obj, seed_offset=seed_offset, max_samples=max_samples)

    if use_cache and max_samples is None:
        _save_cache(samples, dataset_id, split, seed_offset)

    return samples
