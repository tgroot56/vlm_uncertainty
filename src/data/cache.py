from __future__ import annotations

import os
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional

DEFAULT_CACHE_DIR = "cached_datasets"
CACHE_DIR = os.environ.get("VLM_UQ_PREP_CACHE", DEFAULT_CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)


def _stable_kwargs_hash(dataset_kwargs: Optional[Dict[str, Any]]) -> str:
    """
    Stable short hash for dataset_kwargs to separate caches per config.
    - Sort keys for determinism.
    - Convert non-JSONable objects via str(...).
    """
    if not dataset_kwargs:
        return "nokws"
    try:
        s = json.dumps(dataset_kwargs, sort_keys=True, default=str)
    except TypeError:
        s = str(sorted((k, str(v)) for k, v in dataset_kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:10]


def cache_key(
    dataset_id: str,
    split: str,
    seed_offset: int,
    num_samples: int,
    dataset_kwargs: Optional[Dict[str, Any]],
) -> str:
    kw_hash = _stable_kwargs_hash(dataset_kwargs)
    s = f"{dataset_id}|{split}|{seed_offset}|{num_samples}|{kw_hash}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


def cache_path(
    dataset_id: str,
    split: str,
    seed_offset: int,
    num_samples: int,
    dataset_kwargs: Optional[Dict[str, Any]],
) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(
        CACHE_DIR,
        f"prepared_{cache_key(dataset_id, split, seed_offset, num_samples, dataset_kwargs)}.pkl",
    )


def load_cache(
    dataset_id: str,
    split: str,
    seed_offset: int,
    num_samples: int,
    dataset_kwargs: Optional[Dict[str, Any]],
) -> Optional[List[Dict]]:
    path = cache_path(dataset_id, split, seed_offset, num_samples, dataset_kwargs)
    print("searched for cache at:", path)
    if not os.path.exists(path):
        print("No cache file found.")
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
            and meta.get("kwargs_hash") == _stable_kwargs_hash(dataset_kwargs)
        ):
            print(f"Loaded prepared dataset from cache: {path}")
            return payload["samples"]

        print("Cache metadata mismatch; ignoring cache.")
        return None
    except Exception as e:
        print(f"Failed to load cache ({path}): {e}")
        return None


def save_cache(
    samples: List[Dict],
    dataset_id: str,
    split: str,
    seed_offset: int,
    dataset_kwargs: Optional[Dict[str, Any]],
) -> str:
    path = cache_path(dataset_id, split, seed_offset, len(samples), dataset_kwargs)
    payload = {
        "meta": {
            "dataset_id": dataset_id,
            "split": split,
            "seed_offset": seed_offset,
            "num_samples": len(samples),
            "kwargs_hash": _stable_kwargs_hash(dataset_kwargs),
            "dataset_kwargs": dataset_kwargs or {},
        },
        "samples": samples,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved prepared dataset cache: {path}")
    return path


def find_matching_cache(
    dataset_id: str,
    split: str,
    seed_offset: int,
    dataset_kwargs: Optional[Dict[str, Any]],
) -> Optional[List[Dict]]:
    if not os.path.exists(CACHE_DIR):
        return None

    target_hash = _stable_kwargs_hash(dataset_kwargs)

    for fname in os.listdir(CACHE_DIR):
        if not fname.startswith("prepared_") or not fname.endswith(".pkl"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        print(f"Checking cache file: {fname} (this may take a while for large files)...")
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("meta", {})
            if (
                meta.get("dataset_id") == dataset_id
                and meta.get("split") == split
                and meta.get("seed_offset") == seed_offset
                and meta.get("kwargs_hash") == target_hash
            ):
                print(f"Found matching cache: {path}")
                return payload["samples"]
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

    return None
