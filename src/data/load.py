from __future__ import annotations

from typing import Dict, List, Optional, Any

from .registry import get_dataset_spec
from .hf import load_hf_dataset
from .cache import find_matching_cache, load_cache, save_cache

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
        # very defensive fallback
        s = str(sorted((k, str(v)) for k, v in dataset_kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:10]


def load_dataset_prepared(
    dataset_id: str,
    split: Optional[str] = None,
    seed_offset: int = 42,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Load and prepare a dataset split into a list of standardized dict samples.

    Args:
        dataset_id: key in DATASET_SPECS
        split: override split (defaults to spec.default_split)
        seed_offset: offset used for deterministic sampling
        max_samples: optional truncation
        use_cache: if True, use prepared dataset cache (same behavior as before)
        dataset_kwargs: forwarded to dataset-specific prepare_split (e.g. ImageNet-R k_options, CLIP negatives)

    Returns:
        List[Dict] prepared samples.
    """
    dataset_kwargs = dataset_kwargs or {}

    spec = get_dataset_spec(dataset_id)
    split = split or spec.default_split

    # Optimized cache check (optional)
    if use_cache:
        cached_any = find_matching_cache(dataset_id, split, seed_offset, dataset_kwargs)
        if cached_any is not None and (max_samples is None or len(cached_any) >= max_samples):
            return cached_any if max_samples is None else cached_any[:max_samples]

    ds = load_hf_dataset(spec.hf_name)
    split_obj = ds[split]

    # Exact cache check (only when max_samples is None, matching your old behavior)
    if use_cache and max_samples is None:
        cached_exact = load_cache(dataset_id, split, seed_offset, len(split_obj), dataset_kwargs)
        if cached_exact is not None:
            return cached_exact

    print(f"Preparing dataset: dataset_id={dataset_id}, split={split}, num_samples={len(split_obj)}")
    print("This may take a few minutes for large datasets...")

    # IMPORTANT: dataset_kwargs enables ImageNet-R options like k_options and CLIP negatives
    samples = spec.prepare_split(
        split_obj,
        seed_offset=seed_offset,
        max_samples=max_samples,
        **dataset_kwargs,
    )

    print(f"Dataset preparation complete. Processed {len(samples)} samples.")

    if use_cache and max_samples is None:
        save_cache(samples, dataset_id, split, seed_offset, dataset_kwargs)

    return samples
