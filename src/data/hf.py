from __future__ import annotations

from datasets import load_dataset, DownloadConfig
import aiohttp

from src.storage_paths import configure_hf_cache_env, hf_datasets_cache

configure_hf_cache_env()


def load_hf_dataset(hf_name: str, hf_config: str | None = None):
    """
    HuggingFace dataset loader with a long aiohttp timeout.
    Keeps the same behavior as your current code.
    """
    timeout = aiohttp.ClientTimeout(total=60 * 60)  # 1 hour total timeout
    dl_cfg = DownloadConfig(
        storage_options={"client_kwargs": {"timeout": timeout}}
    )
    return load_dataset(
        hf_name,
        hf_config,
        trust_remote_code=True,
        download_config=dl_cfg,
        cache_dir=hf_datasets_cache(),
    )
