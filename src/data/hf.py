from __future__ import annotations

from datasets import load_dataset, DownloadConfig
import aiohttp


def load_hf_dataset(hf_name: str):
    """
    HuggingFace dataset loader with a long aiohttp timeout.
    Keeps the same behavior as your current code.
    """
    timeout = aiohttp.ClientTimeout(total=60 * 60)  # 1 hour total timeout
    dl_cfg = DownloadConfig(
        storage_options={"client_kwargs": {"timeout": timeout}}
    )
    return load_dataset(hf_name, trust_remote_code=True, download_config=dl_cfg)
