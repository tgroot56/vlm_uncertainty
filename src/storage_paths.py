from __future__ import annotations

import getpass
import os

DEFAULT_STORAGE_ROOT = "/projects/prjs2014"


def _current_user() -> str:
    user = os.environ.get("USER")
    if user:
        return user
    try:
        return getpass.getuser()
    except Exception:
        return "unknown_user"


def storage_root() -> str:
    return os.environ.get("VLM_UQ_STORAGE_ROOT", DEFAULT_STORAGE_ROOT)


def user_storage_root() -> str:
    return os.environ.get(
        "VLM_UQ_USER_STORAGE_ROOT",
        os.path.join(storage_root(), _current_user()),
    )


def default_supervision_output_root() -> str:
    return os.environ.get("VLM_UQ_OUTPUT_ROOT", storage_root())


def hf_home() -> str:
    return os.environ.get("HF_HOME", os.path.join(user_storage_root(), "hf_home"))


def hf_datasets_cache() -> str:
    return os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(user_storage_root(), "hf_datasets"),
    )


def transformers_cache() -> str:
    return os.environ.get(
        "TRANSFORMERS_CACHE",
        os.path.join(user_storage_root(), "hf_transformers"),
    )


def prep_cache_dir() -> str:
    return os.environ.get(
        "VLM_UQ_PREP_CACHE",
        os.path.join(user_storage_root(), "vlm_uq_prepared_cache"),
    )


def image_resolver_cache_dir() -> str:
    return os.environ.get(
        "VLM_UQ_IMAGE_RESOLVER_CACHE",
        os.path.join(user_storage_root(), "image_resolver_cache"),
    )


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def configure_hf_cache_env() -> None:
    defaults = {
        "HF_HOME": hf_home(),
        "HF_DATASETS_CACHE": hf_datasets_cache(),
        "TRANSFORMERS_CACHE": transformers_cache(),
        "VLM_UQ_PREP_CACHE": prep_cache_dir(),
    }

    for env_name, path in defaults.items():
        os.environ.setdefault(env_name, path)
        ensure_dir(os.environ[env_name])
