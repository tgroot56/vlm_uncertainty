from __future__ import annotations

import csv
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.storage_paths import ensure_dir, hf_datasets_cache, user_storage_root

try:
    from PIL import Image
except Exception:  # pragma: no cover - the runtime environment should have PIL.
    Image = None  # type: ignore


MSTS_PROMPTS_URL = (
    "https://raw.githubusercontent.com/paul-rottger/msts-multimodal-safety/"
    "main/data/prompts/english_multimodal.csv"
)
MSTS_UNSAFE_IMAGES_URL = (
    "https://raw.githubusercontent.com/paul-rottger/msts-multimodal-safety/"
    "main/data/images/unsafe_images.csv"
)
EXPECTED_SHOULD_I_PROMPTS = 200
MAX_IMAGE_HEIGHT = 1400
_MSTS_HF_IMAGE_INDEX: Optional[Dict[str, object]] = None


def _msts_root() -> Path:
    root = os.environ.get("VLM_UQ_MSTS_ROOT")
    if root:
        return Path(ensure_dir(root))
    return Path(ensure_dir(os.path.join(user_storage_root(), "msts")))


def _download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vlm-uncertainty-msts"})
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            tmp_path.write_bytes(response.read())
        tmp_path.replace(path)
    except (urllib.error.URLError, TimeoutError, OSError):
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _image_extension_from_url(url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def _wikimedia_original_url(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "upload.wikimedia.org" or "/thumb/" not in parsed.path:
        return None

    prefix, thumb_tail = parsed.path.split("/thumb/", 1)
    tail_parts = thumb_tail.split("/")
    if len(tail_parts) < 4:
        return None

    original_path = f"{prefix}/{'/'.join(tail_parts[:-1])}"
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, original_path, "", "", "")
    )


def _prepare_image(path: Path) -> None:
    if Image is None:
        raise RuntimeError("PIL is required to prepare MSTS images.")

    with Image.open(path) as image:
        prepared = image.convert("RGB")
        if prepared.size[1] > MAX_IMAGE_HEIGHT:
            ratio = MAX_IMAGE_HEIGHT / prepared.size[1]
            prepared = prepared.resize(
                (int(ratio * prepared.size[0]), int(ratio * prepared.size[1]))
            )
        prepared.save(path)


def _load_hf_image_index() -> Dict[str, object]:
    global _MSTS_HF_IMAGE_INDEX
    if _MSTS_HF_IMAGE_INDEX is not None:
        return _MSTS_HF_IMAGE_INDEX

    from datasets import load_dataset

    dataset = load_dataset(
        "felfri/MSTS",
        split="english",
        cache_dir=hf_datasets_cache(),
    )
    image_index: Dict[str, object] = {}
    for row in dataset:
        image_id = row.get("unsafe_image_id")
        image = row.get("unsafe_image")
        if image_id and image is not None and image_id not in image_index:
            image_index[image_id] = image

    _MSTS_HF_IMAGE_INDEX = image_index
    return image_index


def _write_hf_fallback_image(image_id: str, image_path: Path) -> bool:
    if Image is None:
        return False

    image = _load_hf_image_index().get(image_id)
    if image is None:
        return False

    image.convert("RGB").save(image_path)
    _prepare_image(image_path)
    return True


def _download_image(image_id: str, image_url: str, image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{image_id}{_image_extension_from_url(image_url)}"
    if image_path.exists() and image_path.stat().st_size > 0:
        _prepare_image(image_path)
        return image_path

    candidate_urls = [image_url]
    wikimedia_original = _wikimedia_original_url(image_url)
    if wikimedia_original is not None:
        candidate_urls.append(wikimedia_original)

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=image_id, suffix=".download", dir=str(image_dir))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    last_error: Optional[BaseException] = None
    try:
        for candidate_url in candidate_urls:
            request = urllib.request.Request(
                candidate_url,
                headers={"User-Agent": "vlm-uncertainty-msts"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    tmp_path.write_bytes(response.read())
                tmp_path.replace(image_path)
                _prepare_image(image_path)
                return image_path
            except Exception as exc:
                last_error = exc
                if tmp_path.exists():
                    tmp_path.unlink()
                continue
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        if image_path.exists():
            image_path.unlink()
        raise

    if image_path.exists():
        image_path.unlink()
    if _write_hf_fallback_image(image_id, image_path):
        return image_path
    raise RuntimeError(
        f"Failed to download MSTS image {image_id} from candidates: {candidate_urls}"
    ) from last_error


def prepare_msts_split(
    split_obj,
    seed_offset: int,
    max_samples: Optional[int],
) -> List[Dict]:
    """Prepare the English MSTS assistance prompts as local image-text samples."""
    del split_obj, seed_offset

    root = _msts_root()
    raw_dir = root / "raw"
    image_dir = root / "unsafe_images"
    prompts_path = raw_dir / "english_multimodal.csv"
    images_path = raw_dir / "unsafe_images.csv"

    _download_file(MSTS_PROMPTS_URL, prompts_path)
    _download_file(MSTS_UNSAFE_IMAGES_URL, images_path)

    prompt_rows = _read_csv(prompts_path)
    image_rows = _read_csv(images_path)
    image_by_id = {row["unsafe_image_id"]: row for row in image_rows}

    should_i_rows = [
        row for row in prompt_rows
        if (row.get("prompt_text") or "").startswith("Should I")
    ]
    if len(should_i_rows) != EXPECTED_SHOULD_I_PROMPTS:
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_SHOULD_I_PROMPTS} MSTS prompts beginning with 'Should I', "
            f"found {len(should_i_rows)}."
        )

    selected_rows = should_i_rows
    if max_samples is not None:
        selected_rows = selected_rows[: min(max_samples, len(selected_rows))]

    samples: List[Dict] = []
    iterator = tqdm(selected_rows, desc="Preparing MSTS", leave=True)
    for idx, row in enumerate(iterator):
        unsafe_image_id = row["unsafe_image_id"]
        image_meta = image_by_id.get(unsafe_image_id)
        if image_meta is None:
            raise RuntimeError(f"MSTS image metadata missing for {unsafe_image_id!r}.")

        image_path = _download_image(
            image_id=unsafe_image_id,
            image_url=image_meta["unsafe_image_url"],
            image_dir=image_dir,
        )

        prompt_text = row["prompt_text"]
        samples.append(
            {
                "idx": idx,
                "dataset_id": "msts",
                "image_path": str(image_path),
                "prompt": prompt_text,
                "prompt_text": prompt_text,
                "prompt_id": row.get("prompt_id"),
                "prompt_type": row.get("prompt_type"),
                "case_id": row.get("case_id"),
                "case_text": row.get("case_text"),
                "hazard_category": row.get("hazard_category"),
                "hazard_subcategory": row.get("hazard_subcategory"),
                "hazard_subsubcategory": row.get("hazard_subsubcategory"),
                "unsafe_image_id": unsafe_image_id,
                "unsafe_image_description": row.get("unsafe_image_description"),
                "unsafe_image_url": image_meta.get("unsafe_image_url"),
                "unsafe_image_license": image_meta.get("unsafe_image_license"),
                "unsafe_image_cw": image_meta.get("unsafe_image_cw"),
                "manual_label": None,
            }
        )

    return samples
