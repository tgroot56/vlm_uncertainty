from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import random
import string
import os
import json

import torch

QUESTION_DEFAULT = "Which object is shown in the image?"


# ----------------------------
# Option keys (supports K>26)
# ----------------------------

def _option_keys(k: int) -> List[str]:
    """
    Produce option keys: A..Z, AA..AZ, BA..BZ, ...
    Supports large K (enough for 200).
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    alphabet = list(string.ascii_uppercase)
    keys: List[str] = []
    i = 0
    while len(keys) < k:
        n = i
        s = ""
        while True:
            s = alphabet[n % 26] + s
            n = n // 26 - 1
            if n < 0:
                break
        keys.append(s)
        i += 1
    return keys


# ----------------------------
# CLIP sampler
# ----------------------------

@dataclass
class ClipNegSamplerConfig:
    """
    mode:
      - "image_to_text": distractors are labels whose text is closest to the image embedding (hard but slower)
      - "gt_to_text": distractors are labels whose text is closest to the GT label text embedding (cheaper)

    top_pool:
      - sample from the top-N closest labels (excluding GT), keeps negatives hard but still randomized.

    clip_model_name:
      - HF CLIP checkpoint

    device:
      - "cuda" recommended if using image_to_text on many samples
    """
    mode: str = "gt_to_text"
    top_pool: int = 30
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32


class ClipHardNegativeSampler:
    def __init__(self, class_names: Sequence[str], cfg: ClipNegSamplerConfig):
        self.class_names = list(class_names)
        self.cfg = cfg

        if cfg.mode not in ("image_to_text", "gt_to_text"):
            raise ValueError(f"Unknown cfg.mode='{cfg.mode}'")

        self.device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._processor = None
        self._model = None

        self._text_emb: Optional[torch.Tensor] = None  # [C, D] normalized
        self._name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(self.class_names)}
        self._gt_text_emb_cache: Dict[str, torch.Tensor] = {}

    def _ensure_clip_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._text_emb is not None:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise ImportError(
                "CLIP hard negatives requested, but transformers is not available. "
                "Install transformers or disable CLIP negatives."
            ) from e

        self._processor = CLIPProcessor.from_pretrained(self.cfg.clip_model_name)
        self._model = CLIPModel.from_pretrained(self.cfg.clip_model_name).to(self.device)
        self._model.eval()

        texts = [f"a photo of a {name}" for name in self.class_names]
        with torch.no_grad():
            inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self._model.get_text_features(**inputs)  # [C, D]
            text_features = text_features.to(self.cfg.dtype)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        self._text_emb = text_features

    @torch.no_grad()
    def _image_embedding(self, pil_image) -> torch.Tensor:
        self._ensure_clip_loaded()
        assert self._processor is not None and self._model is not None

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        img_feat = self._model.get_image_features(**inputs)  # [1, D]
        img_feat = img_feat.to(self.cfg.dtype)
        img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
        return img_feat[0]  # [D]

    def _gt_text_embedding(self, gt_label: str) -> torch.Tensor:
        self._ensure_clip_loaded()
        assert self._text_emb is not None

        if gt_label in self._gt_text_emb_cache:
            return self._gt_text_emb_cache[gt_label]

        idx = self._name_to_idx.get(gt_label)
        if idx is None:
            raise ValueError(f"GT label '{gt_label}' not in class_names.")
        emb = self._text_emb[idx]
        self._gt_text_emb_cache[gt_label] = emb
        return emb

    def sample_distractors(
        self,
        *,
        pil_image,
        gt_label: str,
        num_distractors: int,
        seed: int,
    ) -> List[str]:
        if num_distractors <= 0:
            return []

        self._ensure_clip_loaded()
        assert self._text_emb is not None

        gt_idx = self._name_to_idx.get(gt_label)
        if gt_idx is None:
            raise ValueError(f"GT label '{gt_label}' not in class_names.")

        if self.cfg.mode == "image_to_text":
            query = self._image_embedding(pil_image)  # [D]
        else:
            query = self._gt_text_embedding(gt_label)  # [D]

        sims = self._text_emb @ query  # [C]
        sims = sims.clone()
        sims[gt_idx] = float("-inf")

        top_pool = max(self.cfg.top_pool, num_distractors)
        top_pool = min(top_pool, len(self.class_names) - 1)
        _, top_idx = torch.topk(sims, k=top_pool, largest=True)

        candidate_labels = [self.class_names[i] for i in top_idx.tolist()]

        rng = random.Random(seed)
        if len(candidate_labels) <= num_distractors:
            return candidate_labels
        return rng.sample(candidate_labels, num_distractors)


# ----------------------------
# Prompt building
# ----------------------------

def build_mc_prompt(
    *,
    question: str,
    gt_label: str,
    options: List[str],
) -> Dict[str, Any]:
    k = len(options)
    keys = _option_keys(k)
    option_map = {keys[i]: options[i] for i in range(k)}
    gt_key = next(k for k, v in option_map.items() if v == gt_label)

    prompt = (
        f"{question}\n\n"
        "Choices:\n"
        + "\n".join(f"{k}. {v}" for k, v in option_map.items())
        + f"\n\nAnswer with only one of: {', '.join(keys)}."
    )

    out = {"prompt": prompt, "option_map": option_map, "gt_key": gt_key}

    # Backward compatibility: if K==4 and keys are A/B/C/D, also provide gt_letter
    if k == 4 and list(option_map.keys()) == ["A", "B", "C", "D"]:
        out["gt_letter"] = gt_key

    return out


# ----------------------------
# Debug dump helper (PROMPT + IMG)
# ----------------------------

def _debug_dump_first_sample(sample: Dict[str, Any]) -> None:
    """
    For inspection:
    - prints the exact prompt to stdout
    - saves image to quick_tests/outputs/image_r.png
    - saves prompt/options metadata to quick_tests/outputs/imagenet_r_prompt.json
    """
    out_dir = os.path.join("quick_tests", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Print prompt
    print("\n" + "=" * 100)
    print("DEBUG (ImageNet-R) — First prepared sample prompt:")
    print(sample["prompt"])
    print("=" * 100 + "\n")

    # 2) Save image
    img_path = os.path.join(out_dir, "image_r.png")
    try:
        img = sample.get("image", None)
        if img is not None:
            # HF image objects are usually PIL Images; both .save and PIL.Image.Image work.
            img.save(img_path)
            print(f"Saved ImageNet-R debug image to: {img_path}")
        else:
            print("WARNING: sample['image'] is None, cannot save debug image.")
    except Exception as e:
        print(f"WARNING: Failed to save debug image to {img_path}: {e}")

    # 3) Save JSON with prompt + options
    json_path = os.path.join(out_dir, "imagenet_r_prompt.json")
    payload = {
        "idx": sample.get("idx"),
        "gt_class": sample.get("gt_class"),
        "gt_key": sample.get("gt_key"),
        "gt_letter": sample.get("gt_letter"),  # present only for 4-way
        "k_options": len(sample.get("option_map", {})) if isinstance(sample.get("option_map"), dict) else None,
        "option_map": sample.get("option_map"),
        "prompt": sample.get("prompt"),
    }
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved ImageNet-R prompt/options JSON to: {json_path}")
    except Exception as e:
        print(f"WARNING: Failed to save JSON to {json_path}: {e}")


# ----------------------------
# Split preparation (public)
# ----------------------------

def prepare_imagenet_r_split(
    split_obj,
    seed_offset: int,
    max_samples: Optional[int],
    *,
    k_options: int = 4,
    question: str = QUESTION_DEFAULT,
    use_clip_hard_negatives: bool = False,
    clip_cfg: Optional[ClipNegSamplerConfig] = None,
    debug_dump: bool = False,
) -> List[Dict]:
    """
    Prepare ImageNet-R as K-way MC.
    Defaults match your previous 4-way random distractor logic.

    Returns dicts containing:
      idx, image, gt_class, prompt, option_map, gt_key (and gt_letter if k==4)

    debug_dump:
      If True, dumps the first prepared sample (prompt + image + json) to quick_tests/outputs/
    """
    if k_options < 2:
        raise ValueError("k_options must be >= 2")

    if max_samples is not None:
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    all_class_names = sorted(set(split_obj["class_name"]))
    if k_options > len(all_class_names):
        raise ValueError(f"k_options={k_options} > num_classes={len(all_class_names)}")

    clip_sampler: Optional[ClipHardNegativeSampler] = None
    if use_clip_hard_negatives:
        cfg = clip_cfg or ClipNegSamplerConfig()
        clip_sampler = ClipHardNegativeSampler(all_class_names, cfg)

    samples: List[Dict] = []
    for idx in range(len(split_obj)):
        row = split_obj[idx]
        gt_class = row["class_name"]
        img = row["image"]

        rng_seed = seed_offset + idx
        rng = random.Random(rng_seed)

        num_distractors = k_options - 1

        if clip_sampler is None:
            distractors = rng.sample([c for c in all_class_names if c != gt_class], num_distractors)
        else:
            distractors = clip_sampler.sample_distractors(
                pil_image=img,
                gt_label=gt_class,
                num_distractors=num_distractors,
                seed=rng_seed,
            )

        options = [gt_class] + distractors
        rng.shuffle(options)

        mc = build_mc_prompt(question=question, gt_label=gt_class, options=options)

        sample = {
            "idx": idx,
            "image": img,
            "gt_class": gt_class,
            **mc,
        }

        samples.append(sample)

        # Only dump the first sample once
        if debug_dump and idx == 0:
            _debug_dump_first_sample(sample)

    return samples
