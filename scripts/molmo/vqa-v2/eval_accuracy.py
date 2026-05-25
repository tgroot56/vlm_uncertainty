"""Lightweight accuracy-only evaluation of Molmo2 on VQA-v2.

Runs model.generate directly (no feature extraction) and scores each answer
with src.labeling.vqa.score_vqa, then reports the mean soft accuracy. Intended
as a quick sanity check before committing to a full supervision dataset run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.load import load_dataset_prepared
from src.labeling.vqa import score_vqa
from src.storage_paths import configure_hf_cache_env
from src.vlm import prepare_inputs
from src.vlm.adapters import load_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", default="allenai/Molmo2-O-7B")
    parser.add_argument("--dataset", default="vqa-v2")
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def _predict(model, processor, device, adapter, image, prompt, max_new_tokens):
    inputs = prepare_inputs(adapter, processor, image, prompt, device)
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
    )
    if adapter.family == "molmo":
        generate_kwargs["use_cache"] = False
    out = model.generate(**generate_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    configure_hf_cache_env()
    args = _parse_args()

    print(f"Loading model: {args.vlm}")
    model, processor, device, adapter = load_model(args.vlm, args.device)
    print(f"Adapter family: {adapter.family}")

    samples = load_dataset_prepared(
        dataset_id=args.dataset,
        split=args.dataset_split,
        seed_offset=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples)} samples")

    t0 = time.time()
    scores: list[float] = []
    num_nonempty = 0
    sample_log: list[tuple[str, str, float]] = []

    for sample in tqdm(samples, desc="evaluate"):
        pred = _predict(
            model, processor, device, adapter,
            image=sample["image"],
            prompt=sample["prompt"],
            max_new_tokens=args.max_new_tokens,
        )
        if pred:
            num_nonempty += 1
        result = score_vqa(
            pred_answer=pred,
            gt_answers=sample.get("gt_answers"),
            gt_answer_single=sample.get("gt_answer"),
        )
        scores.append(float(result.score))

        if len(sample_log) < 20:
            gt = sample.get("gt_answer") or (
                sample.get("gt_answers", [""])[0] if sample.get("gt_answers") else ""
            )
            sample_log.append((pred, gt, float(result.score)))

    elapsed = time.time() - t0
    mean_acc = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "=" * 60)
    print(f"Dataset: {args.dataset} (split={args.dataset_split or 'default'})")
    print(f"Model:   {args.vlm}")
    print(f"Samples: {len(scores)}")
    print(f"Non-empty predictions: {num_nonempty}/{len(scores)}")
    print(f"Mean soft accuracy: {mean_acc:.4f}")
    print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/max(len(scores),1):.2f} s/sample)")
    print("=" * 60)

    print("\nFirst 20 samples:")
    for pred, gt, score in sample_log:
        print(f"  pred={pred!r:40s} | gt={gt!r:20s} | score={score:.2f}")


if __name__ == "__main__":
    main()
