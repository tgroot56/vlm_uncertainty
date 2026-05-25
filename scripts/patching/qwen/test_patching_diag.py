"""One-off diagnostic: exercise the calibrate_blur prediction path for Qwen."""
import os
import sys

sys.path.insert(0, "/gpfs/home4/tgroot/vlm_uncertainty")

from src.data.load import load_dataset_prepared
from src.vlm.adapters import load_model
from src.vlm import prepare_inputs
import torch

model, processor, device, adapter = load_model("Qwen/Qwen3.5-9B")
print("adapter.family:", adapter.family)

samples = load_dataset_prepared(
    dataset_id="coco-qa-vi", split=None, seed_offset=42, max_samples=5
)
print("loaded", len(samples), "samples")

for s in samples[:3]:
    prompt = s["prompt"]
    img = s["image"]
    gt = s.get("gt_answer")
    print("\n=== prompt:", prompt[:160])
    print("gt_answer:", gt)

    inputs = prepare_inputs(adapter, processor, img, prompt, device)
    print("input_ids shape:", tuple(inputs["input_ids"].shape),
          "last tokens:", inputs["input_ids"][0, -8:].tolist())

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
        )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    pred_raw = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    pred_skip = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    print("gen token ids:", gen_ids[0].tolist())
    print("raw pred (no skip):", repr(pred_raw))
    print("skipped pred (4):", repr(pred_skip))

    with torch.inference_mode():
        out2 = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            return_dict_in_generate=True,
        )
    gen_ids2 = out2.sequences[:, prompt_len:]
    pred_skip2 = processor.batch_decode(gen_ids2, skip_special_tokens=True)[0].strip()
    print("skipped pred (20):", repr(pred_skip2))
