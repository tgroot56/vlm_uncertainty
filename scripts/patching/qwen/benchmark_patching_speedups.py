"""Benchmark Qwen patching speedups — Qwen-only, non-invasive.

Measures per-generate-call time for 5 variants on 2 samples:

  V0: baseline — matches current repo `_patched_generate` exactly
  V1: V0 + cache `inputs` once per (sample, severity) (skip re-tokenize / re-process)
  V2: V1 + reduce max_new_tokens from 20 to 8 (dataset_default + 4-token preamble)
  V3: V2 + prefill `<think>\\n</think>\\n\\n` at prompt end (skip 4-token preamble)
  V4: V3 + cache fused inputs_embeds (skip vision tower + image-embed scatter)

Emits a table of mean times per call and speedup vs V0.

Run as a slurm job:
  sbatch scripts/patching/qwen/benchmark_patching_speedups.job
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/gpfs/home4/tgroot/vlm_uncertainty")

from src.vlm.adapters import load_model
from src.vlm import prepare_inputs, qwen_patching_max_new_tokens
from src.patching.run_patching import (
    _get_decoder_layers,
    _qwen_post_generate,
)


# -- config ------------------------------------------------------------------
MODEL_ID = os.environ.get("VLM_ID", "Qwen/Qwen3.5-9B")
DATASET_JSON = os.environ.get(
    "PATCHING_DATASET",
    "/projects/prjs2014/patching/Qwen_Qwen3-5-9B/coco-qa-vi/patching_dataset_coco-qa-vi.json",
)
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "2"))
NUM_CALLS_PER_VARIANT = int(os.environ.get("NUM_CALLS", "10"))  # per (sample, severity)
PATCH_LAYER_IDX = int(os.environ.get("PATCH_LAYER", "17"))
DATASET_DEFAULT_MAX_NEW = 4  # coco-qa-vi default


def _format_prompt(q: str) -> str:
    return f"Question: {q}\nProvide a short answer."


def _build_clean_activation(
    model: Any,
    adapter: Any,
    processor: Any,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    layer_idx: int,
) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
    """Run one clean forward, capture the layer_idx output at the final prompt token."""
    inputs = prepare_inputs(adapter, processor, image, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[1])

    captured = {}
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[:, -1, :].detach().clone()
        return output

    decoder_layers = _get_decoder_layers(model, adapter)
    handle = decoder_layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            _ = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    return captured["h"], prompt_len, inputs


# ---------- V0: baseline, exactly matching repo's _patched_generate ----------

def patched_generate_v0(
    *, model, processor, device, adapter,
    image, prompt, max_new_tokens, layer_idx, clean_act,
) -> Dict[str, Any]:
    inputs = prepare_inputs(adapter, processor, image, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[1])

    decoder_layers = _get_decoder_layers(model, adapter)
    target_layer = decoder_layers[layer_idx]
    clean_act_d = clean_act.to(device=device, dtype=next(model.parameters()).dtype)

    patched_flag = {"v": False}
    def hook(module, args, output):
        if patched_flag["v"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < prompt_len:
            return output
        h = h.clone()
        h[0, -1, :] = clean_act_d[0]
        patched_flag["v"] = True
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = target_layer.register_forward_hook(hook)
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=qwen_patching_max_new_tokens(adapter, max_new_tokens),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    gen_logits = torch.stack(out.scores, dim=1) if len(out.scores) > 0 else torch.empty(0)
    gen_ids, gen_logits, raw_text = _qwen_post_generate(adapter, gen_ids, gen_logits, processor)
    return {"raw_text": raw_text, "first_top1": _first_top1(gen_logits)}


# ---------- V1: cache inputs ----------

def patched_generate_v1(
    *, model, processor, device, adapter,
    cached_inputs, prompt_len, max_new_tokens, layer_idx, clean_act,
) -> Dict[str, Any]:
    decoder_layers = _get_decoder_layers(model, adapter)
    target_layer = decoder_layers[layer_idx]
    clean_act_d = clean_act.to(device=device, dtype=next(model.parameters()).dtype)

    patched_flag = {"v": False}
    def hook(module, args, output):
        if patched_flag["v"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < prompt_len:
            return output
        h = h.clone()
        h[0, -1, :] = clean_act_d[0]
        patched_flag["v"] = True
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = target_layer.register_forward_hook(hook)
    try:
        out = model.generate(
            **cached_inputs,
            max_new_tokens=qwen_patching_max_new_tokens(adapter, max_new_tokens),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    gen_logits = torch.stack(out.scores, dim=1) if len(out.scores) > 0 else torch.empty(0)
    gen_ids, gen_logits, raw_text = _qwen_post_generate(adapter, gen_ids, gen_logits, processor)
    return {"raw_text": raw_text, "first_top1": _first_top1(gen_logits)}


# ---------- V2: V1 + reduced max_new_tokens ----------

def patched_generate_v2(
    *, model, processor, device, adapter,
    cached_inputs, prompt_len, max_new_tokens, layer_idx, clean_act,
) -> Dict[str, Any]:
    # max_new_tokens = dataset_default + 4 preamble = 8 for coco-qa-vi
    mnt = max_new_tokens + 4
    decoder_layers = _get_decoder_layers(model, adapter)
    target_layer = decoder_layers[layer_idx]
    clean_act_d = clean_act.to(device=device, dtype=next(model.parameters()).dtype)

    patched_flag = {"v": False}
    def hook(module, args, output):
        if patched_flag["v"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < prompt_len:
            return output
        h = h.clone()
        h[0, -1, :] = clean_act_d[0]
        patched_flag["v"] = True
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = target_layer.register_forward_hook(hook)
    try:
        out = model.generate(
            **cached_inputs,
            max_new_tokens=mnt,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    gen_logits = torch.stack(out.scores, dim=1) if len(out.scores) > 0 else torch.empty(0)
    gen_ids, gen_logits, raw_text = _qwen_post_generate(adapter, gen_ids, gen_logits, processor)
    return {"raw_text": raw_text, "first_top1": _first_top1(gen_logits)}


# ---------- V3: V2 + prefill <think></think> preamble ----------

_QWEN_THINK_START = 248068   # <think>
_QWEN_THINK_END = 248069     # </think>
_QWEN_NEWLINE = 271          # \n\n  (actually \n but 271 appears in gen)


def _prefill_thinking_inputs(cached_inputs: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """Append `<think>\\n</think>\\n\\n` tokens (248068, 271, 248069, 271) to the prompt.

    After this, Qwen starts generating the actual answer at position 0 of gen_ids.
    """
    preamble = torch.tensor([[_QWEN_THINK_START, _QWEN_NEWLINE, _QWEN_THINK_END, _QWEN_NEWLINE]],
                             device=device, dtype=cached_inputs["input_ids"].dtype)
    new_input_ids = torch.cat([cached_inputs["input_ids"], preamble], dim=1)

    new_inputs = {k: v for k, v in cached_inputs.items()}
    new_inputs["input_ids"] = new_input_ids

    if "attention_mask" in cached_inputs:
        pad = torch.ones((1, 4), device=device, dtype=cached_inputs["attention_mask"].dtype)
        new_inputs["attention_mask"] = torch.cat([cached_inputs["attention_mask"], pad], dim=1)

    # Extend mm_token_type_ids if present — new tokens are plain text (type 0)
    if "mm_token_type_ids" in cached_inputs:
        pad = torch.zeros((1, 4), device=device, dtype=cached_inputs["mm_token_type_ids"].dtype)
        new_inputs["mm_token_type_ids"] = torch.cat([cached_inputs["mm_token_type_ids"], pad], dim=1)

    return new_inputs


def patched_generate_v3(
    *, model, processor, device, adapter,
    prefilled_inputs, prompt_len, max_new_tokens, layer_idx, clean_act,
) -> Dict[str, Any]:
    # prompt_len here includes the 4 preamble tokens
    decoder_layers = _get_decoder_layers(model, adapter)
    target_layer = decoder_layers[layer_idx]
    clean_act_d = clean_act.to(device=device, dtype=next(model.parameters()).dtype)

    # IMPORTANT: we want to patch at position -5 (the last real prompt token, BEFORE the preamble)
    # because that's the position the clean activation corresponds to.
    patched_flag = {"v": False}
    patch_pos = prompt_len - 5  # real last prompt token index
    def hook(module, args, output):
        if patched_flag["v"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < prompt_len:
            return output
        h = h.clone()
        h[0, patch_pos, :] = clean_act_d[0]
        patched_flag["v"] = True
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = target_layer.register_forward_hook(hook)
    try:
        out = model.generate(
            **prefilled_inputs,
            max_new_tokens=max_new_tokens,  # no preamble offset needed
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    gen_logits = torch.stack(out.scores, dim=1) if len(out.scores) > 0 else torch.empty(0)
    # No need for _qwen_post_generate strip — preamble is already in prompt
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return {"raw_text": raw_text, "first_top1": _first_top1(gen_logits)}


# ---------- V4: V3 + cache fused inputs_embeds (skip vision tower) ----------

def _compute_fused_inputs_embeds(model, prefilled_inputs):
    """Run the multimodal embedding fusion ONCE, returning inputs_embeds
    with the vision embeddings already scattered into image-token positions.

    After this, we can call model.generate(inputs_embeds=...) which skips
    the vision tower + masked_scatter on each call.
    """
    input_ids = prefilled_inputs["input_ids"]
    pixel_values = prefilled_inputs.get("pixel_values")
    image_grid_thw = prefilled_inputs.get("image_grid_thw")

    # Get the inner multimodal model (not the top-level ForConditionalGeneration)
    mm_model = model.model if hasattr(model, "model") else model

    with torch.inference_mode():
        inputs_embeds = mm_model.get_input_embeddings()(input_ids)
        if pixel_values is not None and image_grid_thw is not None:
            img_out = mm_model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = img_out.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = mm_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds.detach()


def patched_generate_v4(
    *, model, processor, device, adapter,
    fused_embeds, attention_mask, prompt_len, max_new_tokens, layer_idx, clean_act,
) -> Dict[str, Any]:
    decoder_layers = _get_decoder_layers(model, adapter)
    target_layer = decoder_layers[layer_idx]
    clean_act_d = clean_act.to(device=device, dtype=next(model.parameters()).dtype)

    patched_flag = {"v": False}
    patch_pos = prompt_len - 5
    def hook(module, args, output):
        if patched_flag["v"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < prompt_len:
            return output
        h = h.clone()
        h[0, patch_pos, :] = clean_act_d[0]
        patched_flag["v"] = True
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = target_layer.register_forward_hook(hook)
    try:
        # Use inputs_embeds path — skips vision tower on generate
        # Note: not all VLM generate() support this. Try and see.
        out = model.generate(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        handle.remove()

    # With inputs_embeds, out.sequences only contains generated tokens (no input)
    gen_ids = out.sequences
    gen_logits = torch.stack(out.scores, dim=1) if len(out.scores) > 0 else torch.empty(0)
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return {"raw_text": raw_text, "first_top1": _first_top1(gen_logits)}


def _first_top1(gen_logits):
    if gen_logits.numel() == 0 or gen_logits.shape[1] == 0:
        return float("nan")
    probs = F.softmax(gen_logits[0, 0, :].float(), dim=0)
    return float(probs.max().item())


# ---------- timing harness ----------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_variant(fn, warmup: int, n: int, **kwargs) -> Tuple[float, float, Any]:
    """Return (mean_sec, std_sec, last_result)."""
    for _ in range(warmup):
        fn(**kwargs)
    _sync()
    times = []
    last = None
    for _ in range(n):
        _sync()
        t0 = time.perf_counter()
        last = fn(**kwargs)
        _sync()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std, last


# ---------- main ----------

def main():
    print(f"Loading {MODEL_ID}...", flush=True)
    model, processor, device, adapter = load_model(MODEL_ID)
    print(f"  loaded on {device}", flush=True)

    with open(DATASET_JSON) as f:
        data = json.load(f)
    samples = data["samples"][:NUM_SAMPLES]
    print(f"  using {len(samples)} samples, {NUM_CALLS_PER_VARIANT} timed calls per variant", flush=True)

    results = []
    for s_idx, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        prompt = _format_prompt(sample["clean_question"])
        clean_image = Image.open(sample["clean_image_path"]).convert("RGB")

        pert = sample["visual_perturbations"][0]  # extreme
        blurred_image = Image.open(pert["blurred_image_path"]).convert("RGB")

        print(f"\n===== sample {sample_id} (prompt: {sample['clean_question'][:60]}) =====", flush=True)

        # Prepare clean activation at layer PATCH_LAYER_IDX, last prompt token
        clean_act, clean_prompt_len, _ = _build_clean_activation(
            model, adapter, processor, device, clean_image, prompt, PATCH_LAYER_IDX,
        )
        print(f"  clean prompt_len={clean_prompt_len}, clean_act shape={tuple(clean_act.shape)}", flush=True)

        # Pre-compute cached inputs (for V1+) on the blurred image
        cached_inputs = prepare_inputs(adapter, processor, blurred_image, prompt, device)
        blurred_prompt_len = int(cached_inputs["input_ids"].shape[1])
        print(f"  blurred prompt_len={blurred_prompt_len}", flush=True)

        # Pre-compute prefilled inputs (for V3+)
        prefilled_inputs = _prefill_thinking_inputs(cached_inputs, device)
        prefilled_prompt_len = int(prefilled_inputs["input_ids"].shape[1])
        print(f"  prefilled prompt_len={prefilled_prompt_len} (+4 preamble)", flush=True)

        # Pre-compute fused inputs_embeds (for V4)
        try:
            fused_embeds = _compute_fused_inputs_embeds(model, prefilled_inputs)
            fused_attn = prefilled_inputs["attention_mask"]
            v4_ok = True
            print(f"  fused_embeds shape={tuple(fused_embeds.shape)}", flush=True)
        except Exception as e:
            v4_ok = False
            print(f"  [V4 setup FAILED] {type(e).__name__}: {e}", flush=True)

        common = dict(
            model=model, processor=processor, device=device, adapter=adapter,
            layer_idx=PATCH_LAYER_IDX, clean_act=clean_act,
            max_new_tokens=DATASET_DEFAULT_MAX_NEW,
        )

        # V0 baseline
        print("  timing V0 (baseline)...", flush=True)
        v0_mean, v0_std, v0_res = time_variant(
            patched_generate_v0, warmup=2, n=NUM_CALLS_PER_VARIANT,
            image=blurred_image, prompt=prompt, **common,
        )
        print(f"    V0: {v0_mean*1000:.1f} ± {v0_std*1000:.1f} ms | text={v0_res['raw_text'][:40]!r}", flush=True)

        # V1 cached inputs
        print("  timing V1 (cached inputs)...", flush=True)
        v1_mean, v1_std, v1_res = time_variant(
            patched_generate_v1, warmup=2, n=NUM_CALLS_PER_VARIANT,
            cached_inputs=cached_inputs, prompt_len=blurred_prompt_len, **common,
        )
        print(f"    V1: {v1_mean*1000:.1f} ± {v1_std*1000:.1f} ms ({v0_mean/v1_mean:.2f}x) | text={v1_res['raw_text'][:40]!r}", flush=True)

        # V2 reduced max_new_tokens
        print("  timing V2 (+reduced max_new_tokens=8)...", flush=True)
        v2_mean, v2_std, v2_res = time_variant(
            patched_generate_v2, warmup=2, n=NUM_CALLS_PER_VARIANT,
            cached_inputs=cached_inputs, prompt_len=blurred_prompt_len, **common,
        )
        print(f"    V2: {v2_mean*1000:.1f} ± {v2_std*1000:.1f} ms ({v0_mean/v2_mean:.2f}x) | text={v2_res['raw_text'][:40]!r}", flush=True)

        # V3 prefilled thinking
        print("  timing V3 (+prefill <think></think>)...", flush=True)
        v3_mean, v3_std, v3_res = time_variant(
            patched_generate_v3, warmup=2, n=NUM_CALLS_PER_VARIANT,
            prefilled_inputs=prefilled_inputs, prompt_len=prefilled_prompt_len, **common,
        )
        print(f"    V3: {v3_mean*1000:.1f} ± {v3_std*1000:.1f} ms ({v0_mean/v3_mean:.2f}x) | text={v3_res['raw_text'][:40]!r}", flush=True)

        # V4 fused embeds
        if v4_ok:
            try:
                print("  timing V4 (+fused inputs_embeds)...", flush=True)
                v4_mean, v4_std, v4_res = time_variant(
                    patched_generate_v4, warmup=2, n=NUM_CALLS_PER_VARIANT,
                    fused_embeds=fused_embeds, attention_mask=fused_attn,
                    prompt_len=prefilled_prompt_len, **{k: v for k, v in common.items() if k != "image" and k != "prompt"},
                )
                print(f"    V4: {v4_mean*1000:.1f} ± {v4_std*1000:.1f} ms ({v0_mean/v4_mean:.2f}x) | text={v4_res['raw_text'][:40]!r}", flush=True)
            except Exception as e:
                print(f"    V4 FAILED during run: {type(e).__name__}: {e}", flush=True)
                v4_mean = float("nan"); v4_std = float("nan"); v4_res = {"raw_text": "ERR"}
        else:
            v4_mean = float("nan"); v4_std = float("nan"); v4_res = {"raw_text": "SKIPPED"}

        results.append({
            "sample_id": sample_id,
            "V0_ms": v0_mean * 1000, "V1_ms": v1_mean * 1000, "V2_ms": v2_mean * 1000,
            "V3_ms": v3_mean * 1000, "V4_ms": v4_mean * 1000,
            "V1_speedup": v0_mean / v1_mean, "V2_speedup": v0_mean / v2_mean,
            "V3_speedup": v0_mean / v3_mean,
            "V4_speedup": v0_mean / v4_mean if v4_mean == v4_mean else float("nan"),
        })

        clean_image.close(); blurred_image.close()

    print("\n===== SUMMARY =====", flush=True)
    print(f"{'sample':>10} {'V0':>8} {'V1':>8} {'V2':>8} {'V3':>8} {'V4':>8}  (ms)", flush=True)
    for r in results:
        print(f"{r['sample_id']:>10} {r['V0_ms']:>8.0f} {r['V1_ms']:>8.0f} {r['V2_ms']:>8.0f} {r['V3_ms']:>8.0f} {r['V4_ms']:>8.0f}", flush=True)
    print(f"{'speedup':>10} {'-':>8} {'V1':>8} {'V2':>8} {'V3':>8} {'V4':>8}", flush=True)
    for r in results:
        print(f"{r['sample_id']:>10} {'-':>8} {r['V1_speedup']:>8.2f} {r['V2_speedup']:>8.2f} {r['V3_speedup']:>8.2f} {r['V4_speedup']:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
