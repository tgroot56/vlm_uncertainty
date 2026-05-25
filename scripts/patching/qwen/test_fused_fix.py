"""Verify the Qwen vision-cache patching fix.

Runs 2 samples through the updated `_patched_generate` + `_run_baseline_forward`,
comparing the non-fused path (vision tower rerun each call) to the fused path
(vision tower runs ONCE per image, then a monkey-patched `get_image_features`
returns the cached output on every subsequent call). Prints:

  - per-call wall time (fused vs non-fused)
  - top-1 probability and entropy match (should be equal up to numerical noise)
  - generated text match

The fused path keeps the original `input_ids` + `pixel_values` bundle (so
Qwen's M-RoPE 3D position_ids are computed correctly) — it only skips the
expensive vision tower forward, which was the real bottleneck.

This verifies BOTH correctness and speedup in one short run.
"""
import os
import sys
import time
import json

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/gpfs/home4/tgroot/vlm_uncertainty")

from src.vlm.adapters import load_model
from src.vlm import prepare_qwen_fused_inputs_embeds, qwen_vision_cache_context
from src.patching.run_patching import (
    _get_decoder_layers,
    _patched_generate,
    _run_baseline_forward,
)


MODEL_ID = os.environ.get("VLM_ID", "Qwen/Qwen3.5-9B")
DATASET_JSON = os.environ.get(
    "PATCHING_DATASET",
    "/projects/prjs2014/patching/Qwen_Qwen3-5-9B/coco-qa-vi/patching_dataset_coco-qa-vi.json",
)
LAYER = 17
NUM_CALLS = 5


def _format_prompt(q: str) -> str:
    return f"Question: {q}\nProvide a short answer."


def _first_top1_and_entropy(logits):
    if logits.numel() == 0 or logits.shape[1] == 0:
        return float("nan"), float("nan")
    probs = F.softmax(logits[0, 0, :].float(), dim=0)
    top1 = float(probs.max().item())
    entropy = float(-(probs * torch.log(probs + 1e-10)).sum().item())
    return top1, entropy


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_it(fn, n=NUM_CALLS):
    for _ in range(2):  # warmup
        fn()
    _sync()
    times = []
    last = None
    for _ in range(n):
        _sync()
        t0 = time.perf_counter()
        last = fn()
        _sync()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), last


def _capture_clean_activation_at_layer(model, adapter, clean_bundle, layer_idx):
    """Run the clean image through the cached path once, capture the layer
    output at the last prompt-token position. Used as the `clean_activations`
    to inject during patched_generate."""
    decoder_layers = _get_decoder_layers(model, adapter)
    captured = {}

    def hook(m, a, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[:, -1, :].detach().clone()
        return out

    handle = decoder_layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode(), qwen_vision_cache_context(model, clean_bundle):
            _ = model(**clean_bundle["inputs"], use_cache=False)
    finally:
        handle.remove()

    return captured["h"]  # (1, H)


def main():
    print(f"Loading {MODEL_ID}...", flush=True)
    model, processor, device, adapter = load_model(MODEL_ID)

    with open(DATASET_JSON) as f:
        data = json.load(f)
    samples = data["samples"][:2]
    print(f"  testing on {len(samples)} samples", flush=True)

    all_top1_diffs = []
    all_entropy_diffs = []
    all_speedups_baseline = []
    all_speedups_patched = []
    all_text_mismatches = 0

    for sample in samples:
        sid = sample["sample_id"]
        prompt = _format_prompt(sample["clean_question"])
        blurred = Image.open(sample["visual_perturbations"][0]["blurred_image_path"]).convert("RGB")
        clean = Image.open(sample["clean_image_path"]).convert("RGB")

        print(f"\n===== sample {sid}: {sample['clean_question'][:60]} =====", flush=True)

        # Pre-compute fused bundles (one per image). This runs the vision tower
        # ONCE and stores the result in bundle["cached_image_features"].
        blurred_bundle = prepare_qwen_fused_inputs_embeds(
            model, adapter, processor, blurred, prompt, device,
        )
        clean_bundle = prepare_qwen_fused_inputs_embeds(
            model, adapter, processor, clean, prompt, device,
        )
        prompt_len = blurred_bundle["prompt_len"]
        print(f"  prompt_len={prompt_len} (blurred), {clean_bundle['prompt_len']} (clean)", flush=True)

        # Capture clean activation at LAYER, last prompt token
        clean_act = _capture_clean_activation_at_layer(model, adapter, clean_bundle, LAYER)
        print(f"  clean_act shape={tuple(clean_act.shape)}", flush=True)

        # --- Test _run_baseline_forward: non-fused vs fused ---
        def run_baseline_nonfused():
            return _run_baseline_forward(
                model=model, processor=processor, device=device, vlm_adapter=adapter,
                image=blurred, prompt=prompt, max_new_tokens=4,
            )

        def run_baseline_fused():
            return _run_baseline_forward(
                model=model, processor=processor, device=device, vlm_adapter=adapter,
                image=blurred, prompt=prompt, max_new_tokens=4,
                fused_bundle=blurred_bundle,
            )

        t_nonfused, res_nonfused = time_it(run_baseline_nonfused)
        t_fused, res_fused = time_it(run_baseline_fused)
        top1_nf, ent_nf = _first_top1_and_entropy(res_nonfused["gen_step_logits"])
        top1_f, ent_f = _first_top1_and_entropy(res_fused["gen_step_logits"])
        text_match = res_nonfused["raw_text"] == res_fused["raw_text"]
        all_top1_diffs.append(abs(top1_nf - top1_f))
        all_entropy_diffs.append(abs(ent_nf - ent_f))
        all_speedups_baseline.append(t_nonfused / t_fused)
        all_text_mismatches += (0 if text_match else 1)

        print(
            f"  baseline: non-fused={t_nonfused*1000:.0f}ms "
            f"text={res_nonfused['raw_text'][:30]!r} top1={top1_nf:.4f} ent={ent_nf:.4f}",
            flush=True,
        )
        print(
            f"  baseline:     fused={t_fused*1000:.0f}ms "
            f"text={res_fused['raw_text'][:30]!r} top1={top1_f:.4f} ent={ent_f:.4f} "
            f"| speedup={t_nonfused/t_fused:.1f}x  text_match={text_match} "
            f"top1_diff={abs(top1_nf-top1_f):.2e} ent_diff={abs(ent_nf-ent_f):.2e}",
            flush=True,
        )

        # --- Test _patched_generate: non-fused vs fused ---
        def run_patched_nonfused():
            return _patched_generate(
                model=model, processor=processor, device=device, vlm_adapter=adapter,
                image=blurred, prompt=prompt, max_new_tokens=4,
                layer_idx=LAYER, positions=[prompt_len - 1],
                clean_activations=clean_act,
            )

        def run_patched_fused():
            return _patched_generate(
                model=model, processor=processor, device=device, vlm_adapter=adapter,
                image=blurred, prompt=prompt, max_new_tokens=4,
                layer_idx=LAYER, positions=[prompt_len - 1],
                clean_activations=clean_act,
                fused_bundle=blurred_bundle,
            )

        t_nonfused, res_nonfused = time_it(run_patched_nonfused)
        t_fused, res_fused = time_it(run_patched_fused)
        top1_nf, ent_nf = _first_top1_and_entropy(res_nonfused["gen_step_logits"])
        top1_f, ent_f = _first_top1_and_entropy(res_fused["gen_step_logits"])
        text_match = res_nonfused["raw_text"] == res_fused["raw_text"]
        all_top1_diffs.append(abs(top1_nf - top1_f))
        all_entropy_diffs.append(abs(ent_nf - ent_f))
        all_speedups_patched.append(t_nonfused / t_fused)
        all_text_mismatches += (0 if text_match else 1)

        print(
            f"  patched : non-fused={t_nonfused*1000:.0f}ms "
            f"text={res_nonfused['raw_text'][:30]!r} top1={top1_nf:.4f} ent={ent_nf:.4f}",
            flush=True,
        )
        print(
            f"  patched :     fused={t_fused*1000:.0f}ms "
            f"text={res_fused['raw_text'][:30]!r} top1={top1_f:.4f} ent={ent_f:.4f} "
            f"| speedup={t_nonfused/t_fused:.1f}x  text_match={text_match} "
            f"top1_diff={abs(top1_nf-top1_f):.2e} ent_diff={abs(ent_nf-ent_f):.2e}",
            flush=True,
        )

        clean.close(); blurred.close()

    print("\n===== SUMMARY =====", flush=True)
    print(
        f"  baseline speedups: {[f'{s:.1f}x' for s in all_speedups_baseline]}",
        flush=True,
    )
    print(
        f"  patched  speedups: {[f'{s:.1f}x' for s in all_speedups_patched]}",
        flush=True,
    )
    print(
        f"  top1 diffs (max): {max(all_top1_diffs):.2e}  "
        f"(should be <1e-4 with monkey-patch approach)",
        flush=True,
    )
    print(
        f"  entropy diffs (max): {max(all_entropy_diffs):.2e}  "
        f"(should be <1e-3)",
        flush=True,
    )
    print(
        f"  text mismatches: {all_text_mismatches} / {2 * len(samples) * 2}",
        flush=True,
    )


if __name__ == "__main__":
    main()
