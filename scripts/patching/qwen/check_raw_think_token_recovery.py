"""Sanity check Qwen raw first-token recovery under final-token patching.

The stored Qwen patching parquet files report first-token stats after the
leading ``<think>...</think>`` block has been stripped. This script bypasses
that postprocessing and directly compares the next-token logits at the final
prompt position:

  clean prompt/image logits
  corrupted prompt/image logits
  corrupted logits with clean final-prompt-token activation patched in

If the patch is applied at the final decoder layer and the tokenized prompt is
identical, the patched logits should match the clean logits up to numerical
noise.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/gpfs/home4/tgroot/vlm_uncertainty")

from src.patching.collect_activations import _format_prompt
from src.patching.run_patching import _get_decoder_layers
from src.vlm import prepare_inputs
from src.vlm.adapters import load_model


MODEL_ID = os.environ.get("VLM_ID", "Qwen/Qwen3.5-9B")
DEFAULT_DATASET_JSON = (
    "/projects/prjs2014/patching/Qwen_Qwen3-5-9B/coco-qa-vi/"
    "patching_dataset_coco-qa-vi.json"
)
QWEN_THINK_ID = 248068


@dataclass
class LogitStats:
    top_id: int
    top_text: str
    top_prob: float
    think_prob: float
    entropy: float


def _token_text(processor: Any, token_id: int) -> str:
    return processor.tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _stats(processor: Any, logits: torch.Tensor) -> LogitStats:
    probs = F.softmax(logits.float(), dim=-1)
    top_prob, top_id = torch.max(probs, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return LogitStats(
        top_id=int(top_id.item()),
        top_text=_token_text(processor, int(top_id.item())),
        top_prob=float(top_prob.item()),
        think_prob=float(probs[QWEN_THINK_ID].item()),
        entropy=float(entropy.item()),
    )


def _print_stats(name: str, stats: LogitStats) -> None:
    print(
        f"    {name:9s} top={stats.top_id:<7d} {stats.top_text!r:<14s} "
        f"top_p={stats.top_prob:.6f} p(<think>)={stats.think_prob:.6f} "
        f"H={stats.entropy:.4f}",
        flush=True,
    )


def _select_perturbation(sample: dict[str, Any], severity: str) -> dict[str, Any]:
    perturbations = sample.get("visual_perturbations") or []
    for perturbation in perturbations:
        if perturbation.get("severity") == severity:
            return perturbation
    if not perturbations:
        raise ValueError(f"sample {sample.get('sample_id')} has no visual perturbations")
    return perturbations[0]


def _forward_logits_and_capture(
    *,
    model: Any,
    adapter: Any,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_layers = _get_decoder_layers(model, adapter)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[:, -1, :].detach().clone()
        return output

    handle = decoder_layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError(f"did not capture activation at layer {layer_idx}")
    return out.logits[0, -1, :].detach().clone(), captured["activation"]


def _forward_logits_with_patch(
    *,
    model: Any,
    adapter: Any,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
    clean_activation: torch.Tensor,
) -> torch.Tensor:
    decoder_layers = _get_decoder_layers(model, adapter)
    clean_activation = clean_activation.to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    patched = False

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        nonlocal patched
        if patched:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        patched_hidden = hidden.clone()
        patched_hidden[:, -1, :] = clean_activation
        patched = True
        if isinstance(output, tuple):
            return (patched_hidden,) + output[1:]
        return patched_hidden

    handle = decoder_layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if not patched:
        raise RuntimeError(f"did not patch activation at layer {layer_idx}")
    return out.logits[0, -1, :].detach().clone()


def _relative_recovery(clean: float, corrupted: float, patched: float) -> float:
    denom = clean - corrupted
    if abs(denom) < 1e-12:
        return math.nan
    return (patched - corrupted) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", default=os.environ.get("PATCHING_DATASET", DEFAULT_DATASET_JSON))
    parser.add_argument("--dataset-id", default=os.environ.get("DATASET", "coco-qa-vi"))
    parser.add_argument("--vlm", default=MODEL_ID)
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "3")))
    parser.add_argument("--layer", type=int, default=int(os.environ.get("LAYER", "31")))
    parser.add_argument("--severity", default=os.environ.get("SEVERITY", "extreme"))
    args = parser.parse_args()

    print(f"Loading {args.vlm}...", flush=True)
    model, processor, device, adapter = load_model(args.vlm)
    template_signature = inspect.signature(processor.apply_chat_template)
    print(f"adapter={adapter.family} device={device}", flush=True)
    print(
        f"chat_template supports enable_thinking="
        f"{'enable_thinking' in template_signature.parameters}",
        flush=True,
    )

    dataset_path = Path(args.dataset_json)
    with dataset_path.open() as f:
        data = json.load(f)
    samples = data["samples"][: args.num_samples]
    print(f"dataset={dataset_path} dataset_id={args.dataset_id} samples={len(samples)} layer={args.layer}", flush=True)
    print(f"<think> token id={QWEN_THINK_ID} text={_token_text(processor, QWEN_THINK_ID)!r}", flush=True)

    deltas = []
    think_recoveries = []
    top_recoveries = []

    for sample in samples:
        perturbation = _select_perturbation(sample, args.severity)
        prompt = _format_prompt(args.dataset_id, sample["clean_question"])
        clean_image = Image.open(sample["clean_image_path"]).convert("RGB")
        corrupted_image = Image.open(perturbation["blurred_image_path"]).convert("RGB")

        clean_inputs = prepare_inputs(adapter, processor, clean_image, prompt, device)
        corrupted_inputs = prepare_inputs(adapter, processor, corrupted_image, prompt, device)
        clean_len = int(clean_inputs["input_ids"].shape[1])
        corrupted_len = int(corrupted_inputs["input_ids"].shape[1])
        if clean_len != corrupted_len:
            raise RuntimeError(
                f"prompt length mismatch for sample {sample['sample_id']}: "
                f"clean={clean_len}, corrupted={corrupted_len}"
            )

        clean_logits, clean_activation = _forward_logits_and_capture(
            model=model,
            adapter=adapter,
            inputs=clean_inputs,
            layer_idx=args.layer,
        )
        corrupted_logits, _ = _forward_logits_and_capture(
            model=model,
            adapter=adapter,
            inputs=corrupted_inputs,
            layer_idx=args.layer,
        )
        patched_logits = _forward_logits_with_patch(
            model=model,
            adapter=adapter,
            inputs=corrupted_inputs,
            layer_idx=args.layer,
            clean_activation=clean_activation,
        )

        clean_stats = _stats(processor, clean_logits)
        corrupted_stats = _stats(processor, corrupted_logits)
        patched_stats = _stats(processor, patched_logits)

        logit_linf = float((patched_logits.float() - clean_logits.float()).abs().max().item())
        logit_mean_abs = float((patched_logits.float() - clean_logits.float()).abs().mean().item())
        think_abs_delta = abs(patched_stats.think_prob - clean_stats.think_prob)
        top_abs_delta = abs(patched_stats.top_prob - clean_stats.top_prob)
        deltas.append((logit_linf, logit_mean_abs, think_abs_delta, top_abs_delta))
        think_recoveries.append(
            _relative_recovery(clean_stats.think_prob, corrupted_stats.think_prob, patched_stats.think_prob)
        )
        top_recoveries.append(
            _relative_recovery(clean_stats.top_prob, corrupted_stats.top_prob, patched_stats.top_prob)
        )

        print(
            f"\nSample {sample['sample_id']} question={sample['clean_question']!r} "
            f"severity={perturbation.get('severity')} prompt_len={clean_len}",
            flush=True,
        )
        _print_stats("clean", clean_stats)
        _print_stats("corrupt", corrupted_stats)
        _print_stats("patched", patched_stats)
        print(
            f"    patched-clean: logit_linf={logit_linf:.6e} "
            f"logit_mean_abs={logit_mean_abs:.6e} "
            f"|delta p(<think>)|={think_abs_delta:.6e} "
            f"|delta top_p|={top_abs_delta:.6e}",
            flush=True,
        )
        print(
            f"    recovery: p(<think>)={think_recoveries[-1]:.4f} "
            f"top_p={top_recoveries[-1]:.4f}",
            flush=True,
        )

    finite_think = [x for x in think_recoveries if math.isfinite(x)]
    finite_top = [x for x in top_recoveries if math.isfinite(x)]
    print("\nSummary", flush=True)
    print(f"  mean logit_linf={sum(d[0] for d in deltas) / len(deltas):.6e}", flush=True)
    print(f"  mean logit_mean_abs={sum(d[1] for d in deltas) / len(deltas):.6e}", flush=True)
    print(f"  mean |delta p(<think>)|={sum(d[2] for d in deltas) / len(deltas):.6e}", flush=True)
    print(f"  mean |delta top_p|={sum(d[3] for d in deltas) / len(deltas):.6e}", flush=True)
    if finite_think:
        print(f"  mean recovery p(<think>)={sum(finite_think) / len(finite_think):.4f}", flush=True)
    if finite_top:
        print(f"  mean recovery top_p={sum(finite_top) / len(finite_top):.4f}", flush=True)


if __name__ == "__main__":
    main()
