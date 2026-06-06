#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.patching.collect_activations import _format_prompt
from src.labeling.vqa import score_vqa
from src.patching.run_patching import _get_decoder_layers
from src.patching.run_patching import _answer_token_indices
from src.vlm import prepare_inputs
from src.vlm.adapters import load_model


def _tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _resolve_final_norm(model: Any) -> torch.nn.Module:
    candidates = [
        ("model", "language_model", "norm"),
        ("model", "language_model", "model", "norm"),
        ("language_model", "model", "norm"),
        ("language_model", "norm"),
        ("model", "norm"),
    ]
    for path in candidates:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, torch.nn.Module):
            print(f"Resolved final norm: {'.'.join(path)}")
            return node
    raise RuntimeError("Could not resolve final LM norm module.")


@torch.no_grad()
def _forward_next_logits(model: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**inputs, return_dict=True)
    return outputs.logits[0, -1, :].detach().float().cpu()


@torch.no_grad()
def _capture_module_last_token(
    model: Any,
    module: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}
    prompt_len = int(inputs["input_ids"].shape[1])

    def hook(_module, _args, output):
        hidden = _tensor_from_output(output)
        if hidden.ndim == 3 and hidden.shape[1] >= prompt_len:
            captured["activation"] = hidden[0, prompt_len - 1, :].detach().float().cpu()

    handle = module.register_forward_hook(hook)
    try:
        model(**inputs, return_dict=True)
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError(f"Did not capture output at final prompt token from {module}.")
    return captured["activation"]


@torch.no_grad()
def _patched_module_next_logits(
    model: Any,
    module: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    replacement: torch.Tensor,
) -> torch.Tensor:
    prompt_len = int(inputs["input_ids"].shape[1])
    patched = False

    def hook(_module, _args, output):
        nonlocal patched
        if patched:
            return output
        hidden = _tensor_from_output(output)
        if hidden.ndim != 3 or hidden.shape[1] < prompt_len:
            return output

        repl = replacement.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[0, prompt_len - 1, :] = repl
        patched = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = module.register_forward_hook(hook)
    try:
        return _forward_next_logits(model, inputs)
    finally:
        handle.remove()


@torch.no_grad()
def _generate_with_optional_final_norm_patch(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    *,
    patch_module: torch.nn.Module | None = None,
    replacement: torch.Tensor | None = None,
    max_new_tokens: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_len = int(inputs["input_ids"].shape[1])
    handle = None
    patched = False

    if patch_module is not None and replacement is not None:
        def hook(_module, _args, output):
            nonlocal patched
            if patched:
                return output
            hidden = _tensor_from_output(output)
            if hidden.ndim != 3 or hidden.shape[1] < prompt_len:
                return output

            repl = replacement.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden.clone()
            hidden[0, prompt_len - 1, :] = repl
            patched = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = patch_module.register_forward_hook(hook)

    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    finally:
        if handle is not None:
            handle.remove()

    gen_ids = out.sequences[:, prompt_len:]
    step_logits = (
        torch.stack(out.scores, dim=1).detach().float().cpu()
        if len(out.scores) > 0
        else torch.empty((1, 0, model.config.vocab_size), dtype=torch.float32)
    )
    return gen_ids.cpu(), step_logits


def _topk(logits: torch.Tensor, processor: Any, k: int = 5) -> List[Tuple[int, str, float]]:
    probs = F.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)
    rows = []
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        text = processor.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        rows.append((int(token_id), text, float(prob)))
    return rows


def _topk_stripped_answer_step(
    gen_ids: torch.Tensor,
    step_logits: torch.Tensor,
    processor: Any,
    *,
    label: str,
) -> None:
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    answer_indices = [
        idx for idx in _answer_token_indices(gen_ids, processor)
        if idx < step_logits.shape[1]
    ]
    print(f"\n{label} generated_text={decoded!r}")
    print(f"{label} gen_ids={gen_ids[0].tolist()}")
    print(f"{label} stripped_answer_indices={answer_indices}")
    if not answer_indices:
        print(f"{label}: no stripped answer-token step found")
        return
    step_idx = answer_indices[0]
    print(f"{label} stripped-answer-step top-5 (step_idx={step_idx}):")
    for token_id, text, prob in _topk(step_logits[0, step_idx, :], processor):
        print(f"  id={token_id:<6d} p={prob:.6f} token={text!r}")


def _decoded_and_stripped(
    gen_ids: torch.Tensor,
    processor: Any,
) -> Tuple[str, str]:
    full = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return full, full.strip()


def _print_answer_score(
    label: str,
    gen_ids: torch.Tensor,
    processor: Any,
    record: Dict[str, Any],
) -> None:
    full, stripped = _decoded_and_stripped(gen_ids, processor)
    scored = score_vqa(
        pred_answer=stripped,
        gt_answers=record.get("gt_answers"),
        gt_answer_single=record.get("gt_answer"),
    )
    print(f"\n{label} full_generated_answer={full!r}")
    print(f"{label} stripped/evaluated_answer={stripped!r}")
    print(f"{label} VQA_score={scored.score:.3f}")


def _load_record(path: Path, sample_id: str) -> Dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    for record in data["samples"]:
        if str(record["sample_id"]) == str(sample_id):
            return record
    raise KeyError(f"sample_id {sample_id!r} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="243694011")
    parser.add_argument("--dataset_id", default="vqa-v2")
    parser.add_argument("--severity", default="extreme")
    parser.add_argument("--vlm", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--patching_dataset",
        default=(
            "/projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/"
            "vqa-v2/patching_dataset_vqa-v2.json"
        ),
    )
    args = parser.parse_args()

    record = _load_record(Path(args.patching_dataset), args.sample_id)
    pert = next(
        (
            p for p in record.get("visual_perturbations", [])
            if p.get("severity") == args.severity
        ),
        None,
    )
    if pert is None:
        raise RuntimeError(f"No perturbation with severity={args.severity!r}.")

    model, processor, device, adapter = load_model(args.vlm, args.device)
    final_norm = _resolve_final_norm(model)
    final_decoder = _get_decoder_layers(model, adapter)[-1]
    prompt = _format_prompt(args.dataset_id, record["clean_question"])

    clean_image = Image.open(record["clean_image_path"]).convert("RGB")
    degraded_image = Image.open(pert["blurred_image_path"]).convert("RGB")
    try:
        clean_inputs = prepare_inputs(adapter, processor, clean_image, prompt, device)
        degraded_inputs = prepare_inputs(adapter, processor, degraded_image, prompt, device)

        clean_block_last = _capture_module_last_token(model, final_decoder, clean_inputs)
        clean_norm_last = _capture_module_last_token(model, final_norm, clean_inputs)
        clean_logits = _forward_next_logits(model, clean_inputs)
        degraded_logits = _forward_next_logits(model, degraded_inputs)
        patched_block_logits = _patched_module_next_logits(
            model,
            final_decoder,
            degraded_inputs,
            clean_block_last,
        )
        patched_norm_logits = _patched_module_next_logits(
            model,
            final_norm,
            degraded_inputs,
            clean_norm_last,
        )
        clean_gen_ids, clean_step_logits = _generate_with_optional_final_norm_patch(
            model,
            clean_inputs,
            max_new_tokens=4,
        )
        degraded_gen_ids, degraded_step_logits = _generate_with_optional_final_norm_patch(
            model,
            degraded_inputs,
            max_new_tokens=4,
        )
        patched_block_gen_ids, patched_block_step_logits = _generate_with_optional_final_norm_patch(
            model,
            degraded_inputs,
            patch_module=final_decoder,
            replacement=clean_block_last,
            max_new_tokens=4,
        )
        patched_norm_gen_ids, patched_norm_step_logits = _generate_with_optional_final_norm_patch(
            model,
            degraded_inputs,
            patch_module=final_norm,
            replacement=clean_norm_last,
            max_new_tokens=4,
        )
    finally:
        clean_image.close()
        degraded_image.close()

    print(f"sample_id={args.sample_id}")
    print(f"question={record.get('clean_question')!r}")
    print(f"gt_answer={record.get('gt_answer')!r}")
    print(f"gt_answers={record.get('gt_answers')!r}")
    print(f"severity={args.severity}")
    print(f"prompt_len={int(clean_inputs['input_ids'].shape[1])}")
    print("\nclean top-5:")
    for token_id, text, prob in _topk(clean_logits, processor):
        print(f"  id={token_id:<6d} p={prob:.6f} token={text!r}")

    print("\ndegraded top-5:")
    for token_id, text, prob in _topk(degraded_logits, processor):
        print(f"  id={token_id:<6d} p={prob:.6f} token={text!r}")

    print("\npatched final-decoder-output top-5:")
    for token_id, text, prob in _topk(patched_block_logits, processor):
        print(f"  id={token_id:<6d} p={prob:.6f} token={text!r}")

    print("\npatched final-norm-output top-5:")
    for token_id, text, prob in _topk(patched_norm_logits, processor):
        print(f"  id={token_id:<6d} p={prob:.6f} token={text!r}")

    _print_answer_score("clean", clean_gen_ids, processor, record)
    _print_answer_score("degraded", degraded_gen_ids, processor, record)
    _print_answer_score("patched_final_decoder_output", patched_block_gen_ids, processor, record)
    _print_answer_score("patched_final_norm_output", patched_norm_gen_ids, processor, record)

    _topk_stripped_answer_step(
        clean_gen_ids,
        clean_step_logits,
        processor,
        label="clean",
    )
    _topk_stripped_answer_step(
        degraded_gen_ids,
        degraded_step_logits,
        processor,
        label="degraded",
    )
    _topk_stripped_answer_step(
        patched_block_gen_ids,
        patched_block_step_logits,
        processor,
        label="patched_final_decoder_output",
    )
    _topk_stripped_answer_step(
        patched_norm_gen_ids,
        patched_norm_step_logits,
        processor,
        label="patched_final_norm_output",
    )


if __name__ == "__main__":
    main()
