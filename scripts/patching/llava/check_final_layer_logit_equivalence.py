from __future__ import annotations

import argparse
import csv
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
from src.patching.run_layer_sweep_patching import _get_layer_index_pairs
from src.patching.run_patching import _get_decoder_layers
from src.vlm.adapters import load_model
from src.vlm import prepare_inputs


def _tensor_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _capture_module_last_token(
    module: torch.nn.Module,
    fn,
) -> Tuple[torch.Tensor, Any]:
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _args, output):
        hidden = _tensor_from_layer_output(output)
        if hidden.ndim == 3 and hidden.shape[1] > 1:
            captured["last"] = hidden[0, -1, :].detach().float().cpu()

    handle = module.register_forward_hook(hook)
    try:
        result = fn()
    finally:
        handle.remove()

    if "last" not in captured:
        raise RuntimeError(f"Did not capture a full-sequence output from {module.__class__.__name__}.")
    return captured["last"], result


def _patch_module_last_token(
    module: torch.nn.Module,
    replacement: torch.Tensor,
    prompt_len: int,
    fn,
) -> Any:
    replacement_by_device: Dict[torch.device, torch.Tensor] = {}
    patched = False

    def hook(_module, _args, output):
        nonlocal patched
        if patched:
            return output
        hidden = _tensor_from_layer_output(output)
        if hidden.ndim != 3 or hidden.shape[1] < prompt_len:
            return output

        repl = replacement_by_device.get(hidden.device)
        if repl is None:
            repl = replacement.to(device=hidden.device, dtype=hidden.dtype)
            replacement_by_device[hidden.device] = repl

        hidden = hidden.clone()
        hidden[0, prompt_len - 1, :] = repl
        patched = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = module.register_forward_hook(hook)
    try:
        return fn()
    finally:
        handle.remove()


@torch.no_grad()
def _forward_logits(
    model: Any,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(**inputs, return_dict=True)
    return outputs.logits[0, -1, :].detach().float().cpu()


def _top_stats(logits: torch.Tensor, processor: Any) -> Dict[str, Any]:
    probs = F.softmax(logits.float(), dim=-1)
    top_prob, top_idx = probs.max(dim=-1)
    top_id = int(top_idx.item())
    return {
        "top_token_id": top_id,
        "top_token": processor.tokenizer.decode([top_id]).strip(),
        "top1_probability": float(top_prob.item()),
    }


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a.float() - b.float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "rms_diff": float(torch.sqrt(torch.mean(diff ** 2)).item()),
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def _norm_path(model: Any) -> torch.nn.Module:
    candidates = [
        ("model", "language_model", "norm"),
        ("model", "language_model", "model", "norm"),
        ("language_model", "model", "norm"),
        ("language_model", "norm"),
        ("model", "language_model", "model", "norm"),
        ("model", "norm"),
    ]
    for path in candidates:
        node = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, torch.nn.Module):
            return node
    raise RuntimeError("Could not resolve final LM norm module.")


def _load_records(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    return data["samples"][:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="vqa-v2")
    parser.add_argument("--vlm", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument(
        "--patching_dataset",
        default="/projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/vqa-v2/patching_dataset_vqa-v2.json",
    )
    parser.add_argument("--severity", default="extreme")
    parser.add_argument("--max_samples", type=int, default=25)
    parser.add_argument(
        "--output_dir",
        default="/projects/prjs2014/patching/all_datasets/extreme/final_layer_equivalence",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device, adapter = load_model(args.vlm, args.device)
    decoder_layers = _get_decoder_layers(model, adapter)
    final_decoder_idx = len(decoder_layers) - 1
    final_decoder = decoder_layers[final_decoder_idx]
    final_norm = _norm_path(model)

    records = _load_records(Path(args.patching_dataset), args.max_samples)
    rows: List[Dict[str, Any]] = []

    for sample_idx, record in enumerate(records):
        sample_id = str(record["sample_id"])
        prompt = _format_prompt(args.dataset_id, record["clean_question"])
        pert = next(
            (p for p in record.get("visual_perturbations", []) if p.get("severity") == args.severity),
            None,
        )
        if pert is None:
            continue

        clean_image = Image.open(record["clean_image_path"]).convert("RGB")
        degraded_image = Image.open(pert["blurred_image_path"]).convert("RGB")
        clean_inputs = prepare_inputs(adapter, processor, clean_image, prompt, device)
        degraded_inputs = prepare_inputs(adapter, processor, degraded_image, prompt, device)
        prompt_len = int(clean_inputs["input_ids"].shape[1])

        clean_logits = _forward_logits(model, clean_inputs)
        degraded_logits = _forward_logits(model, degraded_inputs)

        clean_block_last, _ = _capture_module_last_token(
            final_decoder,
            lambda: model(**clean_inputs, return_dict=True),
        )
        clean_norm_last, _ = _capture_module_last_token(
            final_norm,
            lambda: model(**clean_inputs, return_dict=True),
        )

        outputs_hs = model(**clean_inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs_hs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states with output_hidden_states=True.")
        layer_pairs = _get_layer_index_pairs(len(hidden_states), len(decoder_layers))
        runner_hidden_idx, runner_patch_idx = layer_pairs[-1]
        runner_clean_last = hidden_states[runner_hidden_idx][0, prompt_len - 1, :].detach().float().cpu()
        if runner_patch_idx != final_decoder_idx:
            raise RuntimeError(
                f"Unexpected final mapping: hidden_state_idx={runner_hidden_idx}, "
                f"patch_idx={runner_patch_idx}, final_decoder_idx={final_decoder_idx}."
            )

        patched_block_logits = _patch_module_last_token(
            final_decoder,
            clean_block_last,
            prompt_len,
            lambda: _forward_logits(model, degraded_inputs),
        )
        patched_runner_logits = _patch_module_last_token(
            final_decoder,
            runner_clean_last,
            prompt_len,
            lambda: _forward_logits(model, degraded_inputs),
        )
        patched_norm_logits = _patch_module_last_token(
            final_norm,
            clean_norm_last,
            prompt_len,
            lambda: _forward_logits(model, degraded_inputs),
        )

        clean_top = _top_stats(clean_logits, processor)
        degraded_top = _top_stats(degraded_logits, processor)
        block_top = _top_stats(patched_block_logits, processor)
        runner_top = _top_stats(patched_runner_logits, processor)
        norm_top = _top_stats(patched_norm_logits, processor)

        row = {
            "sample_idx": sample_idx,
            "sample_id": sample_id,
            "question": record.get("clean_question", ""),
            "prompt_len": prompt_len,
            "num_hidden_states": len(hidden_states),
            "num_decoder_layers": len(decoder_layers),
            "runner_hidden_state_idx": runner_hidden_idx,
            "runner_patch_decoder_idx": runner_patch_idx,
            "clean_top_token": clean_top["top_token"],
            "clean_top1_probability": clean_top["top1_probability"],
            "degraded_top_token": degraded_top["top_token"],
            "degraded_top1_probability": degraded_top["top1_probability"],
            "patched_true_block_top_token": block_top["top_token"],
            "patched_true_block_top1_probability": block_top["top1_probability"],
            "patched_runner_style_top_token": runner_top["top_token"],
            "patched_runner_style_top1_probability": runner_top["top1_probability"],
            "patched_norm_output_top_token": norm_top["top_token"],
            "patched_norm_output_top1_probability": norm_top["top1_probability"],
            "runner_act_cosine_with_block_output": _cosine(runner_clean_last, clean_block_last),
            "runner_act_cosine_with_norm_output": _cosine(runner_clean_last, clean_norm_last),
            "block_output_cosine_with_norm_output": _cosine(clean_block_last, clean_norm_last),
            "runner_act_l2_to_block_output": float(torch.linalg.vector_norm(runner_clean_last - clean_block_last).item()),
            "runner_act_l2_to_norm_output": float(torch.linalg.vector_norm(runner_clean_last - clean_norm_last).item()),
        }
        for prefix, logits in [
            ("true_block_vs_clean", patched_block_logits),
            ("runner_style_vs_clean", patched_runner_logits),
            ("norm_output_vs_clean", patched_norm_logits),
            ("degraded_vs_clean", degraded_logits),
        ]:
            for k, v in _diff_stats(logits, clean_logits).items():
                row[f"{prefix}_{k}"] = v
        row["true_block_top1_delta_from_clean"] = block_top["top1_probability"] - clean_top["top1_probability"]
        row["runner_style_top1_delta_from_clean"] = runner_top["top1_probability"] - clean_top["top1_probability"]
        row["norm_output_top1_delta_from_clean"] = norm_top["top1_probability"] - clean_top["top1_probability"]
        rows.append(row)

        print(
            f"[{sample_idx:03d}] {sample_id}: "
            f"true_block maxdiff={row['true_block_vs_clean_max_abs_diff']:.3e}, "
            f"runner maxdiff={row['runner_style_vs_clean_max_abs_diff']:.3e}, "
            f"norm maxdiff={row['norm_output_vs_clean_max_abs_diff']:.3e}, "
            f"runner cos(block/norm)=({row['runner_act_cosine_with_block_output']:.4f}/"
            f"{row['runner_act_cosine_with_norm_output']:.4f})",
            flush=True,
        )

        clean_image.close()
        degraded_image.close()

    csv_path = output_dir / f"{args.dataset_id}_llava_final_layer_logit_equivalence.csv"
    json_path = output_dir / f"{args.dataset_id}_llava_final_layer_logit_equivalence_summary.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def summarise(name: str) -> Dict[str, float]:
        return {
            "max_abs_diff_mean": float(sum(r[f"{name}_max_abs_diff"] for r in rows) / len(rows)),
            "max_abs_diff_max": float(max(r[f"{name}_max_abs_diff"] for r in rows)),
            "mean_abs_diff_mean": float(sum(r[f"{name}_mean_abs_diff"] for r in rows) / len(rows)),
            "rms_diff_mean": float(sum(r[f"{name}_rms_diff"] for r in rows) / len(rows)),
        }

    summary = {
        "dataset_id": args.dataset_id,
        "severity": args.severity,
        "num_samples": len(rows),
        "num_decoder_layers": len(decoder_layers),
        "final_decoder_idx": final_decoder_idx,
        "true_block_vs_clean": summarise("true_block_vs_clean"),
        "runner_style_vs_clean": summarise("runner_style_vs_clean"),
        "norm_output_vs_clean": summarise("norm_output_vs_clean"),
        "degraded_vs_clean": summarise("degraded_vs_clean"),
        "runner_act_cosine_with_block_output_mean": float(
            sum(r["runner_act_cosine_with_block_output"] for r in rows) / len(rows)
        ),
        "runner_act_cosine_with_norm_output_mean": float(
            sum(r["runner_act_cosine_with_norm_output"] for r in rows) / len(rows)
        ),
        "runner_style_over_clean_top1_count": int(
            sum(r["runner_style_top1_delta_from_clean"] > 1e-6 for r in rows)
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
