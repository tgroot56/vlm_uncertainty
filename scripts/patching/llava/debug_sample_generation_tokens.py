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

from src.patching.collect_activations import _MAX_NEW_TOKENS, _format_prompt
from src.patching.run_layer_sweep_patching import _extract_all_layer_activations
from src.patching.run_patching import _get_decoder_layers
from src.vlm import load_model, prepare_inputs


def _load_record(path: Path, sample_id: str) -> Dict[str, Any]:
    with path.open() as f:
        samples = json.load(f)["samples"]
    for record in samples:
        if str(record["sample_id"]) == str(sample_id):
            return record
    raise RuntimeError(f"Could not find sample_id={sample_id}")


def _severity_image(record: Dict[str, Any], severity: str) -> str:
    for pert in record.get("visual_perturbations", []):
        if pert.get("severity") == severity:
            return pert["blurred_image_path"]
    raise RuntimeError(f"Could not find severity={severity}")


def _decode_piece(processor: Any, token_id: int) -> str:
    return processor.tokenizer.decode([token_id]).replace("\n", "\\n")


def _summarize_generation(
    *,
    label: str,
    out: Any,
    prompt_len: int,
    processor: Any,
    max_steps: int = 4,
) -> None:
    gen_ids = out.sequences[:, prompt_len:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    print(f"\n=== {label} ===")
    print(f"sequences_shape={tuple(out.sequences.shape)} prompt_len={prompt_len}")
    print(f"raw_text={raw_text!r}")
    print(f"gen_ids={gen_ids[0].tolist()}")
    print("generated tokens:")
    for idx, token_id in enumerate(gen_ids[0].tolist()[:max_steps]):
        print(f"  gen[{idx}] id={token_id} text={_decode_piece(processor, token_id)!r}")

    print("score-step top tokens:")
    for step, logits in enumerate(out.scores[:max_steps]):
        probs = F.softmax(logits[0].float(), dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        token_id = int(top_idx.item())
        print(
            f"  score[{step}] top_id={token_id} "
            f"top_text={_decode_piece(processor, token_id)!r} "
            f"top_prob={float(top_prob.item()):.9f}"
        )


def _generate(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
) -> Any:
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )


def _patched_generate(
    *,
    model: Any,
    adapter: Any,
    inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    pos: int,
    clean_act: torch.Tensor,
    max_new_tokens: int,
) -> Any:
    prompt_len = int(inputs["input_ids"].shape[1])
    decoder_layers = _get_decoder_layers(model, adapter)
    target_layer = decoder_layers[layer_idx]
    replacement = clean_act.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    patched = False

    def hook(_module, _args, output):
        nonlocal patched
        if patched:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] < prompt_len:
            return output
        hidden = hidden.clone()
        hidden[0, pos, :] = replacement
        patched = True
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = target_layer.register_forward_hook(hook)
    try:
        return _generate(model, inputs, max_new_tokens)
    finally:
        handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="493612002")
    parser.add_argument("--dataset_id", default="vqa-v2")
    parser.add_argument("--severity", default="extreme")
    parser.add_argument("--vlm", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument(
        "--patching_dataset",
        default="/projects/prjs2014/patching/llava-hf_llava-1-5-7b-hf/vqa-v2/patching_dataset_vqa-v2.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, processor, device, adapter = load_model(args.vlm, args.device)
    record = _load_record(Path(args.patching_dataset), args.sample_id)
    prompt = _format_prompt(args.dataset_id, record["clean_question"])
    max_new_tokens = _MAX_NEW_TOKENS.get(args.dataset_id, 4)

    clean_image = Image.open(record["clean_image_path"]).convert("RGB")
    degraded_image = Image.open(_severity_image(record, args.severity)).convert("RGB")

    clean_inputs = prepare_inputs(adapter, processor, clean_image, prompt, device)
    degraded_inputs = prepare_inputs(adapter, processor, degraded_image, prompt, device)
    prompt_len = int(clean_inputs["input_ids"].shape[1])

    all_layer_h, _ = _extract_all_layer_activations(
        model=model,
        processor=processor,
        device=device,
        vlm_adapter=adapter,
        image=clean_image,
        prompt=prompt,
    )
    final_layer = all_layer_h.shape[0] - 1
    final_pos = prompt_len - 1
    clean_act = all_layer_h[final_layer, final_pos, :]

    clean_out = _generate(model, clean_inputs, max_new_tokens)
    degraded_out = _generate(model, degraded_inputs, max_new_tokens)
    patched_out = _patched_generate(
        model=model,
        adapter=adapter,
        inputs=degraded_inputs,
        layer_idx=final_layer,
        pos=final_pos,
        clean_act=clean_act,
        max_new_tokens=max_new_tokens,
    )

    print(f"sample_id={args.sample_id}")
    print(f"question={record['clean_question']!r}")
    print(f"gt_answers={record.get('gt_answers', [])}")
    print(f"prompt_len={prompt_len} final_layer={final_layer} final_pos={final_pos}")
    _summarize_generation(label="clean", out=clean_out, prompt_len=prompt_len, processor=processor)
    _summarize_generation(label="degraded", out=degraded_out, prompt_len=prompt_len, processor=processor)
    _summarize_generation(label="patched_final_layer", out=patched_out, prompt_len=prompt_len, processor=processor)


if __name__ == "__main__":
    main()
