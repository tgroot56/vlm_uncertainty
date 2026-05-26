#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch

from src.data import load_dataset_prepared
from src.uq_dataset_generation.registry import get_task
from src.vlm import (
    extract_vision_features_bundle,
    infer_image_token_ids,
    load_model,
)
from utils.image_resolver import resolve_image


def _move_batch_to_device(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    if hasattr(batch, "to"):
        batch = batch.to(device)
    return dict(batch)


def prepare_llava_question_before_image_inputs(processor, image, prompt: str, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    return _move_batch_to_device(inputs, device), text_prompt


def _find_last_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle:
        return None
    for idx in range(len(haystack) - len(needle), -1, -1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return None


def _find_first_contiguous_visual_span(
    input_ids: List[int],
    image_token_ids: Tuple[int, ...],
    tokenizer,
) -> Tuple[int, int]:
    if image_token_ids:
        positions = [idx for idx, token_id in enumerate(input_ids) if token_id in set(image_token_ids)]
    else:
        token_strings = tokenizer.convert_ids_to_tokens(input_ids)
        positions = [
            idx
            for idx, token_str in enumerate(token_strings)
            if isinstance(token_str, str)
            and any(marker in token_str.lower() for marker in ("image", "vision", "img"))
        ]

    if not positions:
        raise RuntimeError("Could not find image tokens in swapped-order input_ids.")

    best_start = positions[0]
    best_end = positions[0] + 1
    cur_start = positions[0]
    cur_prev = positions[0]

    for pos in positions[1:]:
        if pos == cur_prev + 1:
            cur_prev = pos
            continue
        if cur_prev + 1 - cur_start > best_end - best_start:
            best_start = cur_start
            best_end = cur_prev + 1
        cur_start = pos
        cur_prev = pos

    if cur_prev + 1 - cur_start > best_end - best_start:
        best_start = cur_start
        best_end = cur_prev + 1

    return best_start, best_end


def _find_assistant_start(input_ids: List[int], prompt_len: int, tokenizer, markers: Tuple[str, ...]) -> int:
    for marker in markers:
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        start = _find_last_subsequence(input_ids[:prompt_len], marker_ids)
        if start is not None:
            return start
    return prompt_len


def _find_question_span(
    input_ids: List[int],
    prompt: str,
    visual_start: int,
    tokenizer,
) -> Tuple[int, int]:
    candidate_texts = [
        prompt,
        " " + prompt,
        "\n" + prompt,
        "\n " + prompt,
    ]
    for candidate in candidate_texts:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        start = _find_last_subsequence(input_ids[:visual_start], candidate_ids)
        if start is not None:
            return start, start + len(candidate_ids)

    return 0, visual_start


def _print_token_block(
    title: str,
    input_ids: List[int],
    tokenizer,
    start: int,
    end: int,
) -> None:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[start:end])
    decoded = tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
    print(f"[{title}] span=[{start}, {end}) length={end - start}")
    print("DECODED SPAN:")
    print(decoded)
    print("TOKENS:")
    for pos, (token_id, token) in enumerate(zip(input_ids[start:end], tokens), start=start):
        print(f"{pos:4d} | id={int(token_id):6d} | token={repr(token)}")


def _run_one_sample(
    *,
    sample_idx: int,
    sample: Dict,
    model,
    processor,
    device: torch.device,
    vlm_adapter,
    max_new_tokens: int,
) -> bool:
    task = get_task("pope")
    prompt = task.predict_kwargs(sample, vlm_adapter=vlm_adapter)["prompt"]
    image = resolve_image(sample)

    print(f"================= SAMPLE {sample_idx + 1} =================", flush=True)
    print(f"SAMPLE INDEX: {sample.get('idx', sample_idx)}")
    print(f"SAMPLE ID: {sample.get('sample_id')}")
    print("QUESTION:")
    print(sample.get("question") or sample.get("statement") or "")
    print("GROUND TRUTH:")
    print(sample.get("gt_answer"))
    print("PROMPT:")
    print(prompt)

    inputs, text_prompt = prepare_llava_question_before_image_inputs(processor, image, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()

    print("SWAPPED CHAT TEMPLATE TEXT:")
    print(text_prompt)
    print("INPUT IDS SHAPE:")
    print(tuple(inputs["input_ids"].shape))

    image_token_ids = infer_image_token_ids(vlm_adapter, model, processor)
    visual_start, visual_end = _find_first_contiguous_visual_span(
        input_ids,
        image_token_ids,
        processor.tokenizer,
    )
    assistant_start = _find_assistant_start(
        input_ids,
        prompt_len,
        processor.tokenizer,
        vlm_adapter.assistant_markers,
    )
    question_start, question_end = _find_question_span(
        input_ids,
        prompt,
        visual_start,
        processor.tokenizer,
    )

    print("TOKEN ORDER:")
    _print_token_block("QUESTION TOKENS", input_ids, processor.tokenizer, question_start, question_end)
    _print_token_block("IMAGE TOKENS", input_ids, processor.tokenizer, visual_start, visual_end)
    _print_token_block("ASSISTANT/PREFILL TOKENS", input_ids, processor.tokenizer, visual_end, prompt_len)

    with torch.inference_mode():
        _ = extract_vision_features_bundle(vlm_adapter, model, inputs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Swapped-order prompt forward did not return hidden states.")

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = out.sequences[:, prompt_len:]
    generated_list = generated_ids[0].detach().cpu().tolist()
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("[GENERATED ANSWER TOKENS]")
    print(f"span=[{prompt_len}, {prompt_len + len(generated_list)}) length={len(generated_list)}")
    _print_token_block(
        "GENERATED ANSWER TOKENS",
        out.sequences[0].detach().cpu().tolist(),
        processor.tokenizer,
        prompt_len,
        prompt_len + len(generated_list),
    )
    print("GENERATED ANSWER:")
    print(generated_answer)
    print("PROMPT HIDDEN STATE SHAPE:")
    print(tuple(hidden_states[-1].shape))
    print("============================================", flush=True)
    return bool(generated_answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA POPE swapped token-order feasibility test.")
    parser.add_argument("--vlm", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset_split", default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, processor, device, vlm_adapter = load_model(args.vlm, args.device)
    if vlm_adapter.family != "llava":
        raise ValueError(f"This diagnostic is LLaVA-only; got adapter family {vlm_adapter.family!r}.")

    dataset = load_dataset_prepared(
        dataset_id="pope",
        split=args.dataset_split,
        seed_offset=args.seed,
        use_cache=True,
        max_samples=args.max_samples,
        dataset_kwargs={},
    )

    print("LLaVA swapped token-order feasibility test")
    print(f"model: {args.vlm}")
    print(f"dataset: pope/{args.dataset_split}")
    print(f"samples: {len(dataset)}")
    print("diagnostic order: question tokens -> image tokens -> generated answer tokens")
    print("generation: do_sample=False, max_new_tokens=15")

    successes = 0
    for sample_idx, sample in enumerate(dataset):
        try:
            if _run_one_sample(
                sample_idx=sample_idx,
                sample=sample,
                model=model,
                processor=processor,
                device=device,
                vlm_adapter=vlm_adapter,
                max_new_tokens=args.max_new_tokens,
            ):
                successes += 1
        except Exception as exc:
            print(f"FAILED SAMPLE {sample_idx + 1}: {type(exc).__name__}: {exc}", flush=True)
            raise

    print("SUMMARY")
    print(f"technical_forward_and_generate_successes: {successes}/{len(dataset)}")


if __name__ == "__main__":
    main()
