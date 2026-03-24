from typing import Dict, List, Optional, Tuple

import torch

from src.vlm import VLMAdapter, infer_image_token_ids


def _find_last_subsequence(haystack: torch.Tensor, needle: List[int]) -> Optional[int]:
    if len(needle) == 0:
        return None
    h = haystack.tolist()
    n = needle
    L, M = len(h), len(n)
    for i in range(L - M, -1, -1):
        if h[i:i + M] == n:
            return i
    return None


def _find_contiguous_visual_span(
    input_ids: torch.Tensor,
    image_token_ids: Tuple[int, ...],
    processor,
) -> Optional[Tuple[int, int]]:
    if image_token_ids:
        image_token_set = set(image_token_ids)
        positions = [i for i, token_id in enumerate(input_ids.tolist()) if token_id in image_token_set]
    else:
        token_strings = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        positions = [
            i
            for i, token_str in enumerate(token_strings)
            if isinstance(token_str, str) and any(marker in token_str.lower() for marker in ("image", "vision", "img"))
        ]

    if not positions:
        return None

    best_start = positions[0]
    best_end = positions[0] + 1
    current_start = positions[0]
    current_prev = positions[0]

    for pos in positions[1:]:
        if pos == current_prev + 1:
            current_prev = pos
            continue

        if (current_prev + 1 - current_start) > (best_end - best_start):
            best_start = current_start
            best_end = current_prev + 1

        current_start = pos
        current_prev = pos

    if (current_prev + 1 - current_start) > (best_end - best_start):
        best_start = current_start
        best_end = current_prev + 1

    return best_start, best_end


def build_token_spans(
    model,
    processor,
    vlm_adapter: VLMAdapter,
    input_ids: torch.Tensor,
    prompt_len: int,
    lm_hidden_len: int,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Build prompt spans in hidden-space indices for a multimodal chat model.
    """
    image_token_ids = infer_image_token_ids(vlm_adapter, model, processor)
    visual_span = _find_contiguous_visual_span(input_ids, image_token_ids, processor)
    if visual_span is None:
        raise RuntimeError("Could not find an image-token span in input_ids.")

    visual_start, visual_end = visual_span
    num_visual_tokens = visual_end - visual_start

    assistant_ids: List[int] = []
    assistant_start = None
    for marker in vlm_adapter.assistant_markers:
        marker_ids = processor.tokenizer.encode(marker, add_special_tokens=False)
        marker_start = _find_last_subsequence(input_ids[:prompt_len], marker_ids)
        if marker_start is not None:
            assistant_ids = marker_ids
            assistant_start = marker_start
            break

    if assistant_start is None:
        assistant_start = prompt_len
        assistant_end = prompt_len
    else:
        assistant_end = min(prompt_len, assistant_start + len(assistant_ids))

    text_pre_start, text_pre_end = 0, visual_start
    text_post_start, text_post_end = visual_end, prompt_len

    question_start = text_post_start
    question_end = max(question_start, assistant_start)

    question_including_spans = [
        (text_pre_start, text_pre_end),
        (text_post_start, text_post_end),
    ]

    fullspan_start, fullspan_end = 0, prompt_len

    if lm_hidden_len != prompt_len:
        pass

    if verbose:
        print("\n" + "=" * 80)
        print("TOKEN SPAN DEBUG")

        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        def _print_span(name, start, end):
            print(f"\n{name}: [{start}, {end})")
            for i in range(start, end):
                print(f"{i:4d} | id={int(input_ids[i])} | token={repr(tokens[i])}")

        _print_span("VISUAL", visual_start, visual_end)
        _print_span("QUESTION", question_start, question_end)
        _print_span("FULLSPAN", fullspan_start, fullspan_end)

        print("\nQUESTION INCLUDING (multi-span):")
        for a, b in question_including_spans:
            _print_span("  segment", a, b)

        _print_span("ASSISTANT MARKER", assistant_start, assistant_end)

        print("=" * 80 + "\n")

    return {
        "visual_start": visual_start,
        "visual_end": visual_end,
        "num_visual_tokens": num_visual_tokens,
        "text_pre_start": text_pre_start,
        "text_pre_end": text_pre_end,
        "text_post_start": text_post_start,
        "text_post_end": text_post_end,
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
        "question_start": question_start,
        "question_end": question_end,
        "question_including_spans": question_including_spans,
        "fullspan_start": fullspan_start,
        "fullspan_end": fullspan_end,
        "image_token_ids": list(image_token_ids),
        "image_token_position": int(visual_start),
        "input_seq_len": int(input_ids.numel()),
        "prompt_len": int(prompt_len),
        "hidden_seq_len": int(lm_hidden_len),
    }
