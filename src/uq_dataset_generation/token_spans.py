from typing import Dict, List, Tuple, Optional
import torch

def _find_subsequence(haystack: torch.Tensor, needle: List[int]) -> Optional[int]:
    """Return start idx of first occurrence of needle in haystack, else None."""
    if len(needle) == 0:
        return None
    h = haystack.tolist()
    n = needle
    L, M = len(h), len(n)
    for i in range(0, L - M + 1):
        if h[i:i+M] == n:
            return i
    return None


def build_token_spans(
    model,
    processor,
    input_ids: torch.Tensor,   # (seq_len,)
    prompt_len: int,           # length of prompt in token-id space (inputs["input_ids"].shape[1])
    lm_hidden_len: int,        # lm_hidden_states[0].shape[1]
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Build spans in *hidden space* indices.
    Assumes prompt tokens align with hidden states (true for LLaVA-style where <image> already expanded in input_ids).
    """
    image_token_id = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_id is None:
        raise RuntimeError("model.config.image_token_index missing; can't find image token id.")

    # --- visual block ---
    positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise RuntimeError("Could not find any image tokens in input_ids.")

    visual_start = int(positions[0].item())
    visual_end = int(positions[-1].item()) + 1
    num_visual_tokens = visual_end - visual_start

    # --- assistant marker inside the prompt ---
    # In your tokenizer this often becomes tokens like: '▁A','SS','IST','ANT',':'
    assistant_ids = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    assistant_start = _find_subsequence(input_ids[:prompt_len], assistant_ids)
    if assistant_start is None:
        # Fallback: try "Assistant:" (some templates differ)
        assistant_ids2 = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
        assistant_start = _find_subsequence(input_ids[:prompt_len], assistant_ids2)
        assistant_ids = assistant_ids2 if assistant_start is not None else assistant_ids

    if assistant_start is None:
        # If we cannot find it, treat "question" as everything after the image to end of prompt
        assistant_start = prompt_len
        assistant_end = prompt_len
    else:
        assistant_end = min(prompt_len, assistant_start + len(assistant_ids))

    # --- prompt text spans ---
    text_pre_start, text_pre_end = 0, visual_start
    text_post_start, text_post_end = visual_end, prompt_len

    # "question" in your new naming: post-image text up to assistant marker
    question_start = text_post_start
    question_end = max(question_start, assistant_start)  # guard

    # "question_including": all text excluding image => two segments
    question_including_spans = [
        (text_pre_start, text_pre_end),
        (text_post_start, text_post_end),
    ]

    # "fullspan": entire prompt including visual
    fullspan_start, fullspan_end = 0, prompt_len

    # Consistency checks: prompt_len should match lm_hidden_len in your current setup
    # (If it doesn't, you can still use prompt_len for token-id slicing and lm_hidden_len for hidden slicing,
    # but then you'd need a mapping. For LLaVA-1.5 HF style, they usually match when <image> is expanded.)
    if lm_hidden_len != prompt_len:
        # Don’t crash; store so you can debug.
        pass
    
    if verbose:
        print("\n" + "=" * 80)
        print("TOKEN SPAN DEBUG")

        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        def _print_span(name, start, end):
            print(f"\n{name}: [{start}, {end})")
            for i in range(start, end):
                print(f"{i:4d} | id={int(input_ids[i])} | token={repr(tokens[i])}")

        # Visual
        _print_span("VISUAL", visual_start, visual_end)

        # Question
        _print_span("QUESTION", question_start, question_end)

        # Full span
        _print_span("FULLSPAN", fullspan_start, fullspan_end)

        # Question including (multi-span)
        print("\nQUESTION INCLUDING (multi-span):")
        for a, b in question_including_spans:
            _print_span("  segment", a, b)

        # Assistant marker
        _print_span("ASSISTANT MARKER", assistant_start, assistant_end)

        print("=" * 80 + "\n")

    return {
        # visual
        "visual_start": visual_start,
        "visual_end": visual_end,
        "num_visual_tokens": num_visual_tokens,

        # text pre/post
        "text_pre_start": text_pre_start,
        "text_pre_end": text_pre_end,
        "text_post_start": text_post_start,
        "text_post_end": text_post_end,

        # assistant marker
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,

        # question (renamed from prompt)
        "question_start": question_start,
        "question_end": question_end,

        # question including (multi-span)
        "question_including_spans": question_including_spans,

        # full span
        "fullspan_start": fullspan_start,
        "fullspan_end": fullspan_end,

        # bookkeeping
        "image_token_id": int(image_token_id),
        "image_token_position": int(visual_start),
        "input_seq_len": int(input_ids.numel()),
        "prompt_len": int(prompt_len),
        "hidden_seq_len": int(lm_hidden_len),
    }