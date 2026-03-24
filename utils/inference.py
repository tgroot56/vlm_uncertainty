"""Inference utilities for model predictions."""

import torch
import re
import time
from typing import Dict, List, Tuple, Optional

from src.uq_dataset_generation.token_spans import build_token_spans
from src.vlm import (
    VLMAdapter,
    build_full_forward_kwargs,
    extract_vision_features_bundle,
    get_vlm_adapter,
    prepare_inputs,
)


def _sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(device: torch.device, fn, *args, **kwargs):
    _sync_device_for_timing(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync_device_for_timing(device)
    return result, time.perf_counter() - start


def _print_verbose_qwen_timings(title: str, timings: Dict[str, float]) -> None:
    total = sum(timings.values())
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for key, value in timings.items():
        print(f"{key:28s}: {value:7.3f}s")
    print(f"{'total':28s}: {total:7.3f}s")
    print("=" * 80 + "\n")


def _extract_prompt_hidden_states_from_generate_output(
    generation_hidden_states,
    prompt_len: int,
) -> Optional[Tuple[torch.Tensor, ...]]:
    if not generation_hidden_states:
        return None

    first_step_hidden_states = generation_hidden_states[0]
    if first_step_hidden_states is None:
        return None

    prompt_hidden_states = []
    for layer_hidden in first_step_hidden_states:
        if layer_hidden is None or layer_hidden.ndim != 3 or layer_hidden.shape[1] < prompt_len:
            return None
        prompt_hidden_states.append(layer_hidden[:, :prompt_len, :])

    return tuple(prompt_hidden_states)


def _build_single_token_answer_hidden_states_from_generate_output(
    prompt_hidden_states: Tuple[torch.Tensor, ...],
    generation_hidden_states,
) -> Optional[Tuple[torch.Tensor, ...]]:
    if not generation_hidden_states or len(generation_hidden_states) < 2:
        return None

    first_generated_token_hidden_states = generation_hidden_states[1]
    if first_generated_token_hidden_states is None:
        return None

    answer_hidden_states = []
    for prompt_layer, generated_layer in zip(prompt_hidden_states, first_generated_token_hidden_states):
        if generated_layer is None or generated_layer.ndim != 3 or generated_layer.shape[1] < 1:
            return None
        answer_hidden_states.append(torch.cat([prompt_layer, generated_layer[:, -1:, :]], dim=1))

    return tuple(answer_hidden_states)


def _build_answer_hidden_states_from_generate_output(
    prompt_hidden_states: Tuple[torch.Tensor, ...],
    generation_hidden_states,
    effective_gen_len: int,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Reconstruct hidden states for prompt + generated answer directly from
    `generate(..., output_hidden_states=True)` output.

    For decoder-only generation in transformers, the first hidden-state entry
    contains the prompt states and subsequent entries contain one newly decoded
    token each. This lets us avoid a separate forward pass over the answer
    tokens for Qwen when generation hidden states are available.
    """
    if effective_gen_len <= 0:
        return None

    if effective_gen_len == 1:
        return _build_single_token_answer_hidden_states_from_generate_output(
            prompt_hidden_states,
            generation_hidden_states,
        )

    if not generation_hidden_states or len(generation_hidden_states) < (effective_gen_len + 1):
        return None

    answer_hidden_states = []
    for layer_idx, prompt_layer in enumerate(prompt_hidden_states):
        generated_tokens = []
        for step_idx in range(1, effective_gen_len + 1):
            step_hidden_states = generation_hidden_states[step_idx]
            if step_hidden_states is None or len(step_hidden_states) <= layer_idx:
                return None

            generated_layer = step_hidden_states[layer_idx]
            if generated_layer is None or generated_layer.ndim != 3 or generated_layer.shape[1] < 1:
                return None

            generated_tokens.append(generated_layer[:, -1:, :])

        answer_hidden_states.append(torch.cat([prompt_layer, *generated_tokens], dim=1))

    return tuple(answer_hidden_states)


def _extract_first_key(text: str, option_keys: List[str]) -> Optional[str]:
    """
    Extract the first standalone option key from text.
    Works for keys like A..Z and also AA/AB if present.
    """
    if not text:
        return None
    t = text.strip().upper()

    # Longest keys first so "AA" matches before "A"
    keys = sorted((k.upper() for k in option_keys), key=len, reverse=True)
    if not keys:
        return None

    # Build regex like: \b(AA|AB|A|B|...)\b
    pat = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    m = re.search(pat, t)
    return m.group(1) if m else None


def _compute_effective_generated_length(
    gen_ids: torch.Tensor,
    eos_id: Optional[int],
    pad_id: Optional[int],
) -> int:
    gen_tokens = gen_ids[0].tolist()
    effective_gen_len = len(gen_tokens)

    while effective_gen_len > 0:
        tid = gen_tokens[effective_gen_len - 1]
        if (eos_id is not None and tid == eos_id) or (pad_id is not None and tid == pad_id):
            effective_gen_len -= 1
        else:
            break

    return effective_gen_len


def _empty_gen_step_logits(model, device: torch.device) -> torch.Tensor:
    return torch.empty((1, 0, model.config.vocab_size), device=device)


def _run_generation_with_features(
    *,
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    verbose: bool,
    vlm_adapter: VLMAdapter,
) -> Dict[str, object]:
    is_verbose_qwen = verbose and vlm_adapter.family == "qwen"
    qwen_timings: Dict[str, float] = {}

    if is_verbose_qwen:
        inputs, elapsed = _timed_call(device, prepare_inputs, vlm_adapter, processor, image, prompt, device)
        qwen_timings["prepare_inputs"] = elapsed
    else:
        inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)

    if verbose:
        input_ids_prompt = inputs["input_ids"][0]

        print("\n" + "=" * 80)
        print("PROMPT INPUT IDS SHAPE:", input_ids_prompt.shape)
        print("PROMPT INPUT IDS:")
        print(input_ids_prompt.tolist())

        tokens_prompt = processor.tokenizer.convert_ids_to_tokens(
            input_ids_prompt.tolist()
        )

        print("\nPROMPT TOKENS (token-by-token):")
        for i, (tid, tok) in enumerate(zip(input_ids_prompt.tolist(), tokens_prompt)):
            print(f"{i:4d} | id={tid:6d} | token={repr(tok)}")

        decoded_prompt = processor.tokenizer.decode(
            input_ids_prompt,
            skip_special_tokens=False
        )

        print("\nPROMPT FULL DECODE:")
        print(decoded_prompt)
        print("=" * 80 + "\n")

    if is_verbose_qwen:
        (vision_hidden_states, vision_merged_states), elapsed = _timed_call(
            device,
            extract_vision_features_bundle,
            vlm_adapter,
            model,
            inputs,
        )
        qwen_timings["vision_features"] = elapsed
    else:
        vision_hidden_states, vision_merged_states = extract_vision_features_bundle(vlm_adapter, model, inputs)

    input_ids = inputs["input_ids"][0]
    prompt_len = int(inputs["input_ids"].shape[1])

    if vlm_adapter.family == "qwen":
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature

        if is_verbose_qwen:
            out, elapsed = _timed_call(
                device,
                model.generate,
                **generate_kwargs,
            )
            qwen_timings["generate_answer"] = elapsed
        else:
            out = model.generate(**generate_kwargs)

        lm_hidden_states = _extract_prompt_hidden_states_from_generate_output(
            getattr(out, "hidden_states", None),
            prompt_len=prompt_len,
        )
        if lm_hidden_states is None:
            if is_verbose_qwen:
                outputs, elapsed = _timed_call(device, model, **inputs, output_hidden_states=True, return_dict=True)
                qwen_timings["prompt_forward_lm_fallback"] = elapsed
            else:
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            lm_hidden_states = getattr(outputs, "hidden_states", None)
            if lm_hidden_states is None:
                raise RuntimeError("Model did not return hidden_states with output_hidden_states=True.")
        elif is_verbose_qwen:
            qwen_timings["prompt_hidden_from_generate"] = 0.0
    else:
        if is_verbose_qwen:
            outputs, elapsed = _timed_call(device, model, **inputs, output_hidden_states=True, return_dict=True)
            qwen_timings["prompt_forward_lm"] = elapsed
        else:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        lm_hidden_states = getattr(outputs, "hidden_states", None)
        if lm_hidden_states is None:
            raise RuntimeError("Model did not return hidden_states with output_hidden_states=True.")

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature

        if is_verbose_qwen:
            out, elapsed = _timed_call(
                device,
                model.generate,
                **generate_kwargs,
            )
            qwen_timings["generate_answer"] = elapsed
        else:
            out = model.generate(**generate_kwargs)

    lm_hidden_len = int(lm_hidden_states[0].shape[1])

    if is_verbose_qwen:
        token_spans, elapsed = _timed_call(
            device,
            build_token_spans,
            model=model,
            processor=processor,
            vlm_adapter=vlm_adapter,
            input_ids=input_ids,
            prompt_len=prompt_len,
            lm_hidden_len=lm_hidden_len,
            verbose=verbose,
        )
        qwen_timings["token_spans"] = elapsed
    else:
        token_spans = build_token_spans(
            model=model,
            processor=processor,
            vlm_adapter=vlm_adapter,
            input_ids=input_ids,
            prompt_len=prompt_len,
            lm_hidden_len=lm_hidden_len,
            verbose=verbose,
        )

    gen_ids = out.sequences[:, prompt_len:]
    if len(out.scores) > 0:
        gen_step_logits = torch.stack(out.scores, dim=1)
    else:
        gen_step_logits = _empty_gen_step_logits(model, device)

    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    effective_gen_len = _compute_effective_generated_length(gen_ids, eos_id=eos_id, pad_id=pad_id)

    prompt_hidden_len = int(lm_hidden_states[0].shape[1])
    answer_start = prompt_hidden_len
    answer_end = answer_start + effective_gen_len

    token_spans["answer_start"] = answer_start
    token_spans["answer_end"] = answer_end
    token_spans["answer_length"] = answer_end - answer_start

    full_sequence = out.sequences
    if pad_id is not None:
        attention_mask = (full_sequence != pad_id).long()
    else:
        attention_mask = torch.ones_like(full_sequence)

    if verbose:
        full_ids = full_sequence[0]
        tokens_full = processor.tokenizer.convert_ids_to_tokens(full_ids.tolist())

        print("\n" + "=" * 80)
        print("FULL SEQUENCE (PROMPT + GENERATED)")
        print("Shape:", full_ids.shape)
        print("IDs:")
        print(full_ids.tolist())

        print("\nFULL SEQUENCE TOKENS (token-by-token):")
        for i, (tid, tok) in enumerate(zip(full_ids.tolist(), tokens_full)):
            print(f"{i:4d} | id={int(tid):6d} | token={repr(tok)}")

        decoded_full = processor.tokenizer.decode(
            full_ids,
            skip_special_tokens=False
        )
        print("\nFULL SEQUENCE FULL DECODE:")
        print(decoded_full)

        print("\nANSWER SPAN INDICES:")
        print("answer_start:", answer_start)
        print("answer_end:  ", answer_end)
        print("answer_length:", answer_end - answer_start)

        print("\nANSWER SPAN TOKENS (TRIMMED, used for features):")
        for i in range(answer_start, answer_end):
            print(f"{i:4d} | id={int(full_ids[i]):6d} | token={repr(tokens_full[i])}")

        decoded_answer = processor.tokenizer.decode(
            full_ids[answer_start:answer_end],
            skip_special_tokens=False
        )
        print("\nDecoded answer span (trimmed):")
        print(decoded_answer)

        raw_answer_end = prompt_hidden_len + int(gen_ids.shape[1])
        print("\nANSWER SPAN TOKENS (RAW, incl. EOS/PAD if present):")
        for i in range(answer_start, raw_answer_end):
            print(f"{i:4d} | id={int(full_ids[i]):6d} | token={repr(tokens_full[i])}")

        decoded_raw_answer = processor.tokenizer.decode(
            full_ids[answer_start:raw_answer_end],
            skip_special_tokens=False
        )
        print("\nDecoded answer span (raw):")
        print(decoded_raw_answer)

        print("=" * 80 + "\n")

    answer_hidden_states = None
    if vlm_adapter.family == "qwen":
        answer_hidden_states = _build_answer_hidden_states_from_generate_output(
            lm_hidden_states,
            getattr(out, "hidden_states", None),
            effective_gen_len=effective_gen_len,
        )
        if is_verbose_qwen and answer_hidden_states is not None:
            qwen_timings["answer_forward_lm"] = 0.0

    if answer_hidden_states is None:
        full_forward_kwargs = build_full_forward_kwargs(inputs, full_sequence, attention_mask)
        if is_verbose_qwen:
            full_outputs, elapsed = _timed_call(
                device,
                model,
                **full_forward_kwargs,
                output_hidden_states=True,
                return_dict=True,
            )
            qwen_timings["answer_forward_lm"] = elapsed
        else:
            full_outputs = model(
                **full_forward_kwargs,
                output_hidden_states=True,
                return_dict=True,
            )
        answer_hidden_states = getattr(full_outputs, "hidden_states", None)
        if answer_hidden_states is None:
            raise RuntimeError("Full forward pass did not return hidden_states.")

    if is_verbose_qwen:
        _print_verbose_qwen_timings("QWEN FEATURE EXTRACTION TIMINGS", qwen_timings)

    return {
        "raw_text": raw_text,
        "vision_hidden_states": vision_hidden_states,
        "vision_merged_states": vision_merged_states,
        "lm_hidden_states": lm_hidden_states,
        "token_spans": token_spans,
        "answer_hidden_states": answer_hidden_states,
        "gen_ids": gen_ids,
        "gen_step_logits": gen_step_logits,
    }



@torch.inference_mode()
def predict_letter_and_logits(
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    option_letters: Optional[List[str]] = None,
    vlm_adapter: Optional[VLMAdapter] = None,
) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[Dict[str, float]], str]:
    if vlm_adapter is None:
        vlm_adapter = get_vlm_adapter(getattr(getattr(model, "config", None), "_name_or_path", ""))

    if option_letters is None:
        option_letters = ["A", "B", "C", "D"]

    inputs = prepare_inputs(vlm_adapter, processor, image, prompt, device)

    out = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # 1D tensor (V,)
    first_step_logits = out.scores[0][0]

    # token ids for each option key, if single-token
    letter_token_ids: Optional[Dict[str, int]] = {}
    for L in option_letters:
        ids = processor.tokenizer.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            letter_token_ids = None
            break
        letter_token_ids[L] = ids[0]

    predicted = None
    option_logits = None
    option_probs = None

    if letter_token_ids is not None:
        option_logits = {L: float(first_step_logits[tid].item()) for L, tid in letter_token_ids.items()}
        logits_tensor = torch.stack([first_step_logits[letter_token_ids[L]] for L in option_letters]).float()
        probs_tensor = torch.softmax(logits_tensor, dim=0)
        option_probs = {L: float(probs_tensor[i].item()) for i, L in enumerate(option_letters)}
        predicted = option_letters[int(torch.argmax(probs_tensor).item())]

    # Always try to parse text and override if valid
    parsed = _extract_first_key(raw_text, option_letters)
    if parsed in option_letters:
        predicted = parsed

    return predicted, option_probs, option_logits, raw_text



@torch.inference_mode()
def predict_letter_and_logits_with_features(
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    option_letters: Optional[List[str]] = None,
    verbose: bool = False,
    vlm_adapter: Optional[VLMAdapter] = None,
) -> Tuple[
    Optional[str],
    Dict[str, float],
    Dict[str, float],
    str,
    Tuple[torch.Tensor, ...],      # vision_hidden_states
    torch.Tensor,                  # vision_merged_states
    Tuple[torch.Tensor, ...],      # lm_hidden_states (prompt forward)
    Dict[str, int],                # token_spans
    Tuple[torch.Tensor, ...],      # answer_hidden_states (prompt + generated answer)
    torch.Tensor,                  # gen_ids: (B, T)
    torch.Tensor,                  # gen_step_logits: (B, T, V)
]:
    if vlm_adapter is None:
        vlm_adapter = get_vlm_adapter(getattr(getattr(model, "config", None), "_name_or_path", ""))

    if option_letters is None:
        option_letters = ["A", "B", "C", "D"]
        print("Warning: option_letters not provided; defaulting to A/B/C/D. If your task has different keys, pass them explicitly for accurate parsing and scoring.")

    feature_bundle = _run_generation_with_features(
        model=model,
        processor=processor,
        device=device,
        image=image,
        prompt=prompt,
        max_new_tokens=3,
        do_sample=False,
        temperature=1.0,
        verbose=verbose,
        vlm_adapter=vlm_adapter,
    )

    raw_text = feature_bundle["raw_text"]
    vision_hidden_states = feature_bundle["vision_hidden_states"]
    vision_merged_states = feature_bundle["vision_merged_states"]
    lm_hidden_states = feature_bundle["lm_hidden_states"]
    token_spans = feature_bundle["token_spans"]
    answer_hidden_states = feature_bundle["answer_hidden_states"]
    gen_ids = feature_bundle["gen_ids"]
    gen_step_logits = feature_bundle["gen_step_logits"]

    first_step_logits = gen_step_logits[:, 0, :] if gen_step_logits.shape[1] > 0 else None

    letter_token_ids: Optional[Dict[str, int]] = {}
    for L in option_letters:
        ids = processor.tokenizer.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            letter_token_ids = None
            break
        letter_token_ids[L] = ids[0]

    if letter_token_ids is not None and first_step_logits is not None:
        option_logits = {L: float(first_step_logits[0, tid].item()) for L, tid in letter_token_ids.items()}
        logits_tensor = torch.stack([first_step_logits[0, letter_token_ids[L]] for L in option_letters]).float()
        probs_tensor = torch.softmax(logits_tensor, dim=0)
        option_probs = {L: float(probs_tensor[i].item()) for i, L in enumerate(option_letters)}
        predicted_letter = option_letters[int(torch.argmax(probs_tensor).item())]
    else:
        option_logits = None
        option_probs = None
        predicted_letter = None

    parsed_letter = _extract_first_key(raw_text, option_letters)
    if parsed_letter in option_letters:
        predicted_letter = parsed_letter

    return (
        predicted_letter,
        option_probs,
        option_logits,
        raw_text,
        vision_hidden_states,
        vision_merged_states,
        lm_hidden_states,
        token_spans,
        answer_hidden_states,
        gen_ids,
        gen_step_logits,
    )


@torch.inference_mode()
def predict_answer_and_features(
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    max_new_tokens: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    verbose: bool = False, 
    vlm_adapter: Optional[VLMAdapter] = None,
) -> Tuple[
    str,                      # pred_answer (decoded)
    Tuple[torch.Tensor, ...], # vision_hidden_states
    torch.Tensor,             # vision_merged_states
    Tuple[torch.Tensor, ...], # lm_hidden_states (prompt forward)
    Dict[str, int],           # token_spans
    Tuple[torch.Tensor, ...], # answer_hidden_states (prompt + generated answer)
    torch.Tensor,             # gen_ids (B, T)
    torch.Tensor,             # gen_step_logits (B, T, V)
]:
    if vlm_adapter is None:
        vlm_adapter = get_vlm_adapter(getattr(getattr(model, "config", None), "_name_or_path", ""))

    feature_bundle = _run_generation_with_features(
        model=model,
        processor=processor,
        device=device,
        image=image,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        verbose=verbose,
        vlm_adapter=vlm_adapter,
    )

    pred_answer = feature_bundle["raw_text"]
    vision_hidden_states = feature_bundle["vision_hidden_states"]
    vision_merged_states = feature_bundle["vision_merged_states"]
    lm_hidden_states = feature_bundle["lm_hidden_states"]
    token_spans = feature_bundle["token_spans"]
    answer_hidden_states = feature_bundle["answer_hidden_states"]
    gen_ids = feature_bundle["gen_ids"]
    gen_step_logits = feature_bundle["gen_step_logits"]

    return (
        pred_answer,
        vision_hidden_states,
        vision_merged_states,
        lm_hidden_states,
        token_spans,
        answer_hidden_states,
        gen_ids,
        gen_step_logits,
    )

    
