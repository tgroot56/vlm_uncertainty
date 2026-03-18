"""Inference utilities for model predictions."""

import torch
import re
from typing import Dict, List, Tuple, Optional

from src.uq_dataset_generation.token_spans import build_token_spans

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



@torch.inference_mode()
def predict_letter_and_logits(
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    option_letters: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[Dict[str, float]], str]:

    if option_letters is None:
        option_letters = ["A", "B", "C", "D"]

    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
) -> Tuple[
    Optional[str],
    Dict[str, float],
    Dict[str, float],
    str,
    Tuple[torch.Tensor, ...],      # vision_hidden_states
    Tuple[torch.Tensor, ...],      # lm_hidden_states (prompt forward)
    Dict[str, int],                # token_spans
    Tuple[torch.Tensor, ...],      # answer_hidden_states (full forward)
    torch.Tensor,                  # gen_ids: (B, T)
    torch.Tensor,                  # gen_step_logits: (B, T, V)
]:
    if option_letters is None:
        option_letters = ["A", "B", "C", "D"]
        print("Warning: option_letters not provided; defaulting to A/B/C/D. If your task has different keys, pass them explicitly for accurate parsing and scoring.")
    
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # -------------------------------------------------
    # DEBUG: Print prompt tokens (exact model input)
    # -------------------------------------------------
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

    # ---------------------------
    # Vision tower hidden states
    # ---------------------------
    vision_hidden_states = None
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_model"):
        if "pixel_values" in inputs:
            vision_outputs = model.vision_tower.vision_model(
                inputs["pixel_values"],
                output_hidden_states=True,
            )
            if isinstance(vision_outputs, tuple):
                vision_hidden_states = vision_outputs[2] if len(vision_outputs) > 2 else None
            else:
                vision_hidden_states = getattr(vision_outputs, "hidden_states", None)

    # ---------------------------
    # Prompt forward pass (LM hidden states + spans)
    # ---------------------------
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    lm_hidden_states = getattr(outputs, "hidden_states", None)
    if lm_hidden_states is None:
        raise RuntimeError("Model did not return hidden_states. Ensure output_hidden_states=True works.")

    input_ids = inputs["input_ids"][0]
    prompt_len = int(inputs["input_ids"].shape[1])
    lm_hidden_len = int(lm_hidden_states[0].shape[1])

    token_spans = build_token_spans(
        model=model,
        processor=processor,
        input_ids=input_ids,
        prompt_len=prompt_len,
        lm_hidden_len=lm_hidden_len,
        verbose=verbose,  # Set to True to enable detailed token span debug printing
    )

    # ---------------------------
    # Generate (need scores)
    # ---------------------------
    out = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]                # (B, T)
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # (B, T, V) logits for each generated step
    gen_step_logits = torch.stack(out.scores, dim=1)


    first_step_logits = out.scores[0]                      

    letter_token_ids: Optional[Dict[str, int]] = {}
    for L in option_letters:
        ids = processor.tokenizer.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            letter_token_ids = None
            break
        letter_token_ids[L] = ids[0]

    if letter_token_ids is not None:
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


    # ---------------------------
    # Full forward pass to get answer hidden states
    # ---------------------------
    gen_len = int(gen_ids.shape[1])
    prompt_hidden_len = int(lm_hidden_states[0].shape[1])  # hidden-space prompt len

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    # gen_ids: (B, T) = generated part only
    g = gen_ids[0].tolist()

    effective_gen_len = len(g)
    while effective_gen_len > 0:
        tid = g[effective_gen_len - 1]
        if (eos_id is not None and tid == eos_id) or (pad_id is not None and tid == pad_id):
            effective_gen_len -= 1
        else:
            break

    answer_start = prompt_hidden_len
    answer_end = answer_start + effective_gen_len  # <-- excludes </s> if present

    token_spans["answer_start"] = answer_start
    token_spans["answer_end"] = answer_end
    token_spans["answer_length"] = answer_end - answer_start

    full_sequence = out.sequences
    if processor.tokenizer.pad_token_id is not None:
        attention_mask = (full_sequence != processor.tokenizer.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(full_sequence)
    
    # -------------------------------------------------
    # DEBUG: Print FULL SEQUENCE (prompt + answer)
    # -------------------------------------------------
    if verbose: 

        full_ids = full_sequence[0]   # (seq_len,)

        print("\n" + "=" * 80)
        print("FULL SEQUENCE (PROMPT + ANSWER)")
        print("Full sequence shape:", full_ids.shape)
        print("Full sequence ids:")
        print(full_ids.tolist())

        tokens_full = processor.tokenizer.convert_ids_to_tokens(
            full_ids.tolist()
        )

        print("\nFULL SEQUENCE TOKENS (token-by-token):")
        for i, (tid, tok) in enumerate(zip(full_ids.tolist(), tokens_full)):
            print(f"{i:4d} | id={tid:6d} | token={repr(tok)}")

        decoded_full = processor.tokenizer.decode(
            full_ids,
            skip_special_tokens=False
        )

        print("\nFULL SEQUENCE FULL DECODE:")
        print(decoded_full)

        print("\nAnswer span indices:")
        print("answer_start:", token_spans["answer_start"])
        print("answer_end:", token_spans["answer_end"])
        print("answer_length:", token_spans["answer_length"])

        print("=" * 80 + "\n")

        full_ids = full_sequence[0]  # (seq_len,)
        tokens_full = processor.tokenizer.convert_ids_to_tokens(
            full_ids.tolist()
        )

        a0 = token_spans["answer_start"]
        a1 = token_spans["answer_end"]

        print("\n" + "=" * 80)
        print("ANSWER SPAN TOKENS")
        print(f"Span: [{a0}, {a1})  length={a1-a0}")

        for i in range(a0, a1):
            print(f"{i:4d} | id={int(full_ids[i])} | token={repr(tokens_full[i])}")

        decoded_answer = processor.tokenizer.decode(
            full_ids[a0:a1],
            skip_special_tokens=False
        )
        print("\nDecoded answer span:")
        print(decoded_answer)
        print("=" * 80 + "\n")

    full_outputs = model(
        input_ids=full_sequence,
        pixel_values=inputs.get("pixel_values", None),
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    answer_hidden_states = getattr(full_outputs, "hidden_states", None)

    return (
        predicted_letter,
        option_probs,
        option_logits,
        raw_text,
        vision_hidden_states,
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
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 1.0,
    verbose: bool = False, 
) -> Tuple[
    str,                      # pred_answer (decoded)
    Tuple[torch.Tensor, ...], # vision_hidden_states
    Tuple[torch.Tensor, ...], # lm_hidden_states (prompt forward)
    Dict[str, int],           # token_spans
    Tuple[torch.Tensor, ...], # answer_hidden_states (full forward)
    torch.Tensor,             # gen_ids (B, T)
    torch.Tensor,             # gen_step_logits (B, T, V)
]:
    # 1) Build multimodal prompt (same as your code)
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # -------------------------------------------------
    # DEBUG: Print prompt tokens (exact model input)
    # -------------------------------------------------
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

    # 2) Vision tower hidden states (unchanged)
    vision_hidden_states = None
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_model"):
        if "pixel_values" in inputs:
            vision_outputs = model.vision_tower.vision_model(
                inputs["pixel_values"],
                output_hidden_states=True,
            )
            vision_hidden_states = getattr(vision_outputs, "hidden_states", None)

    # 3) Prompt forward pass -> lm_hidden_states + token_spans (unchanged)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    lm_hidden_states = getattr(outputs, "hidden_states", None)
    if lm_hidden_states is None:
        raise RuntimeError("Model did not return hidden_states with output_hidden_states=True.")

    input_ids = inputs["input_ids"][0]
    prompt_len = int(inputs["input_ids"].shape[1])
    lm_hidden_len = int(lm_hidden_states[0].shape[1])

    token_spans = build_token_spans(
        model=model,
        processor=processor,
        input_ids=input_ids,
        prompt_len=prompt_len,
        lm_hidden_len=lm_hidden_len,
        verbose=verbose,  # Set to True to enable detailed token span debug printing
    )

    # 4) Generate answer + keep per-step logits (THIS is the key change)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,   # <-- needed for gen_step_logits
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]  # (B, T)

    # (B, T, V) logits per generated token step
    gen_step_logits = torch.stack(out.scores, dim=1) if len(out.scores) > 0 else torch.empty(
        (1, 0, model.config.vocab_size), device=device
    )

    pred_answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # 5) Full forward pass on (prompt + answer) to get answer_hidden_states (unchanged)
    gen_len = int(gen_ids.shape[1])
    prompt_hidden_len = int(lm_hidden_states[0].shape[1])  # hidden-space prompt len

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    # gen_ids: (B, T) = generated part only
    g = gen_ids[0].tolist()

    effective_gen_len = len(g)
    while effective_gen_len > 0:
        tid = g[effective_gen_len - 1]
        if (eos_id is not None and tid == eos_id) or (pad_id is not None and tid == pad_id):
            effective_gen_len -= 1
        else:
            break

    answer_start = prompt_hidden_len
    answer_end = answer_start + effective_gen_len  # <-- excludes </s> if present

    token_spans["answer_start"] = answer_start
    token_spans["answer_end"] = answer_end
    token_spans["answer_length"] = answer_end - answer_start

    full_sequence = out.sequences
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        attention_mask = (full_sequence != pad_id).long()
    else:
        attention_mask = torch.ones_like(full_sequence)

    
    # -------------------------------------------------
    # DEBUG: Print full sequence tokens (prompt + answer)
    # -------------------------------------------------
    if verbose:
        full_ids = full_sequence[0]  # (seq_len,)
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

        # Optional: also show raw generated tokens including EOS
        raw_answer_end = prompt_hidden_len + gen_len
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

    full_outputs = model(
        input_ids=full_sequence,
        pixel_values=inputs.get("pixel_values", None),
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    answer_hidden_states = getattr(full_outputs, "hidden_states", None)
    if answer_hidden_states is None:
        raise RuntimeError("Full forward pass did not return hidden_states.")

    return (
        pred_answer,
        vision_hidden_states,
        lm_hidden_states,
        token_spans,
        answer_hidden_states,
        gen_ids,
        gen_step_logits,
    )

    


