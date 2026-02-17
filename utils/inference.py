"""Inference utilities for model predictions."""

import torch
import re
from typing import Dict, List, Tuple, Optional


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
    s_in = int(input_ids.numel())
    s_hid = int(lm_hidden_states[0].shape[1])

    image_token_id = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_id is None:
        raise RuntimeError("model.config.image_token_index missing; can't find image placeholder reliably.")

    mask = (input_ids == image_token_id)
    if not mask.any():
        raise RuntimeError("Could not find image placeholder token in input_ids; spans unreliable.")
    image_token_position = int(mask.nonzero(as_tuple=True)[0][0].item())

    num_visual_tokens = s_hid - s_in + 1
    if num_visual_tokens <= 0:
        raise RuntimeError(f"Non-positive num_visual_tokens={num_visual_tokens} (s_hid={s_hid}, s_in={s_in}).")

    visual_start = image_token_position
    visual_end = visual_start + num_visual_tokens

    text_pre_start, text_pre_end = 0, visual_start
    text_post_start, text_post_end = visual_end, s_hid

    token_spans = {
        "visual_start": visual_start,
        "visual_end": visual_end,
        "text_pre_start": text_pre_start,
        "text_pre_end": text_pre_end,
        "text_post_start": text_post_start,
        "text_post_end": text_post_end,
        "num_visual_tokens": num_visual_tokens,
        "image_token_id": int(image_token_id),
        "image_token_position": int(image_token_position),
        "input_seq_len": s_in,
        "hidden_seq_len": s_hid,
    }

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
    answer_start = prompt_hidden_len
    answer_end = answer_start + gen_len

    token_spans["answer_start"] = answer_start
    token_spans["answer_end"] = answer_end
    token_spans["answer_length"] = answer_end - answer_start

    full_sequence = out.sequences
    if processor.tokenizer.pad_token_id is not None:
        attention_mask = (full_sequence != processor.tokenizer.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(full_sequence)

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
    s_in = int(input_ids.numel())
    s_hid = int(lm_hidden_states[0].shape[1])

    image_token_id = getattr(getattr(model, "config", None), "image_token_index", None)
    if image_token_id is None:
        raise RuntimeError("model.config.image_token_index missing; can't find image placeholder reliably.")

    mask = (input_ids == image_token_id)
    if not mask.any():
        raise RuntimeError("Could not find image placeholder token in input_ids; spans unreliable.")
    image_token_position = int(mask.nonzero(as_tuple=True)[0][0].item())

    num_visual_tokens = s_hid - s_in + 1
    if num_visual_tokens <= 0:
        raise RuntimeError(f"Non-positive num_visual_tokens={num_visual_tokens} (s_hid={s_hid}, s_in={s_in}).")

    visual_start = image_token_position
    visual_end = visual_start + num_visual_tokens

    token_spans = {
        "visual_start": visual_start,
        "visual_end": visual_end,
        "text_pre_start": 0,
        "text_pre_end": visual_start,
        "text_post_start": visual_end,
        "text_post_end": s_hid,
        "num_visual_tokens": num_visual_tokens,
        "image_token_id": int(image_token_id),
        "image_token_position": int(image_token_position),
        "input_seq_len": s_in,
        "hidden_seq_len": s_hid,
    }

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
    answer_start = prompt_hidden_len
    answer_end = answer_start + gen_len

    token_spans["answer_start"] = answer_start
    token_spans["answer_end"] = answer_end
    token_spans["answer_length"] = answer_end - answer_start

    full_sequence = out.sequences
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        attention_mask = (full_sequence != pad_id).long()
    else:
        attention_mask = torch.ones_like(full_sequence)

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

    


