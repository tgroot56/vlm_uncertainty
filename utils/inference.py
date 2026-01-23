"""Inference utilities for model predictions."""

import torch
import re
from typing import Dict, List, Tuple, Optional

LETTERS = ["A", "B", "C", "D"]


def _extract_first_letter(text: str) -> Optional[str]:
    """
    Extract the first standalone A/B/C/D from model text output.
    
    Args:
        text: Raw model output text
        
    Returns:
        First letter found (A/B/C/D) or None
    """
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


@torch.inference_mode()
def predict_letter_and_logits(
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    option_letters: List[str] = LETTERS,
) -> Tuple[Optional[str], Dict[str, float], Dict[str, float], str]:
    """
    Runs LLaVA on (image, prompt) and returns predictions with logits/probabilities.
    
    Args:
        model: LLaVA model
        processor: AutoProcessor for the model
        device: torch device
        image: PIL Image
        prompt: Text prompt
        option_letters: List of option letters (default: ["A", "B", "C", "D"])
        
    Returns:
        Tuple of:
            - predicted_letter (A/B/C/D or None)
            - option_probs: dict letter -> probability among {A,B,C,D}
            - option_logits: dict letter -> raw logit for that letter
            - raw_text: decoded generation
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate 1–3 tokens; ask for scores so we can get logits for the first generated token
    out = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Decode only the newly generated tokens (continuation)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # First-step logits (vocab-sized) for the first generated token
    # `out.scores` is a list length = num_generated_tokens; take the first
    first_step_logits = out.scores[0][0]  # shape: (vocab,)

    # Map letters -> token ids (most tokenizers have single-token "A"/"B"/"C"/"D")
    letter_token_ids = {
        L: processor.tokenizer.encode(L, add_special_tokens=False)[0]
        for L in option_letters
    }

    option_logits = {L: float(first_step_logits[tid].item()) for L, tid in letter_token_ids.items()}

    # Softmax over ONLY the 4 letter logits
    logits_tensor = torch.tensor([option_logits[L] for L in option_letters], dtype=torch.float32)
    probs_tensor = torch.softmax(logits_tensor, dim=0)
    option_probs = {L: float(probs_tensor[i].item()) for i, L in enumerate(option_letters)}

    predicted_letter = option_letters[int(torch.argmax(probs_tensor).item())]

    # Also parse the text (sometimes it outputs "A." or "A)" etc.)
    parsed_letter = _extract_first_letter(raw_text)
    if parsed_letter in option_letters:
        predicted_letter = parsed_letter

    return predicted_letter, option_probs, option_logits, raw_text


@torch.inference_mode()
def predict_letter_and_logits_with_features(
    model,
    processor,
    device: torch.device,
    image,
    prompt: str,
    option_letters: List[str] = LETTERS,
) -> Tuple[Optional[str], Dict[str, float], Dict[str, float], str, Tuple[torch.Tensor, ...]]:
    """
    Runs LLaVA on (image, prompt) and returns predictions with logits/probabilities
    and vision hidden states for feature extraction.
    
    Args:
        model: LLaVA model
        processor: AutoProcessor for the model
        device: torch device
        image: PIL Image
        prompt: Text prompt
        option_letters: List of option letters (default: ["A", "B", "C", "D"])
        
    Returns:
        Tuple of:
            - predicted_letter (A/B/C/D or None)
            - option_probs: dict letter -> probability among {A,B,C,D}
            - option_logits: dict letter -> raw logit for that letter
            - raw_text: decoded generation
            - vision_hidden_states: tuple of hidden states from vision tower
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # First, get vision hidden states by doing a forward pass through vision tower
    vision_hidden_states = None
    if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'vision_model'):
        # Extract pixel_values for the vision model
        if 'pixel_values' in inputs:
            vision_outputs = model.vision_tower.vision_model(
                inputs['pixel_values'],
                output_hidden_states=True,
            )
            # Handle both tuple and dict outputs
            if isinstance(vision_outputs, tuple):
                # Outputs are (last_hidden_state, pooler_output, hidden_states)
                vision_hidden_states = vision_outputs[2] if len(vision_outputs) > 2 else None
            else:
                # Output is a dict-like object
                vision_hidden_states = vision_outputs.hidden_states if hasattr(vision_outputs, 'hidden_states') else None

    # Generate 1–3 tokens; ask for scores so we can get logits for the first generated token
    out = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Decode only the newly generated tokens (continuation)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[:, prompt_len:]
    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # First-step logits (vocab-sized) for the first generated token
    # `out.scores` is a list length = num_generated_tokens; take the first
    first_step_logits = out.scores[0][0]  # shape: (vocab,)

    # Map letters -> token ids (most tokenizers have single-token "A"/"B"/"C"/"D")
    letter_token_ids = {
        L: processor.tokenizer.encode(L, add_special_tokens=False)[0]
        for L in option_letters
    }

    option_logits = {L: float(first_step_logits[tid].item()) for L, tid in letter_token_ids.items()}

    # Softmax over ONLY the 4 letter logits
    logits_tensor = torch.tensor([option_logits[L] for L in option_letters], dtype=torch.float32)
    probs_tensor = torch.softmax(logits_tensor, dim=0)
    option_probs = {L: float(probs_tensor[i].item()) for i, L in enumerate(option_letters)}

    predicted_letter = option_letters[int(torch.argmax(probs_tensor).item())]

    # Also parse the text (sometimes it outputs "A." or "A)" etc.)
    parsed_letter = _extract_first_letter(raw_text)
    if parsed_letter in option_letters:
        predicted_letter = parsed_letter

    return predicted_letter, option_probs, option_logits, raw_text, vision_hidden_states

