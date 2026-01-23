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
) -> Tuple[Optional[str], Dict[str, float], Dict[str, float], str, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Dict[str, int], Tuple[torch.Tensor, ...]]:
    """
    Runs LLaVA on (image, prompt) and returns predictions with logits/probabilities,
    vision hidden states, language model hidden states, and answer hidden states for feature extraction.
    
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
            - lm_hidden_states: tuple of hidden states from language model (input only)
            - token_spans: dict with 'visual_start', 'visual_end', 'prompt_start', 'prompt_end', 'answer_start', 'answer_end'
            - answer_hidden_states: tuple of hidden states from language model (with answer tokens)
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

    # Do a forward pass to get LM hidden states (before generation)
    lm_hidden_states = None
    token_spans = {}
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        lm_hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        
        # Robustly identify token spans by inspecting the actual model behavior
        input_ids = inputs['input_ids'][0]  # Shape: (seq_len,)
        
        # Step 1: Find the IMAGE_TOKEN_ID in the input sequence
        # LLaVA uses a special token (typically IMAGE_TOKEN_INDEX = -200) as a placeholder
        image_token_id = None
        image_token_position = None
        
        # Try to get the image token ID from the model config
        if hasattr(model, 'config') and hasattr(model.config, 'image_token_index'):
            image_token_id = model.config.image_token_index
        
        # Search for the image token in input_ids
        if image_token_id is not None:
            image_token_mask = (input_ids == image_token_id)
            if image_token_mask.any():
                image_token_position = image_token_mask.nonzero(as_tuple=True)[0][0].item()
        
        # Step 2: Determine the actual number of visual tokens
        # This is the number of patch tokens that replace the image token placeholder
        num_visual_tokens = 576  # Default for ViT-L/14 with 336x336 images (24x24 patches)
        
        if hasattr(model, 'config'):
            # Try various config attributes
            if hasattr(model.config, 'num_query_tokens'):
                num_visual_tokens = model.config.num_query_tokens
            elif hasattr(model.config, 'vision_config') and hasattr(model.config.vision_config, 'image_size'):
                # Calculate from image size and patch size
                image_size = model.config.vision_config.image_size
                patch_size = model.config.vision_config.patch_size if hasattr(model.config.vision_config, 'patch_size') else 14
                num_patches_per_side = image_size // patch_size
                num_visual_tokens = num_patches_per_side * num_patches_per_side
        
        # Alternatively, check the actual hidden states shape if available
        if lm_hidden_states is not None and len(lm_hidden_states) > 0:
            actual_seq_len = lm_hidden_states[0].shape[1]
            input_seq_len = len(input_ids)
            # If hidden states are longer than input_ids, the difference is from visual token expansion
            if actual_seq_len > input_seq_len:
                num_visual_tokens = actual_seq_len - input_seq_len + 1  # +1 because 1 image token is replaced
        
        # Step 3: Compute token spans
        # In LLaVA, the sequence structure after processing is:
        # [BOS] [system_tokens] [visual_tokens] [user_prompt_tokens] [assistant_tokens]
        
        if image_token_position is not None:
            # Visual tokens replace the image token placeholder
            visual_start = image_token_position
            visual_end = visual_start + num_visual_tokens
            
            # Prompt tokens come after visual tokens
            # Everything after visual tokens until the end
            prompt_start = visual_end
            prompt_end = actual_seq_len if lm_hidden_states is not None else len(input_ids) + num_visual_tokens - 1
        else:
            # Fallback: assume standard LLaVA 1.5 layout
            # Visual tokens typically start at position 1 (after BOS)
            visual_start = 1
            visual_end = visual_start + num_visual_tokens
            prompt_start = visual_end
            prompt_end = len(input_ids) + num_visual_tokens - 1
        
        token_spans = {
            'visual_start': visual_start,
            'visual_end': visual_end,
            'prompt_start': prompt_start,
            'prompt_end': prompt_end,
            'num_visual_tokens': num_visual_tokens,
            'image_token_id': image_token_id,
            'image_token_position': image_token_position,
            'input_seq_len': len(input_ids),
            'hidden_seq_len': lm_hidden_states[0].shape[1] if lm_hidden_states is not None else None,
        }

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

    # Extract answer hidden states by doing another forward pass with the full sequence
    answer_hidden_states = None
    answer_start = prompt_len
    answer_end = answer_start + len(gen_ids[0])
    
    # Add answer span to token_spans
    token_spans['answer_start'] = answer_start
    token_spans['answer_end'] = answer_end
    token_spans['answer_length'] = answer_end - answer_start
    
    # Forward pass with the complete sequence (input + generated answer)
    with torch.no_grad():
        full_sequence = out.sequences  # Shape: (batch_size, prompt_len + generated_len)
        
        # Create inputs with the full sequence
        full_outputs = model(
            input_ids=full_sequence,
            pixel_values=inputs['pixel_values'] if 'pixel_values' in inputs else None,
            attention_mask=torch.ones_like(full_sequence),
            output_hidden_states=True,
            return_dict=True,
        )
        answer_hidden_states = full_outputs.hidden_states if hasattr(full_outputs, 'hidden_states') else None

    return predicted_letter, option_probs, option_logits, raw_text, vision_hidden_states, lm_hidden_states, token_spans, answer_hidden_states



