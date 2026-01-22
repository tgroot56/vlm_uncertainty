"""Model loading utilities for LLaVA."""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Tuple


def load_llava_model(
    model_id: str = "llava-hf/llava-1.5-7b-hf"
) -> Tuple[LlavaForConditionalGeneration, AutoProcessor, torch.device]:
    """
    Load LLaVA model and processor.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Tuple of (model, processor, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading LLaVA model on {device.type.upper()}: {model_id}")
    
    if torch.cuda.is_available():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    
    return model, processor, device
