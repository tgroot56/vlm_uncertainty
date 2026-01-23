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
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    
    # Print model architecture information
    print(f"\n{'='*60}")
    print("Model Architecture Information:")
    print(f"{'='*60}")
    
    # Vision tower information
    if hasattr(model, 'vision_tower'):
        vision_tower = model.vision_tower
        print(f"\nVision Tower: {vision_tower.__class__.__name__}")
        
        # Get the underlying vision model (typically CLIP)
        if hasattr(vision_tower, 'vision_model'):
            vision_model = vision_tower.vision_model
            print(f"Vision Model Type: {vision_model.__class__.__name__}")
            
            # Count encoder layers
            if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
                num_layers = len(vision_model.encoder.layers)
                print(f"Number of Vision Encoder Layers: {num_layers}")
            
            # Print config info if available
            if hasattr(vision_model, 'config'):
                config = vision_model.config
                if hasattr(config, 'num_hidden_layers'):
                    print(f"Config num_hidden_layers: {config.num_hidden_layers}")
                if hasattr(config, 'hidden_size'):
                    print(f"Hidden size: {config.hidden_size}")
                if hasattr(config, 'model_type'):
                    print(f"Model type: {config.model_type}")
    else:
        print("No vision tower found in the model.")
    
    # Multi-modal projector
    if hasattr(model, 'multi_modal_projector'):
        projector = model.multi_modal_projector
        print(f"\nMulti-modal Projector: {projector.__class__.__name__}")

        # print hidden size if available
        
    else:
        print("No multi-modal projector found in the model.")
    
    # Language model
    if hasattr(model, 'language_model'):
        language_model = model.language_model
        print(f"\nLanguage Model: {language_model.__class__.__name__}")
        if hasattr(language_model, 'config'):
            lm_config = language_model.config
            if hasattr(lm_config, 'num_hidden_layers'):
                print(f"LM num_hidden_layers: {lm_config.num_hidden_layers}")
            if hasattr(lm_config, 'hidden_size'):
                print(f"LM hidden_size: {lm_config.hidden_size}")
    else:
        print("No language model found in the model.")
    
    print(f"{'='*60}\n")
    
    return model, processor, device
