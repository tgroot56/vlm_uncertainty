from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from src.storage_paths import configure_hf_cache_env
from transformers import AutoProcessor, LlavaForConditionalGeneration

configure_hf_cache_env()

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


@dataclass(frozen=True)
class VLMAdapter:
    family: str
    loader_kind: str
    input_builder: str
    trust_remote_code: bool
    assistant_markers: Tuple[str, ...]
    image_token_text_markers: Tuple[str, ...]
    vision_attr_paths: Tuple[Tuple[str, ...], ...]


LLAVA_ADAPTER = VLMAdapter(
    family="llava",
    loader_kind="llava",
    input_builder="llava_text_and_images",
    trust_remote_code=False,
    assistant_markers=("ASSISTANT:", "Assistant:"),
    image_token_text_markers=("image", "vision"),
    vision_attr_paths=(
        ("model", "vision_tower", "vision_model"),
        ("model", "vision_tower"),
        ("vision_tower", "vision_model"),
        ("vision_tower",),
    ),
)

QWEN_ADAPTER = VLMAdapter(
    family="qwen",
    loader_kind="auto_image_text_to_text",
    input_builder="chat_template_tokenized",
    trust_remote_code=True,
    assistant_markers=(
        "<|im_start|>assistant",
        "<|assistant|>",
        "assistant",
        "ASSISTANT:",
        "Assistant:",
    ),
    image_token_text_markers=("image", "vision", "img"),
    vision_attr_paths=(("visual",), ("model", "visual"), ("vision_tower", "vision_model")),
)


def get_vlm_adapter(model_id: str) -> VLMAdapter:
    model_id_lower = model_id.lower()

    if "llava" in model_id_lower:
        return LLAVA_ADAPTER
    if "qwen" in model_id_lower:
        return QWEN_ADAPTER

    raise ValueError(
        f"Unknown model_id '{model_id}'. "
        "Add a matching adapter in src/vlm/adapters.py."
    )


def load_model(
    model_id: str,
    device: str | None = None,
) -> Tuple[object, object, torch.device, VLMAdapter]:
    adapter = get_vlm_adapter(model_id)
    torch_device = _resolve_device(device)
    dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

    print(f"Loading {adapter.family} model: {model_id}")
    print(f"Device: {torch_device}, dtype: {dtype}")

    if adapter.loader_kind == "llava":
        model = _load_llava_model(model_id, torch_device, dtype, adapter.trust_remote_code)
    elif adapter.loader_kind == "auto_image_text_to_text":
        model = _load_auto_image_text_to_text_model(model_id, torch_device, dtype, adapter.trust_remote_code)
    else:
        raise ValueError(f"Unsupported loader_kind '{adapter.loader_kind}' for model_id '{model_id}'")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=adapter.trust_remote_code,
    )
    model.eval()

    print_model_overview(model, adapter.family)

    return model, processor, torch_device, adapter


def prepare_inputs(
    adapter: VLMAdapter,
    processor: Any,
    image: Any,
    prompt: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if adapter.input_builder == "llava_text_and_images":
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
        return _move_batch_to_device(inputs, device)

    if adapter.input_builder == "chat_template_tokenized":
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]

        supports_enable_thinking = _processor_supports_enable_thinking(processor)

        try:
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if supports_enable_thinking:
                template_kwargs["enable_thinking"] = False

            inputs = processor.apply_chat_template(messages, **template_kwargs)
            inputs = _move_batch_to_device(inputs, device)
            if adapter.family == "qwen":
                inputs = _strip_qwen_trailing_thinking_tokens(inputs)
            return inputs
        except TypeError:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_batch_to_device(inputs, device)
            if adapter.family == "qwen":
                inputs = _strip_qwen_trailing_thinking_tokens(inputs)
            return inputs
        except Exception:
            text_messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }]
            try:
                text_template_kwargs = {
                    "add_generation_prompt": True,
                    "tokenize": False,
                }
                if supports_enable_thinking:
                    text_template_kwargs["enable_thinking"] = False

                text_prompt = processor.apply_chat_template(text_messages, **text_template_kwargs)
            except TypeError:
                text_prompt = processor.apply_chat_template(
                    text_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            if adapter.family == "qwen":
                text_prompt = _strip_qwen_trailing_thinking_text(text_prompt)
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
            return _move_batch_to_device(inputs, device)

    raise ValueError(f"Unsupported input_builder '{adapter.input_builder}'")


def prepare_qwen_visual_only_inputs_batch(
    processor: Any,
    images: Sequence[Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if len(images) == 0:
        raise ValueError("prepare_qwen_visual_only_inputs_batch requires at least one image.")

    # Fast path: let the processor build vision tensors directly from images.
    try:
        inputs = processor(images=list(images), return_tensors="pt")
        inputs = _move_batch_to_device(inputs, device)
        minimal_inputs = _extract_qwen_visual_tensor_inputs(inputs)
        if minimal_inputs is not None:
            return minimal_inputs
    except Exception:
        pass

    # Fallback: build minimal image-only chat inputs per sample, then concatenate
    # only the visual tensors needed by the vision tower.
    per_image_inputs = [
        _prepare_qwen_visual_only_single_inputs(processor=processor, image=image, device=device)
        for image in images
    ]
    return _concat_qwen_visual_only_inputs(per_image_inputs)


def extract_vision_hidden_states(
    adapter: VLMAdapter,
    model: Any,
    inputs: Dict[str, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, ...]]:
    hidden_states, _ = extract_vision_features_bundle(adapter, model, inputs)
    return hidden_states


def extract_vision_features_bundle(
    adapter: VLMAdapter,
    model: Any,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[Optional[Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
    vision_module = None
    for path in adapter.vision_attr_paths:
        vision_module = _resolve_attr_path(model, path)
        if vision_module is not None:
            break

    if vision_module is None:
        return None, None

    if "pixel_values" not in inputs:
        return None, None

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs.get("image_grid_thw")
    try:
        if adapter.family == "qwen" and image_grid_thw is not None:
            vision_outputs = vision_module(pixel_values, image_grid_thw, output_hidden_states=True)
        else:
            vision_kwargs: Dict[str, Any] = {
                "pixel_values": pixel_values,
                "output_hidden_states": True,
            }
            if image_grid_thw is not None:
                vision_kwargs["image_grid_thw"] = image_grid_thw
                vision_kwargs["grid_thw"] = image_grid_thw
            vision_outputs = vision_module(**vision_kwargs)
    except TypeError:
        try:
            if adapter.family == "qwen" and image_grid_thw is not None:
                vision_outputs = vision_module(pixel_values, grid_thw=image_grid_thw, output_hidden_states=True)
            else:
                vision_outputs = vision_module(pixel_values, output_hidden_states=True)
        except Exception:
            return None, None
    except Exception:
        return None, None

    merged_states = getattr(vision_outputs, "pooler_output", None)
    if merged_states is None:
        merged_states = getattr(vision_outputs, "last_hidden_state", None)
    if isinstance(vision_outputs, tuple):
        if len(vision_outputs) > 2:
            return vision_outputs[2], merged_states
        return None, merged_states

    return getattr(vision_outputs, "hidden_states", None), merged_states


def extract_qwen_selected_vision_tensors_batch(
    adapter: VLMAdapter,
    model: Any,
    inputs: Dict[str, torch.Tensor],
    capture_layer_indices: Tuple[int, ...] = (13, -1),
) -> Dict[str, Any]:
    if adapter.family != "qwen":
        raise ValueError("extract_qwen_selected_vision_tensors_batch only supports Qwen adapters.")

    vision_module = None
    for path in adapter.vision_attr_paths:
        vision_module = _resolve_attr_path(model, path)
        if vision_module is not None:
            break
    if vision_module is None:
        raise RuntimeError("Could not resolve Qwen vision module from model.")

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise RuntimeError("Qwen visual-only batching requires both pixel_values and image_grid_thw.")

    blocks = getattr(vision_module, "blocks", None)
    if blocks is None:
        raise RuntimeError("Resolved Qwen vision module does not expose transformer blocks.")

    num_blocks = len(blocks)
    normalized_indices = []
    for idx in capture_layer_indices:
        actual_idx = idx if idx >= 0 else num_blocks + idx
        if actual_idx < 0 or actual_idx >= num_blocks:
            raise IndexError(
                f"Requested Qwen vision layer {idx}, but vision module has {num_blocks} blocks."
            )
        normalized_indices.append(actual_idx)

    captured_layers: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(actual_idx: int):
        def _hook(_module: Any, _module_inputs: Tuple[Any, ...], module_output: Any) -> None:
            if isinstance(module_output, tuple):
                module_output = module_output[0]
            captured_layers[actual_idx] = module_output

        return _hook

    for actual_idx in dict.fromkeys(normalized_indices):
        handles.append(blocks[actual_idx].register_forward_hook(_make_hook(actual_idx)))

    try:
        try:
            vision_outputs = vision_module(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
        except TypeError:
            vision_outputs = vision_module(
                pixel_values,
                grid_thw=image_grid_thw,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    for actual_idx in normalized_indices:
        if actual_idx not in captured_layers:
            raise RuntimeError(f"Failed to capture Qwen vision layer {actual_idx}.")

    pre_merge_split_sizes = image_grid_thw.prod(dim=-1).detach().cpu().tolist()
    spatial_merge_size = int(getattr(vision_module, "spatial_merge_size", 1))
    merged_split_sizes = (image_grid_thw.prod(dim=-1) // (spatial_merge_size ** 2)).detach().cpu().tolist()

    merged_states = getattr(vision_outputs, "pooler_output", None)
    if merged_states is None:
        merged_states = getattr(vision_outputs, "last_hidden_state", None)
    if merged_states is None:
        raise RuntimeError("Qwen vision forward did not return merged visual states.")

    layer_state_splits = tuple(
        tuple(torch.split(captured_layers[actual_idx], pre_merge_split_sizes, dim=0))
        for actual_idx in normalized_indices
    )
    merged_state_splits = tuple(torch.split(merged_states, merged_split_sizes, dim=0))

    return {
        "capture_layer_indices": capture_layer_indices,
        "vision_hidden_state_splits": layer_state_splits,
        "vision_merged_state_splits": merged_state_splits,
    }


def build_full_forward_kwargs(
    inputs: Dict[str, torch.Tensor],
    full_sequence: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    kwargs: Dict[str, torch.Tensor] = {}
    full_seq_len = int(full_sequence.shape[1])

    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask", "position_ids", "cache_position"}:
            continue

        # Qwen provides per-token modality labels for the prompt. When we run a
        # second forward pass on prompt + generated answer, extend those labels
        # so the new answer tokens are treated as plain text tokens.
        if key == "mm_token_type_ids":
            if value.shape[1] > full_seq_len:
                raise ValueError(
                    f"{key} has length {value.shape[1]}, which exceeds full sequence length {full_seq_len}."
                )
            if value.shape[1] < full_seq_len:
                pad = torch.zeros(
                    (value.shape[0], full_seq_len - value.shape[1]),
                    dtype=value.dtype,
                    device=value.device,
                )
                value = torch.cat([value, pad], dim=1)
            kwargs[key] = value
            continue

        kwargs[key] = value

    kwargs["input_ids"] = full_sequence
    kwargs["attention_mask"] = attention_mask
    return kwargs


def print_model_overview(model: Any, family: str) -> None:
    print("\n" + "=" * 60)
    print(f"{family.upper()} Architecture Overview")
    print("=" * 60)

    vision_cfg = None
    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_model"):
        vision_cfg = getattr(model.vision_tower.vision_model, "config", None)
    elif (
        hasattr(model, "model")
        and hasattr(model.model, "vision_tower")
        and hasattr(model.model.vision_tower, "vision_model")
    ):
        vision_cfg = getattr(model.model.vision_tower.vision_model, "config", None)
    elif hasattr(model, "visual") and hasattr(model.visual, "config"):
        vision_cfg = model.visual.config
    elif hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "config"):
        vision_cfg = model.model.visual.config

    if vision_cfg is not None:
        print(f"Vision model type: {getattr(vision_cfg, 'model_type', 'unknown')}")
        if hasattr(vision_cfg, "hidden_size"):
            print(f"  Vision hidden size: {vision_cfg.hidden_size}")
        if hasattr(vision_cfg, "num_hidden_layers"):
            print(f"  Vision layers: {vision_cfg.num_hidden_layers}")
    else:
        print("No vision config found.")

    lm_cfg = None
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        lm_cfg = model.language_model.config
    elif hasattr(model, "model") and hasattr(model.model, "config"):
        lm_cfg = model.model.config
    elif hasattr(model, "config"):
        lm_cfg = model.config

    if lm_cfg is not None:
        print(f"Language model type: {getattr(lm_cfg, 'model_type', 'unknown')}")
        if hasattr(lm_cfg, "hidden_size"):
            print(f"  LM hidden size: {lm_cfg.hidden_size}")
        if hasattr(lm_cfg, "num_hidden_layers"):
            print(f"  LM layers: {lm_cfg.num_hidden_layers}")
    else:
        print("No language model config found.")

    print("=" * 60 + "\n")


def infer_image_token_ids(
    adapter: VLMAdapter,
    model: Any,
    processor: Any,
) -> Tuple[int, ...]:
    token_ids = set()
    config = getattr(model, "config", None)

    for attr_name in ("image_token_index", "image_token_id", "image_token_ids"):
        value = getattr(config, attr_name, None)
        if value is None:
            continue
        if isinstance(value, int):
            token_ids.add(int(value))
        elif isinstance(value, Iterable):
            token_ids.update(int(v) for v in value if isinstance(v, int))

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return tuple(sorted(token_ids))

    special_strings = set()
    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for value in special_map.values():
        if isinstance(value, str):
            special_strings.add(value)
        elif isinstance(value, list):
            special_strings.update(v for v in value if isinstance(v, str))

    additional_specials = getattr(tokenizer, "additional_special_tokens", []) or []
    special_strings.update(tok for tok in additional_specials if isinstance(tok, str))

    common_candidates = (
        "<image>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<img>",
    )
    special_strings.update(common_candidates)

    for token_str in special_strings:
        token_lower = token_str.lower()
        if any(marker in token_lower for marker in adapter.image_token_text_markers):
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(token_id, int) and token_id >= 0:
                token_ids.add(int(token_id))

    return tuple(sorted(token_ids))


def _processor_supports_enable_thinking(processor: Any) -> bool:
    try:
        sig = inspect.signature(processor.apply_chat_template)
    except (TypeError, ValueError):
        return False
    return "enable_thinking" in sig.parameters


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_llava_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    if device.type == "cuda":
        return LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
    return LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).to(device)


def _load_auto_image_text_to_text_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    if AutoModelForImageTextToText is None:
        raise ImportError(
            "Your installed transformers version does not expose "
            "AutoModelForImageTextToText. Upgrade transformers to load Qwen/Qwen3.5-9B."
        )

    try:
        if device.type == "cuda":
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
        return AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).to(device)
    except ValueError as exc:
        if "qwen3_5" in str(exc):
            raise ValueError(
                "The installed transformers build in your 'llava-experiments' environment "
                "does not recognize the Qwen 3.5 architecture yet. "
                "Try upgrading transformers in that environment, or install a newer build "
                "from source if the released version still lacks qwen3_5 support."
            ) from exc
        raise


def _move_batch_to_device(batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    if hasattr(batch, "to"):
        batch = batch.to(device)
    return dict(batch)


def _extract_qwen_visual_tensor_inputs(batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        return None
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def _strip_qwen_trailing_thinking_text(text_prompt: str) -> str:
    think_markers = ("\n<think>\n</think>\n\n", "\n<think>\n")
    for marker in think_markers:
        if text_prompt.endswith(marker):
            return text_prompt[: -len(marker)]
    return text_prompt


def _strip_qwen_trailing_thinking_tokens(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = batch.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return batch

    tokens = input_ids[0]
    think_positions = (tokens == 248068).nonzero(as_tuple=False)
    if think_positions.numel() == 0:
        return batch

    last_think_idx = int(think_positions[-1].item())
    trim_idx = last_think_idx
    trimmed_batch: Dict[str, torch.Tensor] = {}
    seq_len = tokens.shape[0]

    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            trimmed_batch[key] = value
            continue

        if value.ndim >= 2 and value.shape[0] == input_ids.shape[0] and value.shape[1] == seq_len:
            trimmed_batch[key] = value[:, :trim_idx, ...]
        else:
            trimmed_batch[key] = value

    return trimmed_batch


def _prepare_qwen_visual_only_single_inputs(
    processor: Any,
    image: Any,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    supports_enable_thinking = _processor_supports_enable_thinking(processor)
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}],
    }]

    try:
        template_kwargs = {
            "add_generation_prompt": False,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if supports_enable_thinking:
            template_kwargs["enable_thinking"] = False
        inputs = processor.apply_chat_template(messages, **template_kwargs)
        inputs = _move_batch_to_device(inputs, device)
        minimal_inputs = _extract_qwen_visual_tensor_inputs(inputs)
        if minimal_inputs is not None:
            return minimal_inputs
    except Exception:
        pass

    fallback_inputs = prepare_inputs(QWEN_ADAPTER, processor, image, "", device)
    minimal_inputs = _extract_qwen_visual_tensor_inputs(fallback_inputs)
    if minimal_inputs is None:
        raise RuntimeError("Could not build Qwen visual-only inputs with image_grid_thw.")
    return minimal_inputs


def _concat_qwen_visual_only_inputs(
    per_image_inputs: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if len(per_image_inputs) == 0:
        raise ValueError("_concat_qwen_visual_only_inputs requires at least one input batch.")

    return {
        "pixel_values": torch.cat([batch["pixel_values"] for batch in per_image_inputs], dim=0),
        "image_grid_thw": torch.cat([batch["image_grid_thw"] for batch in per_image_inputs], dim=0),
    }


def _resolve_attr_path(root: Any, path: Tuple[str, ...]) -> Any:
    current = root
    for attr in path:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current
