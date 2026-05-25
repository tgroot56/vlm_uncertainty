from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
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


_QWEN_THINK_START_ID = 248068
_QWEN_THINK_END_ID = 248069
_QWEN_NEWLINE_ID = 271
_QWEN_EMPTY_THINKING_PREAMBLE_IDS = (
    _QWEN_THINK_START_ID,
    _QWEN_NEWLINE_ID,
    _QWEN_THINK_END_ID,
    _QWEN_NEWLINE_ID,
)


GEMMA_ADAPTER = VLMAdapter(
    family="gemma",
    loader_kind="auto_image_text_to_text",
    input_builder="chat_template_tokenized",
    trust_remote_code=False,
    assistant_markers=(
        "<start_of_turn>model",
        "model",
        "Model:",
    ),
    image_token_text_markers=("image", "vision", "img"),
    vision_attr_paths=(
        ("model", "vision_tower", "vision_model"),
        ("model", "vision_tower"),
        ("vision_tower", "vision_model"),
        ("vision_tower",),
    ),
)


MOLMO_ADAPTER = VLMAdapter(
    family="molmo",
    loader_kind="auto_image_text_to_text",
    input_builder="chat_template_tokenized",
    trust_remote_code=True,
    assistant_markers=(
        "<|assistant|>",
        "Assistant:",
        "ASSISTANT:",
        "assistant",
    ),
    image_token_text_markers=("image",),
    vision_attr_paths=(
        ("model", "vision_backbone"),
        ("vision_backbone",),
        ("model", "vision_tower", "vision_model"),
        ("model", "vision_tower"),
    ),
)


def get_vlm_adapter(model_id: str) -> VLMAdapter:
    model_id_lower = model_id.lower()

    if "llava" in model_id_lower:
        return LLAVA_ADAPTER
    if "qwen" in model_id_lower:
        return QWEN_ADAPTER
    if "gemma" in model_id_lower:
        return GEMMA_ADAPTER
    if "molmo" in model_id_lower:
        return MOLMO_ADAPTER

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
    if torch_device.type == "cuda":
        # Gemma 3n's _init_weights fills `gradient_clipping` with a constant that
        # overflows fp16 (RuntimeError: value cannot be converted to type at::Half
        # without overflow). bf16 has matching range and is the checkpoint's
        # native dtype.
        dtype = torch.bfloat16 if adapter.family == "gemma" else torch.float16
    else:
        dtype = torch.float32

    print(f"Loading {adapter.family} model: {model_id}")
    print(f"Device: {torch_device}, dtype: {dtype}")

    _ensure_default_rope_init()

    if adapter.loader_kind == "llava":
        model = _load_llava_model(model_id, torch_device, dtype, adapter.trust_remote_code)
    elif adapter.loader_kind == "auto_image_text_to_text":
        model = _load_auto_image_text_to_text_model(model_id, torch_device, dtype, adapter.trust_remote_code)
    else:
        raise ValueError(f"Unsupported loader_kind '{adapter.loader_kind}' for model_id '{model_id}'")

    _ensure_use_cache_attr(model)
    processor = _load_processor(model_id, adapter)
    model.eval()

    print_model_overview(model, adapter.family)

    return model, processor, torch_device, adapter


def prepare_inputs(
    adapter: VLMAdapter,
    processor: Any,
    image: Any,
    prompt: str,
    device: torch.device,
    qwen_prefill_empty_thinking: bool = False,
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
                if qwen_prefill_empty_thinking:
                    inputs = append_qwen_empty_thinking_preamble(inputs, device)
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
                if qwen_prefill_empty_thinking:
                    inputs = append_qwen_empty_thinking_preamble(inputs, device)
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
            inputs = _move_batch_to_device(inputs, device)
            if adapter.family == "qwen" and qwen_prefill_empty_thinking:
                inputs = append_qwen_empty_thinking_preamble(inputs, device)
            return inputs

    raise ValueError(f"Unsupported input_builder '{adapter.input_builder}'")


def prepare_qwen_fused_inputs_embeds(
    model: Any,
    adapter: "VLMAdapter",
    processor: Any,
    image: Any,
    prompt: str,
    device: torch.device,
    qwen_prefill_empty_thinking: bool = False,
) -> Dict[str, Any]:
    """Pre-compute a Qwen vision-tower cache for reuse across patching calls.

    Running the Qwen vision tower on every patching ``generate()`` call is the
    dominant cost (measured ~20-40x per-call slowdown vs reusing a cached
    vision output). This helper runs ``get_image_features`` once; the result
    lives in the returned bundle and can be reused by every subsequent
    patching condition on the same (image, prompt) pair.

    The bundle contains the original ``inputs`` (input_ids, pixel_values,
    image_grid_thw, attention_mask, ...) plus the pre-computed vision output
    in ``cached_image_features``. Downstream calls (see
    ``_patched_generate`` in ``src.patching.run_patching``) temporarily
    monkey-patch ``model.model.get_image_features`` to return this cached
    value, which lets the standard ``input_ids`` path run end-to-end (so
    M-RoPE 3D position_ids are computed correctly) while skipping only the
    expensive vision-encoder forward.

    Only supported for ``adapter.family == "qwen"``.
    """
    if adapter.family != "qwen":
        raise ValueError(
            f"prepare_qwen_fused_inputs_embeds only supports Qwen; got '{adapter.family}'."
        )

    inputs = prepare_inputs(
        adapter,
        processor,
        image,
        prompt,
        device,
        qwen_prefill_empty_thinking=qwen_prefill_empty_thinking,
    )
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    # The multimodal wrapper (Qwen3_5ForConditionalGeneration) exposes
    # ``model`` (the inner multimodal module) which owns ``get_image_features``.
    mm_model = getattr(model, "model", model)

    cached_image_features = None
    if pixel_values is not None and image_grid_thw is not None:
        with torch.inference_mode():
            cached_image_features = mm_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True,
            )

    return {
        "inputs": inputs,
        "cached_image_features": cached_image_features,
        "prompt_len": int(inputs["input_ids"].shape[1]),
    }


def qwen_vision_cache_context(model: Any, fused_bundle: Dict[str, Any]):
    """Context manager that redirects ``model.model.get_image_features`` to
    the pre-computed value in ``fused_bundle``. Restores the original method
    on exit. Use around ``model.generate(**fused_bundle["inputs"], ...)`` to
    skip the vision tower while keeping the original input_ids path
    (preserves M-RoPE position_ids)."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        mm_model = getattr(model, "model", model)
        cached = fused_bundle.get("cached_image_features")
        if cached is None:
            yield
            return
        original = mm_model.get_image_features
        mm_model.get_image_features = lambda *a, **kw: cached
        try:
            yield
        finally:
            mm_model.get_image_features = original

    return _cm()


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
    if adapter.family == "molmo":
        return _extract_molmo_vision_features_bundle(model, inputs)

    vision_module = None
    for path in adapter.vision_attr_paths:
        vision_module = _resolve_attr_path(model, path)
        if vision_module is not None:
            break

    if vision_module is None:
        return None, None

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        pixel_values = inputs.get("images")
    if pixel_values is None:
        return None, None

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


def _extract_molmo_vision_features_bundle(
    model: Any,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[Optional[Tuple[torch.Tensor, ...]], Optional[torch.Tensor]]:
    """Extract Molmo2 ViT block states plus projected image-token features."""
    molmo_model = getattr(model, "model", None)
    if molmo_model is None or not hasattr(molmo_model, "merge_visual_inputs"):
        return None, None

    vision_backbone = getattr(molmo_model, "vision_backbone", None)
    image_vit = getattr(vision_backbone, "image_vit", None)
    if vision_backbone is None or image_vit is None:
        return None, None

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return None, None

    try:
        images, token_pooling = molmo_model.merge_visual_inputs(
            input_ids=inputs.get("input_ids"),
            pixel_values=pixel_values,
            image_token_pooling=inputs.get("image_token_pooling"),
            image_grids=inputs.get("image_grids"),
            image_num_crops=inputs.get("image_num_crops"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_token_pooling=inputs.get("video_token_pooling"),
            video_grids=inputs.get("video_grids"),
        )
    except Exception as exc:
        raise RuntimeError("Failed to build Molmo2 batched image inputs for vision features.") from exc

    if images is None or token_pooling is None:
        return None, None

    try:
        images = images.to(device=vision_backbone.device, dtype=vision_backbone.dtype)
        batch_size, num_crops, num_patches, pixels_per_patch = images.shape
        flat_images = images.reshape(batch_size * num_crops, num_patches, pixels_per_patch)
        raw_layer_states = image_vit(flat_images)
        if not raw_layer_states:
            return None, None

        hidden_states = []
        for layer_state in raw_layer_states:
            layer_state = layer_state.reshape(batch_size, num_crops * num_patches, layer_state.shape[-1])
            # Molmo ViT has no CLS token. Add a dummy slot so the shared
            # CLIP-style mean-pool helper can drop index 0 without losing a
            # real patch token.
            dummy_cls = torch.zeros(
                (batch_size, 1, layer_state.shape[-1]),
                dtype=layer_state.dtype,
                device=layer_state.device,
            )
            hidden_states.append(torch.cat([dummy_cls, layer_state], dim=1))

    except Exception as exc:
        raise RuntimeError("Failed to run Molmo2 vision backbone for hidden-state extraction.") from exc

    return tuple(hidden_states), None


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

        # Qwen and Molmo provide per-token modality labels for the prompt. When
        # we run a second forward pass on prompt + generated answer, extend
        # those labels so the new answer tokens are treated as plain text.
        if key in {"mm_token_type_ids", "token_type_ids"}:
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
    elif hasattr(model, "vision_backbone") and hasattr(model.vision_backbone, "vit_config"):
        vision_cfg = model.vision_backbone.vit_config
    elif (
        hasattr(model, "model")
        and hasattr(model.model, "vision_backbone")
        and hasattr(model.model.vision_backbone, "vit_config")
    ):
        vision_cfg = model.model.vision_backbone.vit_config

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
        "<|image|>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<img>",
        "<im_patch>",
        "<im_col>",
        "<im_end>",
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


def _load_processor(model_id: str, adapter: VLMAdapter) -> Any:
    """Load processor, working around compatibility issues for custom processors."""
    try:
        return AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=adapter.trust_remote_code,
        )
    except TypeError:
        # Some custom processors (e.g. Molmo2) pass extra kwargs to the parent
        # ProcessorMixin.__init__ that older transformers versions reject.
        # Patch ProcessorMixin.__init__ temporarily to accept and store them.
        from transformers.processing_utils import ProcessorMixin

        _orig_init = ProcessorMixin.__init__

        def _patched_init(self, *args, **kwargs):
            extra = {}
            import inspect
            sig = inspect.signature(_orig_init)
            valid_params = set(sig.parameters.keys())
            for k in list(kwargs):
                if k not in valid_params:
                    extra[k] = kwargs.pop(k)
            _orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                setattr(self, k, v)

        ProcessorMixin.__init__ = _patched_init
        try:
            return AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=adapter.trust_remote_code,
            )
        finally:
            ProcessorMixin.__init__ = _orig_init


def _ensure_default_rope_init() -> None:
    """Patch ROPE_INIT_FUNCTIONS with a 'default' entry if missing.

    Newer model code (e.g. Molmo2) expects a 'default' RoPE init function that
    was added in a later transformers release.  If the installed version lacks it,
    register a simple standard-RoPE implementation so the model can load.
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:
        return

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default_rope_parameters(config, device, seq_len=None, **kwargs):
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters


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


def _ensure_use_cache_attr(model: Any) -> None:
    """Backfill config.use_cache for remote-code models that assume it exists.

    Some custom model implementations access ``self.config.use_cache`` directly in
    ``forward`` even when the loaded config class does not define that field.
    Defaulting to ``True`` preserves normal cached decoding behavior for models
    like Molmo that expect image inputs only on the prefill step.
    """
    config_candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "language_model", None), "config", None),
    ]

    for config in config_candidates:
        if config is None or hasattr(config, "use_cache"):
            continue
        setattr(config, "use_cache", True)


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


_QWEN_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", flags=re.DOTALL)


def strip_qwen_thinking_prefix(raw_text: str) -> str:
    """Remove a leading Qwen '<think>...</think>' block that Qwen3.5 emits
    even when the chat template sets enable_thinking=False. Safe to call on
    any string — it is a no-op when no block is present."""
    if not raw_text:
        return raw_text
    return _QWEN_THINK_BLOCK_RE.sub("", raw_text, count=1).strip()


def append_qwen_empty_thinking_preamble(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Append ``<think>\n</think>\n\n`` before Qwen generation.

    With Qwen's empty thinking block prefilled in the prompt, generated token
    step 0 corresponds to the first visible answer token instead of the
    ``<think>`` delimiter. Sequence-shaped masks/type ids are extended so the
    model treats the preamble as normal text.
    """
    input_ids = batch.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return batch

    preamble = torch.tensor(
        [_QWEN_EMPTY_THINKING_PREAMBLE_IDS],
        device=device,
        dtype=input_ids.dtype,
    )
    preamble_len = int(preamble.shape[1])
    seq_len = int(input_ids.shape[1])

    updated: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            updated[key] = value
            continue

        if key == "input_ids":
            updated[key] = torch.cat([value, preamble], dim=1)
            continue

        if value.ndim >= 2 and value.shape[0] == input_ids.shape[0] and value.shape[1] == seq_len:
            if key == "attention_mask":
                pad_value = 1
            else:
                pad_value = 0
            pad = torch.full(
                (value.shape[0], preamble_len, *value.shape[2:]),
                pad_value,
                device=value.device,
                dtype=value.dtype,
            )
            updated[key] = torch.cat([value, pad], dim=1)
        else:
            updated[key] = value

    return updated


def qwen_patching_max_new_tokens(
    adapter: "VLMAdapter", dataset_default: int
) -> int:
    """Qwen needs extra budget to get past the '<think></think>' preamble
    before emitting the real answer. Use at least 20 tokens for Qwen; leave
    other families unchanged."""
    if adapter.family == "qwen":
        return max(dataset_default, 20)
    return dataset_default


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
