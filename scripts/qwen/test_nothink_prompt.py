from __future__ import annotations

import argparse

import torch

from src.data.load import load_dataset_prepared
from src.vlm import load_model, prepare_inputs, strip_qwen_thinking_prefix
from utils.image_resolver import resolve_image


def build_prompt_variants(base_prompt: str) -> dict[str, str]:
    question_prefixed = base_prompt.replace("Question:", "Question: /no_think", 1)
    backslash_question_prefixed = base_prompt.replace("Question:", r"Question: \no_think", 1)

    return {
        "plain": base_prompt,
        "prefix_slash_no_think": f"/no_think\n{base_prompt}",
        "prefix_backslash_no_think": f"\\no_think\n{base_prompt}",
        "question_slash_no_think": question_prefixed,
        "question_backslash_no_think": backslash_question_prefixed,
        "suffix_space_no_think": f"{base_prompt} /no_think",
        "suffix_newline_no_think": f"{base_prompt}\n/no_think",
        "suffix_space_backslash_no_think": f"{base_prompt} \\no_think",
        "suffix_newline_backslash_no_think": f"{base_prompt}\n\\no_think",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen no-thinking prompt variants.")
    parser.add_argument("--dataset", default="coco-qa-vi")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--vlm", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    model, processor, device, adapter = load_model(args.vlm, args.device)
    samples = load_dataset_prepared(
        dataset_id=args.dataset,
        seed_offset=42,
        use_cache=True,
        max_samples=args.sample_index + 1,
    )
    sample = samples[args.sample_index]
    image = resolve_image(sample)
    base_prompt = sample["prompt"]

    print("dataset:", args.dataset)
    print("sample_index:", args.sample_index)
    print("question_id:", sample.get("question_id"))
    print("image_id:", sample.get("image_id"))
    print("gt_answer:", repr(sample.get("gt_answer")))
    print("base_prompt:", repr(base_prompt))
    print("max_new_tokens:", args.max_new_tokens)
    print("=" * 80)

    for name, prompt in build_prompt_variants(base_prompt).items():
        inputs = prepare_inputs(adapter, processor, image, prompt, device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[:, prompt_len:]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        raw_with_specials = processor.batch_decode(gen_ids, skip_special_tokens=False)[0].strip()
        stripped = strip_qwen_thinking_prefix(raw)

        print("variant:", name)
        print("prompt_suffix:", repr(prompt[len(base_prompt):]))
        print("generated_ids:", gen_ids[0].tolist())
        print("raw_skip_special:", repr(raw))
        print("raw_with_specials:", repr(raw_with_specials))
        print("stripped:", repr(stripped))
        print("-" * 80)


if __name__ == "__main__":
    main()
