from __future__ import annotations

from typing import Any, Dict, List, Optional
from tqdm import tqdm


def prepare_default_split(split_obj, seed_offset: int, max_samples: Optional[int]) -> List[Dict]:
    """
    Minimal preparation: just wrap raw rows with an idx.
    Use this for datasets where you don't need custom prompting.
    """
    if max_samples is not None:
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    return [{"idx": i, **split_obj[i]} for i in range(len(split_obj))]


def prepare_vqa_v2_split(split_obj, seed_offset: int, max_samples: Optional[int]) -> List[Dict]:
    """
    VQAv2 preparation — copied from your existing loader.
    """
    if max_samples is not None:
        split_obj = split_obj.shuffle(seed=42 + seed_offset)
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    samples: List[Dict] = []
    for idx, row in enumerate(tqdm(split_obj)):

        raw_answers = row.get("answers", [])
        if isinstance(raw_answers, list) and raw_answers and isinstance(raw_answers[0], dict):
            all_answers = [a.get("answer", "") for a in raw_answers]
        elif isinstance(raw_answers, list):
            all_answers = [str(a) for a in raw_answers]
        else:
            all_answers = []

        prompt = f"Question: {row.get('question','')}\nProvide a short answer."

        samples.append(
            {
                "idx": idx,
                "image": row.get("image"),
                "question": row.get("question", ""),
                "prompt": prompt,
                "gt_answers": all_answers,
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
                "dataset_id": "vqa-v2",
            }
        )
    return samples


def prepare_coco_qa_vi_split(
    split_obj,
    seed_offset: int,
    max_samples: Optional[int],
    use_translated_text: bool = False,
) -> List[Dict]:
    """
    Prepare COCO-QA Vietnamese into the same sample schema used by the VQA pipeline.

    By default we use the non-translated English fields. Set use_translated_text=True
    to use the Vietnamese translations instead. 
    """
    if max_samples is not None:
        split_obj = split_obj.shuffle(seed=42 + seed_offset)
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    question_key = "translated_question" if use_translated_text else "question"
    answer_key = "translated_answer" if use_translated_text else "answer"

    samples: List[Dict] = []
    for idx, row in enumerate(tqdm(split_obj)):
        question = row.get(question_key) or row.get("question", "")
        gt_answer = row.get(answer_key) or row.get("answer", "")

        prompt = f"Question: {question}\nProvide a short answer."

        samples.append(
            {
                "idx": idx,
                "image": row.get("image"),
                "question": question,
                "prompt": prompt,
                "gt_answer": gt_answer,
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
                "question_type": row.get("type"),
                "dataset_id": "coco-qa-vi",
                "source_question": row.get("question", ""),
                "source_answer": row.get("answer", ""),
                "split": row.get("split"),
            }
        )
    return samples


def prepare_pope_split(
    split_obj,
    seed_offset: int,
    max_samples: Optional[int],
) -> List[Dict]:
    """
    Prepare POPE into the shared sample schema used by the VQA-style pipelines.

    The HF dataset schema has varied a bit across snapshots, so we resolve a few
    likely field names defensively.
    """
    if max_samples is not None:
        split_obj = split_obj.shuffle(seed=42 + seed_offset)
        split_obj = split_obj.select(range(min(max_samples, len(split_obj))))

    samples: List[Dict] = []
    for idx, row in enumerate(tqdm(split_obj)):
        question = (
            row.get("question")
            or row.get("text")
            or row.get("query")
            or ""
        )

        gt_answer = (
            row.get("answer")
            or row.get("label")
            or row.get("gt_answer")
            or ""
        )
        if gt_answer is None:
            gt_answer = ""
        gt_answer = str(gt_answer).strip().lower()

        prompt = f"Question: {question}\nAnswer with exactly one word: yes or no."

        sample_id = (
            row.get("sample_id")
            or row.get("id")
            or row.get("question_id")
            or row.get("image_id")
        )

        samples.append(
            {
                "idx": idx,
                "image": row.get("image"),
                "question": question,
                "prompt": prompt,
                "gt_answer": gt_answer,
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
                "sample_id": sample_id,
                "category": row.get("category"),
                "statement": row.get("statement") or question,
                "dataset_id": "pope",
            }
        )
    return samples
