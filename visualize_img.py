from pathlib import Path
import json
from collections import Counter
import string

import torch

from src.data import load_dataset_prepared


SUP_DS_PATH = (
    "outputs/supervised_datasets/"
    "imagenet-r/llava-hf/llava-1.5-7b-hf/"
    "run_b8e74dc670/supervision_dataset.pt"
)


def _to_jsonable(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _analyze_pred_keys(rows):
    preds = []
    for r in rows:
        p = r.get("pred_letter", None)
        if p is None:
            continue
        preds.append(str(p).strip())

    c = Counter(preds)
    total = len(rows)
    present = len(preds)
    unique = len(c)

    print("\n" + "=" * 100)
    print("PREDICTION KEY ANALYSIS (from supervision_dataset.pt rows)")
    print(f"rows total: {total}")
    print(f"pred_letter present: {present}")
    print(f"unique pred_letter values: {unique}")

    print("\nTop-20 most common predicted keys:")
    for k, v in c.most_common(20):
        print(f"  {k}: {v}")

    # Check for keys beyond A-D
    beyond_ad_single = sorted(
        [k for k in c.keys() if len(k) == 1 and k in string.ascii_uppercase and k > "D"]
    )
    multi_letter = sorted([k for k in c.keys() if len(k) >= 2])

    only_ad = set(c.keys()).issubset({"A", "B", "C", "D"})

    print("\nDoes pred_letter contain ONLY A-D?", only_ad)
    print("Any single-letter predictions beyond D?", len(beyond_ad_single) > 0)
    if beyond_ad_single:
        print("  Beyond-D single-letter keys seen:", beyond_ad_single)

    print("Any multi-letter predictions (AA, AB, ...)?", len(multi_letter) > 0)
    if multi_letter:
        # print a few
        preview = multi_letter[:30]
        print("  Multi-letter keys seen (preview):", preview)
        if len(multi_letter) > len(preview):
            print(f"  ... (+{len(multi_letter) - len(preview)} more)")

    # Heuristic: raw_text indicates E/F/... but pred_letter is A-D
    suspicious = []
    for i, r in enumerate(rows):
        rt = r.get("raw_text", "") or ""
        p = str(r.get("pred_letter", "")).strip()
        if p in {"A", "B", "C", "D"}:
            # look for patterns that strongly suggest other letters were produced
            if any(tok in rt for tok in ["Answer: E", "Answer: F", "Answer: G", "Answer: H", "E.", "F.", "G.", "H."]):
                suspicious.append((i, p, rt.replace("\n", " ")[:220]))
        if len(suspicious) >= 10:
            break

    if suspicious:
        print("\nSuspicious cases (raw_text suggests E+ but pred_letter is A-D):")
        for i, p, rt in suspicious:
            print(f"  row_idx={i} pred_letter={p} raw_text='{rt}...'")
    else:
        print("\nNo suspicious raw_text cases found with the simple heuristic.")

    print("=" * 100 + "\n")


def main():
    out_dir = Path("quick_tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load saved supervision dataset (for predictions) ----
    data = torch.load(SUP_DS_PATH, map_location="cpu")
    rows = data["rows"]

    print(f"Loaded supervision dataset rows: {len(rows)}")
    _analyze_pred_keys(rows)

    # ---- Load prepared ImageNet-R (same order as generation) ----
    samples = load_dataset_prepared(
        dataset_id="imagenet-r",
        split="test",
        seed_offset=42,
        max_samples=5,
        use_cache=True,
        dataset_kwargs={
            "k_options": 16,
            "use_clip_hard_negatives": False,
        },
    )

    print(f"Loaded {len(samples)} ImageNet-R samples for inspection")

    # ---- Inspect first 5 samples ----
    for i, (sample, row) in enumerate(zip(samples, rows)):
        print("\n" + "=" * 100)
        print(f"SAMPLE {i}")
        print("- PROMPT:")
        print(sample["prompt"])

        pred_key = row.get("pred_letter")
        gt_key = sample.get("gt_key")

        print("\n- MODEL OUTPUT:")
        print(f"  predicted_key : {pred_key}")
        print(f"  gt_key        : {gt_key}")
        print(f"  gt_class      : {sample.get('gt_class')}")
        print(f"  correct       : {str(pred_key).strip() == str(gt_key).strip()}")

        opt_probs = row.get("option_probs", None)
        if opt_probs is not None:
            print("\n- OPTION PROBS:")
            if torch.is_tensor(opt_probs):
                opt_probs_list = opt_probs.detach().cpu().tolist()
            else:
                opt_probs_list = opt_probs

            if not isinstance(opt_probs_list, (list, tuple)):
                print(f"  (raw) {opt_probs_list}")
            else:
                try:
                    for k, v in zip(sample["option_map"].keys(), opt_probs_list):
                        print(f"  {k}: {float(v):.4f}")
                except Exception:
                    print(f"  (raw) {opt_probs_list}")

        raw_text = row.get("raw_text", None)
        if raw_text is not None:
            print("\n- RAW MODEL TEXT:")
            print(raw_text)

        # Save image
        img_path = out_dir / f"imagenet_r_{i}.png"
        image = sample.get("image")
        if image is not None:
            image.save(img_path)
        else:
            print(f"WARNING: sample {i} has no image")

        # Save JSON with prompt + prediction
        meta = {
            "idx": sample.get("idx"),
            "prompt": sample.get("prompt"),
            "option_map": sample.get("option_map"),
            "gt_class": sample.get("gt_class"),
            "gt_key": gt_key,
            "pred_key": pred_key,
            "correct": str(pred_key).strip() == str(gt_key).strip(),
            "option_probs": _to_jsonable(opt_probs),
            "raw_text": raw_text,
        }

        json_path = out_dir / f"imagenet_r_{i}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"\nSaved image -> {img_path}")
        print(f"Saved JSON  -> {json_path}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
