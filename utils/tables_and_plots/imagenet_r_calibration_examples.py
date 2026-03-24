from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RESULTS_ROOT = Path("probe_results/llava/imagenet-r/16mc-clip")
DEFAULT_DATA_GLOB = Path(
    "/scratch-shared/tgroot/supervised_datasets/imagenet_r_16mc/imagenet-r/llava-hf/llava-1.5-7b-hf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show qualitative calibration improvements for LLaVA / ImageNet-R by comparing "
            "the raw model confidence with the trained MLP probe confidence."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root results directory containing results_brier.csv and the mlp feature folders.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Specific MLP feature folder to analyze. Defaults to the best MLP from results_brier.csv.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional explicit supervision_dataset.pt path. If omitted, the script tries config.json first.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to show for each case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the qualitative report. Defaults to <results-root>/qualitative_examples.",
    )
    return parser.parse_args()


def load_best_mlp_feature(results_root: Path) -> str:
    csv_path = results_root / "results_brier.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find results_brier.csv at: {csv_path}")

    best_feature = None
    best_score = None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mlp_value = row.get("mlp", "")
            try:
                mlp_score = float(mlp_value)
            except (TypeError, ValueError):
                continue
            if best_score is None or mlp_score < best_score:
                best_score = mlp_score
                best_feature = row["feature_label"]

    if best_feature is None:
        raise RuntimeError(f"Could not infer best MLP feature from: {csv_path}")
    return best_feature


def _format_prob(value: float) -> str:
    return f"{float(value):.4f}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_probe_dir(results_root: Path, feature: str) -> Path:
    probe_dir = results_root / "mlp" / feature
    if not probe_dir.exists():
        raise FileNotFoundError(f"Could not find MLP probe directory at: {probe_dir}")
    return probe_dir


def candidate_data_paths(config_data_path: str, override_data_path: Optional[Path]) -> Iterable[Path]:
    seen = set()

    def _yield(path: Optional[Path]) -> Iterable[Path]:
        if path is None:
            return
        resolved = path.expanduser()
        if resolved in seen:
            return
        seen.add(resolved)
        yield resolved

    if override_data_path is not None:
        yield from _yield(override_data_path)

    if config_data_path:
        yield from _yield(Path(config_data_path))

    if DEFAULT_DATA_GLOB.exists():
        for path in sorted(DEFAULT_DATA_GLOB.glob("run_*/supervision_dataset.pt")):
            yield from _yield(path)


def build_test_indices(num_samples: int, train_split: float, val_split: float, seed: int) -> List[int]:
    num_train = int(num_samples * train_split)
    num_val = int(num_samples * val_split)
    num_test = num_samples - num_train - num_val
    generator = torch.Generator().manual_seed(seed)
    _, _, test_subset = torch.utils.data.random_split(
        list(range(num_samples)),
        [num_train, num_val, num_test],
        generator=generator,
    )
    return list(test_subset.indices)


def count_label_mismatches(test_labels: List[float], probe_labels: List[float]) -> int:
    return sum(1 for a, b in zip(test_labels, probe_labels) if abs(float(a) - float(b)) >= 1e-6)


def load_matching_supervision_dataset(
    *,
    config: Dict[str, Any],
    probe_labels: List[float],
    override_data_path: Optional[Path],
) -> Tuple[Path, Dict[str, Any], List[int], int]:
    config_data_path = str(config.get("data_path", "")).strip()
    best_candidate: Optional[Tuple[Path, Dict[str, Any], List[int], int]] = None

    for path in candidate_data_paths(config_data_path, override_data_path):
        if not path.exists():
            continue

        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = payload.get("rows")
        y = payload.get("y")
        if rows is None or y is None:
            continue

        test_indices = build_test_indices(
            num_samples=len(rows),
            train_split=float(config["train_split"]),
            val_split=float(config["val_split"]),
            seed=int(config["seed"]),
        )
        if len(test_indices) != len(probe_labels):
            continue

        test_labels = payload["y"][test_indices].float().tolist()
        mismatches = count_label_mismatches(test_labels, probe_labels)
        candidate = (path, payload, test_indices, mismatches)
        if best_candidate is None or mismatches < best_candidate[3]:
            best_candidate = candidate
        if mismatches == 0:
            return candidate

    if best_candidate is None:
        raise FileNotFoundError(
            "Could not find a supervision_dataset.pt whose reconstructed test split matches the "
            "stored probe labels. Pass --data-path explicitly if needed."
        )

    allowed_mismatches = max(5, int(0.001 * len(probe_labels)))
    if best_candidate[3] > allowed_mismatches:
        raise FileNotFoundError(
            "Found supervision_dataset.pt candidates, but none matched the stored probe labels "
            f"closely enough. Best candidate had {best_candidate[3]} mismatches out of "
            f"{len(probe_labels)} test labels. Pass --data-path explicitly if needed."
        )
    return best_candidate


def top_option_probs(option_probs: Dict[str, float], top_k: int = 3) -> List[Dict[str, Any]]:
    ranked = sorted(option_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"key": key, "prob": float(prob)} for key, prob in ranked]


def raw_model_confidence(row: Dict[str, Any]) -> float:
    option_probs = row.get("option_probs") or {}
    pred_key = row.get("pred_key")
    if pred_key is None:
        return float("nan")
    return float(option_probs.get(pred_key, float("nan")))


def gt_confidence(row: Dict[str, Any]) -> float:
    option_probs = row.get("option_probs") or {}
    gt_key = row.get("gt_key")
    if gt_key is None:
        return float("nan")
    return float(option_probs.get(gt_key, float("nan")))


def format_example(
    *,
    rank: int,
    test_position: int,
    dataset_index: int,
    row: Dict[str, Any],
    probe_confidence: float,
) -> Dict[str, Any]:
    baseline_conf = raw_model_confidence(row)
    return {
        "rank": rank,
        "test_position": test_position,
        "dataset_index": dataset_index,
        "ground_truth_class": row.get("gt_class"),
        "ground_truth_key": row.get("gt_key"),
        "predicted_key": row.get("pred_key"),
        "raw_generation": row.get("raw_text"),
        "raw_model_confidence": baseline_conf,
        "ground_truth_option_confidence": gt_confidence(row),
        "probe_confidence": float(probe_confidence),
        "confidence_delta": float(probe_confidence - baseline_conf),
        "top_option_probs": top_option_probs(row.get("option_probs") or {}),
    }


def collect_examples(
    *,
    rows: List[Dict[str, Any]],
    test_indices: List[int],
    probe_predictions: List[float],
    probe_labels: List[float],
    num_examples: int,
) -> Dict[str, List[Dict[str, Any]]]:
    incorrect_high_conf: List[tuple[float, float, Dict[str, Any]]] = []
    correct_low_conf: List[tuple[float, float, Dict[str, Any]]] = []

    for test_position, (dataset_index, probe_conf, label) in enumerate(
        zip(test_indices, probe_predictions, probe_labels)
    ):
        row = rows[dataset_index]
        baseline_conf = raw_model_confidence(row)
        if not torch.isfinite(torch.tensor(baseline_conf)):
            continue

        example = format_example(
            rank=0,
            test_position=test_position,
            dataset_index=dataset_index,
            row=row,
            probe_confidence=float(probe_conf),
        )

        if float(label) < 0.5 and probe_conf < baseline_conf:
            incorrect_high_conf.append((baseline_conf, baseline_conf - probe_conf, example))
        if float(label) >= 0.5 and probe_conf > baseline_conf:
            correct_low_conf.append((baseline_conf, probe_conf - baseline_conf, example))

    incorrect_high_conf.sort(key=lambda item: (-item[0], -item[1]))
    correct_low_conf.sort(key=lambda item: (item[0], -item[1]))

    incorrect_examples = []
    for rank, (_, _, example) in enumerate(incorrect_high_conf[:num_examples], start=1):
        example["rank"] = rank
        incorrect_examples.append(example)

    correct_examples = []
    for rank, (_, _, example) in enumerate(correct_low_conf[:num_examples], start=1):
        example["rank"] = rank
        correct_examples.append(example)

    return {
        "incorrect_high_confidence_now_calibrated": incorrect_examples,
        "correct_low_confidence_now_calibrated": correct_examples,
    }


def load_imagenet_r_split(num_rows: int):
    from src.data.hf import load_hf_dataset
    from src.data.registry import get_dataset_spec

    spec = get_dataset_spec("imagenet-r")
    dataset = load_hf_dataset(spec.hf_name, spec.hf_config)
    split_obj = dataset[spec.default_split]
    if num_rows < len(split_obj):
        split_obj = split_obj.select(range(num_rows))
    return split_obj


def image_to_data_uri(image: Any, max_side: int = 320) -> Optional[str]:
    if image is None:
        return None

    pil_image = image.copy() if hasattr(image, "copy") else image
    if not hasattr(pil_image, "save"):
        return None

    if getattr(pil_image, "mode", "RGB") not in {"RGB", "L"}:
        pil_image = pil_image.convert("RGB")

    if hasattr(pil_image, "thumbnail"):
        pil_image.thumbnail((max_side, max_side))

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def attach_images_to_examples(
    examples: Dict[str, List[Dict[str, Any]]],
    split_obj: Any,
) -> None:
    for case_examples in examples.values():
        for example in case_examples:
            dataset_index = int(example["dataset_index"])
            raw_row = split_obj[dataset_index]
            image_data_uri = image_to_data_uri(raw_row.get("image"))
            example["image_data_uri"] = image_data_uri
            example["dataset_class_name"] = raw_row.get("class_name")


def render_html_example_card(example: Dict[str, Any]) -> str:
    image_html = '<div class="image-placeholder">Image unavailable</div>'
    if example.get("image_data_uri"):
        image_html = (
            f'<img class="example-image" src="{example["image_data_uri"]}" '
            f'alt="ImageNet-R example {int(example["dataset_index"])}">'
        )

    top_options = "".join(
        "<li>"
        f"{html.escape(str(opt['key']))}: <strong>{_format_prob(opt['prob'])}</strong>"
        "</li>"
        for opt in example.get("top_option_probs", [])
    )

    return f"""
    <article class="example-card">
      <div class="image-wrap">
        {image_html}
      </div>
      <div class="example-body">
        <h3>#{int(example['rank'])} · dataset_idx={int(example['dataset_index'])}</h3>
        <p class="meta"><strong>GT class:</strong> {html.escape(str(example.get('ground_truth_class')))}</p>
        <p class="meta"><strong>GT key:</strong> {html.escape(str(example.get('ground_truth_key')))} · <strong>Pred key:</strong> {html.escape(str(example.get('predicted_key')))}</p>
        <p class="meta"><strong>Raw confidence:</strong> {_format_prob(example['raw_model_confidence'])} · <strong>Probe confidence:</strong> {_format_prob(example['probe_confidence'])} · <strong>Delta:</strong> {_format_prob(example['confidence_delta'])}</p>
        <p class="meta"><strong>GT option confidence:</strong> {_format_prob(example['ground_truth_option_confidence'])}</p>
        <p class="meta"><strong>Raw generation:</strong> <code>{html.escape(str(example.get('raw_generation')))}</code></p>
        <div class="options">
          <strong>Top option probabilities</strong>
          <ul>{top_options}</ul>
        </div>
      </div>
    </article>
    """


def render_html_section(title: str, subtitle: str, examples: List[Dict[str, Any]]) -> str:
    cards = "\n".join(render_html_example_card(example) for example in examples)
    return f"""
    <section class="case-section">
      <div class="section-head">
        <h2>{html.escape(title)}</h2>
        <p>{html.escape(subtitle)}</p>
      </div>
      <div class="card-grid">
        {cards}
      </div>
    </section>
    """


def render_html_report(
    *,
    feature: str,
    probe_dir: Path,
    data_path: Path,
    label_mismatches: int,
    examples: Dict[str, List[Dict[str, Any]]],
) -> str:
    case_1 = render_html_section(
        "Incorrect, overconfident raw predictions",
        "Raw LLaVA was wrong but confident; the trained MLP lowered the correctness confidence.",
        examples["incorrect_high_confidence_now_calibrated"],
    )
    case_2 = render_html_section(
        "Correct, underconfident raw predictions",
        "Raw LLaVA was correct but hesitant; the trained MLP raised the correctness confidence.",
        examples["correct_low_confidence_now_calibrated"],
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ImageNet-R calibration examples · {html.escape(feature)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f2ea;
      --panel: #fffdf8;
      --ink: #1f1a16;
      --muted: #6b6258;
      --line: #d7cec1;
      --accent: #9a3412;
      --accent-soft: #fde9dd;
      --shadow: 0 10px 30px rgba(64, 45, 28, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(154, 52, 18, 0.10), transparent 28rem),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1400px, calc(100vw - 3rem));
      margin: 0 auto;
      padding: 2.5rem 0 4rem;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 2rem;
      box-shadow: var(--shadow);
      margin-bottom: 2rem;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.15; }}
    h1 {{ font-size: clamp(2rem, 4vw, 3.3rem); margin-bottom: 0.75rem; }}
    h2 {{ font-size: clamp(1.4rem, 2.4vw, 2rem); margin-bottom: 0.35rem; }}
    h3 {{ font-size: 1.2rem; margin-bottom: 0.75rem; }}
    p {{ margin: 0; }}
    .hero p, .section-head p {{ color: var(--muted); margin-top: 0.4rem; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 0.9rem;
      margin-top: 1.4rem;
    }}
    .meta-chip {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 0.9rem 1rem;
      background: #fffaf4;
    }}
    .meta-chip strong {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
      margin-bottom: 0.25rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .case-section + .case-section {{ margin-top: 2rem; }}
    .section-head {{ margin-bottom: 1rem; }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 1rem;
    }}
    .example-card {{
      display: grid;
      grid-template-columns: minmax(220px, 300px) 1fr;
      gap: 1rem;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 1rem;
      box-shadow: var(--shadow);
      align-items: start;
    }}
    .image-wrap {{
      border-radius: 16px;
      overflow: hidden;
      background: #efe7da;
      min-height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .example-image {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .image-placeholder {{
      color: var(--muted);
      padding: 1rem;
      text-align: center;
    }}
    .example-body {{
      display: grid;
      gap: 0.45rem;
    }}
    .meta code {{
      background: var(--accent-soft);
      padding: 0.1rem 0.35rem;
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .options {{
      margin-top: 0.35rem;
      padding: 0.8rem 0.9rem;
      border-radius: 14px;
      background: #fff8f1;
      border: 1px solid #eadbc5;
    }}
    .options ul {{
      margin: 0.5rem 0 0;
      padding-left: 1.25rem;
    }}
    .options li + li {{ margin-top: 0.15rem; }}
    @media (max-width: 900px) {{
      .card-grid {{
        grid-template-columns: 1fr;
      }}
      .example-card {{
        grid-template-columns: 1fr;
      }}
      .image-wrap {{
        min-height: 260px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>ImageNet-R calibration examples</h1>
      <p>Qualitative comparison of raw LLaVA confidence versus the trained MLP correctness probe for feature <strong>{html.escape(feature)}</strong>.</p>
      <div class="meta-grid">
        <div class="meta-chip"><strong>Probe directory</strong>{html.escape(str(probe_dir))}</div>
        <div class="meta-chip"><strong>Supervision dataset</strong>{html.escape(str(data_path))}</div>
        <div class="meta-chip"><strong>Saved split mismatches</strong>{int(label_mismatches)}</div>
        <div class="meta-chip"><strong>Examples per case</strong>{len(examples["incorrect_high_confidence_now_calibrated"])}</div>
      </div>
    </section>
    {case_1}
    {case_2}
  </main>
</body>
</html>
"""


def render_text_report(
    *,
    feature: str,
    probe_dir: Path,
    data_path: Path,
    label_mismatches: int,
    examples: Dict[str, List[Dict[str, Any]]],
) -> str:
    lines = [
        f"Feature: {feature}",
        f"Probe directory: {probe_dir}",
        f"Supervision data: {data_path}",
        f"Test-label mismatches vs saved probe split: {label_mismatches}",
        "",
        "Case 1: Incorrect raw predictions with high confidence, but lower calibrated probe confidence",
    ]

    for ex in examples["incorrect_high_confidence_now_calibrated"]:
        lines.extend(
            [
                (
                    f"{ex['rank']}. dataset_idx={ex['dataset_index']} gt_class={ex['ground_truth_class']} "
                    f"pred_key={ex['predicted_key']} raw_conf={ex['raw_model_confidence']:.4f} "
                    f"probe_conf={ex['probe_confidence']:.4f} delta={ex['confidence_delta']:.4f}"
                ),
                f"   top options: {ex['top_option_probs']}",
            ]
        )

    lines.extend(
        [
            "",
            "Case 2: Correct raw predictions with low confidence, but higher calibrated probe confidence",
        ]
    )

    for ex in examples["correct_low_confidence_now_calibrated"]:
        lines.extend(
            [
                (
                    f"{ex['rank']}. dataset_idx={ex['dataset_index']} gt_class={ex['ground_truth_class']} "
                    f"pred_key={ex['predicted_key']} raw_conf={ex['raw_model_confidence']:.4f} "
                    f"probe_conf={ex['probe_confidence']:.4f} delta={ex['confidence_delta']:.4f}"
                ),
                f"   top options: {ex['top_option_probs']}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def strip_image_blobs(
    examples: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    stripped: Dict[str, List[Dict[str, Any]]] = {}
    for case_name, case_examples in examples.items():
        stripped[case_name] = []
        for example in case_examples:
            stripped[case_name].append(
                {key: value for key, value in example.items() if key != "image_data_uri"}
            )
    return stripped


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    feature = args.feature or load_best_mlp_feature(results_root)
    probe_dir = resolve_probe_dir(results_root, feature)

    config = load_json(probe_dir / "config.json")
    predictions_payload = load_json(probe_dir / "test_predictions.json")
    probe_predictions = [float(x) for x in predictions_payload["predictions"]]
    probe_labels = [float(x) for x in predictions_payload["labels"]]

    data_path, supervision_payload, test_indices, label_mismatches = load_matching_supervision_dataset(
        config=config,
        probe_labels=probe_labels,
        override_data_path=args.data_path,
    )

    examples = collect_examples(
        rows=supervision_payload["rows"],
        test_indices=test_indices,
        probe_predictions=probe_predictions,
        probe_labels=probe_labels,
        num_examples=args.num_examples,
    )
    attach_images_to_examples(examples, load_imagenet_r_split(len(supervision_payload["rows"])))

    output_dir = args.output_dir or (results_root / "qualitative_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "feature": feature,
        "probe_dir": str(probe_dir),
        "data_path": str(data_path),
        "test_label_mismatches": int(label_mismatches),
        "num_examples_requested": int(args.num_examples),
        "cases": strip_image_blobs(examples),
    }

    json_path = output_dir / f"{feature}__calibration_examples.json"
    txt_path = output_dir / f"{feature}__calibration_examples.txt"
    html_path = output_dir / f"{feature}__calibration_examples.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    text_report = render_text_report(
        feature=feature,
        probe_dir=probe_dir,
        data_path=data_path,
        label_mismatches=label_mismatches,
        examples=examples,
    )
    txt_path.write_text(text_report, encoding="utf-8")
    html_report = render_html_report(
        feature=feature,
        probe_dir=probe_dir,
        data_path=data_path,
        label_mismatches=label_mismatches,
        examples=examples,
    )
    html_path.write_text(html_report, encoding="utf-8")

    print(text_report)
    print(f"[saved] {json_path}")
    print(f"[saved] {txt_path}")
    print(f"[saved] {html_path}")


if __name__ == "__main__":
    main()
