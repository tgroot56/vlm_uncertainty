"""Generate 2x4 positional-sweep figure: rows=models, cols=datasets.

Each cell shows answer-confidence recovery and accuracy recovery overlaid.
LLaVA uses extreme_answer_confidence parquets; Qwen uses the
extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking runs, including
POPE.

Output: filtered_plots/cross_model/sample_cutoff/two_metrics_positional_sweep_nofilter_100_alltokenspans.pdf/.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import plot_two_metrics_cross_model_positional_sweep

_ROOT = Path("/projects/prjs2014/patching")
CROSS_OUT = _ROOT / "filtered_plots/cross_model/sample_cutoff"

LLAVA_PS = _ROOT / "positional_patching_results_including_visual/llava-hf_llava-1-5-7b-hf"
QWEN_PS = _ROOT / "positional_patching_results_including_visual/Qwen_Qwen3-5-9B"

DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]

LLAVA_SUBDIR = "extreme_answer_confidence"
QWEN_SUBDIR = "extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking"

llava_paths = {
    ds: str(LLAVA_PS / (Path("pope/random") if ds == "pope" else Path(ds)) / LLAVA_SUBDIR / "positional_patching_results.parquet")
    for ds in DATASETS
}
qwen_paths = {
    ds: str(QWEN_PS / (Path("pope/random") if ds == "pope" else Path(ds)) / QWEN_SUBDIR / "positional_patching_results.parquet")
    for ds in DATASETS
}

model_entries = [
    ("LLaVA-1.5-7B", llava_paths),
    ("Qwen3.5-9B", qwen_paths),
]

print("Generating two-metrics positional-sweep figure (no filter, unclipped recovery) ...")
for label, paths in model_entries:
    for ds, p in paths.items():
        exists = Path(p).exists()
        print(f"  {label:15s} {ds:12s}: {'OK' if exists else 'MISSING'} - {p}")

plot_two_metrics_cross_model_positional_sweep(
    model_entries=model_entries,
    datasets=DATASETS,
    output_dir=CROSS_OUT,
    output_name="two_metrics_positional_sweep_nofilter_100_alltokenspans_unclipped",
    min_delta=0.0,
    min_pos_samples=100,
    append_nofilter_suffix=False,
    clip_recovery=False,
)

print("\nDone.")
