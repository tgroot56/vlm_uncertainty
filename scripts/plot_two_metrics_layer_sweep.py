"""Generate 2×4 layer-sweep figure: rows=models, cols=datasets, two metrics per cell.

Each cell shows answer-confidence recovery and accuracy recovery overlaid.
LLaVA uses extreme_answer_confidence parquets; Qwen uses
extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking.

Output: filtered_plots/cross_model/sample_cutoff/two_metrics_layer_sweep_nofilter.pdf/.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.plot_sweep_filtered import plot_two_metrics_cross_model_layer_sweep

_ROOT = Path("/projects/prjs2014/patching")
CROSS_OUT = _ROOT / "filtered_plots/cross_model/sample_cutoff"

LLAVA_LS = _ROOT / "layer_sweep_results/llava-hf_llava-1-5-7b-hf"
QWEN_LS  = _ROOT / "layer_sweep_results/Qwen_Qwen3-5-9B"

DATASETS = ["vqa-v2", "coco-qa-vi", "imagenet-r", "pope"]

LLAVA_SUBDIR = "extreme_answer_confidence"

QWEN_SUBDIR_DEFAULT  = "extreme_answer_confidence_finaltoken_qwen_prefill_empty_thinking"

QWEN_SUBDIRS = {
    "vqa-v2":     QWEN_SUBDIR_DEFAULT,
    "coco-qa-vi": QWEN_SUBDIR_DEFAULT,
    "imagenet-r": QWEN_SUBDIR_DEFAULT,
    "pope":       f"random/{QWEN_SUBDIR_DEFAULT}",
}

llava_paths = {
    ds: str(LLAVA_LS / (Path("pope/random") if ds == "pope" else Path(ds)) / LLAVA_SUBDIR / "layer_sweep_results.parquet")
    for ds in DATASETS
}
qwen_paths = {
    ds: str(QWEN_LS / ds / QWEN_SUBDIRS[ds] / "layer_sweep_results.parquet")
    for ds in DATASETS
}

model_entries = [
    ("LLaVA-1.5-7B", llava_paths),
    ("Qwen3.5-9B",   qwen_paths),
]

print("Generating two-metrics layer-sweep figure (no filter) ...")
for label, paths in model_entries:
    for ds, p in paths.items():
        exists = Path(p).exists()
        print(f"  {label:15s} {ds:12s}: {'OK' if exists else 'MISSING'} — {p}")

plot_two_metrics_cross_model_layer_sweep(
    model_entries=model_entries,
    datasets=DATASETS,
    output_dir=CROSS_OUT,
    output_name="two_metrics_layer_sweep",
    min_delta=0.0,
)

print("\nDone.")
