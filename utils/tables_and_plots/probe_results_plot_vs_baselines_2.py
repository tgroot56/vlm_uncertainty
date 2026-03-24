from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# -----------------------
# CONFIG & PATHS
# -----------------------
PLACEHOLDER_NO_SKILL_BRIER = np.nan

DATASET_CONFIGS = {
    "llava/coco-qa-vi": {
        "true": Path("probe_results/llava/coco-qa-vi"),
        "shuffle": Path("probe_results/llava/coco-qa-vi/baselines"),
        "no_skill_brier": 0.24560517072677612,
    },
    "llava/imagenet-r": {
        "true": Path("probe_results/llava/imagenet-r/16mc-clip"),
        "shuffle": Path("probe_results/llava/imagenet-r/16mc-clip/baselines"),
        "no_skill_brier": 0.18988,
    },
    "llava/vqa-v2": {
        "true": Path("probe_results/llava/vqa-v2"),
        "shuffle": Path("probe_results/llava/vqa-v2/baselines"),
        "no_skill_brier": 0.153334,
    },
    "llava/pope/random": {
        "true": Path("probe_results/llava/pope/random"),
        "shuffle": Path("probe_results/llava/pope/random/baselines"),
        "no_skill_brier": 0.10048888623714447,
    },
    "llava/pope/adversarial": {
        "true": Path("probe_results/llava/pope/adversarial"),
        "shuffle": Path("probe_results/llava/pope/adversarial/baselines"),
        "no_skill_brier": 0.1377750039100647,
    },
    "qwen/coco-qa-vi": {
        "true": Path("probe_results/qwen/coco-qa-vi"),
        "shuffle": Path("probe_results/qwen/coco-qa-vi/baselines"),
        "no_skill_brier": 0.23307399451732635,
    },
    "qwen/imagenet-r": {
        "true": Path("probe_results/qwen/imagenet-r"),
        "shuffle": Path("probe_results/qwen/imagenet-r/baselines"),
        "no_skill_brier": PLACEHOLDER_NO_SKILL_BRIER,
    },
    "qwen/vqa-v2": {
        "true": Path("probe_results/qwen/vqa-v2"),
        "shuffle": Path("probe_results/qwen/vqa-v2/baselines"),
        "no_skill_brier": 0.11618038266897202,
    },
    "qwen/pope/random": {
        "true": Path("probe_results/qwen/pope/random"),
        "shuffle": Path("probe_results/qwen/pope/random/baselines"),
        "no_skill_brier": 0.0727590024471283,
    },
    "qwen/pope/adversarial": {
        "true": Path("probe_results/qwen/pope/adversarial"),
        "shuffle": Path("probe_results/qwen/pope/adversarial/baselines"),
        "no_skill_brier": 0.10686388611793518,
    },
}

OUTDIR = Path("probe_results/_plots/brier_grouped_clean")

DISPLAY_NAMES = {
    "llava/coco-qa-vi": "LLaVA / COCO-QA-VI",
    "llava/imagenet-r": "LLaVA / ImageNet-R",
    "llava/vqa-v2": "LLaVA / VQA-v2",
    "llava/pope/random": "LLaVA / POPE Random",
    "llava/pope/adversarial": "LLaVA / POPE Adversarial",
    "qwen/coco-qa-vi": "Qwen / COCO-QA-VI",
    "qwen/imagenet-r": "Qwen / ImageNet-R",
    "qwen/vqa-v2": "Qwen / VQA-v2",
    "qwen/pope/random": "Qwen / POPE Random",
    "qwen/pope/adversarial": "Qwen / POPE Adversarial",
}

MODEL_TYPES = {"linear", "mlp"}


def dataset_display_name(dataset: str) -> str:
    return DISPLAY_NAMES.get(dataset, dataset.replace("/", " / ").title())


def get_no_skill_brier(dataset: str) -> float:
    return float(DATASET_CONFIGS.get(dataset, {}).get("no_skill_brier", PLACEHOLDER_NO_SKILL_BRIER))


def plot_outdir(dataset: str, probe_type: Optional[str] = None) -> Path:
    outdir = OUTDIR / Path(dataset)
    outdir = outdir / (probe_type if probe_type is not None else "combined")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_plot(fig, dataset: str, filename: str, probe_type: Optional[str] = None) -> None:
    outpath = plot_outdir(dataset, probe_type) / filename
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath} (+ png)")

# -----------------------
# I/O
# -----------------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    
    df.columns = [c.strip() for c in df.columns]
    if "feature_label" not in df.columns:
        return None
        
    for c in ["linear", "mlp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_feature_names(run_dir: Path) -> list[str]:
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            j = _load_json(cfg)
            feats = j.get("feature_names", None)
            if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
                return feats
        except Exception:
            pass

    name = run_dir.name
    parts = [p for p in name.split("__") if p]
    return parts if parts else [name]


def _collect_brier_from_metrics(root: Path) -> Optional[pd.DataFrame]:
    rows = []
    for metrics_path in root.rglob("metrics_test.json"):
        run_dir = metrics_path.parent
        model_type = None
        for parent in (run_dir,) + tuple(run_dir.parents):
            if parent.name in MODEL_TYPES:
                model_type = parent.name
                break
        if model_type is None:
            continue

        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue

        feature_names = _infer_feature_names(run_dir)
        feature_label = feature_names[0] if len(feature_names) == 1 else " + ".join(feature_names)
        rows.append(
            {
                "feature_label": feature_label,
                model_type: metrics.get("test_loss", np.nan),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.groupby("feature_label", as_index=False).first()
    for c in ["linear", "mlp"]:
        if c not in df.columns:
            df[c] = np.nan
    df["linear"] = pd.to_numeric(df["linear"], errors="coerce")
    df["mlp"] = pd.to_numeric(df["mlp"], errors="coerce")
    return df[["feature_label", "linear", "mlp"]]

def load_brier(dataset: str, regime: str) -> Optional[pd.DataFrame]:
    root = DATASET_CONFIGS[dataset][regime]
    df = safe_read_csv(root / "results_brier.csv")
    if df is not None:
        return df
    return _collect_brier_from_metrics(root)

def load_merged_brier(dataset: str) -> Optional[pd.DataFrame]:
    dft = load_brier(dataset, "true")
    dfs = load_brier(dataset, "shuffle")
    if dft is None or dfs is None:
        return None
        
    dft = dft.rename(columns={'linear': 'linear_true', 'mlp': 'mlp_true'})
    dfs = dfs.rename(columns={'linear': 'linear_shuffle', 'mlp': 'mlp_shuffle'})
    
    df = pd.merge(dft, dfs, on='feature_label', how='inner')
    return df

# -----------------------
# HELPER PLOTTING FUNCTION
# -----------------------
def plot_grouped_bars(df: pd.DataFrame, x_col: str, group_col: str, title: str, filename: str, dataset: str, ylabel="Brier Score (Lower is better)", probe_types=None, x_order=None):
    """
    Creates a subplot with grouped bar charts.
    Now includes true (solid) and shuffle (hatched) stacked bars, plus a no-skill line.
    """
    if df.empty:
        print(f"[skip] No data available for {filename}")
        return

    if probe_types is None:
        probe_types = ['linear', 'mlp']
    
    fig, axes = plt.subplots(1, len(probe_types), figsize=(8 * len(probe_types), 6), squeeze=False, sharey=True)
    axes = axes[0]
    
    if x_order:
        x_labels = [l for l in x_order if l in df[x_col].unique()]
    else:
        x_labels = df[x_col].unique()
    
    groups = df[group_col].unique()
    x = np.arange(len(x_labels))
    width = 0.8 / len(groups)
    
    no_skill = get_no_skill_brier(dataset)
    
    for ax_idx, probe_type in enumerate(probe_types):
        ax = axes[ax_idx]
        
        if np.isfinite(no_skill):
            ax.axhspan(0, no_skill, alpha=0.06, color='gray')
            ax.axhline(no_skill, linestyle="--", linewidth=2, color='black', label="No-skill baseline")
        
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group].drop_duplicates(subset=[x_col]).set_index(x_col)
            
            y_true = np.array([group_data.loc[label, f"{probe_type}_true"] if label in group_data.index else np.nan for label in x_labels], dtype=float)
            y_shuf = np.array([group_data.loc[label, f"{probe_type}_shuffle"] if label in group_data.index else np.nan for label in x_labels], dtype=float)
            y_extra = y_shuf - y_true
            
            offset = (i - len(groups) / 2 + 0.5) * width
            
            # Plot True labels (solid)
            bars = ax.bar(x + offset, y_true, width, label=group)
            colors = [b.get_facecolor() for b in bars]
            
            # Plot Shuffle control (hatched on top)
            shuf_label = "Shuffled-label control" if i == 0 else ""
            ax.bar(x + offset, y_extra, width, bottom=y_true, hatch="//", alpha=0.6, color=colors, label=shuf_label)
            
        ax.set_title(f"{probe_type.upper()} Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
        
        # Set y-axis to 0.00-0.45 with 0.05 ticks
        ax.set_ylim(0.00, 0.45)
        ax.set_yticks(np.arange(0.00, 0.46, 0.05))
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Prevent duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left' if (len(probe_types) > 1 and ax_idx == 1) else 'upper right')
        
        if ax_idx == 0:
            ax.set_ylabel(ylabel, fontsize=14)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    save_plot(fig, dataset, filename, probe_type=None if len(probe_types) > 1 else probe_types[0])
    
    # Save individual plots if combined
    if len(probe_types) > 1:
        for probe_type in probe_types:
            plot_grouped_bars(
                df=df, x_col=x_col, group_col=group_col, 
                title=f"{title} ({probe_type.upper()})", 
                filename=f"{Path(filename).stem}_{probe_type}.pdf", 
                dataset=dataset, ylabel=ylabel, 
                probe_types=[probe_type], x_order=x_order
            )

# -----------------------
# 1. COMPONENT COMPARISON
# -----------------------
def plot_components(df: pd.DataFrame, dataset: str):
    selected_features = {
        'vision_mean_layer_-1': 'Vision',
        'lm_visual_mean_layer_-1': 'LM Visual',
        'lm_answer_mean_layer_-1': 'LM Answer',
        'lm_question_mean_layer_-1': 'LM Question',
        'lm_fullspan_mean_layer_-1': 'LM Fullspan',
        'answer_gen_entropy_mean': 'Entropy (Mean)',
        'answer_gen_neglogp_mean': 'NegLogP (Mean)'
    }
    
    subset = df[df['feature_label'].isin(selected_features.keys())].copy()
    if subset.empty:
        return

    subset['Component'] = subset['feature_label'].map(selected_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    components = [v for k, v in selected_features.items() if v in subset['Component'].values]
    
    no_skill = get_no_skill_brier(dataset)
    
    for ax_idx, probe_type in enumerate(['linear', 'mlp']):
        ax = axes[ax_idx]
        
        if np.isfinite(no_skill):
            ax.axhspan(0, no_skill, alpha=0.06, color='gray')
            ax.axhline(no_skill, linestyle="--", linewidth=2, color='black', label="No-skill baseline")
        
        subset_sorted = subset.set_index('Component').reindex(components)
        
        y_true = subset_sorted[f'{probe_type}_true'].to_numpy()
        y_shuf = subset_sorted[f'{probe_type}_shuffle'].to_numpy()
        y_extra = y_shuf - y_true
        
        x = np.arange(len(components))
        
        # Plot True labels (solid)
        bars = ax.bar(x, y_true, color='C0', edgecolor='black', label="Probe (true labels)")
        # Plot Shuffle control (hatched on top)
        ax.bar(x, y_extra, bottom=y_true, hatch="//", alpha=0.6, color='C0', edgecolor='black', label="Shuffled-label control")
        
        ax.set_title(f"{probe_type.upper()} Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right', fontsize=12)
        
        # Set y-axis to 0.00-0.45 with 0.05 ticks
        ax.set_ylim(0.00, 0.45)
        ax.set_yticks(np.arange(0.00, 0.46, 0.05))
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if ax_idx == 0:
            ax.set_ylabel("Brier Score (Lower is better)", fontsize=14)
            ax.legend()
            
    plt.suptitle(f"Component Comparison (Final Layer, Mean Aggregation) - {dataset_display_name(dataset)}", fontsize=16)
    plt.tight_layout()
    save_plot(plt.gcf(), dataset, "1_component_comparison.pdf")

    # Generate individual plots
    for probe_type in ['linear', 'mlp']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        no_skill = get_no_skill_brier(dataset)
        if np.isfinite(no_skill):
            ax.axhspan(0, no_skill, alpha=0.06, color='gray')
            ax.axhline(no_skill, linestyle="--", linewidth=2, color='black', label="No-skill baseline")
        
        subset_sorted = subset.set_index('Component').reindex(components)
        y_true = subset_sorted[f'{probe_type}_true'].to_numpy()
        y_shuf = subset_sorted[f'{probe_type}_shuffle'].to_numpy()
        y_extra = y_shuf - y_true
        
        x = np.arange(len(components))
        bars = ax.bar(x, y_true, color='C0', edgecolor='black', label="Probe (true labels)")
        ax.bar(x, y_extra, bottom=y_true, hatch="//", alpha=0.6, color='C0', edgecolor='black', label="Shuffled-label control")
        
        ax.set_title(f"{probe_type.upper()} Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0.00, 0.45)
        ax.set_yticks(np.arange(0.00, 0.46, 0.05))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel("Brier Score (Lower is better)", fontsize=14)
        ax.legend()
        
        plt.suptitle(f"Component Comparison (Final Layer, Mean Aggregation) - {dataset_display_name(dataset)}", fontsize=16)
        plt.tight_layout()
        save_plot(fig, dataset, f"1_component_comparison_{probe_type}.pdf", probe_type=probe_type)

def plot_components_lasttok(df: pd.DataFrame, dataset: str):
    selected_features = {
        'vision_lasttok_layer_-1': 'Vision',
        'lm_visual_lasttok_layer_-1': 'LM Visual',
        'lm_answer_lasttok_layer_-1': 'LM Answer',
        'lm_question_lasttok_layer_-1': 'LM Question',
        'lm_fullspan_lasttok_layer_-1': 'LM Fullspan',
        'answer_gen_entropy_mean': 'Entropy (Mean)',
        'answer_gen_neglogp_mean': 'NegLogP (Mean)'
    }
    
    subset = df[df['feature_label'].isin(selected_features.keys())].copy()
    if subset.empty:
        # Fallback for datasets where 'vision_lasttok' might not exist or has different name
        # Some loaders might use 'vision_mean' for both if only one exists
        # Let's check for alternatives if subset is small
        if len(subset) < 4:
            alt_features = {
                'vision_mean_layer_-1': 'Vision',
                'lm_visual_lasttok_layer_-1': 'LM Visual',
                'lm_question_lasttok_layer_-1': 'LM Question',
                'lm_answer_lasttok_layer_-1': 'LM Answer',
                'lm_fullspan_lasttok_layer_-1': 'LM Fullspan',
                'answer_gen_entropy_mean': 'Entropy (Mean)',
                'answer_gen_neglogp_mean': 'NegLogP (Mean)'
            }
            subset = df[df['feature_label'].isin(alt_features.keys())].copy()
            subset['Component'] = subset['feature_label'].map(alt_features)
        
    if subset.empty:
        return

    if 'Component' not in subset.columns:
        subset['Component'] = subset['feature_label'].map(selected_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    components = [v for k, v in selected_features.items() if v in subset['Component'].values]
    
    no_skill = get_no_skill_brier(dataset)
    
    # 1. Combined Plot
    for ax_idx, probe_type in enumerate(['linear', 'mlp']):
        ax = axes[ax_idx]
        
        if np.isfinite(no_skill):
            ax.axhspan(0, no_skill, alpha=0.06, color='gray')
            ax.axhline(no_skill, linestyle="--", linewidth=2, color='black', label="No-skill baseline")
        
        subset_sorted = subset.set_index('Component').reindex(components).dropna(subset=[f'{probe_type}_true'])
        current_components = subset_sorted.index.tolist()
        
        y_true = subset_sorted[f'{probe_type}_true'].to_numpy()
        y_shuf = subset_sorted[f'{probe_type}_shuffle'].to_numpy()
        y_extra = y_shuf - y_true
        
        x = np.arange(len(current_components))
        
        # Plot True labels (solid)
        bars = ax.bar(x, y_true, color='C0', edgecolor='black', label="Probe (true labels)")
        # Plot Shuffle control (hatched on top)
        ax.bar(x, y_extra, bottom=y_true, hatch="//", alpha=0.6, color='C0', edgecolor='black', label="Shuffled-label control")
        
        ax.set_title(f"{probe_type.upper()} Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(current_components, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0.00, 0.45)
        ax.set_yticks(np.arange(0.00, 0.46, 0.05))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if ax_idx == 0:
            ax.set_ylabel("Brier Score (Lower is better)", fontsize=14)
            ax.legend()
            
    plt.suptitle(f"Component Comparison (Final Layer, LastTok Aggregation) - {dataset_display_name(dataset)}", fontsize=16)
    plt.tight_layout()
    save_plot(plt.gcf(), dataset, "1_component_comparison_lasttok.pdf")

    # 2. Individual Plots
    for probe_type in ['linear', 'mlp']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        no_skill = get_no_skill_brier(dataset)
        if np.isfinite(no_skill):
            ax.axhspan(0, no_skill, alpha=0.06, color='gray')
            ax.axhline(no_skill, linestyle="--", linewidth=2, color='black', label="No-skill baseline")
        
        subset_sorted = subset.set_index('Component').reindex(components).dropna(subset=[f'{probe_type}_true'])
        current_components = subset_sorted.index.tolist()
        
        y_true = subset_sorted[f'{probe_type}_true'].to_numpy()
        y_shuf = subset_sorted[f'{probe_type}_shuffle'].to_numpy()
        y_extra = y_shuf - y_true
        
        x = np.arange(len(current_components))
        bars = ax.bar(x, y_true, color='C0', edgecolor='black', label="Probe (true labels)")
        ax.bar(x, y_extra, bottom=y_true, hatch="//", alpha=0.6, color='C0', edgecolor='black', label="Shuffled-label control")
        
        ax.set_title(f"{probe_type.upper()} Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(current_components, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0.00, 0.45)
        ax.set_yticks(np.arange(0.00, 0.46, 0.05))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel("Brier Score (Lower is better)", fontsize=14)
        ax.legend()
        
        plt.suptitle(f"Component Comparison (Final Layer, LastTok Aggregation) - {dataset_display_name(dataset)}", fontsize=16)
        plt.tight_layout()
        save_plot(fig, dataset, f"1_component_comparison_lasttok_{probe_type}.pdf", probe_type=probe_type)

# -----------------------
# 2. MEAN VS LAST TOKEN
# -----------------------
def plot_aggregation_strategies(df: pd.DataFrame, dataset: str):
    mask = df['feature_label'].str.contains('layer_-1') & df['feature_label'].str.contains('lm_')
    mask &= ~df['feature_label'].str.contains('including')
    subset = df[mask].copy()
    
    if subset.empty:
        return

    subset['Base_Feature'] = subset['feature_label'].apply(lambda x: x.split('_mean')[0].split('_lasttok')[0])
    subset['Aggregation'] = subset['feature_label'].apply(lambda x: 'Mean' if '_mean_' in x else 'Last Token')
    
    plot_grouped_bars(
        df=subset, 
        x_col='Base_Feature', 
        group_col='Aggregation', 
        title=f"Aggregation Strategy: Mean vs Last Token (Final Layer) - {dataset_display_name(dataset)}", 
        filename="2_aggregation_strategy.pdf",
        dataset=dataset
    )

# -----------------------
# 3. MIDDLE VS FINAL LAYER
# -----------------------
def plot_layer_differences(df: pd.DataFrame, dataset: str):
    mask = df['feature_label'].str.contains('_mean_layer_')
    mask &= ~df['feature_label'].str.contains('including')
    subset = df[mask].copy()
    
    if subset.empty:
        return

    subset['Base_Feature'] = subset['feature_label'].apply(lambda x: x.split('_mean_')[0])
    subset['Layer'] = subset['feature_label'].apply(lambda x: 'Final Layer' if '-1' in x else 'Middle Layer')

    # Mapping for nice names and sorting
    name_map = {
        'vision': 'Vision',
        'lm_visual': 'LM Visual',
        'lm_answer': 'LM Answer',
        'lm_question': 'LM Question',
        'lm_fullspan': 'LM Fullspan'
    }
    subset['Base_Feature_Name'] = subset['Base_Feature'].map(name_map)
    
    # Sort by the order requested
    order = ['Vision', 'LM Visual', 'LM Answer', 'LM Question', 'LM Fullspan']
    
    plot_grouped_bars(
        df=subset, 
        x_col='Base_Feature_Name', 
        group_col='Layer', 
        title=f"Network Depth: Middle vs Final Layer (Mean Aggregation) - {dataset_display_name(dataset)}", 
        filename="3_layer_differences.pdf",
        dataset=dataset,
        probe_types=['linear', 'mlp'],
        x_order=order
    )

# -----------------------
# 4. QUESTION CONTEXT
# -----------------------
def plot_question_context(df: pd.DataFrame, dataset: str):
    mask = df['feature_label'].str.contains('lm_question')
    subset = df[mask].copy()
    
    if subset.empty:
        return

    subset['Context'] = subset['feature_label'].apply(
        lambda x: 'Question + Include' if 'including' in x else 'Question Only'
    )
    
    def get_config(label):
        agg = 'Mean' if '_mean_' in label else 'LastTok'
        layer = 'Final Layer' if '-1' in label else 'Middle Layer'
        return f"{agg} | {layer}"
        
    subset['Configuration'] = subset['feature_label'].apply(get_config)
    
    plot_grouped_bars(
        df=subset, 
        x_col='Configuration', 
        group_col='Context', 
        title=f"Prompt Context: Regular Question vs Question Including Image - {dataset_display_name(dataset)}", 
        filename="4_question_context.pdf",
        dataset=dataset,
        probe_types=['linear', 'mlp']
    )

# -----------------------
# 5. COMBINED LAYER DIFFERENCES (MLP)
# -----------------------
def plot_combined_layer_differences(dfs: dict[str, pd.DataFrame], datasets: list[str]):
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    probe_type = "mlp"
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        df = dfs.get(dataset)
        
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha='center')
            continue

        # Filter logic (same as plot_layer_differences)
        mask = df['feature_label'].str.contains('_mean_layer_')
        mask &= ~df['feature_label'].str.contains('including')
        subset = df[mask].copy()
        
        if subset.empty:
            continue

        subset['Base_Feature'] = subset['feature_label'].apply(lambda x: x.split('_mean_')[0])
        subset['Layer'] = subset['feature_label'].apply(lambda x: 'Final Layer' if '-1' in x else 'Middle Layer')

        name_map = {
            'vision': 'Vision',
            'lm_visual': 'LM Visual',
            'lm_answer': 'LM Answer',
            'lm_question': 'LM Question',
            'lm_fullspan': 'LM Fullspan'
        }
        subset['Base_Feature_Name'] = subset['Base_Feature'].map(name_map)
        
        x_order = ['Vision', 'LM Visual', 'LM Answer', 'LM Question', 'LM Fullspan']
        x_labels = [l for l in x_order if l in subset['Base_Feature_Name'].unique()]
        
        # Explicit sorting: Middle Layer first, then Final Layer
        groups = ['Middle Layer', 'Final Layer']
        
        x = np.arange(len(x_labels))
        width = 0.8 / len(groups)
        no_skill = get_no_skill_brier(dataset)
        if np.isfinite(no_skill):
            ax.axhspan(0, no_skill, alpha=0.06, color='gray')
            ax.axhline(no_skill, linestyle="--", linewidth=2, color='black', label="No-skill baseline")

        for i, group in enumerate(groups):
            group_data = subset[subset['Layer'] == group].drop_duplicates(subset=['Base_Feature_Name']).set_index('Base_Feature_Name')
            
            y_true = np.array([group_data.loc[label, f"{probe_type}_true"] if label in group_data.index else np.nan for label in x_labels], dtype=float)
            y_shuf = np.array([group_data.loc[label, f"{probe_type}_shuffle"] if label in group_data.index else np.nan for label in x_labels], dtype=float)
            y_extra = y_shuf - y_true
            
            offset = (i - len(groups) / 2 + 0.5) * width
            
            bars = ax.bar(x + offset, y_true, width, label=group)
            colors = [b.get_facecolor() for b in bars]
            
            shuf_label = "Shuffled-label control" if i == 0 else ""
            ax.bar(x + offset, y_extra, width, bottom=y_true, hatch="//", alpha=0.6, color=colors, label=shuf_label)

        ax.set_title(dataset_display_name(dataset))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0.00, 0.45)
        ax.set_yticks(np.arange(0.00, 0.46, 0.05))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if ax_idx == 0:
            ax.set_ylabel("Brier Score (Lower is better)", fontsize=14)
            
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.suptitle("Network Depth: Middle vs Final Layer (Mean Aggregation) - MLP Probe", fontsize=16, y=1.02)
    plt.tight_layout()
    save_plot(fig, "_combined", "combined_layer_differences_mlp.pdf", probe_type="mlp")

# -----------------------
# MAIN EXECUTION
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot probe results against shuffled-label baselines for selected model/dataset combinations."
    )
    parser.add_argument(
        "--combo",
        action="append",
        choices=sorted(DATASET_CONFIGS.keys()),
        help=(
            "Exact model/dataset combination to plot. Repeat this flag to run multiple "
            "combinations, for example: --combo llava/coco-qa-vi --combo qwen/vqa-v2"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = args.combo if args.combo else list(DATASET_CONFIGS.keys())
    loaded_dfs = {}
    
    for dataset in datasets:
        # Load merged data containing BOTH true and shuffle columns
        df = load_merged_brier(dataset)
        loaded_dfs[dataset] = df
        
        if df is None:
            print(f"[skip] missing or empty Brier CSVs for {dataset}")
            continue
            
        print(f"\nGenerating plots for {dataset}...")
        plot_components(df, dataset)
        plot_components_lasttok(df, dataset)
        plot_aggregation_strategies(df, dataset)
        plot_layer_differences(df, dataset)
        plot_question_context(df, dataset)

    if len(datasets) > 1:
        print(f"\nGenerating combined plot...")
        plot_combined_layer_differences(loaded_dfs, datasets)

if __name__ == "__main__":
    main()
