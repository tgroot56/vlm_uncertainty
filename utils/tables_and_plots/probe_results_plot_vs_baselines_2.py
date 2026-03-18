from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# -----------------------
# CONFIG & PATHS
# -----------------------
NO_SKILL_BRIER = {
    "imagenet-r": 0.18988,
    "vqa-v2": 0.153334,
}

PATHS = {
    ("imagenet-r", "true"): Path("probe_results/imagenet-r/16mc-clip"),
    ("imagenet-r", "shuffle"): Path("probe_results/imagenet-r/16mc-clip/baselines"),
    ("vqa-v2", "true"): Path("probe_results/vqa-v2"),
    ("vqa-v2", "shuffle"): Path("probe_results/vqa-v2/baselines"),
}

OUTDIR = Path("probe_results/_plots/brier_grouped_clean")

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

def load_brier(dataset: str, regime: str) -> Optional[pd.DataFrame]:
    return safe_read_csv(PATHS[(dataset, regime)] / "results_brier.csv")

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
    
    no_skill = NO_SKILL_BRIER.get(dataset, 0.25)
    
    for ax_idx, probe_type in enumerate(probe_types):
        ax = axes[ax_idx]
        
        # Add baseline formatting
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
    
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTDIR / filename
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath} (+ png)")
    
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
    
    no_skill = NO_SKILL_BRIER.get(dataset, 0.25)
    
    for ax_idx, probe_type in enumerate(['linear', 'mlp']):
        ax = axes[ax_idx]
        
        # Add baseline formatting
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
            
    plt.suptitle(f"Component Comparison (Final Layer, Mean Aggregation) - {dataset.upper()}", fontsize=16)
    plt.tight_layout()
    outpath = OUTDIR / f"{dataset}_1_component_comparison.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[saved] {outpath} (+ png)")

    # Generate individual plots
    for probe_type in ['linear', 'mlp']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        no_skill = NO_SKILL_BRIER.get(dataset, 0.25)
        
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
        
        plt.suptitle(f"Component Comparison (Final Layer, Mean Aggregation) - {dataset.upper()}", fontsize=16)
        plt.tight_layout()
        outpath = OUTDIR / f"{dataset}_1_component_comparison_{probe_type}.pdf"
        plt.savefig(outpath, bbox_inches="tight")
        plt.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[saved] {outpath} (+ png)")

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
    
    no_skill = NO_SKILL_BRIER.get(dataset, 0.25)
    
    # 1. Combined Plot
    for ax_idx, probe_type in enumerate(['linear', 'mlp']):
        ax = axes[ax_idx]
        
        # Add baseline formatting
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
            
    plt.suptitle(f"Component Comparison (Final Layer, LastTok Aggregation) - {dataset.upper()}", fontsize=16)
    plt.tight_layout()
    outpath = OUTDIR / f"{dataset}_1_component_comparison_lasttok.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[saved] {outpath} (+ png)")

    # 2. Individual Plots
    for probe_type in ['linear', 'mlp']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        no_skill = NO_SKILL_BRIER.get(dataset, 0.25)
        
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
        
        plt.suptitle(f"Component Comparison (Final Layer, LastTok Aggregation) - {dataset.upper()}", fontsize=16)
        plt.tight_layout()
        outpath = OUTDIR / f"{dataset}_1_component_comparison_lasttok_{probe_type}.pdf"
        plt.savefig(outpath, bbox_inches="tight")
        plt.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[saved] {outpath} (+ png)")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if ax_idx == 0:
            ax.set_ylabel("Brier Score (Lower is better)", fontsize=14)
            ax.legend()
            
    plt.suptitle(f"Component Comparison (Final Layer, LastTok Aggregation) - {dataset.upper()}", fontsize=16)
    plt.tight_layout()
    outpath = OUTDIR / f"{dataset}_1_component_comparison_lasttok.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[saved] {outpath} (+ png)")

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
        title=f"Aggregation Strategy: Mean vs Last Token (Final Layer) - {dataset.upper()}", 
        filename=f"{dataset}_2_aggregation_strategy.pdf",
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
        title=f"Network Depth: Middle vs Final Layer (Mean Aggregation) - {dataset.upper()}", 
        filename=f"{dataset}_3_layer_differences.pdf",
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
        title=f"Prompt Context: Regular Question vs Question Including Image - {dataset.upper()}", 
        filename=f"{dataset}_4_question_context.pdf",
        dataset=dataset,
        probe_types=['linear', 'mlp']
    )

# -----------------------
# 5. COMBINED LAYER DIFFERENCES (MLP)
# -----------------------
def plot_combined_layer_differences(dfs: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    datasets = ["imagenet-r", "vqa-v2"]
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
        no_skill = NO_SKILL_BRIER.get(dataset, 0.25)
        
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

        dataset_title = "ImageNet-R" if dataset == "imagenet-r" else "VQA-v2"
        ax.set_title(f"{dataset_title}")
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
    
    outpath = OUTDIR / "combined_layer_differences_mlp.pdf"
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {outpath} (+ png)")

# -----------------------
# MAIN EXECUTION
# -----------------------
def main():
    datasets = ["imagenet-r", "vqa-v2"]
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

    # Combined plot
    print(f"\nGenerating combined plot...")
    plot_combined_layer_differences(loaded_dfs)

if __name__ == "__main__":
    main()