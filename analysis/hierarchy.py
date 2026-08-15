import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import json
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text, not paths
# mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']


CONFIG = {
    # 'dataset': 'cfp',
    # 'results_dir': './results/cfp_analysis',
    # 'output_dir': './results/cfp_matrix_analysis',

    'dataset': 'cfp',
    'results_dir': './results/cfp_analysis',
    'output_dir': './results/cfp_matrix_analysis',
    'method': 'Linear',  # Which alignment method to analyze
    'models': [
        'clip', 'align', 'dinov2', 'googlevit',
        'blip2', 'llava', 'kosmos', 'internvl',
        'sam', 'florence'
        # 'arcface', 'adaface', 'kprpe', 'magface'
    ],
}

_sel = os.environ.get("CFE_MODELS")
if _sel:
    _keep = {s.strip() for s in _sel.split(",") if s.strip()}
    CONFIG['models'] = [m for m in CONFIG['models'] if m in _keep]

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# Model metadata (for analysis)
MODEL_INFO = {
    'clip': {'arch': 'ViT', 'objective': 'Contrastive', 'dim': 512},
    'align': {'arch': 'EfficientNet', 'objective': 'Contrastive', 'dim': 640},
    'dinov2': {'arch': 'ViT', 'objective': 'Self-supervised', 'dim': 768},
    'dino': {'arch': 'ViT', 'objective': 'Self-supervised', 'dim': 768},
    'googlevit': {'arch': 'ViT', 'objective': 'Supervised', 'dim': 768},
    'blip2': {'arch': 'ViT+LLM', 'objective': 'Multimodal', 'dim': 1408},
    'llava': {'arch': 'ViT+LLM', 'objective': 'Multimodal', 'dim': 1024},
    'kosmos': {'arch': 'ViT+LLM', 'objective': 'Multimodal', 'dim': 1024},
    'internvl': {'arch': 'ViT+LLM', 'objective': 'Multimodal', 'dim': 512},
    'sam': {'arch': 'ViT', 'objective': 'Segmentation', 'dim': 256},
    'florence': {'arch': 'ViT', 'objective': 'Dense', 'dim': 768},
    'arcface': {'arch': 'ViT', 'objective': 'Dense', 'dim': 768},
    'adaface': {'arch': 'ViT', 'objective': 'Dense', 'dim': 768},
    'magface': {'arch': 'ViT', 'objective': 'Dense', 'dim': 768},
    'kprpe': {'arch': 'ViT', 'objective': 'Dense', 'dim': 768},


}


def build_alignment_matrix(models, results_dir, dataset, method='Linear'):
    """
    Build n×n alignment accuracy matrix from individual CSV result files.
    
    Args:
        models: List of model names
        results_dir: Directory containing result CSV files
        dataset: Dataset name (e.g., 'cfp')
        method: Alignment method to extract (e.g., 'Linear', 'Procrustes')
    
    Returns:
        rank1_matrix: n×n matrix of Rank-1 accuracies
        map_matrix: n×n matrix of mAP values
        available_pairs: List of (modelX, modelY) pairs that have results
    """
    n = len(models)
    rank1_matrix = np.full((n, n), np.nan)
    map_matrix = np.full((n, n), np.nan)
    
    # Diagonal is perfect (same model)
    np.fill_diagonal(rank1_matrix, 100.0)
    np.fill_diagonal(map_matrix, 1.0)
    
    available_pairs = []
    
    for i, modelX in enumerate(models):
        for j, modelY in enumerate(models):
            if i == j:
                continue
            
            # Look for result file
            csv_path = f"{results_dir}/{modelX}_to_{modelY}_{dataset}_aggregated_results.csv"
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Find the row for the specified method
                    method_row = df[df['Method'] == method]
                    
                    if len(method_row) > 0:
                        # Extract mean from "mean ± std" format
                        rank1_str = method_row['RANK1'].values[0]
                        map_str = method_row['MAP'].values[0]
                        
                        # Parse "XX.XX ± YY.YY" format
                        rank1_mean = float(rank1_str.split('±')[0].strip())
                        map_mean = float(map_str.split('±')[0].strip())
                        
                        rank1_matrix[i, j] = rank1_mean
                        map_matrix[i, j] = map_mean
                        
                        available_pairs.append((modelX, modelY))
                        print(f"✓ Loaded {modelX}→{modelY}: Rank-1={rank1_mean:.2f}%")
                    else:
                        print(f"⚠ Method '{method}' not found in {csv_path}")
                        
                except Exception as e:
                    print(f"✗ Error reading {csv_path}: {e}")
            else:
                print(f"✗ File not found: {csv_path}")
    
    return rank1_matrix, map_matrix, available_pairs


print("="*70)
print("BUILDING ALIGNMENT MATRIX FROM RESULTS")
print("="*70)

rank1_matrix, map_matrix, available_pairs = build_alignment_matrix(
    CONFIG['models'],
    CONFIG['results_dir'],
    CONFIG['dataset'],
    CONFIG['method']
)

print(f"\nMatrix shape: {rank1_matrix.shape}")
print(f"Available pairs: {len(available_pairs)}")
print(f"Missing pairs: {np.sum(np.isnan(rank1_matrix)) - len(CONFIG['models'])}")  # Subtract diagonal

# Save matrices
np.save(f"{CONFIG['output_dir']}/rank1_matrix.npy", rank1_matrix)
np.save(f"{CONFIG['output_dir']}/map_matrix.npy", map_matrix)

with open(f"{CONFIG['output_dir']}/available_pairs.json", 'w') as f:
    json.dump(available_pairs, f, indent=2)

# ANALYSIS 1: HIERARCHICAL CLUSTERING (DENDROGRAM)

def plot_dendrogram(matrix, models, save_path):
    """
    Create hierarchical clustering dendrogram.
    """
    # Symmetrize matrix (average of A→B and B→A)
    matrix_sym = np.copy(matrix)
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            if not np.isnan(matrix[i, j]) and not np.isnan(matrix[j, i]):
                avg = (matrix[i, j] + matrix[j, i]) / 2
                matrix_sym[i, j] = avg
                matrix_sym[j, i] = avg
    
    np.fill_diagonal(matrix_sym, 0)
    
    # Convert similarity to distance (100 - accuracy)
    D = 100 - matrix_sym
    
    # Handle NaN values (set to max distance)
    D = np.nan_to_num(D, nan=100)
    
    # Convert to condensed distance matrix
    D_condensed = squareform(D, checks=False)
    
    # Hierarchical clustering
    Z = linkage(D_condensed, method='average')
    # Plot
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=models, leaf_font_size=14)
    plt.xlabel('Model', fontsize=15)
    plt.ylabel('Distance (100 - Rank-1 Accuracy)', fontsize=15)
    plt.title(f'Hierarchical Clustering of Foundation Models\n'
              f'Dataset: {CONFIG["dataset"].upper()}, Method: {CONFIG["method"]}',
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dendrogram saved to: {save_path}")
    plt.close()

plot_dendrogram(
    rank1_matrix, 
    CONFIG['models'], 
    f"{CONFIG['output_dir']}/dendrogram.svg"
)

# ANALYSIS 2: DIRECTIONAL ASYMMETRY

def analyze_asymmetry(matrix, models, save_path):
    """
    Analyze and visualize directional asymmetry.
    """
    # Compute asymmetry matrix
    A = np.abs(matrix - matrix.T)
    np.fill_diagonal(A, 0)
    
    # Find most asymmetric pairs
    asymmetry_pairs = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            if not np.isnan(matrix[i, j]) and not np.isnan(matrix[j, i]):
                asymmetry_pairs.append({
                    'Source': models[i],
                    'Target': models[j],
                    'Forward': matrix[i, j],
                    'Backward': matrix[j, i],
                    'Asymmetry': A[i, j],
                    'Dim_Source': MODEL_INFO[models[i]]['dim'],
                    'Dim_Target': MODEL_INFO[models[j]]['dim'],
                })
    
    df_asym = pd.DataFrame(asymmetry_pairs).sort_values('Asymmetry', ascending=False)
    
    # Save to CSV
    csv_path = f"{CONFIG['output_dir']}/asymmetry_analysis.csv"
    df_asym.to_csv(csv_path, index=False)
    print(f"\n✓ Asymmetry analysis saved to: {csv_path}")
    
    print("\nTop 10 Most Asymmetric Pairs:")
    print(df_asym.head(10)[['Source', 'Target', 'Forward', 'Backward', 
                            'Asymmetry', 'Dim_Source', 'Dim_Target']].to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Asymmetry magnitude heatmap
    ax = axes[0]
    mask = np.isnan(A)
    sns.heatmap(A, annot=True, fmt='.1f', cmap='Reds', 
                xticklabels=models, yticklabels=models,
                ax=ax, cbar_kws={'label': 'Asymmetry (%)'},
                mask=mask)
    ax.set_title('Directional Asymmetry: |R[i→j] - R[j→i]|', fontweight='bold')
    
    # Right: Scatter plot of asymmetry vs dimensionality difference
    ax = axes[1]
    dim_diff = np.abs(df_asym['Dim_Source'] - df_asym['Dim_Target'])
    
    ax.scatter(dim_diff, df_asym['Asymmetry'], s=100, alpha=0.6)
    
    # Add labels for top asymmetric pairs
    for idx in df_asym.head(5).index:
        row = df_asym.loc[idx]
        ax.annotate(f"{row['Source']}→{row['Target']}", 
                   (dim_diff[idx], row['Asymmetry']),
                   fontsize=8, ha='right')
    
    # Correlation
    rho, p = spearmanr(dim_diff, df_asym['Asymmetry'])
    
    ax.set_xlabel('Dimensionality Difference |dim(Source) - dim(Target)|', fontsize=12)
    ax.set_ylabel('Asymmetry (%)', fontsize=12)
    ax.set_title(f'Asymmetry vs Dimensionality Gap\n(Spearman ρ={rho:.3f}, p={p:.4f})',
                fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Asymmetry visualization saved to: {save_path}")
    plt.close()
    
    return df_asym

df_asym = analyze_asymmetry(
    rank1_matrix,
    CONFIG['models'],
    f"{CONFIG['output_dir']}/asymmetry.svg"
)

# ANALYSIS 3: EMBEDDING SPACE CENTRALITY

def compute_centrality(matrix, models, save_path):
    """
    Which model's embedding space is easiest to align TO?
    """
    n = len(models)
    mask = ~np.eye(n, dtype=bool)
    
    centrality_scores = []
    
    for i, model in enumerate(models):
        # Incoming alignment (column average) - how well others align TO this model
        incoming = []
        for j in range(n):
            if i != j and not np.isnan(matrix[j, i]):
                incoming.append(matrix[j, i])
        
        incoming_avg = np.mean(incoming) if len(incoming) > 0 else np.nan
        
        # Outgoing alignment (row average) - how well this model aligns TO others
        outgoing = []
        for j in range(n):
            if i != j and not np.isnan(matrix[i, j]):
                outgoing.append(matrix[i, j])
        
        outgoing_avg = np.mean(outgoing) if len(outgoing) > 0 else np.nan
        
        centrality_scores.append({
            'Model': model,
            'Incoming (Target)': incoming_avg,
            'Outgoing (Source)': outgoing_avg,
            'Overall': (incoming_avg + outgoing_avg) / 2,
            'Architecture': MODEL_INFO[model]['arch'],
            'Objective': MODEL_INFO[model]['objective'],
            'Dimension': MODEL_INFO[model]['dim'],
        })
    
    df_central = pd.DataFrame(centrality_scores).sort_values('Incoming (Target)', ascending=False)
    
    # Save to CSV
    csv_path = f"{CONFIG['output_dir']}/centrality_scores.csv"
    df_central.to_csv(csv_path, index=False)
    print(f"\n✓ Centrality scores saved to: {csv_path}")
    
    print("\nEmbedding Space Centrality (Easiest to align TO):")
    print(df_central[['Model', 'Incoming (Target)', 'Outgoing (Source)', 'Overall', 'Dimension']].to_string(index=False))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    sorted_df = df_central.sort_values('Overall', ascending=False)
    
    ax.barh(x - width/2, sorted_df['Incoming (Target)'], width, 
           label='Incoming (as Target)', alpha=0.8, color='steelblue')
    ax.barh(x + width/2, sorted_df['Outgoing (Source)'], width, 
           label='Outgoing (as Source)', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Average Rank-1 Accuracy (%)', fontsize=12)
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_df['Model'])
    ax.set_title('Model Compatibility: Target vs Source Performance', 
                fontsize=13, fontweight='bold')
    ax.legend()
    # ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Centrality visualization saved to: {save_path}")
    plt.close()
    
    return df_central

df_central = compute_centrality(
    rank1_matrix,
    CONFIG['models'],
    f"{CONFIG['output_dir']}/centrality.svg"
)


# ANALYSIS 5: SYMMETRIZED HEATMAP

def plot_symmetrized_heatmap(matrix, models, save_path):
    """
    Plot symmetrized accuracy matrix: average of A→B and B→A.
    """
    # Symmetrize
    matrix_sym = np.copy(matrix)
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            if not np.isnan(matrix[i, j]) and not np.isnan(matrix[j, i]):
                avg = (matrix[i, j] + matrix[j, i]) / 2
                matrix_sym[i, j] = avg
                matrix_sym[j, i] = avg
    
    plt.figure(figsize=(10, 8))
    mask = np.isnan(matrix_sym)
    sns.heatmap(matrix_sym, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=models, yticklabels=models,
                cbar_kws={'label': 'Rank-1 Accuracy (%)'},
                mask=mask, vmin=0, vmax=100)
    plt.title(f'Symmetrized Cross-Model Alignment Accuracy\n'
              f'Dataset: {CONFIG["dataset"].upper()}, Method: {CONFIG["method"]}',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Symmetrized heatmap saved to: {save_path}")
    plt.close()

plot_symmetrized_heatmap(
    rank1_matrix,
    CONFIG['models'],
    f"{CONFIG['output_dir']}/heatmap_symmetrized.svg"
)


print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Exclude diagonal
mask = ~np.eye(len(CONFIG['models']), dtype=bool)
off_diag = rank1_matrix[mask]
off_diag = off_diag[~np.isnan(off_diag)]

print(f"Overall mean alignment accuracy: {off_diag.mean():.2f}%")
print(f"Overall std: {off_diag.std():.2f}%")
print(f"Min: {off_diag.min():.2f}%")
print(f"Max: {off_diag.max():.2f}%")

# Asymmetry statistics
A = np.abs(rank1_matrix - rank1_matrix.T)
asym_values = A[mask]
asym_values = asym_values[~np.isnan(asym_values)]

print(f"\nMean asymmetry: {asym_values.mean():.2f}%")
print(f"Max asymmetry: {asym_values.max():.2f}%")

# Best and worst pairs
print(f"\nBest alignment pair: {off_diag.max():.2f}%")
best_idx = np.unravel_index(np.nanargmax(rank1_matrix * (1 - np.eye(len(CONFIG['models'])))), 
                             rank1_matrix.shape)
print(f"  {CONFIG['models'][best_idx[0]]} → {CONFIG['models'][best_idx[1]]}")

print(f"\nWorst alignment pair: {off_diag.min():.2f}%")
worst_mask = np.copy(rank1_matrix)
np.fill_diagonal(worst_mask, np.nan)
worst_idx = np.unravel_index(np.nanargmin(worst_mask), worst_mask.shape)
print(f"  {CONFIG['models'][worst_idx[0]]} → {CONFIG['models'][worst_idx[1]]}")

print("\n" + "="*70)
print("ALL ANALYSES COMPLETED!")
print(f"Results saved to: {CONFIG['output_dir']}")
print("="*70)