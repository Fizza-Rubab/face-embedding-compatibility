import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from cfe.methods import *
import json
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text, not paths
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
# ======================================================
# DATA EFFICIENCY EXPERIMENT CONFIGURATION
# ======================================================
DATA_EFFICIENCY_CONFIG = {
    'dataset': 'cfp',
    'meta_path': 'embeddings/cfp/cfp_metadata.npy',
    # Representative pair for the performance-vs-training-data trend (paper Fig. 3).
    # Add more pairs here to average the curves over additional pairs.
    'model_pairs': [
        ('arcface', 'adaface'),
    ],
    'train_ratios': [0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.70],  # Vary training data
    'test_ratio': 0.30,  # Fixed test split
    'out_dir': './results/data_efficiency_cfp',
    'n_seeds': 5,
    'seeds': [42, 123, 217, 7, 531],
    'max_rank': 50,
}

os.makedirs(DATA_EFFICIENCY_CONFIG['out_dir'], exist_ok=True)

# Save configuration
with open(f"{DATA_EFFICIENCY_CONFIG['out_dir']}/config.json", 'w') as f:
    json.dump(DATA_EFFICIENCY_CONFIG, f, indent=2)

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def split_by_identity_with_fixed_test(metadata, train_ratio, test_ratio, seed=42):
    """
    Split dataset with FIXED test set and VARYING train set.
    Test set is always the LAST test_ratio% of identities.
    Train set grows from the FIRST train_ratio% of remaining identities.
    """
    np.random.seed(seed)
    
    # Group image indices by identity
    identity_to_indices = defaultdict(list)
    for idx, rec in enumerate(metadata):
        identity_to_indices[rec['identity']].append(idx)
    
    # Get unique identities and shuffle ONCE
    identities = sorted(identity_to_indices.keys())
    n_total = len(identities)
    
    identities_shuffled = identities.copy()
    np.random.shuffle(identities_shuffled)
    
    # FIXED split: test comes from END
    n_test = int(n_total * test_ratio)
    test_identities = set(identities_shuffled[-n_test:])  # LAST 30%
    
    # VARYING split: train comes from BEGINNING of remaining
    remaining_identities = identities_shuffled[:-n_test]  # First 70%
    n_train = int(len(remaining_identities) * (train_ratio / (1 - test_ratio)))
    train_identities = set(remaining_identities[:n_train])
    
    unused_identities = set(remaining_identities[n_train:])
    
    # Get all image indices
    train_idx = []
    test_idx = []
    unused_idx = []
    
    for identity, indices in identity_to_indices.items():
        if identity in train_identities:
            train_idx.extend(indices)
        elif identity in test_identities:
            test_idx.extend(indices)
        else:
            unused_idx.extend(indices)
    
    # Verify no overlap
    assert len(train_identities.intersection(test_identities)) == 0, "Train/test overlap!"
    assert len(train_identities.intersection(unused_identities)) == 0, "Train/unused overlap!"
    
    split_info = {
        'seed': seed,
        'total_identities': n_total,
        'train_identities': len(train_identities),
        'test_identities': len(test_identities),
        'unused_identities': len(unused_identities),
        'train_images': len(train_idx),
        'test_images': len(test_idx),
        'unused_images': len(unused_idx),
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
    }
    
    return np.array(train_idx), np.array(test_idx), np.array(unused_idx), split_info
def preprocess_embeddings(X_train, Y_train, X_test, Y_test):
    """Preprocess embeddings: center and pad to same dimension."""
    Dx, Dy = X_train.shape[1], Y_train.shape[1]
    Dmax = max(Dx, Dy)
    
    def pad_to_dim(A, target_dim):
        if A.shape[1] == target_dim:
            return A
        return np.pad(A, ((0,0),(0,target_dim-A.shape[1])), mode="constant")
    
    # Center using TRAIN statistics
    mean_X = X_train.mean(0, keepdims=True)
    mean_Y = Y_train.mean(0, keepdims=True)
    
    X_train_proc = pad_to_dim(X_train - mean_X, Dmax)
    Y_train_proc = pad_to_dim(Y_train - mean_Y, Dmax)
    
    # Apply TRAIN statistics to test
    X_test_proc = pad_to_dim(X_test - mean_X, Dmax)
    Y_test_proc = pad_to_dim(Y_test - mean_Y, Dmax)
    
    return X_train_proc, Y_train_proc, X_test_proc, Y_test_proc


def row_norm(a):
    """Normalize rows to unit length"""
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def create_identity_labels(metadata):
    """Create numerical labels from identity strings"""
    identities = [rec['identity'] for rec in metadata]
    unique_identities = sorted(list(set(identities)))
    identity_to_label = {identity: i for i, identity in enumerate(unique_identities)}
    labels = np.array([identity_to_label[identity] for identity in identities])
    return labels


def evaluate_alignment_fast(A, B, labels_A, labels_B, max_rank=50):
    """Fast evaluation for identification."""
    print("    Computing similarity matrix...")
    S = cosine_similarity(row_norm(A), row_norm(B))
    
    print("    Sorting indices...")
    sorted_indices = np.argsort(-S, axis=1)
    sorted_labels = labels_B[sorted_indices]
    
    print("    Computing matches...")
    matches = (sorted_labels == labels_A[:, None])
    
    # Rank-1
    rank1 = np.sum(matches[:, 0]) / len(A) * 100
    
    # mAP
    matches_float = matches.astype(np.float32)
    n_relevant = np.sum(matches_float, axis=1)
    cumsum_matches = np.cumsum(matches_float, axis=1)
    ranks = np.arange(1, matches.shape[1] + 1)
    precisions = cumsum_matches / ranks
    precisions = precisions * matches_float
    ap = np.sum(precisions, axis=1) / np.maximum(n_relevant, 1e-10)
    valid_queries = n_relevant > 0
    map_score = np.mean(ap[valid_queries]) if np.sum(valid_queries) > 0 else 0.0
    
    return {
        'rank1': rank1,
        'map': map_score,
    }


# ======================================================
# SINGLE SEED DATA EFFICIENCY EXPERIMENT
# ======================================================

def run_data_efficiency_single_seed(modelX, modelY, metadata, X, Y, train_ratio, seed, config):
    """
    Run data efficiency experiment for a single seed and training ratio.
    """
    print(f"\n{'='*70}")
    print(f"SEED {seed} | TRAIN RATIO {train_ratio:.0%}: {modelX} → {modelY}")
    print(f"{'='*70}")
    
    # Split data
    train_idx, test_idx, unused_idx, split_info = split_by_identity_with_fixed_test(
        metadata, train_ratio=train_ratio, test_ratio=config['test_ratio'], seed=seed
    )
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    meta_test = metadata[test_idx]
    
    print(f"Training: {split_info['train_identities']} identities ({split_info['train_images']} images)")
    print(f"Testing: {split_info['test_identities']} identities ({split_info['test_images']} images)")
    print(f"Unused: {split_info['unused_identities']} identities ({split_info['unused_images']} images)")
    
    # Preprocess
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)
    
    # Create labels
    test_labels = create_identity_labels(meta_test)
    
    # Storage for results
    seed_results = {
        'seed': seed,
        'train_ratio': train_ratio,
        'split_info': split_info,
        'methods': {}
    }
    
    # Baseline (no alignment)
    print("\nBaseline (No Alignment)...")
    baseline_metrics = evaluate_alignment_fast(
        X_test_proc, Y_test_proc, test_labels, test_labels, max_rank=config['max_rank']
    )
    
    seed_results['methods']['Baseline'] = baseline_metrics
    print(f"  Rank-1: {baseline_metrics['rank1']:.2f}%, mAP: {baseline_metrics['map']:.3f}")
    
    # Alignment methods
    alignment_methods = [
        ProcrustesAlignment(),
        LinearAlignment(),
        RidgeAlignment(alpha=0.1),
    ]
    
    for method in alignment_methods:
        print(f"\n{method.name}...")
        
        try:
            # Train alignment
            method.fit(X_train_proc, Y_train_proc)
            
            # Transform test
            X_test_aligned = method.transform(X_test_proc)
            
            # Evaluate
            metrics = evaluate_alignment_fast(
                X_test_aligned, Y_test_proc, test_labels, test_labels, max_rank=config['max_rank']
            )
            
            seed_results['methods'][method.name] = metrics
            
            print(f"  Rank-1: {metrics['rank1']:.2f}% (Δ={metrics['rank1']-baseline_metrics['rank1']:+.2f}%)")
            print(f"  mAP: {metrics['map']:.3f} (Δ={metrics['map']-baseline_metrics['map']:+.3f})")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            seed_results['methods'][method.name] = {
                'rank1': 0.0, 'map': 0.0, 'error': str(e)
            }
    
    return seed_results


# ======================================================
# MULTI-SEED DATA EFFICIENCY EXPERIMENT
# ======================================================

def run_data_efficiency_experiment(modelX, modelY, config):
    """
    Run data efficiency experiment across multiple seeds and training ratios.
    """
    print(f"\n{'#'*70}")
    print(f"DATA EFFICIENCY: {modelX} → {modelY}")
    print(f"Testing with training ratios: {config['train_ratios']}")
    print(f"{'#'*70}")
    
    # Load data
    X = np.load(f"embeddings/{config['dataset']}/{modelX}.npy")
    Y = np.load(f"embeddings/{config['dataset']}/{modelY}.npy")
    metadata = np.load(config['meta_path'], allow_pickle=True)
    
    print(f"\n{modelX} shape: {X.shape}")
    print(f"{modelY} shape: {Y.shape}")
    
    # Storage for all results
    all_results = defaultdict(list)  # train_ratio -> [seed_results]
    
    # Run for each training ratio
    for train_ratio in config['train_ratios']:
        print(f"\n{'*'*70}")
        print(f"TRAINING RATIO: {train_ratio:.0%}")
        print(f"{'*'*70}")
        
        for seed in config['seeds']:
            seed_results = run_data_efficiency_single_seed(
                modelX, modelY, metadata, X, Y, train_ratio, seed, config
            )
            all_results[train_ratio].append(seed_results)
    
    # Aggregate results per training ratio
    aggregated_results = {}
    for train_ratio in config['train_ratios']:
        aggregated_results[train_ratio] = aggregate_results(all_results[train_ratio])
    
    return all_results, aggregated_results


def aggregate_results(seed_results_list):
    """Aggregate results across seeds for a single training ratio."""
    method_names = list(seed_results_list[0]['methods'].keys())
    
    aggregated = {}
    
    for method_name in method_names:
        metrics_across_seeds = defaultdict(list)
        
        for seed_result in seed_results_list:
            if method_name in seed_result['methods']:
                method_metrics = seed_result['methods'][method_name]
                for metric_name, value in method_metrics.items():
                    if metric_name != 'error':
                        metrics_across_seeds[metric_name].append(value)
        
        aggregated[method_name] = {}
        for metric_name, values in metrics_across_seeds.items():
            values = np.array(values)
            aggregated[method_name][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values.tolist(),
            }
    
    return aggregated


# ======================================================
# RESULTS EXPORT
# ======================================================

def save_data_efficiency_results(all_results, aggregated_results, modelX, modelY, config):
    """Save all data efficiency results."""
    out_dir = config['out_dir']
    prefix = f"{modelX}_to_{modelY}_{config['dataset']}"
    
    # Save raw results
    raw_path = f"{out_dir}/{prefix}_all_results.json"
    
    # Convert to serializable format
    serializable = {}
    for train_ratio, seed_results in all_results.items():
        serializable[str(train_ratio)] = seed_results
    
    with open(raw_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to: {raw_path}")
    
    # Save aggregated table
    table_path = f"{out_dir}/{prefix}_aggregated.csv"
    create_data_efficiency_table(aggregated_results, table_path, config)
    
    # Save plot
    plot_path = f"{out_dir}/{prefix}_plot.svg"
    plot_data_efficiency(aggregated_results, modelX, modelY, plot_path, config)


def create_data_efficiency_table(aggregated_results, save_path, config):
    """Create CSV table with results."""
    rows = []
    
    for train_ratio in config['train_ratios']:
        results = aggregated_results[train_ratio]
        
        for method_name, metrics in results.items():
            row = {
                'Train Ratio': f"{train_ratio:.0%}",
                'Method': method_name,
                'Rank-1': f"{metrics['rank1']['mean']:.2f} ± {metrics['rank1']['std']:.2f}",
                'mAP': f"{metrics['map']['mean']:.3f} ± {metrics['map']['std']:.3f}",
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Table saved to: {save_path}")
    
    print("\n" + "="*80)
    print("DATA EFFICIENCY RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def plot_data_efficiency(aggregated_results, modelX, modelY, save_path, config):
    """Plot performance vs training data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    train_ratios = config['train_ratios']
    methods = ['Baseline', 'Procrustes', 'Linear', 'Ridge(α=0.1)']
    colors = {'Baseline': 'gray', 'Procrustes': 'blue', 'Linear': 'green', 'Ridge(α=0.1)': 'red'}
    
    # Plot Rank-1
    for method in methods:
        means = [aggregated_results[tr][method]['rank1']['mean'] for tr in train_ratios]
        stds = [aggregated_results[tr][method]['rank1']['std'] for tr in train_ratios]
        
        ax1.plot(train_ratios, means, 'o-', label=method, color=colors[method], linewidth=2)
        ax1.fill_between(train_ratios, 
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, color=colors[method])
    
    ax1.set_xlabel('Training Data Ratio', fontsize=12)
    ax1.set_ylabel('Rank-1 Accuracy (%)', fontsize=12)
    ax1.set_title(f'{modelX} → {modelY} (Rank-1)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot mAP
    for method in methods:
        means = [aggregated_results[tr][method]['map']['mean'] for tr in train_ratios]
        stds = [aggregated_results[tr][method]['map']['std'] for tr in train_ratios]
        
        ax2.plot(train_ratios, means, 'o-', label=method, color=colors[method], linewidth=2)
        ax2.fill_between(train_ratios,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, color=colors[method])
    
    ax2.set_xlabel('Training Data Ratio', fontsize=12)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.set_title(f'{modelX} → {modelY} (mAP)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


# ======================================================
# MAIN
# ======================================================

def main():
    """Main execution."""
    print("="*70)
    print("DATA EFFICIENCY EXPERIMENTS - FACE-SPECIFIC MODELS")
    print("="*70)
    
    for modelX, modelY in DATA_EFFICIENCY_CONFIG['model_pairs']:
        all_results, aggregated_results = run_data_efficiency_experiment(
            modelX, modelY, DATA_EFFICIENCY_CONFIG
        )
        
        save_data_efficiency_results(
            all_results, aggregated_results, modelX, modelY, DATA_EFFICIENCY_CONFIG
        )
        
        print(f"\n✓ Completed: {modelX} → {modelY}")
    
    print("\n" + "="*70)
    print("ALL DATA EFFICIENCY EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {DATA_EFFICIENCY_CONFIG['out_dir']}")
    print("="*70)


if __name__ == "__main__":
    main()