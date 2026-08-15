import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from cfe.methods import *
from cfe.config import DEFAULT_PAIRS
import json

CROSS_DATASET_CONFIG = {
    'experiments': [
        # Train the alignment on CFP, evaluate on LFW (paper Table 3).
        {
            'name': 'CFP->LFW',
            'train_dataset': 'cfp',
            'test_dataset': 'lfw',
            'train_meta_path': 'embeddings/cfp/cfp_metadata.npy',
            'test_meta_path': 'embeddings/lfw/lfw_metadata.npy',
        },
    ],
    'model_pairs': DEFAULT_PAIRS,
    'out_dir': './results/cross_dataset_analysis/CFPtoLFW',
    'n_seeds': 1,
    'seeds': [42],
    'max_rank': 50,
}

os.makedirs(CROSS_DATASET_CONFIG['out_dir'], exist_ok=True)

# Save configuration
with open(f"{CROSS_DATASET_CONFIG['out_dir']}/cross_dataset_config.json", 'w') as f:
    json.dump(CROSS_DATASET_CONFIG, f, indent=2)

# UTILITY FUNCTIONS (reuse from main script)

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

def create_identity_labels(metadata):
    """Create numerical labels from identity strings"""
    identities = [rec['identity'] for rec in metadata]
    unique_identities = sorted(list(set(identities)))
    identity_to_label = {identity: i for i, identity in enumerate(unique_identities)}
    labels = np.array([identity_to_label[identity] for identity in identities])
    return labels



def row_norm(a):
    """Normalize rows to unit length"""
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def compute_all_metrics_fast(A, B, labels_A, labels_B, max_rank=50):
    """
    Compute all metrics in one pass to avoid recomputing similarity matrix.
    Memory-efficient for large datasets.
    """
    print("  Computing similarity matrix...")
    S = cosine_similarity(row_norm(A), row_norm(B))
    
    print("  Sorting indices...")
    sorted_indices = np.argsort(-S, axis=1)
    sorted_labels = labels_B[sorted_indices]
    
    print("  Computing matches...")
    matches = (sorted_labels == labels_A[:, None])
    
    # ===== Rank-1 =====
    rank1 = np.sum(matches[:, 0]) / len(A) * 100
    
    # ===== Top-k =====
    rank5 = np.sum(np.any(matches[:, :5], axis=1)) / len(A) * 100
    rank10 = np.sum(np.any(matches[:, :10], axis=1)) / len(A) * 100
    rank20 = np.sum(np.any(matches[:, :20], axis=1)) / len(A) * 100
    
    # ===== CMC =====
    match_positions = np.argmax(matches, axis=1)
    has_match = np.any(matches, axis=1)
    match_positions[~has_match] = max_rank
    
    cmc = np.zeros(max_rank)
    for rank in range(max_rank):
        cmc[rank] = np.sum(match_positions <= rank) / len(A)
    
    # ===== mAP =====
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
        'rank5': rank5,
        'rank10': rank10,
        'rank20': rank20,
        'map': map_score,
        'cmc': cmc,
    }


def compute_all_metrics_chunked(A, B, labels_A, labels_B, max_rank=50, chunk_size=15000):
    """
    Memory-efficient chunked computation with argpartition optimization.
    """
    print(f"  Input shapes: A={A.shape}, B={B.shape}")
    n_queries = len(A)
    n_gallery = len(B)
    n_chunks = (n_queries + chunk_size - 1) // chunk_size
    
    # Normalize once
    print("  Normalizing embeddings...")
    A_norm = row_norm(A)
    B_norm = row_norm(B)
    
    # Accumulators
    total_rank1 = 0
    total_rank5 = 0
    total_rank10 = 0
    total_rank20 = 0
    cmc_cumulative = np.zeros(max_rank)
    aps = []
    
    print(f"  Processing {n_queries} queries in {n_chunks} chunks...")
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_queries)
        chunk_size_actual = end_idx - start_idx
        
        # Process this chunk
        A_chunk = A_norm[start_idx:end_idx]
        labels_A_chunk = labels_A[start_idx:end_idx]
        
        # Compute similarity for this chunk only
        S_chunk = A_chunk @ B_norm.T
        
        # Use argpartition for speed (only need top-k, not full sort)
        k_needed = min(max_rank, n_gallery)
        
        if k_needed < n_gallery:
            # Partition to get top-k
            top_k_indices_unsorted = np.argpartition(-S_chunk, k_needed-1, axis=1)[:, :k_needed]
            
            # Sort just the top-k
            top_k_scores = np.take_along_axis(S_chunk, top_k_indices_unsorted, axis=1)
            sort_order = np.argsort(-top_k_scores, axis=1)
            sorted_indices = np.take_along_axis(top_k_indices_unsorted, sort_order, axis=1)
        else:
            sorted_indices = np.argsort(-S_chunk, axis=1)
        
        # Get labels
        sorted_labels = labels_B[sorted_indices]
        
        # Matches
        matches = (sorted_labels == labels_A_chunk[:, None])
        
        # Rank-1
        rank1_correct = np.sum(matches[:, 0])
        total_rank1 += rank1_correct
        
        # Top-k
        if k_needed >= 5:
            total_rank5 += np.sum(np.any(matches[:, :5], axis=1))
        if k_needed >= 10:
            total_rank10 += np.sum(np.any(matches[:, :10], axis=1))
        if k_needed >= 20:
            total_rank20 += np.sum(np.any(matches[:, :20], axis=1))
        
        match_positions = np.argmax(matches, axis=1)
        has_match = np.any(matches, axis=1)
        match_positions[~has_match] = max_rank
        
        for rank in range(max_rank):
            cmc_cumulative[rank] += np.sum(match_positions <= rank)
        
        # mAP
        matches_float = matches.astype(np.float32)
        n_relevant = np.sum(matches_float, axis=1)
        
        valid_mask = n_relevant > 0
        if np.sum(valid_mask) > 0:
            matches_valid = matches_float[valid_mask]
            cumsum_matches = np.cumsum(matches_valid, axis=1)
            ranks = np.arange(1, matches_valid.shape[1] + 1)
            precisions = cumsum_matches / ranks
            precisions = precisions * matches_valid
            ap = np.sum(precisions, axis=1) / n_relevant[valid_mask]
            aps.extend(ap.tolist())
        
        if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == n_chunks:
            print(f"    Processed {end_idx}/{n_queries} queries "
                  f"({end_idx/n_queries*100:.1f}%)")
    
    return {
        'rank1': (total_rank1 / n_queries) * 100,
        'rank5': (total_rank5 / n_queries) * 100,
        'rank10': (total_rank10 / n_queries) * 100,
        'rank20': (total_rank20 / n_queries) * 100,
        'map': np.mean(aps) if len(aps) > 0 else 0.0,
        'cmc': cmc_cumulative / n_queries,
    }


def evaluate_alignment(A, B, labels_A, labels_B, max_rank=50):
    """
    Smart evaluation wrapper that chooses fast or chunked based on size.
    
    Thresholds:
    - < 10k images: Fast vectorized (loads full similarity matrix)
    - >= 10k images: Chunked with argpartition (memory-efficient)
    """
    n_queries = len(A)
    n_gallery = len(B)
    
    # Estimate memory usage for similarity matrix
    similarity_matrix_gb = (n_queries * n_gallery * 4) / (1024**3)  # 4 bytes per float32
    
    print(f"  Dataset size: {n_queries} queries × {n_gallery} gallery")
    print(f"  Estimated similarity matrix size: {similarity_matrix_gb:.2f} GB")
    
    # Use chunked if dataset is large or similarity matrix > 2GB
    if n_queries >= 15000 or True:
        print("  → Using CHUNKED computation (memory-efficient)")
        return compute_all_metrics_chunked(A, B, labels_A, labels_B, max_rank, chunk_size=15000)
    else:
        print("  → Using FAST vectorized computation")
        return compute_all_metrics_fast(A, B, labels_A, labels_B, max_rank)


def run_cross_dataset_single_seed(modelX, modelY, exp_config, seed, config):
    """
    Run cross-dataset experiment for a single seed.
    
    Train on one dataset, test on another.
    """
    print(f"\n{'='*70}")
    print(f"SEED {seed}: {exp_config['name']} | {modelX} → {modelY}")
    print(f"{'='*70}")
    
    train_dataset = exp_config['train_dataset']
    test_dataset = exp_config['test_dataset']
    
    # Load training data (entire dataset)
    print(f"Loading TRAIN data from {train_dataset}...")
    X_train = np.load(f"embeddings/{train_dataset}/{modelX}.npy")
    Y_train = np.load(f"embeddings/{train_dataset}/{modelY}.npy")
    train_metadata = np.load(exp_config['train_meta_path'], allow_pickle=True)
    
    print(f"  Train: {len(X_train)} images from {train_dataset}")
    
    # Load test data (entire dataset)
    print(f"Loading TEST data from {test_dataset}...")
    X_test = np.load(f"embeddings/{test_dataset}/{modelX}.npy")
    Y_test = np.load(f"embeddings/{test_dataset}/{modelY}.npy")
    test_metadata = np.load(exp_config['test_meta_path'], allow_pickle=True)
    
    print(f"  Test: {len(X_test)} images from {test_dataset}")
    
    # Create identity labels
    test_labels = create_identity_labels(test_metadata)
    
    # Preprocess embeddings
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)
    
    # Define alignment methods
    alignment_methods = [
        ProcrustesAlignment(),
        LinearAlignment(),
        RidgeAlignment(alpha=0.1),
        # RidgeAlignment(alpha=1.0),
        # RidgeAlignment(alpha=10.0),
    ]
    
    # Storage for results
    seed_results = {
        'seed': seed,
        'experiment': exp_config['name'],
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'methods': {}
    }
    
    # Baseline (no alignment)
    print("\nBaseline (No Alignment)...")
    baseline_metrics = evaluate_alignment(
        X_test_proc, Y_test_proc,
        test_labels, test_labels,
        max_rank=config['max_rank']
    )
    
    seed_results['methods']['Baseline'] = baseline_metrics
    print(f"  Rank-1: {baseline_metrics['rank1']:.3f}%, mAP: {baseline_metrics['map']:.3f}")
    
    # Run each alignment method
    for method in alignment_methods:
        print(f"\n{method.name}...")
        
        try:
            # Train alignment on TRAIN dataset
            method.fit(X_train_proc, Y_train_proc)
            
            # Transform TEST dataset
            X_test_aligned = method.transform(X_test_proc)
            
            # Evaluate on TEST dataset
            metrics = evaluate_alignment(
                X_test_aligned, Y_test_proc,
                test_labels, test_labels,
                max_rank=config['max_rank']
            )
            
            seed_results['methods'][method.name] = metrics
            
            print(f"  Rank-1: {metrics['rank1']:.3f}% (Δ={metrics['rank1']-baseline_metrics['rank1']:+.3f}%)")
            print(f"  mAP: {metrics['map']:.3f} (Δ={metrics['map']-baseline_metrics['map']:+.3f})")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            seed_results['methods'][method.name] = {
                'rank1': 0.0, 'rank5': 0.0, 'rank10': 0.0,
                'rank20': 0.0, 'map': 0.0, 'error': str(e)
            }
    
    return seed_results



def run_cross_dataset_multi_seed(modelX, modelY, exp_config, config):
    """
    Run cross-dataset experiment across multiple seeds.
    """
    print(f"\n{'#'*70}")
    print(f"CROSS-DATASET: {exp_config['name']} | {modelX} → {modelY}")
    print(f"Running {config['n_seeds']} experiments with seeds: {config['seeds']}")
    print(f"{'#'*70}")
    
    # Run experiment for each seed
    all_results = []
    for seed in config['seeds']:
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        seed_results = run_cross_dataset_single_seed(
            modelX, modelY, exp_config, seed, config
        )
        all_results.append(seed_results)
    
    # Aggregate results
    aggregated_results = aggregate_multi_seed_results(all_results)
    
    return all_results, aggregated_results


def aggregate_multi_seed_results(all_results):
    """Aggregate results across multiple seeds."""
    method_names = list(all_results[0]['methods'].keys())
    
    aggregated = {}
    
    for method_name in method_names:
        metrics_across_seeds = defaultdict(list)
        
        for seed_result in all_results:
            if method_name in seed_result['methods']:
                method_metrics = seed_result['methods'][method_name]
                for metric_name, value in method_metrics.items():
                    if metric_name != 'cmc' and metric_name != 'error':
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



def create_results_table(aggregated_results, save_path):
    """Create and save results table with mean ± std."""
    rows = []
    
    for method_name, metrics in aggregated_results.items():
        row = {'Method': method_name}
        
        for metric_name in ['rank1', 'rank5', 'rank10', 'rank20', 'map']:
            if metric_name in metrics:
                mean = metrics[metric_name]['mean']
                std = metrics[metric_name]['std']
                
                if metric_name == 'map':
                    row[metric_name.upper()] = f"{mean:.3f} ± {std:.3f}"
                else:
                    row[metric_name.upper()] = f"{mean:.3f} ± {std:.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df[['Method', 'RANK1', 'RANK5', 'RANK10', 'RANK20', 'MAP']]
    
    df.to_csv(save_path, index=False)
    print(f"\nResults table saved to: {save_path}")
    
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (Mean ± Std across all seeds)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def plot_cmc_curves_aggregated(all_results, aggregated_results, exp_name, 
                                modelX, modelY, save_path):
    """Plot CMC curves with mean and confidence intervals."""
    method_names = list(aggregated_results.keys())
    max_rank = len(all_results[0]['methods']['Baseline']['cmc'])
    
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    
    for i, method_name in enumerate(method_names):
        cmc_curves = []
        for seed_result in all_results:
            if method_name in seed_result['methods']:
                if 'cmc' in seed_result['methods'][method_name]:
                    cmc_curves.append(seed_result['methods'][method_name]['cmc'])
        
        if len(cmc_curves) == 0:
            continue
        
        cmc_curves = np.array(cmc_curves) * 100
        cmc_mean = np.mean(cmc_curves, axis=0)
        cmc_std = np.std(cmc_curves, axis=0)
        
        ranks = np.arange(1, max_rank + 1)
        
        linestyle = '--' if method_name == 'Baseline' else '-'
        linewidth = 2.5 if method_name == 'Baseline' else 1.5
        
        plt.plot(ranks, cmc_mean, linestyle=linestyle, linewidth=linewidth,
                label=method_name, color=colors[i])
        
        plt.fill_between(ranks, cmc_mean - cmc_std, cmc_mean + cmc_std,
                        alpha=0.2, color=colors[i])
    
    plt.xlabel('Rank', fontsize=13)
    plt.ylabel('Recognition Rate (%)', fontsize=13)
    plt.title(f'Cross-Dataset CMC: {exp_name} | {modelX} → {modelY}\n'
              f'(Mean ± Std across {len(all_results)} seeds)',
              fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.xlim([1, max_rank])
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"CMC curve saved to: {save_path}")
    plt.close()


def save_cross_dataset_results(all_results, aggregated_results, 
                                exp_name, modelX, modelY, config):
    """Save all cross-dataset results."""
    out_dir = config['out_dir']
    prefix = f"{exp_name.replace('→', '_to_')}_{modelX}_to_{modelY}"
    
    # Save raw results
    raw_results_path = f"{out_dir}/{prefix}_all_seeds.json"
    
    serializable_results = []
    for seed_result in all_results:
        result_copy = seed_result.copy()
        result_copy['methods'] = {}
        for method_name, metrics in seed_result['methods'].items():
            method_copy = metrics.copy()
            if 'cmc' in method_copy:
                method_copy['cmc'] = method_copy['cmc'].tolist()
            result_copy['methods'][method_name] = method_copy
        serializable_results.append(result_copy)
    
    with open(raw_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Raw results saved to: {raw_results_path}")
    
    # Save aggregated table
    table_path = f"{out_dir}/{prefix}_results.csv"
    create_results_table(aggregated_results, table_path)
    
    # Save CMC curves
    # cmc_path = f"{out_dir}/{prefix}_cmc.png"
    # plot_cmc_curves_aggregated(all_results, aggregated_results,
    #                            exp_name, modelX, modelY, cmc_path)



def create_comparison_table(all_experiment_results, config):
    """
    Create a comparison table showing performance across all 
    cross-dataset experiments.
    """
    rows = []
    
    for exp_result in all_experiment_results:
        exp_name = exp_result['exp_name']
        model_pair = exp_result['model_pair']
        aggregated = exp_result['aggregated_results']
        
        for method_name, metrics in aggregated.items():
            row = {
                'Experiment': exp_name,
                'Model Pair': f"{model_pair[0]}→{model_pair[1]}",
                'Method': method_name,
                'RANK1': f"{metrics['rank1']['mean']:.3f} ± {metrics['rank1']['std']:.3f}",
                'RANK5': f"{metrics['rank5']['mean']:.3f} ± {metrics['rank5']['std']:.3f}",
                'MAP': f"{metrics['map']['mean']:.3f} ± {metrics['map']['std']:.3f}",
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    comparison_path = f"{config['out_dir']}/all_cross_dataset_comparison.csv"
    df.to_csv(comparison_path, index=False)
    
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"\nComparison table saved to: {comparison_path}")
    
    return df



def main():
    """Main execution function for cross-dataset experiments."""
    print("="*70)
    print("CROSS-DATASET GENERALIZATION EXPERIMENTS")
    print("="*70)
    
    all_experiment_results = []
    
    # Run each cross-dataset experiment
    for exp_config in CROSS_DATASET_CONFIG['experiments']:
        print(f"\n{'*'*70}")
        print(f"EXPERIMENT: {exp_config['name']}")
        print(f"Train on {exp_config['train_dataset']}, Test on {exp_config['test_dataset']}")
        print(f"{'*'*70}")
        
        # Run for each model pair
        for modelX, modelY in CROSS_DATASET_CONFIG['model_pairs']:
            all_results, aggregated_results = run_cross_dataset_multi_seed(
                modelX, modelY, exp_config, CROSS_DATASET_CONFIG
            )
            
            # Save results
            save_cross_dataset_results(
                all_results, aggregated_results,
                exp_config['name'], modelX, modelY,
                CROSS_DATASET_CONFIG
            )
            
            # Store for comparison table
            all_experiment_results.append({
                'exp_name': exp_config['name'],
                'model_pair': (modelX, modelY),
                'all_results': all_results,
                'aggregated_results': aggregated_results,
            })
            
            print(f"\n✓ Completed: {exp_config['name']} | {modelX} → {modelY}")
    
    # Create comparison table across all experiments
    create_comparison_table(all_experiment_results, CROSS_DATASET_CONFIG)
    
    print("\n" + "="*70)
    print("ALL CROSS-DATASET EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {CROSS_DATASET_CONFIG['out_dir']}")
    print("="*70)


if __name__ == "__main__":
    main()
