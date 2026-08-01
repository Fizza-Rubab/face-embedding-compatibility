import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from cfe.methods import *
from cfe.config import DEFAULT_PAIRS
import json

# ======================================================
# CROSS-DATASET VERIFICATION CONFIGURATION
# ======================================================
CROSS_VERIFICATION_CONFIG = {
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
    'out_dir': './results/cross_dataset_verification_cfp_lfw',
    'n_seeds': 5,
    'seeds': [42, 123, 217, 7, 531],
    
    # Sampling parameters
    'n_genuine_pairs': 10000,   # Number of genuine pairs to sample
    'n_impostor_pairs': 10000,  # Number of impostor pairs to sample (balanced)
}

os.makedirs(CROSS_VERIFICATION_CONFIG['out_dir'], exist_ok=True)

# Save configuration
with open(f"{CROSS_VERIFICATION_CONFIG['out_dir']}/config.json", 'w') as f:
    json.dump(CROSS_VERIFICATION_CONFIG, f, indent=2)

# ======================================================
# UTILITY FUNCTIONS
# ======================================================

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
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


# ======================================================
# SAMPLED VERIFICATION PAIR GENERATION
# ======================================================

def sample_verification_pairs(metadata_test, n_genuine, n_impostor):
    """
    Sample a fixed number of balanced genuine and impostor pairs from test set.
    
    Uses current numpy random state (set by experiment seed).
    
    Args:
        metadata_test: Test set metadata
        n_genuine: Number of genuine pairs to sample
        n_impostor: Number of impostor pairs to sample
    
    Returns:
        pairs: List of (idx1, idx2, label) where label=1 for genuine, 0 for impostor
    """
    # Group images by identity
    identity_to_indices = defaultdict(list)
    for idx, rec in enumerate(metadata_test):
        identity_to_indices[rec['identity']].append(idx)
    
    all_indices = np.arange(len(metadata_test))
    
    pairs = []
    
    # ========================================
    # SAMPLE GENUINE PAIRS
    # ========================================
    print(f"  Sampling {n_genuine:,} genuine pairs...")
    
    # Collect all possible genuine pairs
    all_genuine_pairs = []
    for identity, indices in identity_to_indices.items():
        if len(indices) >= 2:
            # All combinations for this identity
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    all_genuine_pairs.append((indices[i], indices[j], 1))
    
    total_genuine_available = len(all_genuine_pairs)
    print(f"    Total genuine pairs available: {total_genuine_available:,}")
    
    if total_genuine_available < n_genuine:
        print(f"    ⚠ Warning: Only {total_genuine_available:,} genuine pairs available "
              f"(requested {n_genuine:,}). Using all available.")
        sampled_genuine = all_genuine_pairs
    else:
        # Randomly sample n_genuine pairs
        sampled_indices = np.random.choice(len(all_genuine_pairs), 
                                          size=n_genuine, replace=False)
        sampled_genuine = [all_genuine_pairs[i] for i in sampled_indices]
    
    pairs.extend(sampled_genuine)
    print(f"    Sampled {len(sampled_genuine):,} genuine pairs")
    
    # ========================================
    # SAMPLE IMPOSTOR PAIRS
    # ========================================
    print(f"  Sampling {n_impostor:,} impostor pairs...")
    
    # Build identity lookup
    idx_to_identity = {}
    for identity, indices in identity_to_indices.items():
        for idx in indices:
            idx_to_identity[idx] = identity
    
    impostor_pairs_set = set()
    n_attempts = 0
    max_attempts = n_impostor * 10  # Safety limit
    
    while len(impostor_pairs_set) < n_impostor and n_attempts < max_attempts:
        # Randomly sample two images
        idx1, idx2 = np.random.choice(all_indices, size=2, replace=False)
        
        # Check if different identities
        if idx_to_identity[idx1] != idx_to_identity[idx2]:
            # Add in canonical order (smaller index first)
            pair = tuple(sorted([idx1, idx2]))
            impostor_pairs_set.add(pair)
        
        n_attempts += 1
    
    # Convert set to list with label=0
    for idx1, idx2 in impostor_pairs_set:
        pairs.append((idx1, idx2, 0))
    
    n_impostor_sampled = len(impostor_pairs_set)
    print(f"    Sampled {n_impostor_sampled:,} impostor pairs")
    
    if n_impostor_sampled < n_impostor:
        print(f"    ⚠ Warning: Could only generate {n_impostor_sampled} impostor pairs "
              f"(wanted {n_impostor}). May need more test images.")
    
    # Shuffle pairs
    np.random.shuffle(pairs)
    
    print(f"  Final: {len(sampled_genuine):,} genuine + {n_impostor_sampled:,} impostor "
          f"= {len(pairs):,} total pairs")
    print(f"  Balance ratio: {len(sampled_genuine)/max(n_impostor_sampled,1):.2f}:1")
    
    return pairs


# ======================================================
# VERIFICATION METRICS
# ======================================================

def compute_verification_metrics(X, Y, pairs):
    """
    Compute ROC, EER, and TMR@FMR for verification task.
    
    Args:
        X, Y: Normalized embeddings from two models (already aligned if applicable)
        pairs: List of (idx1, idx2, label)
    
    Returns:
        Dictionary with all metrics
    """
    print("    Computing similarities for all pairs...")
    
    # Compute similarity for all pairs
    similarities = []
    labels = []
    
    for idx1, idx2, label in pairs:
        sim = np.dot(X[idx1], Y[idx2])
        similarities.append(sim)
        labels.append(label)
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    print(f"    Processed {len(pairs)} pairs")
    
    # Separate genuine and impostor scores
    genuine_scores = similarities[labels == 1]
    impostor_scores = similarities[labels == 0]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
    auc = roc_auc_score(labels, similarities)
    
    # Compute EER
    fnmr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnmr))
    eer = (fpr[eer_idx] + fnmr[eer_idx]) / 2
    
    # Compute TMR at specific FMR operating points
    tmr_at_fmr = {}
    for target_fmr in [0.0001, 0.001, 0.01, 0.1]:
        idx = np.nanargmin(np.abs(fpr - target_fmr))
        tmr_at_fmr[target_fmr] = tpr[idx] * 100
    
    return {
        'auc': auc,
        'eer': eer * 100,
        'tmr_at_fmr_0.01': tmr_at_fmr[0.0001],
        'tmr_at_fmr_0.1': tmr_at_fmr[0.001],
        'tmr_at_fmr_1': tmr_at_fmr[0.01],
        'tmr_at_fmr_10': tmr_at_fmr[0.1],
        'n_genuine': len(genuine_scores),
        'n_impostor': len(impostor_scores),
    }


# ======================================================
# SINGLE SEED CROSS-DATASET VERIFICATION
# ======================================================

def run_cross_verification_single_seed(modelX, modelY, exp_config, seed, config):
    """
    Run cross-dataset verification for a single random seed.
    
    Train alignment on source dataset, test on sampled pairs from target dataset.
    """
    print(f"\n{'='*70}")
    print(f"SEED {seed}: {exp_config['name']} | {modelX} → {modelY}")
    print(f"{'='*70}")
    
    # Set random seed for reproducible sampling
    np.random.seed(seed)
    
    train_dataset = exp_config['train_dataset']
    test_dataset = exp_config['test_dataset']
    
    # Load training data (entire dataset)
    print(f"Loading TRAIN data from {train_dataset}...")
    X_train = np.load(f"embeddings/{train_dataset}/{modelX}.npy")
    Y_train = np.load(f"embeddings/{train_dataset}/{modelY}.npy")
    
    # Load test data (entire dataset)
    print(f"Loading TEST data from {test_dataset}...")
    X_test = np.load(f"embeddings/{test_dataset}/{modelX}.npy")
    Y_test = np.load(f"embeddings/{test_dataset}/{modelY}.npy")
    test_metadata = np.load(exp_config['test_meta_path'], allow_pickle=True)
    
    print(f"  Train: {len(X_train)} images from {train_dataset}")
    print(f"  Test: {len(X_test)} images from {test_dataset}")
    
    # Preprocess embeddings
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)
    
    # Normalize for verification
    X_test_norm = row_norm(X_test_proc)
    Y_test_norm = row_norm(Y_test_proc)
    
    # Sample verification pairs from test set
    print(f"\nSampling verification pairs from {test_dataset}...")
    pairs = sample_verification_pairs(
        test_metadata,
        n_genuine=config['n_genuine_pairs'],
        n_impostor=config['n_impostor_pairs']
    )
    
    # Storage for results
    seed_results = {
        'seed': seed,
        'experiment': exp_config['name'],
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'n_pairs': len(pairs),
        'n_genuine': config['n_genuine_pairs'],
        'n_impostor': config['n_impostor_pairs'],
        'methods': {}
    }
    
    # Baseline (no alignment)
    print("\nBaseline (No Alignment)...")
    baseline_metrics = compute_verification_metrics(X_test_norm, Y_test_norm, pairs)
    
    seed_results['methods']['Baseline'] = {
        'auc': baseline_metrics['auc'],
        'eer': baseline_metrics['eer'],
        'tmr_at_fmr_0.01': baseline_metrics['tmr_at_fmr_0.01'],
        'tmr_at_fmr_0.1': baseline_metrics['tmr_at_fmr_0.1'],
        'tmr_at_fmr_1': baseline_metrics['tmr_at_fmr_1'],
        'tmr_at_fmr_10': baseline_metrics['tmr_at_fmr_10'],
    }
    
    print(f"  AUC: {baseline_metrics['auc']:.4f}")
    print(f"  EER: {baseline_metrics['eer']:.2f}%")
    print(f"  TMR@FMR=1%: {baseline_metrics['tmr_at_fmr_1']:.2f}%")
    
    # With alignment
    alignment_methods = [
        ProcrustesAlignment(),
        LinearAlignment(),
        RidgeAlignment(alpha=0.1),
    ]
    
    for method in alignment_methods:
        print(f"\n{method.name}...")
        
        try:
            # Train alignment on TRAIN dataset (full CFP)
            print("  Training alignment...")
            method.fit(X_train_proc, Y_train_proc)
            
            # Transform TEST embeddings
            print("  Transforming test embeddings...")
            X_test_aligned = method.transform(X_test_proc)
            X_test_aligned_norm = row_norm(X_test_aligned)
            
            # Evaluate on sampled pairs
            metrics = compute_verification_metrics(X_test_aligned_norm, Y_test_norm, pairs)
            
            seed_results['methods'][method.name] = {
                'auc': metrics['auc'],
                'eer': metrics['eer'],
                'tmr_at_fmr_0.01': metrics['tmr_at_fmr_0.01'],
                'tmr_at_fmr_0.1': metrics['tmr_at_fmr_0.1'],
                'tmr_at_fmr_1': metrics['tmr_at_fmr_1'],
                'tmr_at_fmr_10': metrics['tmr_at_fmr_10'],
            }
            
            print(f"  AUC: {metrics['auc']:.4f} (Δ={metrics['auc']-baseline_metrics['auc']:+.4f})")
            print(f"  EER: {metrics['eer']:.2f}% (Δ={metrics['eer']-baseline_metrics['eer']:+.2f}%)")
            print(f"  TMR@FMR=1%: {metrics['tmr_at_fmr_1']:.2f}% "
                  f"(Δ={metrics['tmr_at_fmr_1']-baseline_metrics['tmr_at_fmr_1']:+.2f}%)")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            seed_results['methods'][method.name] = {
                'auc': 0.0,
                'eer': 100.0,
                'error': str(e)
            }
    
    return seed_results


# ======================================================
# MULTI-SEED EXPERIMENT
# ======================================================

def run_cross_verification_multi_seed(modelX, modelY, exp_config, config):
    """
    Run cross-dataset verification across multiple random seeds.
    """
    print(f"\n{'#'*70}")
    print(f"CROSS-DATASET VERIFICATION: {exp_config['name']} | {modelX} → {modelY}")
    print(f"Running {config['n_seeds']} experiments with seeds: {config['seeds']}")
    print(f"{'#'*70}")
    
    # Run experiment for each seed
    all_results = []
    for seed in config['seeds']:
        seed_results = run_cross_verification_single_seed(
            modelX, modelY, exp_config, seed, config
        )
        all_results.append(seed_results)
    
    # Aggregate results
    aggregated_results = aggregate_verification_results(all_results)
    
    return all_results, aggregated_results


def aggregate_verification_results(all_results):
    """Aggregate verification results across multiple seeds."""
    method_names = list(all_results[0]['methods'].keys())
    
    aggregated = {}
    
    for method_name in method_names:
        metrics_across_seeds = defaultdict(list)
        
        for seed_result in all_results:
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

def create_verification_table(aggregated_results, save_path):
    """Create and save verification results table."""
    rows = []
    
    for method_name, metrics in aggregated_results.items():
        row = {'Method': method_name}
        
        for metric_name in ['auc', 'eer', 'tmr_at_fmr_0.01', 'tmr_at_fmr_0.1',
                           'tmr_at_fmr_1', 'tmr_at_fmr_10']:
            if metric_name in metrics:
                mean = metrics[metric_name]['mean']
                std = metrics[metric_name]['std']
                
                if metric_name == 'auc':
                    row['AUC'] = f"{mean:.4f} ± {std:.4f}"
                elif metric_name == 'eer':
                    row['EER (%)'] = f"{mean:.2f} ± {std:.2f}"
                elif metric_name == 'tmr_at_fmr_0.01':
                    row['TMR@0.01%'] = f"{mean:.2f} ± {std:.2f}"
                elif metric_name == 'tmr_at_fmr_0.1':
                    row['TMR@0.1%'] = f"{mean:.2f} ± {std:.2f}"
                elif metric_name == 'tmr_at_fmr_1':
                    row['TMR@1%'] = f"{mean:.2f} ± {std:.2f}"
                elif metric_name == 'tmr_at_fmr_10':
                    row['TMR@10%'] = f"{mean:.2f} ± {std:.2f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    cols = ['Method', 'AUC', 'EER (%)', 'TMR@0.01%', 'TMR@0.1%', 'TMR@1%', 'TMR@10%']
    df = df[[c for c in cols if c in df.columns]]
    
    df.to_csv(save_path, index=False)
    print(f"\nVerification results table saved to: {save_path}")
    
    print("\n" + "="*80)
    print("AGGREGATED VERIFICATION RESULTS (Mean ± Std)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def save_cross_verification_results(all_results, aggregated_results,
                                    exp_name, modelX, modelY, config):
    """Save all cross-dataset verification results."""
    out_dir = config['out_dir']
    prefix = f"{exp_name.replace('→', '_to_')}_{modelX}_to_{modelY}"
    
    # Save raw results
    raw_results_path = f"{out_dir}/{prefix}_verification_all_seeds.json"
    with open(raw_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to: {raw_results_path}")
    
    # Save aggregated table
    table_path = f"{out_dir}/{prefix}_verification_results.csv"
    create_verification_table(aggregated_results, table_path)


# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    """Main execution function."""
    print("="*70)
    print("CROSS-DATASET VERIFICATION EXPERIMENTS")
    print("="*70)
    
    for exp_config in CROSS_VERIFICATION_CONFIG['experiments']:
        print(f"\n{'*'*70}")
        print(f"EXPERIMENT: {exp_config['name']}")
        print(f"Train on {exp_config['train_dataset']}, Test on {exp_config['test_dataset']}")
        print(f"Sampling {CROSS_VERIFICATION_CONFIG['n_genuine_pairs']:,} genuine + "
              f"{CROSS_VERIFICATION_CONFIG['n_impostor_pairs']:,} impostor pairs per seed")
        print(f"{'*'*70}")
        
        for modelX, modelY in CROSS_VERIFICATION_CONFIG['model_pairs']:
            all_results, aggregated_results = run_cross_verification_multi_seed(
                modelX, modelY, exp_config, CROSS_VERIFICATION_CONFIG
            )
            
            save_cross_verification_results(
                all_results, aggregated_results,
                exp_config['name'], modelX, modelY,
                CROSS_VERIFICATION_CONFIG
            )
            
            print(f"\n✓ Completed: {exp_config['name']} | {modelX} → {modelY}")
    
    print("\n" + "="*70)
    print("ALL CROSS-DATASET VERIFICATION EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {CROSS_VERIFICATION_CONFIG['out_dir']}")
    print("="*70)


if __name__ == "__main__":
    main()