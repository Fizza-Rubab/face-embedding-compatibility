import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from cfe.methods import *
from cfe.config import DEFAULT_PAIRS, DATASET_IMG_DIRS
import json

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    # 'dataset': 'cfp',
    # 'img_dir': "./data/chinafax/cfpw-dataset/cfp-dataset/Data/Images",
    # 'meta_path': "./embeddings/cfp/cfp_metadata.npy",
    # 'out_dir': "./results/cfp_verification",

    'dataset': 'lfw',
    'img_dir': DATASET_IMG_DIRS['lfw'],
    'meta_path': "./embeddings/lfw/lfw_metadata.npy",
    'out_dir': "./results/lfw_verification",

    'model_pairs': DEFAULT_PAIRS,
    'train_ratio': 0.7,
    'n_seeds': 5,
    'seeds': [42, 123, 217, 7, 531],
}

os.makedirs(CONFIG['out_dir'], exist_ok=True)

# Save configuration
with open(f"{CONFIG['out_dir']}/verification_config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("Verification Experiment Configuration:")
print(json.dumps(CONFIG, indent=2))

# ======================================================
# UTILITY FUNCTIONS (SAME AS IDENTIFICATION)
# ======================================================

def split_by_identity(metadata, train_ratio=0.7, seed=42):
    """
    Split dataset by identities (not images).
    IDENTICAL to identification experiment.
    """
    np.random.seed(seed)
    
    # Group image indices by identity
    identity_to_indices = defaultdict(list)
    for idx, rec in enumerate(metadata):
        identity_to_indices[rec['identity']].append(idx)
    
    # Get unique identities and shuffle
    identities = sorted(identity_to_indices.keys())
    n_train = int(len(identities) * train_ratio)
    
    identities_shuffled = identities.copy()
    np.random.shuffle(identities_shuffled)
    
    train_identities = set(identities_shuffled[:n_train])
    test_identities = set(identities_shuffled[n_train:])
    
    # Get all image indices for train/test identities
    train_idx = []
    test_idx = []
    
    for identity, indices in identity_to_indices.items():
        if identity in train_identities:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)
    
    # Verify no overlap
    assert len(train_identities.intersection(test_identities)) == 0, \
        "Identity overlap detected!"
    
    split_info = {
        'seed': seed,
        'total_identities': len(identities),
        'train_identities': len(train_identities),
        'test_identities': len(test_identities),
        'train_images': len(train_idx),
        'test_images': len(test_idx),
    }
    
    return np.array(train_idx), np.array(test_idx), split_info


def preprocess_embeddings(X_train, Y_train, X_test, Y_test):
    """
    Preprocess embeddings: center and pad to same dimension.
    IDENTICAL to identification experiment.
    """
    Dx, Dy = X_train.shape[1], Y_train.shape[1]
    Dmax = max(Dx, Dy)
    
    def pad_to_dim(A, target_dim):
        if A.shape[1] == target_dim:
            return A
        return np.pad(A, ((0,0),(0,target_dim-A.shape[1])), mode="constant")
    
    # Center training data
    mean_X = X_train.mean(0, keepdims=True)
    mean_Y = Y_train.mean(0, keepdims=True)
    
    X_train_centered = X_train - mean_X
    Y_train_centered = Y_train - mean_Y
    
    # Pad to same dimension
    X_train_proc = pad_to_dim(X_train_centered, Dmax)
    Y_train_proc = pad_to_dim(Y_train_centered, Dmax)
    
    # Apply same preprocessing to test (using train statistics)
    X_test_centered = X_test - mean_X
    Y_test_centered = Y_test - mean_Y
    
    X_test_proc = pad_to_dim(X_test_centered, Dmax)
    Y_test_proc = pad_to_dim(Y_test_centered, Dmax)
    
    stats = {
        'dim_X': Dx,
        'dim_Y': Dy,
        'dim_max': Dmax,
        'mean_X': mean_X,
        'mean_Y': mean_Y,
    }
    
    return X_train_proc, Y_train_proc, X_test_proc, Y_test_proc, stats


def row_norm(a):
    """Normalize rows to unit length"""
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


# ======================================================
# VERIFICATION PAIR GENERATION FROM TEST SET
# ======================================================

def create_balanced_verification_pairs_from_test(metadata_test):
    identity_to_indices = defaultdict(list)
    for idx, rec in enumerate(metadata_test):
        identity_to_indices[rec['identity']].append(idx)
    
    all_indices = np.arange(len(metadata_test))
    all_identities = list(identity_to_indices.keys())
    
    pairs = []
    
    # ========================================
    # GENUINE PAIRS: All combinations within same identity (exhaustive)
    # ========================================
    print("  Generating genuine pairs (exhaustive)...")
    
    for identity, indices in identity_to_indices.items():
        if len(indices) >= 2:
            # All combinations for this identity
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    pairs.append((indices[i], indices[j], 1))
    
    n_genuine = len(pairs)
    print(f"    Generated {n_genuine:,} genuine pairs")
    
    # ========================================
    # IMPOSTOR PAIRS: Sample to match genuine count (balanced)
    # ========================================
    print(f"  Sampling {n_genuine:,} impostor pairs (balanced)...")
    
    # Build identity lookup for fast access
    idx_to_identity = {}
    for identity, indices in identity_to_indices.items():
        for idx in indices:
            idx_to_identity[idx] = identity
    
    impostor_pairs_set = set()  # Use set to avoid duplicates
    n_attempts = 0
    max_attempts = n_genuine * 10  # Safety limit
    
    while len(impostor_pairs_set) < n_genuine and n_attempts < max_attempts:
        # Randomly sample two images (uses global numpy random state)
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
    
    n_impostor = len(impostor_pairs_set)
    print(f"    Generated {n_impostor:,} impostor pairs")
    
    if n_impostor < n_genuine:
        print(f"    ⚠ Warning: Could only generate {n_impostor} impostor pairs "
              f"(wanted {n_genuine}). May need more test identities.")
    
    # Shuffle pairs (uses global numpy random state)
    np.random.shuffle(pairs)
    
    print(f"  Final: {n_genuine:,} genuine + {n_impostor:,} impostor = {len(pairs):,} total pairs")
    print(f"  Balance ratio: {n_genuine/max(n_impostor,1):.2f}:1 (target: 1.00:1)")
    
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
    print("    Computing ROC curve...")
    fpr, tpr, thresholds = roc_curve(labels, similarities, pos_label=1)
    # fpr = FMR (False Match Rate), tpr = TMR (True Match Rate)
    
    auc = roc_auc_score(labels, similarities)
    
    # Compute EER
    fnmr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnmr))
    eer = (fpr[eer_idx] + fnmr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    # Compute TMR at specific FMR operating points
    print("    Computing TMR@FMR...")
    tmr_at_fmr = {}
    for target_fmr in [0.0001, 0.001, 0.01, 0.1]:  # 0.01%, 0.1%, 1%, 10%
        # Find threshold where FMR is closest to target
        idx = np.nanargmin(np.abs(fpr - target_fmr))
        tmr_at_fmr[target_fmr] = tpr[idx] * 100  # Convert to percentage
    
    return {
        'fmr': fpr,
        'tmr': tpr,
        'thresholds': thresholds,
        'auc': auc,
        'eer': eer * 100,  # Convert to percentage
        'eer_threshold': eer_threshold,
        'tmr_at_fmr_0.01': tmr_at_fmr[0.0001],
        'tmr_at_fmr_0.1': tmr_at_fmr[0.001],
        'tmr_at_fmr_1': tmr_at_fmr[0.01],
        'tmr_at_fmr_10': tmr_at_fmr[0.1],
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'n_genuine': len(genuine_scores),
        'n_impostor': len(impostor_scores),
    }


# ======================================================
# SINGLE SEED VERIFICATION EXPERIMENT
# ======================================================

def run_single_seed_verification(modelX, modelY, metadata, X, Y, seed, config):
    """
    Run verification experiment for a single random seed.
    IDENTICAL split to identification, but different evaluation.
    """
    print(f"\n{'='*70}")
    print(f"SEED {seed}: {modelX} → {modelY}")
    print(f"{'='*70}")
    
    # ========================================
    # IDENTICAL SPLIT TO IDENTIFICATION
    # ========================================
    train_idx, test_idx, split_info = split_by_identity(
        metadata, train_ratio=config['train_ratio'], seed=seed
    )
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    meta_test = metadata[test_idx]
    
    print(f"Training alignment on {split_info['train_identities']} identities ({split_info['train_images']} images)")
    print(f"Testing on {split_info['test_identities']} identities ({split_info['test_images']} images)")
    
    # ========================================
    # PREPROCESS (IDENTICAL TO IDENTIFICATION)
    # ========================================
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc, preprocess_stats = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)
    
    # Normalize for verification
    X_test_norm = row_norm(X_test_proc)
    Y_test_norm = row_norm(Y_test_proc)
    
    # ========================================
    # GENERATE ALL VERIFICATION PAIRS FROM TEST SET
    # ========================================
    print(f"\nGenerating ALL verification pairs from test set...")
    pairs = create_balanced_verification_pairs_from_test(meta_test)
    
    # Storage for results
    seed_results = {
        'seed': seed,
        'split_info': split_info,
        'n_genuine_pairs': len([p for p in pairs if p[2]==1]),
        'n_impostor_pairs': len([p for p in pairs if p[2]==0]),
        'methods': {}
    }
    
    # ========================================
    # BASELINE (No Alignment)
    # ========================================
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
    print(f"  TMR@FMR=0.1%: {baseline_metrics['tmr_at_fmr_0.1']:.2f}%")
    
    # ========================================
    # WITH ALIGNMENT (SAME METHODS AS IDENTIFICATION)
    # ========================================
    alignment_methods = [
        ProcrustesAlignment(),
        LinearAlignment(),
        RidgeAlignment(alpha=0.1),
        # RidgeAlignment(alpha=1.0),
        # RidgeAlignment(alpha=10.0),
    ]
    
    for method in alignment_methods:
        print(f"\n{method.name}...")
        
        try:
            # Train alignment on training identities (SAME AS IDENTIFICATION)
            print("  Training alignment...")
            method.fit(X_train_proc, Y_train_proc)
            
            # Transform test set
            print("  Transforming test set...")
            X_test_aligned = method.transform(X_test_proc)
            X_test_aligned_norm = row_norm(X_test_aligned)
            
            # Evaluate verification on test set pairs
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
            print(f"  TMR@FMR=1%: {metrics['tmr_at_fmr_1']:.2f}% (Δ={metrics['tmr_at_fmr_1']-baseline_metrics['tmr_at_fmr_1']:+.2f}%)")
            
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
# MULTI-SEED VERIFICATION EXPERIMENT
# ======================================================

def run_multi_seed_verification(modelX, modelY, config):
    """
    Run verification experiments across multiple random seeds.
    IDENTICAL to identification experiment structure.
    """
    print(f"\n{'#'*70}")
    print(f"MULTI-SEED VERIFICATION: {modelX} → {modelY}")
    print(f"Running {config['n_seeds']} experiments with seeds: {config['seeds']}")
    print(f"{'#'*70}")
    
    # Load data
    X = np.load(f"embeddings/{config['dataset']}/{modelX}.npy")
    Y = np.load(f"embeddings/{config['dataset']}/{modelY}.npy")
    metadata = np.load(config['meta_path'], allow_pickle=True)
    
    print(f"\n{modelX} shape: {X.shape}")
    print(f"{modelY} shape: {Y.shape}")
    print(f"Total images: {len(metadata)}")
    
    # Run experiment for each seed
    all_results = []
    for seed in config['seeds']:
        seed_results = run_single_seed_verification(
            modelX, modelY, metadata, X, Y, seed, config
        )
        all_results.append(seed_results)
    
    # Aggregate results across seeds
    aggregated_results = aggregate_verification_results(all_results)
    
    return all_results, aggregated_results


def aggregate_verification_results(all_results):
    """
    Aggregate verification results across multiple seeds.
    IDENTICAL structure to identification aggregation.
    """
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
    """
    Create and save verification results table with mean ± std.
    """
    rows = []
    
    for method_name, metrics in aggregated_results.items():
        row = {'Method': method_name}
        
        # Add metrics
        for metric_name in ['auc', 'eer', 'tmr_at_fmr_0.01', 'tmr_at_fmr_0.1', 
                           'tmr_at_fmr_1', 'tmr_at_fmr_10']:
            if metric_name in metrics:
                mean = metrics[metric_name]['mean']
                std = metrics[metric_name]['std']
                
                if metric_name == 'auc':
                    row['AUC'] = f"{mean:.3f} ± {std:.3f}"
                elif metric_name == 'eer':
                    row['EER (%)'] = f"{mean:.3f} ± {std:.3f}"
                elif metric_name == 'tmr_at_fmr_0.01':
                    row['TMR@0.01%'] = f"{mean:.3f} ± {std:.3f}"
                elif metric_name == 'tmr_at_fmr_0.1':
                    row['TMR@0.1%'] = f"{mean:.3f} ± {std:.3f}"
                elif metric_name == 'tmr_at_fmr_1':
                    row['TMR@1%'] = f"{mean:.3f} ± {std:.3f}"
                elif metric_name == 'tmr_at_fmr_10':
                    row['TMR@10%'] = f"{mean:.3f} ± {std:.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ['Method', 'AUC', 'EER (%)', 'TMR@0.01%', 'TMR@0.1%', 'TMR@1%', 'TMR@10%']
    df = df[[c for c in cols if c in df.columns]]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nVerification results table saved to: {save_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("AGGREGATED VERIFICATION RESULTS (Mean ± Std across all seeds)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def save_verification_results(all_results, aggregated_results, modelX, modelY, config):
    """
    Save all verification results to files.
    """
    out_dir = config['out_dir']
    prefix = f"{modelX}_to_{modelY}_{config['dataset']}"
    
    # Save raw results (all seeds)
    raw_results_path = f"{out_dir}/{prefix}_verification_all_seeds.json"
    
    # Remove numpy arrays for JSON serialization
    serializable_results = []
    for seed_result in all_results:
        result_copy = seed_result.copy()
        serializable_results.append(result_copy)
    
    with open(raw_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nRaw verification results saved to: {raw_results_path}")
    
    # Save aggregated results table
    table_path = f"{out_dir}/{prefix}_verification_results.csv"
    create_verification_table(aggregated_results, table_path)


# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    """
    Main execution function for verification experiments.
    """
    print("="*70)
    print("FACE VERIFICATION EXPERIMENTS WITH ALIGNMENT")
    print("="*70)
    
    # Run experiments for each model pair
    for modelX, modelY in CONFIG['model_pairs']:
        all_results, aggregated_results = run_multi_seed_verification(
            modelX, modelY, CONFIG
        )
        
        # Save results
        save_verification_results(all_results, aggregated_results, 
                                 modelX, modelY, CONFIG)
        
        print(f"\n✓ Completed: {modelX} → {modelY}")
        print(f"  Results saved to: {CONFIG['out_dir']}")
    
    print("\n" + "="*70)
    print("ALL VERIFICATION EXPERIMENTS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()