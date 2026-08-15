import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from cfe.methods import ProcrustesAlignment, LinearAlignment, RidgeAlignment
from scipy.stats import spearmanr
import json

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    'dataset': 'cfp',
    'meta_path': "embeddings/cfp/cfp_metadata.npy",
    'out_dir': "results/cfp_rsa_cka",
    'model_pairs': [
        ('arcface', 'adaface'),
        # ('adaface', 'arcface'),
        ('arcface', 'clip'),
        # ('adaface', 'clip'),
        # ('clip', 'arcface'),
        # ('clip', 'adaface'),
    ],
    'train_ratio': 0.7,
    'n_seeds': 5,
    'seeds': [42, 123, 217, 7, 531],
    # Number of images to subsample for RSA/CKA (NxN matrix - keep manageable)
    'rsa_subsample': 500,
}

# CFE_MODELS="a,b,..." overrides the pair list with (a, b) for partial runs.
_sel = [s.strip() for s in os.environ.get("CFE_MODELS", "").split(",") if s.strip()]
if len(_sel) >= 2:
    CONFIG['model_pairs'] = [(_sel[0], _sel[1])]

os.makedirs(CONFIG['out_dir'], exist_ok=True)

with open(f"{CONFIG['out_dir']}/config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("Configuration:")
print(json.dumps(CONFIG, indent=2))


# ======================================================
# UTILITIES (identical to within_dataset_cfp.py)
# ======================================================

def split_by_identity(metadata, train_ratio=0.7, seed=42):
    np.random.seed(seed)

    identity_to_indices = defaultdict(list)
    for idx, rec in enumerate(metadata):
        identity_to_indices[rec['identity']].append(idx)

    identities = sorted(identity_to_indices.keys())
    n_train = int(len(identities) * train_ratio)

    identities_shuffled = identities.copy()
    np.random.shuffle(identities_shuffled)

    train_identities = set(identities_shuffled[:n_train])
    test_identities  = set(identities_shuffled[n_train:])

    train_idx, test_idx = [], []
    for identity, indices in identity_to_indices.items():
        if identity in train_identities:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    assert len(train_identities & test_identities) == 0
    return np.array(train_idx), np.array(test_idx)


def preprocess_embeddings(X_train, Y_train, X_test, Y_test):
    Dx, Dy = X_train.shape[1], Y_train.shape[1]
    Dmax = max(Dx, Dy)

    def pad(A, d):
        if A.shape[1] == d:
            return A
        return np.pad(A, ((0, 0), (0, d - A.shape[1])), mode='constant')

    mean_X = X_train.mean(0, keepdims=True)
    mean_Y = Y_train.mean(0, keepdims=True)

    X_train_proc = pad(X_train - mean_X, Dmax)
    Y_train_proc = pad(Y_train - mean_Y, Dmax)
    X_test_proc  = pad(X_test  - mean_X, Dmax)
    Y_test_proc  = pad(Y_test  - mean_Y, Dmax)

    return X_train_proc, Y_train_proc, X_test_proc, Y_test_proc


def row_norm(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)


# ======================================================
# RSA  -  Representational Similarity Analysis
# ======================================================
# Compute the NxN cosine-similarity matrix for each model,
# then correlate the upper triangles (Spearman r).
# A high score means the two models rank pairs of images similarly.

def compute_rsa(A, B, subsample=500, seed=42):
    """
    Args:
        A, B : (N, D) embedding matrices (already row-normalised)
        subsample : use at most this many images (NxN matrix gets big fast)
    Returns:
        rsa_score : Spearman r between upper triangles of the two RDMs
    """
    N = min(len(A), len(B), subsample)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=N, replace=False)

    A_sub = row_norm(A[idx])
    B_sub = row_norm(B[idx])

    # Representational Dissimilarity Matrices (1 - cosine_sim)
    RDM_A = 1.0 - (A_sub @ A_sub.T)
    RDM_B = 1.0 - (B_sub @ B_sub.T)

    # Upper triangle (excluding diagonal)
    tri = np.triu_indices(N, k=1)
    rsa_score, _ = spearmanr(RDM_A[tri], RDM_B[tri])
    return float(rsa_score)


# ======================================================
# CKA  -  Linear Centered Kernel Alignment
# ======================================================
# Measures similarity between two kernel matrices derived from the embeddings.
# Invariant to orthogonal transforms and isotropic scaling.

def linear_kernel(A):
    return A @ A.T


def center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_cka(A, B, subsample=500, seed=42):
    """
    Linear CKA between two embedding matrices.
    Returns value in [0, 1]; higher = more similar geometry.
    """
    N = min(len(A), len(B), subsample)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), size=N, replace=False)

    A_sub = row_norm(A[idx])
    B_sub = row_norm(B[idx])

    Ka = center_kernel(linear_kernel(A_sub))
    Kb = center_kernel(linear_kernel(B_sub))

    hsic_ab = np.sum(Ka * Kb)
    hsic_aa = np.sum(Ka * Ka)
    hsic_bb = np.sum(Kb * Kb)

    denom = np.sqrt(hsic_aa * hsic_bb)
    if denom < 1e-10:
        return 0.0
    return float(hsic_ab / denom)


# ======================================================
# SINGLE SEED EXPERIMENT
# ======================================================

def run_single_seed(modelX, modelY, metadata, X, Y, seed):
    print(f"\n{'='*60}")
    print(f"Seed {seed}: {modelX} → {modelY}")
    print(f"{'='*60}")

    train_idx, test_idx = split_by_identity(metadata, CONFIG['train_ratio'], seed)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    X_train_p, Y_train_p, X_test_p, Y_test_p = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)

    alignment_methods = [
        ProcrustesAlignment(),
        LinearAlignment(),
        RidgeAlignment(alpha=0.1),
    ]

    seed_results = {'seed': seed, 'methods': {}}

    # ---- Baseline RSA / CKA (no alignment) ----
    rsa_before = compute_rsa(X_test_p, Y_test_p,
                             subsample=CONFIG['rsa_subsample'], seed=seed)
    cka_before = compute_cka(X_test_p, Y_test_p,
                             subsample=CONFIG['rsa_subsample'], seed=seed)

    seed_results['methods']['Baseline'] = {
        'rsa': rsa_before,
        'cka': cka_before,
    }
    print(f"  Baseline  RSA={rsa_before:.4f}  CKA={cka_before:.4f}")

    # ---- Each alignment method ----
    for method in alignment_methods:
        method.fit(X_train_p, Y_train_p)
        X_test_aligned = method.transform(X_test_p)

        rsa_after = compute_rsa(X_test_aligned, Y_test_p,
                                subsample=CONFIG['rsa_subsample'], seed=seed)
        cka_after = compute_cka(X_test_aligned, Y_test_p,
                                subsample=CONFIG['rsa_subsample'], seed=seed)

        seed_results['methods'][method.name] = {
            'rsa': rsa_after,
            'cka': cka_after,
        }
        print(f"  {method.name:20s}  RSA={rsa_after:.4f} (Δ={rsa_after-rsa_before:+.4f})"
              f"  CKA={cka_after:.4f} (Δ={cka_after-cka_before:+.4f})")

    return seed_results


# ======================================================
# MULTI-SEED AGGREGATION
# ======================================================

def aggregate(all_results):
    method_names = list(all_results[0]['methods'].keys())
    aggregated = {}
    for name in method_names:
        rsa_vals = [r['methods'][name]['rsa'] for r in all_results]
        cka_vals = [r['methods'][name]['cka'] for r in all_results]
        aggregated[name] = {
            'rsa_mean': float(np.mean(rsa_vals)),
            'rsa_std':  float(np.std(rsa_vals)),
            'cka_mean': float(np.mean(cka_vals)),
            'cka_std':  float(np.std(cka_vals)),
        }
    return aggregated


# ======================================================
# RESULTS TABLE
# ======================================================

def save_results(all_pair_results, out_dir):
    rows = []
    for entry in all_pair_results:
        modelX, modelY, agg = entry['modelX'], entry['modelY'], entry['aggregated']
        for method_name, metrics in agg.items():
            rows.append({
                'Model Pair':   f"{modelX} → {modelY}",
                'Method':       method_name,
                'RSA (mean)':   f"{metrics['rsa_mean']:.4f}",
                'RSA (std)':    f"{metrics['rsa_std']:.4f}",
                'RSA':          f"{metrics['rsa_mean']:.4f} ± {metrics['rsa_std']:.4f}",
                'CKA (mean)':   f"{metrics['cka_mean']:.4f}",
                'CKA (std)':    f"{metrics['cka_std']:.4f}",
                'CKA':          f"{metrics['cka_mean']:.4f} ± {metrics['cka_std']:.4f}",
            })

    df = pd.DataFrame(rows)
    csv_path = f"{out_dir}/rsa_cka_results.csv"
    df[['Model Pair', 'Method', 'RSA', 'CKA']].to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "="*70)
    print("RSA / CKA RESULTS  (mean ± std across seeds)")
    print("="*70)
    print(df[['Model Pair', 'Method', 'RSA', 'CKA']].to_string(index=False))
    print("="*70)

    # also dump raw json
    json_path = f"{out_dir}/rsa_cka_raw.json"
    with open(json_path, 'w') as f:
        json.dump(all_pair_results, f, indent=2)

    return df


# ======================================================
# OPTIONAL: heatmap of CKA before/after for all pairs
# ======================================================

def plot_heatmap(all_pair_results, out_dir):
    pairs   = [f"{e['modelX']}→{e['modelY']}" for e in all_pair_results]
    methods = list(all_pair_results[0]['aggregated'].keys())

    for metric in ['cka_mean', 'rsa_mean']:
        label = 'CKA' if 'cka' in metric else 'RSA'
        data  = np.array([
            [e['aggregated'][m][metric] for m in methods]
            for e in all_pair_results
        ])

        fig, ax = plt.subplots(figsize=(max(6, len(methods)*1.5), max(4, len(pairs)*0.6)))
        im = ax.imshow(data, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_yticks(range(len(pairs)));   ax.set_yticklabels(pairs)
        plt.colorbar(im, ax=ax, label=label)

        for i in range(len(pairs)):
            for j in range(len(methods)):
                ax.text(j, i, f"{data[i,j]:.3f}", ha='center', va='center', fontsize=8)

        ax.set_title(f'{label} before and after alignment (CFP)')
        plt.tight_layout()
        path = f"{out_dir}/{label.lower()}_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {path}")


# ======================================================
# MAIN
# ======================================================

def main():
    print("="*60)
    print("RSA / CKA EMBEDDING SIMILARITY ANALYSIS")
    print("="*60)

    metadata = np.load(CONFIG['meta_path'], allow_pickle=True)

    all_pair_results = []

    for modelX, modelY in CONFIG['model_pairs']:
        print(f"\n{'#'*60}")
        print(f"PAIR: {modelX} → {modelY}")
        print(f"{'#'*60}")

        X = np.load(f"embeddings/{CONFIG['dataset']}/{modelX}.npy")
        Y = np.load(f"embeddings/{CONFIG['dataset']}/{modelY}.npy")

        print(f"  {modelX}: {X.shape}  |  {modelY}: {Y.shape}")
        assert len(X) == len(metadata), "Embedding / metadata length mismatch"

        all_seed_results = []
        for seed in CONFIG['seeds']:
            seed_results = run_single_seed(modelX, modelY, metadata, X, Y, seed)
            all_seed_results.append(seed_results)

        aggregated = aggregate(all_seed_results)

        all_pair_results.append({
            'modelX':      modelX,
            'modelY':      modelY,
            'aggregated':  aggregated,
            'all_seeds':   all_seed_results,
        })

        print(f"\nAggregated ({modelX} → {modelY}):")
        for method_name, metrics in aggregated.items():
            print(f"  {method_name:20s}  "
                  f"RSA={metrics['rsa_mean']:.4f}±{metrics['rsa_std']:.4f}  "
                  f"CKA={metrics['cka_mean']:.4f}±{metrics['cka_std']:.4f}")

    save_results(all_pair_results, CONFIG['out_dir'])
    plot_heatmap(all_pair_results, CONFIG['out_dir'])

    print("\nDONE.")


if __name__ == "__main__":
    main()
