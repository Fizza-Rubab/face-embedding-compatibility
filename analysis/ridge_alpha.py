import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from cfe.methods import RidgeAlignment, LinearAlignment
import json

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    'dataset': 'cfp',
    'meta_path': "embeddings/cfp/cfp_metadata.npy",
    'out_dir': "results/cfp_ridge_ablation",
    'model_pairs': [
        ('arcface', 'adaface'),
        # ('arcface', 'clip'),
    ],
    'alphas': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    # Different training ratios - key experiment: Ridge vs Linear at low data
    'train_ratios': [0.1, 0.2, 0.3, 0.5, 0.7],
    'n_seeds': 5,
    'seeds': [42, 123, 217, 7, 531],
    'max_rank': 50,
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
# UTILITIES - identical to within_dataset_cfp.py
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

    return (pad(X_train - mean_X, Dmax),
            pad(Y_train - mean_Y, Dmax),
            pad(X_test  - mean_X, Dmax),
            pad(Y_test  - mean_Y, Dmax))


def row_norm(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)


def compute_rank1(A, B, labels, chunk_size=5000):
    """Chunked Rank-1 to avoid memory issues."""
    A_n = row_norm(A)
    B_n = row_norm(B)
    correct = 0
    for start in range(0, len(A_n), chunk_size):
        end   = min(start + chunk_size, len(A_n))
        S     = A_n[start:end] @ B_n.T
        preds = np.argmax(S, axis=1)
        correct += np.sum(labels[preds] == labels[start:end])
    return correct / len(A_n) * 100


# ======================================================
# SINGLE SEED - runs all train_ratios in one pass
# ======================================================

def run_single_seed(modelX, modelY, metadata, X, Y, seed):
    """
    For each train_ratio: fix a 70% test set (largest split's test identities
    are reused for comparability), vary how many training identities are used
    to fit the alignment W.

    We always evaluate on the same held-out test set (from train_ratio=0.7
    split) so Rank-1 numbers are comparable across ratios.
    """
    # Fix test set using the largest train ratio for consistent evaluation
    _, test_idx_fixed = split_by_identity(metadata, 0.7, seed)
    X_test = X[test_idx_fixed]
    Y_test = Y[test_idx_fixed]

    test_identities = [metadata[i]['identity'] for i in test_idx_fixed]
    unique_ids      = sorted(set(test_identities))
    id_to_label     = {id_: i for i, id_ in enumerate(unique_ids)}
    test_labels     = np.array([id_to_label[id_] for id_ in test_identities])

    ratio_results = {}

    for train_ratio in CONFIG['train_ratios']:
        train_idx, _ = split_by_identity(metadata, train_ratio, seed)

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        n_train_ids = len(set(metadata[i]['identity'] for i in train_idx))

        X_tr, Y_tr, X_te, Y_te = preprocess_embeddings(
            X_train, Y_train, X_test, Y_test
        )

        method_results = {}

        # Baseline (no alignment)
        method_results['Baseline'] = compute_rank1(X_te, Y_te, test_labels)

        # Linear reference
        lin = LinearAlignment()
        lin.fit(X_tr, Y_tr)
        method_results['Linear'] = compute_rank1(
            lin.transform(X_te), Y_te, test_labels
        )

        # Ridge across all alphas
        for alpha in CONFIG['alphas']:
            ridge = RidgeAlignment(alpha=alpha)
            ridge.fit(X_tr, Y_tr)
            method_results[f'Ridge(α={alpha})'] = compute_rank1(
                ridge.transform(X_te), Y_te, test_labels
            )

        ratio_results[train_ratio] = {
            'n_train_ids':    n_train_ids,
            'n_train_images': len(train_idx),
            'methods':        method_results,
        }

        print(f"    ratio={train_ratio} ({n_train_ids} ids)  "
              f"Linear={method_results['Linear']:.1f}%  "
              + "  ".join(f"α={a}:{method_results[f'Ridge(α={a})']:.1f}%"
                          for a in CONFIG['alphas']))

    return ratio_results


# ======================================================
# MULTI-SEED
# ======================================================

def run_pair(modelX, modelY, metadata, X, Y):
    print(f"\n{'#'*60}")
    print(f"PAIR: {modelX} → {modelY}")
    print(f"{'#'*60}")

    all_seed_results = []
    for seed in CONFIG['seeds']:
        print(f"  Seed {seed}:")
        res = run_single_seed(modelX, modelY, metadata, X, Y, seed)
        all_seed_results.append(res)

    # Aggregate across seeds for each (train_ratio, method)
    aggregated = {}
    for train_ratio in CONFIG['train_ratios']:
        aggregated[train_ratio] = {}
        method_names = list(all_seed_results[0][train_ratio]['methods'].keys())
        n_train_ids  = all_seed_results[0][train_ratio]['n_train_ids']

        for name in method_names:
            vals = [r[train_ratio]['methods'][name] for r in all_seed_results]
            aggregated[train_ratio][name] = {
                'mean':        float(np.mean(vals)),
                'std':         float(np.std(vals)),
                'n_train_ids': n_train_ids,
            }

    return aggregated


# ======================================================
# RESULTS TABLE + PLOT
# ======================================================

def save_results(all_pair_results):
    out_dir = CONFIG['out_dir']
    alphas  = CONFIG['alphas']
    ratios  = CONFIG['train_ratios']

    # --- CSV table ---
    rows = []
    for entry in all_pair_results:
        pair = f"{entry['modelX']} → {entry['modelY']}"
        for ratio, methods in entry['aggregated'].items():
            n_ids = methods['Linear']['n_train_ids']
            for method_name, metrics in methods.items():
                rows.append({
                    'Model Pair':    pair,
                    'Train Ratio':   ratio,
                    'Train IDs':     n_ids,
                    'Method':        method_name,
                    'Rank-1':        f"{metrics['mean']:.2f} ± {metrics['std']:.2f}",
                })

    df = pd.DataFrame(rows)
    csv_path = f"{out_dir}/ridge_ablation_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "="*75)
    print("Ridge Regularizaition(α) ABLATION  (Rank-1 mean ± std across seeds)")
    print("="*75)
    print(df.to_string(index=False))
    print("="*75)

    # --- Plot: one subplot per model pair
    #     x = alpha (log), y = Rank-1
    #     one curve per train_ratio, dashed horizontal = Linear at that ratio
    n_pairs = len(all_pair_results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 3.5), sharey=False)
    if n_pairs == 1:
        axes = [axes]

    ratio_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(ratios)))
    pair_labels  = {'arcface': 'ArcFace', 'adaface': 'AdaFace', 'clip': 'CLIP',
                    'kprpe': 'KPRPE', 'sam': 'SAM'}

    for ax, entry in zip(axes, all_pair_results):
        mx   = pair_labels.get(entry['modelX'], entry['modelX'])
        my   = pair_labels.get(entry['modelY'], entry['modelY'])
        agg  = entry['aggregated']

        for j, ratio in enumerate(ratios):
            color   = ratio_colors[j]
            methods = agg[ratio]
            n_ids   = methods['Linear']['n_train_ids']

            lin_mean = methods['Linear']['mean']

            means = [methods[f'Ridge(α={a})']['mean'] for a in alphas]
            stds  = [methods[f'Ridge(α={a})']['std']  for a in alphas]

            lbl = f"{int(ratio*100)}% ({n_ids} ids)"

            ax.errorbar(alphas, means, yerr=stds,
                        label=lbl, color=color,
                        marker='o', markersize=4, linewidth=1.5, capsize=2)

            # Linear reference as dashed horizontal - same color, no label
            ax.axhline(lin_mean, linestyle='--', color=color, alpha=0.6, linewidth=1)

        ax.set_xscale('log')
        ax.set_xlabel('Ridge $\\alpha$', fontsize=10)
        ax.set_ylabel('Rank-1 Accuracy(%)', fontsize=10)
        ax.set_title(f'{mx} $\\to$ {my}', fontsize=10)
        ax.legend(fontsize=7, title='Train size', title_fontsize=7,
                  loc='best', framealpha=0.7)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    # Add a shared annotation explaining dashed lines
    fig.text(0.5, -0.02, 'Dashed lines = Linear map baseline at each train size',
             ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plot_path = f"{out_dir}/ridge_ablation_plot.svg"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")

    with open(f"{out_dir}/ridge_ablation_raw.json", 'w') as f:
        json.dump(all_pair_results, f, indent=2)


# ======================================================
# MAIN
# ======================================================

def main():
    print("="*60)
    print("RIDGE REGULARIZATION α ABLATION")
    print("="*60)

    metadata = np.load(CONFIG['meta_path'], allow_pickle=True)
    all_pair_results = []

    for modelX, modelY in CONFIG['model_pairs']:
        X = np.load(f"embeddings/{CONFIG['dataset']}/{modelX}.npy")
        Y = np.load(f"embeddings/{CONFIG['dataset']}/{modelY}.npy")
        assert len(X) == len(metadata)

        aggregated = run_pair(modelX, modelY, metadata, X, Y)
        all_pair_results.append({
            'modelX': modelX, 'modelY': modelY,
            'aggregated': aggregated,
        })

    save_results(all_pair_results)
    print("\nDONE.")


if __name__ == "__main__":
    main()
