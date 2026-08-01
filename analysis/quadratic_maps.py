
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, MultiTaskLasso
from collections import defaultdict
import json

CONFIG = {
    'dataset': 'cfp',
    'meta_path': "embeddings/cfp/cfp_metadata.npy",
    'out_dir': "cfp_quadratic",
    'model_pairs': [
        ('arcface', 'adaface'),
        # ('adaface', 'kprpe'),
        

        # ('dinov2', 'clip'),
        # ('clip', 'dinov2'),
        # # ('sam', 'dinov2'),

        # ('clip', 'arcface'),
        # ('dinov2', 'arcface'),
        
        # ('arcface', 'clip'),
        

        
    ],
    'n_components': None,
    'train_ratios': [0.7],  # vary train size for instability analysis
    'regularizer': 'ridge',  # 'ridge' or 'lasso'
    'alpha': 10**(-4),       # Ridge penalty
    'lasso_alpha': 1e-4,     # set automatically by sweep; override manually if desired
    'lasso_alpha_sweep': [1e-4],
    'seeds': [42, 123, 217, 7, 531],   # seeds for aggregating main metrics
}

import os
os.makedirs(CONFIG['out_dir'], exist_ok=True)


def split_by_identity(metadata, train_ratio=0.7, seed=42):
    np.random.seed(seed)
    identity_to_indices = defaultdict(list)
    for idx, rec in enumerate(metadata):
        identity_to_indices[rec['identity']].append(idx)
    identities = sorted(identity_to_indices.keys())
    np.random.shuffle(identities := list(identities))
    n_train = int(len(identities) * train_ratio)
    train_ids = set(identities[:n_train])
    train_idx, test_idx = [], []
    for identity, indices in identity_to_indices.items():
        (train_idx if identity in train_ids else test_idx).extend(indices)
    return np.array(train_idx), np.array(test_idx)


def run_pair(modelX, modelY, metadata, X, Y):
    n_components = CONFIG['n_components']
    print(f"\n{'='*60}")
    print(f"{modelX} -> {modelY}  (n_components={'full' if n_components is None else n_components})")
    print(f"{'='*60}")

    seed_results = []
    W_lins_by_ratio, W_quads_by_ratio = [], []   # collect across train sizes (fixed seed)

    for seed in CONFIG['seeds']:
        np.random.seed(seed)
        train_idx, test_idx = split_by_identity(metadata, 0.7, seed)

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Center
        mx, my = X_train.mean(0), Y_train.mean(0)
        X_train_c = X_train - mx;  X_test_c = X_test - mx
        Y_train_c = Y_train - my;  Y_test_c = Y_test - my

        if n_components is None:
            Xtr, Xte = X_train_c, X_test_c
            Ytr, Yte = Y_train_c, Y_test_c
            print(f"  Full dim: X={Xtr.shape[1]}, Y={Ytr.shape[1]}")
        else:
            pca_x = PCA(n_components=n_components).fit(X_train_c)
            pca_y = PCA(n_components=n_components).fit(Y_train_c)
            Xtr = pca_x.transform(X_train_c);  Xte = pca_x.transform(X_test_c)
            Ytr = pca_y.transform(Y_train_c);  Yte = pca_y.transform(Y_test_c)

        W_lin = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]  # (n_components, n_components)
        Y_pred_lin_tr = Xtr @ W_lin
        Y_pred_lin_te = Xte @ W_lin

        mse_lin_train = np.mean((Ytr - Y_pred_lin_tr) ** 2)
        mse_lin_test  = np.mean((Yte - Y_pred_lin_te) ** 2)


        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        Xtr_poly = poly.fit_transform(Xtr)   # (n_train, 1325)
        Xte_poly = poly.transform(Xte)

        n_lin  = Xtr.shape[1]                  # first d cols = linear terms
        n_quad = Xtr_poly.shape[1] - n_lin    # remaining = quadratic terms

        if CONFIG['regularizer'] == 'lasso':
            # MultiTaskLasso: shared sparsity pattern across all output dims.
            # If L1 zeros out quadratic features preferentially, that is strong evidence
            # the quadratic terms carry no signal beyond the linear terms.
            reg = MultiTaskLasso(alpha=CONFIG['lasso_alpha'], fit_intercept=False, max_iter=5000)
        else:
            reg = Ridge(alpha=CONFIG['alpha'], fit_intercept=False)

        reg.fit(Xtr_poly, Ytr)
        W_poly = reg.coef_.T   # (n_features_poly, n_outputs)
        print(f"W_poly.shape {W_poly.shape}  [{CONFIG['regularizer']}]")
        W_poly_lin  = W_poly[:n_lin,  :]
        W_poly_quad = W_poly[n_lin:,  :]

        norm_lin  = np.linalg.norm(W_poly_lin,  'fro')
        norm_quad = np.linalg.norm(W_poly_quad, 'fro')
        ratio     = norm_quad / norm_lin

        Y_pred_poly_tr = Xtr_poly @ W_poly
        Y_pred_poly_te = Xte_poly @ W_poly

        mse_poly_train = np.mean((Ytr - Y_pred_poly_tr) ** 2)
        mse_poly_test  = np.mean((Yte - Y_pred_poly_te) ** 2)

        col_ratios = np.linalg.norm(W_poly_quad, axis=0) / (np.linalg.norm(W_poly_lin, axis=0) + 1e-9)
        quad_outputs = Xte_poly[:, n_lin:] @ W_poly_quad  # (n_test, 512)
        lin_outputs  = Xte_poly[:, :n_lin] @ W_poly_lin   # (n_test, 512)
        per_sample_ratio = (np.linalg.norm(quad_outputs, axis=1) /
                            np.maximum(np.linalg.norm(lin_outputs, axis=1), 1e-9))
        contribution_ratio = float(np.mean(per_sample_ratio))

        # ---- Variance explained by quadratic term ----
        total_outputs = lin_outputs + quad_outputs
        var_quad  = float(np.var(quad_outputs))
        var_total = float(np.var(total_outputs))
        var_ratio = var_quad / (var_total + 1e-9)

        # ---- Lasso-specific sparsity stats ----
        if CONFIG['regularizer'] == 'lasso':
            total_nonzero_lin  = int(np.sum(np.any(W_poly_lin  != 0, axis=1)))
            total_nonzero_quad = int(np.sum(np.any(W_poly_quad != 0, axis=1)))
            nonzero_lin_frac   = total_nonzero_lin  / n_lin
            nonzero_quad_frac  = total_nonzero_quad / n_quad
            lasso_lin_share    = total_nonzero_lin / max(total_nonzero_lin + total_nonzero_quad, 1)
        else:
            total_nonzero_lin = total_nonzero_quad = None
            nonzero_lin_frac = nonzero_quad_frac = lasso_lin_share = float('nan')

        def row_norm(A):
            return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)

        def rank1(queries, gallery, labels_q, labels_g):
            S = row_norm(queries) @ row_norm(gallery).T
            preds = labels_g[np.argmax(S, axis=1)]
            return float(np.mean(preds == labels_q)) * 100

        labels_te = np.array([metadata[i]['identity'] for i in test_idx])
        rank1_lin  = rank1(Y_pred_lin_te,  Yte, labels_te, labels_te)
        rank1_quad = rank1(Y_pred_poly_te, Yte, labels_te, labels_te)

        print(f"  Seed {seed}:")
        print(f"    Linear   block shape: {W_poly_lin.shape}   ||W_lin||={norm_lin:.4f}")
        print(f"    Quadratic block shape: {W_poly_quad.shape}  ||W_quad||={norm_quad:.4f}")
        print(f"    Ratio ||W_quad||/||W_lin|| = {ratio:.4f}")
        print(f"    Contribution ratio (mean per-sample ||quad||/||lin||) = {contribution_ratio:.4f}")
        print(f"    Var(quad output) / Var(total output) = {var_ratio:.4f}")
        print(f"    MSE (linear map, test):    {mse_lin_test:.6f}")
        print(f"    MSE (quadratic map, test): {mse_poly_test:.6f}")
        print(f"    MSE improvement: {(mse_lin_test - mse_poly_test)/mse_lin_test*100:.2f}%")
        print(f"    Rank-1 (linear map):    {rank1_lin:.2f}%")
        print(f"    Rank-1 (quadratic map): {rank1_quad:.2f}%")
        if CONFIG['regularizer'] == 'lasso':
            print(f"    --- Lasso sparsity (alpha={CONFIG['lasso_alpha']}) ---")
            print(f"    Non-zero lin features:  {total_nonzero_lin}/{n_lin}  ({nonzero_lin_frac*100:.1f}%)")
            print(f"    Non-zero quad features: {total_nonzero_quad}/{n_quad}  ({nonzero_quad_frac*100:.1f}%)")
            print(f"    Share of surviving features that are linear: {lasso_lin_share*100:.1f}%")

        seed_results.append({
            'seed': seed,
            'norm_lin': float(norm_lin),
            'norm_quad': float(norm_quad),
            'ratio': float(ratio),
            'mse_lin_test': float(mse_lin_test),
            'mse_poly_test': float(mse_poly_test),
            'mse_improvement_pct': float((mse_lin_test - mse_poly_test) / mse_lin_test * 100),
            'col_ratio_mean': float(col_ratios.mean()),
            'col_ratio_std': float(col_ratios.std()),
            'contribution_ratio': contribution_ratio,
            'var_ratio': var_ratio,
            'rank1_lin': rank1_lin,
            'rank1_quad': rank1_quad,
            'lasso_nonzero_lin_frac': nonzero_lin_frac,
            'lasso_nonzero_quad_frac': nonzero_quad_frac,
            'lasso_lin_share': lasso_lin_share,
        })

    # Aggregate
    ratios = [r['ratio'] for r in seed_results]
    improvements = [r['mse_improvement_pct'] for r in seed_results]
    col_means = [r['col_ratio_mean'] for r in seed_results]
    contrib_ratios = [r['contribution_ratio'] for r in seed_results]
    var_ratios = [r['var_ratio'] for r in seed_results]
    rank1_lins   = [r['rank1_lin']  for r in seed_results]
    rank1_quads  = [r['rank1_quad'] for r in seed_results]
    lasso_lin_shares = [r['lasso_lin_share']         for r in seed_results]
    lasso_quad_fracs = [r['lasso_nonzero_quad_frac'] for r in seed_results]
    lasso_lin_fracs  = [r['lasso_nonzero_lin_frac']  for r in seed_results]

    print(f"\n  SUMMARY ({modelX} -> {modelY}):")
    print(f"    ||W_quad||/||W_lin||  = {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    print(f"    Contribution ratio    = {np.mean(contrib_ratios):.4f} ± {np.std(contrib_ratios):.4f}")
    print(f"    Var(quad)/Var(total)  = {np.mean(var_ratios):.4f} ± {np.std(var_ratios):.4f}")
    print(f"    MSE improvement (quad over linear) = {np.mean(improvements):.2f}% ± {np.std(improvements):.2f}%")
    print(f"    Per-dim ratio mean = {np.mean(col_means):.4f} ± {np.std(col_means):.4f}")
    rank1_gain = np.mean(rank1_quads) - np.mean(rank1_lins)
    print(f"    Rank-1 linear map:    {np.mean(rank1_lins):.2f}% ± {np.std(rank1_lins):.2f}%")
    print(f"    Rank-1 quadratic map: {np.mean(rank1_quads):.2f}% ± {np.std(rank1_quads):.2f}%")
    print(f"    Rank-1 gain (quad - lin): {rank1_gain:+.2f}pp")
    if CONFIG['regularizer'] == 'lasso':
        print(f"    --- Lasso sparsity (alpha={CONFIG['lasso_alpha']}) ---")
        print(f"    Linear features kept:    {np.mean(lasso_lin_fracs)*100:.1f}% ± {np.std(lasso_lin_fracs)*100:.1f}%")
        print(f"    Quadratic features kept: {np.mean(lasso_quad_fracs)*100:.1f}% ± {np.std(lasso_quad_fracs)*100:.1f}%")
        print(f"    Share of surviving features that are linear: {np.mean(lasso_lin_shares)*100:.1f}% ± {np.std(lasso_lin_shares)*100:.1f}%")

    # ---- Weight instability across training sizes (fixed seed=42) ----
    # fit W_lin and W_quad at each train ratio, compare consistency
    # if quadratic fits noise: weights change wildly as training set grows
    print(f"\n  Computing weight instability across train sizes {CONFIG['train_ratios']}...")
    fixed_seed = CONFIG['seeds'][0]
    _, test_idx_fixed = split_by_identity(metadata, 0.7, fixed_seed)
    X_test_f, Y_test_f = X[test_idx_fixed], Y[test_idx_fixed]
    mx_f = X[test_idx_fixed].mean(0)   # use test mean for centering (approx)

    for ratio in CONFIG['train_ratios']:
        tr_idx, _ = split_by_identity(metadata, ratio, fixed_seed)
        Xtr_r, Ytr_r = X[tr_idx] - X[tr_idx].mean(0), Y[tr_idx] - Y[tr_idx].mean(0)
        Xte_r = X_test_f - X[tr_idx].mean(0)

        W_l = np.linalg.lstsq(Xtr_r, Ytr_r, rcond=None)[0]
        poly_r = PolynomialFeatures(degree=2, include_bias=False)
        Xtr_p = poly_r.fit_transform(Xtr_r)
        reg_r = Ridge(alpha=CONFIG['alpha'], fit_intercept=False)
        reg_r.fit(Xtr_p, Ytr_r)
        W_p = reg_r.coef_.T
        W_lins_by_ratio.append(W_l)
        W_quads_by_ratio.append(W_p[Xtr_r.shape[1]:, :])

    W_lin_stack  = np.stack(W_lins_by_ratio,  axis=0)
    W_quad_stack = np.stack(W_quads_by_ratio, axis=0)
    instab_lin  = (np.linalg.norm(W_lin_stack.std(axis=0),  'fro') /
                   (np.linalg.norm(W_lin_stack.mean(axis=0), 'fro') + 1e-9))
    instab_quad = (np.linalg.norm(W_quad_stack.std(axis=0),  'fro') /
                   (np.linalg.norm(W_quad_stack.mean(axis=0), 'fro') + 1e-9))
    print(f"    Weight instability across train sizes (||std||_F / ||mean||_F):")
    print(f"      Linear:    {instab_lin:.4f}")
    print(f"      Quadratic: {instab_quad:.4f}")
    print(f"      Ratio (quad/lin instability): {instab_quad/instab_lin:.2f}x")

    return {
        'modelX': modelX,
        'modelY': modelY,
        'ratio_mean': float(np.mean(ratios)),
        'ratio_std': float(np.std(ratios)),
        'mse_improvement_mean': float(np.mean(improvements)),
        'mse_improvement_std': float(np.std(improvements)),
        'col_ratio_mean': float(np.mean(col_means)),
        'seeds': seed_results,
    }


def run_pair_alpha(modelX, modelY, metadata, X, Y, sweep_alpha):
    """Lightweight version of run_pair: only returns mean rank1 gain (quad - lin) for alpha sweep."""
    seed_gains = []
    for seed in CONFIG['seeds']:
        train_idx, test_idx = split_by_identity(metadata, 0.7, seed)
        Xtr = X[train_idx] - X[train_idx].mean(0)
        Xte = X[test_idx]  - X[train_idx].mean(0)
        Ytr = Y[train_idx] - Y[train_idx].mean(0)
        Yte = Y[test_idx]  - Y[train_idx].mean(0)

        W_lin = np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
        Y_pred_lin_te = Xte @ W_lin

        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        Xtr_poly = poly.fit_transform(Xtr)
        Xte_poly = poly.transform(Xte)

        if CONFIG['regularizer'] == 'lasso':
            reg = MultiTaskLasso(alpha=sweep_alpha, fit_intercept=False, max_iter=5000)
        else:
            reg = Ridge(alpha=sweep_alpha, fit_intercept=False)
        reg.fit(Xtr_poly, Ytr)
        W_poly = reg.coef_.T
        Y_pred_poly_te = Xte_poly @ W_poly

        def row_norm(A):
            return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        def rank1_acc(queries, gallery, labels):
            S = row_norm(queries) @ row_norm(gallery).T
            return float(np.mean(labels[np.argmax(S, axis=1)] == labels)) * 100

        labels_te = np.array([metadata[i]['identity'] for i in test_idx])
        gain = rank1_acc(Y_pred_poly_te, Yte, labels_te) - rank1_acc(Y_pred_lin_te, Yte, labels_te)
        seed_gains.append(gain)
    return float(np.mean(seed_gains))


def main():
    metadata = np.load(CONFIG['meta_path'], allow_pickle=True)
    pairs_data = []
    for modelX, modelY in CONFIG['model_pairs']:
        X = np.load(f"embeddings/{CONFIG['dataset']}/{modelX}.npy")
        Y = np.load(f"embeddings/{CONFIG['dataset']}/{modelY}.npy")
        pairs_data.append((modelX, modelY, X, Y))

    # ---- Alpha sweep (ridge or lasso) ----
    if len(CONFIG.get('lasso_alpha_sweep', [])) > 1:
        alphas = CONFIG['lasso_alpha_sweep']
        reg_label = CONFIG['regularizer'].upper()
        print("\n" + "="*60)
        print(f"{reg_label} ALPHA SWEEP over {alphas}")
        print("="*60)

        alpha_gains = {a: [] for a in alphas}
        for modelX, modelY, X, Y in pairs_data:
            for a in alphas:
                gain = run_pair_alpha(modelX, modelY, metadata, X, Y, sweep_alpha=a)
                alpha_gains[a].append(gain)
                print(f"  {modelX}->{modelY}  alpha={a:.0e}  rank1_gain={gain:+.2f}pp")

        mean_gains = {a: float(np.mean(v)) for a, v in alpha_gains.items()}
        print("\n  Mean rank-1 gain (quad - lin) across all pairs x seeds:")
        for a, g in mean_gains.items():
            marker = "  <-- BEST" if a == max(mean_gains, key=mean_gains.get) else ""
            print(f"    alpha={a:.0e}  mean_gain={g:+.2f}pp{marker}")

        best_alpha = max(mean_gains, key=mean_gains.get)
        print(f"\n{'*'*60}")
        print(f"  BEST ALPHA: {best_alpha:.0e}  (mean rank-1 gain {mean_gains[best_alpha]:+.2f}pp across all pairs)")
        print(f"{'*'*60}")
        CONFIG['lasso_alpha'] = best_alpha
        CONFIG['alpha'] = best_alpha
    else:
        best_alpha = CONFIG['alpha'] if CONFIG['regularizer'] == 'ridge' else CONFIG['lasso_alpha']
        print(f"\nUsing fixed alpha={best_alpha} (no sweep)")

    # ---- Full run with chosen alpha ----
    print("\n" + "="*60)
    print(f"FULL RUN  [{CONFIG['regularizer']}, alpha={CONFIG['lasso_alpha']}]")
    print("="*60)
    all_results = []
    for modelX, modelY, X, Y in pairs_data:
        result = run_pair(modelX, modelY, metadata, X, Y)
        all_results.append(result)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"{r['modelX']:10s} -> {r['modelY']:10s}  "
              f"||W_quad||/||W_lin|| = {r['ratio_mean']:.4f} ± {r['ratio_std']:.4f}  "
              f"MSE gain = {r['mse_improvement_mean']:.2f}%")

    with open(f"{CONFIG['out_dir']}/quadratic_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {CONFIG['out_dir']}/quadratic_results.json")


if __name__ == '__main__':
    main()
