import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from collections import defaultdict
import pandas as pd
from cfe.methods import *
from cfe.config import DEFAULT_PAIRS, DATASET_IMG_DIRS
from tqdm import tqdm
import json
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    'dataset': 'lfw',
    'img_dir': DATASET_IMG_DIRS['lfw'],
    'meta_path': "./embeddings/lfw/lfw_metadata.npy",
    'out_dir': "./results/lfw_analysis",
    'model_pairs': DEFAULT_PAIRS,
    'train_ratio': 0.7,
    'n_seeds': 5,
    'seeds': [42, 123, 217, 7, 531],
    'max_rank': 50,
}

os.makedirs(CONFIG['out_dir'], exist_ok=True)

# Save configuration for reproducibility
with open(f"{CONFIG['out_dir']}/experiment_config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

print("Experiment Configuration:")
print(json.dumps(CONFIG, indent=2))

# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def split_by_identity(metadata, train_ratio=0.7, seed=42):
    """
    Split dataset by identities (not images).
    Returns train/test indices ensuring no identity overlap.
    
    Args:
        metadata: Array of metadata records with 'identity' field
        train_ratio: Proportion of identities for training
        seed: Random seed for reproducibility
    
    Returns:
        train_idx: Indices of training images
        test_idx: Indices of test images
        split_info: Dictionary with split statistics
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
    
    Args:
        X_train, Y_train: Training embeddings
        X_test, Y_test: Test embeddings
    
    Returns:
        Processed embeddings and statistics
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
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)


def compute_all_metrics_fast(A, B, labels_A, labels_B, max_rank=50):
    """
    Compute all metrics in one pass to avoid recomputing similarity matrix.
    Memory-efficient for large datasets.
    
    For WebFace (490k images), this processes in ~30 seconds vs ~30 minutes.
    """
    print("Computing similarity matrix...")
    S = cosine_similarity(row_norm(A), row_norm(B))
    
    print("Sorting indices...")
    sorted_indices = np.argsort(-S, axis=1)
    sorted_labels = labels_B[sorted_indices]
    
    print("Computing matches...")
    matches = (sorted_labels == labels_A[:, None])
    
    # ===== Rank-1 =====
    print("Computing Rank-1...")
    rank1 = np.sum(matches[:, 0]) / len(A) * 100
    
    # ===== Top-k =====
    print("Computing Top-k...")
    rank5 = np.sum(np.any(matches[:, :5], axis=1)) / len(A) * 100
    rank10 = np.sum(np.any(matches[:, :10], axis=1)) / len(A) * 100
    rank20 = np.sum(np.any(matches[:, :20], axis=1)) / len(A) * 100
    
    # ===== CMC =====
    print("Computing CMC...")
    match_positions = np.argmax(matches, axis=1)
    has_match = np.any(matches, axis=1)
    match_positions[~has_match] = max_rank
    
    cmc = np.zeros(max_rank)
    for rank in range(max_rank):
        cmc[rank] = np.sum(match_positions <= rank) / len(A)
    
    # ===== mAP =====
    print("Computing mAP...")
    matches_float = matches.astype(np.float32)
    n_relevant = np.sum(matches_float, axis=1)
    cumsum_matches = np.cumsum(matches_float, axis=1)
    ranks = np.arange(1, matches.shape[1] + 1)
    precisions = cumsum_matches / ranks
    precisions = precisions * matches_float
    ap = np.sum(precisions, axis=1) / np.maximum(n_relevant, 1e-10)
    valid_queries = n_relevant > 0
    map_score = np.mean(ap[valid_queries]) if np.sum(valid_queries) > 0 else 0.0
    
    print("Done!")
    
    return {
        'rank1': rank1,
        'rank5': rank5,
        'rank10': rank10,
        'rank20': rank20,
        'map': map_score,
        'cmc': cmc,
    }


# ======================================================
# CHUNKED VERSION FOR EXTREMELY LARGE DATASETS
# ======================================================
def compute_all_metrics_chunked(A, B, labels_A, labels_B, max_rank=50, chunk_size=10000):
    """
    Memory-efficient chunked computation.
    Uses argpartition instead of full sort for speed.
    """
    print(f"Input shapes: A={A.shape}, B={B.shape}")
    n_queries = len(A)
    n_gallery = len(B)
    n_chunks = (n_queries + chunk_size - 1) // chunk_size
    
    # Normalize once
    print("Normalizing embeddings...")
    A_norm = row_norm(A)
    B_norm = row_norm(B)
    
    # Accumulators
    total_rank1 = 0
    total_rank5 = 0
    total_rank10 = 0
    total_rank20 = 0
    cmc_cumulative = np.zeros(max_rank)
    aps = []
    
    print(f"Processing {n_queries} queries in {n_chunks} chunks of {chunk_size}...")
    import time
    for chunk_idx in range(n_chunks):
        st = time.time()
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_queries)
        chunk_size_actual = end_idx - start_idx
        
        print(f"  Chunk {chunk_idx+1}/{n_chunks}: queries {start_idx}-{end_idx}...", end='')
        
        # Process this chunk
        A_chunk = A_norm[start_idx:end_idx]
        labels_A_chunk = labels_A[start_idx:end_idx]
        
        # Compute similarity for this chunk only
        S_chunk = A_chunk @ B_norm.T  # Shape: (chunk_size, n_gallery)
        
        # =====================================================
        # KEY OPTIMIZATION: Use argpartition instead of argsort
        # Only need top-k, not full sort!
        # =====================================================
        k_needed = min(max_rank, n_gallery)
        
        # Get top-k indices (much faster than full sort!)
        # Note: argpartition doesn't guarantee order within top-k, so we sort those
        if k_needed < n_gallery:
            # Partition to get top-k
            top_k_indices_unsorted = np.argpartition(-S_chunk, k_needed-1, axis=1)[:, :k_needed]
            
            # Sort just the top-k (much smaller!)
            # Get scores for top-k
            top_k_scores = np.take_along_axis(S_chunk, top_k_indices_unsorted, axis=1)
            
            # Sort within top-k
            sort_order = np.argsort(-top_k_scores, axis=1)
            sorted_indices = np.take_along_axis(top_k_indices_unsorted, sort_order, axis=1)
        else:
            # Gallery smaller than k_needed, just sort normally
            sorted_indices = np.argsort(-S_chunk, axis=1)
        
        # Get labels
        sorted_labels = labels_B[sorted_indices]
        
        # Matches
        matches = (sorted_labels == labels_A_chunk[:, None])
        
        # Rank-1
        rank1_correct = np.sum(matches[:, 0])
        total_rank1 += rank1_correct
        
        # Top-k (vectorized)
        if k_needed >= 5:
            total_rank5 += np.sum(np.any(matches[:, :5], axis=1))
        if k_needed >= 10:
            total_rank10 += np.sum(np.any(matches[:, :10], axis=1))
        if k_needed >= 20:
            total_rank20 += np.sum(np.any(matches[:, :20], axis=1))
        
        # CMC
        match_positions = np.argmax(matches, axis=1)
        has_match = np.any(matches, axis=1)
        match_positions[~has_match] = max_rank
        
        for rank in range(max_rank):
            cmc_cumulative[rank] += np.sum(match_positions <= rank)
        
        # mAP (only compute for queries with matches)
        matches_float = matches.astype(np.float32)
        n_relevant = np.sum(matches_float, axis=1)
        
        # Only compute AP for queries with at least one match
        valid_mask = n_relevant > 0
        n_valid = np.sum(valid_mask)
        
        if n_valid > 0:
            matches_valid = matches_float[valid_mask]
            cumsum_matches = np.cumsum(matches_valid, axis=1)
            ranks = np.arange(1, matches_valid.shape[1] + 1)
            precisions = cumsum_matches / ranks
            precisions = precisions * matches_valid
            ap = np.sum(precisions, axis=1) / n_relevant[valid_mask]
            aps.extend(ap.tolist())
        
        print(f" Rank-1: {rank1_correct}/{chunk_size_actual} ({rank1_correct/chunk_size_actual*100:.1f}%)")
        print(f"time:", time.time() - st)
    
    print("Aggregating final results...")
    
    return {
        'rank1': (total_rank1 / n_queries) * 100,
        'rank5': (total_rank5 / n_queries) * 100,
        'rank10': (total_rank10 / n_queries) * 100,
        'rank20': (total_rank20 / n_queries) * 100,
        'map': np.mean(aps) if len(aps) > 0 else 0.0,
        'cmc': cmc_cumulative / n_queries,
    }


# ======================================================
# USAGE EXAMPLE
# ======================================================

def evaluate_alignment(A, B, labels_A, labels_B, max_rank=50, use_chunked=True):
    """
    Fast evaluation wrapper.
    
    Args:
        use_chunked: Set to True for datasets >100k images to avoid memory issues
    """
    if use_chunked or len(A) > 15000:
        print("Using chunked computation for large dataset...")
        return compute_all_metrics_chunked(A, B, labels_A, labels_B, max_rank)
    else:
        print("Using fast vectorized computation...")
        return compute_all_metrics_fast(A, B, labels_A, labels_B, max_rank)
    

def visualize_query_retrieval_before_after(
        modelX, modelY, metadata, X, Y, 
        seed, method,
        n_queries=3,  # Number of query identities to visualize
        top_k=10,     # Top-k retrievals to show
        save_dir=None):
    """
    Visualize retrieval results before and after alignment for specific queries.
    
    Creates a plot showing:
    - Query image
    - Top-k retrievals BEFORE alignment
    - Top-k retrievals AFTER alignment
    
    Args:
        modelX, modelY: Model names (e.g., 'clip', 'dinov2')
        metadata: Metadata array
        X, Y: Embedding arrays
        seed: Random seed used for split
        method: Alignment method (already fitted)
        n_queries: Number of query identities to visualize
        top_k: Number of top retrievals to show
        save_dir: Directory to save plots (if None, just display)
    
    Returns:
        Path to saved plot (if save_dir provided)
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Get test split
    train_idx, test_idx, _ = split_by_identity(
        metadata, train_ratio=CONFIG['train_ratio'], seed=seed
    )
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    meta_test = metadata[test_idx]
    
    # Preprocess
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc, stats = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)
    
    # Get aligned embeddings
    X_test_aligned = method.transform(X_test_proc)
    
    # Create identity labels for test set
    test_identities = [rec['identity'] for rec in meta_test]
    unique_identities = sorted(list(set(test_identities)))
    
    # Select n_queries random identities
    np.random.seed(seed)  # For reproducibility
    selected_identities = np.random.choice(unique_identities, 
                                          size=min(n_queries, len(unique_identities)), 
                                          replace=False)
    
    # For each selected identity, pick one query image
    query_indices = []
    for identity in selected_identities:
        # Get all test images for this identity
        identity_indices = [i for i, rec in enumerate(meta_test) 
                           if rec['identity'] == identity]
        # Pick the first one as query
        if len(identity_indices) > 0:
            query_indices.append(identity_indices[0])
    
    # Create figure with extra columns for spacing
    n_rows = len(query_indices)
    n_cols = 1 + 1 + top_k + 1 + top_k  # Query + gap + Before top-k + gap + After top-k
    
    fig = plt.figure(figsize=(2.5 * (2 + 2*top_k), 3 * n_rows))
    
    for row_idx, query_idx in enumerate(query_indices):
        query_rec = meta_test[query_idx]
        query_identity = query_rec['identity']
        query_img_path = os.path.join(CONFIG['img_dir'], query_rec['rel_path'])
        
        # Compute similarities BEFORE alignment
        sim_before = cosine_similarity(
            row_norm(X_test_proc[query_idx:query_idx+1]), 
            row_norm(Y_test_proc)
        )[0]
        
        # Compute similarities AFTER alignment
        sim_after = cosine_similarity(
            row_norm(X_test_aligned[query_idx:query_idx+1]), 
            row_norm(Y_test_proc)
        )[0]
        
        # Get top-k indices
        top_k_before = np.argsort(-sim_before)[:top_k]
        top_k_after = np.argsort(-sim_after)[:top_k]
        
        # --- Query Image ---
        ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + 1)
        try:
            query_img = Image.open(query_img_path).convert('RGB')
            ax.imshow(query_img)
        except:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
        ax.set_title(f'Query\nID: {query_identity}', fontsize=10, fontweight='bold')
        ax.axis('off')
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, 
                                        transform=ax.transAxes,
                                        linewidth=3, edgecolor='blue', 
                                        facecolor='none'))
        
        # --- Gap column after query (column 2) ---
        # (automatically empty - just skip it)
        
        # --- BEFORE Alignment Retrievals ---
        for k_idx, gallery_idx in enumerate(top_k_before):
            col = 1 + 1 + k_idx + 1  # Query + gap + k_idx + 1
            ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col)
            
            gallery_rec = meta_test[gallery_idx]
            gallery_identity = gallery_rec['identity']
            gallery_img_path = os.path.join(CONFIG['img_dir'], gallery_rec['rel_path'])
            
            # Load and display image
            try:
                gallery_img = Image.open(gallery_img_path).convert('RGB')
                ax.imshow(gallery_img)
            except:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            
            # Title with rank and similarity
            is_correct = gallery_identity == query_identity
            title_color = 'green' if is_correct else 'red'
            
            ax.set_title(f'#{k_idx+1}\nSim: {sim_before[gallery_idx]:.3f}\nID: {gallery_identity}',
                        fontsize=8, color=title_color)
            ax.axis('off')
            
            # Border: green if correct, red if wrong
            border_color = 'green' if is_correct else 'red'
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, 
                                           transform=ax.transAxes,
                                           linewidth=2, edgecolor=border_color, 
                                           facecolor='none'))
        
        # Add "BEFORE" label
        if row_idx == 0:
            ax = plt.subplot(n_rows, n_cols, 1 + 1 + top_k//2 + 1)
            ax.text(0.5, 1.15, 'BEFORE Alignment', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        # --- Gap column between before and after (automatically empty) ---
        
        # --- AFTER Alignment Retrievals ---
        for k_idx, gallery_idx in enumerate(top_k_after):
            col = 1 + 1 + top_k + 1 + k_idx + 1  # Query + gap + Before + gap + k_idx + 1
            ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col)
            
            gallery_rec = meta_test[gallery_idx]
            gallery_identity = gallery_rec['identity']
            gallery_img_path = os.path.join(CONFIG['img_dir'], gallery_rec['rel_path'])
            
            # Load and display image
            try:
                gallery_img = Image.open(gallery_img_path).convert('RGB')
                ax.imshow(gallery_img)
            except:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            
            # Title with rank and similarity
            is_correct = gallery_identity == query_identity
            title_color = 'green' if is_correct else 'red'
            
            ax.set_title(f'#{k_idx+1}\nSim: {sim_after[gallery_idx]:.3f}\nID: {gallery_identity}',
                        fontsize=8, color=title_color)
            ax.axis('off')
            
            # Border: green if correct, red if wrong
            border_color = 'green' if is_correct else 'red'
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, 
                                           transform=ax.transAxes,
                                           linewidth=2, edgecolor=border_color, 
                                           facecolor='none'))
        
        # Add "AFTER" label
        if row_idx == 0:
            ax = plt.subplot(n_rows, n_cols, 1 + 1 + top_k + 1 + top_k//2 + 1)
            ax.text(0.5, 1.15, f'AFTER {method.name}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Overall title
    fig.suptitle(f'Cross-Modal Retrieval: {modelX} → {modelY}\n'
                 f'Method: {method.name}, Seed: {seed}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/before_after_{modelX}_to_{modelY}_{method.name}_seed{seed}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None
    
def visualize_pca_before_after_alignment(
        modelX, modelY, metadata, X, Y,
        seed, method,
        n_identities=10,
        save_dir=None,
        reduction="tsne"   # "pca" or "tsne"
    ):
    """
    Visualize embeddings in 2D space (PCA or t-SNE) before and after alignment.
    """


    if reduction not in ["pca", "tsne"]:
        raise ValueError("reduction must be 'pca' or 'tsne'")

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------
    train_idx, test_idx, _ = split_by_identity(
        metadata, train_ratio=CONFIG['train_ratio'], seed=seed
    )

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    meta_test = metadata[test_idx]

    # Preprocess
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc, stats = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)

    # Align
    X_test_aligned = method.transform(X_test_proc)

    # ---------------------------------------------------------
    # Identity selection
    # ---------------------------------------------------------
    test_identities = [rec['identity'] for rec in meta_test]
    unique_identities = sorted(list(set(test_identities)))

    identity_counts = {
        identity: sum(1 for id in test_identities if id == identity)
        for identity in unique_identities
    }

    top_identities = sorted(identity_counts.items(),
                            key=lambda x: -x[1])[:n_identities]

    selected_identities = [identity for identity, _ in top_identities]

    selected_mask = np.array(
        [identity in selected_identities for identity in test_identities]
    )

    X_test_filtered = X_test_proc[selected_mask]
    Y_test_filtered = Y_test_proc[selected_mask]
    X_test_aligned_filtered = X_test_aligned[selected_mask]

    identities_filtered = [
        test_identities[i]
        for i in range(len(test_identities))
        if selected_mask[i]
    ]

    # ---------------------------------------------------------
    # Color mapping
    # ---------------------------------------------------------
    identity_to_idx = {
        identity: i for i, identity in enumerate(selected_identities)
    }

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_identities)))
    point_colors = [
        colors[identity_to_idx[identity]]
        for identity in identities_filtered
    ]

    # ---------------------------------------------------------
    # Dimensionality reduction helper
    # ---------------------------------------------------------
    def reduce_2d(data):
        if reduction == "pca":
            reducer = PCA(n_components=2)
            Z2 = reducer.fit_transform(data)
            return Z2, reducer.explained_variance_ratio_
        else:  # tsne
            reducer = TSNE(
                n_components=2,
                perplexity=min(30, len(data)//5),
                random_state=seed,
                init="pca",
                learning_rate="auto"
            )
            Z2 = reducer.fit_transform(data)
            return Z2, None

    # ---------------------------------------------------------
    # BEFORE Alignment
    # ---------------------------------------------------------
    combined_before = np.vstack([X_test_filtered, Y_test_filtered])
    combined_before_2d, var_before = reduce_2d(combined_before)

    n = len(X_test_filtered)
    X_before_2d = combined_before_2d[:n]
    Y_before_2d = combined_before_2d[n:]

    # ---------------------------------------------------------
    # AFTER Alignment
    # ---------------------------------------------------------
    combined_after = np.vstack([X_test_aligned_filtered, Y_test_filtered])
    combined_after_2d, var_after = reduce_2d(combined_after)

    X_after_2d = combined_after_2d[:n]
    Y_after_2d = combined_after_2d[n:]

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ================= BEFORE =================
    ax = axes[0]

    for i, color in enumerate(point_colors):
        ax.scatter(Y_before_2d[i, 0], Y_before_2d[i, 1],
                   c=[color], s=100, alpha=0.6,
                   edgecolors='black', linewidth=1,
                   marker='o')

        ax.scatter(X_before_2d[i, 0], X_before_2d[i, 1],
                   c=[color], s=200, alpha=0.8,
                   edgecolors='black', linewidth=1.5,
                   marker='*')

    if reduction == "pca":
        ax.set_xlabel(f'PC1 ({var_before[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_before[1]*100:.1f}%)', fontsize=12)
    else:
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)

    ax.set_title(f'BEFORE Alignment\n{modelX} (★) vs {modelY} (●)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # ================= AFTER =================
    ax = axes[1]

    for i, color in enumerate(point_colors):
        ax.scatter(Y_after_2d[i, 0], Y_after_2d[i, 1],
                   c=[color], s=100, alpha=0.6,
                   edgecolors='black', linewidth=1,
                   marker='o')

        ax.scatter(X_after_2d[i, 0], X_after_2d[i, 1],
                   c=[color], s=200, alpha=0.8,
                   edgecolors='black', linewidth=1.5,
                   marker='*')

    if reduction == "pca":
        ax.set_xlabel(f'PC1 ({var_after[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_after[1]*100:.1f}%)', fontsize=12)
    else:
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)

    ax.set_title(f'AFTER {method.name}\n{modelX} (★) → {modelY} (●)',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------
    handles = [
        mpatches.Patch(color=colors[i], label=f'ID: {identity}')
        for i, identity in enumerate(selected_identities)
    ]

    handles.append(plt.Line2D([0], [0], marker='*', color='w',
                              markerfacecolor='gray', markersize=15,
                              label=f'{modelX} (query)', linestyle='None'))

    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='gray', markersize=10,
                              label=f'{modelY} (gallery)', linestyle='None'))

    fig.legend(handles=handles, loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               fontsize=10, frameon=True)

    fig.suptitle(f'{reduction.upper()} Visualization: {modelX} → {modelY}\n'
                 f'Method: {method.name}, Seed: {seed}',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{reduction}_{modelX}_to_{modelY}_{method.name}_seed{seed}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: {save_path}")
        return save_path
    else:
        plt.show()
        return None
    
# ======================================================
# SINGLE SEED EXPERIMENT
# ======================================================

def run_single_seed_experiment(modelX, modelY, metadata, X, Y, seed, config):
    """
    Run complete experiment for a single random seed.
    NOW WITH VISUALIZATIONS!
    """
    print(f"\n{'='*70}")
    print(f"SEED {seed}: {modelX} → {modelY}")
    print(f"{'='*70}")
    
    print("Metadata length:", len(metadata))
    print("Embedding length:", X.shape[0], "Embedding dimension:", X.shape[1])
    assert len(metadata) == X.shape[0], "Metadata and embeddings misaligned!"

    train_idx, test_idx, split_info = split_by_identity(
        metadata, train_ratio=config['train_ratio'], seed=seed
    )
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    meta_test = metadata[test_idx]
    
    test_identities = [rec['identity'] for rec in meta_test]
    unique_identities = sorted(list(set(test_identities)))
    identity_to_label = {identity: i for i, identity in enumerate(unique_identities)}
    test_labels = np.array([identity_to_label[identity] for identity in test_identities])
    
    X_train_proc, Y_train_proc, X_test_proc, Y_test_proc, preprocess_stats = \
        preprocess_embeddings(X_train, Y_train, X_test, Y_test)
    print(f"Preprocessing stats: dim_X={preprocess_stats['dim_X']}, dim_Y={preprocess_stats['dim_Y']}, ")
    
    # ... [existing alignment methods] ...
    
    alignment_methods = [
        ProcrustesAlignment(),
        LinearAlignment(),
        RidgeAlignment(alpha=0.1),
        # RidgeAlignment(alpha=1.0),
        # RidgeAlignment(alpha=10.0),
    ]
    
    seed_results = {
        'seed': seed,
        'split_info': split_info,
        'preprocess_stats': {k: v for k, v in preprocess_stats.items() 
                            if k not in ['mean_X', 'mean_Y']},
        'methods': {}
    }
    
    # Baseline
    print("\nBaseline (No Alignment)...")
    baseline_metrics = evaluate_alignment(
        X_test_proc, Y_test_proc, test_labels, test_labels, 
        max_rank=config['max_rank']
    )
    seed_results['methods']['Baseline'] = baseline_metrics
    print(f"  Rank-1: {baseline_metrics['rank1']:.2f}%, mAP: {baseline_metrics['map']:.4f}")
    
    # Run each alignment method
    for method in alignment_methods:
        print(f"\n{method.name}...")
        
        try:
            # Train alignment
            method.fit(X_train_proc, Y_train_proc)
            
            # Transform test set
            X_test_aligned = method.transform(X_test_proc)
            
            # Evaluate
            metrics = evaluate_alignment(
                X_test_aligned, Y_test_proc,
                test_labels, test_labels,
                max_rank=config['max_rank']
            )
            
            seed_results['methods'][method.name] = metrics
            
            print(f"  Rank-1: {metrics['rank1']:.2f}% (Δ={metrics['rank1']-baseline_metrics['rank1']:+.2f}%)")
            print(f"  mAP: {metrics['map']:.4f} (Δ={metrics['map']-baseline_metrics['map']:+.4f})")
            
            # ============================================================
            # VISUALIZATIONS
            # ============================================================
            vis_dir = f"{config['out_dir']}/visualizations"
            
            if seed == config['seeds'][0]:
                # 1. Before/After retrieval visualization
                visualize_query_retrieval_before_after(
                    modelX, modelY, metadata, X, Y,
                    seed=seed,
                    method=method,
                    n_queries=3,
                    top_k=10,
                    save_dir=vis_dir
                )
            
            # 2. PCA visualization (only for first seed to avoid too many plots)
            if seed == config['seeds'][0]:
                visualize_pca_before_after_alignment(
                    modelX, modelY, metadata, X, Y,
                    seed=seed,
                    method=method,
                    n_identities=10,
                    save_dir=vis_dir
                )
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            seed_results['methods'][method.name] = {
                'rank1': 0.0, 'rank5': 0.0, 'rank10': 0.0, 
                'rank20': 0.0, 'map': 0.0, 'error': str(e)
            }
    
    return seed_results


# ======================================================
# MULTI-SEED EXPERIMENT
# ======================================================

def run_multi_seed_experiment(modelX, modelY, config):
    """
    Run experiments across multiple random seeds.
    
    Returns:
        all_results: List of results for each seed
        aggregated_results: Mean and std across seeds
    """
    print(f"\n{'#'*70}")
    print(f"MULTI-SEED EXPERIMENT: {modelX} → {modelY}")
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
        seed_results = run_single_seed_experiment(
            modelX, modelY, metadata, X, Y, seed, config
        )
        all_results.append(seed_results)
    
    # Aggregate results across seeds
    aggregated_results = aggregate_multi_seed_results(all_results)
    
    return all_results, aggregated_results


def aggregate_multi_seed_results(all_results):
    """
    Aggregate results across multiple seeds.
    
    Returns:
        Dictionary with mean and std for each metric
    """
    # Get all method names
    method_names = list(all_results[0]['methods'].keys())
    
    aggregated = {}
    
    for method_name in method_names:
        # Collect metrics across all seeds
        metrics_across_seeds = defaultdict(list)
        
        for seed_result in all_results:
            if method_name in seed_result['methods']:
                method_metrics = seed_result['methods'][method_name]
                for metric_name, value in method_metrics.items():
                    if metric_name != 'cmc' and metric_name != 'error':  # Skip CMC curve and errors
                        metrics_across_seeds[metric_name].append(value)
        
        # Compute mean and std
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
# RESULTS VISUALIZATION AND EXPORT
# ======================================================

def create_results_table(aggregated_results, save_path):
    """
    Create and save results table with mean ± std.
    """
    rows = []
    
    for method_name, metrics in aggregated_results.items():
        row = {'Method': method_name}
        
        # Add each metric as "mean ± std"
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
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nResults table saved to: {save_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (Mean ± Std across all seeds)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def plot_cmc_curves_aggregated(all_results, aggregated_results, modelX, modelY, save_path):
    """
    Plot CMC curves with mean and confidence intervals.
    """
    method_names = list(aggregated_results.keys())
    max_rank = len(all_results[0]['methods']['Baseline']['cmc'])
    
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    
    for i, method_name in enumerate(method_names):
        # Collect CMC curves across all seeds
        cmc_curves = []
        for seed_result in all_results:
            if method_name in seed_result['methods']:
                if 'cmc' in seed_result['methods'][method_name]:
                    cmc_curves.append(seed_result['methods'][method_name]['cmc'])
        
        if len(cmc_curves) == 0:
            continue
        
        cmc_curves = np.array(cmc_curves) * 100  # Convert to percentage
        
        # Compute mean and std
        cmc_mean = np.mean(cmc_curves, axis=0)
        cmc_std = np.std(cmc_curves, axis=0)
        
        ranks = np.arange(1, max_rank + 1)
        
        # Plot mean line
        linestyle = '--' if method_name == 'Baseline' else '-'
        linewidth = 2.5 if method_name == 'Baseline' else 1.5
        
        plt.plot(ranks, cmc_mean, linestyle=linestyle, linewidth=linewidth,
                label=method_name, color=colors[i])
        
        # Plot confidence interval (mean ± std)
        plt.fill_between(ranks, cmc_mean - cmc_std, cmc_mean + cmc_std,
                        alpha=0.2, color=colors[i])
    
    plt.xlabel('Rank', fontsize=13)
    plt.ylabel('Recognition Rate (%)', fontsize=13)
    plt.title(f'CMC Curves: {modelX} → {modelY}\n(Mean ± Std across {len(all_results)} seeds)',
              fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.xlim([1, max_rank])
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"CMC curve saved to: {save_path}")
    plt.close()


def save_all_results(all_results, aggregated_results, modelX, modelY, config):
    """
    Save all results to files.
    """
    out_dir = config['out_dir']
    prefix = f"{modelX}_to_{modelY}_{config['dataset']}"
    
    # Save raw results (all seeds)
    raw_results_path = f"{out_dir}/{prefix}_all_seeds.json"
    
    # Convert CMC curves to lists for JSON serialization
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
    
    # Save aggregated results table
    table_path = f"{out_dir}/{prefix}_aggregated_results.csv"
    create_results_table(aggregated_results, table_path)
    
    # Save CMC curves
    cmc_path = f"{out_dir}/{prefix}_cmc_curves.png"
    plot_cmc_curves_aggregated(all_results, aggregated_results, 
                               modelX, modelY, cmc_path)


# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("REPRODUCIBLE MULTI-SEED ALIGNMENT EXPERIMENTS")
    print("="*70)
    
    # Run experiments for each model pair
    for modelX, modelY in CONFIG['model_pairs']:
        all_results, aggregated_results = run_multi_seed_experiment(
            modelX, modelY, CONFIG
        )
        
        # Save all results
        save_all_results(all_results, aggregated_results, 
                        modelX, modelY, CONFIG)
        
        print(f"\n✓ Completed: {modelX} → {modelY}")
        print(f"  Results saved to: {CONFIG['out_dir']}")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()