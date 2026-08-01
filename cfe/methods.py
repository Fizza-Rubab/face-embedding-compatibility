import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA
from collections import defaultdict
from tqdm import tqdm

# torch is only needed for the optional NeuralAlignment baseline.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None



class AlignmentMethod:
    """Base class for alignment methods"""
    def __init__(self, name):
        self.name = name
    
    def fit(self, A_train, B_train):
        """Learn alignment from training data"""
        raise NotImplementedError
    
    def transform(self, A):
        """Apply learned alignment to data"""
        raise NotImplementedError


class ProcrustesAlignment(AlignmentMethod):
    """Orthogonal Procrustes with scaling"""
    def __init__(self):
        super().__init__("Procrustes")
        self.W = None
        self.s = None
    
    def fit(self, A_train, B_train):
        M = A_train.T @ B_train
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        self.W = U @ Vt
        self.s = S.sum() / np.sum(A_train ** 2)
        print(f"  Procrustes: scale s={self.s:.6f}")
        return self
    
    def transform(self, A):
        return (A @ self.W)


class RidgeAlignment(AlignmentMethod):
    """Ridge regression (linear transformation with L2 regularization)"""
    def __init__(self, alpha=1.0):
        super().__init__(f"Ridge(α={alpha})")
        self.model = Ridge(alpha=alpha, fit_intercept=False)
        self.alpha = alpha
    
    def fit(self, A_train, B_train):
        self.model.fit(A_train, B_train)
        print(f"  Ridge: coefficient shape={self.model.coef_.shape}")
        return self
    
    def transform(self, A):
        return self.model.predict(A)




class LinearAlignment(AlignmentMethod):
    """Simple linear transformation (no regularization)"""
    def __init__(self):
        super().__init__("Linear")
        self.W = None
    
    def fit(self, A_train, B_train):
        # Solve: A @ W = B  =>  W = (A.T @ A)^-1 @ A.T @ B
        self.W = np.linalg.lstsq(A_train, B_train, rcond=None)[0]
        print(f"  Linear: W shape={self.W.shape}")
        return self
    
    def transform(self, A):
        return A @ self.W



class CCAAlignment(AlignmentMethod):
    """Canonical Correlation Analysis"""
    def __init__(self, n_components=None):
        super().__init__("CCA")
        self.n_components = n_components
        self.cca = None
    
    def fit(self, A_train, B_train):
        # Determine number of components - BE CONSERVATIVE
        if self.n_components is None:
            # Use much smaller number: min of dims or 50, whichever is smaller
            self.n_components = min(A_train.shape[1], B_train.shape[1], 50)  # Changed from 100
        
        self.cca = CCA(
            n_components=self.n_components,
            max_iter=100,  # Add max iterations
            tol=1e-3,      # Add tolerance
            scale=True     # Add scaling
        )
        self.cca.fit(A_train, B_train)
        print(f"  CCA: n_components={self.n_components}")
        return self
    
    def transform(self, A):
        A_transformed, _ = self.cca.transform(A, np.zeros((A.shape[0], self.n_components)))
        return A_transformed

class NeuralAlignment(AlignmentMethod):
    """Neural network alignment (MLP)"""
    def __init__(self, hidden_dim=512, epochs=50, lr=0.001, device='cuda'):
        super().__init__(f"Neural(h={hidden_dim})")
        if torch is None:
            raise ImportError(
                "NeuralAlignment requires PyTorch. Install torch, or use "
                "Procrustes/Linear/Ridge (the methods reported in the paper)."
            )
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
    
    def fit(self, A_train, B_train):
        input_dim = A_train.shape[1]
        output_dim = B_train.shape[1]
        
        # Build model
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim)
        ).to(self.device)
        
        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        A_tensor = torch.FloatTensor(A_train).to(self.device)
        B_tensor = torch.FloatTensor(B_train).to(self.device)
        
        batch_size = 256
        n_batches = len(A_train) // batch_size
        
        self.model.train()
        print(f"  Neural: Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_loss = 0
            perm = np.random.permutation(len(A_train))
            
            for i in range(n_batches):
                idx = perm[i*batch_size:(i+1)*batch_size]
                A_batch = A_tensor[idx]
                B_batch = B_tensor[idx]
                
                optimizer.zero_grad()
                pred = self.model(A_batch)
                loss = criterion(pred, B_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/n_batches:.6f}")
        
        self.model.eval()
        return self
    
    def transform(self, A):
        with torch.no_grad():
            A_tensor = torch.FloatTensor(A).to(self.device)
            pred = self.model(A_tensor)
            return pred.cpu().numpy()