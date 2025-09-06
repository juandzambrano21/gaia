"""Synthetic Dataset Generation for GAIA"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_classification, make_regression

def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 5,
    n_informative: Optional[int] = None,
    n_redundant: int = 2,
    random_state: int = 42,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic classification dataset
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        random_state: Random seed
        device: Device for tensors
        
    Returns:
        Tuple of (features, labels)
    """
    if n_informative is None:
        n_informative = max(2, n_features // 2)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    return X_tensor, y_tensor

def create_xor_dataset(
    n_samples: int = 1000,
    noise: float = 0.1,
    random_state: int = 42,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create XOR dataset for testing categorical learning
    
    Args:
        n_samples: Number of samples
        noise: Noise level
        random_state: Random seed
        device: Device for tensors
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    # Generate random points in [0,1]^2
    X = np.random.rand(n_samples, 2)
    
    # XOR labels
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    
    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    return X_tensor, y_tensor

def create_regression_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic regression dataset
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        noise: Noise level
        random_state: Random seed
        device: Device for tensors
        
    Returns:
        Tuple of (features, targets)
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    return X_tensor, y_tensor