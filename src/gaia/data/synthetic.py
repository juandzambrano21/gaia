"""Synthetic Dataset Generation for GAIA"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_classification, make_regression

def _get_data_config():
    """Helper function to get data configuration with fallback defaults."""
    try:
        from ..training.config import DataConfig
        return DataConfig()
    except ImportError:
        # Fallback to hardcoded defaults if config system not available
        class FallbackDataConfig:
            n_samples = 1000
            n_features = 20
            n_classes = 5
            n_redundant = 2
            random_seed = 42
            noise_level = 0.1
            n_informative = 10
        return FallbackDataConfig()

def create_synthetic_dataset(
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    n_classes: Optional[int] = None,
    n_informative: Optional[int] = None,
    n_redundant: Optional[int] = None,
    random_state: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic classification dataset
    
    Args:
        n_samples: Number of samples (uses config default if None)
        n_features: Number of features (uses config default if None)
        n_classes: Number of classes (uses config default if None)
        n_informative: Number of informative features (uses config default if None)
        n_redundant: Number of redundant features (uses config default if None)
        random_state: Random seed (uses config default if None)
        device: Device for tensors
        
    Returns:
        Tuple of (features, labels)
    """
    # Get config defaults
    config = _get_data_config()
    
    # Use config values if parameters not provided
    if n_samples is None:
        n_samples = getattr(config, 'n_samples', 1000)
    if n_features is None:
        n_features = getattr(config, 'n_features', 20)
    if n_classes is None:
        n_classes = getattr(config, 'n_classes', 5)
    if n_redundant is None:
        n_redundant = getattr(config, 'n_redundant', 2)
    if random_state is None:
        random_state = getattr(config, 'random_seed', 42)
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
    n_samples: Optional[int] = None,
    noise: Optional[float] = None,
    random_state: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create XOR dataset for testing categorical learning
    
    Args:
        n_samples: Number of samples (uses config default if None)
        noise: Noise level (uses config default if None)
        random_state: Random seed (uses config default if None)
        device: Device for tensors
        
    Returns:
        Tuple of (features, labels)
    """
    # Get config defaults
    config = _get_data_config()
    
    # Use config values if parameters not provided
    if n_samples is None:
        n_samples = getattr(config, 'n_samples', 1000)
    if noise is None:
        noise = getattr(config, 'noise_level', 0.1)
    if random_state is None:
        random_state = getattr(config, 'random_seed', 42)
    
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
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    n_informative: Optional[int] = None,
    noise: Optional[float] = None,
    random_state: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic regression dataset
    
    Args:
        n_samples: Number of samples (uses config default if None)
        n_features: Number of features (uses config default if None)
        n_informative: Number of informative features (uses config default if None)
        noise: Noise level (uses config default if None)
        random_state: Random seed (uses config default if None)
        device: Device for tensors
        
    Returns:
        Tuple of (features, targets)
    """
    # Get config defaults
    config = _get_data_config()
    
    # Use config values if parameters not provided
    if n_samples is None:
        n_samples = getattr(config, 'n_samples', 1000)
    if n_features is None:
        n_features = getattr(config, 'n_features', 20)
    if n_informative is None:
        n_informative = getattr(config, 'n_informative', 10)
    if noise is None:
        noise = getattr(config, 'noise_level', 0.1)
    if random_state is None:
        random_state = getattr(config, 'random_seed', 42)
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