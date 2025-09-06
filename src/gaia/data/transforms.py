"""Data Transforms for Categorical Learning"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod

class CategoricalTransform(ABC):
    """Base class for categorical transforms"""
    
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        pass

class YonedaTransform(CategoricalTransform):
    """Transform data using Yoneda embedding principles"""
    
    def __init__(
        self,
        embedding_dim: int = 64,
        categorical_dims: Optional[List[int]] = None,
        preserve_structure: bool = True
    ):
        """
        Initialize Yoneda transform
        
        Args:
            embedding_dim: Dimension of Yoneda embedding
            categorical_dims: Dimensions to treat as categorical
            preserve_structure: Whether to preserve categorical structure
        """
        self.embedding_dim = embedding_dim
        self.categorical_dims = categorical_dims or []
        self.preserve_structure = preserve_structure
        self.embeddings = {}
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply Yoneda transform"""
        if len(self.categorical_dims) == 0:
            return self._global_yoneda_embedding(data)
        
        transformed_parts = []
        non_categorical_mask = torch.ones(data.shape[-1], dtype=torch.bool)
        
        # Transform categorical dimensions
        for dim in self.categorical_dims:
            if dim < data.shape[-1]:
                categorical_data = data[..., dim:dim+1]
                embedded = self._categorical_yoneda_embedding(categorical_data, dim)
                transformed_parts.append(embedded)
                non_categorical_mask[dim] = False
        
        # Keep non-categorical dimensions
        if non_categorical_mask.any():
            non_categorical_data = data[..., non_categorical_mask]
            transformed_parts.append(non_categorical_data)
        
        return torch.cat(transformed_parts, dim=-1)
    
    def _global_yoneda_embedding(self, data: torch.Tensor) -> torch.Tensor:
        """Apply global Yoneda embedding"""
        # Implement representable functor embedding
        batch_size = data.shape[0]
        input_dim = data.shape[-1]
        
        # Create embedding matrix if not exists
        if 'global' not in self.embeddings:
            self.embeddings['global'] = nn.Linear(input_dim, self.embedding_dim)
        
        return self.embeddings['global'](data)
    
    def _categorical_yoneda_embedding(self, data: torch.Tensor, dim: int) -> torch.Tensor:
        """Apply categorical-specific Yoneda embedding"""
        # Create dimension-specific embedding
        if dim not in self.embeddings:
            self.embeddings[dim] = nn.Embedding(
                num_embeddings=256,  # Assume max 256 categories
                embedding_dim=self.embedding_dim // len(self.categorical_dims)
            )
        
        # Convert to integer indices for embedding
        indices = (data * 255).long().clamp(0, 255)
        return self.embeddings[dim](indices.squeeze(-1))
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Approximate inverse transform"""
        # This is an approximation since Yoneda embedding may not be invertible
        return data

class SimplicalTransform(CategoricalTransform):
    """Transform data to respect simplicial structure"""
    
    def __init__(
        self,
        simplex_dim: int = 2,
        normalize: bool = True,
        add_boundary: bool = True
    ):
        """
        Initialize simplicial transform
        
        Args:
            simplex_dim: Dimension of simplices
            normalize: Whether to normalize to simplex
            add_boundary: Whether to add boundary information
        """
        self.simplex_dim = simplex_dim
        self.normalize = normalize
        self.add_boundary = add_boundary
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply simplicial transform"""
        if self.normalize:
            # Project to simplex
            data = torch.softmax(data, dim=-1)
        
        if self.add_boundary:
            # Add boundary information for simplicial structure
            boundary_info = self._compute_boundary_info(data)
            data = torch.cat([data, boundary_info], dim=-1)
        
        return data
    
    def _compute_boundary_info(self, data: torch.Tensor) -> torch.Tensor:
        """Compute boundary information for simplicial structure"""
        # Compute face information
        faces = []
        for i in range(data.shape[-1]):
            for j in range(i+1, data.shape[-1]):
                face = data[..., i] * data[..., j]
                faces.append(face.unsqueeze(-1))
        
        if faces:
            return torch.cat(faces, dim=-1)
        else:
            return torch.zeros(data.shape[:-1] + (1,), device=data.device)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse simplicial transform"""
        if self.add_boundary:
            # Remove boundary information
            original_dim = data.shape[-1] - self._boundary_dim(data.shape[-1])
            data = data[..., :original_dim]
        
        return data
    
    def _boundary_dim(self, total_dim: int) -> int:
        """Compute dimension of boundary information"""
        # For n features, boundary has n*(n-1)/2 dimensions
        n = int((-1 + np.sqrt(1 + 8*total_dim)) / 2)
        return n * (n - 1) // 2

class RobustNormalization(CategoricalTransform):
    """Robust normalization that handles outliers and missing values"""
    
    def __init__(
        self,
        method: str = "robust",  # "standard", "robust", "minmax"
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,
        fill_missing: bool = True,
        missing_strategy: str = "median"  # "mean", "median", "mode", "zero"
    ):
        """
        Initialize robust normalization
        
        Args:
            method: Normalization method
            handle_outliers: Whether to handle outliers
            outlier_threshold: Threshold for outlier detection (in std devs)
            fill_missing: Whether to fill missing values
            missing_strategy: Strategy for filling missing values
        """
        self.method = method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.fill_missing = fill_missing
        self.missing_strategy = missing_strategy
        self.stats = {}
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply robust normalization"""
        # Handle missing values
        if self.fill_missing:
            data = self._fill_missing_values(data)
        
        # Handle outliers
        if self.handle_outliers:
            data = self._handle_outliers(data)
        
        # Apply normalization
        if self.method == "standard":
            return self._standard_normalize(data)
        elif self.method == "robust":
            return self._robust_normalize(data)
        elif self.method == "minmax":
            return self._minmax_normalize(data)
        else:
            return data
    
    def _fill_missing_values(self, data: torch.Tensor) -> torch.Tensor:
        """Fill missing values"""
        mask = torch.isnan(data) | torch.isinf(data)
        if not mask.any():
            return data
        
        if self.missing_strategy == "zero":
            data[mask] = 0.0
        elif self.missing_strategy == "mean":
            for i in range(data.shape[-1]):
                col_mask = mask[:, i]
                if col_mask.any():
                    col_mean = data[~col_mask, i].mean()
                    data[col_mask, i] = col_mean
        elif self.missing_strategy == "median":
            for i in range(data.shape[-1]):
                col_mask = mask[:, i]
                if col_mask.any():
                    col_median = data[~col_mask, i].median().values
                    data[col_mask, i] = col_median
        
        return data
    
    def _handle_outliers(self, data: torch.Tensor) -> torch.Tensor:
        """Handle outliers using z-score method"""
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        z_scores = torch.abs((data - mean) / (std + 1e-8))
        
        outlier_mask = z_scores > self.outlier_threshold
        
        # Clip outliers to threshold
        data_clipped = data.clone()
        data_clipped[outlier_mask] = torch.sign(data[outlier_mask]) * \
                                   (mean + self.outlier_threshold * std)[outlier_mask]
        
        return data_clipped
    
    def _standard_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Standard normalization (z-score)"""
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        self.stats['mean'] = mean
        self.stats['std'] = std
        return (data - mean) / (std + 1e-8)
    
    def _robust_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Robust normalization using median and MAD"""
        median = data.median(dim=0, keepdim=True).values
        mad = torch.median(torch.abs(data - median), dim=0, keepdim=True).values
        self.stats['median'] = median
        self.stats['mad'] = mad
        return (data - median) / (mad + 1e-8)
    
    def _minmax_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Min-max normalization"""
        min_vals = data.min(dim=0, keepdim=True).values
        max_vals = data.max(dim=0, keepdim=True).values
        self.stats['min'] = min_vals
        self.stats['max'] = max_vals
        return (data - min_vals) / (max_vals - min_vals + 1e-8)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization"""
        if self.method == "standard":
            return data * self.stats['std'] + self.stats['mean']
        elif self.method == "robust":
            return data * self.stats['mad'] + self.stats['median']
        elif self.method == "minmax":
            return data * (self.stats['max'] - self.stats['min']) + self.stats['min']
        else:
            return data