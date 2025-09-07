"""
Module: fuzzy_encoding
Implements the UMAP-adapted data encoding pipeline (F1-F4) for GAIA.

Following Section 2.4 of the theoretical framework, this implements:
(F1) k-nearest neighbors with Euclidean metric
(F2) Normalization of distances and local radii
(F3) Modified singular functor: (X,d_i) → Sing(X,d_i)
(F4) Merge via t-conorms for global fuzzy simplicial set

This is the critical component that connects real-world data to the 
categorical structure of GAIA, enabling the framework to process
point clouds and high-dimensional data.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import warnings

from ..core.fuzzy import (
    FuzzySet, FuzzySimplicialSet, FuzzySimplicialFunctor, 
    TConorm, merge_fuzzy_simplicial_sets, create_discrete_fuzzy_set
)
from ..core import DEVICE


@dataclass
class UMAPConfig:
    """Configuration for UMAP-adapted fuzzy encoding pipeline using global GAIAConfig."""
    n_neighbors: int = field(default_factory=lambda: _get_global_config().n_neighbors)
    metric: str = field(default_factory=lambda: _get_global_config().metric)
    min_dist: float = field(default_factory=lambda: _get_global_config().min_dist)
    spread: float = field(default_factory=lambda: _get_global_config().spread)
    local_connectivity: float = field(default_factory=lambda: _get_global_config().local_connectivity)
    bandwidth: float = field(default_factory=lambda: _get_global_config().bandwidth)
    t_conorm: Callable[[float, float], float] = TConorm.algebraic_sum
    random_state: int = field(default_factory=lambda: getattr(_get_global_config(), 'random_seed', 42))

def _get_global_config():
    """Helper function to get global configuration with fallback defaults."""
    try:
        from ..training.config import DataConfig
        config = DataConfig()
        # Set fallback defaults if config attributes don't exist
        if not hasattr(config, 'n_neighbors'):
            config.n_neighbors = 15
        if not hasattr(config, 'metric'):
            config.metric = "euclidean"
        if not hasattr(config, 'min_dist'):
            config.min_dist = 0.1
        if not hasattr(config, 'spread'):
            config.spread = 1.0
        if not hasattr(config, 'local_connectivity'):
            config.local_connectivity = 1.0
        if not hasattr(config, 'bandwidth'):
            config.bandwidth = 1.0
        return config
    except ImportError:
        # Fallback to hardcoded defaults if config system not available
        class FallbackConfig:
            class DataConfig:
                n_neighbors = 15
                metric = "euclidean"
                min_dist = 0.1
                spread = 1.0
                local_connectivity = 1.0
                bandwidth = 1.0
            class ReproducibilityConfig:
                seed = 42
            data = DataConfig()
            reproducibility = ReproducibilityConfig()
        return FallbackConfig()


class FuzzyEncodingPipeline:
    """
    UMAP-adapted pipeline for encoding data as fuzzy simplicial sets.
    
    Implements the four-step process (F1-F4):
    F1: Compute k-nearest neighbors
    F2: Normalize distances and compute local radii
    F3: Apply modified singular functor
    F4: Merge via t-conorms
    """
    
    def __init__(self, config: UMAPConfig = None):
        self.config = config or UMAPConfig()
        self.nn_model = None
        self.local_radii = None
        self.distance_matrix = None
        self.fuzzy_simplicial_sets = {}
        
    def step_f1_knn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step F1: Compute k-nearest neighbors with Euclidean metric.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            distances: k-NN distances of shape (n_samples, k)
            indices: k-NN indices of shape (n_samples, k)
        """
        self.nn_model = NearestNeighbors(
            n_neighbors=self.config.n_neighbors + 1,  # +1 because point is its own neighbor
            metric=self.config.metric,
            algorithm='auto'
        )
        
        self.nn_model.fit(X)
        distances, indices = self.nn_model.kneighbors(X)
        
        # Remove self-neighbors (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        self.distance_matrix = distances
        self.indices_matrix = indices
        
        return distances, indices
    
    def step_f2_normalize_distances(self, X: np.ndarray, distances: np.ndarray, 
                                   indices: np.ndarray) -> np.ndarray:
        """
        Step F2: Normalize distances and compute local radii.
        
        Computes local radius ρᵢ for each point and normalizes distances.
        
        Args:
            X: Original data matrix
            distances: k-NN distances from F1
            indices: k-NN indices from F1
            
        Returns:
            normalized_distances: Normalized distance matrix
        """
        n_samples = X.shape[0]
        self.local_radii = np.zeros(n_samples)
        
        # Compute local radii ρᵢ (distance to nearest neighbor)
        for i in range(n_samples):
            if distances[i].size > 0:
                self.local_radii[i] = distances[i][0]  # Distance to nearest neighbor
            else:
                self.local_radii[i] = 1.0  # Default radius
        
        # Normalize distances: d'ᵢⱼ = max(0, (dᵢⱼ - ρᵢ) / σᵢ)
        # where σᵢ is chosen so that Σⱼ exp(-d'ᵢⱼ) = log₂(k)
        normalized_distances = np.zeros_like(distances)
        target_sum = np.log2(self.config.n_neighbors)
        
        for i in range(n_samples):
            # Binary search for σᵢ
            sigma = self._find_sigma(distances[i], self.local_radii[i], target_sum)
            
            # Normalize distances
            for j in range(len(distances[i])):
                normalized_distances[i, j] = max(0, (distances[i, j] - self.local_radii[i]) / sigma)
        
        return normalized_distances
    
    def _find_sigma(self, distances: np.ndarray, rho: float, target: float, 
                   tolerance: float = 1e-5, max_iter: int = 64) -> float:
        """
        Binary search to find σ such that Σⱼ exp(-d'ᵢⱼ) = target.
        """
        sigma_min, sigma_max = 1e-20, 1e3
        
        for _ in range(max_iter):
            sigma = (sigma_min + sigma_max) / 2.0
            
            # Compute normalized distances
            norm_distances = np.maximum(0, (distances - rho) / sigma)
            
            # Compute sum of exponentials
            sum_exp = np.sum(np.exp(-norm_distances))
            
            if abs(sum_exp - target) < tolerance:
                return sigma
            elif sum_exp > target:
                sigma_min = sigma
            else:
                sigma_max = sigma
        
        return sigma
    
    def step_f3_singular_functor(self, X: np.ndarray, normalized_distances: np.ndarray,
                                indices: np.ndarray) -> Dict[int, FuzzySimplicialSet]:
        """
        Step F3: Apply modified singular functor (X,dᵢ) → Sing(X,dᵢ).
        
        Creates local fuzzy simplicial sets for each point based on its
        neighborhood structure.
        
        Args:
            X: Original data matrix
            normalized_distances: Normalized distances from F2
            indices: k-NN indices
            
        Returns:
            Dictionary of local fuzzy simplicial sets
        """
        n_samples = X.shape[0]
        local_fuzzy_sets = {}
        
        for i in range(n_samples):
            # Create local fuzzy simplicial set for point i
            fss = FuzzySimplicialSet(f"local_{i}", dimension=1)  # Start with 1-simplices
            
            # Add 0-simplices (vertices)
            fss.add_simplex(0, i, 1.0)  # Point itself has membership 1.0
            
            for j_idx, j in enumerate(indices[i]):
                # Membership strength based on normalized distance
                membership = np.exp(-normalized_distances[i, j_idx])
                fss.add_simplex(0, j, membership)
            
            # Add 1-simplices (edges) between point i and its neighbors
            for j_idx, j in enumerate(indices[i]):
                edge_membership = np.exp(-normalized_distances[i, j_idx])
                edge = (min(i, j), max(i, j))  # Canonical edge representation
                fss.add_simplex(1, edge, edge_membership)
            
            # Add higher-dimensional simplices for mutual neighbors
            self._add_higher_simplices(fss, i, indices, normalized_distances)
            
            local_fuzzy_sets[i] = fss
        
        self.fuzzy_simplicial_sets = local_fuzzy_sets
        return local_fuzzy_sets
    
    def _add_higher_simplices(self, fss: FuzzySimplicialSet, center_point: int,
                             indices: np.ndarray, normalized_distances: np.ndarray):
        """
        Add higher-dimensional simplices for mutual neighbors.
        
        Creates 2-simplices (triangles) when three points are mutually connected.
        """
        center_neighbors = set(indices[center_point])
        
        # Find triangles: center_point + two mutual neighbors
        for i, neighbor1 in enumerate(indices[center_point]):
            for j, neighbor2 in enumerate(indices[center_point]):
                if i >= j:  # Avoid duplicates
                    continue
                
                # Check if neighbor1 and neighbor2 are also connected
                if neighbor2 in indices[neighbor1] or neighbor1 in indices[neighbor2]:
                    # Create triangle
                    triangle = tuple(sorted([center_point, neighbor1, neighbor2]))
                    
                    # Membership is minimum of edge memberships
                    edge_memberships = [
                        np.exp(-normalized_distances[center_point, i]),
                        np.exp(-normalized_distances[center_point, j])
                    ]
                    
                    # Find membership between neighbor1 and neighbor2
                    if neighbor2 in indices[neighbor1]:
                        n2_idx = np.where(indices[neighbor1] == neighbor2)[0]
                        if len(n2_idx) > 0:
                            edge_memberships.append(
                                np.exp(-normalized_distances[neighbor1, n2_idx[0]])
                            )
                    
                    triangle_membership = min(edge_memberships)
                    
                    # Extend dimension if needed
                    if fss.dimension < 2:
                        fss.dimension = 2
                        fss.fuzzy_sets[2] = FuzzySet(set(), lambda x: 0.0, f"{fss.name}_2")
                    
                    fss.add_simplex(2, triangle, triangle_membership)
    
    def step_f4_merge_tconorms(self, local_fuzzy_sets: Dict[int, FuzzySimplicialSet]) -> FuzzySimplicialSet:
        """
        Step F4: Merge local fuzzy simplicial sets via t-conorms.
        
        Combines all local fuzzy simplicial sets into a global one using
        the configured t-conorm operation.
        
        Args:
            local_fuzzy_sets: Dictionary of local fuzzy simplicial sets
            
        Returns:
            Global merged fuzzy simplicial set
        """
        if not local_fuzzy_sets:
            return FuzzySimplicialSet("empty", 0)
        
        # Start with first local set
        first_key = next(iter(local_fuzzy_sets))
        global_fss = local_fuzzy_sets[first_key]
        global_fss.name = "global_merged"
        
        # Merge with all other local sets
        for i, (key, local_fss) in enumerate(local_fuzzy_sets.items()):
            if key == first_key:
                continue
            
            global_fss = merge_fuzzy_simplicial_sets(
                global_fss, local_fss, 
                t_conorm=self.config.t_conorm,
                name=f"global_merged_{i}"
            )
        
        return global_fss
    
    def encode(self, X: np.ndarray) -> FuzzySimplicialSet:
        """
        Complete encoding pipeline: F1 → F2 → F3 → F4.
        
        Args:
            X: Data matrix of shape (n_samples, n_features) or (batch_size, seq_len, n_features)
            
        Returns:
            Global fuzzy simplicial set encoding the data
        """
        # Handle 3D tensors by reshaping to 2D for NearestNeighbors compatibility
        original_shape = X.shape
        if len(X.shape) == 3:
            # Reshape (batch_size, seq_len, n_features) -> (batch_size * seq_len, n_features)
            X = X.reshape(-1, X.shape[-1])
        elif len(X.shape) > 3:
            # Flatten all dimensions except the last one
            X = X.reshape(-1, X.shape[-1])
        
        # F1: k-nearest neighbors
        distances, indices = self.step_f1_knn(X)
        
        # F2: Normalize distances
        normalized_distances = self.step_f2_normalize_distances(X, distances, indices)
        
        # F3: Modified singular functor
        local_fuzzy_sets = self.step_f3_singular_functor(X, normalized_distances, indices)
        
        # F4: Merge via t-conorms
        global_fss = self.step_f4_merge_tconorms(local_fuzzy_sets)
        
        return global_fss
    
    def encode_batch(self, X: torch.Tensor) -> FuzzySimplicialSet:
        """
        Encode batch of data (PyTorch tensor version).
        
        Args:
            X: Data tensor of shape (batch_size, n_features)
            
        Returns:
            Global fuzzy simplicial set encoding the batch
        """
        # Convert to numpy for sklearn compatibility
        X_np = X.detach().cpu().numpy()
        return self.encode(X_np)
    
    def get_adjacency_matrix(self, fss: FuzzySimplicialSet) -> np.ndarray:
        """
        Extract adjacency matrix from fuzzy simplicial set.
        
        Args:
            fss: Fuzzy simplicial set
            
        Returns:
            Adjacency matrix with fuzzy membership values
        """
        if 1 not in fss.fuzzy_sets:
            return np.array([[]])
        
        edges = fss.fuzzy_sets[1].elements
        vertices = fss.fuzzy_sets[0].elements if 0 in fss.fuzzy_sets else set()
        
        if not vertices:
            return np.array([[]])
        
        # Create vertex to index mapping
        vertex_list = sorted(list(vertices))
        vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
        n_vertices = len(vertex_list)
        
        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_vertices, n_vertices))
        
        # Fill adjacency matrix
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                i, j = edge
                if i in vertex_to_idx and j in vertex_to_idx:
                    membership = fss.get_membership(1, edge)
                    idx_i, idx_j = vertex_to_idx[i], vertex_to_idx[j]
                    adj_matrix[idx_i, idx_j] = membership
                    adj_matrix[idx_j, idx_i] = membership  # Symmetric
        
        return adj_matrix
    
    def visualize_fuzzy_complex(self, fss: FuzzySimplicialSet, threshold: float = None):
        """
        Visualize fuzzy simplicial complex structure.
        
        Args:
            fss: Fuzzy simplicial set to visualize
            threshold: Minimum membership threshold for display (uses global config if None)
        """
        # Use global config default if threshold not provided
        if threshold is None:
            global_config = _get_global_config()
            threshold = getattr(global_config.data, 'visualization_threshold', 0.1)
            
        print(f"Fuzzy Simplicial Complex: {fss.name}")
        print(f"Dimension: {fss.dimension}")
        
        for level in range(fss.dimension + 1):
            if level not in fss.fuzzy_sets:
                continue
            
            fuzzy_set = fss.fuzzy_sets[level]
            significant_elements = [
                (elem, fuzzy_set.membership(elem))
                for elem in fuzzy_set.elements
                if fuzzy_set.membership(elem) >= threshold
            ]
            
            print(f"\nLevel {level} ({len(significant_elements)} elements above threshold {threshold}):")
            for elem, membership in sorted(significant_elements, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {elem}: {membership:.3f}")
            
            if len(significant_elements) > 10:
                print(f"  ... and {len(significant_elements) - 10} more")


# Utility functions for common use cases

def encode_point_cloud(points: np.ndarray, n_neighbors: int = None, 
                      min_dist: float = None) -> FuzzySimplicialSet:
    """
    Encode point cloud as fuzzy simplicial set.
    
    Args:
        points: Point cloud of shape (n_points, n_dims)
        n_neighbors: Number of nearest neighbors (uses global config if None)
        min_dist: Minimum distance parameter (uses global config if None)
        
    Returns:
        Fuzzy simplicial set encoding the point cloud
    """
    # Use global config defaults if parameters not provided
    global_config = _get_global_config()
    if n_neighbors is None:
        n_neighbors = global_config.data.n_neighbors
    if min_dist is None:
        min_dist = global_config.data.min_dist
        
    config = UMAPConfig(n_neighbors=n_neighbors, min_dist=min_dist)
    pipeline = FuzzyEncodingPipeline(config)
    return pipeline.encode(points)


def encode_graph_data(adjacency_matrix: np.ndarray, 
                     node_features: Optional[np.ndarray] = None) -> FuzzySimplicialSet:
    """
    Encode graph data as fuzzy simplicial set.
    
    Args:
        adjacency_matrix: Graph adjacency matrix
        node_features: Optional node feature matrix
        
    Returns:
        Fuzzy simplicial set encoding the graph
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Use node features if available, otherwise use adjacency matrix rows
    if node_features is not None:
        X = node_features
    else:
        X = adjacency_matrix
    
    # Create fuzzy simplicial set
    fss = FuzzySimplicialSet("graph_encoding", dimension=1)
    
    # Add vertices
    for i in range(n_nodes):
        fss.add_simplex(0, i, 1.0)
    
    # Add edges with weights from adjacency matrix
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = adjacency_matrix[i, j]
            if weight > 0:
                edge = (i, j)
                fss.add_simplex(1, edge, weight)
    
    return fss


def create_synthetic_fuzzy_complex(n_points: int = None, n_dims: int = None, 
                                  noise_level: float = None) -> FuzzySimplicialSet:
    """
    Create synthetic fuzzy simplicial complex for testing.
    
    Args:
        n_points: Number of points (uses global config if None)
        n_dims: Dimensionality (uses global config if None)
        noise_level: Noise level (uses global config if None)
        
    Returns:
        Synthetic fuzzy simplicial set
    """
    # Use global config defaults if parameters not provided
    global_config = _get_global_config()
    if n_points is None:
        n_points = getattr(global_config.data, 'n_synthetic_points', 100)
    if n_dims is None:
        n_dims = getattr(global_config.data, 'synthetic_dims', 2)
    if noise_level is None:
        noise_level = getattr(global_config.data, 'synthetic_noise_level', 0.1)
    # Generate synthetic data (circle + noise)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    if n_dims > 2:
        # Add extra dimensions with noise
        extra_dims = np.random.normal(0, noise_level, (n_points, n_dims - 2))
        points = np.column_stack([points, extra_dims])
    
    # Add noise to all dimensions
    points += np.random.normal(0, noise_level, points.shape)
    
    return encode_point_cloud(points)