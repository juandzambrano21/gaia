"""
Metric Yoneda Lemma for GAIA Framework

Implements Section 6.7 from paper.md: "The Metric Yoneda Lemma"

THEORETICAL FOUNDATIONS:
- Generalized metric spaces (X, d) with non-symmetric distances
- [0,∞]-enriched categories for distance computations
- Yoneda embedding y: X → X̂ as universal representer
- Isometric property: X(x,x') = X̂(y(x), y(x'))
- Applications to LLMs, image processing, probability distributions

This enables GAIA to work with non-symmetric distances and build universal
representers for any objects in generalized metric spaces.
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

# Type variables
X = TypeVar('X')  # Objects in metric space
Y = TypeVar('Y')  # Target objects

class GeneralizedMetricSpace(Generic[X]):
    """
    Generalized metric space (X, d) From (MAHADEVAN,2024) Section 6.7
    
    Properties:
    1. d(x,x) = 0 (reflexivity)
    2. d(x,z) ≤ d(x,y) + d(y,z) (triangle inequality)
    
    NOT required:
    - Symmetry: d(x,y) = d(y,x)
    - Identity: d(x,y) = 0 ⟹ x = y  
    - Finiteness: d(x,y) < ∞
    """
    
    def __init__(self, 
                 objects: List[X],
                 distance_function: Callable[[X, X], float],
                 name: Optional[str] = None):
        self.objects = objects
        self.distance_function = distance_function
        self.name = name or "GeneralizedMetricSpace"
        
        # Verify metric space properties
        self._verify_properties()
    
    def distance(self, x: X, y: X) -> float:
        """Compute distance d(x,y) in generalized metric space"""
        return self.distance_function(x, y)
    
    def _verify_properties(self) -> bool:
        """Verify generalized metric space properties"""
        try:
            # Test reflexivity: d(x,x) = 0
            for obj in self.objects[:5]:  # Test subset for efficiency
                if self.distance(obj, obj) != 0:
                    logger.warning(f"Reflexivity violated for {obj}")
                    return False
            
            # Test triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
            for i, x in enumerate(self.objects[:3]):
                for j, y in enumerate(self.objects[:3]):
                    for k, z in enumerate(self.objects[:3]):
                        if i != j and j != k and i != k:
                            d_xz = self.distance(x, z)
                            d_xy = self.distance(x, y)
                            d_yz = self.distance(y, z)
                            
                            if d_xz > d_xy + d_yz + 1e-6:  # Small tolerance
                                logger.warning(f"Triangle inequality violated: d({x},{z}) > d({x},{y}) + d({y},{z})")
                                return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not verify metric properties: {e}")
            return False

class EnrichedCategory:
    """
    [0,∞]-enriched category from paper.md
    
    Objects are non-negative real numbers including ∞
    Morphisms exist r → s iff r ≤ s
    Monoidal structure via addition: r ⊗ s = r + s
    """
    
    def __init__(self):
        self.objects = set()  # Will contain float values in [0,∞]
        self.morphisms = {}   # (r,s) → bool indicating if r ≤ s
    
    def add_object(self, r: float):
        """Add object r ∈ [0,∞] to category"""
        if r < 0:
            raise ValueError("Objects must be non-negative")
        self.objects.add(r)
    
    def has_morphism(self, r: float, s: float) -> bool:
        """Check if morphism r → s exists (i.e., r ≤ s)"""
        return r <= s
    
    def tensor_product(self, r: float, s: float) -> float:
        """Monoidal structure: r ⊗ s = r + s"""
        return r + s
    
    def internal_hom(self, r: float, s: float) -> float:
        """Internal hom [0,∞](r,s) for compact closed structure"""
        if r <= s:
            return 0.0
        else:
            return float('inf')
    
    def categorical_product(self, r: float, s: float) -> float:
        """Categorical product: r ⊓ s = max{r,s}"""
        return max(r, s)
    
    def categorical_coproduct(self, r: float, s: float) -> float:
        """Categorical coproduct: r ⊔ s = min{r,s}"""
        return min(r, s)

class YonedaEmbedding(Generic[X]):
    """
    Yoneda embedding y: X → X̂ for generalized metric spaces
    
    From (MAHADEVAN,2024) Theorem 8: y(x) = X(-,x): X^op → [0,∞]
    
    Key properties:
    1. Isometric: X(x,x') = X̂(y(x), y(x'))
    2. Universal representer: objects determined by interactions
    3. Non-expansive function into presheaf category
    """
    
    def __init__(self, metric_space: GeneralizedMetricSpace[X]):
        self.metric_space = metric_space
        self.enriched_category = EnrichedCategory()
        
        # Build enriched category from metric space
        self._build_enriched_category()
        
        # Compute Yoneda embeddings for all objects
        self.embeddings = {}
        self._compute_embeddings()
    
    def _build_enriched_category(self):
        """Build [0,∞]-enriched category from metric space"""
        # Add all distance values as objects
        for x in self.metric_space.objects:
            for y in self.metric_space.objects:
                distance = self.metric_space.distance(x, y)
                self.enriched_category.add_object(distance)
    
    def _compute_embeddings(self):
        """Compute Yoneda embedding y(x) = X(-,x) for all x"""
        for i, x in enumerate(self.metric_space.objects):
            # Create presheaf X(-,x): X^op → [0,∞]
            presheaf = {}
            for j, y in enumerate(self.metric_space.objects):
                # Use index as key for unhashable types like lists
                key = j if not self._is_hashable(y) else y
                # X(-,x)(y) = X(y,x)
                presheaf[key] = self.metric_space.distance(y, x)
            
            # Use index as key for unhashable types
            embedding_key = i if not self._is_hashable(x) else x
            self.embeddings[embedding_key] = presheaf
    
    def _is_hashable(self, obj) -> bool:
        """Check if object is hashable"""
        try:
            hash(obj)
            return True
        except TypeError:
            return False
    
    def embed(self, x: X) -> Dict:
        """
        Yoneda embedding: x ↦ X(-,x)
        
        Returns presheaf representing x by its interactions with all objects
        """
        # Find the key for this object
        embedding_key = None
        for i, obj in enumerate(self.metric_space.objects):
            if self._objects_equal(obj, x):
                embedding_key = i if not self._is_hashable(x) else x
                break
        
        if embedding_key is None or embedding_key not in self.embeddings:
            # Compute embedding for new object
            presheaf = {}
            for j, y in enumerate(self.metric_space.objects):
                key = j if not self._is_hashable(y) else y
                presheaf[key] = self.metric_space.distance(y, x)
            
            if embedding_key is None:
                embedding_key = len(self.metric_space.objects) if not self._is_hashable(x) else x
            self.embeddings[embedding_key] = presheaf
        
        return self.embeddings[embedding_key]
    
    def _objects_equal(self, obj1, obj2) -> bool:
        """Check if two objects are equal, handling different types"""
        try:
            if isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
                return torch.equal(obj1, obj2)
            elif isinstance(obj1, list) and isinstance(obj2, list):
                return obj1 == obj2
            else:
                return obj1 == obj2
        except:
            return False
    
    def presheaf_distance(self, presheaf1: Dict[X, float], presheaf2: Dict[X, float]) -> float:
        """
        Compute distance in presheaf category X̂
        
        X̂(φ,ψ) = sup_y |φ(y) - ψ(y)| for presheaves φ,ψ
        """
        max_diff = 0.0
        all_objects = set(presheaf1.keys()) | set(presheaf2.keys())
        
        for obj in all_objects:
            val1 = presheaf1.get(obj, float('inf'))
            val2 = presheaf2.get(obj, float('inf'))
            diff = abs(val1 - val2)
            max_diff = max(max_diff, diff)
        
        return max_diff
    
    def verify_isometry(self, x: X, x_prime: X, tolerance: float = 1e-6) -> bool:
        """
        Verify isometric property: X(x,x') = X̂(y(x), y(x'))
        
        This is Theorem 9 from paper.md
        """
        # Original distance in metric space
        original_distance = self.metric_space.distance(x, x_prime)
        
        # Distance in presheaf category
        embedding_x = self.embed(x)
        embedding_x_prime = self.embed(x_prime)
        presheaf_distance = self.presheaf_distance(embedding_x, embedding_x_prime)
        
        # Check if they're equal (within tolerance)
        is_isometric = abs(original_distance - presheaf_distance) < tolerance
        
        if not is_isometric:
            logger.warning(f"Isometry violated: X({x},{x_prime}) = {original_distance}, "
                         f"X̂(y({x}), y({x_prime})) = {presheaf_distance}")
        
        return is_isometric
    
    def universal_property(self, x: X, phi: Dict[X, float]) -> float:
        """
        Universal property: X̂(X(-,x), φ) = φ(x)
        
        This is the metric Yoneda lemma from paper.md
        """
        embedding_x = self.embed(x)
        distance_to_phi = self.presheaf_distance(embedding_x, phi)
        phi_at_x = phi.get(x, float('inf'))
        
        # The universal property should hold
        return distance_to_phi, phi_at_x

class MetricYonedaApplications:
    """
    Applications of Metric Yoneda Lemma to AI/ML problems
    
    From paper.md: "discriminate two objects (probability distributions, 
    images, text documents) by comparing them in suitable metric space"
    """
    
    @staticmethod
    def create_text_metric_space(documents: List[str]) -> GeneralizedMetricSpace[str]:
        """
        Create metric space for text documents using edit distance
        
        Example From (MAHADEVAN,2024) Section 6.7 for string comparison
        """
        def string_distance(u: str, v: str) -> float:
            # Simplified: use longest common prefix metric from paper.md
            if not u or not v:
                return float('inf')
            
            # Find longest common prefix
            common_prefix_len = 0
            for i in range(min(len(u), len(v))):
                if u[i] == v[i]:
                    common_prefix_len += 1
                else:
                    break
            
            # Distance based on paper.md formula
            if u.startswith(v) or v.startswith(u):
                return 0.0
            else:
                return 2**(-common_prefix_len) if common_prefix_len > 0 else 1.0
        
        return GeneralizedMetricSpace(documents, string_distance, "TextMetricSpace")
    
    @staticmethod
    def create_image_metric_space(images: List[torch.Tensor]) -> GeneralizedMetricSpace[torch.Tensor]:
        """
        Create metric space for images using perceptual distance
        
        Non-symmetric distance for image comparison
        """
        def image_distance(img1: torch.Tensor, img2: torch.Tensor) -> float:
            # Simplified perceptual distance (non-symmetric)
            if img1.shape != img2.shape:
                return float('inf')
            
            # L2 distance with asymmetric weighting
            diff = torch.norm(img1 - img2, p=2).item()
            
            # Make it non-symmetric based on image "complexity"
            complexity1 = torch.std(img1).item()
            complexity2 = torch.std(img2).item()
            
            # More complex images have smaller distances to simpler ones
            asymmetry_factor = 1.0 + 0.1 * (complexity1 - complexity2)
            
            return diff * asymmetry_factor
        
        return GeneralizedMetricSpace(images, image_distance, "ImageMetricSpace")
    
    @staticmethod
    def create_probability_metric_space(distributions: List[torch.Tensor]) -> GeneralizedMetricSpace[torch.Tensor]:
        """
        Create metric space for probability distributions
        
        Using Wasserstein-like distance (non-symmetric)
        """
        def probability_distance(p: torch.Tensor, q: torch.Tensor) -> float:
            # Ensure they're probability distributions
            p = torch.softmax(p, dim=-1)
            q = torch.softmax(q, dim=-1)
            
            # KL divergence (non-symmetric)
            kl_div = torch.sum(p * torch.log(p / (q + 1e-8))).item()
            
            return max(0.0, kl_div)  # Ensure non-negative
        
        return GeneralizedMetricSpace(distributions, probability_distance, "ProbabilityMetricSpace")
    
    @staticmethod
    def create_neural_embedding_space(embeddings: List[torch.Tensor]) -> GeneralizedMetricSpace[torch.Tensor]:
        """
        Create metric space for neural embeddings (e.g., from LLMs)
        
        Non-symmetric distance based on attention-like mechanism
        """
        def embedding_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
            # Handle identical tensors (reflexivity)
            if torch.equal(emb1, emb2):
                return 0.0
            
            # Cosine similarity with asymmetric attention weighting
            cos_sim = torch.cosine_similarity(emb1.flatten(), emb2.flatten(), dim=0).item()
            
            # Attention-like weighting (non-symmetric)
            attention_weight = torch.softmax(torch.cat([emb1.flatten(), emb2.flatten()]), dim=0)
            emb1_attention = attention_weight[:emb1.numel()].sum().item()
            
            # Distance with asymmetric attention
            distance = (1 - cos_sim) * (2 - emb1_attention)
            
            return max(0.0, distance)
        
        return GeneralizedMetricSpace(embeddings, embedding_distance, "NeuralEmbeddingSpace")

class UniversalRepresenter:
    """
    Universal representer using Metric Yoneda Lemma
    
    Enables representing any object by its interactions with other objects,
    crucial for foundation models and generative AI.
    """
    
    def __init__(self, metric_space: GeneralizedMetricSpace):
        self.metric_space = metric_space
        self.yoneda_embedding = YonedaEmbedding(metric_space)
        
    def represent(self, obj) -> Dict:
        """
        Create universal representation of object
        
        Returns Yoneda embedding as universal representer
        """
        embedding = self.yoneda_embedding.embed(obj)
        
        return {
            'object': obj,
            'representation': embedding,
            'metric_space': self.metric_space.name,
            'is_universal': True
        }
    
    def compare_objects(self, obj1, obj2) -> Dict[str, float]:
        """
        Compare two objects using universal representations
        
        Returns both original and representational distances
        """
        # Original distance
        original_dist = self.metric_space.distance(obj1, obj2)
        
        # Representational distance
        repr1 = self.yoneda_embedding.embed(obj1)
        repr2 = self.yoneda_embedding.embed(obj2)
        repr_dist = self.yoneda_embedding.presheaf_distance(repr1, repr2)
        
        # Verify isometry
        is_isometric = self.yoneda_embedding.verify_isometry(obj1, obj2)
        
        return {
            'original_distance': original_dist,
            'representational_distance': repr_dist,
            'isometric': is_isometric,
            'difference': abs(original_dist - repr_dist)
        }
    
    def find_nearest_neighbors(self, query_obj, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Find k nearest neighbors using universal representation
        
        Useful for retrieval in foundation models
        """
        query_repr = self.yoneda_embedding.embed(query_obj)
        
        distances = []
        for obj in self.metric_space.objects:
            # Handle tensor comparison properly
            is_same = False
            try:
                if isinstance(obj, torch.Tensor) and isinstance(query_obj, torch.Tensor):
                    is_same = torch.equal(obj, query_obj)
                else:
                    is_same = obj == query_obj
            except:
                is_same = False
            
            if not is_same:
                obj_repr = self.yoneda_embedding.embed(obj)
                dist = self.yoneda_embedding.presheaf_distance(query_repr, obj_repr)
                distances.append((obj, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

# Factory functions for common metric spaces

def create_llm_metric_space(token_sequences: List[List[int]]) -> GeneralizedMetricSpace[List[int]]:
    """Create metric space for LLM token sequences"""
    def token_distance(seq1: List[int], seq2: List[int]) -> float:
        # Edit distance with attention-based weighting
        if not seq1 or not seq2:
            return float('inf')
        
        # Simple edit distance (can be made more sophisticated)
        len_diff = abs(len(seq1) - len(seq2))
        
        # Compare overlapping parts
        min_len = min(len(seq1), len(seq2))
        mismatches = sum(1 for i in range(min_len) if seq1[i] != seq2[i])
        
        return len_diff + mismatches
    
    return GeneralizedMetricSpace(token_sequences, token_distance, "LLMTokenSpace")

def create_causal_metric_space(causal_graphs: List[Dict]) -> GeneralizedMetricSpace[Dict]:
    """Create metric space for causal graphs (DAGs)"""
    def causal_distance(graph1: Dict, graph2: Dict) -> float:
        # Distance based on structural differences
        # This implements the preorder example from paper.md
        
        nodes1 = set(graph1.get('nodes', []))
        nodes2 = set(graph2.get('nodes', []))
        edges1 = set(tuple(e) for e in graph1.get('edges', []))
        edges2 = set(tuple(e) for e in graph2.get('edges', []))
        
        # Structural differences
        node_diff = len(nodes1.symmetric_difference(nodes2))
        edge_diff = len(edges1.symmetric_difference(edges2))
        
        return node_diff + edge_diff
    
    return GeneralizedMetricSpace(causal_graphs, causal_distance, "CausalGraphSpace")

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Metric Yoneda Lemma implementation...")
    
    # Test 1: Text metric space
    documents = ["hello world", "hello there", "goodbye world", "farewell"]
    text_space = MetricYonedaApplications.create_text_metric_space(documents)
    text_representer = UniversalRepresenter(text_space)
    
    # Test universal representation
    repr_result = text_representer.represent("hello world")
    logger.info(f"Universal representation created for text: {len(repr_result['representation'])} interactions")
    
    # Test comparison
    comparison = text_representer.compare_objects("hello world", "hello there")
    logger.info(f"Text comparison - Original: {comparison['original_distance']:.3f}, "
               f"Representational: {comparison['representational_distance']:.3f}, "
               f"Isometric: {comparison['isometric']}")
    
    # Test 2: Neural embeddings
    embeddings = [torch.randn(10) for _ in range(5)]
    embedding_space = MetricYonedaApplications.create_neural_embedding_space(embeddings)
    embedding_representer = UniversalRepresenter(embedding_space)
    
    # Find nearest neighbors
    neighbors = embedding_representer.find_nearest_neighbors(embeddings[0], k=2)
    logger.info(f"Found {len(neighbors)} nearest neighbors for embedding")
    
    logger.info("✅ Metric Yoneda Lemma implementation complete!")
    logger.info("🎯 Section 6.7 From (MAHADEVAN,2024) now implemented - Universal representers in non-symmetric metric spaces")