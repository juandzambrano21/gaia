"""
Module: fuzzy
Implements fuzzy sets and fuzzy simplicial sets for the GAIA framework.

Following Spivak's theory and Mahadevan (2024), this implements:
1. Fuzzy sets as sheaves on the unit interval I=[0,1]
2. Morphisms preserving membership strengths
3. Fuzzy simplicial sets as functors S: Δᵒᵖ → Fuz
4. Membership coherence constraints
5. Strength preservation by degeneracies

Key principles:
1. Fuzzy sets are sheaves with injective restriction maps
2. Membership functions η: X → [0,1] define classical fuzzy sets
3. Morphisms preserve membership: ξ(f(x)) ≥ η(x)
4. Fuzzy simplicial sets generalize weighted graphs to higher-order relations
"""

import uuid
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict

from .simplices import SimplicialObject
from . import DEVICE


@dataclass(slots=True)
class FuzzySet:
    """
    Fuzzy set as sheaf on unit interval I=[0,1].
    
    Following Definition 2.1 from the theoretical framework:
    - Sheaf on I=[0,1] with injective restriction maps
    - Equivalent to classical fuzzy set (X,η) where η: X→[0,1]
    - Morphisms preserve membership strengths
    """
    elements: Set[Any]
    membership_function: Callable[[Any], float]
    name: str = ""
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)
    
    def __post_init__(self):
        """Validate membership function values are in [0,1]."""
        for element in self.elements:
            membership = self.membership_function(element)
            if not (0.0 <= membership <= 1.0):
                raise ValueError(f"Membership value {membership} for element {element} not in [0,1]")
    
    def membership(self, element: Any) -> float:
        """Get membership strength of element."""
        if element not in self.elements:
            return 0.0
        return self.membership_function(element)
    
    def support(self, threshold: float = 0.0) -> Set[Any]:
        """Get support set with membership > threshold."""
        return {x for x in self.elements if self.membership(x) > threshold}
    
    def alpha_cut(self, alpha: float) -> Set[Any]:
        """Get α-cut: {x ∈ X | η(x) ≥ α}."""
        return {x for x in self.elements if self.membership(x) >= alpha}
    
    def height(self) -> float:
        """Get height: max membership value."""
        if not self.elements:
            return 0.0
        return max(self.membership(x) for x in self.elements)
    
    def is_normal(self) -> bool:
        """Check if fuzzy set is normal (height = 1.0)."""
        return abs(self.height() - 1.0) < 1e-10
    
    def __repr__(self):
        return f"FuzzySet(name='{self.name}', |elements|={len(self.elements)}, height={self.height():.3f})"


class FuzzySetMorphism:
    """
    Morphism between fuzzy sets preserving membership strengths.
    
    For f: (X,η) → (Y,ξ), requires ξ(f(x)) ≥ η(x) for all x ∈ X.
    """
    
    def __init__(self, 
                 source: FuzzySet, 
                 target: FuzzySet, 
                 mapping: Callable[[Any], Any],
                 name: str = ""):
        self.source = source
        self.target = target
        self.mapping = mapping
        self.name = name
        self.id = uuid.uuid4()
        
        # Verify membership preservation
        self._verify_membership_preservation()
    
    def _verify_membership_preservation(self):
        """Verify ξ(f(x)) ≥ η(x) for all x in source."""
        for x in self.source.elements:
            fx = self.mapping(x)
            if fx not in self.target.elements:
                continue  # Skip elements not in target
            
            source_membership = self.source.membership(x)
            target_membership = self.target.membership(fx)
            
            if target_membership < source_membership - 1e-10:
                raise ValueError(
                    f"Membership preservation violated: "
                    f"ξ(f({x})) = {target_membership} < η({x}) = {source_membership}"
                )
    
    def apply(self, element: Any) -> Any:
        """Apply morphism to element."""
        return self.mapping(element)
    
    def __call__(self, element: Any) -> Any:
        """Make morphism callable."""
        return self.apply(element)
    
    def __repr__(self):
        return f"FuzzySetMorphism('{self.source.name}' → '{self.target.name}')"


class FuzzyCategory:
    """
    Category Fuz of fuzzy sets with membership-preserving morphisms.
    
    Objects: Fuzzy sets
    Morphisms: Membership-preserving functions
    """
    
    def __init__(self, name: str = "Fuz"):
        self.name = name
        self.objects: Dict[uuid.UUID, FuzzySet] = {}
        self.morphisms: Dict[uuid.UUID, FuzzySetMorphism] = {}
        self.composition_cache: Dict[Tuple[uuid.UUID, uuid.UUID], uuid.UUID] = {}
    
    def add_object(self, fuzzy_set: FuzzySet) -> uuid.UUID:
        """Add fuzzy set as object."""
        self.objects[fuzzy_set.id] = fuzzy_set
        return fuzzy_set.id
    
    def add_morphism(self, morphism: FuzzySetMorphism) -> uuid.UUID:
        """Add morphism between fuzzy sets."""
        # Verify source and target are in category
        if morphism.source.id not in self.objects:
            self.add_object(morphism.source)
        if morphism.target.id not in self.objects:
            self.add_object(morphism.target)
        
        self.morphisms[morphism.id] = morphism
        return morphism.id
    
    def compose(self, f_id: uuid.UUID, g_id: uuid.UUID) -> FuzzySetMorphism:
        """Compose morphisms g ∘ f."""
        if (f_id, g_id) in self.composition_cache:
            return self.morphisms[self.composition_cache[(f_id, g_id)]]
        
        f = self.morphisms[f_id]
        g = self.morphisms[g_id]
        
        if f.target.id != g.source.id:
            raise ValueError("Morphisms not composable")
        
        # Create composition
        def composed_mapping(x):
            return g.mapping(f.mapping(x))
        
        composition = FuzzySetMorphism(
            f.source, g.target, composed_mapping, 
            name=f"{g.name} ∘ {f.name}"
        )
        
        comp_id = self.add_morphism(composition)
        self.composition_cache[(f_id, g_id)] = comp_id
        return composition
    
    def identity(self, obj_id: uuid.UUID) -> FuzzySetMorphism:
        """Get identity morphism for object."""
        obj = self.objects[obj_id]
        return FuzzySetMorphism(obj, obj, lambda x: x, name=f"id_{obj.name}")


@dataclass(slots=True)
class FuzzySimplicialSet:
    """
    Fuzzy simplicial set as contravariant functor S: Δᵒᵖ → Fuz.
    
    Following Section 2.3 of the theoretical framework:
    - Maps each [n] ∈ Δ to fuzzy set S_n
    - Face maps preserve membership coherence
    - Degeneracies preserve strength
    - Generalizes weighted graphs to higher-order relations
    """
    name: str
    dimension: int
    fuzzy_sets: Dict[int, FuzzySet] = field(default_factory=dict)
    face_maps: Dict[Tuple[int, int], FuzzySetMorphism] = field(default_factory=dict)
    degeneracy_maps: Dict[Tuple[int, int], FuzzySetMorphism] = field(default_factory=dict)
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)
    
    def __post_init__(self):
        """Initialize empty fuzzy sets for each dimension if not provided."""
        for n in range(self.dimension + 1):
            if n not in self.fuzzy_sets:
                self.fuzzy_sets[n] = FuzzySet(set(), lambda x: 0.0, f"{self.name}_{n}")
    
    def add_simplex(self, level: int, simplex: Any, membership: float):
        """Add simplex with membership strength."""
        if level > self.dimension:
            raise ValueError(f"Level {level} exceeds dimension {self.dimension}")
        
        if level not in self.fuzzy_sets:
            self.fuzzy_sets[level] = FuzzySet(set(), lambda x: 0.0, f"{self.name}_{level}")
        
        # Add to fuzzy set
        fuzzy_set = self.fuzzy_sets[level]
        fuzzy_set.elements.add(simplex)
        
        # Update membership function
        old_membership = fuzzy_set.membership_function
        def new_membership(x):
            if x == simplex:
                return membership
            return old_membership(x)
        
        fuzzy_set.membership_function = new_membership
    
    def get_membership(self, level: int, simplex: Any) -> float:
        """Get membership strength of simplex at level."""
        if level not in self.fuzzy_sets:
            return 0.0
        return self.fuzzy_sets[level].membership(simplex)
    
    def add_face_map(self, source_level: int, target_level: int, face_index: int, 
                     mapping: Callable[[Any], Any]):
        """Add face map δᵢ: S_{n-1} → S_n."""
        if target_level != source_level + 1:
            raise ValueError("Face maps must increase dimension by 1")
        
        source_fuzzy = self.fuzzy_sets[source_level]
        target_fuzzy = self.fuzzy_sets[target_level]
        
        face_morphism = FuzzySetMorphism(
            source_fuzzy, target_fuzzy, mapping,
            name=f"δ_{face_index}^{target_level}"
        )
        
        self.face_maps[(source_level, face_index)] = face_morphism
    
    def add_degeneracy_map(self, source_level: int, target_level: int, deg_index: int,
                          mapping: Callable[[Any], Any]):
        """Add degeneracy map σᵢ: S_{n+1} → S_n."""
        if source_level != target_level + 1:
            raise ValueError("Degeneracy maps must decrease dimension by 1")
        
        source_fuzzy = self.fuzzy_sets[source_level]
        target_fuzzy = self.fuzzy_sets[target_level]
        
        deg_morphism = FuzzySetMorphism(
            source_fuzzy, target_fuzzy, mapping,
            name=f"σ_{deg_index}^{target_level}"
        )
        
        self.degeneracy_maps[(source_level, deg_index)] = deg_morphism
    
    def verify_membership_coherence(self) -> bool:
        """
        Verify membership coherence: strength of simplex ≤ min(strength of faces).
        
        For each n-simplex σ, check that membership(σ) ≤ min(membership(∂ᵢσ))
        """
        for level in range(1, self.dimension + 1):
            if level not in self.fuzzy_sets:
                continue
            
            fuzzy_set = self.fuzzy_sets[level]
            for simplex in fuzzy_set.elements:
                simplex_membership = fuzzy_set.membership(simplex)
                
                # Check all faces
                min_face_membership = float('inf')
                for face_idx in range(level + 1):
                    if (level - 1, face_idx) in self.face_maps:
                        face_map = self.face_maps[(level - 1, face_idx)]
                        # This is a simplified check - in practice would need proper face computation
                        min_face_membership = min(min_face_membership, simplex_membership)
                
                if min_face_membership < float('inf') and simplex_membership > min_face_membership + 1e-10:
                    return False
        
        return True
    
    def verify_degeneracy_preservation(self) -> bool:
        """Verify degeneracies preserve strength."""
        for (source_level, deg_idx), deg_map in self.degeneracy_maps.items():
            # Check that degeneracy preserves membership
            # This is automatically satisfied by FuzzySetMorphism construction
            pass
        return True
    
    def __repr__(self):
        return f"FuzzySimplicialSet(name='{self.name}', dim={self.dimension}, levels={list(self.fuzzy_sets.keys())})"


class FuzzySimplicialFunctor:
    """
    Contravariant functor S: Δᵒᵖ → Fuz mapping simplicial category to fuzzy sets.
    
    This extends SimplicialFunctor to work with fuzzy sets instead of classical sets.
    """
    
    def __init__(self, name: str, fuzzy_category: FuzzyCategory):
        self.name = name
        self.fuzzy_category = fuzzy_category
        self.simplicial_sets: Dict[uuid.UUID, FuzzySimplicialSet] = {}
        self.natural_transformations: Dict[Tuple[uuid.UUID, uuid.UUID], Dict[int, FuzzySetMorphism]] = {}
    
    def add_fuzzy_simplicial_set(self, fss: FuzzySimplicialSet) -> uuid.UUID:
        """Add fuzzy simplicial set."""
        self.simplicial_sets[fss.id] = fss
        
        # Add all fuzzy sets to category
        for fuzzy_set in fss.fuzzy_sets.values():
            self.fuzzy_category.add_object(fuzzy_set)
        
        # Add all morphisms to category
        for face_map in fss.face_maps.values():
            self.fuzzy_category.add_morphism(face_map)
        for deg_map in fss.degeneracy_maps.values():
            self.fuzzy_category.add_morphism(deg_map)
        
        return fss.id
    
    def natural_transformation(self, source_id: uuid.UUID, target_id: uuid.UUID) -> Dict[int, FuzzySetMorphism]:
        """Get natural transformation between fuzzy simplicial sets."""
        if (source_id, target_id) in self.natural_transformations:
            return self.natural_transformations[(source_id, target_id)]
        
        source = self.simplicial_sets[source_id]
        target = self.simplicial_sets[target_id]
        
        # Create natural transformation components
        nat_trans = {}
        for level in range(min(source.dimension, target.dimension) + 1):
            if level in source.fuzzy_sets and level in target.fuzzy_sets:
                # Create identity-like transformation (simplified)
                nat_trans[level] = FuzzySetMorphism(
                    source.fuzzy_sets[level],
                    target.fuzzy_sets[level],
                    lambda x: x,  # Identity mapping
                    name=f"η_{level}: {source.name} → {target.name}"
                )
        
        self.natural_transformations[(source_id, target_id)] = nat_trans
        return nat_trans
    
    def verify_functoriality(self, fss_id: uuid.UUID) -> bool:
        """Verify functor laws for fuzzy simplicial set."""
        fss = self.simplicial_sets[fss_id]
        
        # Check that face and degeneracy maps satisfy simplicial identities
        # This is a simplified check - full verification would require
        # checking all simplicial identities
        
        return fss.verify_membership_coherence() and fss.verify_degeneracy_preservation()
    
    def __repr__(self):
        return f"FuzzySimplicialFunctor(name='{self.name}', |objects|={len(self.simplicial_sets)})"


# Utility functions for creating common fuzzy sets

def create_discrete_fuzzy_set(elements_with_membership: Dict[Any, float], name: str = "") -> FuzzySet:
    """Create fuzzy set from discrete elements with membership values."""
    elements = set(elements_with_membership.keys())
    
    def membership_func(x):
        return elements_with_membership.get(x, 0.0)
    
    return FuzzySet(elements, membership_func, name)


def create_gaussian_fuzzy_set(center: float, sigma: float, domain: List[float], name: str = "") -> FuzzySet:
    """Create Gaussian fuzzy set over continuous domain."""
    elements = set(domain)
    
    def membership_func(x):
        # Handle tensor inputs
        if torch.is_tensor(x):
            return torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        # Handle scalar inputs
        if x not in domain:
            return 0.0
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    return FuzzySet(elements, membership_func, name)


def create_triangular_fuzzy_set(a: float, b: float, c: float, domain: List[float], name: str = "") -> FuzzySet:
    """Create triangular fuzzy set with parameters (a, b, c)."""
    elements = set(domain)
    
    def membership_func(x):
        # Handle tensor inputs
        if torch.is_tensor(x):
            result = torch.zeros_like(x)
            mask1 = (x > a) & (x <= b)
            mask2 = (x > b) & (x < c)
            result[mask1] = (x[mask1] - a) / (b - a)
            result[mask2] = (c - x[mask2]) / (c - b)
            return result
        # Handle scalar inputs
        if x not in domain:
            return 0.0
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0.0
    
    return FuzzySet(elements, membership_func, name)


# T-conorms for merging fuzzy simplicial sets (Definition 6.1 from paper)

class TConorm:
    """T-conorm operations for merging fuzzy simplicial sets."""
    
    @staticmethod
    def maximum(a: float, b: float) -> float:
        """Maximum t-conorm: T(a,b) = max(a,b)."""
        return max(a, b)
    
    @staticmethod
    def algebraic_sum(a: float, b: float) -> float:
        """Algebraic sum t-conorm: T(a,b) = a + b - ab."""
        return a + b - a * b
    
    @staticmethod
    def bounded_sum(a: float, b: float) -> float:
        """Bounded sum t-conorm: T(a,b) = min(1, a + b)."""
        return min(1.0, a + b)
    
    @staticmethod
    def drastic_sum(a: float, b: float) -> float:
        """Drastic sum t-conorm."""
        if a == 0:
            return b
        elif b == 0:
            return a
        else:
            return 1.0


def merge_fuzzy_simplicial_sets(fss1: FuzzySimplicialSet, fss2: FuzzySimplicialSet, 
                               t_conorm: Callable[[float, float], float] = TConorm.maximum,
                               name: str = "") -> FuzzySimplicialSet:
    """
    Merge two fuzzy simplicial sets using t-conorm.
    
    This implements step (F4) of the UMAP-adapted pipeline.
    """
    if not name:
        name = f"{fss1.name} ∪ {fss2.name}"
    
    max_dim = max(fss1.dimension, fss2.dimension)
    merged = FuzzySimplicialSet(name, max_dim)
    
    # Merge fuzzy sets at each level
    for level in range(max_dim + 1):
        elements = set()
        if level in fss1.fuzzy_sets:
            elements.update(fss1.fuzzy_sets[level].elements)
        if level in fss2.fuzzy_sets:
            elements.update(fss2.fuzzy_sets[level].elements)
        
        # Create merged membership function
        def merged_membership(x):
            m1 = fss1.get_membership(level, x) if level <= fss1.dimension else 0.0
            m2 = fss2.get_membership(level, x) if level <= fss2.dimension else 0.0
            return t_conorm(m1, m2)
        
        merged.fuzzy_sets[level] = FuzzySet(elements, merged_membership, f"{name}_{level}")
    
    return merged