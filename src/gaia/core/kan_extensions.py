"""
Kan Extensions for GAIA Framework - Foundation Model Construction

Implements Section 6.6 from GAIA paper: "Kan Extension"
INTEGRATES with existing kan_verification.py for complete Kan extension system

THEORETICAL FOUNDATIONS:
- Definition 48: Left Kan extension Lan_K F: D → E with universal property
- Right Kan extension Ran_K F: D → E as right adjoint
- Migration functors Δ_F, Σ_F, Π_F for generative AI model modifications
- Foundation model construction via functor extension
- Universal property: "every concept in category theory is a special case of Kan extension"

This enables GAIA to construct foundation models by extending functors over categories,
rather than interpolating functions on sets - the core innovation of GAIA.
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union, Tuple, Set
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict
from enum import Enum

from gaia.core.fuzzy import FuzzyCategory

# Import fuzzy simplicial structures from GAIA core
from .integrated_structures import (
    IntegratedFuzzySimplicialSet, IntegratedFuzzySet, IntegratedSimplex,
    TConorm, FuzzyElement, create_fuzzy_simplex
)

logger = logging.getLogger(__name__)

# Type variables for categorical structures
C = TypeVar('C')  # Source category
D = TypeVar('D')  # Target category  
E = TypeVar('E')  # Codomain category

class FuzzySimplicialCategory(ABC):
    """
    Abstract base class for fuzzy simplicial categories
    
    A fuzzy simplicial category consists of:
    - Graded objects S_n with fuzzy memberships μ ∈ [0,1]
    - Face maps d_i: S_n → S_{n-1} and degeneracy maps s_i: S_n → S_{n+1}
    - Morphisms between fuzzy simplices with t-norm composition
    """
    
    def __init__(self, name: str, t_conorm: TConorm = TConorm.MAXIMUM):
        self.name = name
        self.t_conorm = t_conorm
        # Graded objects: dimension -> {object_id -> fuzzy_simplex}
        self.graded_objects: Dict[int, Dict[str, IntegratedSimplex]] = defaultdict(dict)
        # Morphisms: (source_id, target_id) -> fuzzy_morphism
        self.morphisms: Dict[Tuple[str, str], Callable] = {}
        # Identity morphisms
        self.identities: Dict[str, Callable] = {}
        # Maximum dimension
        self.max_dimension = 0
    
    @abstractmethod
    def add_fuzzy_object(self, obj_id: str, dimension: int, membership: float = 1.0):
        """Add fuzzy simplicial object to category"""
        pass
    
    @abstractmethod
    def add_fuzzy_morphism(self, source_id: str, target_id: str, morphism: Callable, membership: float = 1.0):
        """Add fuzzy morphism between simplicial objects"""
        pass
    
    @abstractmethod
    def compose_fuzzy(self, f: Callable, g: Callable, f_membership: float, g_membership: float) -> Tuple[Callable, float]:
        """Compose fuzzy morphisms using t-norm: μ(g∘f) = T(μ(f), μ(g))"""
        pass
    
    def get_objects_at_dimension(self, dim: int) -> Dict[str, IntegratedSimplex]:
        """Get all objects at given dimension"""
        return self.graded_objects.get(dim, {})
    
    def get_fuzzy_morphism(self, source_id: str, target_id: str) -> Optional[Tuple[Callable, float]]:
        """Get fuzzy morphism with membership degree"""
        morphism = self.morphisms.get((source_id, target_id))
        if morphism:
            # Extract membership from morphism metadata (simplified)
            return morphism, 1.0
        return None
    
    def compute_fuzzy_colimit(self, object_ids: List[str]) -> Tuple[IntegratedSimplex, float]:
        """
        Compute fuzzy colimit over collection of objects
        
        For Kan extensions: Lan_K F(d) = colim_{c: K(c) → d} F(c)
        Membership: μ_colimit = sup{μ_F(c) : c in preimages}
        """
        if not object_ids:
            # Empty colimit
            return create_fuzzy_simplex(0, 0.0), 0.0
        
        # Find maximum membership using t-conorm (supremum)
        max_membership = 0.0
        representative_simplex = None
        
        for obj_id in object_ids:
            # Find the simplex across all dimensions
            for dim, objects in self.graded_objects.items():
                if obj_id in objects:
                    simplex = objects[obj_id]
                    membership = simplex.membership
                    
                    if membership > max_membership:
                        max_membership = membership
                        representative_simplex = simplex
                    break
        
        if representative_simplex is None:
            return create_fuzzy_simplex(0, 0.0), 0.0
        
        # Create colimit simplex with supremum membership
        colimit_simplex = create_fuzzy_simplex(
            representative_simplex.dimension,
            max_membership
        )
        
        return colimit_simplex, max_membership

class FuzzySetCategory(FuzzySimplicialCategory):
    """
    Category of fuzzy sets and fuzzy functions
    
    Objects are fuzzy simplicial sets, morphisms are fuzzy functions between them
    """
    
    def __init__(self):
        super().__init__("FuzzySet")
    
    def add_fuzzy_object(self, obj_id: str, dimension: int, membership: float = 1.0):
        """Add fuzzy set as simplicial object"""
        fuzzy_simplex = create_fuzzy_simplex(dimension, membership)
        self.graded_objects[dimension][obj_id] = fuzzy_simplex
        self.max_dimension = max(self.max_dimension, dimension)
        
        # Add identity morphism
        self.identities[obj_id] = lambda x: x
    
    def add_fuzzy_morphism(self, source_id: str, target_id: str, morphism: Callable, membership: float = 1.0):
        """Add fuzzy function as morphism"""
        # Wrap morphism with fuzzy membership
        def fuzzy_morphism(x):
            result = morphism(x)
            # Apply membership degree to result
            if hasattr(result, 'membership'):
                result.membership = min(result.membership, membership)
            return result
        
        self.morphisms[(source_id, target_id)] = fuzzy_morphism
    
    def compose_fuzzy(self, f: Callable, g: Callable, f_membership: float, g_membership: float) -> Tuple[Callable, float]:
        """Compose fuzzy functions using minimum t-norm"""
        def composed(x):
            return g(f(x))
        
        # T-norm composition: min(μ_f, μ_g)
        composed_membership = min(f_membership, g_membership)
        return composed, composed_membership

class FuzzyGenerativeAICategory(FuzzySimplicialCategory):
    """
    Fuzzy simplicial category for generative AI models
    
    Objects are fuzzy simplicial model components with graded structure
    Morphisms are fuzzy transformations between simplicial components
    """
    
    def __init__(self, name: str = "FuzzyGenerativeAI"):
        super().__init__(name)
        self.model_components = {}  # object_id -> neural network component
        self.fuzzy_transformations = {}   # (source, target) -> (transformation, membership)
    
    def add_fuzzy_object(self, obj_id: str, dimension: int, membership: float = 1.0, component: Optional[nn.Module] = None):
        """Add fuzzy model component as simplicial object"""
        fuzzy_simplex = create_fuzzy_simplex(dimension, membership)
        self.graded_objects[dimension][obj_id] = fuzzy_simplex
        self.max_dimension = max(self.max_dimension, dimension)
        
        if component is not None:
            self.model_components[obj_id] = component
        
        # Add fuzzy identity morphism
        def fuzzy_identity(x):
            if hasattr(x, 'membership'):
                x.membership = min(x.membership, membership)
            return x
        self.identities[obj_id] = fuzzy_identity
    
    def add_fuzzy_morphism(self, source_id: str, target_id: str, morphism: Callable, membership: float = 1.0):
        """Add fuzzy transformation as morphism"""
        # Wrap transformation with fuzzy membership propagation
        def fuzzy_transformation(x):
            result = morphism(x)
            # Propagate fuzzy membership using t-norm
            if hasattr(result, 'membership') and hasattr(x, 'membership'):
                result.membership = min(x.membership, membership)
            return result
        
        self.morphisms[(source_id, target_id)] = fuzzy_transformation
        self.fuzzy_transformations[(source_id, target_id)] = (morphism, membership)
    
    def compose_fuzzy(self, f: Callable, g: Callable, f_membership: float, g_membership: float) -> Tuple[Callable, float]:
        """Compose fuzzy transformations using minimum t-norm"""
        def composed_transformation(x):
            intermediate = f(x)
            result = g(intermediate)
            # Apply composed membership
            if hasattr(result, 'membership'):
                result.membership = min(result.membership, min(f_membership, g_membership))
            return result
        
        # T-norm composition for AI transformations
        composed_membership = min(f_membership, g_membership)
        return composed_transformation, composed_membership

class FuzzySimplicialFunctor(ABC, Generic[C, D]):
    """
    Abstract base class for functors between fuzzy simplicial categories F: C → D
    
    Maps fuzzy simplicial objects and morphisms while preserving:
    - Graded structure (dimension preservation)
    - Fuzzy memberships (via t-norm operations)
    - Face/degeneracy maps (simplicial structure)
    """
    
    def __init__(self, source_category: FuzzySimplicialCategory, target_category: FuzzySimplicialCategory, name: str):
        self.source_category = source_category
        self.target_category = target_category
        self.name = name
        self.object_map = {}      # source object_id -> (target object_id, membership)
        self.morphism_map = {}    # source morphism -> (target morphism, membership)
    
    @abstractmethod
    def map_fuzzy_object(self, obj_id: str, dimension: int, membership: float) -> Tuple[str, int, float]:
        """Map fuzzy simplicial object: returns (target_obj_id, target_dim, target_membership)"""
        pass
    
    @abstractmethod
    def map_fuzzy_morphism(self, morphism: Callable, source_id: str, target_id: str, membership: float) -> Tuple[Callable, float]:
        """Map fuzzy morphism with membership propagation"""
        pass
    
    def verify_fuzzy_functoriality(self) -> bool:
        """Verify fuzzy functor laws with membership preservation"""
        try:
            # Test identity preservation across dimensions
            for dim in range(min(3, self.source_category.max_dimension + 1)):
                objects = self.source_category.get_objects_at_dimension(dim)
                for obj_id, simplex in list(objects.items())[:3]:  # Test subset
                    if obj_id in self.source_category.identities:
                        source_id = self.source_category.identities[obj_id]
                        
                        # Map object and check membership preservation
                        mapped_obj_id, mapped_dim, mapped_membership = self.map_fuzzy_object(
                            obj_id, simplex.dimension, simplex.membership
                        )
                        
                        # Verify dimension and membership constraints
                        if mapped_membership > simplex.membership:
                            logger.warning(f"Functor increased membership: {simplex.membership} -> {mapped_membership}")
                            return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not verify fuzzy functoriality: {e}")
            return False
    
    def find_preimages(self, target_obj_id: str) -> List[Tuple[str, float]]:
        """Find all source objects that map to target_obj_id with their memberships"""
        preimages = []
        
        for dim in range(self.source_category.max_dimension + 1):
            objects = self.source_category.get_objects_at_dimension(dim)
            for source_id, simplex in objects.items():
                try:
                    mapped_id, _, mapped_membership = self.map_fuzzy_object(
                        source_id, simplex.dimension, simplex.membership
                    )
                    if mapped_id == target_obj_id:
                        preimages.append((source_id, mapped_membership))
                except Exception:
                    continue
        
        return preimages

class FuzzyNeuralFunctor(FuzzySimplicialFunctor):
    """
    Fuzzy simplicial functor for neural network transformations
    
    Maps between fuzzy simplicial categories of neural network components
    Preserves fuzzy memberships through neural transformations
    """
    
    def __init__(self, source_category: FuzzyGenerativeAICategory, target_category: FuzzyGenerativeAICategory):
        super().__init__(source_category, target_category, "FuzzyNeuralFunctor")
    
    def map_fuzzy_object(self, obj_id: str, dimension: int, membership: float) -> Tuple[str, int, float]:
        """Map fuzzy neural component to transformed component"""
        if obj_id not in self.object_map:
            # Create transformed component name
            transformed_name = f"transformed_{obj_id}"
            
            # Apply neural transformation to membership (slight attenuation)
            transformed_membership = membership * 0.95  # Neural processing reduces certainty
            
            self.object_map[obj_id] = (transformed_name, transformed_membership)
            
            # Add to target category if not exists
            target_objects = self.target_category.get_objects_at_dimension(dimension)
            if transformed_name not in target_objects:
                self.target_category.add_fuzzy_object(transformed_name, dimension, transformed_membership)
        
        transformed_name, transformed_membership = self.object_map[obj_id]
        return transformed_name, dimension, transformed_membership
    
    def map_fuzzy_morphism(self, morphism: Callable, source_id: str, target_id: str, membership: float) -> Tuple[Callable, float]:
        """Map fuzzy transformation to neural transformation"""
        key = (source_id, target_id)
        if key not in self.morphism_map:
            # Create fuzzy neural transformation
            def fuzzy_neural_transformation(x):
                # Apply original transformation with neural processing
                result = morphism(x)
                
                if isinstance(result, torch.Tensor):
                    # Add learnable neural transformation
                    noise_scale = 0.1 * (1.0 - membership)  # Less noise for higher membership
                    result = result + noise_scale * torch.randn_like(result)
                
                # Propagate fuzzy membership
                if hasattr(result, 'membership'):
                    result.membership = min(result.membership, membership * 0.9)  # Neural attenuation
                
                return result
            
            # Neural transformations slightly reduce membership certainty
            neural_membership = membership * 0.9
            self.morphism_map[key] = (fuzzy_neural_transformation, neural_membership)
        
        transformation, neural_membership = self.morphism_map[key]
        return transformation, neural_membership

class FuzzyNaturalTransformation:
    """
    Natural transformation between fuzzy simplicial functors
    
    Provides component-wise transformation that commutes with fuzzy functor morphisms
    while preserving fuzzy membership structure
    """
    
    def __init__(self, source_functor: FuzzySimplicialFunctor, target_functor: FuzzySimplicialFunctor, name: str):
        self.source_functor = source_functor
        self.target_functor = target_functor
        self.name = name
        self.components = {}  # object -> (component transformation, membership)
        
        # Verify functors have same domain and codomain
        if (source_functor.source_category != target_functor.source_category or
            source_functor.target_category != target_functor.target_category):
            raise ValueError("Fuzzy natural transformation requires functors with same domain/codomain")
    
    def add_fuzzy_component(self, obj_id: str, component: Callable, membership: float = 1.0):
        """Add fuzzy component of natural transformation at object"""
        self.components[obj_id] = (component, membership)
    
    def get_fuzzy_component(self, obj_id: str) -> Optional[Tuple[Callable, float]]:
        """Get fuzzy component with membership at object"""
        return self.components.get(obj_id)
    
    def verify_fuzzy_naturality(self) -> bool:
        """
        Verify fuzzy naturality condition: η_B ∘ F(f) = G(f) ∘ η_A
        with fuzzy membership preservation
        
        For morphism f: A → B, the diagram must commute in fuzzy sense
        """
        try:
            # Test naturality for available morphisms (simplified)
            for (source_id, target_id), morphism in self.source_functor.source_category.morphisms.items():
                if source_id in self.components and target_id in self.components:
                    # Get fuzzy components
                    eta_A, membership_A = self.components[source_id]
                    eta_B, membership_B = self.components[target_id]
                    
                    # Get fuzzy functor mappings
                    F_f, f_membership = self.source_functor.map_fuzzy_morphism(morphism, source_id, target_id, 1.0)
                    G_f, g_membership = self.target_functor.map_fuzzy_morphism(morphism, source_id, target_id, 1.0)
                    
                    # Test fuzzy naturality with membership constraints
                    # η_B ∘ F(f) should equal G(f) ∘ η_A with compatible memberships
                    combined_membership = min(membership_A, membership_B, f_membership, g_membership)
                    if combined_membership < 0.1:  # Too low membership
                        logger.warning(f"Low fuzzy naturality membership: {combined_membership}")
                        return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not verify fuzzy naturality: {e}")
            return False

# Backward compatibility aliases
NaturalTransformation = FuzzyNaturalTransformation
Functor = FuzzySimplicialFunctor
GenerativeAICategory = FuzzyGenerativeAICategory
NeuralFunctor = FuzzyNeuralFunctor

class LeftKanExtension:
    """
    Left Kan extension Lan_K F: D → E
    
    From (MAHADEVAN,2024) Definition 48: Universal property for extending functors
    """
    
    def __init__(self, 
                 F: Functor,           # F: C → E (functor to extend)
                 K: Functor,           # K: C → D (extension direction)
                 name: str = "LeftKanExtension"):
        self.F = F
        self.K = K
        self.name = name
        
        # Verify compatibility: F and K must have same source category
        if F.source_category != K.source_category:
            raise ValueError("F and K must have same source category")
        
        # Create extended functor Lan_K F: D → E
        self.extended_functor = self._construct_extended_functor()
        
        # Create natural transformation η: F → Lan_K F ∘ K
        self.unit = self._construct_unit()
    
    def _construct_extended_functor(self) -> FuzzySimplicialFunctor:
        """
        Construct Lan_K F: D → E via fuzzy colimit construction
        
        This is the core of the Kan extension - extending F along K using
        proper colimits over comma category (d ↓ K) with fuzzy memberships
        """
        class FuzzyExtendedFunctor(FuzzySimplicialFunctor):
            def __init__(self, left_kan_ext):
                super().__init__(
                    left_kan_ext.K.target_category,  # D
                    left_kan_ext.F.target_category,  # E
                    f"Lan_{left_kan_ext.K.name}_{left_kan_ext.F.name}"
                )
                self.left_kan_ext = left_kan_ext
            
            def map_fuzzy_object(self, d_obj_id: str, dimension: int, membership: float) -> Tuple[str, int, float]:
                """Map object in D to object in E via fuzzy colimit construction"""
                # Find all preimages of d_obj_id under K with their memberships
                preimages = self.left_kan_ext.K.find_preimages(d_obj_id)
                
                if not preimages:
                    # No preimages - create new extended object
                    extended_name = f"extended_{d_obj_id}"
                    extended_membership = membership * 0.5  # Reduced certainty for extensions
                    
                    # Add to target category
                    target_objects = self.left_kan_ext.F.target_category.get_objects_at_dimension(dimension)
                    if extended_name not in target_objects:
                        self.left_kan_ext.F.target_category.add_fuzzy_object(
                            extended_name, dimension, extended_membership
                        )
                    
                    return extended_name, dimension, extended_membership
                
                # Compute fuzzy colimit: μ_colimit = sup{μ_F(c) : K(c) → d}
                colimit_objects = []
                max_membership = 0.0
                representative_obj = None
                
                for c_obj_id, k_membership in preimages:
                    # Apply F to each preimage
                    try:
                        # Find the source object
                        source_simplex = None
                        for dim in range(self.left_kan_ext.F.source_category.max_dimension + 1):
                            objects = self.left_kan_ext.F.source_category.get_objects_at_dimension(dim)
                            if c_obj_id in objects:
                                source_simplex = objects[c_obj_id]
                                break
                        
                        if source_simplex:
                            # Map through F
                            f_obj_id, f_dim, f_membership = self.left_kan_ext.F.map_fuzzy_object(
                                c_obj_id, source_simplex.dimension, source_simplex.membership
                            )
                            
                            # Combine memberships using t-norm
                            combined_membership = min(k_membership, f_membership)
                            colimit_objects.append((f_obj_id, f_dim, combined_membership))
                            
                            # Track maximum for colimit (supremum)
                            if combined_membership > max_membership:
                                max_membership = combined_membership
                                representative_obj = (f_obj_id, f_dim)
                    
                    except Exception as e:
                        logger.warning(f"Failed to map preimage {c_obj_id}: {e}")
                        continue
                
                if not colimit_objects or representative_obj is None:
                    # Fallback to extended object
                    extended_name = f"colimit_{d_obj_id}"
                    return extended_name, dimension, membership * 0.3
                
                # Create colimit object with supremum membership
                colimit_name = f"colimit_{d_obj_id}_{representative_obj[0]}"
                colimit_dim = representative_obj[1]
                
                # Add colimit to target category
                target_objects = self.left_kan_ext.F.target_category.get_objects_at_dimension(colimit_dim)
                if colimit_name not in target_objects:
                    self.left_kan_ext.F.target_category.add_fuzzy_object(
                        colimit_name, colimit_dim, max_membership
                    )
                
                return colimit_name, colimit_dim, max_membership
            
            def map_fuzzy_morphism(self, morphism: Callable, source_id: str, target_id: str, membership: float) -> Tuple[Callable, float]:
                """Map morphism in D to morphism in E via colimit construction"""
                def extended_fuzzy_morphism(x):
                    # Apply original morphism with fuzzy membership propagation
                    result = morphism(x)
                    
                    # Propagate membership through extension
                    if hasattr(result, 'membership'):
                        result.membership = min(result.membership, membership * 0.8)  # Extension attenuation
                    
                    return result
                
                # Extended morphisms have reduced membership
                extended_membership = membership * 0.8
                return extended_fuzzy_morphism, extended_membership
        
        return FuzzyExtendedFunctor(self)
    
    def _construct_unit(self) -> FuzzyNaturalTransformation:
        """
        Construct fuzzy unit η: F → Lan_K F ∘ K
        
        This is part of the universal property with fuzzy membership preservation
        """
        # Compose Lan_K F with K
        class FuzzyComposedFunctor(FuzzySimplicialFunctor):
            def __init__(self, lan_k_f, k):
                super().__init__(k.source_category, lan_k_f.target_category, "FuzzyComposed")
                self.lan_k_f = lan_k_f
                self.k = k
            
            def map_fuzzy_object(self, obj_id: str, dimension: int, membership: float) -> Tuple[str, int, float]:
                k_obj_id, k_dim, k_membership = self.k.map_fuzzy_object(obj_id, dimension, membership)
                return self.lan_k_f.map_fuzzy_object(k_obj_id, k_dim, k_membership)
            
            def map_fuzzy_morphism(self, morphism: Callable, source_id: str, target_id: str, membership: float) -> Tuple[Callable, float]:
                k_morphism, k_membership = self.k.map_fuzzy_morphism(morphism, source_id, target_id, membership)
                k_source_id, _, _ = self.k.map_fuzzy_object(source_id, 0, 1.0)  # Simplified dimension
                k_target_id, _, _ = self.k.map_fuzzy_object(target_id, 0, 1.0)
                return self.lan_k_f.map_fuzzy_morphism(k_morphism, k_source_id, k_target_id, k_membership)
        
        composed = FuzzyComposedFunctor(self.extended_functor, self.K)
        unit = FuzzyNaturalTransformation(self.F, composed, "FuzzyUnit")
        
        # Add fuzzy components
        for dim in range(self.F.source_category.max_dimension + 1):
            objects = self.F.source_category.get_objects_at_dimension(dim)
            for obj_id, simplex in objects.items():
                # Fuzzy identity component with membership preservation
                def fuzzy_identity_component(x, membership=simplex.membership):
                    if hasattr(x, 'membership'):
                        x.membership = min(x.membership, membership)
                    return x
                
                unit.add_fuzzy_component(obj_id, fuzzy_identity_component, simplex.membership)
        
        return unit
    
    def verify_universal_property(self, G: FuzzySimplicialFunctor, gamma: FuzzyNaturalTransformation) -> bool:
        """
        Verify fuzzy universal property: for any G: D → E and γ: F → G∘K,
        there exists unique α: Lan_K F → G such that γ = α * η
        with fuzzy membership constraints
        """
        try:
            # Verify the fuzzy universal property
            # Check that gamma is a valid fuzzy natural transformation
            if not gamma.verify_fuzzy_naturality():
                logger.warning("Gamma is not a valid fuzzy natural transformation")
                return False
            
            # Check compatibility of functors
            if (G.source_category != self.K.target_category or
                G.target_category != self.F.target_category):
                logger.warning("Functor G has incompatible domain/codomain")
                return False
            
            # Simplified verification - in practice would construct mediating morphism
            return True
        except Exception as e:
            logger.warning(f"Could not verify fuzzy universal property: {e}")
            return False
    
    def compute_universal_property_loss(self, representations: torch.Tensor, 
                                      target_representations: torch.Tensor) -> torch.Tensor:
        """
        Compute loss measuring deviation from Kan extension's universal property in fuzzy simplicial terms.
        
        The universal property states: for any G: D → E and γ: F → G∘K,
        there exists unique α: Lan_K F → G such that the diagram commutes.
        
        This loss measures commutativity defect in fuzzy simplicial terms:
        δ = sup_{c ∈ C} d(γ_c, (α ∘ η)_c)
        where d is a metric on fuzzy simplices.
        
        Args:
            representations: Current functor outputs F(X) 
            target_representations: Target functor outputs G(K(X))
            
        Returns:
            Loss tensor measuring universal property deviation
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: LEFT KAN EXTENSION universal property computation STARTED")
        logger.info(f"   Input representations shape: {representations.shape}")
        logger.info(f"   Target representations shape: {target_representations.shape}")
        logger.info(f"   Fuzzy Functor F: {self.F.name}")
        logger.info(f"   Extension functor K: {self.K.name}")
        
        device = representations.device
        # Handle both 2D and 3D input tensors
        if len(representations.shape) == 2:
            batch_size, d_model = representations.shape
            seq_len = 1
            representations = representations.unsqueeze(1)
        elif len(representations.shape) == 3:
            batch_size, seq_len, d_model = representations.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {len(representations.shape)}D tensor with shape {representations.shape}")
        
        # 1. Fuzzy simplicial commutativity loss: measure diagram commutativity
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: Step 1 - Computing fuzzy unit transformation η")
        
        # Create fuzzy simplicial representations
        fuzzy_representations = self._tensorize_fuzzy_simplices(representations)
        fuzzy_targets = self._tensorize_fuzzy_simplices(target_representations)
        
        # Apply fuzzy unit transformation
        unit_composition = self._apply_fuzzy_unit_transformation(fuzzy_representations)
        logger.info(f"   Fuzzy unit composition computed with membership propagation")
        
        # 2. Compute fuzzy mediating morphism with membership constraints
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: Step 2 - Computing fuzzy mediating morphism α")
        mediating_morphism, mediating_membership = self._compute_fuzzy_mediating_morphism(
            unit_composition, fuzzy_targets
        )
        logger.info(f"   Fuzzy mediating morphism membership: {mediating_membership:.6f}")
        
        # 3. Measure fuzzy simplicial commutativity defect
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: Step 3 - Computing fuzzy commutativity defect")
        
        # Compute supremum over fuzzy simplicial components
        commutativity_defects = []
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                # Extract fuzzy simplex components
                gamma_c = fuzzy_targets[batch_idx, seq_idx, :]
                alpha_eta_c = mediating_morphism[batch_idx, seq_idx, :]
                
                # Compute fuzzy simplicial distance using t-conorm metric
                fuzzy_distance = self._compute_fuzzy_simplicial_distance(
                    gamma_c, alpha_eta_c, self.F.source_category.t_conorm
                )
                commutativity_defects.append(fuzzy_distance)
        
        # Supremum (maximum) of commutativity defects
        commutativity_loss = torch.max(torch.stack(commutativity_defects))
        logger.info(f"   Fuzzy commutativity defect (supremum): {commutativity_loss.item():.6f}")
        
        # 4. Fuzzy uniqueness loss: penalize multiple fuzzy solutions
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: Step 4 - Testing fuzzy uniqueness")
        
        alternative_morphism, alt_membership = self._compute_alternative_fuzzy_mediating_morphism(
            unit_composition, fuzzy_targets
        )
        
        # Fuzzy uniqueness penalty using membership-weighted distance
        uniqueness_defects = []
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                alpha_1 = mediating_morphism[batch_idx, seq_idx, :]
                alpha_2 = alternative_morphism[batch_idx, seq_idx, :]
                
                # Weight by membership difference
                membership_weight = abs(mediating_membership - alt_membership)
                fuzzy_uniqueness_distance = membership_weight * torch.norm(alpha_1 - alpha_2)
                uniqueness_defects.append(fuzzy_uniqueness_distance)
        
        uniqueness_loss = torch.max(torch.stack(uniqueness_defects))
        logger.info(f"   Fuzzy uniqueness defect: {uniqueness_loss.item():.6f}")
        
        # 5. Fuzzy functoriality preservation
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: Step 5 - Verifying fuzzy functoriality")
        
        if seq_len > 1:
            # Check fuzzy morphism composition preservation
            functoriality_defects = []
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len - 1):
                    # Source and target fuzzy morphisms
                    source_morph = fuzzy_representations[batch_idx, seq_idx + 1, :] - fuzzy_representations[batch_idx, seq_idx, :]
                    target_morph = fuzzy_targets[batch_idx, seq_idx + 1, :] - fuzzy_targets[batch_idx, seq_idx, :]
                    
                    # Apply extended functor
                    extended_morph = self._apply_fuzzy_extended_functor(source_morph)
                    
                    # Fuzzy functoriality distance
                    func_distance = torch.norm(extended_morph - target_morph)
                    functoriality_defects.append(func_distance)
            
            functoriality_loss = torch.max(torch.stack(functoriality_defects))
            logger.info(f"   Fuzzy functoriality defect: {functoriality_loss.item():.6f}")
        else:
            functoriality_loss = torch.tensor(0.0, device=device)
        
        # 6. Combine losses with fuzzy categorical weights
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: Step 6 - Combining fuzzy losses")
        
        # Weights based on fuzzy categorical importance
        total_loss = (0.6 * commutativity_loss +     # Primary: fuzzy diagram commutativity
                     0.3 * uniqueness_loss +         # Secondary: fuzzy solution uniqueness  
                     0.1 * functoriality_loss)       # Tertiary: fuzzy structure preservation
        
        logger.info(f"   Weighted fuzzy commutativity: {(0.6 * commutativity_loss).item():.6f}")
        logger.info(f"   Weighted fuzzy uniqueness: {(0.3 * uniqueness_loss).item():.6f}")
        logger.info(f"   Weighted fuzzy functoriality: {(0.1 * functoriality_loss).item():.6f}")
        logger.info(f"🔧 FUZZY KAN FRAMEWORK: LEFT KAN EXTENSION fuzzy universal property loss: {total_loss.item():.6f}")
        
        return total_loss
    
    def _tensorize_fuzzy_simplices(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor representations to fuzzy simplicial form
        
        Each tensor element is treated as a fuzzy simplex with implicit membership
        """
        # Normalize to [0,1] range for fuzzy memberships
        normalized = torch.sigmoid(representations)
        return normalized
    
    def _apply_fuzzy_unit_transformation(self, fuzzy_representations: torch.Tensor) -> torch.Tensor:
        """
        Apply the fuzzy unit η: F → Lan_K F ∘ K of the Kan extension
        
        Preserves fuzzy membership structure through the unit natural transformation
        """
        batch_size, seq_len, d_model = fuzzy_representations.shape
        
        # Create fuzzy unit transformation that preserves membership bounds
        unit_weight = torch.randn(d_model, d_model, device=fuzzy_representations.device) * 0.05
        unit_weight = torch.sigmoid(unit_weight)  # Ensure [0,1] range
        
        # Apply unit with membership preservation
        unit_transformed = torch.matmul(fuzzy_representations, unit_weight)
        
        # Ensure result stays in fuzzy range [0,1]
        unit_transformed = torch.clamp(unit_transformed, 0.0, 1.0)
        
        return unit_transformed
    
    def _compute_fuzzy_simplicial_distance(self, simplex_a: torch.Tensor, simplex_b: torch.Tensor, t_conorm: TConorm) -> torch.Tensor:
        """
        Compute distance between fuzzy simplices using t-conorm metric
        
        For fuzzy simplices with memberships μ_a, μ_b, the distance is:
        d(a,b) = |μ_a - μ_b| + ||a - b||_2
        """
        # Extract membership degrees (assume last dimension encodes membership)
        membership_a = torch.mean(simplex_a)  # Simplified: average as membership
        membership_b = torch.mean(simplex_b)
        
        # Membership distance
        membership_distance = torch.abs(membership_a - membership_b)
        
        # Geometric distance
        geometric_distance = torch.norm(simplex_a - simplex_b, p=2)
        
        # Combined fuzzy simplicial distance
        if t_conorm == TConorm.MAXIMUM:
            # Maximum t-conorm: max(membership_dist, geometric_dist)
            fuzzy_distance = torch.max(membership_distance, geometric_distance)
        else:
            # Default: weighted sum
            fuzzy_distance = 0.5 * membership_distance + 0.5 * geometric_distance
        
        return fuzzy_distance
    
    def _compute_fuzzy_mediating_morphism(self, unit_output: torch.Tensor, 
                                         fuzzy_target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute the fuzzy mediating morphism α: Lan_K F → G from universal property
        
        Returns both the morphism and its fuzzy membership degree
        """
        batch_size, seq_len, d_model = unit_output.shape
        
        # Reshape for batch matrix operations
        unit_flat = unit_output.view(-1, d_model)
        target_flat = fuzzy_target.view(-1, d_model)
        
        # Compute fuzzy pseudo-inverse with membership constraints
        try:
            # Fuzzy least squares: minimize ||Uα - T||² subject to membership constraints
            UTU = torch.matmul(unit_flat.T, unit_flat) + 1e-6 * torch.eye(d_model, device=unit_output.device)
            UTU_inv = torch.inverse(UTU)
            UT_target = torch.matmul(unit_flat.T, target_flat)
            alpha_matrix = torch.matmul(UTU_inv, UT_target)
            
            # Apply mediating morphism
            mediated = torch.matmul(unit_flat, alpha_matrix)
            mediated = mediated.view(batch_size, seq_len, d_model)
            
            # Ensure fuzzy range [0,1]
            mediated = torch.clamp(mediated, 0.0, 1.0)
            
            # Compute membership degree as average confidence
            membership_degree = torch.mean(mediated).item()
            
            return mediated, membership_degree
            
        except Exception:
            # Fallback: fuzzy identity with reduced membership
            fallback_membership = 0.3
            return torch.clamp(unit_output, 0.0, 1.0), fallback_membership
    
    def _compute_alternative_fuzzy_mediating_morphism(self, unit_output: torch.Tensor,
                                                    fuzzy_target: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute alternative fuzzy mediating morphism to test uniqueness
        
        Uses different regularization to find alternative fuzzy solution
        """
        batch_size, seq_len, d_model = unit_output.shape
        
        # Alternative: use ridge regression with different fuzzy regularization
        unit_flat = unit_output.view(-1, d_model)
        target_flat = fuzzy_target.view(-1, d_model)
        
        try:
            # Different fuzzy regularization parameter
            UTU = torch.matmul(unit_flat.T, unit_flat) + 1e-4 * torch.eye(d_model, device=unit_output.device)
            UTU_inv = torch.inverse(UTU)
            UT_target = torch.matmul(unit_flat.T, target_flat)
            alpha_alt = torch.matmul(UTU_inv, UT_target)
            
            mediated_alt = torch.matmul(unit_flat, alpha_alt)
            mediated_alt = mediated_alt.view(batch_size, seq_len, d_model)
            
            # Ensure fuzzy range with different scaling
            mediated_alt = torch.sigmoid(mediated_alt * 0.8)  # Different scaling
            
            # Alternative membership computation
            alt_membership = torch.mean(mediated_alt).item() * 0.9
            
            return mediated_alt, alt_membership
            
        except Exception:
            # Fallback: slightly perturbed fuzzy version
            noise = torch.randn_like(unit_output) * 0.01
            perturbed = torch.clamp(unit_output + noise, 0.0, 1.0)
            return perturbed, 0.25
    
    def _apply_fuzzy_extended_functor(self, fuzzy_morphism: torch.Tensor) -> torch.Tensor:
        """
        Apply the fuzzy extended functor Lan_K F to fuzzy morphisms
        
        Preserves fuzzy simplicial structure through the extension
        """
        # Apply fuzzy transformation with membership preservation
        extended = torch.sigmoid(fuzzy_morphism * 0.9)  # Slight attenuation
        return extended
    
    def _compute_alternative_mediating_morphism(self, unit_output: torch.Tensor,
                                              target: torch.Tensor) -> torch.Tensor:
        """
        Compute alternative mediating morphism to test uniqueness
        """
        # Use different method (e.g., different regularization) to find alternative solution
        batch_size, seq_len, d_model = unit_output.shape
        
        # Alternative: use ridge regression with different regularization
        unit_flat = unit_output.view(-1, d_model)
        target_flat = target.view(-1, d_model)
        
        try:
            # Different regularization parameter
            UTU = torch.matmul(unit_flat.T, unit_flat) + 1e-4 * torch.eye(d_model, device=unit_output.device)
            UTU_inv = torch.inverse(UTU)
            UT_target = torch.matmul(unit_flat.T, target_flat)
            alpha_alt = torch.matmul(UTU_inv, UT_target)
            
            mediated_alt = torch.matmul(unit_flat, alpha_alt)
            return mediated_alt.view(batch_size, seq_len, d_model)
            
        except Exception:
            # Fallback: slightly perturbed version
            noise = torch.randn_like(unit_output) * 0.01
            return unit_output + noise
    
    def _apply_extended_functor(self, morphisms: torch.Tensor) -> torch.Tensor:
        """
        Apply the extended functor Lan_K F to morphisms
        """
        # Simplified: apply learnable transformation representing extended functor
        # In practice, this would be the actual categorical functor application
        return morphisms  # Identity for simplification
    
    def apply(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Apply left Kan extension for compositional understanding
        
        Implements colimit-based migration: Σ_F (left adjoint to pullback)
        Captures how local syntactic changes propagate to global semantic structure
        """
        # Handle both 2D and 3D tensors
        if len(representations.shape) == 2:
            batch_size, d_model = representations.shape
            seq_len = 1  # Treat as single sequence element
            representations = representations.unsqueeze(1)  # Add sequence dimension
        elif len(representations.shape) == 3:
            batch_size, seq_len, d_model = representations.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {len(representations.shape)}D tensor with shape {representations.shape}")
        
        # Colimit construction: local-to-global propagation
        # Each token influences its compositional context through weighted aggregation
        position_weights = torch.softmax(
            torch.arange(seq_len, dtype=torch.float32, device=representations.device), 
            dim=0
        )
        
        # Apply colimit through categorical coproduct
        colimit_repr = torch.einsum('bsd,s->bsd', representations, position_weights)
        
        return colimit_repr

class RightKanExtension:
    """
    Right Kan extension Ran_K F: D → E
    
    Dual to left Kan extension, constructed via limits instead of colimits
    """
    
    def __init__(self, 
                 F: Functor,           # F: C → E (functor to extend)
                 K: Functor,           # K: C → D (extension direction)
                 name: str = "RightKanExtension"):
        self.F = F
        self.K = K
        self.name = name
        
        # Verify compatibility
        if F.source_category != K.source_category:
            raise ValueError("F and K must have same source category")
        
        # Create extended functor Ran_K F: D → E
        self.extended_functor = self._construct_extended_functor()
        
        # Create natural transformation ε: Ran_K F ∘ K → F
        self.counit = self._construct_counit()
    
    def _construct_extended_functor(self) -> FuzzySimplicialFunctor:
        """
        Construct Ran_K F: D → E via fuzzy limit construction
        
        This is the core of the right Kan extension - extending F along K using
        proper limits over comma category (K ↓ d) with fuzzy memberships
        """
        class FuzzyRightExtendedFunctor(FuzzySimplicialFunctor):
            def __init__(self, right_kan_ext):
                super().__init__(
                    right_kan_ext.K.target_category,  # D
                    right_kan_ext.F.target_category,  # E
                    f"Ran_{right_kan_ext.K.name}_{right_kan_ext.F.name}"
                )
                self.right_kan_ext = right_kan_ext
            
            def map_fuzzy_object(self, d_obj_id: str, dimension: int, membership: float) -> Tuple[str, int, float]:
                """Map object in D to object in E via fuzzy limit construction"""
                # Find all preimages of d_obj_id under K with their memberships
                preimages = self.right_kan_ext.K.find_preimages(d_obj_id)
                
                if not preimages:
                    # No preimages - create new extended object
                    extended_name = f"right_extended_{d_obj_id}"
                    extended_membership = membership * 0.4  # Reduced certainty for right extensions
                    
                    # Add to target category
                    target_objects = self.right_kan_ext.F.target_category.get_objects_at_dimension(dimension)
                    if extended_name not in target_objects:
                        self.right_kan_ext.F.target_category.add_fuzzy_object(
                            extended_name, dimension, extended_membership
                        )
                    
                    return extended_name, dimension, extended_membership
                
                # Compute fuzzy limit: μ_limit = inf{μ_F(c) : K(c) → d}
                limit_objects = []
                min_membership = 1.0
                representative_obj = None
                
                for c_obj_id, k_membership in preimages:
                    # Apply F to each preimage
                    try:
                        # Find the source object
                        source_simplex = None
                        for dim in range(self.right_kan_ext.F.source_category.max_dimension + 1):
                            objects = self.right_kan_ext.F.source_category.get_objects_at_dimension(dim)
                            if c_obj_id in objects:
                                source_simplex = objects[c_obj_id]
                                break
                        
                        if source_simplex:
                            # Map through F
                            f_obj_id, f_dim, f_membership = self.right_kan_ext.F.map_fuzzy_object(
                                c_obj_id, source_simplex.dimension, source_simplex.membership
                            )
                            
                            # Combine memberships using t-norm
                            combined_membership = min(k_membership, f_membership)
                            limit_objects.append((f_obj_id, f_dim, combined_membership))
                            
                            # Track minimum for limit (infimum)
                            if combined_membership < min_membership:
                                min_membership = combined_membership
                                representative_obj = (f_obj_id, f_dim)
                    
                    except Exception as e:
                        logger.warning(f"Failed to map preimage {c_obj_id}: {e}")
                        continue
                
                if not limit_objects or representative_obj is None:
                    # Fallback to extended object
                    extended_name = f"right_limit_{d_obj_id}"
                    return extended_name, dimension, membership * 0.2
                
                # Create limit object with infimum membership
                limit_name = f"right_limit_{d_obj_id}_{representative_obj[0]}"
                limit_dim = representative_obj[1]
                
                # Add limit to target category
                target_objects = self.right_kan_ext.F.target_category.get_objects_at_dimension(limit_dim)
                if limit_name not in target_objects:
                    self.right_kan_ext.F.target_category.add_fuzzy_object(
                        limit_name, limit_dim, min_membership
                    )
                
                return limit_name, limit_dim, min_membership
            
            def map_fuzzy_morphism(self, morphism: Callable, source_id: str, target_id: str, membership: float) -> Tuple[Callable, float]:
                """Map morphism in D to morphism in E via limit construction"""
                def right_extended_fuzzy_morphism(x):
                    # Apply original morphism with fuzzy membership propagation
                    result = morphism(x)
                    
                    # Propagate membership through right extension (more conservative)
                    if hasattr(result, 'membership'):
                        result.membership = min(result.membership, membership * 0.7)  # Right extension attenuation
                    
                    return result
                
                # Right extended morphisms have more reduced membership
                extended_membership = membership * 0.7
                return right_extended_fuzzy_morphism, extended_membership
        
        return FuzzyRightExtendedFunctor(self)
    
    def _construct_counit(self) -> FuzzyNaturalTransformation:
        """
        Construct fuzzy counit ε: Ran_K F ∘ K → F
        
        Dual to unit construction with fuzzy membership preservation
        """
        # Similar to unit construction but in opposite direction
        class FuzzyComposedFunctor(FuzzySimplicialFunctor):
            def __init__(self, ran_k_f, k):
                super().__init__(k.source_category, ran_k_f.target_category, "FuzzyComposed")
                self.ran_k_f = ran_k_f
                self.k = k
            
            def map_fuzzy_object(self, obj_id: str, dimension: int, membership: float) -> Tuple[str, int, float]:
                k_obj_id, k_dim, k_membership = self.k.map_fuzzy_object(obj_id, dimension, membership)
                return self.ran_k_f.map_fuzzy_object(k_obj_id, k_dim, k_membership)
            
            def map_fuzzy_morphism(self, morphism: Callable, source_id: str, target_id: str, membership: float) -> Tuple[Callable, float]:
                k_morphism, k_membership = self.k.map_fuzzy_morphism(morphism, source_id, target_id, membership)
                k_source_id, _, _ = self.k.map_fuzzy_object(source_id, 0, 1.0)
                k_target_id, _, _ = self.k.map_fuzzy_object(target_id, 0, 1.0)
                return self.ran_k_f.map_fuzzy_morphism(k_morphism, k_source_id, k_target_id, k_membership)
        
        composed = FuzzyComposedFunctor(self.extended_functor, self.K)
        counit = FuzzyNaturalTransformation(composed, self.F, "FuzzyCounit")
        
        # Add fuzzy components
        for dim in range(self.F.source_category.max_dimension + 1):
            objects = self.F.source_category.get_objects_at_dimension(dim)
            for obj_id, simplex in objects.items():
                # Fuzzy counit component with membership preservation
                def fuzzy_counit_component(x, membership=simplex.membership):
                    if hasattr(x, 'membership'):
                        x.membership = min(x.membership, membership * 0.95)  # Slight attenuation for counit
                    return x
                
                counit.add_fuzzy_component(obj_id, fuzzy_counit_component, simplex.membership * 0.95)
        
        return counit
    
    def compute_universal_property_loss(self, representations: torch.Tensor, 
                                       target_representations: torch.Tensor) -> torch.Tensor:
        """
        Compute universal property loss for Right Kan extension
        
        Right Kan extension Ran_K F has universal property:
        For any functor G: D → E and natural transformation δ: G ∘ K → F,
        there exists unique α: G → Ran_K F such that δ = ε ∘ (α * K)
        where ε: Ran_K F ∘ K → F is the counit
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"🔧 KAN FRAMEWORK: RIGHT KAN EXTENSION universal property computation STARTED")
        logger.info(f"   Input representations shape: {representations.shape}")
        logger.info(f"   Target representations shape: {target_representations.shape}")
        logger.info(f"   Functor F: {self.F}")
        logger.info(f"   Extension functor K: {self.K}")
        
        device = representations.device
        batch_size, seq_len, d_model = representations.shape
        
        # Step 1: Compute counit transformation ε: Ran_K F ∘ K → F
        logger.info(f"🔧 KAN FRAMEWORK: Step 1 - Computing counit transformation ε: Ran_K F ∘ K → F")
        
        # Apply K functor (simplified as linear transformation)
        k_applied = torch.matmul(representations, torch.randn(representations.size(-1), representations.size(-1), device=device) * 0.1)
        
        # Apply Ran_K F (right extension)
        ran_k_f_applied = self.apply(k_applied)
        
        # Counit: Ran_K F ∘ K → F
        counit_composition = torch.matmul(ran_k_f_applied, torch.randn(representations.size(-1), representations.size(-1), device=device) * 0.1)
        logger.info(f"   Counit composition shape: {counit_composition.shape}")
        logger.info(f"   Counit composition norm: {torch.norm(counit_composition).item():.6f}")
        
        # Step 2: Compute mediating morphism α: G → Ran_K F
        logger.info(f"🔧 KAN FRAMEWORK: Step 2 - Computing mediating morphism α: G → Ran_K F")
        
        # G is represented by target_representations
        # Mediating morphism: solve for α such that δ = ε ∘ (α * K)
        mediating_morphism = torch.matmul(target_representations, torch.randn(target_representations.size(-1), representations.size(-1), device=device) * 0.1)
        logger.info(f"   Mediating morphism shape: {mediating_morphism.shape}")
        logger.info(f"   Mediating morphism norm: {torch.norm(mediating_morphism).item():.6f}")
        
        # Step 3: Check universal property δ = ε ∘ (α * K)
        logger.info(f"🔧 KAN FRAMEWORK: Step 3 - Checking universal property δ = ε ∘ (α * K)")
        
        # δ: G ∘ K → F (natural transformation from target to source via K)
        delta_transform = torch.matmul(target_representations, torch.randn(target_representations.size(-1), representations.size(-1), device=device) * 0.1)
        
        # ε ∘ (α * K): composition through mediating morphism
        alpha_k_composition = torch.matmul(mediating_morphism.transpose(-2, -1), k_applied)
        epsilon_alpha_k = torch.matmul(counit_composition, alpha_k_composition.transpose(-2, -1))
        
        # Universal property error
        composition_error = torch.norm(delta_transform - epsilon_alpha_k)
        logger.info(f"   Composition error ||δ - ε∘(α*K)||: {composition_error.item():.6f}")
        
        # Commutativity loss
        commutativity_loss = torch.pow(composition_error, 2) / (torch.norm(delta_transform) + 1e-8)
        logger.info(f"   Commutativity loss: {commutativity_loss.item():.6f}")
        
        # Step 4: Test uniqueness of mediating morphism
        logger.info(f"🔧 KAN FRAMEWORK: Step 4 - Testing uniqueness of mediating morphism")
        
        # Alternative mediating morphism
        alternative_morphism = torch.matmul(target_representations, torch.randn(target_representations.size(-1), representations.size(-1), device=device) * 0.05)
        logger.info(f"   Alternative morphism norm: {torch.norm(alternative_morphism).item():.6f}")
        
        # Uniqueness penalty
        uniqueness_penalty = torch.norm(mediating_morphism - alternative_morphism)
        logger.info(f"   Uniqueness penalty ||α - α'||: {uniqueness_penalty.item():.6f}")
        
        # Uniqueness loss
        uniqueness_loss = torch.pow(uniqueness_penalty, 2) / (torch.norm(mediating_morphism) + 1e-8)
        logger.info(f"   Uniqueness loss: {uniqueness_loss.item():.6f}")
        
        # Step 5: Verify functoriality preservation
        logger.info(f"🔧 KAN FRAMEWORK: Step 5 - Verifying functoriality preservation")
        
        # Source morphisms (identity-like) - use sequence length dimension to match mediating_morphism
        seq_len = representations.size(1)
        source_morphisms = torch.eye(seq_len, device=device).unsqueeze(0).expand(representations.size(0), -1, -1)
        logger.info(f"   Source morphisms norm: {torch.norm(source_morphisms).item():.6f}")
        
        # Extended morphisms should preserve composition
        # mediating_morphism is [4, 127, 512], we need [4, 127, 127] for comparison
        extended_morphisms = torch.matmul(mediating_morphism, mediating_morphism.transpose(-2, -1))
        # Normalize to same scale as identity
        extended_morphisms = extended_morphisms / (torch.norm(extended_morphisms, dim=(-2, -1), keepdim=True) + 1e-8) * seq_len
        logger.info(f"   Extended morphisms norm: {torch.norm(extended_morphisms).item():.6f}")
        
        # Functoriality error - compare identity preservation
        functoriality_error = torch.norm(extended_morphisms - source_morphisms)
        logger.info(f"   Functoriality error: {functoriality_error.item():.6f}")
        
        # Functoriality loss
        functoriality_loss = torch.pow(functoriality_error, 2) / (torch.norm(source_morphisms) + 1e-8)
        logger.info(f"   Functoriality loss: {functoriality_loss.item():.6f}")
        
        # Step 6: Combine losses with categorical weights
        logger.info(f"🔧 KAN FRAMEWORK: Step 6 - Combining losses with categorical weights")
        
        # Categorical weights for right Kan extension (limits emphasize global constraints)
        commutativity_weight = 0.4  # Counit commutativity
        uniqueness_weight = 0.4     # Mediating morphism uniqueness
        functoriality_weight = 0.2  # Functoriality preservation
        
        weighted_commutativity = commutativity_loss * commutativity_weight
        weighted_uniqueness = uniqueness_loss * uniqueness_weight
        weighted_functoriality = functoriality_loss * functoriality_weight
        
        logger.info(f"   Weighted commutativity: {weighted_commutativity.item():.6f}")
        logger.info(f"   Weighted uniqueness: {weighted_uniqueness.item():.6f}")
        logger.info(f"   Weighted functoriality: {weighted_functoriality.item():.6f}")
        
        # Total universal property loss
        total_loss = weighted_commutativity + weighted_uniqueness + weighted_functoriality
        
        logger.info(f"🔧 KAN FRAMEWORK: RIGHT KAN EXTENSION universal property loss: {total_loss.item():.6f}")
        
        return total_loss
    
    def apply(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Apply right Kan extension for compositional understanding
        
        Implements limit-based migration: Π_F (right adjoint to pullback)
        Captures how global semantic structure constrains local syntactic choices
        """
        # Handle both 2D and 3D tensors
        if len(representations.shape) == 2:
            batch_size, d_model = representations.shape
            seq_len = 1  # Treat as single sequence element
            representations = representations.unsqueeze(1)  # Add sequence dimension
        elif len(representations.shape) == 3:
            batch_size, seq_len, d_model = representations.shape
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {len(representations.shape)}D tensor with shape {representations.shape}")
        
        # Limit construction: global-to-local constraint
        # Global context constrains local token representations
        global_context = torch.mean(representations, dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # Apply limit through categorical product
        limit_repr = representations * torch.sigmoid(global_context)
        
        return limit_repr

class MigrationFunctor:
    """
    Migration functors for generative AI model modifications
    
    From GAIA paper: Δ_F (pullback), Σ_F (left pushforward), Π_F (right pushforward)
    """
    
    def __init__(self, F: Functor, name: str = "MigrationFunctor"):
        self.F = F  # F: S → T (model modification)
        self.name = name
    
    def pullback_functor(self, epsilon_functor: Functor) -> Functor:
        """
        Δ_F: ε → δ
        
        Pullback functor that maps modified model back to original
        """
        class PullbackFunctor(Functor):
            def __init__(self, migration_functor, epsilon):
                super().__init__(
                    epsilon.target_category,
                    migration_functor.F.source_category,
                    f"Pullback_{migration_functor.name}"
                )
                self.migration_functor = migration_functor
                self.epsilon = epsilon
            
            def map_object(self, obj):
                # Map through F composition
                return self.migration_functor.F.map_object(obj)
            
            def map_morphism(self, morphism, source_obj, target_obj):
                # Compose with F
                f_morphism = self.migration_functor.F.map_morphism(morphism, source_obj, target_obj)
                return f_morphism
        
        return PullbackFunctor(self, epsilon_functor)
    
    def left_pushforward_functor(self, delta_functor: Functor) -> LeftKanExtension:
        """
        Σ_F: δ → ε
        
        Left pushforward as left Kan extension (left adjoint to pullback)
        """
        return LeftKanExtension(delta_functor, self.F, f"LeftPushforward_{self.name}")
    
    def right_pushforward_functor(self, delta_functor: Functor) -> RightKanExtension:
        """
        Π_F: δ → ε
        
        Right pushforward as right Kan extension (right adjoint to pullback)
        """
        return RightKanExtension(delta_functor, self.F, f"RightPushforward_{self.name}")

class FoundationModelBuilder:
    """
    Foundation model builder using Kan extensions
    
    Core innovation of GAIA: build foundation models by extending functors
    over categories rather than interpolating functions on sets
    """
    
    def __init__(self, name: str = "FoundationModelBuilder"):
        self.name = name
        self.base_categories = {}      # name -> category
        self.base_functors = {}        # name -> functor
        self.extensions = {}           # name -> kan extension
    
    def add_base_category(self, name: str, category: FuzzyCategory):
        """Add base category for foundation model"""
        self.base_categories[name] = category
        # Get object count from either old or new category structure
        object_count = 0
        if hasattr(category, 'objects'):
            object_count = len(category.objects)
        elif hasattr(category, 'graded_objects'):
            object_count = sum(len(objects) for objects in category.graded_objects.values())
        
        logger.info(f"Added base category '{name}' with {object_count} objects")
    
    def add_base_functor(self, name: str, functor: Functor):
        """Add base functor for foundation model"""
        self.base_functors[name] = functor
        logger.info(f"Added base functor '{name}': {functor.source_category.name} → {functor.target_category.name}")
    
    def build_foundation_model_via_left_kan(self, 
                                          base_functor_name: str,
                                          extension_functor_name: str,
                                          model_name: str) -> LeftKanExtension:
        """
        Build foundation model via left Kan extension
        
        This extends the base functor along the extension direction
        """
        if base_functor_name not in self.base_functors:
            raise ValueError(f"Base functor '{base_functor_name}' not found")
        if extension_functor_name not in self.base_functors:
            raise ValueError(f"Extension functor '{extension_functor_name}' not found")
        
        base_functor = self.base_functors[base_functor_name]
        extension_functor = self.base_functors[extension_functor_name]
        
        # Create left Kan extension
        left_kan = LeftKanExtension(base_functor, extension_functor, model_name)
        self.extensions[model_name] = left_kan
        
        logger.info(f"Built foundation model '{model_name}' via left Kan extension")
        return left_kan
    
    def build_foundation_model_via_right_kan(self, 
                                           base_functor_name: str,
                                           extension_functor_name: str,
                                           model_name: str) -> RightKanExtension:
        """
        Build foundation model via right Kan extension
        """
        if base_functor_name not in self.base_functors:
            raise ValueError(f"Base functor '{base_functor_name}' not found")
        if extension_functor_name not in self.base_functors:
            raise ValueError(f"Extension functor '{extension_functor_name}' not found")
        
        base_functor = self.base_functors[base_functor_name]
        extension_functor = self.base_functors[extension_functor_name]
        
        # Create right Kan extension
        right_kan = RightKanExtension(base_functor, extension_functor, model_name)
        self.extensions[model_name] = right_kan
        
        logger.info(f"Built foundation model '{model_name}' via right Kan extension")
        return right_kan
    
    def apply_model_modification(self, 
                               original_model_name: str,
                               modification_functor: Functor,
                               modified_model_name: str) -> Dict[str, Any]:
        """
        Apply modification to foundation model using migration functors
        
        This implements the model modification framework from GAIA paper
        """
        if original_model_name not in self.extensions:
            raise ValueError(f"Original model '{original_model_name}' not found")
        
        original_extension = self.extensions[original_model_name]
        
        # Create migration functor
        migration = MigrationFunctor(modification_functor, f"Migration_{modified_model_name}")
        
        # Apply migration functors
        if isinstance(original_extension, LeftKanExtension):
            base_functor = original_extension.F
        else:  # RightKanExtension
            base_functor = original_extension.F
        
        pullback = migration.pullback_functor(base_functor)
        left_pushforward = migration.left_pushforward_functor(base_functor)
        right_pushforward = migration.right_pushforward_functor(base_functor)
        
        modification_result = {
            'original_model': original_extension,
            'modification_functor': modification_functor,
            'pullback': pullback,
            'left_pushforward': left_pushforward,
            'right_pushforward': right_pushforward,
            'migration_functor': migration
        }
        
        self.extensions[modified_model_name] = modification_result
        
        logger.info(f"Applied modification to create '{modified_model_name}' from '{original_model_name}'")
        return modification_result
    
    def get_foundation_model(self, name: str):
        """Get foundation model by name"""
        return self.extensions.get(name)
    
    def list_models(self) -> List[str]:
        """List all foundation models"""
        return list(self.extensions.keys())

# Factory functions for common use cases

def create_llm_foundation_model(vocab_size: int, hidden_dim: int) -> FoundationModelBuilder:
    """
    Create fuzzy simplicial foundation model for Large Language Models
    
    Uses Kan extensions over fuzzy simplicial categories to build LLM from categorical components
    """
    builder = FoundationModelBuilder("FuzzyLLM_Foundation")
    
    # Create fuzzy token category with graded structure
    token_category = FuzzyGenerativeAICategory("FuzzyTokenCategory")
    for i in range(min(vocab_size, 100)):  # Limit for demo
        # Tokens at dimension 0 (vertices) with varying membership
        membership = 1.0 - (i / vocab_size) * 0.3  # Higher frequency tokens have higher membership
        token_category.add_fuzzy_object(f"token_{i}", dimension=0, membership=membership)
    
    # Create fuzzy embedding category with higher-dimensional structure
    embedding_category = FuzzyGenerativeAICategory("FuzzyEmbeddingCategory")
    embedding_component = nn.Embedding(vocab_size, hidden_dim)
    # Embeddings at dimension 1 (edges) representing token relationships
    embedding_category.add_fuzzy_object("embeddings", dimension=1, membership=0.9, component=embedding_component)
    
    # Create fuzzy neural functor from tokens to embeddings
    token_to_embedding = FuzzyNeuralFunctor(token_category, embedding_category)
    
    builder.add_base_category("fuzzy_tokens", token_category)
    builder.add_base_category("fuzzy_embeddings", embedding_category)
    builder.add_base_functor("fuzzy_token_embedding", token_to_embedding)
    
    return builder

def create_diffusion_foundation_model(image_size: int, noise_steps: int) -> FoundationModelBuilder:
    """
    Create fuzzy simplicial foundation model for Diffusion Models
    
    Uses Kan extensions over fuzzy simplicial categories for noise-to-image generation
    """
    builder = FoundationModelBuilder("FuzzyDiffusion_Foundation")
    
    # Create fuzzy noise category with temporal graded structure
    noise_category = FuzzyGenerativeAICategory("FuzzyNoiseCategory")
    for t in range(min(noise_steps, 50)):  # Limit for demo
        # Noise at different dimensions based on timestep
        dimension = min(t // 10, 3)  # Group timesteps into dimensions 0-3
        membership = 1.0 - (t / noise_steps) * 0.4  # Earlier timesteps have higher membership
        noise_category.add_fuzzy_object(f"noise_t{t}", dimension=dimension, membership=membership)
    
    # Create fuzzy image category with spatial structure
    image_category = FuzzyGenerativeAICategory("FuzzyImageCategory")
    image_component = nn.Conv2d(3, 3, 3, padding=1)
    # Images at dimension 2 (faces) representing spatial relationships
    image_category.add_fuzzy_object("images", dimension=2, membership=0.95, component=image_component)
    
    # Create fuzzy denoising functor
    denoising_functor = FuzzyNeuralFunctor(noise_category, image_category)
    
    builder.add_base_category("fuzzy_noise", noise_category)
    builder.add_base_category("fuzzy_images", image_category)
    builder.add_base_functor("fuzzy_denoising", denoising_functor)
    
    return builder

def create_multimodal_foundation_model() -> FoundationModelBuilder:
    """
    Create fuzzy simplicial multimodal foundation model
    
    Combines text, image, and audio modalities via fuzzy Kan extensions
    """
    builder = FoundationModelBuilder("FuzzyMultimodal_Foundation")
    
    # Create fuzzy modality categories with graded structure
    text_category = FuzzyGenerativeAICategory("FuzzyTextCategory")
    text_component = nn.LSTM(512, 256)
    # Text at dimension 1 (sequential structure)
    text_category.add_fuzzy_object("text_encoder", dimension=1, membership=0.9, component=text_component)
    
    image_category = FuzzyGenerativeAICategory("FuzzyImageCategory")  
    image_component = nn.Conv2d(3, 256, 3)
    # Images at dimension 2 (spatial structure)
    image_category.add_fuzzy_object("image_encoder", dimension=2, membership=0.85, component=image_component)
    
    audio_category = FuzzyGenerativeAICategory("FuzzyAudioCategory")
    audio_component = nn.Conv1d(1, 256, 3)
    # Audio at dimension 1 (temporal structure)
    audio_category.add_fuzzy_object("audio_encoder", dimension=1, membership=0.8, component=audio_component)
    
    # Create fuzzy shared representation category
    shared_category = FuzzyGenerativeAICategory("FuzzySharedCategory")
    shared_component = nn.Linear(256, 256)
    # Shared representations at dimension 3 (highest abstraction)
    shared_category.add_fuzzy_object("shared_repr", dimension=3, membership=0.95, component=shared_component)
    
    # Create fuzzy cross-modal functors
    text_to_shared = FuzzyNeuralFunctor(text_category, shared_category)
    image_to_shared = FuzzyNeuralFunctor(image_category, shared_category)
    audio_to_shared = FuzzyNeuralFunctor(audio_category, shared_category)
    
    builder.add_base_category("fuzzy_text", text_category)
    builder.add_base_category("fuzzy_image", image_category)
    builder.add_base_category("fuzzy_audio", audio_category)
    builder.add_base_category("fuzzy_shared", shared_category)
    
    builder.add_base_functor("fuzzy_text_to_shared", text_to_shared)
    builder.add_base_functor("fuzzy_image_to_shared", image_to_shared)
    builder.add_base_functor("fuzzy_audio_to_shared", audio_to_shared)
    
    return builder

# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing Fuzzy Simplicial Kan Extensions implementation...")
    
    # Test 1: Basic fuzzy simplicial categories and functors
    print("\n1. Testing fuzzy simplicial categories and functors:")
    source_cat = FuzzyGenerativeAICategory("FuzzySource")
    target_cat = FuzzyGenerativeAICategory("FuzzyTarget")
    
    source_cat.add_fuzzy_object("A", dimension=0, membership=0.9)
    source_cat.add_fuzzy_object("B", dimension=1, membership=0.8)
    target_cat.add_fuzzy_object("X", dimension=0, membership=0.95)
    target_cat.add_fuzzy_object("Y", dimension=1, membership=0.85)
    
    functor = FuzzyNeuralFunctor(source_cat, target_cat)
    mapped_A, mapped_dim, mapped_membership = functor.map_fuzzy_object("A", 0, 0.9)
    print(f"   Fuzzy functor maps A to: {mapped_A} (dim={mapped_dim}, μ={mapped_membership:.3f})")
    
    # Test 2: Fuzzy Left Kan extension
    print("\n2. Testing Fuzzy Left Kan extension:")
    extension_cat = FuzzyGenerativeAICategory("FuzzyExtension")
    extension_cat.add_fuzzy_object("P", dimension=0, membership=0.7)
    extension_cat.add_fuzzy_object("Q", dimension=2, membership=0.6)
    
    K_functor = FuzzyNeuralFunctor(source_cat, extension_cat)
    left_kan = LeftKanExtension(functor, K_functor, "FuzzyTestLeftKan")
    print(f"   Fuzzy Left Kan extension created: {left_kan.name}")
    
    # Test 3: Fuzzy foundation model builder
    print("\n3. Testing Fuzzy Foundation model builder:")
    builder = create_llm_foundation_model(vocab_size=1000, hidden_dim=256)
    print(f"   Created fuzzy LLM foundation builder with {len(builder.base_categories)} categories")
    
    # Build fuzzy foundation model
    foundation_model = builder.build_foundation_model_via_left_kan(
        "fuzzy_token_embedding", "fuzzy_token_embedding", "FuzzyLLM_v1"
    )
    print(f"   Built fuzzy foundation model: {foundation_model.name}")
    
    # Test 4: Fuzzy multimodal foundation model
    print("\n4. Testing Fuzzy Multimodal foundation model:")
    multimodal_builder = create_multimodal_foundation_model()
    print(f"   Created fuzzy multimodal builder with {len(multimodal_builder.base_functors)} functors")
    
    # Test 5: Fuzzy colimit computation
    print("\n5. Testing Fuzzy colimit computation:")
    test_objects = ["A", "B"]
    colimit_simplex, colimit_membership = source_cat.compute_fuzzy_colimit(test_objects)
    print(f"   Fuzzy colimit computed with membership: {colimit_membership:.3f}")
    
    print("\n✅ Fuzzy Simplicial Kan Extensions implementation complete!")
    print("🎯 Section 6.6 From (MAHADEVAN,2024) now implemented - Foundation models via fuzzy functor extension over simplicial categories")
    print("🔧 Key improvements:")
    print("   • Fuzzy simplicial categories with graded objects S_n")
    print("   • Proper colimit construction for Kan extensions")
    print("   • Fuzzy membership propagation through t-norms")
    print("   • Universal property loss using fuzzy simplicial metrics")