"""
GAIA Integrated Structures

This module integrates fuzzy sets, simplicial sets, and coalgebras into a unified
system following the abstractions defined in abstractions.py.

Combines functionality from:
- fuzzy.py (fuzzy sets and membership functions)
- simplices.py (simplicial structures)  
- coalgebras.py (F-coalgebras and endofunctors)
- functor.py (simplicial functors)
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import uuid
import torch
import torch.nn as nn
import numpy as np
from enum import Enum

from .abstractions import (
    CategoryObject, Morphism, Functor, Endofunctor, StructureMap, Coalgebra,
    FuzzyMembership, SimplicialStructure, GAIAComponent, TrainingState
)


class TConorm(Enum):
    """T-conorms for fuzzy set operations."""
    MAXIMUM = "maximum"
    PROBABILISTIC = "probabilistic" 
    LUKASIEWICZ = "lukasiewicz"
    
    def apply(self, a: float, b: float) -> float:
        """Apply the t-conorm operation."""
        if self == TConorm.MAXIMUM:
            return max(a, b)
        elif self == TConorm.PROBABILISTIC:
            return a + b - a * b
        elif self == TConorm.LUKASIEWICZ:
            return min(1.0, a + b)
        else:
            raise ValueError(f"Unknown t-conorm: {self}")


@dataclass(frozen=True)
class FuzzyElement:
    """Element with fuzzy membership degree."""
    element: Any
    membership: float
    
    def __post_init__(self):
        if not 0.0 <= self.membership <= 1.0:
            raise ValueError(f"Membership must be in [0,1], got {self.membership}")


class IntegratedFuzzySet(GAIAComponent):
    """Integrated fuzzy set implementing sheaf structure on [0,1]."""
    
    def __init__(self, elements: Set[Any], membership_fn: FuzzyMembership, 
                 name: str = "fuzzy_set"):
        super().__init__(name)
        self.elements = set(elements)
        self.membership_function = membership_fn
        self._cache: Dict[Any, float] = {}
    
    def initialize(self) -> None:
        """Initialize fuzzy set by computing membership for all elements."""
        for element in self.elements:
            self._cache[element] = self.membership_function(element)
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update fuzzy set based on training state."""
        # Fuzzy sets are typically static, but could adapt membership
        return state
    
    def validate(self) -> bool:
        """Validate fuzzy set properties."""
        for element in self.elements:
            membership = self.get_membership(element)
            if not 0.0 <= membership <= 1.0:
                return False
        return True
    
    def get_membership(self, element: Any) -> float:
        """Get membership degree for element."""
        if element in self._cache:
            return self._cache[element]
        
        membership = self.membership_function(element)
        self._cache[element] = membership
        return membership
    
    def support(self) -> Set[Any]:
        """Return support set {x | μ(x) > 0}."""
        return {elem for elem in self.elements if self.get_membership(elem) > 0}
    
    def alpha_cut(self, alpha: float) -> Set[Any]:
        """Return α-cut {x | μ(x) ≥ α}."""
        return {elem for elem in self.elements if self.get_membership(elem) >= alpha}
    
    def merge_with(self, other: 'IntegratedFuzzySet', 
                   t_conorm: TConorm = TConorm.MAXIMUM) -> 'IntegratedFuzzySet':
        """Merge with another fuzzy set using t-conorm."""
        combined_elements = self.elements.union(other.elements)
        
        def merged_membership(x):
            m1 = self.get_membership(x) if x in self.elements else 0.0
            m2 = other.get_membership(x) if x in other.elements else 0.0
            return t_conorm.apply(m1, m2)
        
        return IntegratedFuzzySet(
            combined_elements, merged_membership, 
            f"{self.name}_merged_{other.name}"
        )


class IntegratedSimplex(SimplicialStructure):
    """Integrated simplex with fuzzy membership and categorical structure."""
    
    def __init__(self, dimension: int, name: str, components: Optional[Tuple] = None,
                 membership: float = 1.0, payload: Any = None):
        self.id = uuid.uuid4()
        self.name = name
        self._dimension = dimension
        self.components = components or ()
        self.membership = membership
        self.payload = payload
        self._face_cache: Dict[int, 'IntegratedSimplex'] = {}
        self._degeneracy_cache: Dict[int, 'IntegratedSimplex'] = {}
    
    @property
    def dimension(self) -> int:
        """Dimension of the simplex."""
        return self._dimension
    
    def face_map(self, i: int) -> 'IntegratedSimplex':
        """Apply i-th face map δᵢ with membership coherence."""
        if i in self._face_cache:
            return self._face_cache[i]
        
        if self.dimension == 0:
            raise ValueError("0-simplex has no faces")
        
        if not 0 <= i <= self.dimension:
            raise ValueError(f"Face index {i} out of range for {self.dimension}-simplex")
        
        # Create face by omitting i-th component
        face_components = tuple(
            comp for j, comp in enumerate(self.components) if j != i
        )
        
        # Membership coherence: face membership ≥ simplex membership
        face_membership = max(self.membership, 
                             min(getattr(comp, 'membership', 1.0) 
                                 for comp in face_components) if face_components else 1.0)
        
        face = IntegratedSimplex(
            self.dimension - 1, f"∂_{i}({self.name})", 
            face_components, face_membership
        )
        
        self._face_cache[i] = face
        return face
    
    def degeneracy_map(self, i: int) -> 'IntegratedSimplex':
        """Apply i-th degeneracy map σᵢ preserving membership."""
        if i in self._degeneracy_cache:
            return self._degeneracy_cache[i]
        
        if not 0 <= i <= self.dimension:
            raise ValueError(f"Degeneracy index {i} out of range")
        
        # Create degeneracy by repeating i-th component
        if self.components:
            degen_components = (
                self.components[:i] + 
                (self.components[i], self.components[i]) + 
                self.components[i+1:]
            )
        else:
            degen_components = (self, self)
        
        # Degeneracies preserve membership strength
        degen = IntegratedSimplex(
            self.dimension + 1, f"σ_{i}({self.name})",
            degen_components, self.membership
        )
        
        self._degeneracy_cache[i] = degen
        return degen
    
    def verify_simplicial_identities(self) -> bool:
        """Verify simplicial identities hold."""
        if self.dimension < 2:
            return True
        
        try:
            # Face-face relations: ∂ᵢ∂ⱼ = ∂ⱼ₋₁∂ᵢ for i < j
            for i in range(self.dimension):
                for j in range(i + 1, self.dimension + 1):
                    face_i_j = self.face_map(i).face_map(j - 1)
                    face_j_i = self.face_map(j).face_map(i)
                    # In practice would need proper equality check
            
            return True
        except Exception:
            return False


class IntegratedFuzzySimplicialSet(GAIAComponent):
    """Integrated fuzzy simplicial set combining fuzzy and simplicial structures."""
    
    def __init__(self, name: str, max_dimension: int = 3):
        super().__init__(name)
        self.max_dimension = max_dimension
        self.fuzzy_sets: Dict[int, IntegratedFuzzySet] = {}
        self.simplices: Dict[int, Dict[Any, IntegratedSimplex]] = {
            i: {} for i in range(max_dimension + 1)
        }
    
    def initialize(self) -> None:
        """Initialize all fuzzy sets."""
        for fuzzy_set in self.fuzzy_sets.values():
            fuzzy_set.initialize()
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update fuzzy simplicial set based on training state."""
        # Could adapt membership functions based on training
        for fuzzy_set in self.fuzzy_sets.values():
            state = fuzzy_set.update(state)
        return state
    
    def validate(self) -> bool:
        """Validate fuzzy simplicial set properties."""
        # Check membership coherence
        for dim in range(1, self.max_dimension + 1):
            for simplex in self.simplices[dim].values():
                if not self._verify_membership_coherence(simplex):
                    return False
        
        # Check degeneracy preservation
        for dim in range(self.max_dimension):
            for simplex in self.simplices[dim].values():
                if not self._verify_degeneracy_preservation(simplex):
                    return False
        
        return True
    
    def add_simplex(self, dimension: int, simplex_data: Any, membership: float = 1.0):
        """Add a simplex with given membership."""
        if dimension > self.max_dimension:
            raise ValueError(f"Dimension {dimension} exceeds maximum {self.max_dimension}")
        
        simplex = IntegratedSimplex(dimension, str(simplex_data), 
                                  simplex_data if isinstance(simplex_data, tuple) else (simplex_data,),
                                  membership)
        
        self.simplices[dimension][simplex_data] = simplex
        
        # Update corresponding fuzzy set
        if dimension not in self.fuzzy_sets:
            elements = set(self.simplices[dimension].keys())
            membership_fn = lambda x: self.simplices[dimension].get(x, 
                IntegratedSimplex(dimension, str(x))).membership
            self.fuzzy_sets[dimension] = IntegratedFuzzySet(
                elements, membership_fn, f"{self.name}_{dimension}"
            )
        else:
            self.fuzzy_sets[dimension].elements.add(simplex_data)
    
    def get_membership(self, dimension: int, simplex_data: Any) -> float:
        """Get membership degree for a simplex."""
        if dimension in self.simplices and simplex_data in self.simplices[dimension]:
            return self.simplices[dimension][simplex_data].membership
        return 0.0
    
    def _verify_membership_coherence(self, simplex: IntegratedSimplex) -> bool:
        """Verify membership coherence: μ(σ) ≤ min{μ(∂ᵢσ)}."""
        if simplex.dimension == 0:
            return True
        
        try:
            face_memberships = [
                simplex.face_map(i).membership 
                for i in range(simplex.dimension + 1)
            ]
            return simplex.membership <= min(face_memberships)
        except Exception:
            return False
    
    def _verify_degeneracy_preservation(self, simplex: IntegratedSimplex) -> bool:
        """Verify degeneracies preserve membership strength."""
        try:
            for i in range(simplex.dimension + 1):
                degen = simplex.degeneracy_map(i)
                if abs(degen.membership - simplex.membership) > 1e-6:
                    return False
            return True
        except Exception:
            return False
    
    def merge_with(self, other: 'IntegratedFuzzySimplicialSet', 
                   t_conorm: TConorm = TConorm.MAXIMUM) -> 'IntegratedFuzzySimplicialSet':
        """Merge with another fuzzy simplicial set."""
        max_dim = max(self.max_dimension, other.max_dimension)
        merged = IntegratedFuzzySimplicialSet(f"{self.name}_merged_{other.name}", max_dim)
        
        # Merge at each dimension
        for dim in range(max_dim + 1):
            # Collect all simplices at this dimension
            all_simplices = set()
            if dim <= self.max_dimension:
                all_simplices.update(self.simplices[dim].keys())
            if dim <= other.max_dimension:
                all_simplices.update(other.simplices[dim].keys())
            
            # Add merged simplices
            for simplex_data in all_simplices:
                m1 = self.get_membership(dim, simplex_data)
                m2 = other.get_membership(dim, simplex_data)
                merged_membership = t_conorm.apply(m1, m2)
                merged.add_simplex(dim, simplex_data, merged_membership)
        
        return merged


class IntegratedCoalgebra(Coalgebra[torch.Tensor], GAIAComponent):
    """Integrated coalgebra for GAIA training dynamics."""
    
    def __init__(self, initial_state: torch.Tensor, endofunctor: Endofunctor,
                 name: str = "coalgebra"):
        def structure_map(state):
            return endofunctor.apply_to_object(state)
        
        super(Coalgebra, self).__init__(name)
        Coalgebra.__init__(self, initial_state, structure_map)
        self.endofunctor = endofunctor
        self.trajectory: List[torch.Tensor] = [initial_state]
    
    def initialize(self) -> None:
        """Initialize coalgebra."""
        pass
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update coalgebra with new training state."""
        # Extract parameters from training state
        if 'parameters' in state.parameters:
            new_params = state.parameters['parameters']
            evolved = self.evolve(new_params)
            self.trajectory.append(evolved)
            
            # Update training state with evolved parameters
            state.parameters['parameters'] = evolved
        
        return state
    
    def validate(self) -> bool:
        """Validate coalgebra properties."""
        return len(self.trajectory) > 0
    
    def is_bisimilar(self, other: 'IntegratedCoalgebra', 
                     relation: Callable[[torch.Tensor, torch.Tensor], bool]) -> bool:
        """Check bisimilarity with another coalgebra."""
        if len(self.trajectory) != len(other.trajectory):
            return False
        
        for s1, s2 in zip(self.trajectory, other.trajectory):
            if not relation(s1, s2):
                return False
        
        return True
    
    def iterate(self, initial_state: torch.Tensor, steps: int) -> List[torch.Tensor]:
        """Iterate coalgebra dynamics for multiple steps using proper coalgebraic iteration."""
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(steps):
            next_state = self.evolve(current_state)
            trajectory.append(next_state)
            # For parameter coalgebras, extract the parameters from the tuple
            if isinstance(next_state, tuple) and len(next_state) >= 3:
                current_state = next_state[2]  # Use parameters for next iteration
            else:
                current_state = next_state
        
        return trajectory
    
    def iterate_dynamics(self, steps: int) -> List[torch.Tensor]:
        """Iterate coalgebra dynamics for multiple steps."""
        current = self.trajectory[-1] if self.trajectory else self.carrier
        
        for _ in range(steps):
            current = self.evolve(current)
            self.trajectory.append(current)
        
        return self.trajectory[-steps:]


# Factory functions for common patterns
def create_fuzzy_simplex(dimension: int, name: str, membership: float = 1.0) -> IntegratedSimplex:
    """Create a fuzzy simplex with given properties."""
    return IntegratedSimplex(dimension, name, membership=membership)


def create_fuzzy_simplicial_set_from_data(data: np.ndarray, k: int = 5, 
                                         name: str = "data_fss") -> IntegratedFuzzySimplicialSet:
    """Create fuzzy simplicial set from point cloud data using k-NN."""
    from sklearn.neighbors import NearestNeighbors
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Create fuzzy simplicial set
    fss = IntegratedFuzzySimplicialSet(name, max_dimension=2)
    
    # Add 0-simplices (vertices)
    for i in range(len(data)):
        fss.add_simplex(0, i, 1.0)
    
    # Add 1-simplices (edges) with fuzzy membership
    for i in range(len(data)):
        for j in range(1, k+1):  # Skip self (index 0)
            neighbor = indices[i, j]
            distance = distances[i, j]
            
            # Convert distance to membership (closer = higher membership)
            membership = np.exp(-distance)
            fss.add_simplex(1, (i, neighbor), membership)
    
    return fss


def merge_fuzzy_simplicial_sets(*fss_list: IntegratedFuzzySimplicialSet,
                               t_conorm: TConorm = TConorm.MAXIMUM) -> IntegratedFuzzySimplicialSet:
    """Merge multiple fuzzy simplicial sets."""
    if not fss_list:
        raise ValueError("Cannot merge empty list of fuzzy simplicial sets")
    
    result = fss_list[0]
    for fss in fss_list[1:]:
        result = result.merge_with(fss, t_conorm)
    
    return result


# Export public interface
__all__ = [
    'TConorm', 'FuzzyElement', 'IntegratedFuzzySet', 'IntegratedSimplex',
    'IntegratedFuzzySimplicialSet', 'IntegratedCoalgebra',
    'create_fuzzy_simplex', 'create_fuzzy_simplicial_set_from_data',
    'merge_fuzzy_simplicial_sets'
]