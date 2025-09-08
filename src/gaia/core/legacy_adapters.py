"""
GAIA Legacy Adapters

This module provides adapters to maintain compatibility with existing code
while using the new integrated structures. This ensures no functionality is lost
during the refactoring process.
"""

from typing import Any, Dict, List, Optional, Set, Callable
import torch
import numpy as np

# Import legacy modules (avoiding circular imports)
# from .fuzzy import FuzzySet as LegacyFuzzySet, FuzzySimplicialSet as LegacyFuzzySimplicialSet
# from .coalgebras import FCoalgebra as LegacyCoalgebra
# from .business_units import BusinessUnit as LegacyBusinessUnit
# from .kan_verification import KanComplexVerifier as LegacyKanVerifier

# Import new integrated modules
from .integrated_structures import (
    IntegratedFuzzySet, IntegratedFuzzySimplicialSet, IntegratedCoalgebra,
    TConorm
)
from .abstractions import GAIAComponent


class FuzzySetAdapter:
    """Adapter for legacy FuzzySet interface."""
    
    def __init__(self, integrated_fuzzy_set: IntegratedFuzzySet):
        self._integrated = integrated_fuzzy_set
    
    @property
    def elements(self) -> Set[Any]:
        return self._integrated.elements
    
    @property
    def membership_function(self) -> Callable[[Any], float]:
        return self._integrated.membership_function
    
    def get_membership(self, element: Any) -> float:
        return self._integrated.get_membership(element)
    
    def support(self) -> Set[Any]:
        return self._integrated.support()
    
    def alpha_cut(self, alpha: float) -> Set[Any]:
        return self._integrated.alpha_cut(alpha)


class FuzzySimplicialSetAdapter:
    """Adapter for legacy FuzzySimplicialSet interface."""
    
    def __init__(self, integrated_fss: IntegratedFuzzySimplicialSet):
        self._integrated = integrated_fss
    
    @property
    def name(self) -> str:
        return self._integrated.name
    
    @property
    def dimension(self) -> int:
        return self._integrated.max_dimension
    
    def add_simplex(self, dimension: int, simplex_data: Any, membership: float = 1.0):
        return self._integrated.add_simplex(dimension, simplex_data, membership)
    
    def get_membership(self, dimension: int, simplex_data: Any) -> float:
        return self._integrated.get_membership(dimension, simplex_data)
    
    def verify_membership_coherence(self) -> bool:
        return self._integrated.validate()
    
    def verify_degeneracy_preservation(self) -> bool:
        return self._integrated.validate()


class CoalgebraAdapter:
    """Adapter for legacy Coalgebra interface."""
    
    def __init__(self, integrated_coalgebra: IntegratedCoalgebra):
        self._integrated = integrated_coalgebra
    
    def evolve(self, state: torch.Tensor) -> Any:
        return self._integrated.evolve(state)
    
    def iterate(self, initial_state: torch.Tensor, steps: int) -> List[torch.Tensor]:
        return self._integrated.iterate_dynamics(steps)
    
    def is_bisimilar(self, other: 'CoalgebraAdapter', 
                     relation: Callable[[torch.Tensor, torch.Tensor], bool]) -> bool:
        return self._integrated.is_bisimilar(other._integrated, relation)


def create_legacy_fuzzy_set(elements: Set[Any], membership_fn: Callable[[Any], float],
                           name: str = "legacy_fuzzy_set") -> FuzzySetAdapter:
    """Create a legacy-compatible fuzzy set."""
    integrated = IntegratedFuzzySet(elements, membership_fn, name)
    integrated.initialize()
    return FuzzySetAdapter(integrated)


def create_legacy_fuzzy_simplicial_set(name: str, dimension: int = 3) -> FuzzySimplicialSetAdapter:
    """Create a legacy-compatible fuzzy simplicial set."""
    integrated = IntegratedFuzzySimplicialSet(name, dimension)
    integrated.initialize()
    return FuzzySimplicialSetAdapter(integrated)


def create_legacy_coalgebra(initial_state: torch.Tensor, structure_map: Callable,
                           name: str = "legacy_coalgebra") -> CoalgebraAdapter:
    """Create a legacy-compatible coalgebra."""
    class SimpleEndofunctor:
        def apply_to_object(self, state):
            # Return 3 values to match coalgebra expectations: (activations, gradients, params)
            batch_size = state.shape[0] if state.dim() > 1 else 1
            state_dim = state.shape[-1] if state.dim() > 0 else 1
            
            # Create dummy activations and gradients
            activations = torch.zeros(batch_size, state_dim, device=state.device)
            gradients = torch.zeros(batch_size, state_dim, device=state.device)
            
            # Apply structure map to get updated state
            updated_state = structure_map(state)
            
            return activations, gradients, updated_state
    
    endofunctor = SimpleEndofunctor()
    integrated = IntegratedCoalgebra(initial_state, endofunctor, name)
    integrated.initialize()
    return CoalgebraAdapter(integrated)


def merge_fuzzy_simplicial_sets(fss1: FuzzySimplicialSetAdapter, 
                               fss2: FuzzySimplicialSetAdapter,
                               t_conorm: Any = None) -> FuzzySimplicialSetAdapter:
    """Legacy-compatible merge function."""
    if t_conorm is None:
        t_conorm = TConorm.MAXIMUM
    elif hasattr(t_conorm, 'maximum'):  # Legacy TConorm enum
        t_conorm = TConorm.MAXIMUM
    
    merged = fss1._integrated.merge_with(fss2._integrated, t_conorm)
    return FuzzySimplicialSetAdapter(merged)


# Compatibility classes that match legacy interface
class FuzzySet(FuzzySetAdapter):
    """Legacy-compatible FuzzySet class."""
    def __init__(self, elements: Set[Any], membership_fn: Callable[[Any], float], name: str = "legacy_fuzzy_set"):
        integrated = IntegratedFuzzySet(elements, membership_fn, name)
        integrated.initialize()
        super().__init__(integrated)


class FuzzySimplicialSet(FuzzySimplicialSetAdapter):
    """Legacy-compatible FuzzySimplicialSet class."""
    def __init__(self, name: str, dimension: int = 3):
        integrated = IntegratedFuzzySimplicialSet(name, dimension)
        integrated.initialize()
        super().__init__(integrated)


class FCoalgebra(CoalgebraAdapter):
    """Legacy-compatible FCoalgebra class."""
    def __init__(self, initial_state: torch.Tensor, structure_map: Callable, name: str = "legacy_coalgebra"):
        class SimpleEndofunctor:
            def apply_to_object(self, state):
                # Return 3 values to match coalgebra expectations: (activations, gradients, params)
                batch_size = state.shape[0] if state.dim() > 1 else 1
                state_dim = state.shape[-1] if state.dim() > 0 else 1
                
                # Create dummy activations and gradients
                activations = torch.zeros(batch_size, state_dim, device=state.device)
                gradients = torch.zeros(batch_size, state_dim, device=state.device)
                
                # Apply structure map to get updated state
                updated_state = structure_map(state)
                
                return activations, gradients, updated_state
        
        endofunctor = SimpleEndofunctor()
        integrated = IntegratedCoalgebra(initial_state, endofunctor, name)
        integrated.initialize()
        super().__init__(integrated)

# Export legacy interface
__all__ = [
    'FuzzySetAdapter', 'FuzzySimplicialSetAdapter', 'CoalgebraAdapter',
    'create_legacy_fuzzy_set', 'create_legacy_fuzzy_simplicial_set',
    'create_legacy_coalgebra', 'merge_fuzzy_simplicial_sets',
    'FuzzySet', 'FuzzySimplicialSet', 'FCoalgebra'
]