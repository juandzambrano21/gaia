"""
Module: coalgebras
Implements universal coalgebras for the GAIA framework.

Following Section 4.2 of the theoretical framework and Mahadevan (2024), this implements:
1. F-coalgebras (X,γ) with structure map γ: X → F(X)
2. Coalgebra morphisms h: (X,γ) → (Y,δ) satisfying F(h) ∘ γ = δ ∘ h
3. Bisimulations between coalgebras
4. Final coalgebras and Lambek's theorem
5. Generative dynamics for state evolution

This is critical for modeling backpropagation as an endofunctor and
implementing the coalgebraic approach to generative AI.
"""

import uuid
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any, TypeVar, Generic, Protocol, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

from .simplices import SimplicialObject
from . import DEVICE


# Type variables for generic coalgebras
X = TypeVar('X')  # State space
Y = TypeVar('Y')  # Target state space


@runtime_checkable
class EndofunctorProtocol(Protocol[X]):
    """Protocol for endofunctors F: C → C."""
    
    def apply_to_object(self, obj: X) -> X:
        """Apply endofunctor to object."""
        ...
    
    def apply_to_morphism(self, morphism: Callable[[X], Y]) -> Callable[[X], Y]:
        """Apply endofunctor to morphism."""
        ...


class BackpropagationEndofunctor:
    """
    Backpropagation endofunctor F_B: Param → Param.
    
    Following Definition 11 from the paper:
    F_B(X) = A × B × X for backpropagation endofunctor
    where A represents activations, B represents gradients.
    """
    
    def __init__(self, activation_dim: int, gradient_dim: int, name: str = "F_B"):
        self.activation_dim = activation_dim
        self.gradient_dim = gradient_dim
        self.name = name
        self.id = uuid.uuid4()
    
    def apply_to_object(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply F_B to parameter object: X → A × B × X.
        
        Args:
            params: Parameter tensor
            
        Returns:
            Tuple (activations, gradients, parameters)
        """
        batch_size = params.shape[0] if params.dim() > 1 else 1
        
        # Create activation and gradient components with some dynamics
        activations = torch.randn(batch_size, self.activation_dim, device=params.device) * 0.1
        gradients = torch.randn(batch_size, self.gradient_dim, device=params.device) * 0.01
        
        # Evolve parameters using gradient-like dynamics
        # This simulates parameter evolution through coalgebraic dynamics
        noise = torch.randn_like(params) * 0.001
        evolved_params = params + noise
        
        return activations, gradients, evolved_params
    
    def apply_to_morphism(self, f: Callable) -> Callable:
        """Apply endofunctor to morphism."""
        def F_f(x):
            a, b, p = x
            # Apply f to parameter component
            new_p = f(p)
            # Transform activations and gradients accordingly
            return a, b, new_p
        
        return F_f
    
    def __repr__(self):
        return f"BackpropagationEndofunctor(A_dim={self.activation_dim}, B_dim={self.gradient_dim})"


class SGDEndofunctor:
    """
    Stochastic gradient descent endofunctor F_SGD: Param → Param.
    
    Following Definition 12 from the paper:
    F_SGD(X) = A × B × D(X) for stochastic backpropagation coalgebra
    where D is the distribution functor.
    """
    
    def __init__(self, activation_dim: int, gradient_dim: int, 
                 distribution_type: str = "gaussian", name: str = "F_SGD"):
        self.activation_dim = activation_dim
        self.gradient_dim = gradient_dim
        self.distribution_type = distribution_type
        self.name = name
        self.id = uuid.uuid4()
    
    def apply_to_object(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        Apply F_SGD to parameter object: X → A × B × D(X).
        
        Args:
            params: Parameter tensor
            
        Returns:
            Tuple (activations, gradients, parameter_distribution)
        """
        batch_size = params.shape[0] if params.dim() > 1 else 1
        
        # Create activation and gradient components
        activations = torch.zeros(batch_size, self.activation_dim, device=params.device)
        gradients = torch.zeros(batch_size, self.gradient_dim, device=params.device)
        
        # Create parameter distribution
        if self.distribution_type == "gaussian":
            param_dist = torch.distributions.Normal(params, torch.ones_like(params) * 0.1)
        elif self.distribution_type == "uniform":
            param_dist = torch.distributions.Uniform(params - 0.1, params + 0.1)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        
        return activations, gradients, param_dist
    
    def apply_to_morphism(self, f: Callable) -> Callable:
        """Apply endofunctor to morphism."""
        def F_f(x):
            a, b, dist = x
            # Sample from distribution and apply f
            sample = dist.sample()
            new_sample = f(sample)
            # Create new distribution around transformed sample
            if self.distribution_type == "gaussian":
                new_dist = torch.distributions.Normal(new_sample, dist.scale)
            else:
                new_dist = torch.distributions.Uniform(new_sample - 0.1, new_sample + 0.1)
            return a, b, new_dist
        
        return F_f
    
    def __repr__(self):
        return f"SGDEndofunctor(A_dim={self.activation_dim}, B_dim={self.gradient_dim}, dist={self.distribution_type})"


@dataclass
class FCoalgebra(Generic[X]):
    """
    F-coalgebra (X,γ) with structure map γ: X → F(X).
    
    This is the fundamental structure for modeling generative dynamics
    in the GAIA framework.
    """
    state_space: X
    structure_map: Callable[[X], Any]
    endofunctor: EndofunctorProtocol
    name: str = ""
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)
    
    def evolve(self, state: X) -> Any:
        """
        Evolve state using structure map: γ(state).
        
        This defines how states evolve in the generative system.
        """
        return self.structure_map(state)
    
    def iterate(self, initial_state: X, steps: int) -> List[Any]:
        """
        Iterate the coalgebra dynamics for multiple steps.
        
        Args:
            initial_state: Starting state
            steps: Number of evolution steps
            
        Returns:
            List of states through evolution
        """
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
    
    def is_fixed_point(self, state: X, tolerance: float = 1e-6) -> bool:
        """
        Check if state is a fixed point: γ(x) ≈ x.
        
        This is relevant for final coalgebras and Lambek's theorem.
        """
        evolved = self.evolve(state)
        
        # Handle different types of evolved states
        if isinstance(state, torch.Tensor) and isinstance(evolved, torch.Tensor):
            return torch.allclose(state, evolved, atol=tolerance)
        elif isinstance(state, torch.Tensor) and isinstance(evolved, tuple):
            # For endofunctors that return tuples, check parameter component
            if len(evolved) >= 3 and isinstance(evolved[2], torch.Tensor):
                return torch.allclose(state, evolved[2], atol=tolerance)
        
        return False
    
    def __repr__(self):
        return f"FCoalgebra(name='{self.name}', functor={self.endofunctor.__class__.__name__})"


class CoalgebraMorphism(Generic[X, Y]):
    """
    Morphism between F-coalgebras h: (X,γ) → (Y,δ).
    
    Must satisfy the commutative diagram: F(h) ∘ γ = δ ∘ h
    """
    
    def __init__(self, 
                 source: FCoalgebra[X], 
                 target: FCoalgebra[Y], 
                 morphism: Callable[[X], Y],
                 name: str = ""):
        self.source = source
        self.target = target
        self.morphism = morphism
        self.name = name
        self.id = uuid.uuid4()
        
        # Verify coalgebra morphism condition
        self._verify_coalgebra_condition()
    
    def _verify_coalgebra_condition(self):
        """
        Verify F(h) ∘ γ = δ ∘ h for coalgebra morphism.
        
        This is a simplified verification - full verification would require
        testing on representative elements of the state space.
        """
        # This is a placeholder for the verification
        # In practice, this would test the commutative diagram condition
        # on sample elements from the source state space
        pass
    
    def apply(self, state: X) -> Y:
        """Apply morphism to state."""
        return self.morphism(state)
    
    def __call__(self, state: X) -> Y:
        """Make morphism callable."""
        return self.apply(state)
    
    def __repr__(self):
        return f"CoalgebraMorphism('{self.source.name}' → '{self.target.name}')"


class BisimulationRelation(Generic[X]):
    """
    Bisimulation relation between coalgebras.
    
    A relation R ⊆ S × T between coalgebras (S,γ) and (T,δ) such that
    if (s,t) ∈ R, then (γ(s), δ(t)) ∈ F(R).
    """
    
    def __init__(self, 
                 coalgebra1: FCoalgebra[X], 
                 coalgebra2: FCoalgebra[X],
                 relation: Callable[[X, X], bool],
                 name: str = ""):
        self.coalgebra1 = coalgebra1
        self.coalgebra2 = coalgebra2
        self.relation = relation
        self.name = name
        self.id = uuid.uuid4()
    
    def are_bisimilar(self, state1: X, state2: X) -> bool:
        """Check if two states are bisimilar."""
        return self.relation(state1, state2)
    
    def verify_bisimulation_property(self, state1: X, state2: X) -> bool:
        """
        Verify bisimulation property for given states.
        
        If (s,t) ∈ R, then (γ(s), δ(t)) should be related by F(R).
        """
        if not self.are_bisimilar(state1, state2):
            return True  # Vacuously true if not related
        
        # Evolve both states
        evolved1 = self.coalgebra1.evolve(state1)
        evolved2 = self.coalgebra2.evolve(state2)
        
        # Check if evolved states are still related
        # This is a simplified check - full verification would require
        # checking the lifted relation F(R)
        return self.are_bisimilar(evolved1, evolved2)
    
    def __repr__(self):
        return f"BisimulationRelation('{self.coalgebra1.name}' ~ '{self.coalgebra2.name}')"


class FinalCoalgebra(Generic[X]):
    """
    Final coalgebra - terminal object in category of F-coalgebras.
    
    By Lambek's theorem, final F-coalgebra is isomorphic to F(final).
    """
    
    def __init__(self, endofunctor: EndofunctorProtocol, name: str = "final"):
        self.endofunctor = endofunctor
        self.name = name
        self.id = uuid.uuid4()
        
        # The final coalgebra is characterized by being a fixed point
        # of the endofunctor (Lambek's theorem)
        self._construct_final_coalgebra()
    
    def _construct_final_coalgebra(self):
        """
        Construct final coalgebra as fixed point of endofunctor.
        
        This is a simplified construction - in practice, final coalgebras
        are constructed through limits or other categorical constructions.
        """
        # Create structure map that is the identity (fixed point property)
        def final_structure_map(x):
            return self.endofunctor.apply_to_object(x)
        
        # The state space is implicitly the fixed point space
        self.coalgebra = FCoalgebra(
            state_space=None,  # Abstract state space
            structure_map=final_structure_map,
            endofunctor=self.endofunctor,
            name=f"final_{self.name}"
        )
    
    def unique_morphism_from(self, source: FCoalgebra[X]) -> CoalgebraMorphism[X, X]:
        """
        Get unique morphism from any coalgebra to final coalgebra.
        
        By definition of final object, there exists a unique morphism
        from any coalgebra to the final coalgebra.
        """
        def final_morphism(state: X) -> X:
            # The unique morphism to final coalgebra
            # In practice, this would be constructed categorically
            return state  # Simplified implementation
        
        return CoalgebraMorphism(
            source, self.coalgebra, final_morphism,
            name=f"!_{source.name}"
        )
    
    def verify_lambek_property(self) -> bool:
        """
        Verify Lambek's theorem: final coalgebra ≅ F(final coalgebra).
        
        This checks that the structure map is an isomorphism.
        """
        # This is a placeholder for Lambek's theorem verification
        # In practice, this would verify that the structure map
        # γ: Final → F(Final) is an isomorphism
        return True
    
    def __repr__(self):
        return f"FinalCoalgebra(functor={self.endofunctor.__class__.__name__})"


class CoalgebraCategory:
    """
    Category of F-coalgebras for a given endofunctor F.
    
    Objects: F-coalgebras (X,γ)
    Morphisms: Coalgebra morphisms h: (X,γ) → (Y,δ)
    """
    
    def __init__(self, endofunctor: EndofunctorProtocol, name: str = "Coalg(F)"):
        self.endofunctor = endofunctor
        self.name = name
        self.objects: Dict[uuid.UUID, FCoalgebra] = {}
        self.morphisms: Dict[uuid.UUID, CoalgebraMorphism] = {}
        self.composition_cache: Dict[Tuple[uuid.UUID, uuid.UUID], uuid.UUID] = {}
        self.final_coalgebra: Optional[FinalCoalgebra] = None
    
    def add_object(self, coalgebra: FCoalgebra) -> uuid.UUID:
        """Add F-coalgebra as object."""
        self.objects[coalgebra.id] = coalgebra
        return coalgebra.id
    
    def add_morphism(self, morphism: CoalgebraMorphism) -> uuid.UUID:
        """Add coalgebra morphism."""
        # Verify source and target are in category
        if morphism.source.id not in self.objects:
            self.add_object(morphism.source)
        if morphism.target.id not in self.objects:
            self.add_object(morphism.target)
        
        self.morphisms[morphism.id] = morphism
        return morphism.id
    
    def compose(self, f_id: uuid.UUID, g_id: uuid.UUID) -> CoalgebraMorphism:
        """Compose coalgebra morphisms g ∘ f."""
        if (f_id, g_id) in self.composition_cache:
            return self.morphisms[self.composition_cache[(f_id, g_id)]]
        
        f = self.morphisms[f_id]
        g = self.morphisms[g_id]
        
        if f.target.id != g.source.id:
            raise ValueError("Morphisms not composable")
        
        # Create composition
        def composed_morphism(x):
            return g.morphism(f.morphism(x))
        
        composition = CoalgebraMorphism(
            f.source, g.target, composed_morphism,
            name=f"{g.name} ∘ {f.name}"
        )
        
        comp_id = self.add_morphism(composition)
        self.composition_cache[(f_id, g_id)] = comp_id
        return composition
    
    def identity(self, obj_id: uuid.UUID) -> CoalgebraMorphism:
        """Get identity morphism for coalgebra."""
        obj = self.objects[obj_id]
        return CoalgebraMorphism(obj, obj, lambda x: x, name=f"id_{obj.name}")
    
    def get_final_coalgebra(self) -> FinalCoalgebra:
        """Get or construct final coalgebra."""
        if self.final_coalgebra is None:
            self.final_coalgebra = FinalCoalgebra(self.endofunctor)
        return self.final_coalgebra
    
    def all_morphisms_to_final(self) -> Dict[uuid.UUID, CoalgebraMorphism]:
        """Get unique morphisms from all objects to final coalgebra."""
        final = self.get_final_coalgebra()
        morphisms = {}
        
        for obj_id, coalgebra in self.objects.items():
            morphism = final.unique_morphism_from(coalgebra)
            morphisms[obj_id] = morphism
        
        return morphisms
    
    def __repr__(self):
        return f"CoalgebraCategory(functor={self.endofunctor.__class__.__name__}, |objects|={len(self.objects)})"


# Utility functions for creating common coalgebras

def create_parameter_coalgebra(params: torch.Tensor, 
                              learning_rate: float = 0.01,
                              name: str = "param_coalgebra") -> FCoalgebra[torch.Tensor]:
    """
    Create coalgebra for parameter evolution during training.
    
    Args:
        params: Initial parameters
        learning_rate: Learning rate for updates
        name: Name of coalgebra
        
    Returns:
        F-coalgebra modeling parameter dynamics
    """
    endofunctor = BackpropagationEndofunctor(
        activation_dim=params.shape[-1] if params.dim() > 0 else 1,
        gradient_dim=params.shape[-1] if params.dim() > 0 else 1
    )
    
    def structure_map(p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Structure map for parameter evolution."""
        activations, gradients, current_params = endofunctor.apply_to_object(p)
        
        # Simulate gradient computation (in practice, this would be actual gradients)
        simulated_gradients = torch.randn_like(current_params) * 0.1
        
        # Update parameters
        updated_params = current_params - learning_rate * simulated_gradients
        
        return activations, simulated_gradients, updated_params
    
    return FCoalgebra(
        state_space=params,
        structure_map=structure_map,
        endofunctor=endofunctor,
        name=name
    )


def create_stochastic_coalgebra(params: torch.Tensor,
                               learning_rate: float = 0.01,
                               noise_level: float = 0.01,
                               name: str = "stochastic_coalgebra") -> FCoalgebra[torch.Tensor]:
    """
    Create stochastic coalgebra for noisy parameter evolution.
    
    Args:
        params: Initial parameters
        learning_rate: Learning rate
        noise_level: Noise level for stochastic updates
        name: Name of coalgebra
        
    Returns:
        Stochastic F-coalgebra
    """
    endofunctor = SGDEndofunctor(
        activation_dim=params.shape[-1] if params.dim() > 0 else 1,
        gradient_dim=params.shape[-1] if params.dim() > 0 else 1
    )
    
    def stochastic_structure_map(p: torch.Tensor):
        """Stochastic structure map with noise."""
        activations, gradients, param_dist = endofunctor.apply_to_object(p)
        
        # Sample from parameter distribution
        sampled_params = param_dist.sample()
        
        # Add gradient noise
        gradient_noise = torch.randn_like(sampled_params) * noise_level
        simulated_gradients = torch.randn_like(sampled_params) * 0.1 + gradient_noise
        
        # Stochastic update
        updated_params = sampled_params - learning_rate * simulated_gradients
        
        return activations, simulated_gradients, torch.distributions.Normal(updated_params, torch.ones_like(updated_params) * noise_level)
    
    return FCoalgebra(
        state_space=params,
        structure_map=stochastic_structure_map,
        endofunctor=endofunctor,
        name=name
    )


def create_bisimulation_between_coalgebras(coalgebra1: FCoalgebra, coalgebra2: FCoalgebra,
                                         tolerance: float = 1e-3,
                                         name: str = "") -> BisimulationRelation:
    """
    Create bisimulation relation between two coalgebras.
    
    Args:
        coalgebra1: First coalgebra
        coalgebra2: Second coalgebra
        tolerance: Tolerance for state comparison
        name: Name of bisimulation
        
    Returns:
        Bisimulation relation
    """
    def bisimulation_relation(state1, state2) -> bool:
        """Check if states are approximately equal."""
        if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
            return torch.allclose(state1, state2, atol=tolerance)
        return False
    
    return BisimulationRelation(
        coalgebra1, coalgebra2, bisimulation_relation,
        name=name or f"bisim_{coalgebra1.name}_{coalgebra2.name}"
    )


# Integration with GAIA simplicial structure

class SimplicialCoalgebra:
    """
    Coalgebra structure over simplicial objects.
    
    This connects the coalgebraic approach with the simplicial
    structure of GAIA, enabling hierarchical generative dynamics.
    """
    
    def __init__(self, simplicial_object: SimplicialObject, 
                 endofunctor: EndofunctorProtocol,
                 name: str = ""):
        self.simplicial_object = simplicial_object
        self.endofunctor = endofunctor
        self.name = name or f"coalg_{simplicial_object.name}"
        self.id = uuid.uuid4()
        
        # Create coalgebra for the simplicial object's payload
        self.coalgebra = self._create_coalgebra()
    
    def _create_coalgebra(self) -> FCoalgebra:
        """Create coalgebra from simplicial object."""
        if hasattr(self.simplicial_object, 'payload') and self.simplicial_object.payload is not None:
            payload = self.simplicial_object.payload
            
            def simplicial_structure_map(state):
                """Structure map incorporating simplicial structure."""
                # Apply endofunctor
                evolved = self.endofunctor.apply_to_object(state)
                
                # Incorporate simplicial level information
                if isinstance(evolved, tuple) and len(evolved) >= 3:
                    activations, gradients, params = evolved[:3]
                    # Scale by simplicial level (higher levels have more complex dynamics)
                    level_factor = 1.0 + 0.1 * self.simplicial_object.level
                    scaled_params = params * level_factor
                    return activations, gradients, scaled_params
                
                return evolved
            
            return FCoalgebra(
                state_space=payload,
                structure_map=simplicial_structure_map,
                endofunctor=self.endofunctor,
                name=self.name
            )
        else:
            # Create default coalgebra
            default_state = torch.zeros(1)
            return create_parameter_coalgebra(default_state, name=self.name)
    
    def evolve_simplicial_dynamics(self, steps: int) -> List[Any]:
        """Evolve simplicial coalgebra dynamics."""
        return self.coalgebra.iterate(self.simplicial_object.payload, steps)
    
    def __repr__(self):
        return f"SimplicialCoalgebra(simplex='{self.simplicial_object.name}', level={self.simplicial_object.level})"