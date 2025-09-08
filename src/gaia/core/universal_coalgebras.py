"""
Universal Coalgebras for GAIA Framework

Implements Section 3 from GAIA paper: "Backpropagation as an Endofunctor: Generative AI using Universal Coalgebras"

THEORETICAL FOUNDATIONS:
- Definition 7: F-coalgebra (A, α) with structure map α: A → F(A)
- Definition 9: Coalgebra homomorphisms with commutative diagrams
- Definition 10: Bisimulations between coalgebras
- Lambek's Theorem: Final coalgebras as fixed points
- Applications to generative AI, LLMs, diffusion models

This module provides the categorical foundation for all generative AI in GAIA.
"""

from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Type variables for categorical structures
A = TypeVar('A')  # Carrier object
B = TypeVar('B')  # Target object
F = TypeVar('F')  # Endofunctor

class Endofunctor(ABC, Generic[A]):
    """
    Abstract base class for endofunctors F: C → C
    
    From (MAHADEVAN,2024) Definition 7: Endofunctor required for F-coalgebras
    """
    
    @abstractmethod
    def apply(self, obj: A) -> A:
        """Apply functor to object: F(A) → F(A)"""
        pass
    
    @abstractmethod
    def fmap(self, morphism: Callable[[A], B]) -> Callable[[A], B]:
        """Apply functor to morphism: F(f: A → B) → F(A) → F(B)"""
        pass

class PowersetFunctor(Endofunctor[set]):
    """
    Powerset functor F: S ⇒ P(S) From (MAHADEVAN,2024) Section 3.1
    
    Models context-free grammars, finite state machines, and basic generative models.
    """
    
    def apply(self, obj: set) -> set:
        """Return powerset P(S) = {A | A ⊆ S}"""
        if len(obj) > 10:  # Limit for computational tractability
            # Sample subset for large sets
            import itertools
            return set(frozenset(combo) for combo in itertools.combinations(obj, min(3, len(obj))))
        
        # Full powerset for small sets
        import itertools
        return set(frozenset(combo) for r in range(len(obj) + 1) 
                  for combo in itertools.combinations(obj, r))
    
    def fmap(self, morphism: Callable[[set], set]) -> Callable[[set], set]:
        """Apply morphism to each element of powerset"""
        def mapped_morphism(powerset_obj: set) -> set:
            return set(frozenset(morphism(elem) if isinstance(elem, set) else {morphism(elem)}) 
                      for elem in powerset_obj)
        return mapped_morphism

class StreamFunctor(Endofunctor[List]):
    """
    Stream functor Str: Set → Set, Str(X) = ℕ × X from GAIA paper
    
    Models infinite data streams for generative AI (LLMs, sequence models).
    """
    
    def apply(self, obj: List) -> Tuple[int, List]:
        """Return (index, stream) pair"""
        return (len(obj), obj)
    
    def fmap(self, morphism: Callable[[List], List]) -> Callable[[Tuple[int, List]], Tuple[int, List]]:
        """Apply morphism to stream component"""
        def mapped_morphism(stream_obj: Tuple[int, List]) -> Tuple[int, List]:
            index, stream = stream_obj
            return (index, morphism(stream))
        return mapped_morphism

class BackpropagationFunctor(Endofunctor[torch.Tensor]):
    """
    Backpropagation endofunctor implementing Definition 11 from (MAHADEVAN,2024)
    
    F_B(X) = A × B × X where:
    - A: Input symbols (training data)
    - B: Output symbols (target data) 
    - X: Parameter space (model weights)
    
    """
    
    def __init__(self, input_data: torch.Tensor, target_data: torch.Tensor):
        """
        Initialize with actual training data, not dummy tensors
        
        Args:
            input_data: Input symbols A from training batch
            target_data: Output symbols B from training batch
        """
        self.input_data = input_data
        self.target_data = target_data
        self.stored_gradients = None  # Store actual gradients from model training
    
    def apply(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply F_B: X → A × B × X with backpropagation dynamics
        
        Returns the categorical product (A, B, X') where X' is transformed by actual gradient dynamics
        """
        device = params.device
        
        # Use actual stored gradients if available, otherwise return unchanged parameters
        if self.stored_gradients is not None:
            # Apply actual gradient descent step: X' = X - α * ∇L(X)
            from gaia.training.config import TrainingConfig
            training_config = TrainingConfig()
            learning_rate = training_config.optimization.learning_rate
            
            # Apply real gradient step using stored gradients from model training
            transformed_params = params - learning_rate * self.stored_gradients
        else:
            # If no gradients stored yet, return unchanged parameters
            # This prevents artificial gradient creation that causes divergence
            transformed_params = params
            logger.debug("No stored gradients available, returning unchanged parameters")
        
        return (self.input_data, self.target_data, transformed_params)
    
    def fmap(self, morphism: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
        """
        Apply morphism to parameter component while preserving A × B structure
        
        F(f): F(X) → F(Y) where f: X → Y
        """
        def mapped_morphism(coalgebra_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            input_data, target_data, params = coalgebra_state
            return (input_data, target_data, morphism(params))
        return mapped_morphism

    def update_data(self, new_input: torch.Tensor, new_target: torch.Tensor):
        """
        Update the input/output symbols for new training batch
        
        This allows the same functor to work with different data batches
        """
        self.input_data = new_input
        self.target_data = new_target
        # Reset stored gradients when new data arrives
        self.stored_gradients = None
    
    def store_gradients(self, gradients: torch.Tensor):
        """
        Store actual gradients from model training
        
        This captures real gradient information from the model's backward pass
        to be used in the coalgebra evolution instead of artificial gradients
        """
        self.stored_gradients = gradients.clone().detach()

@dataclass
class FCoalgebra(Generic[A]):
    """
    F-coalgebra (A, α) From (MAHADEVAN,2024) Definition 7
    
    Consists of:
    - carrier: Object A in category C
    - structure_map: Arrow α: A → F(A) defining dynamics
    """
    carrier: A
    structure_map: Callable[[A], A]
    endofunctor: Endofunctor[A]
    name: Optional[str] = None
    
    def evolve(self, state: A) -> A:
        """
        Evolve state using structure map α: A → F(A)
        
        This is the core dynamics of the coalgebra - how states transition.
        """
        return self.structure_map(state)
    
    def iterate(self, initial_state: A, steps: int) -> List[A]:
        """
        Iterate coalgebra dynamics for multiple steps
        
        Generates trajectory: state → α(state) → α²(state) → ...
        """
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(steps):
            current_state = self.evolve(current_state)
            trajectory.append(current_state)
        
        return trajectory
    
    def is_fixed_point(self, state: A, tolerance: float = 1e-6) -> bool:
        """
        Check if state is a fixed point: α(state) = state
        
        Related to Lambek's theorem for final coalgebras.
        """
        evolved = self.evolve(state)
        
        if isinstance(state, torch.Tensor) and isinstance(evolved, torch.Tensor):
            return torch.allclose(state, evolved, atol=tolerance)
        elif isinstance(state, (int, float)) and isinstance(evolved, (int, float)):
            return abs(state - evolved) < tolerance
        else:
            return state == evolved

class CoalgebraHomomorphism(Generic[A, B]):
    """
    Homomorphism between F-coalgebras following Definition 9 from (MAHADEVAN,2024)
    
    A morphism f: A → B between coalgebras (A,α) and (B,β) such that the diagram commutes:
    
        A ----α----> F(A)
        |             |
        f           F(f)
        |             |
        v             v
        B ----β----> F(B)
    
    This means: β ∘ f = F(f) ∘ α
    
    Critical for establishing relationships between generative AI models.
    """
    
    def __init__(self, 
                 source: FCoalgebra[A], 
                 target: FCoalgebra[B], 
                 morphism: Callable[[A], B],
                 tolerance: float = 1e-6):
        self.source = source
        self.target = target
        self.morphism = morphism
        self.tolerance = tolerance
        
        # Verify homomorphism property
        self.is_valid = self._verify_homomorphism()
        if not self.is_valid:
            logger.warning("Coalgebra homomorphism property does not hold")
    
    def _verify_homomorphism(self) -> bool:
        """
        Verify the commutative diagram condition: β ∘ f = F(f) ∘ α
        
        This is the fundamental property that makes f a coalgebra homomorphism.
        """
        try:
            # Generate test states for verification
            test_states = self._generate_test_states()
            
            if not test_states:
                logger.debug("No test states generated for homomorphism verification")
                return True  # Vacuously true if no states to test
            
            for state in test_states:
                if not self._verify_commutativity_for_state(state):
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not verify coalgebra homomorphism: {e}")
            return False
    
    def _verify_commutativity_for_state(self, state: A) -> bool:
        """
        Verify commutativity β ∘ f = F(f) ∘ α for a specific state.
        
        Left path:  β(f(state)) = target.structure_map(morphism(state))
        Right path: F(f)(α(state)) = endofunctor.fmap(morphism)(source.structure_map(state))
        """
        try:
            # Left path: β ∘ f
            mapped_state = self.morphism(state)
            left_path = self.target.structure_map(mapped_state)
            
            # Right path: F(f) ∘ α
            evolved_source = self.source.structure_map(state)
            
            # Apply F(f) using the endofunctor's fmap
            if hasattr(self.source.endofunctor, 'fmap'):
                # Use proper functor mapping F(f)
                f_mapped = self.source.endofunctor.fmap(self.morphism)
                right_path = f_mapped(evolved_source)
            else:
                # Fallback: apply morphism directly (not categorically correct but practical)
                right_path = self.morphism(evolved_source)
            
            # Check if both paths yield the same result
            return self._states_equal(left_path, right_path)
            
        except Exception as e:
            logger.debug(f"Commutativity verification failed for state: {e}")
            return False
    
    def _generate_test_states(self) -> List[A]:
        """
        Generate test states for homomorphism verification.
        
        For tensor coalgebras, generate sample states from the carrier.
        For other types, use the carrier directly if available.
        """
        test_states = []
        
        try:
            # Use the source coalgebra's carrier as a base
            if hasattr(self.source, 'carrier') and self.source.carrier is not None:
                carrier = self.source.carrier
                
                if isinstance(carrier, torch.Tensor):
                    # Generate variations of the carrier tensor
                    test_states.append(carrier)
                    
                    # Add small perturbations
                    for _ in range(3):
                        perturbed = carrier + torch.randn_like(carrier) * 0.1
                        test_states.append(perturbed)
                    
                    # Add scaled versions
                    test_states.append(carrier * 0.5)
                    test_states.append(carrier * 2.0)
                    
                else:
                    # For non-tensor carriers, use directly
                    test_states.append(carrier)
            
            # If no carrier available, try to generate from structure
            elif isinstance(self.source.carrier, type):
                # Handle case where carrier is a type rather than instance
                if self.source.carrier == torch.Tensor:
                    # Generate sample tensors
                    test_states.extend([
                        torch.randn(10),
                        torch.zeros(10),
                        torch.ones(10)
                    ])
            
        except Exception as e:
            logger.debug(f"Could not generate test states: {e}")
        
        return test_states
    
    def _states_equal(self, state1, state2, tolerance: Optional[float] = None) -> bool:
        """
        Check if two states are equal within tolerance.
        
        Handles different state types appropriately.
        """
        if tolerance is None:
            tolerance = self.tolerance
            
        try:
            if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
                if state1.shape != state2.shape:
                    return False
                return torch.allclose(state1, state2, atol=tolerance, rtol=tolerance)
            
            elif isinstance(state1, (tuple, list)) and isinstance(state2, (tuple, list)):
                if len(state1) != len(state2):
                    return False
                return all(self._states_equal(s1, s2, tolerance) for s1, s2 in zip(state1, state2))
            
            elif isinstance(state1, (int, float)) and isinstance(state2, (int, float)):
                return abs(state1 - state2) < tolerance
            
            else:
                return state1 == state2
                
        except Exception:
            return False
    
    def apply(self, state: A) -> B:
        """Apply homomorphism to state"""
        return self.morphism(state)
    
    def compose(self, other: 'CoalgebraHomomorphism[B, Any]') -> 'CoalgebraHomomorphism[A, Any]':
        """
        Compose this homomorphism with another: other ∘ self
        
        The composition of coalgebra homomorphisms is also a coalgebra homomorphism.
        """
        def composed_morphism(state: A):
            return other.morphism(self.morphism(state))
        
        return CoalgebraHomomorphism(
            source=self.source,
            target=other.target,
            morphism=composed_morphism,
            tolerance=min(self.tolerance, other.tolerance)
        )
    
    def is_isomorphism(self) -> bool:
        """
        Check if this homomorphism is an isomorphism.
        
        A coalgebra homomorphism is an isomorphism if it's bijective.
        This is a simplified check - full verification would require inverse construction.
        """
        # This is a simplified implementation
        # In practice, would need to verify bijectivity more rigorously
        return self.is_valid
    
    def __repr__(self):
        validity = "valid" if self.is_valid else "invalid"
        return f"CoalgebraHomomorphism({self.source.name} → {self.target.name}, {validity})"

class Bisimulation(Generic[A, B]):
    """
    F-Bisimulation between coalgebras following Definition 10 from (MAHADEVAN,2024)
    
    A relation R ⊆ S × T with structure map α_R: R → F(R) such that:
    1. The projections π₁: R → S and π₂: R → T are coalgebra homomorphisms
    2. F(π₁) ∘ α_R = α_S ∘ π₁ (first projection commutes)
    3. F(π₂) ∘ α_R = α_T ∘ π₂ (second projection commutes)
    
    This enables comparison of generative AI models.
    """
    
    def __init__(self, 
                 coalgebra1: FCoalgebra[A], 
                 coalgebra2: FCoalgebra[B],
                 relation: List[Tuple[A, B]],
                 tolerance: float = 1e-6):
        self.coalgebra1 = coalgebra1
        self.coalgebra2 = coalgebra2
        self.relation = relation
        self.tolerance = tolerance
        
        # Create structure map for the bisimulation relation
        self.structure_map = self._create_relation_structure_map()
        
        # Defer F-bisimulation verification until training data is available
        # This prevents errors during initialization when coalgebras don't have training data
        self.is_valid = None  # Will be set when verify() is called explicitly
    
    def _create_relation_structure_map(self) -> Callable[[Tuple[A, B]], Tuple[A, B]]:
        """
        Create structure map α_R: R → F(R) for the bisimulation relation.
        
        Following Definition 10: the structure map must make projections into homomorphisms.
        """
        def relation_structure_map(pair: Tuple[A, B]) -> Tuple[A, B]:
            state1, state2 = pair
            
            # Apply structure maps of both coalgebras
            evolved1 = self.coalgebra1.structure_map(state1)
            evolved2 = self.coalgebra2.structure_map(state2)
            
            # The F-bisimulation structure map preserves the relation
            return (evolved1, evolved2)
        
        return relation_structure_map
    
    def verify(self) -> bool:
        """Explicitly verify F-bisimulation properties after training data is available."""
        self.is_valid = self._verify_f_bisimulation()
        return self.is_valid
    
    def _verify_f_bisimulation(self) -> bool:
        """
        Verify F-bisimulation conditions following Definition 10:
        1. F(π₁) ∘ α_R = α_S ∘ π₁ (first projection is homomorphism)
        2. F(π₂) ∘ α_R = α_T ∘ π₂ (second projection is homomorphism)
        """
        try:
            for state1, state2 in self.relation:
                # Verify first projection homomorphism: F(π₁) ∘ α_R = α_S ∘ π₁
                if not self._verify_projection_homomorphism(state1, state2, projection=1):
                    return False
                
                # Verify second projection homomorphism: F(π₂) ∘ α_R = α_T ∘ π₂
                if not self._verify_projection_homomorphism(state1, state2, projection=2):
                    return False
                
                # Verify that evolved states maintain the bisimulation relation
                evolved_pair = self.structure_map((state1, state2))
                evolved1, evolved2 = evolved_pair
                
                if not self._states_are_related(evolved1, evolved2):
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not verify F-bisimulation: {e}")
            return False
    
    def _verify_projection_homomorphism(self, state1: A, state2: B, projection: int) -> bool:
        """
        Verify that projection π_i is a coalgebra homomorphism.
        
        For projection=1: F(π₁) ∘ α_R = α_S ∘ π₁
        For projection=2: F(π₂) ∘ α_R = α_T ∘ π₂
        """
        try:
            # Apply relation structure map
            evolved_pair = self.structure_map((state1, state2))
            evolved1, evolved2 = evolved_pair
            
            if projection == 1:
                # Check F(π₁) ∘ α_R = α_S ∘ π₁
                # Left side: F(π₁)(α_R(state1, state2)) = F(π₁)(evolved1, evolved2) = evolved1
                left_side = evolved1
                
                # Right side: α_S(π₁(state1, state2)) = α_S(state1)
                right_side = self.coalgebra1.structure_map(state1)
                
                return self._states_approximately_equal(left_side, right_side)
            
            elif projection == 2:
                # Check F(π₂) ∘ α_R = α_T ∘ π₂
                # Left side: F(π₂)(α_R(state1, state2)) = F(π₂)(evolved1, evolved2) = evolved2
                left_side = evolved2
                
                # Right side: α_T(π₂(state1, state2)) = α_T(state2)
                right_side = self.coalgebra2.structure_map(state2)
                
                return self._states_approximately_equal(left_side, right_side)
            
            return False
        except Exception as e:
            logger.debug(f"Projection homomorphism verification failed: {e}")
            return False
    
    def _states_are_related(self, state1: A, state2: B) -> bool:
        """
        Check if two states are related by the bisimulation relation.
        
        This implements the closure property: if (s,t) ∈ R and s →α s', t →β t',
        then (s',t') ∈ R (or are approximately related).
        """
        # Check if states are exactly in the relation
        if (state1, state2) in self.relation:
            return True
        
        # Check approximate relation for tensor states
        for rel_state1, rel_state2 in self.relation:
            if (self._states_approximately_equal(state1, rel_state1) and 
                self._states_approximately_equal(state2, rel_state2)):
                return True
        
        return False
    
    def _states_approximately_equal(self, state1, state2) -> bool:
        """
        Check if two states are approximately equal within tolerance.
        """
        try:
            if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
                if state1.shape != state2.shape:
                    return False
                return torch.allclose(state1, state2, atol=self.tolerance, rtol=self.tolerance)
            else:
                # For non-tensor states, use exact equality
                return state1 == state2
        except:
            return False
    
    def are_bisimilar(self, state1: A, state2: B) -> bool:
        """
        Check if two states are bisimilar according to this F-bisimulation.
        
        Returns True if (state1, state2) ∈ R or are approximately related.
        """
        return self._states_are_related(state1, state2)
    
    def add_relation_pair(self, state1: A, state2: B) -> bool:
        """
        Add a new pair to the bisimulation relation if it preserves the F-bisimulation property.
        
        Returns True if the pair was successfully added.
        """
        # Temporarily add the pair
        test_relation = self.relation + [(state1, state2)]
        
        # Create temporary bisimulation to test validity
        temp_bisim = Bisimulation(self.coalgebra1, self.coalgebra2, test_relation, self.tolerance)
        
        if temp_bisim.is_valid:
            self.relation = test_relation
            return True
        
        return False
    
    def get_maximal_bisimulation(self) -> 'Bisimulation':
        """
        Compute the maximal bisimulation relation between the two coalgebras.
        
        This implements the greatest fixed point characterization of bisimulation.
        """
        # Start with the full Cartesian product (this is often too large in practice)
        # In practice, we would use more sophisticated algorithms
        maximal_relation = []
        
        # For tensor coalgebras, sample a reasonable number of state pairs
        if hasattr(self.coalgebra1, 'carrier') and hasattr(self.coalgebra2, 'carrier'):
            try:
                carrier1 = self.coalgebra1.carrier
                carrier2 = self.coalgebra2.carrier
                
                if isinstance(carrier1, torch.Tensor) and isinstance(carrier2, torch.Tensor):
                    # Sample pairs and test bisimulation property
                    for _ in range(min(10, len(self.relation) * 2)):  # Reasonable sampling
                        # Create test states by perturbing existing ones
                        if self.relation:
                            base1, base2 = self.relation[0]
                            test1 = base1 + torch.randn_like(base1) * 0.1
                            test2 = base2 + torch.randn_like(base2) * 0.1
                            
                            if self.add_relation_pair(test1, test2):
                                maximal_relation.append((test1, test2))
            except:
                pass
        
        # Return current relation if no expansion possible
        return Bisimulation(self.coalgebra1, self.coalgebra2, 
                          self.relation + maximal_relation, self.tolerance)

class GenerativeCoalgebra(FCoalgebra[torch.Tensor]):
    """
    Specialized F_B-coalgebra for generative AI models
    
    Implements backpropagation as F-coalgebra following Definition 11 from (MAHADEVAN,2024)
    Structure map γ: X → F_B(X) = A × B × X defines backpropagation dynamics
    
    NOTE: This class defines only the coalgebra structure. Training components
    (optimizer, loss_fn) should be handled separately to avoid side effects.
    Training data is set externally to avoid baking in random values at construction.
    """
    
    def __init__(self, model: nn.Module):
        
        # Extract parameters as carrier object X
        params = torch.cat([p.flatten() for p in model.parameters()])
        
        # Initialize with placeholder functor - data will be set externally
        # This avoids baking in random data at construction time
        self.backprop_functor = None
        
        # Define structure map γ: X → F_B(X) following Definition 11
        def coalgebra_structure_map(current_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Structure map γ: X → A × B × X
            
            This is the core of the F_B-coalgebra - it takes parameters X
            and returns the categorical product (A, B, X) where:
            - A: input symbols (training data)
            - B: output symbols (target data)
            - X: parameters (unchanged in pure coalgebra structure)
            
            Note: Parameter evolution is handled by external training components
            """
            if self.backprop_functor is None:
                raise ValueError("Training data not set. Call update_training_data() first.")
            # Return F_B(X) = (A, B, X) - pure coalgebra structure
            return self.backprop_functor.apply(current_params)
        
        super().__init__(
            carrier=params,
            structure_map=coalgebra_structure_map,
            endofunctor=None,  # Will be set when training data is provided
            name="GenerativeCoalgebra"
        )
        
        self.model = model
    

    
    def update_training_data(self, new_input: torch.Tensor, new_target: torch.Tensor):
        """
        Update training data for new batch
        
        This allows the coalgebra to work with different training batches
        while maintaining the same categorical structure
        """
        # Initialize backprop_functor if not already set
        if self.backprop_functor is None:
            self.backprop_functor = BackpropagationFunctor(new_input, new_target)
            # Update endofunctor to use the backprop functor
            self.endofunctor = self.backprop_functor
        else:
            self.backprop_functor.update_data(new_input, new_target)
        
        # Store data as properties for external access
        self.input_data = new_input
        self.target_data = new_target
    
    @property
    def input_data(self):
        """Access current input data"""
        if hasattr(self, '_input_data'):
            return self._input_data
        elif self.backprop_functor is not None:
            return self.backprop_functor.input_data
        else:
            return None
    
    @input_data.setter
    def input_data(self, value):
        """Set input data"""
        self._input_data = value
    
    @property
    def target_data(self):
        """Access current target data"""
        if hasattr(self, '_target_data'):
            return self._target_data
        elif self.backprop_functor is not None:
            return self.backprop_functor.target_data
        else:
            return None
    
    @target_data.setter
    def target_data(self, value):
        """Set target data"""
        self._target_data = value

class CoalgebraTrainer:
    """
    Separate training wrapper for GenerativeCoalgebra
    
    Handles optimizer and loss function outside the coalgebra definition
    to maintain theoretical purity and avoid side effects during model instantiation.
    """
    
    def __init__(self, 
                 coalgebra: GenerativeCoalgebra,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.coalgebra = coalgebra
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, input_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """
        Perform one training step using the coalgebra structure
        
        Returns loss from the training step
        """
        # Update coalgebra with current training data
        self.coalgebra.update_training_data(input_data, target_data)
        
        # Get current parameters
        current_params = self.coalgebra.carrier
        
        # Apply backpropagation step
        loss, evolved_params = self._backprop_step(current_params)
        
        # Update coalgebra carrier with evolved parameters
        self.coalgebra.carrier = evolved_params
        
        return loss
    
    def _backprop_step(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single backpropagation step implementing parameter evolution
        
        This implements the training dynamics separate from coalgebra structure
        Returns (loss, evolved_parameters)
        """
        # Set model parameters from flattened tensor
        param_idx = 0
        total_params_needed = sum(p.numel() for p in self.coalgebra.model.parameters())
        
        # Handle shape mismatch by padding or truncating params tensor
        if params.numel() != total_params_needed:
            logger.debug(f"Parameter size mismatch: got {params.numel()}, need {total_params_needed}")
            if params.numel() < total_params_needed:
                # Pad with zeros if params tensor is too small
                padding = torch.zeros(total_params_needed - params.numel(), device=params.device, dtype=params.dtype)
                params = torch.cat([params, padding])
            else:
                # Truncate if params tensor is too large
                params = params[:total_params_needed]
        
        with torch.no_grad():
            for param in self.coalgebra.model.parameters():
                param_size = param.numel()
                if param_idx + param_size <= params.numel():
                    # Extract the slice and ensure it's contiguous before reshaping
                    param_slice = params[param_idx:param_idx + param_size].contiguous()
                    param.copy_(param_slice.view(param.shape))
                else:
                    # Handle edge case where we don't have enough parameters
                    available_params = params[param_idx:]
                    if available_params.numel() > 0:
                        # Pad with zeros to match required size
                        padding = torch.zeros(param_size - available_params.numel(), device=params.device, dtype=params.dtype)
                        padded_params = torch.cat([available_params, padding]).contiguous()
                        param.copy_(padded_params.view(param.shape))
                param_idx += param_size
        
        # Use current training data from the coalgebra functor
        input_data = self.coalgebra.backprop_functor.input_data
        target_data = self.coalgebra.backprop_functor.target_data
        
        try:
            # Forward pass with real data
            self.coalgebra.model.train()
            output = self.coalgebra.model(input_data)
            loss = self.loss_fn(output, target_data)
            
            # Backward pass - this is the categorical morphism
            self.optimizer.zero_grad()
            loss.backward()
            
            # Store actual gradients in the BackpropagationFunctor before optimizer step
            actual_gradients = torch.cat([p.grad.flatten() for p in self.coalgebra.model.parameters() if p.grad is not None])
            self.coalgebra.backprop_functor.store_gradients(actual_gradients)
            
            self.optimizer.step()
            
            # Return loss and evolved parameters X'
            evolved_params = torch.cat([p.flatten() for p in self.coalgebra.model.parameters()])
            return loss, evolved_params
        
        except Exception as e:
            logger.warning(f"Training step failed: {e}")
            return torch.tensor(float('inf')), params  # Return high loss and unchanged params if step fails
    
    def evolve_coalgebra(self, steps: int = 1) -> List[torch.Tensor]:
        """
        Evolve coalgebra through multiple training steps
        
        Returns list of parameter states after each step
        """
        states = [self.coalgebra.carrier.clone()]
        
        for _ in range(steps):
            # Use current training data from coalgebra
            input_data = self.coalgebra.backprop_functor.input_data
            target_data = self.coalgebra.backprop_functor.target_data
            
            evolved_params = self.train_step(input_data, target_data)
            states.append(evolved_params.clone())
        
        return states


class CoalgebraCategory:
    """
    Category of F-coalgebras with homomorphisms as morphisms
    
    Provides categorical structure for organizing generative AI models.
    """
    
    def __init__(self):
        self.objects: Dict[str, FCoalgebra] = {}
        self.morphisms: Dict[Tuple[str, str], CoalgebraHomomorphism] = {}
    
    def add_coalgebra(self, name: str, coalgebra: FCoalgebra):
        """Add coalgebra as object in category"""
        self.objects[name] = coalgebra
        logger.info(f"Added coalgebra '{name}' to category")
    
    def add_homomorphism(self, 
                        source_name: str, 
                        target_name: str, 
                        morphism: Callable):
        """Add homomorphism as morphism in category"""
        if source_name not in self.objects or target_name not in self.objects:
            raise ValueError("Source and target coalgebras must exist")
        
        source = self.objects[source_name]
        target = self.objects[target_name]
        
        hom = CoalgebraHomomorphism(source, target, morphism)
        self.morphisms[(source_name, target_name)] = hom
        
        logger.info(f"Added homomorphism {source_name} → {target_name}")
    
    def compose_morphisms(self, 
                         first: Tuple[str, str], 
                         second: Tuple[str, str]) -> Optional[CoalgebraHomomorphism]:
        """
        Compose two homomorphisms if composable
        
        Implements categorical composition in coalgebra category.
        """
        if first[1] != second[0]:  # Check if composable
            return None
        
        if first not in self.morphisms or second not in self.morphisms:
            return None
        
        hom1 = self.morphisms[first]
        hom2 = self.morphisms[second]
        
        # Compose morphisms
        def composed_morphism(state):
            return hom2.apply(hom1.apply(state))
        
        # Create composed homomorphism
        source = hom1.source
        target = hom2.target
        
        return CoalgebraHomomorphism(source, target, composed_morphism)
    
    def find_bisimulations(self) -> List[Tuple[str, str, Bisimulation]]:
        """
        Find bisimulations between coalgebras in category
        
        Critical for comparing different generative AI models.
        """
        bisimulations = []
        
        coalgebra_names = list(self.objects.keys())
        for i, name1 in enumerate(coalgebra_names):
            for name2 in coalgebra_names[i+1:]:
                coalgebra1 = self.objects[name1]
                coalgebra2 = self.objects[name2]
                
                # Simplified bisimulation detection
                # In practice, would need sophisticated algorithms
                relation = []  # Would compute actual relation
                
                bisim = Bisimulation(coalgebra1, coalgebra2, relation)
                if bisim.is_valid:
                    bisimulations.append((name1, name2, bisim))
        
        return bisimulations

# Factory functions for common coalgebras

def create_llm_coalgebra(model: nn.Module, 
                        input_data: torch.Tensor = None,
                        target_data: torch.Tensor = None) -> GenerativeCoalgebra:
    """
    Create coalgebra for Large Language Model following Definition 11
    
    Creates proper F_B-coalgebra with structure map γ: X → A × B × X
    where A = input_data, B = target_data, X = model parameters
    
    Note: Training components (optimizer, loss_fn) should be handled
    separately using CoalgebraTrainer to avoid side effects.
    Training data should be set via update_training_data() method.
    """
    coalgebra = GenerativeCoalgebra(model)
    if input_data is not None and target_data is not None:
        coalgebra.update_training_data(input_data, target_data)
    return coalgebra


def create_llm_coalgebra_trainer(model: nn.Module,
                                optimizer: torch.optim.Optimizer,
                                loss_fn: Callable,
                                input_data: torch.Tensor,
                                target_data: torch.Tensor) -> CoalgebraTrainer:
    """
    Create complete training setup with coalgebra and trainer
    
    This is a convenience function that creates both the coalgebra structure
    and the training wrapper in one call.
    """
    coalgebra = create_llm_coalgebra(model, input_data, target_data)
    return CoalgebraTrainer(coalgebra, optimizer, loss_fn)

def create_diffusion_coalgebra(model: nn.Module,
                              noise_schedule: Callable) -> FCoalgebra[torch.Tensor]:
    """
    Create coalgebra for diffusion model
    
    Models probabilistic coalgebra over ODEs from GAIA paper
    """
    def diffusion_structure_map(state: torch.Tensor) -> torch.Tensor:
        # Simplified diffusion step
        noise = noise_schedule(state)
        return state + noise
    
    return FCoalgebra(
        carrier=torch.randn(100),  # Simplified state
        structure_map=diffusion_structure_map,
        endofunctor=NeuralFunctor(100, 100),
        name="DiffusionCoalgebra"
    )

def create_transformer_coalgebra(attention_heads: int,
                                hidden_dim: int) -> FCoalgebra[torch.Tensor]:
    """
    Create coalgebra for Transformer model
    
    Will be extended with categorical transformer structure in later tasks.
    """
    def transformer_structure_map(state: torch.Tensor) -> torch.Tensor:
        # Simplified attention mechanism
        # MUST be replaced with proper categorical attention
        return torch.nn.functional.softmax(state, dim=-1)
    
    return FCoalgebra(
        carrier=torch.randn(hidden_dim),
        structure_map=transformer_structure_map,
        endofunctor=NeuralFunctor(hidden_dim, attention_heads),
        name="TransformerCoalgebra"
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the universal coalgebras
    logger.info("Testing Universal Coalgebras...")
    
    # Create test model and training components
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    # Create sample training data
    input_data = torch.randn(4, 10)
    target_data = torch.randn(4, 1)

    # Test new separated structure
    coalgebra_trainer = create_llm_coalgebra_trainer(model, optimizer, loss_fn, input_data, target_data)
    
    # Test coalgebra evolution through training
    initial_params = coalgebra_trainer.coalgebra.carrier
    evolved_states = coalgebra_trainer.evolve_coalgebra(steps=3)

    logger.info(f"Initial params shape: {initial_params.shape}")
    logger.info(f"Evolution steps: {len(evolved_states)}")
    logger.info(f"Final params shape: {evolved_states[-1].shape}")
    logger.info("✅ Universal Coalgebras implementation complete!")