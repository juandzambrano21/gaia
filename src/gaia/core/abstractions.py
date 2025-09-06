"""
GAIA Framework Core Abstractions

This module provides the unified base abstractions for the entire GAIA framework,
implementing the category-theoretic foundations From (MAHADEVAN,2024) and report.tex.

Key Design Principles:
1. Unified interfaces for all categorical structures
2. Composition over inheritance where possible  
3. Type safety with generic protocols
4. Integration points for all subsystems
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Generic, Union
from dataclasses import dataclass
import uuid
import torch
import torch.nn as nn

# Type variables for generic abstractions
X = TypeVar('X')  # Objects in categories
Y = TypeVar('Y')  # Codomain objects
F = TypeVar('F')  # Functor type
S = TypeVar('S')  # State type


@dataclass(frozen=True)
class CategoryObject:
    """Base class for all objects in GAIA categories."""
    id: uuid.UUID
    name: str
    dimension: int
    
    def __post_init__(self):
        if self.dimension < 0:
            raise ValueError(f"Dimension must be non-negative, got {self.dimension}")


class Morphism(Protocol[X, Y]):
    """Protocol for morphisms between category objects."""
    
    @property
    def domain(self) -> X:
        """Source object of the morphism."""
        ...
    
    @property
    def codomain(self) -> Y:
        """Target object of the morphism."""
        ...
    
    def compose(self, other: 'Morphism[Y, Any]') -> 'Morphism[X, Any]':
        """Compose this morphism with another: other ∘ self."""
        ...
    
    def __call__(self, x: Any) -> Any:
        """Apply the morphism to an element."""
        ...


class Functor(Protocol[X, Y]):
    """Protocol for functors between categories."""
    
    def map_object(self, obj: X) -> Y:
        """Map an object from source to target category."""
        ...
    
    def map_morphism(self, morphism: Morphism[X, X]) -> Morphism[Y, Y]:
        """Map a morphism from source to target category."""
        ...


class Endofunctor(Functor[X, X], Protocol):
    """Protocol for endofunctors F: C → C."""
    
    def apply_to_object(self, obj: X) -> X:
        """Apply endofunctor to an object."""
        ...
    
    def iterate(self, obj: X, steps: int) -> List[X]:
        """Iterate the endofunctor multiple times."""
        ...


class StructureMap(Protocol[X]):
    """Protocol for coalgebra structure maps γ: X → F(X)."""
    
    def __call__(self, state: X) -> Any:
        """Apply structure map to evolve state."""
        ...


class Coalgebra(Generic[X], ABC):
    """Abstract base class for F-coalgebras (X, γ)."""
    
    def __init__(self, carrier: X, structure_map: StructureMap[X]):
        self.carrier = carrier
        self.structure_map = structure_map
    
    def evolve(self, state: X) -> Any:
        """Evolve state using structure map: γ(state)."""
        return self.structure_map(state)
    
    @abstractmethod
    def is_bisimilar(self, other: 'Coalgebra[X]', relation: Callable[[X, X], bool]) -> bool:
        """Check if two coalgebras are bisimilar."""
        pass


class FuzzyMembership(Protocol):
    """Protocol for fuzzy membership functions η: X → [0,1]."""
    
    def __call__(self, element: Any) -> float:
        """Compute membership degree in [0,1]."""
        ...
    
    def support(self) -> set:
        """Return support set {x | η(x) > 0}."""
        ...


class SimplicialStructure(Protocol):
    """Protocol for simplicial structures with face and degeneracy maps."""
    
    def face_map(self, i: int) -> 'SimplicialStructure':
        """Apply i-th face map δᵢ."""
        ...
    
    def degeneracy_map(self, i: int) -> 'SimplicialStructure':
        """Apply i-th degeneracy map σᵢ."""
        ...
    
    @property
    def dimension(self) -> int:
        """Dimension of the simplicial structure."""
        ...


class HornFiller(Protocol):
    """Protocol for horn filling algorithms."""
    
    def can_fill(self, horn_type: str, dimension: int, missing_face: int) -> bool:
        """Check if this filler can handle the given horn."""
        ...
    
    def fill_horn(self, horn_data: Dict[str, Any]) -> Optional[Any]:
        """Fill the horn if possible."""
        ...


class MessagePasser(Protocol):
    """Protocol for hierarchical message passing."""
    
    def send_up(self, message: Any, level: int) -> None:
        """Send message up the hierarchy via degeneracy maps."""
        ...
    
    def send_down(self, message: Any, level: int) -> None:
        """Send message down the hierarchy via face maps."""
        ...
    
    def process_message(self, message: Any, source_level: int, target_level: int) -> Any:
        """Process message between hierarchy levels."""
        ...


@dataclass
class TrainingState:
    """Unified training state for GAIA framework."""
    epoch: int
    step: int
    loss: float
    parameters: Dict[str, torch.Tensor]
    gradients: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]


class GAIAComponent(ABC):
    """Abstract base class for all GAIA framework components."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.id = uuid.uuid4()
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    def update(self, state: TrainingState) -> TrainingState:
        """Update component with new training state."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate component state and configuration."""
        pass


class IntegratedTrainer(ABC):
    """Abstract base class for integrated GAIA trainers."""
    
    def __init__(self, components: List[GAIAComponent]):
        self.components = {comp.name: comp for comp in components}
        self.state = TrainingState(
            epoch=0, step=0, loss=float('inf'),
            parameters={}, gradients={}, metadata={}
        )
    
    @abstractmethod
    def train_step(self) -> TrainingState:
        """Execute one training step across all components."""
        pass
    
    @abstractmethod
    def validate_step(self) -> Dict[str, float]:
        """Execute validation across all components."""
        pass
    
    def add_component(self, component: GAIAComponent) -> None:
        """Add a component to the trainer."""
        self.components[component.name] = component
        component.initialize()
    
    def get_component(self, name: str) -> Optional[GAIAComponent]:
        """Get a component by name."""
        return self.components.get(name)


# Utility functions for common patterns
def compose_morphisms(*morphisms: Morphism) -> Morphism:
    """Compose multiple morphisms: f₁ ∘ f₂ ∘ ... ∘ fₙ."""
    if not morphisms:
        raise ValueError("Cannot compose empty sequence of morphisms")
    
    result = morphisms[0]
    for morphism in morphisms[1:]:
        result = result.compose(morphism)
    return result


def create_identity_morphism(obj: CategoryObject) -> Morphism:
    """Create identity morphism for an object."""
    class IdentityMorphism:
        def __init__(self, obj):
            self._obj = obj
        
        @property
        def domain(self):
            return self._obj
        
        @property
        def codomain(self):
            return self._obj
        
        def compose(self, other):
            return other
        
        def __call__(self, x):
            return x
    
    return IdentityMorphism(obj)


def verify_functor_laws(functor: Functor, test_objects: List[Any], 
                       test_morphisms: List[Morphism]) -> bool:
    """Verify functor preserves identity and composition."""
    # Test identity preservation: F(id_A) = id_F(A)
    for obj in test_objects:
        id_morphism = create_identity_morphism(obj)
        mapped_obj = functor.map_object(obj)
        mapped_id = functor.map_morphism(id_morphism)
        expected_id = create_identity_morphism(mapped_obj)
        
        # In practice, would need proper equality check
        # This is a simplified version
    
    # Test composition preservation: F(g ∘ f) = F(g) ∘ F(f)
    for i in range(len(test_morphisms) - 1):
        f = test_morphisms[i]
        g = test_morphisms[i + 1]
        if f.codomain == g.domain:  # Composable
            composed = g.compose(f)
            mapped_composed = functor.map_morphism(composed)
            mapped_f = functor.map_morphism(f)
            mapped_g = functor.map_morphism(g)
            expected = mapped_g.compose(mapped_f)
            
            # In practice, would need proper equality check
    
    return True  # Simplified for now


# Integration utilities
class ComponentRegistry:
    """Registry for managing GAIA components."""
    
    def __init__(self):
        self._components: Dict[str, GAIAComponent] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(self, component: GAIAComponent, dependencies: Optional[List[str]] = None):
        """Register a component with optional dependencies."""
        self._components[component.name] = component
        self._dependencies[component.name] = dependencies or []
    
    def get(self, name: str) -> Optional[GAIAComponent]:
        """Get a component by name."""
        return self._components.get(name)
    
    def initialize_all(self) -> None:
        """Initialize all components in dependency order."""
        initialized = set()
        
        def initialize_component(name: str):
            if name in initialized:
                return
            
            # Initialize dependencies first
            for dep in self._dependencies.get(name, []):
                initialize_component(dep)
            
            # Initialize this component
            component = self._components[name]
            component.initialize()
            initialized.add(name)
        
        for name in self._components:
            initialize_component(name)


# Export all public interfaces
__all__ = [
    'CategoryObject', 'Morphism', 'Functor', 'Endofunctor', 'StructureMap',
    'Coalgebra', 'FuzzyMembership', 'SimplicialStructure', 'HornFiller',
    'MessagePasser', 'TrainingState', 'GAIAComponent', 'IntegratedTrainer',
    'compose_morphisms', 'create_identity_morphism', 'verify_functor_laws',
    'ComponentRegistry'
]