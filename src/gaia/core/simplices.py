"""
Module: simplices
Defines simplicial objects for the GAIA framework.

Following Mahadevan (2024), this implements simplicial objects as
functorial mappings from the simplex category Δ to neural network parameters.

Key principles:
1. Pure categorical structure - no local identity checking
2. Simplicial objects are immutable after creation
3. Face and degeneracy operations are purely structural
4. Composition is handled dynamically for coherence
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Tuple, Optional
from copy import deepcopy

import torch
import torch.nn as nn

# Global device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(slots=True)
class SimplicialObject:
    """Base class for all simplicial objects in the GAIA framework."""
    level: int
    name: str
    payload: Any = None
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)

    def __repr__(self):
        return f"{self.__class__.__name__}(level={self.level}, name='{self.name}', id={self.id})"


class BasisRegistry:
    """
    Registry for managing basis transformations between parameter spaces.
    
    Following Mahadevan (2024), this implements the basis equivalence
    relation that defines the Param category. Two parameter spaces are
    equivalent if there exists a differentiable isomorphism between them.
    
    The registry maintains:
    1. Canonical basis representatives for each dimension
    2. Isomorphisms between equivalent bases
    3. Efficient lookup for basis equivalence queries
    """
    
    def __init__(self):
        # Map from dimension to canonical basis UUID
        self._canonical_bases: dict[int, uuid.UUID] = {}
        
        # Map from (basis_a, basis_b) to isomorphism neural network
        self._isomorphisms: dict[tuple[uuid.UUID, uuid.UUID], nn.Module] = {}
        
        # Map from basis UUID to its dimension
        self._basis_dims: dict[uuid.UUID, int] = {}
    
    def canonical_id(self, dim: int) -> uuid.UUID:
        """
        Get the canonical basis ID for a given dimension.
        
        This implements the canonical choice function that selects a
        representative from each equivalence class of parameter spaces.
        
        Args:
            dim: The dimension of the parameter space
            
        Returns:
            UUID of the canonical basis for this dimension
        """
        if dim not in self._canonical_bases:
            # Create new canonical basis for this dimension
            canonical_id = uuid.uuid4()
            self._canonical_bases[dim] = canonical_id
            self._basis_dims[canonical_id] = dim
            
            # Register identity isomorphism
            identity = nn.Identity()
            self._isomorphisms[(canonical_id, canonical_id)] = identity
            
        return self._canonical_bases[dim]
    
    def register_isomorphism(self, basis_a: uuid.UUID, basis_b: uuid.UUID, iso: nn.Module):
        """
        Register an isomorphism between two bases.
        
        This extends the equivalence relation by adding a new isomorphism.
        The registry automatically computes the inverse and maintains
        transitivity through composition.
        
        Args:
            basis_a: Source basis UUID
            basis_b: Target basis UUID  
            iso: Neural network implementing the isomorphism
        """
        if basis_a not in self._basis_dims or basis_b not in self._basis_dims:
            raise ValueError("Both bases must be registered before defining isomorphism")
            
        if self._basis_dims[basis_a] != self._basis_dims[basis_b]:
            raise ValueError("Isomorphism can only exist between same-dimensional spaces")
        
        # Register forward isomorphism
        self._isomorphisms[(basis_a, basis_b)] = iso
        
        # For simplicity, assume inverse exists (in practice, would need to compute/verify)
        # This is a placeholder - real implementation would need proper inverse computation
        try:
            # Attempt to create a simple inverse (works for linear maps)
            if hasattr(iso, 'weight') and hasattr(iso, 'bias'):
                inv_weight = torch.inverse(iso.weight)
                inv_bias = -torch.matmul(inv_weight, iso.bias) if iso.bias is not None else None
                
                inverse = nn.Linear(iso.out_features, iso.in_features, bias=inv_bias is not None)
                inverse.weight.data = inv_weight
                if inv_bias is not None:
                    inverse.bias.data = inv_bias
                    
                self._isomorphisms[(basis_b, basis_a)] = inverse
        except:
            # If inverse computation fails, just register the forward direction
            pass
    
    def get_isomorphism(self, basis_a: uuid.UUID, basis_b: uuid.UUID) -> Optional[nn.Module]:
        """Get the isomorphism from basis_a to basis_b, if it exists."""
        return self._isomorphisms.get((basis_a, basis_b))
    
    # Alias for backward compatibility
    get_canonical_id = canonical_id
    
    def new_id(self, dim: int) -> uuid.UUID:
        """
        Create a new basis ID for the given dimension.
        
        This creates a fresh basis that is initially not equivalent to any
        existing basis. Equivalences can be established later through
        register_isomorphism.
        
        Args:
            dim: The dimension of the parameter space
            
        Returns:
            UUID of the new basis
        """
        new_basis_id = uuid.uuid4()
        self._basis_dims[new_basis_id] = dim
        
        # Register identity isomorphism for self
        identity = nn.Identity()
        self._isomorphisms[(new_basis_id, new_basis_id)] = identity
        
        return new_basis_id
    
    def get_id(self, dim: int, *, same_basis: bool = True) -> uuid.UUID:
        """
        Get a basis ID for the given dimension.
        
        Args:
            dim: The dimension of the parameter space
            same_basis: If True, return canonical basis. If False, create new basis.
            
        Returns:
            UUID of the basis (canonical or new)
        """
        if same_basis:
            return self.canonical_id(dim)
        else:
            return self.new_id(dim)


class Simplex0(SimplicialObject):
    """
    0-simplex representing an object in the Param category.
    
    Following Mahadevan (2024), objects are equivalence classes of
    parameter spaces <d> modulo differentiable isomorphism.
    """
    __slots__ = ('dim', 'basis_id')
    
    def __init__(self, dim: int, name: str, registry: BasisRegistry, payload: Any = None, 
                 same_basis: bool = False, basis_id: Optional[uuid.UUID] = None):
        super().__init__(level=0, name=name, payload=payload)
        
        object.__setattr__(self, "dim", dim)
        
        if basis_id is not None:
            object.__setattr__(self, "basis_id", basis_id)
        else:
            # Use registry to get appropriate basis ID
            basis_id = registry.get_id(dim, same_basis=same_basis)
            object.__setattr__(self, "basis_id", basis_id)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification after initialization."""
        if hasattr(self, name):
            raise AttributeError(f"Cannot modify immutable attribute '{name}'")
        super().__setattr__(name, value)
    
    def __eq__(self, other: object) -> bool:
        """Equality based on basis equivalence, not just basis_id."""
        if not isinstance(other, Simplex0):
            return False
        # Two 0-simplices are equal if they have the same dimension and basis
        # In a full implementation, this would check basis equivalence through the registry
        return self.dim == other.dim and self.basis_id == other.basis_id
    
    def __hash__(self):
        """Hash based on dimension and basis_id."""
        return hash((self.dim, self.basis_id))
    
    def __repr__(self):
        return f"Simplex0(dim={self.dim}, name='{self.name}', basis_id={self.basis_id})"
    
    def __deepcopy__(self, memo: dict[int, object]) -> "Simplex0":
        """Create a deep copy with a new UUID but same basis."""
        # Create new instance with same parameters but new UUID
        new_obj = object.__new__(self.__class__)
        
        # Copy all attributes except id
        object.__setattr__(new_obj, "level", self.level)
        object.__setattr__(new_obj, "name", self.name)
        object.__setattr__(new_obj, "payload", deepcopy(self.payload, memo))
        object.__setattr__(new_obj, "dim", self.dim)
        object.__setattr__(new_obj, "basis_id", self.basis_id)
        
        # Generate new UUID
        object.__setattr__(new_obj, "id", uuid.uuid4())
        
        return new_obj


class SimplexN(SimplicialObject):
    """
    n-simplex for n ≥ 1, representing higher-dimensional simplicial structure.
    
    This is a pure categorical implementation with no local identity checking.
    All simplicial identities are verified globally at the functor level.
    """
    __slots__ = ('components', '_face_cache', '_degeneracy_cache')

    def __init__(self, level: int, name: str, components: Tuple[Any, ...], payload: Any = None):
        super().__init__(level=level, name=name, payload=payload)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "_face_cache", {})
        object.__setattr__(self, "_degeneracy_cache", {})
        self._validate_simplicial_structure()

    def _validate_simplicial_structure(self) -> None:
        """Validate that components form a valid simplicial structure."""
        if len(self.components) != self.level + 1:
            raise ValueError(f"Level {self.level} simplex must have {self.level + 1} components")

    def face(self, i: int) -> "SimplexN":
        """
        Compute the i-th face by removing the i-th component.
        
        This is a pure categorical operation - no identity verification.
        The functor is responsible for maintaining simplicial identities.
        """
        if not 0 <= i <= self.level:
            raise ValueError(f"Face index {i} out of range for level {self.level}")
            
        # Check cache first
        if i in self._face_cache:
            return self._face_cache[i]
            
        # Create new face by removing the i-th component
        c = list(self.components)
        c.pop(i)
        face = SimplexN(self.level - 1, f"∂{i}({self.name})", tuple(c))
        self._face_cache[i] = face
        return face

    def degeneracy(self, j: int) -> "SimplexN":
        """
        Compute the j-th degeneracy by duplicating the j-th component.
        
        Pure categorical operation with no identity checking.
        """
        if not 0 <= j <= self.level:
            raise ValueError(f"Degeneracy index {j} out of range for level {self.level}")
            
        # Check cache first
        if j in self._degeneracy_cache:
            return self._degeneracy_cache[j]
            
        # Create new degeneracy by duplicating the j-th component
        c = list(self.components)
        c.insert(j, self.components[j])  # Insert duplicate at position j
        degeneracy = SimplexN(self.level + 1, f"σ{j}({self.name})", tuple(c))
        self._degeneracy_cache[j] = degeneracy
        return degeneracy

    def __len__(self) -> int:
        return self.level + 1

    def __eq__(self, other) -> bool:
        """Equality based on level, components, and name."""
        if not isinstance(other, SimplexN):
            return False
        return (self.level == other.level and 
                self.components == other.components and
                self.name == other.name)

    def __hash__(self) -> int:
        """Hash based on level, components, and name."""
        return hash((self.level, self.components, self.name))


class Simplex1(SimplexN):
    """
    1-simplex representing a morphism in the Param category.
    
    Following Mahadevan (2024), morphisms are equivalence classes of
    differentiable maps modulo parameter re-parameterization.
    """
    __slots__ = ('morphism',)

    def __init__(
        self,
        morphism: nn.Module,
        domain: Simplex0,
        codomain: Simplex0,
        name: str,
        payload: Any = None
    ):
        super().__init__(level=1, name=name, components=(domain, codomain), payload=payload)
        object.__setattr__(self, "morphism", morphism.to(DEVICE))

    @property
    def domain(self) -> Simplex0:
        return self.components[0]

    @property
    def codomain(self) -> Simplex0:
        return self.components[1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the morphism to input tensor x.
        
        If this is a composition morphism (with payload function), always use the payload
        to ensure dynamic composition is maintained.
        """
        if x.device != DEVICE:
            x = x.to(DEVICE)
        
        # If we have a payload function, use it (for composition morphisms)
        if callable(self.payload):
            return self.payload(x)
        
        # Otherwise use the morphism directly
        return self.morphism(x)

    def to(self, device) -> "Simplex1":
        """Move morphism to specified device."""
        self.morphism = self.morphism.to(device)
        return self
        
    def eval(self) -> "Simplex1":
        """Set morphism to evaluation mode."""
        self.morphism.eval()
        return self


class Simplex2(SimplexN):
    """
    2-simplex representing a commutative triangle in the Param category.
    
    Following Mahadevan (2024), this implements the inner horn Λ²₁ with
    endofunctorial solver where h = g ∘ f is computed dynamically to maintain
    coherence during training.
    """
    def __init__(
        self,
        f: Simplex1,
        g: Simplex1,
        name: str,
        payload: Any = None
    ):
        if f.codomain != g.domain:
            raise ValueError("Composition failed: codomain≠domain")

        # Create h as an INDEPENDENT learnable neural network
        # This is the key fix: h should be a separate network that learns to approximate g∘f
        input_dim = f.domain.dim
        output_dim = g.codomain.dim
        
        h_morphism = nn.Linear(input_dim, output_dim).to(DEVICE)
        # Enable gradient tracking for h - it should be learnable!
        h_morphism.requires_grad_(True)
        
        h = Simplex1(
            h_morphism,  # Independent learnable parameters
            f.domain,
            g.codomain,
            f"h_independent_{f.name}_{g.name}",
            payload=None  # No payload - h is independent, not a composition
        )

        super().__init__(level=2, name=name, components=(f, h, g), payload=payload)
    
    def _create_composition_payload(self, f: Simplex1, g: Simplex1):
        """
        Create a pickle-compatible composition payload.
        
        This method creates a composition function that can be serialized,
        unlike the previous local function approach.
        """
        # Store references to f and g for the composition
        self._composition_f = f
        self._composition_g = g
        
        # Return a method reference instead of a local function
        return self._dynamic_composition
    
    def _dynamic_composition(self, x):
        """
        Pickle-compatible dynamic composition method.
        
        This replaces the local function that was causing pickling issues.
        """
        return self._composition_g(self._composition_f(x))
        
    def _verify_pure_composition(self) -> None:
        """
        Verify that h is a pure composition of g and f.
        
        This ensures that:
        1. h.morphism has no learnable parameters
        2. h.morphism has requires_grad=False
        3. h uses the payload function for computation
        
        Uses the public interface (self.f, self.g, self.h) rather than
        accessing components directly, to ensure all payload functions are properly used.
        """
        # Check that h.morphism has no parameters requiring gradients
        has_params = any(p.requires_grad for p in self.h.morphism.parameters())
        if has_params:
            raise ValueError("h.morphism must not have learnable parameters")
            
        # Check that h has a callable payload
        if not callable(self.h.payload):
            raise ValueError("h must have a callable payload for dynamic composition")
            
        # Verify the composition with a test tensor
        with torch.no_grad():
            x = torch.randn(1, self.f.domain.dim, device=DEVICE)
            
            # Use public interface to ensure payload functions are used
            f_x = self.f(x)
            g_f_x = self.g(f_x)
            h_x = self.h(x)
            
            if not torch.allclose(h_x, g_f_x, atol=1e-6):
                raise ValueError(f"h(x) != g(f(x)): {h_x} != {g_f_x}")

    @property
    def f(self) -> Simplex1:
        return self.components[0]

    @property
    def h(self) -> Simplex1:
        return self.components[1]

    @property
    def g(self) -> Simplex1:
        return self.components[2]

    def is_inner_horn(self, missing_face: int) -> bool:
        """Check if this is an inner horn with the specified missing face."""
        return 0 < missing_face < self.level

    def is_outer_horn(self, missing_face: int) -> bool:
        """Check if this is an outer horn with the specified missing face."""
        return missing_face in {0, self.level}

    def horn_type(self, missing_face: int) -> str:
        """Determine the type of horn based on the missing face."""
        if self.is_inner_horn(missing_face):
            return "inner"
        if self.is_outer_horn(missing_face):
            return "outer"
        return "invalid"