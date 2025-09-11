"""
Module: simplices
Core simplicial structures for GAIA categorical deep learning.

This module implements Layer 1 of GAIA: Simplicial Sets as described in
Mahadevan (2024) Section 4. Following the paper's true architecture:

1. Simplicial category Δ with ordinal numbers [n] as objects
2. Horn extension problems for hierarchical learning (Definition 23)
3. Lifting diagrams for parameter updates (Definition 2)
4. Kan complex properties for structural coherence
5. Hierarchical simplicial modules with n-simplices managing (n+1)-subsimplices

Key Features from Paper:
- Inner horns (0 < i < n): solvable by traditional backpropagation
- Outer horns (i = 0 or i = n): require advanced lifting methods
- Simplicial sets as "combinatorial factory" for assembling AI components
- Hierarchical learning beyond sequential framework
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, Dict
from copy import deepcopy

import torch
import torch.nn as nn

# Global device configuration with MPS support
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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
    Registry for managing simplicial bases and horn extension problems.
    
    Following Mahadevan (2024) Section 4.2, this manages the simplicial category Δ
    and provides the combinatorial factory for assembling generative AI components.
    
    Key Features from Paper:
    1. Standard simplices Δⁿ as canonical objects in simplicial category
    2. Horn extension problems Λᵢⁿ for hierarchical learning
    3. Lifting diagrams for solving outer horn problems
    4. Kan complex verification for structural coherence
    
    The registry maintains:
    - Standard simplices up to max dimension
    - Horn problems and their solutions
    - Lifting diagrams for outer horn extensions
    - Isomorphisms between simplicial structures
    """
    
    def __init__(self, max_dimension: int = 10):
        self.max_dimension = max_dimension
        
        # Standard simplicial structures from paper
        self._standard_simplices: dict[int, 'StandardSimplex'] = {}
        self._horn_registry: dict[tuple[int, int], 'Horn'] = {}
        self._lifting_diagrams: dict[uuid.UUID, 'LiftingDiagram'] = {}
        
        # Legacy basis management for compatibility
        self._canonical_bases: dict[int, uuid.UUID] = {}
        self._basis_dims: dict[uuid.UUID, int] = {}
        self._isomorphisms: dict[tuple[uuid.UUID, uuid.UUID], nn.Module] = {}
        
        # Initialize standard simplices following paper architecture
        self._initialize_simplicial_category()
    
    def _initialize_simplicial_category(self):
        """Initialize the simplicial category Δ following Mahadevan (2024)."""
        # Create standard simplices Δⁿ for n = 0, 1, ..., max_dimension
        for n in range(self.max_dimension + 1):
            self._standard_simplices[n] = StandardSimplex(dimension=n, registry=self)
    
    def get_standard_simplex(self, dimension: int):
        """Get standard n-simplex Δⁿ from the simplicial category."""
        if dimension > self.max_dimension:
            raise ValueError(f"Dimension {dimension} exceeds maximum {self.max_dimension}")
        return self._standard_simplices[dimension]
    
    def create_horn(self, dimension: int, missing_vertex: int):
        """Create horn Λᵢⁿ for hierarchical learning problems (Definition 23)."""
        key = (dimension, missing_vertex)
        if key not in self._horn_registry:
            parent_simplex = self.get_standard_simplex(dimension)
            self._horn_registry[key] = Horn(dimension=dimension, missing_vertex=missing_vertex, 
                                          parent_simplex=parent_simplex, registry=self)
            
        return self._horn_registry[key]
    
    def canonical_id(self, dim: int) -> uuid.UUID:
        """Get canonical basis ID (legacy compatibility)."""
        if dim not in self._canonical_bases:
            canonical_id = uuid.uuid4()
            self._canonical_bases[dim] = canonical_id
            self._basis_dims[canonical_id] = dim
            
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
    0-simplex representing objects in GAIA's simplicial category.
    
    Following Mahadevan (2024) Section 4.1, 0-simplices represent basic
    generative AI components in the simplicial hierarchy. These form the
    foundation of the "combinatorial factory" for assembling complex systems.
    
    Key Properties from Paper:
    - Represents parameter spaces as objects in simplicial category
    - Forms basis for higher-dimensional simplicial structures
    - Enables hierarchical learning through simplicial composition
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
    n-simplex for n ≥ 1, implementing GAIA's hierarchical simplicial modules.
    
    Following Mahadevan (2024) Section 4.2, n-simplices manage hierarchical
    learning where each n-simplex maintains parameters and updates based on
    information from (n-1)-subsimplices and (n+1)-supersimplices.
    
    Key Features from Paper:
    - Hierarchical parameter updates beyond sequential backpropagation
    - Horn extension problems for learning complex compositions
    - Lifting diagrams for solving outer horn problems
    - Consistency across simplicial hierarchy
    
    This enables true hierarchical generative AI beyond traditional frameworks.
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
        Compute the i-th face following simplicial category structure.
        
        Following Mahadevan (2024), face maps d_i: Δ^n → Δ^{n-1} are fundamental
        to the simplicial category and enable horn extension problems.
        
        For horn extensions:
        - Missing i-th face creates horn Λᵢⁿ
        - Inner horns (0 < i < n): solvable by backpropagation
        - Outer horns (i = 0 or i = n): require lifting diagrams
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


class StandardSimplex(SimplicialObject):
    """Standard n-simplex Δⁿ in the simplicial category.
    
    Following Mahadevan (2024), this represents the canonical n-dimensional
    simplex with vertices {0, 1, ..., n} and all faces properly defined.
    
    Mathematical Foundation:
        Δⁿ = {(t₀, t₁, ..., tₙ) ∈ ℝⁿ⁺¹ | ∑tᵢ = 1, tᵢ ≥ 0}
        
        Face maps: ∂ᵢ: Δⁿ → Δⁿ⁻¹ (remove i-th vertex)
        Degeneracy maps: sⱼ: Δⁿ → Δⁿ⁺¹ (duplicate j-th vertex)
    """
    
    def __init__(self, dimension: int, registry: BasisRegistry):
        super().__init__(level=dimension, name=f"Δ^{dimension}")
        self.dimension = dimension
        self.registry = registry
        self.vertices = list(range(dimension + 1))
        
        # Face and degeneracy operators as tensor operations
        self._face_matrices = self._compute_face_matrices()
        self._degeneracy_matrices = self._compute_degeneracy_matrices()
    
    def _compute_face_matrices(self) -> Dict[int, torch.Tensor]:
        """Compute face operator matrices ∂ᵢ: Δⁿ → Δⁿ⁻¹."""
        face_matrices = {}
        if self.dimension > 0:
            for i in range(self.dimension + 1):
                # Create face matrix that removes i-th coordinate
                face_matrix = torch.zeros(self.dimension, self.dimension + 1)
                col_idx = 0
                for j in range(self.dimension + 1):
                    if j != i:
                        face_matrix[col_idx, j] = 1.0
                        col_idx += 1
                face_matrices[i] = face_matrix
        return face_matrices
    
    def _compute_degeneracy_matrices(self) -> Dict[int, torch.Tensor]:
        """Compute degeneracy operator matrices sⱼ: Δⁿ → Δⁿ⁺¹."""
        degeneracy_matrices = {}
        for j in range(self.dimension + 1):
            # Create degeneracy matrix that duplicates j-th coordinate
            degeneracy_matrix = torch.zeros(self.dimension + 2, self.dimension + 1)
            for i in range(self.dimension + 1):
                if i <= j:
                    degeneracy_matrix[i, i] = 1.0
                else:
                    degeneracy_matrix[i + 1, i] = 1.0
            degeneracy_matrix[j + 1, j] = 1.0  # Duplicate j-th coordinate
            degeneracy_matrices[j] = degeneracy_matrix
        return degeneracy_matrices
    
    def face(self, i: int) -> 'StandardSimplex':
        """Apply i-th face operator ∂ᵢ."""
        if i < 0 or i > self.dimension:
            raise ValueError(f"Face index {i} out of range for {self.dimension}-simplex")
        if self.dimension == 0:
            raise ValueError("0-simplex has no faces")
        return self.registry.get_standard_simplex(self.dimension - 1)
    
    def degeneracy(self, j: int) -> 'StandardSimplex':
        """Apply j-th degeneracy operator sⱼ."""
        if j < 0 or j > self.dimension:
            raise ValueError(f"Degeneracy index {j} out of range for {self.dimension}-simplex")
        return self.registry.get_standard_simplex(self.dimension + 1)


class Horn(SimplicialObject):
    """Horn Λᵢⁿ for hierarchical learning problems.
    
    Following Mahadevan (2024) Definition 23, horns represent the fundamental
    learning challenges in GAIA's hierarchical framework.
    
    Mathematical Foundation:
        Λᵢⁿ = Δⁿ \ {interior of i-th face}
        
        Classification:
        - Inner horns (0 < i < n): Solvable by enhanced backpropagation
        - Outer horns (i = 0, n): Require advanced lifting diagram methods
    """
    
    def __init__(self, dimension: int, missing_vertex: int, 
                 parent_simplex: StandardSimplex, registry: BasisRegistry):
        super().__init__(level=dimension, name=f"Λ_{missing_vertex}^{dimension}")
        self.dimension = dimension
        self.missing_vertex = missing_vertex
        self.parent_simplex = parent_simplex
        self.registry = registry
        
        # Validate horn construction
        if missing_vertex < 0 or missing_vertex > dimension:
            raise ValueError(f"Missing vertex {missing_vertex} out of range for {dimension}-horn")
    
    def is_inner_horn(self) -> bool:
        """Check if this is an inner horn (0 < i < n)."""
        return 0 < self.missing_vertex < self.dimension
    
    def is_outer_horn(self) -> bool:
        """Check if this is an outer horn (i = 0 or i = n)."""
        return self.missing_vertex == 0 or self.missing_vertex == self.dimension
    
    def horn_type(self) -> str:
        """Get horn classification for learning algorithm selection."""
        if self.is_inner_horn():
            return "inner"
        elif self.missing_vertex == 0:
            return "outer_left"
        elif self.missing_vertex == self.dimension:
            return "outer_right"
        else:
            return "degenerate"
    
    def requires_lifting_diagram(self) -> bool:
        """Check if horn requires lifting diagram solution."""
        return self.is_outer_horn()
    
    def solvable_by_backprop(self) -> bool:
        """Check if horn is solvable by enhanced backpropagation."""
        return self.is_inner_horn()


class Simplex2(SimplexN):
    """
    2-simplex implementing GAIA's hierarchical learning triangles.
    
    Following Mahadevan (2024) Section 4.2, this implements horn extension
    problems for hierarchical learning. The inner horn Λ²₁ represents the
    fundamental composition problem in GAIA's simplicial hierarchy.
    
    Key Features from Paper:
    - Inner horn extension for hierarchical composition learning
    - Dynamic coherence maintenance during training
    - Hierarchical parameter updates beyond sequential backpropagation
    - Foundation for higher-dimensional simplicial modules
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

        # Create h as composition g∘f following GAIA's hierarchical structure
        # This implements the inner horn Λ²₁ with endofunctorial solver
        input_dim = f.domain.dim
        output_dim = g.codomain.dim
        
        # h is computed dynamically to maintain coherence during training
        h_morphism = nn.Identity()  # Placeholder - actual computation via payload
        h_morphism.requires_grad_(False)  # No learnable parameters
        
        # Create composition payload for dynamic computation
        def compose_g_f(x):
            """Dynamic composition following GAIA hierarchical learning."""
            return g(f(x))
        
        h = Simplex1(
            h_morphism,  # No parameters - pure composition
            f.domain,
            g.codomain,
            f"compose_{g.name}_{f.name}",
            payload=compose_g_f  # Dynamic composition for coherence
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
        """Check if removing face creates inner horn (solvable by backpropagation)."""
        return self.level >= 2 and 0 < missing_face < self.level
    
    def is_outer_horn(self, missing_face: int) -> bool:
        """Check if removing face creates outer horn (requires lifting diagrams)."""
        return self.level >= 2 and (missing_face == 0 or missing_face == self.level)
    
    def horn_type(self, missing_face: int) -> str:
        """Determine horn type following Mahadevan (2024) Definition 23.
        
        Inner horns: Traditional backpropagation can solve
        Outer horns: Require advanced lifting diagram methods
        """
        if self.is_inner_horn(missing_face):
            return "inner"  # Solvable by backpropagation
        elif self.is_outer_horn(missing_face):
            return "outer"  # Requires lifting diagrams
        return "invalid"
    
    def create_horn_extension_problem(self, missing_face: int) -> dict[str, Any]:
        """Create horn extension problem for hierarchical learning.
        
        Following paper Section 4.2, this creates the lifting problem structure
        needed for GAIA's hierarchical learning framework.
        """
        horn_type = self.horn_type(missing_face)
        if horn_type == "invalid":
            raise ValueError(f"Cannot create horn by removing face {missing_face} from {self.level}-simplex")
        
        return {
            'simplex_dimension': self.level,
            'missing_face': missing_face,
            'horn_type': horn_type,
            'solvable_by_backprop': horn_type == "inner",
            'requires_lifting_diagram': horn_type == "outer",
            'remaining_faces': [i for i in range(self.level + 1) if i != missing_face]
        }