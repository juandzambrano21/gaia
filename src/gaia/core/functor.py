"""
Module: functor
Defines the SimplicialFunctor class representing a functor from Δᵒᵖ to Param.

Following Mahadevan (2024), this implements a complete simplicial functor
X: Δᵒᵖ → Param where every [n] ∈ Δ is mapped to X([n]) = <d_n>.

This implementation follows pure category theory principles:
1. The functor DEFINES structure, never discovers it
2. Simplicial identities are verified globally, not locally
3. Horn detection queries structure directly
4. Explicit structure definition methods are provided
5. Complete lifting problem validation
"""

import uuid
import logging
import warnings
from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, Tuple, Optional, Literal, Set, TypeVar, Union, Protocol, runtime_checkable, Any, Callable

# Set up logger
logger = logging.getLogger(__name__)

# Define map types as enum for type safety
class MapType(Enum):
    FACE = "face"
    DEGENERACY = "degeneracy"

class SimplicialError(Exception):
    """Base class for simplicial functor errors."""
    pass

class HornError(SimplicialError):
    """Error raised when a horn is encountered."""
    pass

class MapConflictError(SimplicialError):
    """Error raised when there's a conflict in map registration."""
    pass

class BasisClashError(SimplicialError):
    """Error raised when there's a basis clash."""
    pass

class FaceExpectationError(SimplicialError):
    """Error raised when a registered face doesn't match expected one."""
    pass

@runtime_checkable
class SimplicialObjectProtocol(Protocol):
    """Protocol defining the interface for simplicial objects."""
    id: uuid.UUID
    level: int
    name: str

# Type alias for simplicial objects
Simplex = TypeVar('Simplex', bound=SimplicialObjectProtocol)

from functools import wraps

def _invalidate_cache(method):
    """Decorator to automatically invalidate caches after any state change."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._horn_cache_valid = False
        return result
    return wrapper

class SimplicialFunctor:
    """
    GAIA Simplicial Functor: Δᵒᵖ → Neural-network parameters.
    
    Following Mahadevan (2024), this implements a complete simplicial functor
    where the functor defines all structural relationships explicitly.
    The functor is the sole arbiter of simplicial structure.
    
    Key Principles:
    1. Pure contravariant functor - defines structure, never discovers it
    2. Global simplicial identity verification ONLY
    3. Direct horn detection via structure queries
    4. Explicit structure definition methods
    5. Complete categorical coherence validation
    """

    def __init__(self, name: str, basis_registry):
        self.name = name
        self.basis_registry = basis_registry
        self.registry: Dict[uuid.UUID, Simplex] = {}
        self.graded_registry: Dict[int, Set[uuid.UUID]] = defaultdict(set)
        self.maps: Dict[Tuple[uuid.UUID, int, MapType], uuid.UUID] = {}
        # NO pending caches - structure is queried directly
        self._horn_cache: Dict[Tuple[int, str], List[Tuple[uuid.UUID, int]]] = {}
        self._horn_cache_valid = False

    def __getitem__(self, key: uuid.UUID) -> Simplex:
        """Get a simplex by its UUID."""
        return self.registry[key]

    @_invalidate_cache
    def add(self, simplex: Simplex) -> Set[uuid.UUID]:
        """
        Registers a simplex as an object in the target category.
        
        CRITICAL: This method does NOT infer, discover, or create any structural maps.
        It ONLY registers the simplex as an object. The functor defines structure
        through explicit define_face/define_degeneracy calls.
        
        NO LOCAL CHECKS - all validation is global via validate() method.
        """
        if simplex.id in self.registry:
            return {simplex.id}  # Already exists
            
        # Add the simplex to registry - PURE REGISTRATION ONLY
        self.registry[simplex.id] = simplex
        self.graded_registry[simplex.level].add(simplex.id)
        
        # Invalidate caches
        self._horn_cache_valid = False
        
        return {simplex.id}

    @_invalidate_cache
    def define_face(self, source_id: uuid.UUID, index: int, face_id: uuid.UUID) -> None:
        """
        Explicitly defines the i-th face map, d_i: [n] → [n-1].
        This method is the functor's action on the coface maps in Δᵒᵖ.
        
        This is how the functor DEFINES structure - the sole mechanism
        for creating face relationships in the categorical mapping.
        
        Args:
            source_id: The source simplex UUID
            index: The face index (0 ≤ index ≤ source.level)
            face_id: The target face simplex UUID
        """
        # Validation: Check if source and face exist
        if source_id not in self.registry:
            raise ValueError(f"Source simplex {source_id} not in registry")
        if face_id not in self.registry:
            raise ValueError(f"Face simplex {face_id} not in registry")
            
        source = self.registry[source_id]
        face = self.registry[face_id]
        
        # Validation: Check level consistency
        if face.level != source.level - 1:
            raise ValueError(
                f"Level inconsistency: face level {face.level} != source level - 1 ({source.level - 1})"
            )
            
        # Validation: Check index bounds
        if not (0 <= index <= source.level):
            raise ValueError(f"Face index {index} out of bounds for level {source.level} simplex")
            
        # Check for map conflicts
        map_key = (source_id, index, MapType.FACE)
        if map_key in self.maps and self.maps[map_key] != face_id:
            existing_face = self.registry[self.maps[map_key]]
            raise MapConflictError(
                f"Face map for ({source.name}, {index}) already maps to {existing_face.name}, "
                f"cannot remap to {face.name}"
            )
        
        # Register the map - THIS IS HOW STRUCTURE IS DEFINED
        self.maps[map_key] = face_id
        self._horn_cache_valid = False
        
        logger.debug(f"Defined face map: {source.name}.d_{index} → {face.name}")

    @_invalidate_cache
    def define_degeneracy(self, source_id: uuid.UUID, index: int, degen_id: uuid.UUID) -> None:
        """
        Explicitly defines the j-th degeneracy map, s_j: [n] → [n+1].
        This method is the functor's action on the co-degeneracy maps in Δᵒᵖ.
        
        This is how the functor DEFINES degeneracy structure.
        
        Args:
            source_id: The source simplex UUID
            index: The degeneracy index (0 ≤ index ≤ source.level)
            degen_id: The target degeneracy simplex UUID
        """
        # Validation: Check if source and degeneracy exist
        if source_id not in self.registry:
            raise ValueError(f"Source simplex {source_id} not in registry")
        if degen_id not in self.registry:
            raise ValueError(f"Degeneracy simplex {degen_id} not in registry")
            
        source = self.registry[source_id]
        degen = self.registry[degen_id]
        
        # Validation: Check level consistency
        if degen.level != source.level + 1:
            raise ValueError(
                f"Level inconsistency: degeneracy level {degen.level} != source level + 1 ({source.level + 1})"
            )
            
        # Validation: Check index bounds
        if not (0 <= index <= source.level):
            raise ValueError(f"Degeneracy index {index} out of bounds for level {source.level} simplex")
            
        # Check for map conflicts
        map_key = (source_id, index, MapType.DEGENERACY)
        if map_key in self.maps and self.maps[map_key] != degen_id:
            existing_degen = self.registry[self.maps[map_key]]
            raise MapConflictError(
                f"Degeneracy map for ({source.name}, {index}) already maps to {existing_degen.name}, "
                f"cannot remap to {degen.name}"
            )
        
        # Register the map - THIS IS HOW STRUCTURE IS DEFINED
        self.maps[map_key] = degen_id
        self._horn_cache_valid = False
        
        logger.debug(f"Defined degeneracy map: {source.name}.s_{index} → {degen.name}")

    def face(self, index: int, simplex_id: uuid.UUID) -> Simplex:
        """
        Get the face of a simplex at the given index.
        Queries the functor's defined structure directly.
        """
        key = (simplex_id, index, MapType.FACE)
        if key not in self.maps:
            simplex = self.registry[simplex_id]
            raise HornError(f"Face {index} of {simplex.name} not registered (horn detected)")
        
        return self.registry[self.maps[key]]
    
    def degeneracy(self, index: int, simplex_id: uuid.UUID) -> Simplex:
        """
        Get the degeneracy of a simplex at the given index.
        Queries the functor's defined structure directly.
        """
        key = (simplex_id, index, MapType.DEGENERACY)
        if key not in self.maps:
            simplex = self.registry[simplex_id]
            raise ValueError(f"Degeneracy {index} of {simplex.name} not registered")
        
        return self.registry[self.maps[key]]
    
    def find_horns(self, level: int, horn_type: Literal["inner", "outer", "both"] = "both") -> List[Tuple[uuid.UUID, int]]:
        """
        Locates all n-simplices where a face map is not defined.
        
         This method now queries the functor's structure directly
        via self.maps rather than relying on non-existent pending caches.
        A horn is identified structurally as a missing map in the functor.
        
        Args:
            level: The level of simplices to check for horns
            horn_type: Type of horns to find ("inner", "outer", or "both")
            
        Returns:
            List of (simplex_id, face_index) tuples representing horns
        """
        
        # Use cached results if valid
        cache_key = (level, horn_type)
        if self._horn_cache_valid and cache_key in self._horn_cache:
            cached_horns = self._horn_cache[cache_key]
            return cached_horns
        
        horns = []
        if level not in self.graded_registry:
            return []

        
        # DIRECT STRUCTURE QUERY - no pending caches
        for s_id in self.graded_registry[level]:
            simplex = self.registry[s_id]
            
            for i in range(simplex.level + 1):
                # A horn exists if the face map is not in the maps dictionary
                if (s_id, i, MapType.FACE) not in self.maps:
                    
                    # Check if it matches the requested horn type
                    if horn_type == "both":
                        horns.append((s_id, i))
                    elif horn_type == "inner" and 0 < i < simplex.level:
                        horns.append((s_id, i))
                    elif horn_type == "outer" and (i == 0 or i == simplex.level):
                        horns.append((s_id, i))
                    else:
                        logger.debug(f"🔍 HORN DETECTION: Horn at ({str(s_id)[:8]}..., {i}) doesn't match type '{horn_type}'")
        
        logger.debug(f"🔍 HORN DETECTION: Completed - found {len(horns)} total horns of type '{horn_type}'")
        
        # Update cache
        self._horn_cache[cache_key] = horns
        return horns

    def verify_simplicial_identities(self) -> Dict[str, Any]:
        """
        verification of simplicial identities on a  functor.
        
        This method validates that the functor preserves the simplicial identities
        as required by category theory. It assumes a complete structural definition
        and performs a clean, elegant validation of the three identity classes:
        
        1. Face-face identities: d_i ∘ d_j = d_{j-1} ∘ d_i for i < j
        2. Degeneracy-degeneracy identities: s_i ∘ s_j = s_{j+1} ∘ s_i for i ≤ j  
        3. Mixed identities: Various d_i ∘ s_j relations
        
        Returns:
            Dictionary with validation results for complete structure only
        """
        violations = []
        total_checks = 0
        
        # Only validate if we have a meaningful structure
        if not self.graded_registry or not self.maps:
            return {
                "valid": True,
                "violations": [],
                "total_checks": 0,
                "message": "Empty functor - trivially valid"
            }
        
        max_level = max(self.graded_registry.keys())
        
        # Validate face-face identities: d_i ∘ d_j = d_{j-1} ∘ d_i for i < j
        for level in range(2, max_level + 1):
            for s_id in self.graded_registry[level]:
                simplex = self.registry[s_id]
                
                for i in range(level):
                    for j in range(i + 1, level + 1):
                        total_checks += 1
                        
                        # Get the required maps - if any are missing, structure is incomplete
                        face_j_key = (s_id, j, MapType.FACE)
                        face_i_key = (s_id, i, MapType.FACE)
                        
                        if face_j_key not in self.maps or face_i_key not in self.maps:
                            violations.append({
                                "type": "incomplete_structure",
                                "simplex": simplex.name,
                                "identity": f"d_{i} ∘ d_{j}",
                                "reason": "Missing required face maps for identity verification"
                            })
                            continue
                        
                        # Compute d_i(d_j(s)) and d_{j-1}(d_i(s))
                        face_j_id = self.maps[face_j_key]
                        face_i_id = self.maps[face_i_key]
                        
                        left_key = (face_j_id, i, MapType.FACE)
                        right_key = (face_i_id, j-1, MapType.FACE)
                        
                        if left_key not in self.maps or right_key not in self.maps:
                            violations.append({
                                "type": "incomplete_structure",
                                "simplex": simplex.name,
                                "identity": f"d_{i} ∘ d_{j}",
                                "reason": "Missing composed face maps"
                            })
                            continue
                        
                        if self.maps[left_key] != self.maps[right_key]:
                            violations.append({
                                "type": "face_face_violation",
                                "simplex": simplex.name,
                                "identity": f"d_{i} ∘ d_{j} = d_{j-1} ∘ d_{i}",
                                "left_result": self.registry[self.maps[left_key]].name,
                                "right_result": self.registry[self.maps[right_key]].name
                            })
        
        # Validate degeneracy-degeneracy identities: s_i ∘ s_j = s_{j+1} ∘ s_i for i ≤ j
        for level in range(1, max_level + 1):
            for s_id in self.graded_registry[level]:
                simplex = self.registry[s_id]
                
                for i in range(level + 1):
                    for j in range(i, level + 1):
                        total_checks += 1
                        
                        deg_j_key = (s_id, j, MapType.DEGENERACY)
                        deg_i_key = (s_id, i, MapType.DEGENERACY)
                        
                        if deg_j_key not in self.maps or deg_i_key not in self.maps:
                            continue  # Degeneracies are optional
                        
                        deg_j_id = self.maps[deg_j_key]
                        deg_i_id = self.maps[deg_i_key]
                        
                        left_key = (deg_j_id, i, MapType.DEGENERACY)
                        right_key = (deg_i_id, j+1, MapType.DEGENERACY)
                        
                        if left_key in self.maps and right_key in self.maps:
                            if self.maps[left_key] != self.maps[right_key]:
                                violations.append({
                                    "type": "degeneracy_degeneracy_violation",
                                    "simplex": simplex.name,
                                    "identity": f"s_{i} ∘ s_{j} = s_{j+1} ∘ s_{i}",
                                    "left_result": self.registry[self.maps[left_key]].name,
                                    "right_result": self.registry[self.maps[right_key]].name
                                })
        
        # Validate mixed identities: d_i ∘ s_j relations
        for level in range(1, max_level + 1):
            for s_id in self.graded_registry[level]:
                simplex = self.registry[s_id]
                
                for j in range(level + 1):
                    deg_j_key = (s_id, j, MapType.DEGENERACY)
                    if deg_j_key not in self.maps:
                        continue
                        
                    deg_j_id = self.maps[deg_j_key]
                    
                    for i in range(level + 2):
                        total_checks += 1
                        face_i_key = (deg_j_id, i, MapType.FACE)
                        
                        if face_i_key not in self.maps:
                            continue
                        
                        if i == j or i == j + 1:
                            # Identity case: d_j ∘ s_j = id and d_{j+1} ∘ s_j = id
                            if self.maps[face_i_key] != s_id:
                                violations.append({
                                    "type": "mixed_identity_violation",
                                    "simplex": simplex.name,
                                    "identity": f"d_{i} ∘ s_{j} = id",
                                    "result": self.registry[self.maps[face_i_key]].name,
                                    "expected": simplex.name
                                })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "total_checks": total_checks,
            "message": "All simplicial identities verified" if len(violations) == 0 else f"{len(violations)} identity violations found"
        }

    def diagnose_partial_structure(self) -> Dict[str, Any]:
        """
        SEPARATE diagnostic method for analyzing incomplete structures.
        
        This method is explicitly for debugging and development, providing
        insights into the current state of a potentially incomplete functor.
        It does NOT attempt to validate identities on incomplete structures.
        
        Returns:
            Dictionary with diagnostic information about structure completeness
        """
        diagnostics = {
            "total_simplices": len(self.registry),
            "levels": dict(self.graded_registry),
            "total_maps": len(self.maps),
            "face_maps": len([k for k in self.maps.keys() if k[2] == MapType.FACE]),
            "degeneracy_maps": len([k for k in self.maps.keys() if k[2] == MapType.DEGENERACY]),
            "missing_faces": [],
            "orphaned_simplices": []
        }
        
        # Find missing face maps (potential horns)
        for level in self.graded_registry:
            for s_id in self.graded_registry[level]:
                simplex = self.registry[s_id]
                missing_faces = []
                
                for i in range(simplex.level + 1):
                    if (s_id, i, MapType.FACE) not in self.maps:
                        missing_faces.append(i)
                
                if missing_faces:
                    diagnostics["missing_faces"].append({
                        "simplex": simplex.name,
                        "level": simplex.level,
                        "missing_indices": missing_faces
                    })
        
        # Find orphaned simplices (no incoming or outgoing maps)
        for s_id, simplex in self.registry.items():
            has_incoming = any(target_id == s_id for target_id in self.maps.values())
            has_outgoing = any(source_id == s_id for source_id, _, _ in self.maps.keys())
            
            if not has_incoming and not has_outgoing and simplex.level > 0:
                diagnostics["orphaned_simplices"].append(simplex.name)
        
        return diagnostics

    def _check_global_basis_clashes(self) -> List[Dict[str, Any]]:
        """
        Global check for basis clashes across all registered simplices.
        This is the ONLY correct way to validate basis uniqueness.
        """
        clashes = []
        basis_map = {}
        
        for s_id, simplex in self.registry.items():
            if simplex.level == 0 and hasattr(simplex, 'basis_id') and hasattr(simplex, 'dim'):
                key = (simplex.dim, simplex.basis_id)
                if key in basis_map:
                    existing_simplex = self.registry[basis_map[key]]
                    clashes.append({
                        "type": "basis_clash",
                        "dimension": simplex.dim,
                        "basis_id": simplex.basis_id,
                        "conflicting_simplices": [existing_simplex.name, simplex.name]
                    })
                else:
                    basis_map[key] = s_id
                    
        return clashes

    def _validate_level_consistency(self) -> Dict[str, Any]:
        """
        SEPARATE method for basic level consistency checks.
        
        This method performs the simple type-consistency validation
        that was previously misnamed as "lifting problem validation".
        This is useful for debugging but is NOT lifting problem validation.
        """
        consistency_issues = []
        
        for (source_id, index, map_type), target_id in self.maps.items():
            source = self.registry[source_id]
            target = self.registry[target_id]
            
            if map_type == MapType.FACE:
                expected_level = source.level - 1
                if target.level != expected_level:
                    consistency_issues.append({
                        "type": "face_level_inconsistency",
                        "source": source.name,
                        "target": target.name,
                        "source_level": source.level,
                        "target_level": target.level,
                        "expected_level": expected_level,
                        "face_index": index
                    })
            
            elif map_type == MapType.DEGENERACY:
                expected_level = source.level + 1
                if target.level != expected_level:
                    consistency_issues.append({
                        "type": "degeneracy_level_inconsistency",
                        "source": source.name,
                        "target": target.name,
                        "source_level": source.level,
                        "target_level": target.level,
                        "expected_level": expected_level,
                        "degeneracy_index": index
                    })
        
        return {
            "is_consistent": len(consistency_issues) == 0,
            "issues": consistency_issues,
            "total_issues": len(consistency_issues)
        }

    def _check_basis_clash(self, simplex: Simplex) -> None:
        """
        Check for basis clashes when adding a simplex.
        
        A basis clash occurs when two 0-simplices have the same basis_id
        but different dimensions, which violates the categorical structure.
        """
        if not hasattr(simplex, 'level') or simplex.level != 0:
            return
            
        # For 0-simplices, check if there's already a simplex with the same basis_id
        # but different dimension
        if hasattr(simplex, 'basis_id'):
            for existing_id in self.graded_registry.get(0, set()):
                existing = self.registry[existing_id]
                if (hasattr(existing, 'basis_id') and 
                    existing.basis_id == simplex.basis_id and 
                    hasattr(existing, 'dim') and hasattr(simplex, 'dim') and
                    existing.dim != simplex.dim):
                    raise BasisClashError(
                        f"Basis clash: {existing.name} (dim={existing.dim}) and "
                        f"{simplex.name} (dim={simplex.dim}) have same basis_id {simplex.basis_id}"
                    )
    
    def _check_all_basis_clashes(self) -> None:
        """
        Check for basis clashes across all 0-simplices in the functor.
        """
        zero_simplices = [self.registry[sid] for sid in self.graded_registry.get(0, set())]
        
        for i, s1 in enumerate(zero_simplices):
            for s2 in zero_simplices[i+1:]:
                if (hasattr(s1, 'basis_id') and hasattr(s2, 'basis_id') and
                    s1.basis_id == s2.basis_id and 
                    hasattr(s1, 'dim') and hasattr(s2, 'dim') and
                    s1.dim != s2.dim):
                    raise BasisClashError(
                        f"Basis clash: {s1.name} (dim={s1.dim}) and "
                        f"{s2.name} (dim={s2.dim}) have same basis_id {s1.basis_id}"
                    )

    # Add these methods after the existing validate method, around line 736
    
    def create_object(self, dim: int, name: str, same_basis: bool = True) -> 'Simplex0':
        """
        Factory method to create a 0-simplex (object) with automatic registration.
        
        This is the ONLY correct way to create objects in the functor.
        The functor maintains complete control over structure.
        
        Args:
            dim: Dimension of the object
            name: Name of the object
            same_basis: Whether to use same basis (for basis registry)
            
        Returns:
            Created and registered 0-simplex
        """
        from .simplices import Simplex0  # Import here to avoid circular imports
        
        obj = Simplex0(dim, name, self.basis_registry, same_basis=same_basis)
        self.add(obj)
        
        logger.info(f"Created object: {name} (dim={dim})")
        return obj
    
    def create_morphism(self, network: Any, source: 'Simplex0', target: 'Simplex0', name: str) -> 'Simplex1':
        """
        Factory method to create a 1-simplex (morphism) with automatic registration.
        
        This method creates the morphism AND automatically defines its face maps
        according to the categorical structure.
        
        Args:
            network: Neural network implementing the morphism
            source: Source 0-simplex
            target: Target 0-simplex  
            name: Name of the morphism
            
        Returns:
            Created and registered 1-simplex with face maps defined
        """
        from .simplices import Simplex1  # Import here to avoid circular imports
        
        # Validate that source and target are registered
        if source.id not in self.registry:
            raise ValueError(f"Source {source.name} not registered in functor")
        if target.id not in self.registry:
            raise ValueError(f"Target {target.name} not registered in functor")
        
        # Create the morphism
        morphism = Simplex1(network, source, target, name)
        self.add(morphism)
        
        # AUTOMATICALLY define face maps (this is the key fix!)
        self.define_face(morphism.id, 0, target.id)  # d_0: f → target
        self.define_face(morphism.id, 1, source.id)  # d_1: f → source
        
        logger.info(f"Created morphism: {name} with automatic face maps")
        return morphism
    
    def create_triangle(self, f: 'Simplex1', g: 'Simplex1', name: str) -> 'Simplex2':
        """
        Factory method to create a 2-simplex (triangle) with automatic registration.
        
        This method creates the triangle AND automatically defines all its face maps
        according to the categorical structure.
        
        Args:
            f: First morphism (f: A → B)
            g: Second morphism (g: B → C)
            name: Name of the triangle
            
        Returns:
            Created and registered 2-simplex with all face maps defined
        """
        from .simplices import Simplex2  # Import here to avoid circular imports
        
        # Validate composition
        if not hasattr(f, 'codomain') or not hasattr(g, 'domain'):
            raise ValueError("Morphisms must have domain/codomain attributes")
        if f.codomain.id != g.domain.id:
            raise ValueError(f"Cannot compose {f.name} and {g.name}: codomain/domain mismatch")
        
        # Validate that morphisms are registered
        if f.id not in self.registry:
            raise ValueError(f"Morphism {f.name} not registered in functor")
        if g.id not in self.registry:
            raise ValueError(f"Morphism {g.name} not registered in functor")
        
        # Create the triangle
        triangle = Simplex2(f, g, name)
        self.add(triangle)
        
        # CRITICAL FIX: Register triangle.h before defining face maps:
        # The Simplex2 constructor creates h = g∘f, but it's not registered
        self.add(triangle.h)
        
        # AUTOMATICALLY define all face maps according to categorical structure
        # For a triangle with morphisms f: A → B, g: B → C:
        # d_0: triangle → g (remove first vertex A)
        # d_1: triangle → h=g∘f (remove middle vertex B) 
        # d_2: triangle → f (remove last vertex C)
        
        self.define_face(triangle.id, 0, g.id)        # d_0: remove A → g
        self.define_face(triangle.id, 1, triangle.h.id)  # d_1: remove B → h=g∘f
        self.define_face(triangle.id, 2, f.id)        # d_2: remove C → f
        
        logger.info(f"Created triangle: {name} with automatic face maps")
        return triangle

    
    def has_lift(self, f_id: uuid.UUID, p_id: uuid.UUID) -> List[uuid.UUID]:
        """
        Queries the functor to solve a lifting problem.
    
        This method poses the question: Given morphisms f: A -> B and p: E -> B, 
        which morphisms h: A -> E exist such that p ∘ h = f?
    
        Args:
            f_id: The UUID of the morphism f.
            p_id: The UUID of the morphism p, which must share a codomain with f.
    
        Returns:
            A list of UUIDs for all valid lifting morphisms 'h' found. An empty 
            list indicates no solution was found in the functor's structure.
        """
        f = self.registry.get(f_id)
        p = self.registry.get(p_id)
    
        if not (f and p and f.level == 1 and p.level == 1 and f.codomain.id == p.codomain.id):
            raise ValueError("Invalid lifting problem setup: f and p must be 1-simplices sharing a codomain.")
    
        A, E = f.domain, p.domain
        solutions = []
    
        # 1. Find all candidate diagonal morphisms h: A -> E
        candidate_morphisms = [s for s in self.registry.values() if s.level == 1 and hasattr(s, 'domain') and s.domain.id == A.id and hasattr(s, 'codomain') and s.codomain.id == E.id]
    
        for h_simplex in candidate_morphisms:
            # 2. Check if the diagram p ∘ h = f commutes by looking for a witness 2-simplex
            for s2_id in self.graded_registry.get(2, set()):
                triangle = self.registry[s2_id]
                if (hasattr(triangle, 'f') and hasattr(triangle, 'g') and hasattr(triangle, 'h') and
                    triangle.f.id == h_simplex.id and
                    triangle.g.id == p.id and
                    triangle.h.id == f.id):
                    solutions.append(h_simplex.id)
                    break  # Found witness for this h, move to next candidate
    
        return solutions
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary for serialization (PyTorch compatibility)"""
        return {
            'name': self.name,
            'registry': {str(k): {
                'id': str(v.id),
                'level': v.level,
                'name': v.name,
                'class_name': v.__class__.__name__
            } for k, v in self.registry.items()},
            'graded_registry': {k: [str(uuid_val) for uuid_val in v] for k, v in self.graded_registry.items()},
            'maps': {f"{k[0]}_{k[1]}_{k[2].value}": str(v) for k, v in self.maps.items()},
            '_horn_cache': {f"{k[0]}_{k[1]}": [(str(uuid_val), idx) for uuid_val, idx in v] for k, v in self._horn_cache.items()},
            '_horn_cache_valid': self._horn_cache_valid
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from dictionary (PyTorch compatibility)"""
        # Note: This is a simplified implementation for basic compatibility
        # Full reconstruction would require recreating all simplex objects
        self.name = state_dict.get('name', self.name)
        self._horn_cache_valid = state_dict.get('_horn_cache_valid', False)
        
        # For now, just preserve the basic structure
        # Full implementation would need to reconstruct all simplices
        logger.warning("SimplicialFunctor.load_state_dict: Partial implementation - structure not fully restored")
    
    def register_endofunctor_update(self, simplex_id: uuid.UUID, 
                                   old_state: Any, new_state: Any,
                                   endofunctor_name: str = "F") -> None:
        """
        Register endofunctor update for coalgebraic dynamics.
        
        This method implements the structure map γ: X → F(X) for F-coalgebras
        by recording state transitions under endofunctor application.
        
        Args:
            simplex_id: ID of simplex being updated
            old_state: Previous state
            new_state: New state after endofunctor application
            endofunctor_name: Name of endofunctor
        """
        if simplex_id not in self.registry:
            raise ValueError(f"Simplex {simplex_id} not registered in functor")
        
        simplex = self.registry[simplex_id]
        
        # Create endofunctor update record
        update_record = {
            "simplex_name": simplex.name,
            "simplex_level": simplex.level,
            "old_state": old_state,
            "new_state": new_state,
            "endofunctor": endofunctor_name,
            "timestamp": uuid.uuid4()  # Use UUID as timestamp
        }
        
        # Store in simplex payload if possible
        if hasattr(simplex, 'payload') and isinstance(simplex.payload, dict):
            if 'endofunctor_updates' not in simplex.payload:
                simplex.payload['endofunctor_updates'] = []
            simplex.payload['endofunctor_updates'].append(update_record)
        
        # Log the update
        logger.debug(f"Registered endofunctor update for {simplex.name}: {endofunctor_name}({old_state}) → {new_state}")
        
        # Invalidate caches since structure may have changed
        self._horn_cache_valid = False
    
    def get_endofunctor_trajectory(self, simplex_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Get trajectory of endofunctor updates for a simplex.
        
        This provides the coalgebraic evolution history γ^n(x₀).
        
        Args:
            simplex_id: ID of simplex
            
        Returns:
            List of endofunctor update records
        """
        if simplex_id not in self.registry:
            return []
        
        simplex = self.registry[simplex_id]
        
        if (hasattr(simplex, 'payload') and 
            isinstance(simplex.payload, dict) and 
            'endofunctor_updates' in simplex.payload):
            return simplex.payload['endofunctor_updates']
        
        return []
    
    def create_coalgebra_structure_map(self, simplex_id: uuid.UUID) -> Callable:
        """
        Create structure map γ: X → F(X) from endofunctor update history.
        
        This constructs the coalgebraic structure map from recorded updates.
        
        Args:
            simplex_id: ID of simplex
            
        Returns:
            Structure map function
        """
        trajectory = self.get_endofunctor_trajectory(simplex_id)
        
        def structure_map(state):
            """Structure map derived from update trajectory."""
            # Find most recent update matching current state
            for update in reversed(trajectory):
                if update['old_state'] == state:
                    return update['new_state']
            
            # Default: return state unchanged (identity endofunctor)
            return state
        
        return structure_map

  