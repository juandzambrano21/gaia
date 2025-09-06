"""
Comprehensive tests for the GAIA core components.

These tests verify that the fixed components work correctly:
1. BasisRegistry properly handles canonical bases
2. Simplex0 equality is based on basis_id, not just dimension
3. Simplex1 properly uses payload function for composition
4. Simplex2 ensures h is a pure composition
5. SimplicialFunctor is properly contravariant
"""

import uuid
import torch
import torch.nn as nn
import pytest

from gaia.core import (
    BasisRegistry, Simplex0, Simplex1, Simplex2, SimplexN,
    SimplicialFunctor, DEVICE
)

# ────────────────────────────────────────────────────────────────────────────────
# 1. BasisRegistry Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_basis_registry_canonical_bases():
    """Test that BasisRegistry properly handles canonical bases."""
    registry = BasisRegistry()
    
    # Get canonical basis for dimension 2
    basis_1 = registry.get_canonical_id(2)
    basis_2 = registry.get_canonical_id(2)
    
    # Same canonical basis should be returned
    assert basis_1 is basis_2
    
    # Non-canonical basis should be different
    basis_3 = registry.get_id(2, same_basis=False)
    assert basis_1 is not basis_3
    
    # Different dimensions should have different canonical bases
    basis_4 = registry.get_canonical_id(3)
    assert basis_1 is not basis_4

def test_basis_registry_isomorphisms():
    """Test that BasisRegistry properly handles isomorphisms."""
    registry = BasisRegistry()
    
    # Create two bases for dimension 2
    basis_a = registry.get_id(2)
    basis_b = registry.get_id(2)
    
    # Register an isomorphism
    iso = nn.Linear(2, 2)
    registry.register_isomorphism(basis_a, basis_b, iso)
    
    # Get the isomorphism
    retrieved_iso = registry.get_isomorphism(basis_a, basis_b)
    assert retrieved_iso is not None
    
    # Check that the inverse was also registered
    inverse_iso = registry.get_isomorphism(basis_b, basis_a)
    assert inverse_iso is not None

# ────────────────────────────────────────────────────────────────────────────────
# 2. Simplex0 Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplex0_equality():
    """Test that Simplex0 equality is based on basis_id, not just dimension."""
    registry = BasisRegistry()
    
    # Create two 0-simplices with the same dimension but different bases
    A = Simplex0(2, "A", registry)
    B = Simplex0(2, "B", registry)
    
    # They should not be equal
    assert A != B
    
    # Create two 0-simplices with the same basis
    C = Simplex0(3, "C", registry, same_basis=True)
    D = Simplex0(3, "D", registry, same_basis=True)
    
    # They should be equal
    assert C == D
    
    # Create a 0-simplex with a specific basis_id
    basis_id = uuid.uuid4()
    E = Simplex0(4, "E", registry, basis_id=basis_id)
    F = Simplex0(4, "F", registry, basis_id=basis_id)
    
    # They should be equal
    assert E == F

def test_simplex0_deepcopy():
    """Test that Simplex0 deepcopy preserves the basis_id."""
    registry = BasisRegistry()
    
    # Create a 0-simplex
    A = Simplex0(2, "A", registry)
    
    # Deepcopy it
    import copy
    B = copy.deepcopy(A)
    
    # They should be equal (same basis_id)
    assert A == B
    assert A.basis_id is B.basis_id

# ────────────────────────────────────────────────────────────────────────────────
# 3. Simplex1 Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplex1_call():
    """Test that Simplex1 properly uses payload function for composition."""
    registry = BasisRegistry()
    
    # Create two 0-simplices
    A = Simplex0(2, "A", registry)
    B = Simplex0(3, "B", registry)
    
    # Create a 1-simplex with a morphism
    f = Simplex1(nn.Linear(2, 3), A, B, "f")
    
    # Create a 1-simplex with a payload function
    def payload_func(x):
        return x * 2
        
    g = Simplex1(nn.Identity(), A, A, "g", payload=payload_func)
    
    # Test that f uses the morphism
    x = torch.randn(1, 2)
    assert torch.allclose(f(x), f.morphism(x))
    
    # Test that g uses the payload function
    assert torch.allclose(g(x), payload_func(x))
    assert not torch.allclose(g(x), g.morphism(x))

# ────────────────────────────────────────────────────────────────────────────────
# 4. Simplex2 Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplex2_pure_composition():
    """Test that Simplex2 ensures h is a pure composition."""
    registry = BasisRegistry()
    
    # Create three 0-simplices
    A = Simplex0(2, "A", registry)
    B = Simplex0(3, "B", registry)
    C = Simplex0(4, "C", registry)
    
    # Create two 1-simplices
    f = Simplex1(nn.Linear(2, 3), A, B, "f")
    g = Simplex1(nn.Linear(3, 4), B, C, "g")
    
    # Create a 2-simplex
    s2 = Simplex2(f, g, "triangle")
    
    # Test that h is a pure composition
    x = torch.randn(1, 2)
    assert torch.allclose(s2.h(x), g(f(x)))
    
    # Test that h has no learnable parameters
    assert not any(p.requires_grad for p in s2.h.morphism.parameters())
    
    # Test that h uses the payload function
    assert s2.h.payload is not None
    assert callable(s2.h.payload)

def test_simplex2_horn_classification():
    """Test that Simplex2 correctly classifies horns."""
    registry = BasisRegistry()
    
    # Create three 0-simplices
    A = Simplex0(2, "A", registry)
    B = Simplex0(3, "B", registry)
    C = Simplex0(4, "C", registry)
    
    # Create two 1-simplices
    f = Simplex1(nn.Linear(2, 3), A, B, "f")
    g = Simplex1(nn.Linear(3, 4), B, C, "g")
    
    # Create a 2-simplex
    s2 = Simplex2(f, g, "triangle")
    
    # Test horn classification
    assert s2.is_inner_horn(1)
    assert not s2.is_inner_horn(0)
    assert not s2.is_inner_horn(2)
    
    assert s2.is_outer_horn(0)
    assert s2.is_outer_horn(2)
    assert not s2.is_outer_horn(1)
    
    assert s2.horn_type(0) == "outer"
    assert s2.horn_type(1) == "inner"
    assert s2.horn_type(2) == "outer"

# ────────────────────────────────────────────────────────────────────────────────
# 5. SimplicialFunctor Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplicial_functor_contravariance():
    """Test that SimplicialFunctor is properly contravariant."""
    registry = BasisRegistry()
    functor = SimplicialFunctor("test", registry)
    
    # Create three 0-simplices
    A = Simplex0(2, "A", registry)
    B = Simplex0(3, "B", registry)
    C = Simplex0(4, "C", registry)
    
    # Create two 1-simplices
    f = Simplex1(nn.Linear(2, 3), A, B, "f")
    g = Simplex1(nn.Linear(3, 4), B, C, "g")
    
    # Create a 2-simplex
    s2 = Simplex2(f, g, "triangle")
    
    # Add the 2-simplex to the functor
    functor.add(s2)
    
    # Test that the functor contains all simplices
    assert s2.id in functor.registry
    assert f.id in functor.registry
    assert g.id in functor.registry
    assert s2.h.id in functor.registry
    assert A.id in functor.registry
    assert B.id in functor.registry
    assert C.id in functor.registry
    
    # Test that no face maps are registered yet
    assert (s2.id, 0, 'face') not in functor.maps
    assert (s2.id, 1, 'face') not in functor.maps
    assert (s2.id, 2, 'face') not in functor.maps
    
    # Register face maps
    functor.register_face(s2.id, 0, f.id)
    functor.register_face(s2.id, 1, s2.h.id)
    functor.register_face(s2.id, 2, g.id)
    
    # Test that face maps are now registered
    assert (s2.id, 0, 'face') in functor.maps
    assert (s2.id, 1, 'face') in functor.maps
    assert (s2.id, 2, 'face') in functor.maps
    
    # Test that face() returns the correct simplices
    assert functor.face(0, s2.id) == f
    assert functor.face(1, s2.id) == s2.h
    assert functor.face(2, s2.id) == g
    
    # Test that find_horns() returns no horns (all faces are registered)
    assert len(functor.find_horns(2)) == 0
    
    # Unregister a face map
    del functor.maps[(s2.id, 1, 'face')]
    
    # Test that find_horns() now returns one horn
    horns = functor.find_horns(2, "inner")
    assert len(horns) == 1
    assert horns[0] == (s2.id, 1)
    
    # Test that face() raises an error for the missing face
    with pytest.raises(ValueError):
        functor.face(1, s2.id)

def test_simplicial_functor_register_all_faces():
    """Test that register_all_faces correctly registers all faces."""
    registry = BasisRegistry()
    functor = SimplicialFunctor("test", registry)
    
    # Create three 0-simplices
    A = Simplex0(2, "A", registry)
    B = Simplex0(3, "B", registry)
    C = Simplex0(4, "C", registry)
    
    # Create two 1-simplices
    f = Simplex1(nn.Linear(2, 3), A, B, "f")
    g = Simplex1(nn.Linear(3, 4), B, C, "g")
    
    # Create a 2-simplex
    s2 = Simplex2(f, g, "triangle")
    
    # Add the 2-simplex to the functor
    functor.add(s2)
    
    # Register all faces
    face_ids = functor.register_all_faces(s2.id)
    
    # Test that all face maps are registered
    assert (s2.id, 0, 'face') in functor.maps
    assert (s2.id, 1, 'face') in functor.maps
    assert (s2.id, 2, 'face') in functor.maps
    
    # Test that face_ids contains all face IDs
    assert len(face_ids) == 3
    assert f.id in face_ids
    assert s2.h.id in face_ids
    assert g.id in face_ids