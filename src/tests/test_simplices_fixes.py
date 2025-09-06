"""
Tests for the fixed simplices.py module.

These tests verify that the es work correctly:
1. Simplex0.__eq__ uses == for UUID comparison, not 'is'
2. Simplex0.__hash__ is based solely on basis_id
3. BasisRegistry provides separate methods for canonical vs fresh basis IDs
4. SimplexN.face/degeneracy uses memoization
5. SimplexN._check_simplicial_identities uses object identity
6. BasisRegistry.register_isomorphism checks dimensions
7. Simplex0.__deepcopy__ handles torch modules in payload
8. Simplex2._verify_pure_composition uses public interface
"""

import uuid
import copy
import torch
import torch.nn as nn
import pytest

from gaia.core import (
    BasisRegistry, Simplex0, Simplex1, Simplex2, SimplexN,
    DEVICE
)

# ────────────────────────────────────────────────────────────────────────────────
# 1. Simplex0.__eq__ and __hash__ Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplex0_equality_with_uuid_equality():
    """Test that Simplex0.__eq__ uses == for UUID comparison, not 'is'."""
    registry = BasisRegistry()
    
    # Create a UUID and its copy (different objects, same value)
    basis_id = uuid.uuid4()
    basis_id_copy = copy.deepcopy(basis_id)
    
    # Verify they're equal but not the same object
    assert basis_id == basis_id_copy
    assert basis_id is not basis_id_copy
    
    # Create two simplices with these UUIDs
    A = Simplex0(2, "A", registry, basis_id=basis_id)
    B = Simplex0(2, "B", registry, basis_id=basis_id_copy)
    
    # They should be equal because the UUIDs are equal
    assert A == B
    
    # They should hash to the same value
    assert hash(A) == hash(B)
    
    # They should work correctly in dictionaries and sets
    d = {A: "value"}
    assert B in d
    assert d[B] == "value"
    
    s = {A}
    assert B in s

def test_simplex0_hash_based_on_basis_id_only():
    """Test that Simplex0.__hash__ is based solely on basis_id."""
    registry = BasisRegistry()
    
    # Create a UUID
    basis_id = uuid.uuid4()
    
    # Create two simplices with the same basis_id but different levels
    # (This is a contrived example since Simplex0 always has level=0)
    A = Simplex0(2, "A", registry, basis_id=basis_id)
    object.__setattr__(A, 'level', 1)  # Artificially change level
    
    B = Simplex0(2, "B", registry, basis_id=basis_id)
    
    # They should still hash to the same value
    assert hash(A) == hash(B)

# ────────────────────────────────────────────────────────────────────────────────
# 2. BasisRegistry API Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_basis_registry_canonical_id():
    """Test that BasisRegistry.canonical_id returns the same ID for a given dimension."""
    registry = BasisRegistry()
    
    # Get canonical ID for dimension 2
    id1 = registry.canonical_id(2)
    id2 = registry.canonical_id(2)
    
    # They should be the same object
    assert id1 is id2
    
    # Different dimensions should have different canonical IDs
    id3 = registry.canonical_id(3)
    assert id1 != id3

def test_basis_registry_new_id():
    """Test that BasisRegistry.new_id always returns a fresh ID."""
    registry = BasisRegistry()
    
    # Get new IDs for dimension 2
    id1 = registry.new_id(2)
    id2 = registry.new_id(2)
    
    # They should be different
    assert id1 != id2
    
    # Get canonical ID for dimension 2
    id3 = registry.canonical_id(2)
    
    # It should be different from both new IDs
    assert id1 != id3
    assert id2 != id3

def test_basis_registry_get_id_default_changed():
    """Test that BasisRegistry.get_id defaults to same_basis=True."""
    registry = BasisRegistry()
    
    # Get ID with default parameters
    id1 = registry.get_id(2)
    
    # Get canonical ID
    id2 = registry.canonical_id(2)
    
    # They should be the same
    assert id1 is id2

def test_basis_registry_register_isomorphism_dimension_check():
    """Test that BasisRegistry.register_isomorphism checks dimensions."""
    registry = BasisRegistry()
    
    # Create bases for different dimensions
    basis_2d = registry.new_id(2)
    basis_3d = registry.new_id(3)
    
    # Create a linear map
    iso = nn.Linear(2, 3)
    
    # Should raise ValueError due to dimension mismatch
    with pytest.raises(ValueError):
        registry.register_isomorphism(basis_2d, basis_3d, iso)
    
    # Create bases for the same dimension
    basis_2d_2 = registry.new_id(2)
    
    # Should work fine
    iso2 = nn.Linear(2, 2)
    registry.register_isomorphism(basis_2d, basis_2d_2, iso2)
    
    # Get the isomorphism
    retrieved_iso = registry.get_isomorphism(basis_2d, basis_2d_2)
    assert retrieved_iso is not None

# ────────────────────────────────────────────────────────────────────────────────
# 3. SimplexN Memoization Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplexn_face_memoization():
    """Test that SimplexN.face uses memoization."""
    # Create a 2-simplex
    s = SimplexN(2, "test", (0, 1, 2))
    
    # Get faces
    f0_1 = s.face(0)
    f0_2 = s.face(0)
    
    # They should be the same object
    assert f0_1 is f0_2
    
    # Get another face
    f1 = s.face(1)
    
    # It should be different from f0
    assert f0_1 is not f1
    
    # Get faces of faces
    f0_f1 = f0_1.face(1)
    f1_f0 = f1.face(0)
    
    # They should be the same object due to simplicial identity
    assert f0_f1 is f1_f0

def test_simplexn_degeneracy_memoization():
    """Test that SimplexN.degeneracy uses memoization."""
    # Create a 1-simplex
    s = SimplexN(1, "test", (0, 1))
    
    # Get degeneracies
    d0_1 = s.degeneracy(0)
    d0_2 = s.degeneracy(0)
    
    # They should be the same object
    assert d0_1 is d0_2
    
    # Get another degeneracy
    d1 = s.degeneracy(1)
    
    # It should be different from d0
    assert d0_1 is not d1

def test_simplexn_check_simplicial_identities_uses_object_identity():
    """Test that SimplexN._check_simplicial_identities uses object identity."""
    # Create a 3-simplex
    s = SimplexN(3, "test", (0, 1, 2, 3))
    
    # Get faces in different orders
    f0_f1 = s.face(0).face(1)
    f1_f0 = s.face(1).face(0)
    
    # They should be the same object
    assert f0_f1 is f1_f0
    
    # Check that the cache is working
    assert 0 in s._face_cache
    assert 1 in s._face_cache
    assert 0 in s._face_cache[1]._face_cache
    assert 0 in s._face_cache[0]._face_cache

# ────────────────────────────────────────────────────────────────────────────────
# 4. Simplex0.__deepcopy__ Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplex0_deepcopy_with_torch_module():
    """Test that Simplex0.__deepcopy__ handles torch modules in payload."""
    registry = BasisRegistry()
    
    # Create a simplex with a torch module as payload
    module = nn.Linear(2, 3).to(DEVICE)
    A = Simplex0(2, "A", registry, payload=module)
    
    # Deepcopy it
    B = copy.deepcopy(A)
    
    # The payload should be a different object
    assert A.payload is not B.payload
    
    # But it should be on the same device
    assert A.payload.weight.device == B.payload.weight.device
    
    # And they should be equal
    assert torch.all(A.payload.weight == B.payload.weight)
    assert torch.all(A.payload.bias == B.payload.bias)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Simplex2._verify_pure_composition Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_simplex2_verify_pure_composition_uses_public_interface():
    """Test that Simplex2._verify_pure_composition uses the public interface."""
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
    
    # Test with a custom input
    x = torch.randn(1, 2)
    f_x = f(x)
    g_f_x = g(f_x)
    h_x = s2.h(x)
    
    # h(x) should equal g(f(x))
    assert torch.allclose(h_x, g_f_x)
    
    # Create a broken 2-simplex
    def broken_payload(x):
        return torch.zeros_like(g(f(x)))
        
    h_broken = Simplex1(
        nn.Identity(),
        A, C, "h_broken",
        payload=broken_payload
    )
    
    # This should raise an error during initialization
    with pytest.raises(ValueError):
        Simplex2(f, g, "broken_triangle", components=(f, h_broken, g))