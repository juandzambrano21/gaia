"""
Tests for the simplicial set module.
"""

import unittest
import torch
import torch.nn as nn

from gaia.core import (
    Simplex0, Simplex1, Simplex2, SimplexN, BasisRegistry,
    SimplicialFunctor, id_edge
)

class TestSimplicialSet(unittest.TestCase):
    """Test cases for the simplicial set module."""
    
    def test_basis_registry(self):
        """Test the basis registry."""
        registry = BasisRegistry()
        
        # Get basis IDs
        id1 = registry.get_id(2)
        id2 = registry.get_id(2)
        id3 = registry.get_id(3)
        
        # Same dimension should return same ID
        self.assertEqual(id1, id2)
        
        # Different dimensions should return different IDs
        self.assertNotEqual(id1, id3)
        
        # Test isomorphism registration
        iso = nn.Linear(2, 2)
        registry.register_isomorphism(id1, id3, iso)
        
        # Get registered isomorphism
        retrieved_iso = registry.get_isomorphism(id1, id3)
        self.assertIsNotNone(retrieved_iso)
        self.assertEqual(type(retrieved_iso), type(iso))
    
    def test_simplex0(self):
        """Test 0-simplices."""
        registry = BasisRegistry()
        
        # Create 0-simplices
        A = Simplex0(2, "A", registry)
        B = Simplex0(2, "B", registry)
        C = Simplex0(3, "C", registry)
        
        # Test equality (based on dimension)
        self.assertEqual(A, B)
        self.assertNotEqual(A, C)
        
        # Test deepcopy
        import copy
        A_copy = copy.deepcopy(A)
        self.assertEqual(A, A_copy)
        self.assertEqual(A.basis_id, A_copy.basis_id)
    
    def test_simplex1(self):
        """Test 1-simplices."""
        registry = BasisRegistry()
        
        # Create 0-simplices
        A = Simplex0(2, "A", registry)
        B = Simplex0(3, "B", registry)
        
        # Create 1-simplex
        f = Simplex1(nn.Linear(2, 3), A, B, "f")
        
        # Test properties
        self.assertEqual(f.domain, A)
        self.assertEqual(f.codomain, B)
        self.assertEqual(f.level, 1)
        
        # Test face maps
        f0 = f.face(0)
        f1 = f.face(1)
        self.assertEqual(f0, A)
        self.assertEqual(f1, B)
        
        # Test call
        x = torch.randn(5, 2)
        y = f(x)
        self.assertEqual(y.shape, (5, 3))
    
    def test_simplex2(self):
        """Test 2-simplices."""
        registry = BasisRegistry()
        
        # Create 0-simplices
        A = Simplex0(2, "A", registry)
        B = Simplex0(3, "B", registry)
        C = Simplex0(1, "C", registry)
        
        # Create 1-simplices
        f = Simplex1(nn.Linear(2, 3), A, B, "f")
        g = Simplex1(nn.Linear(3, 1), B, C, "g")
        
        # Create 2-simplex
        s2 = Simplex2(f, g, "triangle")
        
        # Test properties
        self.assertEqual(s2.f, f)
        self.assertEqual(s2.g, g)
        self.assertEqual(s2.h.domain, A)
        self.assertEqual(s2.h.codomain, C)
        
        # Test face maps
        s2_0 = s2.face(0)
        s2_1 = s2.face(1)
        s2_2 = s2.face(2)
        self.assertEqual(s2_0, f)
        self.assertEqual(s2_1, s2.h)
        self.assertEqual(s2_2, g)
        
        # Test horn classification
        self.assertTrue(s2.is_inner_horn(1))
        self.assertTrue(s2.is_outer_horn(0))
        self.assertTrue(s2.is_outer_horn(2))
        self.assertEqual(s2.horn_type(1), "inner")
        self.assertEqual(s2.horn_type(0), "outer")
        self.assertEqual(s2.horn_type(2), "outer")
    
    def test_simplicial_identities(self):
        """Test simplicial identities."""
        # Create a 3-simplex
        tetra = SimplexN(3, "tetra", (0, 1, 2, 3))
        
        # Test face-face identity: ∂ᵢ∂ⱼ = ∂ⱼ₋₁∂ᵢ for i < j
        for i in range(3):
            for j in range(i+1, 4):
                left = tetra.face(j).face(i)
                right = tetra.face(i).face(j-1)
                self.assertEqual(left.components, right.components)
        
        # Test degeneracy
        s0 = tetra.degeneracy(0)
        self.assertEqual(s0.level, 4)
        self.assertEqual(s0.components, (0, 0, 1, 2, 3))
    
    def test_simplicial_functor(self):
        """Test the simplicial functor."""
        registry = BasisRegistry()
        functor = SimplicialFunctor("test", registry)
        
        # Create 0-simplices
        A = Simplex0(2, "A", registry)
        B = Simplex0(3, "B", registry)
        C = Simplex0(1, "C", registry)
        
        # Create 1-simplices
        f = Simplex1(nn.Linear(2, 3), A, B, "f")
        g = Simplex1(nn.Linear(3, 1), B, C, "g")
        
        # Create 2-simplex
        s2 = Simplex2(f, g, "triangle")
        
        # Add to functor
        functor.add(s2)
        
        # Test registry
        self.assertIn(s2.id, functor.registry)
        self.assertIn(f.id, functor.registry)
        self.assertIn(g.id, functor.registry)
        self.assertIn(s2.h.id, functor.registry)
        self.assertIn(A.id, functor.registry)
        self.assertIn(B.id, functor.registry)
        self.assertIn(C.id, functor.registry)
        
        # Test face method
        self.assertEqual(functor.face(0, s2.id), f)
        self.assertEqual(functor.face(1, s2.id), s2.h)
        self.assertEqual(functor.face(2, s2.id), g)
        
        # Test find_horns
        inner_horns = functor.find_horns(2, "inner")
        outer_horns = functor.find_horns(2, "outer")
        self.assertGreaterEqual(len(inner_horns), 0)
        self.assertGreaterEqual(len(outer_horns), 0)
        
        # Test kan_condition_check
        kan_check = functor.kan_condition_check()
        self.assertIn("inner_horns", kan_check)
        self.assertIn("outer_horns", kan_check)
    
    def test_id_edge(self):
        """Test the id_edge function."""
        registry = BasisRegistry()
        functor = SimplicialFunctor("test", registry)
        
        # Create 0-simplex
        A = Simplex0(2, "A", registry)
        
        # Create identity edge
        id_A = id_edge(A, functor)
        
        # Test properties
        self.assertEqual(id_A.domain, A)
        self.assertEqual(id_A.codomain, A)
        self.assertIsInstance(id_A.morphism, nn.Identity)
        
        # Test in functor
        self.assertIn(id_A.id, functor.registry)
        
        # Test reuse
        id_A2 = id_edge(A, functor)
        self.assertEqual(id_A, id_A2)

if __name__ == "__main__":
    unittest.main()