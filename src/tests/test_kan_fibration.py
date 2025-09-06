"""
Tests for the Kan fibration module.
"""

import unittest
import torch
import torch.nn as nn

from gaia.core import (
    Simplex0, Simplex1, Simplex2, BasisRegistry,
    SimplicialFunctor
)
from gaia.train import (
    EndofunctorialSolver, UniversalLiftingSolver, MetricYonedaProxy
)

class TestKanFibration(unittest.TestCase):
    """Test cases for the Kan fibration module."""
    
    def test_endofunctorial_solver(self):
        """Test the endofunctorial solver."""
        # Create basis registry and functor
        basis = BasisRegistry()
        functor = SimplicialFunctor("test", basis)
        
        # Create 0-simplices
        A = Simplex0(2, "A", basis)
        B = Simplex0(3, "B", basis)
        C = Simplex0(1, "C", basis)
        
        # Create 1-simplices
        f = Simplex1(nn.Linear(2, 3), A, B, "f")
        g = Simplex1(nn.Linear(3, 1), B, C, "g")
        
        # Create 2-simplex
        s2 = Simplex2(f, g, "triangle")
        
        # Add to functor
        functor.add(s2)
        
        # Create solver
        solver = EndofunctorialSolver(functor, s2.id, lr=0.01)
        
        # Test properties
        self.assertEqual(solver.functor, functor)
        self.assertEqual(solver.s2_id, s2.id)
        self.assertEqual(solver.f, f)
        self.assertEqual(solver.g, g)
        
        # Test coherence loss
        x = torch.randn(5, 2)
        coherence = solver.coherence_loss(x)
        self.assertGreaterEqual(coherence.item(), 0)
        
        # Test step (minimal)
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)
        losses = solver.step(x, y)
        self.assertIn('task_loss', losses)
        self.assertIn('coherence_loss', losses)
        self.assertIn('total_loss', losses)
        
        # Test validation
        validation = solver.validate_horn_condition(x)
        self.assertGreaterEqual(validation, 0)
    
    def test_metric_yoneda_proxy(self):
        """Test the metric Yoneda proxy."""
        # Create proxy
        yoneda = MetricYonedaProxy(target_dim=4, num_probes=8, pretrain_steps=10)
        
        # Test properties
        self.assertEqual(yoneda.target_dim, 4)
        self.assertEqual(yoneda.probes.shape, (8, 4))
        
        # Test profile
        z = torch.randn(5, 4)
        profile = yoneda._profile(z)
        self.assertEqual(profile.shape, (5, 8, 1))
        
        # Test loss
        pred = torch.randn(5, 4)
        target = torch.randn(5, 4)
        loss = yoneda.loss(pred, target)
        self.assertGreaterEqual(loss.item(), 0)
        
        # Test update_probes
        new_data = torch.randn(10, 4)
        yoneda.update_probes(new_data)
    
    def test_universal_lifting_solver(self):
        """Test the universal lifting solver."""
        # Create basis registry
        basis = BasisRegistry()
        
        # Create 0-simplices
        A = Simplex0(2, "A", basis)
        B = Simplex0(3, "B", basis)
        C = Simplex0(4, "C", basis)
        
        # Create 1-simplices
        f = Simplex1(nn.Linear(2, 3), A, B, "f")
        k = Simplex1(nn.Linear(2, 4), A, C, "k")
        
        # Create Yoneda proxy
        yoneda = MetricYonedaProxy(target_dim=4, num_probes=8, pretrain_steps=10)
        
        # Create solver
        solver = UniversalLiftingSolver(f, k, yoneda, lr=0.01)
        
        # Test properties
        self.assertEqual(solver.f, f)
        self.assertEqual(solver.k, k)
        self.assertEqual(solver.yoneda, yoneda)
        self.assertEqual(solver.m_filler.domain, f.codomain)
        self.assertEqual(solver.m_filler.codomain, k.codomain)
        
        # Test solve (minimal)
        x = torch.randn(5, 2)
        m_fill = solver.solve(x, epochs=5)
        self.assertEqual(m_fill.domain, f.codomain)
        self.assertEqual(m_fill.codomain, k.codomain)

if __name__ == "__main__":
    unittest.main()