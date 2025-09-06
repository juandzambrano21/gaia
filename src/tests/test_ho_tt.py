"""
Tests for the HoTT module.
"""

import unittest
from gaia.ho_tt import (
    Context, Type, Universe, DependentType, IdentityType, BaseType, InductiveType,
    Term, Variable, Lambda, Application, Pair, Refl, TypeChecker
)

class TestHoTT(unittest.TestCase):
    """Test cases for the HoTT module."""
    
    def test_universe_type(self):
        """Test universe types."""
        u0 = Universe(0)
        u1 = Universe(1)
        ctx = Context()
        
        self.assertTrue(u0.check_well_formed(ctx))
        self.assertTrue(u1.check_well_formed(ctx))
        self.assertTrue(u0.equals(Universe(0), ctx))
        self.assertFalse(u0.equals(u1, ctx))
    
    def test_dependent_type(self):
        """Test dependent types (Π and Σ)."""
        ctx = Context()
        nat = BaseType("Nat")
        
        # Π(x: Nat). Nat
        pi_type = DependentType("x", nat, nat, is_pi=True)
        self.assertTrue(pi_type.check_well_formed(ctx))
        
        # Σ(x: Nat). Nat
        sigma_type = DependentType("x", nat, nat, is_pi=False)
        self.assertTrue(sigma_type.check_well_formed(ctx))
        
        # Check equality
        self.assertTrue(pi_type.equals(DependentType("y", nat, nat, is_pi=True), ctx))
        self.assertFalse(pi_type.equals(sigma_type, ctx))
    
    def test_identity_type(self):
        """Test identity types."""
        ctx = Context()
        nat = BaseType("Nat")
        ctx = ctx.extend("x", nat).extend("y", nat)
        
        x = Variable("x")
        y = Variable("y")
        
        # Id_Nat(x, y)
        id_type = IdentityType(nat, x, y)
        self.assertTrue(id_type.check_well_formed(ctx))
        
        # Check equality
        self.assertTrue(id_type.equals(IdentityType(nat, x, y), ctx))
        self.assertFalse(id_type.equals(IdentityType(nat, x, x), ctx))
    
    def test_inductive_type(self):
        """Test inductive types."""
        ctx = Context()
        nat = BaseType("Nat")
        
        # Bool := true: Bool, false: Bool
        bool_type = InductiveType("Bool", [("true", BaseType("Bool")), ("false", BaseType("Bool"))])
        self.assertTrue(bool_type.check_well_formed(ctx))
        
        # Check equality
        same_bool = InductiveType("Bool", [("true", BaseType("Bool")), ("false", BaseType("Bool"))])
        diff_bool = InductiveType("Bool", [("true", BaseType("Bool"))])
        self.assertTrue(bool_type.equals(same_bool, ctx))
        self.assertFalse(bool_type.equals(diff_bool, ctx))
    
    def test_variable_term(self):
        """Test variable terms."""
        ctx = Context()
        nat = BaseType("Nat")
        ctx = ctx.extend("x", nat)
        
        x = Variable("x")
        y = Variable("y")
        
        self.assertEqual(x.check_type(ctx), nat)
        self.assertIsNone(y.check_type(ctx))
        
        self.assertTrue(x.equals(Variable("x"), ctx))
        self.assertFalse(x.equals(y, ctx))
    
    def test_lambda_term(self):
        """Test lambda terms."""
        ctx = Context()
        nat = BaseType("Nat")
        
        # λ(x: Nat). x
        id_fn = Lambda("x", nat, Variable("x"))
        
        # Check type
        id_type = id_fn.check_type(ctx)
        self.assertIsNotNone(id_type)
        self.assertTrue(isinstance(id_type, DependentType))
        self.assertTrue(id_type.is_pi)
        self.assertTrue(id_type.param_type.equals(nat, ctx))
        
        # Check equality
        same_id = Lambda("y", nat, Variable("y"))
        diff_id = Lambda("x", nat, Lambda("y", nat, Variable("y")))
        self.assertTrue(id_fn.equals(same_id, ctx))
        self.assertFalse(id_fn.equals(diff_id, ctx))
    
    def test_application_term(self):
        """Test application terms."""
        ctx = Context()
        nat = BaseType("Nat")
        ctx = ctx.extend("f", DependentType("x", nat, nat, is_pi=True))
        ctx = ctx.extend("a", nat)
        
        f = Variable("f")
        a = Variable("a")
        
        # f a
        app = Application(f, a)
        
        # Check type
        app_type = app.check_type(ctx)
        self.assertIsNotNone(app_type)
        self.assertTrue(app_type.equals(nat, ctx))
        
        # Check equality
        same_app = Application(Variable("f"), Variable("a"))
        diff_app = Application(Variable("f"), Variable("f"))
        self.assertTrue(app.equals(same_app, ctx))
        self.assertFalse(app.equals(diff_app, ctx))
    
    def test_refl_term(self):
        """Test reflexivity terms."""
        ctx = Context()
        nat = BaseType("Nat")
        ctx = ctx.extend("a", nat)
        
        a = Variable("a")
        
        # refl a
        refl_a = Refl(a)
        
        # Check type
        refl_type = refl_a.check_type(ctx)
        self.assertIsNotNone(refl_type)
        self.assertTrue(isinstance(refl_type, IdentityType))
        self.assertTrue(refl_type.base_type.equals(nat, ctx))
        self.assertTrue(refl_type.left.equals(a, ctx))
        self.assertTrue(refl_type.right.equals(a, ctx))
        
        # Check equality
        same_refl = Refl(Variable("a"))
        diff_refl = Refl(Variable("b"))
        self.assertTrue(refl_a.equals(same_refl, ctx))
        self.assertFalse(refl_a.equals(diff_refl, ctx))
    
    def test_type_checker(self):
        """Test the type checker."""
        checker = TypeChecker()
        ctx = Context()
        nat = BaseType("Nat")
        ctx = ctx.extend("a", nat)
        
        a = Variable("a")
        
        # Check type
        checked_type = checker.check_type(a, ctx)
        self.assertIsNotNone(checked_type)
        self.assertTrue(checked_type.equals(nat, ctx))
        
        # Check equality
        self.assertTrue(checker.check_equal(a, Variable("a"), ctx))
        self.assertFalse(checker.check_equal(a, Variable("b"), ctx))
        
        # Check derivation
        derivation = checker.get_derivation()
        self.assertIn("⊢ a : Nat", derivation)
        self.assertIn("⊢ a ≡ a", derivation)

if __name__ == "__main__":
    unittest.main()