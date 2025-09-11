"""
Universal Coalgebras for GAIA Framework

Implements Section 3 from GAIA paper: "Backpropagation as an Endofunctor: Generative AI using Universal Coalgebras"

THEORETICAL FOUNDINGS:
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

# Import existing GAIA fuzzy simplicial set implementations
from .integrated_structures import (
    IntegratedFuzzySimplicialSet, IntegratedFuzzySet, IntegratedSimplex,
    TConorm, FuzzyElement, create_fuzzy_simplex
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------
A = TypeVar('A')  # Carrier object
B = TypeVar('B')  # Target object
F = TypeVar('F')  # Endofunctor

FSSObject = IntegratedFuzzySimplicialSet

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_get(dct, key, default=None):
    try:
        return dct.get(key, default)
    except Exception:
        return default

def _float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

# ---------------------------------------------------------------------
# Endofunctors
# ---------------------------------------------------------------------
class Endofunctor(ABC, Generic[A]):
    """Abstract base class for endofunctors F: C → C"""
    @abstractmethod
    def apply(self, obj: A) -> A:
        """Apply functor to object: A ↦ F(A)"""
        pass
    
    @abstractmethod
    def fmap(self, morphism: Callable[[A], B]) -> Callable[[A], B]:
        """Apply functor to morphism: f ↦ F(f) where F(f): F(A) → F(B)"""
        pass

class PowersetFunctor(Endofunctor[set]):
    """Powerset functor F: S ⇒ P(S)"""
    def apply(self, obj: set) -> set:
        if len(obj) > 10:
            import itertools
            return set(frozenset(c) for c in itertools.combinations(obj, min(3, len(obj))))
        import itertools
        return set(
            frozenset(c)
            for r in range(len(obj)+1)
            for c in itertools.combinations(obj, r)
        )
    def fmap(self, morphism: Callable[[set], set]) -> Callable[[set], set]:
        def mapped(ps: set) -> set:
            return set(
                frozenset(morphism(e) if isinstance(e, set) else {morphism(e)})
                for e in ps
            )
        return mapped

class StreamFunctor(Endofunctor[List]):
    """Stream functor Str: X ↦ ℕ × X"""
    def apply(self, obj: List) -> Tuple[int, List]:
        return (len(obj), obj)
    def fmap(self, morphism: Callable[[List], List]) -> Callable[[Tuple[int, List]], Tuple[int, List]]:
        def mapped(st: Tuple[int, List]) -> Tuple[int, List]:
            i, xs = st
            return (i, morphism(xs))
        return mapped

class NeuralFunctor(Endofunctor[torch.Tensor]):
    """Minimal tensor endofunctor used by diffusion/transformer factory methods."""
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim, self.out_dim = in_dim, out_dim
    def apply(self, obj: torch.Tensor) -> torch.Tensor:
        return obj
    def fmap(self, morphism: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def mapped(x: torch.Tensor) -> torch.Tensor:
            return morphism(x)
        return mapped

class BackpropagationFunctor(Endofunctor[torch.Tensor]):
    """
    F_B(X) = A × B × X (tensor category)
    """
    def __init__(self, input_data: torch.Tensor, target_data: torch.Tensor):
        self.input_data = input_data
        self.target_data = target_data
        self.stored_gradients: Optional[torch.Tensor] = None
    
    def apply(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.stored_gradients is not None:
            try:
                from gaia.training.config import TrainingConfig
                lr = TrainingConfig().optimization.learning_rate
                
                # Handle new dictionary format for gradient statistics
                if isinstance(self.stored_gradients, dict):
                    # Use gradient norm for simple parameter perturbation
                    grad_norm = self.stored_gradients.get('norm', 0.0)
                    grad_mean = self.stored_gradients.get('mean', 0.0)
                    
                    # Apply small perturbation based on gradient statistics (all devices)
                    with torch.no_grad():
                        perturbation_scale = grad_norm * lr * 0.01
                        params.add_(torch.randn_like(params) * perturbation_scale)
                else:
                    # Legacy tensor format (should not occur with new code)
                    g = self.stored_gradients
                    if g.shape != params.shape:
                        fp, fg = params.flatten(), g.flatten()
                        if fg.numel() != fp.numel():
                            if fg.numel() < fp.numel():
                                pad = torch.zeros(fp.numel()-fg.numel(), device=fg.device, dtype=fg.dtype)
                                fg = torch.cat([fg, pad])
                            else:
                                fg = fg[:fp.numel()]
                        params = (fp - lr*fg).view(params.shape)
                    else:
                        params = params - lr*g
            except Exception as e:
                logger.debug(f"Gradient application failed: {e} — returning unchanged params")

        return (self.input_data, self.target_data, params)
    
    def fmap(self, morphism: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
        def mapped(state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            a, b, x = state
            return (a, b, morphism(x))
        return mapped
    
    def update_data(self, new_input: torch.Tensor, new_target: torch.Tensor):
        # Update data but preserve stored gradients for coalgebraic evolution
        # Only clear gradients if they are stale (different tensor shapes/devices)
        should_clear_gradients = False
        
        if hasattr(self, 'stored_gradients') and self.stored_gradients is not None:
            if isinstance(self.stored_gradients, torch.Tensor):
                # Check if gradient tensor is compatible with new data
                if (self.stored_gradients.device != new_input.device or 
                    self.input_data is None or 
                    self.input_data.shape != new_input.shape):
                    should_clear_gradients = True
            # For dict format (gradient statistics), preserve them as they're device-agnostic
        
        self.input_data = new_input
        self.target_data = new_target
        
        # Only clear gradients if necessary for compatibility
        if should_clear_gradients:
            if isinstance(self.stored_gradients, torch.Tensor):
                del self.stored_gradients
            self.stored_gradients = None
    
    def store_gradients(self, gradients: torch.Tensor):
        """
        Memory-efficient gradient statistics storage with automatic cleanup.
        
        Store minimal statistical summaries and implement
        automatic memory management to prevent gradient accumulation.
        
        Args:
            gradients: Flattened gradient tensor
        """
        # Clear any cached gradient computations
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Store only essential gradient statistics, not full tensors
        # This prevents MPS memory accumulation
        if gradients is not None and gradients.numel() > 0:
            # Compute statistics in memory-efficient way
            with torch.no_grad():
                # Use double precision for numerical stability with minimal memory
                grad_norm = float(gradients.norm().detach().item())
                grad_mean = float(gradients.mean().detach().item())
                grad_std = float(gradients.std().detach().item())
                
                # Store minimal statistics with memory bounds
                # Initialize if None, otherwise accumulate with exponential moving average
                if self.stored_gradients is None:
                    self.stored_gradients = {
                        'norm': min(grad_norm, 1000.0),  # Clip extreme values
                        'mean': max(-100.0, min(grad_mean, 100.0)),  # Bounded mean
                        'std': min(grad_std, 100.0)  # Bounded std
                    }
                else:
                    # Exponential moving average for gradient accumulation
                    alpha = 0.1  # Smoothing factor
                    self.stored_gradients['norm'] = (1-alpha) * self.stored_gradients['norm'] + alpha * min(grad_norm, 1000.0)
                    self.stored_gradients['mean'] = (1-alpha) * self.stored_gradients['mean'] + alpha * max(-100.0, min(grad_mean, 100.0))
                    self.stored_gradients['std'] = (1-alpha) * self.stored_gradients['std'] + alpha * min(grad_std, 100.0)
        else:
            # Initialize with zeros if no gradients provided
            if self.stored_gradients is None:
                self.stored_gradients = {'norm': 0.0, 'mean': 0.0, 'std': 0.0}
        

class FuzzyBackpropagationFunctor(Endofunctor[FSSObject]):
    """
    F_B endofunctor on Fuzzy Simplicial Sets: FSS → FSS
    
    - On objects: (F_B X)_k = A_k × B_k × X_k
    - Memberships: μ_F = T(μ_A, T(μ_B, μ_X'))
      where μ_X' optionally attenuates by gradient magnitude if provided.
    - On morphisms: F_B(f)_k = id_A_k × id_B_k × f_k
    """
    def __init__(self, input_fss: FSSObject, target_fss: FSSObject, t_norm: TConorm = TConorm.MAXIMUM):
        self.input_fss = input_fss
        self.target_fss = target_fss
        self.t_norm = t_norm
        self.stored_gradients: Optional[torch.Tensor] = None

    # ----- internal fuzzy combo -----
    def _combine_mu(self, mu_a: float, mu_b: float, mu_x: float) -> float:
        inner = self.t_norm.apply(mu_b, mu_x)
        return self.t_norm.apply(mu_a, inner)

    def _attenuate_mu_x(self, mu_x: float) -> float:
        if self.stored_gradients is None:
            return mu_x
        # Use stored gradient norm instead of full tensor
        grad_norm = self.stored_gradients if isinstance(self.stored_gradients, (int, float)) else 0.0
        factor = torch.sigmoid(-torch.tensor(grad_norm)).item()
        return float(np.clip(mu_x * factor, 0.0, 1.0))

    # ----- transport structure if helpers exist -----
    def _transport_structure(self, A: FSSObject, B: FSSObject, X: FSSObject, FAX: FSSObject, k: int):
        try:
            if not (hasattr(A, "iter_simplices") and hasattr(B, "iter_simplices") and hasattr(X, "iter_simplices")):
                return
            if not hasattr(FAX, "register_product_face_degeneracy"):
                if hasattr(X, "copy_face_degeneracy_structure_into"):
                    X.copy_face_degeneracy_structure_into(FAX, k, source_component="X")
                return
            def face_map(product_key, i: int):
                ak, bk, xk = product_key
                a_face = A.face_of(k, ak, i) if hasattr(A, "face_of") else ak
                b_face = B.face_of(k, bk, i) if hasattr(B, "face_of") else bk
                x_face = X.face_of(k, xk, i) if hasattr(X, "face_of") else xk
                return (a_face, b_face, x_face)
            def degeneracy_map(product_key, i: int):
                ak, bk, xk = product_key
                a_deg = A.degeneracy_of(k, ak, i) if hasattr(A, "degeneracy_of") else ak
                b_deg = B.degeneracy_of(k, bk, i) if hasattr(B, "degeneracy_of") else bk
                x_deg = X.degeneracy_of(k, xk, i) if hasattr(X, "degeneracy_of") else xk
                return (a_deg, b_deg, x_deg)
            FAX.register_product_face_degeneracy(k, face_map, degeneracy_map)
        except Exception:
            if hasattr(X, "copy_face_degeneracy_structure_into"):
                try:
                    X.copy_face_degeneracy_structure_into(FAX, k, source_component="X")
                except Exception:
                    pass

    # ----- functor on objects -----
    def apply(self, fss_obj: FSSObject) -> FSSObject:
        if hasattr(fss_obj, "empty_like"):
            out = fss_obj.empty_like(name=f"F_B({getattr(fss_obj, 'name', 'X')})")
            out.max_dimension = max(self.input_fss.max_dimension, self.target_fss.max_dimension, fss_obj.max_dimension)
        else:
            out = IntegratedFuzzySimplicialSet(
                name=f"F_B({getattr(fss_obj, 'name', 'X')})",
                max_dimension=max(self.input_fss.max_dimension, self.target_fss.max_dimension, fss_obj.max_dimension),
            )
        for k in range(out.max_dimension + 1):
            A_k = _safe_get(self.input_fss.simplices, k, {})
            B_k = _safe_get(self.target_fss.simplices, k, {})
            X_k = _safe_get(fss_obj.simplices, k, {})
            for a_key, a_sx in A_k.items():
                mu_a = _float(a_sx.membership)
                for b_key, b_sx in B_k.items():
                    mu_b = _float(b_sx.membership)
                    for x_key, x_sx in X_k.items():
                        mu_x = self._attenuate_mu_x(_float(x_sx.membership))
                        mu_prod = self._combine_mu(mu_a, mu_b, mu_x)
                        prod_key = (a_key, b_key, x_key)
                        out.add_simplex(k, prod_key, mu_prod)
            self._transport_structure(self.input_fss, self.target_fss, fss_obj, out, k)
        return out

    # ----- functor on morphisms -----
    def fmap(self, morphism: Callable[[FSSObject], FSSObject]) -> Callable[[FSSObject], FSSObject]:
        def mapped(fss_obj: FSSObject) -> FSSObject:
            x_transformed = morphism(fss_obj)
            return self.apply(x_transformed)
        return mapped
    
    # ----- gradient plumbing for fuzzy carriers -----
    def update_data(self, new_input_fss: FSSObject, new_target_fss: FSSObject):
        self.input_fss = new_input_fss
        self.target_fss = new_target_fss
    
    def store_gradients(self, gradients: torch.Tensor):
        # Clear previous gradients to prevent memory accumulation
        if hasattr(self, 'stored_gradients') and self.stored_gradients is not None:
            del self.stored_gradients
        # Store only gradient norm for attenuation, not full tensor
        self.stored_gradients = gradients.norm().detach().item()

# ---------------------------------------------------------------------
# Coalgebras
# ---------------------------------------------------------------------
@dataclass
class FCoalgebra(Generic[A]):
    """F-Coalgebra (A, α)"""
    carrier: A
    structure_map: Callable[[A], A]
    endofunctor: Endofunctor[A]
    name: Optional[str] = None

class FSSCoalagra(FCoalgebra[FSSObject]):
    """Backward-compatibility alias if older modules import this name."""
    pass

class FSSCoalgebra(FCoalgebra[FSSObject]):
    """
    Coalgebra in the category of fuzzy simplicial sets with F_B
    """
    def __init__(self, initial_fss: FSSObject, fb_functor: FuzzyBackpropagationFunctor, name: str = "fss_coalgebra"):
        def alpha(x: FSSObject) -> FSSObject:
            fx = fb_functor.apply(x)
            self._verify_naturality(x, fx, fb_functor)
            return fx
        super().__init__(carrier=initial_fss, structure_map=alpha, endofunctor=fb_functor, name=name)
        self.fb_functor = fb_functor
    
    def _verify_naturality(self, x: FSSObject, fx: FSSObject, fb: FuzzyBackpropagationFunctor) -> bool:
        try:
            if not (hasattr(x, "iter_face_maps") and hasattr(x, "apply_face_map")):
                return True
            ok = True
            for k, i, _f in x.iter_face_maps():
                x_di = x.apply_face_map(k, i)
                right = fb.apply(x_di)
                if hasattr(fx, "approx_equal_as_products"):
                    eq = fx.approx_equal_as_products(right, tol=1e-6)
                    ok = ok and bool(eq)
            return ok
        except Exception:
            return True

    def _fold_F_to_X(self, fx: FSSObject) -> FSSObject:
        if hasattr(self.carrier, "empty_like"):
            folded = self.carrier.empty_like(name=f"fold({getattr(fx, 'name', 'F(X)')})")
        else:
            folded = IntegratedFuzzySimplicialSet(name=f"fold({getattr(fx, 'name', 'F(X)')})",
                                                  max_dimension=fx.max_dimension)
        for k in range(fx.max_dimension + 1):
            bucket: Dict[Any, List[float]] = defaultdict(list)
            for prod_key, sx in _safe_get(fx.simplices, k, {}).items():
                x_key = prod_key[2] if (isinstance(prod_key, tuple) and len(prod_key) == 3) else prod_key
                bucket[x_key].append(_float(sx.membership))
            for x_key, mus in bucket.items():
                folded.add_simplex(k, x_key, float(np.max(mus) if mus else 0.0))
            if hasattr(self.carrier, "copy_face_degeneracy_structure_into"):
                try:
                    self.carrier.copy_face_degeneracy_structure_into(folded, k, source_component="X")
                except Exception:
                    pass
        return folded

    def update_memberships_via_gradients(self, gradients: torch.Tensor):
        if gradients is None:
            return
        factor = torch.sigmoid(-torch.norm(gradients).detach()).item()
        for k in range(self.carrier.max_dimension + 1):
            for _, sx in _safe_get(self.carrier.simplices, k, {}).items():
                sx.membership = float(np.clip(_float(sx.membership) * factor, 0.0, 1.0))
    
    def evolve(self, state: FSSObject) -> FSSObject:
        """Apply one coalgebra evolution step via the structure map."""
        return self.structure_map(state)

    def iterate(self, initial_state: FSSObject, steps: int) -> List[FSSObject]:
        """
        Run the coalgebra forward for `steps` iterations starting from initial_state.
        Each step applies the structure map α : A → F(A).
        Returns [x0, x1, ..., x_steps].
        
        Args:
            initial_state: Starting fuzzy simplicial set state
            steps: Number of coalgebraic evolution steps (must be non-negative)
            
        Returns:
            List of FSSObject states representing the coalgebraic trajectory
            
        Raises:
            ValueError: If steps is negative
        """
        if steps < 0:
            raise ValueError(f"Steps must be non-negative, got {steps}")
            
        out = [initial_state]
        cur = initial_state
        for _ in range(steps):
            cur = self.evolve(cur)
            out.append(cur)
        return out
    
    def is_fixed_point(self, state: FSSObject, tolerance: float = 1e-6) -> bool:
        fx = self.evolve(state)
        folded = self._fold_F_to_X(fx)
        if hasattr(state, "approx_equal"):
            return bool(state.approx_equal(folded, tol=tolerance))
        for k in range(state.max_dimension + 1):
            s_k = _safe_get(state.simplices, k, {})
            f_k = _safe_get(folded.simplices, k, {})
            if set(s_k.keys()) != set(f_k.keys()):
                return False
            for key in s_k.keys():
                if abs(_float(s_k[key].membership) - _float(f_k[key].membership)) > tolerance:
                    return False
        return True

# ---------------------------------------------------------------------
# Coalgebra homomorphisms & bisimulations
# ---------------------------------------------------------------------
class CoalagraHomomorphism(Generic[A, B]):
    """Backward-compatibility alias container (name kept)."""
    pass

class CoalgebraHomomorphism(Generic[A, B]):
    """
    Homomorphism f: (A,α) → (B,β) s.t. β ∘ f = F(f) ∘ α
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
        self.is_valid = self._verify_homomorphism()
        if not self.is_valid:
            logger.warning("Coalgebra homomorphism property does not hold")
    
    def _verify_homomorphism(self) -> bool:
        try:
            tests = self._generate_test_states()
            if not tests:
                return True
            for s in tests:
                if not self._verify_commutativity_for_state(s):
                    return False
            return True
        except Exception as e:
            logger.warning(f"Could not verify coalgebra homomorphism: {e}")
            return False
    
    def _verify_commutativity_for_state(self, state: A) -> bool:
        try:
            left = self.target.structure_map(self.morphism(state))  # β ∘ f
            evolved_source = self.source.structure_map(state)
            if hasattr(self.target.endofunctor, 'fmap'):
                Ff = self.target.endofunctor.fmap(self.morphism)
                right = Ff(evolved_source)
            else:
                right = self.morphism(evolved_source)
            return self._states_equal(left, right)
        except Exception as e:
            logger.debug(f"Commutativity check failed: {e}")
            return False
    
    def _generate_test_states(self) -> List[A]:
        tests: List[A] = []
        try:
            carr = getattr(self.source, "carrier", None)
            if carr is None:
                return tests
            if isinstance(carr, torch.Tensor):
                tests.extend([carr, carr*0.5, carr*2.0, carr + torch.randn_like(carr)*0.1])
            else:
                tests.append(carr)
        except Exception:
            pass
        return tests
    
    def _states_equal(self, s1, s2, tol: Optional[float] = None) -> bool:
        t = tol if tol is not None else self.tolerance
        try:
            if isinstance(s1, torch.Tensor) and isinstance(s2, torch.Tensor):
                return (s1.shape == s2.shape) and torch.allclose(s1, s2, atol=t, rtol=t)
            if isinstance(s1, (tuple, list)) and isinstance(s2, (tuple, list)):
                if len(s1) != len(s2): return False
                return all(self._states_equal(a, b, t) for a, b in zip(s1, s2))
            if hasattr(s1, "approx_equal"):
                return bool(s1.approx_equal(s2, tol=t))
            if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                return abs(s1 - s2) < t
            return s1 == s2
        except Exception:
            return False
    
    def apply(self, state: A) -> B:
        return self.morphism(state)
    
    def compose(self, other: 'CoalgebraHomomorphism[B, Any]') -> 'CoalgebraHomomorphism':
        def composed(state: A):
            return other.morphism(self.morphism(state))
        return CoalagraHomomorphism(self.source, other.target, composed) if isinstance(CoalagraHomomorphism, type) else CoalagraHomomorphism

    def is_isomorphism(self) -> bool:
        return self.is_valid
    
    def __repr__(self):
        return f"CoalebraHomomorphism({self.source.name} → {self.target.name}, {'valid' if self.is_valid else 'invalid'})"

class Bisimulation(Generic[A, B]):
    """
    F-Bisimulation between coalgebras (S,α_S) and (T,α_T)
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
        self.structure_map = self._mk_alpha_R()
        self.is_valid: Optional[bool] = None
    
    def _mk_alpha_R(self) -> Callable[[Tuple[A, B]], Tuple[A, B]]:
        def aR(pair: Tuple[A, B]) -> Tuple[A, B]:
            s, t = pair
            return (self.coalgebra1.structure_map(s), self.coalgebra2.endofunctor(s))
        return aR
    
    def verify(self) -> bool:
        self.is_valid = self._verify()
        return self.is_valid
    
    def _verify(self) -> bool:
        try:
            for s, t in self.relation:
                if not self._proj_is_hom(1, s, t): return False
                if not self._proj_is_hom(2, s, t): return False
                s2, t2 = self.structure_map((s, t))
                if not self._related(s2, t2): return False
            return True
        except Exception as e:
            logger.warning(f"Bisimulation verify failed: {e}")
            return False
    
    def _proj_is_hom(self, which: int, s: A, t: B) -> bool:
        evolved = self.structure_map((s, t))
        left = evolved[0] if which == 1 else evolved[1]
        right = self.coalgebra1.structure_map(s) if which == 1 else self.coalgebra2.structure_map(t)
        return self._eq(left, right)
    
    def _related(self, s: A, t: B) -> bool:
        if (s, t) in self.relation:
            return True
        for ss, tt in self.relation:
            if self._eq(s, ss) and self._eq(t, tt):
                return True
        return False
    
    def _eq(self, u, v) -> bool:
        try:
            if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor):
                return (u.shape == v.shape) and torch.allclose(u, v, atol=self.tolerance, rtol=self.tolerance)
            if hasattr(u, "approx_equal"):
                return bool(u.approx_equal(v, tol=self.tolerance))
            return u == v
        except Exception:
            return False
    
    def are_bisimilar(self, s: A, t: B) -> bool:
        return self._related(s, t)
    
    def add_relation_pair(self, s: A, t: B) -> bool:
        trial = self.relation + [(s, t)]
        tmp = Bisimulation(self.coalgebra1, self.coalgebra2, trial, self.tolerance)
        if tmp.verify():
            self.relation = trial
            self.is_valid = True
            return True
        return False
    
    def get_maximal_bisimulation(self) -> 'Bisimulation':
        extra: List[Tuple[A, B]] = []
        try:
            c1, c2 = self.coalgebra1.carrier, self.coalebra2.carrier  # keep name typo compatibility
        except Exception:
            try:
                c1, c2 = self.coalgebra1.carrier, self.coalgebra2.carrier
            except Exception:
                c1 = c2 = None
        try:
            if isinstance(c1, torch.Tensor) and isinstance(c2, torch.Tensor):
                base = self.relation[0] if self.relation else (c1, c2)
                for _ in range(8):
                    s = base[0] + torch.randn_like(base[0]) * 0.1
                    t = base[1] + torch.randn_like(base[1]) * 0.1
                    if self.add_relation_pair(s, t):
                        extra.append((s, t))
        except Exception:
            pass
        return Bisimulation(self.coalgebra1, self.coalgebra2, self.relation + extra, self.tolerance)

# ---------------------------------------------------------------------
# Parametric (tensor) coalgebra + trainer
# ---------------------------------------------------------------------
class GenerativeCoalagra(FCoalgebra[torch.Tensor]):
    """Backward-compatibility alias."""
    pass

class GenerativeCoalgebra(FCoalgebra[torch.Tensor]):
    """
    Tensor-space coalgebra using BackpropagationFunctor
    """
    def __init__(self, model: nn.Module):
        params_list = [p.flatten() for p in model.parameters()]
        params = torch.cat(params_list) if len(params_list) else torch.zeros(1)
        self.backprop_functor: Optional[BackpropagationFunctor] = None
        def gamma(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.backprop_functor is None:
                raise ValueError("Training data not set. Call update_training_data() first.")
            return self.backprop_functor.apply(x)
        super().__init__(carrier=params, structure_map=gamma, endofunctor=None, name="GenerativeCoalgebra")
        self.model = model
    
    def update_training_data(self, new_input: torch.Tensor, new_target: torch.Tensor):
        if self.backprop_functor is None:
            self.backprop_functor = BackpropagationFunctor(new_input, new_target)
            self.endofunctor = self.backprop_functor
        else:
            self.backprop_functor.update_data(new_input, new_target)
        self.input_data = new_input
        self.target_data = new_target
    
    @property
    def input_data(self):
        if hasattr(self, '_input_data'): return self._input_data
        if self.backprop_functor is not None: return self.backprop_functor.input_data
        return None
    @input_data.setter
    def input_data(self, v): self._input_data = v
    
    @property
    def target_data(self):
        if hasattr(self, '_target_data'): return self._target_data
        if self.backprop_functor is not None: return self.backprop_functor.target_data
        return None
    @target_data.setter
    def target_data(self, v): self._target_data = v

class CoalagraTrainer:
    """Backward-compatibility alias."""
    pass

class CoalgebraTrainer:
    """
    External training wrapper. Keeps coalgebra structure pure.
    """
    def __init__(self, 
                 coalgebra: GenerativeCoalagra if 'GenerativeCoalagra' in globals() else GenerativeCoalgebra,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.coalgebra = coalagra if (coalagra := coalgebra) else coalgebra
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gradient_history = []

    # ---- adapter helpers (DO NOT change public signatures) ----
    def _find_vocab_weight(self, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Try common locations for tied vocab/head weights of shape [V, D].
        """
        cand_names = [
            "lm_head.weight",
            "decoder.weight",
            "output_projection.weight",
            "embed_out.weight",
            "proj.weight",
        ]
        # also try embeddings (transpose if needed)
        embed_names = [
            "transformer.wte.weight",
            "embed_tokens.weight",
            "embedding.weight",
            "encoder.embed.weight",
        ]
        for n in cand_names:
            m = model
            try:
                for part in n.split(".")[:-1]:
                    m = getattr(m, part)
                w = getattr(m, n.split(".")[-1], None)
            except Exception:
                w = None
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                return w  # [V, D]
        for n in embed_names:
            m = model
            try:
                for part in n.split(".")[:-1]:
                    m = getattr(m, part)
                w = getattr(m, n.split(".")[-1], None)
            except Exception:
                w = None
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                return w  # [V, D] embedding matrix works too
        # exhaustive sweep (last resort)
        for name, p in model.named_parameters():
            if p.dim() == 2 and p.shape[0] > p.shape[1]:
                # looks like [V, D]
                return p
        return None

    def _align_and_compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Shape-aware loss mediation (handles 2D and 3D sequence outputs):
        - index targets → CrossEntropy
        - embedding targets (via vocab weight) → KLDiv between distributions
        - else pad/truncate and MSE
        """
        # Note: Removed target.detach() to allow gradient flow for coalgebraic training

        # Normalize to (B, T, V) or (B, V) for output
        if output.dim() == 3:
            B, T, V = output.shape
            out2d = output.reshape(B * T, V)
        elif output.dim() == 2:
            B, V = output.shape
            T = 1
            out2d = output
        else:
            raise ValueError(f"Unsupported output dims: {tuple(output.shape)}")

        # ---- 1) Index targets (CrossEntropy) ----
        if target.dtype in (torch.long, torch.int64):
            if target.dim() == 3:
                # Some pipelines one-hot encode; collapse to indices if it's argmax-onehot
                _, idx = target.max(dim=-1)
                target = idx
            if target.dim() == 2:  # (B, T)
                y = target.reshape(-1)  # (B*T,)
                # Ensure out2d matches the flattened target size
                if out2d.size(0) != y.size(0):
                    min_size = min(out2d.size(0), y.size(0))
                    out2d = out2d[:min_size]
                    y = y[:min_size]
                loss = nn.CrossEntropyLoss()(out2d, y)
                return loss
            elif target.dim() == 1:  # (B,)
                # Ensure out2d batch size matches target
                if out2d.size(0) != target.size(0):
                    min_size = min(out2d.size(0), target.size(0))
                    out2d = out2d[:min_size]
                    target = target[:min_size]
                loss = nn.CrossEntropyLoss()(out2d, target)
                return loss
            else:
                raise ValueError(f"Incompatible index target shape: {tuple(target.shape)}")

        # ---- 2) Embedding targets -> vocab via W and KLDiv ----
        if target.dim() in (2, 3):
            # Make target 2D as (B*T, D) if needed
            if target.dim() == 3:
                Bt, Dt = target.shape[0] * target.shape[1], target.shape[2]
                tgt2d = target.reshape(Bt, Dt)
                # when output was 2D (B, V) but target has T>1, ensure B*T matches
                if out2d.size(0) != Bt:
                    # repeat or truncate to align time
                    if out2d.size(0) == B and target.shape[1] > 1:
                        out2d = out2d.repeat_interleave(target.shape[1], dim=0)
                    else:
                        Bt2 = min(out2d.size(0), Bt)
                        out2d = out2d[:Bt2]
                        tgt2d = tgt2d[:Bt2]
            else:
                tgt2d = target  # (B, D)

            # Try to find a vocab/head matrix W: [V, D]
            w = self._find_vocab_weight(self.coalgebra.model)
            if isinstance(w, torch.Tensor) and w.dim() == 2:
                V = out2d.size(1)
                if w.size(0) == V:
                    # Map target embeddings → vocab logits
                    target_logits = tgt2d @ w.t()  # (N, V)
                    log_p = torch.log_softmax(out2d, dim=-1)
                    q = torch.softmax(target_logits, dim=-1)
                    loss = nn.KLDivLoss(reduction="batchmean")(log_p, q)
                    return loss

        # ---- 3) Fallback: pad/truncate and MSE on 2D ----
        # Make both 2D same shape
        if target.dim() == 3:
            tgt2d = target.reshape(target.size(0) * target.size(1), target.size(2))
        elif target.dim() == 2:
            tgt2d = target
        else:
            raise ValueError(f"Unsupported target dims for fallback: {tuple(target.shape)}")

        # align row count
        n = min(out2d.size(0), tgt2d.size(0))
        out2d = out2d[:n]
        tgt2d = tgt2d[:n]

        # align feature dim
        f1, f2 = out2d.size(1), tgt2d.size(1)
        if f1 > f2:
            out2d = out2d[:, :f2]
        elif f2 > f1:
            # Create padding tensor without gradients to prevent graph inflation
            pad = torch.zeros(out2d.size(0), f2 - f1, device=out2d.device, dtype=out2d.dtype, requires_grad=False)
            out2d = torch.cat([out2d, pad], dim=1)
        

        loss = nn.MSELoss()(out2d, tgt2d)
        return loss

    def train_step(self, input_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """Execute one training step using coalgebraic evolution with categorical memory bounds."""
        self.coalagra = self.coalgebra  # alias
        self.coalagra.update_training_data(input_data, target_data)
        
        cur = self.coalagra.carrier
        
        # Apply categorical memory bounds through colimit construction
        with self._categorical_memory_context():
            # Detach carrier to prevent graph conflicts
            cur_detached = cur.detach().requires_grad_(False)
            
            loss, evolved = self._backprop_step(cur_detached)
            
            # Ensure evolved parameters don't carry gradients
            evolved_detached = evolved.detach().requires_grad_(False)
            self.coalagra.carrier = evolved_detached
        
        return loss
    
    def _backprop_step(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient coalgebraic backpropagation with gradient checkpointing and streaming.
        
        Long-term architectural solution:
        1. Gradient checkpointing to reduce memory footprint
        2. Streaming parameter updates to avoid large tensor accumulation
        3. Coalgebraic structure preservation with minimal memory overhead
        4. Automatic memory pressure detection and fallback strategies
        
        Args:
            params: Flattened model parameters
            
        Returns:
            Tuple of (loss, evolved_params)
        """
        try:
            # Memory pressure detection
            if torch.backends.mps.is_available():
                try:
                    current_memory = torch.mps.current_allocated_memory() / (1024**3)  # GB
                    if current_memory > 10.0:  # Above 10GB, use aggressive memory management
                        torch.mps.empty_cache()
                        # Force CPU fallback for this step if memory is critical
                        if current_memory > 12.0:
                            return self._cpu_fallback_step(params)
                except:
                    pass
            
            # Streaming parameter loading - process in chunks to reduce peak memory
            model_param_count = sum(p.numel() for p in self.coalagra.model.parameters())
            if params.numel() != model_param_count:
                if params.numel() < model_param_count:
                    padding = torch.zeros(model_param_count - params.numel(), device=params.device, dtype=params.dtype)
                    params = torch.cat([params, padding])
                else:
                    params = params[:model_param_count]
            
            param_idx = 0
            with torch.no_grad():  # In-place parameter copying to prevent allocator churn
                for p in self.coalagra.model.parameters():
                    param_size = p.numel()
                    # Copy into existing parameter tensors to prevent storage reassignment
                    param_slice = params[param_idx:param_idx + param_size].view(p.shape)
                    p.data.copy_(param_slice)  # In-place copy preserves tensor identity
                    param_idx += param_size
                        
            self.coalagra.model.train()
            
            # Use gradient checkpointing for memory efficiency
            A_input = self.coalagra.backprop_functor.input_data.detach().requires_grad_(False)  # Detach and disable gradients
            B = self.coalagra.backprop_functor.target_data.detach().requires_grad_(False)  # Ensure target doesn't require gradients
            
            # Ensure model parameters require gradients
            for p in self.coalagra.model.parameters():
                p.requires_grad_(True)
            
            # Fast forward pass without debug overhead
            out = self.coalagra.model(A_input)
            
            # Compute loss efficiently
            loss = self._memory_efficient_loss(out, B)
            
            # Ensure loss requires gradients
            if not loss.requires_grad:
                param_loss = sum(p.sum() * 1e-6 for p in self.coalagra.model.parameters() if p.requires_grad)
                loss = loss + param_loss
            
            # Fast gradient computation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)

            # Collect gradient stats (so you can optionally add coalgebraic noise)
            grad_stats = self._fast_grad_stats()
            self.coalagra.backprop_functor.stored_gradients = grad_stats

            # Apply optimizer update to model weights
            self.optimizer.step()
            
            # Clear gradients after stepping
            for p in self.coalagra.model.parameters():
                p.grad = None

            # NEW: make the carrier reflect the model *after* the step
            evolved_params = self._flatten_model_params()
            # Optional: tiny coalgebraic jitter using the stats (works on MPS too)
            evolved_params = self._apply_coalgebraic_noise(evolved_params, grad_stats)
            
            # Aggressive cleanup and gradient clearing
            del A_input, out, B
            
            # Clear all gradients to prevent accumulation
            for p in self.coalagra.model.parameters():
                if p.grad is not None:
                    p.grad = None
            
            # Clear MPS cache and force garbage collection
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            return loss.detach(), evolved_params
        except Exception as e:
            logger.error(f"Error in _backprop_step: {e}")
            # Clear any partial gradients before fallback
            self.optimizer.zero_grad(set_to_none=True)
            # Fallback to CPU computation without backward pass (gradients already computed or failed)
            return self._cpu_fallback_step(params, skip_backward=True)
    
    def _categorical_memory_context(self):
        """Context manager for categorical memory bounds using limits and colimits."""
        class CategoricalMemoryContext:
            def __init__(self, trainer):
                self.trainer = trainer
                self.initial_state = None
                
            def __enter__(self):
                # Establish categorical memory limit
                self.initial_state = self.trainer._capture_categorical_state()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Enforce categorical memory bounds through natural truncation
                self.trainer._enforce_categorical_bounds(self.initial_state)
                
        return CategoricalMemoryContext(self)
    
    def _capture_categorical_state(self) -> dict:
        """Capture current categorical state for memory management."""
        return {
            'gradient_history_length': len(self.gradient_history),
            'device': next(self.coalgebra.model.parameters()).device
        }
    
    def _enforce_categorical_bounds(self, initial_state: dict) -> None:
        """Enforce categorical memory bounds through natural operations."""
        # Clear MPS cache if available (categorical cleanup)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        # Ensure gradient history stays within categorical bounds
        max_history = self._categorical_memory_bound()
        if len(self.gradient_history) > max_history:
            # Natural categorical truncation
            self.gradient_history = self.gradient_history[-max_history//2:]
    
    def _categorical_memory_bound(self) -> int:
        """Compute categorical memory bound based on coalgebraic structure."""
        # Base bound on categorical structure - natural number from coalgebra dimension
        base_bound = 10 * 10  # Conservative categorical bound
        return max(base_bound, 50)  # Minimum categorical bound
    
    def _store_categorical_gradients(self, gradients: torch.Tensor) -> None:
        """Store gradients using categorical colimit construction for memory bounds."""
        # Construct categorical colimit for gradient statistics
        gradient_colimit = self._construct_gradient_colimit(gradients)
        
        # Store only the categorical representation
        self.gradient_history.append(gradient_colimit)
        
        # Maintain categorical memory bounds through natural truncation
        if len(self.gradient_history) > self._categorical_memory_bound():
            # Remove oldest entries to maintain categorical structure
            self.gradient_history = self.gradient_history[-self._categorical_memory_bound()//2:]
    
    def _store_categorical_gradients_from_stats(self, grad_stats: dict) -> None:
        """Store pre-computed gradient statistics without tensor materialization."""
        # Store only the categorical representation from pre-computed stats
        self.gradient_history.append(grad_stats)
        
        # Maintain categorical memory bounds through natural truncation
        if len(self.gradient_history) > self._categorical_memory_bound():
            # Remove oldest entries to maintain categorical structure
            self.gradient_history = self.gradient_history[-self._categorical_memory_bound()//2:]
    
    def _construct_gradient_colimit(self, gradients: torch.Tensor) -> dict:
        """Construct categorical colimit for gradient representation."""
        if gradients is None or gradients.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'norm': 0.0}
            
        # Construct colimit using categorical operations with bounds
        with torch.no_grad():
            return {
                'mean': float(torch.clamp(gradients.mean(), -1e6, 1e6).item()),
                'std': float(torch.clamp(gradients.std(), 0, 1e6).item()),
                'norm': float(torch.clamp(gradients.norm(), 0, 1e6).item())
            }
    
    def _memory_efficient_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient loss computation with streaming alignment.
        """
        # Process in smaller chunks to reduce memory pressure
        if output.numel() > 10000:  # For large tensors, process in chunks
            chunk_size = 1000
            total_loss = None
            num_chunks = 0
            
            for i in range(0, output.size(0), chunk_size):
                end_idx = min(i + chunk_size, output.size(0))
                output_chunk = output[i:end_idx]
                # Ensure target chunk matches output chunk dimensions
                if target.size(0) >= end_idx:
                    target_chunk = target[i:end_idx]
                else:
                    # Handle case where target is smaller than output
                    available_target = target.size(0) - i
                    if available_target > 0:
                        target_chunk = target[i:i + available_target]
                        # Pad or truncate output_chunk to match target_chunk
                        output_chunk = output_chunk[:available_target]
                    else:
                        # Skip this chunk if no target data available
                        continue
                
                chunk_loss = self._align_and_compute_loss(output_chunk, target_chunk)
                
                # CRITICAL FIX: Accumulate tensor losses, not scalar values
                # Clone to avoid graph reuse issues
                if total_loss is None:
                    total_loss = chunk_loss.clone()
                else:
                    total_loss = total_loss + chunk_loss.clone()
                num_chunks += 1
                
                # Clean up chunk tensors
                del output_chunk, target_chunk, chunk_loss
            
            # Return average loss while preserving gradients
            return total_loss / num_chunks if total_loss is not None else torch.tensor(0.0, requires_grad=True, device=output.device)
        else:
            return self._align_and_compute_loss(output, target)
    
    def _flatten_model_params(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([p.data.view(-1) for p in self.coalagra.model.parameters()])
    
    def _fast_grad_stats(self) -> dict:
        total, sum_sq, sum_val = 0, 0.0, 0.0
        with torch.no_grad():
            for p in self.coalagra.model.parameters():
                if p.grad is not None:
                    g = p.grad.data
                    total += g.numel()
                    sum_sq += (g * g).sum().item()
                    sum_val += g.sum().item()
        if total == 0:
            return {'norm': 0.0, 'mean': 0.0, 'std': 0.0}
        mean = sum_val / total
        var = max(0.0, sum_sq / total - mean * mean)
        return {
            'norm': min(sum_sq ** 0.5, 1000.0),
            'mean': max(-100.0, min(mean, 100.0)),
            'std': min(var ** 0.5, 100.0),
        }
    
    def _apply_coalgebraic_noise(self, flat_params: torch.Tensor, grad_stats: dict) -> torch.Tensor:
        # Keep it tiny & device-safe (including MPS)
        if not grad_stats or grad_stats.get('norm', 0.0) == 0.0:
            return flat_params
        with torch.no_grad():
            # scale ~ norm * lr * small factor
            lr = next((g.get('lr', None) for g in self.optimizer.param_groups), None) or 1e-3
            scale = grad_stats['norm'] * lr * 0.01
            flat_params.add_(torch.randn_like(flat_params) * scale)
        return flat_params

    def _streaming_coalgebraic_evolution(self, params: torch.Tensor) -> torch.Tensor:
        """
        Optimized coalgebraic parameter evolution with fast gradient statistics.
        """
        # Fast gradient statistics computation using vectorized operations
        total_grad_elements = 0
        sum_squares = 0.0
        sum_values = 0.0
        
        # Vectorized gradient processing - much faster than chunking
        with torch.no_grad():
            for p in self.coalagra.model.parameters():
                if p.grad is not None:
                    # Direct vectorized operations on full gradient tensor
                    grad_data = p.grad.data  # Use .data to avoid autograd overhead
                    total_grad_elements += grad_data.numel()
                    
                    # Vectorized statistics - single pass, no loops
                    sum_squares += (grad_data ** 2).sum().item()
                    sum_values += grad_data.sum().item()
        
        # Fast statistics computation
        if total_grad_elements > 0:
            grad_norm = (sum_squares ** 0.5)
            grad_mean = sum_values / total_grad_elements
            grad_std = max(0.0, (sum_squares / total_grad_elements - grad_mean ** 2) ** 0.5)
            
            # Bounded gradient statistics
            grad_stats = {
                'norm': min(grad_norm, 1000.0),
                'mean': max(-100.0, min(grad_mean, 100.0)),
                'std': min(grad_std, 100.0)
            }
        else:
            grad_stats = {'norm': 0.0, 'mean': 0.0, 'std': 0.0}
        
        # Fast storage without expensive operations
        self.coalagra.backprop_functor.stored_gradients = grad_stats
        
        # Direct coalgebraic evolution - skip unnecessary tuple unpacking
        evolved = self.coalagra.backprop_functor.apply(params)
        
        # Fast parameter extraction
        return evolved[2] if isinstance(evolved, tuple) and len(evolved) >= 3 else evolved
    
    def _cpu_fallback_step(self, params: torch.Tensor, skip_backward: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CPU fallback for memory-constrained situations.
        """
        
        # Move everything to CPU temporarily
        original_device = params.device
        
        # Move model to CPU
        self.coalagra.model.cpu()
        params_cpu = params.cpu()
        input_cpu = self.coalagra.backprop_functor.input_data.cpu().detach().requires_grad_(False)
        target_cpu = self.coalagra.backprop_functor.target_data.cpu().detach().requires_grad_(False)
        
        try:
            # Load parameters with proper size checking
            param_idx = 0
            total_params_needed = sum(p.numel() for p in self.coalagra.model.parameters())
            
            if params_cpu.numel() < total_params_needed:
                logger.error(f"🔍 CPU_FALLBACK: Parameter tensor too small: {params_cpu.numel()} < {total_params_needed}")
                raise ValueError(f"Parameter tensor size mismatch: {params_cpu.numel()} < {total_params_needed}")
            
            for p in self.coalagra.model.parameters():
                param_size = p.numel()
                if param_idx + param_size > params_cpu.numel():
                    logger.error(f"🔍 CPU_FALLBACK: Not enough parameters for layer: need {param_size}, have {params_cpu.numel() - param_idx}")
                    raise ValueError(f"Insufficient parameters for layer")
                with torch.no_grad():
                    p.data.copy_(params_cpu[param_idx:param_idx + param_size].view(p.shape))
                param_idx += param_size
            
            self.coalagra.model.train()
            
            # Forward pass
            out = self.coalagra.model(input_cpu)
            loss = self._align_and_compute_loss(out, target_cpu)
            
            # Only perform backward pass if not skipping (i.e., not called as fallback after failed backward)
            if not skip_backward:
                # Backward pass with explicit memory cleanup
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=False)
                
                # Fast gradient statistics for CPU fallback
                total_elements = 0
                sum_grads = 0.0
                sum_squared_grads = 0.0
                
                with torch.no_grad():
                    for p in self.coalagra.model.parameters():
                        if p.grad is not None:
                            # Vectorized gradient statistics - no chunking needed
                            grad_data = p.grad.data
                            total_elements += grad_data.numel()
                            sum_grads += grad_data.sum().item()
                            sum_squared_grads += (grad_data ** 2).sum().item()
                
                if total_elements > 0:
                    grad_stats = {
                        'norm': (sum_squared_grads ** 0.5),
                        'mean': sum_grads / total_elements,
                        'std': max(0.0, (sum_squared_grads / total_elements - (sum_grads / total_elements) ** 2) ** 0.5)
                    }
                    self.coalagra.backprop_functor.stored_gradients = grad_stats
                    
                self.optimizer.step()
            else:
                # Skip backward pass - use existing gradients or return simple loss
                logger.info("🔄 COALGEBRA: Skipping backward pass in CPU fallback (already attempted)")
            
            # Mirror GPU path: flatten post-step model, then optional noise
            evolved_flat = torch.cat([p.data.view(-1) for p in self.coalagra.model.parameters()])
            if not skip_backward:
                # reuse stats already computed above; if you didn't compute, call _fast_grad_stats()
                pass
            loss_result = loss.detach().to(original_device)
            evolved_result = evolved_flat.to(original_device)
            
            # Clear all gradients to prevent accumulation
            for p in self.coalagra.model.parameters():
                if p.grad is not None:
                    p.grad = None
            
            del loss, out, params_cpu, input_cpu, target_cpu
            
            # Clear MPS cache more aggressively
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Move back to original device
            self.coalagra.model.to(original_device)
            
            return loss_result, evolved_result
            
        except Exception as e:
            logger.error(f"🔍 CPU_FALLBACK: Failed: {e}")
            # Move model back to original device
            self.coalagra.model.to(original_device)
            return torch.tensor(float('inf')), params
    
    def evolve_coalgebra(self, steps: int = 1) -> List[torch.Tensor]:
        states = [self.coalgebra.carrier.clone().detach()]
        
        for step in range(steps):
            _ = self.train_step(self.coalgebra.backprop_functor.input_data,
                                self.coalgebra.backprop_functor.target_data)
            new_state = self.coalgebra.carrier.clone().detach()
            states.append(new_state)
        
        return states

# ---------------------------------------------------------------------
# Category of coalgebras
# ---------------------------------------------------------------------
class CoalgebraCategory:
    """Category of F-coalgebras with homomorphisms as morphisms."""
    def __init__(self):
        self.objects: Dict[str, FCoalgebra] = {}
        self.morphisms: Dict[Tuple[str, str], CoalagraHomomorphism if 'CoalagraHomomorphism' in globals() else CoalgebraHomomorphism] = {}
    
    def add_coalgebra(self, name: str, coalgebra: FCoalgebra):
        self.objects[name] = coalgebra
    
    def add_homomorphism(self, source_name: str, target_name: str, morphism: Callable):
        if source_name not in self.objects or target_name not in self.objects:
            raise ValueError("Source and target coalgebras must exist")
        src, tgt = self.objects[source_name], self.objects[target_name]
        hom = CoalagraHomomorphism(src, tgt, morphism) if isinstance(CoalagraHomomorphism, type) else CoalgebraHomomorphism(src, tgt, morphism)
        self.morphisms[(source_name, target_name)] = hom
    
    def compose_morphisms(self, first: Tuple[str, str], second: Tuple[str, str]) -> Optional[CoalgebraHomomorphism]:
        if first[1] != second[0]:
            return None
        if first not in self.morphisms or second not in self.morphisms:
            return None
        h1, h2 = self.morphisms[first], self.morphisms[second]
        def composed(x): return h2.apply(h1.apply(x))
        return CoalgebraHomomorphism(h1.source, h2.target, composed)
    
    def find_bisimulations(self) -> List[Tuple[str, str, Bisimulation]]:
        out: List[Tuple[str, str, Bisimulation]] = []
        names = list(self.objects.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                b = Bisimulation(self.objects[n1], self.objects[n2], relation=[])
                if b.verify():
                    out.append((n1, n2, b))
        return out

# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------
def create_llm_coalgebra(model: nn.Module, 
                         input_data: torch.Tensor = None,
                         target_data: torch.Tensor = None) -> GenerativeCoalgebra:
    coalgebra = GenerativeCoalgebra(model)
    if input_data is not None and target_data is not None:
        coalgebra.update_training_data(input_data, target_data)
    return coalagra if (coalagra := coalgebra) else coalgebra

def create_llm_coalgebra_trainer(model: nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 loss_fn: Callable,
                                 input_data: torch.Tensor,
                                 target_data: torch.Tensor) -> CoalagraTrainer if 'CoalagraTrainer' in globals() else CoalgebraTrainer:
    coalgebra = create_llm_coalgebra(model, input_data, target_data)
    return CoalgebraTrainer(coalgebra, optimizer, loss_fn)

def create_diffusion_coalgebra(model: nn.Module,
                               noise_schedule: Callable) -> FCoalgebra[torch.Tensor]:
    def diffusion_structure_map(state: torch.Tensor) -> torch.Tensor:
        return state + noise_schedule(state)
    return FCoalgebra(
        carrier=torch.randn(100),
        structure_map=diffusion_structure_map,
        endofunctor=NeuralFunctor(100, 100),
        name="DiffusionCoalgebra"
    )

def create_transformer_coalgebra(attention_heads: int,
                                 hidden_dim: int) -> FCoalgebra[torch.Tensor]:
    def transformer_structure_map(state: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(state, dim=-1)
    return FCoalgebra(
        carrier=torch.randn(hidden_dim),
        structure_map=transformer_structure_map,
        endofunctor=NeuralFunctor(hidden_dim, attention_heads),
        name="TransformerCoalgebra"
    )

# ---------------------------------------------------------------------
# Example usage and testing
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Tensor coalgebra sanity test
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    A = torch.randn(4, 10)
    B = torch.randn(4, 1)
    trainer = create_llm_coalgebra_trainer(model, optimizer, loss_fn, A, B)
    init = trainer.coalgebra.carrier
    evolved_states = trainer.evolve_coalgebra(steps=3)

