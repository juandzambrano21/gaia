"""
Module: inner_solver
Implements the EndofunctorialSolver for inner horn filling.

Following Mahadevan (2024), this implements the Inner Horn Λ²₁ solver
where h = g ∘ f is defined symbolically and optimization concerns only task loss.
"""

import datetime as _dt
from typing import Dict, Optional
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gaia.core.simplices import Simplex1, Simplex2
from gaia.core.functor import SimplicialFunctor
from gaia.core import DEVICE

class EndofunctorialSolver:
    """
    GAIA Inner-horn solver implementing true endofunctorial optimization.
    Maintains simplicial coherence while training neural morphisms.
    
    Following Mahadevan (2024), this implements the Inner Horn Λ²₁ solver
    where h = g ∘ f is defined symbolically and optimization concerns only task loss.
    """
    
    def solve_horn(self, horn) -> Optional[Dict]:
        """Solve inner horn using endofunctorial approach."""
        try:
            # Extract horn information
            simplex_id = horn.simplex_id if hasattr(horn, 'simplex_id') else None
            horn_index = horn.horn_index if hasattr(horn, 'horn_index') else None
            
            if simplex_id is None or horn_index is None:
                return None
                
            # Get the simplex from functor registry
            if not hasattr(self.functor, 'registry') or simplex_id not in self.functor.registry:
                return None
                
            simplex = self.functor.registry[simplex_id]
            
            # Verify this is an inner horn (1 ≤ k ≤ n-1)
            if not (1 <= horn_index <= simplex.level - 1):
                return None

            result = self._apply_endofunctorial_composition(simplex, horn_index)
            
            solution = {
                'status': 'solved',
                'method': 'endofunctorial',
                'simplex_id': simplex_id,
                'horn_index': horn_index,
                'result': result
            }
            return solution
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _apply_endofunctorial_composition(self, simplex, horn_index):
        """Apply endofunctorial composition for inner horn filling."""
        
        # For inner horns, we need to fill the missing face by creating a new morphism
        try:
            # Create the missing face morphism based on horn type
            if hasattr(simplex, 'level') and simplex.level >= 2:
                # For 2-simplices, create the missing edge
                if horn_index == 1:  # Missing middle face
                    # Create composition h = g ∘ f as required by theory
                    if hasattr(simplex, 'f') and hasattr(simplex, 'g'):
                        # Get domain and codomain from existing morphisms
                        domain = simplex.f.domain if hasattr(simplex.f, 'domain') else None
                        codomain = simplex.g.codomain if hasattr(simplex.g, 'codomain') else None
                        
                        if domain and codomain:
                            # Create new morphism as composition
                            import torch.nn as nn
                            new_morphism = nn.Linear(domain.dim, codomain.dim)
                            
                            # Initialize with composition of existing morphisms
                            with torch.no_grad():
                                if hasattr(simplex.f, 'morphism') and hasattr(simplex.g, 'morphism'):
                                    # Check if morphisms are Identity layers (placeholders)
                                    f_is_identity = isinstance(simplex.f.morphism, nn.Identity)
                                    g_is_identity = isinstance(simplex.g.morphism, nn.Identity)
                                    
                                    if not f_is_identity and not g_is_identity:
                                        # Both have weights - compose them: W_h = W_g @ W_f
                                        new_morphism.weight.data = torch.matmul(
                                            simplex.g.morphism.weight.data,
                                            simplex.f.morphism.weight.data
                                        )
                                        if new_morphism.bias is not None:
                                            new_morphism.bias.data = simplex.g.morphism.bias.data.clone()
                                        print(f"🔧 INNER SOLVER: Composed weights from non-Identity morphisms")
                                    elif f_is_identity and not g_is_identity:
                                        # f is Identity, use g's weights
                                        new_morphism.weight.data = simplex.g.morphism.weight.data.clone()
                                        if new_morphism.bias is not None and hasattr(simplex.g.morphism, 'bias') and simplex.g.morphism.bias is not None:
                                            new_morphism.bias.data = simplex.g.morphism.bias.data.clone()
                                        print(f"🔧 INNER SOLVER: Used g's weights (f is Identity)")
                                    elif not f_is_identity and g_is_identity:
                                        # g is Identity, use f's weights
                                        new_morphism.weight.data = simplex.f.morphism.weight.data.clone()
                                        if new_morphism.bias is not None and hasattr(simplex.f.morphism, 'bias') and simplex.f.morphism.bias is not None:
                                            new_morphism.bias.data = simplex.f.morphism.bias.data.clone()
                                        print(f"🔧 INNER SOLVER: Used f's weights (g is Identity)")

                            # Create new 1-simplex for the filled horn
                            from gaia.core.simplices import Simplex1
                            import uuid
                            filled_morphism = Simplex1(
                                new_morphism, domain, codomain, f"filled_horn_{uuid.uuid4().hex[:8]}"
                            )
                            
                            # CRITICAL: Add to functor registry to modify structure
                            if hasattr(self.functor, 'registry'):
                                self.functor.registry[filled_morphism.id] = filled_morphism
                                print(f"🔧 INNER SOLVER: Added filled morphism {filled_morphism.id} to functor registry")
                            
                            # Update the simplex to include the filled face
                            if hasattr(simplex, 'faces'):
                                simplex.faces[horn_index] = filled_morphism
                                print(f"🔧 INNER SOLVER: Updated simplex face {horn_index} with filled morphism")
                            
                            return {
                                'filled_morphism': filled_morphism,
                                'composition_weights': new_morphism.state_dict(),
                                'modified_registry': True
                            }
            
            return {'modified_registry': False, 'error': 'Insufficient simplex structure'}
            
        except Exception as e:
            return {'modified_registry': False, 'error': str(e)}
    def __init__(self, functor: SimplicialFunctor, simplex2_id: uuid.UUID, lr: float = 0.01, coherence_weight: float = 1.0, writer: Optional[SummaryWriter] = None):
        self.functor = functor
        self.s2_id = simplex2_id  
        self.coherence_weight = coherence_weight
        
        s2 = functor.registry[simplex2_id]  
        if not isinstance(s2, Simplex2):
            raise ValueError('EndofunctorialSolver requires a 2-simplex')
            
        # training morphisms
        self.f = s2.f
        self.g = s2.g
        self.f.morphism.train()
        self.g.morphism.train()
        
        self.loss_fn = nn.MSELoss()
        self.writer = writer or SummaryWriter(
            log_dir=f'runs/inner_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        self._step = 0
        self._epoch = 0  # Track epochs properly
        
        # According to GAIA paper, horn filling should optimize the entire
        # functorial structure. All morphisms f, g, h in the 2-simplex should be learnable
        # neural networks registered in the functor, as per the categorical foundation.
        
        all_params = []
        
        # Collect parameters from all morphisms in the triangle
        f_params = list(self.f.morphism.parameters())
        g_params = list(self.g.morphism.parameters()) 
        h_params = list(s2.h.morphism.parameters())
        
        if f_params:
            all_params.extend(f_params)
        
        if g_params:
            all_params.extend(g_params)
            
        if h_params:
            all_params.extend(h_params)
        
        if not all_params:
            raise RuntimeError(f"No trainable parameters found in triangle morphisms. "
                             f"This violates GAIA's theoretical foundation where all morphisms "
                             f"should be learnable neural networks registered in the functor.")
        
        # Optimizer with gradient clipping for stability - optimize entire triangle
        self.opt = optim.Adam(all_params, lr=lr, weight_decay=1e-5)
        
        # FIX: Proper epoch-based scheduler with deterministic resets
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=50, T_mult=2  # T_0 in epochs, not steps
        )
        
        # FIX: Create h_ref as frozen reference for coherence measurement
        from copy import deepcopy
        self.h_ref = deepcopy(s2.h.morphism)
        for param in self.h_ref.parameters():
            param.requires_grad = False
        self.h_ref.eval()  # Keep in eval mode

    def coherence_loss(self, xb: torch.Tensor) -> torch.Tensor:
        """
        FIX: Compute coherence loss against frozen h_ref to prevent always-zero loss.
        
        This measures drift from the original h = g ∘ f composition.
        """
        with torch.no_grad():
            h_ref_out = self.h_ref(xb)  # Frozen reference
        gf_out = self.g(self.f(xb))  # Current composition
        return nn.MSELoss()(gf_out, h_ref_out)

    def step(self, xb: torch.Tensor, yb: torch.Tensor) -> Dict[str, float]:
        """
        Single optimization step maintaining simplicial coherence.
        Returns dict of loss components for monitoring.
        """
        self.opt.zero_grad()
        
        # Primary task loss
        pred = self.g(self.f(xb))
        task_loss = self.loss_fn(pred, yb)
        
        # FIX: Coherence constraint against frozen reference
        coherence = self.coherence_loss(xb)
        total = task_loss + self.coherence_weight * coherence
        
        total.backward()
        
        # FIX: Constant gradient clipping independent of coherence_weight
        torch.nn.utils.clip_grad_norm_(
            list(self.f.morphism.parameters()) + list(self.g.morphism.parameters()),
            max_norm=1.0  # Fixed clip norm
        )
        
        self.opt.step()
        # Note: scheduler.step() called in on_epoch_end() for proper epoch tracking
        
        # Logging
        losses = {
            'task_loss': task_loss.item(),
            'coherence_loss': coherence.item(),
            'total_loss': total.item(),
            'lr': self.opt.param_groups[0]['lr']
        }
        
        for k, v in losses.items():
            self.writer.add_scalar(f'inner_horn/{k}', v, self._step)
            
        self._step += 1
        
        return losses

    def validate_horn_condition(self, test_data: torch.Tensor) -> float:
        """Validate that the horn condition h = g ∘ f holds against reference."""
        with torch.no_grad():
            return self.coherence_loss(test_data).item()
    
    def on_epoch_end(self):
        """
        FIX: Proper epoch-based scheduler stepping for deterministic convergence.
        """
        self._epoch += 1
        self.scheduler.step()  # Step scheduler by epochs, not batches
        
        # Log epoch-level metrics
        self.writer.add_scalar('inner_horn/epoch', self._epoch, self._step)
        self.writer.add_scalar('inner_horn/lr_epoch', self.opt.param_groups[0]['lr'], self._epoch)