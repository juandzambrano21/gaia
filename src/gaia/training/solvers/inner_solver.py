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
        
        # Optimizer with gradient clipping for stability
        self.opt = optim.Adam(
            list(self.f.morphism.parameters()) + list(self.g.morphism.parameters()),
            lr=lr, weight_decay=1e-5
        )
        
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