"""
Module: outer_solver
Implements the UniversalLiftingSolver for outer horn filling.

Following Mahadevan (2024), this implements the Outer Horn Λ²₀ solver
using the full Metric Yoneda embedding to minimize the lifting loss.
"""

import datetime as _dt
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from gaia.core.simplices import Simplex1
from gaia.core import DEVICE
from .yoneda_proxy import MetricYonedaProxy

class UniversalLiftingSolver:
    def __init__(
        self,
        f: Simplex1,
        k: Simplex1,
        yoneda: MetricYonedaProxy,
        lr: float = 1e-3,
        writer: Optional[SummaryWriter] = None
    ):
        self.f = f.to(DEVICE).eval()
        self.k = k.to(DEVICE).eval()
        self.yoneda = yoneda
        
        # filler morphism
        self.m_filler = Simplex1(
            nn.Linear(f.codomain.dim, k.codomain.dim).to(DEVICE),
            f.codomain,
            k.codomain,
            'm_filler'
        )
        self.m_filler.morphism.train()
        
        # optimizers: one for filler, one for yoneda (adaptive)
        self.opt = optim.Adam(self.m_filler.morphism.parameters(), lr=lr)
        self.optimizer = self.opt  # Add alias for compatibility
        self.yoneda_proxy = self.yoneda  # Add alias for compatibility
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=250)
        
        self.writer = writer or SummaryWriter(
            log_dir=f'runs/outer_{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        self._step = 0

    
    def verify_universality(self, X_A: torch.Tensor, epsilon: float = 1e-4) -> Dict[str, float]:
        """
        Verify the universality property of the lifting solution.
        
        Following Mahadevan (2024), this checks that:
        1. m ∘ f ≈ k (the lifting equation)
        2. For any other solution m', ||Y_C(m(f(x))) - Y_C(k(x))|| ≤ ||Y_C(m'(f(x))) - Y_C(k(x))||
        
        Returns a dictionary with verification metrics.
        """
        with torch.no_grad():
            f_out = self.f(X_A)
            k_out = self.k(X_A)
            m_out = self.m_filler(f_out)
            
            # Direct error
            direct_error = nn.MSELoss()(m_out, k_out)
            
            # Yoneda error (should be minimal)
            yoneda_error = self.yoneda.loss(m_out, k_out)
            
            # Create a perturbed solution to verify minimality
            perturbed = Simplex1(
                nn.Linear(self.f.codomain.dim, self.k.codomain.dim).to(DEVICE),
                self.f.codomain,
                self.k.codomain,
                'm_perturbed'
            )
            # Copy weights and add small perturbation
            with torch.no_grad():
                for p_src, p_dst in zip(self.m_filler.morphism.parameters(), 
                                        perturbed.morphism.parameters()):
                    p_dst.copy_(p_src + epsilon * torch.randn_like(p_src))
                    
            perturbed_out = perturbed(f_out)
            perturbed_yoneda_error = self.yoneda.loss(perturbed_out, k_out)
            
            # Universality requires our solution to be minimal
            is_universal = yoneda_error <= perturbed_yoneda_error
            
            return {
                "direct_error": direct_error.item(),
                "yoneda_error": yoneda_error.item(),
                "perturbed_yoneda_error": perturbed_yoneda_error.item(),
                "is_universal": is_universal.item() if isinstance(is_universal, torch.Tensor) else is_universal,
                "universality_gap": (perturbed_yoneda_error - yoneda_error).item()
            }
    
    # Update the solve method to use DataLoader for streaming
    def solve(self, D_A: torch.utils.data.Dataset, max_steps: int = 1000, 
             early_stopping: bool = True, patience: int = 50, min_delta: float = 1e-4,
             batch_size: int = 32, seed: int = 42) -> dict:
         """
         Solve the universal lifting problem.
         
         Args:
             D_A: Dataset of points in the domain of f
             max_steps: Maximum number of optimization steps
             early_stopping: Whether to use early stopping
             patience: Number of steps without improvement before stopping
             min_delta: Minimum change in loss to count as improvement
             batch_size: Batch size for DataLoader
             seed: Random seed for deterministic behavior
             
         Returns:
             Dictionary with optimization results
         """
         # Set models to appropriate modes and fix random seed for deterministic behavior
         self.k.eval()
         self.f.eval()
         self.m_filler.morphism.train()  # Ensure filler is in training mode
         torch.random.manual_seed(seed)
         
         # Create DataLoader for streaming data
         data_loader = torch.utils.data.DataLoader(D_A, batch_size=batch_size, shuffle=True)
         
         # Initialize tracking variables
         best_loss = float('inf')
         best_state = None
         steps_without_improvement = 0
         losses = []
         
         # Optimization loop
         for step in range(max_steps):
             # Get next batch (with cycling if needed)
             try:
                 xb, = next(data_loader_iter)
             except (StopIteration, NameError):
                 data_loader_iter = iter(data_loader)
                 xb, = next(data_loader_iter)
             
             # Compute target values for this batch
             with torch.no_grad():
                 kb = self.k(xb)  # Cache k(x) per batch
                 fb = self.f(xb)  # Cache f(x) per batch
             
             # Compute lifting loss with gradients enabled
             self.optimizer.zero_grad()
             mb_fb = self.m_filler(fb)
             loss = self.yoneda_proxy.loss(mb_fb, kb)
             
             # Update
             loss.backward()
             self.optimizer.step()
             self.scheduler.step()
             
             # Track loss
             losses.append(loss.item())
             
             # Check for improvement
             if loss.item() < best_loss - min_delta:
                 best_loss = loss.item()
                 best_state = self.m_filler.morphism.state_dict().copy()  # Fix: use .morphism.state_dict()
                 steps_without_improvement = 0
             else:
                 steps_without_improvement += 1
             
             # Early stopping
             if early_stopping and steps_without_improvement >= patience:
                 print(f"Early stopping at step {step} with loss {best_loss:.6f}")
                 break
         
         # Restore best state
         if best_state is not None:
             self.m_filler.morphism.load_state_dict(best_state)  # Fix: use .morphism.load_state_dict()
         
         return {
             "loss": best_loss,
             "steps": step + 1,
             "losses": losses,
             "early_stopped": step + 1 < max_steps and early_stopping,
         }
    
    # REMOVE the duplicate solve method from lines 130-246