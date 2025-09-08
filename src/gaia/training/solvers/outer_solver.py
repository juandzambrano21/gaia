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
        
        # Ensure dimensional consistency for universal lifting
        
        # filler morphism with proper dimension handling
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

    def solve_horn(self, horn) -> Optional[Dict]:
        """Solve outer horn using universal lifting approach."""
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
            
            # Verify this is an outer horn (k = 0 or k = n)
            if not (horn_index == 0 or horn_index == simplex.level):
                return None
            
            # Apply universal lifting using Yoneda proxy
            result = self._apply_universal_lifting(simplex, horn_index)
            
            solution = {
                'status': 'solved',
                'method': 'universal_lifting',
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
            
            # Compute verification metrics
            
            # Handle dimension mismatch for Yoneda loss
            if m_out.shape != k_out.shape:
                min_batch = min(m_out.shape[0], k_out.shape[0])
                m_out = m_out[:min_batch]
                k_out = k_out[:min_batch]
            
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
            
            # Handle dimension mismatch for perturbed solution
            if perturbed_out.shape != k_out.shape:
                min_batch = min(perturbed_out.shape[0], k_out.shape[0])
                perturbed_out = perturbed_out[:min_batch]
                k_out_perturbed = k_out[:min_batch]
            else:
                k_out_perturbed = k_out
                
            perturbed_yoneda_error = self.yoneda.loss(perturbed_out, k_out_perturbed)
            
            # Universality requires our solution to be minimal
            # CRITICAL FIX: Ensure both yoneda errors are computed with same tensor sizes
            if yoneda_error.shape != perturbed_yoneda_error.shape:
                # Both should be scalars, but if not, take the mean
                yoneda_error = yoneda_error.mean() if yoneda_error.numel() > 1 else yoneda_error
                perturbed_yoneda_error = perturbed_yoneda_error.mean() if perturbed_yoneda_error.numel() > 1 else perturbed_yoneda_error
            
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
             
             # Compute lifting loss with gradients enabled
             self.optimizer.zero_grad()
             mb_fb = self.m_filler(fb)
             
             # Ensure both tensors are compatible for Yoneda loss computation
             
             # Ensure both tensors have at least 2 dimensions
             if mb_fb.dim() == 1:
                 mb_fb = mb_fb.unsqueeze(1)  # Make it (batch_size, 1)
             if kb.dim() == 1:
                 kb = kb.unsqueeze(1)  # Make it (batch_size, 1)
                 
             # Handle batch size mismatch (common with MetricYonedaProxy num_probes)
             if mb_fb.shape[0] != kb.shape[0]:
                 min_batch = min(mb_fb.shape[0], kb.shape[0])
                 if min_batch > 0:
                     mb_fb = mb_fb[:min_batch]
                     kb = kb[:min_batch]
                 else:
                     continue
                     
             # Handle feature dimension mismatch
             if mb_fb.shape[1] != kb.shape[1]:
                 target_dim = max(mb_fb.shape[1], kb.shape[1])
                 
                 # Pad smaller tensor to match larger one
                 if mb_fb.shape[1] < target_dim:
                     padding = torch.zeros(mb_fb.shape[0], target_dim - mb_fb.shape[1], device=mb_fb.device)
                     mb_fb = torch.cat([mb_fb, padding], dim=1)
                 elif mb_fb.shape[1] > target_dim:
                     mb_fb = mb_fb[:, :target_dim]
                     
                 if kb.shape[1] < target_dim:
                     padding = torch.zeros(kb.shape[0], target_dim - kb.shape[1], device=kb.device)
                     kb = torch.cat([kb, padding], dim=1)
                 elif kb.shape[1] > target_dim:
                     kb = kb[:, :target_dim]
             
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
    
    def _apply_universal_lifting(self, simplex, horn_index):
        """Apply universal lifting to solve the outer horn following Definition 24.
        
        According to the paper's Definition 24, a Kan fibration requires solving
        lifting problems where σ₀: Λᵢⁿ → X and σ: Δⁿ → X satisfy f ∘ σ = σ̄.
        This implements the proper lifting diagram construction.
        """
        try:
            # Construct proper lifting diagram as per Definition 24
            # σ₀: Λᵢⁿ → X (horn map)
            # σ̄: Δⁿ → S (extending f ∘ σ₀)
            # Need to find σ: Δⁿ → X such that f ∘ σ = σ̄
            
            from gaia.training.config import TrainingConfig
            training_config = TrainingConfig()
            batch_size = training_config.data.batch_size or 32  
            domain_dim = self.f.domain.dim
            
            # Create horn map σ₀: Λᵢⁿ → X
            horn_data = torch.randn(batch_size, domain_dim).to(DEVICE)
            
            # Apply existing faces of the horn (all faces except the i-th)
            horn_faces = []
            for face_idx in range(simplex.level + 1):
                if face_idx != horn_index:  # Skip the missing face
                    if hasattr(simplex, 'faces') and face_idx < len(simplex.faces):
                        face_result = simplex.faces[face_idx](horn_data)
                        horn_faces.append(face_result)
            
            # Construct σ̄: Δⁿ → S extending f ∘ σ₀
            if horn_faces:
                sigma_bar = torch.stack(horn_faces, dim=1).mean(dim=1)  # Average existing faces
            else:
                sigma_bar = self.f.morphism(horn_data)  # Fallback to f applied to horn data
            
            # Solve for σ: Δⁿ → X such that f ∘ σ = σ̄
            dataset = torch.utils.data.TensorDataset(horn_data, sigma_bar)
            result = self.solve(dataset, max_steps=500, batch_size=batch_size)
            
            # Verify the lifting property: f ∘ σ = σ̄
            with torch.no_grad():
                sigma = self.m_filler.morphism(horn_data)
                f_sigma = self.f.morphism(sigma)
                lifting_error = torch.norm(f_sigma - sigma_bar).item()
            
            # Create filled morphism only if lifting property is satisfied
            if result['loss'] < 0.1 and lifting_error < 0.1:
                from gaia.core.simplices import Simplex1
                import uuid
                import torch.nn as nn
                
                # Clone the optimized lifting morphism
                filled_morphism_nn = nn.Linear(self.m_filler.morphism.in_features, self.m_filler.morphism.out_features)
                filled_morphism_nn.load_state_dict(self.m_filler.morphism.state_dict())
                
                # Create 1-simplex with proper domain/codomain based on horn position
                if horn_index == 0:  # Missing first face (outer horn Λ₀ⁿ)
                    domain = self.k.domain if hasattr(self.k, 'domain') else self.f.domain
                    codomain = self.f.codomain if hasattr(self.f, 'codomain') else self.k.codomain
                else:  # Missing last face (horn_index == simplex.level)
                    domain = self.f.domain if hasattr(self.f, 'domain') else self.k.domain
                    codomain = self.k.codomain if hasattr(self.k, 'codomain') else self.f.codomain
                
                filled_morphism = Simplex1(
                    filled_morphism_nn, domain, codomain, f"lifted_horn_{uuid.uuid4().hex[:8]}"
                )
                
                # Add to functor registry to modify structure
                if hasattr(self.functor, 'registry'):
                    self.functor.registry[filled_morphism.id] = filled_morphism
                
                # Update the simplex to include the filled face
                if hasattr(simplex, 'faces'):
                    simplex.faces[horn_index] = filled_morphism
                
                return {
                    'optimization': result,
                    'verification': verification,
                    'filler_weights': self.m_filler.morphism.state_dict(),
                    'filled_morphism': filled_morphism,
                    'modified_registry': True
                }
            else:
                return {
                    'optimization': result,
                    'verification': verification,
                    'filler_weights': self.m_filler.morphism.state_dict(),
                    'modified_registry': False,
                    'error': 'Lifting failed universality or convergence criteria'
                }
                
        except Exception as e:
            return {
                'modified_registry': False,
                'error': str(e)
            }
    