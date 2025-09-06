"""Training and Validation Loops for GAIA Framework

Implements the complete GAIA hierarchical learning framework based on:
- Horn extension solving (inner Λ²₁ and outer Λ²₀)
- Lifting problem detection and resolution
- Categorical structure verification
- Simplicial learning algorithms
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple
from torch.utils.data import DataLoader
import time
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseLoop(ABC):
    """Base class for GAIA training loops"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Any,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        self.model = model
        self.device = device
        self.config = config
        self.metrics = metrics or {}
        self.step_count = 0
        
    @abstractmethod
    def run_epoch(self, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        """Run one epoch of the loop"""
        pass
        
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for the current batch"""
        results = {}
        for name, metric_fn in self.metrics.items():
            try:
                results[name] = float(metric_fn(outputs, targets))
            except Exception as e:
                logger.warning(f"Failed to compute metric {name}: {e}")
        return results

class TrainingLoop(BaseLoop):
    """GAIA Training loop with complete categorical structure and horn extension solving"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Any,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        gradient_clip: Optional[float] = None,
        accumulation_steps: int = 1,
        categorical_loss_weight: float = 0.3,
        coherence_loss_weight: float = 0.3,
        horn_solving_weight: float = 0.3,
        metrics: Optional[Dict[str, Callable]] = None,
        verify_categorical_structure: bool = True,
        verify_kan_fibration: bool = True
    ):
        super().__init__(model, device, config, metrics)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.categorical_loss_weight = categorical_loss_weight
        self.coherence_loss_weight = coherence_loss_weight
        self.horn_solving_weight = horn_solving_weight
        self.verify_categorical_structure = verify_categorical_structure
        self.verify_kan_fibration = verify_kan_fibration
        
        # GAIA-specific components
        self.horn_solvers = {}
        self.lifting_problems = []
        self.categorical_violations = []
        
    def _detect_horn_problems(self, batch_data: torch.Tensor) -> Dict[str, List]:
        """Detect horn extension problems in the current batch"""
        horn_problems = {'inner_horns': [], 'outer_horns': []}
        
        if hasattr(self.model, 'get_horn_problems'):
            try:
                detected = self.model.get_horn_problems()
                horn_problems.update(detected)
            except Exception as e:
                logger.warning(f"Horn problem detection failed: {e}")
                
        return horn_problems
    
    def _solve_lifting_problems(self, horn_problems: Dict[str, List], batch_data: torch.Tensor) -> Dict[str, float]:
        lifting_losses = {
            # Replace: torch.tensor(0.0, device=self.device)
            'inner_horn_loss': torch.zeros(1, device=self.device, requires_grad=True),
            'outer_horn_loss': torch.zeros(1, device=self.device, requires_grad=True),
        }
        
        # Solve inner horns using endofunctorial solver
        if horn_problems['inner_horns'] and hasattr(self.model, 'solve_inner_horns'):
            try:
                inner_result = self.model.solve_inner_horns(horn_problems['inner_horns'], batch_data)
                lifting_losses['inner_horn_loss'] = inner_result.get('loss', torch.zeros(1, device=self.device, requires_grad=True))
                lifting_losses['lifting_violations'] += inner_result.get('violations', 0)
            except Exception as e:
                logger.warning(f"Inner horn solving failed: {e}")
                # Add penalty for failed inner horn solving
                lifting_losses['inner_horn_loss'] = torch.tensor(0.1, device=self.device)
                lifting_losses['lifting_violations'] += len(horn_problems['inner_horns'])
        
        # Solve outer horns using universal lifting solver
        if horn_problems['outer_horns'] and hasattr(self.model, 'solve_outer_horns'):
            try:
                outer_result = self.model.solve_outer_horns(horn_problems['outer_horns'], batch_data)
                lifting_losses['outer_horn_loss'] = outer_result.get('loss', torch.zeros(1, device=self.device, requires_grad=True))
                lifting_losses['lifting_violations'] += outer_result.get('violations', 0)
            except Exception as e:
                logger.warning(f"Outer horn solving failed: {e}")
                # Add penalty for failed outer horn solving
                lifting_losses['outer_horn_loss'] = torch.tensor(0.2, device=self.device)
                lifting_losses['lifting_violations'] += len(horn_problems['outer_horns'])
        
        return lifting_losses
    
    def _hierarchical_update(self, total_loss: torch.Tensor, lifting_losses: Dict[str, float]) -> None:
        """Perform hierarchical updates according to GAIA's simplicial learning algorithm"""
        # Standard gradient computation
        total_loss.backward(retain_graph=True)
        
        # GAIA hierarchical updates: solve lifting problems at different simplicial levels
        if self.config.hierarchical_learning:
            # Level 0: Object updates (standard neural network parameters)
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Level 1: Morphism updates (connections between layers)
            if hasattr(self.model, 'update_morphisms'):
                try:
                    self.model.update_morphisms(lifting_losses)
                except Exception as e:
                    logger.warning(f"Morphism update failed: {e}")
            
            # Level 2: Triangle updates (compositional coherence)
            if hasattr(self.model, 'update_triangles'):
                try:
                    self.model.update_triangles(lifting_losses)
                except Exception as e:
                    logger.warning(f"Triangle update failed: {e}")
    
    def run_epoch(self, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        """Run one GAIA training epoch with complete categorical structure"""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'categorical_loss': 0.0,
            'coherence_loss': 0.0,
            'inner_horn_loss': 0.0,
            'outer_horn_loss': 0.0,
            'lifting_violations': 0,
            'horn_problems_detected': 0,
            'lr': 0.0
        }
        
        num_batches = len(dataloader)
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None
            
            # GAIA Step 1: Detect horn extension problems
            horn_problems = self._detect_horn_problems(inputs)
            epoch_metrics['horn_problems_detected'] += len(horn_problems['inner_horns']) + len(horn_problems['outer_horns'])
            
            # Forward pass with mixed precision if available
            if self.scaler is not None:
                # Use device-appropriate autocast
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        
                        # Compute main loss
                        if targets is not None:
                            main_loss = self.criterion(outputs, targets)
                        else:
                            main_loss = torch.zeros(1, device=self.device, requires_grad=True)
                elif self.device.type == 'mps':
                    # MPS doesn't support autocast, use regular forward pass
                    outputs = self.model(inputs)
                    
                    # Compute main loss
                    if targets is not None:
                        main_loss = self.criterion(outputs, targets)
                    else:
                        main_loss = torch.zeros(1, device=self.device, requires_grad=True)
                else:
                    # CPU or other devices
                    with torch.cpu.amp.autocast() if self.device.type == 'cpu' else torch.no_grad():
                        outputs = self.model(inputs)
                        
                        # Compute main loss
                        if targets is not None:
                            main_loss = self.criterion(outputs, targets)
                        else:
                            main_loss = torch.zeros(1, device=self.device, requires_grad=True)
            else:
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute main loss
                if targets is not None:
                    main_loss = self.criterion(outputs, targets)
                else:
                    main_loss = torch.zeros(1, device=self.device, requires_grad=True)
            
            # GAIA Step 2: Compute categorical losses
            categorical_loss = torch.tensor(0.0, device=self.device)
            coherence_loss = torch.tensor(0.0, device=self.device)
            
            if hasattr(self.model, 'compute_categorical_loss'):
                categorical_loss = self.model.compute_categorical_loss(inputs)
                
            if hasattr(self.model, 'verify_coherence') and self.verify_categorical_structure:
                coherence_info = self.model.verify_coherence()
                if 'coherence_loss' in coherence_info:
                    coherence_loss = coherence_info['coherence_loss']
            
            # GAIA Step 3: Solve lifting problems
            lifting_losses = self._solve_lifting_problems(horn_problems, inputs)
            
            # GAIA Step 4: Compute total loss with hierarchical components
            total_loss = (
                main_loss + 
                self.categorical_loss_weight * categorical_loss +
                self.coherence_loss_weight * coherence_loss +
                self.horn_solving_weight * (lifting_losses['inner_horn_loss'] + lifting_losses['outer_horn_loss'])
            ) / self.accumulation_steps
            
            # GAIA Step 5: Hierarchical backward pass and updates
            self._hierarchical_update(total_loss, lifting_losses)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
            
            # Update metrics
            epoch_metrics['loss'] += total_loss.item() * self.accumulation_steps
            epoch_metrics['categorical_loss'] += categorical_loss.item()
            epoch_metrics['coherence_loss'] += coherence_loss.item()
            epoch_metrics['inner_horn_loss'] += lifting_losses['inner_horn_loss'].item() if torch.is_tensor(lifting_losses['inner_horn_loss']) else lifting_losses['inner_horn_loss']
            epoch_metrics['outer_horn_loss'] += lifting_losses['outer_horn_loss'].item() if torch.is_tensor(lifting_losses['outer_horn_loss']) else lifting_losses['outer_horn_loss']
            epoch_metrics['lifting_violations'] += lifting_losses['lifting_violations']
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # Compute additional metrics
            if targets is not None:
                batch_metrics = self.compute_metrics(outputs, targets)
                for name, value in batch_metrics.items():
                    if name not in epoch_metrics:
                        epoch_metrics[name] = 0.0
                    epoch_metrics[name] += value
            
            self.step_count += 1
        
        # Average metrics
        for key in epoch_metrics:
            if key not in ['lr', 'lifting_violations', 'horn_problems_detected']:
                epoch_metrics[key] /= num_batches
                
        return epoch_metrics

class ValidationLoop(BaseLoop):
    """GAIA Validation loop with complete categorical structure verification"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        config: Any,
        metrics: Optional[Dict[str, Callable]] = None,
        verify_categorical_structure: bool = True,
        verify_kan_fibration: bool = True
    ):
        super().__init__(model, device, config, metrics)
        self.criterion = criterion
        self.verify_categorical_structure = verify_categorical_structure
        self.verify_kan_fibration = verify_kan_fibration
        
    def val_step(self, batch, state=None) -> Dict[str, float]:
            """Perform a single validation step"""
            self.model.eval()
            
            with torch.no_grad():
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute main loss
                if targets is not None:
                    main_loss = self.criterion(outputs, targets)
                else:
                    main_loss = torch.tensor(0.0, device=self.device)
                
                # Compute categorical losses
                categorical_loss = torch.tensor(0.0, device=self.device)
                coherence_loss = torch.tensor(0.0, device=self.device)
                
                if hasattr(self.model, 'compute_categorical_loss'):
                    categorical_loss = self.model.compute_categorical_loss(inputs)
                    
                if hasattr(self.model, 'verify_coherence') and self.verify_categorical_structure:
                    coherence_info = self.model.verify_coherence()
                    if 'coherence_loss' in coherence_info:
                        coherence_loss = coherence_info['coherence_loss']
                
                # Prepare metrics
                metrics = {
                    'val_loss': main_loss.item(),
                    'val_categorical_loss': categorical_loss.item(),
                    'val_coherence_loss': coherence_loss.item()
                }
                
                # Compute additional metrics
                if targets is not None:
                    batch_metrics = self.compute_metrics(outputs, targets)
                    for name, value in batch_metrics.items():
                        metrics[f'val_{name}'] = value
                
                # GAIA categorical structure verification
                if self.verify_kan_fibration:
                    kan_metrics = self._verify_kan_fibration(inputs)
                    metrics.update(kan_metrics)
                
                return metrics
        
    def _verify_kan_fibration(self, batch_data: torch.Tensor) -> Dict[str, float]:
        """Verify that the model satisfies Kan fibration properties"""
        kan_metrics = {
            'kan_fibration_valid': 1.0,
            'horn_extension_failures': 0.0,
            'lifting_property_violations': 0.0
        }
        
        if hasattr(self.model, 'verify_kan_fibration'):
            try:
                kan_result = self.model.verify_kan_fibration(batch_data)
                kan_metrics.update(kan_result)
            except Exception as e:
                logger.warning(f"Kan fibration verification failed: {e}")
                kan_metrics['kan_fibration_valid'] = 0.0
                
        return kan_metrics
    
    def _verify_simplicial_identities(self) -> Dict[str, float]:
        """Verify all simplicial identities hold"""
        identity_metrics = {
            'simplicial_identities_valid': 1.0,
            'face_map_violations': 0.0,
            'degeneracy_violations': 0.0,
            'composition_violations': 0.0
        }
        
        if hasattr(self.model, 'verify_simplicial_identities'):
            try:
                identity_result = self.model.verify_simplicial_identities()
                if isinstance(identity_result, dict):
                    identity_metrics.update(identity_result)
                else:
                    identity_metrics['simplicial_identities_valid'] = float(identity_result)
            except Exception as e:
                logger.warning(f"Simplicial identity verification failed: {e}")
                identity_metrics['simplicial_identities_valid'] = 0.0
                
        return identity_metrics
        
    def run_epoch(self, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        """Run one GAIA validation epoch with complete categorical verification"""
        self.model.eval()
        epoch_metrics = {
            'val_loss': 0.0,
            'val_categorical_loss': 0.0,
            'val_coherence_loss': 0.0,
            'coherence_violation_rate': 0.0,
            'kan_fibration_valid': 1.0,
            'simplicial_identities_valid': 1.0,
            'horn_extension_failures': 0.0,
            'categorical_structure_score': 1.0
        }
        
        num_batches = len(dataloader)
        coherence_violations = 0
        total_horn_failures = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute main loss
                if targets is not None:
                    main_loss = self.criterion(outputs, targets)
                else:
                    main_loss = torch.tensor(0.0, device=self.device)
                
                # Compute categorical losses
                categorical_loss = torch.tensor(0.0, device=self.device)
                coherence_loss = torch.tensor(0.0, device=self.device)
                
                if hasattr(self.model, 'compute_categorical_loss'):
                    categorical_loss = self.model.compute_categorical_loss()
                    
                if hasattr(self.model, 'verify_coherence') and self.verify_categorical_structure:
                    coherence_info = self.model.verify_coherence()
                    if 'coherence_loss' in coherence_info:
                        coherence_loss = coherence_info['coherence_loss']
                    if not coherence_info.get('is_coherent', True):
                        coherence_violations += 1
                
                # GAIA categorical structure verification
                if self.verify_kan_fibration:
                    kan_metrics = self._verify_kan_fibration(inputs)
                    for key, value in kan_metrics.items():
                        if key not in epoch_metrics:
                            epoch_metrics[key] = 0.0
                        epoch_metrics[key] += value
                    total_horn_failures += kan_metrics.get('horn_extension_failures', 0)
                
                # Update metrics
                epoch_metrics['val_loss'] += main_loss.item()
                epoch_metrics['val_categorical_loss'] += categorical_loss.item()
                epoch_metrics['val_coherence_loss'] += coherence_loss.item()
                
                # Compute additional metrics
                if targets is not None:
                    batch_metrics = self.compute_metrics(outputs, targets)
                    for name, value in batch_metrics.items():
                        val_name = f'val_{name}'
                        if val_name not in epoch_metrics:
                            epoch_metrics[val_name] = 0.0
                        epoch_metrics[val_name] += value
        
        
        # Verify simplicial identities (once per epoch)
        if self.verify_categorical_structure:
            identity_metrics = self._verify_simplicial_identities()
            epoch_metrics.update(identity_metrics)
        
        # Average metrics
        for key in epoch_metrics:
            if key not in ['simplicial_identities_valid']:
                epoch_metrics[key] /= num_batches
        
        # Compute derived metrics
        epoch_metrics['coherence_violation_rate'] = coherence_violations / num_batches
        epoch_metrics['horn_extension_failure_rate'] = total_horn_failures / num_batches
        
        # Overall categorical structure score
        epoch_metrics['categorical_structure_score'] = (
            epoch_metrics['kan_fibration_valid'] * 
            epoch_metrics['simplicial_identities_valid'] * 
            (1.0 - epoch_metrics['coherence_violation_rate']) *
            (1.0 - epoch_metrics['horn_extension_failure_rate'])
        )
        
        return epoch_metrics