"""Lifting Diagram Trainer - GAIA Training with Categorical Updates

Following Mahadevan (2024), this implements training loops that use lifting diagrams
for parameter updates instead of traditional gradient descent, integrating with
the horn extension learning framework.

Key Features from Paper:
- Parameter updates as lifting diagrams over simplicial sets
- Integration with horn extension learning (inner/outer horns)
- Hierarchical learning across simplicial dimensions
- Categorical optimization preserving simplicial structure
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import logging
from dataclasses import dataclass

from ..core.lifting_diagram_optimizer import LiftingDiagramOptimizer, ParameterLiftingProblem
from ..core.horn_extension_learning import HornExtensionSolver, HornExtensionProblem
from ..core.simplices import BasisRegistry
from ..nn import GAIAModule
from ..utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class LiftingTrainingConfig:
    """Configuration for lifting diagram-based training."""
    max_simplicial_dimension: int = 3
    learning_rate: float = 0.001
    use_horn_extensions: bool = True
    use_kan_fibrations: bool = True
    coordinate_hierarchical_learning: bool = True
    lifting_solver_type: str = "adaptive"  # adaptive, gradient, horn, kan, fibration
    

class LiftingDiagramTrainer(GAIAModule):
    """Trainer using lifting diagrams for parameter updates.
    
    This replaces traditional gradient-based training with categorical
    lifting diagram-based updates, implementing the complete GAIA
    hierarchical learning framework from the paper.
    
    Mathematical Foundation:
        Following Mahadevan (2024) Section 4.2, training is formulated as
        solving lifting problems in the category of simplicial sets:
        
        For each parameter update, we solve:
        Given fibration p: E → B and current state f: A → B,
        find lifting h: B → X that provides the categorical update.
        
        This preserves the simplicial structure while enabling both
        inner horn (traditional) and outer horn (advanced) learning.
    
    Architecture:
        - Lifting diagram optimizer: Replaces traditional optimizers
        - Horn extension solver: Handles hierarchical learning problems
        - Simplicial coordinator: Maintains categorical structure
        - Training loop: Integrates lifting with loss computation
    
    Args:
        model: GAIA model to train
        config: Lifting training configuration
        basis_registry: Simplicial basis registry from Layer 1
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: LiftingTrainingConfig,
                 basis_registry: Optional[BasisRegistry] = None):
        super().__init__()
        
        self.model = model
        self.config = config
        self.basis_registry = basis_registry or BasisRegistry()
        
        # Lifting diagram optimizer (replaces traditional optimizer)
        self.lifting_optimizer = LiftingDiagramOptimizer(
            max_dimension=config.max_simplicial_dimension,
            basis_registry=self.basis_registry,
            learning_rate=config.learning_rate,
            use_kan_fibrations=config.use_kan_fibrations
        )
        
        # Horn extension solver for hierarchical learning
        if config.use_horn_extensions:
            self.horn_solver = HornExtensionSolver(
                max_dimension=config.max_simplicial_dimension,
                basis_registry=self.basis_registry,
                learning_rate=config.learning_rate,
                use_lifting_diagrams=True
            )
        else:
            self.horn_solver = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.training_history = []
        
        # Set GAIA metadata
        self.set_gaia_metadata(
            training_method="lifting_diagrams",
            replaces_gradient_descent=True,
            horn_extension_learning=config.use_horn_extensions,
            hierarchical_learning=config.coordinate_hierarchical_learning,
            max_simplicial_dimension=config.max_simplicial_dimension
        )
    
    def train_step(self,
                  batch: Dict[str, torch.Tensor],
                  loss_fn: Callable,
                  simplicial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform one training step using lifting diagrams.
        
        This replaces the traditional forward-backward-step cycle with
        categorical lifting diagram-based parameter updates.
        
        Args:
            batch: Training batch data
            loss_fn: Loss function to compute training loss
            simplicial_context: Simplicial structure context
            
        Returns:
            Dictionary containing loss, metrics, and lifting statistics
        """
        self.model.train()
        
        # Extract simplicial context or create default
        context = simplicial_context or self._create_default_context()
        
        # Forward pass with simplicial context
        outputs = self._forward_with_context(batch, context)
        
        # Compute loss
        loss = loss_fn(outputs, batch)
        
        # Create horn extension problems if enabled
        horn_problems = []
        if self.horn_solver and self.config.use_horn_extensions:
            horn_problems = self._create_horn_problems(loss, context)
        
        # Compute gradients (traditional or horn-based)
        gradients = self._compute_gradients(loss, horn_problems, context)
        
        # Create lifting problems from gradients
        lifting_problems = self._create_lifting_problems(gradients, context)
        
        # Solve lifting problems for parameter updates
        parameter_updates = self._solve_lifting_problems(lifting_problems)
        
        # Apply categorical parameter updates
        self._apply_parameter_updates(parameter_updates)
        
        # Update training state
        self.current_step += 1
        
        # Collect training statistics
        stats = {
            'loss': loss.item(),
            'num_horn_problems': len(horn_problems),
            'num_lifting_problems': len(lifting_problems),
            'simplicial_dimension': context.get('dimension', 1),
            'step': self.current_step
        }
        
        self.training_history.append(stats)
        
        return stats
    
    def train_epoch(self,
                   dataloader,
                   loss_fn: Callable,
                   simplicial_context_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Train for one epoch using lifting diagrams.
        
        Args:
            dataloader: Training data loader
            loss_fn: Loss function
            simplicial_context_fn: Function to generate simplicial context per batch
            
        Returns:
            Epoch training statistics
        """
        epoch_stats = {
            'total_loss': 0.0,
            'num_batches': 0,
            'total_horn_problems': 0,
            'total_lifting_problems': 0
        }
        
        for batch_idx, batch in enumerate(dataloader):
            # Generate simplicial context for this batch
            if simplicial_context_fn:
                context = simplicial_context_fn(batch, batch_idx, self.current_epoch)
            else:
                context = self._create_adaptive_context(batch_idx)
            
            # Perform training step
            step_stats = self.train_step(batch, loss_fn, context)
            
            # Accumulate statistics
            epoch_stats['total_loss'] += step_stats['loss']
            epoch_stats['total_horn_problems'] += step_stats['num_horn_problems']
            epoch_stats['total_lifting_problems'] += step_stats['num_lifting_problems']
            epoch_stats['num_batches'] += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                          f"Loss={step_stats['loss']:.4f}, "
                          f"Horn Problems={step_stats['num_horn_problems']}, "
                          f"Lifting Problems={step_stats['num_lifting_problems']}")
        
        # Compute epoch averages
        if epoch_stats['num_batches'] > 0:
            epoch_stats['avg_loss'] = epoch_stats['total_loss'] / epoch_stats['num_batches']
            epoch_stats['avg_horn_problems'] = epoch_stats['total_horn_problems'] / epoch_stats['num_batches']
            epoch_stats['avg_lifting_problems'] = epoch_stats['total_lifting_problems'] / epoch_stats['num_batches']
        
        self.current_epoch += 1
        
        return epoch_stats
    
    def _forward_with_context(self, 
                            batch: Dict[str, torch.Tensor], 
                            context: Dict[str, Any]) -> torch.Tensor:
        """Forward pass with simplicial context."""
        # Check if model supports simplicial context
        if hasattr(self.model, 'forward') and 'simplicial_context' in self.model.forward.__code__.co_varnames:
            return self.model(batch['input'], simplicial_context=context)
        else:
            return self.model(batch['input'])
    
    def _create_default_context(self) -> Dict[str, Any]:
        """Create default simplicial context."""
        return {
            'dimension': 1,
            'dimension_weights': torch.tensor([0.3, 0.7, 0.0, 0.0]),  # Favor 1-simplices
            'horn_problems': [
                {'horn_type': 'inner', 'dimension': 1},
                {'horn_type': 'outer', 'dimension': 2}
            ]
        }
    
    def _create_adaptive_context(self, batch_idx: int) -> Dict[str, Any]:
        """Create adaptive simplicial context based on training progress."""
        # Gradually increase complexity during training
        progress = min(1.0, self.current_step / 1000.0)
        
        # Start with simple 1-simplices, gradually add higher dimensions
        if progress < 0.3:
            dimension = 1
            weights = torch.tensor([0.2, 0.8, 0.0, 0.0])
        elif progress < 0.7:
            dimension = 2
            weights = torch.tensor([0.1, 0.5, 0.4, 0.0])
        else:
            dimension = 3
            weights = torch.tensor([0.1, 0.3, 0.4, 0.2])
        
        return {
            'dimension': dimension,
            'dimension_weights': weights,
            'horn_problems': [
                {'horn_type': 'inner', 'dimension': min(dimension, 2)},
                {'horn_type': 'outer', 'dimension': dimension}
            ],
            'training_progress': progress
        }
    
    def _create_horn_problems(self, 
                            loss: torch.Tensor, 
                            context: Dict[str, Any]) -> List[HornExtensionProblem]:
        """Create horn extension problems from loss and context."""
        if not self.horn_solver:
            return []
        
        # Get model parameters
        parameters = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Create horn extension problems
        return self.horn_solver.create_horn_problems_from_loss(loss, parameters, context)
    
    def _compute_gradients(self, 
                         loss: torch.Tensor, 
                         horn_problems: List[HornExtensionProblem], 
                         context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute gradients using traditional or horn extension methods."""
        gradients = {}
        
        if horn_problems and self.config.use_horn_extensions:
            # Use horn extension-based gradient computation
            for problem in horn_problems:
                param_name = problem.learning_context['param_name']
                
                # Solve horn extension problem
                updated_params = self.horn_solver.solve_horn_extension(problem)
                
                # Compute gradient as difference
                original_params = problem.current_parameters
                gradient = updated_params - original_params
                
                gradients[param_name] = gradient
        else:
            # Traditional gradient computation
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # Clear gradients for next iteration
            self.model.zero_grad()
        
        return gradients
    
    def _create_lifting_problems(self, 
                               gradients: Dict[str, torch.Tensor], 
                               context: Dict[str, Any]) -> List[ParameterLiftingProblem]:
        """Create lifting problems from computed gradients."""
        # Get model parameters
        parameters = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Create lifting problems
        return self.lifting_optimizer.create_lifting_problems(parameters, gradients, context)
    
    def _solve_lifting_problems(self, 
                              problems: List[ParameterLiftingProblem]) -> Dict[str, torch.Tensor]:
        """Solve lifting problems to get parameter updates."""
        return self.lifting_optimizer.solve_lifting_problems(problems)
    
    def _apply_parameter_updates(self, updates: Dict[str, torch.Tensor]):
        """Apply categorical parameter updates to model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates and param.requires_grad:
                    # Apply lifting diagram-based update
                    param.data += self.config.learning_rate * updates[name]
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint with lifting diagram state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lifting_optimizer_state': self.lifting_optimizer.state_dict(),
            'config': self.config,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'training_history': self.training_history
        }
        
        if self.horn_solver:
            checkpoint['horn_solver_state'] = self.horn_solver.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved lifting diagram training checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint with lifting diagram state."""
        checkpoint = torch.load(filepath, map_location=get_device())
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.lifting_optimizer.load_state_dict(checkpoint['lifting_optimizer_state'])
        self.current_epoch = checkpoint['current_epoch']
        self.current_step = checkpoint['current_step']
        self.training_history = checkpoint['training_history']
        
        if 'horn_solver_state' in checkpoint and self.horn_solver:
            self.horn_solver.load_state_dict(checkpoint['horn_solver_state'])
        
        logger.info(f"Loaded lifting diagram training checkpoint from {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.training_history:
            return {}
        
        recent_history = self.training_history[-100:]  # Last 100 steps
        
        return {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_steps': len(self.training_history),
            'recent_avg_loss': sum(s['loss'] for s in recent_history) / len(recent_history),
            'recent_avg_horn_problems': sum(s['num_horn_problems'] for s in recent_history) / len(recent_history),
            'recent_avg_lifting_problems': sum(s['num_lifting_problems'] for s in recent_history) / len(recent_history),
            'config': self.config
        }


def create_lifting_trainer(model: nn.Module, 
                         config: Optional[LiftingTrainingConfig] = None,
                         basis_registry: Optional[BasisRegistry] = None) -> LiftingDiagramTrainer:
    """Factory function to create lifting diagram trainer.
    
    Args:
        model: GAIA model to train
        config: Training configuration (uses defaults if None)
        basis_registry: Simplicial basis registry
        
    Returns:
        Configured lifting diagram trainer
    """
    if config is None:
        config = LiftingTrainingConfig()
    
    return LiftingDiagramTrainer(model, config, basis_registry)


def train_with_lifting_diagrams(model: nn.Module,
                              train_dataloader,
                              val_dataloader,
                              loss_fn: Callable,
                              num_epochs: int,
                              config: Optional[LiftingTrainingConfig] = None) -> Dict[str, Any]:
    """High-level function to train model with lifting diagrams.
    
    Args:
        model: GAIA model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        loss_fn: Loss function
        num_epochs: Number of training epochs
        config: Training configuration
        
    Returns:
        Training results and statistics
    """
    trainer = create_lifting_trainer(model, config)
    
    training_results = {
        'epoch_stats': [],
        'val_stats': []
    }
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs} with lifting diagrams")
        
        # Training epoch
        epoch_stats = trainer.train_epoch(train_dataloader, loss_fn)
        training_results['epoch_stats'].append(epoch_stats)
        
        # Validation (traditional forward pass)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(batch['input'])
                loss = loss_fn(outputs, batch)
                val_loss += loss.item()
                val_batches += 1
        
        val_stats = {
            'val_loss': val_loss / val_batches if val_batches > 0 else 0.0,
            'epoch': epoch + 1
        }
        training_results['val_stats'].append(val_stats)
        
        logger.info(f"Epoch {epoch + 1} completed: "
                   f"Train Loss={epoch_stats['avg_loss']:.4f}, "
                   f"Val Loss={val_stats['val_loss']:.4f}")
    
    # Add final statistics
    training_results['final_stats'] = trainer.get_training_statistics()
    
    return training_results