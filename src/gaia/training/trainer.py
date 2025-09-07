"""
GAIA Unified Trainer

This module provides a unified training system that integrates all GAIA components:
- Fuzzy simplicial sets for data encoding
- Coalgebras for parameter evolution  
- Hierarchical message passing
- Business unit communication
- Horn filling and Kan conditions
- Endofunctor dynamics

Replaces the massive trainer.py files with an optimal integrated approach.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

from ..core.abstractions import (
    IntegratedTrainer, GAIAComponent, TrainingState, ComponentRegistry
)
from ..core.integrated_structures import (
    IntegratedFuzzySimplicialSet, IntegratedCoalgebra, TConorm,
    create_fuzzy_simplicial_set_from_data
)
from ..core.simplices import Simplex0, Simplex1, BasisRegistry
from ..core.functor import SimplicialFunctor



@dataclass
class GAIATrainingConfig:
    """Configuration for GAIA unified trainer using global config system."""
    # Model parameters (MLP)
    input_dim: int = field(default_factory=lambda: _get_global_config().model.input_dim)
    hidden_dims: List[int] = field(default_factory=lambda: _get_global_config().model.hidden_dims or [256, 128, 64])
    output_dim: int = field(default_factory=lambda: _get_global_config().model.output_dim)
    
    # Transformer parameters
    vocab_size: int = field(default_factory=lambda: _get_global_config().model.vocab_size)
    d_model: int = field(default_factory=lambda: _get_global_config().model.d_model)
    num_heads: int = field(default_factory=lambda: _get_global_config().model.num_heads)
    num_layers: int = field(default_factory=lambda: _get_global_config().model.num_layers)
    seq_len: int = field(default_factory=lambda: _get_global_config().model.seq_len)
    d_ff: int = field(default_factory=lambda: _get_global_config().model.d_ff)
    max_seq_length: int = field(default_factory=lambda: _get_global_config().model.max_seq_length)
    
    # Training parameters
    learning_rate: float = field(default_factory=lambda: _get_global_config().optimization.learning_rate)
    batch_size: int = field(default_factory=lambda: _get_global_config().data.batch_size)
    max_epochs: int = field(default_factory=lambda: _get_global_config().epochs)
    
    # GAIA-specific parameters
    fuzzy_k_neighbors: int = field(default_factory=lambda: _get_global_config().data.n_neighbors)
    coalgebra_steps: int = field(default_factory=lambda: getattr(_get_global_config().model, 'coalgebra_steps', 3))
    message_passing_levels: int = field(default_factory=lambda: getattr(_get_global_config().model, 'message_passing_levels', 3))
    horn_filling_tolerance: float = field(default_factory=lambda: getattr(_get_global_config().optimization, 'horn_filling_tolerance', 1e-6))
    
    # Optimization parameters
    use_hierarchical_updates: bool = field(default_factory=lambda: getattr(_get_global_config().optimization, 'use_hierarchical_updates', True))
    use_business_units: bool = field(default_factory=lambda: getattr(_get_global_config().optimization, 'use_business_units', True))
    use_kan_verification: bool = field(default_factory=lambda: getattr(_get_global_config().optimization, 'use_kan_verification', True))
    verify_coalgebra_dynamics: bool = field(default_factory=lambda: getattr(_get_global_config().optimization, 'verify_coalgebra_dynamics', True))
    
    # Logging
    log_level: str = field(default_factory=lambda: getattr(_get_global_config().logging, 'level', "INFO"))
    checkpoint_dir: str = field(default_factory=lambda: getattr(_get_global_config().logging, 'checkpoint_dir', "checkpoints"))
    log_interval: int = field(default_factory=lambda: getattr(_get_global_config().logging, 'log_interval', 10))

def _get_global_config():
    """Helper function to get global configuration with fallback defaults."""
    try:
        from .config import TrainingConfig
        config = TrainingConfig()
        # Set fallback defaults for missing attributes
        if not hasattr(config.model, 'input_dim'):
            config.model.input_dim = 784
        if not hasattr(config.model, 'output_dim'):
            config.model.output_dim = 10
        if not hasattr(config.model, 'vocab_size'):
            config.model.vocab_size = 1000
        if not hasattr(config.model, 'd_model'):
            config.model.d_model = 256
        if not hasattr(config.model, 'num_heads'):
            config.model.num_heads = 4
        if not hasattr(config.model, 'num_layers'):
            config.model.num_layers = 4
        if not hasattr(config.model, 'seq_len'):
            config.model.seq_len = 32
        if not hasattr(config.model, 'd_ff'):
            config.model.d_ff = 1024
        if not hasattr(config.model, 'max_seq_length'):
            config.model.max_seq_length = 512
        return config
    except ImportError:
        # Fallback to hardcoded defaults if config system not available
        class FallbackConfig:
            class ModelConfig:
                input_dim = 784
                hidden_dims = [256, 128, 64]
                output_dim = 10
                vocab_size = 1000
                d_model = 256
                num_heads = 4
                num_layers = 4
                seq_len = 32
                d_ff = 1024
                max_seq_length = 512
            class OptimizationConfig:
                learning_rate = 1e-3
            class DataConfig:
                batch_size = 32
                n_neighbors = 5
            class LoggingConfig:
                level = "INFO"
                checkpoint_dir = "checkpoints"
                log_interval = 10
            model = ModelConfig()
            optimization = OptimizationConfig()
            data = DataConfig()
            logging = LoggingConfig()
            epochs = 100
        return FallbackConfig()


class FuzzyDataEncoder(GAIAComponent):
    """Component for encoding data as fuzzy simplicial sets."""
    
    def __init__(self, config: GAIATrainingConfig):
        super().__init__("fuzzy_encoder", config.__dict__)
        self.k = config.fuzzy_k_neighbors
        self.current_fss: Optional[IntegratedFuzzySimplicialSet] = None
    
    def initialize(self) -> None:
        """Initialize the fuzzy encoder."""
        logging.info("Initializing fuzzy data encoder")
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update with new data batch."""
        if 'batch_data' in state.metadata:
            data = state.metadata['batch_data']
            self.current_fss = create_fuzzy_simplicial_set_from_data(
                data, self.k, f"batch_{state.step}"
            )
            state.metadata['fuzzy_simplicial_set'] = self.current_fss
        
        return state
    
    def validate(self) -> bool:
        """Validate encoder state."""
        return self.current_fss is None or self.current_fss.validate()


class CoalgebraEvolution(GAIAComponent):
    """Component for coalgebraic parameter evolution."""
    
    def __init__(self, config: GAIATrainingConfig, model: nn.Module):
        super().__init__("coalgebra_evolution", config.__dict__)
        self.model = model
        self.steps = config.coalgebra_steps
        self.coalgebras: Dict[str, IntegratedCoalgebra] = {}
    
    def initialize(self) -> None:
        """Initialize coalgebras for model parameters."""
        logging.info("Initializing coalgebraic evolution")
        
        # Get gradient step size from global config
        gradient_step_size = getattr(_get_global_config().optimization, 'gradient_step_size', 0.01)
        
        # Create coalgebras for each parameter group
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Simple endofunctor: gradient descent step
                def create_endofunctor(p, step_size):
                    def endofunctor_apply(state):
                        if hasattr(p, 'grad') and p.grad is not None:
                            return state - step_size * p.grad  # Configurable gradient step
                        return state
                    return endofunctor_apply
                
                class SimpleEndofunctor:
                    def __init__(self, param, step_size):
                        self.param = param
                        self.step_size = step_size
                    
                    def apply_to_object(self, state):
                        if hasattr(self.param, 'grad') and self.param.grad is not None:
                            return state - self.step_size * self.param.grad
                        return state
                
                endofunctor = SimpleEndofunctor(param, gradient_step_size)
                coalgebra = IntegratedCoalgebra(
                    param.data.clone(), endofunctor, f"coalgebra_{name}"
                )
                self.coalgebras[name] = coalgebra
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update coalgebras with parameter evolution."""
        for name, coalgebra in self.coalgebras.items():
            # Evolve parameters through coalgebra dynamics
            evolved_trajectory = coalgebra.iterate_dynamics(self.steps)
            state.metadata[f'coalgebra_trajectory_{name}'] = evolved_trajectory
        
        return state
    
    def validate(self) -> bool:
        """Validate coalgebra evolution."""
        return all(coalgebra.validate() for coalgebra in self.coalgebras.values())


class HierarchicalCommunication(GAIAComponent):
    """Component for hierarchical message passing and business unit communication."""
    
    def __init__(self, config: GAIATrainingConfig, functor: SimplicialFunctor):
        super().__init__("hierarchical_communication", config.__dict__)
        self.functor = functor
        self.levels = config.message_passing_levels
        self.message_passing: Optional[Any] = None
        self.business_units: Optional[Any] = None
    
    def initialize(self) -> None:
        """Initialize hierarchical communication systems."""
        logging.info("Initializing hierarchical communication")
        
        if self.config.get('use_hierarchical_updates', True):
            # Lazy import to avoid circular dependency
            from .hierarchical_message_passing import HierarchicalMessagePassingSystem
            self.message_passing = HierarchicalMessagePassingSystem(
                self.functor, parameter_dim=64
            )
        
        if self.config.get('use_business_units', True):
            # Lazy import to avoid circular dependency
            from ..core.business_units import BusinessUnitHierarchy
            self.business_units = BusinessUnitHierarchy(self.functor)
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update hierarchical communication."""
        if self.message_passing:
            # Process hierarchical updates (simplified)
            for level in range(self.levels):
                if f'level_{level}_gradients' in state.gradients:
                    gradients = state.gradients[f'level_{level}_gradients']
                    # Process through message passing system
                    if hasattr(self.message_passing, 'process_level_gradients'):
                        processed = self.message_passing.process_level_gradients(level, gradients)
                        state.gradients[f'level_{level}_processed'] = processed
        
        if self.business_units:
            # Update business unit communications (simplified)
            if hasattr(self.business_units, 'update_communications'):
                self.business_units.update_communications(state.metadata)
        
        return state
    
    def validate(self) -> bool:
        """Validate hierarchical communication."""
        valid = True
        if self.message_passing:
            # Simple validation - check if system is initialized
            valid &= hasattr(self.message_passing, 'simplicial_functor')
        if self.business_units:
            # Simple validation - check if hierarchy is initialized
            valid &= hasattr(self.business_units, 'functor')
        return valid


class KanVerification(GAIAComponent):
    """Component for Kan complex verification and horn filling."""
    
    def __init__(self, config: GAIATrainingConfig, functor: SimplicialFunctor):
        super().__init__("kan_verification", config.__dict__)
        self.functor = functor
        self.tolerance = config.horn_filling_tolerance
        self.verifier: Optional[Any] = None
    
    def initialize(self) -> None:
        """Initialize Kan complex verifier."""
        logging.info("Initializing Kan complex verification")
        
        if self.config.get('use_kan_verification', True):
            # Lazy import to avoid circular dependency
            from ..core.kan_verification import KanComplexVerifier
            self.verifier = KanComplexVerifier(self.functor)
    
    def update(self, state: TrainingState) -> TrainingState:
        """Update Kan verification."""
        if self.verifier:
            # Verify horn filling conditions (simplified)
            if hasattr(self.verifier, 'verify_all_conditions'):
                verification_results = self.verifier.verify_all_conditions()
                state.metadata['kan_verification'] = verification_results
                
                # Log any failures
                if isinstance(verification_results, dict) and not all(verification_results.values()):
                    logging.warning(f"Kan verification failures: {verification_results}")
        
        return state
    
    def validate(self) -> bool:
        """Validate Kan verification component."""
        if self.verifier:
            # Simple validation - check if verifier is initialized
            return hasattr(self.verifier, 'functor')
        return True


class GAIATrainer(IntegratedTrainer):
    """Unified trainer integrating all GAIA components."""
    
    def __init__(self, model: nn.Module, config: GAIATrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize core GAIA structures
        self.basis_registry = BasisRegistry()
        self.functor = SimplicialFunctor("gaia_functor", self.basis_registry)
        
        # Create components
        components = [
            FuzzyDataEncoder(config),
            CoalgebraEvolution(config, model),
            HierarchicalCommunication(config, self.functor),
            KanVerification(config, self.functor)
        ]
        
        super().__init__(components)
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.metrics = defaultdict(list)
        self.best_loss = float('inf')
    
    def train_step(self) -> TrainingState:
        """Execute one training step across all components."""
        self.model.train()
        
        # Update training state
        self.state.step += 1
        
        # Process through all components
        for component in self.components.values():
            self.state = component.update(self.state)
        
        # Standard training step
        if 'batch_data' in self.state.metadata and 'batch_targets' in self.state.metadata:
            data = self.state.metadata['batch_data']
            targets = self.state.metadata['batch_targets']
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Store gradients in state
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.state.gradients[name] = param.grad.clone()
            
            # Optimizer step
            self.optimizer.step()
            
            # Update state
            self.state.loss = loss.item()
            self.state.parameters = {name: param.data.clone() 
                                   for name, param in self.model.named_parameters()}
        
        # Log metrics
        if self.state.step % self.config.log_interval == 0:
            self._log_metrics()
        
        return self.state
    
    def validate_step(self) -> Dict[str, float]:
        """Execute validation across all components."""
        self.model.eval()
        
        validation_metrics = {}
        
        # Validate all components
        for name, component in self.components.items():
            validation_metrics[f'{name}_valid'] = float(component.validate())
        
        # Model validation (if validation data available)
        if 'val_data' in self.state.metadata and 'val_targets' in self.state.metadata:
            with torch.no_grad():
                val_data = self.state.metadata['val_data']
                val_targets = self.state.metadata['val_targets']
                
                val_outputs = self.model(val_data)
                val_loss = self.criterion(val_outputs, val_targets)
                
                # Accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                accuracy = (predicted == val_targets).float().mean()
                
                validation_metrics['val_loss'] = val_loss.item()
                validation_metrics['val_accuracy'] = accuracy.item()
        
        return validation_metrics
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Add batch to training state
            self.state.metadata['batch_data'] = data
            self.state.metadata['batch_targets'] = targets
            
            # Execute training step
            self.state = self.train_step()
            
            # Collect metrics
            epoch_metrics['loss'].append(self.state.loss)
            
            # Validation step periodically
            if batch_idx % (len(dataloader) // 4) == 0:
                val_metrics = self.validate_step()
                for key, value in val_metrics.items():
                    epoch_metrics[key].append(value)
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def train(self, train_loader, val_loader=None, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.max_epochs
        
        self.logger.info(f"Starting GAIA unified training for {num_epochs} epochs")
        
        # Initialize all components
        for component in self.components.values():
            component.initialize()
        
        training_history = defaultdict(list)
        
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            
            # Train epoch
            epoch_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)
                epoch_metrics.update(val_metrics)
            
            # Store metrics
            for key, value in epoch_metrics.items():
                training_history[key].append(value)
            
            # Logging
            self.logger.info(f"Epoch {epoch}: {epoch_metrics}")
            
            # Checkpointing
            if epoch_metrics.get('val_loss', epoch_metrics.get('loss', float('inf'))) < self.best_loss:
                self.best_loss = epoch_metrics.get('val_loss', epoch_metrics.get('loss'))
                self._save_checkpoint(epoch)
        
        self.logger.info("Training completed successfully")
        return dict(training_history)
    
    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        val_metrics = defaultdict(list)
        
        self.model.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.state.metadata['val_data'] = data
                self.state.metadata['val_targets'] = targets
                
                batch_val_metrics = self.validate_step()
                for key, value in batch_val_metrics.items():
                    val_metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    def _log_metrics(self):
        """Log current training metrics."""
        self.logger.info(f"Step {self.state.step}: Loss = {self.state.loss:.6f}")
        
        # Log component-specific metrics
        for name, component in self.components.items():
            if hasattr(component, 'get_metrics'):
                component_metrics = component.get_metrics()
                self.logger.debug(f"{name} metrics: {component_metrics}")
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
            'config': self.config,
            'training_state': self.state
        }
        
        checkpoint_path = checkpoint_dir / f'gaia_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state = checkpoint['training_state']
        self.best_loss = checkpoint['loss']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']


# Factory function for easy trainer creation
def create_gaia_trainer(model: nn.Module, config: Optional[GAIATrainingConfig] = None) -> GAIATrainer:
    """Create a GAIA unified trainer with default configuration."""
    if config is None:
        config = GAIATrainingConfig()
    
    return GAIATrainer(model, config)


# Export public interface
__all__ = [
    'GAIATrainingConfig', 'FuzzyDataEncoder', 'CoalgebraEvolution',
    'HierarchicalCommunication', 'KanVerification', 'GAIATrainer',
    'create_gaia_trainer'
]