"""Production GAIA Neural Network Trainer"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with: pip install wandb")

from .config import TrainingConfig
from .engine.state import TrainingState
from .engine.checkpoints import CheckpointManager
from .engine.loops import TrainingLoop, ValidationLoop
from .engine.profiler import GAIAProfiler
from ..core.functor import SimplicialFunctor
from ..callbacks import CallbackManager
from ..metrics import MetricTracker
from ..utils.device import get_device, setup_distributed
from ..utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

class GAIATrainer:
    """ GAIA Neural Network Trainer
    
    Features:
    - Categorical deep learning with simplicial functors
    - Mixed precision training
    - Distributed training support
    - Advanced checkpointing and resuming
    - Comprehensive monitoring and profiling
    - Horn solving and coherence verification
    - Production-grade error handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ):
        """Initialize GAIA Trainer
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            callbacks: List of training callbacks
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Setup device and distributed training
        self.device = get_device(config.device)
        if config.distributed:
            setup_distributed(config.world_size, config.rank)
        
        # Set reproducibility
        set_seed(config.data.random_seed)
        
        # Initialize GAIA categorical components
        self.functor = None
        if config.categorical_training:
            self._setup_categorical_components()
        
        # Setup training components
        self._setup_optimization()
        self._setup_mixed_precision()
        self._setup_monitoring()
        self._setup_checkpointing()
        
        # Initialize training state
        self.state = TrainingState()
        
        # Setup callbacks
        self.callback_manager = CallbackManager(callbacks or [])
        
        # Setup training loops
        self.train_loop = TrainingLoop(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            config=self.config
        )
        
        if self.val_dataloader:
            self.val_loop = ValidationLoop(
                model=self.model,
                criterion=self.criterion,
                device=self.device,
                config=self.config
            )
        
        # Setup profiler
        self.profiler = GAIAProfiler(enabled=kwargs.get('profile', False))
        
        # Setup metrics
        self.metric_tracker = MetricTracker()
        
        logger.info(f"Initialized GAIA Trainer on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_categorical_components(self) -> None:
        """Setup GAIA categorical deep learning components"""
        from ..core.simplices import BasisRegistry
        
        # Initialize basis registry and simplicial functor
        basis_registry = BasisRegistry()
        self.functor = SimplicialFunctor(
            name=f"{self.config.model.name}_functor",
            basis_registry=basis_registry
        )
        
        # Setup horn solvers
        self._setup_horn_solvers()
        
        logger.info("Categorical components initialized")
    
    def _setup_horn_solvers(self) -> None:
        """Setup inner and outer horn solvers"""
        from .solvers.inner_solver import EndofunctorialSolver
        from .solvers.outer_solver import UniversalLiftingSolver
        from .solvers.yoneda_proxy import MetricYonedaProxy
        
        # These will be initialized when needed during training
        self.inner_solver = None
        self.outer_solver = None
        self.yoneda_proxy = MetricYonedaProxy(
            target_dim=self.config.model.categorical_embedding_dim
        )
    
    def _setup_optimization(self) -> None:
        """Setup optimizer and scheduler"""
        from .config import OptimizationType, SchedulerType
        
        opt_config = self.config.optimization
        
        # Create loss function/criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        if opt_config.optimizer == OptimizationType.ADAM:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.optimizer == OptimizationType.ADAMW:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.optimizer == OptimizationType.SGD:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config.optimizer}")
        
        # Create scheduler
        if opt_config.scheduler == SchedulerType.COSINE:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                **opt_config.scheduler_params
            )
        elif opt_config.scheduler == SchedulerType.LINEAR:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                **opt_config.scheduler_params
            )
        elif opt_config.scheduler == SchedulerType.PLATEAU:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.monitor_mode,
                **opt_config.scheduler_params
            )
        else:
            self.scheduler = None
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training"""
        # Only enable mixed precision for CUDA devices
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            if self.config.mixed_precision and self.device.type != 'cuda':
                logger.info(f"Mixed precision disabled for {self.device.type} device")
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring and logging"""
        # TensorBoard
        if self.config.use_tensorboard:
            log_dir = Path("runs") / f"gaia_{int(time.time())}"
            self.tb_writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging: {log_dir}")
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
                name=f"{self.config.model.name}_{int(time.time())}"
            )
            logger.info("W&B logging initialized")
        else:
            self.wandb_enabled = False
    
    def _setup_checkpointing(self) -> None:
        """Setup checkpoint management"""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_checkpoints=self.config.save_top_k,
            monitor_metric=self.config.monitor_metric,
            higher_is_better=(self.config.monitor_mode == 'max')
        )
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting GAIA training...")
        
        try:
            # Pre-training setup
            self.callback_manager.on_train_begin(self.state)
            self._compile_model()
            
            # Training loop
            for epoch in range(self.config.epochs):
                self.state.epoch = epoch
                
                # Epoch callbacks
                self.callback_manager.on_epoch_begin(self.state)
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                if self.val_dataloader and epoch % self.config.eval_frequency == 0:
                    val_metrics = self._validate_epoch()
                    self.state.val_metrics = val_metrics
                
                # GAIA categorical operations
                if self.config.categorical_training:
                    self._categorical_operations(epoch)
                
                # Update state
                self.state.train_metrics = train_metrics
                self.state.step += len(self.train_dataloader)
                
                # Logging
                self._log_metrics(epoch, train_metrics, self.state.val_metrics)
                
                # Checkpointing
                if epoch % self.config.save_frequency == 0:
                    self._save_checkpoint(epoch)
                
                # Early stopping check
                if self._should_stop_early():
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Epoch callbacks
                self.callback_manager.on_epoch_end(self.state)
            
            # Post-training
            self.callback_manager.on_train_end(self.state)
            
            # Final evaluation
            if self.test_dataloader:
                test_metrics = self._test()
                logger.info(f"Test metrics: {test_metrics}")
            
            return self._get_training_summary()
            
        except Exception as e:
            logger.error(f"Training failed: {type(e).__name__}: {str(e)}")
            logger.error(f"Exception details: {repr(e)}")
            
            # Add more detailed error information
            if isinstance(e, KeyError):
                logger.error(f"KeyError details - missing key: {e.args[0] if e.args else 'unknown'}")
                logger.error(f"KeyError key type: {type(e.args[0]) if e.args else 'unknown'}")
                if hasattr(self, 'metric_tracker'):
                    logger.error(f"Available metrics: {list(self.metric_tracker.metrics.keys())}")
                    logger.error(f"Metric tracker state: {self.metric_tracker.__dict__}")
                
                # Debug training state
                logger.error(f"Current epoch: {getattr(self.state, 'epoch', 'unknown')}")
                logger.error(f"Current step: {getattr(self.state, 'step', 'unknown')}")
                logger.error(f"Train metrics: {getattr(self.state, 'train_metrics', 'unknown')}")
                logger.error(f"Val metrics: {getattr(self.state, 'val_metrics', 'unknown')}")
                
                # Debug dataloader info
                logger.error(f"Train dataloader length: {len(self.train_dataloader) if hasattr(self, 'train_dataloader') else 'unknown'}")
                logger.error(f"Val dataloader length: {len(self.val_dataloader) if hasattr(self, 'val_dataloader') and self.val_dataloader else 'unknown'}")
                
                # Print full traceback
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            self.callback_manager.on_train_error(self.state, e)
            raise
        
        finally:
            self._cleanup()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        with self.profiler.profile_section("train_epoch"):
            # Fix: The TrainingLoop's run_epoch handles the iteration over the dataloader.
            # The previous implementation incorrectly looped over the dataloader here
            # and called run_epoch for each batch, which is wrong.
            try:
                logger.debug(f"Starting training epoch with dataloader length: {len(self.train_dataloader)}")
                
                # Call run_epoch once per epoch, not per batch
                epoch_metrics = self.train_loop.run_epoch(self.train_dataloader)
                
                logger.debug(f"Epoch metrics from run_epoch: {epoch_metrics}")
                logger.debug(f"Epoch metrics type: {type(epoch_metrics)}")
                
                # Update metrics tracker
                self.metric_tracker.update(epoch_metrics)
                
                # Compute final metrics for the epoch
                final_metrics = self.metric_tracker.compute()
                logger.debug(f"Final computed metrics: {final_metrics}")
                
                return final_metrics
                
            except Exception as e:
                logger.error(f"Error in _train_epoch: {type(e).__name__}: {str(e)}")
                logger.error(f"Epoch metrics at error: {locals().get('epoch_metrics', 'not set')}")
                logger.error(f"Metric tracker state: {self.metric_tracker.__dict__ if hasattr(self, 'metric_tracker') else 'no tracker'}")
                raise
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        with torch.no_grad(), self.profiler.profile_section("val_epoch"):
            try:
                logger.debug(f"Starting validation epoch with dataloader length: {len(self.val_dataloader)}")
                
                for batch in self.val_dataloader:
                    metrics = self.val_loop.val_step(batch, self.state)
                    logger.debug(f"Validation batch metrics: {metrics}")
                    self.metric_tracker.update(metrics, prefix="val_")
                
                val_metrics = self.metric_tracker.compute(prefix="val_")
                logger.debug(f"Final validation metrics: {val_metrics}")
                
                return val_metrics
                
            except Exception as e:
                logger.error(f"Error in _validate_epoch: {type(e).__name__}: {str(e)}")
                logger.error(f"Metric tracker state: {self.metric_tracker.__dict__ if hasattr(self, 'metric_tracker') else 'no tracker'}")
                raise
    
    def _categorical_operations(self, epoch: int) -> None:
        """Perform GAIA categorical operations"""
        if epoch % self.config.horn_solving_frequency == 0:
            self._solve_horns()
        
        if epoch % self.config.coherence_check_frequency == 0:
            self._verify_coherence()
    
    def _solve_horns(self) -> None:
        """Solve inner and outer horns"""
        if self.functor is None:
            return
        
        try:
            # Inner horn solving
            if self.inner_solver:
                inner_results = self.inner_solver.solve_all_horns()
                logger.debug(f"Inner horn results: {inner_results}")
            
            # Outer horn solving  
            if self.outer_solver:
                outer_results = self.outer_solver.solve_all_horns()
                logger.debug(f"Outer horn results: {outer_results}")
                
        except Exception as e:
            logger.warning(f"Horn solving failed: {e}")
    
    def _verify_coherence(self) -> None:
        """Verify categorical coherence"""
        if self.functor is None:
            return
        
        try:
            coherence_results = self.functor.verify_simplicial_identities()
            logger.debug(f"Coherence verification: {coherence_results}")
            
            # Log coherence metrics
            if self.tb_writer:
                # Fix: Extract a numeric value from the coherence results
                if isinstance(coherence_results, dict):
                    # Use a meaningful metric from the results
                    coherence_value = coherence_results.get('valid', 0.0)
                    if isinstance(coherence_value, bool):
                        coherence_value = float(coherence_value)
                    elif not isinstance(coherence_value, (int, float)):
                        coherence_value = 0.0
                else:
                    coherence_value = float(coherence_results)
                    
                self.tb_writer.add_scalar(
                    "coherence/simplicial_identities",
                    coherence_value,
                    self.state.step
                )
                
        except Exception as e:
            logger.warning(f"Coherence verification failed: {e}")
            
    def _compile_model(self) -> None:
        """Compile model for optimization"""
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def _cleanup(self) -> None:
        """Cleanup resources after training"""
        try:
            # Stop profiler
            if hasattr(self, 'profiler') and self.profiler:
                self.profiler.reset()
            
            # Close TensorBoard writer
            if hasattr(self, 'tb_writer') and self.tb_writer:
                self.tb_writer.close()
            
            # Finish W&B run
            if hasattr(self, 'wandb_enabled') and self.wandb_enabled:
                import wandb
                wandb.finish()
            
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log training metrics"""
        # Console logging
        log_str = f"Epoch {epoch:3d} | "
        log_str += f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
        if val_metrics:
            log_str += f"Val Loss: {val_metrics.get('val_loss', 0):.4f}"
        logger.info(log_str)
        
        # TensorBoard logging
        if self.tb_writer:
            for name, value in train_metrics.items():
                self.tb_writer.add_scalar(f"train/{name}", value, epoch)
            
            if val_metrics:
                for name, value in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{name.replace('val_', '')}", value, epoch)
        
        # W&B logging
        if hasattr(self, 'wandb_enabled') and self.wandb_enabled:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            if val_metrics:
                log_dict.update({f"val/{k.replace('val_', '')}": v for k, v in val_metrics.items()})
            wandb.log(log_dict, step=epoch)
    
    def _log_batch_metrics(self, batch_idx: int, metrics: Dict) -> None:
        """Log batch-level metrics"""
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(
                    f"batch/{name}", 
                    value, 
                    self.state.step + batch_idx
                )
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'state': self.state,
            'metrics': self.metric_tracker.get_history()
        }
        
        # Add GAIA-specific state
        if self.functor:
            checkpoint['functor_state'] = self.functor.state_dict()
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            training_state=self.state,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra_data=checkpoint
        )
    
    def _should_stop_early(self) -> bool:
        """Check early stopping condition"""
        if not self.config.early_stopping or not self.state.val_metrics:
            return False
        
        monitor_value = self.state.val_metrics.get(self.config.monitor_metric)
        if monitor_value is None:
            return False
        
        # Simple early stopping logic (can be enhanced)
        if not hasattr(self.state, 'best_metric'):
            self.state.best_metric = monitor_value
            self.state.patience_counter = 0
            return False
        
        improved = (
            (self.config.monitor_mode == "min" and monitor_value < self.state.best_metric - self.config.min_delta) or
            (self.config.monitor_mode == "max" and monitor_value > self.state.best_metric + self.config.min_delta)
        )
        
        if improved:
            self.state.best_metric = monitor_value
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1
        
        return self.state.patience_counter >= self.config.patience
    
    def _test(self) -> Dict[str, float]:
        """Run final test evaluation"""
        self.model.eval()
        
        with torch.no_grad():
            try:
                logger.debug(f"Starting test evaluation with dataloader length: {len(self.test_dataloader)}")
                
                for batch in self.test_dataloader:
                    metrics = self.val_loop.val_step(batch, self.state)
                    logger.debug(f"Test batch metrics: {metrics}")
                    self.metric_tracker.update(metrics, prefix="test_")
                
                test_metrics = self.metric_tracker.compute(prefix="test_")
                logger.debug(f"Final test metrics: {test_metrics}")
                
                return test_metrics
                
            except Exception as e:
                logger.error(f"Error in _test: {type(e).__name__}: {str(e)}")
                logger.error(f"Metric tracker state: {self.metric_tracker.__dict__ if hasattr(self, 'metric_tracker') else 'no tracker'}")
                raise
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'total_epochs': self.state.epoch + 1,
            'total_steps': self.state.step,
            'best_metrics': getattr(self.state, 'best_metric', None),
            'final_train_metrics': self.state.train_metrics,
            'final_val_metrics': self.state.val_metrics,
            'training_time': getattr(self.state, 'training_time', 0),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'categorical_coherence': self._get_final_coherence() if self.functor else None
        }
    
    def _get_final_coherence(self) -> Dict[str, Any]:
        """Get final categorical coherence metrics"""
        try:
            return {
                'simplicial_identities': self.functor.verify_simplicial_identities(),
                'horn_completeness': self._check_horn_completeness()
            }
        except Exception as e:
            logger.warning(f"Final coherence check failed: {e}")
            return {}
    
   