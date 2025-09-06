"""Checkpoint Management for GAIA Framework"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil
import json
from .state import TrainingState

class CheckpointManager:
    """Manages model checkpoints with categorical structure preservation"""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = 'val_loss',
        higher_is_better: bool = False,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best checkpoint
            monitor_metric: Metric to monitor for best checkpoint
            higher_is_better: Whether higher metric values are better
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.higher_is_better = higher_is_better
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.checkpoint_history = []
        self.best_checkpoint = None
        self.best_metric_value = float('-inf') if higher_is_better else float('inf')
    
    def save_checkpoint(
        self,
        model: nn.Module,
        training_state: TrainingState,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """Save a checkpoint"""
        
        if filename is None:
            filename = f"checkpoint_epoch_{training_state.epoch:04d}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': training_state.epoch,
            'step': training_state.step,
            'model_state_dict': model.state_dict(),
            'training_state': training_state.__dict__,
            'model_config': getattr(model, 'config', {}),
        }
        
        # Add optimizer state
        if optimizer is not None and self.save_optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None and self.save_scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add categorical structure info
        if hasattr(model, 'get_categorical_structure'):
            checkpoint_data['categorical_structure'] = model.get_categorical_structure()
        
        # Add extra data
        if extra_data:
            checkpoint_data['extra_data'] = extra_data
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update checkpoint history
        checkpoint_info = {
            'path': checkpoint_path,
            'epoch': training_state.epoch,
            'step': training_state.step,
            'metrics': training_state.val_metrics if training_state.val_metrics else {}
        }
        
        # Check if this is the best checkpoint
        current_metric = self._get_metric_value(checkpoint_info['metrics'])
        is_best = self._is_better_metric(current_metric)
        
        if is_best:
            self.best_checkpoint = checkpoint_info
            self.best_metric_value = current_metric
            
            # Save best checkpoint separately
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            shutil.copy2(checkpoint_path, best_path)
        
        # Manage checkpoint history
        if not self.save_best_only or is_best:
            self.checkpoint_history.append(checkpoint_info)
            self._cleanup_old_checkpoints()
        
        # Save checkpoint metadata
        self._save_metadata()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        map_location: Optional[str] = None
    ) -> TrainingState:
        """Load a checkpoint"""
        
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
        elif checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoint_history:
                raise ValueError("No checkpoints found")
            checkpoint_path = self.checkpoint_history[-1]['path']
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Restore training state
        training_state = TrainingState(**checkpoint_data['training_state'])
        
        # Restore categorical structure if available
        if hasattr(model, 'restore_categorical_structure') and 'categorical_structure' in checkpoint_data:
            model.restore_categorical_structure(checkpoint_data['categorical_structure'])
        
        return training_state
    
    def _get_metric_value(self, metrics: Dict[str, float]) -> float:
        """Get the monitored metric value"""
        return metrics.get(self.monitor_metric, 
                          float('-inf') if self.higher_is_better else float('inf'))
    
    def _is_better_metric(self, current_value: float) -> bool:
        """Check if current metric is better than best so far"""
        if self.higher_is_better:
            return current_value > self.best_metric_value
        else:
            return current_value < self.best_metric_value
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints"""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Remove oldest checkpoints
            to_remove = self.checkpoint_history[:-self.max_checkpoints]
            for checkpoint_info in to_remove:
                checkpoint_path = checkpoint_info['path']
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            
            # Update history
            self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]
    
    def _save_metadata(self):
        """Save checkpoint metadata"""
        metadata = {
            'checkpoint_history': [
                {
                    'path': str(info['path']),
                    'epoch': info['epoch'],
                    'step': info['step'],
                    'metrics': info['metrics']
                }
                for info in self.checkpoint_history
            ],
            'best_checkpoint': {
                'path': str(self.best_checkpoint['path']),
                'epoch': self.best_checkpoint['epoch'],
                'step': self.best_checkpoint['step'],
                'metrics': self.best_checkpoint['metrics']
            } if self.best_checkpoint else None,
            'best_metric_value': self.best_metric_value,
            'monitor_metric': self.monitor_metric,
            'higher_is_better': self.higher_is_better
        }
        
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_checkpoint_list(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints"""
        return self.checkpoint_history.copy()
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint"""
        return self.best_checkpoint.copy() if self.best_checkpoint else None