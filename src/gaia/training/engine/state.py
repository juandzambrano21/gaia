"""Training State Management for GAIA Framework"""

import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

@dataclass
class TrainingState:
    """Comprehensive training state for GAIA framework"""
    
    # Basic training info
    epoch: int = 0
    step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    
    # Training metrics history
    train_metrics: List[Dict[str, float]] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    
    # Learning rate history
    lr_history: List[float] = field(default_factory=list)
    
    # Categorical structure info
    categorical_info: Dict[str, Any] = field(default_factory=dict)
    coherence_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Training configuration
    config: Optional[Dict[str, Any]] = None
    
    # Timing information
    start_time: Optional[float] = None
    epoch_times: List[float] = field(default_factory=list)
    
    # Model architecture info
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def update_train_metrics(self, metrics: Dict[str, float]):
        """Update training metrics for current epoch"""
        self.train_metrics.append(metrics.copy())
        if 'lr' in metrics:
            self.lr_history.append(metrics['lr'])
    
    def update_val_metrics(self, metrics: Dict[str, float]):
        """Update validation metrics for current epoch"""
        self.val_metrics.append(metrics.copy())
    
    def update_coherence_info(self, coherence_info: Dict[str, Any]):
        """Update categorical coherence information"""
        coherence_entry = {
            'epoch': self.epoch,
            'step': self.step,
            'timestamp': time.time(),
            **coherence_info
        }
        self.coherence_history.append(coherence_entry)
    
    def is_best_model(self, metric_value: float, higher_is_better: bool = False) -> bool:
        """Check if current model is the best so far"""
        if higher_is_better:
            is_best = metric_value > self.best_metric
            if is_best:
                self.best_metric = metric_value
                self.best_epoch = self.epoch
        else:
            is_best = metric_value < self.best_metric
            if is_best:
                self.best_metric = metric_value
                self.best_epoch = self.epoch
        
        return is_best
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        summary = {
            'total_epochs': self.epoch,
            'total_steps': self.step,
            'total_time_hours': total_time / 3600,
            'avg_epoch_time_minutes': avg_epoch_time / 60,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'final_lr': self.lr_history[-1] if self.lr_history else None,
        }
        
        # Add latest metrics with safe access
        if self.train_metrics:
            summary['final_train_metrics'] = self.train_metrics[-1]
        else:
            summary['final_train_metrics'] = {}
            
        if self.val_metrics:
            summary['final_val_metrics'] = self.val_metrics[-1]
        else:
            summary['final_val_metrics'] = {}
        
        # Add categorical info
        if self.coherence_history:
            latest_coherence = self.coherence_history[-1]
            summary['final_coherence_info'] = latest_coherence
            
            # Coherence statistics
            coherent_epochs = sum(1 for entry in self.coherence_history 
                                if entry.get('is_coherent', False))
            summary['coherence_rate'] = coherent_epochs / len(self.coherence_history)
        
        return summary
    
    def save_to_file(self, filepath: Path):
        """Save training state to file"""
        state_dict = {
            'epoch': self.epoch,
            'step': self.step,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'lr_history': self.lr_history,
            'categorical_info': self.categorical_info,
            'coherence_history': self.coherence_history,
            'config': self.config,
            'start_time': self.start_time,
            'epoch_times': self.epoch_times,
            'model_info': self.model_info
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'TrainingState':
        """Load training state from file"""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        return cls(**state_dict)
    
    def reset(self):
        """Reset training state for new training run"""
        self.epoch = 0
        self.step = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.train_metrics.clear()
        self.val_metrics.clear()
        self.lr_history.clear()
        self.coherence_history.clear()
        self.start_time = time.time()
        self.epoch_times.clear()