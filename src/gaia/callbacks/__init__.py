"""GAIA Training Callbacks"""

from typing import List, Callable, Any, Dict
import logging

logger = logging.getLogger(__name__)

class CallbackManager:
    """Manages training callbacks for GAIA trainer"""
    
    def __init__(self, callbacks: List[Callable] = None):
        self.callbacks = callbacks or []
        
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Called at the beginning of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs)
                
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Called at the end of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs)
                
    def on_train_error(self, logs: Dict[str, Any] = None, error: Exception = None):
        """Called when training encounters an error"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_error'):
                callback.on_train_error(logs, error)
                
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch, logs)
                
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Called at the end of each epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, logs)
                
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the beginning of each batch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch, logs)
                
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Called at the end of each batch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch, logs)

class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        if logs is None:
            return
            
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
            
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")

__all__ = ['CallbackManager', 'EarlyStopping']