"""GAIA Metrics Tracking"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class MetricTracker:
    """Tracks and computes metrics during training"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)
        
    def update(self, metrics: Dict[str, Any] = None, prefix: str = "", **kwargs):
        """Update metrics with new values."""
        # Handle both calling patterns
        if metrics is not None:
            # Called with metrics as first argument
            all_metrics = metrics
        else:
            # Called with keyword arguments only
            all_metrics = kwargs
            
        for name, value in all_metrics.items():
            # Enhanced validation for problematic metric names
            if not isinstance(name, str):
                logger.warning(f"Invalid metric name type: {type(name)} for name {repr(name)}. Converting to string.")
                name = str(name)
                
            if name in ['-1', 'None', '', 'nan', 'inf'] or not name or name.isspace():
                logger.warning(f"Skipping invalid metric name: {repr(name)}")
                continue
                
            # Add prefix if provided
            full_name = f"{prefix}{name}" if prefix else name
            
            try:
                # Convert value to float
                float_value = float(value)
                if not np.isfinite(float_value):
                    logger.warning(f"Non-finite value for metric {full_name}: {value}")
                    continue
                    
                self.metrics[full_name].append(float_value)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert metric {full_name} value {value} to float: {e}")
    
    def get_current(self, name: str) -> float:
        """Get the current (most recent) value for a metric."""
        # Input validation and type checking
        if not isinstance(name, str):
            logger.error(f"get_current called with non-string name: {repr(name)} (type: {type(name)})")
            if isinstance(name, (int, float)):
                logger.error(f"Numeric metric name detected. This suggests a bug where list index {name} is being used as dict key.")
            return 0.0
            
        # Handle invalid metric names
        if not name or name.isspace() or name in ['-1', 'None', 'nan', 'inf']:
            logger.error(f"Invalid metric name: {repr(name)}")
            return 0.0
        
        try:
            if name not in self.metrics:
                logger.warning(f"Metric '{name}' not found. Available: {list(self.metrics.keys())[:10]}...")
                return 0.0
            
            if not self.metrics[name]:
                logger.warning(f"No values recorded for metric '{name}'")
                return 0.0
            
            # Safe access to the last value
            return float(self.metrics[name][-1])
            
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Error getting current value for metric '{name}': {e}")
            logger.error(f"Available metrics: {list(self.metrics.keys())[:10]}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error in get_current for metric '{name}': {e}")
            return 0.0
    
    def get_average(self, name: str) -> float:
        """Get the average value for a metric."""
        # Input validation
        if not isinstance(name, str):
            logger.error(f"get_average called with non-string name: {repr(name)} (type: {type(name)})")
            return 0.0
            
        try:
            if name not in self.metrics or not self.metrics[name]:
                return 0.0
            return float(np.mean(self.metrics[name]))
        except Exception as e:
            logger.error(f"Error computing average for metric '{name}': {e}")
            return 0.0
    
    def get_epoch_summary(self) -> Dict[str, float]:
        """Get summary of current epoch metrics"""
        return {name: self.get_average(name) for name in self.metrics.keys()}
        
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        
    def end_epoch(self):
        """Mark end of epoch and store epoch averages"""
        for name, values in self.metrics.items():
            if values:
                self.epoch_metrics[name].append(np.mean(values))
        self.reset()
        
    def get_epoch_summary(self) -> Dict[str, float]:
        """Get summary of current epoch metrics"""
        return {name: self.get_average(name) for name in self.metrics.keys()}
        
    def compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy"""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        correct = (predictions == targets).float().sum()
        total = targets.size(0)
        return (correct / total).item()
        
    def compute_categorical_coherence(self, functor_output: Dict[str, Any]) -> float:
        """Compute categorical coherence metric specific to GAIA"""
        if 'coherence_violations' in functor_output:
            violations = functor_output['coherence_violations']
            total_checks = functor_output.get('total_coherence_checks', 1)
            return 1.0 - (violations / total_checks)
        return 1.0
        
    def compute_horn_completion_rate(self, horn_results: Dict[str, Any]) -> float:
        """Compute horn completion success rate"""
        if 'completed_horns' in horn_results and 'total_horns' in horn_results:
            return horn_results['completed_horns'] / max(horn_results['total_horns'], 1)
        return 0.0
    
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """Compute and return current metrics summary"""
        result = {}
        for name, values in self.metrics.items():
            if values:
                metric_name = f"{prefix}{name}" if prefix else name
                result[metric_name] = np.mean(values)
        return result
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get complete metrics history"""
        return {name: list(values) for name, values in self.metrics.items()}
        

__all__ = ['MetricTracker']