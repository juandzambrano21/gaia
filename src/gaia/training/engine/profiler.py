"""Performance Profiler for GAIA Framework"""

import torch
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import psutil
import threading
from pathlib import Path
import json

@dataclass
class ProfilerStats:
    """Statistics collected by the profiler"""
    
    # Timing statistics
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    data_loading_time: float = 0.0
    
    # Memory statistics
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_cached_mb: float = 0.0
    
    # Categorical operations
    horn_solving_time: float = 0.0
    coherence_check_time: float = 0.0
    categorical_loss_time: float = 0.0
    
    # System statistics
    cpu_percent: float = 0.0
    gpu_utilization: float = 0.0
    
    # Counters
    batch_count: int = 0
    step_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timing': {
                'forward_time': self.forward_time,
                'backward_time': self.backward_time,
                'optimizer_time': self.optimizer_time,
                'data_loading_time': self.data_loading_time,
                'horn_solving_time': self.horn_solving_time,
                'coherence_check_time': self.coherence_check_time,
                'categorical_loss_time': self.categorical_loss_time
            },
            'memory': {
                'peak_memory_mb': self.peak_memory_mb,
                'current_memory_mb': self.current_memory_mb,
                'memory_allocated_mb': self.memory_allocated_mb,
                'memory_cached_mb': self.memory_cached_mb
            },
            'system': {
                'cpu_percent': self.cpu_percent,
                'gpu_utilization': self.gpu_utilization
            },
            'counters': {
                'batch_count': self.batch_count,
                'step_count': self.step_count
            }
        }

class GAIAProfiler:
    """Performance profiler for GAIA training"""
    
    def __init__(
        self,
        enabled: bool = True,
        profile_memory: bool = True,
        profile_categorical: bool = True,
        log_interval: int = 100,
        save_dir: Optional[Path] = None
    ):
        """
        Initialize GAIA profiler
        
        Args:
            enabled: Whether profiling is enabled
            profile_memory: Whether to profile memory usage
            profile_categorical: Whether to profile categorical operations
            log_interval: Interval for logging statistics
            save_dir: Directory to save profiling results
        """
        self.enabled = enabled
        self.profile_memory = profile_memory
        self.profile_categorical = profile_categorical
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = ProfilerStats()
        self.step_stats_history: List[ProfilerStats] = []
        self.current_step_stats = ProfilerStats()
        
        # Timing contexts
        self._timers: Dict[str, float] = {}
        self._active_timers: Dict[str, float] = {}
        
        # Memory monitoring
        self._memory_monitor_active = False
        self._memory_thread: Optional[threading.Thread] = None
        
    def start_memory_monitoring(self):
        """Start continuous memory monitoring"""
        if not self.profile_memory or self._memory_monitor_active:
            return
            
        self._memory_monitor_active = True
        self._memory_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._memory_thread.start()
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring"""
        self._memory_monitor_active = False
        if self._memory_thread:
            self._memory_thread.join(timeout=1.0)
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread"""
        while self._memory_monitor_active:
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**2
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                cached_memory = torch.cuda.memory_reserved() / 1024**2
                
                self.current_step_stats.current_memory_mb = max(
                    self.current_step_stats.current_memory_mb, current_memory
                )
                self.current_step_stats.peak_memory_mb = max(
                    self.current_step_stats.peak_memory_mb, peak_memory
                )
                self.current_step_stats.memory_cached_mb = max(
                    self.current_step_stats.memory_cached_mb, cached_memory
                )
            
            # CPU usage
            self.current_step_stats.cpu_percent = psutil.cpu_percent()
            
            time.sleep(0.1)  # Monitor every 100ms
    
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling a code section"""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._add_timing(section_name, elapsed)
    
    def _add_timing(self, section_name: str, elapsed_time: float):
        """Add timing measurement"""
        if section_name == 'forward':
            self.current_step_stats.forward_time += elapsed_time
        elif section_name == 'backward':
            self.current_step_stats.backward_time += elapsed_time
        elif section_name == 'optimizer':
            self.current_step_stats.optimizer_time += elapsed_time
        elif section_name == 'data_loading':
            self.current_step_stats.data_loading_time += elapsed_time
        elif section_name == 'horn_solving':
            self.current_step_stats.horn_solving_time += elapsed_time
        elif section_name == 'coherence_check':
            self.current_step_stats.coherence_check_time += elapsed_time
        elif section_name == 'categorical_loss':
            self.current_step_stats.categorical_loss_time += elapsed_time
    
    def step(self):
        """Mark the end of a training step"""
        if not self.enabled:
            return
            
        self.current_step_stats.step_count = self.stats.step_count + 1
        self.current_step_stats.batch_count = self.stats.batch_count + 1
        
        # Update cumulative stats
        self.stats.forward_time += self.current_step_stats.forward_time
        self.stats.backward_time += self.current_step_stats.backward_time
        self.stats.optimizer_time += self.current_step_stats.optimizer_time
        self.stats.data_loading_time += self.current_step_stats.data_loading_time
        self.stats.horn_solving_time += self.current_step_stats.horn_solving_time
        self.stats.coherence_check_time += self.current_step_stats.coherence_check_time
        self.stats.categorical_loss_time += self.current_step_stats.categorical_loss_time
        
        self.stats.peak_memory_mb = max(self.stats.peak_memory_mb, 
                                       self.current_step_stats.peak_memory_mb)
        self.stats.current_memory_mb = self.current_step_stats.current_memory_mb
        self.stats.memory_cached_mb = self.current_step_stats.memory_cached_mb
        self.stats.cpu_percent = self.current_step_stats.cpu_percent
        
        self.stats.step_count += 1
        self.stats.batch_count += 1
        
        # Save step stats
        self.step_stats_history.append(self.current_step_stats)
        
        # Log periodically
        if self.stats.step_count % self.log_interval == 0:
            self.log_stats()
        
        # Reset current step stats
        self.current_step_stats = ProfilerStats()
    
    def log_stats(self):
        """Log current profiling statistics"""
        if not self.enabled:
            return
            
        avg_forward = self.stats.forward_time / max(self.stats.step_count, 1)
        avg_backward = self.stats.backward_time / max(self.stats.step_count, 1)
        avg_optimizer = self.stats.optimizer_time / max(self.stats.step_count, 1)
        
        print(f"\n=== GAIA Profiler Stats (Step {self.stats.step_count}) ===")
        print(f"Timing (avg per step):")
        print(f"  Forward:     {avg_forward*1000:.2f}ms")
        print(f"  Backward:    {avg_backward*1000:.2f}ms")
        print(f"  Optimizer:   {avg_optimizer*1000:.2f}ms")
        
        if self.profile_categorical:
            avg_horn = self.stats.horn_solving_time / max(self.stats.step_count, 1)
            avg_coherence = self.stats.coherence_check_time / max(self.stats.step_count, 1)
            print(f"  Horn Solving: {avg_horn*1000:.2f}ms")
            print(f"  Coherence:   {avg_coherence*1000:.2f}ms")
        
        if self.profile_memory:
            print(f"Memory:")
            print(f"  Current:     {self.stats.current_memory_mb:.1f}MB")
            print(f"  Peak:        {self.stats.peak_memory_mb:.1f}MB")
            print(f"  Cached:      {self.stats.memory_cached_mb:.1f}MB")
            print(f"  CPU Usage:   {self.stats.cpu_percent:.1f}%")
        
        print("=" * 50)
    
    def save_stats(self, filename: Optional[str] = None):
        """Save profiling statistics to file"""
        if not self.enabled or not self.save_dir:
            return
            
        if filename is None:
            filename = f"profiler_stats_step_{self.stats.step_count}.json"
        
        filepath = self.save_dir / filename
        
        # Prepare data for saving
        data = {
            'cumulative_stats': self.stats.to_dict(),
            'step_history': [stats.to_dict() for stats in self.step_stats_history[-100:]]  # Last 100 steps
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.enabled or self.stats.step_count == 0:
            return {}
            
        total_time = (
            self.stats.forward_time + 
            self.stats.backward_time + 
            self.stats.optimizer_time
        )
        
        summary = {
            'total_steps': self.stats.step_count,
            'total_time_seconds': total_time,
            'avg_step_time_ms': (total_time / self.stats.step_count) * 1000,
            'time_breakdown': {
                'forward_pct': (self.stats.forward_time / total_time) * 100 if total_time > 0 else 0,
                'backward_pct': (self.stats.backward_time / total_time) * 100 if total_time > 0 else 0,
                'optimizer_pct': (self.stats.optimizer_time / total_time) * 100 if total_time > 0 else 0,
            },
            'memory_peak_mb': self.stats.peak_memory_mb,
            'categorical_time_ms': (self.stats.horn_solving_time + self.stats.coherence_check_time) * 1000
        }
        
        return summary
    
    def reset(self):
        """Reset profiler statistics"""
        self.stats = ProfilerStats()
        self.step_stats_history.clear()
        self.current_step_stats = ProfilerStats()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()