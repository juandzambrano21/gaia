"""Production Training Configuration System"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import yaml
from enum import Enum

import logging
from typing import Optional


class OptimizationType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    LION = "lion"
    CATEGORICAL_ADAM = "categorical_adam"  # GAIA-specific

class SchedulerType(Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    CATEGORICAL_ANNEALING = "categorical_annealing"  # GAIA-specific

@dataclass
class OptimizationConfig:
    """Optimization configuration with GAIA categorical extensions"""
    optimizer: OptimizationType = OptimizationType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # GAIA-specific categorical optimization
    categorical_coherence_weight: float = 1.0
    simplicial_regularization: float = 0.1
    horn_solving_weight: float = 0.5
    
    # Scheduler configuration
    scheduler: SchedulerType = SchedulerType.COSINE
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 1000
    
    # Gradient management
    gradient_clip_norm: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "gaia_model"
    architecture: str = "categorical_mlp"
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = True
    
    # Transformer-specific
    vocab_size: int = 5000
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    max_seq_length: int = 256
    seq_len: int = 128  # For compatibility
    
    # GAIA-specific categorical structure
    simplicial_depth: int = 3
    categorical_embedding_dim: int = 64
    horn_solver_config: Dict[str, Any] = field(default_factory=dict)
    
    # Model initialization
    init_method: str = "xavier_uniform"
    init_gain: float = 1.0
    
@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    dataset_path: Union[str, Path] = ""
    batch_size: int = 4
    num_workers: int = 0  # Change from 4 to 0 for macOS compatibility
    pin_memory: bool = False  # Change from True to False for MPS
    persistent_workers: bool = False  # Change from True to False
    
    # Data preprocessing
    normalize: bool = True
    augmentation: bool = False
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Validation split
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # UMAP configuration for fuzzy encoding pipeline
    n_neighbors: int = 15
    min_dist: float = 0.1
    spread: float = 1.0
    metric: str = "euclidean"
    n_components: int = 2
    local_connectivity: float = 1.0
    bandwidth: float = 1.0
    
    # GAIA-specific categorical data
    categorical_features: List[str] = field(default_factory=list)
    simplicial_structure: Optional[Dict[str, Any]] = None
    
    # Dataset configuration
    use_sample_dataset: bool = True  # Use simple sample dataset to avoid memory issues
    sample_dataset_size: int = 20
    
    # Performance configuration
    enable_hierarchical_messaging: bool = False  # Disable for faster training
    enable_horn_detection: bool = False  # Disable for faster training
    quick_demo_mode: bool = True  # Enable quick demo mode
    
@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Basic training parameters
    epochs: int = 2
    max_steps: Optional[int] = None
    eval_frequency: int = 1000
    save_frequency: int = 5000
    
    # Logging and monitoring
    log_level: str = "INFO"
    log_frequency: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "gaia-training"
    
    # Checkpointing
    checkpoint_dir: Union[str, Path] = "checkpoints"
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Hardware and performance
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = True
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # GAIA-specific training
    categorical_training: bool = True
    hierarchical_learning: bool = True  # Add this line
    horn_solving_frequency: int = 1000
    coherence_check_frequency: int = 500
    
    # Configuration components
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**config_dict)
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        config_dict = self.__dict__.copy()
        
        # Convert dataclass fields to dicts
        config_dict['model'] = self.model.__dict__
        config_dict['data'] = self.data.__dict__
        config_dict['optimization'] = self.optimization.__dict__
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")



class GAIAConfig:
    """Global configuration class for GAIA framework."""
    
    # Global debug flag
    ULTRA_VERBOSE_DEBUG = True
    
    # Default logging configuration
    DEFAULT_LOG_LEVEL = logging.DEBUG
    DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def setup_logging(cls, 
                     level: Optional[int] = None,
                     format_str: Optional[str] = None,
                     enable_ultra_verbose: bool = True) -> None:
        """Setup global logging configuration for GAIA framework.
        
        Args:
            level: Logging level (default: DEBUG)
            format_str: Log format string
            enable_ultra_verbose: Enable ultra-verbose debugging
        """
        if level is None:
            level = cls.DEFAULT_LOG_LEVEL
        if format_str is None:
            format_str = cls.DEFAULT_LOG_FORMAT
            
        # Configure root logging
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),  # Console output
            ],
            force=True  # Override any existing configuration
        )
        
        # Set ALL GAIA modules to DEBUG level
        gaia_loggers = [
            'gaia',
            'gaia.core',
            'gaia.core.universal_coalgebras',
            'gaia.core.kan_extensions',
            'gaia.core.hierarchical_messaging',
            'gaia.core.fuzzy',
            'gaia.core.metric_yoneda',
            'gaia.core.ends_coends',
            'gaia.models',
            'gaia.training',
            'gaia.data',
            'gaia.utils'
        ]
        
        for logger_name in gaia_loggers:
            logging.getLogger(logger_name).setLevel(level)
        
        # Update global debug flag
        cls.ULTRA_VERBOSE_DEBUG = enable_ultra_verbose
        
        # Log configuration status
        logger = logging.getLogger('gaia.config')
        logger.info(f"🔧 GAIA CONFIG: ULTRA-VERBOSE DEBUG MODE = {cls.ULTRA_VERBOSE_DEBUG}")
        logger.info(f"🔧 GAIA CONFIG: Root logging level = {logging.getLogger().level}")
        logger.info(f"🔧 GAIA CONFIG: GAIA logging level = {logging.getLogger('gaia').level}")
        
    @classmethod
    def is_debug_enabled(cls) -> bool:
        """Check if ultra-verbose debugging is enabled."""
        return cls.ULTRA_VERBOSE_DEBUG
        
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with GAIA configuration.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        if cls.ULTRA_VERBOSE_DEBUG:
            logger.setLevel(logging.DEBUG)
        return logger


# Initialize default configuration
GAIAConfig.setup_logging()