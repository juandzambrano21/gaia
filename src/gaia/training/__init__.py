"""
GAIA Training Module

This module contains training components for the GAIA framework:
- Unified trainer integrating all GAIA components
- Legacy trainer for backward compatibility
- Training configuration and hyperparameters
- Hierarchical message passing system
- Training loops and optimization engines
- Solver systems (inner/outer horn solvers)
- Profiling and monitoring utilities

All training components integrate with the core categorical structures.
"""

# New unified training system (recommended)
from .trainer import (
    GAIATrainer, GAIATrainingConfig, FuzzyDataEncoder,
    CoalgebraEvolution, HierarchicalCommunication, KanVerification,
    create_gaia_trainer
)
from .categorical_losses import CategoricalLossComputer, CategoricalLoss
from .categorical_operations import CategoricalOperations, CategoricalOps

# Legacy training system (for backward compatibility)
from .trainer import GAIATrainer as GAIATrainer
from .config import TrainingConfig, ModelConfig, DataConfig, OptimizationConfig

# Categorical solvers
from .solvers import EndofunctorialSolver, UniversalLiftingSolver, MetricYonedaProxy

# Training engine
from .engine import TrainingLoop, ValidationLoop, TrainingState, CheckpointManager, GAIAProfiler

# Training utilities
from .utils import StructureHelpers, ValidationUtils


__all__ = [
    # New unified system
    'GAIATrainer', 'GAIATrainingConfig', 'FuzzyDataEncoder',
    'CoalgebraEvolution', 'HierarchicalCommunication', 'KanVerification',
    'create_gaia_trainer',
    
    # Categorical training components
    'CategoricalLossComputer', 'CategoricalLoss',
    'CategoricalOperations', 'CategoricalOps',
    
    # Legacy training
    'GAIATrainer',
    'TrainingConfig',
    'ModelConfig', 
    'DataConfig',
    'OptimizationConfig',
    
    # Categorical solvers
    'EndofunctorialSolver',
    'UniversalLiftingSolver',
    'MetricYonedaProxy',
    
    # Training engine
    'TrainingLoop',
    'ValidationLoop',
    'TrainingState',
    'CheckpointManager',
    'GAIAProfiler',
    
    # Utilities
    'StructureHelpers',
    'ValidationUtils',
    
    # Hierarchical message passing
    'HierarchicalMessagePassingSystem',
    'SimplexParameters',
    'LocalObjectiveFunction',
    'DegeneracyInstruction',
    'create_hierarchical_system_from_model',
    'integrate_with_training_loop'
]