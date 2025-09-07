"""GAIA Data Processing and Loading"""

from .synthetic import create_synthetic_dataset, create_xor_dataset, create_regression_dataset
from .categorical import CategoricalDataset, SimplicalDataLoader
from .transforms import CategoricalTransform, YonedaTransform, SimplicalTransform, RobustNormalization
from .utils import load_any_dataset, validate_dataset, SKLEARN_DATASETS
from .fuzzy_encoding import (
    FuzzyEncodingPipeline, UMAPConfig, encode_point_cloud, 
    encode_graph_data, create_synthetic_fuzzy_complex
)
from .loaders import (
    GAIADataManager, DatasetConfig, DataLoaderConfig,
    DataLoaders, create_standard_language_loaders
)
from .dataset import Dataset

__all__ = [
    # Synthetic data generation
    'create_synthetic_dataset',
    'create_xor_dataset',
    'create_regression_dataset',
    
    # Dataset and DataLoader classes
    'CategoricalDataset',
    'SimplicalDataLoader',
    
    # Transforms
    'CategoricalTransform',
    'YonedaTransform',
    'SimplicalTransform',
    'RobustNormalization',
    
    # Utilities
    'load_any_dataset',
    'validate_dataset',
    'SKLEARN_DATASETS',
    
    # Fuzzy encoding pipeline
    'FuzzyEncodingPipeline',
    'UMAPConfig',
    'encode_point_cloud',
    'encode_graph_data',
    'create_synthetic_fuzzy_complex',
    
    # Extensible data loading architecture
    'GAIADataManager',
    'DatasetConfig',
    'DataLoaderConfig',
    'DataLoaders',
    'create_standard_language_loaders',
    'Dataset'
]

# Convenience function for quick dataset loading
def quick_load(source, **kwargs):
    """Quick dataset loading with sensible defaults"""
    return load_any_dataset(source, **kwargs)

# Dataset validation decorator
def validate_input(func):
    """Decorator to validate dataset inputs"""
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            X = args[0]
            y = args[1] if len(args) > 1 else None
            report = validate_dataset(X, y)
            if not report['valid']:
                raise ValueError(f"Dataset validation failed: {report['errors']}")
            if report['warnings']:
                import warnings
                warnings.warn(f"Dataset warnings: {report['warnings']}")
        return func(*args, **kwargs)
    return wrapper