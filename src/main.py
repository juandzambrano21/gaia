#!/usr/bin/env python3

"""
GAIA Framework - Generative Algebraic Intelligence Architecture
===============================================================

A production-ready categorical deep learning framework based on category theory.

This is the main entry point for the GAIA library, providing access to all
categorical deep learning components as specified in the theoretical framework.

Usage:
    import gaia
    from gaia.models import GAIATransformer
    from gaia.nn import SpectralLinear, YonedaMetric
    from gaia.core import SimplicialFunctor, FuzzySet
"""

# Core categorical components
from gaia.core import (
    Simplex0, Simplex1, SimplexN,
    SimplicialFunctor,
    IntegratedFuzzySet as FuzzySet,
    IntegratedFuzzySimplicialSet as FuzzySimplicialSet,
    KanComplexVerifier,
    get_training_components,
    get_advanced_components
)

# Neural network components
from gaia.nn import (
    GAIAModule,
    SpectralLinear,
    YonedaMetric,
    SimplicialLayer,
    CategoricalMLP,
    CoalgebraNetwork,
    GeometricTransformer,
    CategoricalActivation,
    CategoricalLoss
)

# Models
from gaia.models.gaia_transformer import GAIATransformer, create_gaia_llm
from gaia.models.categorical_mlp import CategoricalMLP

# Data processing
from gaia.data.transforms import SimplicalTransform as SimplicialTransform
from gaia.data.categorical import CategoricalDataset
from gaia.data.utils import _analyze_categorical_structure as analyze_categorical_structure

# Version info
__version__ = "1.0.0"
__author__ = "GAIA Framework Team"
__description__ = "Generative Algebraic Intelligence Architecture - Categorical Deep Learning Framework"

# Main components for easy access
__all__ = [
    # Core categorical theory
    'Simplex0', 'Simplex1', 'SimplexN',
    'SimplicialFunctor',
    'FuzzySet', 'FuzzySimplicialSet',
    'KanComplexVerifier',
    
    # Neural networks
    'GAIAModule',
    'SpectralLinear', 'YonedaMetric', 'SimplicialLayer',
    'CategoricalMLP', 'CoalgebraNetwork', 'GeometricTransformer',
    'CategoricalActivation', 'CategoricalLoss',
    
    # Models
    'GAIATransformer', 'create_gaia_llm',
    'CategoricalMLP',
    
    # Data processing
    'SimplicialTransform', 'CategoricalDataset',
    'analyze_categorical_structure',
    
    # Utilities
    'get_training_components', 'get_advanced_components',
    
    # Metadata
    '__version__', '__author__', '__description__'
]

def get_framework_info():
    """
    Get information about the GAIA framework.
    
    Returns:
        dict: Framework information including version, components, and capabilities
    """
    return {
        'name': 'GAIA Framework',
        'version': __version__,
        'description': __description__,
        'components': {
            'categorical_theory': [
                'Simplicial Sets', 'Fuzzy Sets', 'Functors', 
                'Horn Solvers', 'Kan Complexes', 'Hierarchical Messaging'
            ],
            'training': [
                'Yoneda Metrics', 'Spectral Normalization',
                'F-Coalgebras', 'Universal Coalgebras'
            ],
            'neural_networks': [
                'Spectral Linear Layers', 'Categorical MLPs',
                'Geometric Transformers', 'Coalgebra Networks'
            ],
            'models': [
                'GAIA Transformer', 'Categorical MLP'
            ]
        },
        'capabilities': [
            'Categorical Deep Learning',
            'Fuzzy Simplicial Sets',
            'Hierarchical Message Passing',
            'Spectral Normalization',
            'Yoneda Enhancement',
            'F-Coalgebra Evolution',
            'Kan Complex Verification'
        ]
    }

def create_model(model_type='transformer', **kwargs):
    """
    Factory function to create GAIA models.
    
    Args:
        model_type (str): Type of model ('transformer', 'mlp')
        **kwargs: Model-specific arguments
        
    Returns:
        GAIA model instance
    """
    if model_type == 'transformer':
        return create_gaia_llm(**kwargs)
    elif model_type == 'mlp':
        return CategoricalMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Framework initialization
def initialize_gaia():
    """Initialize the GAIA framework with all components."""
    try:
        # Verify core components are available
        from gaia.core import get_training_components, get_advanced_components
        
        training_comps = get_training_components()
        advanced_comps = get_advanced_components()
        
        print(f"GAIA Framework v{__version__} initialized successfully")
        print(f"Training components: {len(training_comps)}")
        print(f"Advanced components: {len(advanced_comps)}")
        
        return True
    except Exception as e:
        print(f"GAIA Framework initialization failed: {e}")
        return False

if __name__ == "__main__":
    # When run as script, show framework info
    info = get_framework_info()
    print("=" * 60)
    print(f"{info['name']} v{info['version']}")
    print("=" * 60)
    print(f"Description: {info['description']}")
    print()
    
    print("Components:")
    for category, components in info['components'].items():
        print(f"  {category.replace('_', ' ').title()}:")
        for comp in components:
            print(f"    - {comp}")
    
    print()
    print("Capabilities:")
    for cap in info['capabilities']:
        print(f"  - {cap}")
    
    print()
    print("Usage:")
    print("  import gaia")
    print("  model = gaia.create_model('transformer', vocab_size=10000)")
    print("  from gaia.nn import SpectralLinear")
    print("  layer = SpectralLinear(512, 256)")
    
    # Initialize framework
    print()
    initialize_gaia()