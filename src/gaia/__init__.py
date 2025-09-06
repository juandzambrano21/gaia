"""
GAIA Framework - Generative Algebraic Intelligence Architecture

A  categorical deep learning framework based on category theory.

Usage:
    import gaia
    from gaia.models import GAIATransformer
    from gaia.nn import SpectralLinear
    from gaia.core import SimplicialFunctor
"""

__version__ = "1.0.0"
__author__ = "GAIA Framework Team"
__description__ = "Generative Algebraic Intelligence Architecture - Categorical Deep Learning Framework"

# Core imports
from .core import (
    SimplicialFunctor,
    Simplex0, Simplex1, SimplexN,
    IntegratedFuzzySet, IntegratedFuzzySimplicialSet,
    KanComplexVerifier,
    get_training_components,
    get_advanced_components
)

# Aliases for convenience
FuzzySet = IntegratedFuzzySet
FuzzySimplicialSet = IntegratedFuzzySimplicialSet

# Factory functions
def create_model(model_type='transformer', **kwargs):
    """Create GAIA models."""
    if model_type == 'transformer':
        from .models.gaia_transformer import create_gaia_llm
        return create_gaia_llm(**kwargs)
    elif model_type == 'mlp':
        from .models.categorical_mlp import CategoricalMLP
        return CategoricalMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Version info
def version():
    """Get GAIA version"""
    return __version__

def info():
    """Get GAIA framework information"""
    return {
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'components': {
            'core': True,
            'training': True,
            'advanced': True,
            'pytorch_api': True
        }
    }

# Framework status
def status():
    """Check GAIA framework status"""
    try:
        training_comps = get_training_components()
        advanced_comps = get_advanced_components()
        
        return {
            'status': 'operational',
            'training_components': len(training_comps),
            'advanced_components': len(advanced_comps),
            'total_components': len(training_comps) + len(advanced_comps),
            'pytorch_compatible': True,
            'production_ready': True
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'pytorch_compatible': False,
            'production_ready': False
        }

# Convenience functions
def list_components():
    """List all available GAIA components"""
    try:
        training_comps = get_training_components()
        advanced_comps = get_advanced_components()
        
        return {
            'training': list(training_comps.keys()),
            'advanced': list(advanced_comps.keys())
        }
    except Exception as e:
        return {'error': str(e)}

# Export main classes
__all__ = [
    # Core components
    'SimplicialFunctor',
    'Simplex0', 'Simplex1', 'SimplexN',
    'FuzzySet', 'FuzzySimplicialSet',
    'EndofunctorialSolver', 'UniversalLiftingSolver',
    'KanComplexVerifier',
    'HierarchicalMessagePasser',
    
    # Factory functions
    'create_model',
    
    # Utilities
    'get_training_components',
    'get_advanced_components',
    'version',
    'info',
    'status',
    'list_components'
]
