"""
GAIA Core Module - Clean Architecture

This module provides the core categorical structures for the GAIA framework
"""

# Core constants and utilities
from .simplices import DEVICE

# Basic simplicial structures (no circular dependencies)
from .simplices import (
    SimplicialObject, Simplex0, Simplex1, Simplex2, SimplexN
)

# Functor structures (no circular dependencies)
from .functor import (
    SimplicialFunctor, MapType, HornError
)

# Integrated structures (existing, working)
from .integrated_structures import (
    IntegratedFuzzySet, IntegratedSimplex, IntegratedFuzzySimplicialSet,
    IntegratedCoalgebra, TConorm, FuzzyElement,
    create_fuzzy_simplex, create_fuzzy_simplicial_set_from_data,
    merge_fuzzy_simplicial_sets
)

# Kan verification (existing, working with dynamic imports)
from .kan_verification import KanComplexVerifier, KanConditionType, KanConditionResult

# Dynamic import functions to avoid circular dependencies
def get_training_components():
    """Get training components without circular imports"""
    try:
        from ..training.solvers.yoneda_proxy import SpectralNormalizedMetric, SpectralNormalizedLinear
        from ..training.solvers.inner_solver import EndofunctorialSolver
        from ..training.solvers.outer_solver import UniversalLiftingSolver
        return {
            'SpectralNormalizedMetric': SpectralNormalizedMetric,
            'SpectralNormalizedLinear': SpectralNormalizedLinear,
            'EndofunctorialSolver': EndofunctorialSolver,
            'UniversalLiftingSolver': UniversalLiftingSolver
        }
    except ImportError as e:
        print(f"Warning: Could not import training components: {e}")
        return {}

def get_advanced_components():
    """Get advanced components without circular imports"""
    try:
        # Import advanced components dynamically
        from .universal_coalgebras import (
            FCoalgebra, GenerativeCoalgebra, CoalgebraCategory
        )
        from .coalgebras import (
            create_parameter_coalgebra, BackpropagationEndofunctor
        )
        from .metric_yoneda import (
            GeneralizedMetricSpace, UniversalRepresenter
        )
        from .hierarchical_messaging import (
            HierarchicalMessagePasser
        )
        from .ends_coends import (
            End, Coend, GeometricTransformer
        )
        
        return {
            'FCoalgebra': FCoalgebra,
            'GenerativeCoalgebra': GenerativeCoalgebra,
            'CoalgebraCategory': CoalgebraCategory,
            'create_parameter_coalgebra': create_parameter_coalgebra,
            'BackpropagationEndofunctor': BackpropagationEndofunctor,
            'GeneralizedMetricSpace': GeneralizedMetricSpace,
            'UniversalRepresenter': UniversalRepresenter,
            'HierarchicalMessagePasser': HierarchicalMessagePasser,
            'End': End,
            'Coend': Coend,
            'GeometricTransformer': GeometricTransformer
        }
    except ImportError as e:
        print(f"Warning: Could not import advanced components: {e}")
        return {}

# Export core components (no circular dependencies)
__all__ = [
    # Constants
    'DEVICE',
    
    # Basic structures
    'SimplicialObject', 'Simplex0', 'Simplex1', 'Simplex2', 'SimplexN',
    
    # Functors
    'SimplicialFunctor', 'MapType', 'HornError',
    
    # Integrated structures
    'IntegratedFuzzySet', 'IntegratedSimplex', 'IntegratedFuzzySimplicialSet',
    'IntegratedCoalgebra', 'TConorm', 'FuzzyElement',
    'create_fuzzy_simplex', 'create_fuzzy_simplicial_set_from_data',
    'merge_fuzzy_simplicial_sets',
    
    # Kan verification
    'KanComplexVerifier', 'KanConditionType', 'KanConditionResult',
    
    # Dynamic import functions
    'get_training_components', 'get_advanced_components'
]