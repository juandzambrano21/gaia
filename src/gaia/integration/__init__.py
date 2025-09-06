"""
GAIA Integration Layer - Resolves Circular Imports

This module provides a clean integration layer that resolves circular import issues
and provides a unified interface to all GAIA components.

"""

# Core theoretical components (no circular dependencies)
from ..core.universal_coalgebras import (
    FCoalgebra, CoalgebraHomomorphism, Bisimulation,
    GenerativeCoalgebra, CoalgebraCategory,
    Endofunctor, PowersetFunctor, StreamFunctor, NeuralFunctor,
    create_llm_coalgebra, create_diffusion_coalgebra, create_transformer_coalgebra
)

from ..core.metric_yoneda import (
    GeneralizedMetricSpace, EnrichedCategory, YonedaEmbedding,
    MetricYonedaApplications, UniversalRepresenter,
    create_llm_metric_space, create_causal_metric_space
)

from ..core.integrated_structures import (
    IntegratedFuzzySet, IntegratedSimplex, IntegratedFuzzySimplicialSet,
    IntegratedCoalgebra, TConorm, FuzzyElement,
    create_fuzzy_simplex, create_fuzzy_simplicial_set_from_data,
    merge_fuzzy_simplicial_sets
)

# Training system components (imported separately to avoid circular deps)
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

# Verification components
def get_verification_components():
    """Get verification components without circular imports"""
    try:
        from ..core.kan_verification import KanConditionVerifier, KanConditionType, KanConditionResult
        return {
            'KanConditionVerifier': KanConditionVerifier,
            'KanConditionType': KanConditionType,
            'KanConditionResult': KanConditionResult
        }
    except ImportError as e:
        print(f"Warning: Could not import verification components: {e}")
        return {}

__all__ = [
    # Universal Coalgebras
    'FCoalgebra', 'CoalgebraHomomorphism', 'Bisimulation',
    'GenerativeCoalgebra', 'CoalgebraCategory',
    'Endofunctor', 'PowersetFunctor', 'StreamFunctor', 'NeuralFunctor',
    'create_llm_coalgebra', 'create_diffusion_coalgebra', 'create_transformer_coalgebra',
    
    # Metric Yoneda
    'GeneralizedMetricSpace', 'EnrichedCategory', 'YonedaEmbedding',
    'MetricYonedaApplications', 'UniversalRepresenter',
    'create_llm_metric_space', 'create_causal_metric_space',
    
    # Integrated Structures
    'IntegratedFuzzySet', 'IntegratedSimplex', 'IntegratedFuzzySimplicialSet',
    'IntegratedCoalgebra', 'TConorm', 'FuzzyElement',
    'create_fuzzy_simplex', 'create_fuzzy_simplicial_set_from_data',
    'merge_fuzzy_simplicial_sets',
    
    # Dynamic component getters
    'get_training_components', 'get_verification_components'
]