"""GAIA Categorical Solvers"""

from .inner_solver import EndofunctorialSolver
from .outer_solver import UniversalLiftingSolver
from .yoneda_proxy import MetricYonedaProxy

__all__ = [
    'EndofunctorialSolver',
    'UniversalLiftingSolver', 
    'MetricYonedaProxy'
]