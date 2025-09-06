"""
Module: identity
Provides utility functions for creating identity morphisms.
"""

from typing import Optional
import torch.nn as nn
from .simplices import Simplex0, Simplex1
from . import DEVICE

def id_edge(v: Simplex0, functor: Optional[object] = None) -> Simplex1:
    """
    Create a strong-ref identity edge for 0-simplex v.
    If functor is provided, adds the identity edge to the functor if not already present.
    """
    # reuse existing identity if present
    if not hasattr(v, '_id_edge') or v._id_edge is None:
        v._id_edge = Simplex1(nn.Identity().to(DEVICE), v, v, f'id_{v.name}')
    if functor is not None:
        # add only if not already present
        try:
            if v._id_edge.id not in functor.registry:
                functor.add(v._id_edge)
        except Exception:
            pass
    return v._id_edge