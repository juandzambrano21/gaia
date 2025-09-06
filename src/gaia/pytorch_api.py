"""
GAIA PyTorch API - Base Classes

PyTorch-compatible base classes for GAIA framework components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional

class GAIAModule(nn.Module):
    """
    Base class for GAIA neural network modules
    
    Extends PyTorch nn.Module with categorical deep learning capabilities
    """
    
    def __init__(self):
        super().__init__()
        self._gaia_metadata = {
            'categorical_structure': True,
            'simplicial_dimension': 0,
            'yoneda_enhanced': False,
            'spectral_normalized': False
        }
    
    def set_gaia_metadata(self, **kwargs):
        """Set GAIA-specific metadata"""
        self._gaia_metadata.update(kwargs)
    
    def get_gaia_metadata(self) -> Dict[str, Any]:
        """Get GAIA-specific metadata"""
        return self._gaia_metadata.copy()

class GAIAOptimizer(optim.Optimizer):
    """
    Base class for GAIA optimizers
    
    Extends PyTorch Optimizer with categorical enhancements
    """
    
    def __init__(self, params, lr=1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
        self._gaia_metadata = {
            'endofunctorial': False,
            'yoneda_enhanced': False,
            'coalgebra_aware': False
        }

class GAIALoss(nn.Module):
    """
    Base class for GAIA loss functions
    
    Extends PyTorch nn.Module for loss computation with categorical enhancements
    """
    
    def __init__(self, loss_type: str = 'categorical'):
        super().__init__()
        self.loss_type = loss_type
        self._gaia_metadata = {
            'categorical_structure': True,
            'simplicial_aware': False,
            'yoneda_regularized': False
        }