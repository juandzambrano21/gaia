"""ValidationUtils - Extracted from GAIATrainer

This module contains methods related to validate and check valid operations.
Extracted during refactoring to improve maintainability.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class ValidationUtils:
    """ValidationUtils operations for GAIA trainer"""
    
    def __init__(self, functor, config=None):
        self.functor = functor
        self.config = config
        
