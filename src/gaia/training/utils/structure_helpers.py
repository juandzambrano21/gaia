"""StructureHelpers - Extracted from GAIATrainer

This module contains methods related to structure and has structure operations.
Extracted during refactoring to improve maintainability.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class StructureHelpers:
    """StructureHelpers operations for GAIA trainer"""
    
    def __init__(self, functor, config=None):
        self.functor = functor
        self.config = config
        
