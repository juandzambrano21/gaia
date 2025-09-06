"""Device and distributed training utilities"""

import torch
import torch.distributed as dist
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for training"""
    if device is not None and device != "auto":
        return torch.device(device)
        
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        
    return device

def setup_distributed(world_size: int, rank: int, backend: str = 'nccl'):
    """Setup distributed training"""
    if world_size <= 1:
        return
        
    # Initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank
        )
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
        
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")