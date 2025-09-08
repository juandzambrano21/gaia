"""Base GAIA Model Interface

This module defines the abstract base class for all GAIA models,
providing a standardized interface for model operations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class BaseGAIAModel(nn.Module, ABC):
    """Abstract base class for all GAIA models.
    
    This class defines the standard interface that all GAIA models should implement,
    including methods for training, inference, saving, and loading models.
    """
    
    def __init__(self, config: Any, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_metadata = {
            'model_type': getattr(config, 'model_type', 'base_gaia_model'),
            'version': getattr(config, 'version', '1.0.0'),
            'framework': 'GAIA',
            'created_at': None,
            'trained': False
        }
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def fit(self, dataset: Union[List[str], Any], **kwargs) -> Dict[str, Any]:
        """Train the model on the given dataset.
        
        Args:
            dataset: Training dataset
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics and results
        """
        pass
    
    def generate(self, input_text: str, max_length: int = 100, **kwargs) -> str:
        """Generate text from the model.
        
        Args:
            input_text: Input text to start generation
            max_length: Maximum length of generated text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Default implementation - can be overridden by subclasses
        logger.warning(f"Generate method not implemented for {self.__class__.__name__}")
        return input_text
    
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs) -> None:
        """Save the model and its configuration.
        
        Args:
            save_directory: Directory to save the model
            **kwargs: Additional save parameters
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = save_path / 'pytorch_model.bin'
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = save_path / 'config.json'
        config_dict = self._config_to_dict()
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model metadata
        metadata_path = save_path / 'model_metadata.json'
        self.model_metadata['trained'] = True
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_directory: Union[str, Path], **kwargs):
        """Load a pretrained model.
        
        Args:
            model_directory: Directory containing the saved model
            **kwargs: Additional load parameters
            
        Returns:
            Loaded model instance
        """
        model_path = Path(model_directory)
        
        # Load configuration
        config_path = model_path / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object (this would need to be implemented by subclasses)
        config = cls._dict_to_config(config_dict)
        
        # Create model instance
        model = cls(config, **kwargs)
        
        # Load model state dict
        model_state_path = model_path / 'pytorch_model.bin'
        if model_state_path.exists():
            state_dict = torch.load(model_state_path, map_location=model.device)
            model.load_state_dict(state_dict)
        
        # Load metadata if available
        metadata_path = model_path / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model.model_metadata = json.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization.
        
        Returns:
            Configuration as dictionary
        """
        if hasattr(self.config, '__dict__'):
            return self.config.__dict__
        elif hasattr(self.config, '_asdict'):
            return self.config._asdict()
        else:
            # Fallback for basic config objects
            return {'config_type': str(type(self.config))}
    
    @classmethod
    def _dict_to_config(cls, config_dict: Dict[str, Any]):
        """Convert dictionary to config object.
        
        This method should be overridden by subclasses to properly
        reconstruct their specific config types.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        # This is a placeholder - subclasses should implement proper config reconstruction
        logger.warning(f"Using basic config reconstruction for {cls.__name__}")
        return type('Config', (), config_dict)()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_class': self.__class__.__name__,
            'model_type': self.model_metadata.get('model_type'),
            'version': self.model_metadata.get('version'),
            'device': str(self.device),
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'metadata': self.model_metadata
        }
    
    def to_device(self, device: Optional[str] = None) -> 'BaseGAIAModel':
        """Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        if device:
            self.device = device
        self.to(self.device)
        return self