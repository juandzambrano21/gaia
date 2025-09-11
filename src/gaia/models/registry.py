"""GAIA Model Registry System

This module provides a centralized registry for managing different GAIA model
implementations with proper metadata, versioning, and discovery capabilities.
"""

import json
import os
from typing import Dict, List, Optional, Type, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .base_model import BaseGAIAModel
from ..training.config import GAIAConfig

# Setup logging
logger = GAIAConfig.get_logger('model_registry')


@dataclass
class ModelMetadata:
    """Metadata for a registered GAIA model."""
    name: str
    version: str
    description: str
    model_class: str
    config_class: str
    author: str
    created_at: str
    updated_at: str
    tags: List[str]
    capabilities: List[str]
    requirements: Dict[str, str]
    performance_metrics: Dict[str, float]
    model_size: Optional[int] = None
    training_data: Optional[str] = None
    license: Optional[str] = None
    paper_url: Optional[str] = None
    code_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class GAIAModelRegistry:
    """Registry for managing GAIA model implementations."""
    
    def __init__(self, registry_path: Optional[str] = None):
        """Initialize the model registry.
        
        Args:
            registry_path: Path to store registry data. If None, uses default location.
        """
        if registry_path is None:
            registry_path = os.path.join(os.path.expanduser("~"), ".gaia", "model_registry.json")
        
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._models: Dict[str, ModelMetadata] = {}
        self._model_classes: Dict[str, Type[BaseGAIAModel]] = {}
        self._config_classes: Dict[str, Type] = {}
        
        # Load existing registry
        self._load_registry()
        
        # Register built-in models
        self._register_builtin_models()
    
    def register_model(
        self,
        name: str,
        version: str,
        model_class: Type[BaseGAIAModel],
        config_class: Type,
        description: str = "",
        author: str = "Unknown",
        tags: List[str] = None,
        capabilities: List[str] = None,
        requirements: Dict[str, str] = None,
        performance_metrics: Dict[str, float] = None,
        **kwargs
    ) -> None:
        """Register a new GAIA model.
        
        Args:
            name: Model name
            version: Model version
            model_class: Model class that inherits from BaseGAIAModel
            config_class: Configuration class for the model
            description: Model description
            author: Model author
            tags: List of tags for categorization
            capabilities: List of model capabilities
            requirements: Dictionary of requirements
            performance_metrics: Dictionary of performance metrics
            **kwargs: Additional metadata fields
        """
        if not issubclass(model_class, BaseGAIAModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseGAIAModel")
        
        model_key = f"{name}:{version}"
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            model_class=f"{model_class.__module__}.{model_class.__name__}",
            config_class=f"{config_class.__module__}.{config_class.__name__}",
            author=author,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=tags or [],
            capabilities=capabilities or [],
            requirements=requirements or {},
            performance_metrics=performance_metrics or {},
            **kwargs
        )
        
        # Store in registry
        self._models[model_key] = metadata
        self._model_classes[model_key] = model_class
        self._config_classes[model_key] = config_class
        
        # Save to disk
        self._save_registry()
        
    
    def get_model_class(self, name: str, version: str = "latest") -> Type[BaseGAIAModel]:
        """Get model class by name and version.
        
        Args:
            name: Model name
            version: Model version or 'latest'
            
        Returns:
            Model class
        """
        if version == "latest":
            version = self._get_latest_version(name)
        
        model_key = f"{name}:{version}"
        
        if model_key not in self._model_classes:
            raise ValueError(f"Model {name} v{version} not found in registry")
        
        return self._model_classes[model_key]
    
    def get_config_class(self, name: str, version: str = "latest") -> Type:
        """Get config class by name and version.
        
        Args:
            name: Model name
            version: Model version or 'latest'
            
        Returns:
            Config class
        """
        if version == "latest":
            version = self._get_latest_version(name)
        
        model_key = f"{name}:{version}"
        
        if model_key not in self._config_classes:
            raise ValueError(f"Config for model {name} v{version} not found in registry")
        
        return self._config_classes[model_key]
    
    def get_metadata(self, name: str, version: str = "latest") -> ModelMetadata:
        """Get model metadata by name and version.
        
        Args:
            name: Model name
            version: Model version or 'latest'
            
        Returns:
            Model metadata
        """
        if version == "latest":
            version = self._get_latest_version(name)
        
        model_key = f"{name}:{version}"
        
        if model_key not in self._models:
            raise ValueError(f"Model {name} v{version} not found in registry")
        
        return self._models[model_key]
    
    def list_models(self, tag: Optional[str] = None) -> List[ModelMetadata]:
        """List all registered models.
        
        Args:
            tag: Optional tag to filter by
            
        Returns:
            List of model metadata
        """
        models = list(self._models.values())
        
        if tag:
            models = [m for m in models if tag in m.tags]
        
        return sorted(models, key=lambda x: (x.name, x.version))
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """Search models by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching model metadata
        """
        query = query.lower()
        results = []
        
        for metadata in self._models.values():
            if (query in metadata.name.lower() or
                query in metadata.description.lower() or
                any(query in tag.lower() for tag in metadata.tags)):
                results.append(metadata)
        
        return sorted(results, key=lambda x: (x.name, x.version))
    
    def create_model(self, name: str, version: str = "latest", **config_kwargs) -> BaseGAIAModel:
        """Create a model instance from the registry.
        
        Args:
            name: Model name
            version: Model version or 'latest'
            **config_kwargs: Configuration parameters
            
        Returns:
            Model instance
        """
        model_class = self.get_model_class(name, version)
        config_class = self.get_config_class(name, version)
        
        # Create config instance
        config = config_class(**config_kwargs)
        
        # Create model instance
        model = model_class(config)
        
        return model
    
    def update_metadata(self, name: str, version: str, **updates) -> None:
        """Update model metadata.
        
        Args:
            name: Model name
            version: Model version
            **updates: Fields to update
        """
        model_key = f"{name}:{version}"
        
        if model_key not in self._models:
            raise ValueError(f"Model {name} v{version} not found in registry")
        
        metadata = self._models[model_key]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # Update timestamp
        metadata.updated_at = datetime.now().isoformat()
        
        # Save to disk
        self._save_registry()
        
    
    def remove_model(self, name: str, version: str) -> None:
        """Remove a model from the registry.
        
        Args:
            name: Model name
            version: Model version
        """
        model_key = f"{name}:{version}"
        
        if model_key not in self._models:
            raise ValueError(f"Model {name} v{version} not found in registry")
        
        # Remove from all registries
        del self._models[model_key]
        if model_key in self._model_classes:
            del self._model_classes[model_key]
        if model_key in self._config_classes:
            del self._config_classes[model_key]
        
        # Save to disk
        self._save_registry()
        
    
    def _get_latest_version(self, name: str) -> str:
        """Get the latest version of a model.
        
        Args:
            name: Model name
            
        Returns:
            Latest version string
        """
        versions = []
        for key in self._models.keys():
            model_name, version = key.split(":")
            if model_name == name:
                versions.append(version)
        
        if not versions:
            raise ValueError(f"No versions found for model {name}")
        
        # Simple version sorting (assumes semantic versioning)
        versions.sort(key=lambda x: [int(i) for i in x.split(".") if i.isdigit()], reverse=True)
        return versions[0]
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        if not self.registry_path.exists():
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            
            for key, metadata_dict in data.items():
                self._models[key] = ModelMetadata.from_dict(metadata_dict)
            
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {key: metadata.to_dict() for key, metadata in self._models.items()}
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"💾 Saved registry with {len(self._models)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _register_builtin_models(self) -> None:
        """Register built-in GAIA models."""
        try:
            from .gaia_language_model import GAIALanguageModel
            from ..training.config import GAIALanguageModelConfig
            
            self.register_model(
                name="gaia-language-model",
                version="1.0.0",
                model_class=GAIALanguageModel,
                config_class=GAIALanguageModelConfig,
                description="GAIA Language Model with categorical structures and hierarchical message passing",
                author="GAIA Framework",
                tags=["language-model", "transformer", "categorical", "hierarchical"],
                capabilities=[
                    "text-generation",
                    "language-modeling",
                    "categorical-reasoning",
                    "hierarchical-processing"
                ],
                requirements={
                    "torch": ">=1.9.0",
                    "transformers": ">=4.0.0"
                },
                license="MIT"
            )
            
        except ImportError as e:
            logger.warning(f"Could not register built-in models: {e}")


# Global registry instance
_global_registry: Optional[GAIAModelRegistry] = None


def get_model_registry() -> GAIAModelRegistry:
    """Get the global model registry instance.
    
    Returns:
        Global model registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = GAIAModelRegistry()
    return _global_registry


def register_model(*args, **kwargs) -> None:
    """Register a model in the global registry."""
    registry = get_model_registry()
    registry.register_model(*args, **kwargs)


def create_model(name: str, version: str = "latest", **config_kwargs) -> BaseGAIAModel:
    """Create a model from the global registry.
    
    Args:
        name: Model name
        version: Model version or 'latest'
        **config_kwargs: Configuration parameters
        
    Returns:
        Model instance
    """
    registry = get_model_registry()
    return registry.create_model(name, version, **config_kwargs)


def list_models(tag: Optional[str] = None) -> List[ModelMetadata]:
    """List all models in the global registry.
    
    Args:
        tag: Optional tag to filter by
        
    Returns:
        List of model metadata
    """
    registry = get_model_registry()
    return registry.list_models(tag)


def search_models(query: str) -> List[ModelMetadata]:
    """Search models in the global registry.
    
    Args:
        query: Search query
        
    Returns:
        List of matching model metadata
    """
    registry = get_model_registry()
    return registry.search_models(query)