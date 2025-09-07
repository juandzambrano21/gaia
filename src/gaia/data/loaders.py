"""Extensible data loading architecture for GAIA framework.

This module provides a flexible factory pattern for creating datasets and dataloaders
with various categorical, simplicial, and Yoneda transformations.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

from .categorical import CategoricalDataset, SimplicalDataLoader
from .dataset import LanguageModelingDataset, CategoricalLanguageDataset
from ..training.config import GAIAConfig

logger = GAIAConfig.get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    dataset_type: str = "categorical_language"  # categorical_language, language_modeling, custom
    max_length: int = 512
    apply_yoneda: bool = True
    apply_simplicial: bool = True
    custom_transforms: List[Callable] = field(default_factory=list)
    

@dataclass
class DataLoaderConfig:
    """Configuration for dataloader creation."""
    loader_type: str = "simplicial"  # simplicial, standard, custom
    batch_size: int = 4
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    simplicial_batch_size: Optional[int] = None
    custom_collate_fn: Optional[Callable] = None


class DatasetFactory(ABC):
    """Abstract factory for creating datasets."""
    
    @abstractmethod
    def create_dataset(self, 
                      data: Any, 
                      tokenizer: Any, 
                      config: DatasetConfig) -> torch.utils.data.Dataset:
        """Create a dataset instance."""
        pass


class LanguageDatasetFactory(DatasetFactory):
    """Factory for language modeling datasets."""
    
    def create_dataset(self, 
                      data: List[str], 
                      tokenizer: Any, 
                      config: DatasetConfig) -> torch.utils.data.Dataset:
        """Create language modeling dataset."""
        if config.dataset_type == "categorical_language":
            return CategoricalLanguageDataset(
                data, 
                tokenizer,
                max_length=config.max_length,
                apply_yoneda=config.apply_yoneda,
                apply_simplicial=config.apply_simplicial
            )
        elif config.dataset_type == "language_modeling":
            return LanguageModelingDataset(
                data,
                tokenizer,
                max_length=config.max_length
            )
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset_type}")


class DataLoaderFactory(ABC):
    """Abstract factory for creating dataloaders."""
    
    @abstractmethod
    def create_dataloader(self, 
                         dataset: torch.utils.data.Dataset,
                         config: DataLoaderConfig) -> DataLoader:
        """Create a dataloader instance."""
        pass


class GAIADataLoaderFactory(DataLoaderFactory):
    """Factory for GAIA-specific dataloaders."""
    
    def create_dataloader(self, 
                         dataset: torch.utils.data.Dataset,
                         config: DataLoaderConfig) -> DataLoader:
        """Create GAIA dataloader."""
        if config.loader_type == "simplicial":
            return SimplicalDataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                simplicial_batch_size=config.simplicial_batch_size or config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
        elif config.loader_type == "standard":
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                collate_fn=config.custom_collate_fn
            )
        else:
            raise ValueError(f"Unknown loader type: {config.loader_type}")


class GAIADataManager:
    """High-level data management interface for GAIA framework."""
    
    def __init__(self, 
                 dataset_factory: Optional[DatasetFactory] = None,
                 dataloader_factory: Optional[DataLoaderFactory] = None):
        self.dataset_factory = dataset_factory or LanguageDatasetFactory()
        self.dataloader_factory = dataloader_factory or GAIADataLoaderFactory()
        
    def create_train_val_loaders(self,
                                 train_data: List[str],
                                 val_data: List[str],
                                 tokenizer: Any,
                                 dataset_config: Optional[DatasetConfig] = None,
                                 train_loader_config: Optional[DataLoaderConfig] = None,
                                 val_loader_config: Optional[DataLoaderConfig] = None) -> tuple:
        """Create training and validation dataloaders.
        
        Args:
            train_data: Training text data
            val_data: Validation text data
            tokenizer: Tokenizer instance
            dataset_config: Dataset configuration
            train_loader_config: Training dataloader configuration
            val_loader_config: Validation dataloader configuration
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Use default configs if not provided
        dataset_config = dataset_config or DatasetConfig()
        train_loader_config = train_loader_config or DataLoaderConfig(shuffle=True)
        val_loader_config = val_loader_config or DataLoaderConfig(shuffle=False)
        
        logger.info(f"Creating datasets with config: {dataset_config}")
        
        # Create datasets
        train_dataset = self.dataset_factory.create_dataset(
            train_data, tokenizer, dataset_config
        )
        val_dataset = self.dataset_factory.create_dataset(
            val_data, tokenizer, dataset_config
        )
        
        logger.info(f"Created train dataset: {len(train_dataset)} samples")
        logger.info(f"Created val dataset: {len(val_dataset)} samples")
        
        # Create dataloaders
        train_loader = self.dataloader_factory.create_dataloader(
            train_dataset, train_loader_config
        )
        val_loader = self.dataloader_factory.create_dataloader(
            val_dataset, val_loader_config
        )
        
        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def create_single_loader(self,
                           data: List[str],
                           tokenizer: Any,
                           dataset_config: Optional[DatasetConfig] = None,
                           loader_config: Optional[DataLoaderConfig] = None) -> DataLoader:
        """Create a single dataloader.
        
        Args:
            data: Text data
            tokenizer: Tokenizer instance
            dataset_config: Dataset configuration
            loader_config: Dataloader configuration
            
        Returns:
            DataLoader instance
        """
        dataset_config = dataset_config or DatasetConfig()
        loader_config = loader_config or DataLoaderConfig()
        
        dataset = self.dataset_factory.create_dataset(
            data, tokenizer, dataset_config
        )
        
        return self.dataloader_factory.create_dataloader(
            dataset, loader_config
        )


# Convenience functions for common use cases
def DataLoaders(train_texts: List[str],
                val_texts: List[str],
                tokenizer: Any,
                batch_size: int = 4,
                max_seq_length: int = 512,
                apply_yoneda: bool = True,
                apply_simplicial: bool = True) -> tuple:
    """Create categorical language dataloaders with clean naming.
    
    Args:
        train_texts: Training text data
        val_texts: Validation text data
        tokenizer: Tokenizer instance
        batch_size: Batch size for dataloaders
        max_seq_length: Maximum sequence length
        apply_yoneda: Whether to apply Yoneda transformations
        apply_simplicial: Whether to apply simplicial transformations
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_manager = GAIADataManager()
    
    dataset_config = DatasetConfig(
        dataset_type="categorical_language",
        max_length=max_seq_length,
        apply_yoneda=apply_yoneda,
        apply_simplicial=apply_simplicial
    )
    
    train_loader_config = DataLoaderConfig(
        loader_type="simplicial",
        batch_size=batch_size,
        shuffle=True,
        simplicial_batch_size=batch_size
    )
    
    val_loader_config = DataLoaderConfig(
        loader_type="simplicial",
        batch_size=batch_size,
        shuffle=False,
        simplicial_batch_size=batch_size
    )
    
    return data_manager.create_train_val_loaders(
        train_texts, val_texts, tokenizer,
        dataset_config, train_loader_config, val_loader_config
    )


def create_standard_language_loaders(train_texts: List[str],
                                    val_texts: List[str],
                                    tokenizer: Any,
                                    batch_size: int = 4,
                                    max_seq_length: int = 512) -> tuple:
    """Create standard language modeling dataloaders without categorical features.
    
    Args:
        train_texts: Training text data
        val_texts: Validation text data
        tokenizer: Tokenizer instance
        batch_size: Batch size for dataloaders
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_manager = GAIADataManager()
    
    dataset_config = DatasetConfig(
        dataset_type="language_modeling",
        max_length=max_seq_length,
        apply_yoneda=False,
        apply_simplicial=False
    )
    
    train_loader_config = DataLoaderConfig(
        loader_type="standard",
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader_config = DataLoaderConfig(
        loader_type="standard",
        batch_size=batch_size,
        shuffle=False
    )
    
    return data_manager.create_train_val_loaders(
        train_texts, val_texts, tokenizer,
        dataset_config, train_loader_config, val_loader_config
    )