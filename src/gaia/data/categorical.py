"""Categorical Dataset Handling for GAIA Framework"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import json

class CategoricalDataset(Dataset):
    """Universal dataset class that handles any data type with categorical structure"""
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, pd.DataFrame, str, Path],
        targets: Optional[Union[torch.Tensor, np.ndarray, pd.Series, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        categorical_dims: Optional[List[int]] = None,
        device: str = "cpu",
        auto_preprocess: bool = True,
        task_type: str = "classification"  # "classification", "regression", "multiclass", "multilabel"
    ):
        """
        Initialize categorical dataset
        
        Args:
            data: Input data (tensor, array, DataFrame, or file path)
            targets: Target labels/values
            transform: Transform function for data
            target_transform: Transform function for targets
            categorical_dims: Dimensions that represent categorical structure
            device: Device for tensors
            auto_preprocess: Whether to automatically preprocess data
            task_type: Type of learning task
        """
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.categorical_dims = categorical_dims or []
        self.task_type = task_type
        self.preprocessors = {}
        
        # Load and preprocess data
        self.data, self.targets = self._load_data(data, targets)
        
        if auto_preprocess:
            self.data, self.targets = self._preprocess_data(self.data, self.targets)
            
        # Convert to tensors
        self.data = self._to_tensor(self.data)
        if self.targets is not None:
            self.targets = self._to_tensor(self.targets, is_target=True)
            
        # Validate categorical structure
        self._validate_categorical_structure()
    
    def _load_data(self, data: Any, targets: Any) -> Tuple[Any, Any]:
        """Load data from various sources"""
        # Handle file paths
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
                if targets is None and 'target' in df.columns:
                    targets = df['target']
                    data = df.drop('target', axis=1)
                elif targets is None and 'label' in df.columns:
                    targets = df['label']
                    data = df.drop('label', axis=1)
                else:
                    data = df
            elif data_path.suffix == '.json':
                with open(data_path, 'r') as f:
                    json_data = json.load(f)
                data = json_data.get('data', json_data)
                targets = json_data.get('targets', targets)
            elif data_path.suffix in ['.pkl', '.pickle']:
                with open(data_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple):
                    data, targets = loaded
                else:
                    data = loaded
            elif data_path.suffix == '.npy':
                data = np.load(data_path)
            elif data_path.suffix == '.npz':
                loaded = np.load(data_path)
                data = loaded['data'] if 'data' in loaded else loaded['X']
                targets = loaded.get('targets', loaded.get('y', targets))
        
        # Handle string targets (file paths)
        if isinstance(targets, (str, Path)):
            target_path = Path(targets)
            if target_path.suffix == '.npy':
                targets = np.load(target_path)
            elif target_path.suffix == '.csv':
                targets = pd.read_csv(target_path).iloc[:, 0]
        
        return data, targets
    
    def _preprocess_data(self, data: Any, targets: Any) -> Tuple[Any, Any]:
        """Preprocess data for categorical learning"""
        # Convert DataFrame to numpy
        if isinstance(data, pd.DataFrame):
            # Handle categorical columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                # One-hot encode categorical features
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                categorical_encoded = encoder.fit_transform(data[categorical_cols])
                self.preprocessors['categorical_encoder'] = encoder
                
                # Combine with numerical features
                numerical_cols = data.select_dtypes(exclude=['object', 'category']).columns
                if len(numerical_cols) > 0:
                    numerical_data = data[numerical_cols].values
                    data = np.hstack([numerical_data, categorical_encoded])
                else:
                    data = categorical_encoded
            else:
                data = data.values
        
        # Handle targets
        if isinstance(targets, pd.Series):
            targets = targets.values
        
        # Encode string targets
        if targets is not None and targets.dtype == object:
            label_encoder = LabelEncoder()
            targets = label_encoder.fit_transform(targets)
            self.preprocessors['label_encoder'] = label_encoder
        
        # Normalize numerical features
        if isinstance(data, np.ndarray) and data.dtype in [np.float32, np.float64]:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            self.preprocessors['scaler'] = scaler
        
        return data, targets
    
    def _to_tensor(self, data: Any, is_target: bool = False) -> torch.Tensor:
        """Convert data to tensor"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            if is_target:
                if self.task_type == "regression":
                    return torch.FloatTensor(data).to(self.device)
                else:
                    return torch.LongTensor(data).to(self.device)
            else:
                return torch.FloatTensor(data).to(self.device)
        elif isinstance(data, (list, tuple)):
            return torch.FloatTensor(data).to(self.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _validate_categorical_structure(self):
        """Validate that data has proper categorical structure for GAIA"""
        if len(self.data.shape) < 2:
            raise ValueError("Data must be at least 2-dimensional for categorical structure")
        
        # Ensure categorical dimensions are valid
        for dim in self.categorical_dims:
            if dim >= self.data.shape[1]:
                raise ValueError(f"Categorical dimension {dim} exceeds data dimensions")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.targets is not None:
            target = self.targets[idx]
            if self.target_transform:
                target = self.target_transform(target)
            return sample, target
        else:
            return sample
    
    def get_categorical_structure(self) -> Dict[str, Any]:
        """Get information about categorical structure"""
        return {
            'shape': self.data.shape,
            'categorical_dims': self.categorical_dims,
            'task_type': self.task_type,
            'num_classes': len(torch.unique(self.targets)) if self.targets is not None else None,
            'preprocessors': list(self.preprocessors.keys())
        }

class SimplicalDataLoader(DataLoader):
    """DataLoader with simplicial structure awareness"""
    
    def __init__(
        self,
        dataset: CategoricalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        simplicial_batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize simplicial data loader
        
        Args:
            dataset: CategoricalDataset instance
            batch_size: Standard batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            simplicial_batch_size: Special batch size for simplicial operations
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
        self.simplicial_batch_size = simplicial_batch_size or batch_size
        self.categorical_dims = dataset.categorical_dims
    
    def get_simplicial_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch specifically sized for simplicial operations"""
        # Create a temporary loader with simplicial batch size
        temp_loader = DataLoader(
            self.dataset,
            batch_size=self.simplicial_batch_size,
            shuffle=True,
            num_workers=0
        )
        return next(iter(temp_loader))