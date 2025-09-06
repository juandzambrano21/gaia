"""Data Utilities for GAIA Framework"""

import torch
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_california_housing, fetch_20newsgroups, fetch_openml
)
import requests
import zipfile
import tarfile

def load_any_dataset(
    source: Union[str, Path, Dict, pd.DataFrame, np.ndarray],
    target_column: Optional[str] = None,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, Any, Any, Any]:
    """
    Universal dataset loader that can handle any data source
    
    Args:
        source: Data source (file path, URL, sklearn dataset name, etc.)
        target_column: Name of target column (for DataFrames)
        task_type: Type of task ("classification", "regression", "auto")
        test_size: Proportion of test set
        random_state: Random seed
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load data based on source type
    if isinstance(source, str):
        if source.startswith(('http://', 'https://')):
            X, y = _load_from_url(source, **kwargs)
        elif source in SKLEARN_DATASETS:
            X, y = _load_sklearn_dataset(source, **kwargs)
        elif Path(source).exists():
            X, y = _load_from_file(source, target_column, **kwargs)
        else:
            # Try OpenML
            X, y = _load_from_openml(source, **kwargs)
    elif isinstance(source, pd.DataFrame):
        X, y = _load_from_dataframe(source, target_column)
    elif isinstance(source, np.ndarray):
        X = source
        y = kwargs.get('targets')
    elif isinstance(source, dict):
        X = source.get('data', source.get('X'))
        y = source.get('targets', source.get('y'))
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")
    
    # Auto-detect task type
    if task_type == "auto":
        task_type = _detect_task_type(y)
    
    # Split data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if task_type == "classification" else None
        )
    else:
        # Unsupervised case
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        y_train = y_test = None
    
    return X_train, X_test, y_train, y_test

# Sklearn dataset registry
SKLEARN_DATASETS = {
    'iris': load_iris,
    'wine': load_wine,
    'breast_cancer': load_breast_cancer,
    'digits': load_digits,
    'california_housing': fetch_california_housing,
    '20newsgroups': fetch_20newsgroups
}

def _load_sklearn_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Load sklearn dataset"""
    loader = SKLEARN_DATASETS[name]
    if name == '20newsgroups':
        # Special handling for text data
        data = loader(subset='all', **kwargs)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data.data).toarray()
        y = data.target
    else:
        data = loader(**kwargs)
        X = data.data
        y = data.target
    return X, y

def _load_from_file(
    filepath: Union[str, Path], 
    target_column: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Any]:
    """Load data from file"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath, **kwargs)
        return _load_from_dataframe(df, target_column)
    elif filepath.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, **kwargs)
        return _load_from_dataframe(df, target_column)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        X = data.get('data', data.get('X'))
        y = data.get('targets', data.get('y'))
        return X, y
    elif filepath.suffix in ['.pkl', '.pickle']:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            return data[:2]
        else:
            return data, None
    elif filepath.suffix == '.npy':
        X = np.load(filepath)
        return X, None
    elif filepath.suffix == '.npz':
        data = np.load(filepath)
        X = data.get('X', data.get('data'))
        y = data.get('y', data.get('targets'))
        return X, y
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def _load_from_dataframe(
    df: pd.DataFrame, 
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load data from DataFrame"""
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    else:
        # Try common target column names
        common_targets = ['target', 'label', 'y', 'class', 'output']
        target_col = None
        for col in common_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            X = df.drop(target_col, axis=1)
            y = df[target_col]
        else:
            X = df
            y = None
    
    return X, y

def _load_from_url(url: str, **kwargs) -> Tuple[Any, Any]:
    """Load data from URL"""
    # Download file
    response = requests.get(url)
    response.raise_for_status()
    
    # Determine file type from URL
    if url.endswith('.csv'):
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), **kwargs)
        return _load_from_dataframe(df, kwargs.get('target_column'))
    elif url.endswith('.json'):
        data = response.json()
        X = data.get('data', data.get('X'))
        y = data.get('targets', data.get('y'))
        return X, y
    elif url.endswith(('.zip', '.tar.gz')):
        # Handle compressed files
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.content)
            tmp.flush()
            
            if url.endswith('.zip'):
                with zipfile.ZipFile(tmp.name) as zf:
                    # Extract first data file
                    for filename in zf.namelist():
                        if filename.endswith(('.csv', '.json')):
                            with zf.open(filename) as f:
                                if filename.endswith('.csv'):
                                    df = pd.read_csv(f)
                                    return _load_from_dataframe(df, kwargs.get('target_column'))
                                else:
                                    data = json.load(f)
                                    return data.get('X'), data.get('y')
    
    raise ValueError(f"Cannot determine how to load data from URL: {url}")

def _load_from_openml(dataset_id: Union[str, int], **kwargs) -> Tuple[Any, Any]:
    """Load dataset from OpenML"""
    try:
        dataset = fetch_openml(data_id=dataset_id, as_frame=True, **kwargs)
        return dataset.data, dataset.target
    except Exception as e:
        raise ValueError(f"Failed to load OpenML dataset {dataset_id}: {e}")

def _detect_task_type(y: Any) -> str:
    """Auto-detect task type from targets"""
    if y is None:
        return "unsupervised"
    
    if isinstance(y, (pd.Series, np.ndarray)):
        unique_values = len(np.unique(y))
        if unique_values <= 20 and np.issubdtype(y.dtype, np.integer):
            return "classification"
        elif np.issubdtype(y.dtype, np.floating):
            return "regression"
        else:
            return "classification"
    
    return "classification"

def validate_dataset(
    X: Any, 
    y: Any = None,
    min_samples: int = 10,
    max_features: int = 10000,
    check_categorical: bool = True
) -> Dict[str, Any]:
    """
    Validate dataset for GAIA framework
    
    Args:
        X: Input features
        y: Target values
        min_samples: Minimum number of samples required
        max_features: Maximum number of features allowed
        check_categorical: Whether to check for categorical structure
    
    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'info': {}
    }
    
    # Convert to numpy for analysis
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    elif isinstance(X, torch.Tensor):
        X_array = X.cpu().numpy()
    else:
        X_array = np.array(X)
    
    # Basic shape validation
    if len(X_array.shape) < 2:
        report['errors'].append("Data must be at least 2-dimensional")
        report['valid'] = False
    
    n_samples, n_features = X_array.shape[:2]
    report['info']['n_samples'] = n_samples
    report['info']['n_features'] = n_features
    
    # Sample count validation
    if n_samples < min_samples:
        report['errors'].append(f"Too few samples: {n_samples} < {min_samples}")
        report['valid'] = False
    
    # Feature count validation
    if n_features > max_features:
        report['warnings'].append(f"Many features: {n_features} > {max_features}")
    
    # Missing values check
    missing_count = np.isnan(X_array).sum()
    if missing_count > 0:
        missing_pct = missing_count / X_array.size * 100
        report['warnings'].append(f"Missing values: {missing_pct:.1f}%")
        report['info']['missing_values'] = missing_count
    
    # Infinite values check
    inf_count = np.isinf(X_array).sum()
    if inf_count > 0:
        report['errors'].append(f"Infinite values found: {inf_count}")
        report['valid'] = False
    
    # Target validation
    if y is not None:
        if isinstance(y, pd.Series):
            y_array = y.values
        elif isinstance(y, torch.Tensor):
            y_array = y.cpu().numpy()
        else:
            y_array = np.array(y)
        
        if len(y_array) != n_samples:
            report['errors'].append("Target length doesn't match sample count")
            report['valid'] = False
        
        # Target type analysis
        unique_targets = len(np.unique(y_array))
        report['info']['n_classes'] = unique_targets
        
        if unique_targets == 1:
            report['warnings'].append("Only one unique target value")
        elif unique_targets > n_samples * 0.8:
            report['info']['task_type'] = "regression"
        else:
            report['info']['task_type'] = "classification"
    
    # Categorical structure check
    if check_categorical:
        categorical_info = _analyze_categorical_structure(X_array)
        report['info']['categorical'] = categorical_info
    
    return report

def _analyze_categorical_structure(X: np.ndarray) -> Dict[str, Any]:
    """Analyze categorical structure of data"""
    info = {
        'potential_categorical_dims': [],
        'sparsity': np.mean(X == 0),
        'feature_correlations': [],
        'rank': np.linalg.matrix_rank(X) if X.shape[0] >= X.shape[1] else None
    }
    
    # Find potential categorical dimensions
    for i in range(X.shape[1]):
        unique_vals = len(np.unique(X[:, i]))
        if unique_vals <= 20:  # Heuristic for categorical
            info['potential_categorical_dims'].append(i)
    
    # Compute feature correlations (sample if too many features)
    if X.shape[1] <= 100:
        corr_matrix = np.corrcoef(X.T)
        high_corr_pairs = np.where(np.abs(corr_matrix) > 0.8)
        info['feature_correlations'] = list(zip(high_corr_pairs[0], high_corr_pairs[1]))
    
    return info