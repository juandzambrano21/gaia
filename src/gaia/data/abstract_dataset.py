"""Abstract dataset interface for GAIA models"""

from abc import ABC, abstractmethod
from typing import List, Union, Iterator, Optional, Any, Dict
from torch.utils.data import Dataset
import torch


class GAIADatasetInterface(ABC):
    """Abstract interface for GAIA datasets"""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset"""
        pass
    
    @abstractmethod
    def get_texts(self) -> List[str]:
        """Return all texts in the dataset"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return vocabulary size for tokenizer building"""
        pass


class TextListDataset(GAIADatasetInterface, Dataset):
    """Dataset wrapper for list of texts"""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        self._vocab_size = None
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'text': self.texts[idx],
            'index': idx
        }
    
    def get_texts(self) -> List[str]:
        return self.texts
    
    def get_vocab_size(self) -> int:
        if self._vocab_size is None:
            # Estimate vocab size from unique characters/words
            all_text = ' '.join(self.texts)
            unique_chars = len(set(all_text))
            # Add buffer for special tokens
            self._vocab_size = max(1000, unique_chars * 2)
        return self._vocab_size


class FileDataset(GAIADatasetInterface, Dataset):
    """Dataset that loads texts from a file"""
    
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        self.file_path = file_path
        self.encoding = encoding
        self._texts = None
        self._load_texts()
    
    def _load_texts(self):
        """Load texts from file"""
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            self._texts = [line.strip() for line in f if line.strip()]
    
    def __len__(self) -> int:
        return len(self._texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'text': self._texts[idx],
            'index': idx,
            'source_file': self.file_path
        }
    
    def get_texts(self) -> List[str]:
        return self._texts
    
    def get_vocab_size(self) -> int:
        all_text = ' '.join(self._texts)
        unique_chars = len(set(all_text))
        return max(1000, unique_chars * 2)


class DatasetFactory:
    """Factory for creating GAIA datasets from various sources"""
    
    @staticmethod
    def from_texts(texts: List[str]) -> TextListDataset:
        """Create dataset from list of texts"""
        return TextListDataset(texts)
    
    @staticmethod
    def from_file(file_path: str, encoding: str = 'utf-8') -> FileDataset:
        """Create dataset from text file"""
        return FileDataset(file_path, encoding)
    
    @staticmethod
    def from_directory(directory_path: str, pattern: str = "*.txt") -> GAIADatasetInterface:
        """Create dataset from directory of text files."""
        import glob
        import os
        
        file_paths = glob.glob(os.path.join(directory_path, pattern))
        all_texts = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_texts.extend([line.strip() for line in f if line.strip()])
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        return TextListDataset(all_texts)
    
    @staticmethod
    def from_dataset(dataset: Union[Dataset, GAIADatasetInterface]) -> GAIADatasetInterface:
        """Wrap existing dataset to conform to GAIA interface"""
        if isinstance(dataset, GAIADatasetInterface):
            return dataset
        
        # Wrap generic PyTorch dataset
        class DatasetWrapper(GAIADatasetInterface):
            def __init__(self, wrapped_dataset):
                self.dataset = wrapped_dataset
                self._texts = None
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                item = self.dataset[idx]
                if isinstance(item, str):
                    return {'text': item, 'index': idx}
                elif isinstance(item, dict) and 'text' in item:
                    return item
                else:
                    # Try to extract text from various formats
                    text = str(item) if not isinstance(item, (list, tuple)) else str(item[0])
                    return {'text': text, 'index': idx}
            
            def get_texts(self):
                if self._texts is None:
                    self._texts = [self[i]['text'] for i in range(len(self))]
                return self._texts
            
            def get_vocab_size(self):
                texts = self.get_texts()
                all_text = ' '.join(texts)
                unique_chars = len(set(all_text))
                return max(1000, unique_chars * 2)
        
        return DatasetWrapper(dataset)
    
    @staticmethod
    def create_sample_dataset(size: int = 100) -> GAIADatasetInterface:
        """Create a sample dataset for testing and demonstration."""
        import random
        
        # Sample sentences for training
        base_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "GAIA represents a new paradigm in AI architecture.",
            "Coalgebras provide mathematical foundations for computation.",
            "Neural networks learn complex patterns from data.",
            "Artificial intelligence continues to evolve rapidly.",
            "Deep learning models require substantial computational resources.",
            "Natural language processing enables human-computer interaction.",
            "Transformers revolutionized sequence modeling tasks.",
            "Attention mechanisms focus on relevant information."
        ]
        
        # Generate variations and combinations
        sample_texts = []
        for _ in range(size):
            # Pick random sentences and sometimes combine them
            if random.random() < 0.3:  # 30% chance of combining sentences
                selected = random.sample(base_sentences, 2)
                sample_texts.append(" ".join(selected))
            else:
                sample_texts.append(random.choice(base_sentences))
        
        return TextListDataset(sample_texts)
    



def create_gaia_dataset(source: Union[List[str], str, Dataset, GAIADatasetInterface], 
                       **kwargs) -> GAIADatasetInterface:
    """Unified function to create GAIA dataset from various sources"""
    
    if isinstance(source, list):
        return DatasetFactory.from_texts(source)
    elif isinstance(source, str):
        return DatasetFactory.from_file(source, **kwargs)
    elif isinstance(source, (Dataset, GAIADatasetInterface)):
        return DatasetFactory.from_dataset(source)
    else:
        raise ValueError(f"Unsupported dataset source type: {type(source)}")