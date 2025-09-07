from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
import torch
from gaia.data.transforms import YonedaTransform, SimplicalTransform, RobustNormalization
from gaia.data.categorical import CategoricalDataset, SimplicalDataLoader
from gaia.training.config import GAIAConfig

# Initialize logger for this module
logger = GAIAConfig.get_logger(__name__)


try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class LanguageModelingDataset(Dataset):
    """Legacy dataset for backward compatibility"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = tokenizer.encode(text, max_length=max_length)
            if len(tokens) > 1:  # Need at least 2 tokens for input/target
                self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        # For language modeling: input = tokens[:-1], target = tokens[1:]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Pad to max_length
        input_length = len(input_ids)
        attention_mask = [1] * input_length + [0] * (self.max_length - 1 - input_length)
        input_ids = input_ids + [0] * (self.max_length - 1 - input_length)
        target_ids = target_ids + [-100] * (self.max_length - 1 - len(target_ids))  # -100 for ignore_index
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_length-1], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:self.max_length-1], dtype=torch.long),
            'labels': torch.tensor(target_ids[:self.max_length-1], dtype=torch.long)
        }
    

    


class CategoricalLanguageDataset(CategoricalDataset):
    """Categorical language dataset where sentences are objects in a category"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, 
                 apply_yoneda: bool = True, apply_simplicial: bool = True):
        """
        Initialize categorical language dataset
        
        Args:
            texts: List of text sentences (objects in the category)
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            apply_yoneda: Whether to apply Yoneda transform
            apply_simplicial: Whether to apply simplicial transform
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_yoneda = apply_yoneda
        self.apply_simplicial = apply_simplicial
        
        # Create categorical transforms
        self.transforms = []
        if apply_yoneda:
            self.yoneda_transform = YonedaTransform(
                embedding_dim=64,
                categorical_dims=list(range(max_length)),
                preserve_structure=True
            )
            self.transforms.append(self.yoneda_transform)
        
        if apply_simplicial:
            self.simplicial_transform = SimplicalTransform(
                simplex_dim=2,
                normalize=True,
                add_boundary=True
            )
            self.transforms.append(self.simplicial_transform)
        
        # Process texts into categorical structure
        self.categorical_data = self._create_categorical_structure(texts)
        
        # Initialize parent with processed data
        super().__init__(
            data=self.categorical_data['sentences'],
            targets=self.categorical_data['targets'],
            categorical_dims=list(range(max_length)),
            task_type="classification",
            auto_preprocess=False  # We handle preprocessing ourselves
        )
    
    def _create_categorical_structure(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Create categorical structure from texts"""
        sentences = []  # Objects in the category
        targets = []    # Target objects (next sentences)
        morphisms = []  # Transformations between sentences
        
        for i, text in enumerate(texts):
            # Tokenize sentence (object representation)
            tokens = self.tokenizer.encode(text, max_length=self.max_length)
            if len(tokens) < 2:
                continue
                
            # Sentence as object in category
            sentence_obj = torch.zeros(self.max_length, dtype=torch.float32)
            sentence_obj[:len(tokens)] = torch.tensor(tokens, dtype=torch.float32)
            sentences.append(sentence_obj)
            
            # Target (next token prediction as morphism target)
            target_obj = torch.zeros(self.max_length, dtype=torch.long)
            target_obj[:len(tokens)-1] = torch.tensor(tokens[1:], dtype=torch.long)
            targets.append(target_obj)
            
            # Create morphisms (transformations)
            # Shift morphism: sentence -> shifted sentence
            shift_morphism = self._create_shift_morphism(sentence_obj)
            morphisms.append(shift_morphism)
        
        return {
            'sentences': torch.stack(sentences) if sentences else torch.empty(0, self.max_length),
            'targets': torch.stack(targets) if targets else torch.empty(0, self.max_length, dtype=torch.long),
            'morphisms': torch.stack(morphisms) if morphisms else torch.empty(0, self.max_length, self.max_length)
        }
    
    def _create_shift_morphism(self, sentence: torch.Tensor) -> torch.Tensor:
        """Create shift morphism matrix for sentence transformation"""
        # Create shift matrix (morphism in the category)
        shift_matrix = torch.zeros(self.max_length, self.max_length)
        for i in range(self.max_length - 1):
            shift_matrix[i, i + 1] = 1.0  # Shift right by one position
        return shift_matrix
    
    def apply_categorical_functors(self, data: torch.Tensor) -> torch.Tensor:
        """Apply categorical functors (Yoneda, simplicial) to data"""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def get_morphism(self, idx: int) -> torch.Tensor:
        """Get morphism for sentence at index"""
        return self.categorical_data['morphisms'][idx]
    
    def __getitem__(self, idx):
        """Get categorical item with morphism information"""
        sentence = self.data[idx]  # Object in category
        target = self.targets[idx] if self.targets is not None else None
        morphism = self.get_morphism(idx)  # Morphism in category
        
        # Apply categorical functors
        sentence_transformed = self.apply_categorical_functors(sentence.unsqueeze(0)).squeeze(0)
        
        return {
            'sentence': sentence,  # Original object
            'sentence_transformed': sentence_transformed,  # Functor-transformed object
            'target': target,  # Target object
            'morphism': morphism,  # Morphism between objects
            'input_ids': sentence[:self.max_length-1].long(),
            'attention_mask': (sentence[:self.max_length-1] != 0).long(),
            'labels': target[:self.max_length-1] if target is not None else None
        }

def Dataset() -> List[str]:
    """Create language modeling dataset from TinyStories"""
    logger.info("Creating language modeling dataset...")
    
    if DATASETS_AVAILABLE:
        try:
            # Load TinyStories dataset
            logger.info("Loading TinyStories dataset from HuggingFace...")
            dataset = load_dataset('roneneldan/TinyStories')
            
            # Extract first 12000 stories from the training set
            texts = dataset['train']['text'][:12000]
            logger.info(f"Loaded {len(texts)} stories from TinyStories dataset")
            
            # Filter out empty or very short texts
            filtered_texts = [text.strip() for text in texts if text.strip() and len(text.strip()) > 10]
            logger.info(f"Filtered to {len(filtered_texts)} valid stories")
            
            return filtered_texts
            
        except Exception as e:
            logger.error(f"Failed to load TinyStories dataset: {e}")
            logger.info("Falling back to placeholder data")
    else:
        logger.warning("datasets library not available, using placeholder data")
    
    # Fallback to original placeholder data
    logger.info("Using placeholder dataset for demonstration")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing involves understanding human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Language models can generate coherent and contextual text.",
        "The GAIA framework integrates categorical theory with deep learning.",
        "Fuzzy sets provide a mathematical foundation for uncertainty.",
        "Coalgebras model dynamic systems and state transitions."
    ] * 100  # Repeat for more training data
    
    return texts
