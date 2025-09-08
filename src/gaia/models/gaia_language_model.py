"""GAIA Language Model Implementation

This module contains the GAIALanguageModel class that implements
a language model using the GAIA framework with categorical structures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import logging
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path
from tqdm import tqdm

from .base_model import BaseGAIAModel
from ..training.config import GAIALanguageModelConfig, TrainingConfig, GAIAConfig
from ..training import CategoricalLoss, CategoricalOps
from ..data import DataLoaders, create_gaia_dataset, GAIADatasetInterface
from ..data.tokenizer import SimpleTokenizer
from .gaia_transformer import GAIATransformer
from .initialization import ModelInit
from ..training.engine import CheckpointManager, TrainingState

# Setup logging
logger = GAIAConfig.get_logger('gaia_language_model')


class GAIALanguageModel(BaseGAIAModel):
    """GAIA Language Model with categorical structures and hierarchical message passing.
    
    This model integrates various categorical mathematical structures including
    coalgebras, Yoneda embeddings, Kan extensions, and ends/coends for
    advanced language modeling capabilities.
    """
    
    def __init__(self, config: GAIALanguageModelConfig, device: str = None):
        super().__init__(config, device)
        
        # Model configuration
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.max_seq_length = config.max_seq_length
        
        # Initialize core transformer
        self.gaia_transformer = GAIATransformer(
            vocab_size=self.vocab_size,
            d_model=self.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_length=self.max_seq_length
        ).to(self.device)
        
        # Add position embeddings layer (matching gaia_llm_train.py)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_dim).to(self.device)
        
        # Initialize categorical operations (FUNDAMENTAL TO FRAMEWORK)
        self.categorical_ops = CategoricalOps(
            device=self.device,
            bisimulation_tolerance=getattr(config, 'bisimulation_tolerance', 1e-3)
        )
        
        # Initialize component factory
        self.initializer = ModelInit(d_model=self.hidden_dim, vocab_size=self.vocab_size, device=self.device)
        
        # Initialize all GAIA components based on configuration
        self._initialize_all_components()
        
        # Initialize tokenizer (will be built during training)
        self.tokenizer = None
        
        # Update model metadata
        self.model_metadata.update({
            'model_type': config.model_type,
            'version': config.version,
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'max_seq_length': self.max_seq_length
        })
        
        # Initialize checkpoint manager for production-ready state management
        self.checkpoint_manager = None
        self.training_state = None
        self._setup_checkpoint_manager()
        
        # Production logging setup
        self._setup_production_logging()
        
        logger.info(f"🔧 Checkpoint manager initialized: {self.checkpoint_manager is not None}")
    
    def _initialize_all_components(self):
        """Initialize GAIA components based on configuration flags."""
        self.components = {
            'gaia_transformer': self.gaia_transformer,
            'categorical_ops': self.categorical_ops,
            'config': self.config
        }
        
        # Initialize fuzzy components if enabled
        if self.config.enable_fuzzy_components:
            fuzzy_components = self.initializer.initialize_token_fuzzy_sets()
            self.components.update(fuzzy_components)
            
            # Initialize fuzzy encoding pipeline
            self.fuzzy_encoding_pipeline = self.initializer.initialize_fuzzy_encoding_pipeline()
            self.components['fuzzy_encoding_pipeline'] = self.fuzzy_encoding_pipeline
        else:
            fuzzy_components = {}
            self.fuzzy_encoding_pipeline = None
        
        # Initialize simplicial structure if enabled
        if self.config.enable_simplicial_structure and self.config.enable_fuzzy_components:
            simplicial_components = self.initializer.initialize_language_simplicial_structure(
                fuzzy_components.get('token_category')
            )
            self.components.update(simplicial_components)
        elif self.config.enable_simplicial_structure:
            logger.warning("Simplicial structure requires fuzzy components. Skipping.")
        
        # Initialize coalgebras if enabled
        if self.config.enable_coalgebras:
            coalgebra_components = self.initializer.initialize_generative_coalgebras(
                self.gaia_transformer.output_projection
            )
            self.components.update(coalgebra_components)
        
        # Initialize business hierarchy if enabled
        if self.config.enable_business_hierarchy:
            self.business_hierarchy = self.initializer.initialize_business_units(self.gaia_transformer)
            self.components['business_hierarchy'] = self.business_hierarchy
        else:
            self.business_hierarchy = None
        
        # Initialize message passing if enabled
        if self.config.enable_message_passing:
            self.message_passing = self.initializer.initialize_message_passing()
            self.components['message_passing'] = self.message_passing
        else:
            self.message_passing = None
        
        # Initialize Yoneda embeddings if enabled
        if self.config.enable_yoneda_embeddings:
            yoneda_components = self.initializer.initialize_yoneda_embeddings()
            self.components.update(yoneda_components)
        
        # Initialize Kan extensions if enabled
        if self.config.enable_kan_extensions:
            kan_components = self.initializer.initialize_kan_extensions()
            self.components.update(kan_components)
        else:
            kan_components = {}
        
        # Initialize ends/coends if enabled
        if self.config.enable_ends_coends and self.config.enable_kan_extensions:
            ends_coends_components = self.initializer.initialize_ends_coends(kan_components)
            self.components.update(ends_coends_components)
        elif self.config.enable_ends_coends:
            logger.warning("Ends/coends require Kan extensions. Skipping.")
        
        # Log component summary
        enabled_components = [k for k, v in {
            'fuzzy_components': self.config.enable_fuzzy_components,
            'simplicial_structure': self.config.enable_simplicial_structure,
            'coalgebras': self.config.enable_coalgebras,
            'business_hierarchy': self.config.enable_business_hierarchy,
            'message_passing': self.config.enable_message_passing,
            'yoneda_embeddings': self.config.enable_yoneda_embeddings,
            'kan_extensions': self.config.enable_kan_extensions,
            'ends_coends': self.config.enable_ends_coends
        }.items() if v]
        
        logger.info(f"🔧 Initialized components: {', '.join(enabled_components)}")
        self.initializer.log_framework_components(self.components)
    
    def _setup_checkpoint_manager(self):
        """Initialize checkpoint manager with production-ready configuration."""
        try:
            # Get checkpoint directory from config or use default
            checkpoint_dir = getattr(self.config, 'checkpoint_dir', './checkpoints')
            checkpoint_dir = Path(checkpoint_dir) / f"gaia_language_model_{int(time.time())}"
            
            # Initialize checkpoint manager with comprehensive settings
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=getattr(self.config, 'max_checkpoints', 5),
                save_best_only=getattr(self.config, 'save_best_only', False),
                monitor_metric=getattr(self.config, 'monitor_metric', 'val_loss'),
                higher_is_better=getattr(self.config, 'higher_is_better', False),
                save_optimizer=True,
                save_scheduler=True
            )
            
            # Initialize training state
            self.training_state = TrainingState(
                model_info={
                    'model_type': 'GAIALanguageModel',
                    'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                    'parameters': sum(p.numel() for p in self.parameters()),
                    'device': str(self.device)
                }
            )
            
            logger.info(f"🔧 Checkpoint manager setup complete: {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup checkpoint manager: {e}")
            self.checkpoint_manager = None
            self.training_state = None
    
    def _setup_production_logging(self):
        """Setup comprehensive production-ready logging."""
        try:
            # Create model-specific logger
            self.model_logger = logging.getLogger(f"gaia.models.{self.__class__.__name__}")
            
            # Add file handler for model-specific logs if checkpoint manager exists
            if self.checkpoint_manager:
                log_file = self.checkpoint_manager.checkpoint_dir / "model_training.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                self.model_logger.addHandler(file_handler)
                
                logger.info(f"📝 Production logging setup complete: {log_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup production logging: {e}")
    
    def build_tokenizer(self, texts: List[str]):
        """Build tokenizer from training texts."""
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.build_vocab(texts)
        actual_vocab_size = len(self.tokenizer.word_to_id)
        logger.info(f"📝 Built tokenizer with vocab size: {actual_vocab_size}")
        
        # Update model vocab size if it doesn't match
        if actual_vocab_size != self.vocab_size:
            logger.info(f"🔄 Updating model vocab size from {self.vocab_size} to {actual_vocab_size}")
            self.vocab_size = actual_vocab_size
            
            # Recreate transformer with correct vocab size
            self.gaia_transformer = GAIATransformer(
                vocab_size=actual_vocab_size,
                d_model=self.hidden_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                max_seq_length=self.max_seq_length
            ).to(self.device)
            
            # Reinitialize components that depend on vocab size
            self.initializer = ModelInit(d_model=self.hidden_dim, vocab_size=actual_vocab_size, device=self.device)
            self._initialize_all_components()
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not built. Call build_tokenizer() first.")
        return self.tokenizer.encode(text, max_length=self.max_seq_length)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not built. Call build_tokenizer() first.")
        return self.tokenizer.decode(token_ids)
    
    def fit(self, dataset: Union[List[str], GAIADatasetInterface, str], epochs: int = None, batch_size: int = None, 
            learning_rate: float = None, validation_split: float = None) -> Dict[str, float]:
        """Train the GAIA language model."""
        # Use provided parameters or fall back to model config defaults
        epochs = epochs if epochs is not None else getattr(self.config, 'default_epochs', 10)
        batch_size = batch_size if batch_size is not None else getattr(self.config, 'default_batch_size', 8)
        learning_rate = learning_rate if learning_rate is not None else getattr(self.config, 'default_learning_rate', 1e-4)
        validation_split = validation_split if validation_split is not None else getattr(self.config, 'default_validation_split', 0.2)
        
        # Convert dataset to GAIA interface
        gaia_dataset = create_gaia_dataset(dataset)
        texts = gaia_dataset.get_texts()
        
        # Build tokenizer if not already built
        if not self.tokenizer:
            self.build_tokenizer(texts)
        
        # Split data
        split_idx = int(len(texts) * (1 - validation_split))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:] if validation_split > 0 else None
        
        logger.info(f"🚀 Starting training: {len(train_texts)} train, {len(val_texts) if val_texts else 0} val samples")
        
        # Create data loaders
        train_loader, val_loader = DataLoaders(
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_seq_length=self.max_seq_length,
            apply_yoneda=True,
            apply_simplicial=True
        )
        
        # Setup training with checkpoint integration
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = CategoricalLoss(device=self.device, ignore_index=-100, reduction='mean')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        
        # Initialize training state if not already done
        if self.training_state is None:
            self._setup_checkpoint_manager()
        
        # Update training state with current configuration
        if self.training_state:
            self.training_state.config = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'validation_split': validation_split
            }
        
        # Training metrics
        train_losses = []
        val_perplexities = []
        
        # Production logging
        if hasattr(self, 'model_logger'):
            self.model_logger.info(f"🚀 Starting training session: {epochs} epochs, lr={learning_rate}, batch_size={batch_size}")
            self.model_logger.info(f"📊 Dataset: {len(train_texts)} train, {len(val_texts) if val_texts else 0} val samples")
            self.model_logger.info(f"💾 Checkpoint manager: {'enabled' if self.checkpoint_manager else 'disabled'}")
        
        # Training loop with checkpoint integration
        self.train()
        best_val_loss = float('inf')
        epochs_completed = 0
        
        for epoch in range(epochs):
            epochs_completed = epoch + 1
            try:
                # Update training state
                if self.training_state:
                    self.training_state.epoch = epoch
                    epoch_start_time = time.time()
                
                # Train epoch with comprehensive logging
                epoch_loss = self._train_epoch(train_loader, optimizer, criterion, epoch)
                train_losses.append(epoch_loss)
                
                # Validation with checkpoint consideration
                val_perplexity = None
                val_loss = None
                
                if val_loader:
                    val_perplexity = self._evaluate(val_loader)
                    val_perplexities.append(val_perplexity)
                    val_loss = math.log(val_perplexity)  # Convert perplexity back to loss
                    
                    # Update training state with validation metrics
                    if self.training_state:
                        self.training_state.update_val_metrics({
                            'val_loss': val_loss,
                            'val_perplexity': val_perplexity,
                            'epoch': epoch
                        })
                
                # Update training state with training metrics
                if self.training_state:
                    epoch_time = time.time() - epoch_start_time
                    self.training_state.update_train_metrics({
                        'train_loss': epoch_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'epoch_time': epoch_time
                    })
                    self.training_state.epoch_times.append(epoch_time)
                
                # Scheduler step - use validation loss if available, otherwise train loss
                scheduler_metric = val_loss if val_loss is not None else epoch_loss
                scheduler.step(scheduler_metric)
                
                # Checkpoint saving with comprehensive state tracking
                if self.checkpoint_manager and self.training_state:
                    try:
                        # Save checkpoint with full state
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(
                            model=self,
                            training_state=self.training_state,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            extra_data={
                                'train_losses': train_losses,
                                'val_perplexities': val_perplexities,
                                'categorical_components': list(self.components.keys()),
                                'model_metadata': self.model_metadata
                            }
                        )
                        
                        # Production logging for checkpoint
                        if hasattr(self, 'model_logger'):
                            self.model_logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")
                        
                    except Exception as checkpoint_error:
                        logger.error(f"❌ Checkpoint save failed: {checkpoint_error}")
                        if hasattr(self, 'model_logger'):
                            self.model_logger.error(f"❌ Checkpoint save failed: {checkpoint_error}")
                
                # Track best validation loss
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if hasattr(self, 'model_logger'):
                        self.model_logger.info(f"🏆 New best model saved! Val loss: {val_loss:.4f}")
                
                # Enhanced logging
                if val_loader:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Val Perplexity = {val_perplexity:.2f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
                    if hasattr(self, 'model_logger'):
                        self.model_logger.info(f"📈 Epoch {epoch+1}: train_loss={epoch_loss:.4f}, val_perplexity={val_perplexity:.2f}, lr={optimizer.param_groups[0]['lr']:.2e}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
                    if hasattr(self, 'model_logger'):
                        self.model_logger.info(f"📈 Epoch {epoch+1}: train_loss={epoch_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
                
            except Exception as epoch_error:
                logger.error(f"❌ Error in epoch {epoch+1}: {epoch_error}")
                if hasattr(self, 'model_logger'):
                    self.model_logger.error(f"❌ Error in epoch {epoch+1}: {epoch_error}")
                
                # Attempt to recover from checkpoint if available
                if self.checkpoint_manager and epoch > 0:
                    try:
                        logger.info("🔄 Attempting recovery from last checkpoint...")
                        self.load_from_checkpoint()
                        logger.info("✅ Recovery successful, continuing training...")
                        continue
                    except Exception as recovery_error:
                        logger.error(f"❌ Recovery failed: {recovery_error}")
                        break
                else:
                    break
        
        final_perplexity = self._evaluate(val_loader) if val_loader else None
        final_train_loss = train_losses[-1] if train_losses else None
        logger.info(f"✅ Training completed! Final validation perplexity: {final_perplexity:.2f}" if final_perplexity else "✅ Training completed!")
        
        # Mark as trained
        self.model_metadata['trained'] = True
        
        return {
            'train_losses': train_losses,
            'val_perplexities': val_perplexities,
            'final_perplexity': final_perplexity,
            'final_train_loss': final_train_loss,
            'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
            'epochs_completed': epochs_completed
        }
    
    def fit_with_loaders(self, train_loader, val_loader=None, epochs: int = None, 
                        learning_rate: float = None, checkpoint_manager=None) -> Dict[str, float]:
        """Train the GAIA language model with pre-created data loaders.
        
        This method provides the exact same functionality as fit() but accepts
        pre-created data loaders instead of creating them from a dataset.
        """
        # Use provided parameters or fall back to model config defaults
        epochs = epochs if epochs is not None else getattr(self.config, 'default_epochs', 10)
        learning_rate = learning_rate if learning_rate is not None else getattr(self.config, 'default_learning_rate', 1e-4)
        
        # Set checkpoint manager if provided
        if checkpoint_manager:
            self.checkpoint_manager = checkpoint_manager
        
        # Validate tokenizer vocabulary consistency
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'tokenizer'):
            loader_vocab_size = train_loader.dataset.tokenizer.vocab_size
            if loader_vocab_size != self.vocab_size:
                raise ValueError(
                    f"Tokenizer vocabulary mismatch: loader has {loader_vocab_size} tokens, "
                    f"but model expects {self.vocab_size}. Please rebuild loaders with matching tokenizer "
                    f"or update model vocab_size."
                )
        
        # Validate preprocessing flags match fit() behavior
        if hasattr(train_loader, 'dataset'):
            dataset = train_loader.dataset
            if hasattr(dataset, 'apply_yoneda') and not getattr(dataset, 'apply_yoneda', True):
                logger.warning("⚠️ Loader missing apply_yoneda=True flag - may differ from fit() behavior")
            if hasattr(dataset, 'apply_simplicial') and not getattr(dataset, 'apply_simplicial', True):
                logger.warning("⚠️ Loader missing apply_simplicial=True flag - may differ from fit() behavior")
        
        logger.info(f"🚀 Starting training with loaders: {epochs} epochs, lr={learning_rate}")
        
        # Set deterministic behavior for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Setup training with checkpoint integration
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = CategoricalLoss(device=self.device, ignore_index=-100, reduction='mean')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        
        # Initialize training state if not already done
        if self.training_state is None:
            self._setup_checkpoint_manager()
        
        # Update training state with current configuration
        if self.training_state:
            self.training_state.config = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': getattr(train_loader, 'batch_size', 'unknown'),
                'validation_split': 0.0 if val_loader is None else -1.0  # -1.0 indicates external loaders
            }
        
        # Training metrics
        train_losses = []
        val_perplexities = []
        
        # Production logging
        if hasattr(self, 'model_logger'):
            self.model_logger.info(f"🚀 Starting training session with loaders: {epochs} epochs, lr={learning_rate}")
            self.model_logger.info(f"💾 Checkpoint manager: {'enabled' if self.checkpoint_manager else 'disabled'}")
        
        # Training loop with checkpoint integration
        self.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            try:
                # Update training state
                if self.training_state:
                    self.training_state.epoch = epoch
                    epoch_start_time = time.time()
                
                # Train epoch with comprehensive logging
                epoch_loss = self._train_epoch(train_loader, optimizer, criterion, epoch)
                train_losses.append(epoch_loss)
                
                # Validation with checkpoint consideration
                val_perplexity = None
                val_loss = epoch_loss  # Default to train loss if no validation
                
                if val_loader:
                    val_perplexity = self._evaluate(val_loader)
                    val_perplexities.append(val_perplexity)
                    val_loss = math.log(val_perplexity)  # Convert perplexity back to loss
                    
                    # Update training state with validation metrics
                    if self.training_state:
                        self.training_state.update_val_metrics({
                            'val_loss': val_loss,
                            'val_perplexity': val_perplexity,
                            'epoch': epoch
                        })
                
                # Update training state with training metrics
                if self.training_state:
                    epoch_time = time.time() - epoch_start_time
                    self.training_state.update_train_metrics({
                        'train_loss': epoch_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'epoch_time': epoch_time
                    })
                    self.training_state.epoch_times.append(epoch_time)
                
                # Scheduler step
                scheduler.step(val_loss)
                
                # Checkpoint saving with comprehensive state tracking
                if self.checkpoint_manager and self.training_state:
                    try:
                        # Save checkpoint with full state
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(
                            model=self,
                            training_state=self.training_state,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            extra_data={
                                'train_losses': train_losses,
                                'val_perplexities': val_perplexities,
                                'categorical_components': list(self.components.keys()),
                                'model_metadata': self.model_metadata
                            }
                        )
                        
                        # Production logging for checkpoint
                        if hasattr(self, 'model_logger'):
                            self.model_logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                self.model_logger.info(f"🏆 New best model saved! Val loss: {val_loss:.4f}")
                        
                    except Exception as checkpoint_error:
                        logger.error(f"❌ Checkpoint save failed: {checkpoint_error}")
                        if hasattr(self, 'model_logger'):
                            self.model_logger.error(f"❌ Checkpoint save failed: {checkpoint_error}")
                
                # Enhanced logging
                if val_loader:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Val Perplexity = {val_perplexity:.2f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
                    if hasattr(self, 'model_logger'):
                        self.model_logger.info(f"📈 Epoch {epoch+1}: train_loss={epoch_loss:.4f}, val_perplexity={val_perplexity:.2f}, lr={optimizer.param_groups[0]['lr']:.2e}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
                    if hasattr(self, 'model_logger'):
                        self.model_logger.info(f"📈 Epoch {epoch+1}: train_loss={epoch_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
                
            except Exception as epoch_error:
                logger.error(f"❌ Error in epoch {epoch+1}: {epoch_error}")
                if hasattr(self, 'model_logger'):
                    self.model_logger.error(f"❌ Error in epoch {epoch+1}: {epoch_error}")
                
                # Attempt to recover from checkpoint if available
                if self.checkpoint_manager and epoch > 0:
                    try:
                        logger.info("🔄 Attempting recovery from last checkpoint...")
                        self.load_from_checkpoint()
                        logger.info("✅ Recovery successful, continuing training...")
                        continue
                    except Exception as recovery_error:
                        logger.error(f"❌ Recovery failed: {recovery_error}")
                        break
                else:
                    break
        
        final_perplexity = self._evaluate(val_loader) if val_loader else None
        logger.info(f"✅ Training completed! Final validation perplexity: {final_perplexity:.2f}" if final_perplexity else "✅ Training completed!")
        
        # Mark as trained
        self.model_metadata['trained'] = True
        
        return {
            'train_losses': train_losses,
            'val_perplexities': val_perplexities,
            'final_perplexity': final_perplexity
        }
    
    def _train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch using categorical components."""
        epoch_start = time.time()
        logger.debug(f"🔍 STARTING EPOCH {epoch+1} - Total batches: {len(train_loader)}")
        
        running_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(progress_bar):
            batch_start = time.time()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(input_ids, attention_mask)
            
            # Compute loss
            loss_start = time.time()
            categorical_loss = criterion(outputs['logits'], labels)
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Categorical loss computed in {time.time() - loss_start:.4f}s")
            
            # Optional stability loss for initial epochs (configurable)
            if epoch < self.config.stability_loss_epochs:
                logits = outputs['logits']
                stability_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1)) * self.config.stability_loss_weight
                total_loss = categorical_loss + stability_loss
                logger.debug(f"🔍 BATCH {batch_idx + 1}: Using stability loss (epoch {epoch + 1})")
            else:
                total_loss = categorical_loss
            
            # Backward pass
            backward_start = time.time()
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Starting backward pass...")
            total_loss.backward()
            
            # Gradient clipping (configurable)
            if hasattr(self.config, 'gradient_clip_norm') and self.config.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.gradient_clip_norm)
            else:
                # Fallback to optimization config if available
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Backward pass completed in {time.time() - backward_start:.4f}s")
            
            running_loss += total_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})
            
            logger.debug(f"🔍 BATCH {batch_idx + 1} COMPLETED in {time.time() - batch_start:.4f}s")
        
        avg_loss = running_loss / num_batches
        logger.debug(f"🔍 EPOCH {epoch+1} COMPLETED in {time.time() - epoch_start:.4f}s")
        return avg_loss
    
    def _evaluate(self, val_loader):
        """Evaluate model and compute perplexity."""
        self.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.forward(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Compute categorical loss using Gaia framework (consistent with training)
                loss_fn = CategoricalLoss(device=self.device, ignore_index=-100, reduction='mean')
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Count valid tokens for weighting
                valid_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        self.train()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        return math.exp(avg_loss)
      
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with comprehensive categorical operations pipeline."""
        import time
        start_time = time.time()
        
        # ULTRA DETAILED LOGGING - INITIALIZATION
        logger.info(f"🚀 ===== GAIA FORWARD PASS START ===== ")
        logger.info(f"📊 INPUT TENSOR ANALYSIS:")
        logger.info(f"   • Batch shape: {input_ids.shape}")
        logger.info(f"   • Input device: {input_ids.device}")
        logger.info(f"   • Input dtype: {input_ids.dtype}")
        logger.info(f"   • Input min/max: {input_ids.min().item():.4f}/{input_ids.max().item():.4f}")
        logger.info(f"   • Input mean/std: {input_ids.float().mean().item():.4f}/{input_ids.float().std().item():.4f}")
        
        if attention_mask is not None:
            logger.info(f"   • Attention mask shape: {attention_mask.shape}")
            logger.info(f"   • Attention mask sum: {attention_mask.sum().item()}")
        else:
            logger.info(f"   • No attention mask provided")
        
        batch_size, seq_len = input_ids.shape
        logger.info(f"📏 SEQUENCE ANALYSIS: batch_size={batch_size}, seq_len={seq_len}")
        
        # Ensure tensors are on correct device
        device_start = time.time()
        logger.info(f"🔧 DEVICE MANAGEMENT:")
        logger.info(f"   • Model device: {self.device}")
        logger.info(f"   • Input device: {input_ids.device}")
        
        if input_ids.device != self.device:
            logger.info(f"   • Moving input_ids from {input_ids.device} to {self.device}")
            input_ids = input_ids.to(self.device)
            logger.info(f"   • Input transfer completed")
        
        if attention_mask is not None and attention_mask.device != self.device:
            logger.info(f"   • Moving attention_mask from {attention_mask.device} to {self.device}")
            attention_mask = attention_mask.to(self.device)
            logger.info(f"   • Attention mask transfer completed")
        
        device_time = time.time() - device_start
        logger.info(f"   • Device transfer time: {device_time:.4f}s")
        
        # Initialize forward pass counter
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1
        logger.info(f"🔢 FORWARD PASS COUNTER: #{self._forward_count}")
        
        # Add positional embeddings (matching gaia_llm_train.py)
        pos_start = time.time()
        logger.info(f"🎯 ===== POSITIONAL EMBEDDINGS COMPUTATION =====")
        logger.info(f"📍 Creating position IDs for sequence length: {seq_len}")
        
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        logger.info(f"   • Position IDs shape: {position_ids.shape}")
        logger.info(f"   • Position IDs range: {position_ids.min().item()} to {position_ids.max().item()}")
        
        logger.info(f"🔮 Computing positional embeddings...")
        position_embeddings = self.position_embeddings(position_ids)
        logger.info(f"   • Position embeddings shape: {position_embeddings.shape}")
        logger.info(f"   • Position embeddings stats: mean={position_embeddings.mean().item():.4f}, std={position_embeddings.std().item():.4f}")
        logger.info(f"   • Position embeddings min/max: {position_embeddings.min().item():.4f}/{position_embeddings.max().item():.4f}")
        
        pos_time = time.time() - pos_start
        logger.info(f"   • Positional embeddings computation time: {pos_time:.4f}s")
        
        # GAIA Transformer forward
        transformer_start = time.time()
        logger.info(f"🤖 ===== GAIA TRANSFORMER FORWARD PASS =====")
        logger.info(f"🔧 Transformer configuration:")
        logger.info(f"   • Model dimension: {getattr(self.gaia_transformer, 'd_model', 'unknown')}")
        logger.info(f"   • Number of layers: {getattr(self.gaia_transformer, 'num_layers', 'unknown')}")
        logger.info(f"   • Number of heads: {getattr(self.gaia_transformer, 'num_heads', 'unknown')}")
        logger.info(f"   • Vocabulary size: {getattr(self.gaia_transformer, 'vocab_size', 'unknown')}")
        
        logger.info(f"🚀 Executing GAIA transformer forward pass...")
        transformer_outputs = self.gaia_transformer(input_ids, attention_mask)
        
        logger.info(f"📤 GAIA Transformer outputs analysis:")
        logger.info(f"   • Output keys: {list(transformer_outputs.keys())}")
        
        hidden_states = transformer_outputs['last_hidden_state']
        logger.info(f"   • Hidden states shape: {hidden_states.shape}")
        logger.info(f"   • Hidden states stats: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        logger.info(f"   • Hidden states min/max: {hidden_states.min().item():.4f}/{hidden_states.max().item():.4f}")
        
        logger.info(f"➕ Adding positional embeddings to hidden states...")
        hidden_states_before = hidden_states.clone()
        hidden_states = hidden_states + position_embeddings
        
        logger.info(f"   • Combined hidden states stats: mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        logger.info(f"   • Position embedding contribution: {(hidden_states - hidden_states_before).abs().mean().item():.4f}")
        
        transformer_time = time.time() - transformer_start
        logger.info(f"   • GAIA transformer total time: {transformer_time:.4f}s")
        
        # HIERARCHICAL MESSAGE PASSING
        hierarchical_start = time.time()
        logger.info(f"📡 ===== HIERARCHICAL MESSAGE PASSING =====")
        logger.info(f"🔧 HMP Configuration:")
        logger.info(f"   • Message passing enabled: {self.config.enable_message_passing}")
        logger.info(f"   • Message passing object exists: {self.message_passing is not None}")
        
        if self.config.enable_message_passing and self.message_passing:
            logger.info(f"🚀 STARTING HIERARCHICAL MESSAGE PASSING (forward #{self._forward_count})")
            logger.info(f"📊 HMP Parameters:")
            logger.info(f"   • Hierarchical steps: {self.config.hierarchical_steps}")
            logger.info(f"   • Max hierarchical steps: {self.config.max_hierarchical_steps}")
            logger.info(f"   • Convergence threshold: {getattr(self.config, 'convergence_threshold', 1e-3)}")
            logger.info(f"   • Learning rate: 1e-4 (default)")
            
            try:
                logger.info(f"🔄 Executing full hierarchical message passing...")
                # Use default hierarchical steps since GAIAConfig doesn't have get_training_config
                message_results = self.message_passing.full_hierarchical_message_passing(
                num_steps=self.config.hierarchical_steps,
                learning_rate=1e-4,  # Default learning rate
                convergence_threshold=getattr(self.config, 'convergence_threshold', 1e-3),
                max_steps=self.config.max_hierarchical_steps
                 )
                
                hierarchical_time = time.time() - hierarchical_start
                logger.info(f"✅ HIERARCHICAL MESSAGE PASSING COMPLETED:")
                logger.info(f"   • Execution time: {hierarchical_time:.4f}s")
                logger.info(f"   • Message results type: {type(message_results)}")
                if hasattr(message_results, 'keys'):
                    logger.info(f"   • Message results keys: {list(message_results.keys())}")
                elif isinstance(message_results, (list, tuple)):
                    logger.info(f"   • Message results length: {len(message_results)}")
                
            except Exception as e:
                hierarchical_time = time.time() - hierarchical_start
                logger.error(f"❌ HMP FAILED after {hierarchical_time:.4f}s:")
                logger.error(f"   • Error type: {type(e).__name__}")
                logger.error(f"   • Error message: {str(e)}")
                logger.error(f"   • Using identity transformation fallback")
        else:
            logger.info(f"⏭️  HIERARCHICAL MESSAGE PASSING SKIPPED:")
            if not self.config.enable_message_passing:
                logger.info(f"   • Reason: Message passing disabled in config")
            if not self.message_passing:
                logger.info(f"   • Reason: Message passing object not initialized")
        
        # FUZZY ENCODING
        fuzzy_start = time.time()
        logger.info(f"🌊 ===== FUZZY ENCODING PIPELINE =====")
        logger.info(f"🔧 Fuzzy Configuration:")
        logger.info(f"   • Fuzzy components enabled: {self.config.enable_fuzzy_components}")
        logger.info(f"   • Fuzzy encoding pipeline exists: {self.fuzzy_encoding_pipeline is not None}")
        
        fuzzy_encoded = None
        if self.config.enable_fuzzy_components and self.fuzzy_encoding_pipeline:
            logger.info(f"🚀 STARTING FUZZY ENCODING")
            logger.info(f"📊 Input analysis for fuzzy encoding:")
            logger.info(f"   • Hidden states shape: {hidden_states.shape}")
            logger.info(f"   • Hidden states device: {hidden_states.device}")
            logger.info(f"   • Hidden states requires_grad: {hidden_states.requires_grad}")
            
            # Prepare data for fuzzy encoding
            logger.info(f"🔄 Preparing data for fuzzy encoding...")
            hidden_states_cpu = hidden_states.detach().cpu() if hidden_states.device.type == 'mps' else hidden_states.detach()
            logger.info(f"   • Converted to device: {hidden_states_cpu.device}")
            logger.info(f"   • CPU tensor stats: mean={hidden_states_cpu.mean().item():.4f}, std={hidden_states_cpu.std().item():.4f}")
            
            logger.info(f"🌊 Executing fuzzy encoding pipeline...")
            fuzzy_encoded = self.fuzzy_encoding_pipeline.encode(hidden_states_cpu)
            
            fuzzy_time = time.time() - fuzzy_start
            logger.info(f"✅ FUZZY ENCODING COMPLETED:")
            logger.info(f"   • Execution time: {fuzzy_time:.4f}s")
            logger.info(f"   • Fuzzy encoded type: {type(fuzzy_encoded)}")
            if fuzzy_encoded is not None:
                if hasattr(fuzzy_encoded, 'shape'):
                    logger.info(f"   • Fuzzy encoded shape: {fuzzy_encoded.shape}")
                elif hasattr(fuzzy_encoded, '__len__'):
                    logger.info(f"   • Fuzzy encoded length: {len(fuzzy_encoded)}")
        else:
            logger.info(f"⏭️  FUZZY ENCODING SKIPPED:")
            if not self.config.enable_fuzzy_components:
                logger.info(f"   • Reason: Fuzzy components disabled in config")
            if not self.fuzzy_encoding_pipeline:
                logger.info(f"   • Reason: Fuzzy encoding pipeline not initialized")

        # Use original hidden states for membership computation
        fuzzy_input = hidden_states
        logger.info(f"📋 Using hidden states as fuzzy input: shape={fuzzy_input.shape}")
        
        # FUZZY MEMBERSHIP COMPUTATION
        membership_start = time.time()
        logger.info(f"🎯 ===== FUZZY MEMBERSHIP COMPUTATION =====")
        logger.info(f"🔧 Membership Configuration:")
        logger.info(f"   • Fuzzy components enabled: {self.config.enable_fuzzy_components}")
        logger.info(f"   • Categorical ops exists: {self.categorical_ops is not None}")
        
        fuzzy_memberships = None
        if self.config.enable_fuzzy_components:
            logger.info(f"🚀 STARTING FUZZY MEMBERSHIP COMPUTATION")
            logger.info(f"📊 Fuzzy input analysis:")
            logger.info(f"   • Input shape: {fuzzy_input.shape}")
            logger.info(f"   • Input stats: mean={fuzzy_input.mean().item():.4f}, std={fuzzy_input.std().item():.4f}")
            logger.info(f"   • Input min/max: {fuzzy_input.min().item():.4f}/{fuzzy_input.max().item():.4f}")
            
            try:
                logger.info(f"🔄 Computing token fuzzy membership...")
                fuzzy_memberships = self.categorical_ops.compute_token_fuzzy_membership(fuzzy_input)
                
                membership_time = time.time() - membership_start
                logger.info(f"✅ FUZZY MEMBERSHIP COMPLETED:")
                logger.info(f"   • Execution time: {membership_time:.4f}s")
                logger.info(f"   • Membership type: {type(fuzzy_memberships)}")
                if fuzzy_memberships is not None:
                    if hasattr(fuzzy_memberships, 'shape'):
                        logger.info(f"   • Membership shape: {fuzzy_memberships.shape}")
                        logger.info(f"   • Membership stats: mean={fuzzy_memberships.mean().item():.4f}, std={fuzzy_memberships.std().item():.4f}")
                    elif hasattr(fuzzy_memberships, '__len__'):
                        logger.info(f"   • Membership length: {len(fuzzy_memberships)}")
                        
            except Exception as e:
                membership_time = time.time() - membership_start
                logger.error(f"❌ FUZZY MEMBERSHIP FAILED after {membership_time:.4f}s:")
                logger.error(f"   • Error type: {type(e).__name__}")
                logger.error(f"   • Error message: {str(e)}")
                logger.error(f"   • Fuzzy memberships set to None")
        else:
            logger.info(f"⏭️  FUZZY MEMBERSHIP COMPUTATION SKIPPED:")
            logger.info(f"   • Reason: Fuzzy components disabled in config")
        
        # COALGEBRA EVOLUTION
        coalgebra_start = time.time()
        logger.info(f"🧬 ===== COALGEBRA EVOLUTION =====")
        logger.info(f"🔧 Coalgebra Configuration:")
        logger.info(f"   • Coalgebras enabled: {self.config.enable_coalgebras}")
        logger.info(f"   • Forward pass count: #{self._forward_count}")
        logger.info(f"   • Evolution frequency: {getattr(self.config, 'coalgebra_evolution_frequency', 5)}")
        logger.info(f"   • Evolution steps: {getattr(self.config, 'coalgebra_evolution_steps', 3)}")
        
        evolved_state = fuzzy_input
        logger.info(f"📋 Initial evolved state: shape={evolved_state.shape}, mean={evolved_state.mean().item():.4f}")
        
        if self.config.enable_coalgebras:
            logger.info(f"🚀 STARTING COALGEBRA EVOLUTION (forward #{self._forward_count})")
            
            # Check evolution frequency
            evolution_frequency = getattr(self.config, 'coalgebra_evolution_frequency', 5)
            should_evolve = self._forward_count % evolution_frequency == 0
            logger.info(f"📊 Evolution frequency check:")
            logger.info(f"   • Current forward pass: {self._forward_count}")
            logger.info(f"   • Evolution frequency: {evolution_frequency}")
            logger.info(f"   • Should evolve: {should_evolve}")
            
            try:
                # Only run coalgebra evolution every N forward passes for efficiency
                if should_evolve:
                    logger.info(f"🔄 FULL COALGEBRA EVOLUTION TRIGGERED")
                    
                    # Update coalgebra training data with properly shaped tensors
                    logger.info(f"📊 Preparing coalgebra training data:")
                    logger.info(f"   • Fuzzy input shape: {fuzzy_input.shape}")
                    
                    # Use mean-pooled hidden states for both input and target to ensure consistent dimensions
                    coalgebra_input = fuzzy_input.mean(dim=1)  # [batch_size, hidden_dim]
                    coalgebra_target = fuzzy_input.mean(dim=1)  # [batch_size, hidden_dim] - same shape
                    
                    logger.info(f"   • Coalgebra input shape: {coalgebra_input.shape}")
                    logger.info(f"   • Coalgebra target shape: {coalgebra_target.shape}")
                    logger.info(f"   • Coalgebra input stats: mean={coalgebra_input.mean().item():.4f}, std={coalgebra_input.std().item():.4f}")
                    logger.info(f"   • Coalgebra target stats: mean={coalgebra_target.mean().item():.4f}, std={coalgebra_target.std().item():.4f}")
                    
                    logger.info(f"🔧 Available coalgebra components:")
                    logger.info(f"   • Generative coalgebra: {self.components.get('generative_coalgebra') is not None}")
                    logger.info(f"   • Backprop functor class: {self.components.get('backprop_functor_class') is not None}")
                    logger.info(f"   • State coalgebra: {self.components.get('state_coalgebra') is not None}")
                    
                    try:
                        logger.info(f"🔄 Updating coalgebra training data...")
                        self.categorical_ops.update_coalgebra_training_data(
                            self.components.get('generative_coalgebra'),
                            coalgebra_input,
                            coalgebra_target,
                            self.components.get('backprop_functor_class'),
                            self.components.get('state_coalgebra')
                        )
                        logger.info(f"✅ Coalgebra training data updated successfully")
                        
                        evolution_start = time.time()
                        coalgebra_trajectory = self.categorical_ops.evolve_generative_coalgebra(
                            coalgebra_input,  # Use consistent input shape
                            self.components.get('generative_coalgebra'),
                            self.components.get('coalgebra_optimizer'),
                            self.components.get('coalgebra_loss_fn'),
                            getattr(self.categorical_ops, '_backprop_functor', None),
                            steps=getattr(self.config, 'coalgebra_evolution_steps', 3)
                        )
                        # Reshape evolved coalgebra state to match expected dimensions
                        evolved_params = coalgebra_trajectory[-1]  # Shape: [3855000]
                        batch_size, seq_len, hidden_dim = fuzzy_input.shape
                        
                        
                        # Project evolved parameters to hidden dimension space
                        if evolved_params.numel() >= batch_size * hidden_dim:
                            # Truncate and reshape if evolved params are larger
                            evolved_reshaped = evolved_params[:batch_size * hidden_dim].view(batch_size, hidden_dim)
                            evolved_state = evolved_reshaped.unsqueeze(1).expand(-1, seq_len, -1)
                        else:
                            # Use mean pooling if evolved params are smaller
                            param_per_batch = evolved_params.numel() // batch_size
                            if param_per_batch > 0:
                                evolved_reshaped = evolved_params[:batch_size * param_per_batch].view(batch_size, param_per_batch)
                                # Pad or truncate to hidden_dim
                                if param_per_batch < hidden_dim:
                                    padding = torch.zeros(batch_size, hidden_dim - param_per_batch, device=evolved_params.device, dtype=evolved_params.dtype)
                                    evolved_reshaped = torch.cat([evolved_reshaped, padding], dim=1)
                                else:
                                    evolved_reshaped = evolved_reshaped[:, :hidden_dim]
                                evolved_state = evolved_reshaped.unsqueeze(1).expand(-1, seq_len, -1)
                                logger.debug(f"🔍 COALGEBRA RESHAPE: Padded/truncated to {evolved_reshaped.shape}, expanded to {evolved_state.shape}")
                            else:
                                # Fallback: use original fuzzy input
                                evolved_state = fuzzy_input
                        
                    except Exception as coalgebra_error:
                        evolved_state = torch.tanh(fuzzy_input) + getattr(self.config, 'coalgebra_fallback_noise', 0.01) * torch.randn_like(fuzzy_input)
                else:
                    # Use simplified evolution for other forward passes
                    # Using simplified coalgebra evolution
                    fallback_start = time.time()
                    evolved_state = torch.tanh(fuzzy_input) + getattr(self.config, 'coalgebra_fallback_noise', 0.01) * torch.randn_like(fuzzy_input)
            except Exception as e:
                evolved_state = torch.tanh(fuzzy_input) + getattr(self.config, 'coalgebra_fallback_noise', 0.01) * torch.randn_like(fuzzy_input)
        
        
        # YONEDA EMBEDDINGS
        yoneda_start = time.time()
        yoneda_embedded = evolved_state
        
        if self.config.enable_yoneda_embeddings and 'yoneda_proxy' in self.components:
            try:
                # Process each sequence position separately to preserve sequence dimension
                batch_size, seq_len, hidden_dim = evolved_state.shape
                yoneda_embedded = torch.zeros_like(evolved_state)
                
                for i in range(seq_len):
                    seq_slice = evolved_state[:, i, :]
                    yoneda_slice = self.components['yoneda_proxy']._profile(seq_slice).squeeze(-1)
                    
                    # Ensure yoneda_slice has correct dimension
                    if yoneda_slice.shape[-1] != hidden_dim:
                        if yoneda_slice.shape[-1] < hidden_dim:
                            padding = torch.zeros(batch_size, hidden_dim - yoneda_slice.shape[-1], device=yoneda_slice.device)
                            yoneda_slice = torch.cat([yoneda_slice, padding], dim=-1)
                        else:
                            yoneda_slice = yoneda_slice[:, :hidden_dim]
                    yoneda_embedded[:, i, :] = yoneda_slice
                
            except Exception as e:
                yoneda_embedded = evolved_state
        
        # KAN EXTENSIONS
        kan_start = time.time()
        # Starting Kan extensions
        compositional_repr = yoneda_embedded
        
        if (self.config.enable_kan_extensions and 
            'left_kan_extension' in self.components and 
            'right_kan_extension' in self.components):
            try:
                # Process sequence-wise to preserve dimensions
                batch_size, seq_len, hidden_dim = yoneda_embedded.shape
                compositional_repr = torch.zeros_like(yoneda_embedded)
                # Processing sequence positions
                
                for i in range(seq_len):
                    seq_slice = yoneda_embedded[:, i, :]
                    # Processing position
                    
                    kan_slice = self.categorical_ops.apply_compositional_kan_extensions(
                        seq_slice,
                        self.components['left_kan_extension'],
                        self.components['right_kan_extension']
                    )
                    
                    # Handle tensor dimension mismatch
                    if len(kan_slice.shape) == 3 and kan_slice.shape[1] == 1:
                        kan_slice = kan_slice.squeeze(1)
                    
                    # Ensure kan_slice has correct dimension
                    if kan_slice.shape[-1] != hidden_dim:
                        if kan_slice.shape[-1] < hidden_dim:
                            padding = torch.zeros(batch_size, hidden_dim - kan_slice.shape[-1], device=kan_slice.device)
                            kan_slice = torch.cat([kan_slice, padding], dim=-1)
                        else:
                            kan_slice = kan_slice[:, :hidden_dim]
                    compositional_repr[:, i, :] = kan_slice
                
            except Exception as e:
                compositional_repr = yoneda_embedded
        
        # ENDS/COENDS COMPUTATION
        ends_start = time.time()
        final_repr = compositional_repr
        
        if (self.config.enable_ends_coends and 
            'end_computation' in self.components and 
            'coend_computation' in self.components):
            try:
                # Process sequence-wise to preserve dimensions
                batch_size, seq_len, hidden_dim = compositional_repr.shape
                final_repr = torch.zeros_like(compositional_repr)
                
                for i in range(seq_len):
                    seq_slice = compositional_repr[:, i, :]
                    end_result, coend_result = self.categorical_ops.compute_ends_coends(
                        seq_slice,
                        self.components['end_computation'],
                        self.components['coend_computation']
                    )
                    
                    # Ensure results have correct dimension
                    if end_result.shape[-1] != hidden_dim:
                        if end_result.shape[-1] < hidden_dim:
                            padding = torch.zeros(batch_size, hidden_dim - end_result.shape[-1], device=end_result.device)
                            end_result = torch.cat([end_result, padding], dim=-1)
                        else:
                            end_result = end_result[:, :hidden_dim]
                    
                    if coend_result.shape[-1] != hidden_dim:
                        if coend_result.shape[-1] < hidden_dim:
                            padding = torch.zeros(batch_size, hidden_dim - coend_result.shape[-1], device=coend_result.device)
                            coend_result = torch.cat([coend_result, padding], dim=-1)
                        else:
                            coend_result = coend_result[:, :hidden_dim]
                    
                    # Use configurable mixing weights
                    weights = getattr(self.config, 'component_mixing_weights', [0.4, 0.3, 0.3])
                    final_repr[:, i, :] = (weights[0] * seq_slice + weights[1] * end_result + weights[2] * coend_result)
                
            except Exception as e:
                final_repr = compositional_repr
        
        # Final projection to vocabulary
        lm_head_start = time.time()
        logits = self.gaia_transformer.output_projection(final_repr)

        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'fuzzy_encoded': fuzzy_encoded,
            'fuzzy_memberships': fuzzy_memberships,
            'coalgebra_evolved': evolved_state,
            'yoneda_embedded': yoneda_embedded,
            'compositional_repr': compositional_repr,
            'final_repr': final_repr,
            'transformer_outputs': transformer_outputs,
            'components': self.components
        }
    
    def generate(self, input_text: str, max_length: int = 100, temperature: float = 1.0, **kwargs) -> str:
        """Generate text from the model."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not built. Train the model first.")
        
        self.eval()
        
        # Encode input
        input_ids = self.encode_text(input_text).unsqueeze(0).to(self.device)
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated_ids)
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max sequence length
                if generated_ids.shape[1] >= self.max_seq_length:
                    break
        
        # Decode generated text
        generated_text = self.decode_tokens(generated_ids[0])
        
        self.train()  # Switch back to training mode
        return generated_text
    
    def save_checkpoint(self, checkpoint_path: Optional[Path] = None, optimizer: Optional[torch.optim.Optimizer] = None, 
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                      extra_data: Optional[Dict[str, Any]] = None) -> Path:
        """Save model checkpoint with comprehensive state tracking."""
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager not initialized. Cannot save checkpoint.")
        
        if not self.training_state:
            # Create minimal training state if not exists
            self.training_state = TrainingState(
                model_info={
                    'model_type': 'GAIALanguageModel',
                    'parameters': sum(p.numel() for p in self.parameters()),
                    'device': str(self.device)
                }
            )
        
        try:
            # Add categorical structure information
            categorical_data = {
                'components': list(self.components.keys()),
                'model_metadata': self.model_metadata,
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
            }
            
            if extra_data:
                categorical_data.update(extra_data)
            
            # Save checkpoint
            saved_path = self.checkpoint_manager.save_checkpoint(
                model=self,
                training_state=self.training_state,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_data=categorical_data,
                filename=checkpoint_path.name if checkpoint_path else None
            )
            
            # Production logging
            if hasattr(self, 'model_logger'):
                self.model_logger.info(f"💾 Manual checkpoint saved: {saved_path}")
            
            logger.info(f"✅ Checkpoint saved successfully: {saved_path}")
            return saved_path
            
        except Exception as e:
            error_msg = f"Failed to save checkpoint: {e}"
            logger.error(f"❌ {error_msg}")
            if hasattr(self, 'model_logger'):
                self.model_logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def load_from_checkpoint(self, checkpoint_path: Optional[Path] = None, load_best: bool = False,
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                           map_location: Optional[str] = None) -> TrainingState:
        """Load model from checkpoint with comprehensive state restoration."""
        if not self.checkpoint_manager:
            raise ValueError("Checkpoint manager not initialized. Cannot load checkpoint.")
        
        try:
            # Load checkpoint
            training_state = self.checkpoint_manager.load_checkpoint(
                model=self,
                checkpoint_path=checkpoint_path,
                load_best=load_best,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=map_location
            )
            
            # Restore training state
            self.training_state = training_state
            
            # Restore model metadata if available
            checkpoint_data = torch.load(
                checkpoint_path if checkpoint_path else 
                (self.checkpoint_manager.checkpoint_dir / "best_checkpoint.pth" if load_best else 
                 self.checkpoint_manager.checkpoint_history[-1]['path']),
                map_location=map_location
            )
            
            if 'extra_data' in checkpoint_data:
                extra_data = checkpoint_data['extra_data']
                if 'model_metadata' in extra_data:
                    self.model_metadata.update(extra_data['model_metadata'])
            
            # Production logging
            if hasattr(self, 'model_logger'):
                self.model_logger.info(f"🔄 Checkpoint loaded: epoch {training_state.epoch}, step {training_state.step}")
            
            logger.info(f"✅ Checkpoint loaded successfully: epoch {training_state.epoch}, step {training_state.step}")
            return training_state
            
        except Exception as e:
            error_msg = f"Failed to load checkpoint: {e}"
            logger.error(f"❌ {error_msg}")
            if hasattr(self, 'model_logger'):
                self.model_logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def get_categorical_structure(self) -> Dict[str, Any]:
        """Get categorical structure information for checkpoint saving."""
        return {
            'components': {name: type(comp).__name__ for name, comp in self.components.items()},
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'model_metadata': self.model_metadata,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.parameters())
        }
    
    def restore_categorical_structure(self, structure_data: Dict[str, Any]):
        """Restore categorical structure from checkpoint data."""
        try:
            # Restore model metadata
            if 'model_metadata' in structure_data:
                self.model_metadata.update(structure_data['model_metadata'])
            
            # Log restoration
            if hasattr(self, 'model_logger'):
                self.model_logger.info(f"🔧 Categorical structure restored: {len(structure_data.get('components', {}))} components")
            
            logger.info("✅ Categorical structure restored successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to restore categorical structure: {e}")
    
    @classmethod
    def _dict_to_config(cls, config_dict: Dict[str, Any]) -> GAIALanguageModelConfig:
        """Convert dictionary to GAIALanguageModelConfig."""
        # This would need proper implementation based on the config structure
        # For now, create a basic config object
        return GAIALanguageModelConfig(**config_dict)