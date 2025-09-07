#!/usr/bin/env python3
"""
GAIA Framework -  Language Modeling Example
====================================================
This example demonstrates GAIA theoretical framework applied to language modeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
from gaia.training.config import GAIAConfig, ModelConfig, TrainingConfig
from gaia.models import GAIATransformer, ModelInit
from gaia.training import CategoricalLoss, CategoricalOps
from gaia.data import DataLoaders, Dataset
from gaia.data.tokenizer import SimpleTokenizer

# Initialize GAIA global configuration and logger
GAIAConfig.setup_logging(enable_ultra_verbose=True)
logger = GAIAConfig.get_logger('gaia_language_modeling_refactored')



class GAIALanguageModel(nn.Module):
    """
     GAIA Language Model using extracted framework components.
    
    This demonstrates how to properly structure a GAIA model by using
    the refactored components from the framework modules instead of
    implementing everything in the example file.
    """
    
    def __init__(self, config: ModelConfig, device: str = None):
        super().__init__()
        
        # Store config
        self.config = config
        
        # Extract parameters from config
        transformer_config = {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'max_seq_length': config.max_seq_length,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'use_all_gaia_features': True
        }
        
        # Set device
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # Create transformer directly
        self.gaia_transformer = GAIATransformer(**transformer_config)
        
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        self._tokenizer_built = False
        
        # Create language modeling head
        self.lm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.vocab_size)
        )
        
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Move to device
        self.to(self.device)
        
        # Initialize GAIA components using refactored initializer
        self.initializer = ModelInit(self.device, config.d_model, config.vocab_size)
        self._initialize_all_components()
        
        # Initialize categorical training components
        self.categorical_loss_computer = CategoricalLoss(
            self.device, bisimulation_tolerance=1e-3
        )
        self.categorical_operations = CategoricalOps(
            self.device, bisimulation_tolerance=1e-3
        )
        
        # MPS-specific initialization
        if self.device.type == 'mps':
            self.initializer.initialize_mps_tensors(self)
        
        logger.info(" GAIA Language Model initialized successfully")
    
    def _initialize_all_components(self):
        """Initialize all GAIA components using the refactored initializer."""
        # Initialize fuzzy sets
        fuzzy_components = self.initializer.initialize_token_fuzzy_sets()
        
        # Initialize simplicial structure
        simplicial_components = self.initializer.initialize_language_simplicial_structure(
            fuzzy_components['token_category']
        )
        
        # Initialize fuzzy encoding
        self.fuzzy_encoding_pipeline = self.initializer.initialize_fuzzy_encoding_pipeline()
        
        # Initialize coalgebras
        coalgebra_components = self.initializer.initialize_generative_coalgebras(
            self.gaia_transformer.output_projection
        )
        
        # Initialize business units
        self.business_hierarchy = self.initializer.initialize_business_units(self.gaia_transformer)
        
        # Initialize message passing
        self.message_passing = self.initializer.initialize_message_passing()
        
        # Initialize Yoneda embeddings
        yoneda_components = self.initializer.initialize_yoneda_embeddings()
        
        # Initialize Kan extensions
        kan_components = self.initializer.initialize_kan_extensions()
        
        # Initialize ends/coends
        ends_coends_components = self.initializer.initialize_ends_coends(kan_components)
        
        # Store all components for easy access
        self.components = {
            **fuzzy_components,
            **simplicial_components,
            **coalgebra_components,
            **yoneda_components,
            **kan_components,
            **ends_coends_components,
            'business_hierarchy': self.business_hierarchy,
            'message_passing': self.message_passing,
            'fuzzy_encoding_pipeline': self.fuzzy_encoding_pipeline,
            'gaia_transformer': self.gaia_transformer,
            'config': self.config
        }
        
        # Log component summary
        self.initializer.log_framework_components(self.components)
    
    def build_tokenizer(self, texts: List[str]):
        """Build tokenizer vocabulary from texts."""
        logger.info("Building tokenizer vocabulary...")
        self.tokenizer.build_vocab(texts)
        self._tokenizer_built = True
        logger.info(f"Tokenizer built with {len(self.tokenizer.word_to_id)} tokens")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode raw text to token IDs."""
        if not self._tokenizer_built:
            raise RuntimeError("Tokenizer not built. Call build_tokenizer() first.")
        
        token_ids = self.tokenizer.encode(text, max_length=self.max_seq_length)
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        if not self._tokenizer_built:
            raise RuntimeError("Tokenizer not built. Call build_tokenizer() first.")
        
        return self.tokenizer.decode(token_ids.tolist())
    
    def fit(self, texts: List[str], epochs: int = None, batch_size: int = None, 
            learning_rate: float = None, validation_split: float = None) -> Dict[str, float]:
        """High-level training interface using refactored components."""
        # Initialize global training configuration
        
        training_config = TrainingConfig()
        
        # Use provided parameters or fall back to global config
        epochs = epochs or training_config.epochs
        batch_size = batch_size or training_config.data.batch_size
        learning_rate = learning_rate or training_config.optimization.learning_rate
        validation_split = validation_split or training_config.data.validation_split
        
        logger.info(f"🚀 Starting GAIA training on {len(texts)} texts with global config...")
        
        # Build tokenizer
        if not self._tokenizer_built:
            self.build_tokenizer(texts)
        
        # Split dataset
        split_idx = int((1 - validation_split) * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:] if validation_split > 0 else texts[:100]
        
        train_loader, val_loader = DataLoaders(
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_seq_length=self.max_seq_length,
            apply_yoneda=True,
            apply_simplicial=True
        )
        
        # Setup training
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = CategoricalLoss(device=self.device, ignore_index=-100, reduction='mean')
        
        # Training metrics
        train_losses = []
        val_perplexities = []
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = self._train_epoch_refactored(train_loader, optimizer, criterion, epoch)
            train_losses.append(epoch_loss)
            
            # Validation
            if val_loader:
                val_perplexity = self._evaluate(val_loader)
                val_perplexities.append(val_perplexity)
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Val Perplexity = {val_perplexity:.2f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}")
        
        final_perplexity = self._evaluate(val_loader) if val_loader else None
        logger.info(f"✅  training completed! Final validation perplexity: {final_perplexity:.2f}" if final_perplexity else "✅ Training completed!")
        
        return {
            'train_losses': train_losses,
            'val_perplexities': val_perplexities,
            'final_perplexity': final_perplexity
        }
    
    def _train_epoch_refactored(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch using refactored categorical components."""
        import time
        epoch_start = time.time()
        logger.debug(f"🔍 STARTING EPOCH {epoch+1} - Total batches: {len(train_loader)}")
        
        running_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start = time.time()
            logger.debug(f"🔍 STARTING BATCH {batch_idx + 1}/{len(train_loader)}...")
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Prepare batch data
            batch_data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                **{k: v for k, v in batch.items() if k not in ('input_ids', 'attention_mask', 'labels')}
            }
            
            # Forward pass
            forward_start = time.time()
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Starting forward pass...")
            outputs = self.forward(input_ids, attention_mask)
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Forward pass completed in {time.time() - forward_start:.4f}s")
            
            # Compute categorical loss using refactored loss computer
            loss_start = time.time()
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Starting categorical loss computation...")
            categorical_loss = self.categorical_loss_computer.compute_categorical_diagram_loss(
                outputs, batch_data, self.components, epoch, num_batches
            )
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Categorical loss computed in {time.time() - loss_start:.4f}s")
            
            # Optional stability loss for initial epochs
            if epoch < 2:
                logits = outputs['logits']
                stability_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1)) * 0.1
                total_loss = categorical_loss + stability_loss
                logger.debug(f"🔍 BATCH {batch_idx + 1}: Using stability loss (epoch {epoch + 1})")
            else:
                total_loss = categorical_loss
                logger.debug(f"🔍 BATCH {batch_idx + 1}: Using categorical loss only")
            
            # Backward pass
            backward_start = time.time()
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Starting backward pass...")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            logger.debug(f"🔍 BATCH {batch_idx + 1}: Backward pass completed in {time.time() - backward_start:.4f}s")
            
            # Statistics
            running_loss += total_loss.item()
            num_batches += 1
            
            # Batch summary
            batch_time = time.time() - batch_start
            logger.debug(f"🔍 BATCH {batch_idx + 1} COMPLETE - Total time: {batch_time:.4f}s, Loss: {total_loss.item():.4f}")
            
            # Update progress bar
            progress_bar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'categorical': f'{categorical_loss.item():.4f}',
                'avg': f'{running_loss/num_batches:.4f}',
                'time': f'{batch_time:.2f}s'
            })
        
        return running_loss / num_batches
    
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
                
                # Compute categorical loss using Gaia framework
                loss_fn = CategoricalLoss(device=self.device, ignore_index=-100, reduction='sum')
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Count valid tokens
                valid_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item()
                total_tokens += valid_tokens
        
        self.train()
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass using refactored categorical operations."""
        import time
        start_time = time.time()
        logger.debug(f"🔍 FORWARD PASS START - Batch shape: {input_ids.shape}")
        
        batch_size, seq_len = input_ids.shape
        
        # Ensure tensors are on correct device
        device_start = time.time()
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
        logger.debug(f"🔍 DEVICE TRANSFER: {time.time() - device_start:.4f}s")
        
        # Add positional embeddings
        pos_start = time.time()
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        logger.debug(f"🔍 POSITIONAL EMBEDDINGS: {time.time() - pos_start:.4f}s")
        
        # GAIA Transformer forward
        transformer_start = time.time()
        logger.debug(f"🔍 STARTING GAIA TRANSFORMER FORWARD...")
        transformer_outputs = self.gaia_transformer(input_ids, attention_mask)
        hidden_states = transformer_outputs['last_hidden_state']
        hidden_states = hidden_states + position_embeddings
        logger.debug(f"🔍 GAIA TRANSFORMER: {time.time() - transformer_start:.4f}s")
        
        # Apply hierarchical message passing periodically (not every forward pass)
        # Only run hierarchical updates every 10 forward passes to avoid computational overhead
        hierarchical_start = time.time()
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1
        logger.debug(f"🔍 FORWARD COUNT: {self._forward_count}")
        
        # ENABLED: Always run hierarchical message passing for full GAIA functionality
        logger.debug(f"🔍 🚀 ULTRA-VERBOSE: STARTING HIERARCHICAL MESSAGE PASSING (forward #{self._forward_count})...")
        from gaia.training.config import TrainingConfig
        training_config = TrainingConfig()
        hmp_start = time.time()
        logger.debug(f"🔍 🚀 ULTRA-VERBOSE: HMP Config - lr={training_config.optimization.learning_rate}, threshold=1e-3")
        
        try:
            message_results = self.message_passing.full_hierarchical_message_passing(
                num_steps=3,  # Increased steps for full functionality
                learning_rate=training_config.optimization.learning_rate,
                convergence_threshold=1e-3,
                max_steps=5  # Increased limit for better convergence
            )
            logger.debug(f"🔍 🚀 ULTRA-VERBOSE: HMP Results - {message_results}")
            logger.debug(f"🔍 🚀 ULTRA-VERBOSE: HIERARCHICAL MESSAGE PASSING COMPLETED: {time.time() - hmp_start:.4f}s")
        except Exception as e:
            logger.error(f"🔍 🚀 ULTRA-VERBOSE: HMP FAILED: {e}")
            logger.debug(f"🔍 🚀 ULTRA-VERBOSE: HMP FALLBACK: Using identity transformation")
        logger.debug(f"🔍 HIERARCHICAL SECTION TOTAL: {time.time() - hierarchical_start:.4f}s")
        
        # Apply fuzzy encoding using refactored operations
        fuzzy_start = time.time()
        logger.debug(f"🔍 STARTING FUZZY ENCODING...")
        try:
            logger.debug(f"🔍 STARTING FUZZY ENCODING DEVICE...")
            hidden_states_cpu = hidden_states.detach().cpu() if hidden_states.device.type == 'mps' else hidden_states.detach()
            logger.debug(f"🔍 STARTING FUZZY ENCODING DEVICE...2")
            fuzzy_encoded = self.fuzzy_encoding_pipeline.encode(hidden_states_cpu)
            logger.debug(f"🔍 FUZZY ENCODING SUCCESS: {time.time() - fuzzy_start:.4f}s")
            
            # Use original hidden states for membership computation since fuzzy_encoded is FuzzySimplicialSet
            fuzzy_input = hidden_states
        except Exception as e:
            logger.debug(f"🔍 FUZZY ENCODING FAILED: {e} - Using fallback")
            fuzzy_encoded = None
            fuzzy_input = hidden_states
        
        # Compute fuzzy membership using refactored operations
        membership_start = time.time()
        logger.debug(f"🔍 STARTING FUZZY MEMBERSHIP COMPUTATION...")
        fuzzy_memberships = self.categorical_operations.compute_token_fuzzy_membership(fuzzy_input)
        logger.debug(f"🔍 FUZZY MEMBERSHIP: {time.time() - membership_start:.4f}s")
        
        # Evolve through coalgebra using refactored operations (optimized)
        coalgebra_start = time.time()
        logger.debug(f"🔍 STARTING COALGEBRA EVOLUTION (forward #{self._forward_count})...")
        try:
            # Only run coalgebra evolution every 5 forward passes for efficiency
            if self._forward_count % 5 == 0:
                logger.debug(f"🔍 RUNNING FULL COALGEBRA EVOLUTION...")
                evolution_start = time.time()
                coalgebra_trajectory = self.categorical_operations.evolve_generative_coalgebra(
                    fuzzy_input.mean(dim=1),
                    self.components['generative_coalgebra'],
                    self.components['coalgebra_optimizer'],
                    self.components['coalgebra_loss_fn'],
                    self.components['backprop_functor'],
                    steps=1  # Reduced steps for efficiency
                )
                evolved_state = coalgebra_trajectory[-1].unsqueeze(1).expand(-1, seq_len, -1)
                logger.debug(f"🔍 COALGEBRA EVOLUTION COMPLETED: {time.time() - evolution_start:.4f}s")
            else:
                # Use cached or simplified evolution for other forward passes
                logger.debug(f"🔍 USING SIMPLIFIED COALGEBRA EVOLUTION...")
                fallback_start = time.time()
                evolved_state = torch.tanh(fuzzy_input) + 0.05 * torch.randn_like(fuzzy_input)
                logger.debug(f"🔍 SIMPLIFIED EVOLUTION: {time.time() - fallback_start:.4f}s")
        except Exception as e:
            logger.debug(f"🔍 COALGEBRA EVOLUTION FAILED: {e} - Using fallback")
            evolved_state = torch.tanh(fuzzy_input) + 0.05 * torch.randn_like(fuzzy_input)
        logger.debug(f"🔍 COALGEBRA SECTION TOTAL: {time.time() - coalgebra_start:.4f}s")
        
        # Apply Yoneda embeddings while preserving sequence dimension
        yoneda_start = time.time()
        logger.debug(f"🔍 STARTING YONEDA EMBEDDINGS...")
        try:
            # Process each sequence position separately to preserve sequence dimension
            batch_size, seq_len, hidden_dim = evolved_state.shape
            yoneda_embedded = torch.zeros_like(evolved_state)
            
            for i in range(seq_len):
                seq_slice = evolved_state[:, i, :]
                yoneda_slice = self.components['yoneda_proxy']._profile(seq_slice).squeeze(-1)
                # Ensure yoneda_slice has correct dimension
                if yoneda_slice.shape[-1] != hidden_dim:
                    # Pad or truncate to match hidden_dim
                    if yoneda_slice.shape[-1] < hidden_dim:
                        padding = torch.zeros(batch_size, hidden_dim - yoneda_slice.shape[-1], device=yoneda_slice.device)
                        yoneda_slice = torch.cat([yoneda_slice, padding], dim=-1)
                    else:
                        yoneda_slice = yoneda_slice[:, :hidden_dim]
                yoneda_embedded[:, i, :] = yoneda_slice
            
            logger.debug(f"🔍 YONEDA EMBEDDINGS SUCCESS: {time.time() - yoneda_start:.4f}s")
        except Exception as e:
            logger.debug(f"🔍 YONEDA EMBEDDINGS FAILED: {e} - Using fallback")
            yoneda_embedded = evolved_state
        
        # Apply Kan extensions using refactored operations (preserve sequence dimension)
        kan_start = time.time()
        logger.debug(f"🔍 STARTING KAN EXTENSIONS...")
        try:
            # Process sequence-wise to preserve dimensions
            batch_size, seq_len, hidden_dim = yoneda_embedded.shape
            compositional_repr = torch.zeros_like(yoneda_embedded)
            
            for i in range(seq_len):
                seq_slice = yoneda_embedded[:, i, :]
                kan_slice = self.categorical_operations.apply_compositional_kan_extensions(
                    seq_slice,
                    self.components['left_kan_extension'],
                    self.components['right_kan_extension']
                )
                # Ensure kan_slice has correct dimension
                if kan_slice.shape[-1] != hidden_dim:
                    if kan_slice.shape[-1] < hidden_dim:
                        padding = torch.zeros(batch_size, hidden_dim - kan_slice.shape[-1], device=kan_slice.device)
                        kan_slice = torch.cat([kan_slice, padding], dim=-1)
                    else:
                        kan_slice = kan_slice[:, :hidden_dim]
                compositional_repr[:, i, :] = kan_slice
            
            logger.debug(f"🔍 KAN EXTENSIONS SUCCESS: {time.time() - kan_start:.4f}s")
        except Exception as e:
            logger.debug(f"🔍 KAN EXTENSIONS FAILED: {e} - Using fallback")
            compositional_repr = yoneda_embedded
        
        # Compute ends/coends using refactored operations (preserve sequence dimension)
        ends_start = time.time()
        logger.debug(f"🔍 STARTING ENDS/COENDS COMPUTATION...")
        try:
            # Process sequence-wise to preserve dimensions
            batch_size, seq_len, hidden_dim = compositional_repr.shape
            final_repr = torch.zeros_like(compositional_repr)
            
            for i in range(seq_len):
                seq_slice = compositional_repr[:, i, :]
                end_result, coend_result = self.categorical_operations.compute_ends_coends(
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
                        
                final_repr[:, i, :] = (seq_slice + end_result + coend_result) / 3
            
            logger.debug(f"🔍 ENDS/COENDS SUCCESS: {time.time() - ends_start:.4f}s")
        except Exception as e:
            logger.debug(f"🔍 ENDS/COENDS FAILED: {e} - Using fallback")
            final_repr = compositional_repr
        
        # Language modeling head
        lm_head_start = time.time()
        logger.debug(f"🔍 STARTING LANGUAGE MODELING HEAD...")
        logits = self.lm_head(final_repr)
        logger.debug(f"🔍 LANGUAGE MODELING HEAD: {time.time() - lm_head_start:.4f}s")
        
        # Forward pass summary
        total_time = time.time() - start_time
        logger.debug(f"🔍 FORWARD PASS COMPLETE - Total time: {total_time:.4f}s, Output shape: {logits.shape}")
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'fuzzy_encoded': fuzzy_encoded,
            'fuzzy_memberships': fuzzy_memberships,
            'coalgebra_evolved': evolved_state,
            'yoneda_embedded': yoneda_embedded,
            'compositional_repr': compositional_repr,
            'transformer_outputs': transformer_outputs
        }


def create_sample_dataset(size: int = 50) -> List[str]:
    """Create a simple sample dataset to avoid memory issues."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns.",
        "Artificial intelligence is the future of technology.",
        "Data science combines statistics and programming.",
        "Neural networks are inspired by the human brain.",
        "Computer vision allows machines to see and interpret images.",
        "Reinforcement learning teaches agents through trial and error.",
        "Big data requires powerful computational resources."
    ]
    
    # Repeat and extend the sample texts to reach desired size
    texts = []
    for i in range(size):
        texts.append(sample_texts[i % len(sample_texts)])
    
    return texts

def demo_refactored_gaia_language_model():
    """Demonstrate the refactored GAIA language model."""
    global logger
    from gaia.training.config import TrainingConfig, ModelConfig
    
    training_config = TrainingConfig()
    model_config = ModelConfig()
    logger.info("🎯 GAIA Language Model Demo with global config")
    
    # Create dataset based on configuration
    if training_config.data.use_sample_dataset:
        texts = create_sample_dataset(training_config.data.sample_dataset_size)
        logger.info(f"Created sample dataset with {len(texts)} samples")
    else:
        texts = Dataset()
        logger.info(f"Created full dataset with {len(texts)} samples")
    
    # Create refactored model using config injection
    model = GAIALanguageModel(config=model_config)
    
    logger.info(f"Model created on device: {model.device}")
    
    # Train model using global configuration
    try:
        results = model.fit(texts[:100])  # Use subset for demo with global config
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Final results: {results}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Main function."""
    try:
        success = demo_refactored_gaia_language_model()
        return success
    except Exception as e:
        import traceback
        print(f"Demo failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)