#!/usr/bin/env python3
"""
GAIA Framework Production Training Example
==========================================

Demonstrates complete GAIA framework usage with:
- Proper GAIA logging system
- Simplicial dataset implementation
- Full categorical/simplicial data loading
- Production-ready training pipeline
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# GAIA Framework Imports
from gaia.training.config import GAIAConfig, GAIALanguageModelConfig, TrainingConfig
from gaia.models import GAIALanguageModel
from gaia.data import create_gaia_dataset
from gaia.training.engine import CheckpointManager

# Initialize GAIA framework logging
GAIAConfig.setup_logging(enable_ultra_verbose=True)
logger = GAIAConfig.get_logger('gaia_production_training')


def create_config() -> Dict[str, Any]:
    """This is a showcase of all configs in framework. for further details explore training.config.py"""
    
    model_config = GAIALanguageModelConfig(
        # Core Architecture
        vocab_size=1000,
        hidden_dim=256,
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_seq_length=128,
        seq_len=128,
        
        # GAIA Categorical Components
        enable_fuzzy_components=True,
        enable_simplicial_structure=True,
        enable_coalgebras=True,
        enable_business_hierarchy=True,
        enable_message_passing=True,
        enable_yoneda_embeddings=True,
        enable_kan_extensions=True,
        enable_ends_coends=True,
        
        # UMAP Configuration for Fuzzy Encoding Pipeline (F1-F4)
        # According to GAIA paper: "UMAP forms a Kan complex" - critical for categorical structure
        umap_n_neighbors=21,           # Number of nearest neighbors for k-NN (F1)
        umap_min_dist=0.05,           # Minimum distance for local structure preservation
        umap_spread=1.2,              # Spread parameter for global structure
        umap_metric='cosine',         # Distance metric - cosine for language embeddings
        umap_local_connectivity=1.5,  # Local connectivity parameter
        umap_bandwidth=1.0,           # Bandwidth for fuzzy set membership
        umap_n_components=2,          # Dimensionality for visualization (if needed)
        umap_random_state=42,         # Reproducibility seed
        
        # Production parameters
        hierarchical_steps=4,
        convergence_threshold=1e-4,
        max_hierarchical_steps=8,
        
        # Training Defaults
        default_epochs=5,
        default_learning_rate=1e-3,
        default_validation_split=0.2,
        gradient_clip_norm=1.0,
        
        # Model Metadata
        model_type="gaia_language_model_production",
        version="2.1.0"
    )
    
    # Production Training Configuration with UMAP Integration
    training_config = TrainingConfig(
        epochs=5,
        eval_frequency=25,
        save_frequency=50,
        
        # Logging and monitoring
        log_level="DEBUG",
        log_frequency=10,
        use_tensorboard=True,
        
        # Checkpointing
        checkpoint_dir="checkpoints",
        save_top_k=3,
        monitor_metric="val_loss",
        monitor_mode="min",
        
        # Early stopping
        early_stopping=True,
        patience=3,
        
        # GAIA Training Features
        categorical_training=True,
        hierarchical_learning=True
    )
    
    # Configure UMAP parameters in DataConfig for fuzzy encoding pipeline
    training_config.data.n_neighbors = 21        # k-NN parameter for F1 step
    training_config.data.min_dist = 0.05         # Local structure preservation
    training_config.data.spread = 1.2            # Global structure parameter
    training_config.data.metric = 'cosine'       # Distance metric for language data
    training_config.data.local_connectivity = 1.5 # Local connectivity
    training_config.data.bandwidth = 1.0         # Fuzzy membership bandwidth
    training_config.data.n_components = 2        # Dimensionality for UMAP
    training_config.data.random_seed = 42        # Reproducibility
    
    return {
        'model_config': model_config,
        'training_config': training_config
    }

def create_gaia_dataset():
    """Create dataset using proper GAIA framework data loading utilities"""
    logger.info("🔧 Creating dataset using GAIA framework data utilities...")
    
    try:
        # Import GAIA data utilities - proper framework approach
        from gaia.data.dataset import Dataset
        from gaia.data.loaders import GAIADataManager, DatasetConfig, DataLoaderConfig
        
        logger.info("📚 Loading TinyStories dataset using GAIA Dataset() function...")
        
        # Use GAIA's built-in Dataset() function that handles TinyStories loading
        texts = Dataset()
        logger.info(f"✅ Loaded {len(texts)} text samples using GAIA framework")
        
        # Split into train/validation
        split_idx = int(0.9 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        logger.info(f"📊 Split dataset: {len(train_texts)} train, {len(val_texts)} validation samples")
        
        return {
            'train_texts': train_texts,
            'val_texts': val_texts,
            'total_samples': len(texts)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create GAIA dataset: {e}")
        logger.info("🔄 Using emergency fallback dataset...")
        
        # Emergency fallback with TinyStories-style content
        emergency_texts = [
            "Once upon a time, there was a little cat. The cat liked to play.",
            "A boy named Sam went to the park. He saw many birds there.",
            "The sun was bright and warm. All the children were happy.",
            "There was a magic forest where animals could talk.",
            "A little girl found a treasure chest in her backyard."
        ] * 200
        
        split_idx = int(0.9 * len(emergency_texts))
        return {
            'train_texts': emergency_texts[:split_idx],
            'val_texts': emergency_texts[split_idx:],
            'total_samples': len(emergency_texts)
        }

def run_training():
    """Execute production-ready training with full GAIA framework integration."""
    logger = GAIAConfig.get_logger('gaia_production_training')
    
    logger.info("STARTING GAIA LANGUAGE MODEL TRAINING WITH SIMPLICIAL DATASET")
    logger.info("=" * 60)
    
    try:
        # Create production configuration
        configs = create_config()
        model_config = configs['model_config']
        training_config = configs['training_config']
        
        # Create dataset using GAIA framework
        dataset_info = create_gaia_dataset()

        # Build simple tokenizer from dataset first
        logger.info("📝 Building simple tokenizer from dataset...")
        tokenizer_texts = dataset_info['train_texts'] + dataset_info['val_texts']
        
        # Create simple tokenizer
        from gaia.data.tokenizer import SimpleTokenizer
        simple_tokenizer = SimpleTokenizer(vocab_size=15000)
        simple_tokenizer.build_vocab(tokenizer_texts)
        logger.info(f"📖 Simple tokenizer built with vocab size: {len(simple_tokenizer.word_to_id)}")
        
        # Use GAIA data management with proper DataLoaders
        try:
            from gaia.data.loaders import DataLoaders
            
            # Create GAIA DataLoaders with proper configuration
            logger.info("📊 Creating GAIA DataLoaders...")
            train_loader, val_loader = DataLoaders(
                 train_texts=dataset_info['train_texts'],
                 val_texts=dataset_info['val_texts'],
                 tokenizer=simple_tokenizer,  # Use simple tokenizer
                 batch_size=4,  # Default batch size
                 max_seq_length=128,  # Match model configuration
                 apply_yoneda=True,
                 apply_simplicial=True
             )
            
            logger.info(f"✅ Created GAIA DataLoaders:")
            logger.info(f"   📈 Train batches: {len(train_loader)}")
            logger.info(f"   📉 Val batches: {len(val_loader)}")
            
            # Store loaders for training
            data_loaders = {'train': train_loader, 'val': val_loader}
            
        except Exception as e:
            logger.warning(f"GAIA DataLoaders creation failed: {e}")
            logger.info("🔄 Using dataset info directly...")
            
            # Use dataset info directly
            train_texts = dataset_info['train_texts']
            val_texts = dataset_info['val_texts']
            data_loaders = None
            
            logger.info(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
        
        # Initialize GAIA model with full categorical/simplicial support
        logger.info("🚀 Initializing GAIA Language Model with production configuration...")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {device}")
        
        model = GAIALanguageModel(model_config, device=str(device))
        param_count = sum(p.numel() for p in model.parameters())

        model.tokenizer = simple_tokenizer
        vocab_size = len(model.tokenizer.word_to_id)
        logger.info(f"📖 Using simple tokenizer with vocab size: {vocab_size}")
        
        # Update model vocab size if needed
        if vocab_size != model.vocab_size:
            model.vocab_size = vocab_size
            
            # Recreate transformer with correct vocab size
            from gaia.models.gaia_transformer import GAIATransformer
            model.gaia_transformer = GAIATransformer(
                vocab_size=vocab_size,
                d_model=model.hidden_dim,
                num_heads=model_config.num_heads,
                num_layers=model_config.num_layers,
                max_seq_length=model.max_seq_length
            ).to(model.device)
            
            # Reinitialize components that depend on vocab size
            from gaia.models.initialization import ModelInit
            model.initializer = ModelInit(d_model=model.hidden_dim, vocab_size=vocab_size, device=model.device)
            model._initialize_all_components()
        
        # Setup checkpoint manager for production training
        try:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(training_config.checkpoint_dir),
                max_checkpoints=training_config.save_top_k,
                monitor_metric=training_config.monitor_metric,
                higher_is_better=(training_config.monitor_mode == "max")
            )
            logger.info(f"💾 Checkpoint manager initialized: {checkpoint_manager.checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Checkpoint manager setup failed: {e}")
            checkpoint_manager = None
        
        # Production training with full GAIA framework integration
        logger.info("🎓 Starting production training with GAIA framework...")
        try:
            if data_loaders is not None:
                # Use GAIA training with data loaders
                logger.info("Training with GAIA data loaders...")
                results = model.fit_with_loaders(
                     train_loader=data_loaders['train'],
                     val_loader=data_loaders['val'],
                     epochs=training_config.epochs,
                     learning_rate=1e-3,
                     checkpoint_manager=checkpoint_manager
                 )
            else:
                logger.info("Training with dataset texts...")
                # Combine train and val texts for the dataset parameter
                all_texts = dataset_info['train_texts'] + dataset_info['val_texts']
                results = model.fit(
                     dataset=all_texts,
                     epochs=training_config.epochs,
                     batch_size=4,  # Default batch size
                     learning_rate=1e-3,  # Default learning rate
                     validation_split=0.1  
                 )
            
            logger.info("✅ Production training completed successfully!")
            logger.info(f"📈 Training results: {results}")
            
            # Save final checkpoint
            if checkpoint_manager:
                final_checkpoint = checkpoint_manager.save_checkpoint(
                    model=model,
                    epoch=training_config.epochs,
                    metrics=results
                )
                logger.info(f"💾 Final checkpoint saved: {final_checkpoint}")
            
        except Exception as train_error:
            logger.error(f"❌ Language model training failed: {train_error}")
            logger.info("🔄 Attempting emergency checkpoint save...")
            
            try:
                emergency_path = model.save_checkpoint()
                logger.info(f"💾 Emergency checkpoint saved: {emergency_path}")
            except Exception as checkpoint_error:
                logger.error(f"❌ Emergency checkpoint save failed: {checkpoint_error}")
        
        try:
            # Test with prompts
            test_prompts = [
                "Once upon a time",
                "In a magical kingdom",
                "The brave explorer"
            ]
            
            for prompt in test_prompts:
                generated_text = model.generate(
                    input_text=prompt,
                    max_length=20,
                    temperature=0.8
                )
                logger.info(f"📝 Prompt: '{prompt}' → Generated: '{generated_text}'")
            
        except Exception as gen_error:
            logger.error(f"❌ Text generation failed: {gen_error}")
            logger.info("🔄 Trying basic generation...")
            try:
                basic_gen = model.generate("Hello", max_length=10)
                logger.info(f"📝 Basic generation: {basic_gen}")
            except Exception as basic_error:
                logger.error(f"❌ Basic generation also failed: {basic_error}")
        
        logger.info("🎯 GAIA PRODUCTION TRAINING COMPLETED!")
        return model
        
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {e}")
        import traceback
        logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
        
        # Force flush on error
        try:
            import logging
            for handler in logging.getLogger().handlers:
                handler.flush()
        except Exception:
            pass
        
        raise e


if __name__ == "__main__":
    print("Starting GAIA LLM training.")
    print("=" * 70)
    
    # Initialize GAIA framework logging system
    GAIAConfig.setup_logging()
    logger = GAIAConfig.get_logger('gaia_production_main')
        
    try:
        # Execute production training with full GAIA framework
        model = run_training()
                
        # Display final model capabilities
        if hasattr(model, 'get_model_stats'):
            stats = model.get_model_stats()
            logger.info(f"📊 Final model statistics: {stats}")
            
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            logger.info(f"🔺 Model info: {info}")
                
    except Exception as e:
        logger.error(f"💥 PRODUCTION TRAINING FAILED: {e}")
        print(f"\n💥 Production training failed: {e}")
        
        # Comprehensive error logging
        try:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Complete error traceback:\n{full_traceback}")
            
            if "DataLoaders" in str(e):
                logger.error("Data loading failed - check GAIA data utilities")
            elif "Dataset" in str(e):
                logger.error("Dataset creation failed - check GAIA dataset modules")
            elif "GAIALanguageModel" in str(e):
                logger.error("Model initialization failed - check GAIA model configuration")
            
        except Exception as log_error:
            print(f"Additional logging error: {log_error}")
    
    finally:
        logger.info(" GAIA LLM Training session ended")
        print("\n GAIA Framework Training Session Complete!")