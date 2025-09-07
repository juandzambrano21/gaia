"""Model Initialization Utilities for GAIA Framework.

This module provides initialization utilities for GAIA models, extracted from
the language modeling example to improve code organization and reusability.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ..training.config import GAIAConfig
from ..core.fuzzy import (
    FuzzySimplicialSet, FuzzySimplicialFunctor, FuzzyCategory,
    create_discrete_fuzzy_set, create_gaussian_fuzzy_set
)
from ..data.fuzzy_encoding import FuzzyEncodingPipeline, UMAPConfig
from ..core.universal_coalgebras import (
    FCoalgebra, GenerativeCoalgebra, CoalgebraCategory, 
    BackpropagationFunctor, Bisimulation
)
from ..core.business_units import BusinessUnitHierarchy
from ..core.hierarchical_messaging import HierarchicalMessagePasser
from ..training.solvers.yoneda_proxy import MetricYonedaProxy
from ..core.metric_yoneda import YonedaEmbedding, MetricYonedaApplications
from ..core.kan_extensions import LeftKanExtension, RightKanExtension, GenerativeAICategory, NeuralFunctor
from ..core.ends_coends import End, Coend

logger = GAIAConfig.get_logger(__name__)


class GAIAModelInitializer:
    """Handles initialization of GAIA model components."""
    
    def __init__(self, device: torch.device, d_model: int, vocab_size: int):
        self.device = device
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.bisimulation_tolerance = 1e-3
    
    def initialize_token_fuzzy_sets(self) -> Dict[str, Any]:
        """Initialize fuzzy sets for token probability distributions."""
        logger.debug("Initializing token fuzzy sets...")
        
        # Create fuzzy sets for different token types
        content_word_fuzzy = create_gaussian_fuzzy_set(
            center=0.7, sigma=0.2, domain=list(range(100)), name="content_words"
        )
        function_word_fuzzy = create_gaussian_fuzzy_set(
            center=0.3, sigma=0.15, domain=list(range(100)), name="function_words"
        )
        punctuation_fuzzy = create_discrete_fuzzy_set(
            elements_with_membership={".": 0.1, ",": 0.2, "!": 0.05}, name="punctuation"
        )
        
        # Fuzzy category for token relationships
        token_category = FuzzyCategory(name="token_category")
        token_category.add_object(content_word_fuzzy)
        token_category.add_object(function_word_fuzzy)
        token_category.add_object(punctuation_fuzzy)
        
        logger.debug("Token fuzzy sets initialized")
        
        return {
            'content_word_fuzzy': content_word_fuzzy,
            'function_word_fuzzy': function_word_fuzzy,
            'punctuation_fuzzy': punctuation_fuzzy,
            'token_category': token_category
        }
    
    def initialize_language_simplicial_structure(self, token_category: FuzzyCategory) -> Dict[str, Any]:
        """Initialize fuzzy simplicial sets for language structure."""
        logger.debug("Initializing language simplicial structure...")
        
        # Multi-dimensional simplicial set for language hierarchy
        language_simplicial_set = FuzzySimplicialSet(
            dimension=4,  # 0:tokens, 1:phrases, 2:clauses, 3:sentences, 4:paragraphs
            name="language_structure"
        )
        
        # Simplicial functor for language composition
        composition_functor = FuzzySimplicialFunctor(
            name="compositional_semantics_functor",
            fuzzy_category=token_category
        )
        
        logger.debug("Language simplicial structure initialized")
        
        return {
            'language_simplicial_set': language_simplicial_set,
            'composition_functor': composition_functor
        }
    
    def initialize_fuzzy_encoding_pipeline(self) -> FuzzyEncodingPipeline:
        """Initialize UMAP-adapted fuzzy encoding pipeline."""
        logger.debug("Initializing fuzzy encoding pipeline...")
        
        # UMAP configuration for language encoding
        umap_config = UMAPConfig(
            n_neighbors=20,
            min_dist=0.05,
            metric='cosine'
        )
        
        # Fuzzy encoding pipeline (F1-F4 from Section 2.4)
        fuzzy_encoding_pipeline = FuzzyEncodingPipeline(config=umap_config)
        
        logger.debug("Fuzzy encoding pipeline initialized")
        return fuzzy_encoding_pipeline
    
    def initialize_generative_coalgebras(self, model_for_coalgebra) -> Dict[str, Any]:
        """Initialize universal coalgebras for generative dynamics."""
        logger.debug("Initializing generative coalgebras...")
        
        # Create pure coalgebra structure
        generative_coalgebra = GenerativeCoalgebra(model=model_for_coalgebra)
        
        # Store training components separately
        coalgebra_optimizer = torch.optim.AdamW(
            model_for_coalgebra.parameters(), 
            lr=1e-4, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        coalgebra_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        
        # Initialize empty coalgebra structures
        initial_state = torch.zeros(1, self.d_model, device=self.device)
        
        def state_structure_map(state: torch.Tensor) -> torch.Tensor:
            """Structure map for state evolution using backpropagation dynamics."""
            return state  # Identity until proper data is set
        
        state_coalgebra = FCoalgebra(
            carrier=initial_state,
            structure_map=state_structure_map,
            endofunctor=None  
        )
        
        # Coalgebra category for morphisms
        coalgebra_category = CoalgebraCategory()
        coalgebra_category.add_coalgebra("generative", generative_coalgebra)
        coalgebra_category.add_coalgebra("state", state_coalgebra)
        
        # Bisimulation for model equivalence
        initial_relation = [(
            torch.zeros(self.d_model, device=self.device), 
            torch.zeros(self.d_model, device=self.device)
        )]
        bisimulation = Bisimulation(
            coalgebra1=generative_coalgebra,
            coalgebra2=state_coalgebra,
            relation=initial_relation
        )
        
        logger.debug("Generative coalgebras initialized")
        
        return {
            'generative_coalgebra': generative_coalgebra,
            'coalgebra_optimizer': coalgebra_optimizer,
            'coalgebra_loss_fn': coalgebra_loss_fn,
            'state_coalgebra': state_coalgebra,
            'coalgebra_category': coalgebra_category,
            'bisimulation': bisimulation,
            'backprop_functor': None  # Will be initialized when training data is provided
        }
    
    def initialize_business_units(self, gaia_transformer) -> BusinessUnitHierarchy:
        """Initialize business unit hierarchy."""
        logger.debug("Initializing business unit hierarchy...")
        
        # Use the transformer's actual functor if available
        if hasattr(gaia_transformer, 'functor') and gaia_transformer.functor:
            business_hierarchy = BusinessUnitHierarchy(gaia_transformer.functor)
        else:
            # Fallback: create empty hierarchy with SimplicialFunctor
            from ..core.simplices import BasisRegistry
            from ..core.functor import SimplicialFunctor
            registry = BasisRegistry()
            functor = SimplicialFunctor("language_model", registry)
            business_hierarchy = BusinessUnitHierarchy(functor)
        
        logger.debug(f"Business unit hierarchy initialized: {business_hierarchy.total_units} units")
        return business_hierarchy
    
    def initialize_message_passing(self) -> HierarchicalMessagePasser:
        """Initialize hierarchical message passing."""
        logger.debug("Initializing hierarchical message passing...")
        
        message_passing = HierarchicalMessagePasser(
            max_dimension=3,  # Support up to 3-dimensional simplices
            device=self.device
        )
        
        # Add basic simplicial structure with local objectives
        # 0-simplices (vertices)
        vertex_id = message_passing.add_simplex(
            simplex_id="global_vertex",
            dimension=0,
            parameter_dim=self.d_model
        )
        
        # 1-simplices (edges)
        edge_id = message_passing.add_simplex(
            simplex_id="global_edge",
            dimension=1,
            parameter_dim=self.d_model,
            faces=["global_vertex"]
        )
        
        # 2-simplices (triangles)
        if message_passing.max_dimension >= 2:
            triangle_id = message_passing.add_simplex(
                simplex_id="global_triangle",
                dimension=2,
                parameter_dim=self.d_model,
                faces=["global_vertex", "global_edge"]
            )
        
        # Add local objectives with stability controls
        def vertex_objective(*face_params):
            """Local objective for vertex: parameter norm regularization."""
            vertex_params = message_passing.simplex_parameters["global_vertex"].parameters
            loss = 0.001 * torch.norm(vertex_params) ** 2
            return torch.clamp(loss, 0.0, 10.0)
        
        def edge_objective(*face_params):
            """Local objective for edge: coherence with vertex."""
            edge_params = message_passing.simplex_parameters["global_edge"].parameters
            if face_params:
                vertex_params = face_params[0]
            else:
                vertex_params = message_passing.simplex_parameters["global_vertex"].parameters
            loss = 0.001 * torch.norm(edge_params - vertex_params) ** 2
            return torch.clamp(loss, 0.0, 10.0)
        
        message_passing.add_local_objective("global_vertex", vertex_objective)
        message_passing.add_local_objective("global_edge", edge_objective)
        
        if message_passing.max_dimension >= 2:
            def triangle_objective(*face_params):
                """Local objective for triangle: coherence with faces."""
                triangle_params = message_passing.simplex_parameters["global_triangle"].parameters
                if face_params and len(face_params) > 1:
                    edge_params = face_params[1]
                else:
                    edge_params = message_passing.simplex_parameters["global_edge"].parameters
                return 0.001 * torch.norm(triangle_params - edge_params) ** 2
            
            message_passing.add_local_objective("global_triangle", triangle_objective)
        
        logger.debug("Hierarchical message passing initialized with local objectives")
        return message_passing
    
    def initialize_yoneda_embeddings(self) -> Dict[str, Any]:
        """Initialize Yoneda embeddings for representable functors."""
        logger.debug("Initializing Yoneda embeddings...")
        
        # Yoneda proxy for representable functor computations
        yoneda_proxy = MetricYonedaProxy(
            target_dim=self.d_model,
            num_probes=16,
            lr=1e-3,
            pretrain_steps=200,
            adaptive=True,
            use_direct_metric=True
        )
        
        logger.debug("Yoneda embeddings initialized")
        
        return {
            'yoneda_proxy': yoneda_proxy,
            'metric_yoneda': None,  # Will be initialized deferred
            '_yoneda_initialized': False
        }
    
    def initialize_yoneda_embeddings_deferred(
        self, 
        gaia_transformer, 
        yoneda_components: Dict[str, Any]
    ) -> None:
        """Initialize Yoneda embeddings after model is moved to device."""
        if yoneda_components['_yoneda_initialized']:
            return
        
        try:
            # Use actual transformer embeddings from a sample of vocabulary
            sample_token_ids = torch.randint(
                0, min(1000, self.vocab_size), (20,), device=self.device
            )
            sample_embeddings = []
            
            with torch.no_grad():
                embeddings_matrix = gaia_transformer.token_embedding(sample_token_ids)
                sample_embeddings = [embeddings_matrix[i] for i in range(embeddings_matrix.size(0))]
            
            language_metric_space = MetricYonedaApplications.create_neural_embedding_space(
                sample_embeddings
            )
            yoneda_components['metric_yoneda'] = YonedaEmbedding(language_metric_space)
            yoneda_components['_yoneda_initialized'] = True
            
        except Exception as e:
            logger.warning(f"Deferred Yoneda initialization failed: {e}")
            # Create a simple fallback Yoneda embedding
            dummy_embeddings = [torch.randn(self.d_model, device=self.device) for _ in range(20)]
            language_metric_space = MetricYonedaApplications.create_neural_embedding_space(
                dummy_embeddings
            )
            yoneda_components['metric_yoneda'] = YonedaEmbedding(language_metric_space)
            yoneda_components['_yoneda_initialized'] = True
    
    def initialize_kan_extensions(self) -> Dict[str, Any]:
        """Initialize Kan extensions for compositional understanding."""
        logger.debug("Initializing Kan extensions...")
        
        # Create categories for language modeling
        syntax_category = GenerativeAICategory("Syntax")
        semantics_category = GenerativeAICategory("Semantics")
        pragmatics_category = GenerativeAICategory("Pragmatics")
        
        # Add objects to categories
        syntax_category.add_object("tokens")
        semantics_category.add_object("meanings")
        pragmatics_category.add_object("contexts")
        
        # Create functors for compositional understanding
        syntax_to_semantics = NeuralFunctor(syntax_category, semantics_category)
        syntax_to_pragmatics = NeuralFunctor(syntax_category, pragmatics_category)
        
        # Left Kan extension for compositional semantics
        left_kan_extension = LeftKanExtension(
            syntax_to_semantics,     # F: functor to extend
            syntax_to_pragmatics,    # K: extension direction
            "LanguageLeftKan"        # name
        )
        
        # Right Kan extension for pragmatic inference
        semantics_to_syntax = NeuralFunctor(semantics_category, syntax_category)
        semantics_to_pragmatics = NeuralFunctor(semantics_category, pragmatics_category)
        
        right_kan_extension = RightKanExtension(
            semantics_to_syntax,     # F: functor to extend
            semantics_to_pragmatics, # K: extension direction
            "LanguageRightKan"       # name
        )
        
        logger.debug("Kan extensions initialized")
        
        return {
            'left_kan_extension': left_kan_extension,
            'right_kan_extension': right_kan_extension
        }
    
    def initialize_ends_coends(self, kan_extensions: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize ends and coends for natural transformations."""
        logger.debug("Initializing ends and coends...")
        
        # End for universal properties in language understanding
        end_computation = End(
            functor=kan_extensions['left_kan_extension'].F,
            name="universal_language_understanding"
        )
        
        # Coend for colimits in semantic composition
        coend_computation = Coend(
            functor=kan_extensions['right_kan_extension'].F,
            name="semantic_composition_colimit"
        )
        
        logger.debug("Ends and coends initialized")
        
        return {
            'end_computation': end_computation,
            'coend_computation': coend_computation
        }
    
    def initialize_mps_tensors(self, model: nn.Module) -> None:
        """Initialize all tensors properly for MPS device."""
        with torch.no_grad():
            for param in model.parameters():
                if param.device != self.device:
                    param.data = param.data.to(self.device)
                # Force contiguous memory layout for MPS
                param.data = param.data.contiguous()
            
            # Special handling for embedding layers
            for module in model.modules():
                if isinstance(module, nn.Embedding):
                    # Ensure embedding weights are properly initialized on MPS
                    module.weight.data = module.weight.data.to(
                        dtype=torch.float32, device=self.device
                    ).contiguous()
                    # Reinitialize with proper distribution
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        logger.info(f"Tensor initialization completed for {self.device}")
    
    def log_framework_components(self, components: Dict[str, Any]) -> None:
        """Log all initialized GAIA framework components."""
        logger.info("GAIA Language Model Components Summary:")
        
        fuzzy_sets_count = len([
            components.get('content_word_fuzzy'),
            components.get('function_word_fuzzy'), 
            components.get('punctuation_fuzzy')
        ])
        logger.info(f"  • Token Fuzzy Sets: {fuzzy_sets_count}")
        
        if 'language_simplicial_set' in components:
            logger.info(f"  • Language Simplicial Dimension: {components['language_simplicial_set'].dimension}")
        
        if 'coalgebra_category' in components:
            logger.info(f"  • Coalgebras: {len(components['coalgebra_category'].objects)}")
        
        if 'business_hierarchy' in components:
            logger.info(f"  • Business Units: {components['business_hierarchy'].total_units}")
        
        if 'message_passing' in components:
            logger.info(f"  • Message Passing Max Dimension: {components['message_passing'].max_dimension}")
        
        if 'yoneda_proxy' in components:
            logger.info(f"  • Yoneda Embedding Dim: {components['yoneda_proxy'].target_dim}")
        
        logger.info(f"  • Kan Extensions: Left + Right")
        logger.info(f"  • Ends/Coends: Universal + Colimit")


def ModelInit(
    device: torch.device, 
    d_model: int, 
    vocab_size: int
) -> GAIAModelInitializer:
    """Create a GAIA model initializer with clean naming.
    
    Args:
        device: PyTorch device for computations
        d_model: Model dimension
        vocab_size: Vocabulary size
        
    Returns:
        Configured GAIAModelInitializer instance
    """
    return GAIAModelInitializer(device, d_model, vocab_size)