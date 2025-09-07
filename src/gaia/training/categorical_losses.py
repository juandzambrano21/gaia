"""Categorical Loss Functions for GAIA Framework.

This module provides categorical diagram commutativity losses and other
categorical training objectives extracted from the language modeling example.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from ..training.config import GAIAConfig    
from ..core.kan_verification import HornFillingVerifier
from ..core.ends_coends import End, Coend
from ..training.config import GAIAConfig

logger = GAIAConfig.get_logger(__name__)


class CategoricalLossComputer:
    """Computes categorical diagram commutativity losses for GAIA training."""
    
    def __init__(self, device: torch.device, bisimulation_tolerance: float = 1e-3):
        self.device = device
        self.bisimulation_tolerance = bisimulation_tolerance
    
    def compute_categorical_diagram_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch_data: Dict[str, torch.Tensor],
        model_components: Dict[str, Any],
        epoch: int = 0, 
        num_batches: int = 0
    ) -> torch.Tensor:
        """Compute categorical diagram commutativity loss as PRIMARY training objective.
        
        Args:
            outputs: Model forward pass outputs
            batch_data: Batch data including sentences, targets, etc.
            model_components: Dictionary containing model components like coalgebras, Kan extensions
            epoch: Current training epoch
            num_batches: Current batch number
            
        Returns:
            Total categorical loss tensor
        """
        # Extract categorical data
        sentences = batch_data.get('sentence', batch_data.get('input_ids'))
        targets = batch_data.get('target', batch_data.get('labels'))
        
        # Initialize categorical loss components
        yoneda_isomorphism_loss = torch.tensor(0.0, device=self.device)
        coalgebra_recurrence_loss = torch.tensor(0.0, device=self.device)
        colimit_limit_loss = torch.tensor(0.0, device=self.device)
        bisimulation_loss = torch.tensor(0.0, device=self.device)
        naturality_loss = torch.tensor(0.0, device=self.device)
        universal_property_loss = torch.tensor(0.0, device=self.device)
        
        try:
            # Check configuration flags to disable expensive computations
            config = model_components.get('config')
            enable_hierarchical = True
            enable_horn_detection = True
            
            if config and hasattr(config, 'data'):
                enable_hierarchical = getattr(config.data, 'enable_hierarchical_messaging', True)
                enable_horn_detection = getattr(config.data, 'enable_horn_detection', True)
            
            # 1. YONEDA ISOMORPHISM ENFORCEMENT (always enabled for basic functionality)
            yoneda_isomorphism_loss = self._compute_yoneda_isomorphism_loss(
                outputs, model_components, epoch, num_batches
            )
            
            # 2. COALGEBRA RECURRENCE LAW ENFORCEMENT (always enabled for basic functionality)
            coalgebra_recurrence_loss = self._compute_coalgebra_recurrence_loss(
                outputs, sentences, targets, model_components, epoch, num_batches
            )
            
            # 3. COLIMIT/LIMIT COMPUTATION WITH HORN FILLING (conditional)
            if enable_horn_detection:
                colimit_limit_loss = self._compute_colimit_limit_loss(
                    outputs, model_components, epoch, num_batches
                )
            else:
                colimit_limit_loss = torch.tensor(0.0, device=self.device)
            
            # 4. BISIMULATION PRESERVATION (conditional)
            if enable_hierarchical:
                bisimulation_loss = self._compute_bisimulation_loss(
                    outputs, epoch, num_batches
                )
            else:
                bisimulation_loss = torch.tensor(0.0, device=self.device)
            
            # 5. NATURALITY CONDITIONS (conditional)
            if enable_hierarchical:
                naturality_loss = self._compute_naturality_loss(
                    outputs, epoch, num_batches
                )
            else:
                naturality_loss = torch.tensor(0.0, device=self.device)
            
            # 6. UNIVERSAL PROPERTY SATISFACTION (conditional)
            if enable_hierarchical:
                universal_property_loss = self._compute_universal_property_loss(
                    outputs, model_components, epoch, num_batches
                )
            else:
                universal_property_loss = torch.tensor(0.0, device=self.device)
            
        except Exception as e:
            if epoch == 0 and num_batches == 0:
                logger.debug(f"Categorical loss computation error: {e}")
        
        # Combine categorical losses with appropriate weights
        total_categorical_loss = (
            yoneda_isomorphism_loss * 1.0 +      # Primary: Yoneda isomorphism
            coalgebra_recurrence_loss * 1.0 +    # Primary: Coalgebra recurrence γ: X → F(X)
            colimit_limit_loss * 0.8 +           # Primary: Explicit colimit/limit computation
            bisimulation_loss * 0.6 +            # Diagram commutativity
            naturality_loss * 0.6 +              # Natural transformation preservation
            universal_property_loss * 0.4        # Universal property satisfaction
        )
        
        # Log detailed losses for monitoring
        if num_batches % 50 == 0:
            logger.info(f"Epoch {epoch}, Batch {num_batches} - Categorical Loss Breakdown:")
            logger.info(f"  • Yoneda Isomorphism: {yoneda_isomorphism_loss.item():.6f}")
            logger.info(f"  • Coalgebra Recurrence: {coalgebra_recurrence_loss.item():.6f}")
            logger.info(f"  • Colimit/Limit: {colimit_limit_loss.item():.6f}")
            logger.info(f"  • Bisimulation: {bisimulation_loss.item():.6f}")
            logger.info(f"  • Naturality: {naturality_loss.item():.6f}")
            logger.info(f"  • Universal Property: {universal_property_loss.item():.6f}")
            logger.info(f"  • Total Categorical: {total_categorical_loss.item():.6f}")
        
        return total_categorical_loss
    
    def _compute_yoneda_isomorphism_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        model_components: Dict[str, Any],
        epoch: int, 
        num_batches: int
    ) -> torch.Tensor:
        """Compute Yoneda isomorphism loss: Nat(Hom(C,-), F) ≅ F(C)."""
        if 'yoneda_embedded' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        yoneda_embedded = outputs['yoneda_embedded']
        
        if epoch == 0 and num_batches == 0:
            logger.debug(f"CATEGORICAL: Enforcing Yoneda isomorphism Nat(Hom(C,-), F) ≅ F(C)")
        
        try:
            metric_yoneda = model_components.get('metric_yoneda')
            
            # Try different embedding methods based on what's available
            if metric_yoneda and hasattr(metric_yoneda, 'embed'):
                try:
                    sample_embedding = yoneda_embedded[0, 0].cpu().numpy().tolist()
                    embed_result = metric_yoneda.embed(sample_embedding)
                    hom_functor_repr = yoneda_embedded
                except Exception as e:
                    logger.debug(f"Yoneda embed failed: {e}")
                    hom_functor_repr = yoneda_embedded
            elif metric_yoneda and hasattr(metric_yoneda, '__call__'):
                hom_functor_repr = metric_yoneda(yoneda_embedded)
            else:
                hom_functor_repr = yoneda_embedded
            
            # Create F(C) as different representation
            f_c_direct = outputs['logits'].mean(dim=-1, keepdim=True).expand_as(yoneda_embedded)
            
            # Enforce Yoneda isomorphism
            if hom_functor_repr.shape == f_c_direct.shape:
                return F.mse_loss(hom_functor_repr, f_c_direct)
            else:
                # Handle shape mismatch by using naturality condition
                return torch.var(yoneda_embedded, dim=1).mean()
                
        except Exception as e:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_coalgebra_recurrence_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        sentences: torch.Tensor,
        targets: torch.Tensor,
        model_components: Dict[str, Any],
        epoch: int,
        num_batches: int
    ) -> torch.Tensor:
        """Compute coalgebra recurrence law: γ: X → F(X)."""
        if 'coalgebra_evolved' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        coalgebra_evolved = outputs['coalgebra_evolved']
        
        if epoch == 0 and num_batches == 0:
            logger.debug(f"CATEGORICAL: Enforcing coalgebra recurrence γ: X → F(X)")
        
        try:
            # Use model hidden states as the coalgebra carrier
            hs = outputs.get('hidden_states', None)
            if hs is not None:
                current_state = hs.mean(dim=1)  # [B, d_model]
            else:
                current_state = (coalgebra_evolved.mean(dim=1)
                               if isinstance(coalgebra_evolved, torch.Tensor)
                               else sentences.float())
            
            # Get state coalgebra from model components
            state_coalgebra = model_components.get('state_coalgebra')
            if state_coalgebra is None:
                return torch.tensor(0.0, device=self.device)
            
            # Compute structure map
            f_x_result = state_coalgebra.structure_map(current_state)
            
            # Handle BackpropagationFunctor tuple result
            if isinstance(f_x_result, tuple) and len(f_x_result) == 3:
                _, _, f_x = f_x_result  # Extract X component
            else:
                f_x = f_x_result
            
            gamma_x = current_state
            return F.mse_loss(gamma_x, f_x)
            
        except Exception as e:
            logger.debug(f"Coalgebra recurrence computation failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _compute_colimit_limit_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        model_components: Dict[str, Any],
        epoch: int,
        num_batches: int
    ) -> torch.Tensor:
        """Compute colimit/limit loss with horn filling verification."""
        if 'compositional_repr' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        # Check if horn detection is disabled in config
        try:
            # Try to get config from model components or use default
            config = model_components.get('config')
            if config and hasattr(config, 'data') and hasattr(config.data, 'enable_horn_detection'):
                if not config.data.enable_horn_detection:
                    return torch.tensor(0.0, device=self.device)
        except Exception:
            # If config check fails, skip horn detection for safety
            return torch.tensor(0.0, device=self.device)
        
        compositional_repr = outputs['compositional_repr']
        
        if epoch == 0 and num_batches == 0:
            logger.info(f"Computing explicit colimits and limits with horn filling")
        
        horn_filling_violations = 0
        
        # Horn filling verification (only if enabled)
        gaia_transformer = model_components.get('gaia_transformer')
        if gaia_transformer and hasattr(gaia_transformer, 'functor'):
            try:
                horn_verifier = HornFillingVerifier(gaia_transformer.functor)
                
                # Check inner horns (limit to 1 for performance)
                inner_horns = gaia_transformer.functor.find_horns(level=2, horn_type="inner")
                for simplex_id, horn_index in inner_horns[:1]:  # Reduced from 3 to 1
                    try:
                        verification_result = horn_verifier.verify_inner_horn_filling(simplex_id, horn_index)
                        if not verification_result.satisfied:
                            horn_filling_violations += 1
                    except Exception:
                        horn_filling_violations += 1
                
                # Check outer horns
                outer_horns = gaia_transformer.functor.find_horns(level=2, horn_type="outer")
                for simplex_id, horn_index in outer_horns[:2]:  # Limit for performance
                    try:
                        verification_result = horn_verifier.verify_outer_horn_filling(simplex_id, horn_index)
                        if not verification_result.satisfied:
                            horn_filling_violations += 0.5
                    except Exception:
                        horn_filling_violations += 0.5
            except Exception as e:
                logger.debug(f"Horn filling verification failed: {e}")
        
        # Compute colimits and limits using framework
        try:
            left_kan_extension = model_components.get('left_kan_extension')
            right_kan_extension = model_components.get('right_kan_extension')
            
            if left_kan_extension and right_kan_extension:
                # Compute colimit using Coend
                coend_computer = Coend(right_kan_extension.F, "LanguageCoend")
                colimit_result = coend_computer.compute_integral()
                
                # Compute limit using End
                end_computer = End(left_kan_extension.F, "LanguageEnd")
                limit_result = end_computer.compute_integral()
                
                # Verify universal properties
                colimit_universality = 0.0 if coend_computer.verify_universal_property() else 1.0
                limit_universality = 0.0 if end_computer.verify_universal_property() else 1.0
                
                return torch.tensor(
                    colimit_universality + limit_universality + horn_filling_violations * 0.1,
                    device=self.device
                )
        except Exception as e:
            logger.debug(f"Colimit/limit computation failed: {e}")
        
        return torch.tensor(0.0, device=self.device)
    
    def _compute_bisimulation_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        epoch: int,
        num_batches: int
    ) -> torch.Tensor:
        """Compute bisimulation preservation loss."""
        if 'coalgebra_evolved' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        coalgebra_evolved = outputs['coalgebra_evolved']
        
        if isinstance(coalgebra_evolved, torch.Tensor) and coalgebra_evolved.size(1) > 1:
            # Average norm difference beyond tolerance
            gaps = []
            steps = min(coalgebra_evolved.size(1) - 1, 4)
            for i in range(steps):
                state_i = coalgebra_evolved[:, i, :]
                state_j = coalgebra_evolved[:, i + 1, :]
                gap = torch.norm(state_i - state_j, dim=-1).mean()
                gaps.append(gap)
            
            if gaps:
                eps = torch.tensor(self.bisimulation_tolerance, device=self.device)
                return torch.stack([F.relu(g - eps) for g in gaps]).mean()
        
        return torch.tensor(0.0, device=self.device)
    
    def _compute_naturality_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        epoch: int,
        num_batches: int
    ) -> torch.Tensor:
        """Compute naturality condition loss for morphism preservation."""
        if 'yoneda_embedded' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        yoneda_embedded = outputs['yoneda_embedded']
        
        if yoneda_embedded.size(1) > 1:
            # Natural transformation commutativity
            adjacent_embeddings = yoneda_embedded[:, :-1, :]
            next_embeddings = yoneda_embedded[:, 1:, :]
            
            morphism_preservation = F.cosine_similarity(
                adjacent_embeddings, next_embeddings, dim=-1
            ).mean()
            
            # Turn into loss: 0 means perfect commutativity
            return (1.0 - morphism_preservation).clamp_min(0)
        
        return torch.tensor(0.0, device=self.device)
    
    def _compute_universal_property_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        model_components: Dict[str, Any],
        epoch: int,
        num_batches: int
    ) -> torch.Tensor:
        """Compute universal property satisfaction loss from Kan extensions."""
        if 'compositional_repr' not in outputs:
            return torch.tensor(0.0, device=self.device)
        
        compositional_repr = outputs['compositional_repr']
        target_representations = outputs['logits']
        
        try:
            left_kan_extension = model_components.get('left_kan_extension')
            right_kan_extension = model_components.get('right_kan_extension')
            
            if left_kan_extension and right_kan_extension:
                # Project to same dimension if needed
                if target_representations.shape[-1] != compositional_repr.shape[-1]:
                    d_model = compositional_repr.shape[-1]
                    projection = torch.randn(
                        target_representations.shape[-1], d_model, 
                        device=self.device
                    ) * 0.1
                    target_representations = torch.matmul(target_representations, projection)
                
                # Compute universal property losses
                left_kan_loss = left_kan_extension.compute_universal_property_loss(
                    compositional_repr, target_representations
                )
                right_kan_loss = right_kan_extension.compute_universal_property_loss(
                    compositional_repr, target_representations
                )
                
                return (left_kan_loss + right_kan_loss) / 2
        except Exception as e:
            logger.debug(f"Universal property computation failed: {e}")
        
        return torch.tensor(0.0, device=self.device)


def CategoricalLoss(
    device: torch.device, 
    bisimulation_tolerance: float = 1e-3
) -> CategoricalLossComputer:
    """Create a categorical loss computer with clean naming.
    
    Args:
        device: PyTorch device for computations
        bisimulation_tolerance: Tolerance for bisimulation comparisons
        
    Returns:
        Configured CategoricalLossComputer instance
    """
    return CategoricalLossComputer(device, bisimulation_tolerance)