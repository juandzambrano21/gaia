"""
Module: hierarchical_message_passing
Implements hierarchical message passing for GAIA framework.

Following Section 3.4 of the theoretical framework, this implements:
1. Parameters θ_σ specific to each simplex σ
2. Local objective functions L_σ(θ_{d_0σ},...,θ_{d_nσ}) for each simplex
3. Update rules combining gradient information from (n+1) faces
4. Instructions from degeneracies
5. Hierarchical update scheme with information percolation

This is critical for implementing the true simplicial message passing
where information flows up and down the simplicial complex.
"""

import uuid
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from ..core.simplices import SimplicialObject
from ..core.functor import SimplicialFunctor, MapType
from ..core.coalgebras import FCoalgebra, BackpropagationEndofunctor

logger = logging.getLogger(__name__)


@dataclass
class SimplexParameters:
    """
    Parameters θ_σ specific to each simplex σ.
    
    Following Section 3.4, each simplex maintains its own parameter vector
    that is updated based on information from its faces and degeneracies.
    """
    simplex_id: uuid.UUID
    simplex_name: str
    level: int
    parameters: torch.Tensor
    gradients: Optional[torch.Tensor] = None
    momentum: Optional[torch.Tensor] = None
    learning_rate: float = 0.01
    last_update_step: int = 0
    
    def __post_init__(self):
        """Initialize gradients and momentum if not provided."""
        if self.gradients is None:
            self.gradients = torch.zeros_like(self.parameters)
        if self.momentum is None:
            self.momentum = torch.zeros_like(self.parameters)
    
    def update_parameters(self, gradient: torch.Tensor, momentum_factor: float = 0.9):
        """Update parameters using gradient and momentum."""
        self.gradients = gradient
        self.momentum = momentum_factor * self.momentum + (1 - momentum_factor) * gradient
        self.parameters = self.parameters - self.learning_rate * self.momentum
        self.last_update_step += 1
    
    def get_parameter_norm(self) -> float:
        """Get L2 norm of parameters."""
        return torch.norm(self.parameters).item()
    
    def get_gradient_norm(self) -> float:
        """Get L2 norm of gradients."""
        return torch.norm(self.gradients).item()


class LocalObjectiveFunction:
    """
    Local objective function L_σ(θ_{d_0σ},...,θ_{d_nσ}) for each simplex.
    
    This combines information from all faces of the simplex to compute
    a local loss that guides parameter updates.
    """
    
    def __init__(self, simplex_id: uuid.UUID, simplex_level: int, 
                 face_ids: List[uuid.UUID], name: str = ""):
        self.simplex_id = simplex_id
        self.simplex_level = simplex_level
        self.face_ids = face_ids
        self.name = name or f"L_{simplex_id}"
        self.id = uuid.uuid4()
        
        # Weights for combining face contributions
        self.face_weights = torch.ones(len(face_ids)) / len(face_ids) if face_ids else torch.tensor([])
    
    def compute_loss(self, simplex_params: SimplexParameters, 
                    face_params: List[SimplexParameters],
                    coherence_weight: float = 1.0,
                    consistency_weight: float = 1.0) -> torch.Tensor:
        """
        Compute local objective L_σ(θ_{d_0σ},...,θ_{d_nσ}).
        
        Args:
            simplex_params: Parameters of the simplex
            face_params: Parameters of all faces
            coherence_weight: Weight for coherence term
            consistency_weight: Weight for consistency term
            
        Returns:
            Local loss value
        """
        if not face_params:
            # No faces - return regularization loss
            return 0.5 * torch.norm(simplex_params.parameters) ** 2
        
        total_loss = torch.tensor(0.0, device=simplex_params.parameters.device)
        
        # Coherence loss: simplex parameters should be consistent with face parameters
        coherence_loss = torch.tensor(0.0, device=simplex_params.parameters.device)
        for i, face_param in enumerate(face_params):
            if i < len(self.face_weights):
                weight = self.face_weights[i]
                # Measure consistency between simplex and face parameters
                if simplex_params.parameters.shape == face_param.parameters.shape:
                    coherence_loss += weight * torch.norm(simplex_params.parameters - face_param.parameters) ** 2
                else:
                    # Handle different parameter shapes by projecting
                    min_dim = min(simplex_params.parameters.numel(), face_param.parameters.numel())
                    simplex_flat = simplex_params.parameters.flatten()[:min_dim]
                    face_flat = face_param.parameters.flatten()[:min_dim]
                    coherence_loss += weight * torch.norm(simplex_flat - face_flat) ** 2
        
        total_loss += coherence_weight * coherence_loss
        
        # Consistency loss: faces should be mutually consistent
        consistency_loss = torch.tensor(0.0, device=simplex_params.parameters.device)
        for i in range(len(face_params)):
            for j in range(i + 1, len(face_params)):
                face_i, face_j = face_params[i], face_params[j]
                if face_i.parameters.shape == face_j.parameters.shape:
                    consistency_loss += torch.norm(face_i.parameters - face_j.parameters) ** 2
                else:
                    # Handle different shapes
                    min_dim = min(face_i.parameters.numel(), face_j.parameters.numel())
                    face_i_flat = face_i.parameters.flatten()[:min_dim]
                    face_j_flat = face_j.parameters.flatten()[:min_dim]
                    consistency_loss += torch.norm(face_i_flat - face_j_flat) ** 2
        
        if len(face_params) > 1:
            consistency_loss /= (len(face_params) * (len(face_params) - 1) / 2)
        
        total_loss += consistency_weight * consistency_loss
        
        return total_loss
    
    def compute_gradients(self, simplex_params: SimplexParameters,
                         face_params: List[SimplexParameters]) -> torch.Tensor:
        """
        Compute gradients of local objective with respect to simplex parameters.
        
        Returns:
            Gradient tensor
        """
        # Enable gradient computation
        simplex_params.parameters.requires_grad_(True)
        
        # Compute loss
        loss = self.compute_loss(simplex_params, face_params)
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Check if gradients were computed
        if simplex_params.parameters.grad is not None:
            gradients = simplex_params.parameters.grad.clone()
            # Clear gradients
            simplex_params.parameters.grad.zero_()
        else:
            # If no gradients, compute them manually using autograd
            gradients = torch.autograd.grad(loss, simplex_params.parameters, 
                                          retain_graph=True, create_graph=False)[0]
        
        return gradients


class DegeneracyInstruction:
    """
    Instructions from degeneracies that include σ as a face.
    
    Degeneracies provide "top-down" information flow in the hierarchy.
    """
    
    def __init__(self, source_simplex_id: uuid.UUID, target_simplex_id: uuid.UUID,
                 degeneracy_index: int, instruction_type: str = "parameter_guidance"):
        self.source_simplex_id = source_simplex_id
        self.target_simplex_id = target_simplex_id
        self.degeneracy_index = degeneracy_index
        self.instruction_type = instruction_type
        self.id = uuid.uuid4()
    
    def generate_instruction(self, source_params: SimplexParameters,
                           target_params: SimplexParameters) -> torch.Tensor:
        """
        Generate instruction from higher-dimensional simplex to lower-dimensional one.
        
        Args:
            source_params: Parameters of higher-dimensional simplex
            target_params: Parameters of target simplex
            
        Returns:
            Instruction tensor (gradient-like update)
        """
        if self.instruction_type == "parameter_guidance":
            # Guide target parameters toward source parameters
            if source_params.parameters.shape == target_params.parameters.shape:
                return 0.1 * (source_params.parameters - target_params.parameters)
            else:
                # Handle shape mismatch by projection
                min_dim = min(source_params.parameters.numel(), target_params.parameters.numel())
                source_flat = source_params.parameters.flatten()[:min_dim]
                target_flat = target_params.parameters.flatten()[:min_dim]
                instruction_flat = 0.1 * (source_flat - target_flat)
                
                # Reshape back to target shape
                instruction = torch.zeros_like(target_params.parameters)
                instruction.flatten()[:min_dim] = instruction_flat
                return instruction
        
        elif self.instruction_type == "momentum_transfer":
            # Transfer momentum from source to target
            if hasattr(source_params, 'momentum') and source_params.momentum is not None:
                if source_params.momentum.shape == target_params.parameters.shape:
                    return 0.05 * source_params.momentum
                else:
                    # Handle shape mismatch
                    min_dim = min(source_params.momentum.numel(), target_params.parameters.numel())
                    momentum_flat = source_params.momentum.flatten()[:min_dim]
                    instruction = torch.zeros_like(target_params.parameters)
                    instruction.flatten()[:min_dim] = 0.05 * momentum_flat
                    return instruction
        
        # Default: no instruction
        return torch.zeros_like(target_params.parameters)


class HierarchicalMessagePassingSystem:
    """
    Complete hierarchical message passing system for GAIA.
    
    Implements the full Section 3.4 framework with:
    - Per-simplex parameters θ_σ
    - Local objective functions L_σ
    - Gradient combination from faces
    - Degeneracy instructions
    - Hierarchical update scheme
    """
    
    def __init__(self, simplicial_functor: SimplicialFunctor, 
                 parameter_dim: int = 64,
                 learning_rate: float = 0.01,
                 momentum_factor: float = 0.9):
        self.simplicial_functor = simplicial_functor
        self.parameter_dim = parameter_dim
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        
        # Core data structures
        self.simplex_parameters: Dict[uuid.UUID, SimplexParameters] = {}
        self.local_objectives: Dict[uuid.UUID, LocalObjectiveFunction] = {}
        self.degeneracy_instructions: Dict[Tuple[uuid.UUID, uuid.UUID], DegeneracyInstruction] = {}
        
        # Message passing state
        self.message_queue: List[Tuple[uuid.UUID, torch.Tensor, str]] = []  # (target_id, message, type)
        self.update_order: List[List[uuid.UUID]] = []  # Level-wise update order
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize parameters and objectives for all simplices."""
        # Initialize parameters for each simplex
        for simplex_id, simplex in self.simplicial_functor.registry.items():
            self._initialize_simplex_parameters(simplex_id, simplex)
        
        # Create local objective functions
        for simplex_id, simplex in self.simplicial_functor.registry.items():
            self._create_local_objective(simplex_id, simplex)
        
        # Set up degeneracy instructions
        self._setup_degeneracy_instructions()
        
        # Determine update order (bottom-up by level)
        self._compute_update_order()
    
    def _initialize_simplex_parameters(self, simplex_id: uuid.UUID, simplex: SimplicialObject):
        """Initialize parameters θ_σ for a simplex."""
        # Parameter dimension may depend on simplex level
        param_dim = self.parameter_dim * (simplex.level + 1)  # Higher levels have more parameters
        
        # Initialize parameters with Xavier initialization
        parameters = torch.randn(param_dim) * np.sqrt(2.0 / param_dim)
        
        self.simplex_parameters[simplex_id] = SimplexParameters(
            simplex_id=simplex_id,
            simplex_name=simplex.name,
            level=simplex.level,
            parameters=parameters,
            learning_rate=self.learning_rate
        )
    
    def _create_local_objective(self, simplex_id: uuid.UUID, simplex: SimplicialObject):
        """Create local objective function L_σ for a simplex."""
        # Find all faces of this simplex
        face_ids = []
        for i in range(simplex.level + 1):
            try:
                face = self.simplicial_functor.face(i, simplex_id)
                face_ids.append(face.id)
            except Exception:
                # Face not defined (horn)
                continue
        
        self.local_objectives[simplex_id] = LocalObjectiveFunction(
            simplex_id=simplex_id,
            simplex_level=simplex.level,
            face_ids=face_ids,
            name=f"L_{simplex.name}"
        )
    
    def _setup_degeneracy_instructions(self):
        """Set up degeneracy instructions for top-down information flow."""
        for (source_id, deg_index, map_type), target_id in self.simplicial_functor.maps.items():
            if map_type == MapType.DEGENERACY:
                instruction = DegeneracyInstruction(
                    source_simplex_id=source_id,
                    target_simplex_id=target_id,
                    degeneracy_index=deg_index
                )
                self.degeneracy_instructions[(source_id, target_id)] = instruction
    
    def _compute_update_order(self):
        """Compute level-wise update order for hierarchical updates."""
        # Group simplices by level
        levels = defaultdict(list)
        for simplex_id, simplex in self.simplicial_functor.registry.items():
            levels[simplex.level].append(simplex_id)
        
        # Create update order: bottom-up (level 0 first)
        self.update_order = []
        for level in sorted(levels.keys()):
            self.update_order.append(levels[level])
    
    def compute_face_gradients(self, simplex_id: uuid.UUID) -> torch.Tensor:
        """
        Compute gradients by combining information from (n+1) faces.
        
        This implements the core gradient combination from face maps
        as specified in Section 3.4.
        """
        if simplex_id not in self.simplex_parameters:
            return torch.zeros(self.parameter_dim)
        
        simplex_params = self.simplex_parameters[simplex_id]
        local_objective = self.local_objectives[simplex_id]
        
        # Get face parameters
        face_params = []
        for face_id in local_objective.face_ids:
            if face_id in self.simplex_parameters:
                face_params.append(self.simplex_parameters[face_id])
        
        # Compute gradients from local objective
        if face_params:
            gradients = local_objective.compute_gradients(simplex_params, face_params)
        else:
            # No faces - use regularization gradient
            gradients = simplex_params.parameters.clone()
        
        return gradients
    
    def apply_degeneracy_instructions(self, simplex_id: uuid.UUID) -> torch.Tensor:
        """
        Apply instructions from degeneracies that include σ as a face.
        
        This implements top-down information flow from higher-dimensional simplices.
        """
        total_instruction = torch.zeros_like(self.simplex_parameters[simplex_id].parameters)
        instruction_count = 0
        
        # Find all degeneracies that target this simplex
        for (source_id, target_id), instruction in self.degeneracy_instructions.items():
            if target_id == simplex_id and source_id in self.simplex_parameters:
                source_params = self.simplex_parameters[source_id]
                target_params = self.simplex_parameters[simplex_id]
                
                instruction_tensor = instruction.generate_instruction(source_params, target_params)
                total_instruction += instruction_tensor
                instruction_count += 1
        
        # Average instructions if multiple sources
        if instruction_count > 0:
            total_instruction /= instruction_count
        
        return total_instruction
    
    def hierarchical_update_step(self, global_loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Perform one step of hierarchical message passing updates.
        
        This implements the complete hierarchical update scheme where
        information percolates up and down the simplicial complex.
        
        Args:
            global_loss: Optional global loss for additional gradient information
            
        Returns:
            Dictionary with update statistics
        """
        update_stats = {
            "total_updates": 0,
            "average_gradient_norm": 0.0,
            "average_parameter_norm": 0.0,
            "coherence_loss": 0.0,
            "instruction_strength": 0.0
        }
        
        total_gradient_norm = 0.0
        total_parameter_norm = 0.0
        total_coherence_loss = 0.0
        total_instruction_strength = 0.0
        
        # Update in level order (bottom-up)
        for level_simplices in self.update_order:
            for simplex_id in level_simplices:
                if simplex_id not in self.simplex_parameters:
                    continue
                
                simplex_params = self.simplex_parameters[simplex_id]
                
                # 1. Compute gradients from faces
                face_gradients = self.compute_face_gradients(simplex_id)
                
                # 2. Apply degeneracy instructions
                degeneracy_instruction = self.apply_degeneracy_instructions(simplex_id)
                
                # 3. Combine gradients and instructions
                combined_gradient = face_gradients + 0.1 * degeneracy_instruction
                
                # 4. Add global loss gradient if available
                if global_loss is not None and simplex_params.parameters.requires_grad:
                    if simplex_params.parameters.grad is not None:
                        combined_gradient += 0.1 * simplex_params.parameters.grad
                
                # 5. Update parameters
                simplex_params.update_parameters(combined_gradient, self.momentum_factor)
                
                # 6. Update statistics
                update_stats["total_updates"] += 1
                total_gradient_norm += simplex_params.get_gradient_norm()
                total_parameter_norm += simplex_params.get_parameter_norm()
                
                # Compute local coherence loss
                local_objective = self.local_objectives[simplex_id]
                face_params = [self.simplex_parameters[fid] for fid in local_objective.face_ids 
                              if fid in self.simplex_parameters]
                if face_params:
                    coherence_loss = local_objective.compute_loss(simplex_params, face_params)
                    total_coherence_loss += coherence_loss.item()
                
                total_instruction_strength += torch.norm(degeneracy_instruction).item()
        
        # Compute averages
        if update_stats["total_updates"] > 0:
            update_stats["average_gradient_norm"] = total_gradient_norm / update_stats["total_updates"]
            update_stats["average_parameter_norm"] = total_parameter_norm / update_stats["total_updates"]
            update_stats["coherence_loss"] = total_coherence_loss / update_stats["total_updates"]
            update_stats["instruction_strength"] = total_instruction_strength / update_stats["total_updates"]
        
        return update_stats
    
    def get_simplex_parameters(self, simplex_id: uuid.UUID) -> Optional[SimplexParameters]:
        """Get parameters for a specific simplex."""
        return self.simplex_parameters.get(simplex_id)
    
    def get_all_parameters_as_vector(self) -> torch.Tensor:
        """Get all simplex parameters as a single vector."""
        param_list = []
        for simplex_id in sorted(self.simplex_parameters.keys()):
            param_list.append(self.simplex_parameters[simplex_id].parameters.flatten())
        
        if param_list:
            return torch.cat(param_list)
        else:
            return torch.tensor([])
    
    def set_parameters_from_vector(self, param_vector: torch.Tensor):
        """Set all simplex parameters from a single vector."""
        start_idx = 0
        for simplex_id in sorted(self.simplex_parameters.keys()):
            simplex_params = self.simplex_parameters[simplex_id]
            param_size = simplex_params.parameters.numel()
            
            if start_idx + param_size <= param_vector.numel():
                new_params = param_vector[start_idx:start_idx + param_size]
                simplex_params.parameters = new_params.reshape(simplex_params.parameters.shape)
                start_idx += param_size
            else:
                logger.warning(f"Not enough parameters in vector for simplex {simplex_id}")
                break
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete state of the hierarchical message passing system."""
        return {
            "num_simplices": len(self.simplex_parameters),
            "num_levels": len(self.update_order),
            "total_parameters": sum(p.parameters.numel() for p in self.simplex_parameters.values()),
            "num_objectives": len(self.local_objectives),
            "num_instructions": len(self.degeneracy_instructions),
            "parameter_norms": {str(sid): p.get_parameter_norm() 
                              for sid, p in self.simplex_parameters.items()},
            "gradient_norms": {str(sid): p.get_gradient_norm() 
                             for sid, p in self.simplex_parameters.items()}
        }
    
    def visualize_message_flow(self) -> Dict[str, List[str]]:
        """Visualize the message flow structure."""
        flow_structure = {
            "bottom_up_flow": [],
            "top_down_flow": [],
            "level_structure": {}
        }
        
        # Bottom-up flow (face to simplex)
        for simplex_id, objective in self.local_objectives.items():
            simplex_name = self.simplex_parameters[simplex_id].simplex_name
            face_names = []
            for face_id in objective.face_ids:
                if face_id in self.simplex_parameters:
                    face_names.append(self.simplex_parameters[face_id].simplex_name)
            
            if face_names:
                flow_structure["bottom_up_flow"].append(f"{face_names} → {simplex_name}")
        
        # Top-down flow (degeneracy instructions)
        for (source_id, target_id), instruction in self.degeneracy_instructions.items():
            if source_id in self.simplex_parameters and target_id in self.simplex_parameters:
                source_name = self.simplex_parameters[source_id].simplex_name
                target_name = self.simplex_parameters[target_id].simplex_name
                flow_structure["top_down_flow"].append(f"{source_name} ⇣ {target_name}")
        
        # Level structure
        for level, simplex_ids in enumerate(self.update_order):
            level_names = [self.simplex_parameters[sid].simplex_name 
                          for sid in simplex_ids if sid in self.simplex_parameters]
            flow_structure["level_structure"][f"Level_{level}"] = level_names
        
        return flow_structure
    
    def __repr__(self):
        return (f"HierarchicalMessagePassingSystem("
                f"simplices={len(self.simplex_parameters)}, "
                f"levels={len(self.update_order)}, "
                f"objectives={len(self.local_objectives)})")


# Integration with existing GAIA training infrastructure

def create_hierarchical_system_from_model(model: nn.Module, 
                                        parameter_dim: int = 64) -> Optional[HierarchicalMessagePassingSystem]:
    """
    Create hierarchical message passing system from a GAIA model.
    
    Args:
        model: GAIA model with simplicial structure
        parameter_dim: Dimension of per-simplex parameters
        
    Returns:
        Hierarchical message passing system or None if model doesn't support it
    """
    if hasattr(model, 'functor'):  # Use 'functor' instead of 'simplicial_functor'
        hmp_system = HierarchicalMessagePassingSystem(
            model.functor,
            parameter_dim=parameter_dim
        )
        
        # CRITICAL FIX: Add message passer to the system
        if hasattr(model, 'message_passer'):
            hmp_system.message_passer = model.message_passer
        else:
            from gaia.core.abstractions import MessagePasser
            hmp_system.message_passer = MessagePasser()
            
        return hmp_system
    else:
        logger.warning("Model does not have functor attribute")
        return None


def integrate_with_training_loop(training_loop, hierarchical_system: HierarchicalMessagePassingSystem):
    """
    Integrate hierarchical message passing with existing training loop.
    
    This adds the hierarchical update step to the training process.
    """
    original_step = training_loop._training_step if hasattr(training_loop, '_training_step') else None
    
    def enhanced_training_step(batch, batch_idx):
        # Run original training step
        if original_step:
            loss = original_step(batch, batch_idx)
        else:
            # Default training step
            outputs = training_loop.model(batch[0])
            loss = training_loop.criterion(outputs, batch[1])
        
        # Run hierarchical message passing update
        update_stats = hierarchical_system.hierarchical_update_step(loss)
        
        # Log hierarchical statistics
        if hasattr(training_loop, 'log'):
            for key, value in update_stats.items():
                training_loop.log(f"hierarchical/{key}", value)
        
        return loss
    
    # Replace training step
    training_loop._training_step = enhanced_training_step
    
    return training_loop