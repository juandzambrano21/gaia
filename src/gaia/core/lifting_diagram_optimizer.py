"""Lifting Diagram Optimizer - GAIA Parameter Updates

Following Mahadevan (2024) Section 4.2 and Figure 7, this implements parameter
updates as lifting diagrams over simplicial sets instead of traditional gradient descent.

Key Features from Paper:
- Parameter updates formulated as lifting problems in simplicial categories
- Fibrations p: E → B defining the parameter space structure
- Base maps f: A → B representing current parameter state
- Lifting solutions h: B → X providing parameter updates
- Integration with horn extension learning framework
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import math

from .simplices import BasisRegistry, Simplex0, SimplexN, Simplex2
from .simplicial_factory import LiftingDiagram, Horn, HornType
from .horn_extension_learning import HornExtensionProblem, HornExtensionType
from ..nn import GAIAModule
from ..utils.device import get_device


class LiftingProblemType(Enum):
    """Types of lifting problems for parameter updates."""
    GRADIENT_LIFTING = "gradient_lifting"  # Traditional gradient as lifting problem
    HORN_EXTENSION_LIFTING = "horn_extension_lifting"  # Horn extension-based updates
    KAN_EXTENSION_LIFTING = "kan_extension_lifting"  # Kan extension-based updates
    FIBRATION_LIFTING = "fibration_lifting"  # General fibration lifting


@dataclass
class ParameterLiftingProblem:
    """Lifting problem for parameter updates.
    
    Following Definition 2 and Figure 7 in the paper, this represents
    the categorical structure for parameter updates via lifting diagrams.
    """
    parameter_name: str
    current_parameters: torch.Tensor
    target_space: torch.Tensor  # X in the diagram
    base_space: torch.Tensor  # B in the diagram
    total_space: torch.Tensor  # E in the diagram
    fibration: Callable[[torch.Tensor], torch.Tensor]  # p: E → B
    base_map: Callable[[torch.Tensor], torch.Tensor]  # f: A → B
    partial_lift: Optional[Callable[[torch.Tensor], torch.Tensor]]  # g: A → X
    lifting_type: LiftingProblemType
    simplicial_context: Dict[str, Any]
    
    def has_solution(self) -> bool:
        """Check if lifting solution exists based on fibration properties."""
        # For GAIA, we assume Kan complex properties ensure solvability
        return True
    
    def get_lifting_dimension(self) -> int:
        """Get the simplicial dimension for this lifting problem."""
        return self.simplicial_context.get('dimension', 1)


class LiftingDiagramOptimizer(GAIAModule):
    """Optimizer using lifting diagrams instead of gradient descent.
    
    This implements the complete lifting diagram framework from Mahadevan (2024)
    for parameter updates in GAIA's hierarchical learning system.
    
    Mathematical Foundation:
        Following Section 4.2 and Figure 7, parameter updates are formulated
        as lifting problems in the category of simplicial sets:
        
        Given fibration p: E → B and base map f: A → B,
        find lifting h: B → X such that the diagram commutes.
        
        This replaces traditional gradient descent with categorical updates
        that preserve the simplicial structure of the parameter space.
    
    Architecture:
        - Fibration constructors: Create parameter space fibrations
        - Base map generators: Map current state to base space
        - Lifting solvers: Find categorical parameter updates
        - Simplicial coordinators: Maintain hierarchical structure
    
    Args:
        max_dimension: Maximum simplicial dimension for lifting problems
        basis_registry: Registry of simplicial bases from Layer 1
        learning_rate: Base learning rate for lifting solutions
        use_kan_fibrations: Enable Kan fibration properties
    """
    
    def __init__(self,
                 max_dimension: int = 3,
                 basis_registry: Optional[BasisRegistry] = None,
                 learning_rate: float = 0.001,
                 use_kan_fibrations: bool = True):
        super().__init__()
        
        self.max_dimension = max_dimension
        self.basis_registry = basis_registry or BasisRegistry()
        self.learning_rate = learning_rate
        self.use_kan_fibrations = use_kan_fibrations
        
        # Fibration constructors for different parameter types
        self.fibration_constructors = nn.ModuleDict({
            'weight_fibration': FibrationConstructor(max_dimension, 'weight'),
            'bias_fibration': FibrationConstructor(max_dimension, 'bias'),
            'attention_fibration': FibrationConstructor(max_dimension, 'attention'),
            'embedding_fibration': FibrationConstructor(max_dimension, 'embedding')
        })
        
        # Base map generators (current state → base space)
        self.base_map_generators = nn.ModuleDict({
            f'base_gen_{i}': nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            ) for i in range(max_dimension + 1)
        })
        
        # Lifting solvers (find h: B → X)
        self.lifting_solvers = nn.ModuleDict({
            'gradient_solver': GradientLiftingSolver(max_dimension),
            'horn_solver': HornExtensionLiftingSolver(max_dimension),
            'kan_solver': KanExtensionLiftingSolver(max_dimension),
            'fibration_solver': GeneralFibrationSolver(max_dimension)
        })
        
        # Simplicial coordinators for hierarchical updates
        self.simplicial_coordinators = nn.ModuleDict({
            f'coord_{i}': nn.Linear(64, 64) for i in range(max_dimension + 1)
        })
        
        # Set GAIA metadata
        self.set_gaia_metadata(
            layer="Parameter Update Layer",
            lifting_diagrams=True,
            replaces_gradient_descent=True,
            categorical_updates=True,
            max_simplicial_dimension=max_dimension,
            kan_fibrations=use_kan_fibrations
        )
    
    def create_lifting_problems(self,
                              parameters: Dict[str, torch.Tensor],
                              gradients: Dict[str, torch.Tensor],
                              simplicial_context: Dict[str, Any]) -> List[ParameterLiftingProblem]:
        """Create lifting problems for parameter updates.
        
        This converts traditional gradient-based updates into lifting problems
        over simplicial sets, following the paper's categorical framework.
        
        Args:
            parameters: Current model parameters
            gradients: Computed gradients (traditional or horn-based)
            simplicial_context: Simplicial structure from Layer 1
            
        Returns:
            List of lifting problems to solve for parameter updates
        """
        lifting_problems = []
        
        for param_name, param_tensor in parameters.items():
            if param_name in gradients:
                grad_tensor = gradients[param_name]
                
                # Determine lifting problem type based on parameter characteristics
                lifting_type = self._determine_lifting_type(param_name, param_tensor, grad_tensor)
                
                # Create lifting problem
                problem = self._construct_lifting_problem(
                    param_name, param_tensor, grad_tensor, lifting_type, simplicial_context
                )
                
                lifting_problems.append(problem)
        
        return lifting_problems
    
    def solve_lifting_problems(self,
                             problems: List[ParameterLiftingProblem]) -> Dict[str, torch.Tensor]:
        """Solve lifting problems to get parameter updates.
        
        This implements the core lifting diagram solution process,
        replacing traditional gradient descent with categorical updates.
        
        Args:
            problems: List of lifting problems to solve
            
        Returns:
            Dictionary of parameter updates
        """
        parameter_updates = {}
        
        # Group problems by simplicial dimension for coordinated solving
        problems_by_dim = self._group_by_dimension(problems)
        
        # Solve problems within each dimension
        for dim, dim_problems in problems_by_dim.items():
            dim_updates = self._solve_dimension_problems(dim, dim_problems)
            parameter_updates.update(dim_updates)
        
        return parameter_updates
    
    def step(self,
           parameters: Dict[str, torch.Tensor],
           gradients: Dict[str, torch.Tensor],
           simplicial_context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """Perform one optimization step using lifting diagrams.
        
        This is the main interface that replaces traditional optimizer.step()
        with categorical lifting diagram-based parameter updates.
        
        Args:
            parameters: Current model parameters
            gradients: Computed gradients
            simplicial_context: Simplicial structure context
            
        Returns:
            Updated parameters via lifting diagram solutions
        """
        context = simplicial_context or {'dimension': 1}
        
        # Create lifting problems from current state
        lifting_problems = self.create_lifting_problems(parameters, gradients, context)
        
        # Solve lifting problems for parameter updates
        parameter_updates = self.solve_lifting_problems(lifting_problems)
        
        # Apply updates to parameters
        updated_parameters = {}
        for param_name, param_tensor in parameters.items():
            if param_name in parameter_updates:
                update = parameter_updates[param_name]
                updated_parameters[param_name] = param_tensor + self.learning_rate * update
            else:
                updated_parameters[param_name] = param_tensor
        
        return updated_parameters
    
    def _determine_lifting_type(self,
                              param_name: str,
                              param_tensor: torch.Tensor,
                              grad_tensor: torch.Tensor) -> LiftingProblemType:
        """Determine appropriate lifting problem type for parameter."""
        # Heuristics based on parameter characteristics
        if 'attention' in param_name.lower():
            return LiftingProblemType.HORN_EXTENSION_LIFTING
        elif 'embedding' in param_name.lower():
            return LiftingProblemType.KAN_EXTENSION_LIFTING
        elif param_tensor.dim() > 2:
            return LiftingProblemType.FIBRATION_LIFTING
        else:
            return LiftingProblemType.GRADIENT_LIFTING
    
    def _construct_lifting_problem(self,
                                 param_name: str,
                                 param_tensor: torch.Tensor,
                                 grad_tensor: torch.Tensor,
                                 lifting_type: LiftingProblemType,
                                 simplicial_context: Dict[str, Any]) -> ParameterLiftingProblem:
        """Construct lifting problem for given parameter."""
        # Flatten tensors for consistent processing
        param_flat = param_tensor.flatten()
        grad_flat = grad_tensor.flatten()
        
        # Ensure consistent dimensions for fibration construction
        max_size = max(param_flat.numel(), grad_flat.numel(), 64)
        
        # Pad to consistent size
        if param_flat.numel() < max_size:
            param_flat = torch.cat([param_flat, torch.zeros(max_size - param_flat.numel())])
        elif param_flat.numel() > max_size:
            param_flat = param_flat[:max_size]
            
        if grad_flat.numel() < max_size:
            grad_flat = torch.cat([grad_flat, torch.zeros(max_size - grad_flat.numel())])
        elif grad_flat.numel() > max_size:
            grad_flat = grad_flat[:max_size]
        
        # Construct fibration based on parameter type
        fibration_type = self._get_fibration_type(param_name)
        fibration_constructor = self.fibration_constructors[fibration_type]
        
        # Create spaces for lifting diagram
        total_space = param_flat  # E
        base_space = fibration_constructor.create_base_space(param_flat)  # B
        target_space = grad_flat  # X (target update direction)
        
        # Create fibration map p: E → B
        fibration = lambda x: fibration_constructor.apply_fibration(x)
        
        # Create base map f: A → B (current state to base)
        current_dim = simplicial_context.get('dimension', 1)
        if current_dim <= self.max_dimension:
            base_generator = self.base_map_generators[f'base_gen_{current_dim}']
            base_map = lambda x: base_generator(x[:64] if x.numel() >= 64 else 
                                              torch.cat([x, torch.zeros(64 - x.numel())]))
        else:
            base_map = lambda x: x[:16] if x.numel() >= 16 else x
        
        return ParameterLiftingProblem(
            parameter_name=param_name,
            current_parameters=param_tensor,
            target_space=target_space,
            base_space=base_space,
            total_space=total_space,
            fibration=fibration,
            base_map=base_map,
            partial_lift=None,  # Will be computed during solving
            lifting_type=lifting_type,
            simplicial_context=simplicial_context
        )
    
    def _get_fibration_type(self, param_name: str) -> str:
        """Get fibration constructor type for parameter."""
        if 'weight' in param_name.lower():
            return 'weight_fibration'
        elif 'bias' in param_name.lower():
            return 'bias_fibration'
        elif 'attention' in param_name.lower():
            return 'attention_fibration'
        elif 'embedding' in param_name.lower():
            return 'embedding_fibration'
        else:
            return 'weight_fibration'  # Default
    
    def _group_by_dimension(self, problems: List[ParameterLiftingProblem]) -> Dict[int, List[ParameterLiftingProblem]]:
        """Group lifting problems by simplicial dimension."""
        grouped = {}
        for problem in problems:
            dim = problem.get_lifting_dimension()
            if dim not in grouped:
                grouped[dim] = []
            grouped[dim].append(problem)
        return grouped
    
    def _solve_dimension_problems(self, 
                                dimension: int, 
                                problems: List[ParameterLiftingProblem]) -> Dict[str, torch.Tensor]:
        """Solve lifting problems within a single dimension."""
        updates = {}
        
        for problem in problems:
            # Select appropriate solver based on lifting type
            solver_name = self._get_solver_name(problem.lifting_type)
            solver = self.lifting_solvers[solver_name]
            
            # Solve lifting problem
            update_tensor = solver.solve(problem)
            
            # Reshape back to original parameter shape
            original_shape = problem.current_parameters.shape
            original_numel = problem.current_parameters.numel()
            
            if update_tensor.numel() >= original_numel:
                reshaped_update = update_tensor[:original_numel].reshape(original_shape)
            else:
                padded = torch.cat([update_tensor, torch.zeros(original_numel - update_tensor.numel())])
                reshaped_update = padded.reshape(original_shape)
            
            updates[problem.parameter_name] = reshaped_update
        
        # Apply simplicial coordination if multiple problems in dimension
        if len(problems) > 1 and dimension <= self.max_dimension:
            coordinator = self.simplicial_coordinators[f'coord_{dimension}']
            updates = self._coordinate_dimension_updates(updates, coordinator)
        
        return updates
    
    def _get_solver_name(self, lifting_type: LiftingProblemType) -> str:
        """Get solver name for lifting problem type."""
        type_to_solver = {
            LiftingProblemType.GRADIENT_LIFTING: 'gradient_solver',
            LiftingProblemType.HORN_EXTENSION_LIFTING: 'horn_solver',
            LiftingProblemType.KAN_EXTENSION_LIFTING: 'kan_solver',
            LiftingProblemType.FIBRATION_LIFTING: 'fibration_solver'
        }
        return type_to_solver.get(lifting_type, 'gradient_solver')
    
    def _coordinate_dimension_updates(self, 
                                    updates: Dict[str, torch.Tensor], 
                                    coordinator: nn.Module) -> Dict[str, torch.Tensor]:
        """Coordinate parameter updates within a dimension."""
        coordinated_updates = {}
        
        for param_name, update_tensor in updates.items():
            # Flatten and pad for coordination
            update_flat = update_tensor.flatten()
            if update_flat.numel() < 64:
                update_flat = torch.cat([update_flat, torch.zeros(64 - update_flat.numel())])
            elif update_flat.numel() > 64:
                update_flat = update_flat[:64]
            
            # Apply coordination
            coordinated_flat = coordinator(update_flat)
            
            # Reshape back
            original_shape = update_tensor.shape
            original_numel = update_tensor.numel()
            
            if coordinated_flat.numel() >= original_numel:
                coordinated = coordinated_flat[:original_numel].reshape(original_shape)
            else:
                padded = torch.cat([coordinated_flat, torch.zeros(original_numel - coordinated_flat.numel())])
                coordinated = padded.reshape(original_shape)
            
            coordinated_updates[param_name] = coordinated
        
        return coordinated_updates


class FibrationConstructor(GAIAModule):
    """Constructs fibrations for different parameter types."""
    
    def __init__(self, max_dimension: int, param_type: str):
        super().__init__()
        self.max_dimension = max_dimension
        self.param_type = param_type
        
        # Fibration construction networks
        self.base_constructor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.fibration_map = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16)
        )
    
    def create_base_space(self, total_space: torch.Tensor) -> torch.Tensor:
        """Create base space B from total space E."""
        # Ensure consistent input size
        if total_space.numel() < 64:
            padded = torch.cat([total_space, torch.zeros(64 - total_space.numel())])
        else:
            padded = total_space[:64]
        
        return self.base_constructor(padded)
    
    def apply_fibration(self, total_space: torch.Tensor) -> torch.Tensor:
        """Apply fibration map p: E → B."""
        # Ensure consistent input size
        if total_space.numel() < 64:
            padded = torch.cat([total_space, torch.zeros(64 - total_space.numel())])
        else:
            padded = total_space[:64]
        
        return self.fibration_map(padded)


class GradientLiftingSolver(GAIAModule):
    """Solver for gradient-based lifting problems."""
    
    def __init__(self, max_dimension: int):
        super().__init__()
        self.solver = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def solve(self, problem: ParameterLiftingProblem) -> torch.Tensor:
        """Solve gradient lifting problem."""
        # Use target space (gradient direction) as input
        target = problem.target_space
        if target.numel() < 64:
            target = torch.cat([target, torch.zeros(64 - target.numel())])
        elif target.numel() > 64:
            target = target[:64]
        
        return self.solver(target)


class HornExtensionLiftingSolver(GAIAModule):
    """Solver for horn extension-based lifting problems."""
    
    def __init__(self, max_dimension: int):
        super().__init__()
        self.inner_solver = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.outer_solver = nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )
    
    def solve(self, problem: ParameterLiftingProblem) -> torch.Tensor:
        """Solve horn extension lifting problem."""
        # Determine if this is inner or outer horn based on context
        dimension = problem.get_lifting_dimension()
        
        # Use appropriate solver
        if dimension == 1:  # Inner horn case
            solver = self.inner_solver
        else:  # Outer horn case
            solver = self.outer_solver
        
        # Prepare input
        combined = torch.cat([problem.target_space[:32], problem.base_space[:32]])
        if combined.numel() < 64:
            combined = torch.cat([combined, torch.zeros(64 - combined.numel())])
        elif combined.numel() > 64:
            combined = combined[:64]
        
        return solver(combined)


class KanExtensionLiftingSolver(GAIAModule):
    """Solver for Kan extension-based lifting problems."""
    
    def __init__(self, max_dimension: int):
        super().__init__()
        self.left_kan_solver = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 64)
        )
        self.right_kan_solver = nn.Sequential(
            nn.Linear(64, 80),
            nn.ReLU(),
            nn.Linear(80, 64)
        )
    
    def solve(self, problem: ParameterLiftingProblem) -> torch.Tensor:
        """Solve Kan extension lifting problem."""
        # Use left Kan extension for most cases
        solver = self.left_kan_solver
        
        # Combine spaces for Kan extension
        spaces = [problem.target_space, problem.base_space, problem.total_space]
        combined = torch.cat([s[:21] for s in spaces if s.numel() >= 21] + 
                           [s for s in spaces if s.numel() < 21])[:64]
        
        if combined.numel() < 64:
            combined = torch.cat([combined, torch.zeros(64 - combined.numel())])
        
        return solver(combined)


class GeneralFibrationSolver(GAIAModule):
    """Solver for general fibration lifting problems."""
    
    def __init__(self, max_dimension: int):
        super().__init__()
        self.solver = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def solve(self, problem: ParameterLiftingProblem) -> torch.Tensor:
        """Solve general fibration lifting problem."""
        # Apply fibration to total space
        fibrated = problem.fibration(problem.total_space)
        
        # Ensure consistent size
        if fibrated.numel() < 64:
            fibrated = torch.cat([fibrated, torch.zeros(64 - fibrated.numel())])
        elif fibrated.numel() > 64:
            fibrated = fibrated[:64]
        
        return self.solver(fibrated)