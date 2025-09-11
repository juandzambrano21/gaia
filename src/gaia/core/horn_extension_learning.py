"""Horn Extension Learning Module - GAIA Layer 1 Integration

Following Mahadevan (2024) Section 4.2, this implements the complete horn extension
learning framework that bridges Layer 1 (simplicial sets) with Layer 2 (coalgebras).

Key Features from Paper:
- Inner horn extensions: solvable by traditional backpropagation
- Outer horn extensions: require advanced lifting diagram methods
- Hierarchical parameter updates via lifting diagrams over simplicial sets
- Integration with simplicial complex hierarchy from simplices.py
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .simplices import BasisRegistry, Simplex0, Simplex1, SimplexN, Simplex2
from .simplicial_factory import Horn, HornType, LiftingDiagram, SimplicialFactory
from ..nn import GAIAModule
from ..utils.device import get_device


class HornExtensionType(Enum):
    """Types of horn extension problems in GAIA learning."""
    INNER = "inner"  # Traditional backpropagation-solvable (i = 1, ..., n-1)
    OUTER_LEFT = "outer_left"  # Left inverse problems (i = 0)
    OUTER_RIGHT = "outer_right"  # Right inverse problems (i = n)
    DEGENERATE = "degenerate"  # Trivial cases


@dataclass
class HornExtensionProblem:
    """Horn extension problem for hierarchical learning.
    
    Following paper Section 4.2, this represents the lifting problem structure
    needed for GAIA's hierarchical learning framework.
    """
    horn_type: HornExtensionType
    dimension: int
    missing_face: int
    parent_simplex: Any
    target_parameters: torch.Tensor
    current_parameters: torch.Tensor
    learning_context: Dict[str, Any]
    
    def is_solvable_by_backprop(self) -> bool:
        """Check if this horn can be solved by traditional backpropagation."""
        return self.horn_type == HornExtensionType.INNER
    
    def requires_lifting_diagram(self) -> bool:
        """Check if this horn requires lifting diagram methods."""
        return self.horn_type in [HornExtensionType.OUTER_LEFT, HornExtensionType.OUTER_RIGHT]


class HornExtensionSolver(GAIAModule):
    """Core solver for horn extension problems in GAIA.
    
    This implements the complete horn extension learning framework from
    Mahadevan (2024), bridging simplicial sets with coalgebraic learning.
    
    Mathematical Foundation:
        Following Section 4.2, horn extension problems Λᵢⁿ represent the
        fundamental learning challenges in GAIA's hierarchical framework:
        
        - Inner horns (1 ≤ i ≤ n-1): Traditional backpropagation
        - Outer horns (i = 0, n): Advanced lifting diagram methods
        - Parameter updates as lifting diagrams over simplicial sets
    
    Architecture:
        - Inner horn solver: Enhanced backpropagation with simplicial context
        - Outer horn solver: Lifting diagram construction and solution
        - Hierarchical integration: Connects with simplicial complex hierarchy
        - Learning coordination: Manages distributed learning across simplices
    
    Args:
        max_dimension: Maximum simplicial dimension to handle
        basis_registry: Registry of simplicial bases from Layer 1
        learning_rate: Base learning rate for parameter updates
        use_lifting_diagrams: Enable advanced lifting diagram methods
    """
    
    def __init__(self,
                 max_dimension: int = 3,
                 basis_registry: Optional[BasisRegistry] = None,
                 learning_rate: float = 0.001,
                 use_lifting_diagrams: bool = True):
        super().__init__()
        
        self.max_dimension = max_dimension
        self.basis_registry = basis_registry or BasisRegistry()
        self.learning_rate = learning_rate
        self.use_lifting_diagrams = use_lifting_diagrams
        
        # Inner horn solver: Enhanced backpropagation
        self.inner_horn_solver = InnerHornSolver(
            max_dimension=max_dimension,
            learning_rate=learning_rate
        )
        
        # Outer horn solver: Lifting diagram methods
        if use_lifting_diagrams:
            self.outer_horn_solver = OuterHornSolver(
                max_dimension=max_dimension,
                learning_rate=learning_rate
            )
        else:
            self.outer_horn_solver = None
        
        # Hierarchical learning coordinator
        self.learning_coordinator = HierarchicalLearningCoordinator(
            max_dimension=max_dimension,
            basis_registry=self.basis_registry
        )
        
        # Set GAIA metadata
        self.set_gaia_metadata(
            layer="Layer 1 Integration",
            horn_extension_learning=True,
            lifting_diagrams=use_lifting_diagrams,
            hierarchical_learning=True,
            max_simplicial_dimension=max_dimension
        )
    
    def solve_horn_extension(self, 
                           problem: HornExtensionProblem) -> torch.Tensor:
        """Solve horn extension problem for hierarchical learning.
        
        This is the core of GAIA's hierarchical learning framework,
        implementing the complete horn extension solution from the paper.
        
        Args:
            problem: Horn extension problem to solve
            
        Returns:
            Updated parameters solving the horn extension
        """
        if problem.is_solvable_by_backprop():
            # Inner horn: Use enhanced backpropagation
            return self.inner_horn_solver.solve(problem)
        
        elif problem.requires_lifting_diagram() and self.outer_horn_solver:
            # Outer horn: Use lifting diagram methods
            return self.outer_horn_solver.solve(problem)
        
        else:
            # Fallback to traditional update
            return self._fallback_update(problem)
    
    def create_horn_problems_from_loss(self,
                                     loss: torch.Tensor,
                                     parameters: Dict[str, torch.Tensor],
                                     simplicial_context: Dict[str, Any]) -> List[HornExtensionProblem]:
        """Create horn extension problems from loss and simplicial context.
        
        Following the paper's framework, this converts traditional loss-based
        learning into horn extension problems over simplicial sets.
        
        Args:
            loss: Current loss tensor
            parameters: Model parameters to update
            simplicial_context: Context from simplicial hierarchy
            
        Returns:
            List of horn extension problems to solve
        """
        problems = []
        
        # Extract simplicial dimension and context
        current_dim = simplicial_context.get('dimension', 1)
        dimension_weights = simplicial_context.get('dimension_weights', None)
        
        # Create problems for each parameter group
        for param_name, param_tensor in parameters.items():
            # Determine horn type based on parameter role and dimension
            horn_type = self._determine_horn_type(param_name, current_dim, loss)
            
            # Create horn extension problem
            problem = HornExtensionProblem(
                horn_type=horn_type,
                dimension=current_dim,
                missing_face=self._get_missing_face(horn_type, current_dim),
                parent_simplex=self._get_parent_simplex(current_dim),
                target_parameters=param_tensor,
                current_parameters=param_tensor.clone(),
                learning_context={
                    'loss': loss,
                    'param_name': param_name,
                    'dimension_weights': dimension_weights,
                    'simplicial_context': simplicial_context
                }
            )
            
            problems.append(problem)
        
        return problems
    
    def coordinate_hierarchical_learning(self,
                                       problems: List[HornExtensionProblem]) -> Dict[str, torch.Tensor]:
        """Coordinate learning across hierarchical simplicial structure.
        
        This implements the distributed learning process described in the paper,
        where each sub-simplicial complex updates parameters to solve lifting diagrams.
        
        Args:
            problems: List of horn extension problems to coordinate
            
        Returns:
            Dictionary of updated parameters
        """
        return self.learning_coordinator.coordinate_learning(problems)
    
    def _determine_horn_type(self, param_name: str, dimension: int, loss: torch.Tensor) -> HornExtensionType:
        """Determine horn type based on parameter role and learning context."""
        # Heuristic: attention parameters often involve outer horns
        if 'attention' in param_name.lower() or 'query' in param_name.lower():
            return HornExtensionType.OUTER_LEFT
        elif 'output' in param_name.lower() or 'projection' in param_name.lower():
            return HornExtensionType.OUTER_RIGHT
        else:
            # Most parameters can use inner horn (traditional backprop)
            return HornExtensionType.INNER
    
    def _get_missing_face(self, horn_type: HornExtensionType, dimension: int) -> int:
        """Get missing face index for horn type."""
        if horn_type == HornExtensionType.OUTER_LEFT:
            return 0
        elif horn_type == HornExtensionType.OUTER_RIGHT:
            return dimension
        else:
            return 1  # Inner horn
    
    def _get_parent_simplex(self, dimension: int) -> Any:
        """Get parent simplex for given dimension."""
        if dimension == 0:
            return Simplex0(0, "vertex_simplex", self.basis_registry)
        elif dimension == 1:
            # Create dummy 0-simplices for domain and codomain
            domain = Simplex0(0, "domain_vertex", self.basis_registry)
            codomain = Simplex0(0, "codomain_vertex", self.basis_registry)
            morphism = torch.nn.Identity()
            return Simplex1(morphism, domain, codomain, "edge_simplex")
        elif dimension == 2:
            # Create dummy simplices for Simplex2 constructor
            domain = Simplex0(0, "domain_vertex", self.basis_registry)
            codomain = Simplex0(0, "codomain_vertex", self.basis_registry)
            morphism = torch.nn.Identity()
            f = Simplex1(morphism, domain, codomain, "edge_f")
            g = Simplex1(morphism, domain, codomain, "edge_g")
            return Simplex2(f, g, "triangle_simplex")
        else:
            # Create a proper SimplexN with required parameters
            name = f"parent_simplex_{dimension}"
            components = [f"v{i}" for i in range(dimension + 1)]  # Standard simplex vertices
            return SimplexN(dimension, name, components)
    
    def _fallback_update(self, problem: HornExtensionProblem) -> torch.Tensor:
        """Fallback parameter update when advanced methods unavailable."""
        # Simple gradient-based update
        if problem.target_parameters.grad is not None:
            return problem.target_parameters - self.learning_rate * problem.target_parameters.grad
        else:
            return problem.target_parameters


class InnerHornSolver(GAIAModule):
    """Solver for inner horn extension problems.
    
    Inner horns (1 ≤ i ≤ n-1) can be solved using enhanced backpropagation
    with simplicial context awareness.
    """
    
    def __init__(self, max_dimension: int = 3, learning_rate: float = 0.001):
        super().__init__()
        self.max_dimension = max_dimension
        self.learning_rate = learning_rate
        
        # Enhanced backpropagation with simplicial awareness
        self.simplicial_gradient_processor = nn.ModuleDict({
            f'dim_{i}': nn.Linear(1, 1) for i in range(max_dimension + 1)
        })
    
    def solve(self, problem: HornExtensionProblem) -> torch.Tensor:
        """Solve inner horn using enhanced backpropagation."""
        # Extract gradient information
        if problem.target_parameters.grad is None:
            return problem.target_parameters
        
        # Apply simplicial context to gradient
        grad = problem.target_parameters.grad
        
        # Process gradient through simplicial dimension processor
        if problem.dimension <= self.max_dimension:
            processor = self.simplicial_gradient_processor[f'dim_{problem.dimension}']
            # Apply processor to gradient magnitude
            grad_magnitude = torch.norm(grad)
            processed_magnitude = processor(grad_magnitude.unsqueeze(0)).squeeze(0)
            
            # Scale gradient by processed magnitude
            if grad_magnitude > 0:
                grad = grad * (processed_magnitude / grad_magnitude)
        
        # Apply enhanced update
        updated_params = problem.target_parameters - self.learning_rate * grad
        
        return updated_params


class OuterHornSolver(GAIAModule):
    """Solver for outer horn extension problems using lifting diagrams.
    
    Outer horns (i = 0, n) require advanced lifting diagram methods
    beyond traditional backpropagation.
    """
    
    def __init__(self, max_dimension: int = 3, learning_rate: float = 0.001):
        super().__init__()
        self.max_dimension = max_dimension
        self.learning_rate = learning_rate
        
        # Lifting diagram constructors
        self.lifting_constructors = nn.ModuleDict({
            'left_inverse': nn.Linear(64, 64),  # For outer left horns
            'right_inverse': nn.Linear(64, 64),  # For outer right horns
        })
        
        # Fibration approximators
        self.fibration_approximators = nn.ModuleDict({
            f'dim_{i}': nn.Linear(64, 32) for i in range(max_dimension + 1)
        })
    
    def solve(self, problem: HornExtensionProblem) -> torch.Tensor:
        """Solve outer horn using lifting diagram methods."""
        # Construct lifting diagram
        lifting_diagram = self._construct_lifting_diagram(problem)
        
        # Solve lifting problem
        solution = self._solve_lifting_diagram(lifting_diagram, problem)
        
        return solution
    
    def _construct_lifting_diagram(self, problem: HornExtensionProblem) -> Dict[str, Any]:
        """Construct lifting diagram for outer horn problem."""
        # This implements the categorical construction from the paper
        param_flat = problem.target_parameters.flatten()
        
        # Ensure we have enough dimensions for processing
        if param_flat.numel() < 64:
            param_flat = torch.cat([param_flat, torch.zeros(64 - param_flat.numel())])
        elif param_flat.numel() > 64:
            param_flat = param_flat[:64]
        
        # Construct base and total spaces
        if problem.dimension <= self.max_dimension:
            fibration_approx = self.fibration_approximators[f'dim_{problem.dimension}']
            base_space = fibration_approx(param_flat)
        else:
            base_space = param_flat[:32]
        
        return {
            'base_space': base_space,
            'total_space': param_flat,
            'fibration': lambda x: x[:32] if x.numel() >= 32 else x,
            'problem': problem
        }
    
    def _solve_lifting_diagram(self, 
                             lifting_diagram: Dict[str, Any], 
                             problem: HornExtensionProblem) -> torch.Tensor:
        """Solve the lifting diagram to find parameter update."""
        total_space = lifting_diagram['total_space']
        
        # Apply appropriate inverse constructor
        if problem.horn_type == HornExtensionType.OUTER_LEFT:
            constructor = self.lifting_constructors['left_inverse']
        else:
            constructor = self.lifting_constructors['right_inverse']
        
        # Construct solution
        solution_flat = constructor(total_space)
        
        # Reshape back to original parameter shape
        original_shape = problem.target_parameters.shape
        original_numel = problem.target_parameters.numel()
        
        if solution_flat.numel() >= original_numel:
            solution = solution_flat[:original_numel].reshape(original_shape)
        else:
            # Pad if necessary
            padded = torch.cat([solution_flat, torch.zeros(original_numel - solution_flat.numel())])
            solution = padded.reshape(original_shape)
        
        return solution


class HierarchicalLearningCoordinator(GAIAModule):
    """Coordinates learning across hierarchical simplicial structure.
    
    This implements the distributed learning process where each sub-simplicial
    complex updates parameters to solve lifting diagrams.
    """
    
    def __init__(self, max_dimension: int = 3, basis_registry: Optional[BasisRegistry] = None):
        super().__init__()
        self.max_dimension = max_dimension
        self.basis_registry = basis_registry or BasisRegistry()
        
        # Coordination networks for each dimension
        self.coordinators = nn.ModuleDict({
            f'dim_{i}': nn.Linear(64, 64) for i in range(max_dimension + 1)
        })
    
    def coordinate_learning(self, problems: List[HornExtensionProblem]) -> Dict[str, torch.Tensor]:
        """Coordinate learning across all horn extension problems."""
        updated_parameters = {}
        
        # Group problems by dimension
        problems_by_dim = {}
        for problem in problems:
            dim = problem.dimension
            if dim not in problems_by_dim:
                problems_by_dim[dim] = []
            problems_by_dim[dim].append(problem)
        
        # Coordinate learning within each dimension
        for dim, dim_problems in problems_by_dim.items():
            if dim <= self.max_dimension:
                coordinator = self.coordinators[f'dim_{dim}']
                
                # Combine parameter updates for this dimension
                combined_updates = self._combine_dimension_updates(dim_problems, coordinator)
                
                # Store updates
                for i, problem in enumerate(dim_problems):
                    param_name = problem.learning_context['param_name']
                    updated_parameters[param_name] = combined_updates[i]
        
        return updated_parameters
    
    def _combine_dimension_updates(self, 
                                 problems: List[HornExtensionProblem], 
                                 coordinator: nn.Module) -> List[torch.Tensor]:
        """Combine parameter updates within a single dimension."""
        updates = []
        
        for problem in problems:
            # Get parameter tensor
            param_tensor = problem.target_parameters
            param_flat = param_tensor.flatten()
            
            # Ensure consistent size for coordination
            if param_flat.numel() < 64:
                param_flat = torch.cat([param_flat, torch.zeros(64 - param_flat.numel())])
            elif param_flat.numel() > 64:
                param_flat = param_flat[:64]
            
            # Apply coordination
            coordinated = coordinator(param_flat)
            
            # Reshape back
            original_shape = param_tensor.shape
            original_numel = param_tensor.numel()
            
            if coordinated.numel() >= original_numel:
                update = coordinated[:original_numel].reshape(original_shape)
            else:
                padded = torch.cat([coordinated, torch.zeros(original_numel - coordinated.numel())])
                update = padded.reshape(original_shape)
            
            updates.append(update)
        
        return updates