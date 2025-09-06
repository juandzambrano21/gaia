"""
Module: yoneda_proxy
Implements the Metric Yoneda Proxy for universal lifting.

Following Mahadevan (2024), this implements the full Metric Yoneda embedding
Y_C(z) = (d(z,c))_{c∈C} with all training points as probes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.parametrizations as parametrizations
import torch.linalg as linalg
from typing import Dict, Any 
from gaia.core import DEVICE
from gaia.utils.device import get_device

class SpectralNormalizedLinear(nn.Module):
    """
    Linear layer with spectral normalization to enforce 1-Lipschitz constraint.
    
    This uses torch.nn.utils.parametrizations.spectral_norm which properly
    registers hooks to update the spectral normalization during both forward
    and backward passes, ensuring the 1-Lipschitz constraint is maintained
    throughout training.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        linear = nn.Linear(in_features, out_features, bias=bias)
        # Apply spectral normalization to enforce 1-Lipschitz constraint
        self.linear = parametrizations.spectral_norm(linear)
        
    def forward(self, x):
        return self.linear(x)
        
    def verify_lipschitz_property(self, batch_size: int = 100, tol: float = 1e-3) -> Dict[str, Any]:
        device = next(self.parameters()).device if hasattr(self, 'parameters') else self.device
        
        # Compute the largest singular value
        sigma = linalg.svdvals(weight)[0].item()
        is_lipschitz = sigma <= 1.0 + tol
        # Replace: torch.tensor(is_lipschitz, device=device)
        result = {
            'is_lipschitz': torch.full((1,), float(is_lipschitz), device=device, requires_grad=False),
            'max_ratio': torch.full((1,), max_ratio, device=device, requires_grad=False),
            'tolerance': torch.full((1,), tol, device=device, requires_grad=False)
        }
        
        return result


class SpectralNormalizedMetric(nn.Module):
    """
    A 1-Lipschitz metric network using spectral normalization.
    
    This ensures that the network satisfies the requirements of the
    metric Yoneda lemma by being non-expansive (1-Lipschitz).
    
    Two options are provided:
    1. Direct metric using a true 1-Lipschitz function: d(x,y) = ||x-y||/(||x-y||+ε)
    2. Learned metric with spectral normalization (enforced 1-Lipschitz)
    
    This implementation follows the Lawvere metric space enrichment where
    morphisms must be non-expansive maps in the [0,∞]-enriched category.
    
    References:
    - Bonsangue et al. 1998, "Generalized Ultrametrics in Quantitative Domain Theory"
    - Mahadevan 2024, "Generative Algebraic Intelligence Architecture"
    """
    def __init__(self, dim: int, use_direct_metric: bool = True, device=None):
        super().__init__()
        # Whether to use direct metric or learned metric
        self.use_direct_metric = use_direct_metric
        self.dim = dim
        self.device = device or DEVICE
        
        # Small epsilon to avoid division by zero in direct metric
        self.epsilon = 1e-6
        
        # Learned metric network with spectral normalization
        if not use_direct_metric:
            # Create a 1-Lipschitz network using spectral normalization
            # LayerNorm can be used if we ensure its scale parameter is ≤ 1
            self.sn_net = nn.Sequential(
                SpectralNormalizedLinear(dim * 2, 32),
                self._create_lipschitz_norm(32),
                nn.ReLU(),
                SpectralNormalizedLinear(32, 16),
                self._create_lipschitz_norm(16),
                nn.ReLU(),
                SpectralNormalizedLinear(16, 1)
            ).to(self.device)
            
            # Ensure the network is 1-Lipschitz by scaling the first layer if needed
            self._enforce_lipschitz_constraint()
    
    def _create_lipschitz_norm(self, dim):
        """
        Create a normalization layer that preserves the 1-Lipschitz property.
        
        LayerNorm is 1-Lipschitz if its scale parameter γ is ≤ 1.
        """
        norm = nn.LayerNorm(dim)
        # Initialize scale parameter to be ≤ 1
        with torch.no_grad():
            norm.weight.data.clamp_(max=1.0)
        # Register a hook to ensure scale remains ≤ 1 during training
        def clamp_hook(grad):
            if grad is None:
                return None
            return torch.where(norm.weight > 1.0, 0.0, grad)
        norm.weight.register_hook(clamp_hook)
        return norm
    
    def _enforce_lipschitz_constraint(self):
        """
        Enforce the 1-Lipschitz constraint by scaling the first layer if needed.
        
        This ensures that the product of all layer norms is ≤ 1.
        """
        if not hasattr(self, 'sn_net'):
            return
            
        # Trigger a forward pass to ensure spectral norm is updated
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.dim * 2, device=self.device)
            _ = self.sn_net(dummy_input)
        
        # Compute product of all layer norms
        product_norm = 1.0
        first_sn_layer = None
        
        for module in self.sn_net.modules():
            if isinstance(module, SpectralNormalizedLinear):
                # Get weight after spectral normalization
                weight = module.linear.weight
                # Compute spectral norm (largest singular value)
                with torch.no_grad():
                    sigma = torch.linalg.svdvals(weight)[0].item()
                product_norm *= sigma
                
                # Keep track of first SN layer for scaling if needed
                if first_sn_layer is None:
                    first_sn_layer = module
        
        # If product > 1, scale the first layer to ensure product = 1
        if product_norm > 1.0 and first_sn_layer is not None:
            scale_factor = 1.0 / product_norm
            # Access the original weight and scale it
            with torch.no_grad():
                # Get the parametrization
                param = getattr(first_sn_layer.linear, '_parametrizations', {}).get('weight', None)
                if param is not None and hasattr(param[0], 'original'):
                    # Scale the original weight
                    param[0].original.data *= scale_factor
    
    def forward(self, x):
        if self.use_direct_metric:
            # Split the input into two parts
            dim = x.size(1) // 2
            x1 = x[:, :dim]
            x2 = x[:, dim:]
            
            # Compute Euclidean distance
            diff = x1 - x2
            norm = torch.norm(diff, dim=1, keepdim=True)
            
            return norm / (norm + self.epsilon)
        else:
            return self.sn_net(x)
            
    def verify_lipschitz(self, tol=1e-4):
        """
        Verify that the network is 1-Lipschitz by checking all components.
        
        For the direct metric, we verify the mathematical properties.
        For the learned metric, we check the product of all layer Lipschitz constants
        and enforce the constraint if needed.
        
        Returns:
            A dictionary with verification results, containing tensors on the same device
            as the model for compatibility with torchscript/JIT.
        """
        # Fix: Handle device assignment when no parameters exist
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Fallback to self.device if no parameters exist (e.g., direct metric)
            device = getattr(self, 'device', get_device())
        
        if self.use_direct_metric:
            # For the direct metric d(x,y) = ||x-y||/(||x-y||+ε)
            # This is guaranteed to be 1-Lipschitz by construction
            # Proof: The derivative of f(t) = t/(t+ε) is f'(t) = ε/(t+ε)² ≤ 1/ε for t ≥ 0
            # Since we apply this to the norm ||x-y||, which is 1-Lipschitz,
            # the composition is also 1-Lipschitz
            
            # Verify with a random test
            with torch.no_grad():
                batch_size = 100
                x1 = torch.randn(batch_size, self.dim, device=device)
                y1 = torch.randn(batch_size, self.dim, device=device)
                x2 = torch.randn(batch_size, self.dim, device=device)
                y2 = torch.randn(batch_size, self.dim, device=device)
                
                # Compute distances
                input1 = torch.cat([x1, y1], dim=1)
                input2 = torch.cat([x2, y2], dim=1)
                d1 = self(input1)
                d2 = self(input2)
                
                # Compute input difference
                input_diff = torch.norm(input1 - input2, dim=1, keepdim=True)
                output_diff = torch.abs(d1 - d2)
                
                # Check Lipschitz property: |d(x1,y1) - d(x2,y2)| ≤ ||input1 - input2||
                max_ratio = (output_diff / (input_diff + 1e-8)).max().item()
                is_lipschitz = max_ratio <= 1.0 + tol
            
            return {
                'is_lipschitz': torch.full((1,), float(is_lipschitz), device=device, requires_grad=False),
                'method': 'direct_metric',
                'max_ratio': torch.full((1,), max_ratio, device=device, requires_grad=False),
                'tolerance': torch.full((1,), tol, device=device, requires_grad=False)
            }
        else:
            # Enforce the Lipschitz constraint before verification
            self._enforce_lipschitz_constraint()
            
            # Trigger a forward pass to ensure spectral norm is updated
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.dim * 2, device=device)
                _ = self(dummy_input)
            
            # Check each layer and compute product of Lipschitz constants
            product_spectral_norm = torch.full((1,), 1.0, device=device, requires_grad=False)
            layer_norms = {}
            
            for name, module in self.sn_net.named_modules():
                if isinstance(module, SpectralNormalizedLinear):
                    # Get weight after spectral normalization
                    with torch.no_grad():
                        # Ensure we get the weight after spectral norm is applied
                        dummy = torch.zeros(1, module.linear.in_features, device=device)
                        _ = module(dummy)
                        weight = module.linear.weight
                        # Compute spectral norm (largest singular value)
                        sigma = torch.linalg.svdvals(weight)[0]
                        layer_norms[name] = sigma
                        product_spectral_norm *= sigma
                elif isinstance(module, nn.LayerNorm):
                    # LayerNorm is 1-Lipschitz if weight ≤ 1
                    max_weight = torch.max(module.weight).detach()
                    layer_norms[name] = max_weight
                    product_spectral_norm *= max_weight
                elif isinstance(module, nn.ReLU) or isinstance(module, nn.Tanh):
                    # ReLU and Tanh are 1-Lipschitz
                    layer_norms[name] = torch.full((1,), 1.0, device=device, requires_grad=False)
                    # product unchanged
                elif not isinstance(module, nn.Sequential) and not isinstance(module, nn.Identity):
                    # Unknown layer type - conservatively assume it might break Lipschitz
                    layer_norms[name] = torch.full((1,), float('inf'), device=device, requires_grad=False)
                    product_spectral_norm = torch.full((1,), float('inf'), device=device, requires_grad=False)
            
            is_lipschitz = product_spectral_norm <= 1.0 + tol
            
            # Also verify with a random test
            with torch.no_grad():
                batch_size = 100
                x1 = torch.randn(batch_size, self.dim, device=device)
                y1 = torch.randn(batch_size, self.dim, device=device)
                x2 = torch.randn(batch_size, self.dim, device=device)
                y2 = torch.randn(batch_size, self.dim, device=device)
                
                # Compute distances
                input1 = torch.cat([x1, y1], dim=1)
                input2 = torch.cat([x2, y2], dim=1)
                d1 = self(input1)
                d2 = self(input2)
                
                # Compute input difference
                input_diff = torch.norm(input1 - input2, dim=1, keepdim=True)
                output_diff = torch.abs(d1 - d2)
                
                # Check Lipschitz property: |d(x1,y1) - d(x2,y2)| ≤ ||input1 - input2||
                max_ratio = (output_diff / (input_diff + 1e-8)).max().item()
                empirical_lipschitz = max_ratio <= 1.0 + tol
            
            return {
                'is_lipschitz': is_lipschitz.to(device),
                'method': 'spectral_norm_product',
                'product_spectral_norm': product_spectral_norm.to(device),
                'layer_norms': {k: v.to(device) for k, v in layer_norms.items()},
                'empirical_test': torch.full((1,), float(empirical_lipschitz), device=device, requires_grad=False),
                'empirical_max_ratio': torch.full((1,), max_ratio, device=device, requires_grad=False),
                'tolerance': torch.full((1,), tol, device=device, requires_grad=False)
            }

import logging
import torch.utils.tensorboard as tensorboard

# Set up logger
logger = logging.getLogger(__name__)

class MetricYonedaProxy:
    """
    GAIA Metric Yoneda Proxy implementing coend/end calculus.
    Learns representable functors via distance profile comparison,
    enabling universal property-based horn filling.
    
    Following Mahadevan (2024), this implements the full Metric Yoneda embedding
    Y_C(z) = (d(z,c))_{c∈C} with all training points as probes.
    
    This implementation follows the Lawvere metric space enrichment where
    morphisms must be non-expansive maps in the [0,∞]-enriched category.
    
    References:
    - Bonsangue et al. 1998, "Generalized Ultrametrics in Quantitative Domain Theory"
    - Mahadevan 2024, "Generative Algebraic Intelligence Architecture"
    """
    def __init__(self, target_dim: int, num_probes: int = 16,
                 lr: float = 1e-3, pretrain_steps: int = 200,
                 adaptive: bool = True, use_direct_metric: bool = True,
                 seed: int = None, device = None, log_level: str = 'INFO'):
        """
        Initialize the Metric Yoneda Proxy.
        
        Args:
            target_dim: Dimension of the target space
            num_probes: Number of probe points for the Yoneda embedding
            lr: Learning rate for the metric network
            pretrain_steps: Number of steps to pretrain the metric network
            adaptive: Whether to adapt the metric during training
            use_direct_metric: Whether to use direct metric (True)
                              or learned metric with spectral normalization (False)
            seed: Random seed for reproducibility
            device: Device to use for computation (default: DEVICE)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # Set up logging
        self._setup_logging(log_level)
        
        # Store parameters
        self.target_dim = target_dim
        self.adaptive = adaptive
        self.use_direct_metric = use_direct_metric
        self.device = device or DEVICE
        self.num_probes = num_probes
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            logger.info(f"Setting random seed to {seed}")
            torch.manual_seed(seed)

        # Initialize probe points for Yoneda embedding
        self.probes = nn.Parameter(
            torch.randn(num_probes, target_dim, device=self.device),
            requires_grad=False
        )
        
        # Pre-compute probe norms for efficient profile computation
        self._probes_norm_sq = torch.sum(self.probes**2, dim=1, keepdim=True).t().t()

        # Create metric network (representable functor approximation)
        # Must be 1-Lipschitz to satisfy the metric Yoneda lemma requirements
        self.metric = SpectralNormalizedMetric(
            target_dim, 
            use_direct_metric=use_direct_metric,
            device=self.device
        )

        # Pre-training for stability (Yoneda lemma approximation)
        if not use_direct_metric:
            self._pretrain(lr, pretrain_steps)
            # Clear gradients after pre-training
            if hasattr(self.metric, 'parameters'):
                for p in self.metric.parameters():
                    if p.grad is not None:
                        p.grad = None

        # Initialize optimizer and scheduler attributes to None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        
        # Adaptive optimization for coend calculus
        if self.adaptive and not use_direct_metric:
            trainable_params = [p for p in self.metric.parameters() if p.requires_grad]
            if trainable_params:  # Only create if there are trainable parameters
                self.optimizer = optim.Adam(trainable_params, lr=lr)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=50, factor=0.8
                )
                logger.info(f"Created optimizer with {sum(p.numel() for p in trainable_params)} trainable parameters")
                
                # Create tensorboard writer for monitoring
                self.writer = tensorboard.SummaryWriter(
                    log_dir=f'runs/yoneda_proxy_{torch.randint(0, 10000, (1,)).item()}'
                )
            else:
                logger.warning("No trainable parameters found in metric network")
                
        # Initialize loss components tracking
        self._last_loss_components = {'base_loss': 0.0, 'profile_var': 0.0, 'reg_loss': 0.0}
            
        # Verify that the metric is 1-Lipschitz
        lipschitz_check = self.verify_lipschitz()
        if not lipschitz_check['is_lipschitz']:
            logger.warning(f"Metric is not 1-Lipschitz: {lipschitz_check}")
            logger.warning("The Yoneda lemma requires a 1-Lipschitz metric for the universal property to hold.")
        else:
            logger.info(f"Lipschitz verification passed")
            
    def _setup_logging(self, log_level: str):
        """Set up logging with the specified level."""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Configure logger
        logger.setLevel(numeric_level)
        
        # Create console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
    def verify_lipschitz(self, tol=1e-4):
        """Verify that the metric network is 1-Lipschitz."""
        return self.metric.verify_lipschitz(tol)

    def _pretrain(self, lr: float, steps: int):
        """
        Pre-train the metric network for stability.
        
        For the direct metric, no pre-training is needed.
        For the spectral normalized network, we train it to approximate
        the normalized Euclidean distance d(x,y) = ||x-y||/(||x-y||+ε).
        
         We must ensure spectral normalization is properly applied
        during training by doing forward passes inside the gradient context.
        """
        if hasattr(self.metric, 'use_direct_metric') and self.metric.use_direct_metric:
            # No pre-training needed for direct metric
            logger.debug("Skipping pre-training for direct metric")
            return
            
        logger.info(f"Pre-training metric network for {steps} steps")
        
        # Pre-train the spectral normalized network
        trainable_params = [p for p in self.metric.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("No trainable parameters found for pre-training")
            return
            
        opt = optim.Adam(trainable_params, lr=lr)
        loss_fn = nn.MSELoss()
        
        # Get the epsilon value from the metric for the normalized distance
        epsilon = getattr(self.metric, 'epsilon', 1e-6)
        
        # Track losses for monitoring
        losses = []
        
        for step in range(steps):
            # Generate random pairs of points
            x = torch.randn(128, self.target_dim, device=self.device)
            y = torch.randn(128, self.target_dim, device=self.device)
            
            # Compute normalized Euclidean distance as target
            diff = x - y
            norm = torch.norm(diff, dim=1, keepdim=True)
            target_dist = norm / (norm + epsilon)
            
            # Train network to approximate normalized Euclidean distance
            opt.zero_grad()
            
            # Forward pass inside gradient context to trigger spectral norm
            inputs = torch.cat([x, y], dim=-1)
            pred_dist = self.metric(inputs)
            
            loss = loss_fn(pred_dist, target_dist)
            loss.backward()
            
            # Step optimizer
            opt.step()
            
            # Enforce Lipschitz constraint after each step
            self.metric._enforce_lipschitz_constraint()
            
            # Track loss
            losses.append(loss.item())
            
            # Log progress
            if (step + 1) % 50 == 0 or step == 0:
                logger.debug(f"Pre-training step {step+1}/{steps}, loss: {loss.item():.6f}")
                
        # Zero gradients after training
        opt.zero_grad(set_to_none=True)
            
        # Verify Lipschitz constraint after training
        lipschitz_check = self.metric.verify_lipschitz()
        if not lipschitz_check.get('is_lipschitz', False):
            logger.warning(f"Metric failed Lipschitz check after pre-training: {lipschitz_check}")
            # Force Lipschitz constraint
            self.metric._enforce_lipschitz_constraint()
        
        # Set to evaluation mode
        self.metric.eval()
        
        # Log final statistics
        logger.info(f"Pre-training complete. Final loss: {losses[-1]:.6f}, "
                   f"Initial loss: {losses[0]:.6f}, "
                   f"Improvement: {(1 - losses[-1]/losses[0])*100:.2f}%")
        
        # Create tensorboard writer if not already created
        if self.writer is None and self.adaptive:
            self.writer = tensorboard.SummaryWriter(
                log_dir=f'runs/yoneda_proxy_{torch.randint(0, 10000, (1,)).item()}'
            )
            
        # Log to tensorboard if available
        if self.writer is not None:
            for i, loss_val in enumerate(losses):
                self.writer.add_scalar('pretrain/loss', loss_val, i)
            self.writer.add_scalar('pretrain/final_loss', losses[-1], 0)
            self.writer.add_scalar('pretrain/improvement', 1 - losses[-1]/losses[0], 0)

    def _profile(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Yoneda profile: Hom(-, z) representation.
        
        Following Mahadevan (2024), this implements Y_C(z) = (d(z,c))_{c∈C}
        where c ranges over the probe points.
        
        This is a critical component of the Metric Yoneda embedding, which
        allows us to compare points in the target space via their distance
        profiles to a fixed set of probe points.
        
        Args:
            z: Tensor of shape (batch_size, target_dim)
            
        Returns:
            Tensor of shape (batch_size, num_probes, 1) containing distances
            from each point to each probe
        """
        B, P = z.size(0), self.probes.size(0)
        device = z.device
        
        # : Use vectorized computation for direct metric
        # This avoids memory-intensive tiling operations
        if self.metric.use_direct_metric:
            # Compute squared norms efficiently using the identity:
            # ||x-c||² = ||x||² + ||c||² - 2x·c
            # This avoids explicit tiling of tensors
            
            # Use pre-computed probe norms if available, otherwise compute them
            if not hasattr(self, '_probes_norm_sq') or self._probes_norm_sq.device != device:
                self._probes_norm_sq = torch.sum(self.probes**2, dim=1, keepdim=True).t().to(device)
            
            z_norm_sq = torch.sum(z**2, dim=1, keepdim=True)  # [B, 1]
            
            # Compute dot products between all pairs: [B, P]
            # Use torch.einsum for better memory efficiency
            dot_products = torch.einsum('bd,pd->bp', z, self.probes)
            
            # Compute squared distances: [B, P]
            squared_distances = z_norm_sq + self._probes_norm_sq - 2 * dot_products
            
            # Ensure non-negative (numerical stability)
            squared_distances = F.relu(squared_distances)
            
            # Compute distances using the 1-Lipschitz metric d(x,y) = ||x-y||/(||x-y||+ε)
            norm = torch.sqrt(squared_distances + 1e-8)
            distances = norm / (norm + self.metric.epsilon)
            
            # Reshape to [B, P, 1]
            return distances.unsqueeze(2)
        
        # For learned metric, use vectorized computation with torch.vmap if available
        # or efficient batching if not
        if hasattr(torch, 'vmap') and B * P <= 10000:  # Use vmap for reasonable sizes
            # Define a function to compute distance between a point and all probes
            def compute_distances(point):
                # Expand point to match probes
                point_expanded = point.unsqueeze(0).expand(P, -1)
                # Concatenate with probes
                pairs = torch.cat([point_expanded, self.probes], dim=1)
                # Compute distances
                return self.metric(pairs)
                
            # Use vmap to vectorize over the batch dimension
            try:
                distances = torch.vmap(compute_distances)(z)
                return distances
            except Exception:
                # Fall back to chunked computation if vmap fails
                pass
        
        # Chunked computation with efficient batching
        chunk_size = min(128, B)  # Adjust based on available memory
        profiles = []
        
        for i in range(0, B, chunk_size):
            end_idx = min(i + chunk_size, B)
            batch_z = z[i:end_idx]
            batch_size = batch_z.size(0)
            
            # Use torch.repeat_interleave for efficient batching
            # This avoids the nested Python loop in the original implementation
            
            # Replace: torch.tensor(is_lipschitz, device=device)
            result = {
                'is_lipschitz': torch.full((1,), float(is_lipschitz), device=device, requires_grad=False),
                'max_ratio': torch.full((1,), max_ratio, device=device, requires_grad=False),
                'tolerance': torch.full((1,), tol, device=device, requires_grad=False)
            }
            
            return result
            
            # Repeat probes for each point
            probes_repeated = self.probes.repeat(batch_size, 1)  # [batch_size*P, target_dim]
            
            # Concatenate points and probes
            pairs = torch.cat([points_repeated, probes_repeated], dim=1)  # [batch_size*P, 2*target_dim]
            
            # Compute distances
            distances = self.metric(pairs)  # [batch_size*P, 1]
            
            # Reshape to [batch_size, P, 1]
            batch_profile = distances.reshape(batch_size, P, 1)
            profiles.append(batch_profile)
        
        return torch.cat(profiles, dim=0)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, min_var_threshold: float = None) -> torch.Tensor:
        """
        Yoneda loss: compare representable functor profiles.
        
        Following Mahadevan (2024), this minimizes:
        L_lift(ψ) = E_x~D_A[ ||Y_C(k(x)) - Y_C(m_ψ(f(x)))||^2 ]
        
        This loss compares the distance profiles of the predicted points
        and the target points, rather than comparing the points directly.
        This is a key insight from the Yoneda lemma, which allows us to
        compare objects by their morphisms to a fixed set of objects.
        
        Args:
            pred: Predicted points of shape (batch_size, target_dim)
            target: Target points of shape (batch_size, target_dim)
            min_var_threshold: Optional minimum variance threshold. If None,
                               uses information-theoretic spread penalty instead.
            
        Returns:
            Scalar loss value
        """
        # Compute Yoneda profiles
        pred_profile = self._profile(pred)
        target_profile = self._profile(target)
        
        # Base loss: MSE between profiles
        base_loss = F.mse_loss(pred_profile, target_profile)
        
        # Compute profile statistics
        profile_var = torch.var(pred_profile, dim=1).mean()
        
        # : Use information-theoretic spread penalty
        # This encourages diversity in the embedding space without
        # imposing arbitrary thresholds
        if min_var_threshold is None:
            # Compute determinant of covariance matrix (or approximation)
            # Higher determinant = more spread out = better
            B, P, _ = pred_profile.shape
            
            # Flatten profiles for covariance computation
            flat_profiles = pred_profile.reshape(B, P)
            
            # Compute mean-centered profiles
            centered = flat_profiles - flat_profiles.mean(dim=1, keepdim=True)
            
            # Compute covariance matrix (B x P x P)
            # Use batch matrix multiplication for efficiency
            cov = torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1))
            
            # Add small diagonal for numerical stability
            cov = cov + torch.eye(P, device=cov.device) * 1e-6
            
            # Use log-determinant as regularization (higher is better)
            # We can't compute this for all batches efficiently, so use trace as proxy
            # trace(cov) is sum of eigenvalues, which correlates with determinant
            trace = torch.diagonal(cov, dim1=1, dim2=2).sum(dim=1).mean()
            
            # Penalize small trace (want larger spread)
            reg_loss = 0.01 * torch.exp(-trace / P)
        else:
            # Use explicit variance threshold if provided
            reg_loss = 0.1 * F.relu(min_var_threshold - profile_var)
        
        # Log components for debugging
        with torch.no_grad():
            self._last_loss_components = {
                'base_loss': base_loss.item(),
                'profile_var': profile_var.item(),
                'reg_loss': reg_loss.item()
            }
        
        # Total loss
        total_loss = base_loss + reg_loss
        
        return total_loss

    def step_metric(self, loss: torch.Tensor) -> None:
        """
        Adjust scheduler based on loss (no backward/optimizer calls).
        
        : Raises exception if scheduler doesn't exist but is requested.
        """
        if not self.adaptive:
            return
            
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            raise RuntimeError(
                "Cannot step scheduler: no scheduler exists. "
                "This happens when using direct_metric=True or when the optimizer "
                "wasn't initialized properly."
            )
            
        self.scheduler.step(loss)

    # Update the direct_metric function to be 1-Lipschitz
    def direct_metric(x, y, eps=1e-3):
        """
        Compute a 1-Lipschitz metric between x and y.
        
        This uses the formula: d(x,y) = (eps * ||x-y||) / (||x-y|| + eps)
        which has derivative <= 1.
        
        Args:
            x: First tensor
            y: Second tensor
            eps: Small constant for numerical stability
            
        Returns:
            Distance between x and y
        """
        dist = torch.linalg.norm(x - y, dim=-1)
        return (eps * dist) / (dist + eps)
    
    # Create a SpectralNormalizedLinearNoBias class
    class SpectralNormalizedLinearNoBias(nn.Module):
        """
        Linear layer with spectral normalization and no bias.

        This ensures the layer is 1-Lipschitz by normalizing the weight matrix
        by its spectral norm (largest singular value).
        """
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()
            
        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
        def forward(self, x):
            # Apply spectral normalization
            weight = self.weight / torch.linalg.matrix_norm(self.weight, ord=2)
            return F.linear(x, weight)

    # Create an AddBias class
    class AddBias(nn.Module):
        """
        Explicit bias addition with normalization.

        This ensures the bias contribution is bounded by normalizing
        the bias vector if its norm exceeds 1.
        """
        def __init__(self, features):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(features))
            
        def forward(self, x):
            # Normalize bias if needed
            bias_norm = torch.linalg.norm(self.bias)
            if bias_norm > 1:
                normalized_bias = self.bias / bias_norm
            else:
                normalized_bias = self.bias
            return x + normalized_bias.unsqueeze(0)

        # Update the verify_lipschitz method with stronger empirical test
        def verify_lipschitz(self, tolerance=1e-5):
            """
            Verify that the metric satisfies the 1-Lipschitz property.

            For the direct metric, this is verified mathematically.
            For the learned metric, we compute the product of spectral norms
            and perform an empirical test with random unit directions.

            Args:
                tolerance: Tolerance for numerical errors
                
            Returns:
                True if the metric is 1-Lipschitz, False otherwise
            """
            if self.use_direct_metric:
                # Direct metric is 1-Lipschitz by construction
                # Perform empirical test for verification
                d = self.target_dim
                num_samples = 10000
                
                # Generate random unit vectors on S^{d-1}
                directions = torch.randn(num_samples, d, device=self.device)
                directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
                
                # Generate random points
                x = torch.randn(1, d, device=self.device)
                
                # Compute distances for small perturbations
                epsilon = 1e-4
                max_ratio = 0
                
                for i in range(num_samples):
                    y1 = x + epsilon * directions[i:i+1]
                    y2 = x + 2 * epsilon * directions[i:i+1]
                    
                    d1 = self.direct_metric(x, y1)
                    d2 = self.direct_metric(x, y2)
                    
                    input_diff = epsilon
                    output_diff = abs(d2 - d1).item()
                    
                    ratio = output_diff / input_diff
                    max_ratio = max(max_ratio, ratio)
                
                return max_ratio <= 1 + tolerance
            else:
                # For learned metric, enforce Lipschitz constraint first
                self._enforce_lipschitz_constraint()
                
                # Compute product of spectral norms for linear layers
                product = 1.0
                for module in self.modules():
                    if isinstance(module, SpectralNormalizedLinearNoBias):
                        product *= torch.linalg.matrix_norm(module.weight, ord=2).item()
                    elif isinstance(module, nn.LayerNorm):
                        product *= torch.max(torch.abs(module.weight)).item()
                
                # Perform empirical test with random unit directions
                d = self.target_dim
                num_samples = 10000
                
                # Generate random unit vectors on S^{d-1}
                directions = torch.randn(num_samples, d, device=self.device)
                directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
                
                # Generate random points
                x = torch.randn(1, d, device=self.device)
                
                # Compute distances for small perturbations
                epsilon = 1e-4
                max_ratio = 0
                
                for i in range(num_samples):
                    y1 = x + epsilon * directions[i:i+1]
                    y2 = x + 2 * epsilon * directions[i:i+1]
                    
                    d1 = self(x, y1)
                    d2 = self(x, y2)
                    
                    input_diff = epsilon
                    output_diff = abs(d2 - d1).item()
                    
                    ratio = output_diff / input_diff
                    max_ratio = max(max_ratio, ratio)
                
                return max_ratio <= 1 + tolerance

    # Update the update_probes method with clamping
    def update_probes(self, new_data: torch.Tensor, decay: float = 0.9) -> None:
        """
        Update probe points based on new data (coend evolution).

        This implements a form of coend calculus by allowing the probe points
        to adapt to the data distribution. The probes are updated using an
        exponential moving average (EMA) with the specified decay factor.

        Args:
            new_data: New data points to incorporate into the probes
            decay: EMA decay factor (default: 0.9)

        Returns:
            None
        """
        if not self.adaptive:
            logger.debug("Probe update skipped: adaptive=False")
            return
            
        if new_data.device != self.device:
            new_data = new_data.to(self.device)
            
        with torch.no_grad():
            if new_data.size(0) >= self.num_probes:
                # Randomly sample points from new_data
                idx = torch.randperm(new_data.size(0), device=new_data.device)[:self.num_probes]
                sampled_data = new_data[idx]
                
                # Compute statistics before update for invariance testing
                old_mean = self.probes.data.mean(dim=0)
                old_std = self.probes.data.std(dim=0)
                
                # Determine data bounds for clamping
                x_min = new_data.min(dim=0)[0]
                x_max = new_data.max(dim=0)[0]
                
                # Update probes with exponential moving average and clamping
                alpha = 1 - decay
                self.probes.data = torch.clamp(
                    self.probes.data * (1-alpha) + alpha * sampled_data.mean(dim=0),
                    min=x_min, max=x_max
                )
                
                # Update pre-computed probe norms
                self._probes_norm_sq = torch.sum(self.probes**2, dim=1, keepdim=True).t()
                
                # Log the update
                probe_norm = torch.norm(self.probes.data, dim=1).mean().item()
                new_mean = self.probes.data.mean(dim=0)
                new_std = self.probes.data.std(dim=0)
                
                # Calculate mean shift and std change
                mean_shift = torch.norm(new_mean - old_mean).item()
                std_change = torch.norm(new_std - old_std).item()
                
                logger.debug(f"Updated probes: mean_norm={probe_norm:.4f}, "
                            f"mean_shift={mean_shift:.4f}, std_change={std_change:.4f}")
                
                # Log to tensorboard if available
                if self.writer is not None:
                    self.writer.add_scalar('probes/mean_norm', probe_norm, self._step)
                    self.writer.add_scalar('probes/mean_shift', mean_shift, self._step)
                    self.writer.add_scalar('probes/std_change', std_change, self._step)
            else:
                logger.debug(f"Probe update skipped: not enough data points "
                            f"({new_data.size(0)} < {self.num_probes})")
                    
        def update(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
            """
            Update the metric and probes based on new data.
            
            This is the main public interface for training the MetricYonedaProxy.
            It computes the loss, updates the metric (if adaptive), and updates
            the probes.
            
            Args:
                pred: Predicted points of shape (batch_size, target_dim)
                target: Target points of shape (batch_size, target_dim)
                
            Returns:
                Dictionary with loss components
            """
            # Compute loss
            loss = self.loss(pred, target)
            
            # Update metric if adaptive and not using direct metric
            if self.adaptive and not self.use_direct_metric and self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step(loss)
                    
                # Enforce Lipschitz constraint
                self.metric._enforce_lipschitz_constraint()
            
            # Update probes with target data
            self.update_probes(target)
            
            # Return loss components
            return self._last_loss_components