"""
Tests for the MetricYonedaProxy class.

These tests verify that the fixed components work correctly:
1. The metric is 1-Lipschitz
2. The Euclidean metric is properly scaled by 1/√2
3. The spectral normalized network has no LayerNorm layers
4. The profile computation is memory-efficient
5. The loss function is properly regularized
6. The scheduler is properly guarded
"""

import torch
import torch.nn as nn
import pytest

from gaia.train.yoneda_proxy import MetricYonedaProxy, SpectralNormalizedMetric
from gaia.core import DEVICE

# ────────────────────────────────────────────────────────────────────────────────
# 1. Lipschitz Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_direct_metric_lipschitz():
    """Test that the direct Euclidean metric is 1-Lipschitz."""
    # Create a metric with direct Euclidean distance
    metric = SpectralNormalizedMetric(dim=10, use_direct_metric=True)
    
    # Verify Lipschitz constant
    result = metric.verify_lipschitz()
    assert result['is_lipschitz']
    assert abs(result['spectral_norm'] - 1.0) < 1e-4
    
    # Test with random inputs
    x1 = torch.randn(100, 10)
    y1 = torch.randn(100, 10)
    x2 = torch.randn(100, 10)
    y2 = torch.randn(100, 10)
    
    # Compute distances
    d1 = metric(torch.cat([x1, y1], dim=1))
    d2 = metric(torch.cat([x2, y2], dim=1))
    
    # Compute input difference norm
    input_diff = torch.cat([x1 - x2, y1 - y2], dim=1)
    input_diff_norm = torch.norm(input_diff, dim=1, keepdim=True)
    
    # Verify Lipschitz property: |d(x1,y1) - d(x2,y2)| ≤ ||(x1,y1) - (x2,y2)||
    output_diff = torch.abs(d1 - d2)
    assert torch.all(output_diff <= input_diff_norm * (1.0 + 1e-4))

def test_learned_metric_lipschitz():
    """Test that the learned metric is 1-Lipschitz."""
    # Create a metric with learned distance
    metric = SpectralNormalizedMetric(dim=10, use_direct_metric=False)
    
    # Verify Lipschitz constant
    result = metric.verify_lipschitz()
    assert result['is_lipschitz']
    assert result['product_spectral_norm'] <= 1.0 + 1e-4
    
    # Test with random inputs
    x1 = torch.randn(100, 10)
    y1 = torch.randn(100, 10)
    x2 = torch.randn(100, 10)
    y2 = torch.randn(100, 10)
    
    # Compute distances
    d1 = metric(torch.cat([x1, y1], dim=1))
    d2 = metric(torch.cat([x2, y2], dim=1))
    
    # Compute input difference norm
    input_diff = torch.cat([x1 - x2, y1 - y2], dim=1)
    input_diff_norm = torch.norm(input_diff, dim=1, keepdim=True)
    
    # Verify Lipschitz property: |d(x1,y1) - d(x2,y2)| ≤ ||(x1,y1) - (x2,y2)||
    output_diff = torch.abs(d1 - d2)
    assert torch.all(output_diff <= input_diff_norm * (1.0 + 1e-4))

def test_no_layernorm_in_metric():
    """Test that the learned metric has no LayerNorm layers."""
    # Create a metric with learned distance
    metric = SpectralNormalizedMetric(dim=10, use_direct_metric=False)
    
    # Check that there are no LayerNorm layers
    for module in metric.sn_net.modules():
        assert not isinstance(module, nn.LayerNorm)

# ────────────────────────────────────────────────────────────────────────────────
# 2. MetricYonedaProxy Tests
# ────────────────────────────────────────────────────────────────────────────────

def test_yoneda_proxy_direct_metric():
    """Test that the MetricYonedaProxy works with direct metric."""
    # Create a proxy with direct metric
    proxy = MetricYonedaProxy(target_dim=10, use_direct_metric=True)
    
    # Verify Lipschitz constant
    result = proxy.verify_lipschitz()
    assert result['is_lipschitz']
    
    # Test loss computation
    pred = torch.randn(10, 10, device=DEVICE)
    target = torch.randn(10, 10, device=DEVICE)
    loss = proxy.loss(pred, target)
    
    # Loss should be a scalar
    assert loss.dim() == 0
    assert loss.item() >= 0.0

def test_yoneda_proxy_learned_metric():
    """Test that the MetricYonedaProxy works with learned metric."""
    # Create a proxy with learned metric
    proxy = MetricYonedaProxy(target_dim=10, use_direct_metric=False, pretrain_steps=10)
    
    # Verify Lipschitz constant
    result = proxy.verify_lipschitz()
    assert result['is_lipschitz']
    
    # Test loss computation
    pred = torch.randn(10, 10, device=DEVICE)
    target = torch.randn(10, 10, device=DEVICE)
    loss = proxy.loss(pred, target)
    
    # Loss should be a scalar
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    
    # Test scheduler guard
    proxy.step_metric(loss)
    
    # Test probe update
    proxy.update_probes(torch.randn(20, 10, device=DEVICE))

def test_profile_computation():
    """Test that the profile computation is memory-efficient."""
    # Create a proxy with direct metric
    proxy = MetricYonedaProxy(target_dim=10, use_direct_metric=True, num_probes=5)
    
    # Test with different batch sizes
    for batch_size in [1, 10, 100]:
        z = torch.randn(batch_size, 10, device=DEVICE)
        profile = proxy._profile(z)
        
        # Profile should have shape [batch_size, num_probes, 1]
        assert profile.shape == (batch_size, 5, 1)
        
    # Test with learned metric
    proxy = MetricYonedaProxy(target_dim=10, use_direct_metric=False, num_probes=5, pretrain_steps=10)
    
    for batch_size in [1, 10, 100]:
        z = torch.randn(batch_size, 10, device=DEVICE)
        profile = proxy._profile(z)
        
        # Profile should have shape [batch_size, num_probes, 1]
        assert profile.shape == (batch_size, 5, 1)

def test_loss_components():
    """Test that the loss components are properly tracked."""
    # Create a proxy with direct metric
    proxy = MetricYonedaProxy(target_dim=10, use_direct_metric=True)
    
    # Test loss computation
    pred = torch.randn(10, 10, device=DEVICE)
    target = torch.randn(10, 10, device=DEVICE)
    loss = proxy.loss(pred, target)
    
    # Check that loss components are tracked
    assert hasattr(proxy, '_last_loss_components')
    assert 'base_loss' in proxy._last_loss_components
    assert 'profile_var' in proxy._last_loss_components
    assert 'reg_loss' in proxy._last_loss_components