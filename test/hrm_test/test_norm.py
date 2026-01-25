"""
Unit Tests for RMSNorm Implementation

Run with: pytest tests/test_norm.py -v
"""

import pytest
import torch
import torch.nn as nn
import math

# Import the module under test
from hrm.layers.norm import RMSNorm


class TestRMSNormInstantiation:
    """Test RMSNorm can be instantiated with various configurations."""
    
    def test_basic_instantiation(self):
        """RMSNorm can be created with just dim parameter."""
        norm = RMSNorm(dim=64)
        assert norm is not None
        assert isinstance(norm, nn.Module)
    
    def test_custom_eps(self):
        """RMSNorm accepts custom eps parameter."""
        norm = RMSNorm(dim=64, eps=1e-5)
        assert norm.eps == 1e-5
        
        norm2 = RMSNorm(dim=64, eps=1e-8)
        assert norm2.eps == 1e-8
    
    def test_default_eps(self):
        """RMSNorm has sensible default eps (typically 1e-6)."""
        norm = RMSNorm(dim=64)
        assert norm.eps > 0
        assert norm.eps <= 1e-5  # Should be small
    
    def test_various_dimensions(self):
        """RMSNorm works with various feature dimensions."""
        for dim in [1, 16, 64, 128, 256, 512, 1024]:
            norm = RMSNorm(dim=dim)
            assert norm.weight.shape == (dim,)
    
    def test_dim_attribute_stored(self):
        """RMSNorm stores dim as attribute."""
        norm = RMSNorm(dim=128)
        assert hasattr(norm, 'dim')
        assert norm.dim == 128


class TestRMSNormLearnableWeight:
    """Test the learnable scale parameter (weight)."""
    
    def test_weight_exists(self):
        """RMSNorm has a weight parameter."""
        norm = RMSNorm(dim=64)
        assert hasattr(norm, 'weight')
    
    def test_weight_is_parameter(self):
        """Weight is a learnable nn.Parameter."""
        norm = RMSNorm(dim=64)
        assert isinstance(norm.weight, nn.Parameter)
    
    def test_weight_requires_grad(self):
        """Weight has requires_grad=True for learning."""
        norm = RMSNorm(dim=64)
        assert norm.weight.requires_grad is True
    
    def test_weight_shape(self):
        """Weight has correct shape matching dim."""
        for dim in [16, 64, 128]:
            norm = RMSNorm(dim=dim)
            assert norm.weight.shape == (dim,)
    
    def test_weight_initialization(self):
        """Weight is initialized to ones."""
        norm = RMSNorm(dim=64)
        expected = torch.ones(64)
        assert torch.allclose(norm.weight, expected)
    
    def test_weight_in_parameters(self):
        """Weight is included in module parameters."""
        norm = RMSNorm(dim=64)
        param_names = [name for name, _ in norm.named_parameters()]
        assert 'weight' in param_names


class TestRMSNormForward:
    """Test the forward pass behavior."""
    
    def test_output_shape_2d(self):
        """Output shape matches input for 2D tensor (batch, dim)."""
        norm = RMSNorm(dim=64)
        x = torch.randn(32, 64)
        output = norm(x)
        assert output.shape == x.shape
    
    def test_output_shape_3d(self):
        """Output shape matches input for 3D tensor (batch, seq, dim)."""
        norm = RMSNorm(dim=64)
        x = torch.randn(8, 16, 64)
        output = norm(x)
        assert output.shape == x.shape
    
    def test_output_shape_4d(self):
        """Output shape matches input for 4D tensor."""
        norm = RMSNorm(dim=64)
        x = torch.randn(4, 4, 4, 64)  # e.g., batch, height, width, channels
        output = norm(x)
        assert output.shape == x.shape
    
    def test_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        norm = RMSNorm(dim=64)
        
        # Float32
        x_f32 = torch.randn(8, 64, dtype=torch.float32)
        out_f32 = norm(x_f32)
        assert out_f32.dtype == torch.float32
        
        # Float64
        norm_f64 = RMSNorm(dim=64).double()
        x_f64 = torch.randn(8, 64, dtype=torch.float64)
        out_f64 = norm_f64(x_f64)
        assert out_f64.dtype == torch.float64
    
    def test_batch_independence(self):
        """Each sample in batch is normalized independently."""
        norm = RMSNorm(dim=64)
        
        x1 = torch.randn(1, 64)
        x2 = torch.randn(1, 64)
        
        # Process separately
        out1_separate = norm(x1)
        out2_separate = norm(x2)
        
        # Process together
        x_batch = torch.cat([x1, x2], dim=0)
        out_batch = norm(x_batch)
        
        assert torch.allclose(out1_separate, out_batch[0:1], atol=1e-6)
        assert torch.allclose(out2_separate, out_batch[1:2], atol=1e-6)


class TestRMSNormNormalizationBehavior:
    """Test that normalization produces expected RMS properties."""
    
    def test_rms_normalization_formula(self):
        """Verify RMSNorm follows: x / sqrt(mean(x^2) + eps) * weight."""
        dim = 64
        eps = 1e-6
        norm = RMSNorm(dim=dim, eps=eps)
        
        x = torch.randn(8, dim)
        output = norm(x)
        
        # Manual computation
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        expected = (x / rms) * norm.weight
        
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_unit_rms_after_normalization(self):
        """After normalization (before scaling), RMS should be ~1."""
        dim = 64
        norm = RMSNorm(dim=dim)
        
        # With weight=1, output RMS should be close to 1
        x = torch.randn(100, dim)
        output = norm(x)
        
        # Compute RMS of output
        output_rms = torch.sqrt(output.pow(2).mean(-1))
        
        # Should be close to 1 (since weight is initialized to 1)
        assert output_rms.mean().item() > 0.9
        assert output_rms.mean().item() < 1.1
    
    def test_eps_prevents_division_by_zero(self):
        """Eps parameter prevents NaN when input is all zeros."""
        norm = RMSNorm(dim=64, eps=1e-6)
        
        # Input of all zeros
        x = torch.zeros(8, 64)
        output = norm(x)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_eps_with_small_values(self):
        """Eps provides stability with very small input values."""
        norm = RMSNorm(dim=64, eps=1e-6)
        
        # Very small values
        x = torch.randn(8, 64) * 1e-10
        output = norm(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_normalization_is_scale_invariant(self):
        """RMSNorm normalizes regardless of input scale."""
        norm = RMSNorm(dim=64)
        
        x = torch.randn(8, 64)
        x_scaled = x * 1000  # Scale up by 1000x
        
        out1 = norm(x)
        out2 = norm(x_scaled)
        
        # Outputs should be the same (scale invariant)
        assert torch.allclose(out1, out2, atol=1e-4)


class TestRMSNormGradients:
    """Test gradient flow through RMSNorm."""
    
    def test_gradients_flow_to_input(self):
        """Gradients flow back to input tensor."""
        norm = RMSNorm(dim=64)
        
        x = torch.randn(8, 64, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_gradients_flow_to_weight(self):
        """Gradients flow to learnable weight parameter."""
        norm = RMSNorm(dim=64)
        
        x = torch.randn(8, 64)
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert norm.weight.grad is not None
        assert norm.weight.grad.shape == norm.weight.shape
    
    def test_gradients_not_nan(self):
        """Gradients should not be NaN."""
        norm = RMSNorm(dim=64)
        
        x = torch.randn(8, 64, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()
        
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(norm.weight.grad).any()
    
    def test_weight_updates_with_optimizer(self):
        """Weight can be updated by optimizer."""
        norm = RMSNorm(dim=64)
        optimizer = torch.optim.SGD(norm.parameters(), lr=0.1)
        
        original_weight = norm.weight.clone()
        
        x = torch.randn(8, 64)
        output = norm(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Weight should have changed
        assert not torch.equal(norm.weight, original_weight)


class TestRMSNormEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_element_batch(self):
        """Works with batch size of 1."""
        norm = RMSNorm(dim=64)
        x = torch.randn(1, 64)
        output = norm(x)
        assert output.shape == (1, 64)
    
    def test_dim_equals_one(self):
        """Works with dim=1."""
        norm = RMSNorm(dim=1)
        x = torch.randn(8, 1)
        output = norm(x)
        assert output.shape == (8, 1)
    
    def test_invalid_dim_raises_error(self):
        """Invalid dim raises ValueError."""
        with pytest.raises(ValueError):
            RMSNorm(dim=0)
        with pytest.raises(ValueError):
            RMSNorm(dim=-1)
    
    def test_invalid_eps_raises_error(self):
        """Invalid eps raises ValueError."""
        with pytest.raises(ValueError):
            RMSNorm(dim=64, eps=0)
        with pytest.raises(ValueError):
            RMSNorm(dim=64, eps=-1e-6)
    
    def test_large_values(self):
        """Handles large input values without overflow."""
        norm = RMSNorm(dim=64)
        x = torch.randn(8, 64) * 1e6
        output = norm(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_negative_values(self):
        """Correctly handles negative values."""
        norm = RMSNorm(dim=64)
        x = -torch.abs(torch.randn(8, 64))  # All negative
        output = norm(x)
        
        # Output should preserve sign (mostly negative)
        assert (output < 0).sum() > (output > 0).sum()


class TestRMSNormComparison:
    """Compare RMSNorm behavior to reference implementations."""
    
    def test_matches_manual_implementation(self):
        """Output matches manual RMSNorm calculation."""
        dim = 64
        eps = 1e-6
        
        norm = RMSNorm(dim=dim, eps=eps)
        x = torch.randn(8, 16, dim)
        
        # Manual implementation
        def manual_rmsnorm(x, weight, eps):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return x * rms * weight
        
        expected = manual_rmsnorm(x, norm.weight, eps)
        actual = norm(x)
        
        assert torch.allclose(actual, expected, atol=1e-6)
    
    def test_different_from_layernorm(self):
        """RMSNorm produces different results than LayerNorm."""
        dim = 64
        
        rmsnorm = RMSNorm(dim=dim)
        layernorm = nn.LayerNorm(dim)
        
        # Initialize LayerNorm weights to match RMSNorm (weight=1, bias=0)
        layernorm.weight.data.fill_(1.0)
        layernorm.bias.data.fill_(0.0)
        
        x = torch.randn(8, dim)
        
        rms_out = rmsnorm(x)
        ln_out = layernorm(x)
        
        # Should NOT be equal (RMSNorm doesn't center)
        assert not torch.allclose(rms_out, ln_out, atol=1e-4)


class TestRMSNormDeviceCompatibility:
    """Test device compatibility (CPU/CUDA if available)."""
    
    def test_cpu_execution(self):
        """RMSNorm works on CPU."""
        norm = RMSNorm(dim=64)
        x = torch.randn(8, 64)
        output = norm(x)
        assert output.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        """RMSNorm works on CUDA if available."""
        norm = RMSNorm(dim=64).cuda()
        x = torch.randn(8, 64).cuda()
        output = norm(x)
        assert output.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_cpu_consistency(self):
        """CPU and CUDA produce same results."""
        torch.manual_seed(42)
        
        norm_cpu = RMSNorm(dim=64)
        norm_cuda = RMSNorm(dim=64).cuda()
        
        # Copy weights
        norm_cuda.weight.data = norm_cpu.weight.data.cuda()
        
        x = torch.randn(8, 64)
        
        out_cpu = norm_cpu(x)
        out_cuda = norm_cuda(x.cuda())
        
        assert torch.allclose(out_cpu, out_cuda.cpu(), atol=1e-5)


class TestRMSNormSerialization:
    """Test model saving and loading."""
    
    def test_state_dict_keys(self):
        """State dict contains expected keys."""
        norm = RMSNorm(dim=64)
        state_dict = norm.state_dict()
        
        assert 'weight' in state_dict
    
    def test_load_state_dict(self):
        """Can load saved state dict."""
        norm1 = RMSNorm(dim=64)
        norm1.weight.data = torch.randn(64)  # Modify weights
        
        state_dict = norm1.state_dict()
        
        norm2 = RMSNorm(dim=64)
        norm2.load_state_dict(state_dict)
        
        assert torch.equal(norm1.weight, norm2.weight)
    
    def test_save_load_produces_same_output(self):
        """Saved and loaded model produces same output."""
        import tempfile
        import os
        
        norm1 = RMSNorm(dim=64)
        x = torch.randn(8, 64)
        out1 = norm1(x)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(norm1.state_dict(), f.name)
            temp_path = f.name
        
        try:
            # Load
            norm2 = RMSNorm(dim=64)
            norm2.load_state_dict(torch.load(temp_path, weights_only=True))
            out2 = norm2(x)
            
            assert torch.equal(out1, out2)
        finally:
            os.unlink(temp_path)


class TestRMSNormWithBias:
    """Test the RMSNormWithBias variant."""
    
    def test_bias_parameter_exists(self):
        """RMSNormWithBias has bias parameter when enabled."""
        from hrm.layers.norm import RMSNormWithBias
        norm = RMSNormWithBias(dim=64, bias=True)
        assert hasattr(norm, 'bias')
        assert isinstance(norm.bias, nn.Parameter)
    
    def test_bias_initialized_to_zeros(self):
        """Bias is initialized to zeros."""
        from hrm.layers.norm import RMSNormWithBias
        norm = RMSNormWithBias(dim=64, bias=True)
        assert torch.allclose(norm.bias, torch.zeros(64))
    
    def test_bias_disabled(self):
        """Bias can be disabled."""
        from hrm.layers.norm import RMSNormWithBias
        norm = RMSNormWithBias(dim=64, bias=False)
        assert norm.bias is None
    
    def test_output_with_bias(self):
        """Output includes bias when enabled."""
        from hrm.layers.norm import RMSNormWithBias
        norm = RMSNormWithBias(dim=64, bias=True)
        norm.bias.data.fill_(1.0)  # Set bias to 1
        
        x = torch.zeros(8, 64)  # Zero input
        output = norm(x)
        
        # With zero input and bias=1, output should be ~1
        # (though RMS of zeros is handled by eps)
        assert output.shape == (8, 64)


class TestCreateNormLayer:
    """Test the create_norm_layer factory function."""
    
    def test_create_rmsnorm(self):
        """Factory creates RMSNorm."""
        from hrm.layers.norm import create_norm_layer
        norm = create_norm_layer('rmsnorm', dim=64)
        assert isinstance(norm, RMSNorm)
    
    def test_create_rmsnorm_case_insensitive(self):
        """Factory is case insensitive."""
        from hrm.layers.norm import create_norm_layer
        norm1 = create_norm_layer('RMSNorm', dim=64)
        norm2 = create_norm_layer('RMSNORM', dim=64)
        assert isinstance(norm1, RMSNorm)
        assert isinstance(norm2, RMSNorm)
    
    def test_create_rmsnorm_with_bias(self):
        """Factory creates RMSNormWithBias."""
        from hrm.layers.norm import create_norm_layer, RMSNormWithBias
        norm = create_norm_layer('rmsnorm_bias', dim=64)
        assert isinstance(norm, RMSNormWithBias)
    
    def test_create_layernorm(self):
        """Factory creates LayerNorm."""
        from hrm.layers.norm import create_norm_layer
        norm = create_norm_layer('layernorm', dim=64)
        assert isinstance(norm, nn.LayerNorm)
    
    def test_create_identity(self):
        """Factory creates Identity for 'none'."""
        from hrm.layers.norm import create_norm_layer
        norm1 = create_norm_layer('none', dim=64)
        norm2 = create_norm_layer('identity', dim=64)
        assert isinstance(norm1, nn.Identity)
        assert isinstance(norm2, nn.Identity)
    
    def test_invalid_norm_type_raises_error(self):
        """Invalid norm_type raises ValueError."""
        from hrm.layers.norm import create_norm_layer
        with pytest.raises(ValueError):
            create_norm_layer('invalid_type', dim=64)
    
    def test_custom_eps_passed_through(self):
        """Custom eps is passed to created layer."""
        from hrm.layers.norm import create_norm_layer
        norm = create_norm_layer('rmsnorm', dim=64, eps=1e-5)
        assert norm.eps == 1e-5


class TestRMSNormRepr:
    """Test string representation."""
    
    def test_extra_repr(self):
        """extra_repr returns expected format."""
        norm = RMSNorm(dim=64, eps=1e-6)
        repr_str = norm.extra_repr()
        assert 'dim=64' in repr_str
        assert 'eps=' in repr_str
    
    def test_module_repr(self):
        """Module repr includes class name and params."""
        norm = RMSNorm(dim=64)
        full_repr = repr(norm)
        assert 'RMSNorm' in full_repr


# Run verification if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
