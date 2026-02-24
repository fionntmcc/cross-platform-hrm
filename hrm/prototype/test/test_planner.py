"""
Unit Tests for PlannerModule

Focused tests verifying acceptance criteria:
- Accepts h_H_prev and h_L_final inputs
- Returns updated h_H state with correct shape
- Post-norm layers present
- Gradient flow for training
"""

import pytest
import torch

from hrm.layers.planner import PlannerModule, create_planner_module


# =============================================================================
# Test Input/Output Shapes (Core Acceptance Criteria)
# =============================================================================

class TestPlannerShapes:
    """Verify Planner accepts h_H_prev, h_L_final and returns h_H."""
    
    def test_output_shape_matches_input(self):
        """Output h_H has same shape as input h_H_prev."""
        planner = PlannerModule(hidden_dim=64)
        h_H_prev = torch.randn(8, 64)
        h_L_final = torch.randn(8, 64)
        
        h_H_new = planner(h_H_prev, h_L_final)
        
        assert h_H_new.shape == h_H_prev.shape
        assert h_H_new.shape == (8, 64)
    
    def test_different_hidden_dims(self):
        """Works with various hidden_dim values."""
        for hidden_dim in [32, 64, 128]:
            planner = PlannerModule(hidden_dim=hidden_dim)
            h_H_prev = torch.randn(4, hidden_dim)
            h_L_final = torch.randn(4, hidden_dim)
            
            h_H_new = planner(h_H_prev, h_L_final)
            assert h_H_new.shape == (4, hidden_dim)
    
    def test_single_sample(self):
        """Works with batch size 1."""
        planner = PlannerModule(hidden_dim=64)
        h_H_prev = torch.randn(1, 64)
        h_L_final = torch.randn(1, 64)
        
        h_H_new = planner(h_H_prev, h_L_final)
        assert h_H_new.shape == (1, 64)


# =============================================================================
# Test Architecture Components
# =============================================================================

class TestPlannerArchitecture:
    """Verify architecture matches requirements."""
    
    def test_has_post_norm_layer(self):
        """Planner has RMSNorm for post-normalisation."""
        planner = PlannerModule(hidden_dim=64)
        
        assert hasattr(planner, 'norm')
        assert planner.norm.dim == 64
    
    def test_has_mlp_block(self):
        """Planner has MLP transformation block."""
        planner = PlannerModule(hidden_dim=64)
        
        assert hasattr(planner, 'mlp')
        assert hasattr(planner, 'input_proj')
    
    def test_mlp_ratio_configurable(self):
        """MLP expansion ratio is configurable."""
        planner = PlannerModule(hidden_dim=64, mlp_ratio=4)
        assert planner.mlp_ratio == 4


# =============================================================================
# Test Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Verify gradients flow for training."""
    
    def test_gradients_flow_backward(self):
        """Gradients flow to all parameters."""
        planner = PlannerModule(hidden_dim=64)
        h_H_prev = torch.randn(4, 64)
        h_L_final = torch.randn(4, 64)
        
        h_H_new = planner(h_H_prev, h_L_final)
        loss = h_H_new.sum()
        loss.backward()
        
        assert planner.input_proj.weight.grad is not None
        assert planner.norm.weight.grad is not None


# =============================================================================
# Test Hierarchical Behaviour
# =============================================================================

class TestHierarchicalBehaviour:
    """Verify behaviour supports hierarchical convergence."""
    
    def test_different_l_states_produce_different_outputs(self):
        """Different h_L_final produces different h_H_new (reset mechanism)."""
        planner = PlannerModule(hidden_dim=64)
        planner.eval()
        
        h_H_prev = torch.randn(1, 64)
        h_L_final_1 = torch.randn(1, 64)
        h_L_final_2 = torch.randn(1, 64)
        
        h_H_new_1 = planner(h_H_prev, h_L_final_1)
        h_H_new_2 = planner(h_H_prev, h_L_final_2)
        
        # Different L states should produce different H updates
        assert not torch.allclose(h_H_new_1, h_H_new_2)
    
    def test_deterministic_in_eval_mode(self):
        """Same inputs produce same outputs in eval mode."""
        planner = PlannerModule(hidden_dim=64)
        planner.eval()
        
        h_H_prev = torch.randn(4, 64)
        h_L_final = torch.randn(4, 64)
        
        out1 = planner(h_H_prev, h_L_final)
        out2 = planner(h_H_prev, h_L_final)
        
        assert torch.allclose(out1, out2)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Verify proper error handling."""
    
    def test_invalid_hidden_dim_raises_error(self):
        """Invalid hidden_dim raises ValueError."""
        with pytest.raises(ValueError):
            PlannerModule(hidden_dim=0)
        with pytest.raises(ValueError):
            PlannerModule(hidden_dim=-1)


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactory:
    """Test create_planner_module helper."""
    
    def test_creates_configured_planner(self):
        """Factory creates correctly configured PlannerModule."""
        planner = create_planner_module(hidden_dim=64, mlp_ratio=4)
        
        assert isinstance(planner, PlannerModule)
        assert planner.hidden_dim == 64
        assert planner.mlp_ratio == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
