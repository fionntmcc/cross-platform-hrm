"""
Unit Tests for OutputNetwork

Focused tests verifying acceptance criteria:
- cell_head: Linear(hidden_dim → num_cells)
- digit_head: Linear(hidden_dim → num_digits)  
- Returns tuple of (cell_logits, digit_logits)
- Support configurable grid sizes
- Softmax option for inference
"""

import pytest
import torch

from hrm.layers.output_network import OutputNetwork, create_output_network


# =============================================================================
# Test Output Shapes (Core Acceptance Criteria)
# =============================================================================

class TestOutputShapes:
    """Verify output shapes match specification."""
    
    def test_4x4_sudoku_output_shapes(self):
        """4x4 Sudoku: cell_logits (batch, 16), digit_logits (batch, 4)."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(8, 64)
        
        cell_logits, digit_logits = net(h)
        
        assert cell_logits.shape == (8, 16)  # 4*4 = 16 cells
        assert digit_logits.shape == (8, 4)   # digits 1-4
    
    def test_9x9_sudoku_output_shapes(self):
        """9x9 Sudoku: cell_logits (batch, 81), digit_logits (batch, 9)."""
        net = OutputNetwork(hidden_dim=128, grid_size=9)
        h = torch.randn(4, 128)
        
        cell_logits, digit_logits = net(h)
        
        assert cell_logits.shape == (4, 81)  # 9*9 = 81 cells
        assert digit_logits.shape == (4, 9)   # digits 1-9
    
    def test_returns_tuple(self):
        """Forward returns tuple of (cell_logits, digit_logits)."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(2, 64)
        
        result = net(h)
        
        assert isinstance(result, tuple)
        assert len(result) == 2


# =============================================================================
# Test Configurable Grid Sizes
# =============================================================================

class TestConfigurableGridSizes:
    """Verify support for different grid sizes."""
    
    def test_various_grid_sizes(self):
        """Works with various grid sizes."""
        for grid_size in [4, 9, 16]:
            net = OutputNetwork(hidden_dim=64, grid_size=grid_size)
            h = torch.randn(2, 64)
            
            cell_logits, digit_logits = net(h)
            
            assert cell_logits.shape == (2, grid_size * grid_size)
            assert digit_logits.shape == (2, grid_size)
    
    def test_stores_grid_configuration(self):
        """Network stores grid configuration correctly."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        
        assert net.grid_size == 4
        assert net.num_cells == 16
        assert net.num_digits == 4


# =============================================================================
# Test Softmax Option
# =============================================================================

class TestSoftmaxOption:
    """Verify softmax option for inference."""
    
    def test_without_softmax_returns_logits(self):
        """Without softmax, returns raw logits (can be negative)."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(4, 64)
        
        cell_logits, digit_logits = net(h, apply_softmax=False)
        
        # Raw logits can be negative and don't sum to 1
        assert cell_logits.min() < 0 or cell_logits.max() > 1
    
    def test_with_softmax_returns_probabilities(self):
        """With softmax, returns valid probability distributions."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(4, 64)
        
        cell_probs, digit_probs = net(h, apply_softmax=True)
        
        # Probabilities should be in [0, 1] and sum to 1
        assert (cell_probs >= 0).all() and (cell_probs <= 1).all()
        assert (digit_probs >= 0).all() and (digit_probs <= 1).all()
        assert torch.allclose(cell_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.allclose(digit_probs.sum(dim=-1), torch.ones(4), atol=1e-5)


# =============================================================================
# Test Prediction Methods
# =============================================================================

class TestPredictionMethods:
    """Verify prediction convenience methods."""
    
    def test_predict_returns_indices(self):
        """predict() returns cell and digit indices."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(8, 64)
        
        cell_idx, digit_idx = net.predict(h)
        
        assert cell_idx.shape == (8,)
        assert digit_idx.shape == (8,)
        assert (cell_idx >= 0).all() and (cell_idx < 16).all()
        assert (digit_idx >= 0).all() and (digit_idx < 4).all()
    
    def test_predict_with_confidence(self):
        """predict_with_confidence() returns indices and confidences."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(4, 64)
        
        cell_idx, digit_idx, cell_conf, digit_conf = net.predict_with_confidence(h)
        
        assert cell_idx.shape == (4,)
        assert digit_idx.shape == (4,)
        assert cell_conf.shape == (4,)
        assert digit_conf.shape == (4,)
        # Confidences should be valid probabilities
        assert (cell_conf >= 0).all() and (cell_conf <= 1).all()


# =============================================================================
# Test Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Verify gradients flow for training."""
    
    def test_gradients_flow_to_both_heads(self):
        """Gradients flow to cell_head and digit_head."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(4, 64)
        
        cell_logits, digit_logits = net(h)
        loss = cell_logits.sum() + digit_logits.sum()
        loss.backward()
        
        assert net.cell_head.weight.grad is not None
        assert net.digit_head.weight.grad is not None


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Verify proper error handling."""
    
    def test_invalid_hidden_dim_raises_error(self):
        """Invalid hidden_dim raises ValueError."""
        with pytest.raises(ValueError):
            OutputNetwork(hidden_dim=0, grid_size=4)
    
    def test_invalid_grid_size_raises_error(self):
        """Invalid grid_size raises ValueError."""
        with pytest.raises(ValueError):
            OutputNetwork(hidden_dim=64, grid_size=0)
    
    def test_wrong_input_shape_raises_error(self):
        """Wrong input hidden_dim raises ValueError."""
        net = OutputNetwork(hidden_dim=64, grid_size=4)
        h = torch.randn(4, 32)  # Wrong dim
        
        with pytest.raises(ValueError):
            net(h)


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactory:
    """Test create_output_network helper."""
    
    def test_creates_configured_network(self):
        """Factory creates correctly configured OutputNetwork."""
        net = create_output_network(hidden_dim=64, grid_size=4)
        
        assert isinstance(net, OutputNetwork)
        assert net.hidden_dim == 64
        assert net.grid_size == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
