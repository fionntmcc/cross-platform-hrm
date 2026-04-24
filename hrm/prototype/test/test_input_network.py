# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Unit Tests for InputNetwork

Focused tests verifying acceptance criteria:
- Output shapes match specification
- Configurable vocab_size, embed_dim, hidden_dim
- Gradient flow for training
"""

import pytest
import torch
import torch.nn as nn

from hrm.layers.input_network import InputNetwork, create_input_network


# Test Output Shapes (Core Acceptance Criteria)


class TestOutputShapes:
    """Verify output shape: (batch, hidden_dim)."""

    def test_4x4_sudoku_output_shape(self):
        """4x4 Sudoku: (batch, 4, 4) -> (batch, hidden_dim)."""
        net = InputNetwork(vocab_size=5, grid_size=4, embed_dim=16, hidden_dim=64)
        x = torch.randint(0, 5, (8, 4, 4))
        output = net(x)

        assert output.shape == (8, 64)

    def test_9x9_sudoku_output_shape(self):
        """9x9 Sudoku: (batch, 9, 9) -> (batch, hidden_dim)."""
        net = InputNetwork(vocab_size=10, grid_size=9, embed_dim=32, hidden_dim=128)
        x = torch.randint(0, 10, (4, 9, 9))
        output = net(x)

        assert output.shape == (4, 128)

    def test_single_sample(self):
        """Works with batch size 1."""
        net = InputNetwork(vocab_size=5, grid_size=4, embed_dim=16, hidden_dim=64)
        x = torch.randint(0, 5, (1, 4, 4))
        output = net(x)

        assert output.shape == (1, 64)


# Test Configurable Parameters


class TestConfigurableParameters:
    """Verify vocab_size, embed_dim, hidden_dim are configurable."""

    def test_different_vocab_sizes(self):
        """Various vocab_size values work correctly."""
        for vocab_size in [5, 10, 17]:  # 4x4, 9x9, 16x16
            net = InputNetwork(vocab_size=vocab_size, grid_size=4, embed_dim=16, hidden_dim=64)
            assert net.embedding.num_embeddings == vocab_size

    def test_different_embed_dims(self):
        """Various embed_dim values work correctly."""
        for embed_dim in [8, 16, 32]:
            net = InputNetwork(vocab_size=5, grid_size=4, embed_dim=embed_dim, hidden_dim=64)
            assert net.embedding.embedding_dim == embed_dim

    def test_different_hidden_dims(self):
        """Various hidden_dim values produce correct output shape."""
        for hidden_dim in [32, 64, 128]:
            net = InputNetwork(vocab_size=5, grid_size=4, embed_dim=16, hidden_dim=hidden_dim)
            x = torch.randint(0, 5, (2, 4, 4))
            output = net(x)
            assert output.shape == (2, hidden_dim)


# Test Gradient Flow (Essential for Training)


class TestGradientFlow:
    """Verify gradients flow for training."""

    def test_gradients_flow_backward(self):
        """Gradients flow to all learnable parameters."""
        net = InputNetwork(vocab_size=5, grid_size=4, embed_dim=16, hidden_dim=64)
        x = torch.randint(0, 5, (4, 4, 4))

        output = net(x)
        loss = output.sum()
        loss.backward()

        # Check key parameters have gradients
        assert net.embedding.weight.grad is not None
        assert net.norm.weight.grad is not None


# Test Error Handling


class TestErrorHandling:
    """Verify proper error handling."""

    def test_wrong_grid_size_raises_error(self):
        """Wrong input grid size raises ValueError."""
        net = InputNetwork(vocab_size=5, grid_size=4, embed_dim=16, hidden_dim=64)

        with pytest.raises(ValueError):
            net(torch.randint(0, 5, (2, 9, 9)))  # 9x9 to 4x4 network

    def test_invalid_vocab_size_raises_error(self):
        """Invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError):
            InputNetwork(vocab_size=0, grid_size=4, embed_dim=16, hidden_dim=64)


# Test Factory Function


class TestFactory:
    """Test create_input_network helper."""

    def test_creates_configured_network(self):
        """Factory creates correctly configured InputNetwork."""
        net = create_input_network(vocab_size=5, grid_size=4, embed_dim=16, hidden_dim=64)

        assert isinstance(net, InputNetwork)
        assert net.vocab_size == 5
        assert net.hidden_dim == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
