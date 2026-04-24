# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Tests for the WorkerModule.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from hrm.layers.worker import WorkerModule, WorkerModuleWithGating


@pytest.fixture
def worker():
    """Create a worker module for testing."""
    return WorkerModule(
        hidden_dim=64,
        mlp_ratio=2,
        dropout=0.0,
        use_input_proj=True,
    )


class TestWorkerModule:
    """Essential tests for WorkerModule."""

    def test_output_shape(self, worker):
        """Test output has correct shape."""
        batch_size = 8
        h_L = torch.randn(batch_size, 64)
        h_H = torch.randn(batch_size, 64)
        x_in = torch.randn(batch_size, 64)

        output = worker(h_L, h_H, x_in)

        assert output.shape == (batch_size, 64)

    def test_missing_x_in_raises_error(self, worker):
        """Test that missing x_in raises error when use_input_proj=True."""
        h_L = torch.randn(4, 64)
        h_H = torch.randn(4, 64)

        with pytest.raises(ValueError, match="x_in is required"):
            worker(h_L, h_H)

    def test_wrong_hidden_dim_raises_error(self, worker):
        """Test that wrong input dimensions raise error."""
        h_L = torch.randn(4, 128)  # Wrong dim
        h_H = torch.randn(4, 64)
        x_in = torch.randn(4, 64)

        with pytest.raises(ValueError, match="h_L_prev last dim must be"):
            worker(h_L, h_H, x_in)

    def test_no_nan_in_output(self, worker):
        """Test output contains no NaN values."""
        h_L = torch.randn(4, 64)
        h_H = torch.randn(4, 64)
        x_in = torch.randn(4, 64)

        output = worker(h_L, h_H, x_in)

        assert not torch.isnan(output).any()

    def test_gradients_flow(self, worker):
        """Test gradients flow back through the network."""
        h_L = torch.randn(4, 64, requires_grad=True)
        h_H = torch.randn(4, 64, requires_grad=True)
        x_in = torch.randn(4, 64, requires_grad=True)

        output = worker(h_L, h_H, x_in)
        loss = output.sum()
        loss.backward()

        assert h_L.grad is not None
        assert h_H.grad is not None
        assert x_in.grad is not None

    def test_iterate_output_shape(self, worker):
        """Test iterate returns correct shape after multiple steps."""
        h_L = torch.randn(4, 64)
        h_H = torch.randn(4, 64)
        x_in = torch.randn(4, 64)

        output = worker.iterate(h_L, h_H, x_in, num_iterations=5)

        assert output.shape == (4, 64)

    def test_multiple_iterations_stable(self, worker):
        """Test many iterations don't cause numerical instability."""
        h_L = torch.randn(4, 64)
        h_H = torch.randn(4, 64)
        x_in = torch.randn(4, 64)

        output = worker.iterate(h_L, h_H, x_in, num_iterations=50)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_without_input_proj(self):
        """Test worker works without x_in input."""
        worker = WorkerModule(hidden_dim=64, mlp_ratio=2, dropout=0.0, use_input_proj=False)
        h_L = torch.randn(4, 64)
        h_H = torch.randn(4, 64)

        output = worker(h_L, h_H)

        assert output.shape == (4, 64)

    def test_gated_worker_output_shape(self):
        """Test gated variant has correct output shape."""
        worker = WorkerModuleWithGating(hidden_dim=64, mlp_ratio=2, dropout=0.0)
        h_L = torch.randn(8, 64)
        h_H = torch.randn(8, 64)
        x_in = torch.randn(8, 64)

        output = worker(h_L, h_H, x_in)

        assert output.shape == (8, 64)

    def test_invalid_params_raise_errors(self):
        """Test invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            WorkerModule(hidden_dim=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            WorkerModule(dropout=1.5)
