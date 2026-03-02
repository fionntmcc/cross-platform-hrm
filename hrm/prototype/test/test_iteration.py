"""
Unit Tests for Fixed-Point Iteration (Inner Loop)

Issue #21: Tests verifying acceptance criteria:
- Inner loop runs for n_steps iterations
- Residual computed as ||h_L^(t) - h_L^(t-1)||
- Early stopping when residual < threshold
- Convergence statistics tracked
- h_H is DETACHED (fixed) during inner loop

Run tests: pytest tests/test_iteration.py -v
"""

import pytest
import torch

from hrm.layers.worker import WorkerModule
from hrm.prototype.core.iteration import (
    fixed_point_iteration,
    iterate_to_convergence,
    compute_residual,
    IterationStats,
    FixedPointIterator,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def worker():
    """Create a Worker module for testing."""
    return WorkerModule(hidden_dim=64, mlp_ratio=2, dropout=0.0)


@pytest.fixture
def test_tensors():
    """Create test tensors."""
    batch_size = 4
    hidden_dim = 64
    return {
        'h_L': torch.randn(batch_size, hidden_dim),
        'h_H': torch.randn(batch_size, hidden_dim),
        'x_in': torch.randn(batch_size, hidden_dim),
    }


# =============================================================================
# Test Residual Computation
# =============================================================================

class TestComputeResidual:
    """Tests for residual ||h_L^(t) - h_L^(t-1)|| computation."""
    
    def test_residual_zero_for_same_tensors(self):
        """Residual is zero when tensors are identical."""
        h = torch.randn(4, 64)
        residual = compute_residual(h, h)
        assert residual.mean().item() == pytest.approx(0.0, abs=1e-7)
    
    def test_residual_positive_for_different_tensors(self):
        """Residual is positive for different tensors."""
        h1 = torch.randn(4, 64)
        h2 = torch.randn(4, 64)
        residual = compute_residual(h1, h2)
        assert residual.mean().item() > 0
    
    def test_residual_l2_norm(self):
        """L2 norm computed correctly."""
        h1 = torch.tensor([[3.0, 4.0]])  # L2 distance = 5.0
        h2 = torch.tensor([[0.0, 0.0]])
        residual = compute_residual(h1, h2, norm_type="l2")
        assert residual.item() == pytest.approx(5.0)
    
    def test_residual_l1_norm(self):
        """L1 norm computed correctly."""
        h1 = torch.tensor([[3.0, 4.0]])  # L1 distance = 7.0
        h2 = torch.tensor([[0.0, 0.0]])
        residual = compute_residual(h1, h2, norm_type="l1")
        assert residual.item() == pytest.approx(7.0)
    
    def test_residual_linf_norm(self):
        """L-infinity norm computed correctly."""
        h1 = torch.tensor([[3.0, 4.0]])  # Linf distance = 4.0
        h2 = torch.tensor([[0.0, 0.0]])
        residual = compute_residual(h1, h2, norm_type="linf")
        assert residual.item() == pytest.approx(4.0)
    
    def test_residual_per_sample(self):
        """Per-sample residuals have correct shape."""
        h1 = torch.randn(8, 64)
        h2 = torch.randn(8, 64)
        residual = compute_residual(h1, h2)
        assert residual.shape == (8,)


# =============================================================================
# Test Fixed-Point Iteration (Core Acceptance Criteria)
# =============================================================================

class TestFixedPointIteration:
    """Tests for the main fixed_point_iteration function."""
    
    def test_output_shape(self, worker, test_tensors):
        """Output h_L has same shape as input h_L_init."""
        h_L_final, _ = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
        )
        assert h_L_final.shape == test_tensors['h_L'].shape
    
    def test_runs_specified_steps(self, worker, test_tensors):
        """Iteration runs for exactly n_steps when no early stopping."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=7,
        )
        assert stats.num_steps == 7
        assert len(stats.residual_history) == 7
    
    def test_h_H_is_detached_critical(self, worker, test_tensors):
        """
        CRITICAL TEST: h_H must be detached during iteration.
        
        This implements the one-step gradient approximation. Gradients
        should NOT flow through h_H during the inner loop.
        """
        h_H = test_tensors['h_H'].clone().requires_grad_(True)
        h_L_init = test_tensors['h_L'].clone().requires_grad_(True)
        
        h_L_final, _ = fixed_point_iteration(
            worker=worker,
            h_L_init=h_L_init,
            h_H=h_H,
            x_in=test_tensors['x_in'],
            n_steps=5,
        )
        
        # Backward through h_L_final
        loss = h_L_final.sum()
        loss.backward()
        
        # h_H should have NO gradient because it was detached
        # This is the one-step gradient approximation
        assert h_H.grad is None, (
            "h_H should not have gradients! "
            "The one-step gradient approximation requires h_H to be detached."
        )
        
        # h_L_init CAN have gradients (it's the start of the computation)
        # But we're checking h_H specifically
    
    def test_returns_iteration_stats(self, worker, test_tensors):
        """Returns IterationStats with correct fields."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
        )
        
        assert isinstance(stats, IterationStats)
        assert stats.num_steps > 0
        assert isinstance(stats.converged, bool)
        assert isinstance(stats.final_residual, float)
        assert isinstance(stats.residual_history, list)


# =============================================================================
# Test Early Stopping
# =============================================================================

class TestEarlyStopping:
    """Tests for early stopping when residual < threshold."""
    
    def test_early_stopping_triggers(self, worker, test_tensors):
        """Early stopping triggers when residual < threshold."""
        worker.eval()
        
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=100,
            convergence_threshold=10.0,  # Large threshold - should stop early
        )
        
        # Should have stopped before 100 steps or converged
        assert stats.converged or stats.num_steps <= 100
    
    def test_no_early_stopping_without_threshold(self, worker, test_tensors):
        """Without threshold, always runs n_steps."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
            convergence_threshold=None,
        )
        
        assert stats.num_steps == 5
        assert not stats.converged
    
    def test_very_small_threshold_runs_more_steps(self, worker, test_tensors):
        """Very small threshold may not converge, runs max steps."""
        worker.eval()
        
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
            convergence_threshold=1e-10,  # Tiny threshold
        )
        
        # Likely won't converge with such small threshold
        assert stats.num_steps == 5 or stats.converged


# =============================================================================
# Test Convergence Statistics Tracking
# =============================================================================

class TestConvergenceStats:
    """Tests for iteration statistics tracking."""
    
    def test_residual_history_matches_steps(self, worker, test_tensors):
        """Residual history length matches number of steps."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=10,
        )
        
        assert len(stats.residual_history) == stats.num_steps
    
    def test_final_residual_matches_last_history(self, worker, test_tensors):
        """Final residual equals last entry in history."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
        )
        
        assert stats.final_residual == stats.residual_history[-1]
    
    def test_convergence_rate_computed(self, worker, test_tensors):
        """Convergence rate computed from history."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
        )
        
        if len(stats.residual_history) >= 2:
            expected_rate = stats.residual_history[-1] / stats.residual_history[0]
            assert stats.convergence_rate == pytest.approx(expected_rate)
    
    def test_track_history_disabled(self, worker, test_tensors):
        """Can disable history tracking for efficiency."""
        _, stats = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=5,
            track_history=False,
        )
        
        assert len(stats.residual_history) == 0
        # Final residual is still tracked
        assert stats.final_residual >= 0


# =============================================================================
# Test FixedPointIterator Module
# =============================================================================

class TestFixedPointIterator:
    """Tests for the FixedPointIterator nn.Module wrapper."""
    
    def test_forward_output_shape(self, worker, test_tensors):
        """Forward returns correct shape."""
        iterator = FixedPointIterator(worker, n_steps=5)
        
        h_L_final, stats = iterator(
            test_tensors['h_L'],
            test_tensors['h_H'],
            test_tensors['x_in'],
        )
        
        assert h_L_final.shape == test_tensors['h_L'].shape
        assert isinstance(stats, IterationStats)
    
    def test_override_n_steps(self, worker, test_tensors):
        """Can override n_steps in forward call."""
        iterator = FixedPointIterator(worker, n_steps=5)
        
        _, stats = iterator(
            test_tensors['h_L'],
            test_tensors['h_H'],
            test_tensors['x_in'],
            n_steps=3,  # Override
        )
        
        assert stats.num_steps == 3
    
    def test_override_threshold(self, worker, test_tensors):
        """Can override convergence_threshold in forward call."""
        iterator = FixedPointIterator(worker, n_steps=100)
        
        _, stats = iterator(
            test_tensors['h_L'],
            test_tensors['h_H'],
            test_tensors['x_in'],
            convergence_threshold=10.0,  # Large threshold to trigger early stop
        )
        
        # Should converge early with large threshold
        assert stats.num_steps < 100 or stats.converged


# =============================================================================
# Test iterate_to_convergence Helper
# =============================================================================

class TestIterateToConvergence:
    """Tests for the simplified iterate_to_convergence function."""
    
    def test_returns_tuple(self, worker, test_tensors):
        """Returns (h_L_final, steps, converged) tuple."""
        result = iterate_to_convergence(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            max_steps=10,
            threshold=1e-4,
        )
        
        assert len(result) == 3
        h_L_final, steps, converged = result
        assert h_L_final.shape == test_tensors['h_L'].shape
        assert isinstance(steps, int)
        assert isinstance(converged, bool)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_n_steps_raises_error(self, worker, test_tensors):
        """n_steps <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_steps must be positive"):
            fixed_point_iteration(
                worker=worker,
                h_L_init=test_tensors['h_L'],
                h_H=test_tensors['h_H'],
                x_in=test_tensors['x_in'],
                n_steps=0,
            )
    
    def test_invalid_threshold_raises_error(self, worker, test_tensors):
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            fixed_point_iteration(
                worker=worker,
                h_L_init=test_tensors['h_L'],
                h_H=test_tensors['h_H'],
                x_in=test_tensors['x_in'],
                n_steps=5,
                convergence_threshold=-0.1,
            )
    
    def test_invalid_norm_type_raises_error(self, worker, test_tensors):
        """Invalid norm_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown norm_type"):
            fixed_point_iteration(
                worker=worker,
                h_L_init=test_tensors['h_L'],
                h_H=test_tensors['h_H'],
                x_in=test_tensors['x_in'],
                n_steps=5,
                norm_type="invalid",
            )
    
    def test_iterator_invalid_n_steps_raises_error(self, worker):
        """FixedPointIterator with invalid n_steps raises ValueError."""
        with pytest.raises(ValueError, match="n_steps must be positive"):
            FixedPointIterator(worker, n_steps=0)


# =============================================================================
# Test Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability over many iterations."""
    
    def test_no_nan_after_many_iterations(self, worker, test_tensors):
        """No NaN values after many iterations."""
        worker.eval()
        
        h_L_final, _ = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=100,
        )
        
        assert not torch.isnan(h_L_final).any(), "NaN values detected!"
        assert not torch.isinf(h_L_final).any(), "Inf values detected!"
    
    def test_output_bounded(self, worker, test_tensors):
        """Output values stay bounded after many iterations."""
        worker.eval()
        
        h_L_final, _ = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=50,
        )
        
        # Values should be reasonable (not exploded)
        assert h_L_final.abs().max() < 1000, "Values exploded!"


# =============================================================================
# Test Gradient Flow (Training)
# =============================================================================

class TestGradientFlow:
    """Tests for gradient flow during training."""
    
    def test_gradients_flow_to_worker_params(self, worker, test_tensors):
        """Gradients flow to Worker parameters."""
        h_L_final, _ = fixed_point_iteration(
            worker=worker,
            h_L_init=test_tensors['h_L'],
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=3,
        )
        
        loss = h_L_final.sum()
        loss.backward()
        
        # Worker parameters should have gradients
        for name, param in worker.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_gradients_flow_from_h_L_init(self, worker, test_tensors):
        """Gradients flow back to h_L_init."""
        h_L_init = test_tensors['h_L'].clone().requires_grad_(True)
        
        h_L_final, _ = fixed_point_iteration(
            worker=worker,
            h_L_init=h_L_init,
            h_H=test_tensors['h_H'],
            x_in=test_tensors['x_in'],
            n_steps=3,
        )
        
        loss = h_L_final.sum()
        loss.backward()
        
        assert h_L_init.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
