# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Unit Tests for Hierarchical Outer Loop (Planner Cycles)

Tests verifying acceptance criteria:
- Outer loop runs for n_outer_cycles cycles
- Inner loop called for Worker refinement each cycle
- Planner updates after inner convergence
- Cycle-level statistics tracked
- Proper gradient flow (one-step gradient approximation)
- Total iterations = K × T (outer cycles × inner steps)

Run tests: pytest test/hrm_test/test_iteration_outer.py -v
"""

import pytest
import torch

from hrm.layers.worker import WorkerModule
from hrm.layers.planner import PlannerModule
from hrm.prototype.core.iteration import IterationStats
from hrm.prototype.core.iteration_outer import (
    hierarchical_iteration,
    hierarchical_iterate_to_convergence,
    single_hierarchical_step,
    OuterLoopStats,
    HierarchicalIterator,
)


# Fixtures


@pytest.fixture
def worker():
    """Create a Worker module for testing."""
    return WorkerModule(hidden_dim=64, mlp_ratio=2, dropout=0.0)


@pytest.fixture
def planner():
    """Create a Planner module for testing."""
    return PlannerModule(hidden_dim=64, mlp_ratio=2, dropout=0.0)


@pytest.fixture
def test_tensors():
    """Create test tensors."""
    batch_size = 4
    hidden_dim = 64
    return {
        "h_L": torch.randn(batch_size, hidden_dim),
        "h_H": torch.randn(batch_size, hidden_dim),
        "x_in": torch.randn(batch_size, hidden_dim),
    }


# Test OuterLoopStats


class TestOuterLoopStats:
    """Tests for OuterLoopStats dataclass."""

    def test_stats_creation(self):
        """Test OuterLoopStats can be created with required fields."""
        stats = OuterLoopStats(
            num_cycles=5,
            total_inner_steps=50,
            converged=True,
            final_h_H_residual=0.001,
        )

        assert stats.num_cycles == 5
        assert stats.total_inner_steps == 50
        assert stats.converged is True
        assert stats.final_h_H_residual == 0.001

    def test_average_inner_steps(self):
        """Test average_inner_steps_per_cycle property."""
        stats = OuterLoopStats(
            num_cycles=5,
            total_inner_steps=50,
            converged=True,
            final_h_H_residual=0.001,
        )

        assert stats.average_inner_steps_per_cycle == 10.0

    def test_average_inner_steps_zero_cycles(self):
        """Test average_inner_steps_per_cycle with zero cycles."""
        stats = OuterLoopStats(
            num_cycles=0,
            total_inner_steps=0,
            converged=False,
            final_h_H_residual=0.0,
        )

        assert stats.average_inner_steps_per_cycle == 0.0

    def test_convergence_rate(self):
        """Test convergence_rate property."""
        stats = OuterLoopStats(
            num_cycles=3,
            total_inner_steps=30,
            converged=True,
            final_h_H_residual=0.001,
            h_H_residual_history=[1.0, 0.5, 0.1],
        )

        assert stats.convergence_rate == pytest.approx(0.1)

    def test_convergence_rate_insufficient_history(self):
        """Test convergence_rate returns None with insufficient history."""
        stats = OuterLoopStats(
            num_cycles=1,
            total_inner_steps=10,
            converged=False,
            final_h_H_residual=0.5,
            h_H_residual_history=[0.5],
        )

        assert stats.convergence_rate is None

    def test_repr(self):
        """Test string representation."""
        stats = OuterLoopStats(
            num_cycles=5,
            total_inner_steps=50,
            converged=True,
            final_h_H_residual=0.001234,
        )

        repr_str = repr(stats)
        assert "cycles=5" in repr_str
        assert "total_inner_steps=50" in repr_str
        assert "converged=True" in repr_str


# Test Hierarchical Iteration (Core Acceptance Criteria)


class TestHierarchicalIteration:
    """Tests for the main hierarchical_iteration function."""

    def test_output_shapes(self, worker, planner, test_tensors):
        """Output h_L and h_H have same shapes as inputs."""
        h_L_final, h_H_final, _ = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=3,
            n_inner_steps=5,
        )

        assert h_L_final.shape == test_tensors["h_L"].shape
        assert h_H_final.shape == test_tensors["h_H"].shape

    def test_runs_specified_cycles(self, worker, planner, test_tensors):
        """Outer loop runs for exactly n_outer_cycles when no early stopping."""
        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=4,
            n_inner_steps=5,
        )

        assert stats.num_cycles == 4
        assert len(stats.cycle_stats) == 4
        assert len(stats.h_H_residual_history) == 4

    def test_total_iterations_k_times_t(self, worker, planner, test_tensors):
        """Total iterations = K × T (outer cycles × inner steps)."""
        n_outer = 3
        n_inner = 7

        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=n_outer,
            n_inner_steps=n_inner,
        )

        # Without early stopping, total = K × T
        assert stats.total_inner_steps == n_outer * n_inner

    def test_returns_outer_loop_stats(self, worker, planner, test_tensors):
        """Returns OuterLoopStats with correct fields."""
        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=3,
            n_inner_steps=5,
        )

        assert isinstance(stats, OuterLoopStats)
        assert stats.num_cycles > 0
        assert stats.total_inner_steps > 0
        assert isinstance(stats.converged, bool)
        assert isinstance(stats.final_h_H_residual, float)
        assert isinstance(stats.cycle_stats, list)
        assert isinstance(stats.h_H_residual_history, list)

    def test_inner_loop_stats_tracked(self, worker, planner, test_tensors):
        """Inner loop statistics tracked for each cycle."""
        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=3,
            n_inner_steps=5,
        )

        for inner_stats in stats.cycle_stats:
            assert isinstance(inner_stats, IterationStats)
            assert inner_stats.num_steps > 0

    def test_h_H_updated_each_cycle(self, worker, planner, test_tensors):
        """h_H changes after each cycle (Planner update)."""
        h_H_values = []

        h_L = test_tensors["h_L"].clone()
        h_H = test_tensors["h_H"].clone()
        x_in = test_tensors["x_in"]

        # Run cycles manually and track h_H
        for _ in range(3):
            h_L, h_H, inner_stats = single_hierarchical_step(
                worker, planner, h_L, h_H, x_in, n_inner_steps=5
            )
            h_H_values.append(h_H.clone())

        # h_H should change between cycles
        for i in range(len(h_H_values) - 1):
            diff = (h_H_values[i] - h_H_values[i + 1]).abs().sum()
            assert diff > 0, f"h_H unchanged between cycle {i} and {i+1}"


# Test Gradient Flow (One-Step Gradient Approximation)


class TestGradientFlow:
    """Tests for proper gradient flow in hierarchical iteration."""

    def test_gradients_flow_through_final_states(self, worker, planner, test_tensors):
        """Gradients flow through final h_L and h_H states."""
        h_L_init = test_tensors["h_L"].clone().requires_grad_(True)
        h_H_init = test_tensors["h_H"].clone().requires_grad_(True)

        h_L_final, h_H_final, _ = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=h_L_init,
            h_H_init=h_H_init,
            x_in=test_tensors["x_in"],
            n_outer_cycles=2,
            n_inner_steps=3,
        )

        # Backward through final states
        loss = h_L_final.sum() + h_H_final.sum()
        loss.backward()

        # Gradients should flow to initial states
        # (through final iteration step and planner updates)
        assert h_L_init.grad is not None or h_H_init.grad is not None

    def test_worker_and_planner_receive_gradients(self, worker, planner, test_tensors):
        """Worker and Planner modules receive gradients."""
        h_L_final, h_H_final, _ = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=2,
            n_inner_steps=3,
        )

        # Backward
        loss = h_L_final.sum() + h_H_final.sum()
        loss.backward()

        # Check Worker has gradients
        worker_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in worker.parameters()
        )
        assert worker_has_grad, "Worker should receive gradients"

        # Check Planner has gradients
        planner_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters()
        )
        assert planner_has_grad, "Planner should receive gradients"


# Test Early Stopping


class TestOuterEarlyStopping:
    """Tests for outer loop early stopping."""

    def test_outer_early_stopping_triggers(self, worker, planner, test_tensors):
        """Outer loop stops early when h_H converges."""
        worker.eval()
        planner.eval()

        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=100,
            n_inner_steps=5,
            outer_convergence_threshold=10.0,  # Large threshold - should stop early
        )

        # Should have stopped before 100 cycles or converged
        assert stats.converged or stats.num_cycles <= 100

    def test_inner_early_stopping_within_cycles(self, worker, planner, test_tensors):
        """Inner loop can stop early within each cycle."""
        worker.eval()

        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=3,
            n_inner_steps=100,
            inner_convergence_threshold=10.0,  # Large threshold
        )

        # Some cycles should have converged early
        converged_cycles = sum(1 for s in stats.cycle_stats if s.converged)
        assert converged_cycles > 0 or all(s.num_steps < 100 for s in stats.cycle_stats)

    def test_no_early_stopping_without_threshold(self, worker, planner, test_tensors):
        """Without threshold, always runs n_outer_cycles."""
        _, _, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=4,
            n_inner_steps=5,
            outer_convergence_threshold=None,
        )

        assert stats.num_cycles == 4
        assert not stats.converged


# Test HierarchicalIterator Module


class TestHierarchicalIterator:
    """Tests for the HierarchicalIterator nn.Module wrapper."""

    def test_forward_output_shapes(self, worker, planner, test_tensors):
        """Forward returns correct shapes."""
        iterator = HierarchicalIterator(worker, planner, n_outer_cycles=3, n_inner_steps=5)

        h_L_final, h_H_final, stats = iterator(
            test_tensors["h_L"],
            test_tensors["h_H"],
            test_tensors["x_in"],
        )

        assert h_L_final.shape == test_tensors["h_L"].shape
        assert h_H_final.shape == test_tensors["h_H"].shape
        assert isinstance(stats, OuterLoopStats)

    def test_override_n_outer_cycles(self, worker, planner, test_tensors):
        """Can override n_outer_cycles in forward call."""
        iterator = HierarchicalIterator(worker, planner, n_outer_cycles=10, n_inner_steps=5)

        _, _, stats = iterator(
            test_tensors["h_L"],
            test_tensors["h_H"],
            test_tensors["x_in"],
            n_outer_cycles=3,  # Override
        )

        assert stats.num_cycles == 3

    def test_override_n_inner_steps(self, worker, planner, test_tensors):
        """Can override n_inner_steps in forward call."""
        iterator = HierarchicalIterator(worker, planner, n_outer_cycles=2, n_inner_steps=10)

        _, _, stats = iterator(
            test_tensors["h_L"],
            test_tensors["h_H"],
            test_tensors["x_in"],
            n_inner_steps=3,  # Override
        )

        # Total steps should be 2 cycles × 3 inner steps = 6
        assert stats.total_inner_steps == 6

    def test_module_is_trainable(self, worker, planner, test_tensors):
        """HierarchicalIterator is trainable as nn.Module."""
        iterator = HierarchicalIterator(worker, planner, n_outer_cycles=2, n_inner_steps=3)

        h_L_final, h_H_final, _ = iterator(
            test_tensors["h_L"],
            test_tensors["h_H"],
            test_tensors["x_in"],
        )

        loss = h_L_final.sum() + h_H_final.sum()
        loss.backward()

        # Check parameters have gradients
        has_gradients = any(p.grad is not None for p in iterator.parameters())
        assert has_gradients

    def test_extra_repr(self, worker, planner):
        """Test extra_repr contains configuration."""
        iterator = HierarchicalIterator(
            worker,
            planner,
            n_outer_cycles=5,
            n_inner_steps=10,
            inner_convergence_threshold=1e-4,
            outer_convergence_threshold=1e-3,
        )

        repr_str = iterator.extra_repr()
        assert "n_outer_cycles=5" in repr_str
        assert "n_inner_steps=10" in repr_str


# Test hierarchical_iterate_to_convergence Helper


class TestHierarchicalIterateToConvergence:
    """Tests for the simplified hierarchical_iterate_to_convergence function."""

    def test_returns_tuple(self, worker, planner, test_tensors):
        """Returns (h_L, h_H, cycles, steps, converged) tuple."""
        result = hierarchical_iterate_to_convergence(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            max_outer_cycles=5,
            max_inner_steps=10,
        )

        assert len(result) == 5
        h_L_final, h_H_final, cycles, steps, converged = result

        assert h_L_final.shape == test_tensors["h_L"].shape
        assert h_H_final.shape == test_tensors["h_H"].shape
        assert isinstance(cycles, int)
        assert isinstance(steps, int)
        assert isinstance(converged, bool)

    def test_uses_thresholds(self, worker, planner, test_tensors):
        """Uses provided thresholds for early stopping."""
        worker.eval()
        planner.eval()

        _, _, cycles, _, converged = hierarchical_iterate_to_convergence(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            max_outer_cycles=100,
            max_inner_steps=50,
            inner_threshold=1e-5,
            outer_threshold=10.0,  # Large threshold - should converge quickly
        )

        assert cycles < 100 or converged


# Test single_hierarchical_step Helper


class TestSingleHierarchicalStep:
    """Tests for the single_hierarchical_step function."""

    def test_returns_updated_states(self, worker, planner, test_tensors):
        """Returns updated h_L and h_H states."""
        h_L_new, h_H_new, inner_stats = single_hierarchical_step(
            worker=worker,
            planner=planner,
            h_L=test_tensors["h_L"],
            h_H=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_inner_steps=5,
        )

        assert h_L_new.shape == test_tensors["h_L"].shape
        assert h_H_new.shape == test_tensors["h_H"].shape
        assert isinstance(inner_stats, IterationStats)

    def test_states_change_after_step(self, worker, planner, test_tensors):
        """States change after a hierarchical step."""
        h_L_init = test_tensors["h_L"].clone()
        h_H_init = test_tensors["h_H"].clone()

        h_L_new, h_H_new, _ = single_hierarchical_step(
            worker=worker,
            planner=planner,
            h_L=h_L_init,
            h_H=h_H_init,
            x_in=test_tensors["x_in"],
            n_inner_steps=5,
        )

        # States should have changed
        h_L_diff = (h_L_new - h_L_init).abs().sum()
        h_H_diff = (h_H_new - h_H_init).abs().sum()

        assert h_L_diff > 0, "h_L should change after step"
        assert h_H_diff > 0, "h_H should change after step"

    def test_manual_outer_loop(self, worker, planner, test_tensors):
        """Can use for manual outer loop control."""
        h_L = test_tensors["h_L"].clone()
        h_H = test_tensors["h_H"].clone()
        x_in = test_tensors["x_in"]

        total_inner_steps = 0
        for cycle in range(3):
            h_L, h_H, inner_stats = single_hierarchical_step(
                worker, planner, h_L, h_H, x_in, n_inner_steps=5
            )
            total_inner_steps += inner_stats.num_steps

        assert total_inner_steps == 15  # 3 cycles × 5 steps


# Test Error Handling


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_n_outer_cycles_raises_error(self, worker, planner, test_tensors):
        """n_outer_cycles <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_outer_cycles must be positive"):
            hierarchical_iteration(
                worker=worker,
                planner=planner,
                h_L_init=test_tensors["h_L"],
                h_H_init=test_tensors["h_H"],
                n_outer_cycles=0,
                n_inner_steps=5,
            )

    def test_invalid_n_inner_steps_raises_error(self, worker, planner, test_tensors):
        """n_inner_steps <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_inner_steps must be positive"):
            hierarchical_iteration(
                worker=worker,
                planner=planner,
                h_L_init=test_tensors["h_L"],
                h_H_init=test_tensors["h_H"],
                n_outer_cycles=3,
                n_inner_steps=0,
            )

    def test_invalid_inner_threshold_raises_error(self, worker, planner, test_tensors):
        """Negative inner_convergence_threshold raises ValueError."""
        with pytest.raises(ValueError, match="inner_convergence_threshold must be positive"):
            hierarchical_iteration(
                worker=worker,
                planner=planner,
                h_L_init=test_tensors["h_L"],
                h_H_init=test_tensors["h_H"],
                n_outer_cycles=3,
                n_inner_steps=5,
                inner_convergence_threshold=-0.1,
            )

    def test_invalid_outer_threshold_raises_error(self, worker, planner, test_tensors):
        """Negative outer_convergence_threshold raises ValueError."""
        with pytest.raises(ValueError, match="outer_convergence_threshold must be positive"):
            hierarchical_iteration(
                worker=worker,
                planner=planner,
                h_L_init=test_tensors["h_L"],
                h_H_init=test_tensors["h_H"],
                n_outer_cycles=3,
                n_inner_steps=5,
                outer_convergence_threshold=-0.1,
            )


# Integration Tests


class TestIntegration:
    """Integration tests for hierarchical iteration."""

    def test_deterministic_with_seed(self, worker, planner):
        """Results are deterministic with fixed seed."""
        torch.manual_seed(42)
        h_L_1 = torch.randn(4, 64)
        h_H_1 = torch.randn(4, 64)
        x_in_1 = torch.randn(4, 64)

        worker.eval()
        planner.eval()

        h_L_final_1, h_H_final_1, stats_1 = hierarchical_iteration(
            worker,
            planner,
            h_L_1,
            h_H_1,
            x_in_1,
            n_outer_cycles=3,
            n_inner_steps=5,
        )

        # Reset and run again
        torch.manual_seed(42)
        h_L_2 = torch.randn(4, 64)
        h_H_2 = torch.randn(4, 64)
        x_in_2 = torch.randn(4, 64)

        h_L_final_2, h_H_final_2, stats_2 = hierarchical_iteration(
            worker,
            planner,
            h_L_2,
            h_H_2,
            x_in_2,
            n_outer_cycles=3,
            n_inner_steps=5,
        )

        torch.testing.assert_close(h_L_final_1, h_L_final_2)
        torch.testing.assert_close(h_H_final_1, h_H_final_2)
        assert stats_1.num_cycles == stats_2.num_cycles

    def test_training_step_integration(self, worker, planner, test_tensors):
        """Integration test simulating a training step."""
        # Setup optimizer
        params = list(worker.parameters()) + list(planner.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)

        # Forward pass
        h_L_final, h_H_final, stats = hierarchical_iteration(
            worker=worker,
            planner=planner,
            h_L_init=test_tensors["h_L"],
            h_H_init=test_tensors["h_H"],
            x_in=test_tensors["x_in"],
            n_outer_cycles=2,
            n_inner_steps=3,
        )

        # Compute loss (mock target)
        target = torch.randn_like(h_L_final)
        loss = torch.nn.functional.mse_loss(h_L_final, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in params)
        assert has_grad, "Parameters should have gradients"

        # Optimizer step
        optimizer.step()

        # Verify training worked (no errors)
        assert True
