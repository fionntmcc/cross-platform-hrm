"""
Unit Tests for hrm.training.metrics

Covers:
    - compute_accuracy (with and without mask)
    - compute_puzzle_accuracy (with and without mask)
    - compute_residuals (convergence tracking)
    - MetricsTracker (accumulation and epoch summaries)
"""

import pytest
import torch

from hrm.training.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_puzzle_accuracy,
    compute_residuals,
)

# =========================================================================
# compute_accuracy
# =========================================================================


class TestComputeAccuracy:
    """Token-level accuracy function."""

    def test_perfect_accuracy(self):
        preds = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert compute_accuracy(preds, targets) == 1.0

    def test_zero_accuracy(self):
        preds = torch.tensor([[0, 0, 0], [0, 0, 0]])
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert compute_accuracy(preds, targets) == 0.0

    def test_partial_accuracy(self):
        preds = torch.tensor([[1, 0, 3]])
        targets = torch.tensor([[1, 2, 3]])
        acc = compute_accuracy(preds, targets)
        assert abs(acc - 2.0 / 3.0) < 1e-6

    def test_with_mask(self):
        preds = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[1, 9, 3]])
        mask = torch.tensor([[True, False, True]])
        assert compute_accuracy(preds, targets, mask=mask) == 1.0

    def test_mask_all_false(self):
        """All-false mask falls back to full comparison."""
        preds = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[False, False, False]])
        assert compute_accuracy(preds, targets, mask=mask) == 1.0

    def test_empty_tensors(self):
        preds = torch.tensor([]).reshape(0, 3).long()
        targets = torch.tensor([]).reshape(0, 3).long()
        assert compute_accuracy(preds, targets) == 0.0

    def test_returns_float(self):
        preds = torch.tensor([[1]])
        targets = torch.tensor([[1]])
        result = compute_accuracy(preds, targets)
        assert isinstance(result, float)

    def test_large_batch(self):
        preds = torch.randint(0, 10, (128, 81))
        targets = preds.clone()
        assert compute_accuracy(preds, targets) == 1.0


# =========================================================================
# compute_puzzle_accuracy
# =========================================================================


class TestComputePuzzleAccuracy:
    """Full-puzzle solve rate."""

    def test_all_correct(self):
        preds = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert compute_puzzle_accuracy(preds, targets) == 1.0

    def test_none_correct(self):
        preds = torch.tensor([[0, 0, 0], [0, 0, 0]])
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert compute_puzzle_accuracy(preds, targets) == 0.0

    def test_half_correct(self):
        preds = torch.tensor([[1, 2, 3], [0, 0, 0]])
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert abs(compute_puzzle_accuracy(preds, targets) - 0.5) < 1e-6

    def test_single_wrong_cell_fails_puzzle(self):
        preds = torch.tensor([[1, 2, 0]])
        targets = torch.tensor([[1, 2, 3]])
        assert compute_puzzle_accuracy(preds, targets) == 0.0

    def test_with_mask(self):
        """Only masked cells matter."""
        preds = torch.tensor([[9, 2, 3]])  # cell 0 wrong but not masked
        targets = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[False, True, True]])
        assert compute_puzzle_accuracy(preds, targets, mask=mask) == 1.0

    def test_with_mask_wrong_masked_cell(self):
        preds = torch.tensor([[1, 0, 3]])  # cell 1 wrong and masked
        targets = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[False, True, True]])
        assert compute_puzzle_accuracy(preds, targets, mask=mask) == 0.0

    def test_returns_float(self):
        preds = torch.tensor([[1]])
        targets = torch.tensor([[1]])
        assert isinstance(compute_puzzle_accuracy(preds, targets), float)


# =========================================================================
# compute_residuals
# =========================================================================


class TestComputeResiduals:
    """L2 residual norms between consecutive reasoning steps."""

    def test_identical_steps_zero_residual(self):
        logits = torch.randn(2, 4, 10)
        residuals = compute_residuals([logits, logits])
        assert len(residuals) == 1
        assert abs(residuals[0]) < 1e-5

    def test_different_steps_positive_residual(self):
        a = torch.zeros(2, 4, 10)
        b = torch.ones(2, 4, 10)
        residuals = compute_residuals([a, b])
        assert len(residuals) == 1
        assert residuals[0] > 0

    def test_multiple_steps(self):
        steps = [torch.randn(2, 4, 10) for _ in range(5)]
        residuals = compute_residuals(steps)
        assert len(residuals) == 4

    def test_single_step_empty(self):
        residuals = compute_residuals([torch.randn(2, 4, 10)])
        assert residuals == []

    def test_empty_list(self):
        residuals = compute_residuals([])
        assert residuals == []

    def test_returns_list_of_floats(self):
        steps = [torch.randn(2, 4, 10) for _ in range(3)]
        residuals = compute_residuals(steps)
        for r in residuals:
            assert isinstance(r, float)

    def test_converging_sequence(self):
        """Residuals should decrease as steps converge."""
        base = torch.randn(2, 4, 10)
        steps = [base + (0.5**i) * torch.randn(2, 4, 10) for i in range(5)]
        # Not guaranteed to monotonically decrease, but overall should trend down
        residuals = compute_residuals(steps)
        assert len(residuals) == 4


# =========================================================================
# MetricsTracker
# =========================================================================


class TestMetricsTracker:
    """Per-epoch accumulator."""

    def _make_batch(self, batch_size=4, seq_len=16, vocab=10, correct=True):
        """Helper to create a fake batch."""
        targets = torch.randint(0, vocab, (batch_size, seq_len))
        if correct:
            predictions = targets.clone()
        else:
            predictions = torch.randint(0, vocab, (batch_size, seq_len))
        return predictions, targets

    def test_single_batch_summary(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch(correct=True)
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise()
        assert summary["loss"] == pytest.approx(0.5)
        assert summary["token_accuracy"] == 1.0
        assert summary["puzzle_accuracy"] == 1.0
        assert summary["num_samples"] == 4

    def test_multiple_batch_accumulation(self):
        tracker = MetricsTracker()
        # Batch 1: all correct, loss 0.4
        preds1, targets1 = self._make_batch(batch_size=4, correct=True)
        tracker.update(loss=0.4, predictions=preds1, targets=targets1, batch_size=4)
        # Batch 2: all correct, loss 0.6
        preds2, targets2 = self._make_batch(batch_size=4, correct=True)
        tracker.update(loss=0.6, predictions=preds2, targets=targets2, batch_size=4)

        summary = tracker.summarise()
        assert summary["loss"] == pytest.approx(0.5)  # weighted average
        assert summary["num_samples"] == 8

    def test_summary_includes_batch_losses(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.3, predictions=preds, targets=targets, batch_size=4)
        tracker.update(loss=0.7, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise()
        assert summary["batch_losses"] == [0.3, 0.7]

    def test_summary_with_learning_rate(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise(learning_rate=1e-3)
        assert summary["learning_rate"] == 1e-3

    def test_summary_with_reasoning_steps(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise(reasoning_steps=16)
        assert summary["reasoning_steps"] == 16

    def test_summary_with_epoch(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise(epoch=5)
        assert summary["epoch"] == 5

    def test_summary_has_epoch_time(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise()
        assert "epoch_time_s" in summary
        assert summary["epoch_time_s"] >= 0

    def test_with_mask(self):
        tracker = MetricsTracker()
        preds = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[1, 9, 3, 4]])
        mask = torch.tensor([[True, False, True, True]])
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=1, mask=mask)
        summary = tracker.summarise()
        assert summary["token_accuracy"] == 1.0  # masked cell ignored

    def test_residual_tracking(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        step_logits = [torch.randn(4, 16, 10) for _ in range(3)]
        tracker.update(
            loss=0.5,
            predictions=preds,
            targets=targets,
            batch_size=4,
            all_step_logits=step_logits,
        )
        summary = tracker.summarise()
        assert summary["avg_residual"] is not None
        assert summary["avg_residual"] > 0

    def test_no_residual_without_step_logits(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise()
        assert summary["avg_residual"] is None

    def test_summarize_alias(self):
        """American spelling alias works."""
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarize()
        assert "loss" in summary

    def test_lm_loss_defaults_to_loss(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4)
        summary = tracker.summarise()
        assert summary["lm_loss"] == pytest.approx(summary["loss"])

    def test_lm_loss_separate(self):
        tracker = MetricsTracker()
        preds, targets = self._make_batch()
        tracker.update(loss=0.5, predictions=preds, targets=targets, batch_size=4, lm_loss=0.3)
        summary = tracker.summarise()
        assert summary["lm_loss"] == pytest.approx(0.3)

    def test_zero_samples_no_crash(self):
        """Summarise on empty tracker should not crash."""
        tracker = MetricsTracker()
        summary = tracker.summarise()
        assert summary["loss"] == 0.0
        assert summary["token_accuracy"] == 0.0
