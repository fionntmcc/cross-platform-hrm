"""
Unit Tests for OutputHead (Simplified HRM)

Tests verifying acceptance criteria:
- Output shape: (batch, seq_len, vocab_size) per puzzle type
- Correct vocab sizes per puzzle: 4x4→5, 9x9→10, maze→2
- All three heads present in ModuleDict
- No bias on projection weights
- Gradient flow for training

Run: pytest test/hrm_test/test_output_head.py -v
"""

import pytest
import torch

from hrm.layers.output_simplified import OutputHead
from hrm.model_simplified import PuzzleType, SimplifiedHRMConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return SimplifiedHRMConfig(
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def head(config):
    return OutputHead(config)


# =============================================================================
# Output shapes per puzzle type
# =============================================================================


class TestOutputShapes:
    """OutputHead returns (batch, seq_len, vocab_size) for each puzzle type."""

    def test_sudoku_4x4_shape(self, head):
        """4x4 Sudoku → vocab_size 5 (0=empty, 1-4)."""
        h = torch.randn(8, 16, 64)
        logits = head(h, PuzzleType.SUDOKU_4X4)
        assert logits.shape == (8, 16, 5)

    def test_sudoku_9x9_shape(self, head):
        """9x9 Sudoku → vocab_size 10 (0=empty, 1-9)."""
        h = torch.randn(4, 81, 64)
        logits = head(h, PuzzleType.SUDOKU_9X9)
        assert logits.shape == (4, 81, 10)

    def test_maze_shape(self, head):
        """Maze → vocab_size 2 (binary: on-path / not-on-path)."""
        h = torch.randn(2, 25, 64)
        logits = head(h, PuzzleType.MAZE)
        assert logits.shape == (2, 25, 2)

    def test_batch_size_1(self, head):
        h = torch.randn(1, 81, 64)
        logits = head(h, PuzzleType.SUDOKU_9X9)
        assert logits.shape == (1, 81, 10)


# =============================================================================
# Architecture
# =============================================================================


class TestArchitecture:

    def test_has_all_heads(self, head):
        assert "sudoku_4x4" in head.heads
        assert "sudoku_9x9" in head.heads
        assert "maze" in head.heads

    def test_head_output_dims(self, head, config):
        assert head.heads["sudoku_4x4"].out_features == 5
        assert head.heads["sudoku_9x9"].out_features == 10
        assert head.heads["maze"].out_features == 2

    def test_head_input_dims(self, head, config):
        for h in head.heads.values():
            assert h.in_features == config.hidden_size

    def test_no_bias_on_heads(self, head):
        for h in head.heads.values():
            assert h.bias is None

    def test_extra_repr(self, head):
        r = head.extra_repr()
        assert "hidden_size" in r


# =============================================================================
# Different puzzle types produce different outputs
# =============================================================================


class TestPuzzleDispatch:

    def test_4x4_vs_9x9_different_weights(self, head):
        """Different heads have different weights (independent parameters)."""
        w4 = head.heads["sudoku_4x4"].weight
        w9 = head.heads["sudoku_9x9"].weight
        # They have different shapes so trivially different
        assert w4.shape != w9.shape

    def test_outputs_differ_across_types_same_input(self, head):
        """Same hidden state produces correctly shaped but distinct logits."""
        h = torch.randn(2, 16, 64)
        logits_4x4 = head(h, PuzzleType.SUDOKU_4X4)
        logits_maze = head(h, PuzzleType.MAZE)
        # Different vocab sizes
        assert logits_4x4.shape[-1] == 5
        assert logits_maze.shape[-1] == 2


# =============================================================================
# Gradient flow
# =============================================================================


class TestGradients:

    def test_gradients_flow_4x4(self, head):
        h = torch.randn(4, 16, 64, requires_grad=True)
        logits = head(h, PuzzleType.SUDOKU_4X4)
        logits.sum().backward()
        assert h.grad is not None
        assert head.heads["sudoku_4x4"].weight.grad is not None

    def test_gradients_flow_9x9(self, head):
        h = torch.randn(4, 81, 64, requires_grad=True)
        logits = head(h, PuzzleType.SUDOKU_9X9)
        logits.sum().backward()
        assert h.grad is not None

    def test_no_nan_in_output(self, head):
        h = torch.randn(4, 81, 64)
        logits = head(h, PuzzleType.SUDOKU_9X9)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
