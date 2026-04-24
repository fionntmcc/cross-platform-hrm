# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Unit Tests for InputEmbedding (Simplified HRM)

Tests verifying acceptance criteria:
- Output shape: (batch, seq_len, hidden_size)
- Supports all PuzzleType variants
- Puzzle-type embedding is applied
- Gradient flow for training
- Extra repr for debugging

Run: pytest test/hrm_test/test_input_embedding.py -v
"""

import pytest
import torch

from hrm.layers.input_simplified import InputEmbedding
from hrm.model_simplified import PuzzleType, SimplifiedHRMConfig

# Fixtures


@pytest.fixture
def config():
    """Small config for fast testing."""
    return SimplifiedHRMConfig(
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        vocab_size=10,
        num_puzzle_types=3,
        dropout=0.0,
    )


@pytest.fixture
def embed(config):
    return InputEmbedding(config)


# Output shape


class TestOutputShape:
    """Forward pass produces (batch, seq_len, hidden_size)."""

    def test_sudoku_9x9_shape(self, embed, config):
        x = torch.randint(0, 10, (4, 81))
        out = embed(x, PuzzleType.SUDOKU_9X9)
        assert out.shape == (4, 81, config.hidden_size)

    def test_sudoku_4x4_shape(self, embed, config):
        x = torch.randint(0, 5, (8, 16))
        out = embed(x, PuzzleType.SUDOKU_4X4)
        assert out.shape == (8, 16, config.hidden_size)

    def test_batch_size_1(self, embed, config):
        x = torch.randint(0, 5, (1, 16))
        out = embed(x, PuzzleType.SUDOKU_4X4)
        assert out.shape == (1, 16, config.hidden_size)

    def test_larger_batch(self, embed, config):
        x = torch.randint(0, 10, (16, 81))
        out = embed(x, PuzzleType.SUDOKU_9X9)
        assert out.shape == (16, 81, config.hidden_size)


# All puzzle types supported


class TestPuzzleTypes:
    """All PuzzleType variants work without error."""

    def test_sudoku_4x4(self, embed, config):
        x = torch.randint(0, 5, (2, 16))
        out = embed(x, PuzzleType.SUDOKU_4X4)
        assert out.shape == (2, 16, config.hidden_size)

    def test_sudoku_9x9(self, embed, config):
        x = torch.randint(0, 10, (2, 81))
        out = embed(x, PuzzleType.SUDOKU_9X9)
        assert out.shape == (2, 81, config.hidden_size)

    def test_maze(self, embed, config):
        x = torch.randint(0, 4, (2, 25))
        out = embed(x, PuzzleType.MAZE)
        assert out.shape == (2, 25, config.hidden_size)

    def test_different_puzzle_types_produce_different_outputs(self, embed):
        """Puzzle-type embedding causes outputs to differ across puzzle types."""
        x = torch.zeros(1, 16, dtype=torch.long)
        out_4x4 = embed(x, PuzzleType.SUDOKU_4X4)
        out_9x9 = embed(x, PuzzleType.SUDOKU_9X9)
        assert not torch.allclose(out_4x4, out_9x9)


# Gradient flow


class TestGradients:

    def test_grad_flows_through_embedding(self, embed):
        x = torch.randint(0, 10, (4, 81))
        out = embed(x, PuzzleType.SUDOKU_9X9)
        loss = out.sum()
        loss.backward()
        assert embed.tok_emb.weight.grad is not None
        assert embed.puzzle_emb.weight.grad is not None
        assert embed.input_proj.weight.grad is not None

    def test_no_nan_in_output(self, embed):
        x = torch.randint(0, 10, (4, 81))
        out = embed(x, PuzzleType.SUDOKU_9X9)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# Architecture


class TestArchitecture:

    def test_has_token_embedding(self, embed, config):
        assert hasattr(embed, "tok_emb")
        assert embed.tok_emb.num_embeddings == config.vocab_size
        assert embed.tok_emb.embedding_dim == config.hidden_size

    def test_has_puzzle_embedding(self, embed, config):
        assert hasattr(embed, "puzzle_emb")
        assert embed.puzzle_emb.num_embeddings == config.num_puzzle_types
        assert embed.puzzle_emb.embedding_dim == config.hidden_size

    def test_has_input_proj(self, embed, config):
        assert hasattr(embed, "input_proj")
        assert embed.input_proj.in_features == config.hidden_size
        assert embed.input_proj.out_features == config.hidden_size

    def test_no_bias_on_proj(self, embed):
        assert embed.input_proj.bias is None

    def test_extra_repr(self, embed):
        r = embed.extra_repr()
        assert "vocab_size" in r
        assert "hidden_size" in r
