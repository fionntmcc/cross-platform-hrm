"""
Input Embedding for Simplified HRM (L-Module Only)

Token + puzzle-type embedding network for the simplified HRM variant
that eliminates the H-module (Planner) entirely.

Converts token sequences to hidden representations suitable for
transformer-based iterative reasoning.

Key differences from InputNetworkTransformer:
    - Puzzle-type embedding via PuzzleType enum (not integer IDs)
    - sqrt(d) scaling for stable training
    - No bias on input projection (matches Sapient)
    - Truncated normal initialisation

Based on: Ge, Liao & Poggio (2025) "Hierarchical Reasoning Models:
Perspectives and Misconceptions" (arXiv:2510.00355v2)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from hrm.layers.transformer import trunc_normal_init_

if TYPE_CHECKING:
    from hrm.prototype.models.model_simplified import SimplifiedHRMConfig, PuzzleType


class InputEmbedding(nn.Module):
    """
    Token + puzzle-type embedding network for the Simplified HRM.

    Converts token sequences to hidden representations. Puzzle-type
    embedding enables multi-task learning across Sudoku sizes and mazes.

    Architecture:
        1. Token embedding: (batch, seq_len) → (batch, seq_len, hidden_size)
        2. Puzzle-type embedding: broadcast to all positions
        3. Scale by sqrt(hidden_size) for stable training
        4. Linear projection (no bias, matches Sapient)
        5. Dropout

    Args:
        config: SimplifiedHRMConfig with model hyperparameters.

    Shape:
        - Input x: (batch, seq_len) integer tokens in [0, vocab_size)
        - Input puzzle_type: PuzzleType enum value
        - Output: (batch, seq_len, hidden_size)

    Example:
        >>> from hrm.model_simplified import SimplifiedHRMConfig, PuzzleType
        >>> config = SimplifiedHRMConfig(hidden_size=256, vocab_size=10)
        >>> input_net = InputEmbedding(config)
        >>> tokens = torch.randint(0, 10, (8, 81))
        >>> hidden = input_net(tokens, PuzzleType.SUDOKU_9X9)
        >>> hidden.shape
        torch.Size([8, 81, 256])
    """

    def __init__(self, config: SimplifiedHRMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        # Puzzle type embedding for multi-task
        self.puzzle_emb = nn.Embedding(
            config.num_puzzle_types, config.hidden_size
        )

        # Input projection (linear, no bias — matches Sapient)
        self.input_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

        self.dropout = nn.Dropout(config.dropout)

        # Initialise with truncated normal
        embed_std = 1.0 / math.sqrt(config.hidden_size)
        trunc_normal_init_(self.tok_emb.weight, std=embed_std)
        trunc_normal_init_(self.puzzle_emb.weight, std=embed_std)
        trunc_normal_init_(self.input_proj.weight, std=embed_std)

    def forward(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
    ) -> torch.Tensor:
        """
        Embed tokens with puzzle-type context.

        Args:
            x: Token indices (batch, seq_len).
            puzzle_type: Type of puzzle being solved.

        Returns:
            Hidden states (batch, seq_len, hidden_size).
        """
        h = self.tok_emb(x)

        # Add puzzle type embedding (broadcast to all positions)
        puzzle_idx = torch.tensor(
            [puzzle_type.value - 1], device=x.device, dtype=torch.long
        )
        h = h + self.puzzle_emb(puzzle_idx).unsqueeze(0)

        # Scale by sqrt(d) for stable training
        h = h * math.sqrt(self.config.hidden_size)

        h = self.input_proj(h)
        h = self.dropout(h)
        return h

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, "
            f"hidden_size={self.config.hidden_size}, "
            f"num_puzzle_types={self.config.num_puzzle_types}"
        )
