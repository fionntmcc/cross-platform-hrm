"""
Output Head for Simplified HRM (L-Module Only)

Puzzle-specific LM heads for converting hidden states to logits.
Separate linear projections per puzzle type handle different vocab
sizes without wasting capacity on unused output dimensions.

Key differences from OutputNetworkTransformer:
    - Multiple puzzle-type heads in a single module
    - PuzzleType enum dispatch (not string keys)
    - No bias on projection layers
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
    from hrm.model_simplified import PuzzleType, SimplifiedHRMConfig


class OutputHead(nn.Module):
    """
    Puzzle-specific LM heads for the Simplified HRM.

    Separate linear projections per puzzle type handle different vocab
    sizes without wasting capacity on unused output dimensions.

    Heads:
        - sudoku_4x4: hidden_size → 5  (0=empty, 1-4)
        - sudoku_9x9: hidden_size → 10 (0=empty, 1-9)
        - maze:       hidden_size → 4

    Args:
        config: SimplifiedHRMConfig with model hyperparameters.

    Shape:
        - Input: (batch, seq_len, hidden_size)
        - Output: (batch, seq_len, vocab_size) where vocab_size depends
          on puzzle_type

    Example:
        >>> from hrm.model_simplified import SimplifiedHRMConfig, PuzzleType
        >>> config = SimplifiedHRMConfig(hidden_size=256)
        >>> head = OutputHead(config)
        >>> h = torch.randn(8, 81, 256)
        >>> logits = head(h, PuzzleType.SUDOKU_9X9)
        >>> logits.shape
        torch.Size([8, 81, 10])
    """

    def __init__(self, config: SimplifiedHRMConfig):
        super().__init__()
        self.config = config

        # One head per puzzle type
        self.heads = nn.ModuleDict(
            {
                "sudoku_4x4": nn.Linear(config.hidden_size, 5, bias=False),
                "sudoku_9x9": nn.Linear(config.hidden_size, 10, bias=False),
                "maze": nn.Linear(config.hidden_size, 4, bias=False),
            }
        )

        # Initialise with truncated normal
        head_std = 1.0 / math.sqrt(config.hidden_size)
        for head in self.heads.values():
            trunc_normal_init_(head.weight, std=head_std)

    def forward(
        self,
        h: torch.Tensor,
        puzzle_type: PuzzleType,
    ) -> torch.Tensor:
        """
        Project to vocabulary logits.

        Args:
            h: Hidden states (batch, seq_len, hidden_size).
            puzzle_type: Puzzle type for head selection.

        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        # Import here to avoid circular import at module level
        from hrm.model_simplified import PuzzleType

        if puzzle_type == PuzzleType.SUDOKU_4X4:
            return self.heads["sudoku_4x4"](h)
        elif puzzle_type == PuzzleType.SUDOKU_9X9:
            return self.heads["sudoku_9x9"](h)
        else:
            return self.heads["maze"](h)

    def extra_repr(self) -> str:
        return f"hidden_size={self.config.hidden_size}, heads=sudoku_4x4/sudoku_9x9/maze"
