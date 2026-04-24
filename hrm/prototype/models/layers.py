# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Layers for the HRM model.

Attention - Applies weights to input tokens based on their relevance to each other.
SwiGLU - Swish Gated Linear Unit activation - Gating adds more control over which parts of input activate.
RMS Norm - Root Mean Square Normalisation maintains stable activations and not computationally expensive compared to Linear Normalisation.

Both modules use Attention -> RMS Norm -> SwiGLU -> RMS Norm sequence per block.
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Multi-head Attention class.
    This class supports Grouped Query Attention (GQA) by allowing multiple key heads.

    Output can be accessed via self.output_proj after forward pass.

    Args:
        latent_size: Size of the input latent vectors.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        num_key_heads: Number of key heads for GQA. If None, defaults to num_heads.
        causal: Whether to apply causal masking.

    """

    def __init__(
        self,
        latent_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_key_heads: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.head_dim = head_dim or (latent_size // num_heads)
        self.output_size = self.head_dim * num_heads
        self.causal = causal
        self.num_key_heads = num_key_heads or num_heads

        # Initialise the linear projections (Query, Key, Value)
        self.qkv_proj = nn.Linear(
            self.latent_size,
            (3 * self.num_heads + 2 * self.num_key_heads) * self.head_dim,
            bias=False,
        )
        # Output projection
        self.output_proj = nn.Linear(self.output_size, latent_size, bias=False)
