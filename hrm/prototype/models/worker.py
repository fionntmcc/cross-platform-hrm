# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Worker (Low-Level) Module for HRM

Issue #3: Worker Module Implementation

This module provides the Worker module (f_L) that performs fast, detailed
computations within each outer planning cycle. The Worker iterates to
convergence, refining the low-level hidden state.

Architecture Decision:
    Option A (CHOSEN): MLP-based with residual connections
    Option B (Alternative): Attention-based

    Rationale for MLP-based approach:
    1. Simpler architecture, easier to debug and maintain
    2. Better ONNX export compatibility for cross-platform deployment
    3. Lower computational overhead per iteration (important since Worker
       iterates multiple times per Planner step)
    4. Sufficient representational power for local refinement tasks
    5. Residual connections provide gradient highways for deep iteration

    The attention-based approach would offer:
    - Better long-range dependency modeling
    - More expressive representations
    But these benefits are less critical for the Worker's local refinement
    role, where the Planner handles high-level reasoning.

Reference:
    - HRM Architecture Specification
    - "Deep Residual Learning for Image Recognition" (He et al., 2015)
"""

import torch
import torch.nn as nn
from typing import Optional

from hrm.layers.norm import RMSNorm


class WorkerModule(nn.Module):
    """
    Worker (Low-Level) module for detailed, iterative refinement.

    The Worker performs fast local computations, iterating to convergence
    within each outer Planner cycle. It combines information from:
    - h_L_prev: Previous low-level hidden state
    - h_H: High-level guidance from Planner
    - x_in: Input embedding (optional, for grounding)

    Architecture:
        1. Concatenate/combine inputs
        2. Two-layer MLP with GELU activation
        3. Residual connection from h_L_prev
        4. Post-norm RMSNorm for stability

    Args:
        hidden_dim: Dimension of hidden states (h_L, h_H). Default: 256
        mlp_ratio: Ratio for MLP hidden dimension (hidden_dim * mlp_ratio). Default: 4
        dropout: Dropout probability. Default: 0.1
        use_input_proj: Whether to use x_in input. Default: True
        norm_eps: Epsilon for RMSNorm. Default: 1e-6

    Shape:
        - h_L_prev: (batch, hidden_dim) - Previous low-level state
        - h_H: (batch, hidden_dim) - High-level guidance from Planner
        - x_in: (batch, hidden_dim) - Input embedding (optional)
        - Output: (batch, hidden_dim) - Updated low-level state

    Example:
        >>> worker = WorkerModule(hidden_dim=256)
        >>> h_L = torch.randn(32, 256)  # Previous low-level state
        >>> h_H = torch.randn(32, 256)  # Planner guidance
        >>> x_in = torch.randn(32, 256)  # Input embedding
        >>> h_L_new = worker(h_L, h_H, x_in)
        >>> h_L_new.shape
        torch.Size([32, 256])
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_input_proj: bool = True,
        norm_eps: float = 1e-6,
    ):
        """
        Initialise the Worker module.

        Args:
            hidden_dim: Dimension of all hidden states.
            mlp_ratio: Expansion ratio for MLP intermediate layer.
            dropout: Dropout probability for regularisation.
            use_input_proj: Whether to incorporate x_in input.
            norm_eps: Epsilon for normalisation layers.
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.dropout_p = dropout
        self.use_input_proj = use_input_proj

        # Calculate input dimension based on what inputs we combine
        # h_L_prev + h_H + (optionally) x_in
        input_dim = hidden_dim * (3 if use_input_proj else 2)
        mlp_hidden = hidden_dim * mlp_ratio

        # Input combination projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Core MLP with residual connection
        # Two-layer MLP: hidden_dim -> mlp_hidden -> hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Post-norm for stability (applied after residual)
        self.post_norm = RMSNorm(hidden_dim, eps=norm_eps)

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise network weights."""
        # Xavier initialisation for linear layers
        for module in [self.input_proj, *self.mlp.modules()]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        h_L_prev: torch.Tensor,
        h_H: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute one Worker iteration step.

        Combines previous low-level state with high-level guidance (and
        optionally input embedding) to produce refined low-level state.

        Args:
            h_L_prev: Previous low-level hidden state of shape (batch, hidden_dim).
            h_H: High-level hidden state from Planner of shape (batch, hidden_dim).
            x_in: Optional input embedding of shape (batch, hidden_dim).
                  Required if use_input_proj=True was set during init.

        Returns:
            Updated low-level hidden state of shape (batch, hidden_dim).

        Raises:
            ValueError: If use_input_proj=True but x_in is None.
            ValueError: If input shapes don't match hidden_dim.
        """
        # Validate inputs
        if self.use_input_proj and x_in is None:
            raise ValueError("x_in is required when use_input_proj=True")

        if h_L_prev.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"h_L_prev last dim must be {self.hidden_dim}, got {h_L_prev.shape[-1]}"
            )
        if h_H.shape[-1] != self.hidden_dim:
            raise ValueError(f"h_H last dim must be {self.hidden_dim}, got {h_H.shape[-1]}")

        # Combine inputs by concatenation
        if self.use_input_proj and x_in is not None:
            if x_in.shape[-1] != self.hidden_dim:
                raise ValueError(f"x_in last dim must be {self.hidden_dim}, got {x_in.shape[-1]}")
            combined = torch.cat([h_L_prev, h_H, x_in], dim=-1)
        else:
            combined = torch.cat([h_L_prev, h_H], dim=-1)

        # Project combined input to hidden_dim
        projected = self.input_proj(combined)

        # MLP transformation
        mlp_out = self.mlp(projected)

        # Residual connection from previous state
        h_L_new = h_L_prev + mlp_out

        # Post-norm for training stability
        h_L_new = self.post_norm(h_L_new)

        return h_L_new

    def iterate(
        self,
        h_L_init: torch.Tensor,
        h_H: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
        num_iterations: int = 5,
        convergence_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run multiple Worker iterations until convergence or max iterations.

        This is a convenience method for running the Worker loop within
        a single Planner step.

        Args:
            h_L_init: Initial low-level hidden state of shape (batch, hidden_dim).
            h_H: High-level hidden state (fixed during Worker iterations).
            x_in: Optional input embedding (fixed during Worker iterations).
            num_iterations: Maximum number of iterations. Default: 5.
            convergence_threshold: Optional L2 distance threshold for early
                stopping. If None, always runs num_iterations.

        Returns:
            Final low-level hidden state after iterations.
        """
        h_L = h_L_init

        for i in range(num_iterations):
            h_L_prev = h_L
            h_L = self.forward(h_L_prev, h_H, x_in)

            # Early stopping if converged
            if convergence_threshold is not None:
                diff = (h_L - h_L_prev).pow(2).mean().sqrt()
                if diff < convergence_threshold:
                    break

        return h_L

    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (
            f"hidden_dim={self.hidden_dim}, mlp_ratio={self.mlp_ratio}, "
            f"dropout={self.dropout_p}, use_input_proj={self.use_input_proj}"
        )


class WorkerModuleWithGating(WorkerModule):
    """
    Worker module variant with gated residual connection.

    Adds a learnable gate to control how much of the MLP output is added
    to the residual. This can help with training stability and allows the
    model to learn when to update vs. preserve the hidden state.

    The gate is computed as: gate = sigmoid(W_g @ [h_L_prev, mlp_out])
    Output: h_L_new = h_L_prev + gate * mlp_out

    Args:
        hidden_dim: Dimension of hidden states. Default: 256
        mlp_ratio: Ratio for MLP hidden dimension. Default: 4
        dropout: Dropout probability. Default: 0.1
        use_input_proj: Whether to use x_in input. Default: True
        norm_eps: Epsilon for RMSNorm. Default: 1e-6

    Example:
        >>> worker = WorkerModuleWithGating(hidden_dim=256)
        >>> h_L_new = worker(h_L, h_H, x_in)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_input_proj: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_input_proj=use_input_proj,
            norm_eps=norm_eps,
        )

        # Gating mechanism: learns when to update vs. preserve state
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Initialise gate to start with ~0.5 (balanced)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(
        self,
        h_L_prev: torch.Tensor,
        h_H: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute one Worker iteration with gated residual.

        Args:
            h_L_prev: Previous low-level hidden state.
            h_H: High-level hidden state from Planner.
            x_in: Optional input embedding.

        Returns:
            Updated low-level hidden state.
        """
        # Validate inputs (same as parent)
        if self.use_input_proj and x_in is None:
            raise ValueError("x_in is required when use_input_proj=True")

        # Combine inputs
        if self.use_input_proj and x_in is not None:
            combined = torch.cat([h_L_prev, h_H, x_in], dim=-1)
        else:
            combined = torch.cat([h_L_prev, h_H], dim=-1)

        # Project and MLP
        projected = self.input_proj(combined)
        mlp_out = self.mlp(projected)

        # Compute gate value
        gate_input = torch.cat([h_L_prev, mlp_out], dim=-1)
        gate_value = self.gate(gate_input)

        # Gated residual connection
        h_L_new = h_L_prev + gate_value * mlp_out

        # Post-norm
        h_L_new = self.post_norm(h_L_new)

        return h_L_new


class WorkerTransformer(nn.Module):
    """
    Transformer-based Worker module (Sapient-style L-level).

    Replaces the MLP-based Worker with a transformer-based ReasoningModule
    that uses self-attention + SwiGLU MLP blocks with post-normalisation.

    This matches Sapient's L_level implementation which applies input
    injection (adding h_H + input_embeddings) before transformer layers.

    Args:
        hidden_size: Model hidden dimension. Default: 256
        num_heads: Number of attention heads. Default: 4
        num_layers: Number of transformer blocks. Default: 4
        expansion: MLP expansion factor. Default: 4.0
        rms_norm_eps: Epsilon for RMS normalisation. Default: 1e-5
        max_seq_len: Maximum sequence length for RoPE. Default: 1024
        rope_base: Base for RoPE frequency computation. Default: 10000.0
        causal: Whether attention is causal. Default: False

    Shape:
        - z_L: (batch, seq_len, hidden_size) - L-level state
        - injection: (batch, seq_len, hidden_size) - Input to inject (h_H + x_in)
        - Output: (batch, seq_len, hidden_size) - Updated L-level state

    Example:
        >>> worker = WorkerTransformer(hidden_size=256, num_heads=4, num_layers=4)
        >>> z_L = torch.randn(8, 81, 256)  # 9x9 Sudoku flattened
        >>> injection = torch.randn(8, 81, 256)  # h_H + input_embeddings
        >>> z_L_new = worker(z_L, injection)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        expansion: float = 4.0,
        rms_norm_eps: float = 1e-5,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
        causal: bool = False,
    ):
        super().__init__()

        from hrm.layers.transformer import ReasoningModule, RotaryEmbedding

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # RoPE for positional encoding
        head_dim = hidden_size // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_base,
        )

        # Transformer reasoning module
        self.reasoning = ReasoningModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            expansion=expansion,
            rms_norm_eps=rms_norm_eps,
            causal=causal,
        )

    def forward(
        self,
        z_L: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply L-level transformer update.

        Args:
            z_L: Current L-level state of shape (batch, seq_len, hidden_size).
            injection: Input injection (typically z_H + input_embeddings).

        Returns:
            Updated L-level state of shape (batch, seq_len, hidden_size).
        """
        cos_sin = self.rotary_emb()
        return self.reasoning(z_L, injection, cos_sin=cos_sin)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}"
        )
