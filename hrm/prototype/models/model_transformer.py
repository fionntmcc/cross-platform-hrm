# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Transformer-based HRM Model (Sapient-compatible)

This module implements a transformer-based Hierarchical Reasoning Model
that closely follows Sapient Inc's architecture for solving variable-size
puzzles (9x9 Sudoku, 30x30 mazes, etc.).

Key differences from MLP-based HRM:
- Sequence-based I/O: (batch, seq_len) tokens instead of 2D grids
- Transformer blocks with self-attention + SwiGLU MLP
- RoPE positional encoding instead of learned position embeddings
- Functional RMSNorm (no learnable parameters)
- nn.Buffer for initial states (non-trainable)
- Per-position Q-heads for halting decisions

Architecture:
    1. InputNetworkTransformer: Token embedding + optional puzzle embedding
    2. WorkerTransformer: L-level transformer reasoning
    3. PlannerTransformer: H-level transformer reasoning
    4. OutputNetworkTransformer: Single LM head for vocab logits
    5. QHaltingHeadTransformer: Per-position Q-values for ACT

Reference:
    - Sapient Inc HRM Implementation
    - "Hierarchical Reasoning Model for Long-Horizon Planning"
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.transformer import trunc_normal_init_
from hrm.layers.input_network import InputNetworkTransformer
from hrm.layers.output_network import OutputNetworkTransformer
from hrm.layers.worker import WorkerTransformer
from hrm.layers.planner import PlannerTransformer
from hrm.prototype.core.halting import QHaltingHeadTransformer


@dataclass
class HRMTransformerConfig:
    """
    Configuration for transformer-based HRM.

    Attributes:
        vocab_size: Size of token vocabulary (e.g., 10 for 0-9 Sudoku).
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_layers_L: Number of transformer blocks in L-level Worker.
        num_layers_H: Number of transformer blocks in H-level Planner.
        expansion: MLP expansion factor.
        max_seq_len: Maximum sequence length for RoPE.
        H_cycles: Number of outer (H) iterations.
        L_steps: Number of inner (L) iterations per H cycle.
        num_puzzles: Number of puzzle types for multi-task.
        use_puzzle_embedding: Whether to use puzzle type embedding.
        rms_norm_eps: Epsilon for RMS normalisation.
        causal: Whether to use causal attention.
        dtype: Computation dtype.
    """

    vocab_size: int = 10  # 0-9 for Sudoku
    hidden_size: int = 256
    num_heads: int = 4
    num_layers_L: int = 4
    num_layers_H: int = 4
    expansion: float = 4.0
    max_seq_len: int = 1024
    H_cycles: int = 4
    L_steps: int = 8
    num_puzzles: int = 1
    use_puzzle_embedding: bool = False
    rms_norm_eps: float = 1e-5
    causal: bool = False
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def for_sudoku_9x9(cls) -> "HRMTransformerConfig":
        """Configuration for 9x9 Sudoku (seq_len=81)."""
        return cls(
            vocab_size=10,  # 0=empty, 1-9=digits
            hidden_size=256,
            num_heads=4,
            num_layers_L=4,
            num_layers_H=4,
            max_seq_len=128,  # 81 + padding
            H_cycles=4,
            L_steps=8,
        )

    @classmethod
    def for_maze_30x30(cls) -> "HRMTransformerConfig":
        """Configuration for 30x30 mazes (seq_len=900)."""
        return cls(
            vocab_size=4,  # wall, empty, start, goal
            hidden_size=256,
            num_heads=4,
            num_layers_L=4,
            num_layers_H=4,
            max_seq_len=1024,  # 900 + padding
            H_cycles=6,
            L_steps=12,
        )


class HRMTransformer(nn.Module):
    """
    Transformer-based Hierarchical Reasoning Model (Sapient-compatible).

    This model implements the full HRM architecture using transformer blocks
    with self-attention, matching Sapient's proven implementation that can
    solve 9x9 Sudoku and 30x30 mazes.

    Architecture Overview:
        Input Stage:
            x (batch, seq_len) -> InputNetworkTransformer -> z_input (batch, seq_len, hidden_size)

        Hierarchical Iteration:
            for h in H_cycles:
                for l in L_steps:
                    z_L = WorkerTransformer(z_L, z_input)  # L-level refinement
                z_H = PlannerTransformer(z_H, z_L)  # H-level update

        Output Stage:
            OutputNetworkTransformer(z_H) -> logits (batch, seq_len, vocab_size)

    Args:
        config: HRMTransformerConfig with model hyperparameters.

    Attributes:
        input_net: Token embedding and projection.
        worker: L-level transformer reasoning module.
        planner: H-level transformer reasoning module.
        output_net: LM head for vocabulary prediction.
        q_head_L: Q-halting head for L-level.
        q_head_H: Q-halting head for H-level.
        z_L_init: Buffer for initial L-level state (non-trainable).
        z_H_init: Buffer for initial H-level state (non-trainable).

    Example:
        >>> config = HRMTransformerConfig.for_sudoku_9x9()
        >>> model = HRMTransformer(config)
        >>> x = torch.randint(0, 10, (8, 81))  # Batch of 9x9 Sudoku
        >>> output = model(x)
        >>> output['logits'].shape
        torch.Size([8, 81, 10])
    """

    def __init__(self, config: HRMTransformerConfig):
        super().__init__()

        self.config = config

        # Input Network
        self.input_net = InputNetworkTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_puzzles=config.num_puzzles,
            use_puzzle_embedding=config.use_puzzle_embedding,
            dtype=config.dtype,
        )

        # L-Level Worker (Transformer)
        self.worker = WorkerTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers_L,
            expansion=config.expansion,
            rms_norm_eps=config.rms_norm_eps,
            max_seq_len=config.max_seq_len,
            causal=config.causal,
        )

        # H-Level Planner (Transformer)
        self.planner = PlannerTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers_H,
            expansion=config.expansion,
            rms_norm_eps=config.rms_norm_eps,
            max_seq_len=config.max_seq_len,
            causal=config.causal,
        )

        # Output Network (LM Head)
        self.output_net = OutputNetworkTransformer(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            tie_weights=False,
            dtype=config.dtype,
        )

        # Q-Halting Heads (for ACT)
        self.q_head_L = QHaltingHeadTransformer(
            hidden_size=config.hidden_size,
            dtype=config.dtype,
        )
        self.q_head_H = QHaltingHeadTransformer(
            hidden_size=config.hidden_size,
            dtype=config.dtype,
        )

        # Initial States (Non-trainable Buffers)
        # Sapient uses nn.Buffer with truncated normal init (not trainable)
        # Shape: (1, 1, hidden_size) for broadcasting to (batch, seq_len, hidden_size)
        self.register_buffer("z_L_init", torch.empty(1, 1, config.hidden_size))
        self.register_buffer("z_H_init", torch.empty(1, 1, config.hidden_size))

        # Initialise buffers with truncated normal
        self._init_buffers()

    def _init_buffers(self) -> None:
        """Initialise state buffers with truncated normal."""
        trunc_normal_init_(self.z_L_init, std=0.02)
        trunc_normal_init_(self.z_H_init, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        puzzle_ids: Optional[torch.Tensor] = None,
        H_cycles: Optional[int] = None,
        L_steps: Optional[int] = None,
        return_intermediates: bool = False,
        use_act: bool = False,
    ) -> Dict[str, Any]:
        """
        Run transformer HRM forward pass.

        Args:
            x: Input tokens of shape (batch, seq_len) in [0, vocab_size).
            puzzle_ids: Optional puzzle type indices of shape (batch,).
            H_cycles: Override number of H cycles (default: config.H_cycles).
            L_steps: Override number of L steps (default: config.L_steps).
            return_intermediates: Include z_L, z_H history in output.
            use_act: Enable adaptive computation time with Q-halting.

        Returns:
            Dictionary containing:
                - 'logits': (batch, seq_len, vocab_size) vocabulary logits
                - 'predictions': (batch, seq_len) argmax predictions
                - 'z_L_final': Final L-level state
                - 'z_H_final': Final H-level state
                - 'h_cycles_used': Number of H cycles executed
                - 'l_steps_used': List of L steps per H cycle
                - 'intermediates': Optional dict with z_L_history, z_H_history
        """
        batch_size, seq_len = x.shape
        H_cycles = H_cycles or self.config.H_cycles
        L_steps = L_steps or self.config.L_steps

        device = x.device
        dtype = self.config.dtype

        # Step 1: Embed input tokens
        # (batch, seq_len) -> (batch, seq_len, hidden_size)
        z_input = self.input_net(x, puzzle_ids)

        # Step 2: Initialise states by broadcasting buffers
        # (1, 1, hidden_size) -> (batch, seq_len, hidden_size)
        z_L = self.z_L_init.expand(batch_size, seq_len, -1).to(dtype)
        z_H = self.z_H_init.expand(batch_size, seq_len, -1).to(dtype)

        # Tracking
        z_L_history: List[torch.Tensor] = []
        z_H_history: List[torch.Tensor] = []
        l_steps_used: List[int] = []
        h_cycles_used = 0

        # Step 3: Hierarchical iteration
        for h in range(H_cycles):
            # L-level iteration (Worker)
            l_count = 0
            for l in range(L_steps):
                z_L = self.worker(z_L, z_input)
                l_count += 1

                if return_intermediates:
                    z_L_history.append(z_L.detach())

                # Optional ACT halting for L-level
                if use_act and l >= 2:  # Minimum 2 L steps
                    should_halt, _ = self.q_head_L.should_halt(
                        z_L, l, min_cycles=2, training=self.training
                    )
                    if should_halt:
                        break

            l_steps_used.append(l_count)

            # H-level update (Planner)
            z_H = self.planner(z_H, z_L)
            h_cycles_used += 1

            if return_intermediates:
                z_H_history.append(z_H.detach())

            # Optional ACT halting for H-level
            if use_act and h >= 2:  # Minimum 2 H cycles
                should_halt, _ = self.q_head_H.should_halt(
                    z_H, h, min_cycles=2, training=self.training
                )
                if should_halt:
                    break

        # Step 4: Output projection
        # (batch, seq_len, hidden_size) -> (batch, seq_len, vocab_size)
        logits = self.output_net(z_H)
        predictions = logits.argmax(dim=-1)

        # Build output dict
        output = {
            "logits": logits,
            "predictions": predictions,
            "z_L_final": z_L,
            "z_H_final": z_H,
            "h_cycles_used": h_cycles_used,
            "l_steps_used": l_steps_used,
        }

        if return_intermediates:
            output["intermediates"] = {
                "z_L_history": z_L_history,
                "z_H_history": z_H_history,
            }

        return output

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get predictions for input tokens.

        Args:
            x: Input tokens of shape (batch, seq_len).
            **kwargs: Additional arguments for forward().

        Returns:
            Predictions of shape (batch, seq_len).
        """
        with torch.no_grad():
            output = self.forward(x, **kwargs)
        return output["predictions"]

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_buffers(self) -> int:
        """Total number of non-trainable buffer elements."""
        return sum(b.numel() for b in self.buffers())

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.config.vocab_size}, "
            f"hidden_size={self.config.hidden_size}, "
            f"H_cycles={self.config.H_cycles}, "
            f"L_steps={self.config.L_steps}"
        )


def create_hrm_transformer(puzzle_type: str = "sudoku_9x9", **kwargs) -> HRMTransformer:
    """
    Factory function to create HRMTransformer for specific puzzle types.

    Args:
        puzzle_type: One of 'sudoku_9x9', 'maze_30x30', or 'custom'.
        **kwargs: Override config parameters.

    Returns:
        Configured HRMTransformer model.

    Example:
        >>> model = create_hrm_transformer('sudoku_9x9')
        >>> model = create_hrm_transformer('maze_30x30', H_cycles=8)
        >>> model = create_hrm_transformer('custom', vocab_size=20, hidden_size=512)
    """
    if puzzle_type == "sudoku_9x9":
        config = HRMTransformerConfig.for_sudoku_9x9()
    elif puzzle_type == "maze_30x30":
        config = HRMTransformerConfig.for_maze_30x30()
    elif puzzle_type == "custom":
        config = HRMTransformerConfig()
    else:
        raise ValueError(f"Unknown puzzle_type: {puzzle_type}")

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    return HRMTransformer(config)
