# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
HRM_4x4: Complete Hierarchical Reasoning Model for 4x4 Sudoku

Issue #8: Assemble Complete HRM Model

This module provides the complete Hierarchical Reasoning Model (HRM) for
solving 4x4 Sudoku puzzles. It integrates all components:
    - InputNetwork: Converts grid to hidden representation
    - WorkerModule: Low-level iterative refinement (f_L)
    - PlannerModule: High-level planning updates (f_H)
    - OutputNetwork: Decodes actions (cell + digit predictions)

The HRM uses hierarchical iteration:
    - Outer Loop (K cycles): Planner updates h_H
    - Inner Loop (T steps): Worker refines h_L under fixed h_H

This architecture enables sustained computation (200+ effective steps)
vs. 20-30 for standard RNNs, preventing premature convergence.

Configuration Defaults (4x4 Sudoku):
    - hidden_dim: 64
    - n_outer_cycles: 5 (K)
    - n_inner_steps: 10 (T)
    - convergence_threshold: 1e-3
    - dropout: 0.1

Example:
    >>> # Create model for 4x4 Sudoku
    >>> model = HRM_4x4()
    >>>
    >>> # Input puzzle (0 = empty, 1-4 = digits)
    >>> puzzle = torch.tensor([[
    ...     [0, 2, 0, 4],
    ...     [4, 0, 2, 0],
    ...     [0, 4, 0, 2],
    ...     [2, 0, 4, 0]
    ... ]])
    >>>
    >>> # Forward pass with execution trace
    >>> outputs = model(puzzle)
    >>> cell_logits = outputs['cell_logits']      # (batch, 16)
    >>> digit_logits = outputs['digit_logits']    # (batch, 4)
    >>> trace = outputs['trace']                   # Execution trace
    >>>
    >>> # Get predictions
    >>> cell_idx = cell_logits.argmax(dim=-1)      # Which cell to fill
    >>> digit_idx = digit_logits.argmax(dim=-1)    # Which digit (0-indexed, add 1)
    >>>
    >>> # For training: compute losses
    >>> halt_penalty = model.get_halt_penalty(outputs)
    >>> convergence_loss = model.get_convergence_loss(outputs)

Reference:
    - HRM Architecture Specification
    - hrm.core.iteration_outer: Hierarchical iteration implementation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.input_network import InputNetwork
from hrm.layers.worker import WorkerModule
from hrm.layers.planner import PlannerModule
from hrm.layers.output_network import OutputNetwork
from hrm.prototype.core.iteration_outer import (
    hierarchical_iteration,
    OuterLoopStats,
)
from hrm.prototype.core.iteration import compute_residual


@dataclass
class ExecutionTrace:
    """
    Trace of HRM execution for analysis and debugging.

    Records the evolution of hidden states and convergence statistics
    throughout the hierarchical iteration process.

    Attributes:
        outer_stats: Statistics from hierarchical iteration.
        h_L_history: List of h_L states at end of each outer cycle.
        h_H_history: List of h_H states at each outer cycle.
        x_in: Input embedding used during execution.
        final_h_L: Final low-level hidden state.
        final_h_H: Final high-level hidden state.
    """

    outer_stats: OuterLoopStats
    h_L_history: List[torch.Tensor] = field(default_factory=list)
    h_H_history: List[torch.Tensor] = field(default_factory=list)
    x_in: Optional[torch.Tensor] = None
    final_h_L: Optional[torch.Tensor] = None
    final_h_H: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return (
            f"ExecutionTrace(cycles={self.outer_stats.num_cycles}, "
            f"total_steps={self.outer_stats.total_inner_steps}, "
            f"converged={self.outer_stats.converged})"
        )

    @property
    def total_computation_steps(self) -> int:
        """Total Worker iterations across all cycles."""
        return self.outer_stats.total_inner_steps

    @property
    def effective_depth(self) -> int:
        """Effective computational depth (outer cycles × inner steps)."""
        return (
            self.outer_stats.num_cycles * max(s.num_steps for s in self.outer_stats.cycle_stats)
            if self.outer_stats.cycle_stats
            else self.outer_stats.total_inner_steps
        )


class HRM_4x4(nn.Module):
    """
    Hierarchical Reasoning Model for 4x4 Sudoku.

    Integrates all HRM components into a complete model that:
    1. Embeds input puzzles into hidden space
    2. Runs hierarchical iteration (Planner + Worker loops)
    3. Decodes final state to cell and digit predictions

    The model maintains learned initial states for both the Planner (h_H)
    and Worker (h_L), which are expanded to batch size during forward pass.

    Args:
        hidden_dim: Dimension of hidden states. Default: 64.
        embed_dim: Embedding dimension per cell. Default: 16.
        n_outer_cycles: Number of Planner cycles (K). Default: 5.
        n_inner_steps: Worker iterations per cycle (T). Default: 10.
        convergence_threshold: Threshold for early stopping. Default: 1e-3.
        dropout: Dropout probability. Default: 0.1.
        track_history: Whether to record execution trace. Default: True.

    Attributes:
        input_network: Converts grid to hidden representation.
        worker: Low-level refinement module (f_L).
        planner: High-level planning module (f_H).
        output_network: Decodes to action predictions.
        h_L_init: Learned initial low-level state.
        h_H_init: Learned initial high-level state.

    Shape:
        - Input: (batch, 4, 4) integer grid with values in [0, 4]
        - Output: Dict with:
            - 'cell_logits': (batch, 16) cell selection logits
            - 'digit_logits': (batch, 4) digit selection logits
            - 'trace': ExecutionTrace with convergence info

    Example:
        >>> model = HRM_4x4(hidden_dim=64, n_outer_cycles=5)
        >>> puzzle = torch.randint(0, 5, (8, 4, 4))
        >>> outputs = model(puzzle)
        >>> cell_logits = outputs['cell_logits']
        >>> digit_logits = outputs['digit_logits']
        >>>
        >>> # Training loss
        >>> cell_loss = F.cross_entropy(cell_logits, target_cells)
        >>> digit_loss = F.cross_entropy(digit_logits, target_digits)
        >>> halt_penalty = model.get_halt_penalty(outputs)
        >>> total_loss = cell_loss + digit_loss + 0.01 * halt_penalty
    """

    # Class constants for 4x4 Sudoku
    VOCAB_SIZE = 5  # 0=empty, 1-4=digits
    GRID_SIZE = 4
    NUM_CELLS = 16
    NUM_DIGITS = 4

    def __init__(
        self,
        hidden_dim: int = 64,
        embed_dim: int = 16,
        n_outer_cycles: int = 5,
        n_inner_steps: int = 10,
        convergence_threshold: float = 1e-3,
        dropout: float = 0.1,
        track_history: bool = True,
    ):
        """
        Initialise HRM_4x4 model.

        Args:
            hidden_dim: Dimension of all hidden states.
            embed_dim: Cell embedding dimension before projection.
            n_outer_cycles: Maximum Planner update cycles (K).
            n_inner_steps: Worker iterations per cycle (T).
            convergence_threshold: L2 threshold for convergence detection.
            dropout: Dropout probability for regularisation.
            track_history: Whether to track execution trace.
        """
        super().__init__()

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if n_outer_cycles <= 0:
            raise ValueError(f"n_outer_cycles must be positive, got {n_outer_cycles}")
        if n_inner_steps <= 0:
            raise ValueError(f"n_inner_steps must be positive, got {n_inner_steps}")
        if convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold must be positive, got {convergence_threshold}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Store configuration
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_outer_cycles = n_outer_cycles
        self.n_inner_steps = n_inner_steps
        self.convergence_threshold = convergence_threshold
        self.dropout = dropout
        self.track_history = track_history

        # Component 1: Input Network (embedding + projection)
        self.input_network = InputNetwork(
            vocab_size=self.VOCAB_SIZE,
            grid_size=self.GRID_SIZE,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_positional=True,
        )

        # Component 2: Worker Module (low-level refinement f_L)
        self.worker = WorkerModule(
            hidden_dim=hidden_dim,
            mlp_ratio=4,
            dropout=dropout,
            use_input_proj=True,  # Worker uses x_in for grounding
        )

        # Component 3: Planner Module (high-level planning f_H)
        self.planner = PlannerModule(
            hidden_dim=hidden_dim,
            mlp_ratio=2,
            dropout=dropout,
        )

        # Component 4: Output Network (action decoder f_O)
        self.output_network = OutputNetwork(
            hidden_dim=hidden_dim,
            grid_size=self.GRID_SIZE,
            use_shared_mlp=True,
            mlp_ratio=2,
            dropout=dropout,
        )

        # Learned Initial States
        # These are learned parameters that provide starting points for
        # hierarchical iteration. They are expanded to batch size in forward().
        self.h_L_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.h_H_init = nn.Parameter(torch.zeros(1, hidden_dim))

        # Initialise learned states
        self._init_states()

    def _init_states(self) -> None:
        """Initialise learned initial states."""
        # Small random initialisation for diversity
        nn.init.normal_(self.h_L_init, mean=0.0, std=0.02)
        nn.init.normal_(self.h_H_init, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete HRM forward pass.

        Processes a batch of 4x4 Sudoku puzzles through:
        1. Input embedding
        2. Hierarchical iteration (Worker + Planner)
        3. Output decoding

        Args:
            x: Input tensor of shape (batch, 4, 4) with integer values
               in range [0, 4] where 0=empty, 1-4=digits.
            return_intermediates: If True, include h_L and h_H history
               in the trace. Default: False.

        Returns:
            Dictionary containing:
                - 'cell_logits': (batch, 16) cell selection logits
                - 'digit_logits': (batch, 4) digit selection logits
                - 'trace': ExecutionTrace with convergence statistics
                - 'h_L_final': Final low-level hidden state
                - 'h_H_final': Final high-level hidden state

        Raises:
            ValueError: If input shape is not (batch, 4, 4).
        """
        # Validate input shape
        if x.dim() != 3 or x.shape[1:] != (self.GRID_SIZE, self.GRID_SIZE):
            raise ValueError(f"Expected input shape (batch, 4, 4), got {x.shape}")

        batch_size = x.shape[0]
        device = x.device

        # Step 1: Embed input puzzle
        x_in = self.input_network(x)  # (batch, hidden_dim)

        # Step 2: Expand learned initial states to batch size
        h_L = self.h_L_init.expand(batch_size, -1).to(device)  # (batch, hidden_dim)
        h_H = self.h_H_init.expand(batch_size, -1).to(device)  # (batch, hidden_dim)

        # Step 3: Hierarchical iteration
        h_L_final, h_H_final, outer_stats = hierarchical_iteration(
            worker=self.worker,
            planner=self.planner,
            h_L_init=h_L,
            h_H_init=h_H,
            x_in=x_in,
            n_outer_cycles=self.n_outer_cycles,
            n_inner_steps=self.n_inner_steps,
            inner_convergence_threshold=self.convergence_threshold,
            outer_convergence_threshold=self.convergence_threshold,
            track_history=self.track_history,
        )

        # Step 4: Decode to action predictions
        # Use final h_H state for prediction (high-level reasoning result)
        cell_logits, digit_logits = self.output_network(h_H_final)

        # Step 5: Build execution trace
        trace = ExecutionTrace(
            outer_stats=outer_stats,
            x_in=x_in if return_intermediates else None,
            final_h_L=h_L_final,
            final_h_H=h_H_final,
        )

        return {
            "cell_logits": cell_logits,
            "digit_logits": digit_logits,
            "trace": trace,
            "h_L_final": h_L_final,
            "h_H_final": h_H_final,
        }

    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action predictions for inference.

        Convenience method that runs forward pass and returns
        predicted cell and digit indices.

        Args:
            x: Input tensor of shape (batch, 4, 4).

        Returns:
            Tuple of (cell_indices, digit_indices):
                - cell_indices: (batch,) predicted cell index [0, 15]
                - digit_indices: (batch,) predicted digit index [0, 3]
                  (add 1 to get actual Sudoku digit 1-4)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            cell_indices = outputs["cell_logits"].argmax(dim=-1)
            digit_indices = outputs["digit_logits"].argmax(dim=-1)
        return cell_indices, digit_indices

    def get_halt_penalty(
        self,
        outputs: Dict[str, Any],
        lambda_halt: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute halting penalty for encouraging efficient computation.

        The halt penalty encourages the model to converge early by penalising
        the number of computation steps taken. This creates a speed-accuracy
        trade-off controlled by lambda_halt.

        Formula:
            penalty = lambda_halt * (total_inner_steps / max_possible_steps)

        Args:
            outputs: Output dictionary from forward() containing 'trace'.
            lambda_halt: Weight for the halt penalty. Default: 0.01.

        Returns:
            Scalar tensor with the halt penalty value.

        Note:
            Research shows that for HRM-style architectures, using maximum
            iterations often yields best results. This penalty is provided
            for experimentation but may not improve performance.
        """
        trace: ExecutionTrace = outputs["trace"]
        max_steps = self.n_outer_cycles * self.n_inner_steps
        actual_steps = trace.outer_stats.total_inner_steps

        # Normalised ponder cost
        ponder_cost = actual_steps / max_steps

        return torch.tensor(lambda_halt * ponder_cost, device=outputs["cell_logits"].device)

    def get_convergence_loss(
        self,
        outputs: Dict[str, Any],
        target_residual: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute convergence loss to encourage stable fixed-points.

        This loss penalises the final residual, encouraging the model
        to reach a stable equilibrium state. Lower final residuals
        indicate better convergence.

        Formula:
            loss = max(0, final_residual - target_residual)

        Args:
            outputs: Output dictionary from forward() containing 'trace'.
            target_residual: Target residual to achieve. Default: 1e-4.

        Returns:
            Scalar tensor with the convergence loss value.

        Note:
            This loss encourages the Worker to converge to a stable
            fixed point within each Planner cycle. It helps training
            stability but may not be necessary for all applications.
        """
        trace: ExecutionTrace = outputs["trace"]
        final_residual = trace.outer_stats.final_h_H_residual

        # Hinge-style loss: only penalise if above target
        loss = max(0.0, final_residual - target_residual)

        return torch.tensor(loss, device=outputs["cell_logits"].device)

    def get_state_dynamics(
        self,
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyse the dynamics of hidden state evolution.

        Provides metrics about how the hidden states evolved during
        hierarchical iteration, useful for debugging and analysis.

        Args:
            outputs: Output dictionary from forward().

        Returns:
            Dictionary with dynamics metrics:
                - 'h_H_residuals': Per-cycle h_H change magnitudes
                - 'inner_convergence_rates': Per-cycle Worker convergence
                - 'total_steps': Total computation steps
                - 'outer_converged': Whether outer loop converged
        """
        trace: ExecutionTrace = outputs["trace"]
        stats = trace.outer_stats

        return {
            "h_H_residuals": stats.h_H_residual_history,
            "inner_convergence_rates": [
                s.convergence_rate for s in stats.cycle_stats if s.convergence_rate is not None
            ],
            "total_steps": stats.total_inner_steps,
            "outer_converged": stats.converged,
            "cycles_completed": stats.num_cycles,
        }

    def extra_repr(self) -> str:
        """Return string representation of model configuration."""
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"embed_dim={self.embed_dim}, "
            f"n_outer_cycles={self.n_outer_cycles}, "
            f"n_inner_steps={self.n_inner_steps}, "
            f"convergence_threshold={self.convergence_threshold}, "
            f"dropout={self.dropout}"
        )


def create_hrm_4x4(
    hidden_dim: int = 64, n_outer_cycles: int = 5, n_inner_steps: int = 10, **kwargs
) -> HRM_4x4:
    """
    Factory function to create HRM_4x4 model.

    Provides sensible defaults for 4x4 Sudoku while allowing customisation.

    Args:
        hidden_dim: Dimension of hidden states. Default: 64.
        n_outer_cycles: Planner cycles (K). Default: 5.
        n_inner_steps: Worker iterations per cycle (T). Default: 10.
        **kwargs: Additional arguments passed to HRM_4x4.

    Returns:
        Configured HRM_4x4 model instance.

    Example:
        >>> # Default configuration
        >>> model = create_hrm_4x4()
        >>>
        >>> # Custom configuration
        >>> model = create_hrm_4x4(
        ...     hidden_dim=128,
        ...     n_outer_cycles=10,
        ...     n_inner_steps=20,
        ...     dropout=0.2
        ... )
    """
    return HRM_4x4(
        hidden_dim=hidden_dim, n_outer_cycles=n_outer_cycles, n_inner_steps=n_inner_steps, **kwargs
    )
