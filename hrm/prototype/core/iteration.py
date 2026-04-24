# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Fixed-Point Iteration for HRM (Inner Loop)

Issue #21: Implement Fixed-Point Iteration Loop

This module implements the inner loop of the HRM where the Worker module
iteratively refines the low-level hidden state h_L until convergence or
maximum steps are reached.

Key Properties:
    1. h_H is FIXED (detached) during inner loop - one-step gradient approximation
    2. Worker iterates: h_L^(t+1) = f_L(h_L^t, h_H_fixed, x_in)
    3. Convergence measured by residual: ||h_L^(t) - h_L^(t-1)||
    4. Early stopping when residual < threshold

The one-step gradient approximation is critical for training stability:
    - Gradients only flow through the FINAL iteration step
    - This avoids expensive backpropagation through time (BPTT)
    - Memory complexity is O(1) instead of O(T)

Reference:
    - HRM Paper: Fixed-point iteration for low-level refinement
    - Deep Equilibrium Models (Bai et al., 2019)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn


@dataclass
class IterationStats:
    """
    Statistics from fixed-point iteration.

    Tracks convergence information from the inner loop, useful for
    monitoring training progress and debugging.

    Attributes:
        num_steps: Number of iteration steps taken.
        converged: Whether convergence threshold was reached.
        final_residual: Final residual ||h_L^(t) - h_L^(t-1)||.
        residual_history: List of residuals at each step (if tracked).
    """

    num_steps: int
    converged: bool
    final_residual: float
    residual_history: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"IterationStats(steps={self.num_steps}, "
            f"converged={self.converged}, "
            f"final_residual={self.final_residual:.6f})"
        )

    @property
    def convergence_rate(self) -> Optional[float]:
        """
        Compute average convergence rate from residual history.

        Returns ratio of final to initial residual, or None if history
        has fewer than 2 entries.
        """
        if len(self.residual_history) < 2:
            return None
        if self.residual_history[0] == 0:
            return 0.0
        return self.residual_history[-1] / self.residual_history[0]


def compute_residual(
    h_new: torch.Tensor,
    h_prev: torch.Tensor,
    norm_type: str = "l2",
) -> torch.Tensor:
    """
    Compute residual between consecutive hidden states.

    The residual measures how much the hidden state changed between
    iterations, used to determine convergence.

    Args:
        h_new: New hidden state of shape (batch, hidden_dim).
        h_prev: Previous hidden state of shape (batch, hidden_dim).
        norm_type: Type of norm to use. One of:
            - "l2": L2 norm (Euclidean distance) - DEFAULT
            - "l1": L1 norm (Manhattan distance)
            - "linf": L-infinity norm (max absolute difference)

    Returns:
        Residual tensor of shape (batch,) with per-sample residuals.

    Example:
        >>> h1 = torch.tensor([[1.0, 0.0, 0.0]])
        >>> h2 = torch.tensor([[0.0, 0.0, 0.0]])
        >>> residual = compute_residual(h1, h2)
        >>> residual
        tensor([1.])
    """
    diff = h_new - h_prev

    if norm_type == "l2":
        # ||h_new - h_prev||_2 for each sample
        residual = torch.norm(diff, p=2, dim=-1)
    elif norm_type == "l1":
        residual = torch.norm(diff, p=1, dim=-1)
    elif norm_type == "linf":
        residual = torch.abs(diff).max(dim=-1).values
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. " f"Expected one of: 'l2', 'l1', 'linf'")

    return residual


def fixed_point_iteration(
    worker: nn.Module,
    h_L_init: torch.Tensor,
    h_H: torch.Tensor,
    x_in: Optional[torch.Tensor] = None,
    n_steps: int = 10,
    convergence_threshold: Optional[float] = None,
    norm_type: str = "l2",
    track_history: bool = True,
) -> Tuple[torch.Tensor, IterationStats]:
    """
    Run fixed-point iteration with the Worker module.

    This implements the inner loop of HRM where the Worker iteratively
    refines h_L while h_H remains FIXED (detached). This is critical for
    the one-step gradient approximation used in training.

    Algorithm:
        h_H_fixed = h_H.detach()  # Critical: detach for one-step gradient
        for step in range(n_steps):
            h_L_prev = h_L
            h_L = worker(h_L_prev, h_H_fixed, x_in)
            residual = ||h_L - h_L_prev||
            if residual < threshold:
                break  # Early stopping

    Args:
        worker: WorkerModule instance for computing h_L updates.
        h_L_init: Initial low-level hidden state of shape (batch, hidden_dim).
        h_H: High-level hidden state of shape (batch, hidden_dim).
            Will be DETACHED during iteration (one-step gradient approx).
        x_in: Optional input embedding of shape (batch, hidden_dim).
        n_steps: Maximum number of iteration steps. Default: 10.
        convergence_threshold: Optional threshold for early stopping.
            If mean residual < threshold, iteration stops early.
            If None, always runs n_steps iterations.
        norm_type: Norm type for residual computation. Default: "l2".
        track_history: Whether to track residual history. Default: True.

    Returns:
        Tuple of (h_L_final, stats):
            - h_L_final: Final low-level hidden state (batch, hidden_dim).
            - stats: IterationStats with convergence information.

    Critical Property:
        h_H is DETACHED during the inner loop. This implements the
        one-step gradient approximation from the HRM paper, where
        gradients only flow through the final iteration step, not
        through all T steps. This:
        - Reduces memory from O(T) to O(1)
        - Avoids vanishing/exploding gradients through deep iteration
        - Enables training with many iteration steps

    Example:
        >>> worker = WorkerModule(hidden_dim=64)
        >>> h_L = torch.randn(8, 64)
        >>> h_H = torch.randn(8, 64)
        >>> x_in = torch.randn(8, 64)
        >>> h_L_final, stats = fixed_point_iteration(
        ...     worker, h_L, h_H, x_in,
        ...     n_steps=10,
        ...     convergence_threshold=1e-4
        ... )
        >>> print(stats)
        IterationStats(steps=7, converged=True, final_residual=0.000089)
    """
    # Validate inputs
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if convergence_threshold is not None and convergence_threshold <= 0:
        raise ValueError(f"convergence_threshold must be positive, got {convergence_threshold}")

    # CRITICAL: Detach h_H for one-step gradient approximation
    # This ensures gradients only flow through the final iteration step,
    # not through all T steps. This is essential for:
    # 1. Memory efficiency: O(1) instead of O(T)
    # 2. Training stability: no vanishing/exploding gradients through time
    # 3. Matching the HRM paper's training methodology
    h_H_fixed = h_H.detach()

    # Initialise iteration state
    h_L = h_L_init
    residual_history: List[float] = []
    converged = False
    final_residual = 0.0
    steps_taken = 0

    # Fixed-point iteration loop (inner loop)
    for step in range(n_steps):
        # Store previous state for residual computation
        h_L_prev = h_L

        # Worker update: h_L^(t+1) = f_L(h_L^t, h_H_fixed, x_in)
        h_L = worker(h_L_prev, h_H_fixed, x_in)

        # Compute residual: ||h_L^(t+1) - h_L^(t)||
        residual = compute_residual(h_L, h_L_prev, norm_type=norm_type)
        mean_residual = residual.mean().item()
        final_residual = mean_residual
        steps_taken = step + 1

        # Track history if requested
        if track_history:
            residual_history.append(mean_residual)

        # Early stopping check
        if convergence_threshold is not None and mean_residual < convergence_threshold:
            converged = True
            break

    # Build statistics
    stats = IterationStats(
        num_steps=steps_taken,
        converged=converged,
        final_residual=final_residual,
        residual_history=residual_history if track_history else [],
    )

    return h_L, stats


class FixedPointIterator(nn.Module):
    """
    Module wrapper for fixed-point iteration.

    Provides a nn.Module interface for the fixed-point iteration,
    making it easier to integrate into larger models and handle
    training/eval mode switching.

    The iterator wraps a Worker module and handles:
    - h_H detachment (one-step gradient approximation)
    - Convergence tracking
    - Early stopping
    - Statistics collection

    Args:
        worker: WorkerModule for computing h_L updates.
        n_steps: Maximum number of iteration steps. Default: 10.
        convergence_threshold: Optional threshold for early stopping.
        norm_type: Norm type for residual computation. Default: "l2".
        track_history: Whether to track residual history. Default: True.

    Example:
        >>> worker = WorkerModule(hidden_dim=64)
        >>> iterator = FixedPointIterator(worker, n_steps=10)
        >>> h_L_final, stats = iterator(h_L_init, h_H, x_in)
    """

    def __init__(
        self,
        worker: nn.Module,
        n_steps: int = 10,
        convergence_threshold: Optional[float] = None,
        norm_type: str = "l2",
        track_history: bool = True,
    ):
        """
        Initialise FixedPointIterator.

        Args:
            worker: WorkerModule for low-level refinement.
            n_steps: Maximum iteration steps.
            convergence_threshold: Early stopping threshold (optional).
            norm_type: Norm for residual computation.
            track_history: Whether to record residual history.
        """
        super().__init__()

        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if convergence_threshold is not None and convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold must be positive, got {convergence_threshold}")

        self.worker = worker
        self.n_steps = n_steps
        self.convergence_threshold = convergence_threshold
        self.norm_type = norm_type
        self.track_history = track_history

    def forward(
        self,
        h_L_init: torch.Tensor,
        h_H: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, IterationStats]:
        """
        Run fixed-point iteration.

        Args:
            h_L_init: Initial low-level hidden state (batch, hidden_dim).
            h_H: High-level hidden state (batch, hidden_dim).
                Will be detached internally.
            x_in: Optional input embedding (batch, hidden_dim).
            n_steps: Override default number of steps (optional).
            convergence_threshold: Override default threshold (optional).

        Returns:
            Tuple of (h_L_final, stats):
                - h_L_final: Converged low-level state (batch, hidden_dim).
                - stats: IterationStats with convergence info.
        """
        # Use provided values or fall back to defaults
        actual_n_steps = n_steps if n_steps is not None else self.n_steps
        actual_threshold = (
            convergence_threshold
            if convergence_threshold is not None
            else self.convergence_threshold
        )

        return fixed_point_iteration(
            worker=self.worker,
            h_L_init=h_L_init,
            h_H=h_H,
            x_in=x_in,
            n_steps=actual_n_steps,
            convergence_threshold=actual_threshold,
            norm_type=self.norm_type,
            track_history=self.track_history,
        )

    def extra_repr(self) -> str:
        """Return string representation of module configuration."""
        return (
            f"n_steps={self.n_steps}, "
            f"convergence_threshold={self.convergence_threshold}, "
            f"norm_type='{self.norm_type}'"
        )


def iterate_to_convergence(
    worker: nn.Module,
    h_L_init: torch.Tensor,
    h_H: torch.Tensor,
    x_in: Optional[torch.Tensor] = None,
    max_steps: int = 100,
    threshold: float = 1e-5,
    norm_type: str = "l2",
) -> Tuple[torch.Tensor, int, bool]:
    """
    Simplified iteration function that runs until convergence.

    A convenience wrapper that always uses early stopping and returns
    a simpler output format without full statistics.

    Args:
        worker: WorkerModule for h_L updates.
        h_L_init: Initial low-level state (batch, hidden_dim).
        h_H: High-level state (batch, hidden_dim). Will be detached.
        x_in: Optional input embedding (batch, hidden_dim).
        max_steps: Maximum iterations before giving up. Default: 100.
        threshold: Convergence threshold. Default: 1e-5.
        norm_type: Norm for residual. Default: "l2".

    Returns:
        Tuple of (h_L_final, steps_taken, converged):
            - h_L_final: Final h_L state
            - steps_taken: Number of iterations run
            - converged: Whether threshold was reached

    Example:
        >>> h_L_final, steps, converged = iterate_to_convergence(
        ...     worker, h_L, h_H, x_in,
        ...     max_steps=50, threshold=1e-4
        ... )
        >>> print(f"Converged in {steps} steps: {converged}")
    """
    h_L_final, stats = fixed_point_iteration(
        worker=worker,
        h_L_init=h_L_init,
        h_H=h_H,
        x_in=x_in,
        n_steps=max_steps,
        convergence_threshold=threshold,
        norm_type=norm_type,
        track_history=False,  # Skip history for efficiency
    )

    return h_L_final, stats.num_steps, stats.converged
