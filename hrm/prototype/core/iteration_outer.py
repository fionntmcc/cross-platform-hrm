"""
Hierarchical Outer Loop for HRM (Planner Cycles)

This module implements the outer loop of the HRM where the Planner module
updates after each inner loop (Worker) convergence, creating the hierarchical
convergence mechanism.

Key Properties:
    1. Outer loop runs K cycles
    2. Each cycle: Inner loop runs T steps for Worker refinement
    3. Planner updates AFTER Worker converges
    4. Total iterations = K × T
    5. One-step gradient approximation maintained

Hierarchical Structure:
    Outer Loop (K cycles):
        → Fix h_H for this cycle
        → Inner Loop (T steps):
            → Worker refines toward local equilibrium
        → Planner updates with converged h_L
        → New context for next cycle

The hierarchical reset mechanism:
    - Planner provides h_H which defines Worker's equilibrium target
    - Worker converges to h_L_final under fixed h_H
    - Planner updates h_H using converged h_L_final
    - Next cycle: Worker converges to NEW equilibrium under new h_H
    - This prevents premature convergence (200+ steps vs 20-30 for RNNs)

Reference:
    - HRM Paper: Hierarchical convergence mechanism
    - Inner loop: hrm.core.iteration (fixed_point_iteration)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from hrm.prototype.core.iteration import (
    fixed_point_iteration,
    IterationStats,
    compute_residual,
)


@dataclass
class OuterLoopStats:
    """
    Statistics from hierarchical outer loop.
    
    Tracks convergence information across all cycles, useful for
    monitoring training progress and debugging the hierarchical mechanism.
    
    Attributes:
        num_cycles: Number of outer cycles completed.
        total_inner_steps: Total Worker iterations across all cycles (K × T).
        converged: Whether outer loop convergence threshold was reached.
        final_h_H_residual: Final residual ||h_H^(k) - h_H^(k-1)||.
        cycle_stats: List of IterationStats from each inner loop.
        h_H_residual_history: List of h_H residuals at each cycle.
    """
    
    num_cycles: int
    total_inner_steps: int
    converged: bool
    final_h_H_residual: float
    cycle_stats: List[IterationStats] = field(default_factory=list)
    h_H_residual_history: List[float] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"OuterLoopStats(cycles={self.num_cycles}, "
            f"total_inner_steps={self.total_inner_steps}, "
            f"converged={self.converged}, "
            f"final_h_H_residual={self.final_h_H_residual:.6f})"
        )
    
    @property
    def average_inner_steps_per_cycle(self) -> float:
        """
        Compute average inner steps per cycle.
        
        Returns average Worker iterations per Planner update.
        """
        if self.num_cycles == 0:
            return 0.0
        return self.total_inner_steps / self.num_cycles
    
    @property
    def convergence_rate(self) -> Optional[float]:
        """
        Compute average convergence rate from h_H residual history.
        
        Returns ratio of final to initial h_H residual, or None if history
        has fewer than 2 entries.
        """
        if len(self.h_H_residual_history) < 2:
            return None
        if self.h_H_residual_history[0] == 0:
            return 0.0
        return self.h_H_residual_history[-1] / self.h_H_residual_history[0]
    
    @property
    def inner_convergence_rate(self) -> Optional[float]:
        """
        Compute average inner loop convergence rate across cycles.
        
        Returns average of each cycle's convergence rate.
        """
        rates = [
            s.convergence_rate for s in self.cycle_stats 
            if s.convergence_rate is not None
        ]
        if not rates:
            return None
        return sum(rates) / len(rates)


def hierarchical_iteration(
    worker: nn.Module,
    planner: nn.Module,
    h_L_init: torch.Tensor,
    h_H_init: torch.Tensor,
    x_in: Optional[torch.Tensor] = None,
    n_outer_cycles: int = 5,
    n_inner_steps: int = 10,
    inner_convergence_threshold: Optional[float] = None,
    outer_convergence_threshold: Optional[float] = None,
    norm_type: str = "l2",
    track_history: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, OuterLoopStats]:
    """
    Run hierarchical iteration with Planner and Worker modules.
    
    This implements the outer loop of HRM where:
    1. Worker iterates to convergence (inner loop) with fixed h_H
    2. Planner updates h_H using the converged h_L
    3. Repeat for K outer cycles
    
    Algorithm:
        for cycle in range(n_outer_cycles):
            # Inner loop: Worker refines h_L with h_H FIXED
            h_L, inner_stats = fixed_point_iteration(
                worker, h_L, h_H, x_in, n_inner_steps
            )
            
            # Planner update: new h_H from converged h_L
            h_H_prev = h_H
            h_H = planner(h_H_prev, h_L)
            
            # Check outer convergence
            if ||h_H - h_H_prev|| < outer_threshold:
                break
    
    Args:
        worker: WorkerModule instance for inner loop refinement.
        planner: PlannerModule instance for outer loop updates.
        h_L_init: Initial low-level hidden state of shape (batch, hidden_dim).
        h_H_init: Initial high-level hidden state of shape (batch, hidden_dim).
        x_in: Optional input embedding of shape (batch, hidden_dim).
        n_outer_cycles: Maximum number of outer cycles (K). Default: 5.
        n_inner_steps: Maximum inner loop steps per cycle (T). Default: 10.
        inner_convergence_threshold: Optional threshold for inner loop early
            stopping. If None, inner loop always runs n_inner_steps.
        outer_convergence_threshold: Optional threshold for outer loop early
            stopping based on h_H change. If None, always runs n_outer_cycles.
        norm_type: Norm type for residual computation. Default: "l2".
        track_history: Whether to track residual history. Default: True.
    
    Returns:
        Tuple of (h_L_final, h_H_final, stats):
            - h_L_final: Final low-level hidden state (batch, hidden_dim).
            - h_H_final: Final high-level hidden state (batch, hidden_dim).
            - stats: OuterLoopStats with hierarchical convergence info.

    Important Properties:
        - Total iterations = K × T (outer cycles × inner steps)
        - h_H is DETACHED in inner loop (one-step gradient approximation)
        - Gradients flow through final iteration of each cycle
        - Memory complexity is O(K) instead of O(K × T)
    
    Example:
        >>> worker = WorkerModule(hidden_dim=64)
        >>> planner = PlannerModule(hidden_dim=64)
        >>> h_L = torch.randn(8, 64)
        >>> h_H = torch.randn(8, 64)
        >>> x_in = torch.randn(8, 64)
        >>> h_L_final, h_H_final, stats = hierarchical_iteration(
        ...     worker, planner, h_L, h_H, x_in,
        ...     n_outer_cycles=5,
        ...     n_inner_steps=10,
        ...     outer_convergence_threshold=1e-4
        ... )
        >>> print(stats)
        OuterLoopStats(cycles=4, total_inner_steps=40, converged=True, ...)
    """
    # Validate inputs
    if n_outer_cycles <= 0:
        raise ValueError(f"n_outer_cycles must be positive, got {n_outer_cycles}")
    if n_inner_steps <= 0:
        raise ValueError(f"n_inner_steps must be positive, got {n_inner_steps}")
    if inner_convergence_threshold is not None and inner_convergence_threshold <= 0:
        raise ValueError(
            f"inner_convergence_threshold must be positive, "
            f"got {inner_convergence_threshold}"
        )
    if outer_convergence_threshold is not None and outer_convergence_threshold <= 0:
        raise ValueError(
            f"outer_convergence_threshold must be positive, "
            f"got {outer_convergence_threshold}"
        )
    
    # Initialise iteration state
    h_L = h_L_init
    h_H = h_H_init
    cycle_stats: List[IterationStats] = []
    h_H_residual_history: List[float] = []
    total_inner_steps = 0
    converged = False
    final_h_H_residual = 0.0
    cycles_completed = 0
    
    # ==========================================================================
    # Hierarchical outer loop (K cycles)
    # ==========================================================================
    for cycle in range(n_outer_cycles):
        # Store previous h_H for convergence check
        h_H_prev = h_H
        
        # ======================================================================
        # Inner loop: Worker refines h_L with FIXED h_H
        # ======================================================================
        # Note: fixed_point_iteration handles h_H detachment internally
        h_L, inner_stats = fixed_point_iteration(
            worker=worker,
            h_L_init=h_L,
            h_H=h_H,
            x_in=x_in,
            n_steps=n_inner_steps,
            convergence_threshold=inner_convergence_threshold,
            norm_type=norm_type,
            track_history=track_history,
        )
        
        # Track inner loop statistics
        total_inner_steps += inner_stats.num_steps
        if track_history:
            cycle_stats.append(inner_stats)
        
        # ======================================================================
        # Planner update: new h_H from converged h_L
        # ======================================================================
        # Planner receives: previous h_H state and converged h_L
        h_H = planner(h_H_prev, h_L)
        
        # Compute h_H residual: ||h_H^(k) - h_H^(k-1)||
        h_H_residual = compute_residual(h_H, h_H_prev, norm_type=norm_type)
        mean_h_H_residual = h_H_residual.mean().item()
        final_h_H_residual = mean_h_H_residual
        cycles_completed = cycle + 1
        
        # Track h_H residual history
        if track_history:
            h_H_residual_history.append(mean_h_H_residual)
        
        # ======================================================================
        # Outer convergence check
        # ======================================================================
        if (outer_convergence_threshold is not None and 
                mean_h_H_residual < outer_convergence_threshold):
            converged = True
            break
    
    # Build statistics
    stats = OuterLoopStats(
        num_cycles=cycles_completed,
        total_inner_steps=total_inner_steps,
        converged=converged,
        final_h_H_residual=final_h_H_residual,
        cycle_stats=cycle_stats if track_history else [],
        h_H_residual_history=h_H_residual_history if track_history else [],
    )
    
    return h_L, h_H, stats


class HierarchicalIterator(nn.Module):
    """
    Module wrapper for hierarchical iteration.
    
    Provides a nn.Module interface for the hierarchical outer loop,
    making it easier to integrate into larger models and handle
    training/eval mode switching.
    
    The iterator manages:
    - Outer loop (Planner updates)
    - Inner loop (Worker refinement via FixedPointIterator)
    - Convergence tracking at both levels
    - Statistics collection
    
    Args:
        worker: WorkerModule for low-level refinement.
        planner: PlannerModule for high-level planning.
        n_outer_cycles: Maximum outer cycles (K). Default: 5.
        n_inner_steps: Maximum inner steps per cycle (T). Default: 10.
        inner_convergence_threshold: Inner loop early stopping threshold.
        outer_convergence_threshold: Outer loop early stopping threshold.
        norm_type: Norm type for residual computation. Default: "l2".
        track_history: Whether to track residual history. Default: True.
    
    Example:
        >>> worker = WorkerModule(hidden_dim=64)
        >>> planner = PlannerModule(hidden_dim=64)
        >>> iterator = HierarchicalIterator(
        ...     worker, planner,
        ...     n_outer_cycles=5,
        ...     n_inner_steps=10
        ... )
        >>> h_L_final, h_H_final, stats = iterator(h_L_init, h_H_init, x_in)
    """
    
    def __init__(
        self,
        worker: nn.Module,
        planner: nn.Module,
        n_outer_cycles: int = 5,
        n_inner_steps: int = 10,
        inner_convergence_threshold: Optional[float] = None,
        outer_convergence_threshold: Optional[float] = None,
        norm_type: str = "l2",
        track_history: bool = True,
    ):
        """
        Initialise HierarchicalIterator.
        
        Args:
            worker: WorkerModule for low-level refinement.
            planner: PlannerModule for high-level planning.
            n_outer_cycles: Maximum outer cycles.
            n_inner_steps: Maximum inner steps per cycle.
            inner_convergence_threshold: Inner early stopping threshold.
            outer_convergence_threshold: Outer early stopping threshold.
            norm_type: Norm for residual computation.
            track_history: Whether to record residual history.
        """
        super().__init__()
        
        if n_outer_cycles <= 0:
            raise ValueError(f"n_outer_cycles must be positive, got {n_outer_cycles}")
        if n_inner_steps <= 0:
            raise ValueError(f"n_inner_steps must be positive, got {n_inner_steps}")
        if inner_convergence_threshold is not None and inner_convergence_threshold <= 0:
            raise ValueError(
                f"inner_convergence_threshold must be positive, "
                f"got {inner_convergence_threshold}"
            )
        if outer_convergence_threshold is not None and outer_convergence_threshold <= 0:
            raise ValueError(
                f"outer_convergence_threshold must be positive, "
                f"got {outer_convergence_threshold}"
            )
        
        self.worker = worker
        self.planner = planner
        self.n_outer_cycles = n_outer_cycles
        self.n_inner_steps = n_inner_steps
        self.inner_convergence_threshold = inner_convergence_threshold
        self.outer_convergence_threshold = outer_convergence_threshold
        self.norm_type = norm_type
        self.track_history = track_history
    
    def forward(
        self,
        h_L_init: torch.Tensor,
        h_H_init: torch.Tensor,
        x_in: Optional[torch.Tensor] = None,
        n_outer_cycles: Optional[int] = None,
        n_inner_steps: Optional[int] = None,
        inner_convergence_threshold: Optional[float] = None,
        outer_convergence_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, OuterLoopStats]:
        """
        Run hierarchical iteration.
        
        Args:
            h_L_init: Initial low-level hidden state (batch, hidden_dim).
            h_H_init: Initial high-level hidden state (batch, hidden_dim).
            x_in: Optional input embedding (batch, hidden_dim).
            n_outer_cycles: Override default outer cycles (optional).
            n_inner_steps: Override default inner steps (optional).
            inner_convergence_threshold: Override inner threshold (optional).
            outer_convergence_threshold: Override outer threshold (optional).
        
        Returns:
            Tuple of (h_L_final, h_H_final, stats):
                - h_L_final: Converged low-level state (batch, hidden_dim).
                - h_H_final: Final high-level state (batch, hidden_dim).
                - stats: OuterLoopStats with convergence info.
        """
        # Use provided values or fall back to defaults
        actual_outer_cycles = (
            n_outer_cycles if n_outer_cycles is not None 
            else self.n_outer_cycles
        )
        actual_inner_steps = (
            n_inner_steps if n_inner_steps is not None 
            else self.n_inner_steps
        )
        actual_inner_threshold = (
            inner_convergence_threshold 
            if inner_convergence_threshold is not None 
            else self.inner_convergence_threshold
        )
        actual_outer_threshold = (
            outer_convergence_threshold 
            if outer_convergence_threshold is not None 
            else self.outer_convergence_threshold
        )
        
        return hierarchical_iteration(
            worker=self.worker,
            planner=self.planner,
            h_L_init=h_L_init,
            h_H_init=h_H_init,
            x_in=x_in,
            n_outer_cycles=actual_outer_cycles,
            n_inner_steps=actual_inner_steps,
            inner_convergence_threshold=actual_inner_threshold,
            outer_convergence_threshold=actual_outer_threshold,
            norm_type=self.norm_type,
            track_history=self.track_history,
        )
    
    def extra_repr(self) -> str:
        """Return string representation of module configuration."""
        return (
            f"n_outer_cycles={self.n_outer_cycles}, "
            f"n_inner_steps={self.n_inner_steps}, "
            f"inner_threshold={self.inner_convergence_threshold}, "
            f"outer_threshold={self.outer_convergence_threshold}, "
            f"norm_type='{self.norm_type}'"
        )


def hierarchical_iterate_to_convergence(
    worker: nn.Module,
    planner: nn.Module,
    h_L_init: torch.Tensor,
    h_H_init: torch.Tensor,
    x_in: Optional[torch.Tensor] = None,
    max_outer_cycles: int = 20,
    max_inner_steps: int = 50,
    inner_threshold: float = 1e-5,
    outer_threshold: float = 1e-4,
    norm_type: str = "l2",
) -> Tuple[torch.Tensor, torch.Tensor, int, int, bool]:
    """
    Simplified hierarchical iteration that runs until convergence.
    
    A convenience wrapper that always uses early stopping and returns
    a simpler output format without full statistics.
    
    Args:
        worker: WorkerModule for h_L updates.
        planner: PlannerModule for h_H updates.
        h_L_init: Initial low-level state (batch, hidden_dim).
        h_H_init: Initial high-level state (batch, hidden_dim).
        x_in: Optional input embedding (batch, hidden_dim).
        max_outer_cycles: Maximum outer cycles. Default: 20.
        max_inner_steps: Maximum inner steps per cycle. Default: 50.
        inner_threshold: Inner loop convergence threshold. Default: 1e-5.
        outer_threshold: Outer loop convergence threshold. Default: 1e-4.
        norm_type: Norm for residual. Default: "l2".
    
    Returns:
        Tuple of (h_L_final, h_H_final, cycles, total_steps, converged):
            - h_L_final: Final h_L state
            - h_H_final: Final h_H state
            - cycles: Number of outer cycles completed
            - total_steps: Total inner steps across all cycles
            - converged: Whether outer threshold was reached
    
    Example:
        >>> h_L, h_H, cycles, steps, converged = hierarchical_iterate_to_convergence(
        ...     worker, planner, h_L, h_H, x_in,
        ...     max_outer_cycles=10, outer_threshold=1e-4
        ... )
        >>> print(f"Converged in {cycles} cycles ({steps} total steps): {converged}")
    """
    h_L_final, h_H_final, stats = hierarchical_iteration(
        worker=worker,
        planner=planner,
        h_L_init=h_L_init,
        h_H_init=h_H_init,
        x_in=x_in,
        n_outer_cycles=max_outer_cycles,
        n_inner_steps=max_inner_steps,
        inner_convergence_threshold=inner_threshold,
        outer_convergence_threshold=outer_threshold,
        norm_type=norm_type,
        track_history=False,  # Skip history for efficiency
    )
    
    return (
        h_L_final,
        h_H_final,
        stats.num_cycles,
        stats.total_inner_steps,
        stats.converged,
    )


def single_hierarchical_step(
    worker: nn.Module,
    planner: nn.Module,
    h_L: torch.Tensor,
    h_H: torch.Tensor,
    x_in: Optional[torch.Tensor] = None,
    n_inner_steps: int = 10,
    inner_convergence_threshold: Optional[float] = None,
    norm_type: str = "l2",
) -> Tuple[torch.Tensor, torch.Tensor, IterationStats]:
    """
    Execute a single hierarchical step (one outer cycle).
    
    Useful for manual control over the outer loop, allowing custom
    logic between Planner updates.
    
    Algorithm:
        1. Run inner loop (Worker refinement with fixed h_H)
        2. Update Planner (h_H_new from h_H and converged h_L)
        3. Return updated states and inner loop stats
    
    Args:
        worker: WorkerModule for h_L refinement.
        planner: PlannerModule for h_H update.
        h_L: Current low-level state (batch, hidden_dim).
        h_H: Current high-level state (batch, hidden_dim).
        x_in: Optional input embedding (batch, hidden_dim).
        n_inner_steps: Maximum inner loop steps. Default: 10.
        inner_convergence_threshold: Inner loop early stopping threshold.
        norm_type: Norm for residual computation. Default: "l2".
    
    Returns:
        Tuple of (h_L_new, h_H_new, inner_stats):
            - h_L_new: Converged low-level state after inner loop.
            - h_H_new: Updated high-level state after Planner update.
            - inner_stats: IterationStats from the inner loop.
    
    Example:
        >>> # Manual outer loop with custom logic
        >>> for cycle in range(10):
        ...     h_L, h_H, stats = single_hierarchical_step(
        ...         worker, planner, h_L, h_H, x_in
        ...     )
        ...     print(f"Cycle {cycle}: {stats.num_steps} inner steps")
        ...     if custom_convergence_check(h_L, h_H):
        ...         break
    """
    # Inner loop: Worker refines h_L with fixed h_H
    h_L_converged, inner_stats = fixed_point_iteration(
        worker=worker,
        h_L_init=h_L,
        h_H=h_H,
        x_in=x_in,
        n_steps=n_inner_steps,
        convergence_threshold=inner_convergence_threshold,
        norm_type=norm_type,
        track_history=True,
    )
    
    # Planner update: new h_H from converged h_L
    h_H_new = planner(h_H, h_L_converged)
    
    return h_L_converged, h_H_new, inner_stats
