"""
HRM Core Logic

This module contains the core iteration and convergence logic:
    - Fixed-point iteration (inner loop) - Issue #21
    - Hierarchical convergence (outer loop) - Issue #7
    - Adaptive Computation Time (halting) - ACT Extension
"""

from hrm.core.iteration import (
    # Main iteration function
    fixed_point_iteration,
    # Module wrapper
    FixedPointIterator,
    # Convenience function
    iterate_to_convergence,
    # Helper functions
    compute_residual,
    # Data classes
    IterationStats,
)

from hrm.core.iteration_outer import (
    # Main hierarchical iteration function
    hierarchical_iteration,
    # Module wrapper
    HierarchicalIterator,
    # Convenience functions
    hierarchical_iterate_to_convergence,
    single_hierarchical_step,
    # Data classes
    OuterLoopStats,
)

from hrm.core.halting import (
    # Q-learning halting head
    QHaltingHead,
    QHaltingHeadTransformer,  # Sapient-compatible
    # Halting policy
    HaltingPolicy,
    # Statistics
    ACTStats,
    # Q-trainer
    HaltingQTrainer,
    # Helper functions
    compute_ponder_cost,
    compute_act_loss,
    create_halting_components,
)

__all__ = [
    # Issue #21 - Fixed-point iteration (inner loop)
    "fixed_point_iteration",
    "FixedPointIterator",
    "iterate_to_convergence",
    "compute_residual",
    "IterationStats",
    # Issue #7 - Hierarchical convergence (outer loop)
    "hierarchical_iteration",
    "HierarchicalIterator",
    "hierarchical_iterate_to_convergence",
    "single_hierarchical_step",
    "OuterLoopStats",
    # ACT Extension - Adaptive Computation Time
    "QHaltingHead",
    "QHaltingHeadTransformer",
    "HaltingPolicy",
    "ACTStats",
    "HaltingQTrainer",
    "compute_ponder_cost",
    "compute_act_loss",
    "create_halting_components",
]