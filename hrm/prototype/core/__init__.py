"""
HRM Core Logic — Prototype Archive

These modules implement the core iteration and halting logic for the full
Unified HRM (H+L with ACT). They have been moved here from hrm/core/ as they
are no longer in the active SimplifiedHRM dependency chain.

Active codebase (SimplifiedHRM) uses fixed reasoning steps with no ACT halting,
so none of these are needed for training/inference with train_simplified.py.

hrm.core still re-exports everything here for backward compatibility with
the existing test suite.
"""

from hrm.prototype.core.iteration import (
    fixed_point_iteration,
    FixedPointIterator,
    iterate_to_convergence,
    compute_residual,
    IterationStats,
)

from hrm.prototype.core.iteration_outer import (
    hierarchical_iteration,
    HierarchicalIterator,
    hierarchical_iterate_to_convergence,
    single_hierarchical_step,
    OuterLoopStats,
)

from hrm.prototype.core.halting import (
    QHaltingHead,
    QHaltingHeadTransformer,
    HaltingPolicy,
    ACTStats,
    HaltingQTrainer,
    compute_ponder_cost,
    compute_act_loss,
    create_halting_components,
)

__all__ = [
    "fixed_point_iteration",
    "FixedPointIterator",
    "iterate_to_convergence",
    "compute_residual",
    "IterationStats",
    "hierarchical_iteration",
    "HierarchicalIterator",
    "hierarchical_iterate_to_convergence",
    "single_hierarchical_step",
    "OuterLoopStats",
    "QHaltingHead",
    "QHaltingHeadTransformer",
    "HaltingPolicy",
    "ACTStats",
    "HaltingQTrainer",
    "compute_ponder_cost",
    "compute_act_loss",
    "create_halting_components",
]
