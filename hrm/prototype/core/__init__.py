# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
HRM Core Logic — Prototype Archive

These modules implement the core iteration and halting logic for the full
Unified HRM (H+L with ACT). In earlier revisions these lived under hrm/core/;
they are now kept only in this prototype archive and are no longer in the
active SimplifiedHRM dependency chain.

Active codebase (SimplifiedHRM) uses fixed reasoning steps with no ACT halting,
so none of these are needed for training/inference with train_simplified.py.
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
