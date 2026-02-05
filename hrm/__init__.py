"""
Cross-Platform Hierarchical Reasoning Model (HRM)

A platform-agnostic implementation of the Hierarchical Reasoning Model
for constraint-satisfaction problems, specifically Sudoku puzzle solving.

Authors:
    - Fionn McCarthy (G00414386)
    - Kyrylo Kozlovskyi (G00425385)

Supervisor: Dr. John Healy
Atlantic Technological University
"""

__version__ = "0.1.0"
__author__ = "Fionn McCarthy, Kyrylo Kozlovskyi"

# Issue #8: Complete HRM Model
from hrm.model import HRM_4x4, ExecutionTrace, create_hrm_4x4

# Simplified L-Module Only Variant (Ge et al. 2025)
from hrm.model_simple import HRM_4x4_Simple, SimpleExecutionTrace, create_hrm_4x4_simple

# Layers - Neural Network Components
from hrm.layers import (
    # Normalisation (Issue #1)
    RMSNorm,
    RMSNormWithBias,
    create_norm_layer,
    # Input Embedding (Issue #2)
    InputNetwork,
    create_input_network,
    # Worker Module (Issue #3)
    WorkerModule,
    WorkerModuleWithGating,
    # Planner Module (Issue #4)
    PlannerModule,
    create_planner_module,
    # Output Network (Issue #5)
    OutputNetwork,
    create_output_network,
)

# Core - Iteration and Convergence Logic
from hrm.core import (
    # Fixed-point iteration (Issue #21)
    fixed_point_iteration,
    FixedPointIterator,
    iterate_to_convergence,
    compute_residual,
    IterationStats,
    # Hierarchical iteration (Issue #7)
    hierarchical_iteration,
    HierarchicalIterator,
    hierarchical_iterate_to_convergence,
    single_hierarchical_step,
    OuterLoopStats,
    # Adaptive Computation Time
    QHaltingHead,
    HaltingPolicy,
    ACTStats,
    HaltingQTrainer,
    compute_ponder_cost,
    compute_act_loss,
    create_halting_components,
)

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    # Complete Model (Issue #8)
    "HRM_4x4",
    "ExecutionTrace",
    "create_hrm_4x4",
    # Simplified L-Module Only Variant (Ge et al. 2025)
    "HRM_4x4_Simple",
    "SimpleExecutionTrace",
    "create_hrm_4x4_simple",
    # Normalisation (Issue #1)
    "RMSNorm",
    "RMSNormWithBias",
    "create_norm_layer",
    # Input Network (Issue #2)
    "InputNetwork",
    "create_input_network",
    # Worker Module (Issue #3)
    "WorkerModule",
    "WorkerModuleWithGating",
    # Planner Module (Issue #4)
    "PlannerModule",
    "create_planner_module",
    # Output Network (Issue #5)
    "OutputNetwork",
    "create_output_network",
    # Fixed-point iteration (Issue #21)
    "fixed_point_iteration",
    "FixedPointIterator",
    "iterate_to_convergence",
    "compute_residual",
    "IterationStats",
    # Hierarchical iteration (Issue #7)
    "hierarchical_iteration",
    "HierarchicalIterator",
    "hierarchical_iterate_to_convergence",
    "single_hierarchical_step",
    "OuterLoopStats",
    # Adaptive Computation Time
    "QHaltingHead",
    "HaltingPolicy",
    "ACTStats",
    "HaltingQTrainer",
    "compute_ponder_cost",
    "compute_act_loss",
    "create_halting_components",
]
