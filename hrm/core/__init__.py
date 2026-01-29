"""
HRM Core Logic

This module contains the core iteration and convergence logic:
    - Fixed-point iteration (inner loop) - Issue #21
    - Hierarchical convergence (outer loop) - Issue #7
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

__all__ = [
    # Issue #21 - Fixed-point iteration
    "fixed_point_iteration",
    "FixedPointIterator",
    "iterate_to_convergence",
    "compute_residual",
    "IterationStats",
    # Issue #7 - Will be added later
    # "hierarchical_convergence",
]
