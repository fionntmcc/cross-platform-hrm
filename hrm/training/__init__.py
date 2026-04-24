# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Training utilities: metrics tracking and structured logging.

Modules:
    metrics       — Accuracy, solve rate, residual convergence, batch tracking.
    logger        — Structured JSON logging, CSV export, TensorBoard, training curves.
    seed_analysis — Multi-seed aggregation, comparison plots, variance reporting.
"""

from hrm.training.logger import TrainingLogger
from hrm.training.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_puzzle_accuracy,
    compute_residuals,
)
from hrm.training.seed_analysis import (
    aggregate_seeds,
    plot_seed_comparison,
)
from hrm.training.seed_analysis import (
    print_summary as print_seed_summary,
)
from hrm.training.seed_analysis import (
    save_summary as save_seed_summary,
)

__all__ = [
    "compute_accuracy",
    "compute_puzzle_accuracy",
    "compute_residuals",
    "MetricsTracker",
    "TrainingLogger",
    "aggregate_seeds",
    "plot_seed_comparison",
    "print_seed_summary",
    "save_seed_summary",
]
