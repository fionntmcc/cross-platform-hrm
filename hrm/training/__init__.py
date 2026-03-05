"""
Training utilities: metrics tracking and structured logging.

Modules:
    metrics — Accuracy, solve rate, residual convergence, batch tracking.
    logger  — Structured JSON logging, CSV export, TensorBoard, training curves.
"""

from hrm.training.metrics import (
    compute_accuracy,
    compute_puzzle_accuracy,
    compute_residuals,
    MetricsTracker,
)
from hrm.training.logger import TrainingLogger

__all__ = [
    "compute_accuracy",
    "compute_puzzle_accuracy",
    "compute_residuals",
    "MetricsTracker",
    "TrainingLogger",
]
