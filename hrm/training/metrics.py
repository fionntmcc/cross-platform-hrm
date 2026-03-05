"""
Metrics tracking for Simplified HRM training.

Provides reusable accuracy / convergence functions and a per-epoch
aggregation helper (``MetricsTracker``), replacing the inline metric
computation previously scattered in ``train_simplified.py``.

Functions:
    compute_accuracy        — Token-level accuracy (with optional mask).
    compute_puzzle_accuracy — Full-puzzle solve rate.
    compute_residuals       — L2 residuals between consecutive reasoning steps.
    compute_solve_rate      — Evaluate a model's solve rate over a dataset.

Classes:
    MetricsTracker — Accumulates per-batch stats, computes epoch summaries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================================================================
# Accuracy helpers
# =========================================================================


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Token-level accuracy, optionally restricted to masked positions.

    Args:
        predictions: ``(batch, seq_len)`` predicted token ids.
        targets: ``(batch, seq_len)`` ground-truth token ids.
        mask: Optional ``(batch, seq_len)`` boolean mask.  When provided,
              only positions where ``mask == True`` are counted.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    with torch.no_grad():
        if mask is not None and mask.any():
            correct = (predictions[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
        else:
            correct = (predictions == targets).sum().item()
            total = targets.numel()
    return correct / total if total > 0 else 0.0


def compute_puzzle_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Full-puzzle solve rate (every cell correct).

    Args:
        predictions: ``(batch, seq_len)`` predicted token ids.
        targets: ``(batch, seq_len)`` ground-truth token ids.
        mask: Optional ``(batch, seq_len)`` boolean mask.  If provided,
              a puzzle counts as correct when *all masked cells* match
              AND all given (unmasked) cells are preserved.

    Returns:
        Fraction of puzzles fully solved in ``[0, 1]``.
    """
    with torch.no_grad():
        if mask is not None and mask.any():
            per_puzzle = ((predictions == targets) | ~mask).all(dim=1)
        else:
            per_puzzle = (predictions == targets).all(dim=1)
        return per_puzzle.float().mean().item()


# =========================================================================
# Residual / convergence helpers
# =========================================================================


def compute_residuals(
    all_step_logits: list[torch.Tensor],
) -> list[float]:
    """Compute L2 residual norms between consecutive reasoning steps.

    This measures how much the model's output distribution changes from
    one reasoning step to the next — a proxy for convergence of the
    iterative reasoning process.

    Args:
        all_step_logits: List of ``(batch, seq, vocab)`` tensors, one
            per reasoning step (as returned by ``model.forward()``).

    Returns:
        List of mean-per-sample L2 norms of length ``len(all_step_logits) - 1``.
        Empty list if fewer than 2 steps.
    """
    if len(all_step_logits) < 2:
        return []

    residuals: list[float] = []
    with torch.no_grad():
        for i in range(1, len(all_step_logits)):
            diff = all_step_logits[i].float() - all_step_logits[i - 1].float()
            # L2 norm per sample, averaged over batch
            per_sample = diff.view(diff.size(0), -1).norm(dim=1)
            residuals.append(per_sample.mean().item())
    return residuals


# =========================================================================
# MetricsTracker — per-epoch accumulator
# =========================================================================


@dataclass
class _BatchStats:
    """Accumulated statistics from a single batch."""

    loss: float = 0.0
    lm_loss: float = 0.0
    correct_tokens: int = 0
    total_tokens: int = 0
    puzzles_correct: int = 0
    puzzles_total: int = 0
    num_samples: int = 0
    final_residual: float = 0.0
    residual_count: int = 0


class MetricsTracker:
    """Accumulates per-batch metrics and computes epoch-level summaries.

    Typical usage inside a training / evaluation loop::

        tracker = MetricsTracker()
        for batch in dataloader:
            output = model(batch)
            tracker.update(
                loss=loss.item(),
                predictions=output['predictions'],
                targets=batch['target'],
                mask=batch.get('empty_mask'),
                batch_size=batch['input'].size(0),
                all_step_logits=output.get('all_step_logits'),
            )
        summary = tracker.summarise()
    """

    def __init__(self) -> None:
        self._acc = _BatchStats()
        self._batch_losses: list[float] = []
        self._batch_residuals: list[list[float]] = []
        self._start_time: float = time.monotonic()

    # -----------------------------------------------------------------
    # Per-batch update
    # -----------------------------------------------------------------

    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        mask: torch.Tensor | None = None,
        lm_loss: float | None = None,
        all_step_logits: list[torch.Tensor] | None = None,
    ) -> None:
        """Record metrics for one mini-batch.

        Args:
            loss: Scalar loss for the batch (already un-scaled).
            predictions: ``(B, seq)`` predicted tokens.
            targets: ``(B, seq)`` target tokens.
            batch_size: Number of samples in this batch.
            mask: Optional empty-cell mask ``(B, seq)``.
            lm_loss: Separate LM loss (defaults to *loss*).
            all_step_logits: Per-step logits for residual tracking.
        """
        a = self._acc
        a.loss += loss * batch_size
        a.lm_loss += (lm_loss if lm_loss is not None else loss) * batch_size
        a.num_samples += batch_size

        # Token accuracy
        with torch.no_grad():
            if mask is not None and mask.any():
                a.correct_tokens += (predictions[mask] == targets[mask]).sum().item()
                a.total_tokens += mask.sum().item()
                per_puzzle = ((predictions == targets) | ~mask).all(dim=1)
            else:
                a.correct_tokens += (predictions == targets).sum().item()
                a.total_tokens += targets.numel()
                per_puzzle = (predictions == targets).all(dim=1)
            a.puzzles_correct += per_puzzle.sum().item()
            a.puzzles_total += batch_size

        # Per-batch loss (for convergence / loss-curve granularity)
        self._batch_losses.append(loss)

        # Residual tracking
        if all_step_logits is not None:
            residuals = compute_residuals(all_step_logits)
            if residuals:
                self._batch_residuals.append(residuals)
                a.final_residual += residuals[-1] * batch_size
                a.residual_count += batch_size

    # -----------------------------------------------------------------
    # Epoch summary
    # -----------------------------------------------------------------

    def summarise(
        self,
        learning_rate: float | None = None,
        reasoning_steps: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """Compute epoch-level summary metrics.

        Args:
            learning_rate: Current LR (from scheduler).
            reasoning_steps: Number of reasoning steps used.
            epoch: Current epoch number (1-based).

        Returns:
            Dictionary with all tracked metrics.  Keys include at least:
            ``loss``, ``lm_loss``, ``token_accuracy``, ``puzzle_accuracy``,
            ``avg_residual``, ``epoch_time_s``.
        """
        a = self._acc
        n = max(a.num_samples, 1)
        t = max(a.total_tokens, 1)
        elapsed = time.monotonic() - self._start_time

        summary: dict[str, Any] = {
            "loss": a.loss / n,
            "lm_loss": a.lm_loss / n,
            "token_accuracy": a.correct_tokens / t,
            "puzzle_accuracy": a.puzzles_correct / max(a.puzzles_total, 1),
            "avg_residual": (a.final_residual / a.residual_count if a.residual_count > 0 else None),
            "num_samples": a.num_samples,
            "epoch_time_s": round(elapsed, 2),
        }

        if learning_rate is not None:
            summary["learning_rate"] = learning_rate
        if reasoning_steps is not None:
            summary["reasoning_steps"] = reasoning_steps
        if epoch is not None:
            summary["epoch"] = epoch

        # Per-batch loss series (useful for loss-curve plots)
        summary["batch_losses"] = list(self._batch_losses)

        return summary

    # Alias for British/American spelling
    summarize = summarise


# =========================================================================
# Standalone evaluation helper
# =========================================================================


@torch.no_grad()
def compute_solve_rate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model's solve rate over an entire dataset.

    This is a convenience wrapper that runs :class:`MetricsTracker` over
    a dataloader in eval mode, returning the summary.

    Args:
        model: A ``SimplifiedHRM`` (or compatible) model.
        dataloader: DataLoader yielding dicts with keys
                    ``input``, ``target``, ``puzzle_type``, and optionally
                    ``empty_mask``.
        device: Device to run inference on.

    Returns:
        Dictionary with ``token_accuracy``, ``puzzle_accuracy``, ``loss``,
        ``avg_residual``.
    """
    model.eval()
    tracker = MetricsTracker()

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        puzzle_type = batch["puzzle_type"]
        empty_mask = batch.get("empty_mask")
        if empty_mask is not None:
            empty_mask = empty_mask.to(device)

        output = model(inputs, puzzle_type, targets=targets)
        predictions = output["predictions"]

        loss = F.cross_entropy(
            (
                output["logits"][empty_mask].float()
                if (empty_mask is not None and empty_mask.any())
                else output["logits"].view(-1, output["logits"].size(-1)).float()
            ),
            (
                targets[empty_mask]
                if (empty_mask is not None and empty_mask.any())
                else targets.view(-1)
            ),
        ).item()

        tracker.update(
            loss=loss,
            predictions=predictions,
            targets=targets,
            batch_size=inputs.size(0),
            mask=empty_mask,
            all_step_logits=output.get("all_step_logits"),
        )

    return tracker.summarise()
