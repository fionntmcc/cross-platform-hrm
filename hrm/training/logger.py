"""
Structured logging for Simplified HRM training.

Provides a single :class:`TrainingLogger` that:
- Writes per-epoch JSON records to a ``.jsonl`` file.
- Maintains a cumulative ``.csv`` for easy import into spreadsheets/pandas.
- Optionally writes to TensorBoard (``torch.utils.tensorboard``).
- Generates matplotlib training-curve plots.

All external dependencies (tensorboard, matplotlib) are **soft** — the
logger silently degrades if they are not installed.

Usage::

    logger = TrainingLogger(log_dir="model", run_name="sudoku_9x9")
    for epoch in range(1, epochs + 1):
        train_summary = train_tracker.summarise(epoch=epoch, ...)
        val_summary   = val_tracker.summarise(epoch=epoch, ...)
        logger.log_epoch(epoch, train_summary, val_summary)
    logger.plot_training_curves()
    logger.close()
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


# =========================================================================
# Soft imports
# =========================================================================

def _try_import_tensorboard():
    """Attempt to import TensorBoard SummaryWriter."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except ImportError:
        return None


def _try_import_matplotlib():
    """Attempt to import matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# =========================================================================
# TrainingLogger
# =========================================================================

class TrainingLogger:
    """Structured logger with JSON, CSV, TensorBoard, and plot support.

    Files created under ``log_dir/``:

    =====================  ==============================================
    File                   Contents
    =====================  ==============================================
    ``{run_name}.jsonl``   One JSON object per epoch.
    ``{run_name}.csv``     Flat CSV with all scalar metrics per epoch.
    ``tb/{run_name}/``     TensorBoard event files.
    ``{run_name}_curves.png``  Training curve plot.
    =====================  ==============================================

    Args:
        log_dir: Directory for all log artefacts.
        run_name: Identifier for this training run.
        use_tensorboard: Enable TensorBoard logging.  Set to ``True``
            to attempt import; ``False`` to disable.  Defaults to ``True``.
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str = "training",
        use_tensorboard: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # JSONL (append-only structured log)
        self._jsonl_path = self.log_dir / f"{run_name}.jsonl"
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

        # CSV
        self._csv_path = self.log_dir / f"{run_name}.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer: csv.DictWriter | None = None  # initialised on first write
        self._csv_fields: list[str] = []

        # TensorBoard (optional)
        self._tb_writer = None
        if use_tensorboard:
            SummaryWriter = _try_import_tensorboard()
            if SummaryWriter is not None:
                tb_dir = self.log_dir / "tb" / run_name
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # In-memory history (for plotting)
        self._history: list[dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Core logging
    # -----------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log one epoch's metrics to all configured sinks.

        Args:
            epoch: 1-based epoch number.
            train_metrics: Summary dict from ``MetricsTracker.summarise()``.
            val_metrics: Optional validation summary dict.

        Returns:
            The merged record that was written (useful for printing).
        """
        record = self._build_record(epoch, train_metrics, val_metrics)
        self._write_jsonl(record)
        self._write_csv(record)
        self._write_tensorboard(epoch, record)
        self._history.append(record)
        return record

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------

    def plot_training_curves(
        self,
        save_path: str | Path | None = None,
    ) -> Path | None:
        """Generate a matplotlib figure with loss and accuracy curves.

        Args:
            save_path: Override the default save location.

        Returns:
            Path to the saved image, or ``None`` if matplotlib is
            unavailable or no data has been logged.
        """
        plt = _try_import_matplotlib()
        if plt is None:
            print("Warning: matplotlib not installed — skipping training curve plot.")
            return None

        if not self._history:
            return None

        epochs = [r["epoch"] for r in self._history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves — {self.run_name}", fontsize=14)

        # --- Loss ---
        ax = axes[0, 0]
        train_loss = [r.get("train_loss") for r in self._history]
        val_loss = [r.get("val_loss") for r in self._history]
        if any(v is not None for v in train_loss):
            ax.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
        if any(v is not None for v in val_loss):
            ax.plot(epochs, val_loss, label="Val Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Token Accuracy ---
        ax = axes[0, 1]
        train_acc = [r.get("train_token_accuracy") for r in self._history]
        val_acc = [r.get("val_token_accuracy") for r in self._history]
        if any(v is not None for v in train_acc):
            ax.plot(epochs, train_acc, label="Train Token Acc", linewidth=1.5)
        if any(v is not None for v in val_acc):
            ax.plot(epochs, val_acc, label="Val Token Acc", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Token Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # --- Puzzle Accuracy ---
        ax = axes[1, 0]
        val_puzzle = [r.get("val_puzzle_accuracy") for r in self._history]
        if any(v is not None for v in val_puzzle):
            ax.plot(epochs, val_puzzle, label="Val Puzzle Acc",
                    linewidth=1.5, color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Puzzle Accuracy (Full Solve)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # --- Residual / Learning Rate ---
        ax = axes[1, 1]
        residuals = [r.get("train_avg_residual") for r in self._history]
        lr_vals = [r.get("learning_rate") for r in self._history]
        has_residual = any(v is not None for v in residuals)
        has_lr = any(v is not None for v in lr_vals)

        if has_residual:
            ax.plot(epochs, residuals, label="Avg Residual",
                    linewidth=1.5, color="purple")
            ax.set_ylabel("Residual (L2)")
        if has_lr:
            ax2 = ax.twinx()
            ax2.plot(epochs, lr_vals, label="Learning Rate",
                     linewidth=1.0, color="orange", linestyle="--")
            ax2.set_ylabel("Learning Rate")
            ax2.legend(loc="upper right")
        ax.set_xlabel("Epoch")
        ax.set_title("Convergence / Learning Rate")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(save_path) if save_path else self.log_dir / f"{self.run_name}_curves.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {out}")
        return out

    # -----------------------------------------------------------------
    # Export helpers
    # -----------------------------------------------------------------

    def export_history_json(
        self,
        path: str | Path | None = None,
    ) -> Path:
        """Export the full per-epoch history as a single JSON file.

        This is the format compatible with the existing
        ``training_history_*.json`` files produced by ``train_simplified.py``.
        """
        out = Path(path) if path else self.log_dir / f"training_history_{self.run_name}.json"
        # Reshape into the legacy list-per-metric format
        legacy: dict[str, list] = {}
        for record in self._history:
            for key, val in record.items():
                if key == "epoch" or isinstance(val, (list, dict)):
                    continue
                legacy.setdefault(key, []).append(val)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(legacy, f, indent=2)
        return out

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open file handles."""
        self._jsonl_file.close()
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_record(
        epoch: int,
        train: dict[str, Any],
        val: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge train/val summaries into a single flat record."""
        record: dict[str, Any] = {"epoch": epoch}

        # Prefix train metrics
        for key, val_ in train.items():
            if isinstance(val_, (list, dict)):
                continue  # skip batch_losses etc. for the flat record
            record[f"train_{key}"] = val_

        # Prefix val metrics
        if val is not None:
            for key, val_ in val.items():
                if isinstance(val_, (list, dict)):
                    continue
                record[f"val_{key}"] = val_

        return record

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        """Append one JSON line."""
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

    def _write_csv(self, record: dict[str, Any]) -> None:
        """Write / append to CSV.  Field set is locked on first write."""
        if self._csv_writer is None:
            self._csv_fields = list(record.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fields,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def _write_tensorboard(self, epoch: int, record: dict[str, Any]) -> None:
        """Write scalar metrics to TensorBoard."""
        if self._tb_writer is None:
            return
        for key, val_ in record.items():
            if key == "epoch":
                continue
            if isinstance(val_, (int, float)) and val_ is not None:
                self._tb_writer.add_scalar(key, val_, global_step=epoch)
        self._tb_writer.flush()
