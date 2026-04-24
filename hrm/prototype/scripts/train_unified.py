# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Training Script for Unified HRM

This script trains the UnifiedHRM model on multiple puzzle types:
- 4x4 Sudoku
- 9x9 Sudoku
- Maze puzzles

Usage:
    # Train on 9x9 Sudoku (default)
    python train_unified.py --puzzle sudoku_9x9 --epochs 100

    # Train on 4x4 Sudoku
    python train_unified.py --puzzle sudoku_4x4 --epochs 50

    # Train on all puzzle types (multi-task)
    python train_unified.py --puzzle all --epochs 100

    # Generate new training data first
    python train_unified.py --puzzle sudoku_9x9 --generate-data --num-samples 10000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.model_unified import UnifiedHRM, UnifiedHRMConfig, PuzzleType, PUZZLE_CONFIGS


# Dataset Classes


class SudokuDataset(Dataset):
    """Dataset for Sudoku puzzles (4x4 or 9x9)."""

    def __init__(
        self,
        puzzles: np.ndarray,
        solutions: np.ndarray,
        puzzle_type: PuzzleType,
    ):
        """
        Args:
            puzzles: Array of shape (N, grid_size, grid_size).
            solutions: Array of shape (N, grid_size, grid_size).
            puzzle_type: SUDOKU_4X4 or SUDOKU_9X9.
        """
        self.puzzles = torch.from_numpy(puzzles).long()
        self.solutions = torch.from_numpy(solutions).long()
        self.puzzle_type = puzzle_type

        # Flatten to sequences
        self.puzzles = self.puzzles.view(len(puzzles), -1)
        self.solutions = self.solutions.view(len(solutions), -1)

        # Create mask: True for empty cells (the ones we need to predict)
        self.empty_mask = self.puzzles == 0

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input": self.puzzles[idx],
            "target": self.solutions[idx],
            "empty_mask": self.empty_mask[idx],
            "puzzle_type": self.puzzle_type,
        }


class MazeDataset(Dataset):
    """Dataset for maze puzzles."""

    def __init__(
        self,
        mazes: np.ndarray,
        solutions: np.ndarray,
    ):
        """
        Args:
            mazes: Array of shape (N, H, W) with values 0-3.
            solutions: Array of shape (N, H, W) with path marked.
        """
        self.mazes = torch.from_numpy(mazes).long()
        self.solutions = torch.from_numpy(solutions).long()
        self.puzzle_type = PuzzleType.MAZE

        # Flatten
        N = len(mazes)
        self.mazes = self.mazes.view(N, -1)
        self.solutions = self.solutions.view(N, -1)

    def __len__(self) -> int:
        return len(self.mazes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input": self.mazes[idx],
            "target": self.solutions[idx],
            "puzzle_type": self.puzzle_type,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that handles mixed puzzle types."""
    inputs = torch.stack([item["input"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    # All items in batch should have same puzzle type
    puzzle_type = batch[0]["puzzle_type"]

    result = {
        "input": inputs,
        "target": targets,
        "puzzle_type": puzzle_type,
    }

    # Include empty_mask if available (Sudoku datasets)
    if "empty_mask" in batch[0]:
        result["empty_mask"] = torch.stack([item["empty_mask"] for item in batch])

    return result


# Data Loading Functions


def load_sudoku_data(
    data_path: str,
    puzzle_type: PuzzleType,
    max_samples: Optional[int] = None,
) -> SudokuDataset:
    """Load Sudoku data from .npz file."""
    print(f"Loading Sudoku data from {data_path}...")

    data = np.load(data_path)
    puzzles = data["puzzles"]
    solutions = data["solutions"]

    if max_samples and len(puzzles) > max_samples:
        indices = np.random.choice(len(puzzles), max_samples, replace=False)
        puzzles = puzzles[indices]
        solutions = solutions[indices]

    print(f"  Loaded {len(puzzles)} puzzles")
    return SudokuDataset(puzzles, solutions, puzzle_type)


def load_maze_data(
    data_path: str,
    max_samples: Optional[int] = None,
) -> MazeDataset:
    """Load maze data from .npz file."""
    print(f"Loading maze data from {data_path}...")

    data = np.load(data_path)
    mazes = data["mazes"]
    solutions = data["solutions"]

    if max_samples and len(mazes) > max_samples:
        indices = np.random.choice(len(mazes), max_samples, replace=False)
        mazes = mazes[indices]
        solutions = solutions[indices]

    print(f"  Loaded {len(mazes)} mazes")
    return MazeDataset(mazes, solutions)


def generate_sudoku_data(
    grid_size: int,
    num_samples: int,
    difficulty: str = "medium",
    output_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Sudoku training data."""
    print(f"Generating {num_samples} {grid_size}x{grid_size} Sudoku puzzles...")

    if grid_size == 9:
        from hrm.data.sudoku_generator import SudokuGenerator, Difficulty

        generator = SudokuGenerator(grid_size=9)

        # Map string difficulty to enum
        diff_map = {
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
            "mixed": Difficulty.MIXED,
        }
        diff_enum = diff_map.get(difficulty, Difficulty.MEDIUM)

        # For mixed, cycle through easy/medium/hard
        if diff_enum == Difficulty.MIXED:
            cycle_diffs = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
        else:
            cycle_diffs = None

        puzzles = []
        solutions = []

        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples}")

            current_diff = cycle_diffs[i % 3] if cycle_diffs else diff_enum
            puzzle, solution = generator.create_puzzle(difficulty=current_diff)
            puzzles.append(puzzle)
            solutions.append(solution)

        puzzles = np.array(puzzles)
        solutions = np.array(solutions)

    elif grid_size == 4:
        from hrm.prototype.generators.generator_4x4 import generate_puzzle

        puzzles = []
        solutions = []

        # Map difficulty to number of clues
        clues_map = {"easy": 10, "medium": 8, "hard": 6, "mixed": None}
        num_clues = clues_map.get(difficulty, 8)

        # For mixed difficulty on 4x4, cycle through clue counts
        mixed_clues = [10, 8, 6]  # easy, medium, hard

        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples}")

            if num_clues is None:  # mixed
                current_clues = mixed_clues[i % 3]
            else:
                current_clues = num_clues

            puzzle, solution = generate_puzzle(num_clues=current_clues)
            puzzles.append(puzzle)
            solutions.append(solution)

        puzzles = np.array(puzzles)
        solutions = np.array(solutions)

    else:
        raise ValueError(f"Unsupported grid size: {grid_size}")

    if output_path:
        np.savez(output_path, puzzles=puzzles, solutions=solutions)
        print(f"  Saved to {output_path}")

    return puzzles, solutions


# Training Functions


def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    empty_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute cross-entropy loss only on empty cells (the actual task).

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        empty_mask: (batch, seq_len) True for cells the model must predict
        label_smoothing: Label smoothing factor for regularization
    """
    if empty_mask is not None and empty_mask.any():
        # Only compute loss on empty cells
        masked_logits = logits[empty_mask]  # (num_empty, vocab_size)
        masked_targets = targets[empty_mask]  # (num_empty,)
        return F.cross_entropy(masked_logits, masked_targets, label_smoothing=label_smoothing)
    else:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            label_smoothing=label_smoothing,
        )


def compute_multistep_loss(
    all_step_logits: List[torch.Tensor],
    targets: torch.Tensor,
    empty_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute weighted loss across all outer ACT steps.

    Later steps are weighted more heavily—this trains the prediction
    feedback loop to produce useful intermediate predictions that
    the model can iteratively refine (backtracking).

    Weight schedule: step_i gets weight (i+1)/N
    """
    n_steps = len(all_step_logits)
    total_loss = torch.tensor(0.0, device=targets.device)
    total_weight = 0.0

    for step_idx, step_logits in enumerate(all_step_logits):
        weight = (step_idx + 1) / n_steps  # 1/N, 2/N, ..., 1.0
        step_loss = compute_masked_loss(step_logits, targets, empty_mask, label_smoothing)
        total_loss = total_loss + weight * step_loss
        total_weight += weight

    return total_loss / total_weight


def sudoku_constraint_penalty(
    logits: torch.Tensor,
    puzzle_type,
) -> torch.Tensor:
    """
    Differentiable Sudoku constraint violation penalty.

    Penalizes having the same digit appear multiple times in a row,
    column, or box. Uses softmax probabilities for differentiability.

    For each digit d and each row/col/box, the sum of probabilities
    for d should be 1.0. Deviation is penalized with squared error.
    """
    if puzzle_type == PuzzleType.SUDOKU_9X9:
        grid_size, box_size, num_digits = 9, 3, 9
    elif puzzle_type == PuzzleType.SUDOKU_4X4:
        grid_size, box_size, num_digits = 4, 2, 4
    else:
        return torch.tensor(0.0, device=logits.device)

    batch = logits.shape[0]
    probs = F.softmax(logits.float(), dim=-1)  # (batch, seq_len, vocab)
    probs = probs.view(batch, grid_size, grid_size, -1)

    penalty = torch.tensor(0.0, device=logits.device)
    for d in range(1, num_digits + 1):
        dp = probs[:, :, :, d]  # (batch, grid, grid)

        # Row constraint: digit d appears once per row
        penalty = penalty + ((dp.sum(dim=2) - 1.0) ** 2).mean()

        # Column constraint: digit d appears once per column
        penalty = penalty + ((dp.sum(dim=1) - 1.0) ** 2).mean()

        # Box constraint: digit d appears once per box
        for br in range(0, grid_size, box_size):
            for bc in range(0, grid_size, box_size):
                box_sum = dp[:, br : br + box_size, bc : bc + box_size].sum(dim=(1, 2))
                penalty = penalty + ((box_sum - 1.0) ** 2).mean()

    return penalty / num_digits


def train_epoch(
    model: UnifiedHRM,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    use_multistep_loss: bool = False,
    constraint_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch with optional mixed-precision."""
    model.train()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_q_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        puzzle_type = batch["puzzle_type"]
        empty_mask = batch.get("empty_mask")
        if empty_mask is not None:
            empty_mask = empty_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(inputs, puzzle_type, targets=targets)

            # LM loss: multi-step or final-step only
            if (
                use_multistep_loss
                and "all_step_logits" in output
                and len(output["all_step_logits"]) > 1
            ):
                lm_loss = compute_multistep_loss(
                    output["all_step_logits"],
                    targets,
                    empty_mask,
                    label_smoothing=label_smoothing,
                )
            else:
                lm_loss = compute_masked_loss(
                    output["logits"],
                    targets,
                    empty_mask,
                    label_smoothing=label_smoothing,
                )

            # Optional Sudoku constraint penalty
            loss = lm_loss + 0.5 * output["q_halt_loss"]
            if constraint_weight > 0:
                constraint_loss = sudoku_constraint_penalty(output["logits"], puzzle_type)
                loss = loss + constraint_weight * constraint_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Statistics — measure accuracy only on empty cells
        predictions = output["predictions"]
        if empty_mask is not None and empty_mask.any():
            total_correct += (predictions[empty_mask] == targets[empty_mask]).sum().item()
            total_tokens += empty_mask.sum().item()
        else:
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
        total_loss += loss.item() * inputs.size(0)
        total_lm_loss += lm_loss.item() * inputs.size(0)
        total_q_loss += output["q_halt_loss"].item() * inputs.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f} "
                f"(LM: {lm_loss.item():.4f}, "
                f"Q: {output['q_halt_loss'].item():.4f})"
            )

    n = len(dataloader.dataset)
    return {
        "loss": total_loss / n,
        "lm_loss": total_lm_loss / n,
        "q_halt_loss": total_q_loss / n,
        "accuracy": total_correct / total_tokens,
    }


@torch.no_grad()
def evaluate(
    model: UnifiedHRM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation/test set with proper empty-cell metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    puzzle_correct = 0
    puzzle_total = 0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        puzzle_type = batch["puzzle_type"]
        empty_mask = batch.get("empty_mask")
        if empty_mask is not None:
            empty_mask = empty_mask.to(device)

        output = model(inputs, puzzle_type, targets=targets)
        predictions = output["predictions"]

        # Compute masked loss for fair comparison
        if "all_step_logits" in output and len(output["all_step_logits"]) > 1:
            lm_loss = compute_multistep_loss(
                output["all_step_logits"], targets, empty_mask, label_smoothing=0.0
            )
        else:
            lm_loss = compute_masked_loss(
                output["logits"], targets, empty_mask, label_smoothing=0.0
            )
        total_loss += lm_loss.item() * inputs.size(0)

        if empty_mask is not None and empty_mask.any():
            # Token accuracy on empty cells only
            total_correct += (predictions[empty_mask] == targets[empty_mask]).sum().item()
            total_tokens += empty_mask.sum().item()

            # Puzzle-level: all EMPTY cells correct
            per_puzzle_correct = ((predictions == targets) | ~empty_mask).all(dim=1)
            puzzle_correct += per_puzzle_correct.sum().item()
        else:
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            puzzle_correct += (predictions == targets).all(dim=1).sum().item()
        puzzle_total += inputs.size(0)

    return {
        "loss": total_loss / len(dataloader.dataset),
        "token_accuracy": total_correct / total_tokens,
        "puzzle_accuracy": puzzle_correct / puzzle_total,
    }


def train(
    model: UnifiedHRM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: torch.device,
    save_dir: str,
    puzzle_name: str,
    use_amp: bool = False,
    use_compile: bool = False,
    use_multistep_loss: bool = False,
    constraint_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> Dict:
    """Main training loop with optional mixed-precision and torch.compile."""
    print(f"\nTraining on {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"  GPU Memory: {gpu_mem:.1f} GB")
        print(f"  Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")
        print(f"  torch.compile: {'enabled' if use_compile else 'disabled'}")
    print(f"Model parameters: {model.num_parameters:,}")

    model = model.to(device)

    # Optional torch.compile for faster training (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_token_accuracy": [],
        "val_puzzle_accuracy": [],
    }

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    patience = 30  # Early stopping patience
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
            use_multistep_loss=use_multistep_loss,
            constraint_weight=constraint_weight,
            label_smoothing=label_smoothing,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])

        print(
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"(LM: {train_metrics['lm_loss']:.4f}, Q: {train_metrics['q_halt_loss']:.4f}), "
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        # Validate
        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_token_accuracy"].append(val_metrics["token_accuracy"])
            history["val_puzzle_accuracy"].append(val_metrics["puzzle_accuracy"])

            print(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Token Acc: {val_metrics['token_accuracy']:.4f}, "
                f"Puzzle Acc: {val_metrics['puzzle_accuracy']:.4f}"
            )

            # Save best model (track both puzzle accuracy and val loss)
            improved = False
            if val_metrics["puzzle_accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["puzzle_accuracy"]
                improved = True
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                improved = True

            if improved:
                patience_counter = 0
                save_path = os.path.join(save_dir, f"unified_hrm_{puzzle_name}_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": best_val_accuracy,
                        "config": model.config,
                    },
                    save_path,
                )
                print(
                    f"  Saved best model (accuracy: {best_val_accuracy:.4f}, val_loss: {best_val_loss:.4f})"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)"
                    )
                    break

        scheduler.step()

    # Save final model
    save_path = os.path.join(save_dir, f"unified_hrm_{puzzle_name}_final.pt")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": model.config,
        },
        save_path,
    )
    print(f"\nSaved final model to {save_path}")

    # Save history
    history_path = os.path.join(save_dir, f"training_history_{puzzle_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# Main


def main():
    parser = argparse.ArgumentParser(description="Train Unified HRM")

    # Data arguments
    parser.add_argument(
        "--puzzle",
        type=str,
        default="sudoku_9x9",
        choices=["sudoku_4x4", "sudoku_9x9", "maze", "all"],
        help="Puzzle type to train on",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to training data (.npz)")
    parser.add_argument("--generate-data", action="store_true", help="Generate new training data")
    parser.add_argument(
        "--num-samples", type=int, default=10000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard", "mixed"],
        help='Puzzle difficulty ("mixed" = equal parts easy/medium/hard)',
    )

    # Model arguments
    parser.add_argument("--hidden-size", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument(
        "--halt-max-steps",
        type=int,
        default=None,
        help="Override max outer ACT halting steps (default: from PUZZLE_CONFIGS)",
    )
    parser.add_argument(
        "--l-cycles",
        type=int,
        default=None,
        help="Override L-cycles per H-cycle (default: from PUZZLE_CONFIGS)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")

    # Other arguments
    parser.add_argument("--save-dir", type=str, default="model", help="Directory to save models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")
    parser.add_argument(
        "--amp", action="store_true", help="Enable mixed-precision training (AMP) for GPU speedup"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for faster training (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--multistep-loss",
        action="store_true",
        help="Train on all outer ACT steps (default: final step only)",
    )
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=0.0,
        help="Weight for Sudoku constraint penalty (0=off, try 0.1)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing factor (0=off)"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Data paths
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # Load or generate data based on puzzle type
    if args.puzzle == "sudoku_9x9":
        puzzle_type = PuzzleType.SUDOKU_9X9
        default_data_path = data_dir / "sudoku_9x9_train.npz"

        if args.generate_data or not default_data_path.exists():
            puzzles, solutions = generate_sudoku_data(
                grid_size=9,
                num_samples=args.num_samples,
                difficulty=args.difficulty,
                output_path=str(default_data_path),
            )
            dataset = SudokuDataset(puzzles, solutions, puzzle_type)
        else:
            data_path = args.data_path or str(default_data_path)
            dataset = load_sudoku_data(data_path, puzzle_type)

    elif args.puzzle == "sudoku_4x4":
        puzzle_type = PuzzleType.SUDOKU_4X4
        default_data_path = data_dir / "sudoku_4x4_train.npz"

        if args.generate_data or not default_data_path.exists():
            puzzles, solutions = generate_sudoku_data(
                grid_size=4,
                num_samples=args.num_samples,
                difficulty=args.difficulty,
                output_path=str(default_data_path),
            )
            dataset = SudokuDataset(puzzles, solutions, puzzle_type)
        else:
            data_path = args.data_path or str(default_data_path)
            dataset = load_sudoku_data(data_path, puzzle_type)

    elif args.puzzle == "maze":
        # For mazes, we need pre-generated data or a maze generator
        puzzle_type = PuzzleType.MAZE
        if args.data_path:
            dataset = load_maze_data(args.data_path)
        else:
            print("Error: Maze training requires --data-path to maze data")
            print("Generate maze data first using hrm/prototype/generators/maze_generator.py")
            sys.exit(1)

    else:  # 'all' - multi-task
        print("Multi-task training not yet implemented. Train on individual puzzle types.")
        sys.exit(1)

    # Train/val split
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    print(f"\nDataset: {len(dataset)} total, {n_train} train, {n_val} val")

    # DataLoaders — use pin_memory for GPU, num_workers for parallel loading
    use_cuda = device.type == "cuda"
    loader_kwargs = dict(
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        persistent_workers=True if use_cuda else False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Create model
    config_kwargs = dict(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers_L=args.num_layers,
        num_layers_H=args.num_layers,
    )
    if args.halt_max_steps is not None:
        config_kwargs["halt_max_steps"] = args.halt_max_steps
    if args.l_cycles is not None:
        config_kwargs["L_cycles"] = args.l_cycles
    config = UnifiedHRMConfig(**config_kwargs)
    model = UnifiedHRM(config)

    print(f"\nModel Config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num layers (L/H): {config.num_layers_L}/{config.num_layers_H}")
    print(f"  H_cycles: {config.H_cycles}, L_cycles: {config.L_cycles}")
    print(f"  halt_max_steps: {config.halt_max_steps}")
    print(f"  Parameters: {model.num_parameters:,}")

    # Determine AMP usage (only on CUDA)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not device.type == "cuda":
        print("Warning: --amp ignored (only supported on CUDA devices)")

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        puzzle_name=args.puzzle,
        use_amp=use_amp,
        use_compile=args.compile,
        use_multistep_loss=args.multistep_loss,
        constraint_weight=args.constraint_weight,
        label_smoothing=args.label_smoothing,
    )

    print("\nTraining complete!")
    print(f"Best validation puzzle accuracy: {max(history['val_puzzle_accuracy']):.4f}")


if __name__ == "__main__":
    main()
