"""
Training Script for Simplified HRM (L-Module Only)

This script trains the SimplifiedHRM model — a simplified HRM that
eliminates the H-module entirely, using an expanded 8-layer transformer
as the sole reasoning engine.

Based on: Ge, Liao & Poggio (2025) "Hierarchical Reasoning Models:
Perspectives and Misconceptions" (arXiv:2510.00355v2)

Key differences from train_unified.py:
    - No H-module / Planner (L-module only)
    - No ACT halting (fixed reasoning steps; paper shows max steps is optimal)
    - No Q-learning loss (no halting decision)
    - One-step gradient training (equivalent to LCM/diffusion training)
    - Prediction feedback / self-conditioning between reasoning steps

Usage:
    # Paper-aligned: 1K hard puzzles, 100 epochs, lr=3e-4, wd=0.1
    python train_simplified.py --puzzle sudoku_9x9 --generate-data \\
        --num-samples 1000 --difficulty hard --epochs 100

    # Augmented: more data for potentially better results
    python train_simplified.py --puzzle sudoku_9x9 --generate-data \\
        --num-samples 50000 --difficulty medium --epochs 150 \\
        --batch-size 64 --amp \\
        --data-output-path data/sudoku_9x9_medium_50k.npz

    # Train on 4x4 Sudoku
    python train_simplified.py --puzzle sudoku_4x4 --epochs 50

    # Small model for quick experiments / RPi5
    python train_simplified.py --puzzle sudoku_4x4 --hidden-size 128 \\
        --num-layers 4 --reasoning-steps 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.model_simplified import (
    SimplifiedHRM,
    SimplifiedHRMConfig,
    PuzzleType,
    PUZZLE_DEFAULTS,
    create_simplified_hrm,
    create_small_simplified_hrm,
)


# =============================================================================
# Dataset Classes
# =============================================================================

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
        self.empty_mask = (self.puzzles == 0)

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': self.puzzles[idx],
            'target': self.solutions[idx],
            'empty_mask': self.empty_mask[idx],
            'puzzle_type': self.puzzle_type,
        }


class MazeDataset(Dataset):
    """Dataset for maze puzzles."""

    def __init__(
        self,
        mazes: np.ndarray,
        solutions: np.ndarray,
    ):
        self.mazes = torch.from_numpy(mazes).long()
        self.solutions = torch.from_numpy(solutions).long()
        self.puzzle_type = PuzzleType.MAZE

        N = len(mazes)
        self.mazes = self.mazes.view(N, -1)
        self.solutions = self.solutions.view(N, -1)

    def __len__(self) -> int:
        return len(self.mazes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': self.mazes[idx],
            'target': self.solutions[idx],
            'puzzle_type': self.puzzle_type,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that handles mixed puzzle types."""
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    puzzle_type = batch[0]['puzzle_type']

    result = {
        'input': inputs,
        'target': targets,
        'puzzle_type': puzzle_type,
    }

    if 'empty_mask' in batch[0]:
        result['empty_mask'] = torch.stack([item['empty_mask'] for item in batch])

    return result


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_sudoku_data(
    data_path: str,
    puzzle_type: PuzzleType,
    max_samples: Optional[int] = None,
) -> SudokuDataset:
    """Load Sudoku data from .npz file."""
    print(f"Loading Sudoku data from {data_path}...")

    data = np.load(data_path)
    puzzles = data['puzzles']
    solutions = data['solutions']

    if max_samples and len(puzzles) > max_samples:
        indices = np.random.choice(len(puzzles), max_samples, replace=False)
        puzzles = puzzles[indices]
        solutions = solutions[indices]

    print(f"  Loaded {len(puzzles)} puzzles")
    return SudokuDataset(puzzles, solutions, puzzle_type)


def generate_sudoku_data(
    grid_size: int,
    num_samples: int,
    difficulty: str = 'medium',
    output_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Sudoku training data."""
    print(f"Generating {num_samples} {grid_size}x{grid_size} Sudoku puzzles...")

    if grid_size == 9:
        from generators.sudoku_generator import SudokuGenerator, Difficulty
        generator = SudokuGenerator(grid_size=9)

        diff_map = {
            'easy': Difficulty.EASY,
            'medium': Difficulty.MEDIUM,
            'hard': Difficulty.HARD,
            'mixed': Difficulty.MIXED,
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

        clues_map = {'easy': 10, 'medium': 8, 'hard': 6, 'mixed': None}
        num_clues = clues_map.get(difficulty, 8)
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


# =============================================================================
# Training Functions
# =============================================================================

def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    empty_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,                      # FIX 1: was 0.1
) -> torch.Tensor:
    """
    Compute cross-entropy loss only on empty cells (the actual task).

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        empty_mask: (batch, seq_len) True for cells the model must predict
        label_smoothing: Label smoothing factor (paper: 0.0)
    """
    if empty_mask is not None and empty_mask.any():
        masked_logits = logits[empty_mask]
        masked_targets = targets[empty_mask]
        return F.cross_entropy(
            masked_logits, masked_targets, label_smoothing=label_smoothing
        )
    else:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1),
            label_smoothing=label_smoothing,
        )


def sudoku_constraint_penalty(
    logits: torch.Tensor,
    puzzle_type: PuzzleType,
) -> torch.Tensor:
    """
    Differentiable Sudoku constraint violation penalty.

    Not in the paper — our own extension for experimentation.
    Penalises duplicate digits in rows, columns, and boxes using
    softmax probabilities for differentiability.
    """
    if puzzle_type == PuzzleType.SUDOKU_9X9:
        grid_size, box_size, num_digits = 9, 3, 9
    elif puzzle_type == PuzzleType.SUDOKU_4X4:
        grid_size, box_size, num_digits = 4, 2, 4
    else:
        return torch.tensor(0.0, device=logits.device)

    batch = logits.shape[0]
    probs = F.softmax(logits.float(), dim=-1)
    probs = probs.view(batch, grid_size, grid_size, -1)

    penalty = torch.tensor(0.0, device=logits.device)
    for d in range(1, num_digits + 1):
        dp = probs[:, :, :, d]

        # Row constraint
        penalty = penalty + ((dp.sum(dim=2) - 1.0) ** 2).mean()
        # Column constraint
        penalty = penalty + ((dp.sum(dim=1) - 1.0) ** 2).mean()
        # Box constraint
        for br in range(0, grid_size, box_size):
            for bc in range(0, grid_size, box_size):
                box_sum = dp[:, br:br+box_size, bc:bc+box_size].sum(dim=(1, 2))
                penalty = penalty + ((box_sum - 1.0) ** 2).mean()

    return penalty / num_digits


def train_epoch(
    model: SimplifiedHRM,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    constraint_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch. Loss = final-step CE only (paper-aligned)."""
    model.train()
    total_loss = 0.0
    total_lm_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['input'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        puzzle_type = batch['puzzle_type']
        empty_mask = batch.get('empty_mask')
        if empty_mask is not None:
            empty_mask = empty_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(inputs, puzzle_type, targets=targets)

            # FIX 5: Always final-step loss only (paper-aligned).
            # The paper does NOT use deep supervision — only the final
            # step's logits are used for cross-entropy.
            lm_loss = compute_masked_loss(
                output['logits'], targets, empty_mask,
                label_smoothing=label_smoothing,
            )

            loss = lm_loss

            # Optional Sudoku constraint penalty (our extension)
            if constraint_weight > 0:
                constraint_loss = sudoku_constraint_penalty(
                    output['logits'], puzzle_type
                )
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

        # Statistics — accuracy on empty cells only
        predictions = output['predictions']
        if empty_mask is not None and empty_mask.any():
            total_correct += (predictions[empty_mask] == targets[empty_mask]).sum().item()
            total_tokens += empty_mask.sum().item()
        else:
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()

        total_loss += loss.item() * inputs.size(0)
        total_lm_loss += lm_loss.item() * inputs.size(0)

        # FIX 6: Removed dead output['deep_supervision_loss'] tracking
        # (model never outputs this key)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")

    n = len(dataloader.dataset)
    return {
        'loss': total_loss / n,
        'lm_loss': total_lm_loss / n,
        'accuracy': total_correct / total_tokens,
    }


@torch.no_grad()
def evaluate(
    model: SimplifiedHRM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model — always final-step loss only (paper-aligned)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    puzzle_correct = 0
    puzzle_total = 0

    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        puzzle_type = batch['puzzle_type']
        empty_mask = batch.get('empty_mask')
        if empty_mask is not None:
            empty_mask = empty_mask.to(device)

        output = model(inputs, puzzle_type, targets=targets)
        predictions = output['predictions']

        # FIX 5: Always final-step loss (was routing through multistep)
        lm_loss = compute_masked_loss(
            output['logits'], targets, empty_mask, label_smoothing=0.0,
        )
        total_loss += lm_loss.item() * inputs.size(0)

        if empty_mask is not None and empty_mask.any():
            total_correct += (predictions[empty_mask] == targets[empty_mask]).sum().item()
            total_tokens += empty_mask.sum().item()
            per_puzzle_correct = ((predictions == targets) | ~empty_mask).all(dim=1)
            puzzle_correct += per_puzzle_correct.sum().item()
        else:
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            puzzle_correct += (predictions == targets).all(dim=1).sum().item()
        puzzle_total += inputs.size(0)

    return {
        'loss': total_loss / len(dataloader.dataset),
        'token_accuracy': total_correct / total_tokens,
        'puzzle_accuracy': puzzle_correct / puzzle_total,
    }


def train(
    model: SimplifiedHRM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    weight_decay: float,                                # FIX 2: was hardcoded
    device: torch.device,
    save_dir: str,
    puzzle_name: str,
    use_amp: bool = False,
    use_compile: bool = False,
    constraint_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> Dict:
    """Main training loop with optional mixed-precision and torch.compile."""
    print(f"\nTraining on {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"  GPU Memory: {gpu_mem:.1f} GB")
        print(f"  Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")
        print(f"  torch.compile: {'enabled' if use_compile else 'disabled'}")
    print(f"Model parameters: {model.num_parameters:,}")

    model = model.to(device)

    # Optional torch.compile for faster training (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    # FIX 2: weight_decay from CLI arg (paper: 0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # FIX 8: eta_min = lr * 0.1 (paper uses lr_min_ratio=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1,
    )

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_token_accuracy': [],
        'val_puzzle_accuracy': [],
    }

    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp,
            constraint_weight=constraint_weight,
            label_smoothing=label_smoothing,
        )
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])

        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_token_accuracy'].append(val_metrics['token_accuracy'])
            history['val_puzzle_accuracy'].append(val_metrics['puzzle_accuracy'])

            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Token Acc: {val_metrics['token_accuracy']:.4f}, "
                  f"Puzzle Acc: {val_metrics['puzzle_accuracy']:.4f}")

            # Save best model
            improved = False
            if val_metrics['puzzle_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['puzzle_accuracy']
                improved = True
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                improved = True

            if improved:
                patience_counter = 0
                save_path = os.path.join(save_dir, f'simplified_hrm_{puzzle_name}_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_accuracy,
                    'config': model.config,
                }, save_path)
                print(f"  Saved best model (accuracy: {best_val_accuracy:.4f}, val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        scheduler.step()

    # Save final model
    save_path = os.path.join(save_dir, f'simplified_hrm_{puzzle_name}_final.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, save_path)
    print(f"\nSaved final model to {save_path}")

    # Save history
    history_path = os.path.join(save_dir, f'training_history_simplified_{puzzle_name}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Simplified HRM (L-Module Only)'
    )

    # Data arguments
    parser.add_argument('--puzzle', type=str, default='sudoku_9x9',
                        choices=['sudoku_4x4', 'sudoku_9x9', 'maze'],
                        help='Puzzle type to train on')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to training data (.npz)')
    parser.add_argument('--data-output-path', type=str, default=None,   # FIX 7
                        help='Custom output path for generated data (.npz)')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate new training data')
    parser.add_argument('--num-samples', type=int, default=1000,        # FIX 4: was 10000
                        help='Number of samples to generate (paper: 1000)')
    parser.add_argument('--difficulty', type=str, default='hard',       # Paper: hard
                        choices=['easy', 'medium', 'hard', 'mixed'],
                        help='Puzzle difficulty (paper: hard; "mixed" = equal parts easy/medium/hard)')

    # Model arguments
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Model hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=8,
                        help='Number of transformer layers in L-module (paper: 8)')
    parser.add_argument('--reasoning-steps', type=int, default=16,
                        help='Number of reasoning steps (paper: 16, max steps is optimal)')
    parser.add_argument('--no-feedback', action='store_true',
                        help='Disable prediction feedback / self-conditioning')
    parser.add_argument('--small', action='store_true',
                        help='Use small model config (128d, 4L, 8 steps) for quick experiments / RPi5')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (paper: 100)')
    parser.add_argument('--batch-size', type=int, default=64,           # Practical default
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,               # FIX 3: was 1e-4
                        help='Learning rate (paper: 3e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.1,      # FIX 2: new arg
                        help='Weight decay (paper: 0.1)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')

    # Other arguments
    parser.add_argument('--save-dir', type=str, default='model',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed-precision training (AMP) for GPU speedup')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--constraint-weight', type=float, default=0.0,
                        help='Sudoku constraint penalty weight (not in paper, 0=off)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (not in paper, 0=off)')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Data paths
    data_dir = PROJECT_ROOT / 'data'
    data_dir.mkdir(exist_ok=True)

    # Load or generate data
    if args.puzzle == 'sudoku_9x9':
        puzzle_type = PuzzleType.SUDOKU_9X9
        default_data_path = data_dir / 'sudoku_9x9_train.npz'

        if args.generate_data or not default_data_path.exists():
            out_path = args.data_output_path or str(default_data_path)  # FIX 7
            puzzles, solutions = generate_sudoku_data(
                grid_size=9,
                num_samples=args.num_samples,
                difficulty=args.difficulty,
                output_path=out_path,
            )
            dataset = SudokuDataset(puzzles, solutions, puzzle_type)
        else:
            data_path = args.data_path or str(default_data_path)
            dataset = load_sudoku_data(data_path, puzzle_type)

    elif args.puzzle == 'sudoku_4x4':
        puzzle_type = PuzzleType.SUDOKU_4X4
        default_data_path = data_dir / 'sudoku_4x4_train.npz'

        if args.generate_data or not default_data_path.exists():
            out_path = args.data_output_path or str(default_data_path)  # FIX 7
            puzzles, solutions = generate_sudoku_data(
                grid_size=4,
                num_samples=args.num_samples,
                difficulty=args.difficulty,
                output_path=out_path,
            )
            dataset = SudokuDataset(puzzles, solutions, puzzle_type)
        else:
            data_path = args.data_path or str(default_data_path)
            dataset = load_sudoku_data(data_path, puzzle_type)

    elif args.puzzle == 'maze':
        puzzle_type = PuzzleType.MAZE
        if args.data_path:
            from scripts.train_unified import load_maze_data
            dataset = load_maze_data(args.data_path)
        else:
            print("Error: Maze training requires --data-path to maze data")
            print("Generate maze data first using hrm/prototype/generators/maze_generator.py")
            sys.exit(1)

    # Train/val split
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    print(f"\nDataset: {len(dataset)} total, {n_train} train, {n_val} val")

    # DataLoaders
    use_cuda = device.type == 'cuda'
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
    if args.small:
        model = create_small_simplified_hrm(
            use_prediction_feedback=not args.no_feedback,
        )
        print("\nUsing SMALL model configuration (for quick experiments / RPi5)")
    else:
        config = SimplifiedHRMConfig(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_reasoning_steps=args.reasoning_steps,
            use_prediction_feedback=not args.no_feedback,
        )
        model = SimplifiedHRM(config)

    print(f"\nModel Config (Simplified HRM — L-Module Only):")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num heads: {model.config.num_heads}")
    print(f"  Num layers (L-only): {model.config.num_layers}")
    print(f"  Reasoning steps: {model.config.num_reasoning_steps}")
    print(f"  Prediction feedback: {model.config.use_prediction_feedback}")
    print(f"  Parameters: {model.num_parameters:,}")

    # AMP
    use_amp = args.amp and device.type == 'cuda'
    if args.amp and not device.type == 'cuda':
        print("Warning: --amp ignored (only supported on CUDA devices)")

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,                     # FIX 2: pass through
        device=device,
        save_dir=args.save_dir,
        puzzle_name=args.puzzle,
        use_amp=use_amp,
        use_compile=args.compile,
        constraint_weight=args.constraint_weight,
        label_smoothing=args.label_smoothing,
    )

    print("\nTraining complete!")
    if history['val_puzzle_accuracy']:
        print(f"Best validation puzzle accuracy: {max(history['val_puzzle_accuracy']):.4f}")


if __name__ == '__main__':
    main()
