# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
HRM 4x4 Sudoku Solver - Training and Demonstration
Trains the model and demonstrates it solving puzzles
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add hrm to path
sys.path.insert(0, str(Path(__file__).parent / "hrm"))

from hrm.models.model_4x4_layers import HRM_4x4
from hrm.generators.generator_4x4 import generate_puzzle, generate_dataset, print_puzzle


class SudokuDataset(Dataset):
    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray):
        self.puzzles = torch.from_numpy(puzzles).long()

        # Target: first empty cell and its digit
        self.targets = []
        for puzzle, solution in zip(puzzles, solutions):
            empty = np.argwhere(puzzle == 0)
            if len(empty) > 0:
                r, c = empty[0]
                cell_idx = r * 4 + c
                digit = solution[r, c] - 1  # 0-indexed
            else:
                cell_idx = 0
                digit = 0
            self.targets.append((cell_idx, digit))

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return self.puzzles[idx], self.targets[idx][0], self.targets[idx][1]


def train_model(num_epochs=20, train_size=1000):
    """Train the model"""
    print("\n" + "=" * 70)
    print("HRM 4x4 SUDOKU SOLVER - TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Generate data
    print(f"\nGenerating {train_size} training puzzles...")
    train_puzzles, train_solutions = generate_dataset(train_size)
    val_puzzles, val_solutions = generate_dataset(100)

    train_data = SudokuDataset(train_puzzles, train_solutions)
    val_data = SudokuDataset(val_puzzles, val_solutions)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Create model
    print("\nInitializing model...")
    model = HRM_4x4(hidden_dim=64, n_heads=4, n_outer_cycles=5, n_inner_steps=10, dropout=0.1).to(
        device
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Iterations per sample: {5 * 10} (5 cycles x 10 steps)")

    best_val_acc = 0

    # Training loop
    print(f"\n{'-'*70}")
    print(f"Training for {num_epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for puzzles, cells, digits in pbar:
            puzzles = puzzles.to(device)
            cells = cells.to(device).long()
            digits = digits.to(device).long()

            optimizer.zero_grad()

            # Forward pass
            cell_logits, digit_logits, traces = model(puzzles, return_traces=True)

            # Loss: predict correct cell and digit
            loss = nn.functional.cross_entropy(
                cell_logits, cells
            ) + 2.0 * nn.functional.cross_entropy(digit_logits, digits)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            cell_pred = torch.argmax(cell_logits, dim=1)
            digit_pred = torch.argmax(digit_logits, dim=1)
            correct += ((cell_pred == cells) & (digit_pred == digits)).sum().item()
            total += len(puzzles)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct/total:.1%}"})

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for puzzles, cells, digits in val_loader:
                puzzles = puzzles.to(device)
                cells = cells.to(device).long()
                digits = digits.to(device).long()

                cell_logits, digit_logits, _ = model(puzzles, return_traces=False)

                cell_pred = torch.argmax(cell_logits, dim=1)
                digit_pred = torch.argmax(digit_logits, dim=1)
                val_correct += ((cell_pred == cells) & (digit_pred == digits)).sum().item()
                val_total += len(puzzles)

        val_acc = val_correct / val_total

        # Update best
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            status = " [BEST]"

        print(f"Epoch {epoch:2d}: Train={train_acc:.1%}, Val={val_acc:.1%}{status}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1%}")

    return model


def demonstrate_model(model):
    """Demonstrate the trained model on puzzles"""
    print("\n" + "=" * 70)
    print("DEMONSTRATION - Solving Random Puzzles")
    print("=" * 70)

    device = next(model.parameters()).device
    model.eval()

    correct = 0
    total = 5

    for i in range(total):
        print(f"\n{'-'*70}")
        print(f"Puzzle {i+1}/{total}")
        print("-" * 70)

        # Generate puzzle
        puzzle, solution = generate_puzzle(num_clues=10)

        print_puzzle(puzzle, "Input Puzzle")

        # Get prediction
        puzzle_tensor = torch.from_numpy(puzzle).unsqueeze(0).long().to(device)

        with torch.no_grad():
            cell_logits, digit_logits, traces = model(puzzle_tensor, return_traces=True)

            # Get predictions
            cell_pred = torch.argmax(cell_logits[0]).item()
            digit_pred = torch.argmax(digit_logits[0]).item() + 1

            # Get confidence
            cell_conf = torch.softmax(cell_logits[0], dim=0)[cell_pred].item()
            digit_conf = torch.softmax(digit_logits[0], dim=0)[digit_pred - 1].item()

        # Convert to row, col
        pred_row, pred_col = cell_pred // 4, cell_pred % 4

        # Find actual target
        empty_cells = np.argwhere(puzzle == 0)
        if len(empty_cells) > 0:
            target_row, target_col = empty_cells[0]
            target_digit = solution[target_row, target_col]
        else:
            continue

        # Display prediction
        print(f"\nModel Prediction:")
        print(f"  Cell:  Row {pred_row}, Column {pred_col}  (confidence: {cell_conf:.1%})")
        print(f"  Digit: {digit_pred}                    (confidence: {digit_conf:.1%})")

        print(f"\nCorrect Answer:")
        print(f"  Cell:  Row {target_row}, Column {target_col}")
        print(f"  Digit: {target_digit}")

        # Check correctness
        is_correct = (
            pred_row == target_row and pred_col == target_col and digit_pred == target_digit
        )

        if is_correct:
            print(f"\nResult: CORRECT")
            correct += 1
        else:
            print(f"\nResult: INCORRECT")

        print(
            f"\nIterations: {traces['num_iterations']}, "
            f"Final residual: {traces['final_residual']:.6f}"
        )

        print_puzzle(solution, "Complete Solution")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"\nThe model learned to solve 4x4 Sudoku puzzles through")
    print(f"hierarchical reasoning with {5} planning cycles and")
    print(f"{10} refinement steps per cycle.")
    print("=" * 70 + "\n")


def main():
    """Main function"""
    print("\n" + "-" * 70)
    print("HIERARCHICAL REASONING MODEL - 4x4 SUDOKU SOLVER")
    print("Training and Demonstration")
    print("-" * 70)

    # Train model
    model = train_model(num_epochs=20, train_size=1000)

    # Demonstrate the trained model
    demonstrate_model(model)


if __name__ == "__main__":
    main()
