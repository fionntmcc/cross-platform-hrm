# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Demo Script for Unified HRM

Shows how to load a trained model and solve different puzzle types.

Usage:
    # Solve a 9x9 Sudoku
    python demo_unified.py --puzzle sudoku_9x9 --model model/unified_hrm_sudoku_9x9_best.pt

    # Solve a 4x4 Sudoku
    python demo_unified.py --puzzle sudoku_4x4 --model model/unified_hrm_sudoku_4x4_best.pt

    # Interactive mode
    python demo_unified.py --interactive
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.model_unified import UnifiedHRM, UnifiedHRMConfig, PuzzleType


def load_model(model_path: str, device: torch.device) -> UnifiedHRM:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use default
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = UnifiedHRMConfig()

    model = UnifiedHRM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded model with {model.num_parameters:,} parameters")
    return model


def print_sudoku_4x4(grid: np.ndarray, title: str = ""):
    """Pretty print a 4x4 Sudoku grid."""
    if title:
        print(f"\n{title}")
    print("┌───┬───┐")
    for i in range(4):
        row = " ".join(str(int(x)) if x > 0 else "." for x in grid[i])
        if i == 2:
            print("├───┼───┤")
        print(f"│{row[:3]}│{row[4:]}│")
    print("└───┴───┘")


def print_sudoku_9x9(grid: np.ndarray, title: str = ""):
    """Pretty print a 9x9 Sudoku grid."""
    if title:
        print(f"\n{title}")
    print("┌───────┬───────┬───────┐")
    for i in range(9):
        row = [str(int(x)) if x > 0 else "." for x in grid[i]]
        formatted = f"│ {' '.join(row[0:3])} │ {' '.join(row[3:6])} │ {' '.join(row[6:9])} │"
        print(formatted)
        if i == 2 or i == 5:
            print("├───────┼───────┼───────┤")
    print("└───────┴───────┴───────┘")


def validate_sudoku(solution: np.ndarray, grid_size: int) -> bool:
    """Check if a Sudoku solution is valid."""
    expected = set(range(1, grid_size + 1))

    # Check rows
    for i in range(grid_size):
        if set(solution[i]) != expected:
            return False

    # Check columns
    for j in range(grid_size):
        if set(solution[:, j]) != expected:
            return False

    # Check boxes
    box_size = int(np.sqrt(grid_size))
    for bi in range(box_size):
        for bj in range(box_size):
            box = solution[bi * box_size : (bi + 1) * box_size, bj * box_size : (bj + 1) * box_size]
            if set(box.flatten()) != expected:
                return False

    return True


def demo_sudoku_9x9(model: UnifiedHRM, device: torch.device):
    """Demo solving 9x9 Sudoku puzzles."""
    print("\n" + "=" * 50)
    print("9x9 SUDOKU DEMO")
    print("=" * 50)

    # Example puzzles (0 = empty)
    examples = [
        # Easy puzzle
        np.array(
            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
        ),
    ]

    for i, puzzle in enumerate(examples):
        print_sudoku_9x9(puzzle, f"Puzzle {i+1}:")

        # Convert to tensor
        puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)

        # Solve
        with torch.no_grad():
            solution = model.solve_sudoku_9x9(puzzle_tensor)

        solution_np = solution[0].cpu().numpy()
        print_sudoku_9x9(solution_np, "Solution:")

        # Validate
        is_valid = validate_sudoku(solution_np, 9)
        print(f"Valid: {'✓' if is_valid else '✗'}")


def demo_sudoku_4x4(model: UnifiedHRM, device: torch.device):
    """Demo solving 4x4 Sudoku puzzles."""
    print("\n" + "=" * 50)
    print("4x4 SUDOKU DEMO")
    print("=" * 50)

    examples = [
        np.array(
            [
                [1, 0, 0, 4],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [4, 0, 0, 1],
            ]
        ),
        np.array(
            [
                [0, 2, 0, 0],
                [0, 0, 3, 1],
                [1, 3, 0, 0],
                [0, 0, 1, 0],
            ]
        ),
    ]

    for i, puzzle in enumerate(examples):
        print_sudoku_4x4(puzzle, f"Puzzle {i+1}:")

        puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)

        with torch.no_grad():
            solution = model.solve_sudoku_4x4(puzzle_tensor)

        solution_np = solution[0].cpu().numpy()
        print_sudoku_4x4(solution_np, "Solution:")

        is_valid = validate_sudoku(solution_np, 4)
        print(f"Valid: {'✓' if is_valid else '✗'}")


def interactive_mode(model: UnifiedHRM, device: torch.device):
    """Interactive puzzle solving mode."""
    print("\n" + "=" * 50)
    print("INTERACTIVE MODE")
    print("=" * 50)
    print("Enter puzzles as comma-separated values (use 0 for empty cells)")
    print("Type 'quit' to exit\n")

    while True:
        print("\nSelect puzzle type:")
        print("  1. 4x4 Sudoku (16 values)")
        print("  2. 9x9 Sudoku (81 values)")
        print("  q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice in ("q", "quit"):
            break

        if choice == "1":
            grid_size = 4
            print(f"\nEnter {grid_size*grid_size} values (0-4, comma-separated):")
        elif choice == "2":
            grid_size = 9
            print(f"\nEnter {grid_size*grid_size} values (0-9, comma-separated):")
        else:
            print("Invalid choice")
            continue

        try:
            values = input("Values: ").strip()
            values = [int(x.strip()) for x in values.split(",")]

            if len(values) != grid_size * grid_size:
                print(f"Expected {grid_size*grid_size} values, got {len(values)}")
                continue

            puzzle = np.array(values).reshape(grid_size, grid_size)

            if grid_size == 4:
                print_sudoku_4x4(puzzle, "Input puzzle:")
                puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)
                with torch.no_grad():
                    solution = model.solve_sudoku_4x4(puzzle_tensor)
                solution_np = solution[0].cpu().numpy()
                print_sudoku_4x4(solution_np, "Solution:")
                is_valid = validate_sudoku(solution_np, 4)
            else:
                print_sudoku_9x9(puzzle, "Input puzzle:")
                puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)
                with torch.no_grad():
                    solution = model.solve_sudoku_9x9(puzzle_tensor)
                solution_np = solution[0].cpu().numpy()
                print_sudoku_9x9(solution_np, "Solution:")
                is_valid = validate_sudoku(solution_np, 9)

            print(f"Valid: {'✓' if is_valid else '✗'}")

        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Demo Unified HRM")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument(
        "--puzzle",
        type=str,
        default="sudoku_9x9",
        choices=["sudoku_4x4", "sudoku_9x9", "all"],
        help="Puzzle type to demo",
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")

    args = parser.parse_args()

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

    print(f"Using device: {device}")

    # Load or create model
    if args.model and Path(args.model).exists():
        model = load_model(args.model, device)
    else:
        print("\nNo trained model provided. Creating new model for demo...")
        print("Note: Predictions will be random until model is trained!\n")
        model = UnifiedHRM()
        model = model.to(device)
        model.eval()

    # Run demo
    if args.interactive:
        interactive_mode(model, device)
    elif args.puzzle == "sudoku_4x4":
        demo_sudoku_4x4(model, device)
    elif args.puzzle == "sudoku_9x9":
        demo_sudoku_9x9(model, device)
    else:  # all
        demo_sudoku_4x4(model, device)
        demo_sudoku_9x9(model, device)


if __name__ == "__main__":
    main()
