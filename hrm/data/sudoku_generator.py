# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Sudoku puzzle generator that produces unique-solution puzzles with adjustable difficulty.

Supports 4x4 (development) and 9x9 (production) grids.

Output structure for NN training:
- problems: np.array of shape (num_puzzles, grid_size, grid_size)
- solutions: np.array of shape (num_puzzles, grid_size, grid_size)

Usage:
    python sudoku_generator.py --num 100 --size 9 --difficulty medium --seed 42
"""

import argparse
import math
import random
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Difficulty(Enum):
    """Difficulty levels for Sudoku puzzles."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"


class SudokuGenerator:
    """
    Generates Sudoku puzzles with unique solutions and adjustable difficulty.

    Supports 4x4 and 9x9 grids with configurable difficulty levels.
    """

    # Difficulty configuration for 4x4 grids (based on empty cells)
    DIFFICULTY_4X4 = {
        Difficulty.EASY: (6, 8),  # 6-8 empty cells
        Difficulty.MEDIUM: (9, 11),  # 9-11 empty cells
        Difficulty.HARD: (12, 12),  # 12+ empty cells (max 12 for solvability on 4x4)
    }

    # Difficulty configuration for 9x9 grids (based on backtracks during solve)
    DIFFICULTY_9X9_BACKTRACKS = {
        Difficulty.EASY: (0, 4),  # 0-4 backtracks
        Difficulty.MEDIUM: (5, 15),  # 5-15 backtracks
        Difficulty.HARD: (15, 100),  # 15+ backtracks
    }

    def __init__(self, grid_size: int = 9, seed: int | None = None):
        """
        Initialize the Sudoku generator.

        Args:
            grid_size: Size of the grid (4 or 9). Default is 9.
            seed: Random seed for reproducibility. If None, uses system randomness.
        """
        if grid_size not in (4, 9):
            raise ValueError("Grid size must be 4 or 9")

        self.grid_size = grid_size
        self.box_size = int(math.sqrt(grid_size))
        self.seed = seed
        self._rng = random.Random(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set a new random seed for reproducibility."""
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_full_grid(self) -> list[list[int]]:
        """
        Create a valid complete Sudoku grid.

        Returns:
            A completely filled valid Sudoku grid as a 2D list.
        """
        grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self._fill_grid(grid)
        return grid

    def _fill_grid(self, grid: list[list[int]]) -> bool:
        """
        Fill the grid using backtracking with randomized number selection.

        Args:
            grid: The grid to fill (modified in place).

        Returns:
            True if successfully filled, False otherwise.
        """
        # Find next empty cell
        empty_cell = self._find_empty_cell(grid)
        if empty_cell is None:
            return True  # Grid is complete

        row, col = empty_cell

        # Try numbers in random order for variety
        numbers = list(range(1, self.grid_size + 1))
        self._rng.shuffle(numbers)

        for num in numbers:
            if self._is_valid_placement(grid, row, col, num):
                grid[row][col] = num
                if self._fill_grid(grid):
                    return True
                grid[row][col] = 0

        return False

    def _find_empty_cell(self, grid: list[list[int]]) -> tuple[int, int] | None:
        """Find the first empty cell (value 0) in the grid."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if grid[row][col] == 0:
                    return (row, col)
        return None

    def _is_valid_placement(self, grid: list[list[int]], row: int, col: int, num: int) -> bool:
        """
        Check if placing num at (row, col) is valid according to Sudoku rules.

        Args:
            grid: The current grid state.
            row: Row index.
            col: Column index.
            num: Number to place.

        Returns:
            True if placement is valid, False otherwise.
        """
        # Check row
        if num in grid[row]:
            return False

        # Check column
        for r in range(self.grid_size):
            if grid[r][col] == num:
                return False

        # Check box
        box_row_start = (row // self.box_size) * self.box_size
        box_col_start = (col // self.box_size) * self.box_size

        for r in range(box_row_start, box_row_start + self.box_size):
            for c in range(box_col_start, box_col_start + self.box_size):
                if grid[r][c] == num:
                    return False

        return True

    def create_puzzle(
        self, difficulty: Difficulty = Difficulty.MEDIUM
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Create a puzzle by removing cells while maintaining a unique solution.

        Args:
            difficulty: The difficulty level (EASY, MEDIUM, HARD).

        Returns:
            A tuple of (puzzle, solution) where puzzle has empty cells (0s)
            and solution is the complete grid.
        """
        # Generate a complete grid as the solution
        solution = self.generate_full_grid()

        # Create a copy for the puzzle
        puzzle = [row[:] for row in solution]

        if self.grid_size == 4:
            return self._create_puzzle_4x4(puzzle, solution, difficulty)
        else:
            return self._create_puzzle_9x9(puzzle, solution, difficulty)

    def _create_puzzle_4x4(
        self, puzzle: list[list[int]], solution: list[list[int]], difficulty: Difficulty
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Create a 4x4 puzzle based on number of empty cells.

        For 4x4:
        - Easy: 6-8 empty cells
        - Medium: 9-11 empty cells
        - Hard: 12+ empty cells
        """
        min_empty, max_empty = self.DIFFICULTY_4X4[difficulty]
        target_empty = self._rng.randint(min_empty, max_empty)

        # Get all cell positions and shuffle them
        positions = [(r, c) for r in range(4) for c in range(4)]
        self._rng.shuffle(positions)

        empty_count = 0

        for row, col in positions:
            if empty_count >= target_empty:
                break

            # Remember the original value
            original = puzzle[row][col]
            puzzle[row][col] = 0

            # Check if puzzle still has unique solution
            if self._has_unique_solution(puzzle):
                empty_count += 1
            else:
                # Restore the value if removing creates multiple solutions
                puzzle[row][col] = original

        return puzzle, solution

    def _create_puzzle_9x9(
        self, puzzle: list[list[int]], solution: list[list[int]], difficulty: Difficulty
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Create a 9x9 puzzle based on backtrack difficulty.

        For 9x9:
        - Easy: 0-4 backtracks
        - Medium: 5-15 backtracks
        - Hard: 15+ backtracks
        """
        min_backtracks, max_backtracks = self.DIFFICULTY_9X9_BACKTRACKS[difficulty]

        # Get all cell positions and shuffle them
        positions = [(r, c) for r in range(9) for c in range(9)]
        self._rng.shuffle(positions)

        for row, col in positions:
            original = puzzle[row][col]
            puzzle[row][col] = 0

            # Check if puzzle still has unique solution
            if not self._has_unique_solution(puzzle):
                puzzle[row][col] = original
                continue

            # Check backtrack count
            backtracks = self._count_backtracks(puzzle)

            # If we've exceeded the max backtracks for this difficulty,
            # restore the cell and stop removing
            if backtracks > max_backtracks:
                puzzle[row][col] = original
                break

        # Verify we meet minimum backtrack requirement
        # If not, try to adjust (but priority is unique solution)
        current_backtracks = self._count_backtracks(puzzle)

        # If puzzle is too easy, try removing more cells
        if current_backtracks < min_backtracks:
            # Try removing more cells to increase difficulty
            remaining_positions = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0]
            self._rng.shuffle(remaining_positions)

            for row, col in remaining_positions:
                original = puzzle[row][col]
                puzzle[row][col] = 0

                if not self._has_unique_solution(puzzle):
                    puzzle[row][col] = original
                    continue

                backtracks = self._count_backtracks(puzzle)
                if backtracks >= min_backtracks:
                    if backtracks > max_backtracks:
                        puzzle[row][col] = original
                    break

        return puzzle, solution

    def _has_unique_solution(self, puzzle: list[list[int]]) -> bool:
        """
        Check if the puzzle has exactly one solution using backtracking.

        Args:
            puzzle: The puzzle grid with empty cells (0s).

        Returns:
            True if exactly one solution exists, False otherwise.
        """
        # Make a copy to avoid modifying the original
        grid = [row[:] for row in puzzle]
        solutions = [0]

        def solve(grid: list[list[int]]) -> bool:
            """Recursive solver that counts solutions."""
            if solutions[0] > 1:
                return False  # Already found multiple solutions

            empty_cell = self._find_empty_cell(grid)
            if empty_cell is None:
                solutions[0] += 1
                return solutions[0] == 1

            row, col = empty_cell

            for num in range(1, self.grid_size + 1):
                if self._is_valid_placement(grid, row, col, num):
                    grid[row][col] = num
                    solve(grid)
                    if solutions[0] > 1:
                        return False
                    grid[row][col] = 0

            return False

        solve(grid)
        return solutions[0] == 1

    def _count_backtracks(self, puzzle: list[list[int]]) -> int:
        """
        Count the number of backtracks needed to solve the puzzle.

        This is used as a difficulty metric for 9x9 grids.

        Args:
            puzzle: The puzzle grid with empty cells (0s).

        Returns:
            Number of backtracks during solving.
        """
        grid = [row[:] for row in puzzle]
        backtrack_count = [0]

        def solve(grid: list[list[int]]) -> bool:
            empty_cell = self._find_empty_cell(grid)
            if empty_cell is None:
                return True

            row, col = empty_cell

            for num in range(1, self.grid_size + 1):
                if self._is_valid_placement(grid, row, col, num):
                    grid[row][col] = num
                    if solve(grid):
                        return True
                    grid[row][col] = 0
                    backtrack_count[0] += 1

            return False

        solve(grid)
        return backtrack_count[0]

    def is_valid_grid(self, grid: list[list[int]]) -> bool:
        """
        Validate that a grid follows all Sudoku rules.

        Args:
            grid: The grid to validate.

        Returns:
            True if the grid is valid, False otherwise.
        """
        if len(grid) != self.grid_size:
            return False

        for row in grid:
            if len(row) != self.grid_size:
                return False

        # Check all rows
        for row in grid:
            non_zero = [x for x in row if x != 0]
            if len(non_zero) != len(set(non_zero)):
                return False
            if any(x < 0 or x > self.grid_size for x in row):
                return False

        # Check all columns
        for col in range(self.grid_size):
            column = [grid[row][col] for row in range(self.grid_size)]
            non_zero = [x for x in column if x != 0]
            if len(non_zero) != len(set(non_zero)):
                return False

        # Check all boxes
        for box_row in range(self.box_size):
            for box_col in range(self.box_size):
                box = []
                for r in range(box_row * self.box_size, (box_row + 1) * self.box_size):
                    for c in range(box_col * self.box_size, (box_col + 1) * self.box_size):
                        box.append(grid[r][c])
                non_zero = [x for x in box if x != 0]
                if len(non_zero) != len(set(non_zero)):
                    return False

        return True

    def is_complete_grid(self, grid: list[list[int]]) -> bool:
        """
        Check if a grid is completely filled and valid.

        Args:
            grid: The grid to check.

        Returns:
            True if the grid is complete and valid, False otherwise.
        """
        if not self.is_valid_grid(grid):
            return False

        return all(0 not in row for row in grid)

    def count_empty_cells(self, grid: list[list[int]]) -> int:
        """Count the number of empty cells (0s) in a grid."""
        return sum(row.count(0) for row in grid)


def generate_full_grid(grid_size: int = 9, seed: int | None = None) -> list[list[int]]:
    """
    Convenience function to generate a complete valid Sudoku grid.

    Args:
        grid_size: Size of the grid (4 or 9). Default is 9.
        seed: Random seed for reproducibility.

    Returns:
        A completely filled valid Sudoku grid.
    """
    generator = SudokuGenerator(grid_size=grid_size, seed=seed)
    return generator.generate_full_grid()


def create_puzzle(
    difficulty: str = "medium", grid_size: int = 9, seed: int | None = None
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Convenience function to create a Sudoku puzzle.

    Args:
        difficulty: Difficulty level ("easy", "medium", "hard"). Default is "medium".
        grid_size: Size of the grid (4 or 9). Default is 9.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (puzzle, solution).
    """
    difficulty_enum = Difficulty(difficulty.lower())
    generator = SudokuGenerator(grid_size=grid_size, seed=seed)
    return generator.create_puzzle(difficulty_enum)


def generate_sudoku_dataset(
    num_puzzles: int = 100,
    grid_size: int = 9,
    difficulty: str = "medium",
    seed: int | None = None,
    verbose: bool = True,
    ensure_unique: bool = True,
) -> dict[str, Any]:
    """
    Generate a dataset of Sudoku puzzles with solutions for neural network training.

    Args:
        num_puzzles: Number of puzzles to generate.
        grid_size: Size of the grid (4 or 9). Default is 9.
        difficulty: Difficulty level ("easy", "medium", "hard"). Default is "medium".
        seed: Random seed for reproducibility.
        verbose: Print progress information.
        ensure_unique: If True, guarantees all puzzles are unique (default: True).

    Returns:
        dict with keys:
            - 'problems': np.array of shape (num_puzzles, grid_size, grid_size)
            - 'solutions': np.array of shape (num_puzzles, grid_size, grid_size)
            - 'metadata': list of dicts with empty_cells, difficulty, backtracks for each puzzle
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    difficulty_enum = Difficulty(difficulty.lower())
    generator = SudokuGenerator(grid_size=grid_size, seed=seed)

    # For mixed difficulty, cycle through easy/medium/hard equally
    if difficulty_enum == Difficulty.MIXED:
        base_difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    else:
        base_difficulties = None

    problems = []
    solutions = []
    metadata = []
    seen_puzzles: set = set()  # Track unique puzzles by their tuple representation

    generated = 0
    attempts = 0
    max_attempts = num_puzzles * 10  # Prevent infinite loops

    while generated < num_puzzles and attempts < max_attempts:
        attempts += 1

        # Pick difficulty for this puzzle
        if base_difficulties is not None:
            current_diff = base_difficulties[generated % 3]
        else:
            current_diff = difficulty_enum

        puzzle, solution = generator.create_puzzle(current_diff)

        # Check uniqueness if required
        if ensure_unique:
            puzzle_key = tuple(tuple(row) for row in puzzle)
            if puzzle_key in seen_puzzles:
                continue  # Skip duplicate
            seen_puzzles.add(puzzle_key)

        empty_cells = generator.count_empty_cells(puzzle)
        backtracks = generator._count_backtracks(puzzle)

        problems.append(puzzle)
        solutions.append(solution)
        metadata.append(
            {
                "puzzle_id": generated,
                "empty_cells": empty_cells,
                "difficulty": current_diff.value,
                "backtracks": backtracks,
                "grid_size": grid_size,
            }
        )

        generated += 1
        if verbose and generated % 10 == 0:
            print(f"Generated {generated}/{num_puzzles} puzzles...")

    if generated < num_puzzles:
        print(
            f"Warning: Only generated {generated}/{num_puzzles} unique puzzles after {max_attempts} attempts"
        )

    if verbose:
        print(f"Successfully generated {num_puzzles} puzzles")

    return {
        "problems": np.array(problems, dtype=np.int8),
        "solutions": np.array(solutions, dtype=np.int8),
        "metadata": metadata,
    }


def save_dataset(dataset: dict[str, Any], filename_prefix: str, save_format: str = "npz") -> None:
    """
    Save the dataset to files.

    Args:
        dataset: The dataset dict from generate_sudoku_dataset.
        filename_prefix: Prefix for output files.
        save_format: 'npz' (compressed numpy) or 'npy' (separate files).
    """
    # Ensure parent directory exists
    output_path = Path(filename_prefix)
    if output_path.parent and str(output_path.parent) != ".":
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "npz":
        np.savez_compressed(
            f"{filename_prefix}.npz", problems=dataset["problems"], solutions=dataset["solutions"]
        )
        print(f"Saved to {filename_prefix}.npz")

    elif save_format == "npy":
        np.save(f"{filename_prefix}_problems.npy", dataset["problems"])
        np.save(f"{filename_prefix}_solutions.npy", dataset["solutions"])
        print(f"Saved to {filename_prefix}_problems.npy and {filename_prefix}_solutions.npy")

    # Save metadata as text
    with open(f"{filename_prefix}_metadata.txt", "w") as f:
        # Write header with dataset info
        if dataset["metadata"]:
            first_meta = dataset["metadata"][0]
            f.write(
                f"# Sudoku Dataset: {len(dataset['metadata'])} puzzles, "
                f"grid_size={first_meta['grid_size']}, difficulty={first_meta['difficulty']}\n"
            )

        for meta in dataset["metadata"]:
            f.write(
                f"Puzzle {meta['puzzle_id']}: "
                f"empty_cells={meta['empty_cells']}, "
                f"backtracks={meta['backtracks']}, "
                f"difficulty={meta['difficulty']}\n"
            )
    print(f"Metadata saved to {filename_prefix}_metadata.txt")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Sudoku puzzle datasets for neural network training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 medium 9x9 puzzles
  python sudoku_generator.py --num 100 --size 9 --difficulty medium

  # Generate 50 easy 4x4 puzzles with seed for reproducibility
  python sudoku_generator.py --num 50 --size 4 --difficulty easy --seed 42

  # Generate hard puzzles and save to custom location
  python sudoku_generator.py --num 200 --difficulty hard --output ./data/hard_puzzles

Difficulty Levels (4x4):
  easy:   6-8 empty cells
  medium: 9-11 empty cells
  hard:   12+ empty cells

Difficulty Levels (9x9):
  easy:   0-4 backtracks to solve
  medium: 5-15 backtracks to solve
  hard:   15+ backtracks to solve
""",
    )

    parser.add_argument(
        "--num", "-n", type=int, default=100, help="Number of puzzles to generate (default: 100)"
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        choices=[4, 9],
        default=9,
        help="Grid size: 4 (development) or 9 (production) (default: 9)",
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        type=str,
        choices=["easy", "medium", "hard", "mixed"],
        default="medium",
        help='Difficulty level (default: medium). "mixed" generates equal parts easy/medium/hard.',
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output filename prefix (default: sudoku_<size>x<size>_<difficulty>)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["npz", "npy"],
        default="npz",
        help="Output format: npz (compressed) or npy (separate files) (default: npz)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    return parser.parse_args()


def main() -> None:
    """Main entry point for command-line usage."""
    args = parse_args()

    # Set default output filename if not specified
    if args.output is None:
        args.output = f"sudoku_{args.size}x{args.size}_{args.difficulty}"

    print(f"Generating {args.num} {args.difficulty} {args.size}x{args.size} Sudoku puzzles...")
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")

    # Generate the dataset
    dataset = generate_sudoku_dataset(
        num_puzzles=args.num,
        grid_size=args.size,
        difficulty=args.difficulty,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Save the dataset
    save_dataset(dataset, args.output, args.format)

    # Print summary
    print("\nDataset Summary:")
    print(f"  Puzzles: {len(dataset['problems'])}")
    print(f"  Grid size: {args.size}x{args.size}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Problems shape: {dataset['problems'].shape}")
    print(f"  Solutions shape: {dataset['solutions'].shape}")

    # Print difficulty statistics
    empty_cells = [m["empty_cells"] for m in dataset["metadata"]]
    backtracks = [m["backtracks"] for m in dataset["metadata"]]
    print(
        f"  Empty cells: min={min(empty_cells)}, max={max(empty_cells)}, avg={sum(empty_cells)/len(empty_cells):.1f}"
    )
    print(
        f"  Backtracks: min={min(backtracks)}, max={max(backtracks)}, avg={sum(backtracks)/len(backtracks):.1f}"
    )


if __name__ == "__main__":
    main()
