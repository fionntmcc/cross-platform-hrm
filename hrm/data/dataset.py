"""
SudokuDataset: PyTorch Dataset for HRM Training and Validation

Generates Sudoku puzzle samples for the L-Module Only HRM (model_simple.py).
Each sample provides a puzzle grid, a target empty cell to fill, the correct
digit for that cell, and the full solution for reference.

Supports on-the-fly generation and pre-generated caching for reproducible
experiments. Difficulty filtering allows curriculum-style training.

Authors:
    - Kyrylo Kozlovskyi (G00425385)
    - Fionn McCarthy (G00414386)

Reference:
    - HRM_4x4_Simple (model_simple.py): L-Module Only Variant
    - SudokuGenerator (sudoku_generator.py): Puzzle generation
    - validator.py: Constraint checking utilities
"""

import random
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from hrm.data.sudoku_generator import Difficulty, SudokuGenerator


class SudokuDataset(Dataset):
    """
    PyTorch Dataset that yields (puzzle, target_cell, target_digit, solution)
    tuples for training the HRM on single-step Sudoku prediction.

    Each __getitem__ call selects one random empty cell from the puzzle as the
    prediction target. The model learns to predict both *which* cell to fill
    and *what* digit to place there.

    The dataset operates in two modes:
        - **Cached** (default): Puzzles are generated once at init and stored
          in memory. Guarantees identical data across epochs for reproducibility.
        - **On-the-fly**: A new puzzle is generated per access. Useful for
          large-scale training with virtually unlimited variety.

    Args:
        num_puzzles: Number of puzzles in the dataset.
        grid_size: Sudoku grid dimension (4 or 9). Default: 4.
        difficulty: Difficulty level ('easy', 'medium', 'hard') or None for
            mixed difficulties. Default: 'medium'.
        seed: Random seed for reproducibility. Default: None.
        cache: If True, pre-generate and cache all puzzles. Default: True.
        transform: Optional callable applied to each sample dict.

    Shape (per sample):
        - puzzle:       torch.LongTensor  (grid_size, grid_size)
        - solution:     torch.LongTensor  (grid_size, grid_size)
        - target_cell:  int   flat index in [0, num_cells)
        - target_digit: int   0-indexed in [0, num_digits) for CE loss
        - difficulty:   str   'easy' / 'medium' / 'hard'

    Training integration with HRM_4x4_Simple:
        >>> from torch.utils.data import DataLoader
        >>> ds = SudokuDataset(num_puzzles=1000, grid_size=4, seed=42)
        >>> loader = DataLoader(ds, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     outputs = model(batch['puzzle'])
        ...     cell_loss = F.cross_entropy(outputs['cell_logits'], batch['target_cell'])
        ...     digit_loss = F.cross_entropy(outputs['digit_logits'], batch['target_digit'])
    """

    def __init__(
        self,
        num_puzzles: int,
        grid_size: int = 4,
        difficulty: Optional[str] = "medium",
        seed: Optional[int] = None,
        cache: bool = True,
        transform=None,
    ):
        if num_puzzles <= 0:
            raise ValueError(f"num_puzzles must be positive, got {num_puzzles}")
        if grid_size not in (4, 9):
            raise ValueError(f"grid_size must be 4 or 9, got {grid_size}")
        if difficulty is not None and difficulty not in ("easy", "medium", "hard"):
            raise ValueError(
                f"difficulty must be 'easy', 'medium', 'hard', or None; " f"got '{difficulty}'"
            )

        self.num_puzzles = num_puzzles
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.num_digits = grid_size
        self.difficulty = difficulty
        self.seed = seed
        self.cache = cache
        self.transform = transform

        # Internal RNG for target-cell selection (separate from generator RNG)
        self._rng = random.Random(seed)

        # Generator for on-the-fly mode
        self._generator = SudokuGenerator(grid_size=grid_size, seed=seed)

        # Difficulty levels for mixed-difficulty mode
        self._difficulties = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]

        # Pre-generate cache if requested
        self._puzzles: Optional[np.ndarray] = None
        self._solutions: Optional[np.ndarray] = None
        self._metadata: Optional[list[dict[str, Any]]] = None

        if cache:
            self._build_cache()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _build_cache(self) -> None:
        """Pre-generate all puzzles and store them in memory."""
        puzzles = []
        solutions = []
        metadata = []

        gen = SudokuGenerator(grid_size=self.grid_size, seed=self.seed)

        for i in range(self.num_puzzles):
            diff_enum = self._resolve_difficulty(i)
            puzzle, solution = gen.create_puzzle(diff_enum)

            puzzles.append(puzzle)
            solutions.append(solution)
            metadata.append(
                {
                    "puzzle_id": i,
                    "difficulty": diff_enum.value,
                    "grid_size": self.grid_size,
                }
            )

        self._puzzles = np.array(puzzles, dtype=np.int64)
        self._solutions = np.array(solutions, dtype=np.int64)
        self._metadata = metadata

    def _resolve_difficulty(self, index: int) -> Difficulty:
        """Return the Difficulty enum for puzzle at *index*."""
        if self.difficulty is not None:
            return Difficulty(self.difficulty)
        # Mixed: cycle through difficulties deterministically
        return self._difficulties[index % len(self._difficulties)]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_puzzles

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return a single training sample.

        Args:
            index: Dataset index in [0, num_puzzles).

        Returns:
            Dict with keys:
                puzzle:       torch.LongTensor  (grid_size, grid_size)
                solution:     torch.LongTensor  (grid_size, grid_size)
                target_cell:  int   flat index [0, num_cells)
                target_digit: int   0-indexed digit [0, num_digits) for CE
                difficulty:   str   'easy' / 'medium' / 'hard'
        """
        if index < 0 or index >= self.num_puzzles:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_puzzles}")

        if self.cache and self._puzzles is not None:
            puzzle_np = self._puzzles[index]
            solution_np = self._solutions[index]
            difficulty_str = self._metadata[index]["difficulty"]
        else:
            diff_enum = self._resolve_difficulty(index)
            puzzle_list, solution_list = self._generator.create_puzzle(diff_enum)
            puzzle_np = np.array(puzzle_list, dtype=np.int64)
            solution_np = np.array(solution_list, dtype=np.int64)
            difficulty_str = diff_enum.value

        # --- pick a random empty cell as the prediction target ---
        empty_mask = puzzle_np == 0
        empty_indices = np.argwhere(empty_mask)  # shape (N, 2)

        if len(empty_indices) == 0:
            # Degenerate edge case: no empty cells, use cell (0, 0)
            row, col = 0, 0
        else:
            choice = self._rng.randint(0, len(empty_indices) - 1)
            row = int(empty_indices[choice, 0])
            col = int(empty_indices[choice, 1])

        target_cell = row * self.grid_size + col  # flat index for CE
        target_digit = int(solution_np[row, col]) - 1  # 0-indexed for CE

        sample = {
            "puzzle": torch.as_tensor(puzzle_np, dtype=torch.long),
            "solution": torch.as_tensor(solution_np, dtype=torch.long),
            "target_cell": target_cell,
            "target_digit": target_digit,
            "difficulty": difficulty_str,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Batch generation helper
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_size: int,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Generate a fresh batch of samples independent of the cached dataset.

        Useful for on-demand evaluation or curriculum scheduling where you
        want a batch at a specific difficulty without iterating through the
        full dataset with a DataLoader.

        Args:
            batch_size: Number of samples to generate.
            difficulty: Override difficulty for this batch.
                        If None, uses dataset-level difficulty.
            seed: Optional seed for this batch (does not affect dataset RNG).

        Returns:
            Dict with batched tensors:
                puzzles:       (batch_size, grid_size, grid_size) LongTensor
                solutions:     (batch_size, grid_size, grid_size) LongTensor
                target_cells:  (batch_size,) LongTensor
                target_digits: (batch_size,) LongTensor
                difficulties:  list[str]
        """
        diff_str = difficulty or self.difficulty or "medium"
        diff_enum = Difficulty(diff_str)
        gen = SudokuGenerator(grid_size=self.grid_size, seed=seed)
        rng = random.Random(seed)

        puzzles, solutions = [], []
        target_cells, target_digits = [], []
        difficulties = []

        for _ in range(batch_size):
            puzzle, solution = gen.create_puzzle(diff_enum)
            puzzle_np = np.array(puzzle, dtype=np.int64)
            solution_np = np.array(solution, dtype=np.int64)

            empty_indices = np.argwhere(puzzle_np == 0)
            if len(empty_indices) == 0:
                row, col = 0, 0
            else:
                choice = rng.randint(0, len(empty_indices) - 1)
                row = int(empty_indices[choice, 0])
                col = int(empty_indices[choice, 1])

            puzzles.append(puzzle_np)
            solutions.append(solution_np)
            target_cells.append(row * self.grid_size + col)
            target_digits.append(int(solution_np[row, col]) - 1)
            difficulties.append(diff_str)

        return {
            "puzzles": torch.tensor(np.array(puzzles), dtype=torch.long),
            "solutions": torch.tensor(np.array(solutions), dtype=torch.long),
            "target_cells": torch.tensor(target_cells, dtype=torch.long),
            "target_digits": torch.tensor(target_digits, dtype=torch.long),
            "difficulties": difficulties,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_difficulty_subset(self, difficulty: str) -> list[int]:
        """
        Return indices of cached puzzles matching the given difficulty.

        Only works when cache=True. Raises RuntimeError otherwise.

        Args:
            difficulty: 'easy', 'medium', or 'hard'.

        Returns:
            List of dataset indices whose difficulty matches.
        """
        if not self.cache or self._metadata is None:
            raise RuntimeError("get_difficulty_subset requires cache=True")
        return [i for i, m in enumerate(self._metadata) if m["difficulty"] == difficulty]

    def get_statistics(self) -> dict[str, Any]:
        """
        Return summary statistics about the cached dataset.

        Returns:
            Dict with grid_size, num_puzzles, difficulty counts,
            and empty-cell statistics.
        """
        if not self.cache or self._puzzles is None:
            return {
                "grid_size": self.grid_size,
                "num_puzzles": self.num_puzzles,
                "cached": False,
            }

        empty_counts = [int((self._puzzles[i] == 0).sum()) for i in range(len(self))]
        diff_counts: dict[str, int] = {}
        for m in self._metadata:
            d = m["difficulty"]
            diff_counts[d] = diff_counts.get(d, 0) + 1

        return {
            "grid_size": self.grid_size,
            "num_puzzles": self.num_puzzles,
            "cached": True,
            "difficulty_counts": diff_counts,
            "avg_empty_cells": sum(empty_counts) / len(empty_counts),
            "min_empty_cells": min(empty_counts),
            "max_empty_cells": max(empty_counts),
        }

    def __repr__(self) -> str:
        return (
            f"SudokuDataset(num_puzzles={self.num_puzzles}, "
            f"grid_size={self.grid_size}, "
            f"difficulty={self.difficulty!r}, "
            f"cache={self.cache}, "
            f"seed={self.seed})"
        )
