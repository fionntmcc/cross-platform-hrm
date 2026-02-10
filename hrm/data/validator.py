"""
Sudoku Validation Utilities

Provides constraint-checking and solution validation for Sudoku grids.
Supports both 4x4 (development) and 9x9 (production) grid sizes.

Used by the training pipeline to verify generated puzzles and to
evaluate model predictions against ground-truth solutions.

Authors:
    - Kyrylo Kozlovskyi (G00425385)
    - Fionn McCarthy (G00414386)

Reference:
    - HRM_4x4_Simple (model_simple.py): L-Module Only Variant
    - SudokuGenerator (sudoku_generator.py): Puzzle generation
"""

import math
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Supported grid sizes and their box dimensions
# ---------------------------------------------------------------------------
SUPPORTED_SIZES = {4, 9}


def _validate_grid_input(
    grid: Union[list[list[int]], np.ndarray],
) -> tuple[np.ndarray, int, int]:
    """
    Normalise input to a NumPy array and extract grid metadata.

    Args:
        grid: 2-D Sudoku grid as a nested list or NumPy array.

    Returns:
        Tuple of (grid_array, grid_size, box_size).

    Raises:
        ValueError: If the grid is not square or has an unsupported size.
        TypeError:  If the grid is not a list or ndarray.
    """
    if isinstance(grid, list):
        grid = np.array(grid, dtype=int)
    elif not isinstance(grid, np.ndarray):
        raise TypeError(f"Expected list or np.ndarray, got {type(grid).__name__}")

    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2-D, got {grid.ndim}-D array")

    rows, cols = grid.shape
    if rows != cols:
        raise ValueError(f"Grid must be square, got shape {grid.shape}")

    grid_size = rows
    if grid_size not in SUPPORTED_SIZES:
        raise ValueError(
            f"Grid size {grid_size} not supported. " f"Supported sizes: {sorted(SUPPORTED_SIZES)}"
        )

    box_size = int(math.sqrt(grid_size))
    return grid, grid_size, box_size


# ---------------------------------------------------------------------------
# Core validation helpers
# ---------------------------------------------------------------------------


def is_valid_placement(
    grid: Union[list[list[int]], np.ndarray],
    row: int,
    col: int,
    num: int,
) -> bool:
    """
    Check whether placing *num* at (row, col) violates Sudoku constraints.

    The placement is valid if *num* does not already appear in:
      1. The same row,
      2. The same column, or
      3. The same box (2x2 for 4x4 grids, 3x3 for 9x9 grids).

    The cell at (row, col) is treated as empty regardless of its current
    value so that this function can also verify existing placements.

    Args:
        grid: 2-D Sudoku grid (0 = empty cell).
        row:  Row index (0-based).
        col:  Column index (0-based).
        num:  Digit to place (1-grid_size).

    Returns:
        True if the placement is valid, False otherwise.

    Raises:
        ValueError: If row/col is out of bounds or num is out of range.

    Example:
        >>> grid = [[1, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0]]
        >>> is_valid_placement(grid, 0, 1, 1)
        False
        >>> is_valid_placement(grid, 0, 1, 2)
        True
    """
    grid, grid_size, box_size = _validate_grid_input(grid)

    # --- bounds / range checks ---
    if not (0 <= row < grid_size):
        raise ValueError(f"row must be in [0, {grid_size}), got {row}")
    if not (0 <= col < grid_size):
        raise ValueError(f"col must be in [0, {grid_size}), got {col}")
    if not (1 <= num <= grid_size):
        raise ValueError(f"num must be in [1, {grid_size}], got {num}")

    # --- row constraint ---
    for c in range(grid_size):
        if c != col and grid[row, c] == num:
            return False

    # --- column constraint ---
    for r in range(grid_size):
        if r != row and grid[r, col] == num:
            return False

    # --- box constraint ---
    box_row_start = (row // box_size) * box_size
    box_col_start = (col // box_size) * box_size
    for r in range(box_row_start, box_row_start + box_size):
        for c in range(box_col_start, box_col_start + box_size):
            if (r, c) != (row, col) and grid[r, c] == num:
                return False

    return True


def is_valid_solution(
    grid: Union[list[list[int]], np.ndarray],
) -> bool:
    """
    Verify that a completed Sudoku grid is a valid solution.

    A valid solution satisfies ALL of the following:
      1. No empty cells (zeros).
      2. Every row contains each digit exactly once.
      3. Every column contains each digit exactly once.
      4. Every box contains each digit exactly once.

    Args:
        grid: 2-D Sudoku grid (should contain no zeros).

    Returns:
        True if the grid is a valid, complete solution.

    Example:
        >>> solution = [[4, 3, 1, 2],
        ...             [1, 2, 4, 3],
        ...             [2, 4, 3, 1],
        ...             [3, 1, 2, 4]]
        >>> is_valid_solution(solution)
        True
    """
    grid, grid_size, box_size = _validate_grid_input(grid)

    expected = set(range(1, grid_size + 1))

    # 1. No empty cells
    if 0 in grid:
        return False

    # 2. All values within valid range
    if grid.min() < 1 or grid.max() > grid_size:
        return False

    # 3. Row uniqueness
    for r in range(grid_size):
        if set(grid[r].tolist()) != expected:
            return False

    # 4. Column uniqueness
    for c in range(grid_size):
        if set(grid[:, c].tolist()) != expected:
            return False

    # 5. Box uniqueness
    for box_r in range(box_size):
        for box_c in range(box_size):
            r_start = box_r * box_size
            c_start = box_c * box_size
            block = grid[r_start : r_start + box_size, c_start : c_start + box_size]
            if set(block.flatten().tolist()) != expected:
                return False

    return True


def get_empty_cells(
    grid: Union[list[list[int]], np.ndarray],
) -> list[tuple[int, int]]:
    """
    Return the (row, col) coordinates of every empty cell (value == 0).

    Results are returned in row-major order (top-left to bottom-right).

    Args:
        grid: 2-D Sudoku grid where 0 denotes an empty cell.

    Returns:
        List of (row, col) tuples for empty cells.

    Example:
        >>> grid = [[1, 0, 3, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 4]]
        >>> get_empty_cells(grid)
        [(0, 1), (0, 3), (1, 0), (1, 1), ...]
    """
    grid, grid_size, _ = _validate_grid_input(grid)

    empty: list[tuple[int, int]] = []
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] == 0:
                empty.append((r, c))
    return empty


# ---------------------------------------------------------------------------
# Extended utilities (useful for training & evaluation)
# ---------------------------------------------------------------------------


def is_valid_puzzle(
    grid: Union[list[list[int]], np.ndarray],
) -> bool:
    """
    Check that a partially-filled puzzle has no constraint violations.

    Unlike *is_valid_solution*, empty cells (zeros) are permitted.
    Every non-zero value must satisfy row, column, and box uniqueness.

    Args:
        grid: 2-D Sudoku grid (0 = empty).

    Returns:
        True if the puzzle state has no violations.
    """
    grid, grid_size, box_size = _validate_grid_input(grid)

    # Values must be in [0, grid_size]
    if grid.min() < 0 or grid.max() > grid_size:
        return False

    # Check each non-zero cell for constraint violations
    for r in range(grid_size):
        for c in range(grid_size):
            val = int(grid[r, c])
            if val == 0:
                continue
            if not is_valid_placement(grid, r, c, val):
                return False

    return True


def count_filled_cells(
    grid: Union[list[list[int]], np.ndarray],
) -> int:
    """
    Return the number of non-empty cells in the grid.

    Args:
        grid: 2-D Sudoku grid.

    Returns:
        Integer count of cells with value != 0.
    """
    grid, _, _ = _validate_grid_input(grid)
    return int(np.count_nonzero(grid))


def get_valid_candidates(
    grid: Union[list[list[int]], np.ndarray],
    row: int,
    col: int,
) -> list[int]:
    """
    Return the list of digits that can legally be placed at (row, col).

    Args:
        grid: 2-D Sudoku grid.
        row:  Row index (0-based).
        col:  Column index (0-based).

    Returns:
        Sorted list of valid digit candidates.

    Example:
        >>> grid = [[1, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0]]
        >>> get_valid_candidates(grid, 0, 1)
        [2, 3, 4]
    """
    grid, grid_size, _ = _validate_grid_input(grid)

    return [num for num in range(1, grid_size + 1) if is_valid_placement(grid, row, col, num)]
