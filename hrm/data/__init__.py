"""
Data utilities, validators, and generators for the HRM system.

This module provides data validation, processing utilities, and puzzle
generators used throughout the HRM training pipeline.

Includes:
- Sudoku validation (validator)
- Sudoku puzzle generation (sudoku_generator)
- Weighted maze generation (weighted_maze_generator)
"""

from hrm.data.validator import (
    count_filled_cells,
    get_empty_cells,
    get_valid_candidates,
    is_valid_placement,
    is_valid_puzzle,
    is_valid_solution,
)

from hrm.data.sudoku_generator import (
    Difficulty,
    SudokuGenerator,
    create_puzzle,
    generate_full_grid,
    generate_sudoku_dataset,
    save_dataset,
)

from hrm.data.weighted_maze_generator import (
    WeightedMazeGenerator,
    generate_weighted_maze_dataset,
    WALL,
    PATH,
    START,
    GOAL,
    MIN_WEIGHT,
    MAX_WEIGHT,
)

__all__ = [
    # Validator
    "count_filled_cells",
    "get_empty_cells",
    "get_valid_candidates",
    "is_valid_placement",
    "is_valid_puzzle",
    "is_valid_solution",
    # Sudoku generator
    "Difficulty",
    "SudokuGenerator",
    "create_puzzle",
    "generate_full_grid",
    "generate_sudoku_dataset",
    "save_dataset",
    # Maze generator
    "WeightedMazeGenerator",
    "generate_weighted_maze_dataset",
    "WALL",
    "PATH",
    "START",
    "GOAL",
    "MIN_WEIGHT",
    "MAX_WEIGHT",
]
