"""
Data utilities, validators, and generators for the HRM system.

This module provides data validation, processing utilities, and puzzle
generators used throughout the HRM training pipeline.

Includes:
- Sudoku validation (validator)
- Sudoku puzzle generation (sudoku_generator)
- Weighted maze generation (weighted_maze_generator)
- Dataset I/O in JSON/CSV formats (io)
"""

from hrm.data.sudoku_generator import (
    Difficulty,
    SudokuGenerator,
    create_puzzle,
    generate_full_grid,
    generate_sudoku_dataset,
    save_dataset,
)
from hrm.data.validator import (
    count_filled_cells,
    get_empty_cells,
    get_valid_candidates,
    is_valid_placement,
    is_valid_puzzle,
    is_valid_solution,
)
from hrm.data.io import (
    load_dataset,
    save_dataset as save_dataset_io,
)
from hrm.data.weighted_maze_generator import (
    GOAL,
    MAX_WEIGHT,
    MIN_WEIGHT,
    PATH,
    START,
    WALL,
    WeightedMazeGenerator,
    generate_weighted_maze_dataset,
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
    # Dataset I/O (JSON/CSV)
    "save_dataset_io",
    "load_dataset",
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
