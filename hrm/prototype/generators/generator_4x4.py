# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

import numpy as np
import random
from typing import Tuple


def generate_puzzle(num_clues: int = 10, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 4x4 Sudoku puzzle

    Args:
        num_clues: Number of given cells (10 = easy)
        seed: Random seed for reproducibility

    Returns:
        puzzle: 4x4 array with 0 for empty cells
        solution: 4x4 array with complete solution
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate complete board
    board = np.zeros((4, 4), dtype=int)
    fill_board(board)
    solution = board.copy()

    # Remove cells to create puzzle
    positions = [(i, j) for i in range(4) for j in range(4)]
    random.shuffle(positions)

    for i in range(16 - num_clues):
        row, col = positions[i]
        board[row, col] = 0

    return board, solution


def fill_board(board: np.ndarray) -> bool:
    """Fill board using backtracking"""
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                digits = list(range(1, 5))
                random.shuffle(digits)

                for digit in digits:
                    if is_valid(board, i, j, digit):
                        board[i, j] = digit
                        if fill_board(board):
                            return True
                        board[i, j] = 0
                return False
    return True


def is_valid(board: np.ndarray, row: int, col: int, digit: int) -> bool:
    """Check if placing digit is valid"""
    # Check row, column, box
    if digit in board[row, :]:
        return False
    if digit in board[:, col]:
        return False

    box_row, box_col = 2 * (row // 2), 2 * (col // 2)
    if digit in board[box_row : box_row + 2, box_col : box_col + 2]:
        return False

    return True


def print_puzzle(puzzle: np.ndarray, title: str = "Puzzle"):
    """Print puzzle to console with proper 2x2 box borders
    Args:
        puzzle: (4, 4) array
        title: Title to display
    """
    print(f"\n{title}:")
    print("┌─────┬─────┐")
    for i in range(4):
        row_str = "│ "
        for j in range(4):
            row_str += "·" if puzzle[i, j] == 0 else str(puzzle[i, j])
            row_str += " " if j != 1 else " │ "
        row_str += "│"
        print(row_str)
        if i == 1:
            print("├─────┼─────┤")
    print("└─────┴─────┘")


def generate_dataset(size: int, num_clues: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 'n' puzzles
    Args:
        size: Number of puzzles to generate
        num_clues: Number of given cells in each puzzle
    Returns:
        puzzles: (size, 4, 4) array of puzzles
        solutions: (size, 4, 4) array of solutions
    """
    puzzles = []
    solutions = []

    for _ in range(size):
        puzzle, solution = generate_puzzle(num_clues=num_clues)
        puzzles.append(puzzle)
        solutions.append(solution)

    return np.array(puzzles), np.array(solutions)
