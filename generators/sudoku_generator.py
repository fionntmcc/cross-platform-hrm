"""
Sudoku puzzle generator that produces unique-solution puzzles with adjustable difficulty.

Supports 4x4 (development) and 9x9 (production) grids.
"""

import random
import math
from typing import Optional, Tuple, List
from enum import Enum


class Difficulty(Enum):
    """Difficulty levels for Sudoku puzzles."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SudokuGenerator:
    """
    Generates Sudoku puzzles with unique solutions and adjustable difficulty.
    
    Supports 4x4 and 9x9 grids with configurable difficulty levels.
    """
    
    # Difficulty configuration for 4x4 grids (based on empty cells)
    DIFFICULTY_4X4 = {
        Difficulty.EASY: (6, 8),      # 6-8 empty cells
        Difficulty.MEDIUM: (9, 11),   # 9-11 empty cells
        Difficulty.HARD: (12, 12),    # 12+ empty cells (max 12 for solvability on 4x4)
    }
    
    # Difficulty configuration for 9x9 grids (based on backtracks during solve)
    DIFFICULTY_9X9_BACKTRACKS = {
        Difficulty.EASY: (0, 4),      # 0-4 backtracks
        Difficulty.MEDIUM: (5, 15),   # 5-15 backtracks
        Difficulty.HARD: (15, 100),   # 15+ backtracks
    }
    
    def __init__(self, grid_size: int = 9, seed: Optional[int] = None):
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
    
    def set_seed(self, seed: Optional[int]) -> None:
        """Set a new random seed for reproducibility."""
        self.seed = seed
        self._rng = random.Random(seed)
    
    def generate_full_grid(self) -> List[List[int]]:
        """
        Create a valid complete Sudoku grid.
        
        Returns:
            A completely filled valid Sudoku grid as a 2D list.
        """
        grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        self._fill_grid(grid)
        return grid
    
    def _fill_grid(self, grid: List[List[int]]) -> bool:
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
    
    def _find_empty_cell(self, grid: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find the first empty cell (value 0) in the grid."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if grid[row][col] == 0:
                    return (row, col)
        return None
    
    def _is_valid_placement(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
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
    
    def create_puzzle(self, difficulty: Difficulty = Difficulty.MEDIUM) -> Tuple[List[List[int]], List[List[int]]]:
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
        self, 
        puzzle: List[List[int]], 
        solution: List[List[int]], 
        difficulty: Difficulty
    ) -> Tuple[List[List[int]], List[List[int]]]:
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