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
    
    def _create_puzzle_9x9(
        self, 
        puzzle: List[List[int]], 
        solution: List[List[int]], 
        difficulty: Difficulty
    ) -> Tuple[List[List[int]], List[List[int]]]:
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
            remaining_positions = [(r, c) for r in range(9) for c in range(9) 
                                   if puzzle[r][c] != 0]
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
    
    def _has_unique_solution(self, puzzle: List[List[int]]) -> bool:
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
        
        def solve(grid: List[List[int]]) -> bool:
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
    
    def _count_backtracks(self, puzzle: List[List[int]]) -> int:
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
        
        def solve(grid: List[List[int]]) -> bool:
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
    
    def is_valid_grid(self, grid: List[List[int]]) -> bool:
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
    
    def is_complete_grid(self, grid: List[List[int]]) -> bool:
        """
        Check if a grid is completely filled and valid.
        
        Args:
            grid: The grid to check.
            
        Returns:
            True if the grid is complete and valid, False otherwise.
        """
        if not self.is_valid_grid(grid):
            return False
        
        for row in grid:
            if 0 in row:
                return False
        
        return True
    
    def count_empty_cells(self, grid: List[List[int]]) -> int:
        """Count the number of empty cells (0s) in a grid."""
        return sum(row.count(0) for row in grid)


def generate_full_grid(grid_size: int = 9, seed: Optional[int] = None) -> List[List[int]]:
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
    difficulty: str = "medium",
    grid_size: int = 9,
    seed: Optional[int] = None
) -> Tuple[List[List[int]], List[List[int]]]:
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
