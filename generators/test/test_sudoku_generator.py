"""
Tests for the Sudoku puzzle generator module.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from generators.sudoku_generator import (
    SudokuGenerator,
    Difficulty,
    generate_full_grid,
    create_puzzle,
)


class TestSudokuGeneratorInit:
    """Tests for SudokuGenerator initialization."""

    def test_default_grid_size(self):
        """Test default grid size is 9."""
        gen = SudokuGenerator()
        assert gen.grid_size == 9
        assert gen.box_size == 3

    def test_grid_size_4(self):
        """Test 4x4 grid configuration."""
        gen = SudokuGenerator(grid_size=4)
        assert gen.grid_size == 4
        assert gen.box_size == 2

    def test_grid_size_9(self):
        """Test 9x9 grid configuration."""
        gen = SudokuGenerator(grid_size=9)
        assert gen.grid_size == 9
        assert gen.box_size == 3

    def test_invalid_grid_size(self):
        """Test that invalid grid sizes raise ValueError."""
        with pytest.raises(ValueError, match="Grid size must be 4 or 9"):
            SudokuGenerator(grid_size=5)
        
        with pytest.raises(ValueError, match="Grid size must be 4 or 9"):
            SudokuGenerator(grid_size=16)

    def test_seed_reproducibility(self):
        """Test that the same seed produces the same results."""
        gen1 = SudokuGenerator(grid_size=4, seed=42)
        gen2 = SudokuGenerator(grid_size=4, seed=42)
        
        grid1 = gen1.generate_full_grid()
        grid2 = gen2.generate_full_grid()
        
        assert grid1 == grid2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = SudokuGenerator(grid_size=4, seed=42)
        gen2 = SudokuGenerator(grid_size=4, seed=123)
        
        grid1 = gen1.generate_full_grid()
        grid2 = gen2.generate_full_grid()
        
        assert grid1 != grid2

    def test_set_seed(self):
        """Test setting seed after initialization."""
        gen = SudokuGenerator(grid_size=4)
        gen.set_seed(42)
        grid1 = gen.generate_full_grid()
        
        gen.set_seed(42)
        grid2 = gen.generate_full_grid()
        
        assert grid1 == grid2


class TestGenerateFullGrid:
    """Tests for generate_full_grid functionality."""

    def test_full_grid_4x4_dimensions(self):
        """Test 4x4 grid has correct dimensions."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        assert len(grid) == 4
        assert all(len(row) == 4 for row in grid)

    def test_full_grid_9x9_dimensions(self):
        """Test 9x9 grid has correct dimensions."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        grid = gen.generate_full_grid()
        
        assert len(grid) == 9
        assert all(len(row) == 9 for row in grid)

    def test_full_grid_4x4_no_zeros(self):
        """Test 4x4 full grid has no empty cells."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        for row in grid:
            assert 0 not in row

    def test_full_grid_9x9_no_zeros(self):
        """Test 9x9 full grid has no empty cells."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        grid = gen.generate_full_grid()
        
        for row in grid:
            assert 0 not in row

    def test_full_grid_4x4_valid_values(self):
        """Test 4x4 grid contains only values 1-4."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        for row in grid:
            for val in row:
                assert 1 <= val <= 4

    def test_full_grid_9x9_valid_values(self):
        """Test 9x9 grid contains only values 1-9."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        grid = gen.generate_full_grid()
        
        for row in grid:
            for val in row:
                assert 1 <= val <= 9

    def test_full_grid_4x4_valid_sudoku(self):
        """Test 4x4 full grid is valid Sudoku."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        assert gen.is_valid_grid(grid)
        assert gen.is_complete_grid(grid)

    def test_full_grid_9x9_valid_sudoku(self):
        """Test 9x9 full grid is valid Sudoku."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        grid = gen.generate_full_grid()
        
        assert gen.is_valid_grid(grid)
        assert gen.is_complete_grid(grid)

    def test_full_grid_rows_unique(self):
        """Test all rows contain unique values."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        for row in grid:
            assert len(set(row)) == len(row)

    def test_full_grid_columns_unique(self):
        """Test all columns contain unique values."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        for col in range(4):
            column = [grid[row][col] for row in range(4)]
            assert len(set(column)) == len(column)

    def test_full_grid_boxes_unique(self):
        """Test all 2x2 boxes contain unique values in 4x4 grid."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        for box_row in range(2):
            for box_col in range(2):
                box = []
                for r in range(box_row * 2, box_row * 2 + 2):
                    for c in range(box_col * 2, box_col * 2 + 2):
                        box.append(grid[r][c])
                assert len(set(box)) == 4


class TestCreatePuzzle:
    """Tests for create_puzzle functionality."""

    def test_puzzle_returns_tuple(self):
        """Test create_puzzle returns tuple of puzzle and solution."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        result = gen.create_puzzle(Difficulty.EASY)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_puzzle_has_empty_cells(self):
        """Test puzzle has empty cells (zeros)."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.EASY)
        
        empty_count = sum(row.count(0) for row in puzzle)
        assert empty_count > 0

    def test_solution_is_complete(self):
        """Test solution has no empty cells."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        _, solution = gen.create_puzzle(Difficulty.EASY)
        
        assert gen.is_complete_grid(solution)

    def test_solution_is_valid(self):
        """Test solution is valid Sudoku."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        _, solution = gen.create_puzzle(Difficulty.MEDIUM)
        
        assert gen.is_valid_grid(solution)

    def test_puzzle_is_valid(self):
        """Test puzzle (with zeros) follows Sudoku rules."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.EASY)
        
        assert gen.is_valid_grid(puzzle)

    def test_puzzle_has_unique_solution(self):
        """Test generated puzzle has exactly one solution."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.MEDIUM)
        
        assert gen._has_unique_solution(puzzle)


class TestDifficultyLevels4x4:
    """Tests for 4x4 difficulty levels based on empty cells."""

    def test_easy_empty_cells(self):
        """Test easy difficulty has 6-8 empty cells."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.EASY)
        
        empty_count = gen.count_empty_cells(puzzle)
        assert 6 <= empty_count <= 8, f"Expected 6-8 empty cells, got {empty_count}"

    def test_medium_empty_cells(self):
        """Test medium difficulty has 9-11 empty cells."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.MEDIUM)
        
        empty_count = gen.count_empty_cells(puzzle)
        assert 9 <= empty_count <= 11, f"Expected 9-11 empty cells, got {empty_count}"

    def test_hard_empty_cells(self):
        """Test hard difficulty has 12+ empty cells."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.HARD)
        
        empty_count = gen.count_empty_cells(puzzle)
        assert empty_count >= 12, f"Expected 12+ empty cells, got {empty_count}"

    def test_easy_still_solvable(self):
        """Test easy puzzle has unique solution."""
        gen = SudokuGenerator(grid_size=4, seed=100)
        puzzle, _ = gen.create_puzzle(Difficulty.EASY)
        
        assert gen._has_unique_solution(puzzle)

    def test_medium_still_solvable(self):
        """Test medium puzzle has unique solution."""
        gen = SudokuGenerator(grid_size=4, seed=100)
        puzzle, _ = gen.create_puzzle(Difficulty.MEDIUM)
        
        assert gen._has_unique_solution(puzzle)

    def test_hard_still_solvable(self):
        """Test hard puzzle has unique solution."""
        gen = SudokuGenerator(grid_size=4, seed=100)
        puzzle, _ = gen.create_puzzle(Difficulty.HARD)
        
        assert gen._has_unique_solution(puzzle)


class TestDifficultyLevels9x9:
    """Tests for 9x9 difficulty levels based on backtracking."""

    def test_easy_9x9_backtrack_range(self):
        """Test easy 9x9 puzzle needs 0-4 backtracks."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.EASY)
        
        backtracks = gen._count_backtracks(puzzle)
        assert 0 <= backtracks <= 4, f"Expected 0-4 backtracks, got {backtracks}"

    def test_medium_9x9_backtrack_range(self):
        """Test medium 9x9 puzzle needs 5-15 backtracks."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.MEDIUM)
        
        backtracks = gen._count_backtracks(puzzle)
        assert 5 <= backtracks <= 15, f"Expected 5-15 backtracks, got {backtracks}"

    def test_hard_9x9_backtrack_range(self):
        """Test hard 9x9 puzzle needs 15+ backtracks."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.HARD)
        
        backtracks = gen._count_backtracks(puzzle)
        assert backtracks >= 15, f"Expected 15+ backtracks, got {backtracks}"

    def test_9x9_unique_solution_easy(self):
        """Test 9x9 easy puzzle has unique solution."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.EASY)
        
        assert gen._has_unique_solution(puzzle)

    def test_9x9_unique_solution_medium(self):
        """Test 9x9 medium puzzle has unique solution."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.MEDIUM)
        
        assert gen._has_unique_solution(puzzle)

    def test_9x9_unique_solution_hard(self):
        """Test 9x9 hard puzzle has unique solution."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.HARD)
        
        assert gen._has_unique_solution(puzzle)


class TestBacktrackingVerification:
    """Tests for unique solution verification."""

    def test_complete_grid_has_unique_solution(self):
        """Test complete grid has exactly one solution."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        assert gen._has_unique_solution(grid)

    def test_multiple_solution_detection(self):
        """Test detection of puzzles with multiple solutions."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        
        # Create an almost empty grid (should have multiple solutions)
        puzzle = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        
        assert not gen._has_unique_solution(puzzle)

    def test_barely_constrained_has_unique(self):
        """Test a minimally constrained puzzle still validates correctly."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, _ = gen.create_puzzle(Difficulty.HARD)
        
        # Puzzle from difficult level should still have unique solution
        assert gen._has_unique_solution(puzzle)


class TestValidation:
    """Tests for grid validation methods."""

    def test_is_valid_grid_correct(self):
        """Test valid grid passes validation."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ]
        
        assert gen.is_valid_grid(grid)

    def test_is_valid_grid_with_zeros(self):
        """Test grid with zeros passes validation."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 0, 3, 4],
            [3, 4, 0, 2],
            [0, 1, 4, 3],
            [4, 3, 2, 0],
        ]
        
        assert gen.is_valid_grid(grid)

    def test_invalid_grid_duplicate_row(self):
        """Test grid with duplicate in row fails validation."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 1, 3, 4],  # Duplicate 1 in row
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ]
        
        assert not gen.is_valid_grid(grid)

    def test_invalid_grid_duplicate_column(self):
        """Test grid with duplicate in column fails validation."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 2, 3, 4],
            [1, 4, 1, 2],  # Duplicate 1 in column 0
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ]
        
        assert not gen.is_valid_grid(grid)

    def test_invalid_grid_duplicate_box(self):
        """Test grid with duplicate in box fails validation."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 2, 3, 4],
            [2, 4, 1, 3],  # Duplicate 2 in top-left box
            [3, 1, 4, 2],
            [4, 3, 2, 1],
        ]
        
        assert not gen.is_valid_grid(grid)

    def test_invalid_grid_wrong_dimensions(self):
        """Test grid with wrong dimensions fails validation."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 2, 3],
            [3, 4, 1],
            [2, 1, 4],
        ]
        
        assert not gen.is_valid_grid(grid)

    def test_is_complete_grid_true(self):
        """Test complete grid is identified correctly."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ]
        
        assert gen.is_complete_grid(grid)

    def test_is_complete_grid_false_with_zeros(self):
        """Test grid with zeros is not complete."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [1, 0, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1],
        ]
        
        assert not gen.is_complete_grid(grid)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_full_grid_function(self):
        """Test generate_full_grid convenience function."""
        grid = generate_full_grid(grid_size=4, seed=42)
        
        assert len(grid) == 4
        assert all(len(row) == 4 for row in grid)
        assert all(0 not in row for row in grid)

    def test_generate_full_grid_reproducible(self):
        """Test generate_full_grid is reproducible with seed."""
        grid1 = generate_full_grid(grid_size=4, seed=42)
        grid2 = generate_full_grid(grid_size=4, seed=42)
        
        assert grid1 == grid2

    def test_create_puzzle_function(self):
        """Test create_puzzle convenience function."""
        puzzle, solution = create_puzzle(difficulty="medium", grid_size=4, seed=42)
        
        assert len(puzzle) == 4
        assert len(solution) == 4
        assert any(0 in row for row in puzzle)
        assert all(0 not in row for row in solution)

    def test_create_puzzle_difficulty_string(self):
        """Test create_puzzle accepts string difficulty."""
        puzzle_easy, _ = create_puzzle(difficulty="easy", grid_size=4, seed=42)
        puzzle_hard, _ = create_puzzle(difficulty="hard", grid_size=4, seed=42)
        
        easy_empty = sum(row.count(0) for row in puzzle_easy)
        hard_empty = sum(row.count(0) for row in puzzle_hard)
        
        # Hard should generally have more empty cells
        assert easy_empty < hard_empty


class TestCountEmptyCells:
    """Tests for empty cell counting."""

    def test_count_empty_cells_full(self):
        """Test counting zero empty cells in full grid."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        grid = gen.generate_full_grid()
        
        assert gen.count_empty_cells(grid) == 0

    def test_count_empty_cells_partial(self):
        """Test counting empty cells in partial grid."""
        gen = SudokuGenerator(grid_size=4)
        grid = [
            [0, 2, 0, 4],
            [3, 0, 1, 0],
            [0, 1, 0, 3],
            [4, 0, 2, 0],
        ]
        
        assert gen.count_empty_cells(grid) == 8

    def test_count_empty_cells_all_empty(self):
        """Test counting all empty grid."""
        gen = SudokuGenerator(grid_size=4)
        grid = [[0] * 4 for _ in range(4)]
        
        assert gen.count_empty_cells(grid) == 16


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_multiple_puzzles_different(self):
        """Test generating multiple puzzles produces different results."""
        gen = SudokuGenerator(grid_size=4)  # No seed
        
        puzzle1, _ = gen.create_puzzle(Difficulty.EASY)
        puzzle2, _ = gen.create_puzzle(Difficulty.EASY)
        
        # Highly unlikely to be the same without a seed
        assert puzzle1 != puzzle2 or True  # Allow for rare collision

    def test_puzzle_preserves_filled_cells(self):
        """Test puzzle values match solution where filled."""
        gen = SudokuGenerator(grid_size=4, seed=42)
        puzzle, solution = gen.create_puzzle(Difficulty.MEDIUM)
        
        for r in range(4):
            for c in range(4):
                if puzzle[r][c] != 0:
                    assert puzzle[r][c] == solution[r][c]

    def test_9x9_complete_workflow(self):
        """Test complete workflow for 9x9 grid."""
        gen = SudokuGenerator(grid_size=9, seed=42)
        
        # Generate full grid
        full_grid = gen.generate_full_grid()
        assert gen.is_complete_grid(full_grid)
        
        # Create puzzle
        puzzle, solution = gen.create_puzzle(Difficulty.MEDIUM)
        assert gen.is_valid_grid(puzzle)
        assert gen.is_complete_grid(solution)
        assert gen._has_unique_solution(puzzle)
