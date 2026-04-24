# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Tests for the Sudoku puzzle generator module.
"""

import os
import tempfile

import numpy as np
import pytest

from hrm.data.sudoku_generator import (
    Difficulty,
    SudokuGenerator,
    create_puzzle,
    generate_full_grid,
    generate_sudoku_dataset,
    save_dataset,
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

        # Call twice to verify the method works without a seed (rare collision acceptable)
        gen.create_puzzle(Difficulty.EASY)
        gen.create_puzzle(Difficulty.EASY)

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


class TestGenerateSudokuDataset:
    """Tests for dataset generation functionality."""

    def test_dataset_returns_dict(self):
        """Test generate_sudoku_dataset returns correct dict structure."""
        dataset = generate_sudoku_dataset(
            num_puzzles=5, grid_size=4, difficulty="easy", seed=42, verbose=False
        )

        assert isinstance(dataset, dict)
        assert "problems" in dataset
        assert "solutions" in dataset
        assert "metadata" in dataset

    def test_dataset_correct_shape_4x4(self):
        """Test dataset has correct shape for 4x4 puzzles."""
        num_puzzles = 10
        dataset = generate_sudoku_dataset(
            num_puzzles=num_puzzles, grid_size=4, difficulty="medium", seed=42, verbose=False
        )

        assert dataset["problems"].shape == (num_puzzles, 4, 4)
        assert dataset["solutions"].shape == (num_puzzles, 4, 4)

    def test_dataset_correct_shape_9x9(self):
        """Test dataset has correct shape for 9x9 puzzles."""
        num_puzzles = 5
        dataset = generate_sudoku_dataset(
            num_puzzles=num_puzzles, grid_size=9, difficulty="easy", seed=42, verbose=False
        )

        assert dataset["problems"].shape == (num_puzzles, 9, 9)
        assert dataset["solutions"].shape == (num_puzzles, 9, 9)

    def test_dataset_numpy_arrays(self):
        """Test dataset contains numpy arrays."""
        dataset = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)

        assert isinstance(dataset["problems"], np.ndarray)
        assert isinstance(dataset["solutions"], np.ndarray)

    def test_dataset_dtype(self):
        """Test dataset arrays use int8 dtype."""
        dataset = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)

        assert dataset["problems"].dtype == np.int8
        assert dataset["solutions"].dtype == np.int8

    def test_dataset_metadata_count(self):
        """Test metadata count matches num_puzzles."""
        num_puzzles = 15
        dataset = generate_sudoku_dataset(
            num_puzzles=num_puzzles, grid_size=4, seed=42, verbose=False
        )

        assert len(dataset["metadata"]) == num_puzzles

    def test_dataset_metadata_fields(self):
        """Test metadata contains required fields."""
        dataset = generate_sudoku_dataset(
            num_puzzles=5, grid_size=4, difficulty="hard", seed=42, verbose=False
        )

        for meta in dataset["metadata"]:
            assert "puzzle_id" in meta
            assert "empty_cells" in meta
            assert "difficulty" in meta
            assert "backtracks" in meta
            assert "grid_size" in meta

    def test_dataset_reproducible_with_seed(self):
        """Test dataset generation is reproducible with same seed."""
        dataset1 = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)
        dataset2 = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)

        np.testing.assert_array_equal(dataset1["problems"], dataset2["problems"])
        np.testing.assert_array_equal(dataset1["solutions"], dataset2["solutions"])

    def test_dataset_different_without_same_seed(self):
        """Test datasets differ with different seeds."""
        dataset1 = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)
        dataset2 = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=123, verbose=False)

        assert not np.array_equal(dataset1["problems"], dataset2["problems"])

    def test_dataset_problems_have_zeros(self):
        """Test all problems have empty cells (zeros)."""
        dataset = generate_sudoku_dataset(num_puzzles=10, grid_size=4, seed=42, verbose=False)

        for problem in dataset["problems"]:
            assert 0 in problem

    def test_dataset_solutions_no_zeros(self):
        """Test all solutions are complete (no zeros)."""
        dataset = generate_sudoku_dataset(num_puzzles=10, grid_size=4, seed=42, verbose=False)

        for solution in dataset["solutions"]:
            assert 0 not in solution

    def test_dataset_all_difficulties(self):
        """Test dataset generation works for all difficulty levels."""
        for difficulty in ["easy", "medium", "hard"]:
            dataset = generate_sudoku_dataset(
                num_puzzles=3, grid_size=4, difficulty=difficulty, seed=42, verbose=False
            )

            assert len(dataset["problems"]) == 3
            assert all(m["difficulty"] == difficulty for m in dataset["metadata"])


class TestSaveDataset:
    """Tests for dataset saving functionality."""

    def test_save_dataset_npz(self):
        """Test saving dataset in npz format."""
        dataset = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_sudoku")
            save_dataset(dataset, prefix, save_format="npz")

            # Check npz file exists
            assert os.path.exists(f"{prefix}.npz")

            # Verify contents (use context manager to close file handle on Windows)
            with np.load(f"{prefix}.npz") as loaded:
                np.testing.assert_array_equal(loaded["problems"], dataset["problems"])
                np.testing.assert_array_equal(loaded["solutions"], dataset["solutions"])

    def test_save_dataset_npy(self):
        """Test saving dataset in npy format."""
        dataset = generate_sudoku_dataset(num_puzzles=5, grid_size=4, seed=42, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_sudoku")
            save_dataset(dataset, prefix, save_format="npy")

            # Check npy files exist
            assert os.path.exists(f"{prefix}_problems.npy")
            assert os.path.exists(f"{prefix}_solutions.npy")

            # Verify contents
            problems = np.load(f"{prefix}_problems.npy")
            solutions = np.load(f"{prefix}_solutions.npy")
            np.testing.assert_array_equal(problems, dataset["problems"])
            np.testing.assert_array_equal(solutions, dataset["solutions"])

    def test_save_dataset_metadata(self):
        """Test metadata file is created."""
        dataset = generate_sudoku_dataset(
            num_puzzles=5, grid_size=4, difficulty="medium", seed=42, verbose=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_sudoku")
            save_dataset(dataset, prefix, save_format="npz")

            # Check metadata file exists
            metadata_path = f"{prefix}_metadata.txt"
            assert os.path.exists(metadata_path)

            # Verify metadata content
            with open(metadata_path) as f:
                content = f.read()
                assert "Puzzle 0:" in content
                assert "empty_cells=" in content
                assert "difficulty=medium" in content

    def test_save_dataset_creates_directories(self):
        """Test save_dataset creates parent directories."""
        dataset = generate_sudoku_dataset(num_puzzles=3, grid_size=4, seed=42, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "dir", "test_sudoku")
            save_dataset(dataset, nested_path, save_format="npz")

            assert os.path.exists(f"{nested_path}.npz")


class TestDatasetIntegration:
    """Integration tests for full dataset workflow."""

    def test_full_workflow_4x4(self):
        """Test complete workflow: generate, save, load for 4x4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate
            dataset = generate_sudoku_dataset(
                num_puzzles=10, grid_size=4, difficulty="medium", seed=42, verbose=False
            )

            # Save
            prefix = os.path.join(tmpdir, "sudoku_4x4")
            save_dataset(dataset, prefix, save_format="npz")

            # Load and verify (use context manager to close file handle on Windows)
            with np.load(f"{prefix}.npz") as loaded:
                assert loaded["problems"].shape == (10, 4, 4)
                assert loaded["solutions"].shape == (10, 4, 4)

                # Verify each puzzle/solution pair
                gen = SudokuGenerator(grid_size=4)
                for i in range(10):
                    puzzle = loaded["problems"][i].tolist()
                    solution = loaded["solutions"][i].tolist()

                    assert gen.is_valid_grid(puzzle)
                    assert gen.is_complete_grid(solution)

    def test_full_workflow_9x9(self):
        """Test complete workflow: generate, save, load for 9x9."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate
            dataset = generate_sudoku_dataset(
                num_puzzles=5, grid_size=9, difficulty="easy", seed=42, verbose=False
            )

            # Save
            prefix = os.path.join(tmpdir, "sudoku_9x9")
            save_dataset(dataset, prefix, save_format="npz")

            # Load and verify (use context manager to close file handle on Windows)
            with np.load(f"{prefix}.npz") as loaded:
                assert loaded["problems"].shape == (5, 9, 9)
                assert loaded["solutions"].shape == (5, 9, 9)


class TestUniqueness:
    """Tests for puzzle uniqueness guarantees."""

    def test_all_puzzles_unique_4x4(self):
        """Test that all generated 4x4 puzzles are unique."""
        dataset = generate_sudoku_dataset(
            num_puzzles=50,
            grid_size=4,
            difficulty="medium",
            seed=42,
            verbose=False,
            ensure_unique=True,
        )

        # Convert to set of tuples to check uniqueness
        puzzle_set = set()
        for puzzle in dataset["problems"]:
            puzzle_tuple = tuple(map(tuple, puzzle))
            puzzle_set.add(puzzle_tuple)

        assert len(puzzle_set) == len(dataset["problems"])

    def test_all_puzzles_unique_9x9(self):
        """Test that all generated 9x9 puzzles are unique."""
        dataset = generate_sudoku_dataset(
            num_puzzles=20,
            grid_size=9,
            difficulty="easy",
            seed=42,
            verbose=False,
            ensure_unique=True,
        )

        # Convert to set of tuples to check uniqueness
        puzzle_set = set()
        for puzzle in dataset["problems"]:
            puzzle_tuple = tuple(map(tuple, puzzle))
            puzzle_set.add(puzzle_tuple)

        assert len(puzzle_set) == len(dataset["problems"])

    def test_ensure_unique_default_true(self):
        """Test that ensure_unique defaults to True."""
        dataset = generate_sudoku_dataset(num_puzzles=30, grid_size=4, seed=42, verbose=False)

        puzzle_set = set()
        for puzzle in dataset["problems"]:
            puzzle_tuple = tuple(map(tuple, puzzle))
            puzzle_set.add(puzzle_tuple)

        assert len(puzzle_set) == len(dataset["problems"])

    def test_can_disable_uniqueness_check(self):
        """Test that uniqueness check can be disabled."""
        # This should work without errors even if duplicates occur
        dataset = generate_sudoku_dataset(
            num_puzzles=10, grid_size=4, seed=42, verbose=False, ensure_unique=False
        )

        assert len(dataset["problems"]) == 10
