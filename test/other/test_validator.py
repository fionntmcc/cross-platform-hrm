"""
Unit Tests for Sudoku Validation Utilities

Comprehensive tests covering:
    - is_valid_placement: row, column, box constraints
    - is_valid_solution: complete solution validation
    - get_empty_cells: empty cell detection
    - is_valid_puzzle: partial puzzle validation
    - count_filled_cells: cell counting
    - get_valid_candidates: candidate enumeration
    - Edge cases: empty grids, full grids, invalid inputs, both 4x4 and 9x9

Author: Kyrylo Kozlovskyi (G00425385)
"""

import numpy as np
import pytest

from hrm.data.validator import (
    _validate_grid_input,
    count_filled_cells,
    get_empty_cells,
    get_valid_candidates,
    is_valid_placement,
    is_valid_puzzle,
    is_valid_solution,
)

# ===================================================================
# Test fixtures
# ===================================================================

# --- 4x4 fixtures ---

VALID_4X4_SOLUTION = [
    [4, 3, 1, 2],
    [1, 2, 4, 3],
    [2, 4, 3, 1],
    [3, 1, 2, 4],
]

VALID_4X4_PUZZLE = [
    [4, 0, 0, 2],
    [0, 2, 4, 0],
    [2, 4, 0, 0],
    [0, 0, 2, 4],
]

EMPTY_4X4 = [[0] * 4 for _ in range(4)]

# --- 9x9 fixtures ---

VALID_9X9_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

VALID_9X9_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

EMPTY_9X9 = [[0] * 9 for _ in range(9)]


# ===================================================================
# _validate_grid_input
# ===================================================================


class TestValidateGridInput:
    """Tests for internal grid input validation."""

    def test_list_input_converted(self):
        _grid, size, box = _validate_grid_input(VALID_4X4_SOLUTION)
        assert size == 4
        assert box == 2

    def test_ndarray_input_accepted(self):
        arr = np.array(VALID_9X9_SOLUTION)
        _grid, size, box = _validate_grid_input(arr)
        assert size == 9
        assert box == 3

    def test_unsupported_size_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            _validate_grid_input([[0] * 5 for _ in range(5)])

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            _validate_grid_input(np.zeros((4, 5), dtype=int))

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            _validate_grid_input("not a grid")

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            _validate_grid_input(np.zeros(16, dtype=int))


# ===================================================================
# is_valid_placement - 4x4
# ===================================================================


class TestIsValidPlacement4x4:
    """Placement constraint checks on 4x4 grids."""

    def test_valid_placement_empty_grid(self):
        assert is_valid_placement(EMPTY_4X4, 0, 0, 1) is True

    def test_row_conflict(self):
        grid = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert is_valid_placement(grid, 0, 1, 1) is False

    def test_column_conflict(self):
        grid = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert is_valid_placement(grid, 1, 0, 1) is False

    def test_box_conflict(self):
        grid = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # (0,0) and (1,1) share the top-left 2x2 box
        assert is_valid_placement(grid, 1, 1, 1) is False

    def test_no_conflict(self):
        grid = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # Place 1 in bottom-right box (no row/col/box overlap)
        assert is_valid_placement(grid, 2, 2, 1) is True

    def test_verify_existing_placement(self):
        """A cell's own value should not conflict with itself."""
        sol = np.array(VALID_4X4_SOLUTION)
        assert is_valid_placement(sol, 0, 0, int(sol[0, 0])) is True

    def test_all_digits_on_empty_grid(self):
        for num in range(1, 5):
            assert is_valid_placement(EMPTY_4X4, 0, 0, num) is True

    def test_out_of_bounds_row_raises(self):
        with pytest.raises(ValueError, match="row"):
            is_valid_placement(EMPTY_4X4, 4, 0, 1)

    def test_out_of_bounds_col_raises(self):
        with pytest.raises(ValueError, match="col"):
            is_valid_placement(EMPTY_4X4, 0, -1, 1)

    def test_num_zero_raises(self):
        with pytest.raises(ValueError, match="num"):
            is_valid_placement(EMPTY_4X4, 0, 0, 0)

    def test_num_too_large_raises(self):
        with pytest.raises(ValueError, match="num"):
            is_valid_placement(EMPTY_4X4, 0, 0, 5)


# ===================================================================
# is_valid_placement - 9x9
# ===================================================================


class TestIsValidPlacement9x9:
    """Placement constraint checks on 9x9 grids."""

    def test_valid_on_empty(self):
        assert is_valid_placement(EMPTY_9X9, 4, 4, 5) is True

    def test_row_conflict_9x9(self):
        grid = np.array(VALID_9X9_PUZZLE)
        # Row 0 has 5 at (0,0); placing 5 elsewhere in row 0 should fail
        assert is_valid_placement(grid, 0, 2, 5) is False

    def test_column_conflict_9x9(self):
        grid = np.array(VALID_9X9_PUZZLE)
        # Col 0 has 5 at (0,0); placing 5 lower in col 0 should fail
        assert is_valid_placement(grid, 2, 0, 5) is False

    def test_box_conflict_9x9(self):
        grid = np.array(VALID_9X9_PUZZLE)
        # Top-left 3x3 box has 5 at (0,0); (1,1) is in same box
        assert is_valid_placement(grid, 1, 1, 5) is False

    def test_valid_placement_in_puzzle(self):
        grid = np.array(VALID_9X9_PUZZLE)
        # (0,2) is empty, solution has 4 there
        assert is_valid_placement(grid, 0, 2, 4) is True

    def test_num_9_valid(self):
        assert is_valid_placement(EMPTY_9X9, 0, 0, 9) is True

    def test_num_10_raises(self):
        with pytest.raises(ValueError, match="num"):
            is_valid_placement(EMPTY_9X9, 0, 0, 10)


# ===================================================================
# is_valid_solution
# ===================================================================


class TestIsValidSolution:
    """Complete solution validation."""

    def test_valid_4x4_solution(self):
        assert is_valid_solution(VALID_4X4_SOLUTION) is True

    def test_valid_9x9_solution(self):
        assert is_valid_solution(VALID_9X9_SOLUTION) is True

    def test_incomplete_grid_fails(self):
        assert is_valid_solution(VALID_4X4_PUZZLE) is False

    def test_empty_grid_fails(self):
        assert is_valid_solution(EMPTY_4X4) is False

    def test_duplicate_in_row(self):
        bad = [
            [1, 1, 3, 4],
            [3, 4, 1, 2],
            [2, 3, 4, 1],
            [4, 2, 1, 3],
        ]
        assert is_valid_solution(bad) is False

    def test_duplicate_in_column(self):
        bad = [
            [1, 2, 3, 4],
            [1, 4, 2, 3],  # col 0 has duplicate 1
            [3, 1, 4, 2],
            [4, 3, 1, 2],
        ]
        assert is_valid_solution(bad) is False

    def test_duplicate_in_box(self):
        # Rows and columns correct, but top-left box has duplicate
        bad = [
            [1, 2, 3, 4],
            [2, 1, 4, 3],  # top-left box: {1,2,2,1} — duplicate
            [3, 4, 1, 2],
            [4, 3, 2, 1],
        ]
        assert is_valid_solution(bad) is False

    def test_out_of_range_value(self):
        bad = np.array(VALID_4X4_SOLUTION, dtype=int)
        bad[0, 0] = 5  # out of range for 4x4
        assert is_valid_solution(bad) is False

    def test_negative_value(self):
        bad = np.array(VALID_4X4_SOLUTION, dtype=int)
        bad[0, 0] = -1
        assert is_valid_solution(bad) is False

    def test_numpy_input(self):
        arr = np.array(VALID_4X4_SOLUTION)
        assert is_valid_solution(arr) is True

    def test_valid_9x9_as_numpy(self):
        arr = np.array(VALID_9X9_SOLUTION)
        assert is_valid_solution(arr) is True


# ===================================================================
# get_empty_cells
# ===================================================================


class TestGetEmptyCells:
    """Empty cell detection."""

    def test_full_grid_no_empties(self):
        assert get_empty_cells(VALID_4X4_SOLUTION) == []

    def test_empty_4x4_grid(self):
        empties = get_empty_cells(EMPTY_4X4)
        assert len(empties) == 16
        assert empties[0] == (0, 0)
        assert empties[-1] == (3, 3)

    def test_empty_9x9_grid(self):
        empties = get_empty_cells(EMPTY_9X9)
        assert len(empties) == 81

    def test_partial_4x4_puzzle(self):
        empties = get_empty_cells(VALID_4X4_PUZZLE)
        # Count zeros in the fixture
        expected = sum(1 for row in VALID_4X4_PUZZLE for v in row if v == 0)
        assert len(empties) == expected

    def test_row_major_order(self):
        grid = [[0, 1, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        empties = get_empty_cells(grid)
        # First empty should be (0,0), second (0,2)
        assert empties[0] == (0, 0)
        assert empties[1] == (0, 2)

    def test_single_empty_cell(self):
        almost_full = [row[:] for row in VALID_4X4_SOLUTION]
        almost_full[2][3] = 0
        assert get_empty_cells(almost_full) == [(2, 3)]

    def test_numpy_input(self):
        arr = np.array(VALID_9X9_PUZZLE)
        empties = get_empty_cells(arr)
        assert all(isinstance(e, tuple) and len(e) == 2 for e in empties)


# ===================================================================
# is_valid_puzzle (partial grid)
# ===================================================================


class TestIsValidPuzzle:
    """Partial puzzle constraint validation."""

    def test_valid_4x4_puzzle(self):
        assert is_valid_puzzle(VALID_4X4_PUZZLE) is True

    def test_valid_9x9_puzzle(self):
        assert is_valid_puzzle(VALID_9X9_PUZZLE) is True

    def test_empty_grid_valid(self):
        assert is_valid_puzzle(EMPTY_4X4) is True

    def test_full_solution_also_valid_puzzle(self):
        assert is_valid_puzzle(VALID_4X4_SOLUTION) is True

    def test_invalid_puzzle_duplicate_row(self):
        bad = [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert is_valid_puzzle(bad) is False

    def test_invalid_puzzle_duplicate_col(self):
        bad = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert is_valid_puzzle(bad) is False

    def test_invalid_puzzle_duplicate_box(self):
        bad = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert is_valid_puzzle(bad) is False

    def test_out_of_range_value(self):
        bad = [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert is_valid_puzzle(bad) is False

    def test_negative_value(self):
        bad = np.zeros((4, 4), dtype=int)
        bad[0, 0] = -1
        assert is_valid_puzzle(bad) is False


# ===================================================================
# count_filled_cells
# ===================================================================


class TestCountFilledCells:
    """Cell counting utility."""

    def test_full_4x4(self):
        assert count_filled_cells(VALID_4X4_SOLUTION) == 16

    def test_full_9x9(self):
        assert count_filled_cells(VALID_9X9_SOLUTION) == 81

    def test_empty_4x4(self):
        assert count_filled_cells(EMPTY_4X4) == 0

    def test_partial_puzzle(self):
        filled = sum(1 for row in VALID_4X4_PUZZLE for v in row if v != 0)
        assert count_filled_cells(VALID_4X4_PUZZLE) == filled


# ===================================================================
# get_valid_candidates
# ===================================================================


class TestGetValidCandidates:
    """Candidate digit enumeration."""

    def test_empty_grid_all_candidates(self):
        candidates = get_valid_candidates(EMPTY_4X4, 0, 0)
        assert candidates == [1, 2, 3, 4]

    def test_empty_9x9_all_candidates(self):
        candidates = get_valid_candidates(EMPTY_9X9, 0, 0)
        assert candidates == list(range(1, 10))

    def test_single_candidate(self):
        # Only one value missing from solution
        almost = [row[:] for row in VALID_4X4_SOLUTION]
        original_val = almost[0][0]
        almost[0][0] = 0
        candidates = get_valid_candidates(almost, 0, 0)
        assert candidates == [original_val]

    def test_single_candidate_full_row(self):
        # Row has 1,2,3, so only 4 is a valid candidate at position (0,3)
        grid = [[1, 2, 3, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        candidates = get_valid_candidates(grid, 0, 3)
        assert 4 in candidates
        # 1,2,3 are in the row already, so only 4 should work
        assert candidates == [4]

    def test_returns_sorted(self):
        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        candidates = get_valid_candidates(grid, 2, 2)
        assert candidates == sorted(candidates)

    def test_puzzle_candidates_from_9x9(self):
        grid = np.array(VALID_9X9_PUZZLE)
        # (0,2) is empty, solution has 4
        candidates = get_valid_candidates(grid, 0, 2)
        assert 4 in candidates
        # Verify all candidates are actually valid
        for num in candidates:
            assert is_valid_placement(grid, 0, 2, num) is True


# ===================================================================
# Integration: validator + generator consistency
# ===================================================================


class TestIntegrationConsistency:
    """Cross-check that valid solutions pass all validators."""

    def test_solution_cells_all_valid_placements_4x4(self):
        sol = np.array(VALID_4X4_SOLUTION)
        for r in range(4):
            for c in range(4):
                assert is_valid_placement(sol, r, c, int(sol[r, c]))

    def test_solution_cells_all_valid_placements_9x9(self):
        sol = np.array(VALID_9X9_SOLUTION)
        for r in range(9):
            for c in range(9):
                assert is_valid_placement(sol, r, c, int(sol[r, c]))

    def test_puzzle_empty_cells_match(self):
        empties = get_empty_cells(VALID_4X4_PUZZLE)
        filled = count_filled_cells(VALID_4X4_PUZZLE)
        assert len(empties) + filled == 16

    def test_puzzle_empty_cells_match_9x9(self):
        empties = get_empty_cells(VALID_9X9_PUZZLE)
        filled = count_filled_cells(VALID_9X9_PUZZLE)
        assert len(empties) + filled == 81


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
