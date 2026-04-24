# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Test suite for the weighted maze generator.

Tests cover:
  - Token encoding correctness
  - Grid structure validation (borders, start/goal uniqueness)
  - Dijkstra solver correctness (known mazes)
  - Solution validity (connected, optimal)
  - Dataset generation and output format
  - Reproducibility with seeds
  - Edge cases

Usage:
    python -m pytest test/other/test_weighted_maze_generator.py -v
"""

import numpy as np
import pytest

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

# Token encoding tests


class TestTokenEncoding:
    """Verify that token constants are unique and correctly defined."""

    def test_tokens_are_unique(self):
        tokens = {WALL, PATH, START, GOAL}
        weight_tokens = set(range(MIN_WEIGHT, MAX_WEIGHT + 1))
        assert len(tokens) == 4, "Core tokens must be unique"
        assert tokens.isdisjoint(weight_tokens), "Weight tokens must not overlap with core tokens"

    def test_token_values(self):
        assert WALL == 0
        assert PATH == 1
        assert START == 2
        assert GOAL == 3
        assert MIN_WEIGHT == 4
        assert MAX_WEIGHT == 9

    def test_cell_cost(self):
        gen = WeightedMazeGenerator(grid_size=7, seed=0)
        assert gen._cell_cost(WALL) == -1
        assert gen._cell_cost(PATH) == 1
        assert gen._cell_cost(START) == 1
        assert gen._cell_cost(GOAL) == 1
        for w in range(MIN_WEIGHT, MAX_WEIGHT + 1):
            assert gen._cell_cost(w) == w


# Grid structure tests


class TestGridStructure:
    """Verify maze grid structural integrity."""

    @pytest.fixture
    def gen(self):
        return WeightedMazeGenerator(grid_size=15, seed=42)

    def test_grid_dimensions(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        puzzle, solution, _meta = result
        n = gen.grid_size
        assert len(puzzle) == n
        assert all(len(row) == n for row in puzzle)
        assert len(solution) == n
        assert all(len(row) == n for row in solution)

    def test_outer_border_is_walls(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        puzzle, _, _ = result
        n = gen.grid_size
        # Top and bottom rows
        for c in range(n):
            assert puzzle[0][c] == WALL, f"Top border at col {c} is not WALL"
            assert puzzle[n - 1][c] == WALL, f"Bottom border at col {c} is not WALL"
        # Left and right columns
        for r in range(n):
            assert puzzle[r][0] == WALL, f"Left border at row {r} is not WALL"
            assert puzzle[r][n - 1] == WALL, f"Right border at row {r} is not WALL"

    def test_exactly_one_start_and_goal(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        puzzle, _, _ = result
        flat = [cell for row in puzzle for cell in row]
        assert flat.count(START) == 1, "Must have exactly one START"
        assert flat.count(GOAL) == 1, "Must have exactly one GOAL"

    def test_all_tokens_valid(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        puzzle, _, _ = result
        valid = {WALL, PATH, START, GOAL} | set(range(MIN_WEIGHT, MAX_WEIGHT + 1))
        for r, row in enumerate(puzzle):
            for c, cell in enumerate(row):
                assert cell in valid, f"Invalid token {cell} at ({r},{c})"

    def test_is_valid_puzzle(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        puzzle, _, _ = result
        assert gen.is_valid_puzzle(puzzle)

    def test_forced_odd_grid(self):
        gen = WeightedMazeGenerator(grid_size=14, seed=0)
        assert gen.grid_size == 15, "Even grid sizes should be forced to odd"

    def test_minimum_grid_size(self):
        with pytest.raises(ValueError):
            WeightedMazeGenerator(grid_size=5, seed=0)


# Dijkstra solver tests


class TestDijkstraSolver:
    """Test the Dijkstra pathfinding implementation with known mazes."""

    def test_simple_straight_path(self):
        """
        Maze:  W W W W W
               W S · · W
               W W W · W
               W · · G W
               W W W W W
        Optimal path: S→(1,2)→(1,3)→(2,3)→(3,3)→G  cost = 5
        """
        gen = WeightedMazeGenerator(grid_size=7, seed=0)
        # Override grid_size for this test
        gen.grid_size = 5
        grid = [
            [WALL, WALL, WALL, WALL, WALL],
            [WALL, START, PATH, PATH, WALL],
            [WALL, WALL, WALL, PATH, WALL],
            [WALL, PATH, PATH, GOAL, WALL],
            [WALL, WALL, WALL, WALL, WALL],
        ]
        start = (1, 1)
        goal = (3, 3)
        path = gen.solve_dijkstra(grid, start, goal)

        assert path is not None, "Path should exist"
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) == 5

    def test_weighted_path_prefers_cheaper(self):
        """
        7x7 Maze with a costly cell blocking the direct top route.
        Dijkstra should route around the 9-weight cell.
        """
        gen = WeightedMazeGenerator(grid_size=7, seed=0)
        grid = [
            [WALL, WALL, WALL, WALL, WALL, WALL, WALL],
            [WALL, START, PATH, 9, PATH, GOAL, WALL],
            [WALL, PATH, WALL, WALL, WALL, PATH, WALL],
            [WALL, PATH, PATH, PATH, PATH, PATH, WALL],
            [WALL, PATH, WALL, WALL, WALL, PATH, WALL],
            [WALL, PATH, PATH, PATH, PATH, PATH, WALL],
            [WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        ]
        start = (1, 1)
        goal = (1, 5)
        path = gen.solve_dijkstra(grid, start, goal)

        assert path is not None
        path_set = set(path)
        assert (1, 3) not in path_set, "Should avoid the costly cell (token 9)"

    def test_no_path_returns_none(self):
        """Completely walled off goal should return None."""
        gen = WeightedMazeGenerator(grid_size=7, seed=0)
        gen.grid_size = 5
        grid = [
            [WALL, WALL, WALL, WALL, WALL],
            [WALL, START, PATH, WALL, WALL],
            [WALL, WALL, WALL, WALL, WALL],
            [WALL, WALL, WALL, GOAL, WALL],
            [WALL, WALL, WALL, WALL, WALL],
        ]
        path = gen.solve_dijkstra(grid, (1, 1), (3, 3))
        assert path is None

    def test_adjacent_start_goal(self):
        """Start and goal right next to each other."""
        gen = WeightedMazeGenerator(grid_size=7, seed=0)
        gen.grid_size = 5
        grid = [
            [WALL, WALL, WALL, WALL, WALL],
            [WALL, START, GOAL, PATH, WALL],
            [WALL, PATH, PATH, PATH, WALL],
            [WALL, PATH, PATH, PATH, WALL],
            [WALL, WALL, WALL, WALL, WALL],
        ]
        path = gen.solve_dijkstra(grid, (1, 1), (1, 2))
        assert path is not None
        assert len(path) == 2
        assert path == [(1, 1), (1, 2)]


# Solution validity tests


class TestSolutionValidity:
    """Test that generated solutions are correct and optimal."""

    @pytest.fixture
    def gen(self):
        return WeightedMazeGenerator(grid_size=15, seed=42)

    def test_solution_is_binary(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        _, solution, _ = result
        for row in solution:
            for cell in row:
                assert cell in (0, 1), f"Solution must be binary, got {cell}"

    def test_solution_includes_start_and_goal(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        _puzzle, solution, meta = result
        sr, sc = meta["start"]
        gr, gc = meta["goal"]
        assert solution[sr][sc] == 1, "Start must be on solution path"
        assert solution[gr][gc] == 1, "Goal must be on solution path"

    def test_solution_path_is_connected(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        _puzzle, solution, meta = result

        path_cells = set()
        for r in range(gen.grid_size):
            for c in range(gen.grid_size):
                if solution[r][c] == 1:
                    path_cells.add((r, c))

        # BFS from start
        start = meta["start"]
        visited = {start}
        queue = [start]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in path_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        assert visited == path_cells, "Solution path must be connected"

    def test_solution_is_optimal(self, gen):
        """Verify the solution matches an independent Dijkstra solve."""
        result = gen.create_puzzle()
        assert result is not None
        puzzle, solution, meta = result

        # Re-solve independently
        path = gen.solve_dijkstra(puzzle, meta["start"], meta["goal"])
        assert path is not None

        expected_cost = gen._compute_path_cost(puzzle, path)
        actual_cost = sum(
            max(gen._cell_cost(puzzle[r][c]), 0)
            for r in range(gen.grid_size)
            for c in range(gen.grid_size)
            if solution[r][c] == 1
        )
        assert (
            actual_cost == expected_cost
        ), f"Solution cost {actual_cost} != optimal cost {expected_cost}"

    def test_is_valid_solution_method(self, gen):
        result = gen.create_puzzle()
        assert result is not None
        puzzle, solution, _ = result
        assert gen.is_valid_solution(puzzle, solution)

    def test_solution_cells_are_traversable(self, gen):
        """No wall should be marked as part of the solution."""
        result = gen.create_puzzle()
        assert result is not None
        puzzle, solution, _ = result
        for r in range(gen.grid_size):
            for c in range(gen.grid_size):
                if solution[r][c] == 1:
                    assert puzzle[r][c] != WALL, f"Solution cell ({r},{c}) is a wall"

    def test_unique_optimal_path(self, gen):
        """Generated puzzles must have exactly one cheapest path."""
        result = gen.create_puzzle()
        assert result is not None
        puzzle, _solution, meta = result
        assert gen._has_unique_optimal_path(
            puzzle, meta["start"], meta["goal"]
        ), "Puzzle must have exactly one optimal path"

    def test_unique_optimal_path_multiple_puzzles(self):
        """Verify uniqueness guarantee across several generated puzzles."""
        gen = WeightedMazeGenerator(grid_size=11, seed=99)
        for _ in range(10):
            result = gen.create_puzzle(max_attempts=100)
            assert result is not None
            puzzle, _, meta = result
            assert gen._has_unique_optimal_path(
                puzzle, meta["start"], meta["goal"]
            ), "Every generated puzzle must have a unique optimal path"


# Dataset generation tests


class TestDatasetGeneration:
    """Test the full dataset generation pipeline."""

    def test_output_shapes(self):
        dataset = generate_weighted_maze_dataset(
            num_puzzles=5, grid_size=11, seed=42, verbose=False
        )
        assert dataset["problems"].shape == (5, 11, 11)
        assert dataset["solutions"].shape == (5, 11, 11)
        assert len(dataset["metadata"]) == 5

    def test_output_dtype(self):
        dataset = generate_weighted_maze_dataset(num_puzzles=3, grid_size=9, seed=42, verbose=False)
        assert dataset["problems"].dtype == np.int8
        assert dataset["solutions"].dtype == np.int8

    def test_metadata_fields(self):
        dataset = generate_weighted_maze_dataset(
            num_puzzles=3, grid_size=11, seed=42, verbose=False
        )
        required_fields = {
            "puzzle_id",
            "grid_size",
            "path_length",
            "path_cost",
            "start",
            "goal",
            "num_walls",
            "num_weighted",
        }
        for meta in dataset["metadata"]:
            assert required_fields.issubset(
                meta.keys()
            ), f"Missing fields: {required_fields - meta.keys()}"

    def test_uniqueness(self):
        dataset = generate_weighted_maze_dataset(
            num_puzzles=10, grid_size=11, seed=42, verbose=False, ensure_unique=True
        )
        seen = set()
        for i in range(len(dataset["problems"])):
            key = tuple(map(tuple, dataset["problems"][i]))
            assert key not in seen, f"Duplicate puzzle at index {i}"
            seen.add(key)

    def test_even_grid_size_forced_odd(self):
        dataset = generate_weighted_maze_dataset(
            num_puzzles=2, grid_size=10, seed=42, verbose=False
        )
        # Should be 11x11 (forced odd)
        assert dataset["problems"].shape[1] == 11
        assert dataset["problems"].shape[2] == 11


# Reproducibility tests


class TestReproducibility:
    """Test that seeded generation is deterministic."""

    def test_same_seed_same_puzzle(self):
        gen1 = WeightedMazeGenerator(grid_size=15, seed=12345)
        gen2 = WeightedMazeGenerator(grid_size=15, seed=12345)
        r1 = gen1.create_puzzle(max_attempts=100)
        r2 = gen2.create_puzzle(max_attempts=100)
        assert r1 is not None and r2 is not None
        assert r1[0] == r2[0], "Same seed should produce same puzzle"
        assert r1[1] == r2[1], "Same seed should produce same solution"

    def test_different_seed_different_puzzle(self):
        gen1 = WeightedMazeGenerator(grid_size=15, seed=111)
        gen2 = WeightedMazeGenerator(grid_size=15, seed=222)
        r1 = gen1.create_puzzle(max_attempts=100)
        r2 = gen2.create_puzzle(max_attempts=100)
        assert r1 is not None and r2 is not None
        assert r1[0] != r2[0], "Different seeds should produce different puzzles"

    def test_dataset_reproducibility(self):
        d1 = generate_weighted_maze_dataset(num_puzzles=5, grid_size=11, seed=42, verbose=False)
        d2 = generate_weighted_maze_dataset(num_puzzles=5, grid_size=11, seed=42, verbose=False)
        np.testing.assert_array_equal(d1["problems"], d2["problems"])
        np.testing.assert_array_equal(d1["solutions"], d2["solutions"])


# Visualization test


class TestVisualization:
    """Test text rendering of mazes."""

    def test_print_grid_runs(self):
        gen = WeightedMazeGenerator(grid_size=11, seed=42)
        result = gen.create_puzzle()
        assert result is not None
        puzzle, solution, _ = result
        text = gen.print_grid(puzzle, solution)
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain S and G
        assert "S" in text
        assert "G" in text
        # Should contain path markers
        assert "*" in text


# Integration: end-to-end HRM compatibility check


class TestHRMCompatibility:
    """
    Verify the output format is compatible with HRM training pipeline.

    HRM expects:
      - problems and solutions as np arrays of shape (N, H, W)
      - Flattened token sequences for seq2seq training
      - Small, bounded vocabulary (token range)
    """

    def test_vocabulary_is_bounded(self):
        dataset = generate_weighted_maze_dataset(
            num_puzzles=10, grid_size=15, seed=42, verbose=False
        )
        unique_problem_tokens = set(np.unique(dataset["problems"]))
        unique_solution_tokens = set(np.unique(dataset["solutions"]))

        expected_problem_tokens = {WALL, PATH, START, GOAL} | set(range(MIN_WEIGHT, MAX_WEIGHT + 1))
        assert unique_problem_tokens.issubset(
            expected_problem_tokens
        ), f"Unexpected tokens in problems: {unique_problem_tokens - expected_problem_tokens}"

        assert unique_solution_tokens.issubset(
            {0, 1}
        ), f"Solution tokens must be binary, got {unique_solution_tokens}"

    def test_flattening_preserves_info(self):
        """HRM flattens 2D grids to 1D sequences. Verify round-trip."""
        dataset = generate_weighted_maze_dataset(
            num_puzzles=3, grid_size=11, seed=42, verbose=False
        )
        for i in range(len(dataset["problems"])):
            grid_2d = dataset["problems"][i]
            flat = grid_2d.flatten()
            restored = flat.reshape(grid_2d.shape)
            np.testing.assert_array_equal(grid_2d, restored)

    def test_vocab_size(self):
        """Ensure total vocabulary is small (fits in small embedding table)."""
        # Problem tokens: 0-9 (10 tokens)
        # Solution tokens: 0-1 (2 tokens)
        # Combined: 10 unique values
        assert MAX_WEIGHT + 1 <= 16, "Vocabulary should be small for HRM (typically < 16 tokens)"


# Run tests

if __name__ == "__main__":
    # Run with pytest if available, otherwise manual run
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
