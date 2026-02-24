"""
Tests for the maze_generator module.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from hrm.prototype.generators.maze_generator import (
    generate_single_maze,
    find_shortest_path,
    find_path_length,
    place_start_exit,
    create_solution_matrix,
    generate_maze_dataset,
)


class TestGenerateSingleMaze:
    """Tests for generate_single_maze function."""

    def test_returns_valid_maze(self):
        """Test that a valid maze is generated."""
        maze, start_pos, exit_pos = generate_single_maze(width=15, height=15, min_path_length=5)
        
        assert maze is not None
        assert start_pos is not None
        assert exit_pos is not None

    def test_maze_dimensions(self):
        """Test maze has correct dimensions."""
        width, height = 20, 15
        maze, _, _ = generate_single_maze(width=width, height=height, min_path_length=5)
        
        assert maze.shape == (height, width)

    def test_maze_contains_required_values(self):
        """Test maze contains start (-2), exit (0), and walls (-1)."""
        maze, start_pos, exit_pos = generate_single_maze(width=15, height=15, min_path_length=5)
        
        # Check start marker
        assert maze[start_pos] == -2
        # Check exit marker
        assert maze[exit_pos] == 0
        # Check walls exist
        assert -1 in maze

    def test_start_and_exit_different(self):
        """Test start and exit positions are different."""
        maze, start_pos, exit_pos = generate_single_maze(width=15, height=15, min_path_length=5)
        
        assert start_pos != exit_pos

    def test_maze_has_passages(self):
        """Test that maze has passages (non-wall, non-special cells)."""
        maze, _, _ = generate_single_maze(width=15, height=15, min_path_length=5)
        
        # Passages should have Manhattan distance values (positive integers)
        passage_count = np.sum(maze > 0)
        assert passage_count > 0

    def test_small_maze_may_fail(self):
        """Test that very small maze with long path requirement may fail."""
        # 5x5 maze with path length 50 is impossible
        maze, start_pos, exit_pos = generate_single_maze(width=5, height=5, min_path_length=50)
        
        # Should return None when impossible
        assert maze is None
        assert start_pos is None
        assert exit_pos is None


class TestFindShortestPath:
    """Tests for find_shortest_path function."""

    @pytest.fixture
    def simple_maze(self):
        """Create a simple maze for testing path finding."""
        # 5x5 maze with clear path
        # -1 = wall, positive = passage
        maze = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  1,  1,  1, -1],
            [-1,  1, -1,  1, -1],
            [-1,  1,  1,  1, -1],
            [-1, -1, -1, -1, -1],
        ])
        return maze

    def test_finds_path_in_simple_maze(self, simple_maze):
        """Test path finding in simple maze."""
        start = (1, 1)
        end = (3, 3)
        path = find_shortest_path(simple_maze, start, end)
        
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == end

    def test_path_avoids_walls(self, simple_maze):
        """Test that path doesn't include wall cells."""
        start = (1, 1)
        end = (3, 3)
        path = find_shortest_path(simple_maze, start, end)
        
        for y, x in path:
            assert simple_maze[y, x] != -1

    def test_no_path_returns_empty(self):
        """Test returns empty list when no path exists."""
        # Maze with completely blocked path
        maze = np.array([
            [-1, -1, -1],
            [ 1, -1,  1],
            [-1, -1, -1],
        ])
        path = find_shortest_path(maze, (1, 0), (1, 2))
        
        assert path == []

    def test_same_start_and_end(self, simple_maze):
        """Test path from cell to itself."""
        start = (1, 1)
        path = find_shortest_path(simple_maze, start, start)
        
        assert len(path) == 1
        assert path[0] == start


class TestFindPathLength:
    """Tests for find_path_length function."""

    @pytest.fixture
    def linear_maze(self):
        """Create a linear maze for predictable path length."""
        # Horizontal corridor
        maze = np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1,  1,  1,  1,  1,  1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ])
        return maze

    def test_path_length_correct(self, linear_maze):
        """Test path length is correct for linear maze."""
        length = find_path_length(linear_maze, (1, 1), (1, 5))
        
        # Path from (1,1) to (1,5) is 4 steps
        assert length == 4

    def test_no_path_returns_negative(self):
        """Test returns -1 when no path exists."""
        maze = np.array([
            [-1, -1, -1],
            [ 1, -1,  1],
            [-1, -1, -1],
        ])
        length = find_path_length(maze, (1, 0), (1, 2))
        
        assert length == -1

    def test_same_position_zero_length(self, linear_maze):
        """Test path length is 0 for same start and end."""
        length = find_path_length(linear_maze, (1, 1), (1, 1))
        
        assert length == 0


class TestPlaceStartExit:
    """Tests for place_start_exit function."""

    @pytest.fixture
    def open_maze(self):
        """Create an open maze with many passages."""
        maze = np.ones((10, 10), dtype=int)
        maze[0, :] = -1
        maze[-1, :] = -1
        maze[:, 0] = -1
        maze[:, -1] = -1
        return maze

    def test_places_valid_positions(self, open_maze):
        """Test that valid positions are placed."""
        passage_cells = [(y, x) for y in range(10) for x in range(10) if open_maze[y, x] != -1]
        
        exit_pos, start_pos = place_start_exit(open_maze, 10, 10, 5, passage_cells)
        
        assert exit_pos is not None
        assert start_pos is not None
        assert exit_pos in passage_cells
        assert start_pos in passage_cells

    def test_positions_meet_min_path_length(self, open_maze):
        """Test that placed positions meet minimum path requirement."""
        passage_cells = [(y, x) for y in range(10) for x in range(10) if open_maze[y, x] != -1]
        min_path = 5
        
        exit_pos, start_pos = place_start_exit(open_maze, 10, 10, min_path, passage_cells)
        
        if exit_pos is not None:
            path_len = find_path_length(open_maze, start_pos, exit_pos)
            assert path_len >= min_path


class TestCreateSolutionMatrix:
    """Tests for create_solution_matrix function."""

    @pytest.fixture
    def test_maze_with_positions(self):
        """Create a test maze with known path."""
        maze = np.array([
            [-1, -1, -1, -1, -1],
            [-1, -2,  2,  1, -1],  # -2 is start
            [-1,  3, -1,  0, -1],  # 0 is exit
            [-1, -1, -1, -1, -1],
        ])
        start_pos = (1, 1)
        exit_pos = (2, 3)
        return maze, start_pos, exit_pos

    def test_binary_solution_shape(self, test_maze_with_positions):
        """Test binary solution has correct shape."""
        maze, start_pos, exit_pos = test_maze_with_positions
        solution = create_solution_matrix(maze, start_pos, exit_pos, 'binary')
        
        assert solution.shape == maze.shape

    def test_binary_solution_values(self, test_maze_with_positions):
        """Test binary solution contains only 0 and 1."""
        maze, start_pos, exit_pos = test_maze_with_positions
        solution = create_solution_matrix(maze, start_pos, exit_pos, 'binary')
        
        unique_values = np.unique(solution)
        assert all(v in [0, 1] for v in unique_values)

    def test_binary_solution_marks_path(self, test_maze_with_positions):
        """Test binary solution marks start and exit as on path."""
        maze, start_pos, exit_pos = test_maze_with_positions
        solution = create_solution_matrix(maze, start_pos, exit_pos, 'binary')
        
        assert solution[start_pos] == 1
        assert solution[exit_pos] == 1

    def test_path_marked_solution_preserves_markers(self, test_maze_with_positions):
        """Test path_marked solution preserves start and exit markers."""
        maze, start_pos, exit_pos = test_maze_with_positions
        solution = create_solution_matrix(maze, start_pos, exit_pos, 'path_marked')
        
        assert solution[start_pos] == -2
        assert solution[exit_pos] == 0

    def test_invalid_format_raises_error(self, test_maze_with_positions):
        """Test invalid solution format raises ValueError."""
        maze, start_pos, exit_pos = test_maze_with_positions
        
        with pytest.raises(ValueError, match="Unknown solution format"):
            create_solution_matrix(maze, start_pos, exit_pos, 'invalid_format')


class TestGenerateMazeDataset:
    """Tests for generate_maze_dataset function."""

    def test_generates_correct_number(self):
        """Test dataset contains requested number of mazes."""
        dataset = generate_maze_dataset(
            num_mazes=5,
            width=15,
            height=15,
            min_path_length=5,
            seed=42,
            verbose=False
        )
        
        assert dataset['problems'].shape[0] == 5
        assert dataset['solutions'].shape[0] == 5
        assert len(dataset['metadata']) == 5

    def test_dataset_shapes(self):
        """Test dataset arrays have correct shapes."""
        width, height, num = 12, 10, 3
        dataset = generate_maze_dataset(
            num_mazes=num,
            width=width,
            height=height,
            min_path_length=5,
            seed=42,
            verbose=False
        )
        
        assert dataset['problems'].shape == (num, height, width)
        assert dataset['solutions'].shape == (num, height, width)

    def test_seed_reproducibility(self):
        """Test same seed produces identical results."""
        params = dict(
            num_mazes=3,
            width=15,
            height=15,
            min_path_length=5,
            seed=12345,
            verbose=False
        )
        
        dataset1 = generate_maze_dataset(**params)
        dataset2 = generate_maze_dataset(**params)
        
        np.testing.assert_array_equal(dataset1['problems'], dataset2['problems'])
        np.testing.assert_array_equal(dataset1['solutions'], dataset2['solutions'])

    def test_metadata_contains_required_fields(self):
        """Test metadata contains start_pos, exit_pos, path_length."""
        dataset = generate_maze_dataset(
            num_mazes=2,
            width=15,
            height=15,
            min_path_length=5,
            seed=42,
            verbose=False
        )
        
        for meta in dataset['metadata']:
            assert 'start_pos' in meta
            assert 'exit_pos' in meta
            assert 'path_length' in meta

    def test_path_lengths_meet_minimum(self):
        """Test all generated mazes meet minimum path length."""
        min_path = 10
        dataset = generate_maze_dataset(
            num_mazes=5,
            width=20,
            height=20,
            min_path_length=min_path,
            seed=42,
            verbose=False
        )
        
        for meta in dataset['metadata']:
            assert meta['path_length'] >= min_path

    def test_binary_solution_format(self):
        """Test binary solution format produces 0/1 arrays."""
        dataset = generate_maze_dataset(
            num_mazes=2,
            width=15,
            height=15,
            min_path_length=5,
            seed=42,
            solution_format='binary',
            verbose=False
        )
        
        unique = np.unique(dataset['solutions'])
        assert all(v in [0, 1] for v in unique)

    def test_path_marked_solution_format(self):
        """Test path_marked solution format preserves maze structure."""
        dataset = generate_maze_dataset(
            num_mazes=2,
            width=15,
            height=15,
            min_path_length=5,
            seed=42,
            solution_format='path_marked',
            verbose=False
        )
        
        # Should contain -2 (start), -1 (walls), 0 (exit), and path markers
        assert -2 in dataset['solutions']
        assert -1 in dataset['solutions']
        assert 0 in dataset['solutions']


class TestMazeIntegrity:
    """Integration tests to verify maze integrity."""

    def test_solution_path_is_valid(self):
        """Test that solution path is actually traversable in the maze."""
        dataset = generate_maze_dataset(
            num_mazes=3,
            width=15,
            height=15,
            min_path_length=5,
            seed=42,
            solution_format='binary',
            verbose=False
        )
        
        for i, (maze, solution, meta) in enumerate(zip(
            dataset['problems'], 
            dataset['solutions'], 
            dataset['metadata']
        )):
            # Get path cells from solution
            path_cells = set(zip(*np.where(solution == 1)))
            
            # Verify start and exit are on path
            assert meta['start_pos'] in path_cells, f"Maze {i}: Start not on path"
            assert meta['exit_pos'] in path_cells, f"Maze {i}: Exit not on path"
            
            # Verify path cells are not walls in original maze
            for y, x in path_cells:
                assert maze[y, x] != -1, f"Maze {i}: Path goes through wall at ({y}, {x})"

    def test_path_connectivity(self):
        """Test that the solution path is connected."""
        dataset = generate_maze_dataset(
            num_mazes=3,
            width=15,
            height=15,
            min_path_length=5,
            seed=42,
            solution_format='binary',
            verbose=False
        )
        
        for i, (solution, meta) in enumerate(zip(dataset['solutions'], dataset['metadata'])):
            path_cells = list(zip(*np.where(solution == 1)))
            
            # Simple connectivity check: verify path from start to exit exists
            start = meta['start_pos']
            exit_pos = meta['exit_pos']
            
            # Create a mask maze from solution for pathfinding
            path_maze = solution.copy()
            path_len = find_path_length(path_maze, start, exit_pos)
            
            assert path_len > 0, f"Maze {i}: Path is not connected"
