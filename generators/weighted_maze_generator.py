"""
Weighted maze puzzle generator for HRM training.

Generates grid-based weighted mazes where the task is to find the minimum-cost
path from start to goal. Solutions are computed via Dijkstra's algorithm, producing
optimal paths that require non-trivial search — a task well-suited to HRM's
hierarchical reasoning capabilities.

Token encoding (unique integers for each cell type):
    0  = WALL      (impassable obstacle)
    1  = PATH      (traversable, weight 1 — default/background)
    2  = START     (unique start position)
    3  = GOAL      (unique goal position)
    4+ = WEIGHTED  (traversable cells with cost = token value)
                    e.g., token 4 = cost 4, token 9 = cost 9

Solution encoding (output grid):
    0  = not on optimal path
    1  = on optimal path (including start and goal cells)

Output structure for NN training (matches Sudoku generator):
    - problems:  np.array of shape (num_puzzles, grid_size, grid_size)
    - solutions: np.array of shape (num_puzzles, grid_size, grid_size)

Usage:
    python weighted_maze_generator.py --num 100 --size 15 --seed 42
"""

import random
import heapq
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
from collections import defaultdict

import numpy as np


# =============================================================================
# Token definitions — each cell type has a unique integer ID
# =============================================================================
WALL  = 0
PATH  = 1   # traversable, cost 1
START = 2
GOAL  = 3
# Tokens 4..MAX_WEIGHT represent weighted traversable cells (cost = token value)
MIN_WEIGHT = 4
MAX_WEIGHT = 9


class WeightedMazeGenerator:
    """
    Generates weighted maze puzzles with optimal-path solutions.

    Mazes are NxN grids where:
      - Outer border is always walls
      - Interior contains a mix of walls, open paths, and weighted cells
      - Exactly one START and one GOAL cell
      - Solution is a binary mask of the minimum-cost path (Dijkstra)

    """

    # Maze generation parameters
    # wall_frac: fraction of interior cells that are walls
    # weight_frac: fraction of non-wall interior cells that get random weights
    # min_path_len_ratio: minimum optimal path length as fraction of grid_size
    MAZE_CONFIG = {
        'wall_frac': (0.25, 0.40),
        'weight_frac': (0.20, 0.45),
        'min_path_len_ratio': 2.0,
        'max_weight': MAX_WEIGHT,  # weights 4-9
    }

    def __init__(self, grid_size: int = 15, seed: Optional[int] = None):
        """
        Initialize the weighted maze generator.

        Args:
            grid_size: Size of the grid (must be odd, >= 7). Default 15.
            seed: Random seed for reproducibility.
        """
        if grid_size < 7:
            raise ValueError("Grid size must be >= 7")
        # Force odd size for clean maze structure
        if grid_size % 2 == 0:
            grid_size += 1
        self.grid_size = grid_size
        self.seed = seed
        self._rng = random.Random(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Set a new random seed."""
        self.seed = seed
        self._rng = random.Random(seed)

    # -----------------------------------------------------------------
    # Core maze generation
    # -----------------------------------------------------------------

    def _generate_base_maze(self) -> List[List[int]]:
        """
        Generate a base maze using randomized DFS (recursive backtracker).

        This creates a perfect maze (exactly one path between any two cells)
        on the odd-indexed interior cells, then we selectively open extra
        passages to create multiple routes.

        Returns:
            2D grid filled with WALL and PATH tokens.
        """
        n = self.grid_size
        grid = [[WALL] * n for _ in range(n)]

        # Carve passages on odd-indexed cells using DFS
        start_r, start_c = 1, 1
        grid[start_r][start_c] = PATH
        stack = [(start_r, start_c)]
        visited: Set[Tuple[int, int]] = {(start_r, start_c)}

        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        while stack:
            r, c = stack[-1]
            # Find unvisited neighbors 2 steps away
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < n - 1 and 1 <= nc < n - 1 and (nr, nc) not in visited:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))

            if neighbors:
                nr, nc, wr, wc = neighbors[self._rng.randint(0, len(neighbors) - 1)]
                grid[wr][wc] = PATH  # knock down wall between
                grid[nr][nc] = PATH
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid

    def _open_extra_passages(self, grid: List[List[int]], open_frac: float) -> None:
        """
        Remove some walls to create multiple route options.

        Args:
            grid: The maze grid (modified in place).
            open_frac: Fraction of removable interior walls to open.
        """
        n = self.grid_size
        # Collect interior walls that border at least two PATH cells
        candidates = []
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                if grid[r][c] == WALL:
                    adj_paths = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == PATH:
                            adj_paths += 1
                    if adj_paths >= 2:
                        candidates.append((r, c))

        self._rng.shuffle(candidates)
        num_to_open = int(len(candidates) * open_frac)
        for i in range(min(num_to_open, len(candidates))):
            grid[candidates[i][0]][candidates[i][1]] = PATH

    def _place_weights(self, grid: List[List[int]], weight_frac: float,
                       max_weight: int) -> None:
        """
        Assign random weights to a fraction of PATH cells.

        Args:
            grid: The maze grid (modified in place).
            weight_frac: Fraction of PATH cells to convert to weighted.
            max_weight: Maximum weight token to use.
        """
        n = self.grid_size
        path_cells = []
        for r in range(n):
            for c in range(n):
                if grid[r][c] == PATH:
                    path_cells.append((r, c))

        self._rng.shuffle(path_cells)
        num_weighted = int(len(path_cells) * weight_frac)

        for i in range(min(num_weighted, len(path_cells))):
            r, c = path_cells[i]
            grid[r][c] = self._rng.randint(MIN_WEIGHT, max_weight)

    def _place_start_goal(self, grid: List[List[int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Place START and GOAL on traversable cells, maximizing their distance.

        Args:
            grid: The maze grid (modified in place).

        Returns:
            (start_pos, goal_pos) as (row, col) tuples.
        """
        n = self.grid_size
        traversable = []
        for r in range(n):
            for c in range(n):
                if grid[r][c] != WALL:
                    traversable.append((r, c))

        if len(traversable) < 2:
            raise ValueError("Not enough traversable cells to place start and goal")

        # Pick start and goal to be far apart (use Manhattan distance heuristic,
        # then verify with actual shortest path)
        best_pair = None
        best_dist = -1

        # Sample candidate pairs for efficiency on large grids
        sample_size = min(len(traversable), 20)
        candidates = self._rng.sample(traversable, sample_size)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                r1, c1 = candidates[i]
                r2, c2 = candidates[j]
                dist = abs(r1 - r2) + abs(c1 - c2)
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (candidates[i], candidates[j])

        start_pos, goal_pos = best_pair
        grid[start_pos[0]][start_pos[1]] = START
        grid[goal_pos[0]][goal_pos[1]] = GOAL

        return start_pos, goal_pos

    # -----------------------------------------------------------------
    # Pathfinding (Dijkstra)
    # -----------------------------------------------------------------

    @staticmethod
    def _cell_cost(token: int) -> int:
        """Return the traversal cost for a cell token."""
        if token == WALL:
            return -1  # impassable
        if token in (PATH, START, GOAL):
            return 1
        # Weighted cell: cost equals the token value
        return token

    def solve_dijkstra(self, grid: List[List[int]],
                       start: Tuple[int, int],
                       goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the minimum-cost path from start to goal using Dijkstra's algorithm.

        Args:
            grid: The maze grid.
            start: (row, col) of the start cell.
            goal: (row, col) of the goal cell.

        Returns:
            List of (row, col) cells on the optimal path (including start/goal),
            or None if no path exists.
        """
        n = self.grid_size
        dist = defaultdict(lambda: float('inf'))
        dist[start] = 0
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        pq = [(0, start)]
        visited: Set[Tuple[int, int]] = set()

        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if (r, c) == goal:
                # Reconstruct path
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return path[::-1]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    cost = self._cell_cost(grid[nr][nc])
                    if cost < 0:
                        continue  # wall
                    new_dist = d + cost
                    if new_dist < dist[(nr, nc)]:
                        dist[(nr, nc)] = new_dist
                        prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (new_dist, (nr, nc)))

        return None  # no path

    def _has_unique_optimal_path(self, grid: List[List[int]],
                                  start: Tuple[int, int],
                                  goal: Tuple[int, int]) -> bool:
        """
        Check whether there is exactly one minimum-cost path from start to goal.

        Uses Dijkstra to compute shortest distances, then counts the number of
        distinct shortest paths via dynamic programming on the DAG of optimal edges.

        Args:
            grid: The maze grid.
            start: (row, col) of the start cell.
            goal: (row, col) of the goal cell.

        Returns:
            True if exactly one optimal path exists, False otherwise.
        """
        n = self.grid_size
        dist: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        dist[start] = 0
        pq = [(0, start)]
        visited: Set[Tuple[int, int]] = set()

        # Phase 1: Dijkstra to get shortest distances to all reachable cells
        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    cost = self._cell_cost(grid[nr][nc])
                    if cost < 0:
                        continue
                    new_dist = d + cost
                    if new_dist < dist[(nr, nc)]:
                        dist[(nr, nc)] = new_dist
                        heapq.heappush(pq, (new_dist, (nr, nc)))

        if dist[goal] == float('inf'):
            return False  # no path at all

        # Phase 2: count paths on the shortest-path DAG
        # Process cells in order of increasing distance
        num_paths: Dict[Tuple[int, int], int] = defaultdict(int)
        num_paths[start] = 1

        for cell in sorted(visited, key=lambda c: dist[c]):
            r, c = cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    cost = self._cell_cost(grid[nr][nc])
                    if cost < 0:
                        continue
                    # (r,c) -> (nr,nc) is on the shortest-path DAG
                    if dist[(nr, nc)] == dist[cell] + cost:
                        num_paths[(nr, nc)] += num_paths[cell]

        return num_paths[goal] == 1

    def _path_to_solution_grid(self, path: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Convert a path to a binary solution grid.

        Args:
            path: List of (row, col) on the optimal path.

        Returns:
            NxN grid where 1 = on path, 0 = not on path.
        """
        n = self.grid_size
        sol = [[0] * n for _ in range(n)]
        for r, c in path:
            sol[r][c] = 1
        return sol

    # -----------------------------------------------------------------
    # Puzzle creation
    # -----------------------------------------------------------------

    def create_puzzle(self, max_attempts: int = 50) -> Optional[Tuple[List[List[int]], List[List[int]], Dict[str, Any]]]:
        """
        Create a weighted maze puzzle with its optimal-path solution.

        Args:
            max_attempts: Maximum generation attempts before giving up.

        Returns:
            (puzzle_grid, solution_grid, metadata) or None if generation fails.
        """
        config = self.MAZE_CONFIG
        wall_frac_lo, wall_frac_hi = config['wall_frac']
        wt_frac_lo, wt_frac_hi = config['weight_frac']
        min_path_len = int(config['min_path_len_ratio'] * self.grid_size)
        max_weight = config['max_weight']

        for attempt in range(max_attempts):
            # 1. Generate base perfect maze
            grid = self._generate_base_maze()

            # 2. Open extra passages (more openness → need weights for difficulty)
            open_frac = self._rng.uniform(0.15, 0.35)
            self._open_extra_passages(grid, open_frac)

            # 3. Optionally add more walls to hit target wall density
            self._adjust_wall_density(grid, self._rng.uniform(wall_frac_lo, wall_frac_hi))

            # 4. Place weights on traversable cells
            weight_frac = self._rng.uniform(wt_frac_lo, wt_frac_hi)
            self._place_weights(grid, weight_frac, max_weight)

            # 5. Place start and goal
            try:
                start, goal_pos = self._place_start_goal(grid)
            except ValueError:
                continue

            # 6. Solve with Dijkstra
            path = self.solve_dijkstra(grid, start, goal_pos)
            if path is None:
                continue

            # 7. Check minimum path length requirement
            if len(path) < min_path_len:
                continue

            # 8. Ensure the optimal path is unique
            if not self._has_unique_optimal_path(grid, start, goal_pos):
                continue

            # 9. Compute solution grid and metadata
            solution = self._path_to_solution_grid(path)
            total_cost = self._compute_path_cost(grid, path)

            metadata = {
                'grid_size': self.grid_size,
                'path_length': len(path),
                'path_cost': total_cost,
                'start': start,
                'goal': goal_pos,
                'num_walls': sum(1 for r in grid for c in r if c == WALL),
                'num_weighted': sum(1 for r in grid for c in r if c >= MIN_WEIGHT),
                'attempt': attempt + 1,
            }

            return grid, solution, metadata

        return None  # failed after max_attempts

    def _adjust_wall_density(self, grid: List[List[int]], target_frac: float) -> None:
        """
        Adjust interior wall density toward a target fraction.

        Only adds walls (never removes the DFS-carved paths entirely).
        """
        n = self.grid_size
        interior_cells = []
        wall_count = 0
        total_interior = 0

        for r in range(1, n - 1):
            for c in range(1, n - 1):
                total_interior += 1
                if grid[r][c] == WALL:
                    wall_count += 1
                elif grid[r][c] == PATH:
                    interior_cells.append((r, c))

        if total_interior == 0:
            return

        current_frac = wall_count / total_interior
        if current_frac >= target_frac:
            return

        # Add walls to PATH cells (not weighted, not start/goal)
        self._rng.shuffle(interior_cells)
        needed = int((target_frac - current_frac) * total_interior)

        for i in range(min(needed, len(interior_cells))):
            r, c = interior_cells[i]
            # Only add wall if it doesn't disconnect the grid too much
            # Simple heuristic: ensure cell has at least 2 adjacent non-wall neighbors
            adj_open = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < n and 0 <= c + dc < n and grid[r + dr][c + dc] != WALL
            )
            if adj_open >= 3:  # safe to wall off
                grid[r][c] = WALL

    def _compute_path_cost(self, grid: List[List[int]],
                           path: List[Tuple[int, int]]) -> int:
        """Compute the total traversal cost of a path."""
        total = 0
        for r, c in path:
            cost = self._cell_cost(grid[r][c])
            total += max(cost, 0)
        return total

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def is_valid_puzzle(self, grid: List[List[int]]) -> bool:
        """
        Validate that a puzzle grid is well-formed.

        Checks:
          - Correct dimensions
          - Outer border is all walls
          - Exactly one START and one GOAL
          - All tokens are in the valid set
        """
        n = self.grid_size
        if len(grid) != n:
            return False
        for row in grid:
            if len(row) != n:
                return False

        valid_tokens = {WALL, PATH, START, GOAL} | set(range(MIN_WEIGHT, MAX_WEIGHT + 1))

        start_count = 0
        goal_count = 0

        for r in range(n):
            for c in range(n):
                token = grid[r][c]
                if token not in valid_tokens:
                    return False
                # Border must be wall (except we allow start/goal on border edge
                # for some variants — here we enforce walls)
                if r == 0 or r == n - 1 or c == 0 or c == n - 1:
                    if token != WALL:
                        return False
                if token == START:
                    start_count += 1
                elif token == GOAL:
                    goal_count += 1

        return start_count == 1 and goal_count == 1

    def is_valid_solution(self, grid: List[List[int]], solution: List[List[int]]) -> bool:
        """
        Validate that a solution is a valid optimal path.

        Checks:
          - Solution cells form a connected path from START to GOAL
          - Path cost matches Dijkstra's optimal cost
        """
        n = self.grid_size
        # Find start and goal
        start = goal_pos = None
        for r in range(n):
            for c in range(n):
                if grid[r][c] == START:
                    start = (r, c)
                elif grid[r][c] == GOAL:
                    goal_pos = (r, c)

        if start is None or goal_pos is None:
            return False

        # Check that start and goal are on the solution path
        if solution[start[0]][start[1]] != 1 or solution[goal_pos[0]][goal_pos[1]] != 1:
            return False

        # Extract path cells and check connectivity
        path_cells = set()
        for r in range(n):
            for c in range(n):
                if solution[r][c] == 1:
                    path_cells.add((r, c))

        # BFS from start along solution cells
        visited = {start}
        queue = [start]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in path_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        if visited != path_cells:
            return False

        # Check optimality
        optimal = self.solve_dijkstra(grid, start, goal_pos)
        if optimal is None:
            return False

        optimal_cost = self._compute_path_cost(grid, optimal)
        solution_cost = sum(
            max(self._cell_cost(grid[r][c]), 0)
            for r, c in path_cells
        )

        return solution_cost == optimal_cost

    # -----------------------------------------------------------------
    # Visualization (text)
    # -----------------------------------------------------------------

    @staticmethod
    def print_grid(grid: List[List[int]], solution: Optional[List[List[int]]] = None) -> str:
        """
        Return a human-readable string representation of the maze.

        Token display:
            ██  = WALL
            ·   = PATH (cost 1)
            S   = START
            G   = GOAL
            4-9 = weighted cell (digit shown)
            *   = on optimal path (when solution provided)
        """
        n = len(grid)
        lines = []
        for r in range(n):
            row_str = []
            for c in range(n):
                on_path = solution is not None and solution[r][c] == 1
                token = grid[r][c]
                if on_path and token not in (START, GOAL):
                    row_str.append(' *')
                elif token == WALL:
                    row_str.append('██')
                elif token == PATH:
                    row_str.append(' ·')
                elif token == START:
                    row_str.append(' S')
                elif token == GOAL:
                    row_str.append(' G')
                else:
                    row_str.append(f' {token}')
            lines.append(''.join(row_str))
        return '\n'.join(lines)


# =========================================================================
# Dataset generation (mirrors Sudoku generator API)
# =========================================================================

def generate_weighted_maze_dataset(
    num_puzzles: int = 100,
    grid_size: int = 15,
    seed: Optional[int] = None,
    verbose: bool = True,
    ensure_unique: bool = True
) -> Dict[str, Any]:
    """
    Generate a dataset of weighted maze puzzles with optimal-path solutions.

    Args:
        num_puzzles: Number of puzzles to generate.
        grid_size: Size of the grid (>= 7, will be forced odd). Default 15.
        seed: Random seed for reproducibility.
        verbose: Print progress information.
        ensure_unique: If True, guarantees all puzzles are unique.

    Returns:
        dict with keys:
            - 'problems': np.array of shape (num_puzzles, grid_size, grid_size)
            - 'solutions': np.array of shape (num_puzzles, grid_size, grid_size)
            - 'metadata': list of dicts per puzzle
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    generator = WeightedMazeGenerator(grid_size=grid_size, seed=seed)
    # Update grid_size after potential odd-forcing
    actual_grid_size = generator.grid_size

    problems = []
    solutions = []
    metadata = []
    seen_puzzles: set = set()

    generated = 0
    attempts = 0
    max_attempts = num_puzzles * 20

    while generated < num_puzzles and attempts < max_attempts:
        attempts += 1
        result = generator.create_puzzle()
        if result is None:
            continue

        puzzle, solution, meta = result

        # Check uniqueness
        if ensure_unique:
            puzzle_key = tuple(tuple(row) for row in puzzle)
            if puzzle_key in seen_puzzles:
                continue
            seen_puzzles.add(puzzle_key)

        meta['puzzle_id'] = generated
        problems.append(puzzle)
        solutions.append(solution)
        metadata.append(meta)

        generated += 1
        if verbose and generated % 10 == 0:
            print(f"Generated {generated}/{num_puzzles} puzzles...")

    if generated < num_puzzles:
        print(f"Warning: Only generated {generated}/{num_puzzles} unique puzzles "
              f"after {max_attempts} attempts")

    if verbose:
        print(f"Successfully generated {generated} puzzles "
              f"(grid: {actual_grid_size}x{actual_grid_size})")

    return {
        'problems': np.array(problems, dtype=np.int8),
        'solutions': np.array(solutions, dtype=np.int8),
        'metadata': metadata
    }


def save_dataset(dataset: Dict[str, Any], filename_prefix: str,
                 save_format: str = 'npz') -> None:
    """
    Save the dataset to files.

    Args:
        dataset: The dataset dict from generate_weighted_maze_dataset.
        filename_prefix: Prefix for output files.
        save_format: 'npz' (compressed numpy) or 'npy' (separate files).
    """
    output_path = Path(filename_prefix)
    if output_path.parent and str(output_path.parent) != '.':
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == 'npz':
        np.savez_compressed(
            f'{filename_prefix}.npz',
            problems=dataset['problems'],
            solutions=dataset['solutions']
        )
        print(f"Saved to {filename_prefix}.npz")
    elif save_format == 'npy':
        np.save(f'{filename_prefix}_problems.npy', dataset['problems'])
        np.save(f'{filename_prefix}_solutions.npy', dataset['solutions'])
        print(f"Saved to {filename_prefix}_problems.npy and "
              f"{filename_prefix}_solutions.npy")

    # Save metadata
    with open(f'{filename_prefix}_metadata.txt', 'w') as f:
        if dataset['metadata']:
            first = dataset['metadata'][0]
            f.write(f"# Weighted Maze Dataset: {len(dataset['metadata'])} puzzles, "
                    f"grid_size={first['grid_size']}\n")
            f.write(f"# Token encoding: 0=WALL, 1=PATH(cost1), 2=START, "
                    f"3=GOAL, 4-9=WEIGHTED(cost=token)\n")
            f.write(f"# Solution encoding: 0=not_on_path, 1=on_optimal_path\n\n")

        for meta in dataset['metadata']:
            f.write(f"Puzzle {meta['puzzle_id']}: "
                    f"path_length={meta['path_length']}, "
                    f"path_cost={meta['path_cost']}, "
                    f"num_walls={meta['num_walls']}, "
                    f"num_weighted={meta['num_weighted']}\n")
    print(f"Metadata saved to {filename_prefix}_metadata.txt")


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate weighted maze puzzle datasets for HRM training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python weighted_maze_generator.py --num 100 --size 15
  python weighted_maze_generator.py --num 1000 --size 15 --seed 42

Token Encoding:
  0 = WALL (impassable)
  1 = PATH (traversable, cost 1)
  2 = START
  3 = GOAL
  4-9 = WEIGHTED (traversable, cost = token value)

Solution Encoding:
  0 = not on optimal path
  1 = on optimal path
"""
    )
    parser.add_argument('--num', '-n', type=int, default=100,
                        help='Number of puzzles to generate (default: 100)')
    parser.add_argument('--size', '-s', type=int, default=15,
                        help='Grid size (>= 7, forced odd; default: 15)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename prefix')
    parser.add_argument('--format', '-f', type=str, choices=['npz', 'npy'],
                        default='npz', help='Output format (default: npz)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    return parser.parse_args()


def main() -> None:
    """Main entry point for command-line usage."""
    args = parse_args()

    if args.output is None:
        args.output = f'weighted_maze_{args.size}x{args.size}'

    print(f"Generating {args.num} {args.size}x{args.size} "
          f"weighted maze puzzles...")
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")

    dataset = generate_weighted_maze_dataset(
        num_puzzles=args.num,
        grid_size=args.size,
        seed=args.seed,
        verbose=not args.quiet
    )

    save_dataset(dataset, args.output, args.format)

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Puzzles: {len(dataset['problems'])}")
    print(f"  Grid size: {dataset['problems'].shape[1]}x{dataset['problems'].shape[2]}")
    print(f"  Problems shape: {dataset['problems'].shape}")
    print(f"  Solutions shape: {dataset['solutions'].shape}")

    path_lengths = [m['path_length'] for m in dataset['metadata']]
    path_costs = [m['path_cost'] for m in dataset['metadata']]
    num_weighted = [m['num_weighted'] for m in dataset['metadata']]

    print(f"  Path length: min={min(path_lengths)}, max={max(path_lengths)}, "
          f"avg={sum(path_lengths)/len(path_lengths):.1f}")
    print(f"  Path cost:   min={min(path_costs)}, max={max(path_costs)}, "
          f"avg={sum(path_costs)/len(path_costs):.1f}")
    print(f"  Weighted cells: min={min(num_weighted)}, max={max(num_weighted)}, "
          f"avg={sum(num_weighted)/len(num_weighted):.1f}")


if __name__ == '__main__':
    main()