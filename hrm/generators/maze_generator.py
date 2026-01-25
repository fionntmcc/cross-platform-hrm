"""
Maze Generator for HRM Training
Generates maze puzzles with solutions:
- Obstructions: -1
- Exit: 0
- Start: -2
- Other cells: Manhattan distance from exit

Solution formats:
- 'binary': Binary mask where 1 = shortest path, 0 = not on path
- 'path_marked': Same as input but path cells marked with value 1

Output structure for NN training:
- problems: np.array of shape (num_mazes, height, width)
- solutions: np.array of shape (num_mazes, height, width)

Usage:
    python maze_generator.py --num 100 --width 20 --height 20 --min-path 30 --seed 42
"""

import random
import argparse
from collections import deque
import numpy as np


def generate_single_maze(width=30, height=30, min_path_length=50):
    """
    Generate a single maze puzzle.
    
    Args:
        width: Width of the maze (number of columns)
        height: Height of the maze (number of rows)
        min_path_length: Minimum number of cells to travel from start to exit
    
    Returns:
        tuple: (maze array, start_pos, exit_pos) or (None, None, None) if failed
    """
    # Initialize maze with all walls (-1)
    maze = np.full((height, width), -1, dtype=int)
    
    # Use recursive backtracking to carve passages
    start_carve = (1, 1)
    maze[start_carve] = 1  # Mark as passage temporarily
    
    stack = [start_carve]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while stack:
        current = stack[-1]
        
        neighbors = []
        for dy, dx in directions:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 < ny < height - 1 and 0 < nx < width - 1 and maze[ny, nx] == -1:
                neighbors.append((ny, nx))
        
        if neighbors:
            next_cell = random.choice(neighbors)
            wall_y = (current[0] + next_cell[0]) // 2
            wall_x = (current[1] + next_cell[1]) // 2
            maze[wall_y, wall_x] = 1
            maze[next_cell] = 1
            stack.append(next_cell)
        else:
            stack.pop()
    
    # Find all passage cells
    passage_cells = [(y, x) for y in range(height) for x in range(width) if maze[y, x] != -1]
    
    # Place exit and start with minimum path length requirement
    exit_pos, start_pos = place_start_exit(maze, width, height, min_path_length, passage_cells)
    
    if exit_pos is None:
        return None, None, None
    
    # Calculate Manhattan distances from exit for all passage cells
    for y, x in passage_cells:
        if (y, x) != exit_pos and (y, x) != start_pos:
            distance = abs(y - exit_pos[0]) + abs(x - exit_pos[1])
            maze[y, x] = distance
    
    # Set exit and start values
    maze[exit_pos] = 0
    maze[start_pos] = -2
    
    return maze, start_pos, exit_pos


def find_shortest_path(maze, start, end):
    """
    Find the shortest path between start and end using BFS.
    
    Returns:
        list: List of (y, x) coordinates representing the path, or empty list if no path
    """
    height, width = maze.shape
    visited = set()
    # Store (position, path_so_far)
    queue = deque([(start, [start])])
    visited.add(start)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (y, x), path = queue.popleft()
        
        if (y, x) == end:
            return path
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and 
                (ny, nx) not in visited and maze[ny, nx] != -1):
                visited.add((ny, nx))
                queue.append(((ny, nx), path + [(ny, nx)]))
    
    return []


def find_path_length(maze, start, end):
    """
    Find the shortest path length between start and end using BFS.
    Returns -1 if no path exists.
    """
    path = find_shortest_path(maze, start, end)
    return len(path) - 1 if path else -1


def place_start_exit(maze, width, height, min_path_length, passage_cells):
    """
    Place start and exit positions ensuring minimum path length.
    """
    random.shuffle(passage_cells)
    
    # Prefer corners and edges for exit
    corners_edges = [c for c in passage_cells if 
                    c[0] <= 2 or c[0] >= height - 3 or c[1] <= 2 or c[1] >= width - 3]
    
    exit_candidates = corners_edges if corners_edges else passage_cells[:len(passage_cells)//2]
    
    for exit_pos in exit_candidates:
        for start_pos in passage_cells:
            if start_pos == exit_pos:
                continue
            
            path_len = find_path_length(maze, start_pos, exit_pos)
            
            if path_len >= min_path_length:
                return exit_pos, start_pos
    
    # If no valid placement found, try with any two distant points
    for exit_pos in passage_cells:
        for start_pos in reversed(passage_cells):
            if start_pos == exit_pos:
                continue
            
            path_len = find_path_length(maze, start_pos, exit_pos)
            
            if path_len >= min_path_length:
                return exit_pos, start_pos
    
    return None, None


def create_solution_matrix(maze, start_pos, exit_pos, solution_format='binary'):
    """
    Create a solution matrix showing the shortest path.
    
    Args:
        maze: The maze array
        start_pos: Starting position (y, x)
        exit_pos: Exit position (y, x)
        solution_format: 'binary' or 'path_marked'
            - 'binary': 1 = on path, 0 = not on path
            - 'path_marked': Copy of maze with path cells set to 1
    
    Returns:
        numpy array representing the solution
    """
    height, width = maze.shape
    path = find_shortest_path(maze, start_pos, exit_pos)
    
    if solution_format == 'binary':
        solution = np.zeros((height, width), dtype=int)
        for y, x in path:
            solution[y, x] = 1
    
    elif solution_format == 'path_marked':
        solution = maze.copy()
        for y, x in path:
            solution[y, x] = 1
        # Keep start and exit marked
        solution[start_pos] = -2
        solution[exit_pos] = 0
    
    else:
        raise ValueError(f"Unknown solution format: {solution_format}")
    
    return solution


def generate_maze_dataset(
    num_mazes=100,
    width=20,
    height=20,
    min_path_length=30,
    seed=None,
    solution_format='binary',
    verbose=True
):
    """
    Generate a dataset of mazes with solutions for neural network training.
    
    Args:
        num_mazes: Number of mazes to generate
        width: Width of each maze
        height: Height of each maze
        min_path_length: Minimum path length from start to exit
        seed: Random seed for reproducibility
        solution_format: 'binary' or 'path_marked'
        verbose: Print progress information
    
    Returns:
        dict with keys:
            - 'problems': np.array of shape (num_mazes, height, width)
            - 'solutions': np.array of shape (num_mazes, height, width)
            - 'metadata': list of dicts with start_pos, exit_pos, path_length for each maze
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    problems = []
    solutions = []
    metadata = []
    
    generated = 0
    attempts = 0
    max_attempts = num_mazes * 10  # Prevent infinite loops
    
    while generated < num_mazes and attempts < max_attempts:
        attempts += 1
        
        maze, start_pos, exit_pos = generate_single_maze(width, height, min_path_length)
        
        if maze is None:
            continue
        
        solution = create_solution_matrix(maze, start_pos, exit_pos, solution_format)
        path_length = find_path_length(maze, start_pos, exit_pos)
        
        problems.append(maze)
        solutions.append(solution)
        metadata.append({
            'start_pos': start_pos,
            'exit_pos': exit_pos,
            'path_length': path_length
        })
        
        generated += 1
        
        if verbose and generated % 10 == 0:
            print(f"Generated {generated}/{num_mazes} mazes...")
    
    if generated < num_mazes:
        print(f"Warning: Only generated {generated}/{num_mazes} mazes after {max_attempts} attempts")
    
    return {
        'problems': np.array(problems),
        'solutions': np.array(solutions),
        'metadata': metadata
    }


def save_dataset(dataset, filename_prefix, save_format='npz'):
    """
    Save the dataset to files.
    
    Args:
        dataset: The dataset dict from generate_maze_dataset
        filename_prefix: Prefix for output files
        save_format: 'npz' (compressed numpy) or 'npy' (separate files)
    """
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
        print(f"Saved to {filename_prefix}_problems.npy and {filename_prefix}_solutions.npy")
    
    # Save metadata as text
    with open(f'{filename_prefix}_metadata.txt', 'w') as f:
        for i, meta in enumerate(dataset['metadata']):
            f.write(f"Maze {i}: start={meta['start_pos']}, exit={meta['exit_pos']}, path_length={meta['path_length']}\n")
    print(f"Metadata saved to {filename_prefix}_metadata.txt")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate maze datasets for neural network training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 mazes of size 20x20
  python maze_generator.py --num 100 --width 20 --height 20

  # Generate with specific path length and seed
  python maze_generator.py --num 50 --min-path 40 --seed 123

  # Generate with path_marked solution format
  python maze_generator.py --num 100 --solution-format path_marked

  # Save as both npz and npy formats
  python maze_generator.py --num 100 --format both

Visualize generated mazes with:
  python maze_visualizer.py maze_dataset.npz --num 5
        """
    )
    
    # Generation parameters
    parser.add_argument('-n', '--num', type=int, default=5,
                        help='Number of mazes to generate (default: 5)')
    parser.add_argument('-W', '--width', type=int, default=20,
                        help='Width of maze in cells (default: 20)')
    parser.add_argument('-H', '--height', type=int, default=20,
                        help='Height of maze in cells (default: 20)')
    parser.add_argument('-m', '--min-path', type=int, default=30,
                        help='Minimum path length from start to exit (default: 30)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: random)')
    parser.add_argument('-f', '--solution-format', choices=['binary', 'path_marked'],
                        default='binary',
                        help='Solution format: binary or path_marked (default: binary)')
    
    # Output parameters
    parser.add_argument('-o', '--output', type=str, default='maze_dataset',
                        help='Output filename prefix (default: maze_dataset)')
    parser.add_argument('--format', choices=['npz', 'npy', 'both'], default='npz',
                        help='Save format: npz, npy, or both (default: npz)')
    
    # Other options
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress output')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not args.quiet:
        print("=" * 60)
        print("MAZE DATASET GENERATOR")
        print("=" * 60)
        print(f"\nParameters:")
        print(f"  Number of mazes: {args.num}")
        print(f"  Dimensions: {args.width}x{args.height}")
        print(f"  Minimum path length: {args.min_path}")
        print(f"  Seed: {args.seed if args.seed else 'random'}")
        print(f"  Solution format: {args.solution_format}")
        print(f"  Output: {args.output}")
        print()
    
    # Generate dataset
    dataset = generate_maze_dataset(
        num_mazes=args.num,
        width=args.width,
        height=args.height,
        min_path_length=args.min_path,
        seed=args.seed,
        solution_format=args.solution_format,
        verbose=not args.quiet
    )
    
    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Problems shape: {dataset['problems'].shape}")
        print(f"Solutions shape: {dataset['solutions'].shape}")
        print(f"Data type: {dataset['problems'].dtype}")
        
        path_lengths = [m['path_length'] for m in dataset['metadata']]
        print(f"\nPath length statistics:")
        print(f"  Min: {min(path_lengths)}")
        print(f"  Max: {max(path_lengths)}")
        print(f"  Mean: {np.mean(path_lengths):.1f}")
    
    # Save dataset
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SAVING DATASET")
        print("=" * 60)
    
    if args.format in ['npz', 'both']:
        save_dataset(dataset, args.output, save_format='npz')
    if args.format in ['npy', 'both']:
        save_dataset(dataset, args.output, save_format='npy')
    
    if not args.quiet:
        print(f"\nTo visualize: python maze_visualizer.py {args.output}.npz")
