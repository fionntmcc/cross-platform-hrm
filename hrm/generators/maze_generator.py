"""
Maze Generator
Generates NxN maze puzzles with:
- Obstructions: -1
- Exit: 0
- Start: -2
- Other cells: Manhattan distance from exit
"""

import random
from collections import deque
import numpy as np
import argparse
import os


def generate_maze(width=30, height=30, min_path_length=50, seed=None):
    """
    Generate a maze puzzle with specified parameters.
    
    Args:
        width: Width of the maze (number of columns)
        height: Height of the maze (number of rows)
        min_path_length: Minimum number of cells to travel from start to exit
        seed: Random seed for reproducibility
    
    Returns:
        numpy array representing the maze
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize maze with all walls (-1)
    maze = np.full((height, width), -1, dtype=int)
    
    # Use recursive backtracking to carve passages
    # Start carving from position (1, 1)
    start_carve = (1, 1)
    maze[start_carve] = 1  # Mark as passage temporarily
    
    # Stack for backtracking: stores (current_cell, list_of_unvisited_neighbors)
    stack = [start_carve]
    
    # Directions: up, right, down, left (moving by 2 to skip walls)
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while stack:
        current = stack[-1]
        
        # Find unvisited neighbors (2 cells away)
        neighbors = []
        for dy, dx in directions:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 < ny < height - 1 and 0 < nx < width - 1 and maze[ny, nx] == -1:
                neighbors.append((ny, nx))
        
        if neighbors:
            # Choose random neighbor
            next_cell = random.choice(neighbors)
            
            # Carve passage to neighbor (remove wall between)
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
        raise ValueError(f"Could not place start and exit with minimum path length {min_path_length}")
    
    # Calculate Manhattan distances from exit for all passage cells
    for y, x in passage_cells:
        if (y, x) != exit_pos and (y, x) != start_pos:
            distance = abs(y - exit_pos[0]) + abs(x - exit_pos[1])
            maze[y, x] = distance
    
    # Set exit and start values
    maze[exit_pos] = 0
    maze[start_pos] = -2
    
    return maze, start_pos, exit_pos


def find_path_length(maze, start, end):
    """
    Find the shortest path length between start and end using BFS.
    Returns -1 if no path exists.
    """
    height, width = maze.shape
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (y, x), dist = queue.popleft()
        
        if (y, x) == end:
            return dist
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and 
                (ny, nx) not in visited and maze[ny, nx] != -1):
                visited.add((ny, nx))
                queue.append(((ny, nx), dist + 1))
    
    return -1


def place_start_exit(maze, width, height, min_path_length, passage_cells):
    """
    Place start and exit positions ensuring minimum path length.
    """
    # Try multiple combinations to find valid start/exit positions
    random.shuffle(passage_cells)
    
    # Prefer corners and edges for exit
    corners_edges = [c for c in passage_cells if 
                    c[0] <= 2 or c[0] >= height - 3 or c[1] <= 2 or c[1] >= width - 3]
    
    if corners_edges:
        exit_candidates = corners_edges
    else:
        exit_candidates = passage_cells[:len(passage_cells)//2]
    
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


def print_maze(maze):
    """Print the maze in a readable format."""
    size = maze.shape[0]
    
    print("\nMaze Legend: -1=Wall, 0=Exit, -2=Start, other=distance from exit")
    print("-" * (size * 4 + 1))
    
    for row in maze:
        print("|", end="")
        for cell in row:
            if cell == -1:
                print(" ██", end="|")
            elif cell == -2:
                print(" S ", end="|")
            elif cell == 0:
                print(" E ", end="|")
            else:
                print(f"{cell:3}", end="|")
        print()
    print("-" * (size * 4 + 1))


def print_maze_compact(maze):
    """Print a more compact visual representation."""
    print("\nCompact view (█=wall, S=start, E=exit, ·=path):")
    for row in maze:
        for cell in row:
            if cell == -1:
                print("██", end="")
            elif cell == -2:
                print("S ", end="")
            elif cell == 0:
                print("E ", end="")
            else:
                print("· ", end="")
        print()


def get_maze_stats(maze, start_pos, exit_pos):
    """Get statistics about the generated maze."""
    height, width = maze.shape
    total_cells = height * width
    wall_cells = np.sum(maze == -1)
    passage_cells = total_cells - wall_cells
    
    path_length = find_path_length(maze, start_pos, exit_pos)
    
    return {
        'width': width,
        'height': height,
        'total_cells': total_cells,
        'wall_cells': wall_cells,
        'passage_cells': passage_cells,
        'wall_percentage': wall_cells / total_cells * 100,
        'path_length': path_length,
        'start_pos': start_pos,
        'exit_pos': exit_pos
    }


if __name__ == "__main__":
    # Parameters
    MAZE_WIDTH = 30   # Number of columns
    MAZE_HEIGHT = 30  # Number of rows
    MIN_PATH_LENGTH = 50  # Minimum cells to travel from start to exit
    SEED = 42  # For reproducibility
    
    print(f"Generating {MAZE_WIDTH}x{MAZE_HEIGHT} maze...")
    print(f"Minimum path length: {MIN_PATH_LENGTH}")
    print(f"Seed: {SEED}")
    
    # Generate maze
    maze, start_pos, exit_pos = generate_maze(
        width=MAZE_WIDTH,
        height=MAZE_HEIGHT,
        min_path_length=MIN_PATH_LENGTH,
        seed=SEED
    )
    
    # Get and print statistics
    stats = get_maze_stats(maze, start_pos, exit_pos)
    print(f"\nMaze Statistics:")
    print(f"  Dimensions: {stats['width']}x{stats['height']}")
    print(f"  Wall cells: {stats['wall_cells']} ({stats['wall_percentage']:.1f}%)")
    print(f"  Passage cells: {stats['passage_cells']}")
    print(f"  Start position: {stats['start_pos']}")
    print(f"  Exit position: {stats['exit_pos']}")
    print(f"  Actual path length: {stats['path_length']}")
    
    # Print compact visualization
    print_maze_compact(maze)
    
    # Print raw matrix
    print("\nRaw maze matrix:")
    print(maze)


def generate_dataset(n_mazes, width, height, min_path_length, seed=None):
    """
    Generate a dataset of mazes.
    
    Args:
        n_mazes: Number of mazes to generate
        width: Width of each maze
        height: Height of each maze
        min_path_length: Minimum path length for each maze
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (problems_array, solutions_array) where each is shape (n_mazes, height, width)
    """
    problems = []
    solutions = []
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for i in range(n_mazes):
        if (i + 1) % max(1, n_mazes // 10) == 0:
            print(f"  Generated {i + 1}/{n_mazes} mazes...")
        
        maze, start_pos, exit_pos = generate_maze(
            width=width,
            height=height,
            min_path_length=min_path_length,
            seed=None if seed is None else seed + i
        )
        
        # Problem is the maze itself
        problems.append(maze)
        
        # Solution is a binary map showing the shortest path
        solution = np.zeros((height, width), dtype=int)
        path = find_shortest_path(maze, start_pos, exit_pos)
        if path:
            for y, x in path:
                solution[y, x] = 1
        solutions.append(solution)
    
    return np.array(problems), np.array(solutions)


def find_shortest_path(maze, start, end):
    """
    Find the shortest path from start to end using BFS.
    Returns list of coordinates forming the path.
    """
    height, width = maze.shape
    visited = {start: None}
    queue = deque([start])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current = queue.popleft()
        
        if current == end:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = visited[node]
            return list(reversed(path))
        
        y, x = current
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and 
                (ny, nx) not in visited and maze[ny, nx] != -1):
                visited[(ny, nx)] = current
                queue.append((ny, nx))
    
    return None


def save_dataset(problems, solutions, output_prefix):
    """
    Save the dataset to files.
    
    Args:
        problems: Numpy array of problem mazes
        solutions: Numpy array of solution paths
        output_prefix: Path prefix for output files (will add .npz extension)
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as compressed npz
    npz_path = f"{output_prefix}.npz"
    np.savez_compressed(
        npz_path,
        problems=problems,
        solutions=solutions
    )
    print(f"Saved to {npz_path}")
    
    # Also save as separate npy files for flexibility
    problems_path = f"{output_prefix}_problems.npy"
    solutions_path = f"{output_prefix}_solutions.npy"
    np.save(problems_path, problems)
    np.save(solutions_path, solutions)
    print(f"Also saved to {problems_path} and {solutions_path}")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate maze datasets for neural network training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 mazes and save to training_data.npz
  python maze_generator.py -n 1000 -W 20 -H 20 -o training_data
  
  # Generate single maze with custom dimensions
  python maze_generator.py -n 1 -W 40 -H 40 -s 42
  
  # Generate to specific output directory
  python maze_generator.py -n 500 -W 30 -H 30 -o ../data/mazes
        """
    )
    
    parser.add_argument('-n', '--num', type=int, default=1,
                        help='Number of mazes to generate (default: 1)')
    
    parser.add_argument('-W', '--width', type=int, default=30,
                        help='Width of mazes (default: 30)')
    
    parser.add_argument('-H', '--height', type=int, default=30,
                        help='Height of mazes (default: 30)')
    
    parser.add_argument('-m', '--min-path', type=int, default=50,
                        help='Minimum path length (default: 50)')
    
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    parser.add_argument('-o', '--output', type=str, default='mazes',
                        help='Output file prefix (default: mazes)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num < 1:
        print("Error: --num must be at least 1")
        return
    
    if args.width < 5 or args.height < 5:
        print("Error: Width and height must be at least 5")
        return
    
    # Generate dataset
    print(f"Generating {args.num} mazes ({args.width}x{args.height})...")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    
    problems, solutions = generate_dataset(
        n_mazes=args.num,
        width=args.width,
        height=args.height,
        min_path_length=args.min_path,
        seed=args.seed
    )
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Total mazes: {args.num}")
    print(f"Maze dimensions: {args.width}x{args.height}")
    print(f"Problems shape: {problems.shape}")
    print(f"Solutions shape: {solutions.shape}")
    
    # Save dataset
    print(f"\n{'='*60}")
    print(f"SAVING DATASET")
    print(f"{'='*60}")
    save_dataset(problems, solutions, args.output)
    
    print(f"\n{'='*60}")
    print(f"USAGE IN TRAINING SCRIPT")
    print(f"{'='*60}")
    print(f"""
# Load the dataset:
data = np.load('{args.output}.npz')
problems = data['problems']      # Shape: ({args.num}, {args.height}, {args.width})
solutions = data['solutions']    # Shape: ({args.num}, {args.height}, {args.width})

# For PyTorch (add channel dimension for CNN):
problems = torch.from_numpy(problems[:, np.newaxis, :, :]).float()
solutions = torch.from_numpy(solutions[:, np.newaxis, :, :]).float()
    """)


if __name__ == "__main__":
    main()

