"""
Maze Generator
Generates 30x30 maze puzzles with:
- Obstructions: -1
- Exit: 0
- Start: -2
- Other cells: Manhattan distance from exit
"""

import random
from collections import deque
import numpy as np


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
    
    # Save to file
    np.savetxt('/home/claude/maze_output.txt', maze, fmt='%3d')
    print("\nMaze saved to maze_output.txt")
