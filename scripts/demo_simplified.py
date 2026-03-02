"""
Demo Script for Simplified HRM (L-Module Only)

Shows how to load a trained model and solve Sudoku and Maze puzzles.

Usage:
    # Solve 4x4 Sudoku puzzles
    python scripts/demo_simplified.py --puzzle sudoku_4x4 --model model/simplified_hrm_sudoku_4x4_final.pt

    # Solve 9x9 Sudoku puzzles
    python scripts/demo_simplified.py --puzzle sudoku_9x9 --model model/simplified_hrm_sudoku_9x9_best.pt

    # Solve weighted mazes
    python scripts/demo_simplified.py --puzzle maze --model model/simplified_hrm_maze_best.pt

    # Solve mazes from an existing dataset
    python scripts/demo_simplified.py --puzzle maze --model model/simplified_hrm_maze_best.pt \
        --data data/maze_15x15_train.npz

    # Interactive mode
    python scripts/demo_simplified.py --model model/simplified_hrm_sudoku_4x4_final.pt --interactive
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Unicode box-drawing characters display correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hrm.model_simplified import SimplifiedHRM, SimplifiedHRMConfig, PuzzleType


# ── ANSI colours ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"


def load_model(model_path: str, device: torch.device) -> SimplifiedHRM:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config')
    if config is None:
        config = SimplifiedHRMConfig()

    model = SimplifiedHRM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Parameters      : {model.num_parameters:,}")
    print(f"  Hidden size     : {config.hidden_size}")
    print(f"  Layers          : {config.num_layers}")
    print(f"  Reasoning steps : {config.num_reasoning_steps}")
    return model


# ── Sudoku display helpers ──────────────────────────────────────────────────

def print_sudoku_4x4(grid: np.ndarray, title: str = ""):
    """Pretty print a 4x4 Sudoku grid."""
    if title:
        print(f"\n{title}")
    print("┌───┬───┐")
    for i in range(4):
        row = " ".join(str(int(x)) if x > 0 else '.' for x in grid[i])
        if i == 2:
            print("├───┼───┤")
        print(f"│{row[:3]}│{row[4:]}│")
    print("└───┴───┘")


def print_sudoku_9x9(grid: np.ndarray, title: str = ""):
    """Pretty print a 9x9 Sudoku grid."""
    if title:
        print(f"\n{title}")
    print("┌───────┬───────┬───────┐")
    for i in range(9):
        row = [str(int(x)) if x > 0 else '.' for x in grid[i]]
        formatted = f"│ {' '.join(row[0:3])} │ {' '.join(row[3:6])} │ {' '.join(row[6:9])} │"
        print(formatted)
        if i == 2 or i == 5:
            print("├───────┼───────┼───────┤")
    print("└───────┴───────┴───────┘")


def validate_sudoku(solution: np.ndarray, grid_size: int) -> bool:
    """Check if a Sudoku solution is valid."""
    expected = set(range(1, grid_size + 1))
    box_size = int(np.sqrt(grid_size))

    for i in range(grid_size):
        if set(solution[i]) != expected:
            return False
        if set(solution[:, i]) != expected:
            return False

    for bi in range(box_size):
        for bj in range(box_size):
            box = solution[bi*box_size:(bi+1)*box_size, bj*box_size:(bj+1)*box_size]
            if set(box.flatten()) != expected:
                return False
    return True


# ── Maze display helpers ────────────────────────────────────────────────────

_MAZE_TOKENS = {0: '█', 1: ' ', 2: 'S', 3: 'G'}


def _maze_cell(token: int) -> str:
    """Return a display character for a maze token."""
    if token in _MAZE_TOKENS:
        return _MAZE_TOKENS[token]
    return str(token)  # weighted cells 4-9


def print_maze_grid(maze_grid: np.ndarray, pred_grid: np.ndarray = None,
                    solution: np.ndarray = None, title: str = ""):
    """Print a weighted maze with optional coloured path overlay.

    Colours:
        green  = correctly predicted path cell
        red    = wrongly predicted path cell (false positive)
        cyan   = missed path cell (false negative)
        yellow = predicted path cell (when no ground truth given)
        dim    = wall
    """
    if title:
        print(f"\n{title}")
    rows, cols = maze_grid.shape
    for r in range(rows):
        parts = []
        for c in range(cols):
            tok = int(maze_grid[r, c])
            ch = _maze_cell(tok)
            if pred_grid is not None and int(pred_grid[r, c]) == 1:
                if solution is not None:
                    if int(solution[r, c]) == 1:
                        parts.append(f"{GREEN}{ch}{RESET}")
                    else:
                        parts.append(f"{RED}{ch}{RESET}")
                else:
                    parts.append(f"{YELLOW}{ch}{RESET}")
            elif (solution is not None and int(solution[r, c]) == 1
                  and (pred_grid is None or int(pred_grid[r, c]) != 1)):
                parts.append(f"{CYAN}{ch}{RESET}")   # missed path
            else:
                if tok == 0:
                    parts.append(f"{DIM}{ch}{RESET}")
                else:
                    parts.append(ch)
        print('  ' + ' '.join(parts))


# ── Sudoku demos ────────────────────────────────────────────────────────────

def demo_sudoku_4x4(model: SimplifiedHRM, device: torch.device):
    """Demo solving 4x4 Sudoku puzzles."""
    print("\n" + "="*50)
    print("4x4 SUDOKU DEMO")
    print("="*50)

    examples = [
        np.array([
            [1, 0, 0, 4],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [4, 0, 0, 1],
        ]),
        np.array([
            [0, 2, 0, 0],
            [0, 0, 3, 1],
            [1, 3, 0, 0],
            [0, 0, 1, 0],
        ]),
        np.array([
            [0, 0, 3, 0],
            [3, 0, 0, 2],
            [2, 0, 0, 1],
            [0, 1, 0, 0],
        ]),
    ]

    for i, puzzle in enumerate(examples):
        print_sudoku_4x4(puzzle, f"Puzzle {i+1}:")

        puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)

        with torch.no_grad():
            solution = model.solve_sudoku_4x4(puzzle_tensor)

        solution_np = solution[0].cpu().numpy()
        print_sudoku_4x4(solution_np, "Solution:")

        is_valid = validate_sudoku(solution_np, 4)
        print(f"Valid: {'Yes' if is_valid else 'No'}")


def demo_sudoku_9x9(model: SimplifiedHRM, device: torch.device):
    """Demo solving 9x9 Sudoku puzzles."""
    print("\n" + "="*50)
    print("9x9 SUDOKU DEMO")
    print("="*50)

    examples = [
        np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]),
    ]

    for i, puzzle in enumerate(examples):
        print_sudoku_9x9(puzzle, f"Puzzle {i+1}:")

        puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)

        with torch.no_grad():
            solution = model.solve_sudoku_9x9(puzzle_tensor)

        solution_np = solution[0].cpu().numpy()
        print_sudoku_9x9(solution_np, "Solution:")

        is_valid = validate_sudoku(solution_np, 9)
        print(f"Valid: {'Yes' if is_valid else 'No'}")


# ── Maze demo ───────────────────────────────────────────────────────────────

def demo_maze(model: SimplifiedHRM, device: torch.device,
              data_path: str = None, num_examples: int = 3):
    """Demo solving weighted maze puzzles.

    If *data_path* is given, loads puzzles from the .npz file so the
    ground-truth optimal path is available for colour-coded comparison.
    Otherwise, generates fresh mazes on the fly.
    """
    print("\n" + "="*50)
    print("WEIGHTED MAZE DEMO")
    print("="*50)
    print(f"Legend: {GREEN}green{RESET}=correct path  "
          f"{RED}red{RESET}=wrong prediction  "
          f"{CYAN}cyan{RESET}=missed path  "
          f"{DIM}dim{RESET}=wall\n")

    mazes = []       # list of (puzzle_2d, solution_2d | None, label)

    if data_path and Path(data_path).exists():
        # ── Load from dataset ──
        data = np.load(data_path)
        key = 'problems' if 'problems' in data else 'puzzles'
        problems = data[key]
        solutions = data['solutions']
        indices = np.random.choice(len(problems),
                                   size=min(num_examples, len(problems)),
                                   replace=False)
        for idx in indices:
            mazes.append((problems[idx], solutions[idx], f"Maze (dataset #{idx})"))
        print(f"Loaded {len(indices)} mazes from {data_path}\n")

    else:
        # ── Generate fresh mazes ──
        try:
            from hrm.data.weighted_maze_generator import WeightedMazeGenerator
            gen = WeightedMazeGenerator(grid_size=15, seed=42)
            print(f"Generating {num_examples} fresh 15x15 mazes...\n")
            for i in range(num_examples):
                result = gen.create_puzzle()
                if result is None:
                    print(f"  Maze generation failed on attempt {i+1}, skipping.")
                    continue
                puzzle_2d, solution_2d, meta = result
                label = (f"Maze {i+1}  (path_len={meta['path_length']}, "
                         f"cost={meta['path_cost']})")
                mazes.append((np.array(puzzle_2d), np.array(solution_2d), label))
        except ImportError:
            print("Error: generators.weighted_maze_generator not found.")
            print("Provide --data with a .npz file, or ensure the generator is on PYTHONPATH.")
            return

    if not mazes:
        print("No mazes to solve.")
        return

    # ── Solve and display ──
    for puzzle_2d, solution_2d, label in mazes:
        grid_size = puzzle_2d.shape[0]
        puzzle_flat = puzzle_2d.flatten()
        puzzle_tensor = (torch.from_numpy(puzzle_flat).long()
                         .unsqueeze(0).to(device))

        with torch.no_grad():
            pred = model.solve_maze(puzzle_tensor)

        pred_2d = pred[0].cpu().numpy().reshape(grid_size, grid_size)

        # Metrics
        if solution_2d is not None:
            total_cells = solution_2d.size
            n_correct = (pred_2d == solution_2d).sum()
            cell_acc = n_correct / total_cells

            path_mask = solution_2d == 1
            path_correct = (pred_2d[path_mask] == 1).sum() if path_mask.any() else 0
            path_total = path_mask.sum()

            fully_correct = (pred_2d == solution_2d).all()
            status = f"{GREEN}SOLVED ✓{RESET}" if fully_correct else (
                f"{n_correct}/{total_cells} cells ({cell_acc:.1%}), "
                f"path {path_correct}/{path_total}")
        else:
            status = "(no ground truth)"

        print(f"{BOLD}-- {label} -- {status}{RESET}")
        print_maze_grid(puzzle_2d, pred_grid=pred_2d, solution=solution_2d,
                        title="Model prediction:")
        if solution_2d is not None:
            print_maze_grid(puzzle_2d, pred_grid=solution_2d,
                            solution=solution_2d,
                            title="Ground truth:")
        print()


# ── Interactive mode ────────────────────────────────────────────────────────

def interactive_mode(model: SimplifiedHRM, device: torch.device):
    """Interactive puzzle solving mode."""
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Enter puzzles as comma-separated values (use 0 for empty cells)")
    print("Type 'quit' to exit\n")

    while True:
        print("\nSelect puzzle type:")
        print("  1. 4x4 Sudoku (16 values)")
        print("  2. 9x9 Sudoku (81 values)")
        print("  3. Maze (NxN values, tokens 0-9)")
        print("  q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice in ('q', 'quit'):
            break

        if choice == '1':
            grid_size = 4
            print(f"\nEnter {grid_size*grid_size} values (0-4, comma-separated):")
        elif choice == '2':
            grid_size = 9
            print(f"\nEnter {grid_size*grid_size} values (0-9, comma-separated):")
        elif choice == '3':
            try:
                grid_size = int(input("\nGrid size (e.g. 15): ").strip())
            except ValueError:
                print("Invalid grid size")
                continue
            print(f"\nEnter {grid_size*grid_size} values (0-9, comma-separated):")
            print("  Tokens: 0=wall, 1=path, 2=start, 3=goal, 4-9=weighted")
        else:
            print("Invalid choice")
            continue

        try:
            values = input("Values: ").strip()
            values = [int(x.strip()) for x in values.split(',')]

            if len(values) != grid_size * grid_size:
                print(f"Expected {grid_size*grid_size} values, got {len(values)}")
                continue

            puzzle = np.array(values).reshape(grid_size, grid_size)

            if choice == '1':
                print_sudoku_4x4(puzzle, "Input puzzle:")
                puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)
                with torch.no_grad():
                    solution = model.solve_sudoku_4x4(puzzle_tensor)
                solution_np = solution[0].cpu().numpy()
                print_sudoku_4x4(solution_np, "Solution:")
                is_valid = validate_sudoku(solution_np, 4)
                print(f"Valid: {'Yes' if is_valid else 'No'}")

            elif choice == '2':
                print_sudoku_9x9(puzzle, "Input puzzle:")
                puzzle_tensor = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)
                with torch.no_grad():
                    solution = model.solve_sudoku_9x9(puzzle_tensor)
                solution_np = solution[0].cpu().numpy()
                print_sudoku_9x9(solution_np, "Solution:")
                is_valid = validate_sudoku(solution_np, 9)
                print(f"Valid: {'Yes' if is_valid else 'No'}")

            elif choice == '3':
                print_maze_grid(puzzle, title="Input maze:")
                puzzle_tensor = (torch.from_numpy(puzzle.flatten()).long()
                                 .unsqueeze(0).to(device))
                with torch.no_grad():
                    pred = model.solve_maze(puzzle_tensor)
                pred_np = pred[0].cpu().numpy().reshape(grid_size, grid_size)
                print_maze_grid(puzzle, pred_grid=pred_np, title="Predicted path:")

        except Exception as e:
            print(f"Error: {e}")
            continue


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Demo Simplified HRM')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--puzzle', type=str, default='sudoku_4x4',
                        choices=['sudoku_4x4', 'sudoku_9x9', 'maze', 'all'],
                        help='Puzzle type to demo')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to .npz data file (used for maze demo)')
    parser.add_argument('--num-examples', type=int, default=3,
                        help='Number of maze examples to display')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, mps)')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    if args.model and Path(args.model).exists():
        model = load_model(args.model, device)
    else:
        print("\nNo trained model provided. Creating new model for demo...")
        print("Note: Predictions will be random until model is trained!\n")
        model = SimplifiedHRM()
        model = model.to(device)
        model.eval()

    # Run demo
    if args.interactive:
        interactive_mode(model, device)
    elif args.puzzle == 'sudoku_4x4':
        demo_sudoku_4x4(model, device)
    elif args.puzzle == 'sudoku_9x9':
        demo_sudoku_9x9(model, device)
    elif args.puzzle == 'maze':
        demo_maze(model, device, data_path=args.data,
                  num_examples=args.num_examples)
    else:  # all
        demo_sudoku_4x4(model, device)
        demo_sudoku_9x9(model, device)
        demo_maze(model, device, data_path=args.data,
                  num_examples=args.num_examples)


if __name__ == '__main__':
    main()