# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Run trained Simplified HRM on Sudoku or Maze puzzles.

Loads a trained checkpoint, evaluates on held-out puzzles from the dataset,
and displays example solutions side-by-side with ground truth.

Usage:
    python scripts/run_simplified.py --model model/simplified_hrm_sudoku_4x4_final.pt --puzzle sudoku_4x4
    python scripts/run_simplified.py --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9
    python scripts/run_simplified.py --model model/simplified_hrm_maze_best.pt --puzzle maze --data data/maze_15x15_train.npz
    python scripts/run_simplified.py --model model/simplified_hrm_maze_best.pt --puzzle maze --data data/samples/sample_maze_15x15.json
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

from hrm.model_simplified import PuzzleType, SimplifiedHRM, SimplifiedHRMConfig

# ── ANSI helpers ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"


# ── Pretty-printing ──────────────────────────────────────────────────────────

def fmt_cell(val, given, correct):
    """Format a single cell: given=bold, correct=green, wrong=red."""
    v = str(int(val))
    if given:
        return f"\033[1m{v}\033[0m"  # bold (given clue)
    elif correct:
        return f"\033[92m{v}\033[0m"  # green
    else:
        return f"\033[91m{v}\033[0m"  # red


def print_9x9(grid, given_mask=None, solution=None, title=""):
    """Pretty-print a 9x9 grid with colour-coded cells."""
    if title:
        print(f"\n  {title}")
    print("  ┌───────┬───────┬───────┐")
    for i in range(9):
        cells = []
        for j in range(9):
            is_given = given_mask[i, j] if given_mask is not None else False
            is_correct = (grid[i, j] == solution[i, j]) if solution is not None else True
            cells.append(fmt_cell(grid[i, j], is_given, is_correct))
        row = f"  │ {cells[0]} {cells[1]} {cells[2]} │ {cells[3]} {cells[4]} {cells[5]} │ {cells[6]} {cells[7]} {cells[8]} │"
        print(row)
        if i in (2, 5):
            print("  ├───────┼───────┼───────┤")
    print("  └───────┴───────┴───────┘")


def print_4x4(grid, given_mask=None, solution=None, title=""):
    """Pretty-print a 4x4 grid with colour-coded cells."""
    if title:
        print(f"\n  {title}")
    print("  ┌───┬───┐")
    for i in range(4):
        cells = []
        for j in range(4):
            is_given = given_mask[i, j] if given_mask is not None else False
            is_correct = (grid[i, j] == solution[i, j]) if solution is not None else True
            cells.append(fmt_cell(grid[i, j], is_given, is_correct))
        row = f"  │{cells[0]} {cells[1]}│{cells[2]} {cells[3]}│"
        print(row)
        if i == 1:
            print("  ├───┼───┤")
    print("  └───┴───┘")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_sudoku(sol, size):
    """Check if a Sudoku solution is valid."""
    expected = set(range(1, size + 1))
    box = int(size ** 0.5)
    for i in range(size):
        if set(int(x) for x in sol[i]) != expected:
            return False
        if set(int(x) for x in sol[:, i]) != expected:
            return False
    for bi in range(box):
        for bj in range(box):
            block = sol[bi*box:(bi+1)*box, bj*box:(bj+1)*box]
            if set(int(x) for x in block.flatten()) != expected:
                return False
    return True


# ── Maze printing ─────────────────────────────────────────────────────────────

_MAZE_TOKENS = {0: '█', 1: ' ', 2: 'S', 3: 'G'}


def _maze_cell(token: int) -> str:
    if token in _MAZE_TOKENS:
        return _MAZE_TOKENS[token]
    return str(token)


def print_maze(maze_grid, pred_grid=None, solution=None, title=""):
    """Print a weighted maze with path overlay."""
    if title:
        print(f"\n  {title}")
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
            elif solution is not None and int(solution[r, c]) == 1 and (
                pred_grid is None or int(pred_grid[r, c]) != 1
            ):
                parts.append(f"{CYAN}{ch}{RESET}")  # missed path
            else:
                if tok == 0:
                    parts.append(f"{DIM}{ch}{RESET}")
                else:
                    parts.append(ch)
        print('  ' + ' '.join(parts))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run trained Simplified HRM on Sudoku or Maze puzzles")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--puzzle", type=str, default="sudoku_4x4",
                        choices=["sudoku_4x4", "sudoku_9x9", "maze"])
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz data file (auto-detected if omitted)")
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Number of example puzzles to display")
    parser.add_argument("--eval-count", type=int, default=200,
                        help="Number of puzzles to evaluate for accuracy stats")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    is_maze = args.puzzle == "maze"

    if is_maze:
        grid_size = None  # determined from data
        puzzle_type = PuzzleType.MAZE
        print_grid = None  # will use print_maze
    else:
        grid_size = 9 if "9x9" in args.puzzle else 4
        puzzle_type = PuzzleType.SUDOKU_9X9 if grid_size == 9 else PuzzleType.SUDOKU_4X4
        print_grid = print_9x9 if grid_size == 9 else print_4x4

    # ── Load model ────────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    print(f"Loading checkpoint: {model_path}")

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    config = checkpoint.get("config")
    if config is None:
        config = SimplifiedHRMConfig()

    model = SimplifiedHRM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_accuracy", None)
    print(f"  Parameters      : {model.num_parameters:,}")
    print(f"  Epoch           : {epoch}")
    print(f"  Hidden size     : {config.hidden_size}")
    print(f"  Layers          : {config.num_layers}")
    print(f"  Reasoning steps : {config.num_reasoning_steps}")
    if val_acc is not None:
        print(f"  Val Acc         : {val_acc:.4f}")
    print(f"  Device          : {device}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    data_path = args.data
    if data_path is None:
        if is_maze:
            candidates = [
                PROJECT_ROOT / "data" / "maze_15x15_train.npz",
            ]
        else:
            candidates = [
                PROJECT_ROOT / "data" / f"{args.puzzle}_train.npz",
                PROJECT_ROOT / "data" / f"{args.puzzle}_mixed_5000.npz",
                PROJECT_ROOT / "data" / f"{args.puzzle}_mixed_50000.npz",
            ]
        for c in candidates:
            if c.exists():
                data_path = c
                break
    else:
        data_path = Path(data_path)

    if data_path is None or not Path(data_path).exists():
        print("Error: No data file found. Please provide a --data path to a .npz/.json/.csv file.")
        sys.exit(1)
    else:
        ext = Path(data_path).suffix.lower()
        if ext in (".json", ".csv"):
            from hrm.data.io import load_dataset

            ds = load_dataset(data_path)
            puzzles = ds["problems"]
            solutions = ds["solutions"]
        else:
            npz = np.load(str(data_path))
            key = "problems" if "problems" in npz else "puzzles"
            puzzles = npz[key]
            solutions = npz["solutions"]
        if is_maze:
            grid_size = puzzles.shape[1]
        print(f"Loaded {len(puzzles)} puzzles from {data_path}")

    # ── Batch evaluation ──────────────────────────────────────────────────
    n_eval = min(args.eval_count, len(puzzles))
    # Use the LAST n_eval puzzles (least likely to have been heavily trained on)
    eval_idx = np.arange(len(puzzles) - n_eval, len(puzzles))
    eval_puzzles = puzzles[eval_idx]
    eval_solutions = solutions[eval_idx]

    print(f"\nEvaluating on {n_eval} puzzles ...")
    batch_size = 64
    all_preds = []
    with torch.no_grad():
        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)
            batch = torch.from_numpy(eval_puzzles[start:end]).long().view(end - start, -1).to(device)
            out = model.forward(batch, puzzle_type)
            preds = out["predictions"].cpu().numpy().reshape(-1, grid_size, grid_size)
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)

    # Metrics
    if is_maze:
        # Maze: binary output — compare all cells
        total_cells = eval_solutions.size
        cell_correct = (all_preds == eval_solutions).sum()
        cell_acc = cell_correct / total_cells

        puzzle_correct = sum(
            (all_preds[i] == eval_solutions[i]).all() for i in range(n_eval)
        )

        print(f"\n{'='*50}")
        print(f"  EVALUATION RESULTS  ({n_eval} mazes, {grid_size}x{grid_size})")
        print(f"{'='*50}")
        print(f"  Cell accuracy   : {cell_acc:.1%}")
        print(f"  Puzzle accuracy : {puzzle_correct}/{n_eval}  ({puzzle_correct/n_eval:.1%})")
        print(f"{'='*50}")
    else:
        empty_masks = (eval_puzzles == 0)
        token_correct = (all_preds == eval_solutions) & empty_masks
        total_empty = empty_masks.sum()
        token_acc = token_correct.sum() / total_empty if total_empty > 0 else 0

        puzzle_correct = 0
        valid_solutions = 0
        for i in range(n_eval):
            mask_i = empty_masks[i]
            if (all_preds[i][mask_i] == eval_solutions[i][mask_i]).all():
                puzzle_correct += 1
            if validate_sudoku(all_preds[i], grid_size):
                valid_solutions += 1

        print(f"\n{'='*50}")
        print(f"  EVALUATION RESULTS  ({n_eval} puzzles)")
        print(f"{'='*50}")
        print(f"  Token accuracy (empty cells) : {token_acc:.1%}")
        print(f"  Puzzle accuracy (all correct): {puzzle_correct}/{n_eval}  ({puzzle_correct/n_eval:.1%})")
        print(f"  Valid Sudoku solutions       : {valid_solutions}/{n_eval}  ({valid_solutions/n_eval:.1%})")
        print(f"{'='*50}")

    # ── Display examples ──────────────────────────────────────────────────
    n_show = args.num_examples
    show_idx = np.random.choice(len(eval_puzzles), size=min(n_show, len(eval_puzzles)), replace=False)
    show_puzzles = eval_puzzles[show_idx]
    show_solutions = eval_solutions[show_idx]
    show_preds = all_preds[show_idx]

    print(f"\n{'='*50}")
    print("  EXAMPLE SOLUTIONS")
    if is_maze:
        print("  Legend: \033[92mgreen\033[0m=correct path  \033[91mred\033[0m=wrong path  \033[96mcyan\033[0m=missed")
    else:
        print("  Legend: \033[1mbold\033[0m=given  \033[92mgreen\033[0m=correct  \033[91mred\033[0m=wrong")
    print(f"{'='*50}")

    for i in range(min(n_show, len(show_puzzles))):
        puzzle_np = show_puzzles[i]
        pred_np = show_preds[i]

        if is_maze:
            sol_np = show_solutions[i]
            n_total = pred_np.size
            n_correct = (pred_np == sol_np).sum()
            print(f"\n  -- Maze {i+1} -- {n_correct}/{n_total} cells correct --")
            print_maze(puzzle_np, pred_grid=pred_np, solution=sol_np,
                       title="Model prediction:")
            print_maze(puzzle_np, pred_grid=sol_np, solution=sol_np,
                       title="Ground truth:")
        else:
            given_mask = (puzzle_np != 0)
            # Merge givens into prediction display
            display = pred_np.copy()
            display[given_mask] = puzzle_np[given_mask]

            sol_np = show_solutions[i]
            empty_mask = ~given_mask
            n_empty = empty_mask.sum()
            n_correct = ((display == sol_np) & empty_mask).sum()
            is_valid = validate_sudoku(display, grid_size)

            print(f"\n  -- Puzzle {i+1} ({'valid' if is_valid else 'invalid'}) "
                  f"-- {n_correct}/{n_empty} empty cells correct --")
            print_grid(display, given_mask=given_mask, solution=sol_np,
                       title="Model prediction:")
            print_grid(sol_np, given_mask=given_mask, solution=sol_np,
                       title="Ground truth:")


if __name__ == "__main__":
    main()
