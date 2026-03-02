"""
Visualise Simplified HRM Reasoning Steps

Shows the model's predicted grid after each reasoning step, so you can
watch how the solution evolves over the iterative refinement process.

The Simplified HRM uses prediction feedback (self-conditioning): after
each step the model re-embeds its current predictions merged with the
original givens, enabling iterative self-correction like denoising in
diffusion models.

Usage:
    # 4x4 Sudoku
    python scripts/visualise_simplified.py \
        --model model/simplified_hrm_sudoku_4x4_final.pt --puzzle sudoku_4x4

    # 9x9 Sudoku
    python scripts/visualise_simplified.py \
        --model model/simplified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9

    # Weighted maze (from dataset)
    python scripts/visualise_simplified.py \
        --model model/simplified_hrm_maze_best.pt --puzzle maze \
        --data data/maze_15x15_train.npz

    # Custom puzzle (row-major, 0=empty)
    python scripts/visualise_simplified.py \
        --model model/simplified_hrm_sudoku_4x4_final.pt --puzzle sudoku_4x4 \
        --input "1,0,0,4,0,0,1,0,0,1,0,0,4,0,0,1"

    # Slow playback
    python scripts/visualise_simplified.py \
        --model model/simplified_hrm_sudoku_4x4_final.pt --puzzle sudoku_4x4 --delay 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.model_simplified import SimplifiedHRM, SimplifiedHRMConfig, PuzzleType, PUZZLE_DEFAULTS

# ── ANSI colours ────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
GREY    = "\033[90m"


# ── Grid Printers ───────────────────────────────────────────────────────────

def colour_cell(value: int, original: int, target: int = -1) -> str:
    """Colour a cell: dim=given, green=correct, red=wrong, yellow=filled."""
    s = str(value) if value > 0 else "."
    if original > 0:
        return f"{DIM}{s}{RESET}"
    if value == 0:
        return f"{GREY}.{RESET}"
    if target > 0:
        return f"{GREEN}{s}{RESET}" if value == target else f"{RED}{s}{RESET}"
    return f"{YELLOW}{s}{RESET}"


def print_sudoku_grid(grid, original, target=None, grid_size=9):
    """Print a Sudoku grid with box lines and colour coding."""
    box = int(np.sqrt(grid_size))
    seg = "-" * (2 * box + 1)
    print("+" + "+".join([seg] * box) + "+")
    for i in range(grid_size):
        parts = []
        for bj in range(box):
            cells = []
            for j in range(box):
                col = bj * box + j
                t = int(target[i, col]) if target is not None else -1
                cells.append(colour_cell(int(grid[i, col]), int(original[i, col]), t))
            parts.append(" ".join(cells))
        print("| " + " | ".join(parts) + " |")
        if (i + 1) % box == 0 and i < grid_size - 1:
            print("+" + "+".join([seg] * box) + "+")
    print("+" + "+".join([seg] * box) + "+")


def compute_accuracy(pred, target, original):
    """Cell-level accuracy on empty cells only (Sudoku) or all cells (Maze)."""
    mask = original.flatten() == 0
    if mask.sum() == 0:
        # Maze or fully given — compare all cells
        return float((pred.flatten() == target.flatten()).mean())
    return float((pred.flatten()[mask] == target.flatten()[mask]).mean())


# ── Maze token symbols ──────────────────────────────────────────────────────

_MAZE_TOKENS = {0: '█', 1: ' ', 2: 'S', 3: 'G'}

def _maze_cell(token: int) -> str:
    """Return a display character for a maze token."""
    if token in _MAZE_TOKENS:
        return _MAZE_TOKENS[token]
    return str(token)  # weighted cells 4-9


def print_maze_grid(maze_grid, pred_grid=None, target_grid=None):
    """Print a weighted-maze grid with optional path overlay.

    If *pred_grid* is not None, cells predicted as on-path (1) are shown
    in colour.  If *target_grid* is also given, correct path cells are
    green and wrong predictions are red.
    """
    rows, cols = maze_grid.shape
    for r in range(rows):
        parts = []
        for c in range(cols):
            tok = int(maze_grid[r, c])
            ch = _maze_cell(tok)
            if pred_grid is not None and int(pred_grid[r, c]) == 1:
                if target_grid is not None:
                    if int(target_grid[r, c]) == 1:
                        parts.append(f"{GREEN}{ch}{RESET}")
                    else:
                        parts.append(f"{RED}{ch}{RESET}")
                else:
                    parts.append(f"{YELLOW}{ch}{RESET}")
            elif target_grid is not None and int(target_grid[r, c]) == 1 and (pred_grid is None or int(pred_grid[r, c]) != 1):
                # Missed path cell
                parts.append(f"{CYAN}{ch}{RESET}")
            else:
                if tok == 0:
                    parts.append(f"{DIM}{ch}{RESET}")
                else:
                    parts.append(ch)
        print(' '.join(parts))


# ── Sample puzzles ──────────────────────────────────────────────────────────

SAMPLE_4X4 = np.array([
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [4, 0, 0, 1],
])

SAMPLE_9X9 = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])


# ── Main logic ──────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> SimplifiedHRM:
    """Load trained SimplifiedHRM from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', SimplifiedHRMConfig())
    model = SimplifiedHRM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, config


def visualise(model: SimplifiedHRM, config: SimplifiedHRMConfig,
              puzzle_flat: np.ndarray, puzzle_type: PuzzleType,
              target_flat: np.ndarray = None,
              num_steps: int = None, delay: float = 0.3,
              grid_size: int = None):
    """
    Run the model and print the grid after each reasoning step.

    Args:
        model: Trained SimplifiedHRM.
        config: Model config.
        puzzle_flat: 1-D input array (row-major).
        puzzle_type: Which puzzle type.
        target_flat: Optional 1-D target solution for colour coding.
        num_steps: Override number of reasoning steps.
        delay: Seconds to pause between printing each step.
        grid_size: Grid side length (auto-detected from puzzle_flat if None).
    """
    device = next(model.parameters()).device
    is_maze = puzzle_type == PuzzleType.MAZE

    if grid_size is None:
        defaults = PUZZLE_DEFAULTS[puzzle_type]
        grid_size = defaults['grid_size']
    grid_shape = (grid_size, grid_size)

    x = torch.from_numpy(puzzle_flat).long().unsqueeze(0).to(device)
    n_steps = num_steps or config.num_reasoning_steps

    with torch.no_grad():
        output = model(
            x,
            puzzle_type=puzzle_type,
            num_reasoning_steps=n_steps,
            return_intermediates=True,
        )

    step_preds = output['intermediates']['step_predictions']
    original_grid = puzzle_flat.reshape(grid_shape)
    target_grid = target_flat.reshape(grid_shape) if target_flat is not None else None

    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}  Simplified HRM Step-by-Step Visualisation{RESET}")
    print(f"{BOLD}{'='*50}{RESET}")
    print(f"  Puzzle type      : {puzzle_type.name}")
    print(f"  Grid size        : {grid_size}x{grid_size}")
    print(f"  Reasoning steps  : {n_steps}")
    print(f"  Model layers     : {config.num_layers}")
    print(f"  Hidden size      : {config.hidden_size}")
    print(f"  Pred. feedback   : {'On' if config.use_prediction_feedback else 'Off'}")

    # Colour legend
    if is_maze:
        print(f"\n  Legend: {GREEN}green{RESET}=correct path  {RED}red{RESET}=wrong path  "
              f"{CYAN}cyan{RESET}=missed path  {DIM}dim{RESET}=wall")
    else:
        print(f"\n  Legend: {DIM}dim{RESET}=given  ", end="")
        if target_grid is not None:
            print(f"{GREEN}green{RESET}=correct  {RED}red{RESET}=wrong")
        else:
            print(f"{YELLOW}yellow{RESET}=predicted")

    # Show input
    print(f"\n{BOLD}-- Input Puzzle --{RESET}")
    if is_maze:
        print_maze_grid(original_grid, target_grid=target_grid)
    else:
        print_sudoku_grid(original_grid, original_grid, target_grid, grid_size)

    # Show each reasoning step
    for i, preds_tensor in enumerate(step_preds):
        preds = preds_tensor[0].cpu().numpy().reshape(grid_shape)

        if delay > 0 and i > 0:
            time.sleep(delay)

        label = "Final Output" if i == n_steps - 1 else f"Step {i + 1}/{n_steps}"

        acc_str = ""
        if target_grid is not None:
            if is_maze:
                acc = float((preds.flatten() == target_grid.flatten()).mean())
            else:
                acc = compute_accuracy(preds, target_grid, original_grid)
            acc_str = f"  (accuracy: {acc:.1%})"

        # Count changes from previous step
        changes_str = ""
        if i > 0:
            prev = step_preds[i-1][0].cpu().numpy().reshape(grid_shape)
            n_changed = (preds != prev).sum()
            changes_str = f"  [{n_changed} cells changed]"

        print(f"\n{BOLD}-- {label}{acc_str}{changes_str} --{RESET}")
        if is_maze:
            print_maze_grid(original_grid, pred_grid=preds, target_grid=target_grid)
        else:
            print_sudoku_grid(preds, original_grid, target_grid, grid_size)

    # Summary
    if target_grid is not None:
        final_preds = step_preds[-1][0].cpu().numpy().reshape(grid_shape)
        if is_maze:
            final_acc = float((final_preds.flatten() == target_grid.flatten()).mean())
            n_total = target_grid.size
            n_correct = (final_preds.flatten() == target_grid.flatten()).sum()
            print(f"\n{BOLD}Summary:{RESET}  {n_correct}/{n_total} cells correct ({final_acc:.1%})")
        else:
            final_acc = compute_accuracy(final_preds, target_grid, original_grid)
            n_empty = (original_grid == 0).sum()
            n_correct = ((final_preds.flatten() == target_grid.flatten()) &
                         (original_grid.flatten() == 0)).sum()
            print(f"\n{BOLD}Summary:{RESET}  {n_correct}/{n_empty} empty cells correct ({final_acc:.1%})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Visualise Simplified HRM reasoning at each step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--puzzle', type=str, required=True,
                        choices=['sudoku_4x4', 'sudoku_9x9', 'maze'],
                        help='Puzzle type')
    parser.add_argument('--input', type=str, default=None,
                        help='Puzzle values as comma-separated integers (row-major, 0=empty)')
    parser.add_argument('--target', type=str, default=None,
                        help='Target solution as comma-separated integers (for colour coding)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to .npz data file (picks a random example for maze)')
    parser.add_argument('--index', type=int, default=None,
                        help='Index of puzzle to pick from --data file')
    parser.add_argument('--steps', type=int, default=None,
                        help='Override number of reasoning steps')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Seconds to pause between steps (default: 0.3)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu, cuda, mps)')

    args = parser.parse_args()
    device = torch.device(args.device)

    puzzle_type_map = {
        'sudoku_4x4': PuzzleType.SUDOKU_4X4,
        'sudoku_9x9': PuzzleType.SUDOKU_9X9,
        'maze': PuzzleType.MAZE,
    }
    puzzle_type = puzzle_type_map[args.puzzle]

    # Load model
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, device)
    print(f"  {model.num_parameters:,} parameters")

    # Determine grid size
    grid_size = None  # auto-detect

    # Parse or use sample puzzle
    target_flat = None

    if args.data:
        # Load from .npz dataset
        data = np.load(args.data)
        key_problems = 'problems' if 'problems' in data else 'puzzles'
        key_solutions = 'solutions'
        problems = data[key_problems]
        solutions = data[key_solutions]
        idx = args.index if args.index is not None else np.random.randint(len(problems))
        print(f"  Using puzzle {idx} from {args.data}")
        puzzle_flat = problems[idx].flatten().astype(np.int64)
        target_flat = solutions[idx].flatten().astype(np.int64)
        grid_size = problems[idx].shape[0]
    elif args.input:
        puzzle_flat = np.array([int(x.strip()) for x in args.input.split(',')], dtype=np.int64)
    else:
        if puzzle_type == PuzzleType.SUDOKU_4X4:
            puzzle_flat = SAMPLE_4X4.flatten()
        elif puzzle_type == PuzzleType.SUDOKU_9X9:
            puzzle_flat = SAMPLE_9X9.flatten()
        else:
            print("Error: Maze visualisation requires --data or --input")
            sys.exit(1)

    if args.target:
        target_flat = np.array([int(x.strip()) for x in args.target.split(',')], dtype=np.int64)

    visualise(
        model, config, puzzle_flat, puzzle_type,
        target_flat=target_flat,
        num_steps=args.steps,
        delay=args.delay,
        grid_size=grid_size,
    )


if __name__ == '__main__':
    main()
