# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Visualise HRM Reasoning Steps

Shows the model's predicted grid after each H-cycle, so you can watch
how the solution evolves over the hierarchical iteration timesteps.

Works with any puzzle type: 4x4 Sudoku, 9x9 Sudoku, or Mazes.

Usage:
    # 9x9 Sudoku (from trained model)
    python scripts/visualise_steps.py --model model/unified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9

    # 4x4 Sudoku
    python scripts/visualise_steps.py --model model/unified_hrm_sudoku_4x4_best.pt --puzzle sudoku_4x4

    # Custom number of H-cycles to see more intermediate steps
    python scripts/visualise_steps.py --model model/unified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9 --h-cycles 8

    # Provide a puzzle inline (row-major, 0=empty)
    python scripts/visualise_steps.py --model model/unified_hrm_sudoku_9x9_best.pt --puzzle sudoku_9x9 \
        --input "5,3,0,0,7,0,0,0,0,6,0,0,1,9,5,0,0,0,0,9,8,0,0,0,0,6,0,8,0,0,0,6,0,0,0,3,4,0,0,8,0,3,0,0,1,7,0,0,0,2,0,0,0,6,0,6,0,0,0,0,2,8,0,0,0,0,4,1,9,0,0,5,0,0,0,0,8,0,0,7,9"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.model_unified import UnifiedHRM, UnifiedHRMConfig, PuzzleType, PUZZLE_CONFIGS

# ── ANSI colours ────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREY = "\033[90m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"


# ── Grid Printers ───────────────────────────────────────────────────────────


def _colour_cell(value: int, original: int, target: int = -1) -> str:
    """Colour a cell value: grey if given, green if correct, red if wrong, yellow if filled."""
    s = str(value) if value > 0 else "."
    if original > 0:
        # Given clue — keep dim
        return f"{DIM}{s}{RESET}"
    if value == 0:
        return f"{GREY}.{RESET}"
    if target > 0:
        if value == target:
            return f"{GREEN}{s}{RESET}"
        else:
            return f"{RED}{s}{RESET}"
    # Filled but no target to compare
    return f"{YELLOW}{s}{RESET}"


def print_sudoku_grid(
    grid: np.ndarray, original: np.ndarray, target: np.ndarray = None, grid_size: int = 9
):
    """Print a Sudoku grid with box lines and colour coding."""
    box_h = int(np.sqrt(grid_size))
    box_w = grid_size // box_h  # handles non-square boxes (e.g., 2x3 for 6x6)

    # Top border
    seg = "─" * (2 * box_w + 1)
    print("┌" + "┬".join([seg] * box_h) + "┐")

    for i in range(grid_size):
        row_parts = []
        for bj in range(box_h):
            cells = []
            for j in range(box_w):
                col = bj * box_w + j
                t = int(target[i, col]) if target is not None else -1
                cells.append(_colour_cell(int(grid[i, col]), int(original[i, col]), t))
            row_parts.append(" ".join(cells))
        print("│ " + " │ ".join(row_parts) + " │")

        # Box row separator
        if (i + 1) % box_h == 0 and i < grid_size - 1:
            print("├" + "┼".join([seg] * box_h) + "┤")

    print("└" + "┴".join([seg] * box_h) + "┘")


def print_maze_grid(grid: np.ndarray, original: np.ndarray, target: np.ndarray = None):
    """Print a maze grid with symbols."""
    MAZE_SYMBOLS = {0: "█", 1: " ", 2: "S", 3: "G"}
    rows, cols = grid.shape
    print("┌" + "─" * cols + "┐")
    for i in range(rows):
        row = ""
        for j in range(cols):
            v = int(grid[i, j])
            o = int(original[i, j])
            sym = MAZE_SYMBOLS.get(v, "?")
            if o == v:
                row += f"{DIM}{sym}{RESET}"
            elif target is not None:
                t = int(target[i, j])
                if v == t:
                    row += f"{GREEN}{sym}{RESET}"
                else:
                    row += f"{RED}{sym}{RESET}"
            else:
                row += f"{YELLOW}{sym}{RESET}"
        print(f"│{row}│")
    print("└" + "─" * cols + "┘")


def print_grid(
    grid: np.ndarray, original: np.ndarray, puzzle_type: PuzzleType, target: np.ndarray = None
):
    """Dispatch to the right printer."""
    if puzzle_type in (PuzzleType.SUDOKU_4X4, PuzzleType.SUDOKU_9X9):
        gs = PUZZLE_CONFIGS[puzzle_type]["grid_size"]
        print_sudoku_grid(grid, original, target, grid_size=gs)
    else:
        print_maze_grid(grid, original, target)


# ── Accuracy helpers ────────────────────────────────────────────────────────


def compute_accuracy(pred: np.ndarray, target: np.ndarray, original: np.ndarray) -> float:
    """Cell-level accuracy on empty cells only."""
    mask = original.flatten() == 0
    if mask.sum() == 0:
        return 1.0
    return float((pred.flatten()[mask] == target.flatten()[mask]).mean())


# ── Sample puzzles ──────────────────────────────────────────────────────────

SAMPLE_4X4 = np.array(
    [
        [1, 0, 0, 4],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [4, 0, 0, 1],
    ]
)

SAMPLE_9X9 = np.array(
    [
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
)


# ── Main logic ──────────────────────────────────────────────────────────────


def load_model(model_path: str, device: torch.device) -> UnifiedHRM:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", UnifiedHRMConfig())
    model = UnifiedHRM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def parse_puzzle_type(name: str) -> PuzzleType:
    mapping = {
        "sudoku_4x4": PuzzleType.SUDOKU_4X4,
        "sudoku_9x9": PuzzleType.SUDOKU_9X9,
        "maze": PuzzleType.MAZE,
    }
    if name not in mapping:
        raise ValueError(f"Unknown puzzle type '{name}'. Choose from: {list(mapping.keys())}")
    return mapping[name]


def get_grid_shape(puzzle_type: PuzzleType, seq_len: int):
    """Return (rows, cols) for reshaping flat predictions to a grid."""
    cfg = PUZZLE_CONFIGS[puzzle_type]
    if cfg["grid_size"] is not None:
        gs = cfg["grid_size"]
        return (gs, gs)
    # Maze: infer square grid from seq_len
    side = int(np.sqrt(seq_len))
    assert side * side == seq_len, f"Cannot infer grid shape from seq_len={seq_len}"
    return (side, side)


def visualise(
    model: UnifiedHRM,
    puzzle_flat: np.ndarray,
    puzzle_type: PuzzleType,
    target_flat: np.ndarray = None,
    h_cycles: int = None,
    l_steps: int = None,
    halt_max_steps: int = None,
    delay: float = 0.0,
):
    """
    Run the model and print the grid after each outer ACT step.

    Args:
        model: Trained UnifiedHRM.
        puzzle_flat: 1-D input array (row-major).
        puzzle_type: Which puzzle type.
        target_flat: Optional 1-D target solution for colour coding.
        h_cycles: Override number of H-cycles (more = more snapshots).
        l_steps: Override L-steps per cycle.
        delay: Seconds to pause between printing each step.
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(puzzle_flat).long().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            x,
            puzzle_type=puzzle_type,
            halt_max_steps=halt_max_steps,
            H_cycles=h_cycles,
            L_cycles=l_steps,
            return_intermediates=True,
        )

    step_preds = output["intermediates"]["step_predictions"]
    q_halt_logits = output["intermediates"].get("q_halt_logits", [])
    grid_shape = get_grid_shape(puzzle_type, puzzle_flat.shape[0])
    original_grid = puzzle_flat.reshape(grid_shape)
    target_grid = target_flat.reshape(grid_shape) if target_flat is not None else None

    total_steps = len(step_preds)
    outer_steps = output["outer_steps_used"]
    h_per_step = output["h_cycles_per_step"]
    l_per_step = output["l_cycles_per_step"]

    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}  HRM Step-by-Step Visualisation{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"  Puzzle type    : {puzzle_type.name}")
    print(f"  Grid size      : {grid_shape[0]}×{grid_shape[1]}")
    print(f"  Outer ACT steps: {outer_steps}")
    print(f"  H×L per step   : {h_per_step}×{l_per_step}")
    print(f"  Total steps    : {total_steps}")

    # Show input
    print(f"\n{BOLD}── Input Puzzle ──{RESET}")
    print_grid(original_grid, original_grid, puzzle_type, target_grid)

    # Show each intermediate step
    for i, preds_tensor in enumerate(step_preds):
        preds = preds_tensor[0].cpu().numpy().reshape(grid_shape)

        if delay > 0 and i > 0:
            time.sleep(delay)

        label = f"Final Output" if i == total_steps - 1 else f"After outer step {i + 1}"
        # Show Q-halt value if available
        q_str = ""
        if i < len(q_halt_logits):
            q_val = torch.sigmoid(q_halt_logits[i]).mean().item()
            q_str = f"  [Q(halt)={q_val:.3f}]"
        acc_str = ""
        if target_grid is not None:
            acc = compute_accuracy(preds, target_grid, original_grid)
            acc_str = f"  (empty-cell accuracy: {acc:.1%})"

        print(f"\n{BOLD}── {label}{q_str}{acc_str} ──{RESET}")
        print_grid(preds, original_grid, puzzle_type, target_grid)

    # Summary
    if target_grid is not None:
        final_preds = step_preds[-1][0].cpu().numpy().reshape(grid_shape)
        final_acc = compute_accuracy(final_preds, target_grid, original_grid)
        n_empty = (original_grid == 0).sum()
        n_correct = (
            (final_preds.flatten() == target_grid.flatten()) & (original_grid.flatten() == 0)
        ).sum()
        print(
            f"\n{BOLD}Summary:{RESET}  {n_correct}/{n_empty} empty cells correct ({final_acc:.1%})"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualise HRM reasoning at each timestep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--puzzle",
        type=str,
        required=True,
        choices=["sudoku_4x4", "sudoku_9x9", "maze"],
        help="Puzzle type",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Puzzle values as comma-separated integers (row-major, 0=empty). "
        "If omitted, uses a built-in sample puzzle.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target solution as comma-separated integers (for colour coding)",
    )
    parser.add_argument(
        "--halt-max-steps",
        type=int,
        default=None,
        help="Override max outer ACT halting steps (more = more snapshots)",
    )
    parser.add_argument(
        "--h-cycles", type=int, default=None, help="Override H-cycles per inner call"
    )
    parser.add_argument("--l-steps", type=int, default=None, help="Override L-cycles per H-cycle")
    parser.add_argument(
        "--delay", type=float, default=0.3, help="Seconds to pause between steps (default: 0.3)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")

    args = parser.parse_args()
    device = torch.device(args.device)
    puzzle_type = parse_puzzle_type(args.puzzle)

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    print(f"  {model.num_parameters:,} parameters")

    # Parse or use sample puzzle
    if args.input:
        puzzle_flat = np.array([int(x.strip()) for x in args.input.split(",")], dtype=np.int64)
    else:
        if puzzle_type == PuzzleType.SUDOKU_4X4:
            puzzle_flat = SAMPLE_4X4.flatten()
        elif puzzle_type == PuzzleType.SUDOKU_9X9:
            puzzle_flat = SAMPLE_9X9.flatten()
        else:
            print("No built-in sample maze. Please provide --input.")
            sys.exit(1)

    # Parse target if given
    target_flat = None
    if args.target:
        target_flat = np.array([int(x.strip()) for x in args.target.split(",")], dtype=np.int64)

    visualise(
        model,
        puzzle_flat,
        puzzle_type,
        target_flat=target_flat,
        h_cycles=args.h_cycles,
        l_steps=args.l_steps,
        halt_max_steps=args.halt_max_steps,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
