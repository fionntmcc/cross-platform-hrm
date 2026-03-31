"""
ONNX Inference & Benchmarking for Simplified HRM

Runs trained SimplifiedHRM models (exported via export_onnx.py) on CPU
using ONNX Runtime.  Designed for Raspberry Pi 5 and edge devices.

**No PyTorch dependency** -- only numpy and onnxruntime required.

Supports all three puzzle types:
    - sudoku_4x4: 16 tokens, vocab 5, 4x4 grid display
    - sudoku_9x9: 81 tokens, vocab 10, 9x9 grid display
    - maze:       grid_size^2 tokens, binary output, path overlay display

Four modes:
    - Demo (default): generate and solve random puzzles with colour output
    - Dataset eval:   load .npz, compute accuracy, show sample predictions
    - Benchmark:      warmup + timed runs, latency stats, throughput
    - Interactive:    type puzzle values, get instant solution

Usage:
    # Demo -- solve random puzzles
    python scripts/solve_onnx.py \\
        --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9

    # Evaluate on dataset
    python scripts/solve_onnx.py \\
        --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \\
        --data data/sudoku_9x9_train.npz

    # Benchmark latency
    python scripts/solve_onnx.py \\
        --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \\
        --benchmark

    # Interactive mode
    python scripts/solve_onnx.py \\
        --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \\
        --interactive

    # Maze (specify grid size)
    python scripts/solve_onnx.py \\
        --model model/simplified_hrm_maze_11x11_s4.onnx --puzzle maze \\
        --maze-size 11 --benchmark

Authors:
    - Kyrylo Kozlovskyi (G00425385)
    - Fionn McCarthy (G00414386)
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed.")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


# -----------------------------------------------------------------------
# Puzzle config
# -----------------------------------------------------------------------
PUZZLE_CONFIG = {
    "sudoku_4x4": {"seq_len": 16, "grid_size": 4, "vocab_size": 5, "type": "sudoku"},
    "sudoku_9x9": {"seq_len": 81, "grid_size": 9, "vocab_size": 10, "type": "sudoku"},
    "maze": {"seq_len": None, "grid_size": None, "vocab_size": 10, "type": "maze"},
}

# ANSI colour codes
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# -----------------------------------------------------------------------
# Sudoku display
# -----------------------------------------------------------------------
def format_sudoku_grid(flat, grid_size):
    """Format a flat array as a Sudoku grid string with box lines."""
    grid = flat.reshape(grid_size, grid_size)
    box = int(grid_size ** 0.5)
    lines = []

    for r in range(grid_size):
        if r > 0 and r % box == 0:
            lines.append("-" * (grid_size * 2 + box - 1))
        row_parts = []
        for c in range(grid_size):
            if c > 0 and c % box == 0:
                row_parts.append("|")
            val = grid[r, c]
            row_parts.append(str(val) if val != 0 else ".")
        lines.append(" ".join(row_parts))

    return "\n".join(lines)


def colour_sudoku_diff(puzzle, prediction, solution, grid_size):
    """Format Sudoku grid with coloured predictions.

    Colours: dim=given clue, green=correct fill, red=wrong fill.
    """
    grid_pred = prediction.reshape(grid_size, grid_size)
    grid_puz = puzzle.reshape(grid_size, grid_size)
    grid_sol = solution.reshape(grid_size, grid_size)
    box = int(grid_size ** 0.5)
    lines = []

    for r in range(grid_size):
        if r > 0 and r % box == 0:
            lines.append("-" * (grid_size * 2 + box - 1))
        row_parts = []
        for c in range(grid_size):
            if c > 0 and c % box == 0:
                row_parts.append("|")
            val = grid_pred[r, c]
            if grid_puz[r, c] != 0:
                row_parts.append("{}{}{}".format(DIM, val, RESET))
            elif val == grid_sol[r, c]:
                row_parts.append("{}{}{}".format(GREEN, val, RESET))
            else:
                row_parts.append("{}{}{}".format(RED, val, RESET))
        lines.append(" ".join(row_parts))

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Maze display
# -----------------------------------------------------------------------
_MAZE_TOKENS = {0: "\u2588", 1: " ", 2: "S", 3: "G"}


def _maze_cell(token):
    """Return a display character for a maze token."""
    if token in _MAZE_TOKENS:
        return _MAZE_TOKENS[token]
    return str(token)  # weighted cells 4-9


def format_maze_grid(maze_flat, grid_size, pred_flat=None, target_flat=None):
    """Format a maze grid with optional path overlay.

    Colours:
        green  = correct path cell
        red    = predicted path but wrong
        cyan   = missed path cell (in target but not predicted)
        dim    = wall
    """
    maze = maze_flat.reshape(grid_size, grid_size)
    pred = pred_flat.reshape(grid_size, grid_size) if pred_flat is not None else None
    target = target_flat.reshape(grid_size, grid_size) if target_flat is not None else None

    lines = []
    for r in range(grid_size):
        parts = []
        for c in range(grid_size):
            tok = int(maze[r, c])
            ch = _maze_cell(tok)

            if pred is not None and int(pred[r, c]) == 1:
                if target is not None:
                    if int(target[r, c]) == 1:
                        parts.append("{}{}{}".format(GREEN, ch, RESET))
                    else:
                        parts.append("{}{}{}".format(RED, ch, RESET))
                else:
                    parts.append("{}{}{}".format(YELLOW, ch, RESET))
            elif (
                target is not None
                and int(target[r, c]) == 1
                and (pred is None or int(pred[r, c]) != 1)
            ):
                parts.append("{}{}{}".format(CYAN, ch, RESET))
            else:
                if tok == 0:
                    parts.append("{}{}{}".format(DIM, ch, RESET))
                else:
                    parts.append(ch)
        lines.append(" ".join(parts))

    return "\n".join(lines)


def compute_maze_metrics(prediction, solution):
    """Compute maze-specific metrics.

    Returns dict with cell_accuracy, path_precision, path_recall, path_f1.
    """
    pred = prediction.flatten()
    sol = solution.flatten()

    cell_acc = float((pred == sol).mean())

    # Path-specific metrics (class 1 = on-path)
    true_path = sol == 1
    pred_path = pred == 1

    tp = int((pred_path & true_path).sum())
    fp = int((pred_path & ~true_path).sum())
    fn = int((~pred_path & true_path).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "cell_accuracy": cell_acc,
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": f1,
    }


# -----------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------
class ONNXSolver:
    """ONNX Runtime solver for SimplifiedHRM."""

    def __init__(self, model_path):
        """Load ONNX model."""
        print("Loading ONNX model: {}".format(model_path))
        file_size = Path(model_path).stat().st_size / 1024 ** 2
        print("  File size: {:.1f} MB".format(file_size))

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        print("  Input:  {} {} ({})".format(inp.name, inp.shape, inp.type))
        print("  Output: {} {} ({})".format(out.name, out.shape, out.type))
        print("  Provider: {}".format(self.session.get_providers()[0]))

    def solve(self, puzzle):
        """Solve puzzle(s).

        Args:
            puzzle: Integer tokens, shape (batch, seq_len) or (seq_len,).

        Returns:
            Predictions as numpy array, shape (batch, seq_len).
        """
        if puzzle.ndim == 1:
            puzzle = puzzle.reshape(1, -1)

        puzzle = puzzle.astype(np.int64)
        result = self.session.run(None, {"puzzle": puzzle})[0]
        return result

    def benchmark(self, seq_len, vocab_size, num_warmup=5, num_runs=50):
        """Benchmark inference latency.

        Returns dict with mean_ms, std_ms, min_ms, max_ms, median_ms,
        throughput.
        """
        inp = np.random.randint(0, vocab_size, (1, seq_len)).astype(np.int64)

        for _ in range(num_warmup):
            self.solve(inp)

        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.solve(inp)
            times.append(time.perf_counter() - t0)

        times_ms = np.array(times) * 1000
        return {
            "mean_ms": float(np.mean(times_ms)),
            "std_ms": float(np.std(times_ms)),
            "min_ms": float(np.min(times_ms)),
            "max_ms": float(np.max(times_ms)),
            "median_ms": float(np.median(times_ms)),
            "throughput": float(1000 / np.mean(times_ms)),
        }


# -----------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------
def demo_random(solver, puzzle_type, grid_size, num_puzzles=3):
    """Generate and solve random puzzles."""
    cfg = PUZZLE_CONFIG[puzzle_type]
    seq_len = grid_size * grid_size
    is_maze = cfg["type"] == "maze"

    # Try to use the Sudoku generator if available
    generator_available = False
    if not is_maze:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from generators.sudoku_generator import SudokuGenerator, Difficulty

            gen = SudokuGenerator(grid_size=grid_size)
            generator_available = True
        except ImportError:
            pass

    print("\nSolving {} random {} puzzles:\n".format(num_puzzles, puzzle_type))

    for i in range(num_puzzles):
        if generator_available:
            puzzle_2d, solution_2d = gen.create_puzzle(difficulty=Difficulty.EASY)
            puzzle = np.array(puzzle_2d, dtype=np.int64).flatten()
            solution = np.array(solution_2d, dtype=np.int64).flatten()
        else:
            puzzle = np.random.randint(
                0, cfg["vocab_size"], (seq_len,)
            ).astype(np.int64)
            solution = None

        t0 = time.perf_counter()
        prediction = solver.solve(puzzle)[0]
        latency = (time.perf_counter() - t0) * 1000

        print("--- Puzzle {} ---".format(i + 1))
        print("Input:")

        if is_maze:
            print(format_maze_grid(puzzle, grid_size, pred_flat=prediction,
                                   target_flat=solution))
        else:
            print(format_sudoku_grid(puzzle, grid_size))

        if solution is not None:
            if is_maze:
                metrics = compute_maze_metrics(prediction, solution)
                solved = "yes" if metrics["cell_accuracy"] == 1.0 else "no"
                print(
                    "\nCell acc: {:.2%}  |  Path F1: {:.2%}  |  "
                    "Solved: {}  ({:.1f} ms)".format(
                        metrics["cell_accuracy"], metrics["path_f1"],
                        solved, latency,
                    )
                )
            else:
                correct = (prediction == solution).all()
                token_acc = float((prediction == solution).mean())
                print("\nPrediction ({:.1f} ms):".format(latency))
                print(colour_sudoku_diff(puzzle, prediction, solution, grid_size))
                status = "\u2713" if correct else "\u2717"
                print(
                    "\nToken accuracy: {:.2%}  |  Fully correct: {}".format(
                        token_acc, status,
                    )
                )
        else:
            print("\nPrediction ({:.1f} ms):".format(latency))
            if is_maze:
                print(format_maze_grid(puzzle, grid_size, pred_flat=prediction))
            else:
                print(format_sudoku_grid(prediction, grid_size))

        print()


def evaluate_dataset(solver, data_path, puzzle_type, grid_size):
    """Evaluate ONNX model on a dataset."""
    cfg = PUZZLE_CONFIG[puzzle_type]
    is_maze = cfg["type"] == "maze"

    print("\nLoading dataset: {}".format(data_path))
    data = np.load(data_path)

    if "puzzles" in data:
        puzzles = data["puzzles"]
        solutions = data["solutions"]
    elif "problems" in data:
        puzzles = data["problems"]
        solutions = data["solutions"]
    else:
        print("  ERROR: Unknown keys in npz: {}".format(list(data.keys())))
        return

    if puzzles.ndim == 3:
        puzzles = puzzles.reshape(puzzles.shape[0], -1)
        solutions = solutions.reshape(solutions.shape[0], -1)

    n = len(puzzles)
    print("  Samples: {}".format(n))
    print("  Grid size: {}x{}".format(grid_size, grid_size))

    # Solve all
    batch_size = 64
    all_preds = []
    t0 = time.perf_counter()

    for i in range(0, n, batch_size):
        batch = puzzles[i : i + batch_size].astype(np.int64)
        preds = solver.solve(batch)
        all_preds.append(preds)

    total_time = time.perf_counter() - t0
    predictions = np.concatenate(all_preds, axis=0)

    # Compute metrics
    token_correct = float((predictions == solutions).mean())
    puzzle_correct = float((predictions == solutions).all(axis=1).mean())

    print("\n{}".format("=" * 55))
    print("  ONNX Evaluation Results")
    print("{}".format("=" * 55))
    print("  Token accuracy:       {:.4%}".format(token_correct))
    print("  Puzzle accuracy:      {:.4%}".format(puzzle_correct))

    if is_maze:
        all_metrics = compute_maze_metrics(predictions, solutions)
        print("  Path precision:       {:.4%}".format(all_metrics["path_precision"]))
        print("  Path recall:          {:.4%}".format(all_metrics["path_recall"]))
        print("  Path F1:              {:.4%}".format(all_metrics["path_f1"]))
    else:
        empty_mask = puzzles == 0
        if empty_mask.any():
            empty_acc = float(
                (predictions[empty_mask] == solutions[empty_mask]).mean()
            )
            print("  Empty-cell accuracy:  {:.4%}".format(empty_acc))

    print("  Total time:           {:.2f}s".format(total_time))
    print("  Per-puzzle latency:   {:.1f} ms".format(total_time / n * 1000))
    print("  Throughput:           {:.0f} puzzles/sec".format(n / total_time))
    print("{}".format("=" * 55))

    # Show a few examples
    num_show = min(3, n)
    print("\nSample predictions (first {}):\n".format(num_show))
    for i in range(num_show):
        correct = (predictions[i] == solutions[i]).all()
        status = "\u2713" if correct else "\u2717"

        print("--- Sample {} ---".format(i + 1))
        if is_maze:
            metrics = compute_maze_metrics(predictions[i], solutions[i])
            print("Maze with predicted path overlay:")
            print(
                format_maze_grid(
                    puzzles[i], grid_size,
                    pred_flat=predictions[i],
                    target_flat=solutions[i],
                )
            )
            print(
                "\nCell acc: {:.2%}  |  Path F1: {:.2%}  |  "
                "Correct: {}".format(
                    metrics["cell_accuracy"], metrics["path_f1"], status,
                )
            )
        else:
            tok_acc = float((predictions[i] == solutions[i]).mean())
            print("Input:")
            print(format_sudoku_grid(puzzles[i], grid_size))
            print("\nPrediction:")
            print(
                colour_sudoku_diff(
                    puzzles[i], predictions[i], solutions[i], grid_size
                )
            )
            print("\nToken acc: {:.2%}  |  Correct: {}".format(tok_acc, status))
        print()


def benchmark_mode(solver, puzzle_type, grid_size):
    """Run latency benchmark."""
    cfg = PUZZLE_CONFIG[puzzle_type]
    seq_len = grid_size * grid_size

    print("\n{}".format("=" * 55))
    print("  ONNX Benchmark -- {}".format(puzzle_type))
    if cfg["type"] == "maze":
        print("  Grid: {}x{} ({} tokens)".format(grid_size, grid_size, seq_len))
    print("{}".format("=" * 55))
    print("  Platform:   {}".format(platform.machine()))
    print("  Processor:  {}".format(platform.processor() or "unknown"))
    print("  Python:     {}".format(platform.python_version()))
    print("  ORT:        {}".format(ort.__version__))

    results = solver.benchmark(
        seq_len=seq_len,
        vocab_size=cfg["vocab_size"],
        num_warmup=10,
        num_runs=100,
    )

    print("\n  Latency (100 runs):")
    print("    Mean:   {:.1f} ms".format(results["mean_ms"]))
    print("    Std:    {:.1f} ms".format(results["std_ms"]))
    print("    Min:    {:.1f} ms".format(results["min_ms"]))
    print("    Max:    {:.1f} ms".format(results["max_ms"]))
    print("    Median: {:.1f} ms".format(results["median_ms"]))
    print("  Throughput: {:.0f} puzzles/sec".format(results["throughput"]))
    print("{}".format("=" * 55))

    return results


def interactive_mode(solver, puzzle_type, grid_size):
    """Interactive puzzle solving."""
    cfg = PUZZLE_CONFIG[puzzle_type]
    is_maze = cfg["type"] == "maze"
    seq_len = grid_size * grid_size

    print("\nInteractive {} solver (ONNX)".format(puzzle_type))
    print("Grid: {}x{} ({} values)".format(grid_size, grid_size, seq_len))
    print("Enter {} values (0=empty), comma-separated.".format(seq_len))
    print("Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("{}> ".format(puzzle_type)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break

        try:
            values = [int(v.strip()) for v in raw.split(",")]
            if len(values) != seq_len:
                print("  Expected {} values, got {}".format(seq_len, len(values)))
                continue

            puzzle = np.array(values, dtype=np.int64).reshape(1, -1)
            t0 = time.perf_counter()
            prediction = solver.solve(puzzle)[0]
            latency = (time.perf_counter() - t0) * 1000

            print("\nPrediction ({:.1f} ms):".format(latency))
            if is_maze:
                print(
                    format_maze_grid(
                        puzzle[0], grid_size, pred_flat=prediction
                    )
                )
            else:
                print(format_sudoku_grid(prediction, grid_size))
            print()

        except ValueError:
            print("  Invalid input. Use comma-separated integers.")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX inference & benchmarking for SimplifiedHRM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Solve random Sudoku puzzles
  python scripts/solve_onnx.py \\
      --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9

  # Evaluate on dataset
  python scripts/solve_onnx.py \\
      --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \\
      --data data/sudoku_9x9_train.npz

  # Benchmark latency
  python scripts/solve_onnx.py \\
      --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \\
      --benchmark

  # Maze with grid size
  python scripts/solve_onnx.py \\
      --model model/simplified_hrm_maze_11x11_s4.onnx --puzzle maze \\
      --maze-size 11 --benchmark

  # Interactive mode
  python scripts/solve_onnx.py \\
      --model model/simplified_hrm_sudoku_9x9.onnx --puzzle sudoku_9x9 \\
      --interactive
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .onnx model file",
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        required=True,
        choices=["sudoku_4x4", "sudoku_9x9", "maze"],
        help="Puzzle type",
    )
    parser.add_argument(
        "--maze-size",
        type=int,
        default=11,
        help="Maze grid size (default: 11)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to .npz dataset for evaluation",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive puzzle solving mode",
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=3,
        help="Number of random puzzles to demo (default: 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = PUZZLE_CONFIG[args.puzzle]
    if args.puzzle == "maze":
        grid_size = args.maze_size
    else:
        grid_size = cfg["grid_size"]

    solver = ONNXSolver(args.model)

    if args.benchmark:
        benchmark_mode(solver, args.puzzle, grid_size)
    elif args.data:
        evaluate_dataset(solver, args.data, args.puzzle, grid_size)
    elif args.interactive:
        interactive_mode(solver, args.puzzle, grid_size)
    else:
        demo_random(solver, args.puzzle, grid_size, num_puzzles=args.num_puzzles)


if __name__ == "__main__":
    main()
