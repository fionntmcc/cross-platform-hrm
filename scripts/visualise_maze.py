# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Maze Visualiser — Publication-Quality Figures for HRM Dissertation

Generates matplotlib figures showing:
  - The maze structure (walls, open cells, weighted cells, start/goal)
  - The predicted optimal path overlaid on the maze
  - Side-by-side comparison of prediction vs ground truth
  - Per-step reasoning trajectory (animated or tiled)

The output is designed for inclusion in academic reports: high DPI,
clean legends, colour-blind-safe palette, and LaTeX-ready labels.

Usage:
    # Evaluate and visualise from a dataset
    python scripts/visualise_maze.py \
        --model model/simplified_hrm_maze_best.pt \
        --data data/maze_15x15_train.npz \
        --num 5 --save-dir figures/maze

    # Visualise reasoning steps for a single puzzle
    python scripts/visualise_maze.py \
        --model model/simplified_hrm_maze_best.pt \
        --data data/maze_15x15_train.npz \
        --index 42 --steps --save-dir figures/maze

    # Generate fresh mazes (no dataset needed)
    python scripts/visualise_maze.py \
        --model model/simplified_hrm_maze_best.pt \
        --generate --num 5 --maze-size 11

    # Display on screen instead of saving
    python scripts/visualise_maze.py \
        --model model/simplified_hrm_maze_best.pt \
        --data data/maze_15x15_train.npz \
        --show

    # Use exported ONNX model for visualisation
    python scripts/visualise_maze.py \
        --model model/simplified_hrm_maze_11x11_s16.onnx \
        --data data/maze_11x11_train.npz \
        --num 5 --save-dir figures/maze_onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy matplotlib import (allows --help without a display)
_mpl_cache = None

def _import_mpl(interactive: bool = False):
    global _mpl_cache
    if _mpl_cache is not None:
        return _mpl_cache
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")  # non-interactive backend (for saving only)
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    _mpl_cache = plt, mcolors, mpatches, ListedColormap, BoundaryNorm
    return _mpl_cache


# Maze token constants (must match weighted_maze_generator.py)
WALL  = 0
PATH  = 1
START = 2
GOAL  = 3
MIN_WEIGHT = 4
MAX_WEIGHT = 9

# Colour palette — colour-blind-safe, print-friendly
# Maze structure colours
CLR_WALL       = "#2d2d2d"    # near-black
CLR_FLOOR      = "#f0ece3"    # warm off-white
CLR_START      = "#4dabf7"    # sky blue
CLR_GOAL       = "#f76707"    # orange
CLR_WEIGHT_LO  = "#e8d5b7"   # light tan  (weight 4)
CLR_WEIGHT_HI  = "#a0522d"   # sienna     (weight 9)

# Path overlay colours
CLR_PATH_CORRECT   = "#2ecc71"  # green — correctly predicted on-path
CLR_PATH_WRONG     = "#e74c3c"  # red   — false positive (predicted but not on path)
CLR_PATH_MISSED    = "#9b59b6"  # purple — false negative (on path but not predicted)
CLR_PATH_ONLY      = "#3498db"  # blue  — predicted path (no ground truth available)
CLR_TRUTH_PATH     = "#2ecc71"  # green — ground truth path overlay


# ═══════════════════════════════════════════════════════════════════════════
#  Core rendering
# ═══════════════════════════════════════════════════════════════════════════

def _weight_colour(w: int) -> str:
    """Interpolate between light tan (4) and sienna (9)."""
    t = (w - MIN_WEIGHT) / max(MAX_WEIGHT - MIN_WEIGHT, 1)
    r1, g1, b1 = int(CLR_WEIGHT_LO[1:3], 16), int(CLR_WEIGHT_LO[3:5], 16), int(CLR_WEIGHT_LO[5:7], 16)
    r2, g2, b2 = int(CLR_WEIGHT_HI[1:3], 16), int(CLR_WEIGHT_HI[3:5], 16), int(CLR_WEIGHT_HI[5:7], 16)
    r = int(r1 + t * (r2 - r1))
    g = int(g1 + t * (g2 - g1))
    b = int(b1 + t * (b2 - b1))
    return f"#{r:02x}{g:02x}{b:02x}"


def render_maze(ax, maze: np.ndarray,
                pred: np.ndarray = None,
                truth: np.ndarray = None,
                title: str = "",
                show_weights: bool = True,
                show_legend: bool = True,
                cell_size: float = 0.9):
    """
    Render a single maze panel onto a matplotlib Axes.

    Args:
        ax: Matplotlib Axes to draw on.
        maze: 2D int array of maze tokens (0-9).
        pred: 2D int array of predicted path (0/1), or None.
        truth: 2D int array of ground-truth path (0/1), or None.
        title: Panel title string.
        show_weights: Whether to annotate weighted cells with cost numbers.
        show_legend: Whether to add a colour legend.
        cell_size: Size of each cell square (0-1, fraction of grid spacing).
    """
    _, _, mpatches, _, _ = _import_mpl()
    rows, cols = maze.shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # y-axis inverted so (0,0) is top-left
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    half = cell_size / 2
    legend_items = {}

    for r in range(rows):
        for c in range(cols):
            tok = int(maze[r, c])

            # ── Determine base cell colour ──
            if tok == WALL:
                bg = CLR_WALL
            elif tok == START:
                bg = CLR_START
            elif tok == GOAL:
                bg = CLR_GOAL
            elif tok >= MIN_WEIGHT:
                bg = _weight_colour(tok)
            else:
                bg = CLR_FLOOR

            # ── Draw base cell ──
            rect = mpatches.Rectangle((c - half, r - half), cell_size, cell_size,
                                 facecolor=bg, edgecolor="none", linewidth=0)
            ax.add_patch(rect)

            # ── Weight annotation ──
            if show_weights and tok >= MIN_WEIGHT:
                ax.text(c, r, str(tok), ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#ffffff",
                        fontfamily="monospace")

            # ── Start / Goal labels ──
            if tok == START:
                ax.text(c, r, "S", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#ffffff",
                        fontfamily="monospace")
                legend_items["Start"] = CLR_START
            elif tok == GOAL:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#ffffff",
                        fontfamily="monospace")
                legend_items["Goal"] = CLR_GOAL

            # ── Path overlay ──
            has_pred = pred is not None and int(pred[r, c]) == 1
            has_truth = truth is not None and int(truth[r, c]) == 1

            if has_pred and has_truth:
                # Correct prediction
                overlay = mpatches.Rectangle((c - half, r - half), cell_size, cell_size,
                                        facecolor=CLR_PATH_CORRECT, alpha=0.65,
                                        edgecolor="none")
                ax.add_patch(overlay)
                legend_items["Correct path"] = CLR_PATH_CORRECT
            elif has_pred and truth is not None and not has_truth:
                # False positive
                overlay = mpatches.Rectangle((c - half, r - half), cell_size, cell_size,
                                        facecolor=CLR_PATH_WRONG, alpha=0.65,
                                        edgecolor="none")
                ax.add_patch(overlay)
                legend_items["False positive"] = CLR_PATH_WRONG
            elif not has_pred and has_truth:
                # Missed path cell
                overlay = mpatches.Rectangle((c - half, r - half), cell_size, cell_size,
                                        facecolor=CLR_PATH_MISSED, alpha=0.55,
                                        edgecolor="none")
                ax.add_patch(overlay)
                legend_items["Missed path"] = CLR_PATH_MISSED
            elif has_pred and truth is None:
                # Predicted path (no ground truth)
                overlay = mpatches.Rectangle((c - half, r - half), cell_size, cell_size,
                                        facecolor=CLR_PATH_ONLY, alpha=0.55,
                                        edgecolor="none")
                ax.add_patch(overlay)
                legend_items["Predicted path"] = CLR_PATH_ONLY

    # ── Thin grid lines ──
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#cccccc", linewidth=0.3, zorder=0)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#cccccc", linewidth=0.3, zorder=0)

    # Background fill for the full grid area
    ax.set_facecolor("#e0dcd4")

    # ── Legend ──
    if show_legend and legend_items:
        legend_items["Wall"] = CLR_WALL
        legend_items["Floor"] = CLR_FLOOR
        if any(maze.flatten() >= MIN_WEIGHT):
            legend_items["Weighted (4-9)"] = _weight_colour(6)
        handles = [mpatches.Patch(facecolor=clr, edgecolor="none", label=lbl)
                   for lbl, clr in legend_items.items()]
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), fontsize=7,
                  frameon=True, fancybox=True, framealpha=0.85,
                  edgecolor="#cccccc")


def render_truth_only(ax, maze: np.ndarray, truth: np.ndarray,
                      title: str = "Ground truth"):
    """Render a maze panel showing only the ground-truth path."""
    _, _, mpatches, _, _ = _import_mpl()
    rows, cols = maze.shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    half = 0.9 / 2

    for r in range(rows):
        for c in range(cols):
            tok = int(maze[r, c])
            on_path = truth is not None and int(truth[r, c]) == 1

            if tok == WALL:
                bg = CLR_WALL
            elif tok == START:
                bg = CLR_START
            elif tok == GOAL:
                bg = CLR_GOAL
            elif on_path:
                bg = CLR_TRUTH_PATH
            elif tok >= MIN_WEIGHT:
                bg = _weight_colour(tok)
            else:
                bg = CLR_FLOOR

            rect = mpatches.Rectangle((c - half, r - half), 0.9, 0.9,
                                 facecolor=bg, edgecolor="none")
            ax.add_patch(rect)

            if tok == START:
                ax.text(c, r, "S", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#fff",
                        fontfamily="monospace")
            elif tok == GOAL:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#fff",
                        fontfamily="monospace")
            elif tok >= MIN_WEIGHT and not on_path:
                ax.text(c, r, str(tok), ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#fff",
                        fontfamily="monospace")

    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#cccccc", linewidth=0.3, zorder=0)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#cccccc", linewidth=0.3, zorder=0)
    ax.set_facecolor("#e0dcd4")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure composers
# ═══════════════════════════════════════════════════════════════════════════

def figure_comparison(maze: np.ndarray, pred: np.ndarray,
                      truth: np.ndarray, idx: int = 0,
                      metrics: dict = None) -> Figure:
    """
    Create a side-by-side figure: prediction vs ground truth.

    Returns a matplotlib Figure.
    """
    plt, _, _, _, _ = _import_mpl()

    grid_size = maze.shape[0]
    fig_w = max(10, grid_size * 0.6 * 2 + 3)
    fig_h = max(5, grid_size * 0.6 + 1.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # ── Metrics string ──
    metric_str = ""
    if metrics:
        parts = []
        if "cell_acc" in metrics:
            parts.append(f"Cell acc: {metrics['cell_acc']:.1%}")
        if "path_acc" in metrics:
            parts.append(f"Path acc: {metrics['path_acc']:.1%}")
        if "correct" in metrics:
            tag = "✓ SOLVED" if metrics["correct"] else "✗ INCORRECT"
            parts.append(tag)
        metric_str = "  —  ".join(parts)

    render_maze(ax1, maze, pred=pred, truth=truth,
                title="Model prediction", show_legend=True)
    render_truth_only(ax2, maze, truth, title="Ground truth")

    suptitle = f"Maze #{idx}"
    if metric_str:
        suptitle += f"    ({metric_str})"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


def figure_reasoning_steps(maze: np.ndarray, step_preds: list,
                           truth: np.ndarray = None,
                           idx: int = 0) -> Figure:
    """
    Create a tiled figure showing predicted path at each reasoning step.
    """
    plt, _, _, _, _ = _import_mpl()

    n_steps = len(step_preds)
    cols = min(n_steps, 4)
    rows_fig = (n_steps + cols - 1) // cols

    grid_size = maze.shape[0]
    cell_w = max(3.5, grid_size * 0.35)
    cell_h = max(3.5, grid_size * 0.35)
    fig, axes = plt.subplots(rows_fig, cols,
                             figsize=(cell_w * cols + 1, cell_h * rows_fig + 1),
                             squeeze=False)

    for i, pred in enumerate(step_preds):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        pred_2d = pred.reshape(grid_size, grid_size) if pred.ndim == 1 else pred

        # Compute step accuracy
        acc_str = ""
        if truth is not None:
            truth_flat = truth.flatten()
            pred_flat = pred_2d.flatten()
            acc = (pred_flat == truth_flat).mean()
            path_mask = truth_flat == 1
            path_acc = (pred_flat[path_mask] == 1).mean() if path_mask.any() else 0
            n_changed = (pred_flat != (step_preds[i-1].flatten()
                         if i > 0 else np.zeros_like(pred_flat))).sum()
            acc_str = f"  (cell {acc:.0%}, path {path_acc:.0%}, Δ{n_changed})"

        label = f"Step {i+1}" if i < n_steps - 1 else "Final"
        render_maze(ax, maze, pred=pred_2d, truth=truth,
                    title=f"{label}{acc_str}",
                    show_weights=False, show_legend=False, cell_size=0.95)

    # Hide unused axes
    for i in range(n_steps, rows_fig * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Maze #{idx} — Reasoning trajectory ({n_steps} steps)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Model inference helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_pt_model(model_path: str, device):
    import torch

    from hrm.model_simplified import SimplifiedHRM, SimplifiedHRMConfig
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", SimplifiedHRMConfig())
    model = SimplifiedHRM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config


def _load_onnx_model(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX visualisation. "
            "Install with: pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    config = SimpleNamespace(num_reasoning_steps="N/A (ONNX)")
    return session, config, inp.name, out.name


def _predict_pt(model, maze_2d: np.ndarray, device, return_steps=False):
    import torch

    from hrm.model_simplified import PuzzleType
    flat = torch.from_numpy(maze_2d.flatten()).long().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(flat, PuzzleType.MAZE, return_intermediates=return_steps)
    pred = out["predictions"][0].cpu().numpy().reshape(maze_2d.shape)
    steps = None
    if return_steps and "intermediates" in out:
        steps = [s[0].cpu().numpy() for s in out["intermediates"]["step_predictions"]]
    return pred, steps


def _predict_onnx(session, input_name: str, maze_2d: np.ndarray):
    flat = maze_2d.flatten().astype(np.int64).reshape(1, -1)
    pred = session.run(None, {input_name: flat})[0][0].reshape(maze_2d.shape)
    return pred, None


def _resolve_backend(model_path: str, backend_flag: str) -> str:
    if backend_flag != "auto":
        return backend_flag
    ext = Path(model_path).suffix.lower()
    if ext == ".onnx":
        return "onnx"
    return "pt"


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualise HRM maze predictions with publication-quality figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to trained maze checkpoint (.pt) or exported model (.onnx)")
    parser.add_argument("--backend", choices=["auto", "pt", "onnx"],
                        default="auto",
                        help="Inference backend (default: auto by model extension)")
    parser.add_argument("--data", default=None,
                        help="Path to maze .npz dataset")
    parser.add_argument("--generate", action="store_true",
                        help="Generate fresh mazes instead of loading from --data")
    parser.add_argument("--maze-size", type=int, default=11,
                        help="Grid size for generated mazes (default: 11)")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of mazes to visualise")
    parser.add_argument("--index", type=int, default=None,
                        help="Specific puzzle index from dataset (overrides --num)")
    parser.add_argument("--steps", action="store_true",
                        help="Show per-step reasoning trajectory")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save figures (default: display only)")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively (requires display)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure DPI (default: 200)")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    backend = _resolve_backend(args.model, args.backend)

    plt, _, _, _, _ = _import_mpl()

    # ── Load model ──
    print(f"Loading model from {args.model} [{backend}]...")
    device = None
    if backend == "pt":
        import torch

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        model, config = _load_pt_model(args.model, device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Reasoning steps: {config.num_reasoning_steps}")
    else:
        model, config, input_name, output_name = _load_onnx_model(args.model)
        print(f"  Provider: {model.get_providers()[0]}")
        print(f"  Input: {input_name} {model.get_inputs()[0].shape}")
        print(f"  Output: {output_name} {model.get_outputs()[0].shape}")

    if args.steps and backend == "onnx":
        print("Warning: --steps is only available for PyTorch checkpoints (.pt).")
        print("         Continuing with comparison figures only.")

    # ── Load / generate mazes ──
    mazes = []  # list of (maze_2d, solution_2d_or_None)

    if args.generate:
        from hrm.data.weighted_maze_generator import WeightedMazeGenerator
        gen = WeightedMazeGenerator(grid_size=args.maze_size, seed=42)
        print(f"Generating {args.num} fresh {gen.grid_size}x{gen.grid_size} mazes...")
        for _ in range(args.num * 3):
            result = gen.create_puzzle()
            if result is None:
                continue
            p, s, _ = result
            mazes.append((np.array(p), np.array(s)))
            if len(mazes) >= args.num:
                break
    elif args.data:
        ext = Path(args.data).suffix.lower()
        if ext in (".json", ".csv"):
            from hrm.data.io import load_dataset

            ds = load_dataset(args.data)
            problems, solutions = ds["problems"], ds["solutions"]
        else:
            data = np.load(args.data)
            key = "problems" if "problems" in data else "puzzles"
            problems, solutions = data[key], data["solutions"]
        print(f"Loaded {len(problems)} mazes from {args.data}")

        if args.index is not None:
            indices = [args.index]
        else:
            indices = np.random.choice(len(problems),
                                       size=min(args.num, len(problems)),
                                       replace=False)
        for idx in indices:
            mazes.append((problems[idx], solutions[idx]))
    else:
        print("Error: provide --data or --generate")
        sys.exit(1)

    if not mazes:
        print("No mazes to visualise.")
        sys.exit(1)

    # ── Save dir ──
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # ── Render ──
    for i, (maze_2d, truth_2d) in enumerate(mazes):
        idx = args.index if args.index is not None else i
        if backend == "pt":
            pred_2d, step_preds = _predict_pt(
                model, maze_2d, device, return_steps=args.steps
            )
        else:
            pred_2d, step_preds = _predict_onnx(model, input_name, maze_2d)

        # Metrics
        metrics = {}
        if truth_2d is not None:
            metrics["cell_acc"] = (pred_2d == truth_2d).mean()
            path_mask = truth_2d == 1
            if path_mask.any():
                metrics["path_acc"] = (pred_2d[path_mask] == 1).mean()
            metrics["correct"] = (pred_2d == truth_2d).all()

        # Comparison figure
        fig = figure_comparison(maze_2d, pred_2d, truth_2d,
                                idx=idx, metrics=metrics)
        if save_dir:
            out_path = save_dir / f"maze_{idx:04d}_comparison.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                        facecolor="white")
            print(f"  Saved {out_path}")
        if args.show:
            plt.show()
        plt.close(fig)

        # Reasoning steps figure
        if args.steps and step_preds:
            fig_steps = figure_reasoning_steps(maze_2d, step_preds,
                                               truth=truth_2d, idx=idx)
            if save_dir:
                out_path = save_dir / f"maze_{idx:04d}_steps.png"
                fig_steps.savefig(out_path, dpi=args.dpi,
                                  bbox_inches="tight", facecolor="white")
                print(f"  Saved {out_path}")
            if args.show:
                plt.show()
            plt.close(fig_steps)

    # ── Summary ──
    if len(mazes) > 1:
        accs = []
        for maze_2d, truth_2d in mazes:
            if truth_2d is not None:
                if backend == "pt":
                    pred, _ = _predict_pt(model, maze_2d, device)
                else:
                    pred, _ = _predict_onnx(model, input_name, maze_2d)
                accs.append((pred == truth_2d).mean())
        if accs:
            print(f"\nOverall cell accuracy: {np.mean(accs):.1%} "
                  f"(n={len(accs)}, std={np.std(accs):.1%})")

    print("Done.")


if __name__ == "__main__":
    main()
