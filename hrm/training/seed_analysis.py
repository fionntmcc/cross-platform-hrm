"""
Multi-seed result aggregation and comparison plots.

Loads per-seed training history JSON files, computes mean += std across
seeds for every metric, and generates comparison plots.

Usage (standalone)::

    python -m hrm.training.seed_analysis model/seed_run \
        --seeds 123 456 789 --puzzle sudoku_9x9

Or via the public API::

    from hrm.training.seed_analysis import aggregate_seeds, plot_seed_comparison

    summary = aggregate_seeds("model/seed_run", seeds=[123, 456, 789],
                              puzzle="sudoku_9x9")
    plot_seed_comparison(summary, save_dir="model/seed_run")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# =========================================================================
# Aggregation
# =========================================================================

def _load_history(path: Path) -> dict[str, list]:
    """Load a training_history JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_seeds(
    run_dir: str | Path,
    seeds: list[int],
    puzzle: str,
) -> dict[str, Any]:
    """Aggregate per-seed training histories into mean ± std summary.

    Expects files at ``{run_dir}/seed_{s}/training_history_simplified_{puzzle}.json``
    for each seed *s*.

    Args:
        run_dir: Root directory containing per-seed subdirectories.
        seeds: List of seed values used.
        puzzle: Puzzle name (e.g. ``"sudoku_9x9"``).

    Returns:
        Dictionary with:
        - ``seeds``: list of seeds
        - ``per_seed``: ``{seed: history_dict}``
        - ``metrics``: ``{metric_name: {"mean": [...], "std": [...], "per_seed": {seed: [...]}}}``
        - ``final``: ``{metric_name: {"mean": float, "std": float, "values": [...]}}``
        - ``variance_pct``: ``{metric_name: float}`` — std / mean × 100 for final-epoch metrics
    """
    run_dir = Path(run_dir)
    per_seed: dict[int, dict[str, list]] = {}

    for seed in seeds:
        hist_path = run_dir / f"seed_{seed}" / f"training_history_simplified_{puzzle}.json"
        if not hist_path.exists():
            raise FileNotFoundError(
                f"Missing history for seed {seed}: {hist_path}"
            )
        per_seed[seed] = _load_history(hist_path)

    # Discover common metrics (lists of floats/ints)
    all_keys: set[str] = set()
    for hist in per_seed.values():
        for k, v in hist.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                all_keys.add(k)

    # Align to the shortest run length (in case a seed stopped early)
    min_len = min(
        len(v) for hist in per_seed.values() for k, v in hist.items()
        if k in all_keys
    )

    metrics: dict[str, dict[str, Any]] = {}
    final: dict[str, dict[str, Any]] = {}
    variance_pct: dict[str, float] = {}

    for key in sorted(all_keys):
        # Collect arrays: (num_seeds, min_len)
        arrays = []
        for seed in seeds:
            vals = per_seed[seed].get(key, [])
            arrays.append(vals[:min_len])
        arr = np.array(arrays, dtype=np.float64)  # (num_seeds, min_len)

        mean = arr.mean(axis=0).tolist()
        std = arr.std(axis=0).tolist()
        per_seed_vals = {seed: arrays[i] for i, seed in enumerate(seeds)}

        metrics[key] = {"mean": mean, "std": std, "per_seed": per_seed_vals}

        # Final-epoch summary
        final_vals = arr[:, -1]
        f_mean = float(final_vals.mean())
        f_std = float(final_vals.std())
        final[key] = {
            "mean": f_mean,
            "std": f_std,
            "values": final_vals.tolist(),
        }

        # Variance as percentage (avoid division by zero)
        if abs(f_mean) > 1e-12:
            variance_pct[key] = (f_std / abs(f_mean)) * 100.0
        else:
            variance_pct[key] = 0.0

    return {
        "seeds": seeds,
        "per_seed": {s: per_seed[s] for s in seeds},
        "metrics": metrics,
        "final": final,
        "variance_pct": variance_pct,
        "num_epochs": min_len,
    }


# =========================================================================
# Reporting
# =========================================================================

def print_summary(summary: dict[str, Any]) -> str:
    """Pretty-print the aggregated summary to stdout. Returns the text."""
    lines: list[str] = []
    seeds = summary["seeds"]
    final = summary["final"]
    var_pct = summary["variance_pct"]

    lines.append("=" * 65)
    lines.append(f"  Multi-Seed Summary  ({len(seeds)} seeds: {seeds})")
    lines.append(f"  Epochs completed: {summary['num_epochs']}")
    lines.append("=" * 65)

    # Key metrics we care about most
    priority = [
        "val_puzzle_accuracy",
        "val_token_accuracy",
        "val_loss",
        "train_loss",
        "train_accuracy",
    ]
    shown = set()

    for key in priority:
        if key in final:
            f = final[key]
            v = var_pct[key]
            flag = " *** HIGH VARIANCE" if v > 1.5 and "accuracy" in key else ""
            lines.append(
                f"  {key:30s}  {f['mean']:.6f} ± {f['std']:.6f}  "
                f"(var {v:.2f}%){flag}"
            )
            shown.add(key)

    # Remaining metrics
    for key in sorted(final.keys()):
        if key in shown:
            continue
        f = final[key]
        v = var_pct[key]
        lines.append(
            f"  {key:30s}  {f['mean']:.6f} ± {f['std']:.6f}  (var {v:.2f}%)"
        )

    lines.append("=" * 65)

    # Variance assessment
    acc_vars = {
        k: v for k, v in var_pct.items() if "accuracy" in k and k in final
    }
    if acc_vars:
        max_var_key = max(acc_vars, key=acc_vars.get)  # type: ignore[arg-type]
        max_var = acc_vars[max_var_key]
        if max_var <= 1.5:
            lines.append(f"  PASS: Max accuracy variance {max_var:.2f}% <= 1.5% target")
        else:
            lines.append(
                f"  WARN: {max_var_key} variance {max_var:.2f}% > 1.5% target"
            )
    lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


# =========================================================================
# Plotting
# =========================================================================

def plot_seed_comparison(
    summary: dict[str, Any],
    save_dir: str | Path | None = None,
    puzzle: str = "",
) -> Path | None:
    """Generate a multi-panel comparison plot across seeds.

    Creates a 2×2 figure: loss, token accuracy, puzzle accuracy, and a
    bar chart of final-epoch metrics with error bars.

    Args:
        summary: Output of :func:`aggregate_seeds`.
        save_dir: Directory for saving the plot.  Defaults to cwd.
        puzzle: Puzzle name for the plot title.

    Returns:
        Path to the saved PNG, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed — skipping seed comparison plot.")
        return None

    seeds = summary["seeds"]
    metrics = summary["metrics"]
    num_epochs = summary["num_epochs"]
    epochs = list(range(1, num_epochs + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Multi-Seed Comparison — {puzzle} ({len(seeds)} seeds)",
        fontsize=14,
    )

    # Colour palette for seeds
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # --- Panel 1: Loss ---
    ax = axes[0, 0]
    for i, seed in enumerate(seeds):
        if "val_loss" in metrics:
            vals = metrics["val_loss"]["per_seed"][seed]
            ax.plot(epochs[:len(vals)], vals, label=f"Seed {seed}",
                    color=colours[i % len(colours)], alpha=0.7)
    if "val_loss" in metrics:
        mean = metrics["val_loss"]["mean"]
        std = metrics["val_loss"]["std"]
        ax.plot(epochs, mean, color="black", linewidth=2, label="Mean")
        ax.fill_between(
            epochs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.15, color="black",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Token Accuracy ---
    ax = axes[0, 1]
    for i, seed in enumerate(seeds):
        if "val_token_accuracy" in metrics:
            vals = metrics["val_token_accuracy"]["per_seed"][seed]
            ax.plot(epochs[:len(vals)], vals, label=f"Seed {seed}",
                    color=colours[i % len(colours)], alpha=0.7)
    if "val_token_accuracy" in metrics:
        mean = metrics["val_token_accuracy"]["mean"]
        std = metrics["val_token_accuracy"]["std"]
        ax.plot(epochs, mean, color="black", linewidth=2, label="Mean")
        ax.fill_between(
            epochs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.15, color="black",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Token Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # --- Panel 3: Puzzle Accuracy ---
    ax = axes[1, 0]
    for i, seed in enumerate(seeds):
        if "val_puzzle_accuracy" in metrics:
            vals = metrics["val_puzzle_accuracy"]["per_seed"][seed]
            ax.plot(epochs[:len(vals)], vals, label=f"Seed {seed}",
                    color=colours[i % len(colours)], alpha=0.7)
    if "val_puzzle_accuracy" in metrics:
        mean = metrics["val_puzzle_accuracy"]["mean"]
        std = metrics["val_puzzle_accuracy"]["std"]
        ax.plot(epochs, mean, color="black", linewidth=2, label="Mean")
        ax.fill_between(
            epochs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.15, color="black",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Puzzle Accuracy (Solve Rate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # --- Panel 4: Final-epoch bar chart with error bars ---
    ax = axes[1, 1]
    bar_keys = ["val_puzzle_accuracy", "val_token_accuracy", "train_accuracy"]
    bar_keys = [k for k in bar_keys if k in summary["final"]]
    if bar_keys:
        x = np.arange(len(bar_keys))
        means = [summary["final"][k]["mean"] for k in bar_keys]
        stds = [summary["final"][k]["std"] for k in bar_keys]
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colours[:len(bar_keys)],
                       alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_", "\n") for k in bar_keys], fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Final-Epoch Metrics (mean ± std)")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate bars with variance %
        for i, k in enumerate(bar_keys):
            var = summary["variance_pct"][k]
            ax.text(i, means[i] + stds[i] + 0.02, f"{var:.2f}%",
                    ha="center", fontsize=9, color="red" if var > 1.5 else "green")
    else:
        ax.text(0.5, 0.5, "No accuracy metrics", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out_dir = Path(save_dir) if save_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_comparison_{puzzle}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Seed comparison plot saved to {out_path}")
    return out_path


# =========================================================================
# Save summary JSON
# =========================================================================

def save_summary(
    summary: dict[str, Any],
    save_dir: str | Path,
    puzzle: str = "",
) -> Path:
    """Write the aggregated summary to a JSON file.

    The ``per_seed`` raw arrays are included so the JSON is self-contained
    and can be reloaded for further analysis.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_summary_{puzzle}.json"

    # Make JSON-serialisable (numpy → list)
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_convert(summary), f, indent=2)

    print(f"Seed summary saved to {out_path}")
    return out_path


# =========================================================================
# CLI entry point
# =========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed training results"
    )
    parser.add_argument("run_dir", type=str,
                        help="Directory containing seed_* subdirectories")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789],
                        help="Seeds to aggregate (default: 123 456 789)")
    parser.add_argument("--puzzle", type=str, default="sudoku_9x9",
                        help="Puzzle name used in history filenames")
    args = parser.parse_args()

    summary = aggregate_seeds(args.run_dir, args.seeds, args.puzzle)
    print_summary(summary)
    plot_seed_comparison(summary, save_dir=args.run_dir, puzzle=args.puzzle)
    save_summary(summary, save_dir=args.run_dir, puzzle=args.puzzle)


if __name__ == "__main__":
    main()
