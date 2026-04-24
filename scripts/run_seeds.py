# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Automated multi-seed training runner.

Trains the Simplified HRM across multiple random seeds, then aggregates
results to measure performance variance.  Designed for local / dev use
(NOT intended for CI as training is too expensive).

Seeds default to [123, 456, 789] (3-seed setup).  All ``train_simplified.py``
arguments are forwarded, with ``--seed`` overridden per run.

Each seed writes its artefacts into a subfolder::

    {output_dir}/
    ├── seed_42/
    │   ├── simplified_hrm_{puzzle}_best.pt
    │   ├── training_history_simplified_{puzzle}.json
    │   └── ...
    ├── seed_123/
    │   └── ...
    ├── seed_456/
    │   └── ...
    ├── seed_summary_{puzzle}.json
    └── seed_comparison_{puzzle}.png

Usage:
    # Run 3-seed experiment on maze (forwards all args to train_simplified.py)
    python scripts/run_seeds.py --puzzle maze --generate-data \\
        --num-samples 500 --maze-size 15 --epochs 100

    # Custom seeds and output directory
    python scripts/run_seeds.py --puzzle sudoku_9x9 --epochs 50 \\
        --seed-list 1 2 3 --output-dir model/seed_experiment

    # Aggregate-only mode (skip training, just recompute stats + plots)
    python scripts/run_seeds.py --puzzle maze --aggregate-only \\
        --output-dir model/seed_experiment
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hrm.training.seed_analysis import (
    aggregate_seeds,
    plot_seed_comparison,
    print_summary,
    save_summary,
)

# Helpers

def _build_train_command(
    seed: int,
    save_dir: Path,
    forwarded_args: list[str],
) -> list[str]:
    """Build the command line for a single training run.

    Injects ``--seed`` and ``--save-dir`` while forwarding every other
    argument the user passed.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_simplified.py"),
    ]
    cmd += forwarded_args
    cmd += ["--seed", str(seed)]
    cmd += ["--save-dir", str(save_dir)]
    return cmd


def _filter_forwarded_args(argv: list[str]) -> list[str]:
    """Strip run_seeds-specific arguments, keeping only train_simplified args.

    Removes: --seed-list, --output-dir, --aggregate-only and their values.
    """
    result: list[str] = []
    skip_flags = {"--seed-list", "--output-dir"}
    bool_flags = {"--aggregate-only"}

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in bool_flags:
            i += 1
            continue

        if arg in skip_flags:
            # Skip flag and consume subsequent values until next flag
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue

        # Also skip --seed (single value) since we override it
        if arg == "--seed":
            i += 2
            continue

        # Also skip --save-dir since we override it
        if arg == "--save-dir":
            i += 2
            continue

        result.append(arg)
        i += 1

    return result


# Main

def main() -> None:
    # Parse our own arguments first
    parser = argparse.ArgumentParser(
        description="Run multi-seed training experiments (dev/production only)",
        add_help=False,  # Don't conflict with train_simplified --help
    )
    parser.add_argument(
        "--seed-list", type=int, nargs="+", default=[123, 456, 789],
        help="Seeds to train with (default: 123 456 789)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="model/seed_experiment",
        help="Root output directory for seed runs",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip training; only aggregate existing results and plot",
    )
    # We need --puzzle to know the history filename
    parser.add_argument(
        "--puzzle", type=str, default="sudoku_9x9",
        choices=["sudoku_4x4", "sudoku_9x9", "maze"],
    )

    args, _remaining = parser.parse_known_args()

    seeds = args.seed_list
    output_dir = Path(args.output_dir)
    puzzle = args.puzzle

    # Build forwarded args (everything the user typed, minus our flags)
    forwarded = _filter_forwarded_args(sys.argv[1:])

    print("=" * 65)
    print("  Multi-Seed Experiment Runner")
    print(f"  Seeds:      {seeds}")
    print(f"  Puzzle:     {puzzle}")
    print(f"  Output dir: {output_dir}")
    print(f"  Mode:       {'aggregate-only' if args.aggregate_only else 'train + aggregate'}")
    print("=" * 65)

    # Training phase
    if not args.aggregate_only:
        total_start = time.time()

        for i, seed in enumerate(seeds, 1):
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            cmd = _build_train_command(seed, seed_dir, forwarded)

            print(f"\n{'='*65}")
            print(f"  [{i}/{len(seeds)}] Training with seed={seed}")
            print(f"  Output: {seed_dir}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"{'='*65}\n")

            seed_start = time.time()

            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

            seed_elapsed = time.time() - seed_start
            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            print(f"\n  Seed {seed} finished in {seed_elapsed:.1f}s — {status}")

            if result.returncode != 0:
                print(f"\n  ERROR: Training failed for seed {seed}. Stopping.")
                sys.exit(result.returncode)

        total_elapsed = time.time() - total_start
        print(f"\n  All {len(seeds)} seeds trained in {total_elapsed:.1f}s")

    # Aggregation phase
    print(f"\n{'='*65}")
    print(f"  Aggregating results across {len(seeds)} seeds...")
    print(f"{'='*65}\n")

    try:
        summary = aggregate_seeds(output_dir, seeds, puzzle)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Make sure training completed for all seeds, or use --aggregate-only")
        sys.exit(1)

    print_summary(summary)
    plot_seed_comparison(summary, save_dir=output_dir, puzzle=puzzle)
    save_summary(summary, save_dir=output_dir, puzzle=puzzle)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
