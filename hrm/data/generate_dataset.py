"""
CLI tool for generating puzzle datasets in JSON/CSV formats.

Generates Sudoku (4x4, 9x9) and weighted-maze datasets, writing them to
JSON or CSV files with full metadata (difficulty, seed, timestamp).

Usage:
    python -m hrm.data.generate_dataset sudoku --size 4 --num 10 --seed 42 -o data/sample_sudoku_4x4.json
    python -m hrm.data.generate_dataset maze --size 15 --num 10 --seed 42 -o data/sample_maze_15x15.csv
"""

import argparse
import sys

from hrm.data.io import save_dataset
from hrm.data.sudoku_generator import generate_sudoku_dataset
from hrm.data.weighted_maze_generator import generate_weighted_maze_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate puzzle datasets in JSON or CSV format.",
    )

    sub = parser.add_subparsers(dest="puzzle_type", required=True)

    # -- Sudoku subcommand ------------------------------------------------
    sp_sudoku = sub.add_parser("sudoku", help="Generate Sudoku dataset")
    sp_sudoku.add_argument(
        "--size", type=int, default=9, choices=[4, 9], help="Grid size (default: 9)"
    )
    sp_sudoku.add_argument("--num", type=int, default=100, help="Number of puzzles (default: 100)")
    sp_sudoku.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard", "mixed"],
        help="Difficulty level (default: medium)",
    )
    sp_sudoku.add_argument("--seed", type=int, default=None, help="Random seed")
    sp_sudoku.add_argument("-o", "--output", required=True, help="Output file path (.json or .csv)")
    sp_sudoku.add_argument(
        "--format",
        default=None,
        choices=["json", "csv"],
        help="Output format (auto-detected from extension if omitted)",
    )

    # -- Maze subcommand --------------------------------------------------
    sp_maze = sub.add_parser("maze", help="Generate weighted-maze dataset")
    sp_maze.add_argument("--size", type=int, default=15, help="Grid size (>= 7, default: 15)")
    sp_maze.add_argument("--num", type=int, default=100, help="Number of puzzles (default: 100)")
    sp_maze.add_argument("--seed", type=int, default=None, help="Random seed")
    sp_maze.add_argument("-o", "--output", required=True, help="Output file path (.json or .csv)")
    sp_maze.add_argument(
        "--format",
        default=None,
        choices=["json", "csv"],
        help="Output format (auto-detected from extension if omitted)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Determine output format
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "json"

    # Generate dataset
    if args.puzzle_type == "sudoku":
        print(
            f"Generating {args.num} Sudoku puzzles "
            f"(size={args.size}, difficulty={args.difficulty}, seed={args.seed})..."
        )
        dataset = generate_sudoku_dataset(
            num_puzzles=args.num,
            grid_size=args.size,
            difficulty=args.difficulty,
            seed=args.seed,
        )
        difficulty = args.difficulty
    else:
        print(f"Generating {args.num} maze puzzles (size={args.size}, seed={args.seed})...")
        dataset = generate_weighted_maze_dataset(
            num_puzzles=args.num,
            grid_size=args.size,
            seed=args.seed,
        )
        difficulty = None

    # Save
    out_path = save_dataset(dataset, args.output, fmt=fmt, seed=args.seed, difficulty=difficulty)
    print(f"Dataset saved to {out_path}")


if __name__ == "__main__":
    main()
