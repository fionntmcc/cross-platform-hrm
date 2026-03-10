"""
Dataset I/O utilities for saving and loading puzzle datasets in JSON/CSV formats.

Supports both Sudoku and weighted-maze datasets produced by the HRM generators.
Includes auto-format detection on load and metadata (difficulty, timestamp, seed).

Usage:
    from hrm.data.io import save_dataset, load_dataset

    save_dataset(dataset, "data/sudoku_4x4.json", fmt="json")
    dataset = load_dataset("data/sudoku_4x4.json")
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_dataset(
    dataset: dict[str, Any],
    path: str | Path,
    fmt: str = "json",
    seed: int | None = None,
    difficulty: str | None = None,
) -> Path:
    """
    Save a puzzle dataset to disk in JSON or CSV format.

    Args:
        dataset: Dict with keys ``problems`` (np.ndarray), ``solutions``
            (np.ndarray), and optionally ``metadata`` (list[dict]).
        path: Destination file path.  The extension is normalised to match
            *fmt* if it doesn't already.
        fmt: ``"json"`` (default) or ``"csv"``.
        seed: Random seed used during generation (stored in file metadata).
        difficulty: Difficulty label (stored in file metadata).

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        ValueError: If *fmt* is not ``"json"`` or ``"csv"``.
    """
    fmt = fmt.lower()
    if fmt not in ("json", "csv"):
        raise ValueError(f"Unsupported format {fmt!r}. Use 'json' or 'csv'.")

    path = _normalise_path(path, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)

    problems: np.ndarray = np.asarray(dataset["problems"])
    solutions: np.ndarray = np.asarray(dataset["solutions"])
    per_puzzle_meta: list[dict] = dataset.get("metadata", [])

    grid_size = int(problems.shape[1])
    count = int(problems.shape[0])
    puzzle_type = _infer_puzzle_type(per_puzzle_meta, grid_size)

    file_metadata = {
        "puzzle_type": puzzle_type,
        "grid_size": grid_size,
        "count": count,
        "seed": seed,
        "difficulty": difficulty,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    if fmt == "json":
        _write_json(path, file_metadata, problems, solutions, per_puzzle_meta)
    else:
        _write_csv(path, file_metadata, problems, solutions, per_puzzle_meta)

    return path


def load_dataset(path: str | Path) -> dict[str, Any]:
    """
    Load a puzzle dataset from a JSON or CSV file.

    The format is detected automatically from the file extension.

    Args:
        path: Path to the dataset file (``.json`` or ``.csv``).

    Returns:
        Dict with keys:
            - ``problems``: np.ndarray of shape ``(N, grid, grid)``
            - ``solutions``: np.ndarray of shape ``(N, grid, grid)``
            - ``metadata``: list[dict] (per-puzzle metadata, may be empty)
            - ``file_metadata``: dict with ``grid_size``, ``count``, ``seed``,
              ``difficulty``, ``generated_at``, ``puzzle_type``.

    Raises:
        ValueError: If the file extension is not ``.json`` or ``.csv``.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".json":
        return _read_json(path)
    if ext == ".csv":
        return _read_csv(path)

    raise ValueError(f"Cannot auto-detect format for extension {ext!r}. Use .json or .csv.")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _write_json(
    path: Path,
    file_metadata: dict,
    problems: np.ndarray,
    solutions: np.ndarray,
    per_puzzle_meta: list[dict],
) -> None:
    puzzles = []
    for i in range(len(problems)):
        entry: dict[str, Any] = {
            "problem": problems[i].tolist(),
            "solution": solutions[i].tolist(),
        }
        if i < len(per_puzzle_meta):
            entry["metadata"] = _serialisable(per_puzzle_meta[i])
        puzzles.append(entry)

    doc = {"metadata": file_metadata, "puzzles": puzzles}

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)

    file_metadata = doc.get("metadata", {})
    puzzles = doc.get("puzzles", [])

    problems = np.array([p["problem"] for p in puzzles], dtype=np.int8)
    solutions = np.array([p["solution"] for p in puzzles], dtype=np.int8)
    per_puzzle_meta = [p.get("metadata", {}) for p in puzzles]

    return {
        "problems": problems,
        "solutions": solutions,
        "metadata": per_puzzle_meta,
        "file_metadata": file_metadata,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_META_PREFIX = "# "


def _flatten_grid(grid: np.ndarray) -> str:
    """Flatten a 2-D grid to a space-separated string of ints."""
    return " ".join(str(int(v)) for v in grid.ravel())


def _unflatten_grid(flat: str, grid_size: int) -> np.ndarray:
    """Reconstruct a 2-D grid from a flat space-separated string."""
    vals = [int(v) for v in flat.split()]
    return np.array(vals, dtype=np.int8).reshape(grid_size, grid_size)


def _write_csv(
    path: Path,
    file_metadata: dict,
    problems: np.ndarray,
    solutions: np.ndarray,
    per_puzzle_meta: list[dict],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        # Embed file-level metadata as comment header
        fh.write(f"{_CSV_META_PREFIX}{json.dumps(file_metadata)}\n")

        writer = csv.writer(fh)
        writer.writerow(["index", "problem", "solution", "metadata"])

        for i in range(len(problems)):
            meta_json = ""
            if i < len(per_puzzle_meta):
                meta_json = json.dumps(_serialisable(per_puzzle_meta[i]))
            writer.writerow([i, _flatten_grid(problems[i]), _flatten_grid(solutions[i]), meta_json])


def _read_csv(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        first_line = fh.readline()

    # Parse file-level metadata from comment header
    file_metadata: dict[str, Any] = {}
    if first_line.startswith(_CSV_META_PREFIX):
        try:
            file_metadata = json.loads(first_line[len(_CSV_META_PREFIX) :])
        except json.JSONDecodeError:
            pass

    grid_size = int(file_metadata.get("grid_size", 0))

    problems_list: list[np.ndarray] = []
    solutions_list: list[np.ndarray] = []
    per_puzzle_meta: list[dict] = []

    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            # Skip comment lines and header
            if not row or row[0].startswith("#") or row[0] == "index":
                continue

            _, prob_flat, sol_flat = row[0], row[1], row[2]

            # Infer grid_size from first data row if not in metadata
            if grid_size == 0:
                n_vals = len(prob_flat.split())
                grid_size = int(n_vals**0.5)

            problems_list.append(_unflatten_grid(prob_flat, grid_size))
            solutions_list.append(_unflatten_grid(sol_flat, grid_size))

            meta_str = row[3] if len(row) > 3 else ""
            if meta_str:
                try:
                    per_puzzle_meta.append(json.loads(meta_str))
                except json.JSONDecodeError:
                    per_puzzle_meta.append({})
            else:
                per_puzzle_meta.append({})

    # Update file_metadata grid_size if it was inferred
    if "grid_size" not in file_metadata and grid_size:
        file_metadata["grid_size"] = grid_size

    return {
        "problems": np.array(problems_list, dtype=np.int8) if problems_list else np.empty((0,)),
        "solutions": np.array(solutions_list, dtype=np.int8) if solutions_list else np.empty((0,)),
        "metadata": per_puzzle_meta,
        "file_metadata": file_metadata,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_path(path: str | Path, fmt: str) -> Path:
    """Ensure the file has the correct extension for *fmt*."""
    path = Path(path)
    expected = f".{fmt}"
    if path.suffix.lower() != expected:
        path = path.with_suffix(expected)
    return path


def _infer_puzzle_type(per_puzzle_meta: list[dict], grid_size: int) -> str:
    """Best-effort inference of puzzle type from metadata fields."""
    if per_puzzle_meta:
        first = per_puzzle_meta[0]
        if "path_length" in first or "path_cost" in first:
            return "maze"
        if "backtracks" in first or "empty_cells" in first:
            return "sudoku"
    # Fallback heuristic: mazes are typically larger odd grids
    if grid_size >= 7 and grid_size % 2 == 1:
        return "maze"
    return "sudoku"


def _serialisable(obj: Any) -> Any:
    """Convert numpy / tuple types so ``json.dump`` succeeds."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
