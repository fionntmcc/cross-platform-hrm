# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""Tests for hrm.data.io — JSON/CSV dataset save and load."""

import json
from pathlib import Path

import numpy as np
import pytest

from hrm.data.io import load_dataset, save_dataset

# Fixtures


def _make_sudoku_dataset(n: int = 5, grid_size: int = 4) -> dict:
    """Create a small synthetic Sudoku-style dataset."""
    rng = np.random.default_rng(0)
    return {
        "problems": rng.integers(0, grid_size + 1, size=(n, grid_size, grid_size), dtype=np.int8),
        "solutions": rng.integers(1, grid_size + 1, size=(n, grid_size, grid_size), dtype=np.int8),
        "metadata": [
            {
                "puzzle_id": i,
                "empty_cells": int(rng.integers(4, 10)),
                "difficulty": "easy",
                "backtracks": 0,
                "grid_size": grid_size,
            }
            for i in range(n)
        ],
    }


def _make_maze_dataset(n: int = 5, grid_size: int = 15) -> dict:
    """Create a small synthetic maze-style dataset."""
    rng = np.random.default_rng(1)
    return {
        "problems": rng.integers(0, 10, size=(n, grid_size, grid_size), dtype=np.int8),
        "solutions": rng.integers(0, 2, size=(n, grid_size, grid_size), dtype=np.int8),
        "metadata": [
            {
                "puzzle_id": i,
                "grid_size": grid_size,
                "path_length": int(rng.integers(10, 30)),
                "path_cost": int(rng.integers(20, 60)),
                "start": (1, 1),
                "goal": (13, 13),
                "num_walls": int(rng.integers(40, 80)),
                "num_weighted": int(rng.integers(10, 30)),
            }
            for i in range(n)
        ],
    }


# JSON round-trip tests


class TestJsonRoundTrip:
    def test_sudoku_json_round_trip(self, tmp_path: Path):
        ds = _make_sudoku_dataset()
        out = save_dataset(ds, tmp_path / "sudoku.json", fmt="json", seed=42, difficulty="easy")

        loaded = load_dataset(out)
        np.testing.assert_array_equal(loaded["problems"], ds["problems"])
        np.testing.assert_array_equal(loaded["solutions"], ds["solutions"])
        assert loaded["file_metadata"]["seed"] == 42
        assert loaded["file_metadata"]["difficulty"] == "easy"
        assert loaded["file_metadata"]["puzzle_type"] == "sudoku"
        assert loaded["file_metadata"]["count"] == 5

    def test_maze_json_round_trip(self, tmp_path: Path):
        ds = _make_maze_dataset()
        out = save_dataset(ds, tmp_path / "maze.json", fmt="json", seed=7)

        loaded = load_dataset(out)
        np.testing.assert_array_equal(loaded["problems"], ds["problems"])
        np.testing.assert_array_equal(loaded["solutions"], ds["solutions"])
        assert loaded["file_metadata"]["puzzle_type"] == "maze"
        assert loaded["file_metadata"]["grid_size"] == 15

    def test_json_per_puzzle_metadata_preserved(self, tmp_path: Path):
        ds = _make_sudoku_dataset(n=2)
        out = save_dataset(ds, tmp_path / "test.json")
        loaded = load_dataset(out)
        assert len(loaded["metadata"]) == 2
        assert loaded["metadata"][0]["difficulty"] == "easy"

    def test_json_structure_matches_spec(self, tmp_path: Path):
        ds = _make_sudoku_dataset(n=1, grid_size=4)
        out = save_dataset(ds, tmp_path / "spec.json", seed=42)

        with open(out, encoding="utf-8") as f:
            doc = json.load(f)

        assert "metadata" in doc
        assert "puzzles" in doc
        assert "grid_size" in doc["metadata"]
        assert "count" in doc["metadata"]
        assert "seed" in doc["metadata"]
        assert "generated_at" in doc["metadata"]
        assert len(doc["puzzles"]) == 1
        assert "problem" in doc["puzzles"][0]
        assert "solution" in doc["puzzles"][0]


# CSV round-trip tests


class TestCsvRoundTrip:
    def test_sudoku_csv_round_trip(self, tmp_path: Path):
        ds = _make_sudoku_dataset()
        out = save_dataset(ds, tmp_path / "sudoku.csv", fmt="csv", seed=42, difficulty="easy")

        loaded = load_dataset(out)
        np.testing.assert_array_equal(loaded["problems"], ds["problems"])
        np.testing.assert_array_equal(loaded["solutions"], ds["solutions"])
        assert loaded["file_metadata"]["seed"] == 42

    def test_maze_csv_round_trip(self, tmp_path: Path):
        ds = _make_maze_dataset()
        out = save_dataset(ds, tmp_path / "maze.csv", fmt="csv", seed=7)

        loaded = load_dataset(out)
        np.testing.assert_array_equal(loaded["problems"], ds["problems"])
        np.testing.assert_array_equal(loaded["solutions"], ds["solutions"])
        assert loaded["file_metadata"]["puzzle_type"] == "maze"


# Error handling


class TestErrorHandling:
    def test_unsupported_format_raises(self, tmp_path: Path):
        ds = _make_sudoku_dataset(n=1)
        with pytest.raises(ValueError, match="Unsupported format"):
            save_dataset(ds, tmp_path / "test.xyz", fmt="xyz")

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file_12345.json")

    def test_load_unknown_extension_raises(self, tmp_path: Path):
        bad = tmp_path / "data.parquet"
        bad.write_text("")
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            load_dataset(bad)


# Extension normalisation


class TestExtensionNormalisation:
    def test_json_extension_added(self, tmp_path: Path):
        ds = _make_sudoku_dataset(n=1)
        out = save_dataset(ds, tmp_path / "noext", fmt="json")
        assert out.suffix == ".json"

    def test_csv_extension_added(self, tmp_path: Path):
        ds = _make_sudoku_dataset(n=1)
        out = save_dataset(ds, tmp_path / "noext", fmt="csv")
        assert out.suffix == ".csv"


# Dataset without metadata


class TestNoPerPuzzleMetadata:
    def test_json_no_metadata(self, tmp_path: Path):
        ds = {
            "problems": np.zeros((2, 4, 4), dtype=np.int8),
            "solutions": np.ones((2, 4, 4), dtype=np.int8),
        }
        out = save_dataset(ds, tmp_path / "bare.json")
        loaded = load_dataset(out)
        assert loaded["problems"].shape == (2, 4, 4)
        assert len(loaded["metadata"]) == 2

    def test_csv_no_metadata(self, tmp_path: Path):
        ds = {
            "problems": np.zeros((2, 4, 4), dtype=np.int8),
            "solutions": np.ones((2, 4, 4), dtype=np.int8),
        }
        out = save_dataset(ds, tmp_path / "bare.csv", fmt="csv")
        loaded = load_dataset(out)
        assert loaded["problems"].shape == (2, 4, 4)
