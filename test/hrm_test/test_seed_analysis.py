"""Tests for hrm.training.seed_analysis — aggregation and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hrm.training.seed_analysis import (
    aggregate_seeds,
    print_summary,
    save_summary,
)

# =========================================================================
# Fixtures
# =========================================================================

def _make_history(seed: int, num_epochs: int = 10) -> dict[str, list]:
    """Generate a fake training history with small seed-dependent noise."""
    rng = np.random.RandomState(seed)
    base_loss = 1.5 - np.linspace(0, 0.8, num_epochs)
    noise = rng.normal(0, 0.01, num_epochs)

    return {
        "train_loss": (base_loss + noise).tolist(),
        "train_accuracy": np.clip(
            np.linspace(0.3, 0.9, num_epochs) + rng.normal(0, 0.005, num_epochs),
            0, 1,
        ).tolist(),
        "val_loss": (base_loss + 0.05 + rng.normal(0, 0.01, num_epochs)).tolist(),
        "val_token_accuracy": np.clip(
            np.linspace(0.25, 0.85, num_epochs) + rng.normal(0, 0.005, num_epochs),
            0, 1,
        ).tolist(),
        "val_puzzle_accuracy": np.clip(
            np.linspace(0.0, 0.6, num_epochs) + rng.normal(0, 0.005, num_epochs),
            0, 1,
        ).tolist(),
    }


@pytest.fixture
def seed_run_dir(tmp_path: Path) -> tuple[Path, list[int]]:
    """Create a temporary seed_run directory with 3 fake history files."""
    seeds = [42, 123, 456]
    puzzle = "sudoku_9x9"

    for seed in seeds:
        seed_dir = tmp_path / f"seed_{seed}"
        seed_dir.mkdir()
        hist = _make_history(seed)
        hist_path = seed_dir / f"training_history_simplified_{puzzle}.json"
        with open(hist_path, "w") as f:
            json.dump(hist, f)

    return tmp_path, seeds


# =========================================================================
# Tests
# =========================================================================

class TestAggregateSeeds:
    """Tests for aggregate_seeds()."""

    def test_basic_aggregation(self, seed_run_dir: tuple[Path, list[int]]):
        run_dir, seeds = seed_run_dir
        summary = aggregate_seeds(run_dir, seeds, "sudoku_9x9")

        assert summary["seeds"] == seeds
        assert summary["num_epochs"] == 10
        assert "val_puzzle_accuracy" in summary["metrics"]
        assert "train_loss" in summary["metrics"]

    def test_mean_std_shapes(self, seed_run_dir: tuple[Path, list[int]]):
        run_dir, seeds = seed_run_dir
        summary = aggregate_seeds(run_dir, seeds, "sudoku_9x9")

        for key, data in summary["metrics"].items():
            assert len(data["mean"]) == 10, f"{key} mean length wrong"
            assert len(data["std"]) == 10, f"{key} std length wrong"
            assert len(data["per_seed"]) == 3, f"{key} per_seed count wrong"

    def test_final_epoch_values(self, seed_run_dir: tuple[Path, list[int]]):
        run_dir, seeds = seed_run_dir
        summary = aggregate_seeds(run_dir, seeds, "sudoku_9x9")

        for _key, data in summary["final"].items():
            assert "mean" in data
            assert "std" in data
            assert len(data["values"]) == 3

    def test_variance_pct_computed(self, seed_run_dir: tuple[Path, list[int]]):
        run_dir, seeds = seed_run_dir
        summary = aggregate_seeds(run_dir, seeds, "sudoku_9x9")

        for _key, var in summary["variance_pct"].items():
            assert isinstance(var, float)
            assert var >= 0.0

    def test_low_variance_with_deterministic_data(self, tmp_path: Path):
        """Identical histories across seeds should give 0% variance."""
        seeds = [1, 2, 3]
        puzzle = "test"
        hist = {"val_puzzle_accuracy": [0.5, 0.6, 0.7]}

        for seed in seeds:
            d = tmp_path / f"seed_{seed}"
            d.mkdir()
            with open(d / f"training_history_simplified_{puzzle}.json", "w") as f:
                json.dump(hist, f)

        summary = aggregate_seeds(tmp_path, seeds, puzzle)
        assert summary["variance_pct"]["val_puzzle_accuracy"] == pytest.approx(0.0)

    def test_missing_seed_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            aggregate_seeds(tmp_path, [42], "sudoku_9x9")

    def test_unequal_lengths_truncates(self, tmp_path: Path):
        """When one seed trained fewer epochs, all are truncated to min."""
        seeds = [1, 2]
        puzzle = "test"

        for i, seed in enumerate(seeds):
            d = tmp_path / f"seed_{seed}"
            d.mkdir()
            n = 10 if i == 0 else 7  # seed 2 has fewer epochs
            hist = {"train_loss": list(range(n))}
            with open(d / f"training_history_simplified_{puzzle}.json", "w") as f:
                json.dump(hist, f)

        summary = aggregate_seeds(tmp_path, seeds, puzzle)
        assert summary["num_epochs"] == 7


class TestPrintSummary:
    def test_produces_output(self, seed_run_dir: tuple[Path, list[int]], capsys):
        run_dir, seeds = seed_run_dir
        summary = aggregate_seeds(run_dir, seeds, "sudoku_9x9")
        text = print_summary(summary)

        assert "Multi-Seed Summary" in text
        assert "val_puzzle_accuracy" in text


class TestSaveSummary:
    def test_saves_json(self, seed_run_dir: tuple[Path, list[int]]):
        run_dir, seeds = seed_run_dir
        summary = aggregate_seeds(run_dir, seeds, "sudoku_9x9")
        path = save_summary(summary, run_dir, "sudoku_9x9")

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["seeds"] == seeds
