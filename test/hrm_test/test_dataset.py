"""
Unit tests for SudokuDataset.

Covers: init validation, __len__, __getitem__ output format, target ranges,
difficulty filtering, caching vs on-the-fly, generate_batch, reproducibility,
DataLoader integration, and edge cases.

Run:
    python -m pytest test_dataset.py -v
"""

import pytest
import torch
from torch.utils.data import DataLoader

from hrm.data.dataset import SudokuDataset

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def ds_4x4():
    """Small cached 4x4 dataset."""
    return SudokuDataset(num_puzzles=20, grid_size=4, difficulty="medium", seed=42)


@pytest.fixture
def ds_4x4_easy():
    return SudokuDataset(num_puzzles=10, grid_size=4, difficulty="easy", seed=99)


@pytest.fixture
def ds_4x4_hard():
    return SudokuDataset(num_puzzles=10, grid_size=4, difficulty="hard", seed=99)


@pytest.fixture
def ds_4x4_mixed():
    return SudokuDataset(num_puzzles=12, grid_size=4, difficulty=None, seed=42)


@pytest.fixture
def ds_nocache():
    return SudokuDataset(num_puzzles=10, grid_size=4, difficulty="easy", seed=42, cache=False)


# =====================================================================
# Init validation
# =====================================================================


class TestInit:
    def test_valid_4x4(self, ds_4x4):
        assert ds_4x4.grid_size == 4
        assert ds_4x4.num_cells == 16
        assert ds_4x4.num_digits == 4

    def test_valid_9x9(self):
        ds = SudokuDataset(num_puzzles=5, grid_size=9, difficulty="easy", seed=1)
        assert ds.grid_size == 9
        assert ds.num_cells == 81

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError, match="grid_size must be 4 or 9"):
            SudokuDataset(num_puzzles=5, grid_size=6)

    def test_invalid_num_puzzles(self):
        with pytest.raises(ValueError, match="num_puzzles must be positive"):
            SudokuDataset(num_puzzles=0, grid_size=4)

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError, match="difficulty must be"):
            SudokuDataset(num_puzzles=5, grid_size=4, difficulty="extreme")

    def test_none_difficulty_allowed(self):
        ds = SudokuDataset(num_puzzles=3, grid_size=4, difficulty=None, seed=1)
        assert ds.difficulty is None


# =====================================================================
# __len__
# =====================================================================


class TestLen:
    def test_length_matches(self, ds_4x4):
        assert len(ds_4x4) == 20

    def test_length_small(self):
        ds = SudokuDataset(num_puzzles=1, grid_size=4, seed=1)
        assert len(ds) == 1


# =====================================================================
# __getitem__ output format
# =====================================================================


class TestGetItem:
    def test_returns_dict(self, ds_4x4):
        sample = ds_4x4[0]
        assert isinstance(sample, dict)

    def test_required_keys(self, ds_4x4):
        sample = ds_4x4[0]
        expected = {"puzzle", "solution", "target_cell", "target_digit", "difficulty"}
        assert set(sample.keys()) == expected

    def test_puzzle_shape(self, ds_4x4):
        assert ds_4x4[0]["puzzle"].shape == (4, 4)

    def test_solution_shape(self, ds_4x4):
        assert ds_4x4[0]["solution"].shape == (4, 4)

    def test_puzzle_dtype(self, ds_4x4):
        assert ds_4x4[0]["puzzle"].dtype == torch.long

    def test_solution_dtype(self, ds_4x4):
        assert ds_4x4[0]["solution"].dtype == torch.long

    def test_puzzle_values_in_range(self, ds_4x4):
        p = ds_4x4[0]["puzzle"]
        assert p.min() >= 0 and p.max() <= 4

    def test_solution_no_zeros(self, ds_4x4):
        s = ds_4x4[0]["solution"]
        assert s.min() >= 1 and s.max() <= 4

    def test_puzzle_has_empty_cells(self, ds_4x4):
        assert (ds_4x4[0]["puzzle"] == 0).sum() > 0

    def test_9x9_shapes(self):
        ds = SudokuDataset(num_puzzles=3, grid_size=9, difficulty="easy", seed=1)
        sample = ds[0]
        assert sample["puzzle"].shape == (9, 9)
        assert sample["solution"].shape == (9, 9)

    def test_index_out_of_range(self, ds_4x4):
        with pytest.raises(IndexError):
            ds_4x4[20]

    def test_negative_index_raises(self, ds_4x4):
        with pytest.raises(IndexError):
            ds_4x4[-1]


# =====================================================================
# Target cell and digit ranges
# =====================================================================


class TestTargetRanges:
    def test_target_cell_in_range(self, ds_4x4):
        for i in range(len(ds_4x4)):
            assert 0 <= ds_4x4[i]["target_cell"] < 16

    def test_target_digit_in_range(self, ds_4x4):
        for i in range(len(ds_4x4)):
            assert 0 <= ds_4x4[i]["target_digit"] < 4

    def test_target_cell_is_empty(self, ds_4x4):
        """Target cell should be zero in the puzzle grid."""
        for i in range(len(ds_4x4)):
            s = ds_4x4[i]
            row, col = divmod(s["target_cell"], 4)
            assert s["puzzle"][row, col].item() == 0

    def test_target_digit_matches_solution(self, ds_4x4):
        """target_digit + 1 should equal solution value at target cell."""
        for i in range(len(ds_4x4)):
            s = ds_4x4[i]
            row, col = divmod(s["target_cell"], 4)
            assert s["target_digit"] == s["solution"][row, col].item() - 1

    def test_9x9_target_ranges(self):
        ds = SudokuDataset(num_puzzles=5, grid_size=9, difficulty="easy", seed=1)
        for i in range(len(ds)):
            s = ds[i]
            assert 0 <= s["target_cell"] < 81
            assert 0 <= s["target_digit"] < 9


# =====================================================================
# Difficulty
# =====================================================================


class TestDifficulty:
    def test_difficulty_string(self, ds_4x4):
        assert ds_4x4[0]["difficulty"] == "medium"

    def test_easy(self, ds_4x4_easy):
        assert ds_4x4_easy[0]["difficulty"] == "easy"

    def test_hard(self, ds_4x4_hard):
        assert ds_4x4_hard[0]["difficulty"] == "hard"

    def test_mixed_has_all(self, ds_4x4_mixed):
        diffs = {ds_4x4_mixed[i]["difficulty"] for i in range(12)}
        assert {"easy", "medium", "hard"} == diffs

    def test_subset_filter(self, ds_4x4_mixed):
        easy_idx = ds_4x4_mixed.get_difficulty_subset("easy")
        assert len(easy_idx) > 0
        for i in easy_idx:
            assert ds_4x4_mixed[i]["difficulty"] == "easy"

    def test_subset_requires_cache(self, ds_nocache):
        with pytest.raises(RuntimeError):
            ds_nocache.get_difficulty_subset("easy")


# =====================================================================
# Caching
# =====================================================================


class TestCaching:
    def test_cached_arrays(self, ds_4x4):
        assert ds_4x4._puzzles is not None
        assert ds_4x4._puzzles.shape == (20, 4, 4)

    def test_nocache_no_arrays(self, ds_nocache):
        assert ds_nocache._puzzles is None

    def test_nocache_getitem(self, ds_nocache):
        s = ds_nocache[0]
        assert s["puzzle"].shape == (4, 4)

    def test_cached_same_puzzle_twice(self):
        ds = SudokuDataset(num_puzzles=5, grid_size=4, seed=42, cache=True)
        assert torch.equal(ds[0]["puzzle"], ds[0]["puzzle"])


# =====================================================================
# Reproducibility
# =====================================================================


class TestReproducibility:
    def test_same_seed(self):
        ds1 = SudokuDataset(num_puzzles=10, grid_size=4, seed=42)
        ds2 = SudokuDataset(num_puzzles=10, grid_size=4, seed=42)
        for i in range(10):
            assert torch.equal(ds1[i]["puzzle"], ds2[i]["puzzle"])
            assert torch.equal(ds1[i]["solution"], ds2[i]["solution"])

    def test_different_seed(self):
        ds1 = SudokuDataset(num_puzzles=10, grid_size=4, seed=42)
        ds2 = SudokuDataset(num_puzzles=10, grid_size=4, seed=99)
        differ = any(not torch.equal(ds1[i]["puzzle"], ds2[i]["puzzle"]) for i in range(10))
        assert differ


# =====================================================================
# generate_batch
# =====================================================================


class TestGenerateBatch:
    def test_shapes(self, ds_4x4):
        b = ds_4x4.generate_batch(8)
        assert b["puzzles"].shape == (8, 4, 4)
        assert b["solutions"].shape == (8, 4, 4)
        assert b["target_cells"].shape == (8,)
        assert b["target_digits"].shape == (8,)
        assert len(b["difficulties"]) == 8

    def test_dtypes(self, ds_4x4):
        b = ds_4x4.generate_batch(4)
        assert b["puzzles"].dtype == torch.long
        assert b["target_cells"].dtype == torch.long

    def test_difficulty_override(self, ds_4x4):
        b = ds_4x4.generate_batch(4, difficulty="hard")
        assert all(d == "hard" for d in b["difficulties"])

    def test_target_ranges(self, ds_4x4):
        b = ds_4x4.generate_batch(16)
        assert (b["target_cells"] >= 0).all() and (b["target_cells"] < 16).all()
        assert (b["target_digits"] >= 0).all() and (b["target_digits"] < 4).all()

    def test_reproducibility(self, ds_4x4):
        b1 = ds_4x4.generate_batch(5, seed=7)
        b2 = ds_4x4.generate_batch(5, seed=7)
        assert torch.equal(b1["puzzles"], b2["puzzles"])

    def test_9x9_batch(self):
        ds = SudokuDataset(num_puzzles=3, grid_size=9, difficulty="easy", seed=1)
        b = ds.generate_batch(4, seed=10)
        assert b["puzzles"].shape == (4, 9, 9)


# =====================================================================
# DataLoader integration
# =====================================================================


class TestDataLoader:
    def test_basic(self, ds_4x4):
        loader = DataLoader(ds_4x4, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["puzzle"].shape == (4, 4, 4)
        assert batch["target_cell"].shape == (4,)

    def test_full_epoch(self, ds_4x4):
        loader = DataLoader(ds_4x4, batch_size=8, shuffle=True)
        total = sum(b["puzzle"].shape[0] for b in loader)
        assert total == 20


# =====================================================================
# Statistics and repr
# =====================================================================


class TestMisc:
    def test_statistics_cached(self, ds_4x4):
        stats = ds_4x4.get_statistics()
        assert stats["cached"] is True
        assert stats["num_puzzles"] == 20
        assert stats["avg_empty_cells"] > 0

    def test_statistics_uncached(self, ds_nocache):
        assert ds_nocache.get_statistics()["cached"] is False

    def test_repr(self, ds_4x4):
        r = repr(ds_4x4)
        assert "SudokuDataset" in r and "num_puzzles=20" in r


# =====================================================================
# Transform
# =====================================================================


class TestTransform:
    def test_transform_applied(self):
        ds = SudokuDataset(
            num_puzzles=3,
            grid_size=4,
            seed=1,
            transform=lambda s: {**s, "flag": True},
        )
        assert ds[0]["flag"] is True


# =====================================================================
# Solution validity
# =====================================================================


class TestSolutionValidity:
    def _valid_4x4(self, sol):
        exp = {1, 2, 3, 4}
        for r in range(4):
            if set(sol[r].tolist()) != exp:
                return False
        for c in range(4):
            if set(sol[:, c].tolist()) != exp:
                return False
        for br in range(2):
            for bc in range(2):
                blk = sol[br * 2 : (br + 1) * 2, bc * 2 : (bc + 1) * 2]
                if set(blk.flatten().tolist()) != exp:
                    return False
        return True

    def test_solutions_valid(self, ds_4x4):
        for i in range(len(ds_4x4)):
            assert self._valid_4x4(ds_4x4[i]["solution"])

    def test_puzzle_matches_solution(self, ds_4x4):
        for i in range(len(ds_4x4)):
            s = ds_4x4[i]
            mask = s["puzzle"] != 0
            assert torch.equal(s["puzzle"][mask], s["solution"][mask])
