# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Tests for the Dataset classes used in training.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "hrm"))

from hrm.prototype.generators.generator_4x4 import generate_dataset


class Dataset_4x4:
    """Copy of Dataset_4x4 for testing purposes"""

    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray):
        self.puzzles = torch.from_numpy(puzzles).long()

        # Target: first empty cell and its digit
        self.targets = []
        for puzzle, solution in zip(puzzles, solutions):
            empty = np.argwhere(puzzle == 0)
            if len(empty) > 0:
                r, c = empty[0]
                cell_idx = r * 4 + c
                digit = solution[r, c] - 1  # 0-indexed
            else:
                cell_idx = 0
                digit = 0
            self.targets.append((cell_idx, digit))

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return self.puzzles[idx], self.targets[idx][0], self.targets[idx][1]


class TestDataset4x4:
    """Tests for the Dataset_4x4 class"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample puzzles and solutions"""
        puzzles, solutions = generate_dataset(10, num_clues=10)
        return puzzles, solutions

    def test_initialization(self, sample_data):
        """Test dataset initializes correctly"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        assert len(dataset) == 10
        assert len(dataset.targets) == 10

    def test_puzzle_tensor_type(self, sample_data):
        """Test puzzles are converted to long tensors"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        assert dataset.puzzles.dtype == torch.long

    def test_puzzle_shape(self, sample_data):
        """Test puzzles maintain correct shape"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        assert dataset.puzzles.shape == (10, 4, 4)

    def test_getitem_returns_tuple(self, sample_data):
        """Test __getitem__ returns correct format"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        puzzle, cell_idx, digit = dataset[0]

        assert isinstance(puzzle, torch.Tensor)
        assert isinstance(cell_idx, (int, np.integer))
        assert isinstance(digit, (int, np.integer))

    def test_cell_index_range(self, sample_data):
        """Test cell indices are in valid range (0-15)"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        for i in range(len(dataset)):
            _, cell_idx, _ = dataset[i]
            assert 0 <= cell_idx < 16

    def test_digit_range(self, sample_data):
        """Test digits are in valid range (0-3, 0-indexed)"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        for i in range(len(dataset)):
            _, _, digit = dataset[i]
            assert 0 <= digit < 4

    def test_target_corresponds_to_empty_cell(self, sample_data):
        """Test that target cell index corresponds to an empty cell"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        for i in range(len(dataset)):
            puzzle, cell_idx, digit = dataset[i]

            row = cell_idx // 4
            col = cell_idx % 4

            # Cell should be empty in puzzle (0)
            assert puzzle[row, col].item() == 0

    def test_target_digit_matches_solution(self, sample_data):
        """Test that target digit matches the solution"""
        puzzles, solutions = sample_data
        dataset = Dataset_4x4(puzzles, solutions)

        for i in range(len(dataset)):
            puzzle, cell_idx, digit = dataset[i]

            row = cell_idx // 4
            col = cell_idx % 4

            # Digit should match solution (convert from 0-indexed)
            expected_digit = solutions[i, row, col] - 1
            assert digit == expected_digit

    def test_empty_puzzle_handling(self):
        """Test dataset handles puzzle with no empty cells"""
        # Create a fully filled puzzle
        puzzles = np.array([[[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]])
        solutions = puzzles.copy()

        dataset = Dataset_4x4(puzzles, solutions)
        _, cell_idx, digit = dataset[0]

        # Should default to 0, 0
        assert cell_idx == 0
        assert digit == 0


class TestDataLoaderCompatibility:
    """Tests for PyTorch DataLoader compatibility"""

    @pytest.fixture
    def dataset(self):
        """Create dataset fixture"""
        puzzles, solutions = generate_dataset(20, num_clues=10)
        return Dataset_4x4(puzzles, solutions)

    def test_dataloader_batching(self, dataset):
        """Test dataset works with DataLoader batching"""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        puzzles, cells, digits = batch

        assert puzzles.shape == (4, 4, 4)
        assert cells.shape == (4,)
        assert digits.shape == (4,)

    def test_dataloader_shuffling(self, dataset):
        """Test dataset works with DataLoader shuffling"""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Should be able to iterate
        for puzzles, cells, digits in loader:
            assert puzzles.shape[0] <= 4
            break

    def test_dataloader_iteration(self, dataset):
        """Test complete iteration through DataLoader"""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        total_samples = 0
        for puzzles, cells, digits in loader:
            total_samples += puzzles.shape[0]

        assert total_samples == len(dataset)


class TestDataPreprocessing:
    """Tests for data preprocessing and augmentation"""

    def test_value_range(self):
        """Test that puzzle values are in expected range (0-4)"""
        puzzles, solutions = generate_dataset(10)
        dataset = Dataset_4x4(puzzles, solutions)

        assert dataset.puzzles.min() >= 0
        assert dataset.puzzles.max() <= 4

    def test_solution_value_range(self):
        """Test that solution values are in expected range (1-4)"""
        puzzles, solutions = generate_dataset(10)

        assert solutions.min() >= 1
        assert solutions.max() <= 4

    def test_first_empty_cell_selection(self):
        """Test that first empty cell is consistently selected"""
        # Create puzzle with known empty cells
        puzzle = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        solution = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]])

        puzzles = np.array([puzzle])
        solutions = np.array([solution])

        dataset = Dataset_4x4(puzzles, solutions)
        _, cell_idx, digit = dataset[0]

        # First empty cell is (0, 1) = index 1
        assert cell_idx == 1
        assert digit == 1  # solution[0, 1] - 1 = 2 - 1 = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
