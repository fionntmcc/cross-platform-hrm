"""
Pytest configuration and shared fixtures for HRM tests.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add hrm to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / 'hrm'))


@pytest.fixture(scope="session")
def device():
    """Provide device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_puzzle():
    """Provide a sample 4x4 Sudoku puzzle"""
    puzzle = np.array([
        [1, 2, 0, 0],
        [0, 0, 1, 2],
        [2, 1, 0, 0],
        [0, 0, 2, 1]
    ])
    return puzzle


@pytest.fixture
def sample_solution():
    """Provide a sample 4x4 Sudoku solution"""
    solution = np.array([
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]
    ])
    return solution


@pytest.fixture
def sample_puzzle_tensor(sample_puzzle):
    """Provide sample puzzle as tensor"""
    return torch.from_numpy(sample_puzzle).unsqueeze(0).long()


@pytest.fixture
def batch_puzzles():
    """Provide a batch of random puzzles"""
    return torch.randint(0, 5, (8, 4, 4)).long()


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    yield


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
