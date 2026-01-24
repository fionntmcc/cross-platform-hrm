import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class HRM_4x4(nn.Module):
    def __init__(self, hidden_dim: int = 64, max_iterations: int = 10):
        super().__init__()
        self.max_iterations = max_iterations
        
        # Embedding layer for puzzle state (0-4)
        self.embed = nn.Embedding(5, 16)
        
        # Simple feedforward layers for planner. This needs to be much more complex to solve 9x9 Sudoku grid.
        # This should work for a 4x4 grid.
        self.planner = nn.Sequential(
            nn.Linear(16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # High-level planner: puzzle -> abstract representation
        # Linear applies correlations across all 16 cells
        # ReLU applies non-linearity, allowing for "if-then" reasoning (needed for Sudoku rules)
        # Linear reduces the reasoning to a final state that the decoder can use to make a prediction
        self.worker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: predict cell and digit
        self.cell_decoder = nn.Linear(hidden_dim, 16)   # 16 cells
        self.digit_decoder = nn.Linear(hidden_dim, 4)   # 4 digits

        
    def forward(self, puzzle: torch.Tensor, return_traces: bool = False) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass of the HRM model for 4x4 Sudoku.
        Args:
            puzzle: (batch, 4, 4) integers 0-4
            return_traces: Whether to return execution traces
        Returns:
            cell_logits: (batch, 16) 
            digit_logits: (batch, 4)
            traces: dict with execution stats
        """
        batch_size = puzzle.shape[0]
        
        # Get embeddings and flatten them
        x = self.embed(puzzle).view(batch_size, -1)
        
        # H-module
        abstract = self.planner(x)
        
        # L-module
        refined = self.worker(abstract)
        
        # Decode actions
        cell_logits = self.cell_decoder(refined)
        digit_logits = self.digit_decoder(refined)
        
        traces = {
            'num_iterations': 1,
            'residuals': [0.0]
        }
        
        return cell_logits, digit_logits, traces

    def get_halt_penalty(self, traces: dict, target_iterations: int) -> torch.Tensor:
        return torch.tensor(0.0, device=self.embed.weight.device)
