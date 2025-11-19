"""
HRM Model for 4×4 Sudoku
Simpler version for proof of concept
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelPlanner4x4(nn.Module):
    """High-level reasoning module H - generates abstract plan"""
    
    def __init__(self, abstract_dim: int = 64):
        super().__init__()
        self.abstract_dim = abstract_dim
        
        # Embed 4×4 board (16 cells, values 0-4), each to a 32-dimensional vector
        self.embedding = nn.Embedding(5, 32)  # 5 values (0-4)
        
        # Process board to abstract representation
        # Model will learn to extract high-level features
        # Simple feedforward network to start
        # Linear layer to reduce dimensionality
        # Activation function is ReLU
        self.encoder = nn.Sequential(
            nn.Linear(16 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, abstract_dim)
        )
    
    def forward(self, puzzle: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of board state to abstract plan
        
        Args:
            puzzle: (batch, 4, 4) board state
        Returns:
            abstract_state: (batch, abstract_dim) high-level plan
        """
        batch_size = puzzle.shape[0]
        
        # Embed each cell
        embedded = self.embedding(puzzle)
        embedded = embedded.view(batch_size, -1)  # (batch, 512)
        
        # Generate abstract plan based on embedded board
        abstract_state = self.encoder(embedded)  # (batch, abstract_dim)
        
        return abstract_state


class LowLevelWorker4x4(nn.Module):
    """Low-level worker L - refines abstract plan through fixed-point iteration"""
    
    def __init__(self, abstract_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.abstract_dim = abstract_dim
        self.hidden_dim = hidden_dim
        
        # Worker processes abstract state
        # 3-layer feedforward
        # Up to 5 iterations of refinement
        # Stops if change is below tolerance (convergence)
        self.worker = nn.Sequential(
            nn.Linear(abstract_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, abstract_dim)
        )
        
        self.max_iterations = 5
        self.tolerance = 1e-3
    
    def forward(self, abstract_state: torch.Tensor) -> tuple:
        """
        Refine abstract state through fixed-point iteration
        
        Args:
            abstract_state: (batch, abstract_dim)
        Returns:
            refined_state: (batch, abstract_dim)
            metrics: dict with convergence info
        """
        state = abstract_state
        converged = False
        
        for iteration in range(self.max_iterations):
            new_state = self.worker(state)
            
            # Check convergence
            residual = torch.norm(new_state - state, dim=1).mean()
            
            if residual < self.tolerance:
                converged = True
                break
            
            state = new_state
        
        metrics = {
            'iterations': iteration + 1,
            'converged': converged,
            'residual': residual.item()
        }
        
        return state, metrics


class ActionDecoder4x4(nn.Module):
    """Decoder - converts refined state to action (cell + digit)"""
    
    def __init__(self, abstract_dim: int = 64):
        super().__init__()
        
        # Predict which cell to fill
        self.cell_head = nn.Sequential(
            nn.Linear(abstract_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        # Predict which digit (1-4)
        self.digit_head = nn.Sequential(
            nn.Linear(abstract_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, refined_state: torch.Tensor) -> tuple:
        """
        Args:
            refined_state: (batch, abstract_dim)
        Returns:
            cell_logits: (batch, 16) unnormalized scores for each cell
            digit_logits: (batch, 4) unnormalized scores for each digit
        """
        # Get logits for both cell and digit -> simple version that only gets 1 cell at a time
        cell_logits = self.cell_head(refined_state)
        digit_logits = self.digit_head(refined_state)
        
        return cell_logits, digit_logits