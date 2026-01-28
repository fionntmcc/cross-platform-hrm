"""
Output Network for HRM (Action Decoder f_O)

This module provides the output decoder that maps the final hidden state
to Sudoku actions. The decoder produces two outputs:
    1. Cell selection: Which cell to fill next
    2. Digit selection: Which digit to place in the selected cell

For Sudoku:
    - 4x4 puzzles: 16 cells, 4 possible digits (1-4)
    - 9x9 puzzles: 81 cells, 9 possible digits (1-9)

The decoder takes the final h_H state from the Planner after hierarchical
convergence and produces action logits that can be used for:
    - Training: Cross-entropy loss against ground truth actions
    - Inference: Argmax or sampling to select actions
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.norm import RMSNorm


class OutputNetwork(nn.Module):
    """
    Output decoder network for mapping hidden states to Sudoku actions.
    
    Takes the final hidden representation from the HRM and produces:
    - cell_logits: Probability distribution over which cell to fill
    - digit_logits: Probability distribution over which digit to place
    
    Architecture:
        1. Shared feature extraction (optional MLP)
        2. Cell head: Linear projection to num_cells
        3. Digit head: Linear projection to num_digits
        4. Optional softmax for inference
    
    Args:
        hidden_dim: Dimension of the input hidden state.
        grid_size: Size of the Sudoku grid (4 for 4x4, 9 for 9x9).
        use_shared_mlp: Whether to use shared MLP before heads. Default: True
        mlp_ratio: Expansion ratio for shared MLP. Default: 2
        dropout: Dropout probability. Default: 0.1
    
    Shape:
        - Input: (batch, hidden_dim) - Final hidden state
        - cell_logits: (batch, grid_size²) - Logits for cell selection
        - digit_logits: (batch, grid_size) - Logits for digit selection
    
    Example:
        >>> # 4x4 Sudoku configuration
        >>> output_net = OutputNetwork(hidden_dim=64, grid_size=4)
        >>> h_final = torch.randn(8, 64)  # Final hidden state
        >>> cell_logits, digit_logits = output_net(h_final)
        >>> cell_logits.shape
        torch.Size([8, 16])
        >>> digit_logits.shape
        torch.Size([8, 4])
    """
    
    def __init__(
        self,
        hidden_dim: int,
        grid_size: int,
        use_shared_mlp: bool = True,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialise OutputNetwork.
        
        Args:
            hidden_dim: Dimension of input hidden states.
            grid_size: Size of the Sudoku grid (e.g., 4 for 4x4).
            use_shared_mlp: Whether to apply shared MLP before heads.
            mlp_ratio: Expansion ratio for shared MLP intermediate dim.
            dropout: Dropout probability for regularisation.
        """
        super().__init__()
        
        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.num_digits = grid_size  # Digits 1 to grid_size
        self.use_shared_mlp = use_shared_mlp
        
        # Shared feature extraction (optional)
        if use_shared_mlp:
            intermediate_dim = hidden_dim * mlp_ratio
            self.shared_mlp = nn.Sequential(
                nn.Linear(hidden_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            self.norm = RMSNorm(dim=hidden_dim)
        else:
            self.shared_mlp = None
            self.norm = None
        
        # Cell selection head: hidden_dim -> num_cells
        self.cell_head = nn.Linear(hidden_dim, self.num_cells)
        
        # Digit selection head: hidden_dim -> num_digits
        self.digit_head = nn.Linear(hidden_dim, self.num_digits)
        
        # Initialise weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialise weights using Xavier initialisation."""
        # Shared MLP initialisation
        if self.shared_mlp is not None:
            for module in self.shared_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Head initialisation
        nn.init.xavier_uniform_(self.cell_head.weight)
        nn.init.zeros_(self.cell_head.bias)
        
        nn.init.xavier_uniform_(self.digit_head.weight)
        nn.init.zeros_(self.digit_head.bias)
    
    def forward(
        self,
        h: torch.Tensor,
        apply_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode hidden state to Sudoku action logits.
        
        Args:
            h: Hidden state tensor of shape (batch, hidden_dim).
                Typically the final h_H state after hierarchical convergence.
            apply_softmax: Whether to apply softmax to outputs. Default: False.
                Set to True during inference for probability distributions.
        
        Returns:
            Tuple of (cell_logits, digit_logits):
                - cell_logits: (batch, num_cells) - Which cell to fill
                - digit_logits: (batch, num_digits) - Which digit to place
        
        Note:
            During training, use raw logits with CrossEntropyLoss.
            During inference, set apply_softmax=True for probabilities.
        """
        # Validate input shape
        if h.dim() != 2 or h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected input shape (batch, {self.hidden_dim}), got {h.shape}"
            )
        
        # Shared feature extraction
        if self.shared_mlp is not None:
            h = h + self.shared_mlp(h)  # Residual connection
            h = self.norm(h)
        
        # Compute logits from heads
        cell_logits = self.cell_head(h)
        digit_logits = self.digit_head(h)
        
        # Apply softmax if requested (for inference)
        if apply_softmax:
            cell_logits = F.softmax(cell_logits, dim=-1)
            digit_logits = F.softmax(digit_logits, dim=-1)
        
        return cell_logits, digit_logits
    
    def predict(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicted cell and digit indices for inference.
        
        Convenience method that returns argmax of logits.
        
        Args:
            h: Hidden state tensor of shape (batch, hidden_dim).
        
        Returns:
            Tuple of (cell_indices, digit_indices):
                - cell_indices: (batch,) - Predicted cell index [0, num_cells)
                - digit_indices: (batch,) - Predicted digit index [0, num_digits)
                
        Note:
            digit_indices are 0-indexed. Add 1 to get actual Sudoku digits.
        """
        cell_logits, digit_logits = self.forward(h, apply_softmax=False)
        
        cell_indices = cell_logits.argmax(dim=-1)
        digit_indices = digit_logits.argmax(dim=-1)
        
        return cell_indices, digit_indices
    
    def predict_with_confidence(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.
        
        Args:
            h: Hidden state tensor of shape (batch, hidden_dim).
        
        Returns:
            Tuple of (cell_indices, digit_indices, cell_confidence, digit_confidence):
                - cell_indices: (batch,) - Predicted cell index
                - digit_indices: (batch,) - Predicted digit index  
                - cell_confidence: (batch,) - Confidence for cell prediction
                - digit_confidence: (batch,) - Confidence for digit prediction
        """
        cell_probs, digit_probs = self.forward(h, apply_softmax=True)
        
        cell_confidence, cell_indices = cell_probs.max(dim=-1)
        digit_confidence, digit_indices = digit_probs.max(dim=-1)
        
        return cell_indices, digit_indices, cell_confidence, digit_confidence
    
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"grid_size={self.grid_size}, "
            f"num_cells={self.num_cells}, "
            f"num_digits={self.num_digits}, "
            f"use_shared_mlp={self.use_shared_mlp}"
        )


def create_output_network(
    hidden_dim: int,
    grid_size: int,
    **kwargs
) -> OutputNetwork:
    """
    Factory function to create OutputNetwork.
    
    Args:
        hidden_dim: Dimension of input hidden states.
        grid_size: Size of the Sudoku grid.
        **kwargs: Additional arguments (use_shared_mlp, mlp_ratio, dropout).
    
    Returns:
        Configured OutputNetwork instance.
    
    Example:
        >>> # 4x4 Sudoku decoder
        >>> output_net = create_output_network(hidden_dim=64, grid_size=4)
        
        >>> # 9x9 Sudoku decoder
        >>> output_net = create_output_network(hidden_dim=128, grid_size=9)
    """
    return OutputNetwork(
        hidden_dim=hidden_dim,
        grid_size=grid_size,
        **kwargs
    )
