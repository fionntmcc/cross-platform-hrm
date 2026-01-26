"""
Input Network for HRM

Issue #2: Implement InputNetwork (Embedding + Projection)

This module provides the input embedding network that converts raw Sudoku
puzzles into the hidden representation space used by the HRM architecture.

The InputNetwork performs two main operations:
1. Embedding: Convert discrete cell values to continuous representations
2. Projection: Transform flattened embeddings to the hidden dimension

For Sudoku:
    - 4x4 puzzles: vocab_size=5 (0=empty, 1-4=digits)
    - 9x9 puzzles: vocab_size=10 (0=empty, 1-9=digits)
"""

from typing import Optional

import torch
import torch.nn as nn

from hrm.layers.norm import RMSNorm


class InputNetwork(nn.Module):
    """
    Input embedding network for converting Sudoku grids to hidden representations.
    
    Takes a 2D grid of integer cell values and produces a single hidden vector
    that represents the entire puzzle state. This hidden representation is then
    used by the Planner (H-module) and Worker (L-module) for iterative reasoning.
    
    Architecture:
        1. Embedding layer: (batch, H, W) -> (batch, H, W, embed_dim)
        2. Positional encoding: adds spatial information
        3. Flatten: (batch, H, W, embed_dim) -> (batch, H*W*embed_dim)
        4. Projection MLP: (batch, H*W*embed_dim) -> (batch, hidden_dim)
        5. Normalisation: RMSNorm for training stability
    
    Args:
        vocab_size: Size of the vocabulary (number of possible cell values).
            For 4x4 Sudoku: 5 (0=empty, 1-4=digits)
            For 9x9 Sudoku: 10 (0=empty, 1-9=digits)
        grid_size: Size of the Sudoku grid (4 for 4x4, 9 for 9x9).
        embed_dim: Dimension of the embedding for each cell.
        hidden_dim: Dimension of the output hidden representation.
        dropout: Dropout probability. Default: 0.1
        use_positional: Whether to add positional encoding. Default: True
    
    Shape:
        - Input: (batch, grid_size, grid_size) with integer values in [0, vocab_size)
        - Output: (batch, hidden_dim) embedded representation
    
    Attributes:
        embedding: nn.Embedding layer for cell values
        pos_embedding: Positional embedding for grid positions
        projection: MLP that projects flattened embeddings to hidden_dim
        norm: RMSNorm layer for output normalisation
    
    Example:
        >>> # 4x4 Sudoku configuration
        >>> input_net = InputNetwork(
        ...     vocab_size=5,
        ...     grid_size=4,
        ...     embed_dim=16,
        ...     hidden_dim=64
        ... )
        >>> puzzle = torch.randint(0, 5, (8, 4, 4))  # batch of 8 puzzles
        >>> hidden = input_net(puzzle)
        >>> hidden.shape
        torch.Size([8, 64])
    """
    
    def __init__(
        self,
        vocab_size: int,
        grid_size: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_positional: bool = True,
    ):
        """
        Initialise InputNetwork.
        
        Args:
            vocab_size: Number of possible cell values (e.g., 5 for 4x4 Sudoku).
            grid_size: Size of the square grid (e.g., 4 for 4x4 Sudoku).
            embed_dim: Embedding dimension per cell.
            hidden_dim: Output hidden dimension.
            dropout: Dropout probability for regularisation.
            use_positional: Whether to use positional encoding.
        """
        super().__init__()
        
        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        # Store configuration
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_positional = use_positional
        self.num_cells = grid_size * grid_size
        
        # Cell value embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        
        # Positional encoding for grid positions
        if use_positional:
            self.pos_embedding = nn.Embedding(
                num_embeddings=self.num_cells,
                embedding_dim=embed_dim,
            )
            # Register position indices as buffer (not a parameter)
            self.register_buffer(
                'position_ids',
                torch.arange(self.num_cells)
            )
        else:
            self.pos_embedding = None
            self.register_buffer('position_ids', None)
        
        # Calculate flattened dimension
        flat_dim = self.num_cells * embed_dim
        
        # Projection MLP: flat_dim -> hidden_dim
        # Two-layer MLP with GELU activation for expressiveness
        self.projection = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Output normalisation for training stability
        self.norm = RMSNorm(dim=hidden_dim)
        
        # Dropout for regularisation
        self.dropout = nn.Dropout(dropout)
        
        # Initialise weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialise weights using Xavier/Glorot initialisation."""
        # Embedding initialisation
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Linear layer initialisation
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert Sudoku grid to hidden representation.
        
        Args:
            x: Input tensor of shape (batch, grid_size, grid_size) containing
               integer cell values in range [0, vocab_size).
        
        Returns:
            Hidden representation of shape (batch, hidden_dim).
        
        Raises:
            ValueError: If input shape doesn't match expected grid_size.
        """
        batch_size = x.shape[0]
        
        # Validate input shape
        if x.shape[1:] != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Expected input shape (batch, {self.grid_size}, {self.grid_size}), "
                f"got {x.shape}"
            )
        
        # Flatten grid to sequence: (batch, H, W) -> (batch, H*W)
        x_flat = x.view(batch_size, self.num_cells)
        
        # Embed cell values: (batch, num_cells) -> (batch, num_cells, embed_dim)
        embeddings = self.embedding(x_flat)
        
        # Add positional encoding if enabled
        if self.use_positional and self.pos_embedding is not None:
            # position_ids: (num_cells,) -> broadcast to (batch, num_cells, embed_dim)
            pos_emb = self.pos_embedding(self.position_ids)
            embeddings = embeddings + pos_emb
        
        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)
        
        # Flatten all embeddings: (batch, num_cells, embed_dim) -> (batch, num_cells*embed_dim)
        flat_embeddings = embeddings.view(batch_size, -1)
        
        # Project to hidden dimension: (batch, flat_dim) -> (batch, hidden_dim)
        hidden = self.projection(flat_embeddings)
        
        # Apply normalisation
        hidden = self.norm(hidden)
        
        return hidden
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Get the embedding weight matrix.
        
        Useful for analysis and visualisation of learned representations.
        
        Returns:
            Embedding weight tensor of shape (vocab_size, embed_dim).
        """
        return self.embedding.weight.data
    
    def extra_repr(self) -> str:
        """Return a string representation of module parameters."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"grid_size={self.grid_size}, "
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"use_positional={self.use_positional}"
        )


def create_input_network(
    vocab_size: int,
    grid_size: int,
    embed_dim: int,
    hidden_dim: int,
    **kwargs
) -> InputNetwork:
    """
    Factory function to create InputNetwork.
    
    Args:
        vocab_size: Size of the vocabulary.
        grid_size: Size of the Sudoku grid.
        embed_dim: Embedding dimension per cell.
        hidden_dim: Output hidden dimension.
        **kwargs: Additional arguments (dropout, use_positional).
    
    Returns:
        Configured InputNetwork module.
    
    Example:
        >>> net = create_input_network(
        ...     vocab_size=5,
        ...     grid_size=4,
        ...     embed_dim=16,
        ...     hidden_dim=64
        ... )
    """
    return InputNetwork(
        vocab_size=vocab_size,
        grid_size=grid_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        **kwargs
    )
