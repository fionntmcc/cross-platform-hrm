"""
Planner Module for HRM (High-Level Module f_H)

The Planner operates on a SLOW timescale, updating once per outer cycle
after the Worker (L-module) has converged. This hierarchical structure
enables sustained computational activity across 200+ steps, compared to
20-30 steps for standard RNNs.

Hierarchical Reset Mechanism:
    When the Planner updates, it incorporates the converged Worker state
    and produces a new h_H. This new h_H effectively "resets" the Worker's
    computational space, allowing the Worker to converge to a NEW equilibrium
    in the next cycle. This prevents the premature convergence that plagues
    standard recurrent networks.

    Cycle n:
        1. Worker iterates: h_L^(t+1) = f_L(h_L^t, h_H, x) until convergence
        2. Planner updates: h_H^(n+1) = f_H(h_H^n, h_L_converged)
        3. New h_H resets Worker's target equilibrium
        4. Repeat for cycle n+1
"""

import torch
import torch.nn as nn

from hrm.layers.norm import RMSNorm


class PlannerModule(nn.Module):
    """
    High-level planning module (f_H) for hierarchical reasoning.
    
    The Planner operates on a slow timescale, updating once per outer cycle
    after the Worker module has converged. It integrates the converged Worker
    state with its own previous state to produce updated planning context.
    
    Architecture:
        1. Concatenate h_H_prev and h_L_final: (batch, 2*hidden_dim)
        2. MLP transformation with residual connection
        3. Post-normalisation with RMSNorm
    
    The Planner's update "resets" the Worker's computational space by
    providing new context, enabling sustained activity across 200+ steps.
    
    Args:
        hidden_dim: Dimension of hidden states (both H and L modules).
        mlp_ratio: Expansion ratio for MLP intermediate dimension. Default: 2
        dropout: Dropout probability. Default: 0.1
    
    Shape:
        - h_H_prev: (batch, hidden_dim) - Previous Planner state
        - h_L_final: (batch, hidden_dim) - Converged Worker state
        - Output: (batch, hidden_dim) - Updated Planner state
    
    Example:
        >>> planner = PlannerModule(hidden_dim=64)
        >>> h_H_prev = torch.randn(8, 64)   # Previous planner state
        >>> h_L_final = torch.randn(8, 64)  # Converged worker state
        >>> h_H_new = planner(h_H_prev, h_L_final)
        >>> h_H_new.shape
        torch.Size([8, 64])
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialise PlannerModule.
        
        Args:
            hidden_dim: Dimension of hidden state vectors.
            mlp_ratio: Expansion ratio for intermediate MLP dimension.
            dropout: Dropout probability for regularisation.
        """
        super().__init__()
        
        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        
        # Input projection: concatenated states -> hidden_dim
        # Input is [h_H_prev; h_L_final] with dim 2*hidden_dim
        self.input_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # MLP block for state transformation
        intermediate_dim = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Post-normalisation (applied after residual)
        self.norm = RMSNorm(dim=hidden_dim)
        
        # Initialise weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialise weights using Xavier initialisation."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        h_H_prev: torch.Tensor,
        h_L_final: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update Planner state using previous state and converged Worker state.
        
        This update occurs once per outer cycle, after the Worker has converged.
        The new h_H will "reset" the Worker's target equilibrium for the next
        cycle, enabling hierarchical convergence.
        
        Args:
            h_H_prev: Previous Planner hidden state of shape (batch, hidden_dim).
            h_L_final: Converged Worker hidden state of shape (batch, hidden_dim).
        
        Returns:
            Updated Planner hidden state of shape (batch, hidden_dim).
        
        Note:
            The hierarchical reset mechanism:
            - Old h_H defined Worker's equilibrium target
            - Worker converged to h_L_final under that target
            - New h_H incorporates h_L_final information
            - Next cycle: Worker converges to NEW equilibrium under new h_H
        """
        # Concatenate previous H-state and converged L-state
        # Shape: (batch, 2*hidden_dim)
        combined = torch.cat([h_H_prev, h_L_final], dim=-1)
        
        # Project to hidden dimension
        # Shape: (batch, hidden_dim)
        h = self.input_proj(combined)
        
        # MLP transformation with residual connection
        # Residual from h_H_prev maintains continuity across cycles
        h = h + self.mlp(h)
        
        # Post-normalisation for training stability
        h_H_new = self.norm(h)
        
        return h_H_new
    
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return f"hidden_dim={self.hidden_dim}, mlp_ratio={self.mlp_ratio}"


def create_planner_module(hidden_dim: int, **kwargs) -> PlannerModule:
    """
    Factory function to create PlannerModule.
    
    Args:
        hidden_dim: Dimension of hidden states.
        **kwargs: Additional arguments (mlp_ratio, dropout).
    
    Returns:
        Configured PlannerModule instance.
    """
    return PlannerModule(hidden_dim=hidden_dim, **kwargs)
