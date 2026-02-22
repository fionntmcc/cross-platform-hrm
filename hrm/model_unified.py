"""
Unified HRM Model for Multiple Puzzle Types

This module provides a single transformer-based HRM that can solve:
- 4x4 Sudoku (seq_len=16, vocab=5)
- 9x9 Sudoku (seq_len=81, vocab=10)
- Mazes of various sizes (e.g., 15x15, 30x30)

The model uses puzzle-type embeddings for multi-task learning and
dynamically handles variable sequence lengths.

Usage:
    >>> from hrm.model_unified import UnifiedHRM, PuzzleType
    >>> model = UnifiedHRM()
    >>> 
    >>> # 9x9 Sudoku
    >>> sudoku_9x9 = torch.randint(0, 10, (8, 81))
    >>> output = model(sudoku_9x9, puzzle_type=PuzzleType.SUDOKU_9X9)
    >>> 
    >>> # 4x4 Sudoku  
    >>> sudoku_4x4 = torch.randint(0, 5, (8, 16))
    >>> output = model(sudoku_4x4, puzzle_type=PuzzleType.SUDOKU_4X4)
    >>> 
    >>> # 15x15 Maze
    >>> maze = torch.randint(0, 4, (8, 225))
    >>> output = model(maze, puzzle_type=PuzzleType.MAZE)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.norm import rms_norm
from hrm.layers.transformer import (
    RotaryEmbedding,
    TransformerBlock,
    ReasoningModule,
    trunc_normal_init_,
)


class PuzzleType(Enum):
    """Supported puzzle types."""
    SUDOKU_4X4 = auto()   # 4x4 grid, vocab=5 (0=empty, 1-4)
    SUDOKU_9X9 = auto()   # 9x9 grid, vocab=10 (0=empty, 1-9)
    MAZE = auto()         # Variable size, vocab=4 (wall, empty, start, goal)


@dataclass
class UnifiedHRMConfig:
    """
    Configuration for Unified HRM.
    
    Architecture follows Sapient's HRM with two levels of iteration:
    - Inner: H_cycles × L_cycles Worker/Planner steps (with 1-step gradient)
    - Outer: Up to halt_max_steps calls to the inner loop (ACT halting)
    
    Attributes:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_layers_L: Transformer blocks in L-level Worker.
        num_layers_H: Transformer blocks in H-level Planner.
        expansion: MLP expansion factor.
        max_seq_len: Maximum sequence length (for largest puzzle).
        H_cycles: H-level cycles per inner call.
        L_cycles: L-level steps per H-cycle.
        halt_max_steps: Maximum outer ACT steps.
        halt_exploration_prob: Exploration probability for Q-learning during training.
        vocab_size: Unified vocabulary size (max across puzzles).
        num_puzzle_types: Number of puzzle types for embedding.
        rms_norm_eps: Epsilon for RMS normalisation.
        dropout: Dropout rate.
        dtype: Computation dtype.
    """
    hidden_size: int = 256
    num_heads: int = 4
    num_layers_L: int = 2
    num_layers_H: int = 2
    expansion: float = 4.0
    max_seq_len: int = 1024  # Supports up to 32x32 grids
    H_cycles: int = 1
    L_cycles: int = 8
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    vocab_size: int = 10  # Max vocab (Sudoku 9x9)
    num_puzzle_types: int = 3
    rms_norm_eps: float = 1e-5
    dropout: float = 0.1        # Regularisation to prevent overfitting
    dtype: torch.dtype = torch.float32  # Use float32 for CPU; bfloat16/float16 for GPU


# Puzzle-specific configurations
# Inner compute per outer step = H_cycles × L_cycles Worker + H_cycles Planner
# Total compute = halt_max_steps × inner compute
PUZZLE_CONFIGS = {
    PuzzleType.SUDOKU_4X4: {
        'vocab_size': 5,      # 0=empty, 1-4=digits
        'seq_len': 16,        # 4x4 grid
        'grid_size': 4,
        'H_cycles': 1,        # H cycles per inner call
        'L_cycles': 4,        # L steps per H cycle
        'halt_max_steps': 4,  # Outer ACT steps (4×4 + 4×1 = 20 total ops)
    },
    PuzzleType.SUDOKU_9X9: {
        'vocab_size': 10,     # 0=empty, 1-9=digits
        'seq_len': 81,        # 9x9 grid
        'grid_size': 9,
        'H_cycles': 1,        # H cycles per inner call
        'L_cycles': 8,        # L steps per H cycle
        'halt_max_steps': 8,  # Outer ACT steps (8×8 + 8×1 = 72 total ops)
    },
    PuzzleType.MAZE: {
        'vocab_size': 4,      # 0=wall, 1=empty, 2=start, 3=goal
        'seq_len': None,      # Variable (e.g., 225 for 15x15, 900 for 30x30)
        'grid_size': None,    # Variable
        'H_cycles': 1,
        'L_cycles': 8,
        'halt_max_steps': 8,
    },
}


class UnifiedInputNetwork(nn.Module):
    """
    Input network supporting multiple puzzle types.
    
    Uses a shared token embedding with puzzle-type embeddings to
    differentiate between Sudoku and maze tasks.
    """
    
    def __init__(self, config: UnifiedHRMConfig):
        super().__init__()
        self.config = config
        
        # Shared token embedding (unified vocab)
        self.tok_emb = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
        )
        
        # Puzzle type embedding for multi-task learning
        self.puzzle_emb = nn.Embedding(
            config.num_puzzle_types,
            config.hidden_size,
        )
        
        # Input projection
        self.input_proj = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=False,
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialise with std = 1/sqrt(hidden_size)
        embed_init_std = 1.0 / math.sqrt(config.hidden_size)
        trunc_normal_init_(self.tok_emb.weight, std=embed_init_std)
        trunc_normal_init_(self.puzzle_emb.weight, std=embed_init_std)
        trunc_normal_init_(self.input_proj.weight, std=embed_init_std)
    
    def forward(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
    ) -> torch.Tensor:
        """
        Embed input tokens with puzzle-type context.
        
        Args:
            x: Input tokens (batch, seq_len).
            puzzle_type: Type of puzzle being solved.
        
        Returns:
            Hidden states (batch, seq_len, hidden_size).
        """
        # Token embedding
        h = self.tok_emb(x)
        
        # Add puzzle type embedding (broadcast to all positions)
        puzzle_idx = torch.tensor(
            [puzzle_type.value - 1], 
            device=x.device, 
            dtype=torch.long
        )
        puzzle_embed = self.puzzle_emb(puzzle_idx)  # (1, hidden_size)
        h = h + puzzle_embed.unsqueeze(0)  # Broadcast to (batch, seq_len, hidden_size)
        
        # Scale by sqrt(hidden_size)
        h = h * math.sqrt(self.config.hidden_size)
        
        return h


class UnifiedOutputNetwork(nn.Module):
    """
    Output network with puzzle-specific heads.
    
    Uses a shared backbone with separate output projections for
    different vocabulary sizes.
    """
    
    def __init__(self, config: UnifiedHRMConfig):
        super().__init__()
        self.config = config
        
        # Separate LM heads for each puzzle type's vocab
        self.lm_heads = nn.ModuleDict({
            'sudoku_4x4': nn.Linear(config.hidden_size, 5, bias=False),
            'sudoku_9x9': nn.Linear(config.hidden_size, 10, bias=False),
            'maze': nn.Linear(config.hidden_size, 4, bias=False),
        })
        
        # Initialise with std = 1/sqrt(fan_in)
        lm_head_std = 1.0 / math.sqrt(config.hidden_size)
        for head in self.lm_heads.values():
            trunc_normal_init_(head.weight, std=lm_head_std)
    
    def forward(
        self,
        h: torch.Tensor,
        puzzle_type: PuzzleType,
    ) -> torch.Tensor:
        """
        Project to vocabulary logits.
        
        Args:
            h: Hidden states (batch, seq_len, hidden_size).
            puzzle_type: Type of puzzle for selecting output head.
        
        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        # Select appropriate head
        if puzzle_type == PuzzleType.SUDOKU_4X4:
            return self.lm_heads['sudoku_4x4'](h)
        elif puzzle_type == PuzzleType.SUDOKU_9X9:
            return self.lm_heads['sudoku_9x9'](h)
        else:
            return self.lm_heads['maze'](h)


class UnifiedHRM(nn.Module):
    """
    Unified Hierarchical Reasoning Model for multiple puzzle types.
    
    This model handles:
    - 4x4 Sudoku (16 cells, vocab 0-4)
    - 9x9 Sudoku (81 cells, vocab 0-9)
    - Mazes (variable size, vocab 0-3)
    
    Architecture (matches Sapient's HRM with ACT halting):
        Outer loop (up to halt_max_steps):
            Inner loop (1-step gradient):
                for H in H_cycles:
                    for L in L_cycles:
                        z_L = Worker(z_L, z_H + input)
                    z_H = Planner(z_H, z_L)
            logits = OutputHead(z_H)
            q_halt, q_continue = QHead(z_H)  # Halting decision
            carry (z_H, z_L) persists to next outer step
    
    Components:
        1. UnifiedInputNetwork: Token + puzzle-type embedding
        2. WorkerTransformer: L-level iterative refinement
        3. PlannerTransformer: H-level planning updates
        4. UnifiedOutputNetwork: Puzzle-specific output heads
        5. Q-Head: Halting decision via Q-learning
    
    Args:
        config: UnifiedHRMConfig with model hyperparameters.
    
    Example:
        >>> model = UnifiedHRM()
        >>> 
        >>> # Solve 9x9 Sudoku
        >>> puzzle = torch.randint(0, 10, (4, 81))
        >>> output = model(puzzle, puzzle_type=PuzzleType.SUDOKU_9X9)
        >>> predictions = output['predictions']  # (4, 81)
        >>> 
        >>> # Solve 4x4 Sudoku
        >>> puzzle = torch.randint(0, 5, (4, 16))
        >>> output = model(puzzle, puzzle_type=PuzzleType.SUDOKU_4X4)
    """
    
    def __init__(self, config: Optional[UnifiedHRMConfig] = None):
        super().__init__()
        
        self.config = config or UnifiedHRMConfig()
        cfg = self.config
        
        # =====================================================================
        # Input Network
        # =====================================================================
        self.input_net = UnifiedInputNetwork(cfg)
        
        # =====================================================================
        # RoPE for positional encoding
        # =====================================================================
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=cfg.max_seq_len,
            base=10000.0,
        )
        
        # =====================================================================
        # L-Level Worker (Transformer)
        # =====================================================================
        self.worker = ReasoningModule(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers_L,
            expansion=cfg.expansion,
            rms_norm_eps=cfg.rms_norm_eps,
            causal=False,
        )
        
        # =====================================================================
        # H-Level Planner (Transformer)
        # =====================================================================
        self.planner = ReasoningModule(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers_H,
            expansion=cfg.expansion,
            rms_norm_eps=cfg.rms_norm_eps,
            causal=False,
        )
        
        # =====================================================================
        # Output Network
        # =====================================================================
        self.output_net = UnifiedOutputNetwork(cfg)
        
        # =====================================================================
        # Q-Head for ACT Halting (predicts halt vs continue)
        # =====================================================================
        self.q_head = nn.Linear(cfg.hidden_size, 2, bias=True)
        # Init Q to nearly zero for faster bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        
        # =====================================================================
        # Initial States (Non-trainable Buffers)
        # =====================================================================
        self.register_buffer(
            'z_L_init',
            torch.empty(1, 1, cfg.hidden_size)
        )
        self.register_buffer(
            'z_H_init',
            torch.empty(1, 1, cfg.hidden_size)
        )
        
        # Initialise buffers
        trunc_normal_init_(self.z_L_init, std=1.0)
        trunc_normal_init_(self.z_H_init, std=1.0)
    
    def _inner_forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        z_input: torch.Tensor,
        cos_sin,
        puzzle_type: PuzzleType,
        H_cycles: int,
        L_cycles: int,
    ):
        """
        Single inner H×L block with 1-step gradient.
        
        All iterations except the last run in no_grad. The final
        L-step + H-step runs with gradients to allow backpropagation.
        
        Args:
            z_H: H-level state (batch, seq_len, hidden_size).
            z_L: L-level state (batch, seq_len, hidden_size).
            z_input: Input embeddings (batch, seq_len, hidden_size).
            cos_sin: RoPE embeddings.
            puzzle_type: Puzzle type for output head.
            H_cycles: Number of H cycles.
            L_cycles: Number of L steps per H cycle.
        
        Returns:
            Tuple of (z_H_new, z_L_new, logits, q_halt_logits, q_continue_logits).
            z_H_new and z_L_new are detached (no gradient flow between outer steps).
        """
        with torch.no_grad():
            for h_step in range(H_cycles):
                for l_step in range(L_cycles):
                    is_last = (h_step == H_cycles - 1) and (l_step == L_cycles - 1)
                    if not is_last:
                        z_L = self.worker(z_L, z_H + z_input, cos_sin=cos_sin)
                
                if h_step < H_cycles - 1:
                    z_H = self.planner(z_H, z_L, cos_sin=cos_sin)
        
        # 1-step gradient: only the final L+H step is differentiable
        z_L = self.worker(z_L, z_H + z_input, cos_sin=cos_sin)
        z_H = self.planner(z_H, z_L, cos_sin=cos_sin)
        
        # LM output
        logits = self.output_net(z_H, puzzle_type)
        
        # Q-head for halting (uses mean-pooled hidden state)
        q_logits = self.q_head(z_H.mean(dim=1)).to(torch.float32)  # (batch, 2)
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        # Detach carry for next outer step (no gradient flow between outer steps)
        return z_H.detach(), z_L.detach(), logits, q_halt, q_continue
    
    def forward(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
        targets: Optional[torch.Tensor] = None,
        halt_max_steps: Optional[int] = None,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with outer ACT loop.
        
        The model runs up to halt_max_steps outer iterations. Each outer
        step calls the inner H×L block, produces logits and Q-values,
        and persists the carry (z_H, z_L) to the next step.
        
        During training, Q-learning losses are accumulated at each step.
        During eval, always runs all halt_max_steps (matching Sapient).
        
        Args:
            x: Input tokens (batch, seq_len) with values in [0, vocab_size).
            puzzle_type: Type of puzzle (SUDOKU_4X4, SUDOKU_9X9, MAZE).
            targets: Optional target tokens for loss computation.
            halt_max_steps: Override max outer ACT steps.
            H_cycles: Override H cycles per inner call.
            L_cycles: Override L cycles per H cycle.
            return_intermediates: Include per-step predictions in output.
        
        Returns:
            Dictionary containing:
                - 'logits': (batch, seq_len, vocab_size) from final step
                - 'predictions': (batch, seq_len) argmax predictions
                - 'loss': Total loss (LM + Q) if targets provided
                - 'lm_loss': Language model cross-entropy loss
                - 'q_halt_loss': Q-learning halting loss
                - 'outer_steps_used': Number of outer ACT steps run
                - 'intermediates': Per-step predictions (if requested)
        """
        batch_size, seq_len = x.shape
        device = x.device
        dtype = self.config.dtype
        
        # Get puzzle-specific defaults
        puzzle_cfg = PUZZLE_CONFIGS[puzzle_type]
        H_cycles = H_cycles or puzzle_cfg.get('H_cycles', self.config.H_cycles)
        L_cycles = L_cycles or puzzle_cfg.get('L_cycles', self.config.L_cycles)
        halt_max = halt_max_steps or puzzle_cfg.get('halt_max_steps', self.config.halt_max_steps)
        
        # Step 1: Embed input
        z_input = self.input_net(x, puzzle_type)
        
        # Step 2: Initialise carry states
        z_L = self.z_L_init.expand(batch_size, seq_len, -1).to(dtype)
        z_H = self.z_H_init.expand(batch_size, seq_len, -1).to(dtype)
        
        # Get RoPE embeddings
        cos_sin = self.rotary_emb()
        
        # Tracking
        step_predictions: List[torch.Tensor] = []
        all_q_halt: List[torch.Tensor] = []
        all_q_continue: List[torch.Tensor] = []
        all_logits_for_loss: List[torch.Tensor] = []
        
        # Step 3: Outer ACT loop with prediction feedback.
        # After each step, the model's predictions are merged back with
        # the original givens and re-embedded as input for the next step.
        # This enables iterative refinement / backtracking: the model can
        # see its own answers, detect constraint violations, and correct them.
        logits = None
        given_mask = (x != 0)  # True for given clues (don't overwrite)
        
        for outer_step in range(halt_max):
            z_H, z_L, logits, q_halt, q_continue = self._inner_forward(
                z_H, z_L, z_input, cos_sin, puzzle_type, H_cycles, L_cycles,
            )
            
            all_q_halt.append(q_halt)
            all_q_continue.append(q_continue)
            all_logits_for_loss.append(logits)
            
            if return_intermediates:
                step_predictions.append(logits.argmax(dim=-1).detach())
            
            # Prediction feedback: merge current predictions with original
            # givens and re-embed for the next outer step.
            # Given clues are preserved; empty cells get current predictions.
            if outer_step < halt_max - 1:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)  # (batch, seq_len)
                    refined_input = torch.where(given_mask, x, preds)
                z_input = self.input_net(refined_input, puzzle_type)
        
        # Step 4: Final predictions from last outer step
        assert logits is not None
        predictions = logits.argmax(dim=-1)
        
        # Build output
        output: Dict[str, Any] = {
            'logits': logits,
            'predictions': predictions,
            'all_step_logits': all_logits_for_loss,
            'z_L_final': z_L,
            'z_H_final': z_H,
            'outer_steps_used': halt_max,
            'h_cycles_per_step': H_cycles,
            'l_cycles_per_step': L_cycles,
        }
        
        # Compute losses if targets provided
        if targets is not None:
            targets_flat = targets.view(-1)
            
            # LM loss: from the final outer step
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_flat)
            
            # Q-halt loss: teach Q to predict correctness at each step
            # Target: 1 if the step's prediction matches all targets, 0 otherwise
            q_halt_loss = torch.tensor(0.0, device=device)
            for step_idx in range(halt_max):
                step_logits = all_logits_for_loss[step_idx]
                with torch.no_grad():
                    step_preds = step_logits.argmax(dim=-1)
                    is_correct = (step_preds == targets).all(dim=-1).float()
                q_halt_loss = q_halt_loss + F.binary_cross_entropy_with_logits(
                    all_q_halt[step_idx], is_correct,
                )
            q_halt_loss = q_halt_loss / halt_max
            
            # Total loss = LM + 0.5 * Q
            output['loss'] = lm_loss + 0.5 * q_halt_loss
            output['lm_loss'] = lm_loss
            output['q_halt_loss'] = q_halt_loss
        
        if return_intermediates:
            output['intermediates'] = {
                'step_predictions': step_predictions,
                'q_halt_logits': [q.detach() for q in all_q_halt],
                'q_continue_logits': [q.detach() for q in all_q_continue],
            }
        
        return output
    
    def predict(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get predictions for input puzzle.
        
        Args:
            x: Input tokens (batch, seq_len).
            puzzle_type: Type of puzzle.
            **kwargs: Additional forward arguments.
        
        Returns:
            Predictions (batch, seq_len).
        """
        with torch.no_grad():
            output = self.forward(x, puzzle_type, **kwargs)
        return output['predictions']
    
    def solve_sudoku_4x4(
        self,
        puzzle: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Solve a 4x4 Sudoku puzzle.
        
        Args:
            puzzle: Input grid (batch, 4, 4) or flattened (batch, 16).
        
        Returns:
            Solution grid (batch, 4, 4).
        """
        # Flatten if needed
        if puzzle.dim() == 3:
            batch_size = puzzle.shape[0]
            puzzle = puzzle.view(batch_size, -1)
        else:
            batch_size = puzzle.shape[0]
        
        predictions = self.predict(puzzle, PuzzleType.SUDOKU_4X4, **kwargs)
        return predictions.view(batch_size, 4, 4)
    
    def solve_sudoku_9x9(
        self,
        puzzle: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Solve a 9x9 Sudoku puzzle.
        
        Args:
            puzzle: Input grid (batch, 9, 9) or flattened (batch, 81).
        
        Returns:
            Solution grid (batch, 9, 9).
        """
        if puzzle.dim() == 3:
            batch_size = puzzle.shape[0]
            puzzle = puzzle.view(batch_size, -1)
        else:
            batch_size = puzzle.shape[0]
        
        predictions = self.predict(puzzle, PuzzleType.SUDOKU_9X9, **kwargs)
        return predictions.view(batch_size, 9, 9)
    
    def solve_maze(
        self,
        maze: torch.Tensor,
        grid_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Solve a maze puzzle.
        
        Args:
            maze: Input maze (batch, H, W) or flattened (batch, seq_len).
            grid_size: Size for reshaping output (inferred if 2D input).
        
        Returns:
            Solution path markers (batch, H, W) or (batch, seq_len).
        """
        if maze.dim() == 3:
            batch_size, h, w = maze.shape
            maze_flat = maze.view(batch_size, -1)
            predictions = self.predict(maze_flat, PuzzleType.MAZE, **kwargs)
            return predictions.view(batch_size, h, w)
        else:
            return self.predict(maze, PuzzleType.MAZE, **kwargs)
    
    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.config.hidden_size}, "
            f"num_heads={self.config.num_heads}, "
            f"layers_L={self.config.num_layers_L}, "
            f"layers_H={self.config.num_layers_H}"
        )


def create_unified_hrm(
    hidden_size: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    **kwargs,
) -> UnifiedHRM:
    """
    Factory function for creating UnifiedHRM.
    
    Args:
        hidden_size: Model hidden dimension.
        num_heads: Attention heads.
        num_layers: Layers for both L and H modules.
        **kwargs: Additional config overrides.
    
    Returns:
        Configured UnifiedHRM model.
    """
    config = UnifiedHRMConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers_L=num_layers,
        num_layers_H=num_layers,
        **kwargs,
    )
    return UnifiedHRM(config)