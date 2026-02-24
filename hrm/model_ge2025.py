"""
L-Module Only HRM — Aligned with Ge et al. (2025)

"Hierarchical Reasoning Models: Perspectives and Misconceptions"
(arXiv:2510.00355v2, Ge, Liao & Poggio, MIT CBMM)

Key findings from the paper that this model implements:

1. The H-module (Planner) provides minimal benefit. An 8-layer L-module
   only HRM performs comparably to the original 4L+4H HRM while training
   2.4x faster (1h 48m vs 4h 21m on A100). Confirmed by ARC Foundation.

2. HRM's one-step gradient training is equivalent to Latent Consistency
   Model (LCM) training from the diffusion model literature. The model
   learns to map intermediate reasoning states directly to solutions.

3. ACT does not improve inference — running max steps is optimal.
   This suggests the model functions as a very deep feedforward network
   rather than a genuinely adaptive recurrent architecture.

4. Well-trained models can solve Sudoku in as few as 2-4 reasoning steps,
   exhibiting diffusion-like behaviour rather than step-by-step constraint
   propagation.

Architecture (this implementation):
    Input → [N reasoning steps: 8-layer Transformer with weight sharing] → Output

    For each reasoning step m (one-step gradient):
        z_L^{m-1} = detach(z_L^{m-1})             # break gradient chain
        z_L^m = f_L(z_L^{m-1} + x_input; θ_L)     # WITH gradients
        logits^m = f_O(z_L^m; θ_O)                  # WITH gradients

    Every step has full gradient flow through one application of f_L,
    but no BPTT across steps (detach between steps). This matches
    the official HRM's _inner_forward pattern and is equivalent to
    Latent Consistency Model / diffusion-style training.

    Prediction feedback: after each step, the model's predictions are
    merged with original givens and re-embedded, enabling iterative
    self-correction (like denoising in diffusion models).

One-Step Gradient — How It Works:
    The key insight is that "one-step gradient" does NOT mean "only the
    last step has gradients". It means: at each step, gradients flow
    through ONE application of f_L (not through the entire recurrence).
    
    In the official HRM (model_unified._inner_forward):
        - Each outer step runs an inner loop of H×L iterations
        - All but the LAST inner iteration run in no_grad
        - The LAST inner iteration runs WITH gradients
        - Logits from that outer step have gradient through one f_L call
        - States are detached before the next outer step
    
    For L-module only with L_cycles=1 (this model):
        - The inner loop has only 1 step, so it always has gradients
        - Each reasoning step = one f_L call WITH gradients
        - Detach z_L between steps (no BPTT, constant memory)
        - Loss at final step trains the full model through the chain:
          loss → logits → output_head → z_L → reasoning_module

Reference:
    Ge, R., Liao, Q., & Poggio, T. (2025). Hierarchical Reasoning Models:
    Perspectives and Misconceptions. arXiv:2510.00355v2.

Usage:
    >>> from hrm.model_ge2025 import LModuleOnlyHRM, LModuleOnlyConfig
    >>> model = LModuleOnlyHRM()
    >>> puzzle = torch.randint(0, 10, (8, 81))  # batch of 8, 9x9 Sudoku
    >>> output = model(puzzle, PuzzleType.SUDOKU_9X9, targets=solution)
    >>> loss = output['loss']
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.norm import rms_norm
from hrm.layers.transformer import (
    RotaryEmbedding,
    ReasoningModule,
    CastedLinear,
    trunc_normal_init_,
)


# ---------------------------------------------------------------------------
# Puzzle type enum (shared with model_unified for compatibility)
# ---------------------------------------------------------------------------
class PuzzleType(Enum):
    """Supported puzzle types."""
    SUDOKU_4X4 = auto()   # 4x4 grid, vocab=5 (0=empty, 1-4)
    SUDOKU_9X9 = auto()   # 9x9 grid, vocab=10 (0=empty, 1-9)
    MAZE = auto()         # Variable size, vocab=4


# Per-puzzle defaults
PUZZLE_DEFAULTS = {
    PuzzleType.SUDOKU_4X4: {'vocab_size': 5,  'seq_len': 16, 'grid_size': 4},
    PuzzleType.SUDOKU_9X9: {'vocab_size': 10, 'seq_len': 81, 'grid_size': 9},
    PuzzleType.MAZE:       {'vocab_size': 4,  'seq_len': None, 'grid_size': None},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LModuleOnlyConfig:
    """
    Configuration for the L-module only HRM (Ge et al. 2025 variant).

    The key difference from the full HRM is the absence of the H-module.
    All computation budget goes into the L-module transformer, which is
    expanded to 8 layers (matching the combined 4L+4H of the original).

    Attributes:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads in the transformer.
        num_layers: Number of transformer blocks in the L-module.
            Ge et al. used 8 to match the original 4L+4H HRM capacity.
        expansion: SwiGLU MLP expansion factor.
        max_seq_len: Maximum sequence length (for RoPE cache).
        num_reasoning_steps: Number of reasoning iterations (outer steps).
            Paper shows max steps gives best results; 16 is their default.
        vocab_size: Unified vocabulary size (max across all puzzle types).
        num_puzzle_types: Number of puzzle types for embedding.
        rms_norm_eps: Epsilon for RMS normalisation.
        dropout: Dropout rate for regularisation.
        dtype: Computation dtype.
        use_prediction_feedback: Whether to re-embed predictions between
            reasoning steps (self-conditioning / diffusion-style refinement).
    """
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 8          # 8 layers (paper: expanded L-module)
    expansion: float = 4.0
    max_seq_len: int = 1024
    num_reasoning_steps: int = 16  # Paper default; max steps is optimal
    vocab_size: int = 10           # Max vocab (Sudoku 9x9)
    num_puzzle_types: int = 3
    rms_norm_eps: float = 1e-5
    dropout: float = 0.1
    dtype: torch.dtype = torch.float32
    use_prediction_feedback: bool = True  # Self-conditioning between steps


# ---------------------------------------------------------------------------
# Input Network
# ---------------------------------------------------------------------------
class InputEmbedding(nn.Module):
    """
    Token + puzzle-type embedding network.

    Converts token sequences to hidden representations. Puzzle-type
    embedding enables multi-task learning across Sudoku sizes and mazes.
    """

    def __init__(self, config: LModuleOnlyConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        # Puzzle type embedding for multi-task
        self.puzzle_emb = nn.Embedding(
            config.num_puzzle_types, config.hidden_size
        )

        # Input projection (linear, no bias — matches Sapient)
        self.input_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

        self.dropout = nn.Dropout(config.dropout)

        # Initialise
        embed_std = 1.0 / math.sqrt(config.hidden_size)
        trunc_normal_init_(self.tok_emb.weight, std=embed_std)
        trunc_normal_init_(self.puzzle_emb.weight, std=embed_std)
        trunc_normal_init_(self.input_proj.weight, std=embed_std)

    def forward(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
    ) -> torch.Tensor:
        """
        Embed tokens with puzzle-type context.

        Args:
            x: Token indices (batch, seq_len).
            puzzle_type: Type of puzzle being solved.

        Returns:
            Hidden states (batch, seq_len, hidden_size).
        """
        h = self.tok_emb(x)

        # Add puzzle type embedding (broadcast to all positions)
        puzzle_idx = torch.tensor(
            [puzzle_type.value - 1], device=x.device, dtype=torch.long
        )
        h = h + self.puzzle_emb(puzzle_idx).unsqueeze(0)

        # Scale by sqrt(d) for stable training
        h = h * math.sqrt(self.config.hidden_size)

        h = self.input_proj(h)
        h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# Output Network
# ---------------------------------------------------------------------------
class OutputHead(nn.Module):
    """
    Puzzle-specific LM heads for converting hidden states to logits.

    Separate linear projections per puzzle type handle different vocab
    sizes without wasting capacity on unused output dimensions.
    """

    def __init__(self, config: LModuleOnlyConfig):
        super().__init__()
        self.config = config

        # One head per puzzle type
        self.heads = nn.ModuleDict({
            'sudoku_4x4': nn.Linear(config.hidden_size, 5, bias=False),
            'sudoku_9x9': nn.Linear(config.hidden_size, 10, bias=False),
            'maze': nn.Linear(config.hidden_size, 4, bias=False),
        })

        # Initialise
        head_std = 1.0 / math.sqrt(config.hidden_size)
        for head in self.heads.values():
            trunc_normal_init_(head.weight, std=head_std)

    def forward(
        self,
        h: torch.Tensor,
        puzzle_type: PuzzleType,
    ) -> torch.Tensor:
        """
        Project to vocabulary logits.

        Args:
            h: Hidden states (batch, seq_len, hidden_size).
            puzzle_type: Puzzle type for head selection.

        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        if puzzle_type == PuzzleType.SUDOKU_4X4:
            return self.heads['sudoku_4x4'](h)
        elif puzzle_type == PuzzleType.SUDOKU_9X9:
            return self.heads['sudoku_9x9'](h)
        else:
            return self.heads['maze'](h)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class LModuleOnlyHRM(nn.Module):
    """
    L-Module Only HRM (Ge et al. 2025 variant).

    Eliminates the H-module entirely, using an expanded 8-layer
    transformer as the sole reasoning engine. This matches the paper's
    finding that an 8-layer L-module only model performs similarly to
    the original 4L+4H HRM while training 2.4x faster.

    The model performs N reasoning steps where the same transformer
    is applied repeatedly with weight sharing (recurrence). Each step:

        z_L = detach(z_L)                       # one-step: break chain
        z_L = Transformer(z_L + x_input)        # with gradients
        logits = OutputHead(z_L)                 # with gradients

    One-step gradient means: every step has gradient flow through ONE
    application of the reasoning module, but gradients do not flow
    across steps (no BPTT). This achieves O(1) memory regardless of
    reasoning depth and is equivalent to LCM / diffusion training.

    Loss is computed from the FINAL step's logits only, matching the
    official HRM's training procedure (model_unified.py line 535).

    Args:
        config: LModuleOnlyConfig with model hyperparameters.

    Example:
        >>> model = LModuleOnlyHRM()
        >>>
        >>> # 9x9 Sudoku
        >>> puzzle = torch.randint(0, 10, (4, 81))
        >>> target = torch.randint(1, 10, (4, 81))
        >>> output = model(puzzle, PuzzleType.SUDOKU_9X9, targets=target)
        >>> loss = output['loss']
        >>>
        >>> # 4x4 Sudoku
        >>> puzzle_4 = torch.randint(0, 5, (4, 16))
        >>> output_4 = model(puzzle_4, PuzzleType.SUDOKU_4X4)
        >>> predictions = output_4['predictions']
    """

    def __init__(self, config: Optional[LModuleOnlyConfig] = None):
        super().__init__()

        self.config = config or LModuleOnlyConfig()
        cfg = self.config

        # =================================================================
        # Input Embedding
        # =================================================================
        self.input_net = InputEmbedding(cfg)

        # =================================================================
        # RoPE for positional encoding
        # =================================================================
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=cfg.max_seq_len,
            base=10000.0,
        )

        # =================================================================
        # L-Module: 8-layer Transformer (the ONLY reasoning module)
        # =================================================================
        # This single module replaces both the Worker and Planner from
        # the original HRM. Weight sharing: the same 8 transformer layers
        # are applied at every reasoning step (recurrence).
        self.reasoning = ReasoningModule(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            expansion=cfg.expansion,
            rms_norm_eps=cfg.rms_norm_eps,
            causal=False,  # Non-causal for puzzle solving
        )

        # =================================================================
        # Output Head (puzzle-specific)
        # =================================================================
        self.output_head = OutputHead(cfg)

        # =================================================================
        # Initial latent state z_L^0 (non-trainable buffer)
        # =================================================================
        # Shape: (1, 1, hidden_size) — broadcast to (batch, seq_len, d)
        # Matches model_unified.py which also uses register_buffer for
        # z_L_init and z_H_init (lines 353-364).
        self.register_buffer(
            'z_L_init',
            torch.empty(1, 1, cfg.hidden_size),
        )
        trunc_normal_init_(self.z_L_init, std=1.0)

    def _reasoning_step(
        self,
        z_L: torch.Tensor,
        z_input: torch.Tensor,
        cos_sin,
        puzzle_type: PuzzleType,
    ):
        """
        Single reasoning step with one-step gradient.

        Mirrors model_unified._inner_forward but simplified for L-only:
        - No inner H*L loop (L_cycles=1, no H-module)
        - The single f_L call IS the one-step gradient
        - Returns detached carry + logits with gradient

        Args:
            z_L: Current latent state (batch, seq_len, hidden_size).
                 This should already be detached from previous step.
            z_input: Input embeddings (batch, seq_len, hidden_size).
            cos_sin: RoPE cache.
            puzzle_type: Puzzle type for output head.

        Returns:
            Tuple of:
                - z_L_carry: Detached latent state for next step
                - logits: (batch, seq_len, vocab_size) WITH gradient
                  through one application of reasoning + output_head
        """
        # One application of f_L WITH gradients
        # This is the "one-step" — gradients flow through this single
        # call but not through the recurrence (z_L was detached).
        z_L = self.reasoning(z_L, z_input, cos_sin=cos_sin)

        # Compute logits (has gradient through reasoning + output_head)
        logits = self.output_head(z_L, puzzle_type)

        # Detach carry for next step (no BPTT across steps)
        # Matches model_unified._inner_forward line 418:
        #   return z_H.detach(), z_L.detach(), logits, ...
        return z_L.detach(), logits

    def forward(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
        targets: Optional[torch.Tensor] = None,
        num_reasoning_steps: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with iterative reasoning.

        Runs N reasoning steps where the same 8-layer transformer is
        applied repeatedly with weight sharing. Uses one-step gradient:
        every step has gradient through one f_L call, but no BPTT
        across steps (states detached between steps).

        Prediction feedback (when enabled): after each step, the model's
        current predictions are merged with the original puzzle givens
        and re-embedded as input for the next step, enabling iterative
        self-correction (like denoising in diffusion models).

        Loss is computed from the FINAL step only, matching the official
        HRM (model_unified.py computes lm_loss from last outer step).

        Args:
            x: Input tokens (batch, seq_len) with values in [0, vocab_size).
            puzzle_type: Type of puzzle (SUDOKU_4X4, SUDOKU_9X9, MAZE).
            targets: Optional target tokens for loss computation.
            num_reasoning_steps: Override number of reasoning steps.
            return_intermediates: Whether to include per-step predictions.

        Returns:
            Dictionary containing:
                - 'logits': (batch, seq_len, vocab_size) final logits
                - 'predictions': (batch, seq_len) argmax predictions
                - 'loss': Cross-entropy loss (if targets provided)
                - 'lm_loss': LM cross-entropy from final step
                - 'all_step_logits': List of logits from each step
                - 'reasoning_steps_used': Number of steps executed
                - 'intermediates': Per-step predictions (if requested)
        """
        batch_size, seq_len = x.shape
        device = x.device
        n_steps = num_reasoning_steps or self.config.num_reasoning_steps
        use_feedback = self.config.use_prediction_feedback

        # Step 1: Embed input tokens
        z_input = self.input_net(x, puzzle_type)

        # Step 2: Initialise latent state
        z_L = self.z_L_init.expand(batch_size, seq_len, -1).clone()

        # Step 3: Pre-compute RoPE
        cos_sin = self.rotary_emb()

        # Step 4: Tracking
        all_step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        given_mask = (x != 0)  # True for given clues

        # =================================================================
        # Step 5: Iterative reasoning with one-step gradient
        # =================================================================
        # Each step mirrors one call to model_unified._inner_forward:
        #   1. z_L enters detached (no BPTT across steps)
        #   2. f_L runs WITH gradients (one-step gradient)
        #   3. logits have full gradient through reasoning + output_head
        #   4. z_L is detached for next step's carry
        #
        # This is equivalent to LCM training (Ge et al. Section 3.1):
        # the model learns to map each intermediate state to the solution
        # independently, without backpropagating through the chain.
        # =================================================================
        logits = None

        for step in range(n_steps):
            is_last_step = (step == n_steps - 1)

            if is_last_step:
                # Final step: WITH gradients (one-step gradient).
                # This is the only step that contributes to the loss.
                z_L, logits = self._reasoning_step(
                    z_L, z_input, cos_sin, puzzle_type
                )
            else:
                # Intermediate steps: no gradients needed.
                # Matches model_unified._inner_forward which runs all
                # but the last L-step inside torch.no_grad().
                # Without this, N reasoning steps retain N separate
                # autograd graphs (one per step), causing O(N*L) memory
                # instead of O(L).
                with torch.no_grad():
                    z_L, logits = self._reasoning_step(
                        z_L, z_input, cos_sin, puzzle_type
                    )

            all_step_logits.append(logits if is_last_step else logits.detach())

            if return_intermediates:
                step_predictions.append(logits.argmax(dim=-1).detach())

            # Prediction feedback (self-conditioning)
            # Re-embed the model's current predictions merged with
            # original givens for the next reasoning step.
            if use_feedback and not is_last_step:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    refined = torch.where(given_mask, x, preds)
                # Re-embed WITH gradients so the final step can
                # backpropagate through input_net.
                z_input = self.input_net(refined, puzzle_type)

        # Step 6: Final output
        assert logits is not None
        predictions = logits.argmax(dim=-1)

        output: Dict[str, Any] = {
            'logits': logits,
            'predictions': predictions,
            'all_step_logits': all_step_logits,
            'z_L_final': z_L,
            'reasoning_steps_used': n_steps,
        }

        # =================================================================
        # Step 7: Loss computation
        # =================================================================
        # Matches official HRM: lm_loss from FINAL step only.
        # (model_unified.py line 535)
        #
        # Note: the official HRM does NOT use deep supervision for LM
        # loss — it only uses the final outer step's logits for cross-
        # entropy. The intermediate logits are used for Q-learning
        # (halting), which we omit since Ge et al. showed ACT provides
        # no inference benefit.
        # =================================================================
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            output['loss'] = lm_loss
            output['lm_loss'] = lm_loss

        if return_intermediates:
            output['intermediates'] = {
                'step_predictions': step_predictions,
            }

        return output

    # -------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------
    def predict(
        self,
        x: torch.Tensor,
        puzzle_type: PuzzleType,
        **kwargs,
    ) -> torch.Tensor:
        """Get predictions (no gradient)."""
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
            puzzle: (batch, 4, 4) or (batch, 16) integer grid.

        Returns:
            Solution grid (batch, 4, 4).
        """
        if puzzle.dim() == 3:
            batch_size = puzzle.shape[0]
            puzzle = puzzle.view(batch_size, -1)
        else:
            batch_size = puzzle.shape[0]

        preds = self.predict(puzzle, PuzzleType.SUDOKU_4X4, **kwargs)
        return preds.view(batch_size, 4, 4)

    def solve_sudoku_9x9(
        self,
        puzzle: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Solve a 9x9 Sudoku puzzle.

        Args:
            puzzle: (batch, 9, 9) or (batch, 81) integer grid.

        Returns:
            Solution grid (batch, 9, 9).
        """
        if puzzle.dim() == 3:
            batch_size = puzzle.shape[0]
            puzzle = puzzle.view(batch_size, -1)
        else:
            batch_size = puzzle.shape[0]

        preds = self.predict(puzzle, PuzzleType.SUDOKU_9X9, **kwargs)
        return preds.view(batch_size, 9, 9)

    def solve_maze(
        self,
        maze: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Solve a maze puzzle."""
        if maze.dim() == 3:
            batch_size, h, w = maze.shape
            maze_flat = maze.view(batch_size, -1)
            preds = self.predict(maze_flat, PuzzleType.MAZE, **kwargs)
            return preds.view(batch_size, h, w)
        return self.predict(maze, PuzzleType.MAZE, **kwargs)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.config.hidden_size}, "
            f"num_heads={self.config.num_heads}, "
            f"num_layers={self.config.num_layers} (L-only), "
            f"reasoning_steps={self.config.num_reasoning_steps}, "
            f"feedback={self.config.use_prediction_feedback}"
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def create_lmodule_only_hrm(
    hidden_size: int = 256,
    num_heads: int = 4,
    num_layers: int = 8,
    num_reasoning_steps: int = 16,
    **kwargs,
) -> LModuleOnlyHRM:
    """
    Create an L-module only HRM (Ge et al. 2025 variant).

    Default configuration matches the paper's 8-layer L-module only
    setting with 16 reasoning steps.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Transformer layers in the L-module.
        num_reasoning_steps: Number of iterative reasoning steps.
        **kwargs: Additional config overrides.

    Returns:
        Configured LModuleOnlyHRM model.
    """
    config = LModuleOnlyConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        num_reasoning_steps=num_reasoning_steps,
        **kwargs,
    )
    return LModuleOnlyHRM(config)


def create_small_lmodule_hrm(**kwargs) -> LModuleOnlyHRM:
    """
    Create a smaller L-module only HRM for quick experiments / RPi5.

    Reduced dimensions for faster training and lower memory footprint,
    suitable for 4x4 Sudoku and Raspberry Pi deployment.

    Returns:
        Small LModuleOnlyHRM model (~1-2M parameters).
    """
    defaults = dict(
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        num_reasoning_steps=8,
        expansion=2.0,
        max_seq_len=256,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return create_lmodule_only_hrm(**defaults)
