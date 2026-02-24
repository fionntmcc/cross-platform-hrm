"""
HRM_4x4_Simple: L-Module Only Variant of the Hierarchical Reasoning Model

Based on Ge et al. (2025) "Hierarchical Reasoning Models: Perspectives and
Misconceptions" (arXiv:2510.00355v2), which found that:

1. The H-module (Planner) is not essential — an 8-layer L-module only HRM
   performs similarly to the original 4-layer L + 4-layer H HRM.
2. Training is 2.4x faster (1h 48m vs 4h 21m on A100 GPU).
3. Comparable accuracy on constraint-satisfaction tasks.
4. Confirmed independently by ARC Foundation.

Architecture:
    ORIGINAL (HRM_4x4):
        Input → [Outer Loop (K cycles): [Inner Loop (T steps): Worker] → Planner] → Output
        Total: K × T iterations with Planner updates

    SIMPLIFIED (HRM_4x4_Simple):
        Input → [Single Loop (N steps): Worker] → Output
        Total: N iterations (where N ≈ K × T for equivalence)

The key insight is that computational depth via iteration is the important
factor, not the specific hierarchical H/L architecture. The simplified model
uses x_in (input embedding) as the fixed context for the Worker, replacing
the role of h_H from the Planner.

Reference:
    - Ge et al. (2025), arXiv:2510.00355v2
    - hrm.model: Original hierarchical HRM_4x4
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.input_network import InputNetwork
from hrm.layers.worker import WorkerModule
from hrm.layers.output_network import OutputNetwork
from hrm.core.iteration import compute_residual


@dataclass
class SimpleExecutionTrace:
    """
    Trace of simplified HRM execution for analysis and debugging.

    Records iteration count, convergence status, and residual history
    from the flat iteration loop (no outer/inner distinction).

    Attributes:
        num_steps: Number of iteration steps taken.
        max_steps: Maximum iterations allowed.
        converged: Whether convergence threshold was reached.
        final_residual: Final residual ||h_L^(t) - h_L^(t-1)||.
        residual_history: List of residuals at each step.
        x_in: Input embedding used during execution (if tracked).
        final_h_L: Final low-level hidden state.
    """
    num_steps: int
    max_steps: int
    converged: bool
    final_residual: float
    residual_history: List[float] = field(default_factory=list)
    x_in: Optional[torch.Tensor] = None
    final_h_L: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return (
            f"SimpleExecutionTrace(steps={self.num_steps}/{self.max_steps}, "
            f"converged={self.converged}, "
            f"final_residual={self.final_residual:.6f})"
        )

    @property
    def total_computation_steps(self) -> int:
        """Total Worker iterations executed."""
        return self.num_steps

    @property
    def effective_depth(self) -> int:
        """Effective computational depth (same as num_steps for flat loop)."""
        return self.num_steps


class HRM_4x4_Simple(nn.Module):
    """
    Simplified HRM using only the L-module (Worker) without hierarchical structure.

    Based on Ge et al. (2025) findings that the H-module (Planner) provides
    minimal benefit. Trains 2.4x faster while maintaining comparable accuracy.

    The model replaces the hierarchical Planner + Worker loops with a single
    flat iteration loop. The Worker uses x_in (input embedding) as context
    instead of h_H from the Planner, and iterates N times (equivalent to
    K×T in the full HRM) with one-step gradient approximation.

    Args:
        hidden_dim: Dimension of hidden states. Default: 64.
        embed_dim: Embedding dimension per cell. Default: 16.
        n_iterations: Number of flat iterations (N ≈ K×T). Default: 50.
        convergence_threshold: Threshold for early stopping. Default: 1e-3.
        n_worker_layers: Number of stacked Worker MLP blocks. Default: 8.
        dropout: Dropout probability. Default: 0.1.
        track_history: Whether to record execution trace. Default: True.

    Attributes:
        input_network: Converts grid to hidden representation.
        worker_layers: List of Worker MLP blocks for expanded capacity.
        output_network: Decodes to action predictions.
        h_L_init: Learned initial low-level state.

    Shape:
        - Input: (batch, 4, 4) integer grid with values in [0, 4]
        - Output: Dict with:
            - 'cell_logits': (batch, 16) cell selection logits
            - 'digit_logits': (batch, 4) digit selection logits
            - 'trace': SimpleExecutionTrace with convergence info

    Example:
        >>> model = HRM_4x4_Simple()
        >>> puzzle = torch.randint(0, 5, (8, 4, 4))
        >>> outputs = model(puzzle)
        >>> cell_logits = outputs['cell_logits']
        >>> digit_logits = outputs['digit_logits']
        >>>
        >>> # Training loss
        >>> cell_loss = F.cross_entropy(cell_logits, target_cells)
        >>> digit_loss = F.cross_entropy(digit_logits, target_digits)
        >>> halt_penalty = model.get_halt_penalty(outputs)
        >>> total_loss = cell_loss + digit_loss + 0.01 * halt_penalty
    """

    # Class constants for 4x4 Sudoku
    VOCAB_SIZE = 5   # 0=empty, 1-4=digits
    GRID_SIZE = 4
    NUM_CELLS = 16
    NUM_DIGITS = 4

    def __init__(
        self,
        hidden_dim: int = 64,
        embed_dim: int = 16,
        n_iterations: int = 50,
        convergence_threshold: float = 1e-3,
        n_worker_layers: int = 8,
        dropout: float = 0.1,
        track_history: bool = True,
    ):
        """
        Initialise HRM_4x4_Simple model.

        Args:
            hidden_dim: Dimension of all hidden states.
            embed_dim: Cell embedding dimension before projection.
            n_iterations: Maximum flat iterations (N). Default: 50.
            convergence_threshold: L2 threshold for convergence detection.
            n_worker_layers: Number of stacked Worker MLP blocks. Default: 8.
            dropout: Dropout probability for regularisation.
            track_history: Whether to track execution trace.
        """
        super().__init__()

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if n_iterations <= 0:
            raise ValueError(f"n_iterations must be positive, got {n_iterations}")
        if convergence_threshold <= 0:
            raise ValueError(
                f"convergence_threshold must be positive, got {convergence_threshold}"
            )
        if n_worker_layers <= 0:
            raise ValueError(
                f"n_worker_layers must be positive, got {n_worker_layers}"
            )
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Store configuration
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.n_worker_layers = n_worker_layers
        self.dropout = dropout
        self.track_history = track_history

        # =====================================================================
        # Component 1: Input Network (embedding + projection)
        # =====================================================================
        self.input_network = InputNetwork(
            vocab_size=self.VOCAB_SIZE,
            grid_size=self.GRID_SIZE,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_positional=True,
        )

        # =====================================================================
        # Component 2: Worker Layers (expanded L-module, no Planner)
        # =====================================================================
        # Use x_in as context instead of h_H. The Worker concatenates
        # [h_L_prev, h_H, x_in] — we pass x_in for both h_H and x_in slots,
        # so we keep use_input_proj=True and provide x_in as the h_H argument.
        self.worker_layers = nn.ModuleList([
            WorkerModule(
                hidden_dim=hidden_dim,
                mlp_ratio=4,
                dropout=dropout,
                use_input_proj=True,
            )
            for _ in range(n_worker_layers)
        ])

        # =====================================================================
        # Component 3: Output Network (action decoder f_O)
        # =====================================================================
        self.output_network = OutputNetwork(
            hidden_dim=hidden_dim,
            grid_size=self.GRID_SIZE,
            use_shared_mlp=True,
            mlp_ratio=2,
            dropout=dropout,
        )

        # =====================================================================
        # Learned Initial State (h_L only — no h_H needed)
        # =====================================================================
        self.h_L_init = nn.Parameter(torch.zeros(1, hidden_dim))

        # Initialise learned state
        self._init_states()

    def _init_states(self) -> None:
        """Initialise learned initial state."""
        nn.init.normal_(self.h_L_init, mean=0.0, std=0.02)

    def _worker_block(
        self,
        h_L: torch.Tensor,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply all Worker layers sequentially for one iteration step.

        Each layer receives h_L as the previous state and x_in as both
        the high-level context (replacing h_H) and input grounding.

        Args:
            h_L: Current low-level hidden state of shape (batch, hidden_dim).
            x_in: Input embedding of shape (batch, hidden_dim).

        Returns:
            Updated hidden state of shape (batch, hidden_dim).
        """
        for worker in self.worker_layers:
            h_L = worker(h_L, h_H=x_in, x_in=x_in)
        return h_L

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Run simplified HRM forward pass (flat iteration, no Planner).

        Processes a batch of 4x4 Sudoku puzzles through:
        1. Input embedding
        2. Flat iteration loop (Worker only, N steps)
        3. Output decoding from final h_L

        One-step gradient approximation is applied: h_L is detached before
        each iteration so gradients only flow through the final step.

        Args:
            x: Input tensor of shape (batch, 4, 4) with integer values
               in range [0, 4] where 0=empty, 1-4=digits.
            return_intermediates: If True, include x_in in the trace.

        Returns:
            Dictionary containing:
                - 'cell_logits': (batch, 16) cell selection logits
                - 'digit_logits': (batch, 4) digit selection logits
                - 'trace': SimpleExecutionTrace with convergence statistics
                - 'h_L_final': Final low-level hidden state
        """
        # Validate input shape
        if x.dim() != 3 or x.shape[1:] != (self.GRID_SIZE, self.GRID_SIZE):
            raise ValueError(
                f"Expected input shape (batch, 4, 4), got {x.shape}"
            )

        batch_size = x.shape[0]
        device = x.device

        # =================================================================
        # Step 1: Embed input puzzle
        # =================================================================
        x_in = self.input_network(x)  # (batch, hidden_dim)

        # =================================================================
        # Step 2: Expand learned initial state to batch size
        # =================================================================
        h_L = self.h_L_init.expand(batch_size, -1).to(device)  # (batch, hidden_dim)

        # =================================================================
        # Step 3: Flat iteration loop (no Planner, no outer/inner split)
        # =================================================================
        residual_history: List[float] = []
        converged = False
        num_steps = 0
        final_residual = 0.0

        for step in range(self.n_iterations):
            # One-step gradient approximation: detach previous state
            h_L_prev = h_L.detach()

            # Apply all worker layers for this iteration
            h_L = self._worker_block(h_L_prev, x_in)

            num_steps = step + 1

            # Compute residual for convergence check
            residual = compute_residual(h_L, h_L_prev)
            mean_residual = residual.mean().item()
            final_residual = mean_residual

            if self.track_history:
                residual_history.append(mean_residual)

            # Early stopping on convergence
            if mean_residual < self.convergence_threshold:
                converged = True
                break

        # =================================================================
        # Step 4: Decode to action predictions from final h_L
        # =================================================================
        # In the simplified model, h_L is the sole hidden state — no h_H.
        cell_logits, digit_logits = self.output_network(h_L)

        # =================================================================
        # Step 5: Build execution trace
        # =================================================================
        trace = SimpleExecutionTrace(
            num_steps=num_steps,
            max_steps=self.n_iterations,
            converged=converged,
            final_residual=final_residual,
            residual_history=residual_history,
            x_in=x_in if return_intermediates else None,
            final_h_L=h_L,
        )

        return {
            'cell_logits': cell_logits,
            'digit_logits': digit_logits,
            'trace': trace,
            'h_L_final': h_L,
        }

    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action predictions for inference.

        Convenience method that runs forward pass and returns
        predicted cell and digit indices.

        Args:
            x: Input tensor of shape (batch, 4, 4).

        Returns:
            Tuple of (cell_indices, digit_indices):
                - cell_indices: (batch,) predicted cell index [0, 15]
                - digit_indices: (batch,) predicted digit index [0, 3]
                  (add 1 to get actual Sudoku digit 1-4)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            cell_indices = outputs['cell_logits'].argmax(dim=-1)
            digit_indices = outputs['digit_logits'].argmax(dim=-1)
        return cell_indices, digit_indices

    def get_halt_penalty(
        self,
        outputs: Dict[str, Any],
        lambda_halt: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute halting penalty for encouraging efficient computation.

        Penalises the number of computation steps taken, creating a
        speed-accuracy trade-off controlled by lambda_halt.

        Formula:
            penalty = lambda_halt * (steps_taken / max_steps)

        Args:
            outputs: Output dictionary from forward() containing 'trace'.
            lambda_halt: Weight for the halt penalty. Default: 0.01.

        Returns:
            Scalar tensor with the halt penalty value.
        """
        trace: SimpleExecutionTrace = outputs['trace']
        ponder_cost = trace.num_steps / trace.max_steps

        return torch.tensor(
            lambda_halt * ponder_cost,
            device=outputs['cell_logits'].device,
        )

    def get_convergence_loss(
        self,
        outputs: Dict[str, Any],
        target_residual: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute convergence loss to encourage stable fixed-points.

        Penalises the final residual using a hinge-style loss.

        Formula:
            loss = max(0, final_residual - target_residual)

        Args:
            outputs: Output dictionary from forward() containing 'trace'.
            target_residual: Target residual to achieve. Default: 1e-4.

        Returns:
            Scalar tensor with the convergence loss value.
        """
        trace: SimpleExecutionTrace = outputs['trace']
        loss = max(0.0, trace.final_residual - target_residual)

        return torch.tensor(loss, device=outputs['cell_logits'].device)

    def get_state_dynamics(
        self,
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyse the dynamics of hidden state evolution.

        Provides metrics about how h_L evolved during flat iteration.

        Args:
            outputs: Output dictionary from forward().

        Returns:
            Dictionary with dynamics metrics:
                - 'residual_history': Per-step residuals
                - 'total_steps': Total computation steps
                - 'converged': Whether iteration converged
                - 'final_residual': Final residual value
        """
        trace: SimpleExecutionTrace = outputs['trace']

        return {
            'residual_history': trace.residual_history,
            'total_steps': trace.num_steps,
            'converged': trace.converged,
            'final_residual': trace.final_residual,
        }

    def extra_repr(self) -> str:
        """Return string representation of model configuration."""
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"embed_dim={self.embed_dim}, "
            f"n_iterations={self.n_iterations}, "
            f"n_worker_layers={self.n_worker_layers}, "
            f"convergence_threshold={self.convergence_threshold}, "
            f"dropout={self.dropout}"
        )


def create_hrm_4x4_simple(
    hidden_dim: int = 64,
    n_iterations: int = 50,
    n_worker_layers: int = 8,
    **kwargs,
) -> HRM_4x4_Simple:
    """
    Factory function to create HRM_4x4_Simple model.

    Provides sensible defaults for the L-module only variant.

    Args:
        hidden_dim: Dimension of hidden states. Default: 64.
        n_iterations: Flat iteration count (N). Default: 50.
        n_worker_layers: Number of Worker MLP blocks. Default: 8.
        **kwargs: Additional arguments passed to HRM_4x4_Simple.

    Returns:
        Configured HRM_4x4_Simple model instance.

    Example:
        >>> model = create_hrm_4x4_simple()
        >>> model = create_hrm_4x4_simple(hidden_dim=128, n_iterations=100)
    """
    return HRM_4x4_Simple(
        hidden_dim=hidden_dim,
        n_iterations=n_iterations,
        n_worker_layers=n_worker_layers,
        **kwargs,
    )
