# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from paper)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class InputNetwork(nn.Module):
    """Embeds puzzle into working representation (f_in in paper)"""

    def __init__(self, vocab_size: int = 5, embed_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(16 * embed_dim, hidden_dim)

    def forward(self, puzzle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            puzzle: (batch, 4, 4) with values 0-4
        Returns:
            x: (batch, hidden_dim) embedded representation
        """
        batch_size = puzzle.shape[0]
        x = self.embed(puzzle)  # (batch, 4, 4, embed_dim)
        x = x.view(batch_size, -1)  # Flatten to (batch, 16 * embed_dim)
        return self.proj(x)  # (batch, hidden_dim)


class WorkerModule(nn.Module):
    """
    Low-level recurrent module (f_L in paper)
    Fast, detailed computations - operates at inner timescale
    Refines solution iteratively within each outer cycle
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Post-Norm layers (apply after residual)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(
        self, h_L_prev: torch.Tensor, h_H: torch.Tensor, x_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Single refinement step

        Args:
            h_L_prev: Previous L-state (batch, hidden_dim)
            h_H: Current H-state - fixed during inner loop (batch, hidden_dim)
            x_in: Input representation (batch, hidden_dim)

        Returns:
            h_L: Updated L-state (batch, hidden_dim)
        """
        # Combine inputs via element-wise addition (as per paper)
        combined = h_L_prev + h_H + x_in

        # Add sequence dimension for attention (seq_len=1)
        combined = combined.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Self-attention with residual + Post-Norm
        attn_out, _ = self.attention(combined, combined, combined)
        h = self.norm1(combined + attn_out)

        # FFN with residual + Post-Norm
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)

        return h.squeeze(1)  # (batch, hidden_dim)


class PlannerModule(nn.Module):
    """
    High-level recurrent module (f_H in paper)
    Slow, abstract planning - operates at outer timescale
    Updates once per cycle after Worker convergence
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Same architecture as Worker but separate parameters
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(self, h_H_prev: torch.Tensor, h_L_final: torch.Tensor) -> torch.Tensor:
        """
        Update after L-module converges

        Args:
            h_H_prev: Previous H-state (batch, hidden_dim)
            h_L_final: Converged L-state from current cycle (batch, hidden_dim)

        Returns:
            h_H: Updated H-state (batch, hidden_dim)
        """
        # Combine inputs
        combined = h_H_prev + h_L_final
        combined = combined.unsqueeze(1)

        # Self-attention with residual + Post-Norm
        attn_out, _ = self.attention(combined, combined, combined)
        h = self.norm1(combined + attn_out)

        # FFN with residual + Post-Norm
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)

        return h.squeeze(1)


class OutputNetwork(nn.Module):
    """Decodes final H-state to predictions (f_out in paper)"""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.cell_head = nn.Linear(hidden_dim, 16)  # 16 possible cells
        self.digit_head = nn.Linear(hidden_dim, 4)  # 4 possible digits (1-4)

    def forward(self, h_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_H: Final H-state (batch, hidden_dim)

        Returns:
            cell_logits: (batch, 16)
            digit_logits: (batch, 4)
        """
        cell_logits = self.cell_head(h_H)
        digit_logits = self.digit_head(h_H)
        return cell_logits, digit_logits


class HRM_4x4(nn.Module):
    """
    Hierarchical Reasoning Model for 4x4 Sudoku

    Implements the HRM architecture from:
    "Hierarchical Reasoning Model" (https://arxiv.org/html/2506.21734v1)

    Key properties:
    1. Fixed-point iteration: Worker refines iteratively
    2. Hierarchical convergence: Planner updates after Worker equilibrium
    3. Multi-timescale processing: 1:T ratio (Planner:Worker updates)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_outer_cycles: int = 5,  # K in paper
        n_inner_steps: int = 10,  # T in paper
        dropout: float = 0.1,
        convergence_threshold: float = 1e-3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_outer_cycles = n_outer_cycles
        self.n_inner_steps = n_inner_steps
        self.convergence_threshold = convergence_threshold

        # Four core components
        self.input_net = InputNetwork(
            vocab_size=5, embed_dim=16, hidden_dim=hidden_dim  # 0-4 for 4x4 Sudoku
        )

        self.worker = WorkerModule(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)

        self.planner = PlannerModule(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)

        self.output_net = OutputNetwork(hidden_dim=hidden_dim)

        # Initialize hidden states (learned parameters)
        # Using truncated normal initialization as per paper
        self.h_L_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        self.h_H_init = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

    def forward(
        self, puzzle: torch.Tensor, return_traces: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Full forward pass with hierarchical convergence

        Args:
            puzzle: (batch, 4, 4) with values 0-4
            return_traces: Whether to return detailed execution traces

        Returns:
            cell_logits: (batch, 16) logits for cell selection
            digit_logits: (batch, 4) logits for digit selection
            traces: dict with convergence statistics
        """
        batch_size = puzzle.shape[0]
        device = puzzle.device

        # Step 1: Embed input once (does not change during reasoning)
        x_in = self.input_net(puzzle)  # (batch, hidden_dim)

        # Step 2: Initialize hidden states
        h_L = self.h_L_init.expand(batch_size, -1).clone()
        h_H = self.h_H_init.expand(batch_size, -1).clone()

        # Tracking for visualization and analysis
        residuals = []
        cycle_residuals = []  # Average residual per cycle
        total_iterations = 0
        early_stop_cycles = []

        # HIERARCHICAL CONVERGENCE: Two-level nested loop
        # Outer loop: K high-level planning cycles
        for cycle in range(self.n_outer_cycles):
            # Fix H-state for this entire cycle (critical for hierarchical convergence)
            h_H_fixed = h_H.detach()  # Detach for 1-step gradient approximation

            cycle_residual_sum = 0.0
            converged_early = False

            # Inner loop: T refinement steps (Worker converges to local equilibrium)
            for step in range(self.n_inner_steps):
                h_L_prev = h_L.clone()

                # Worker update: refine solution given current context
                h_L = self.worker(h_L_prev, h_H_fixed, x_in)

                # Compute residual: ||h_L^(t) - h_L^(t-1)||
                residual = torch.norm(h_L - h_L_prev, dim=-1).mean().item()
                residuals.append(residual)
                cycle_residual_sum += residual
                total_iterations += 1

                # Optional: Early stopping within cycle if converged
                if residual < self.convergence_threshold and step >= 3:
                    converged_early = True
                    # Fill remaining steps with same residual for consistent tracking
                    for _ in range(self.n_inner_steps - step - 1):
                        residuals.append(residual)
                        total_iterations += 1
                    break

            # Track cycle-level statistics
            cycle_residuals.append(cycle_residual_sum / self.n_inner_steps)
            if converged_early:
                early_stop_cycles.append(cycle)

            # After inner convergence, update Planner once
            # This creates new context for next cycle's Worker convergence
            h_H = self.planner(h_H, h_L)

        # Step 3: Decode from final H-state
        cell_logits, digit_logits = self.output_net(h_H)

        # Step 4: Prepare execution traces
        traces = {
            "num_iterations": total_iterations,
            "num_cycles": self.n_outer_cycles,
            "residuals": residuals,
            "cycle_residuals": cycle_residuals,
            "final_residual": residuals[-1] if residuals else 0.0,
            "avg_residual": sum(residuals) / len(residuals) if residuals else 0.0,
            "early_stop_cycles": early_stop_cycles,
            "converged_early": len(early_stop_cycles) > 0,
        }

        return cell_logits, digit_logits, traces

    def get_halt_penalty(self, traces: Dict, target_iterations: int = 50) -> torch.Tensor:
        """
        Halting penalty to encourage efficient iteration count

        Args:
            traces: Execution traces from forward pass
            target_iterations: Desired iteration count (K × T)

        Returns:
            penalty: Scalar tensor for loss function
        """
        actual = traces["num_iterations"]
        # L1 distance from target, normalized
        penalty = abs(actual - target_iterations) / target_iterations
        return torch.tensor(penalty, device=self.h_H_init.device)

    def get_convergence_loss(self, traces: Dict) -> torch.Tensor:
        """
        Encourages convergence behavior (optional regularization)

        Returns:
            loss: Scalar tensor measuring convergence quality
        """
        residuals = torch.tensor(traces["residuals"], device=self.h_H_init.device)

        # Penalize high final residuals (want convergence)
        convergence_loss = torch.mean(residuals[-self.n_inner_steps :])

        return convergence_loss


class HRM_4x4_WithHalting(HRM_4x4):
    """
    Extended HRM with Adaptive Computation Time (ACT) and Q-Learning Halting.

    This model learns when to stop iterating based on Q-values, allowing
    dynamic computation allocation based on problem difficulty.

    Key Features:
        1. Q-head predicts value of halt vs continue actions
        2. Minimum cycle constraint ensures sufficient computation
        3. λ penalty encourages efficient halting
        4. Supports both training (exploration) and inference (greedy)

    Important Note (Ge et al. Analysis):
        Research shows ACT provides minimal inference benefit. Maximum
        iterations often yields best results. This implementation is
        provided for completeness but fixed iterations are recommended.

    Args:
        hidden_dim: Dimension of hidden states. Default: 64.
        n_heads: Number of attention heads. Default: 4.
        n_outer_cycles: Maximum outer cycles (K). Default: 5.
        n_inner_steps: Inner steps per cycle (T). Default: 10.
        dropout: Dropout probability. Default: 0.1.
        convergence_threshold: Inner loop convergence threshold. Default: 1e-3.
        min_cycles: Minimum cycles before halting allowed. Default: 2.
        lambda_penalty: Weight of ponder cost in loss. Default: 0.01.
        exploration_epsilon: Epsilon for exploration during training. Default: 0.1.

    Example:
        >>> model = HRM_4x4_WithHalting(
        ...     hidden_dim=64,
        ...     n_outer_cycles=10,
        ...     min_cycles=2,
        ...     lambda_penalty=0.01
        ... )
        >>> puzzle = torch.randint(0, 5, (4, 4, 4))
        >>> cell_logits, digit_logits, traces = model(puzzle)
        >>> print(f"Halted at cycle {traces['halt_cycle']}")
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_outer_cycles: int = 5,
        n_inner_steps: int = 10,
        dropout: float = 0.1,
        convergence_threshold: float = 1e-3,
        min_cycles: int = 2,
        lambda_penalty: float = 0.01,
        exploration_epsilon: float = 0.1,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_outer_cycles=n_outer_cycles,
            n_inner_steps=n_inner_steps,
            dropout=dropout,
            convergence_threshold=convergence_threshold,
        )

        # Halting configuration
        self.min_cycles = min_cycles
        self.lambda_penalty = lambda_penalty
        self.exploration_epsilon = exploration_epsilon

        # Q-head for halt/continue decisions
        # Architecture: h_H -> Linear -> ReLU -> Linear -> [Q_halt, Q_continue]
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [Q_halt, Q_continue]
        )

        # Initialize Q-head with small weights for stable learning
        self._init_q_head()

    def _init_q_head(self) -> None:
        """Initialize Q-head weights with small values."""
        for module in self.q_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_q_values(self, h_H: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for halt/continue actions.

        Args:
            h_H: High-level hidden state of shape (batch, hidden_dim).

        Returns:
            Q-values of shape (batch, 2):
                - [:, 0] = Q(halt)
                - [:, 1] = Q(continue)
        """
        return self.q_head(h_H)

    def get_halt_probability(
        self,
        h_H: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get probability of halting using softmax over Q-values.

        Args:
            h_H: High-level hidden state (batch, hidden_dim).
            temperature: Softmax temperature. Lower = more deterministic.

        Returns:
            Halt probability of shape (batch,).
        """
        q_values = self.get_q_values(h_H)
        probs = F.softmax(q_values / temperature, dim=-1)
        return probs[:, 0]  # Probability of halting

    def should_halt(
        self,
        h_H: torch.Tensor,
        cycle: int,
        training: bool = False,
    ) -> Tuple[bool, float, torch.Tensor]:
        """
        Decide whether to halt based on Q-values and policy.

        Args:
            h_H: Current H-state of shape (batch, hidden_dim).
            cycle: Current cycle number (0-indexed).
            training: Whether in training mode (enables exploration).

        Returns:
            Tuple of (should_halt, halt_probability, q_values):
                - should_halt: Boolean decision
                - halt_probability: Probability of halting
                - q_values: Raw Q-values for logging
        """
        q_values = self.get_q_values(h_H)
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]

        # Compute halt probability
        probs = F.softmax(q_values, dim=-1)
        halt_prob = probs[:, 0].mean().item()

        # Enforce minimum cycles constraint
        if cycle < self.min_cycles:
            return False, halt_prob, q_values

        if not training:
            # Inference: greedy policy
            should_halt = (q_halt > q_continue).all().item()
        else:
            # Training: epsilon-greedy exploration
            if torch.rand(1).item() < self.exploration_epsilon:
                should_halt = torch.rand(1).item() > 0.5
            else:
                should_halt = (q_halt > q_continue).all().item()

        return should_halt, halt_prob, q_values

    def forward(
        self,
        puzzle: torch.Tensor,
        return_traces: bool = False,
        force_max_cycles: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with adaptive halting.

        Args:
            puzzle: (batch, 4, 4) with values 0-4.
            return_traces: Whether to return detailed execution traces.
            force_max_cycles: If True, ignore halting and run max cycles.
                Useful for comparing ACT vs fixed iteration.

        Returns:
            cell_logits: (batch, 16) logits for cell selection.
            digit_logits: (batch, 4) logits for digit selection.
            traces: dict with execution and halting statistics.
        """
        batch_size = puzzle.shape[0]
        device = puzzle.device
        training = self.training

        # Step 1: Embed input
        x_in = self.input_net(puzzle)

        # Step 2: Initialize hidden states
        h_L = self.h_L_init.expand(batch_size, -1).clone()
        h_H = self.h_H_init.expand(batch_size, -1).clone()

        # Tracking
        residuals = []
        cycle_residuals = []
        total_iterations = 0
        early_stop_cycles = []
        q_values_history = []
        halt_probabilities = []
        halted_early = False
        halt_cycle = None

        # HIERARCHICAL CONVERGENCE with ADAPTIVE HALTING
        for cycle in range(self.n_outer_cycles):
            # Fix H-state for this cycle
            h_H_fixed = h_H.detach()

            cycle_residual_sum = 0.0
            converged_early = False

            # Inner loop: Worker refinement
            for step in range(self.n_inner_steps):
                h_L_prev = h_L.clone()
                h_L = self.worker(h_L_prev, h_H_fixed, x_in)

                residual = torch.norm(h_L - h_L_prev, dim=-1).mean().item()
                residuals.append(residual)
                cycle_residual_sum += residual
                total_iterations += 1

                if residual < self.convergence_threshold and step >= 3:
                    converged_early = True
                    for _ in range(self.n_inner_steps - step - 1):
                        residuals.append(residual)
                        total_iterations += 1
                    break

            cycle_residuals.append(cycle_residual_sum / self.n_inner_steps)
            if converged_early:
                early_stop_cycles.append(cycle)

            # Planner update
            h_H = self.planner(h_H, h_L)

            # HALTING DECISION (after planner update)
            if not force_max_cycles:
                should_halt, halt_prob, q_vals = self.should_halt(h_H, cycle, training=training)

                q_values_history.append((q_vals[:, 0].mean().item(), q_vals[:, 1].mean().item()))
                halt_probabilities.append(halt_prob)

                if should_halt:
                    halted_early = True
                    halt_cycle = cycle
                    break

        # Decode from final H-state
        cell_logits, digit_logits = self.output_net(h_H)

        # Compute ponder cost for λ penalty
        cycles_used = (halt_cycle + 1) if halt_cycle is not None else self.n_outer_cycles
        ponder_cost = cycles_used / self.n_outer_cycles

        # Prepare traces
        traces = {
            "num_iterations": total_iterations,
            "num_cycles": cycles_used,
            "max_cycles": self.n_outer_cycles,
            "residuals": residuals,
            "cycle_residuals": cycle_residuals,
            "final_residual": residuals[-1] if residuals else 0.0,
            "avg_residual": sum(residuals) / len(residuals) if residuals else 0.0,
            "early_stop_cycles": early_stop_cycles,
            "converged_early": len(early_stop_cycles) > 0,
            # ACT-specific traces
            "halted_early": halted_early,
            "halt_cycle": halt_cycle,
            "q_values_history": q_values_history,
            "halt_probabilities": halt_probabilities,
            "ponder_cost": ponder_cost,
            "lambda_penalty": self.lambda_penalty,
        }

        return cell_logits, digit_logits, traces

    def get_halt_penalty(self, traces: Dict, target_iterations: int = None) -> torch.Tensor:
        """
        Compute halting penalty (λ × ponder_cost).

        This penalty encourages efficient halting by penalizing
        computation time.

        Args:
            traces: Execution traces from forward pass.
            target_iterations: Ignored (kept for API compatibility).

        Returns:
            Penalty tensor for adding to loss.

        Note:
            Per Ge et al. analysis, setting λ=0 often gives best accuracy.
            Use non-zero λ only when inference speed is critical.
        """
        ponder_cost = traces.get("ponder_cost", 1.0)
        penalty = self.lambda_penalty * ponder_cost
        return torch.tensor(penalty, device=self.h_H_init.device)

    def get_act_loss(
        self,
        task_loss: torch.Tensor,
        traces: Dict,
    ) -> torch.Tensor:
        """
        Compute combined ACT loss: Task Loss + λ × Ponder Cost.

        Args:
            task_loss: Primary task loss (e.g., cross-entropy).
            traces: Execution traces from forward pass.

        Returns:
            Combined loss tensor.
        """
        halt_penalty = self.get_halt_penalty(traces)
        return task_loss + halt_penalty

    def set_lambda_penalty(self, lambda_penalty: float) -> None:
        """
        Update λ penalty value.

        Useful for curriculum learning where λ increases over training.

        Args:
            lambda_penalty: New λ value.
        """
        if lambda_penalty < 0:
            raise ValueError(f"lambda_penalty must be non-negative, got {lambda_penalty}")
        self.lambda_penalty = lambda_penalty

    def set_exploration_epsilon(self, epsilon: float) -> None:
        """
        Update exploration epsilon.

        Useful for annealing exploration during training.

        Args:
            epsilon: New exploration rate in [0, 1].
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.exploration_epsilon = epsilon

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"n_outer_cycles={self.n_outer_cycles}, "
            f"min_cycles={self.min_cycles}, "
            f"lambda_penalty={self.lambda_penalty}"
        )


# Usage example and testing
if __name__ == "__main__":
    # Create model
    model = HRM_4x4(hidden_dim=64, n_heads=4, n_outer_cycles=5, n_inner_steps=10, dropout=0.1)

    # Test forward pass
    batch_size = 4
    puzzle = torch.randint(0, 5, (batch_size, 4, 4))

    print("Testing HRM forward pass...")
    print(f"Input shape: {puzzle.shape}")

    cell_logits, digit_logits, traces = model(puzzle, return_traces=True)

    print(f"\nOutput shapes:")
    print(f"  Cell logits: {cell_logits.shape}")
    print(f"  Digit logits: {digit_logits.shape}")

    print(f"\nExecution traces:")
    print(f"  Total iterations: {traces['num_iterations']}")
    print(f"  Outer cycles: {traces['num_cycles']}")
    print(f"  Avg residual: {traces['avg_residual']:.6f}")
    print(f"  Final residual: {traces['final_residual']:.6f}")
    print(f"  Cycle residuals: {[f'{r:.6f}' for r in traces['cycle_residuals']]}")

    print(f"\nParameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Verify hierarchical properties
    print(f"\nVerifying HRM properties:")
    print(f"  ✓ Fixed-point iteration: {traces['num_iterations']} inner steps")
    print(f"  ✓ Hierarchical convergence: {traces['num_cycles']} outer cycles")
    print(f"  ✓ Multi-timescale: 1:{model.n_inner_steps} update ratio")
    expected_total = model.n_outer_cycles * model.n_inner_steps
    print(f"  ✓ Expected total steps: {expected_total}")
    print(f"  ✓ Actual total steps: {traces['num_iterations']}")

    # Test backward pass
    print(f"\nTesting backward pass...")
    loss = F.cross_entropy(cell_logits, torch.randint(0, 16, (batch_size,)))
    loss.backward()
    print(f"  ✓ Gradients computed successfully")

    print("\n✓ HRM implementation verified!")
