"""
Adaptive Computation Time (ACT) Extension with Q-Learning

This module implements learned halting decisions for the HRM, allowing
the model to dynamically decide when to stop iterating based on Q-values.

Key Components:
    1. QHaltingHead: Neural network that outputs Q-values for halt/continue
    2. HaltingPolicy: Manages halting decisions with exploration
    3. ACTStats: Statistics tracking for halting behavior

Important Note (Ge et al. Analysis):
    Research shows ACT provides minimal inference benefit for HRM-style
    architectures. Maximum iterations often yields best results because:
    - The model learns to use all available computation
    - Early halting can miss important refinements
    - The λ penalty creates a speed-accuracy trade-off
    
    This implementation is provided for completeness and experimentation,
    but the default recommendation is to use fixed iteration counts.

Reference:
    - Adaptive Computation Time for Recurrent Neural Networks (Graves, 2016)
    - HRM Paper: Fixed-point iteration analysis
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACTStats:
    """
    Statistics from Adaptive Computation Time execution.
    
    Attributes:
        num_cycles_used: Actual number of outer cycles executed.
        max_cycles: Maximum cycles allowed.
        total_inner_steps: Total Worker iterations.
        halted_early: Whether execution halted before max_cycles.
        halt_cycle: Cycle at which halting occurred (None if ran to max).
        q_values_history: Q-values at each cycle for analysis.
        halt_probabilities: Softmax probabilities of halting at each cycle.
        ponder_cost: Cumulative computation cost (for λ penalty).
    """
    
    num_cycles_used: int
    max_cycles: int
    total_inner_steps: int
    halted_early: bool
    halt_cycle: Optional[int] = None
    q_values_history: List[Tuple[float, float]] = field(default_factory=list)
    halt_probabilities: List[float] = field(default_factory=list)
    ponder_cost: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"ACTStats(cycles={self.num_cycles_used}/{self.max_cycles}, "
            f"halted_early={self.halted_early}, "
            f"ponder_cost={self.ponder_cost:.4f})"
        )
    
    @property
    def efficiency(self) -> float:
        """
        Compute computation efficiency (cycles saved / max cycles).
        
        Higher values indicate more efficient early halting.
        Returns 0.0 if no early halting occurred.
        """
        if not self.halted_early:
            return 0.0
        return (self.max_cycles - self.num_cycles_used) / self.max_cycles
    
    @property
    def average_halt_probability(self) -> Optional[float]:
        """Average probability of halting across cycles."""
        if not self.halt_probabilities:
            return None
        return sum(self.halt_probabilities) / len(self.halt_probabilities)


class QHaltingHead(nn.Module):
    """
    Q-network head for halt/continue decisions.
    
    Outputs Q-values for two actions:
        - Q(halt): Expected value of stopping now
        - Q(continue): Expected value of continuing iteration
    
    Architecture:
        h_H → Linear → ReLU → Linear → [Q_halt, Q_continue]
    
    Args:
        hidden_dim: Dimension of input hidden state.
        intermediate_dim: Dimension of intermediate layer. Default: 32.
    
    Example:
        >>> q_head = QHaltingHead(hidden_dim=64)
        >>> h_H = torch.randn(8, 64)
        >>> q_values = q_head(h_H)  # (8, 2)
        >>> q_halt, q_continue = q_values[:, 0], q_values[:, 1]
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 32,
    ):
        super().__init__()
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")
        
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 2),  # [Q_halt, Q_continue]
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with small values for stable Q-learning start."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, h_H: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for halt/continue actions.
        
        Args:
            h_H: High-level hidden state of shape (batch, hidden_dim).
        
        Returns:
            Q-values of shape (batch, 2) where:
                - [:, 0] = Q(halt)
                - [:, 1] = Q(continue)
        """
        return self.network(h_H)
    
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
        q_values = self.forward(h_H)
        probs = F.softmax(q_values / temperature, dim=-1)
        return probs[:, 0]  # Probability of halting
    
    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}"


class HaltingPolicy:
    """
    Policy for making halting decisions with exploration.
    
    Supports different exploration strategies:
        - greedy: Always take best action (Q_halt > Q_continue)
        - epsilon_greedy: Random action with probability ε
        - softmax: Sample from softmax distribution over Q-values
    
    Args:
        min_cycles: Minimum cycles before halting allowed. Default: 2.
        exploration_strategy: One of 'greedy', 'epsilon_greedy', 'softmax'.
        epsilon: Exploration rate for epsilon-greedy. Default: 0.1.
        temperature: Temperature for softmax exploration. Default: 1.0.
    
    Example:
        >>> policy = HaltingPolicy(min_cycles=2, exploration_strategy='greedy')
        >>> q_values = torch.tensor([[0.5, 0.3]])  # Q_halt > Q_continue
        >>> should_halt = policy.should_halt(q_values, cycle=3, training=False)
        >>> should_halt
        True
    """
    
    def __init__(
        self,
        min_cycles: int = 2,
        exploration_strategy: str = "greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
    ):
        if min_cycles < 0:
            raise ValueError(f"min_cycles must be non-negative, got {min_cycles}")
        if exploration_strategy not in ("greedy", "epsilon_greedy", "softmax"):
            raise ValueError(
                f"Unknown exploration_strategy: {exploration_strategy}. "
                f"Expected one of: 'greedy', 'epsilon_greedy', 'softmax'"
            )
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        
        self.min_cycles = min_cycles
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.temperature = temperature
    
    def should_halt(
        self,
        q_values: torch.Tensor,
        cycle: int,
        training: bool = False,
    ) -> Tuple[bool, float]:
        """
        Decide whether to halt based on Q-values and policy.
        
        Args:
            q_values: Q-values of shape (batch, 2) from QHaltingHead.
            cycle: Current cycle number (0-indexed).
            training: Whether in training mode (enables exploration).
        
        Returns:
            Tuple of (should_halt, halt_probability):
                - should_halt: Boolean decision
                - halt_probability: Probability of halting (for logging)
        """
        # Enforce minimum cycles constraint
        if cycle < self.min_cycles:
            return False, 0.0
        
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]
        
        # Compute halt probability for logging
        probs = F.softmax(q_values / self.temperature, dim=-1)
        halt_prob = probs[:, 0].mean().item()
        
        if not training:
            # Inference: always use greedy policy
            should_halt = (q_halt > q_continue).all().item()
            return should_halt, halt_prob
        
        # Training: apply exploration strategy
        if self.exploration_strategy == "greedy":
            should_halt = (q_halt > q_continue).all().item()
        
        elif self.exploration_strategy == "epsilon_greedy":
            if torch.rand(1).item() < self.epsilon:
                # Random action
                should_halt = torch.rand(1).item() > 0.5
            else:
                should_halt = (q_halt > q_continue).all().item()
        
        elif self.exploration_strategy == "softmax":
            # Sample from softmax distribution
            action = torch.multinomial(probs.mean(dim=0, keepdim=True), 1).item()
            should_halt = (action == 0)
        
        else:
            should_halt = False
        
        return should_halt, halt_prob


def compute_ponder_cost(
    num_cycles: int,
    max_cycles: int,
    cost_per_cycle: float = 1.0,
) -> float:
    """
    Compute ponder cost for ACT penalty.
    
    The ponder cost encourages efficient computation by penalizing
    the number of cycles used.
    
    Args:
        num_cycles: Number of cycles actually used.
        max_cycles: Maximum cycles allowed.
        cost_per_cycle: Cost per cycle. Default: 1.0.
    
    Returns:
        Normalized ponder cost in [0, 1].
    """
    return (num_cycles * cost_per_cycle) / max_cycles


def compute_act_loss(
    ponder_cost: float,
    task_loss: torch.Tensor,
    lambda_penalty: float = 0.01,
) -> torch.Tensor:
    """
    Compute combined ACT loss with λ penalty.
    
    ACT Loss = Task Loss + λ × Ponder Cost
    
    The λ penalty encourages the model to halt early when confident,
    trading off computation time for accuracy.
    
    Args:
        ponder_cost: Normalized ponder cost from compute_ponder_cost.
        task_loss: Primary task loss (e.g., cross-entropy).
        lambda_penalty: Weight of ponder cost penalty. Default: 0.01.
            - Higher λ: More aggressive halting (faster, less accurate)
            - Lower λ: More computation (slower, more accurate)
    
    Returns:
        Combined loss tensor.
    
    Note (Ge et al.):
        In practice, λ=0 (no penalty) often gives best accuracy.
        Use non-zero λ only when inference speed is critical.
    """
    ponder_tensor = torch.tensor(
        ponder_cost, 
        device=task_loss.device, 
        dtype=task_loss.dtype
    )
    return task_loss + lambda_penalty * ponder_tensor


class HaltingQTrainer:
    """
    Trainer for the Q-halting head using temporal difference learning.
    
    Uses Q-learning to train the halting head:
        Q(s, a) ← Q(s, a) + α × (r + γ × max_a' Q(s', a') - Q(s, a))
    
    Where:
        - s = h_H (hidden state)
        - a = halt/continue
        - r = -task_loss (negative loss as reward)
        - γ = discount factor
    
    Args:
        q_head: QHaltingHead module to train.
        learning_rate: Learning rate for Q-updates. Default: 0.001.
        discount_factor: γ for future reward discounting. Default: 0.99.
        target_update_freq: Steps between target network updates. Default: 100.
    
    Example:
        >>> trainer = HaltingQTrainer(q_head, learning_rate=0.001)
        >>> # After each cycle:
        >>> trainer.store_transition(h_H, action, reward, h_H_next, done)
        >>> # Periodically:
        >>> q_loss = trainer.update()
    """
    
    def __init__(
        self,
        q_head: QHaltingHead,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        target_update_freq: int = 100,
    ):
        self.q_head = q_head
        self.discount_factor = discount_factor
        self.target_update_freq = target_update_freq
        
        # Create target network (for stable Q-learning)
        self.target_q_head = QHaltingHead(
            hidden_dim=q_head.hidden_dim,
            intermediate_dim=q_head.intermediate_dim,
        )
        self.target_q_head.load_state_dict(q_head.state_dict())
        self.target_q_head.eval()
        
        # Optimizer for Q-head only
        self.optimizer = torch.optim.Adam(q_head.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer: List[Dict] = []
        self.max_buffer_size = 10000
        self.batch_size = 32
        
        self.update_count = 0
    
    def store_transition(
        self,
        h_H: torch.Tensor,
        action: int,  # 0 = halt, 1 = continue
        reward: float,
        h_H_next: Optional[torch.Tensor],
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            h_H: Current hidden state.
            action: Action taken (0=halt, 1=continue).
            reward: Reward received (typically -task_loss).
            h_H_next: Next hidden state (None if terminal).
            done: Whether episode ended.
        """
        transition = {
            'h_H': h_H.detach().clone(),
            'action': action,
            'reward': reward,
            'h_H_next': h_H_next.detach().clone() if h_H_next is not None else None,
            'done': done,
        }
        
        self.replay_buffer.append(transition)
        
        # Remove oldest if buffer full
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
    
    def update(self) -> Optional[float]:
        """
        Perform one Q-learning update step.
        
        Returns:
            Q-loss value if update performed, None if buffer too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        import random
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Stack tensors - handle varying input shapes by taking mean over batch dim
        h_H_list = []
        for t in batch:
            h = t['h_H']
            # If h_H has batch dimension, take mean to get single vector
            if h.dim() > 1:
                h = h.mean(dim=0)
            h_H_list.append(h)
        
        h_H_batch = torch.stack(h_H_list)  # (batch_size, hidden_dim)
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long, device=h_H_batch.device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32, device=h_H_batch.device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32, device=h_H_batch.device)
        
        # Get current Q-values
        current_q = self.q_head(h_H_batch)  # (batch_size, 2)
        current_q_selected = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get target Q-values
        with torch.no_grad():
            next_q_values = []
            for t in batch:
                if t['done'] or t['h_H_next'] is None:
                    next_q_values.append(0.0)
                else:
                    h_next = t['h_H_next']
                    if h_next.dim() > 1:
                        h_next = h_next.mean(dim=0, keepdim=True)
                    next_q = self.target_q_head(h_next)
                    next_q_values.append(next_q.max().item())
            
            target_q = rewards + self.discount_factor * torch.tensor(next_q_values, device=h_H_batch.device) * (1 - dones)
        
        # Q-learning loss
        q_loss = F.mse_loss(current_q_selected, target_q)
        
        # Update
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        
        # Update target network periodically
        if self.update_count % self.target_update_freq == 0:
            self.target_q_head.load_state_dict(self.q_head.state_dict())
        
        return q_loss.item()


def create_halting_components(
    hidden_dim: int,
    min_cycles: int = 2,
    exploration_strategy: str = "epsilon_greedy",
    epsilon: float = 0.1,
) -> Tuple[QHaltingHead, HaltingPolicy]:
    """
    Factory function to create halting components.
    
    Args:
        hidden_dim: Dimension of hidden states.
        min_cycles: Minimum cycles before halting allowed.
        exploration_strategy: Exploration strategy for training.
        epsilon: Exploration rate for epsilon-greedy.
    
    Returns:
        Tuple of (q_head, policy).
    """
    q_head = QHaltingHead(hidden_dim=hidden_dim)
    policy = HaltingPolicy(
        min_cycles=min_cycles,
        exploration_strategy=exploration_strategy,
        epsilon=epsilon,
    )
    return q_head, policy


class QHaltingHeadTransformer(nn.Module):
    """
    Sapient-compatible Q-halting head for transformer-based HRM.
    
    Unlike QHaltingHead which outputs (batch, 2) for unified halt/continue,
    this outputs per-position Q-values suitable for sequence-based reasoning.
    
    Uses separate linear projections for Q_halt and Q_continue, matching
    Sapient's architecture where halting decisions can be made per-position
    or aggregated across the sequence.
    
    Args:
        hidden_size: Model hidden dimension. Default: 256
        dtype: Data type for computation. Default: torch.bfloat16
    
    Shape:
        - Input: (batch, seq_len, hidden_size) - Reasoning state
        - q_halt: (batch, seq_len, 1) - Q-values for halting
        - q_continue: (batch, seq_len, 1) - Q-values for continuing
    
    Example:
        >>> q_head = QHaltingHeadTransformer(hidden_size=256)
        >>> z = torch.randn(8, 81, 256)  # Hidden states
        >>> q_halt, q_continue = q_head(z)
        >>> # Decision: aggregate over sequence
        >>> should_halt = q_halt.mean(dim=1) > q_continue.mean(dim=1)
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        # Separate Q-heads matching Sapient's q_halt, q_continue
        self.q_halt = nn.Linear(hidden_size, 1)
        self.q_continue = nn.Linear(hidden_size, 1)
        
        # Small initialization for stable training
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize with small weights for stable Q-learning."""
        nn.init.xavier_uniform_(self.q_halt.weight, gain=0.1)
        nn.init.zeros_(self.q_halt.bias)
        nn.init.xavier_uniform_(self.q_continue.weight, gain=0.1)
        nn.init.zeros_(self.q_continue.bias)
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values for halt and continue actions.
        
        Args:
            z: Reasoning hidden state of shape (batch, seq_len, hidden_size).
        
        Returns:
            Tuple of (q_halt, q_continue):
                - q_halt: (batch, seq_len, 1) Q-values for halting
                - q_continue: (batch, seq_len, 1) Q-values for continuing
        """
        return self.q_halt(z), self.q_continue(z)
    
    def get_aggregated_q_values(
        self,
        z: torch.Tensor,
        aggregation: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence-aggregated Q-values for global halting decision.
        
        Args:
            z: Hidden state of shape (batch, seq_len, hidden_size).
            aggregation: How to aggregate over sequence ('mean' or 'max').
        
        Returns:
            Tuple of (q_halt, q_continue) each of shape (batch,).
        """
        q_halt, q_continue = self.forward(z)
        
        if aggregation == "mean":
            return q_halt.mean(dim=(1, 2)), q_continue.mean(dim=(1, 2))
        elif aggregation == "max":
            return q_halt.max(dim=1)[0].squeeze(-1), q_continue.max(dim=1)[0].squeeze(-1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def should_halt(
        self,
        z: torch.Tensor,
        cycle: int,
        min_cycles: int = 2,
        aggregation: str = "mean",
        temperature: float = 1.0,
        training: bool = False,
    ) -> Tuple[bool, float]:
        """
        Make halting decision based on Q-values.
        
        Args:
            z: Hidden state of shape (batch, seq_len, hidden_size).
            cycle: Current iteration cycle (0-indexed).
            min_cycles: Minimum cycles before halting allowed.
            aggregation: How to aggregate Q-values over sequence.
            temperature: Softmax temperature for exploration.
            training: Whether in training mode (enables exploration).
        
        Returns:
            Tuple of (should_halt, halt_probability).
        """
        # Enforce minimum cycles
        if cycle < min_cycles:
            return False, 0.0
        
        q_halt, q_continue = self.get_aggregated_q_values(z, aggregation)
        
        # Stack for softmax: (batch, 2)
        q_values = torch.stack([q_halt, q_continue], dim=-1)
        probs = F.softmax(q_values / temperature, dim=-1)
        halt_prob = probs[:, 0].mean().item()
        
        if training:
            # Softmax exploration during training
            action = torch.multinomial(probs.mean(dim=0, keepdim=True), 1).item()
            should_halt = (action == 0)
        else:
            # Greedy during inference
            should_halt = (q_halt > q_continue).all().item()
        
        return should_halt, halt_prob
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}"