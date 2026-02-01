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
    
    