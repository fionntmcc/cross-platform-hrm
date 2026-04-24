# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Cross-Platform Hierarchical Reasoning Model (HRM)

A platform-agnostic implementation of the Hierarchical Reasoning Model
for constraint-satisfaction problems, specifically Sudoku puzzle solving.

Active model: SimplifiedHRM (L-Module Only, Ge et al. 2025)
    - hrm.model_simplified — SimplifiedHRM, SimplifiedHRMConfig, PuzzleType
    - scripts/train_simplified.py — training
    - scripts/run_simplified.py  — inference & evaluation
    - scripts/demo_simplified.py — interactive demo
    - scripts/visualise_simplified.py — step visualisation

Archived models (hrm/prototype/):
    - hrm/prototype/models/model.py          — Original HRM_4x4 (H+L, MLP)
    - hrm/prototype/models/model_simple.py   — Single-cell HRM variant
    - hrm/prototype/models/model_unified.py  — UnifiedHRM with ACT/Q-learning
    - hrm/prototype/models/model_transformer.py — Transformer HRM
"""

__version__ = "0.1.0"
__author__ = "Fionn McCarthy, Kyrylo Kozlovskyi"

# Simplified HRM — L-Module Only (Ge et al. 2025)  [ACTIVE]
from hrm.model_simplified import (
    PUZZLE_DEFAULTS,
    LModuleOnlyConfig,
    # Backward-compatible aliases
    LModuleOnlyHRM,
    PuzzleType,
    SimplifiedHRM,
    SimplifiedHRMConfig,
    create_lmodule_only_hrm,
    create_simplified_hrm,
    create_small_lmodule_hrm,
    create_small_simplified_hrm,
)

# Simplified PuzzleType alias (kept for scripts that import SimplifiedPuzzleType)
SimplifiedPuzzleType = PuzzleType

# Active Layers
from hrm.layers import (  # noqa: E402
    Attention,
    # Simplified HRM layers (L-Module Only)
    InputEmbedding,
    OutputHead,
    ReasoningModule,
    # Normalisation
    RMSNorm,
    RMSNormWithBias,
    RotaryEmbedding,
    # Transformer Building Blocks
    SwiGLU,
    TransformerBlock,
    create_norm_layer,
    rms_norm,
)

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    # Simplified HRM — L-Module Only (Ge et al. 2025)  [ACTIVE]
    "SimplifiedHRM",
    "SimplifiedHRMConfig",
    "PuzzleType",
    "SimplifiedPuzzleType",
    "PUZZLE_DEFAULTS",
    "create_simplified_hrm",
    "create_small_simplified_hrm",
    # Backward-compatible aliases
    "LModuleOnlyHRM",
    "LModuleOnlyConfig",
    "create_lmodule_only_hrm",
    "create_small_lmodule_hrm",
    # Normalisation
    "RMSNorm",
    "RMSNormWithBias",
    "create_norm_layer",
    "rms_norm",
    # Simplified HRM layers
    "InputEmbedding",
    "OutputHead",
    # Transformer Building Blocks
    "SwiGLU",
    "RotaryEmbedding",
    "Attention",
    "TransformerBlock",
    "ReasoningModule",
]
