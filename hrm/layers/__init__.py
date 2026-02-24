"""
HRM Neural Network Layers

Active layers for the Simplified HRM (L-Module Only):
    - RMSNorm: Root Mean Square Layer Normalisation
    - InputEmbedding: Token + puzzle-type embedding for Simplified HRM
    - OutputHead: Puzzle-specific LM heads for Simplified HRM
    - transformer: SwiGLU, RoPE, Attention, TransformerBlock, ReasoningModule

Legacy layers (InputNetwork, WorkerModule, PlannerModule, OutputNetwork) have
been moved to hrm/prototype/models/ as they are only used by the archived
Unified HRM, not by the active SimplifiedHRM training/inference pipeline.
"""

# Simplified HRM layers (L-Module Only)
from hrm.layers.input_simplified import InputEmbedding
from hrm.layers.norm import RMSNorm, RMSNormWithBias, create_norm_layer, rms_norm
from hrm.layers.output_simplified import OutputHead

# Transformer components
from hrm.layers.transformer import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    ReasoningModule,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    trunc_normal_init_,
)

__all__ = [
    # Normalisation
    "RMSNorm",
    "RMSNormWithBias",
    "create_norm_layer",
    "rms_norm",
    # Simplified HRM layers (L-Module Only)
    "InputEmbedding",
    "OutputHead",
    # Transformer Building Blocks
    "SwiGLU",
    "RotaryEmbedding",
    "Attention",
    "TransformerBlock",
    "ReasoningModule",
    "CastedLinear",
    "CastedEmbedding",
    "trunc_normal_init_",
]
