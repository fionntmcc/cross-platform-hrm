"""
HRM Neural Network Layers

This module contains all the building blocks for the Hierarchical Reasoning Model:
    - RMSNorm: Root Mean Square Layer Normalisation (Issue #1)
    - InputNetwork: Puzzle embedding network f_I (Issue #2)
    - WorkerModule: Low-level refinement module f_L (Issue #3)
    - PlannerModule: High-level planning module f_H (Issue #4)
    - OutputNetwork: Action decoder f_O (Issue #5)

Transformer variants (Sapient-compatible):
    - transformer: SwiGLU, RoPE, Attention, TransformerBlock, ReasoningModule
    - InputNetworkTransformer: Sequence-based token embedding
    - WorkerTransformer: Transformer-based L-level module
    - PlannerTransformer: Transformer-based H-level module
    - OutputNetworkTransformer: LM head for vocab logits

Simplified HRM (L-Module Only):
    - InputEmbedding: Token + puzzle-type embedding for Simplified HRM
    - OutputHead: Puzzle-specific LM heads for Simplified HRM
"""

from hrm.layers.norm import RMSNorm, RMSNormWithBias, create_norm_layer, rms_norm
from hrm.layers.input_network import InputNetwork, create_input_network, InputNetworkTransformer
from hrm.layers.worker import WorkerModule, WorkerModuleWithGating, WorkerTransformer
from hrm.layers.planner import PlannerModule, create_planner_module, PlannerTransformer
from hrm.layers.output_network import OutputNetwork, create_output_network, OutputNetworkTransformer

# Simplified HRM layers (L-Module Only)
from hrm.layers.input_simplified import InputEmbedding
from hrm.layers.output_simplified import OutputHead

# Transformer components
from hrm.layers.transformer import (
    SwiGLU,
    RotaryEmbedding,
    Attention,
    TransformerBlock,
    ReasoningModule,
    CastedLinear,
    CastedEmbedding,
    trunc_normal_init_,
)

__all__ = [
    # Issue #1 - Normalisation
    "RMSNorm",
    "RMSNormWithBias",
    "create_norm_layer",
    "rms_norm",  # Functional RMSNorm
    # Issue #2 - Input Embedding
    "InputNetwork",
    "create_input_network",
    "InputNetworkTransformer",
    # Issue #3 - Worker (Low-level)
    "WorkerModule",
    "WorkerModuleWithGating",
    "WorkerTransformer",
    # Issue #4 - Planner (High-level)
    "PlannerModule",
    "create_planner_module",
    "PlannerTransformer",
    # Issue #5 - Output Decoder
    "OutputNetwork",
    "create_output_network",
    "OutputNetworkTransformer",
    # Simplified HRM (L-Module Only)
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
