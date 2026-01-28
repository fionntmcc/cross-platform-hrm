"""
HRM Neural Network Layers

This module contains all the building blocks for the Hierarchical Reasoning Model:
    - RMSNorm: Root Mean Square Layer Normalisation (Issue #1)
    - InputNetwork: Puzzle embedding network f_I (Issue #2)
    - WorkerModule: Low-level refinement module f_L (Issue #3)
    - PlannerModule: High-level planning module f_H (Issue #4)
    - OutputNetwork: Action decoder f_O (Issue #5)
"""

from hrm.layers.norm import RMSNorm, RMSNormWithBias, create_norm_layer
from hrm.layers.input_network import InputNetwork, create_input_network
from hrm.layers.worker import WorkerModule, WorkerModuleWithGating
from hrm.layers.planner import PlannerModule, create_planner_module
from hrm.layers.output_network import OutputNetwork, create_output_network

__all__ = [
    # Issue #1 - Normalisation
    "RMSNorm",
    "RMSNormWithBias",
    "create_norm_layer",
    # Issue #2 - Input Embedding
    "InputNetwork",
    "create_input_network",
    # Issue #3 - Worker (Low-level)
    "WorkerModule",
    "WorkerModuleWithGating",
    # Issue #4 - Planner (High-level)
    "PlannerModule",
    "create_planner_module",
    # Issue #5 - Output Decoder
    "OutputNetwork",
    "create_output_network",
]
