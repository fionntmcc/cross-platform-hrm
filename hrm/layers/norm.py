# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Normalisation Layers for HRM

Issue #1: Implement RMSNorm and Base Normalisation Layers

This module provides Root Mean Square Layer Normalisation (RMSNorm),
which is used throughout the HRM architecture for stable training.

RMSNorm is preferred over LayerNorm in modern architectures because:
1. Simpler computation (no mean subtraction)
2. Comparable or better performance
3. Lower computational overhead

Reference:
    - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    - Used in LLaMA, Mistral, and HRM architectures
"""

import torch
import torch.nn as nn


def rms_norm(
    hidden_states: torch.Tensor,
    variance_epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Functional RMS normalisation (Sapient-style, no learnable parameters).

    This is the preferred normalisation for transformer blocks in HRM,
    matching Sapient's implementation which uses post-norm without
    learnable scale parameters.

    Formula:
        rms_norm(x) = x / sqrt(mean(x^2) + eps)

    Args:
        hidden_states: Input tensor of shape (*, dim).
        variance_epsilon: Small constant for numerical stability.

    Returns:
        Normalised tensor of the same shape as input.

    Example:
        >>> x = torch.randn(32, 128, 256)  # (batch, seq, hidden)
        >>> normed = rms_norm(x, variance_epsilon=1e-5)
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return hidden_states.to(input_dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Normalises inputs by their root mean square, then applies a learned
    scale parameter. Unlike LayerNorm, RMSNorm does not center the inputs
    (no mean subtraction), which simplifies computation while maintaining
    effectiveness.

    Formula:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Args:
        dim: The dimension of the input features to normalise over.
        eps: Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: (*, dim) where * means any number of leading dimensions
        - Output: (*, dim) same shape as input

    Attributes:
        weight: Learnable scale parameter of shape (dim,), initialised to ones.
        eps: Numerical stability constant.

    Example:
        >>> norm = RMSNorm(dim=64)
        >>> x = torch.randn(32, 64)  # (batch, features)
        >>> output = norm(x)
        >>> output.shape
        torch.Size([32, 64])
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialise RMSNorm layer.

        Args:
            dim: Feature dimension to normalise over (last dimension).
            eps: Small constant added to denominator for numerical stability.
        """
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = eps

        # Learnable scale parameter, initialised to ones
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalisation to input tensor.

        Args:
            x: Input tensor of shape (*, dim) where * is any number of
               leading batch dimensions.

        Returns:
            Normalised tensor of the same shape as input.
        """
        # Compute reciprocal of RMS: 1/sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalise and scale
        return x * rms * self.weight

    def extra_repr(self) -> str:
        """Return a string representation of module parameters."""
        return f"dim={self.dim}, eps={self.eps}"


class RMSNormWithBias(RMSNorm):
    """
    RMSNorm variant with optional learnable bias.

    Formula:
        RMSNormWithBias(x) = x / sqrt(mean(x^2) + eps) * weight + bias

    Args:
        dim: The dimension of the input features.
        eps: Small constant for numerical stability. Default: 1e-6
        bias: Whether to include learnable bias. Default: True
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True):
        super().__init__(dim=dim, eps=eps)

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation with optional bias."""
        output = super().forward(x)

        if self.use_bias and self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.use_bias}"


def create_norm_layer(norm_type: str, dim: int, eps: float = 1e-6, **kwargs) -> nn.Module:
    """
    Factory function to create normalisation layers.

    Args:
        norm_type: Type of normalisation. One of:
            - 'rmsnorm': RMSNorm (default for HRM)
            - 'rmsnorm_bias': RMSNorm with bias
            - 'layernorm': Standard LayerNorm
            - 'none': Identity (no normalisation)
        dim: Feature dimension.
        eps: Numerical stability constant.
        **kwargs: Additional arguments passed to the layer.

    Returns:
        Normalisation layer module.
    """
    norm_type = norm_type.lower()

    if norm_type == "rmsnorm":
        return RMSNorm(dim=dim, eps=eps)
    elif norm_type == "rmsnorm_bias":
        return RMSNormWithBias(dim=dim, eps=eps, **kwargs)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, **kwargs)
    elif norm_type in ("none", "identity"):
        return nn.Identity()
    else:
        raise ValueError(
            f"Unknown norm_type: {norm_type}. "
            f"Expected one of: 'rmsnorm', 'rmsnorm_bias', 'layernorm', 'none'"
        )
