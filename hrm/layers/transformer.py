# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Transformer Components for HRM (Sapient-style Architecture)

This module provides transformer building blocks matching Sapient's HRM:
    - SwiGLU: Gated MLP with SiLU activation
    - RotaryEmbedding: Rotary Position Embeddings (RoPE)
    - Attention: Multi-head self-attention with FlashAttention support
    - TransformerBlock: Combined attention + SwiGLU with post-norm

These replace the MLP-only Worker/Planner modules to match Sapient's
architecture which uses transformer blocks for both H-level and L-level.

Reference:
    - Sapient Inc HRM: https://github.com/sapientinc/HRM
    - SwiGLU: "GLU Variants Improve Transformer" (Shazeer, 2020)
    - RoPE: "RoFormer" (Su et al., 2021)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrm.layers.norm import rms_norm

# Type alias for cosine/sine cache
CosSin = tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a: int, b: int) -> int:
    """Find smallest multiple of b >= a (ceiling division * b)."""
    return (-(a // -b)) * b


def trunc_normal_init_(
    tensor: torch.Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """
    Truncated normal initialisation (JAX-style).

    This matches Sapient's initialisation which follows JAX's truncated
    normal, not PyTorch's (which has incorrect std deviation).

    Args:
        tensor: Tensor to initialise in-place.
        std: Standard deviation.
        lower: Lower truncation bound (in standard deviations).
        upper: Upper truncation bound (in standard deviations).

    Returns:
        The initialised tensor.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_a = c * math.exp(-0.5 * lower**2)
            pdf_b = c * math.exp(-0.5 * upper**2)

            # Variance correction for truncation
            var_correction = 1 + (lower * pdf_a - upper * pdf_b) / z
            adjusted_std = std / math.sqrt(var_correction)

            # Sample from truncated normal
            # Note: erfinv expects input in (-1, 1), uniform should be in (a, b)
            tensor.uniform_(a, b).erfinv_().mul_(sqrt2 * adjusted_std)

    return tensor


class CastedLinear(nn.Module):
    """
    Linear layer with automatic dtype casting (Sapient-style).

    Uses truncated normal initialisation and casts weights to input dtype
    during forward pass for mixed-precision training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()

        # Truncated LeCun normal init: std = 1/sqrt(fan_in)
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)), std=1.0 / math.sqrt(in_features)
            )
        )

        self.bias_param = None
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(x.dtype)
        bias = self.bias_param.to(x.dtype) if self.bias_param is not None else None
        return F.linear(x, weight, bias)


class CastedEmbedding(nn.Module):
    """
    Embedding layer with automatic dtype casting (Sapient-style).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.cast_to = cast_to

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.embedding_weight.to(self.cast_to))


class SwiGLU(nn.Module):
    """
    SwiGLU MLP.

    Uses SiLU (Swish) gating instead of GELU. The architecture:
        gate, up = linear(x).chunk(2)
        out = down(silu(gate) * up)

    This matches Sapient's implementation with expansion factor and
    intermediate dimension rounded to multiple of 256.

    Args:
        hidden_size: Input and output dimension.
        expansion: Expansion factor for intermediate dimension. Default: 4.0

    Example:
        >>> mlp = SwiGLU(hidden_size=256, expansion=4.0)
        >>> x = torch.randn(8, 128, 256)
        >>> out = mlp(x)  # (8, 128, 256)
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float = 4.0,
    ):
        super().__init__()

        # Intermediate dimension: 2/3 * expansion * hidden, rounded to 256
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        # Gate and up projection combined (split in forward)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim)
        cos: Cosine cache of shape (max_seq_len, head_dim)
        sin: Sine cache of shape (max_seq_len, head_dim)

    Returns:
        Tuple of (q_embed, k_embed) with RoPE applied.
    """
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    # Get actual sequence length and slice cos/sin
    seq_len = q.shape[1]
    cos = cos[:seq_len]  # (seq_len, head_dim)
    sin = sin[:seq_len]  # (seq_len, head_dim)

    # cos, sin: [seq_len, head_dim] -> unsqueeze for heads
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer attention.

    Pre-computes and caches sin/cos values for efficient forward pass.

    Args:
        dim: Head dimension (typically hidden_size // num_heads).
        max_position_embeddings: Maximum sequence length to cache.
        base: Base for frequency computation. Default: 10000.0

    Example:
        >>> rope = RotaryEmbedding(dim=64, max_position_embeddings=1024)
        >>> cos, sin = rope()
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        # Position indices
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)

        # Outer product: positions x frequencies
        freqs = torch.outer(t, inv_freq)

        # Concatenate for full dimension (duplicate for rotation)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        """Return cached (cos, sin) embeddings."""
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """
    Multi-head self-attention with RoPE support.

    Supports FlashAttention when available, falls back to standard attention.
    Uses fused QKV projection for efficiency.

    Args:
        hidden_size: Model hidden dimension.
        head_dim: Dimension per attention head.
        num_heads: Number of attention heads.
        num_key_value_heads: Number of key/value heads (for GQA). Default: same as num_heads.
        causal: Whether to use causal masking. Default: False.

    Example:
        >>> attn = Attention(hidden_size=256, head_dim=64, num_heads=4)
        >>> rope = RotaryEmbedding(dim=64, max_position_embeddings=128)
        >>> cos_sin = rope()
        >>> x = torch.randn(8, 128, 256)
        >>> out = attn(cos_sin, x)
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int | None = None,
        causal: bool = False,
    ):
        super().__init__()

        if num_key_value_heads is None:
            num_key_value_heads = num_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        # Fused QKV projection
        qkv_dim = (num_heads + 2 * num_key_value_heads) * head_dim
        self.qkv_proj = CastedLinear(hidden_size, qkv_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

        # Check for FlashAttention
        self._has_flash_attn = False
        try:
            from flash_attn import flash_attn_func

            self._flash_attn_func = flash_attn_func
            self._has_flash_attn = True
        except ImportError:
            pass

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback standard attention when FlashAttention unavailable."""
        # q, k, v: (batch, seq, heads, head_dim)
        _batch, seq_len, _num_heads, head_dim = q.shape

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back: (batch, seq, heads, head_dim)
        return attn_output.transpose(1, 2)

    def forward(
        self,
        cos_sin: CosSin | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            cos_sin: Tuple of (cos, sin) from RotaryEmbedding, or None to skip RoPE.
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)

        # Split into heads: (batch, seq, num_heads + 2*kv_heads, head_dim)
        qkv = qkv.view(
            batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )

        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Attention
        if self._has_flash_attn:
            attn_output = self._flash_attn_func(q=query, k=key, v=value, causal=self.causal)
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
        else:
            attn_output = self._standard_attention(query, key, value)

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    """
    Transformer block with post-norm (Sapient-style).

    Architecture:
        h = rms_norm(h + self_attn(h))
        h = rms_norm(h + mlp(h))

    This uses post-normalisation (norm after residual) which differs
    from standard pre-norm transformers but matches Sapient's HRM.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        expansion: MLP expansion factor. Default: 4.0
        rms_norm_eps: Epsilon for RMS normalisation. Default: 1e-5
        causal: Whether attention is causal. Default: False

    Example:
        >>> block = TransformerBlock(hidden_size=256, num_heads=4)
        >>> rope = RotaryEmbedding(dim=64, max_position_embeddings=128)
        >>> cos_sin = rope()
        >>> x = torch.randn(8, 128, 256)
        >>> out = block(cos_sin, x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float = 4.0,
        rms_norm_eps: float = 1e-5,
        causal: bool = False,
    ):
        super().__init__()

        head_dim = hidden_size // num_heads

        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=causal,
        )

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=expansion,
        )

        self.norm_eps = rms_norm_eps

    def forward(
        self,
        cos_sin: CosSin | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with post-normalisation.

        Args:
            cos_sin: RoPE embeddings or None.
            hidden_states: Input of shape (batch, seq_len, hidden_size).

        Returns:
            Output of shape (batch, seq_len, hidden_size).
        """
        # Self-attention with post-norm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin, hidden_states),
            variance_epsilon=self.norm_eps,
        )

        # MLP with post-norm
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )

        return hidden_states


class ReasoningModule(nn.Module):
    """
    Reasoning module (H-level or L-level) with input injection.

    This matches Sapient's HierarchicalReasoningModel_ACTV1ReasoningModule.
    It applies input injection (additive) before passing through transformer
    blocks.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        expansion: MLP expansion factor. Default: 4.0
        rms_norm_eps: Epsilon for RMS normalisation. Default: 1e-5
        causal: Whether attention is causal. Default: False

    Example:
        >>> h_level = ReasoningModule(hidden_size=256, num_heads=4, num_layers=4)
        >>> rope = RotaryEmbedding(dim=64, max_position_embeddings=128)
        >>> cos_sin = rope()
        >>> h = torch.randn(8, 128, 256)  # Current state
        >>> injection = torch.randn(8, 128, 256)  # Input to inject
        >>> out = h_level(h, injection, cos_sin=cos_sin)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        expansion: float = 4.0,
        rms_norm_eps: float = 1e-5,
        causal: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    expansion=expansion,
                    rms_norm_eps=rms_norm_eps,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: CosSin | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with input injection.

        Args:
            hidden_states: Current state of shape (batch, seq_len, hidden_size).
            input_injection: Input to inject (added to hidden_states).
            cos_sin: RoPE embeddings or None.

        Returns:
            Updated hidden states of shape (batch, seq_len, hidden_size).
        """
        # Input injection (additive, like Sapient)
        hidden_states = hidden_states + input_injection

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(cos_sin, hidden_states)

        return hidden_states
