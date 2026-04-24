# Project: Hierarchical Reasoning Model for Puzzle Solving
# Authors: Kyrylo Kozlovskyi (G00425385), Fionn McCarthy (G00414386)
# Supervisor: Dr. John Healy
# Institution: Atlantic Technological University
# Duration: 2025/2026

"""
Unit Tests for Transformer Components (Simplified HRM)

Tests for:
- RotaryEmbedding: cos/sin cache shape and values
- SwiGLU: output shape, gating, gradient flow
- TransformerBlock: output shape, post-norm architecture
- ReasoningModule: output shape, input injection, multi-layer stacking

Run: pytest test/hrm_test/test_transformer.py -v
"""

import pytest
import torch

from hrm.layers.transformer import (
    Attention,
    ReasoningModule,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    trunc_normal_init_,
)

# Shared constants

HIDDEN = 64  # Small hidden dim for fast tests
N_HEADS = 4
HEAD_DIM = HIDDEN // N_HEADS  # 16
SEQ_LEN = 16
BATCH = 4


# RotaryEmbedding


class TestRotaryEmbedding:

    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(dim=HEAD_DIM, max_position_embeddings=128)

    def test_forward_returns_tuple(self, rope):
        result = rope()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cos_sin_shape(self, rope):
        cos, sin = rope()
        assert cos.shape == (128, HEAD_DIM)
        assert sin.shape == (128, HEAD_DIM)

    def test_different_dim(self):
        rope = RotaryEmbedding(dim=32, max_position_embeddings=64)
        cos, _ = rope()
        assert cos.shape == (64, 32)

    def test_cos_values_in_range(self, rope):
        cos, _ = rope()
        assert cos.min() >= -1.0 - 1e-6
        assert cos.max() <= 1.0 + 1e-6

    def test_sin_values_in_range(self, rope):
        _, sin = rope()
        assert sin.min() >= -1.0 - 1e-6
        assert sin.max() <= 1.0 + 1e-6

    def test_no_nan_in_cache(self, rope):
        cos, sin = rope()
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()


# SwiGLU


class TestSwiGLU:

    @pytest.fixture
    def mlp(self):
        return SwiGLU(hidden_size=HIDDEN, expansion=4.0)

    def test_output_shape_3d(self, mlp):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = mlp(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_output_shape_2d(self, mlp):
        x = torch.randn(BATCH, HIDDEN)
        out = mlp(x)
        assert out.shape == (BATCH, HIDDEN)

    def test_no_nan(self, mlp):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = mlp(x)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, mlp):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    def test_has_gate_up_proj(self, mlp):
        assert hasattr(mlp, "gate_up_proj")

    def test_has_down_proj(self, mlp):
        assert hasattr(mlp, "down_proj")


# Attention


class TestAttention:

    @pytest.fixture
    def attn(self):
        return Attention(
            hidden_size=HIDDEN,
            head_dim=HEAD_DIM,
            num_heads=N_HEADS,
            causal=False,
        )

    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(dim=HEAD_DIM, max_position_embeddings=128)

    def test_output_shape(self, attn, rope):
        cos_sin = rope()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = attn(cos_sin, x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_no_rope(self, attn):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = attn(None, x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_no_nan(self, attn, rope):
        cos_sin = rope()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = attn(cos_sin, x)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, attn, rope):
        cos_sin = rope()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
        out = attn(cos_sin, x)
        out.sum().backward()
        assert x.grad is not None


# TransformerBlock


class TestTransformerBlock:

    @pytest.fixture
    def block(self):
        return TransformerBlock(
            hidden_size=HIDDEN,
            num_heads=N_HEADS,
            expansion=4.0,
            rms_norm_eps=1e-5,
            causal=False,
        )

    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(dim=HEAD_DIM, max_position_embeddings=128)

    def test_output_shape(self, block, rope):
        cos_sin = rope()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = block(cos_sin, x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_no_rope(self, block):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = block(None, x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_no_nan(self, block, rope):
        cos_sin = rope()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = block(cos_sin, x)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, block, rope):
        cos_sin = rope()
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
        out = block(cos_sin, x)
        out.sum().backward()
        assert x.grad is not None

    def test_has_attention(self, block):
        assert hasattr(block, "self_attn")
        assert isinstance(block.self_attn, Attention)

    def test_has_mlp(self, block):
        assert hasattr(block, "mlp")
        assert isinstance(block.mlp, SwiGLU)


# ReasoningModule


class TestReasoningModule:

    @pytest.fixture
    def module(self):
        return ReasoningModule(
            hidden_size=HIDDEN,
            num_heads=N_HEADS,
            num_layers=2,
            expansion=4.0,
            rms_norm_eps=1e-5,
            causal=False,
        )

    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(dim=HEAD_DIM, max_position_embeddings=128)

    def test_output_shape(self, module, rope):
        cos_sin = rope()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        injection = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = module(h, injection, cos_sin=cos_sin)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_without_cos_sin(self, module):
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        injection = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = module(h, injection, cos_sin=None)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_input_injection_applied(self, module):
        """Injection changes the output compared to zero injection."""
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        injection = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        zero_injection = torch.zeros_like(injection)
        out_with = module(h, injection, cos_sin=None)
        out_without = module(h, zero_injection, cos_sin=None)
        assert not torch.allclose(out_with, out_without)

    def test_no_nan(self, module, rope):
        cos_sin = rope()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        injection = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = module(h, injection, cos_sin=cos_sin)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradients_flow(self, module, rope):
        cos_sin = rope()
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
        injection = torch.randn(BATCH, SEQ_LEN, HIDDEN, requires_grad=True)
        out = module(h, injection, cos_sin=cos_sin)
        out.sum().backward()
        assert h.grad is not None
        assert injection.grad is not None

    def test_num_layers(self, module):
        assert len(module.layers) == 2

    def test_layers_are_transformer_blocks(self, module):
        for layer in module.layers:
            assert isinstance(layer, TransformerBlock)

    def test_single_layer(self):
        m = ReasoningModule(hidden_size=HIDDEN, num_heads=N_HEADS, num_layers=1)
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        inj = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        out = m(h, inj)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN)

    def test_multiple_iterations_stable(self, module):
        """Repeated application does not cause numerical instability."""
        h = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        inj = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        for _ in range(10):
            h = module(h.detach(), inj, cos_sin=None)
        assert not torch.isnan(h).any()


# trunc_normal_init_


class TestTruncNormalInit:

    def test_initialises_tensor(self):
        t = torch.empty(64, 64)
        result = trunc_normal_init_(t, std=0.02)
        assert result is t
        assert not torch.isnan(t).any()

    def test_zero_std(self):
        t = torch.empty(16, 16)
        trunc_normal_init_(t, std=0.0)
        assert torch.all(t == 0.0)

    def test_roughly_normal(self):
        """Mean should be near 0, std near specified value."""
        t = torch.empty(10000)
        trunc_normal_init_(t, std=0.1)
        assert abs(t.mean().item()) < 0.01
