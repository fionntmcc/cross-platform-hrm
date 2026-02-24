"""
Unit Tests for SimplifiedHRM (L-Module Only, Ge et al. 2025)

Tests verifying acceptance criteria:
- Model initialises with correct architecture
- Forward pass returns correct output dict keys
- Output shapes for 4x4 and 9x9 Sudoku
- Loss computed when targets provided
- Prediction feedback (self-conditioning) works
- predict(), solve_sudoku_4x4(), solve_sudoku_9x9() helpers
- Return intermediates option
- Factory functions create valid models
- num_parameters property
- One-step gradient: intermediate steps detached, last step has gradient

Run: pytest test/hrm_test/test_simplified_hrm.py -v
"""

import pytest
import torch

from hrm.layers.input_simplified import InputEmbedding
from hrm.layers.output_simplified import OutputHead
from hrm.layers.transformer import ReasoningModule, RotaryEmbedding
from hrm.model_simplified import (
    LModuleOnlyConfig,
    LModuleOnlyHRM,
    PuzzleType,
    SimplifiedHRM,
    SimplifiedHRMConfig,
    create_simplified_hrm,
    create_small_simplified_hrm,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Minimal config for fast tests."""
    return SimplifiedHRMConfig(
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        num_reasoning_steps=2,
        vocab_size=10,
        dropout=0.0,
        use_prediction_feedback=True,
    )


@pytest.fixture
def model(config):
    return SimplifiedHRM(config)


@pytest.fixture
def puzzle_4x4():
    """Batch of 4x4 Sudoku inputs (flat, batch=4)."""
    return torch.randint(0, 5, (4, 16))


@pytest.fixture
def puzzle_9x9():
    """Batch of 9x9 Sudoku inputs (flat, batch=4)."""
    return torch.randint(0, 10, (4, 81))


@pytest.fixture
def target_4x4():
    return torch.randint(1, 5, (4, 16))


@pytest.fixture
def target_9x9():
    return torch.randint(1, 10, (4, 81))


# =============================================================================
# Configuration
# =============================================================================


class TestConfig:

    def test_default_config(self):
        cfg = SimplifiedHRMConfig()
        assert cfg.hidden_size == 256
        assert cfg.num_heads == 4
        assert cfg.num_layers == 8
        assert cfg.num_reasoning_steps == 16
        assert cfg.use_prediction_feedback is True

    def test_custom_config(self):
        cfg = SimplifiedHRMConfig(hidden_size=128, num_layers=4, num_reasoning_steps=8)
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 4
        assert cfg.num_reasoning_steps == 8

    def test_backward_compat_alias(self):
        assert LModuleOnlyConfig is SimplifiedHRMConfig


# =============================================================================
# Model initialisation
# =============================================================================


class TestModelInitialisation:

    def test_has_input_net(self, model):
        assert hasattr(model, "input_net")
        assert isinstance(model.input_net, InputEmbedding)

    def test_has_reasoning_module(self, model):
        assert hasattr(model, "reasoning")
        assert isinstance(model.reasoning, ReasoningModule)

    def test_has_output_head(self, model):
        assert hasattr(model, "output_head")
        assert isinstance(model.output_head, OutputHead)

    def test_has_rotary_emb(self, model):
        assert hasattr(model, "rotary_emb")
        assert isinstance(model.rotary_emb, RotaryEmbedding)

    def test_has_z_L_init_buffer(self, model, config):
        assert hasattr(model, "z_L_init")
        assert model.z_L_init.shape == (1, 1, config.hidden_size)

    def test_no_planner(self, model):
        assert not hasattr(model, "planner")
        assert not hasattr(model, "worker")

    def test_backward_compat_alias(self):
        assert LModuleOnlyHRM is SimplifiedHRM

    def test_num_layers_in_reasoning(self, model, config):
        assert len(model.reasoning.layers) == config.num_layers


# =============================================================================
# Forward pass — output dict
# =============================================================================


class TestForwardOutputDict:

    def test_required_keys_present(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert "logits" in out
        assert "predictions" in out
        assert "reasoning_steps_used" in out
        assert "all_step_logits" in out

    def test_loss_absent_without_targets(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert "loss" not in out

    def test_loss_present_with_targets(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        assert "loss" in out
        assert "lm_loss" in out

    def test_reasoning_steps_used(self, model, config, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert out["reasoning_steps_used"] == config.num_reasoning_steps

    def test_step_override(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, num_reasoning_steps=1)
        assert out["reasoning_steps_used"] == 1

    def test_all_step_logits_length(self, model, config, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert len(out["all_step_logits"]) == config.num_reasoning_steps


# =============================================================================
# Output shapes
# =============================================================================


class TestOutputShapes:

    def test_logits_shape_4x4(self, model, puzzle_4x4):
        out = model(puzzle_4x4, PuzzleType.SUDOKU_4X4)
        assert out["logits"].shape == (4, 16, 5)

    def test_predictions_shape_4x4(self, model, puzzle_4x4):
        out = model(puzzle_4x4, PuzzleType.SUDOKU_4X4)
        assert out["predictions"].shape == (4, 16)

    def test_logits_shape_9x9(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert out["logits"].shape == (4, 81, 10)

    def test_predictions_shape_9x9(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert out["predictions"].shape == (4, 81)

    def test_predictions_are_valid_tokens_4x4(self, model, puzzle_4x4):
        out = model(puzzle_4x4, PuzzleType.SUDOKU_4X4)
        preds = out["predictions"]
        assert preds.min() >= 0
        assert preds.max() <= 4

    def test_predictions_are_valid_tokens_9x9(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        preds = out["predictions"]
        assert preds.min() >= 0
        assert preds.max() <= 9


# =============================================================================
# Loss computation
# =============================================================================


class TestLossComputation:

    def test_loss_is_scalar(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        assert out["loss"].ndim == 0

    def test_loss_is_positive(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        assert out["loss"].item() > 0

    def test_loss_has_gradient(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        assert out["loss"].requires_grad

    def test_backward_runs(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        out["loss"].backward()
        # Check a parameter has gradient
        assert model.output_head.heads["sudoku_9x9"].weight.grad is not None

    def test_lm_loss_equals_loss(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        assert torch.allclose(out["loss"], out["lm_loss"])

    def test_4x4_loss(self, model, puzzle_4x4, target_4x4):
        out = model(puzzle_4x4, PuzzleType.SUDOKU_4X4, targets=target_4x4)
        assert out["loss"].item() > 0


# =============================================================================
# One-step gradient behaviour
# =============================================================================


class TestOneStepGradient:

    def test_intermediate_step_logits_detached(self, config, puzzle_9x9, target_9x9):
        """All step logits except the last should be detached."""
        config_multi = SimplifiedHRMConfig(
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            num_reasoning_steps=3,
            dropout=0.0,
        )
        model = SimplifiedHRM(config_multi)
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        # The last step logit should have gradient, intermediate ones detached
        assert out["all_step_logits"][-1].requires_grad
        for step_logits in out["all_step_logits"][:-1]:
            assert not step_logits.requires_grad


# =============================================================================
# Return intermediates
# =============================================================================


class TestReturnIntermediates:

    def test_intermediates_key_absent_by_default(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert "intermediates" not in out

    def test_intermediates_key_present_when_requested(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, return_intermediates=True)
        assert "intermediates" in out

    def test_intermediates_step_predictions(self, model, config, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, return_intermediates=True)
        step_preds = out["intermediates"]["step_predictions"]
        assert len(step_preds) == config.num_reasoning_steps
        for sp in step_preds:
            assert sp.shape == (4, 81)


# =============================================================================
# Convenience helpers
# =============================================================================


class TestHelpers:

    def test_predict_no_grad(self, model, puzzle_9x9):
        preds = model.predict(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert preds.shape == (4, 81)
        assert preds.requires_grad is False

    def test_solve_sudoku_4x4_flat_input(self, model):
        puzzle = torch.randint(0, 5, (2, 16))
        sol = model.solve_sudoku_4x4(puzzle)
        assert sol.shape == (2, 4, 4)

    def test_solve_sudoku_4x4_grid_input(self, model):
        puzzle = torch.randint(0, 5, (2, 4, 4))
        sol = model.solve_sudoku_4x4(puzzle)
        assert sol.shape == (2, 4, 4)

    def test_solve_sudoku_9x9_flat_input(self, model):
        puzzle = torch.randint(0, 10, (2, 81))
        sol = model.solve_sudoku_9x9(puzzle)
        assert sol.shape == (2, 9, 9)

    def test_solve_sudoku_9x9_grid_input(self, model):
        puzzle = torch.randint(0, 10, (2, 9, 9))
        sol = model.solve_sudoku_9x9(puzzle)
        assert sol.shape == (2, 9, 9)

    def test_num_parameters_positive(self, model):
        assert model.num_parameters > 0

    def test_num_parameters_int(self, model):
        assert isinstance(model.num_parameters, int)

    def test_extra_repr(self, model):
        r = model.extra_repr()
        assert "hidden_size" in r
        assert "L-only" in r


# =============================================================================
# Factory functions
# =============================================================================


class TestFactories:

    def test_create_simplified_hrm_default(self):
        model = create_simplified_hrm()
        assert isinstance(model, SimplifiedHRM)
        assert model.config.hidden_size == 256
        assert model.config.num_layers == 8

    def test_create_simplified_hrm_custom(self):
        model = create_simplified_hrm(hidden_size=128, num_layers=4, num_reasoning_steps=4)
        assert model.config.hidden_size == 128
        assert model.config.num_layers == 4
        assert model.config.num_reasoning_steps == 4

    def test_create_small_simplified_hrm(self):
        model = create_small_simplified_hrm()
        assert isinstance(model, SimplifiedHRM)
        assert model.config.hidden_size == 128
        assert model.config.num_layers == 4
        # Should be smaller than default
        default = create_simplified_hrm()
        assert model.num_parameters < default.num_parameters

    def test_factory_models_forward(self):
        model = create_small_simplified_hrm(num_reasoning_steps=1)
        x = torch.randint(0, 5, (2, 16))
        out = model(x, PuzzleType.SUDOKU_4X4)
        assert "logits" in out


# =============================================================================
# No NaN
# =============================================================================


class TestNumericalStability:

    def test_no_nan_forward(self, model, puzzle_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9)
        assert not torch.isnan(out["logits"]).any()

    def test_no_nan_after_backward(self, model, puzzle_9x9, target_9x9):
        out = model(puzzle_9x9, PuzzleType.SUDOKU_9X9, targets=target_9x9)
        out["loss"].backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
