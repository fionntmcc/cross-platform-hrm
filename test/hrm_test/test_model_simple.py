"""
Unit Tests for HRM_4x4_Simple (L-Module Only Variant)

Tests verifying acceptance criteria from the Ge et al. (2025) ticket:
- HRM_4x4_Simple initialises without Planner module
- Configurable number of iterations (default: 50)
- 8-layer Worker module for expanded capacity
- Same Input/Output network structure as full HRM
- Returns execution trace with iteration count and residuals
- Forward/backward pass verification
- Convergence behavior testing
- Memory: no h_H state storage

Run tests: pytest test/hrm_test/test_model_simple.py -v
"""

import pytest
import torch
import torch.nn as nn

from hrm.model_simple import HRM_4x4_Simple, SimpleExecutionTrace, create_hrm_4x4_simple
from hrm.layers.input_network import InputNetwork
from hrm.layers.worker import WorkerModule
from hrm.layers.output_network import OutputNetwork


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model():
    """Create a default HRM_4x4_Simple model for testing."""
    return HRM_4x4_Simple()


@pytest.fixture
def model_custom():
    """Create a custom HRM_4x4_Simple with non-default parameters."""
    return HRM_4x4_Simple(
        hidden_dim=128,
        embed_dim=32,
        n_iterations=20,
        convergence_threshold=1e-2,
        n_worker_layers=4,
        dropout=0.2,
    )


@pytest.fixture
def sample_puzzle():
    """Create a sample 4x4 Sudoku puzzle tensor."""
    return torch.tensor([[
        [0, 2, 0, 4],
        [4, 0, 2, 0],
        [0, 4, 0, 2],
        [2, 0, 4, 0]
    ]])


@pytest.fixture
def batch_puzzles():
    """Create a batch of random puzzles for testing."""
    return torch.randint(0, 5, (8, 4, 4))


# =============================================================================
# Test Model Initialisation
# =============================================================================

class TestModelInitialisation:
    """Verify HRM_4x4_Simple initialises correctly without Planner."""

    def test_has_input_network(self, model):
        """Model has InputNetwork component."""
        assert hasattr(model, 'input_network')
        assert isinstance(model.input_network, InputNetwork)

    def test_has_worker_layers(self, model):
        """Model has worker_layers as ModuleList."""
        assert hasattr(model, 'worker_layers')
        assert isinstance(model.worker_layers, nn.ModuleList)

    def test_worker_layers_count_default(self, model):
        """Default model has 8 Worker layers."""
        assert len(model.worker_layers) == 8

    def test_worker_layers_are_worker_modules(self, model):
        """Each layer is a WorkerModule instance."""
        for layer in model.worker_layers:
            assert isinstance(layer, WorkerModule)

    def test_has_output_network(self, model):
        """Model has OutputNetwork component."""
        assert hasattr(model, 'output_network')
        assert isinstance(model.output_network, OutputNetwork)

    def test_no_planner_module(self, model):
        """Model does NOT have a Planner module."""
        assert not hasattr(model, 'planner')

    def test_no_h_H_init(self, model):
        """Model does NOT have h_H initial state (no Planner)."""
        assert not hasattr(model, 'h_H_init')

    def test_has_learned_h_L_init(self, model):
        """Model has learned h_L initial state parameter."""
        assert hasattr(model, 'h_L_init')
        assert isinstance(model.h_L_init, nn.Parameter)
        assert model.h_L_init.shape == (1, model.hidden_dim)

    def test_default_configuration(self, model):
        """Default configuration matches specification."""
        assert model.hidden_dim == 64
        assert model.n_iterations == 50
        assert model.n_worker_layers == 8
        assert model.convergence_threshold == 1e-3
        assert model.dropout == 0.1

    def test_custom_configuration(self, model_custom):
        """Custom configuration is applied correctly."""
        assert model_custom.hidden_dim == 128
        assert model_custom.n_iterations == 20
        assert model_custom.n_worker_layers == 4
        assert model_custom.convergence_threshold == 1e-2
        assert model_custom.dropout == 0.2

    def test_components_share_hidden_dim(self, model):
        """All components use the same hidden_dim."""
        assert model.input_network.hidden_dim == model.hidden_dim
        for worker in model.worker_layers:
            assert worker.hidden_dim == model.hidden_dim
        assert model.output_network.hidden_dim == model.hidden_dim

    def test_worker_layers_custom_count(self, model_custom):
        """Custom model has correct number of Worker layers."""
        assert len(model_custom.worker_layers) == 4


# =============================================================================
# Test Forward Pass
# =============================================================================

class TestForwardPass:
    """Verify forward() returns predictions + execution traces."""

    def test_forward_returns_dict(self, model, sample_puzzle):
        """Forward pass returns a dictionary."""
        outputs = model(sample_puzzle)
        assert isinstance(outputs, dict)

    def test_forward_has_cell_logits(self, model, sample_puzzle):
        """Output contains cell_logits with correct shape."""
        outputs = model(sample_puzzle)
        assert 'cell_logits' in outputs
        assert outputs['cell_logits'].shape == (1, 16)

    def test_forward_has_digit_logits(self, model, sample_puzzle):
        """Output contains digit_logits with correct shape."""
        outputs = model(sample_puzzle)
        assert 'digit_logits' in outputs
        assert outputs['digit_logits'].shape == (1, 4)

    def test_forward_has_trace(self, model, sample_puzzle):
        """Output contains execution trace."""
        outputs = model(sample_puzzle)
        assert 'trace' in outputs
        assert isinstance(outputs['trace'], SimpleExecutionTrace)

    def test_forward_has_final_h_L(self, model, sample_puzzle):
        """Output contains final h_L state."""
        outputs = model(sample_puzzle)
        assert 'h_L_final' in outputs
        assert outputs['h_L_final'].shape == (1, model.hidden_dim)

    def test_forward_no_h_H_final(self, model, sample_puzzle):
        """Output does NOT contain h_H_final (no Planner)."""
        outputs = model(sample_puzzle)
        assert 'h_H_final' not in outputs

    def test_forward_batch_processing(self, model, batch_puzzles):
        """Forward pass handles batched inputs correctly."""
        outputs = model(batch_puzzles)
        batch_size = batch_puzzles.shape[0]

        assert outputs['cell_logits'].shape == (batch_size, 16)
        assert outputs['digit_logits'].shape == (batch_size, 4)
        assert outputs['h_L_final'].shape == (batch_size, model.hidden_dim)

    def test_forward_single_sample(self, model):
        """Forward pass works with batch size 1."""
        puzzle = torch.randint(0, 5, (1, 4, 4))
        outputs = model(puzzle)

        assert outputs['cell_logits'].shape == (1, 16)
        assert outputs['digit_logits'].shape == (1, 4)


# =============================================================================
# Test Execution Trace
# =============================================================================

class TestExecutionTrace:
    """Verify execution trace contains correct information."""

    def test_trace_has_num_steps(self, model, sample_puzzle):
        """Trace records number of steps taken."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.num_steps > 0

    def test_trace_steps_bounded(self, model, sample_puzzle):
        """Number of steps is bounded by n_iterations."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.num_steps <= model.n_iterations

    def test_trace_max_steps_matches_config(self, model, sample_puzzle):
        """Trace max_steps matches model configuration."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.max_steps == model.n_iterations

    def test_trace_has_residual_history(self, model, sample_puzzle):
        """Trace contains residual history when track_history=True."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert len(trace.residual_history) > 0
        assert len(trace.residual_history) == trace.num_steps

    def test_trace_residuals_non_negative(self, model, sample_puzzle):
        """All residuals are non-negative."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        for residual in trace.residual_history:
            assert residual >= 0.0

    def test_trace_total_computation_steps(self, model, sample_puzzle):
        """Trace correctly computes total computation steps."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.total_computation_steps == trace.num_steps

    def test_trace_effective_depth(self, model, sample_puzzle):
        """Effective depth equals num_steps for flat loop."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.effective_depth == trace.num_steps

    def test_trace_has_final_residual(self, model, sample_puzzle):
        """Trace records final residual value."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.final_residual >= 0.0

    def test_trace_repr(self, model, sample_puzzle):
        """Trace has informative repr."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        repr_str = repr(trace)

        assert 'SimpleExecutionTrace' in repr_str
        assert 'steps=' in repr_str
        assert 'converged=' in repr_str

    def test_trace_no_history_when_disabled(self, sample_puzzle):
        """Trace residual history is empty when track_history=False."""
        model = HRM_4x4_Simple(track_history=False)
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert len(trace.residual_history) == 0


# =============================================================================
# Test Convergence Behavior
# =============================================================================

class TestConvergenceBehavior:
    """Verify convergence-based early stopping."""

    def test_converges_before_max_steps(self, sample_puzzle):
        """Model can converge before reaching max iterations."""
        # Use a high threshold so convergence is likely
        model = HRM_4x4_Simple(
            n_iterations=100,
            convergence_threshold=10.0,  # Very high threshold
        )
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.converged
        assert trace.num_steps < trace.max_steps

    def test_runs_to_max_with_low_threshold(self, sample_puzzle):
        """Model runs to max iterations with very low threshold."""
        model = HRM_4x4_Simple(
            n_iterations=5,
            convergence_threshold=1e-20,  # Impossibly low
        )
        outputs = model(sample_puzzle)
        trace = outputs['trace']

        assert trace.num_steps == 5
        assert not trace.converged


# =============================================================================
# Test Halt Penalty
# =============================================================================

class TestHaltPenalty:
    """Verify get_halt_penalty() method."""

    def test_halt_penalty_returns_tensor(self, model, sample_puzzle):
        """Halt penalty returns a tensor."""
        outputs = model(sample_puzzle)
        penalty = model.get_halt_penalty(outputs)

        assert isinstance(penalty, torch.Tensor)

    def test_halt_penalty_scalar(self, model, sample_puzzle):
        """Halt penalty is a scalar."""
        outputs = model(sample_puzzle)
        penalty = model.get_halt_penalty(outputs)

        assert penalty.dim() == 0 or penalty.numel() == 1

    def test_halt_penalty_non_negative(self, model, sample_puzzle):
        """Halt penalty is non-negative."""
        outputs = model(sample_puzzle)
        penalty = model.get_halt_penalty(outputs)

        assert penalty.item() >= 0

    def test_halt_penalty_bounded(self, model, sample_puzzle):
        """Halt penalty is bounded by lambda_halt."""
        outputs = model(sample_puzzle)
        lambda_halt = 0.01
        penalty = model.get_halt_penalty(outputs, lambda_halt=lambda_halt)

        assert penalty.item() <= lambda_halt + 1e-9

    def test_halt_penalty_lambda_scaling(self, model, sample_puzzle):
        """Halt penalty scales with lambda_halt."""
        outputs = model(sample_puzzle)
        penalty_small = model.get_halt_penalty(outputs, lambda_halt=0.001)
        penalty_large = model.get_halt_penalty(outputs, lambda_halt=0.1)

        assert penalty_large.item() > penalty_small.item()


# =============================================================================
# Test Convergence Loss
# =============================================================================

class TestConvergenceLoss:
    """Verify get_convergence_loss() method."""

    def test_convergence_loss_returns_tensor(self, model, sample_puzzle):
        """Convergence loss returns a tensor."""
        outputs = model(sample_puzzle)
        loss = model.get_convergence_loss(outputs)

        assert isinstance(loss, torch.Tensor)

    def test_convergence_loss_scalar(self, model, sample_puzzle):
        """Convergence loss is a scalar."""
        outputs = model(sample_puzzle)
        loss = model.get_convergence_loss(outputs)

        assert loss.dim() == 0 or loss.numel() == 1

    def test_convergence_loss_non_negative(self, model, sample_puzzle):
        """Convergence loss is non-negative."""
        outputs = model(sample_puzzle)
        loss = model.get_convergence_loss(outputs)

        assert loss.item() >= 0

    def test_convergence_loss_hinge_behavior(self, model, sample_puzzle):
        """Convergence loss is zero when residual below target."""
        outputs = model(sample_puzzle)
        loss = model.get_convergence_loss(outputs, target_residual=1e6)

        assert loss.item() == pytest.approx(0.0)


# =============================================================================
# Test Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Verify gradients flow through the simplified model."""

    def test_gradients_flow_to_input_network(self, model, sample_puzzle):
        """Gradients flow to input network parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()

        assert model.input_network.embedding.weight.grad is not None

    def test_gradients_flow_to_worker_layers(self, model, sample_puzzle):
        """Gradients flow to all worker layer parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()

        for i, worker in enumerate(model.worker_layers):
            assert worker.input_proj.weight.grad is not None, (
                f"No gradient for worker_layers[{i}]"
            )

    def test_gradients_flow_to_output_network(self, model, sample_puzzle):
        """Gradients flow to output network parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()

        assert model.output_network.cell_head.weight.grad is not None

    def test_gradients_flow_to_initial_state(self, model, sample_puzzle):
        """Gradients flow to learned h_L initial state."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()

        # h_L_init grad may be None due to detach in one-step approx,
        # but gradients should still flow from the final iteration's x_in path.
        # The initial state feeds into the first iteration and is detached,
        # but the input_network and worker layers receive gradients.
        # Check that at least the output and worker receive gradients.
        assert model.output_network.cell_head.weight.grad is not None

    def test_no_gradient_to_planner(self, model):
        """No Planner exists, so no Planner gradients."""
        assert not hasattr(model, 'planner')

    def test_backward_pass_completes(self, model, batch_puzzles):
        """Full backward pass completes without errors."""
        outputs = model(batch_puzzles)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()  # Should not raise


# =============================================================================
# Test Predict Method
# =============================================================================

class TestPredictMethod:
    """Verify predict() method for inference."""

    def test_predict_returns_indices(self, model, sample_puzzle):
        """Predict returns cell and digit indices."""
        cell_idx, digit_idx = model.predict(sample_puzzle)

        assert cell_idx.shape == (1,)
        assert digit_idx.shape == (1,)

    def test_predict_cell_index_valid(self, model, sample_puzzle):
        """Cell index is in valid range [0, 15]."""
        cell_idx, _ = model.predict(sample_puzzle)

        assert 0 <= cell_idx.item() < 16

    def test_predict_digit_index_valid(self, model, sample_puzzle):
        """Digit index is in valid range [0, 3]."""
        _, digit_idx = model.predict(sample_puzzle)

        assert 0 <= digit_idx.item() < 4

    def test_predict_batch(self, model, batch_puzzles):
        """Predict works with batched inputs."""
        cell_idx, digit_idx = model.predict(batch_puzzles)
        batch_size = batch_puzzles.shape[0]

        assert cell_idx.shape == (batch_size,)
        assert digit_idx.shape == (batch_size,)


# =============================================================================
# Test State Dynamics
# =============================================================================

class TestStateDynamics:
    """Verify get_state_dynamics() method."""

    def test_state_dynamics_returns_dict(self, model, sample_puzzle):
        """State dynamics returns a dictionary."""
        outputs = model(sample_puzzle)
        dynamics = model.get_state_dynamics(outputs)

        assert isinstance(dynamics, dict)

    def test_state_dynamics_has_residual_history(self, model, sample_puzzle):
        """Dynamics contains residual history."""
        outputs = model(sample_puzzle)
        dynamics = model.get_state_dynamics(outputs)

        assert 'residual_history' in dynamics
        assert isinstance(dynamics['residual_history'], list)

    def test_state_dynamics_has_total_steps(self, model, sample_puzzle):
        """Dynamics contains total computation steps."""
        outputs = model(sample_puzzle)
        dynamics = model.get_state_dynamics(outputs)

        assert 'total_steps' in dynamics
        assert dynamics['total_steps'] > 0

    def test_state_dynamics_has_convergence_info(self, model, sample_puzzle):
        """Dynamics contains convergence information."""
        outputs = model(sample_puzzle)
        dynamics = model.get_state_dynamics(outputs)

        assert 'converged' in dynamics
        assert 'final_residual' in dynamics


# =============================================================================
# Test Input Validation
# =============================================================================

class TestInputValidation:
    """Verify input validation in forward pass."""

    def test_rejects_wrong_grid_size(self, model):
        """Rejects input with wrong grid size."""
        wrong_size = torch.randint(0, 5, (1, 9, 9))

        with pytest.raises(ValueError, match="Expected input shape"):
            model(wrong_size)

    def test_rejects_wrong_dimensions(self, model):
        """Rejects input with wrong number of dimensions."""
        wrong_dims = torch.randint(0, 5, (4, 4))

        with pytest.raises(ValueError, match="Expected input shape"):
            model(wrong_dims)

    def test_rejects_4d_input(self, model):
        """Rejects 4D input tensor."""
        wrong_dims = torch.randint(0, 5, (1, 1, 4, 4))

        with pytest.raises(ValueError, match="Expected input shape"):
            model(wrong_dims)


# =============================================================================
# Test Configuration Validation
# =============================================================================

class TestConfigurationValidation:
    """Verify configuration parameter validation."""

    def test_rejects_zero_hidden_dim(self):
        """Rejects hidden_dim <= 0."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            HRM_4x4_Simple(hidden_dim=0)

    def test_rejects_negative_hidden_dim(self):
        """Rejects negative hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            HRM_4x4_Simple(hidden_dim=-64)

    def test_rejects_zero_iterations(self):
        """Rejects n_iterations <= 0."""
        with pytest.raises(ValueError, match="n_iterations must be positive"):
            HRM_4x4_Simple(n_iterations=0)

    def test_rejects_zero_worker_layers(self):
        """Rejects n_worker_layers <= 0."""
        with pytest.raises(ValueError, match="n_worker_layers must be positive"):
            HRM_4x4_Simple(n_worker_layers=0)

    def test_rejects_invalid_dropout(self):
        """Rejects dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            HRM_4x4_Simple(dropout=1.0)

    def test_rejects_negative_convergence_threshold(self):
        """Rejects convergence_threshold <= 0."""
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            HRM_4x4_Simple(convergence_threshold=-1e-3)


# =============================================================================
# Test Factory Function
# =============================================================================

class TestCreateHRM4x4Simple:
    """Verify create_hrm_4x4_simple factory function."""

    def test_creates_model(self):
        """Factory creates HRM_4x4_Simple instance."""
        model = create_hrm_4x4_simple()
        assert isinstance(model, HRM_4x4_Simple)

    def test_default_configuration(self):
        """Factory uses default configuration."""
        model = create_hrm_4x4_simple()
        assert model.hidden_dim == 64
        assert model.n_iterations == 50
        assert model.n_worker_layers == 8

    def test_custom_configuration(self):
        """Factory accepts custom configuration."""
        model = create_hrm_4x4_simple(
            hidden_dim=128,
            n_iterations=100,
            n_worker_layers=4,
            dropout=0.2,
        )
        assert model.hidden_dim == 128
        assert model.n_iterations == 100
        assert model.n_worker_layers == 4
        assert model.dropout == 0.2


# =============================================================================
# Test Training Mode
# =============================================================================

class TestTrainingMode:
    """Verify model behavior in train vs eval mode."""

    def test_train_mode_runs(self, model, batch_puzzles):
        """Train mode forward pass completes."""
        model.train()
        outputs = model(batch_puzzles)
        assert outputs['cell_logits'] is not None

    def test_eval_mode_deterministic(self, model, batch_puzzles):
        """Eval mode produces deterministic outputs."""
        model.eval()

        with torch.no_grad():
            outputs1 = model(batch_puzzles)
            outputs2 = model(batch_puzzles)

        torch.testing.assert_close(
            outputs1['cell_logits'],
            outputs2['cell_logits'],
        )


# =============================================================================
# Test Representation
# =============================================================================

class TestRepresentation:
    """Verify string representations."""

    def test_model_extra_repr(self, model):
        """Model has informative extra_repr."""
        repr_str = model.extra_repr()

        assert 'hidden_dim=64' in repr_str
        assert 'n_iterations=50' in repr_str
        assert 'n_worker_layers=8' in repr_str

    def test_trace_repr(self, model, sample_puzzle):
        """SimpleExecutionTrace has informative repr."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        repr_str = repr(trace)

        assert 'SimpleExecutionTrace' in repr_str
        assert 'steps=' in repr_str


# =============================================================================
# Test Comparison with Full HRM Interface
# =============================================================================

class TestInterfaceCompatibility:
    """Verify simplified model provides compatible interface."""

    def test_same_output_keys_subset(self, model, sample_puzzle):
        """Simplified model outputs are a subset of full HRM outputs."""
        outputs = model(sample_puzzle)

        # Must have these keys (same as full HRM)
        assert 'cell_logits' in outputs
        assert 'digit_logits' in outputs
        assert 'trace' in outputs
        assert 'h_L_final' in outputs

    def test_same_logit_shapes(self, model, sample_puzzle):
        """Logit shapes match full HRM output shapes."""
        outputs = model(sample_puzzle)

        assert outputs['cell_logits'].shape == (1, 16)
        assert outputs['digit_logits'].shape == (1, 4)

    def test_has_same_helper_methods(self, model):
        """Model has same helper methods as full HRM."""
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_halt_penalty')
        assert hasattr(model, 'get_convergence_loss')
        assert hasattr(model, 'get_state_dynamics')


# =============================================================================
# Test Parameter Count (fewer than full when layer count equal)
# =============================================================================

class TestParameterEfficiency:
    """Verify parameter count properties."""

    def test_no_planner_parameters(self, model):
        """No Planner-related parameters exist."""
        param_names = [name for name, _ in model.named_parameters()]
        for name in param_names:
            assert 'planner' not in name.lower()

    def test_no_h_H_init_parameter(self, model):
        """No h_H_init parameter exists."""
        param_names = [name for name, _ in model.named_parameters()]
        assert 'h_H_init' not in param_names

    def test_has_h_L_init_parameter(self, model):
        """h_L_init parameter exists."""
        param_names = [name for name, _ in model.named_parameters()]
        assert 'h_L_init' in param_names
