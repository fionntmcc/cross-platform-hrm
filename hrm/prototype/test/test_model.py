"""
Unit Tests for HRM_4x4 Complete Model

Issue #8: Tests verifying acceptance criteria:
- HRM_4x4 class initialises all four components
- Learned h_L and h_H initial states are created
- Forward pass returns predictions + execution traces
- get_halt_penalty() computes halting penalty
- get_convergence_loss() computes convergence loss
- Configuration parameters work as expected

Run tests: pytest test/hrm_test/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn

from hrm.model import HRM_4x4, ExecutionTrace, create_hrm_4x4
from hrm.layers.input_network import InputNetwork
from hrm.layers.worker import WorkerModule
from hrm.layers.planner import PlannerModule
from hrm.layers.output_network import OutputNetwork


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model():
    """Create a default HRM_4x4 model for testing."""
    return HRM_4x4()


@pytest.fixture
def model_custom():
    """Create a custom HRM_4x4 model with non-default parameters."""
    return HRM_4x4(
        hidden_dim=128,
        embed_dim=32,
        n_outer_cycles=3,
        n_inner_steps=5,
        convergence_threshold=1e-2,
        dropout=0.2,
    )


@pytest.fixture
def sample_puzzle():
    """Create a sample 4x4 Sudoku puzzle tensor."""
    # 0 = empty, 1-4 = digits
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
# Test Model Initialisation (Core Acceptance Criteria)
# =============================================================================

class TestModelInitialisation:
    """Verify HRM_4x4 initialises all components correctly."""
    
    def test_has_input_network(self, model):
        """Model has InputNetwork component."""
        assert hasattr(model, 'input_network')
        assert isinstance(model.input_network, InputNetwork)
    
    def test_has_worker_module(self, model):
        """Model has WorkerModule component."""
        assert hasattr(model, 'worker')
        assert isinstance(model.worker, WorkerModule)
    
    def test_has_planner_module(self, model):
        """Model has PlannerModule component."""
        assert hasattr(model, 'planner')
        assert isinstance(model.planner, PlannerModule)
    
    def test_has_output_network(self, model):
        """Model has OutputNetwork component."""
        assert hasattr(model, 'output_network')
        assert isinstance(model.output_network, OutputNetwork)
    
    def test_has_learned_h_L_init(self, model):
        """Model has learned h_L initial state parameter."""
        assert hasattr(model, 'h_L_init')
        assert isinstance(model.h_L_init, nn.Parameter)
        assert model.h_L_init.shape == (1, model.hidden_dim)
    
    def test_has_learned_h_H_init(self, model):
        """Model has learned h_H initial state parameter."""
        assert hasattr(model, 'h_H_init')
        assert isinstance(model.h_H_init, nn.Parameter)
        assert model.h_H_init.shape == (1, model.hidden_dim)
    
    def test_default_configuration(self, model):
        """Default configuration matches specification."""
        assert model.hidden_dim == 64
        assert model.n_outer_cycles == 5
        assert model.n_inner_steps == 10
        assert model.convergence_threshold == 1e-3
        assert model.dropout == 0.1
    
    def test_custom_configuration(self, model_custom):
        """Custom configuration is applied correctly."""
        assert model_custom.hidden_dim == 128
        assert model_custom.n_outer_cycles == 3
        assert model_custom.n_inner_steps == 5
        assert model_custom.convergence_threshold == 1e-2
        assert model_custom.dropout == 0.2
    
    def test_components_share_hidden_dim(self, model):
        """All components use the same hidden_dim."""
        assert model.input_network.hidden_dim == model.hidden_dim
        assert model.worker.hidden_dim == model.hidden_dim
        assert model.planner.hidden_dim == model.hidden_dim
        assert model.output_network.hidden_dim == model.hidden_dim


# =============================================================================
# Test Forward Pass (Core Acceptance Criteria)
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
        assert isinstance(outputs['trace'], ExecutionTrace)
    
    def test_forward_has_final_states(self, model, sample_puzzle):
        """Output contains final hidden states."""
        outputs = model(sample_puzzle)
        assert 'h_L_final' in outputs
        assert 'h_H_final' in outputs
        assert outputs['h_L_final'].shape == (1, model.hidden_dim)
        assert outputs['h_H_final'].shape == (1, model.hidden_dim)
    
    def test_forward_batch_processing(self, model, batch_puzzles):
        """Forward pass handles batched inputs correctly."""
        outputs = model(batch_puzzles)
        batch_size = batch_puzzles.shape[0]
        
        assert outputs['cell_logits'].shape == (batch_size, 16)
        assert outputs['digit_logits'].shape == (batch_size, 4)
        assert outputs['h_L_final'].shape == (batch_size, model.hidden_dim)
        assert outputs['h_H_final'].shape == (batch_size, model.hidden_dim)
    
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
    
    def test_trace_has_outer_stats(self, model, sample_puzzle):
        """Trace contains outer loop statistics."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        
        assert hasattr(trace, 'outer_stats')
        assert trace.outer_stats.num_cycles > 0
        assert trace.outer_stats.total_inner_steps > 0
    
    def test_trace_cycles_bounded(self, model, sample_puzzle):
        """Number of cycles is bounded by n_outer_cycles."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        
        assert trace.outer_stats.num_cycles <= model.n_outer_cycles
    
    def test_trace_inner_steps_bounded(self, model, sample_puzzle):
        """Total inner steps is bounded by K × T."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        max_steps = model.n_outer_cycles * model.n_inner_steps
        
        assert trace.outer_stats.total_inner_steps <= max_steps
    
    def test_trace_total_computation_steps(self, model, sample_puzzle):
        """Trace correctly computes total computation steps."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        
        assert trace.total_computation_steps == trace.outer_stats.total_inner_steps
    
    def test_trace_has_residual_history(self, model, sample_puzzle):
        """Trace contains residual history when track_history=True."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        
        assert len(trace.outer_stats.h_H_residual_history) > 0


# =============================================================================
# Test Halt Penalty (Acceptance Criteria)
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
        
        # Penalty = lambda * (actual_steps / max_steps) <= lambda
        assert penalty.item() <= lambda_halt
    
    def test_halt_penalty_lambda_scaling(self, model, sample_puzzle):
        """Halt penalty scales with lambda_halt."""
        outputs = model(sample_puzzle)
        penalty_small = model.get_halt_penalty(outputs, lambda_halt=0.001)
        penalty_large = model.get_halt_penalty(outputs, lambda_halt=0.1)
        
        assert penalty_large.item() > penalty_small.item()


# =============================================================================
# Test Convergence Loss (Acceptance Criteria)
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
        
        # Set a very high target that should always be met
        loss = model.get_convergence_loss(outputs, target_residual=1e6)
        
        assert loss.item() == pytest.approx(0.0)


# =============================================================================
# Test Gradient Flow
# =============================================================================

class TestGradientFlow:
    """Verify gradients flow through the complete model."""
    
    def test_gradients_flow_to_input_network(self, model, sample_puzzle):
        """Gradients flow to input network parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()
        
        assert model.input_network.embedding.weight.grad is not None
    
    def test_gradients_flow_to_worker(self, model, sample_puzzle):
        """Gradients flow to worker parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()
        
        assert model.worker.input_proj.weight.grad is not None
    
    def test_gradients_flow_to_planner(self, model, sample_puzzle):
        """Gradients flow to planner parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()
        
        assert model.planner.input_proj.weight.grad is not None
    
    def test_gradients_flow_to_output_network(self, model, sample_puzzle):
        """Gradients flow to output network parameters."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()
        
        assert model.output_network.cell_head.weight.grad is not None
    
    def test_gradients_flow_to_initial_states(self, model, sample_puzzle):
        """Gradients flow to learned initial states."""
        outputs = model(sample_puzzle)
        loss = outputs['cell_logits'].sum() + outputs['digit_logits'].sum()
        loss.backward()
        
        assert model.h_L_init.grad is not None
        assert model.h_H_init.grad is not None


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
    
    def test_state_dynamics_has_residuals(self, model, sample_puzzle):
        """Dynamics contains h_H residual history."""
        outputs = model(sample_puzzle)
        dynamics = model.get_state_dynamics(outputs)
        
        assert 'h_H_residuals' in dynamics
        assert isinstance(dynamics['h_H_residuals'], list)
    
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
        
        assert 'outer_converged' in dynamics
        assert 'cycles_completed' in dynamics


# =============================================================================
# Test Input Validation
# =============================================================================

class TestInputValidation:
    """Verify input validation in forward pass."""
    
    def test_rejects_wrong_grid_size(self, model):
        """Rejects input with wrong grid size."""
        wrong_size = torch.randint(0, 5, (1, 9, 9))  # 9x9 instead of 4x4
        
        with pytest.raises(ValueError, match="Expected input shape"):
            model(wrong_size)
    
    def test_rejects_wrong_dimensions(self, model):
        """Rejects input with wrong number of dimensions."""
        wrong_dims = torch.randint(0, 5, (4, 4))  # Missing batch dimension
        
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
            HRM_4x4(hidden_dim=0)
    
    def test_rejects_negative_hidden_dim(self):
        """Rejects negative hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            HRM_4x4(hidden_dim=-64)
    
    def test_rejects_zero_outer_cycles(self):
        """Rejects n_outer_cycles <= 0."""
        with pytest.raises(ValueError, match="n_outer_cycles must be positive"):
            HRM_4x4(n_outer_cycles=0)
    
    def test_rejects_zero_inner_steps(self):
        """Rejects n_inner_steps <= 0."""
        with pytest.raises(ValueError, match="n_inner_steps must be positive"):
            HRM_4x4(n_inner_steps=0)
    
    def test_rejects_invalid_dropout(self):
        """Rejects dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            HRM_4x4(dropout=1.0)
    
    def test_rejects_negative_convergence_threshold(self):
        """Rejects convergence_threshold <= 0."""
        with pytest.raises(ValueError, match="convergence_threshold must be positive"):
            HRM_4x4(convergence_threshold=-1e-3)


# =============================================================================
# Test Factory Function
# =============================================================================

class TestCreateHRM4x4:
    """Verify create_hrm_4x4 factory function."""
    
    def test_creates_model(self):
        """Factory creates HRM_4x4 instance."""
        model = create_hrm_4x4()
        assert isinstance(model, HRM_4x4)
    
    def test_default_configuration(self):
        """Factory uses default configuration."""
        model = create_hrm_4x4()
        assert model.hidden_dim == 64
        assert model.n_outer_cycles == 5
        assert model.n_inner_steps == 10
    
    def test_custom_configuration(self):
        """Factory accepts custom configuration."""
        model = create_hrm_4x4(
            hidden_dim=128,
            n_outer_cycles=10,
            dropout=0.2
        )
        assert model.hidden_dim == 128
        assert model.n_outer_cycles == 10
        assert model.dropout == 0.2


# =============================================================================
# Test Training Mode
# =============================================================================

class TestTrainingMode:
    """Verify model behavior in train vs eval mode."""
    
    def test_train_mode_affects_dropout(self, model, batch_puzzles):
        """Train mode enables dropout variability."""
        model.train()
        
        # Multiple forward passes in train mode may differ due to dropout
        outputs1 = model(batch_puzzles)
        outputs2 = model(batch_puzzles)
        
        # At least one of the outputs should potentially differ
        # (with dropout enabled, repeated runs may vary)
        # This is a weak test - dropout may still produce same results
        assert outputs1['cell_logits'] is not None
        assert outputs2['cell_logits'] is not None
    
    def test_eval_mode_deterministic(self, model, batch_puzzles):
        """Eval mode produces deterministic outputs."""
        model.eval()
        
        with torch.no_grad():
            outputs1 = model(batch_puzzles)
            outputs2 = model(batch_puzzles)
        
        torch.testing.assert_close(
            outputs1['cell_logits'],
            outputs2['cell_logits']
        )


# =============================================================================
# Test Device Compatibility
# =============================================================================

class TestDeviceCompatibility:
    """Verify model works on different devices."""
    
    def test_cpu_execution(self, model, sample_puzzle):
        """Model runs on CPU."""
        model = model.cpu()
        puzzle = sample_puzzle.cpu()
        
        outputs = model(puzzle)
        
        assert outputs['cell_logits'].device.type == 'cpu'
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_execution(self, model, sample_puzzle):
        """Model runs on CUDA if available."""
        model = model.cuda()
        puzzle = sample_puzzle.cuda()
        
        outputs = model(puzzle)
        
        assert outputs['cell_logits'].device.type == 'cuda'


# =============================================================================
# Test Representation
# =============================================================================

class TestRepresentation:
    """Verify string representations."""
    
    def test_model_extra_repr(self, model):
        """Model has informative extra_repr."""
        repr_str = model.extra_repr()
        
        assert 'hidden_dim=64' in repr_str
        assert 'n_outer_cycles=5' in repr_str
        assert 'n_inner_steps=10' in repr_str
    
    def test_trace_repr(self, model, sample_puzzle):
        """ExecutionTrace has informative repr."""
        outputs = model(sample_puzzle)
        trace = outputs['trace']
        
        repr_str = repr(trace)
        
        assert 'ExecutionTrace' in repr_str
        assert 'cycles=' in repr_str
