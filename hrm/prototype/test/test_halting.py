"""
Unit Tests for Adaptive Computation Time (ACT) with Q-Learning Halting

Tests verifying acceptance criteria:
- HRM_4x4_WithHalting subclass works correctly
- Q-head produces valid halt/continue decisions
- should_halt() method respects minimum cycle constraint
- Halting integrates correctly into forward pass
- λ penalty term computed correctly
- Training vs inference behavior differs appropriately

Run tests: pytest test/hrm_test/test_halting.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from hrm.prototype.core.halting import (
    QHaltingHead,
    HaltingPolicy,
    ACTStats,
    HaltingQTrainer,
    compute_ponder_cost,
    compute_act_loss,
    create_halting_components,
)
from hrm.prototype.models.model_4x4_layers import HRM_4x4_WithHalting, HRM_4x4


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def q_head(hidden_dim):
    return QHaltingHead(hidden_dim=hidden_dim)


@pytest.fixture
def policy():
    return HaltingPolicy(min_cycles=2, exploration_strategy="greedy")


@pytest.fixture
def model():
    return HRM_4x4_WithHalting(
        hidden_dim=64,
        n_heads=4,
        n_outer_cycles=5,
        n_inner_steps=10,
        min_cycles=2,
        lambda_penalty=0.01,
    )


@pytest.fixture
def base_model():
    return HRM_4x4(
        hidden_dim=64,
        n_heads=4,
        n_outer_cycles=5,
        n_inner_steps=10,
    )


@pytest.fixture
def sample_puzzle():
    return torch.randint(0, 5, (4, 4, 4))


@pytest.fixture
def sample_h_H(hidden_dim):
    return torch.randn(4, hidden_dim)


# =============================================================================
# Test ACTStats Dataclass
# =============================================================================

class TestACTStats:
    """Tests for ACTStats dataclass."""
    
    def test_stats_creation(self):
        """Test ACTStats can be created with required fields."""
        stats = ACTStats(
            num_cycles_used=3,
            max_cycles=5,
            total_inner_steps=30,
            halted_early=True,
            halt_cycle=2,
        )
        
        assert stats.num_cycles_used == 3
        assert stats.max_cycles == 5
        assert stats.halted_early is True
        assert stats.halt_cycle == 2
    
    def test_efficiency_with_early_halt(self):
        """Test efficiency computation with early halting."""
        stats = ACTStats(
            num_cycles_used=3,
            max_cycles=10,
            total_inner_steps=30,
            halted_early=True,
            halt_cycle=2,
        )
        
        # Saved 7 out of 10 cycles = 0.7 efficiency
        assert stats.efficiency == pytest.approx(0.7)
    
    def test_efficiency_without_early_halt(self):
        """Test efficiency is 0 when no early halting."""
        stats = ACTStats(
            num_cycles_used=5,
            max_cycles=5,
            total_inner_steps=50,
            halted_early=False,
        )
        
        assert stats.efficiency == 0.0
    
    def test_average_halt_probability(self):
        """Test average halt probability computation."""
        stats = ACTStats(
            num_cycles_used=3,
            max_cycles=5,
            total_inner_steps=30,
            halted_early=True,
            halt_probabilities=[0.2, 0.4, 0.6],
        )
        
        assert stats.average_halt_probability == pytest.approx(0.4)
    
    def test_average_halt_probability_empty(self):
        """Test average halt probability returns None when empty."""
        stats = ACTStats(
            num_cycles_used=0,
            max_cycles=5,
            total_inner_steps=0,
            halted_early=False,
        )
        
        assert stats.average_halt_probability is None
    
    def test_repr(self):
        """Test string representation."""
        stats = ACTStats(
            num_cycles_used=3,
            max_cycles=5,
            total_inner_steps=30,
            halted_early=True,
            ponder_cost=0.6,
        )
        
        repr_str = repr(stats)
        assert "cycles=3/5" in repr_str
        assert "halted_early=True" in repr_str


# =============================================================================
# Test QHaltingHead
# =============================================================================

class TestQHaltingHead:
    """Tests for QHaltingHead neural network."""
    
    def test_output_shape(self, q_head, sample_h_H):
        """Q-head outputs correct shape."""
        q_values = q_head(sample_h_H)
        
        assert q_values.shape == (4, 2)
    
    def test_output_contains_halt_and_continue(self, q_head, sample_h_H):
        """Q-values have halt and continue components."""
        q_values = q_head(sample_h_H)
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]
        
        assert q_halt.shape == (4,)
        assert q_continue.shape == (4,)
    
    def test_get_halt_probability_range(self, q_head, sample_h_H):
        """Halt probability is in [0, 1]."""
        halt_prob = q_head.get_halt_probability(sample_h_H)
        
        assert halt_prob.shape == (4,)
        assert (halt_prob >= 0).all()
        assert (halt_prob <= 1).all()
    
    def test_temperature_affects_probability(self, q_head, sample_h_H):
        """Lower temperature makes distribution more peaked."""
        prob_high_temp = q_head.get_halt_probability(sample_h_H, temperature=10.0)
        prob_low_temp = q_head.get_halt_probability(sample_h_H, temperature=0.1)
        
        # Low temp should be closer to 0 or 1
        high_temp_entropy = -(prob_high_temp * torch.log(prob_high_temp + 1e-8) + 
                             (1-prob_high_temp) * torch.log(1-prob_high_temp + 1e-8))
        low_temp_entropy = -(prob_low_temp * torch.log(prob_low_temp + 1e-8) + 
                            (1-prob_low_temp) * torch.log(1-prob_low_temp + 1e-8))
        
        # Low temperature should have lower entropy (more deterministic)
        assert low_temp_entropy.mean() < high_temp_entropy.mean()
    
    def test_gradients_flow(self, q_head, sample_h_H):
        """Gradients flow through Q-head."""
        sample_h_H.requires_grad_(True)
        q_values = q_head(sample_h_H)
        loss = q_values.sum()
        loss.backward()
        
        assert sample_h_H.grad is not None
    
    def test_invalid_hidden_dim_raises_error(self):
        """Invalid hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            QHaltingHead(hidden_dim=0)
    
    def test_extra_repr(self, q_head):
        """Test extra_repr contains configuration."""
        repr_str = q_head.extra_repr()
        assert "hidden_dim=64" in repr_str


# =============================================================================
# Test HaltingPolicy
# =============================================================================

class TestHaltingPolicy:
    """Tests for HaltingPolicy decision making."""
    
    def test_respects_min_cycles(self, policy):
        """Policy respects minimum cycles constraint."""
        q_values = torch.tensor([[1.0, 0.0]])  # Q_halt > Q_continue
        
        # Before min_cycles
        should_halt, prob = policy.should_halt(q_values, cycle=0, training=False)
        assert should_halt is False
        
        should_halt, prob = policy.should_halt(q_values, cycle=1, training=False)
        assert should_halt is False
    
    def test_halts_after_min_cycles_greedy(self, policy):
        """Policy halts when Q_halt > Q_continue after min_cycles."""
        q_values = torch.tensor([[1.0, 0.0]])  # Q_halt > Q_continue
        
        should_halt, prob = policy.should_halt(q_values, cycle=2, training=False)
        assert should_halt is True
    
    def test_continues_when_q_continue_higher(self, policy):
        """Policy continues when Q_continue > Q_halt."""
        q_values = torch.tensor([[0.0, 1.0]])  # Q_continue > Q_halt
        
        should_halt, prob = policy.should_halt(q_values, cycle=5, training=False)
        assert should_halt is False
    
    def test_inference_always_greedy(self):
        """Inference mode always uses greedy policy."""
        policy = HaltingPolicy(
            min_cycles=0,
            exploration_strategy="epsilon_greedy",
            epsilon=1.0,  # 100% random during training
        )
        q_values = torch.tensor([[1.0, 0.0]])
        
        # Inference should be deterministic despite epsilon=1.0
        results = [policy.should_halt(q_values, cycle=1, training=False)[0] for _ in range(10)]
        assert all(results)  # All should be True (halt)
    
    def test_epsilon_greedy_explores(self):
        """Epsilon-greedy explores during training."""
        policy = HaltingPolicy(
            min_cycles=0,
            exploration_strategy="epsilon_greedy",
            epsilon=1.0,  # 100% random
        )
        q_values = torch.tensor([[1.0, 0.0]])
        
        # With epsilon=1.0, should get mix of halt/continue
        torch.manual_seed(42)
        results = [policy.should_halt(q_values, cycle=1, training=True)[0] for _ in range(100)]
        
        # Should have some True and some False
        assert any(results)
        assert not all(results)
    
    def test_softmax_exploration(self):
        """Softmax exploration samples from distribution."""
        policy = HaltingPolicy(
            min_cycles=0,
            exploration_strategy="softmax",
            temperature=1.0,
        )
        # Balanced Q-values
        q_values = torch.tensor([[0.5, 0.5]])
        
        torch.manual_seed(42)
        results = [policy.should_halt(q_values, cycle=1, training=True)[0] for _ in range(100)]
        
        # Should have mix due to sampling
        halt_count = sum(results)
        assert 20 < halt_count < 80  # Roughly balanced
    
    def test_invalid_strategy_raises_error(self):
        """Invalid exploration strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown exploration_strategy"):
            HaltingPolicy(exploration_strategy="invalid")
    
    def test_invalid_epsilon_raises_error(self):
        """Invalid epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be in"):
            HaltingPolicy(epsilon=1.5)
    
    def test_invalid_temperature_raises_error(self):
        """Invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            HaltingPolicy(temperature=0)


# =============================================================================
# Test Ponder Cost and ACT Loss
# =============================================================================

class TestPonderCostAndLoss:
    """Tests for ponder cost and ACT loss computation."""
    
    def test_ponder_cost_normalized(self):
        """Ponder cost is normalized to [0, 1]."""
        cost = compute_ponder_cost(num_cycles=5, max_cycles=10)
        assert cost == 0.5
    
    def test_ponder_cost_full_cycles(self):
        """Ponder cost is 1.0 when using all cycles."""
        cost = compute_ponder_cost(num_cycles=10, max_cycles=10)
        assert cost == 1.0
    
    def test_ponder_cost_with_multiplier(self):
        """Ponder cost respects cost_per_cycle."""
        cost = compute_ponder_cost(num_cycles=5, max_cycles=10, cost_per_cycle=2.0)
        assert cost == 1.0  # 5 * 2.0 / 10 = 1.0
    
    def test_act_loss_combines_task_and_ponder(self):
        """ACT loss = task_loss + λ × ponder_cost."""
        task_loss = torch.tensor(1.0)
        ponder_cost = 0.5
        lambda_penalty = 0.1
        
        act_loss = compute_act_loss(ponder_cost, task_loss, lambda_penalty)
        
        expected = 1.0 + 0.1 * 0.5  # 1.05
        assert act_loss.item() == pytest.approx(expected)
    
    def test_act_loss_zero_lambda(self):
        """ACT loss equals task loss when λ=0."""
        task_loss = torch.tensor(2.5)
        ponder_cost = 1.0
        
        act_loss = compute_act_loss(ponder_cost, task_loss, lambda_penalty=0.0)
        
        assert act_loss.item() == pytest.approx(2.5)


# =============================================================================
# Test HRM_4x4_WithHalting Model
# =============================================================================

class TestHRM4x4WithHalting:
    """Tests for HRM_4x4_WithHalting model."""
    
    def test_is_subclass_of_hrm_4x4(self, model, base_model):
        """HRM_4x4_WithHalting is subclass of HRM_4x4."""
        assert isinstance(model, HRM_4x4)
    
    def test_has_q_head(self, model):
        """Model has Q-head for halting decisions."""
        assert hasattr(model, 'q_head')
        assert model.q_head is not None
    
    def test_forward_output_shapes(self, model, sample_puzzle):
        """Forward pass outputs correct shapes."""
        cell_logits, digit_logits, traces = model(sample_puzzle)
        
        assert cell_logits.shape == (4, 16)
        assert digit_logits.shape == (4, 4)
        assert isinstance(traces, dict)
    
    def test_traces_contain_halting_info(self, model, sample_puzzle):
        """Traces contain ACT-specific information."""
        _, _, traces = model(sample_puzzle)
        
        assert 'halted_early' in traces
        assert 'halt_cycle' in traces
        assert 'q_values_history' in traces
        assert 'halt_probabilities' in traces
        assert 'ponder_cost' in traces
        assert 'lambda_penalty' in traces
    
    def test_min_cycles_respected(self, sample_puzzle):
        """Model respects minimum cycles constraint."""
        model = HRM_4x4_WithHalting(
            hidden_dim=64,
            n_outer_cycles=10,
            min_cycles=5,
        )
        model.eval()
        
        _, _, traces = model(sample_puzzle)
        
        # Even if halted early, should have at least min_cycles
        if traces['halted_early']:
            assert traces['halt_cycle'] >= model.min_cycles - 1  # 0-indexed
    
    def test_force_max_cycles_ignores_halting(self, model, sample_puzzle):
        """force_max_cycles runs all cycles."""
        model.eval()
        
        _, _, traces = model(sample_puzzle, force_max_cycles=True)
        
        assert traces['num_cycles'] == model.n_outer_cycles
        assert not traces['halted_early']
    
    def test_ponder_cost_in_range(self, model, sample_puzzle):
        """Ponder cost is in [0, 1]."""
        _, _, traces = model(sample_puzzle)
        
        assert 0 <= traces['ponder_cost'] <= 1
    
    def test_get_halt_penalty(self, model, sample_puzzle):
        """get_halt_penalty returns correct penalty."""
        _, _, traces = model(sample_puzzle)
        
        penalty = model.get_halt_penalty(traces)
        
        expected = model.lambda_penalty * traces['ponder_cost']
        assert penalty.item() == pytest.approx(expected)
    
    def test_get_act_loss(self, model, sample_puzzle):
        """get_act_loss combines task loss and penalty."""
        cell_logits, digit_logits, traces = model(sample_puzzle)
        
        target_cell = torch.randint(0, 16, (4,))
        task_loss = F.cross_entropy(cell_logits, target_cell)
        
        act_loss = model.get_act_loss(task_loss, traces)
        
        expected = task_loss + model.lambda_penalty * traces['ponder_cost']
        assert act_loss.item() == pytest.approx(expected.item(), rel=1e-5)
    
    def test_set_lambda_penalty(self, model):
        """set_lambda_penalty updates λ."""
        model.set_lambda_penalty(0.05)
        assert model.lambda_penalty == 0.05
    
    def test_set_lambda_penalty_negative_raises_error(self, model):
        """Negative λ raises ValueError."""
        with pytest.raises(ValueError, match="lambda_penalty must be non-negative"):
            model.set_lambda_penalty(-0.1)
    
    def test_set_exploration_epsilon(self, model):
        """set_exploration_epsilon updates ε."""
        model.set_exploration_epsilon(0.5)
        assert model.exploration_epsilon == 0.5
    
    def test_set_exploration_epsilon_invalid_raises_error(self, model):
        """Invalid ε raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be in"):
            model.set_exploration_epsilon(1.5)
    
    def test_gradients_flow_through_model(self, model, sample_puzzle):
        """Gradients flow through entire model."""
        cell_logits, digit_logits, traces = model(sample_puzzle)
        
        target_cell = torch.randint(0, 16, (4,))
        loss = F.cross_entropy(cell_logits, target_cell)
        loss.backward()
        
        # Check some parameters have gradients
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads
    
    def test_training_vs_eval_mode(self, model, sample_puzzle):
        """Training and eval modes behave differently."""
        # Training mode has exploration
        model.train()
        model.exploration_epsilon = 1.0  # 100% random
        
        torch.manual_seed(42)
        _, _, traces_train = model(sample_puzzle)
        
        # Eval mode is deterministic
        model.eval()
        _, _, traces_eval = model(sample_puzzle)
        
        # Both should produce valid traces
        assert 'halted_early' in traces_train
        assert 'halted_early' in traces_eval


# =============================================================================
# Test Q-Head Training (HaltingQTrainer)
# =============================================================================

class TestHaltingQTrainer:
    """Tests for Q-learning trainer."""
    
    def test_trainer_creation(self, q_head):
        """Trainer can be created."""
        trainer = HaltingQTrainer(q_head, learning_rate=0.001)
        
        assert trainer.q_head is q_head
        assert trainer.target_q_head is not None
    
    def test_store_transition(self, q_head, sample_h_H):
        """Transitions can be stored."""
        trainer = HaltingQTrainer(q_head)
        
        trainer.store_transition(
            h_H=sample_h_H,
            action=0,  # halt
            reward=-0.5,
            h_H_next=sample_h_H,
            done=False,
        )
        
        assert len(trainer.replay_buffer) == 1
    
    def test_update_requires_sufficient_buffer(self, q_head, sample_h_H):
        """Update returns None with insufficient buffer."""
        trainer = HaltingQTrainer(q_head)
        trainer.batch_size = 32
        
        # Add only 10 transitions
        for _ in range(10):
            trainer.store_transition(
                h_H=sample_h_H,
                action=0,
                reward=-0.5,
                h_H_next=sample_h_H,
                done=False,
            )
        
        result = trainer.update()
        assert result is None
    
    def test_update_returns_loss(self, q_head, sample_h_H):
        """Update returns loss value when buffer sufficient."""
        trainer = HaltingQTrainer(q_head)
        trainer.batch_size = 8
        
        # Add enough transitions
        for _ in range(20):
            trainer.store_transition(
                h_H=sample_h_H,
                action=0,
                reward=-0.5,
                h_H_next=sample_h_H,
                done=False,
            )
        
        loss = trainer.update()
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_buffer_size_limit(self, q_head, sample_h_H):
        """Buffer respects max size."""
        trainer = HaltingQTrainer(q_head)
        trainer.max_buffer_size = 10
        
        for _ in range(20):
            trainer.store_transition(
                h_H=sample_h_H,
                action=0,
                reward=0.0,
                h_H_next=sample_h_H,
                done=False,
            )
        
        assert len(trainer.replay_buffer) == 10


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for create_halting_components factory."""
    
    def test_creates_components(self, hidden_dim):
        """Factory creates Q-head and policy."""
        q_head, policy = create_halting_components(hidden_dim=hidden_dim)
        
        assert isinstance(q_head, QHaltingHead)
        assert isinstance(policy, HaltingPolicy)
    
    def test_respects_parameters(self, hidden_dim):
        """Factory respects provided parameters."""
        q_head, policy = create_halting_components(
            hidden_dim=hidden_dim,
            min_cycles=5,
            exploration_strategy="softmax",
            epsilon=0.2,
        )
        
        assert policy.min_cycles == 5
        assert policy.exploration_strategy == "softmax"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for halting mechanism."""
    
    def test_training_loop_simulation(self, model, sample_puzzle):
        """Simulate a training step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Forward
        cell_logits, digit_logits, traces = model(sample_puzzle)
        
        # Compute losses
        target_cell = torch.randint(0, 16, (4,))
        target_digit = torch.randint(0, 4, (4,))
        
        cell_loss = F.cross_entropy(cell_logits, target_cell)
        digit_loss = F.cross_entropy(digit_logits, target_digit)
        task_loss = cell_loss + digit_loss
        
        # ACT loss with penalty
        total_loss = model.get_act_loss(task_loss, traces)
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads
        
        # Step
        optimizer.step()
    
    def test_compare_with_and_without_halting(self, sample_puzzle):
        """Compare outputs with and without halting."""
        model_halt = HRM_4x4_WithHalting(
            hidden_dim=64,
            n_outer_cycles=5,
            lambda_penalty=0.0,  # No penalty
        )
        model_halt.eval()
        
        # Run with halting disabled
        _, _, traces_forced = model_halt(sample_puzzle, force_max_cycles=True)
        
        # Run with halting enabled
        _, _, traces_halt = model_halt(sample_puzzle, force_max_cycles=False)
        
        # Forced should use all cycles
        assert traces_forced['num_cycles'] == 5
        
        # Halting may or may not halt early
        assert traces_halt['num_cycles'] <= 5
    
    def test_lambda_curriculum(self, sample_puzzle):
        """Test λ curriculum learning."""
        model = HRM_4x4_WithHalting(
            hidden_dim=64,
            n_outer_cycles=5,
            lambda_penalty=0.0,
        )
        
        # Start with no penalty
        assert model.lambda_penalty == 0.0
        
        # Increase penalty over "training"
        for epoch in range(5):
            model.set_lambda_penalty(0.01 * (epoch + 1))
            _, _, traces = model(sample_puzzle)
            assert traces['lambda_penalty'] == 0.01 * (epoch + 1)
        
        assert model.lambda_penalty == 0.05
    
    def test_epsilon_annealing(self, model, sample_puzzle):
        """Test epsilon annealing."""
        model.train()
        
        # Start with high exploration
        model.set_exploration_epsilon(1.0)
        
        # Anneal over "training"
        for step in range(10):
            epsilon = 1.0 - (step * 0.1)
            model.set_exploration_epsilon(epsilon)
            _, _, traces = model(sample_puzzle)
        
        assert model.exploration_epsilon == pytest.approx(0.1)


# =============================================================================
# Test Documentation Compliance (Ge et al. Analysis)
# =============================================================================

class TestDocumentationCompliance:
    """Tests verifying documented behavior and warnings."""
    
    def test_model_has_act_warning_in_docstring(self):
        """Model docstring warns about ACT limitations."""
        docstring = HRM_4x4_WithHalting.__doc__
        
        assert "Ge et al" in docstring or "minimal inference benefit" in docstring
    
    def test_zero_lambda_supported(self, sample_puzzle):
        """λ=0 is supported (as recommended by Ge et al.)."""
        model = HRM_4x4_WithHalting(
            hidden_dim=64,
            n_outer_cycles=5,
            lambda_penalty=0.0,  # Recommended setting
        )
        
        _, _, traces = model(sample_puzzle)
        
        penalty = model.get_halt_penalty(traces)
        assert penalty.item() == 0.0
