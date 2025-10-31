"""
Tests for mc_utils optimization integration - TDD Cycle 1.3: Utils Integration

This module tests dynamic parameter passing to mc_utils functions
for the evolutionary optimization framework.
"""
import pytest
from unittest.mock import Mock, patch
from agents.mc_utils import (
    calculate_recency_weight, sample_weighted_dice, 
    calculate_trust_weighted_accuracy
)
from agents.optimizable_mc_agent import OptimizableMCAgent


class TestMCUtilsOptimization:
    """Test suite for mc_utils parameter injection."""
    
    @pytest.fixture
    def optimizable_agent(self):
        """Create OptimizableMCAgent instance for testing."""
        return OptimizableMCAgent(
            name="test_agent",
            n=200,
            chunk_size=8,
            early_stop_margin=0.1
        )
    
    @pytest.fixture
    def mock_obs(self):
        """Create mock observation for testing."""
        obs = {
            'current_bid': (3, 2),  # bid qty=3, face=2
            'my_hand': [1, 2, 3, 4, 5],
            'dice_counts': [5, 4, 3, 2, 1],
            'maputa_active': False,
            '_simulator': Mock(ones_are_wild=True)
        }
        return obs
    
    def test_calculate_recency_weight_default_params(self):
        """Test calculate_recency_weight with default parameters."""
        # Test with default decay_factor (should be 0.8)
        current_round = 10
        bid_round = 8  # 2 rounds ago
        
        weight = calculate_recency_weight(bid_round, current_round)
        
        # Should use default decay_factor of 0.8
        expected_weight = max(0.1, 0.8 ** 2)  # decay_factor^rounds_ago
        assert abs(weight - expected_weight) < 0.001
    
    def test_calculate_recency_weight_custom_params(self, optimizable_agent):
        """Test calculate_recency_weight with custom decay_factor from agent."""
        # Set custom decay factor in agent
        optimizable_agent.update_parameters({'decay_factor': 0.9})
        
        current_round = 10
        bid_round = 7  # 3 rounds ago
        
        # Test the modified function that accepts agent parameter
        weight = calculate_recency_weight(bid_round, current_round, 
                                        decay_factor=optimizable_agent.get_utils_param('decay_factor', 0.8))
        
        # Should use custom decay_factor of 0.9
        expected_weight = max(0.1, 0.9 ** 3)  # custom_decay^rounds_ago
        assert abs(weight - expected_weight) < 0.001
    
    def test_sample_weighted_dice_default_params(self, optimizable_agent, mock_obs):
        """Test sample_weighted_dice with default parameters."""
        num_dice = 5
        
        # Mock the agent's random number generator for reproducibility
        optimizable_agent.rng.random = Mock(side_effect=[0.1, 0.3, 0.5, 0.7, 0.9])
        optimizable_agent.rng.randint = Mock(side_effect=[1, 2, 3, 4, 5])
        
        dice = sample_weighted_dice(optimizable_agent, mock_obs, num_dice)
        
        # Should return dice with proper length
        assert len(dice) == num_dice
        assert all(1 <= d <= 6 for d in dice)
    
    def test_sample_weighted_dice_custom_scaling_factor(self, optimizable_agent, mock_obs):
        """Test sample_weighted_dice with custom scaling_factor from agent."""
        # Set custom scaling factor in agent
        optimizable_agent.update_parameters({'scaling_factor': 2.5})
        
        num_dice = 3
        
        # Test that the function can access the custom parameter
        # We'll verify this by checking that the function doesn't crash
        # and returns valid dice (detailed logic testing would require more complex mocking)
        dice = sample_weighted_dice(optimizable_agent, mock_obs, num_dice)
        
        assert len(dice) == num_dice
        assert all(1 <= d <= 6 for d in dice)
    
    def test_sample_weighted_dice_custom_face_weight_boost(self, optimizable_agent, mock_obs):
        """Test sample_weighted_dice with custom face_weight_boost from agent."""
        # Set custom face weight boost in agent
        optimizable_agent.update_parameters({'face_weight_boost': 2.2})
        
        num_dice = 4
        
        dice = sample_weighted_dice(optimizable_agent, mock_obs, num_dice)
        
        assert len(dice) == num_dice
        assert all(1 <= d <= 6 for d in dice)
    
    def test_calculate_trust_weighted_accuracy_default_params(self):
        """Test calculate_trust_weighted_accuracy with default parameters."""
        # Create mock history and trust params
        mock_history = Mock()
        mock_history.entries = []
        mock_history.current_round = 10
        
        trust_params = [0.5, 0.6, 0.7, 0.8, 0.9]
        face = 2
        
        accuracy = calculate_trust_weighted_accuracy(mock_history, trust_params, face)
        
        # With no entries, should return default neutral accuracy
        assert accuracy == 0.5
    
    def test_calculate_trust_weighted_accuracy_with_entries(self):
        """Test calculate_trust_weighted_accuracy with historical entries."""
        # Create mock history with entries
        mock_entry = Mock()
        mock_entry.action = ('bid', 0, 2)  # bid action with face=2
        mock_entry.actual_hand = [1, 2, 2, 3, 4]  # hand with some face=2 dice
        mock_entry.player_idx = 0
        mock_entry.round_num = 8
        
        mock_history = Mock()
        mock_history.entries = [mock_entry]
        mock_history.current_round = 10
        
        trust_params = [0.8, 0.6, 0.7, 0.8, 0.9]  # trust for player 0 = 0.8
        face = 2
        
        accuracy = calculate_trust_weighted_accuracy(mock_history, trust_params, face)
        
        # Should return some calculated accuracy (not default 0.5)
        assert accuracy != 0.5
        assert 0.0 <= accuracy <= 1.0
    
    def test_mc_utils_functions_accept_agent_parameter(self, optimizable_agent, mock_obs):
        """Test that mc_utils functions can accept OptimizableMCAgent as parameter."""
        # This test verifies that we can pass the agent to utils functions
        # and they can extract parameters using get_utils_param
        
        # Set some custom parameters
        optimizable_agent.update_parameters({
            'decay_factor': 0.85,
            'scaling_factor': 2.3,
            'face_weight_boost': 2.1,
            'min_weight_threshold': 0.12
        })
        
        # Test that functions can extract parameters from agent
        decay_factor = optimizable_agent.get_utils_param('decay_factor', 0.8)
        scaling_factor = optimizable_agent.get_utils_param('scaling_factor', 2.0)
        face_weight_boost = optimizable_agent.get_utils_param('face_weight_boost', 2.0)
        min_weight_threshold = optimizable_agent.get_utils_param('min_weight_threshold', 0.1)
        
        # Verify custom values are retrieved
        assert decay_factor == 0.85
        assert scaling_factor == 2.3
        assert face_weight_boost == 2.1
        assert min_weight_threshold == 0.12
    
    def test_backward_compatibility_preserved(self, mock_obs):
        """Test that modified mc_utils functions maintain backward compatibility."""
        # Test that original function signatures still work
        
        # Test calculate_recency_weight with original parameters
        weight = calculate_recency_weight(5, 8)  # bid_round=5, current_round=8
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0
        
        # Test sample_weighted_dice with regular MC agent
        from agents.mc_agent import MonteCarloAgent
        regular_agent = MonteCarloAgent(n=100)
        
        dice = sample_weighted_dice(regular_agent, mock_obs, 5)
        assert len(dice) == 5
        assert all(1 <= d <= 6 for d in dice)
    
    def test_utils_parameter_fallback_to_defaults(self, optimizable_agent):
        """Test that utils functions fall back to defaults when parameters not set."""
        # Don't set any custom parameters
        
        # Function should use defaults when parameters not provided
        decay_factor = optimizable_agent.get_utils_param('decay_factor', 0.8)
        scaling_factor = optimizable_agent.get_utils_param('scaling_factor', 2.0)
        
        # Should return defaults since no custom values set
        assert decay_factor == 0.8  # default
        assert scaling_factor == 2.0  # default
    
    def test_multiple_parameter_types_integration(self, optimizable_agent, mock_obs):
        """Test integration of multiple parameter types (int, float)."""
        # Set mixed parameter types
        optimizable_agent.update_parameters({
            # Core parameters (various types)
            'n': 500,  # int
            'chunk_size': 12,  # int
            'early_stop_margin': 0.25,  # float
            # Utils microparameters (all float)
            'decay_factor': 0.88,  # float
            'scaling_factor': 1.7  # float
        })
        
        # Verify all parameters are accessible and properly typed
        assert optimizable_agent.N == 500
        assert isinstance(optimizable_agent.N, int)
        assert optimizable_agent.chunk_size == 12
        assert isinstance(optimizable_agent.chunk_size, int)
        assert optimizable_agent.early_stop_margin == 0.25
        assert isinstance(optimizable_agent.early_stop_margin, float)
        
        # Verify utils parameters
        assert optimizable_agent.get_utils_param('decay_factor', 0.8) == 0.88
        assert optimizable_agent.get_utils_param('scaling_factor', 2.0) == 1.7
    
    def test_parameter_injection_does_not_break_existing_functionality(self, optimizable_agent, mock_obs):
        """Test that parameter injection doesn't break existing MC agent functionality."""
        # Update parameters
        optimizable_agent.update_parameters({
            'n': 150,
            'chunk_size': 10,
            'decay_factor': 0.75
        })
        
        # Test that core MC functionality still works
        assert hasattr(optimizable_agent, 'sample_determinization')
        assert hasattr(optimizable_agent, 'evaluate_action')
        assert hasattr(optimizable_agent, 'select_action')
        
        # Test sample_determinization still works (basic smoke test)
        # This is a more complex test that would require a full simulation setup,
        # but we can at least verify the method exists and is callable
        assert callable(optimizable_agent.sample_determinization)
        
        # Verify parameter changes took effect
        assert optimizable_agent.N == 150
        assert optimizable_agent.chunk_size == 10
        assert optimizable_agent.get_utils_param('decay_factor', 0.8) == 0.75