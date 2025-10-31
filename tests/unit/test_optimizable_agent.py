"""
Tests for OptimizableMCAgent class - TDD Cycle 1.2: Parameter Injection

This module tests runtime parameter modification and injection functionality
for the evolutionary optimization framework.
"""
import pytest
from agents.optimizable_mc_agent import OptimizableMCAgent
from agents.mc_agent import MonteCarloAgent


class TestOptimizableMCAgent:
    """Test suite for OptimizableMCAgent parameter injection."""
    
    @pytest.fixture
    def base_agent(self):
        """Create OptimizableMCAgent instance for testing."""
        return OptimizableMCAgent(
            name="test_agent",
            n=200,
            chunk_size=8,
            early_stop_margin=0.1,
            trust_learning_rate=0.1,
            history_memory_rounds=10,
            num_workers=2
        )
    
    def test_optimizable_agent_initialization(self, base_agent):
        """Test OptimizableMCAgent inherits from MonteCarloAgent properly."""
        # Should inherit from MonteCarloAgent
        assert isinstance(base_agent, MonteCarloAgent)
        assert isinstance(base_agent, OptimizableMCAgent)
        
        # Should have base MC agent properties
        assert base_agent.name == "test_agent"
        assert base_agent.N == 200
        assert base_agent.chunk_size == 8
        assert base_agent.early_stop_margin == 0.1
        assert base_agent.trust_learning_rate == 0.1
        assert base_agent.history_memory_rounds == 10
        
        # Should initialize utils parameters storage
        assert hasattr(base_agent, '_utils_params')
        assert isinstance(base_agent._utils_params, dict)
    
    def test_update_core_parameters(self, base_agent):
        """Test updating core MC agent parameters."""
        new_params = {
            'n': 500,
            'chunk_size': 16,
            'early_stop_margin': 0.2,
            'trust_learning_rate': 0.15,
            'history_memory_rounds': 15,
            'num_workers': 4
        }
        
        base_agent.update_parameters(new_params)
        
        # Check that core parameters were updated
        assert base_agent.N == 500
        assert base_agent.chunk_size == 16
        assert base_agent.early_stop_margin == 0.2
        assert base_agent.trust_learning_rate == 0.15
        assert base_agent.history_memory_rounds == 15
        assert base_agent.num_workers == 4
    
    def test_update_utils_microparameters(self, base_agent):
        """Test updating utils microparameters."""
        utils_params = {
            'decay_factor': 0.85,
            'scaling_factor': 2.5,
            'face_weight_boost': 2.2,
            'min_weight_threshold': 0.15
        }
        
        base_agent.update_parameters(utils_params)
        
        # Check that utils parameters were stored
        assert base_agent._utils_params['decay_factor'] == 0.85
        assert base_agent._utils_params['scaling_factor'] == 2.5
        assert base_agent._utils_params['face_weight_boost'] == 2.2
        assert base_agent._utils_params['min_weight_threshold'] == 0.15
    
    def test_update_mixed_parameters(self, base_agent):
        """Test updating both core and utils parameters together."""
        mixed_params = {
            # Core parameters
            'n': 800,
            'chunk_size': 12,
            'early_stop_margin': 0.25,
            # Utils microparameters
            'decay_factor': 0.75,
            'scaling_factor': 1.8,
            'face_weight_boost': 1.5,
            'min_weight_threshold': 0.08
        }
        
        base_agent.update_parameters(mixed_params)
        
        # Check core parameters
        assert base_agent.N == 800
        assert base_agent.chunk_size == 12
        assert base_agent.early_stop_margin == 0.25
        
        # Check utils parameters
        assert base_agent._utils_params['decay_factor'] == 0.75
        assert base_agent._utils_params['scaling_factor'] == 1.8
        assert base_agent._utils_params['face_weight_boost'] == 1.5
        assert base_agent._utils_params['min_weight_threshold'] == 0.08
    
    def test_get_utils_param_existing(self, base_agent):
        """Test getting existing utils parameter."""
        # Set some utils parameters
        base_agent.update_parameters({
            'decay_factor': 0.9,
            'scaling_factor': 2.8
        })
        
        # Should return stored values
        assert base_agent.get_utils_param('decay_factor', 0.8) == 0.9
        assert base_agent.get_utils_param('scaling_factor', 2.0) == 2.8
    
    def test_get_utils_param_default(self, base_agent):
        """Test getting non-existing utils parameter returns default."""
        # Parameter not set, should return default
        default_decay = 0.8
        default_scaling = 2.0
        
        assert base_agent.get_utils_param('decay_factor', default_decay) == default_decay
        assert base_agent.get_utils_param('scaling_factor', default_scaling) == default_scaling
    
    def test_parameter_type_conversion(self, base_agent):
        """Test that parameters are properly type-converted."""
        params_with_wrong_types = {
            'n': 500.7,  # Should convert to int
            'chunk_size': 16.9,  # Should convert to int
            'early_stop_margin': 0.2,  # Should stay float
            'trust_learning_rate': 0.15,  # Should stay float
            'history_memory_rounds': 12.3,  # Should convert to int
            'num_workers': 3.8  # Should convert to int
        }
        
        base_agent.update_parameters(params_with_wrong_types)
        
        # Check that int parameters were converted
        assert base_agent.N == 500  # Converted from 500.7
        assert isinstance(base_agent.N, int)
        assert base_agent.chunk_size == 16  # Converted from 16.9
        assert isinstance(base_agent.chunk_size, int)
        assert base_agent.history_memory_rounds == 12  # Converted from 12.3
        assert isinstance(base_agent.history_memory_rounds, int)
        assert base_agent.num_workers == 3  # Converted from 3.8
        assert isinstance(base_agent.num_workers, int)
        
        # Check that float parameters stayed float
        assert base_agent.early_stop_margin == 0.2
        assert isinstance(base_agent.early_stop_margin, float)
        assert base_agent.trust_learning_rate == 0.15
        assert isinstance(base_agent.trust_learning_rate, float)
    
    def test_parameter_bounds_enforcement(self, base_agent):
        """Test that parameter bounds are enforced."""
        # Test minimum bounds
        base_agent.update_parameters({'n': 50})  # Below minimum of 100
        assert base_agent.N >= 1  # Should be clamped to minimum of 1
        
        base_agent.update_parameters({'chunk_size': 0})  # Below minimum of 1
        assert base_agent.chunk_size >= 1  # Should be clamped to minimum of 1
        
        # Test that negative values are handled
        base_agent.update_parameters({'early_stop_margin': -0.1})
        assert base_agent.early_stop_margin >= 0  # Should handle negative values
    
    def test_partial_parameter_update(self, base_agent):
        """Test updating only some parameters leaves others unchanged."""
        # Set initial state
        initial_n = base_agent.N
        initial_chunk_size = base_agent.chunk_size
        initial_margin = base_agent.early_stop_margin
        
        # Update only one parameter
        base_agent.update_parameters({'n': 1000})
        
        # Check that only n was changed
        assert base_agent.N == 1000
        assert base_agent.chunk_size == initial_chunk_size
        assert base_agent.early_stop_margin == initial_margin
    
    def test_empty_parameter_update(self, base_agent):
        """Test updating with empty dictionary doesn't break anything."""
        # Store initial values
        initial_n = base_agent.N
        initial_chunk_size = base_agent.chunk_size
        initial_margin = base_agent.early_stop_margin
        
        # Update with empty dict
        base_agent.update_parameters({})
        
        # Should be unchanged
        assert base_agent.N == initial_n
        assert base_agent.chunk_size == initial_chunk_size
        assert base_agent.early_stop_margin == initial_margin
    
    def test_unknown_parameter_ignored(self, base_agent):
        """Test that unknown parameters are ignored gracefully."""
        # Store initial values
        initial_n = base_agent.N
        
        # Update with unknown parameter
        base_agent.update_parameters({
            'n': 600,
            'unknown_param': 999  # Should be ignored
        })
        
        # Known parameter should be updated, unknown ignored
        assert base_agent.N == 600
        assert not hasattr(base_agent, 'unknown_param')
    
    def test_backward_compatibility(self, base_agent):
        """Test that OptimizableMCAgent maintains backward compatibility."""
        # Should have all the methods of the base MonteCarloAgent
        assert hasattr(base_agent, 'select_action')
        assert hasattr(base_agent, 'evaluate_action')
        assert hasattr(base_agent, 'sample_determinization')
        
        # Should be callable like a normal MC agent
        assert callable(base_agent.select_action)
        assert callable(base_agent.evaluate_action)
        assert callable(base_agent.sample_determinization)
    
    def test_utils_params_integration(self, base_agent):
        """Test that utils parameters are properly accessible for mc_utils functions."""
        # Set utils parameters
        utils_params = {
            'decay_factor': 0.88,
            'scaling_factor': 2.3,
            'face_weight_boost': 2.1,
            'min_weight_threshold': 0.12
        }
        base_agent.update_parameters(utils_params)
        
        # Test that they can be retrieved with defaults
        assert base_agent.get_utils_param('decay_factor', 0.8) == 0.88
        assert base_agent.get_utils_param('scaling_factor', 2.0) == 2.3
        assert base_agent.get_utils_param('face_weight_boost', 2.0) == 2.1
        assert base_agent.get_utils_param('min_weight_threshold', 0.1) == 0.12
        
        # Test non-existent parameters return defaults
        assert base_agent.get_utils_param('nonexistent_param', 42) == 42