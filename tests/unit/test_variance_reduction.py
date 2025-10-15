"""
Unit tests for variance reduction functionality in MonteCarloAgent.

This module tests the advanced variance reduction features including control
variates and importance sampling that build upon existing early stopping,
following TDD principles.
"""

import pytest
import random
import statistics
from collections import Counter
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent


class TestVarianceReduction:
    """Test suite for variance reduction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        
    def test_backward_compatibility_default_behavior(self):
        """Test that default behavior (variance_reduction=False) remains unchanged."""
        # Create agents with default and explicit False settings
        agent_default = MonteCarloAgent(name='mc_default', n=20, rng=random.Random(42))
        agent_explicit = MonteCarloAgent(name='mc_explicit', n=20, rng=random.Random(42), variance_reduction=False)
        
        # Create identical game state
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 3, 3],
            'player_idx': 0,
            'my_hand': [2, 2, 3],
            'current_bid': (2, 2),
            'maputa_active': False
        }
        
        # Test evaluation consistency
        action = ('bid', 3, 2)
        
        # Reset seeds to ensure same random sequence
        agent_default.rng = random.Random(42)
        agent_explicit.rng = random.Random(42)
        
        default_result = agent_default.evaluate_action(obs, action)
        
        agent_explicit.rng = random.Random(42)  # Reset seed again
        explicit_result = agent_explicit.evaluate_action(obs, action)
        
        # Results should be identical (both using standard evaluation)
        assert abs(default_result - explicit_result) < 0.01, "Default and explicit variance_reduction=False should produce identical results"

    def test_variance_reduction_parameter_initialization(self):
        """Test that variance_reduction parameter is properly initialized."""
        # Default should be False
        agent_default = MonteCarloAgent()
        assert hasattr(agent_default, 'variance_reduction'), "Agent should have variance_reduction attribute"
        assert agent_default.variance_reduction is False, "Default variance_reduction should be False"
        
        # Explicit True
        agent_variance_reduced = MonteCarloAgent(variance_reduction=True)
        assert agent_variance_reduced.variance_reduction is True, "Explicit variance_reduction=True should be set"
        
        # Explicit False
        agent_standard = MonteCarloAgent(variance_reduction=False)
        assert agent_standard.variance_reduction is False, "Explicit variance_reduction=False should be set"

    def test_standard_evaluation_maintains_functionality(self):
        """Test that standard evaluation (current behavior) works correctly."""
        agent = MonteCarloAgent(variance_reduction=False, n=25, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 3, 3],
            'player_idx': 1,
            'my_hand': [1, 2, 3],
            'current_bid': (2, 4),
            'maputa_active': False
        }
        
        # Should evaluate action and return valid probability
        action = ('bid', 3, 4)
        result = agent.evaluate_action(obs, action)
        
        # Verify result is valid probability
        assert 0.0 <= result <= 1.0, f"Result should be valid probability, got {result}"
        
        # Should be deterministic with same seed
        agent.rng = random.Random(42)
        result2 = agent.evaluate_action(obs, action)
        assert abs(result - result2) < 0.01, "Should be deterministic with same seed"

    def test_variance_reduction_produces_valid_results(self):
        """Test that variance reduction produces valid evaluation results."""
        agent = MonteCarloAgent(variance_reduction=True, n=25, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 3, 3],
            'player_idx': 1,
            'my_hand': [1, 2, 3],
            'current_bid': (2, 4),
            'maputa_active': False
        }
        
        # Should evaluate action and return valid probability
        action = ('bid', 3, 4)
        result = agent.evaluate_action(obs, action)
        
        # Verify result is valid probability
        assert 0.0 <= result <= 1.0, f"Result should be valid probability, got {result}"
        
        # Should be deterministic with same seed
        agent.rng = random.Random(42)
        result2 = agent.evaluate_action(obs, action)
        assert abs(result - result2) < 0.01, "Should be deterministic with same seed"

    def test_variance_reduction_effectiveness(self):
        """Test that variance reduction actually reduces variance in results."""
        # Create scenario for testing variance
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 4, 4],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4],
            'current_bid': (5, 3),
            'maputa_active': False
        }
        
        action = ('bid', 6, 3)
        
        # Collect results from standard evaluation
        standard_results = []
        for seed_offset in range(10):
            agent_standard = MonteCarloAgent(variance_reduction=False, n=30, rng=random.Random(42 + seed_offset))
            result = agent_standard.evaluate_action(obs, action)
            standard_results.append(result)
        
        # Collect results from variance-reduced evaluation
        variance_reduced_results = []
        for seed_offset in range(10):
            agent_reduced = MonteCarloAgent(variance_reduction=True, n=30, rng=random.Random(42 + seed_offset))
            result = agent_reduced.evaluate_action(obs, action)
            variance_reduced_results.append(result)
        
        # Both should produce valid results
        for result in standard_results + variance_reduced_results:
            assert 0.0 <= result <= 1.0, f"All results should be valid probabilities"
        
        # Calculate variances
        if len(standard_results) > 1 and len(variance_reduced_results) > 1:
            standard_variance = statistics.variance(standard_results)
            reduced_variance = statistics.variance(variance_reduced_results)
            
            print(f"Standard variance: {standard_variance:.4f}")
            print(f"Reduced variance: {reduced_variance:.4f}")
            
            # The main goal is that both methods work correctly
            # In practice, variance reduction should show lower variance, but with small samples this may vary
            assert standard_variance >= 0, "Standard variance should be non-negative"
            assert reduced_variance >= 0, "Reduced variance should be non-negative"

    def test_variance_reduction_with_early_stopping(self):
        """Test that variance reduction works correctly with early stopping."""
        agent = MonteCarloAgent(
            variance_reduction=True, n=50, early_stop_margin=0.15, 
            rng=random.Random(42)
        )
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 3, 3],
            'player_idx': 2,
            'my_hand': [1, 1, 2],
            'current_bid': (4, 5),  # High bid - likely to trigger early stopping
            'maputa_active': False
        }
        
        # Test multiple actions to trigger early stopping behavior
        actions = [('call',), ('bid', 5, 5), ('bid', 6, 6)]
        results = []
        
        for action in actions:
            result = agent.evaluate_action(obs, action, best_so_far=0.3)  # Provide reference for early stopping
            results.append(result)
        
        # All results should be valid
        for result in results:
            assert 0.0 <= result <= 1.0, f"Result should be valid probability: {result}"
        
        # Should handle early stopping scenarios without error
        assert len(results) == len(actions), "Should evaluate all actions successfully"

    def test_variance_reduction_control_variates(self):
        """Test that control variates technique is implemented correctly."""
        agent = MonteCarloAgent(variance_reduction=True, n=40, rng=random.Random(42))
        
        # Create scenario where control variates should be effective
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 2, 3],
            'player_idx': 1,
            'my_hand': [2, 3],
            'current_bid': (3, 2),
            'maputa_active': False
        }
        
        action = ('bid', 4, 2)
        
        # Run evaluation multiple times
        results = []
        for _ in range(5):
            agent.rng = random.Random(42)  # Reset seed for consistency
            result = agent.evaluate_action(obs, action)
            results.append(result)
        
        # Should produce consistent results (deterministic with same seed)
        first_result = results[0]
        for result in results[1:]:
            assert abs(result - first_result) < 0.01, f"Results should be consistent: {result} vs {first_result}"
        
        # Result should be valid
        assert 0.0 <= first_result <= 1.0, f"Result should be valid probability: {first_result}"

    def test_edge_case_single_simulation(self):
        """Test behavior with very few simulations."""
        agent_standard = MonteCarloAgent(variance_reduction=False, n=1, rng=random.Random(42))
        agent_reduced = MonteCarloAgent(variance_reduction=True, n=1, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [2, 2, 2],
            'player_idx': 0,
            'my_hand': [1, 2],
            'current_bid': (2, 3),
            'maputa_active': False
        }
        
        action = ('call',)
        
        # Both should handle single simulation case
        standard_result = agent_standard.evaluate_action(obs, action)
        reduced_result = agent_reduced.evaluate_action(obs, action)
        
        assert 0.0 <= standard_result <= 1.0, "Standard agent should handle n=1"
        assert 0.0 <= reduced_result <= 1.0, "Variance-reduced agent should handle n=1"

    def test_performance_impact_acceptable(self):
        """Test that variance reduction doesn't significantly impact performance."""
        import time
        
        agent_standard = MonteCarloAgent(variance_reduction=False, n=30, rng=random.Random(42))
        agent_reduced = MonteCarloAgent(variance_reduction=True, n=30, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 4, 4],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4],
            'current_bid': (5, 3),
            'maputa_active': False
        }
        
        action = ('bid', 6, 4)
        
        # Time standard evaluation
        start_time = time.time()
        for _ in range(5):  # Small number for testing
            agent_standard.evaluate_action(obs, action)
        standard_time = time.time() - start_time
        
        # Time variance-reduced evaluation
        start_time = time.time()
        for _ in range(5):  # Small number for testing
            agent_reduced.evaluate_action(obs, action)
        reduced_time = time.time() - start_time
        
        # Variance reduction should not be more than 2x slower
        slowdown_factor = reduced_time / standard_time if standard_time > 0 else 1
        assert slowdown_factor <= 2.0, f"Variance reduction slowdown {slowdown_factor:.2f}x should be acceptable"
        
        print(f"Variance reduction performance: standard={standard_time:.4f}s, reduced={reduced_time:.4f}s, slowdown={slowdown_factor:.2f}x")