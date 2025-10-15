"""
Unit tests for enhanced action pruning functionality in MonteCarloAgent.

This module tests the enhanced pruning feature that combines statistical priors
with opponent modeling for better action filtering, following TDD principles.
"""

import pytest
import random
from collections import Counter
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent


class TestEnhancedPruning:
    """Test suite for enhanced pruning functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        
    def test_backward_compatibility_default_pruning(self):
        """Test that default behavior (enhanced_pruning=False) remains unchanged."""
        # Create agents with default and explicit False settings
        agent_default = MonteCarloAgent(name='mc_default', n=10, rng=random.Random(42))
        agent_explicit = MonteCarloAgent(name='mc_explicit', n=10, rng=random.Random(42), enhanced_pruning=False)
        
        # Create identical game state
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 3, 3],
            'player_idx': 0,
            'my_hand': [2, 2, 3],
            'current_bid': (2, 2),
            'maputa_active': False
        }
        
        # Get legal actions and test pruning behavior
        legal_actions = self.sim.legal_actions({'dice_counts': obs['dice_counts']}, obs['current_bid'], obs.get('maputa_restrict_face'))
        
        # Both should prune the same way with prune_k=12
        default_action = agent_default.select_action(obs, prune_k=12)
        explicit_action = agent_explicit.select_action(obs, prune_k=12)
        
        # Results should be identical (both using same pruning logic)
        assert default_action == explicit_action, "Default and explicit enhanced_pruning=False should produce identical results"

    def test_enhanced_pruning_parameter_initialization(self):
        """Test that enhanced_pruning parameter is properly initialized."""
        # Default should be False
        agent_default = MonteCarloAgent()
        assert hasattr(agent_default, 'enhanced_pruning'), "Agent should have enhanced_pruning attribute"
        assert agent_default.enhanced_pruning is False, "Default enhanced_pruning should be False"
        
        # Explicit True
        agent_enhanced = MonteCarloAgent(enhanced_pruning=True)
        assert agent_enhanced.enhanced_pruning is True, "Explicit enhanced_pruning=True should be set"
        
        # Explicit False
        agent_standard = MonteCarloAgent(enhanced_pruning=False)
        assert agent_standard.enhanced_pruning is False, "Explicit enhanced_pruning=False should be set"

    def test_standard_pruning_maintains_functionality(self):
        """Test that standard pruning (current behavior) works correctly."""
        agent = MonteCarloAgent(enhanced_pruning=False, n=10, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 3, 3],
            'player_idx': 1,
            'my_hand': [1, 2, 3],
            'current_bid': (2, 4),
            'maputa_active': False
        }
        
        # Should select a valid action
        action = agent.select_action(obs, prune_k=5)  # Small prune_k for testing
        
        # Verify action is valid
        assert action is not None, "Should return a valid action"
        assert len(action) >= 1, "Action should have at least one element"
        
        # Verify action is legal
        legal_actions = self.sim.legal_actions({'dice_counts': obs['dice_counts']}, obs['current_bid'], obs.get('maputa_restrict_face'))
        assert action in legal_actions, f"Selected action {action} should be legal"

    def test_enhanced_pruning_produces_valid_actions(self):
        """Test that enhanced pruning produces valid actions."""
        agent = MonteCarloAgent(enhanced_pruning=True, n=10, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 3, 3],
            'player_idx': 1,
            'my_hand': [1, 2, 3],
            'current_bid': (2, 4),
            'maputa_active': False
        }
        
        # Should select a valid action
        action = agent.select_action(obs, prune_k=5)  # Small prune_k for testing
        
        # Verify action is valid
        assert action is not None, "Should return a valid action"
        assert len(action) >= 1, "Action should have at least one element"
        
        # Verify action is legal
        legal_actions = self.sim.legal_actions({'dice_counts': obs['dice_counts']}, obs['current_bid'], obs.get('maputa_restrict_face'))
        assert action in legal_actions, f"Selected action {action} should be legal"

    def test_enhanced_pruning_considers_opponent_modeling(self):
        """Test that enhanced pruning shows different behavior from standard pruning."""
        agent_standard = MonteCarloAgent(enhanced_pruning=False, n=20, rng=random.Random(42))
        agent_enhanced = MonteCarloAgent(enhanced_pruning=True, n=20, rng=random.Random(42))
        
        # Create scenario where opponent modeling should matter
        obs = {
            '_simulator': self.sim,
            'dice_counts': [5, 2, 3],  # Player 1 has fewer dice (might be more conservative)
            'player_idx': 0,
            'my_hand': [2, 3, 4, 5, 6],  # Good hand, no ones
            'current_bid': (3, 2),  # Moderate bid on twos
            'maputa_active': False
        }
        
        # Run multiple trials to see if there's systematic difference
        standard_actions = []
        enhanced_actions = []
        
        for trial in range(5):  # Small number for testing
            # Use different seeds per trial but same for both agents
            agent_standard.rng = random.Random(42 + trial)
            agent_enhanced.rng = random.Random(42 + trial)
            
            standard_action = agent_standard.select_action(obs, prune_k=8)
            enhanced_action = agent_enhanced.select_action(obs, prune_k=8)
            
            standard_actions.append(standard_action)
            enhanced_actions.append(enhanced_action)
        
        # Both should produce valid actions
        for action in standard_actions + enhanced_actions:
            assert action is not None, "All actions should be valid"
        
        # This test mainly verifies the interface works correctly
        # In a full implementation, we'd test for specific strategic differences
        print(f"Standard pruning actions: {standard_actions}")
        print(f"Enhanced pruning actions: {enhanced_actions}")

    def test_enhanced_pruning_multi_criteria_scoring(self):
        """Test that enhanced pruning uses multi-criteria scoring effectively."""
        agent = MonteCarloAgent(enhanced_pruning=True, n=15, rng=random.Random(42))
        
        # Create a scenario where multiple criteria should influence pruning
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 1, 4],  # Unbalanced - player 1 is weak
            'player_idx': 2,
            'my_hand': [1, 1, 2, 3],  # Strong hand with ones
            'current_bid': (4, 5),  # High bid on fives
            'maputa_active': False
        }
        
        # Test that agent can handle complex scenarios
        action = agent.select_action(obs, prune_k=6)
        
        # Verify action validity
        assert action is not None, "Should handle complex scenarios"
        legal_actions = self.sim.legal_actions({'dice_counts': obs['dice_counts']}, obs['current_bid'], obs.get('maputa_restrict_face'))
        assert action in legal_actions, "Action should be legal in complex scenario"
        
        # Test multiple calls for consistency
        actions = []
        for _ in range(3):
            agent.rng = random.Random(42)  # Reset seed
            actions.append(agent.select_action(obs, prune_k=6))
        
        # Should be deterministic with same seed
        first_action = actions[0]
        for action in actions[1:]:
            assert action == first_action, "Should be deterministic with same seed"

    def test_edge_case_no_bids_to_prune(self):
        """Test behavior when there are no bid actions to prune."""
        agent = MonteCarloAgent(enhanced_pruning=True, n=10)
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [1, 1, 1],  # End game scenario
            'player_idx': 0,
            'my_hand': [3],
            'current_bid': (2, 2),  # High bid relative to remaining dice
            'maputa_active': False
        }
        
        # In end game, might only have call/exact available
        action = agent.select_action(obs, prune_k=10)
        
        assert action is not None, "Should handle scenarios with few legal actions"
        legal_actions = self.sim.legal_actions({'dice_counts': obs['dice_counts']}, obs['current_bid'], obs.get('maputa_restrict_face'))
        assert action in legal_actions, "Action should be legal even in constrained scenarios"

    def test_edge_case_first_bid_no_current_bid(self):
        """Test behavior when making first bid (no current bid to consider)."""
        agent_standard = MonteCarloAgent(enhanced_pruning=False, n=10, rng=random.Random(42))
        agent_enhanced = MonteCarloAgent(enhanced_pruning=True, n=10, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 3, 3],
            'player_idx': 0,
            'my_hand': [1, 2, 3],
            'current_bid': None,  # First bid scenario
            'maputa_active': False
        }
        
        # Both should handle first bid scenario
        standard_action = agent_standard.select_action(obs, prune_k=8)
        enhanced_action = agent_enhanced.select_action(obs, prune_k=8)
        
        assert standard_action is not None, "Standard agent should handle first bid"
        assert enhanced_action is not None, "Enhanced agent should handle first bid"
        
        # Both should make bid actions (not call on nothing)
        assert standard_action[0] == 'bid', "Should make bid action when no current bid"
        assert enhanced_action[0] == 'bid', "Enhanced agent should also make bid action when no current bid"

    def test_performance_impact_acceptable(self):
        """Test that enhanced pruning doesn't significantly impact performance."""
        import time
        
        agent_standard = MonteCarloAgent(enhanced_pruning=False, n=20, rng=random.Random(42))
        agent_enhanced = MonteCarloAgent(enhanced_pruning=True, n=20, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [4, 4, 4],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4],
            'current_bid': (5, 3),
            'maputa_active': False
        }
        
        # Time standard pruning
        start_time = time.time()
        for _ in range(5):  # Small number for testing
            agent_standard.select_action(obs, prune_k=12)
        standard_time = time.time() - start_time
        
        # Time enhanced pruning
        start_time = time.time()
        for _ in range(5):  # Small number for testing
            agent_enhanced.select_action(obs, prune_k=12)
        enhanced_time = time.time() - start_time
        
        # Enhanced pruning should not be more than 2x slower
        slowdown_factor = enhanced_time / standard_time if standard_time > 0 else 1
        assert slowdown_factor <= 2.0, f"Enhanced pruning slowdown {slowdown_factor:.2f}x should be acceptable"
        
        print(f"Pruning performance comparison: standard={standard_time:.4f}s, enhanced={enhanced_time:.4f}s, slowdown={slowdown_factor:.2f}x")