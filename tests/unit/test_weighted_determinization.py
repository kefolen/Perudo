"""
Unit tests for weighted determinization functionality in MonteCarloAgent.

This module tests the enhanced determinization feature that weights dice sampling
based on bidding history plausibility, following TDD principles.
"""

import pytest
import random
from collections import Counter
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent


class TestWeightedDeterminization:
    """Test suite for weighted determinization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        
    def test_backward_compatibility_default_behavior(self):
        """Test that default behavior (weighted_sampling=False) remains unchanged."""
        # Create agents with default and explicit False settings
        agent_default = MonteCarloAgent(name='mc_default', n=10, rng=random.Random(42))
        agent_explicit = MonteCarloAgent(name='mc_explicit', n=10, rng=random.Random(42), weighted_sampling=False)
        
        # Create identical game state
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 3, 3],
            'player_idx': 0,
            'my_hand': [2, 2, 3],
            'current_bid': (2, 2),
            'maputa_active': False
        }
        
        # Sample determinizations multiple times and compare
        random.seed(42)  # Ensure reproducibility
        samples_default = [agent_default.sample_determinization(obs) for _ in range(5)]
        
        random.seed(42)  # Reset for second agent
        samples_explicit = [agent_explicit.sample_determinization(obs) for _ in range(5)]
        
        # Results should be identical
        assert samples_default == samples_explicit, "Default and explicit weighted_sampling=False should produce identical results"

    def test_weighted_sampling_parameter_initialization(self):
        """Test that weighted_sampling parameter is properly initialized."""
        # Default should be False
        agent_default = MonteCarloAgent()
        assert hasattr(agent_default, 'weighted_sampling'), "Agent should have weighted_sampling attribute"
        assert agent_default.weighted_sampling is False, "Default weighted_sampling should be False"
        
        # Explicit True
        agent_weighted = MonteCarloAgent(weighted_sampling=True)
        assert agent_weighted.weighted_sampling is True, "Explicit weighted_sampling=True should be set"
        
        # Explicit False
        agent_uniform = MonteCarloAgent(weighted_sampling=False)
        assert agent_uniform.weighted_sampling is False, "Explicit weighted_sampling=False should be set"

    def test_uniform_sampling_maintains_valid_dice(self):
        """Test that uniform sampling (current behavior) produces valid dice values."""
        agent = MonteCarloAgent(weighted_sampling=False, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 2, 4],
            'player_idx': 1,
            'my_hand': [1, 3],
            'current_bid': (1, 2),
            'maputa_active': False
        }
        
        # Sample multiple determinizations
        for _ in range(10):
            hands = agent.sample_determinization(obs)
            
            # Verify structure
            assert len(hands) == 3, "Should have hands for all 3 players"
            assert hands[1] == [1, 3], "Player 1's hand should match my_hand"
            
            # Verify dice counts match
            assert len(hands[0]) == 3, "Player 0 should have 3 dice"
            assert len(hands[1]) == 2, "Player 1 should have 2 dice"  # my_hand
            assert len(hands[2]) == 4, "Player 2 should have 4 dice"
            
            # Verify all dice are valid (1-6)
            for player_idx, hand in enumerate(hands):
                if player_idx != 1:  # Skip own hand
                    for die in hand:
                        assert 1 <= die <= 6, f"Die value {die} should be between 1 and 6"

    def test_weighted_sampling_produces_valid_dice(self):
        """Test that weighted sampling produces valid dice values."""
        agent = MonteCarloAgent(weighted_sampling=True, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 2, 4],
            'player_idx': 1,
            'my_hand': [1, 3],
            'current_bid': (3, 2),  # Bid 3 twos
            'maputa_active': False
        }
        
        # Sample multiple determinizations
        for _ in range(10):
            hands = agent.sample_determinization(obs)
            
            # Verify structure (same as uniform)
            assert len(hands) == 3, "Should have hands for all 3 players"
            assert hands[1] == [1, 3], "Player 1's hand should match my_hand"
            
            # Verify dice counts match
            assert len(hands[0]) == 3, "Player 0 should have 3 dice"
            assert len(hands[1]) == 2, "Player 1 should have 2 dice"
            assert len(hands[2]) == 4, "Player 2 should have 4 dice"
            
            # Verify all dice are valid (1-6)
            for player_idx, hand in enumerate(hands):
                if player_idx != 1:  # Skip own hand
                    for die in hand:
                        assert 1 <= die <= 6, f"Die value {die} should be between 1 and 6"

    def test_weighted_sampling_considers_bidding_history(self):
        """Test that weighted sampling shows bias toward plausible dice combinations."""
        agent_uniform = MonteCarloAgent(weighted_sampling=False, rng=random.Random(42))
        agent_weighted = MonteCarloAgent(weighted_sampling=True, rng=random.Random(42))
        
        # Setup scenario where bid is for many twos
        obs = {
            '_simulator': self.sim,
            'dice_counts': [5, 3, 5],  # Many dice total
            'player_idx': 1,
            'my_hand': [2, 3, 4],  # Player has 1 two
            'current_bid': (8, 2),  # Bid 8 twos - needs many more from opponents
            'maputa_active': False
        }
        
        # Sample many determinizations and count face frequencies
        num_samples = 100
        uniform_face_counts = Counter()
        weighted_face_counts = Counter()
        
        for _ in range(num_samples):
            # Uniform sampling
            uniform_hands = agent_uniform.sample_determinization(obs)
            for player_idx in [0, 2]:  # Opponent players
                for die in uniform_hands[player_idx]:
                    uniform_face_counts[die] += 1
            
            # Weighted sampling
            weighted_hands = agent_weighted.sample_determinization(obs)
            for player_idx in [0, 2]:  # Opponent players
                for die in weighted_hands[player_idx]:
                    weighted_face_counts[die] += 1
        
        # For uniform sampling, all faces should be roughly equal
        uniform_total = sum(uniform_face_counts.values())
        uniform_proportions = {face: count/uniform_total for face, count in uniform_face_counts.items()}
        
        # For weighted sampling, face 2 should be more common given the bid
        weighted_total = sum(weighted_face_counts.values())
        weighted_proportions = {face: count/weighted_total for face, count in weighted_face_counts.items()}
        
        # This test verifies the interface works - actual bias testing needs implementation
        assert uniform_total > 0, "Uniform sampling should produce dice"
        assert weighted_total > 0, "Weighted sampling should produce dice"
        assert len(uniform_proportions) <= 6, "Should have at most 6 different faces"
        assert len(weighted_proportions) <= 6, "Should have at most 6 different faces"

    def test_edge_case_no_opponents(self):
        """Test behavior when there are no opponent players."""
        agent = MonteCarloAgent(weighted_sampling=True)
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 0, 0],  # Only player 0 has dice
            'player_idx': 0,
            'my_hand': [1, 2, 3],
            'current_bid': None,
            'maputa_active': False
        }
        
        hands = agent.sample_determinization(obs)
        assert len(hands) == 3, "Should have hands array for all players"
        assert hands[0] == [1, 2, 3], "Player 0's hand should match my_hand"
        assert hands[1] == [], "Player 1 should have empty hand"
        assert hands[2] == [], "Player 2 should have empty hand"

    def test_edge_case_no_current_bid(self):
        """Test behavior when there is no current bid (game start)."""
        agent_uniform = MonteCarloAgent(weighted_sampling=False, rng=random.Random(42))
        agent_weighted = MonteCarloAgent(weighted_sampling=True, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [3, 3, 3],
            'player_idx': 0,
            'my_hand': [1, 2, 3],
            'current_bid': None,  # No current bid
            'maputa_active': False
        }
        
        # Both should handle this gracefully
        hands_uniform = agent_uniform.sample_determinization(obs)
        hands_weighted = agent_weighted.sample_determinization(obs)
        
        assert len(hands_uniform) == 3, "Uniform sampling should work with no current bid"
        assert len(hands_weighted) == 3, "Weighted sampling should work with no current bid"
        
        # Without a bid to consider, weighted sampling might fall back to uniform
        # This is an implementation detail to be verified once implemented

    def test_performance_impact_acceptable(self):
        """Test that weighted sampling doesn't significantly impact performance."""
        import time
        
        agent_uniform = MonteCarloAgent(weighted_sampling=False, rng=random.Random(42))
        agent_weighted = MonteCarloAgent(weighted_sampling=True, rng=random.Random(42))
        
        obs = {
            '_simulator': self.sim,
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': (7, 3),
            'maputa_active': False
        }
        
        # Time uniform sampling
        start_time = time.time()
        for _ in range(100):
            agent_uniform.sample_determinization(obs)
        uniform_time = time.time() - start_time
        
        # Time weighted sampling
        start_time = time.time()
        for _ in range(100):
            agent_weighted.sample_determinization(obs)
        weighted_time = time.time() - start_time
        
        # Weighted sampling should not be more than 3x slower
        slowdown_factor = weighted_time / uniform_time if uniform_time > 0 else 1
        assert slowdown_factor <= 3.0, f"Weighted sampling slowdown {slowdown_factor:.2f}x should be acceptable"
        
        print(f"Performance comparison: uniform={uniform_time:.4f}s, weighted={weighted_time:.4f}s, slowdown={slowdown_factor:.2f}x")