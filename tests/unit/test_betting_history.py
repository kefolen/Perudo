"""
Unit tests for betting history tracking functionality.

Tests the BettingHistoryEntry, GameBettingHistory, and PlayerTrustManager classes
along with related utility functions for collective plausibility and Bayesian sampling.
"""

import unittest
import random
from agents.mc_utils import (
    BettingHistoryEntry, GameBettingHistory, PlayerTrustManager,
    calculate_recency_weight, calculate_trust_weighted_accuracy,
    compute_collective_plausibility, sample_bayesian_player_dice,
    sample_dice_with_probabilities
)
from agents.mc_agent import MonteCarloAgent
from sim.perudo import PerudoSimulator


class TestBettingHistoryEntry(unittest.TestCase):
    """Test BettingHistoryEntry class."""
    
    def test_entry_creation(self):
        """Test basic entry creation and attribute access."""
        entry = BettingHistoryEntry(
            player_idx=0,
            action=('bid', 3, 2),
            round_num=1,
            dice_count=4,
            actual_hand=[1, 2, 2, 5],
            bid_result='won_round'
        )
        
        self.assertEqual(entry.player_idx, 0)
        self.assertEqual(entry.action, ('bid', 3, 2))
        self.assertEqual(entry.round_num, 1)
        self.assertEqual(entry.dice_count, 4)
        self.assertEqual(entry.actual_hand, [1, 2, 2, 5])
        self.assertEqual(entry.bid_result, 'won_round')
    
    def test_entry_partial_data(self):
        """Test entry creation with minimal required data."""
        entry = BettingHistoryEntry(
            player_idx=1,
            action=('call',),
            round_num=0,
            dice_count=3
        )
        
        self.assertEqual(entry.player_idx, 1)
        self.assertEqual(entry.action, ('call',))
        self.assertIsNone(entry.actual_hand)
        self.assertIsNone(entry.bid_result)


class TestGameBettingHistory(unittest.TestCase):
    """Test GameBettingHistory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.history = GameBettingHistory(num_players=3)
        
        # Add some test entries
        self.entry1 = BettingHistoryEntry(0, ('bid', 2, 3), 0, 5, [1, 3, 3, 4, 6])
        self.entry2 = BettingHistoryEntry(1, ('bid', 3, 3), 0, 4, [2, 3, 3, 5])
        self.entry3 = BettingHistoryEntry(2, ('call',), 0, 3, [1, 4, 6])
        self.entry4 = BettingHistoryEntry(0, ('bid', 1, 1), 1, 4, [1, 2, 4, 5])
        
        self.history.add_entry(self.entry1)
        self.history.add_entry(self.entry2)
        self.history.add_entry(self.entry3)
        self.history.add_entry(self.entry4)
    
    def test_history_initialization(self):
        """Test history initialization."""
        new_history = GameBettingHistory(num_players=4)
        self.assertEqual(new_history.num_players, 4)
        self.assertEqual(len(new_history.entries), 0)
        self.assertEqual(len(new_history.player_entries), 4)
        self.assertEqual(len(new_history.round_entries), 0)
        self.assertEqual(new_history.current_round, 0)
    
    def test_add_entry(self):
        """Test adding entries to history."""
        self.assertEqual(len(self.history.entries), 4)
        self.assertEqual(len(self.history.player_entries[0]), 2)  # Player 0 has 2 entries
        self.assertEqual(len(self.history.player_entries[1]), 1)  # Player 1 has 1 entry
        self.assertEqual(len(self.history.player_entries[2]), 1)  # Player 2 has 1 entry
        self.assertEqual(len(self.history.round_entries), 2)      # 2 rounds of entries
    
    def test_get_player_bids(self):
        """Test getting player bids."""
        player_0_bids = self.history.get_player_bids(0)
        self.assertEqual(len(player_0_bids), 2)
        self.assertEqual(player_0_bids[0].action, ('bid', 2, 3))
        self.assertEqual(player_0_bids[1].action, ('bid', 1, 1))
        
        # Test filtering by face
        face_3_bids = self.history.get_player_bids(0, face=3)
        self.assertEqual(len(face_3_bids), 1)
        self.assertEqual(face_3_bids[0].action, ('bid', 2, 3))
    
    def test_get_face_popularity(self):
        """Test face popularity calculation."""
        # Face 3 was bid on twice out of 3 total bids
        popularity_3 = self.history.get_face_popularity(3)
        expected = 2.0 / 3.0  # 2 bids on face 3 out of 3 total bids
        self.assertAlmostEqual(popularity_3, expected, places=5)
        
        # Face 1 was bid on once out of 3 total bids
        popularity_1 = self.history.get_face_popularity(1)
        expected = 1.0 / 3.0  # 1 bid on face 1 out of 3 total bids
        self.assertAlmostEqual(popularity_1, expected, places=5)
        
        # Face 2 was never bid on
        popularity_2 = self.history.get_face_popularity(2)
        expected = 0.0  # 0 bids on face 2
        self.assertAlmostEqual(popularity_2, expected, places=5)
    
    def test_get_face_popularity_recent(self):
        """Test face popularity with recent rounds filter."""
        # Only consider round 1 (most recent)
        self.history.current_round = 1
        popularity_1_recent = self.history.get_face_popularity(1, recent_rounds=1)
        expected = 1.0 / 1.0  # 1 bid on face 1 out of 1 total bid in round 1
        self.assertAlmostEqual(popularity_1_recent, expected, places=5)
        
        popularity_3_recent = self.history.get_face_popularity(3, recent_rounds=1)
        expected = 0.0  # 0 bids on face 3 in round 1
        self.assertAlmostEqual(popularity_3_recent, expected, places=5)
    
    def test_get_round_entries(self):
        """Test getting entries by round."""
        round_0_entries = self.history.get_round_entries(0)
        self.assertEqual(len(round_0_entries), 3)
        
        round_1_entries = self.history.get_round_entries(1)
        self.assertEqual(len(round_1_entries), 1)
        
        # Test getting last round
        last_round_entries = self.history.get_round_entries(-1)
        self.assertEqual(len(last_round_entries), 1)  # Should be round 1
    
    def test_analyze_player_accuracy(self):
        """Test player accuracy analysis."""
        # Player 0 accuracy: First bid on face 3 has 2 face-3 dice + 1 one (3/5 = 0.6, scaled to 1.0)
        # Second bid on face 1 has 1 face-1 die (1/4 = 0.25, scaled to 0.5)
        # Average: (1.0 + 0.5) / 2 = 0.75
        accuracy_0 = self.history.analyze_player_accuracy(0)
        expected = (min(1.0, (3/5) * 2) + min(1.0, (1/4) * 2)) / 2
        self.assertAlmostEqual(accuracy_0, expected, places=5)
        
        # Player with no bids should return default
        accuracy_empty = GameBettingHistory(1).analyze_player_accuracy(0)
        self.assertEqual(accuracy_empty, 0.5)


class TestPlayerTrustManager(unittest.TestCase):
    """Test PlayerTrustManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trust_manager = PlayerTrustManager(num_players=3, initial_trust=0.5)
        self.history = GameBettingHistory(num_players=3)
        
        # Add test entries with actual hands
        entry1 = BettingHistoryEntry(0, ('bid', 2, 3), 0, 5, [1, 3, 3, 4, 6])
        entry2 = BettingHistoryEntry(1, ('bid', 3, 4), 0, 4, [2, 4, 4, 5])
        
        self.history.add_entry(entry1)
        self.history.add_entry(entry2)
    
    def test_initialization(self):
        """Test trust manager initialization."""
        self.assertEqual(len(self.trust_manager.trust_params), 3)
        self.assertEqual(self.trust_manager.trust_params[0], 0.5)
        self.assertEqual(len(self.trust_manager.accuracy_history), 3)
    
    def test_calculate_round_accuracy(self):
        """Test round accuracy calculation."""
        # Player 0: bid on face 3, has 2 face-3 dice + 1 one (wild)
        # Accuracy = min(1.0, (3/5) * 2) = 1.0
        accuracy = self.trust_manager.calculate_round_accuracy(0, None, self.history)
        expected = min(1.0, (3/5) * 2)
        self.assertAlmostEqual(accuracy, expected, places=5)
        
        # Player 1: bid on face 4, has 2 face-4 dice
        # Accuracy = min(1.0, (2/4) * 2) = 1.0
        accuracy = self.trust_manager.calculate_round_accuracy(1, None, self.history)
        expected = min(1.0, (2/4) * 2)
        self.assertAlmostEqual(accuracy, expected, places=5)
        
        # Player with no bids should return None
        accuracy = self.trust_manager.calculate_round_accuracy(2, None, self.history)
        self.assertIsNone(accuracy)
    
    def test_update_trust_after_round(self):
        """Test trust parameter updates."""
        initial_trust = self.trust_manager.trust_params[0]
        
        # Mock round result
        round_result = {'loser': 2, 'bid_true': True, 'actual_count': 3, 'bid': (2, 3)}
        
        # Update trust
        self.trust_manager.update_trust_after_round(round_result, self.history)
        
        # Trust should have changed for players with bids
        self.assertNotEqual(self.trust_manager.trust_params[0], initial_trust)
        
        # Check that trust is in valid range
        for trust in self.trust_manager.trust_params:
            self.assertGreaterEqual(trust, 0.0)
            self.assertLessEqual(trust, 1.0)
    
    def test_get_trust(self):
        """Test getting trust values."""
        self.assertEqual(self.trust_manager.get_trust(0), 0.5)
        self.assertEqual(self.trust_manager.get_trust(1), 0.5)
        self.assertEqual(self.trust_manager.get_trust(2), 0.5)
        
        # Out of bounds should return default
        self.assertEqual(self.trust_manager.get_trust(10), 0.5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for betting history analysis."""
    
    def test_calculate_recency_weight(self):
        """Test recency weight calculation."""
        # Same round should have weight 1.0
        weight = calculate_recency_weight(5, 5)
        self.assertEqual(weight, 1.0)
        
        # One round ago should have weight 0.8
        weight = calculate_recency_weight(4, 5)
        self.assertAlmostEqual(weight, 0.8, places=5)
        
        # Two rounds ago should have weight 0.64
        weight = calculate_recency_weight(3, 5)
        self.assertAlmostEqual(weight, 0.64, places=5)
        
        # Very old rounds should have minimum weight
        weight = calculate_recency_weight(0, 10)
        self.assertGreaterEqual(weight, 0.1)
    
    def test_sample_dice_with_probabilities(self):
        """Test dice sampling with custom probabilities."""
        rng = random.Random(42)
        
        # Test with uniform probabilities
        uniform_probs = [1/6] * 6
        dice = sample_dice_with_probabilities(rng, uniform_probs, 100)
        self.assertEqual(len(dice), 100)
        for die in dice:
            self.assertIn(die, [1, 2, 3, 4, 5, 6])
        
        # Test with biased probabilities (heavily favor face 1)
        biased_probs = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        dice = sample_dice_with_probabilities(rng, biased_probs, 1000)
        face_1_count = dice.count(1)
        # Should have roughly 900 face-1 dice (allow some variance)
        self.assertGreater(face_1_count, 800)
    
    def test_compute_collective_plausibility_fallback(self):
        """Test collective plausibility without betting history."""
        # Create mock observation without betting history
        obs = {
            'dice_counts': [3, 3, 3],
            'my_hand': [1, 2, 3],
            'player_idx': 0,
            'maputa_active': False,
            '_simulator': PerudoSimulator()
        }
        
        # Should fall back to base probability calculation
        plausibility = compute_collective_plausibility(obs, 3, 2)
        self.assertIsInstance(plausibility, float)
        self.assertGreaterEqual(plausibility, 0.0)
        self.assertLessEqual(plausibility, 1.0)


class TestBayesianSampling(unittest.TestCase):
    """Test Bayesian player dice sampling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MonteCarloAgent(n=100, bayesian_sampling=True)
        self.history = GameBettingHistory(num_players=3)
        self.trust_manager = PlayerTrustManager(num_players=3)
        
        # Add betting history for player 1 (favors face 3)
        for i in range(3):
            entry = BettingHistoryEntry(1, ('bid', 2, 3), i, 4, [1, 3, 3, 5])
            self.history.add_entry(entry)
    
    def test_bayesian_sampling_with_history(self):
        """Test Bayesian sampling with betting history."""
        obs = {
            'betting_history': self.history,
            'player_trust': self.trust_manager.trust_params,
            'current_round': 3
        }
        
        # Sample dice for player 1 who has history of bidding on face 3
        dice = sample_bayesian_player_dice(self.agent, obs, 1, 100)
        self.assertEqual(len(dice), 100)
        
        # Should have higher proportion of face 3 due to bidding history
        face_3_count = dice.count(3)
        # Should be more than random (16.67% = 16-17 out of 100)
        self.assertGreater(face_3_count, 20)
    
    def test_bayesian_sampling_fallback(self):
        """Test Bayesian sampling fallback behavior."""
        # Test without history
        obs = {}
        dice = sample_bayesian_player_dice(self.agent, obs, 0, 20)
        self.assertEqual(len(dice), 20)
        
        # Test with weighted sampling agent
        weighted_agent = MonteCarloAgent(weighted_sampling=True, bayesian_sampling=True)
        obs = {'current_bid': (3, 2), '_simulator': PerudoSimulator(), 'my_hand': [1, 2], 'dice_counts': [2, 3, 3]}
        dice = sample_bayesian_player_dice(weighted_agent, obs, 1, 10)
        self.assertEqual(len(dice), 10)


if __name__ == '__main__':
    unittest.main()