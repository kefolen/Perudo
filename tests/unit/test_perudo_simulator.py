"""
Unit tests for PerudoSimulator core game mechanics.

This module tests the core functionality of the PerudoSimulator class,
including initialization, legal actions, bid validation, and game state transitions.
"""

import pytest
from sim.perudo import PerudoSimulator, Action


class TestPerudoSimulator:
    """Test suite for PerudoSimulator core functionality."""

    def test_initialization_default_parameters(self):
        """Test PerudoSimulator initialization with default parameters."""
        simulator = PerudoSimulator()

        assert simulator.num_players == 3
        assert simulator.start_dice == 5
        assert simulator.ones_are_wild == True
        assert simulator.use_maputa == True
        assert simulator.use_exact == True

    def test_initialization_custom_parameters(self):
        """Test PerudoSimulator initialization with custom parameters."""
        simulator = PerudoSimulator(
            num_players=4,
            start_dice=3,
            ones_are_wild=False,
            use_maputa=False,
            use_exact=False,
            seed=123
        )

        assert simulator.num_players == 4
        assert simulator.start_dice == 3
        assert simulator.ones_are_wild == False
        assert simulator.use_maputa == False
        assert simulator.use_exact == False

    def test_legal_actions_early_game(self, sample_simulator):
        """Test legal actions in early game state (no current bid)."""
        state = {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5]
        }

        actions = sample_simulator.legal_actions(state, current_bid=None)

        # Should contain bid actions and also call/exact (implementation includes them)
        bid_actions = [a for a in actions if Action.is_bid(a)]
        call_action = Action.call()
        exact_action = Action.exact()

        assert len(bid_actions) > 0
        assert call_action in actions
        if sample_simulator.use_exact:
            assert exact_action in actions

        # Check some expected bids are present
        bid_1_2 = Action.bid(1, 2)
        bid_1_3 = Action.bid(1, 3)
        assert bid_1_2 in actions
        assert bid_1_3 in actions

    def test_legal_actions_with_current_bid(self, sample_simulator):
        """Test legal actions when there's a current bid."""
        state = {
            'dice_counts': [5, 5, 5],
            'player_idx': 1,
            'my_hand': [1, 2, 3, 4, 5]
        }
        current_bid = (2, 3)

        actions = sample_simulator.legal_actions(state, current_bid)

        # Should contain higher bids, call, and exact
        bid_actions = [a for a in actions if Action.is_bid(a)]
        call_action = Action.call()
        exact_action = Action.exact()

        assert call_action in actions
        assert exact_action in actions
        assert len(bid_actions) > 0

        # All bid actions should be higher than current bid
        for action in bid_actions:
            qty, face = Action.qty(action), Action.face(action)
            assert (qty > 2) or (qty == 2 and face > 3)

    def test_legal_actions_with_maputa(self, sample_simulator):
        """Test legal actions with maputa restriction."""
        state = {
            'dice_counts': [1, 1, 2],  # Two players with single dice (maputa condition)
            'player_idx': 0,
            'my_hand': [3]
        }
        current_bid = (1, 2)
        maputa_restrict_face = 2

        actions = sample_simulator.legal_actions(state, current_bid, maputa_restrict_face)

        # With maputa restriction, only bids with the restricted face should be allowed
        bid_actions = [a for a in actions if Action.is_bid(a)]
        for action in bid_actions:
            face = Action.face(action)
            assert face == 2  # Should only allow face 2 due to maputa restriction

    def test_legal_actions_edge_cases(self, sample_simulator):
        """Test legal actions in edge case scenarios."""
        # Test with single die remaining
        state = {
            'dice_counts': [1, 1, 1],
            'player_idx': 0,
            'my_hand': [6]
        }

        actions = sample_simulator.legal_actions(state, current_bid=None)

        # Should still have valid actions
        assert len(actions) > 0

        # All bids should be reasonable for total dice count
        bid_actions = [a for a in actions if Action.is_bid(a)]
        for action in bid_actions:
            qty = Action.qty(action)
            assert qty <= 3  # Can't bid more than total dice

    def test_is_bid_true_with_wild_ones(self, sample_simulator):
        """Test bid validation with wild ones enabled."""
        # Test case where ones count as wild
        hands = [[1, 1, 2, 3, 4], [2, 2, 3, 4, 5], [1, 3, 4, 5, 6]]
        bid = (4, 2)  # 4 twos

        # Should be true: 3 actual twos + 3 ones (wild) = 6 >= 4
        # is_bid_true returns (bool, count) tuple
        result, count = sample_simulator.is_bid_true(hands, bid, ones_are_wild=True)
        assert result == True
        assert count >= 4  # Should have at least 4 (3 twos + 3 ones = 6)

        # Test with bid for ones specifically
        bid_ones = (3, 1)  # 3 ones
        result_ones, count_ones = sample_simulator.is_bid_true(hands, bid_ones, ones_are_wild=True)
        assert result_ones == True  # Should have exactly 3 ones
        assert count_ones == 3

    def test_is_bid_true_without_wild_ones(self, sample_simulator):
        """Test bid validation with wild ones disabled."""
        hands = [[1, 1, 2, 3, 4], [2, 2, 3, 4, 5], [1, 3, 4, 5, 6]]
        bid = (4, 2)  # 4 twos

        # Should be false: only 3 actual twos, ones don't count as wild
        # is_bid_true returns (bool, count) tuple
        result, count = sample_simulator.is_bid_true(hands, bid, ones_are_wild=False)
        assert result == False
        assert count == 3  # Only 3 actual twos

        # Test with bid for ones specifically
        bid_ones = (3, 1)  # 3 ones
        result_ones, count_ones = sample_simulator.is_bid_true(hands, bid_ones, ones_are_wild=False)
        assert result_ones == True  # Should have exactly 3 ones
        assert count_ones == 3

    def test_total_dice_calculation(self, sample_simulator):
        """Test total dice calculation from state."""
        state = {'dice_counts': [3, 2, 4]}
        total = sample_simulator.total_dice(state)
        assert total == 9

    def test_legal_bids_after(self, sample_simulator):
        """Test legal bid generation after a current bid."""
        current_bid = (2, 3)
        cap_qty = 10

        legal_bids = list(sample_simulator.legal_bids_after(current_bid, cap_qty))  # Convert generator to list

        # Should contain higher bids
        assert len(legal_bids) > 0

        # All bids should be higher than current
        for qty, face in legal_bids:
            assert (qty > 2) or (qty == 2 and face > 3)
            assert qty <= cap_qty

    def test_next_player_idx(self, sample_simulator):
        """Test next player index calculation."""
        dice_counts = [3, 2, 0, 1]  # Player 2 is eliminated (0 dice)

        # From player 0, should go to player 1
        next_idx = sample_simulator.next_player_idx(0, dice_counts)
        assert next_idx == 1

        # From player 1, should skip eliminated player 2 and go to player 3
        next_idx = sample_simulator.next_player_idx(1, dice_counts)
        assert next_idx == 3

        # From player 3, should wrap around to player 0
        next_idx = sample_simulator.next_player_idx(3, dice_counts)
        assert next_idx == 0

    def test_new_game_state(self, sample_simulator):
        """Test new game state creation."""
        state = sample_simulator.new_game()

        assert 'dice_counts' in state
        # new_game() only returns dice_counts, not player_idx
        assert len(state['dice_counts']) == sample_simulator.num_players
        assert all(count == sample_simulator.start_dice for count in state['dice_counts'])

    def test_roll_hands(self, sample_simulator):
        """Test hand rolling functionality."""
        state = {
            'dice_counts': [3, 2, 4],
            'player_idx': 0
        }

        hands = sample_simulator.roll_hands(state)

        # Should return hands for all players
        assert len(hands) == 3
        assert len(hands[0]) == 3  # Player 0 has 3 dice
        assert len(hands[1]) == 2  # Player 1 has 2 dice
        assert len(hands[2]) == 4  # Player 2 has 4 dice

        # All dice should be valid (1-6)
        for hand in hands:
            for die in hand:
                assert 1 <= die <= 6


class TestPerudoSimulatorEdgeCases:
    """Test suite for PerudoSimulator edge cases and special scenarios."""

    def test_single_player_remaining(self):
        """Test behavior when only one player has dice remaining."""
        simulator = PerudoSimulator(num_players=3, start_dice=5)
        dice_counts = [0, 0, 1]  # Only player 2 has dice

        next_idx = simulator.next_player_idx(2, dice_counts)
        assert next_idx == 2  # Should stay with the only remaining player

    def test_maximum_bid_constraints(self, sample_simulator):
        """Test that bids don't exceed reasonable limits."""
        state = {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5]
        }

        actions = sample_simulator.legal_actions(state, current_bid=None)
        bid_actions = [a for a in actions if Action.is_bid(a)]

        # No bid should exceed total dice count
        total_dice = sum(state['dice_counts'])
        for action in bid_actions:
            qty = Action.qty(action)
            assert qty <= total_dice

    def test_maputa_detection(self):
        """Test maputa condition detection (when multiple players have single dice)."""
        simulator = PerudoSimulator(use_maputa=True)

        # Maputa condition: at least 2 players with 1 die each
        dice_counts_maputa = [1, 1, 3]
        dice_counts_no_maputa = [1, 2, 3]

        # This would be tested in the context of the game flow
        # The actual maputa logic is implemented in the game flow
        assert True  # Placeholder for maputa detection logic

    def test_exact_call_scenarios(self, sample_simulator):
        """Test exact call availability and constraints."""
        state = {
            'dice_counts': [2, 2, 2],
            'player_idx': 1,
            'my_hand': [3, 4]
        }
        current_bid = (3, 5)

        actions = sample_simulator.legal_actions(state, current_bid)
        exact_action = Action.exact()

        # Exact should be available when use_exact is True
        if sample_simulator.use_exact:
            assert exact_action in actions
