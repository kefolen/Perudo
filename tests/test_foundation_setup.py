"""
Foundation setup verification test.

This test verifies that the Phase 1 foundation setup is working correctly,
including fixtures, directory structure, and basic pytest functionality.
"""

import pytest
from sim.perudo import PerudoSimulator, Action
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from tests.fixtures.sample_game_states import GameStates
from tests.fixtures.test_scenarios import EdgeCases
from tests.fixtures.expected_outcomes import ExpectedLegalActions


class TestFoundationSetup:
    """Test class to verify foundation setup is working correctly"""


    def test_fixtures_available(self, sample_simulator, sample_game_state, all_agent_types):
        """Test that all fixtures are available and working"""
        # Test simulator fixture
        assert isinstance(sample_simulator, PerudoSimulator)
        assert sample_simulator.num_players == 3
        assert sample_simulator.start_dice == 5

        # Test game state fixture
        assert isinstance(sample_game_state, dict)
        assert 'dice_counts' in sample_game_state
        assert 'player_idx' in sample_game_state
        assert 'my_hand' in sample_game_state

        # Test agent fixtures
        assert 'random' in all_agent_types
        assert 'baseline' in all_agent_types
        assert 'mc' in all_agent_types
        assert isinstance(all_agent_types['random'], RandomAgent)
        assert isinstance(all_agent_types['baseline'], BaselineAgent)

    def test_sample_game_states_fixture(self):
        """Test that sample game states fixture is working"""
        early_game = GameStates.early_game_no_bid()
        assert isinstance(early_game, dict)
        assert early_game['dice_counts'] == [5, 5, 5]
        assert early_game['current_bid'] is None

        mid_game = GameStates.mid_game_balanced()
        assert isinstance(mid_game, dict)
        assert sum(mid_game['dice_counts']) == 8  # Total dice should be 8
        assert mid_game['current_bid'] is not None

    def test_edge_cases_fixture(self):
        """Test that edge cases fixture is working"""
        single_player = EdgeCases.single_player_remaining()
        assert isinstance(single_player, dict)
        assert single_player['dice_counts'].count(0) == 2  # Two players eliminated

        min_players = EdgeCases.minimum_players()
        assert isinstance(min_players, dict)
        assert len(min_players['dice_counts']) == 2  # Only 2 players

    def test_expected_outcomes_fixture(self):
        """Test that expected outcomes fixture is working"""
        legal_actions = ExpectedLegalActions.early_game_no_bid()
        assert isinstance(legal_actions, list)
        assert len(legal_actions) > 0

        # Should contain bid actions for different faces
        bid_actions = [action for action in legal_actions if action[0] == 'bid']
        assert len(bid_actions) > 0

    def test_deterministic_behavior(self, deterministic_seed, random_agent):
        """Test that deterministic seed produces consistent results"""
        # Create observation with simulator
        sim = PerudoSimulator(seed=deterministic_seed)
        obs = {
            'dice_counts': [3, 3, 3],
            'player_idx': 0,
            'my_hand': [1, 2, 3],
            'current_bid': None,
            'maputa_restrict_face': None,
            '_simulator': sim
        }

        # Get two actions with same seed - should be identical
        action1 = random_agent.select_action(obs)

        # Reset the agent with same seed
        random_agent_2 = RandomAgent(name='test_random_2', seed=deterministic_seed)
        action2 = random_agent_2.select_action(obs)

        assert action1 == action2, "Deterministic seed should produce identical results"

    def test_action_creation(self, sample_actions):
        """Test that Action creation is working through fixtures"""
        assert 'bid_low' in sample_actions
        assert 'call' in sample_actions
        assert 'exact' in sample_actions

        bid_action = sample_actions['bid_low']
        assert Action.is_bid(bid_action)
        assert Action.qty(bid_action) == 1
        assert Action.face(bid_action) == 2

        call_action = sample_actions['call']
        assert call_action == ('call',)

        exact_action = sample_actions['exact']
        assert exact_action == ('exact',)

    def test_simulator_basic_functionality(self, sample_simulator):
        """Test basic simulator functionality"""
        # Test game state creation
        state = sample_simulator.new_game()
        assert isinstance(state, dict)

        # Test legal actions
        legal_actions = sample_simulator.legal_actions(
            {'dice_counts': [5, 5, 5]}, 
            None, 
            None
        )
        assert isinstance(legal_actions, list)
        assert len(legal_actions) > 0

        # Test bid validation
        hands = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [1, 1, 2, 3, 4]]
        result = sample_simulator.is_bid_true(hands, (3, 2), ones_are_wild=True)
        # is_bid_true returns a tuple (bool, count), we check the first element
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)

    def test_agent_interfaces(self, all_agent_types, observation_with_simulator):
        """Test that all agents implement the required interface"""
        for agent_name, agent in all_agent_types.items():
            # Each agent should have a select_action method
            assert hasattr(agent, 'select_action'), f"{agent_name} missing select_action method"

            # Each agent should have a name attribute
            assert hasattr(agent, 'name'), f"{agent_name} missing name attribute"

            # select_action should return a valid action
            action = agent.select_action(observation_with_simulator)
            assert isinstance(action, tuple), f"{agent_name} should return tuple action"
            assert len(action) >= 1, f"{agent_name} action should have at least one element"
            assert action[0] in ['bid', 'call', 'exact'], f"{agent_name} returned invalid action type"


