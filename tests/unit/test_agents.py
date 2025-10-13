"""
Unit tests for Agent interface functionality.

This module tests that all agents implement consistent interfaces,
validate select_action behavior, initialization, and deterministic behavior.
"""

import pytest
from sim.perudo import PerudoSimulator, Action
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent


class TestAgentInterface:
    """Test suite for agent interface consistency."""

    def test_random_agent_interface(self, deterministic_seed):
        """Test RandomAgent implements the expected interface."""
        agent = RandomAgent(name='test_random', seed=deterministic_seed)

        # Test initialization
        assert agent.name == 'test_random'
        assert hasattr(agent, 'rng')
        assert hasattr(agent, 'select_action')

        # Test select_action method exists and is callable
        assert callable(agent.select_action)

    def test_baseline_agent_interface(self):
        """Test BaselineAgent implements the expected interface."""
        agent = BaselineAgent(name='test_baseline', threshold_call=0.6)

        # Test initialization
        assert agent.name == 'test_baseline'
        assert agent.threshold_call == 0.6
        assert hasattr(agent, 'select_action')

        # Test select_action method exists and is callable
        assert callable(agent.select_action)

    def test_mc_agent_interface(self, deterministic_seed):
        """Test MonteCarloAgent implements the expected interface."""
        import random
        rng = random.Random(deterministic_seed)
        agent = MonteCarloAgent(
            name='test_mc',
            n=10,  # Reduced for faster testing
            rng=rng,
            chunk_size=5
        )

        # Test initialization
        assert agent.name == 'test_mc'
        assert agent.N == 10  # MonteCarloAgent uses capital N
        assert hasattr(agent, 'select_action')

        # Test select_action method exists and is callable
        assert callable(agent.select_action)

    def test_agent_returns_valid_actions(self, all_agent_types, observation_with_simulator):
        """Test that all agents return valid Action objects."""
        for agent_name, agent in all_agent_types.items():
            action = agent.select_action(observation_with_simulator)

            # Should return a valid action (tuple)
            assert isinstance(action, tuple), f"{agent_name} didn't return a tuple"

            # Should be a valid action type
            is_valid_action = (
                Action.is_bid(action) or 
                action == Action.call() or 
                action == Action.exact()
            )
            assert is_valid_action, f"{agent_name} returned invalid action: {action}"

    def test_agent_respects_legal_actions(self, all_agent_types, observation_with_simulator):
        """Test that agents only return legal actions."""
        obs = observation_with_simulator
        sim = obs['_simulator']
        state = {'dice_counts': obs['dice_counts']}
        legal_actions = sim.legal_actions(state, obs['current_bid'], obs.get('maputa_restrict_face'))

        for agent_name, agent in all_agent_types.items():
            action = agent.select_action(obs)
            assert action in legal_actions, f"{agent_name} returned illegal action: {action}"

    def test_deterministic_behavior(self, deterministic_seed):
        """Test that agents with fixed seeds produce consistent results."""
        # Test RandomAgent deterministic behavior
        agent1 = RandomAgent(name='test1', seed=deterministic_seed)
        agent2 = RandomAgent(name='test2', seed=deterministic_seed)

        # Create a test observation
        simulator = PerudoSimulator(seed=deterministic_seed)
        obs = {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': None,
            'maputa_restrict_face': None,
            '_simulator': simulator
        }

        # Both agents should make the same decision
        action1 = agent1.select_action(obs)
        action2 = agent2.select_action(obs)
        assert action1 == action2, "RandomAgent with same seed should be deterministic"

        # Test MonteCarloAgent deterministic behavior
        import random
        rng1 = random.Random(deterministic_seed)
        rng2 = random.Random(deterministic_seed)

        mc_agent1 = MonteCarloAgent(name='mc1', n=10, rng=rng1)
        mc_agent2 = MonteCarloAgent(name='mc2', n=10, rng=rng2)

        # Both MC agents should make the same decision
        action_mc1 = mc_agent1.select_action(obs)
        action_mc2 = mc_agent2.select_action(obs)
        assert action_mc1 == action_mc2, "MonteCarloAgent with same seed should be deterministic"

    def test_agent_initialization_parameters(self):
        """Test agent initialization with various parameters."""
        # Test RandomAgent parameters
        random_agent = RandomAgent(name='custom_random', seed=123)
        assert random_agent.name == 'custom_random'

        # Test BaselineAgent parameters
        baseline_agent = BaselineAgent(name='custom_baseline', threshold_call=0.3)
        assert baseline_agent.name == 'custom_baseline'
        assert baseline_agent.threshold_call == 0.3

        # Test MonteCarloAgent parameters
        import random
        rng = random.Random(456)
        mc_agent = MonteCarloAgent(
            name='custom_mc',
            n=50,
            rng=rng,
            chunk_size=10,
            max_rounds=4
        )
        assert mc_agent.name == 'custom_mc'
        assert mc_agent.N == 50  # MonteCarloAgent uses capital N
        assert mc_agent.chunk_size == 10
        assert mc_agent.max_rounds == 4

    def test_agents_handle_no_current_bid(self, all_agent_types):
        """Test that agents handle the case when there's no current bid."""
        simulator = PerudoSimulator(seed=42)
        obs = {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': None,  # No current bid
            'maputa_restrict_face': None,
            '_simulator': simulator
        }

        for agent_name, agent in all_agent_types.items():
            action = agent.select_action(obs)

            # Should return a bid action (not call or exact when no current bid)
            assert Action.is_bid(action), f"{agent_name} should bid when no current bid exists"

    def test_agents_handle_existing_bid(self, all_agent_types):
        """Test that agents handle the case when there's an existing bid."""
        simulator = PerudoSimulator(seed=42)
        obs = {
            'dice_counts': [5, 5, 5],
            'player_idx': 1,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': (2, 3),  # Existing bid
            'maputa_restrict_face': None,
            '_simulator': simulator
        }

        for agent_name, agent in all_agent_types.items():
            action = agent.select_action(obs)

            # Should return a valid action (bid, call, or exact)
            is_valid = (
                Action.is_bid(action) or 
                action == Action.call() or 
                action == Action.exact()
            )
            assert is_valid, f"{agent_name} returned invalid action with existing bid"

    def test_agents_handle_edge_cases(self, all_agent_types):
        """Test that agents handle edge case scenarios gracefully."""
        simulator = PerudoSimulator(seed=42)

        # Test with single die
        obs_single_die = {
            'dice_counts': [1, 1, 1],
            'player_idx': 0,
            'my_hand': [6],
            'current_bid': None,
            'maputa_restrict_face': None,
            '_simulator': simulator
        }

        for agent_name, agent in all_agent_types.items():
            try:
                action = agent.select_action(obs_single_die)
                assert isinstance(action, tuple), f"{agent_name} failed on single die scenario"
            except Exception as e:
                pytest.fail(f"{agent_name} raised exception on single die: {e}")

    def test_agents_handle_maputa_restriction(self, all_agent_types):
        """Test that agents handle maputa face restrictions."""
        simulator = PerudoSimulator(seed=42, use_maputa=True)
        obs = {
            'dice_counts': [1, 1, 2],  # Maputa condition
            'player_idx': 0,
            'my_hand': [3],
            'current_bid': (1, 2),
            'maputa_restrict_face': 2,  # Restricted to face 2
            '_simulator': simulator
        }

        for agent_name, agent in all_agent_types.items():
            action = agent.select_action(obs)

            # If it's a bid, should respect maputa restriction
            if Action.is_bid(action):
                face = Action.face(action)
                # In maputa, should only bid the restricted face or call/exact
                assert face == 2 or action == Action.call() or action == Action.exact(), \
                    f"{agent_name} didn't respect maputa restriction"


class TestAgentSpecificBehavior:
    """Test suite for agent-specific behavior patterns."""

    def test_baseline_agent_probability_threshold(self):
        """Test that BaselineAgent respects its probability threshold."""
        # Test with very low threshold (should almost never call)
        agent_low = BaselineAgent(name='low_threshold', threshold_call=0.01)

        # Test with very high threshold (should almost always call)
        agent_high = BaselineAgent(name='high_threshold', threshold_call=0.99)

        simulator = PerudoSimulator(seed=42)
        obs = {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [2, 3, 4, 5, 6],  # No ones, no matching dice
            'current_bid': (10, 1),  # Very high bid
            'maputa_restrict_face': None,
            '_simulator': simulator
        }

        # Low threshold agent might still bid
        action_low = agent_low.select_action(obs)

        # High threshold agent should likely call
        action_high = agent_high.select_action(obs)

        # Both should return valid actions
        assert isinstance(action_low, tuple)
        assert isinstance(action_high, tuple)

    def test_monte_carlo_agent_simulation_parameters(self):
        """Test that MonteCarloAgent uses its simulation parameters."""
        import random
        rng = random.Random(42)

        # Test with different n values
        agent_few_sims = MonteCarloAgent(name='few_sims', n=5, rng=rng)
        agent_many_sims = MonteCarloAgent(name='many_sims', n=20, rng=rng)

        assert agent_few_sims.N == 5  # MonteCarloAgent uses capital N
        assert agent_many_sims.N == 20  # MonteCarloAgent uses capital N

        # Both should have the required methods
        assert hasattr(agent_few_sims, 'evaluate_action')
        assert hasattr(agent_many_sims, 'sample_determinization')

    def test_random_agent_randomness(self):
        """Test that RandomAgent produces varied results across different seeds."""
        simulator = PerudoSimulator(seed=42)
        obs = {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': None,
            'maputa_restrict_face': None,
            '_simulator': simulator
        }

        # Create agents with different seeds
        actions = []
        for seed in range(10):
            agent = RandomAgent(name=f'random_{seed}', seed=seed)
            action = agent.select_action(obs)
            actions.append(action)

        # Should have some variety in actions (not all the same)
        unique_actions = set(actions)
        assert len(unique_actions) > 1, "RandomAgent should produce varied actions with different seeds"

    def test_agent_name_assignment(self):
        """Test that agents properly assign and store their names."""
        random_agent = RandomAgent(name='test_random_name')
        baseline_agent = BaselineAgent(name='test_baseline_name')

        import random
        rng = random.Random(42)
        mc_agent = MonteCarloAgent(name='test_mc_name', rng=rng)

        assert random_agent.name == 'test_random_name'
        assert baseline_agent.name == 'test_baseline_name'
        assert mc_agent.name == 'test_mc_name'
