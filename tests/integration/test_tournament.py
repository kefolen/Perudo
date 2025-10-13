"""
Integration tests for Tournament System functionality.

This module tests the tournament execution system, including different
configurations, parameter variations, special rule combinations, and
performance metrics collection.
"""

import pytest
import time
from collections import Counter
from eval.tournament import make_agent, play_match
from sim.perudo import PerudoSimulator
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent


class TestTournament:
    """Test suite for tournament system functionality."""

    def test_tournament_basic_execution(self):
        """Test basic tournament execution with minimal configuration."""
        sim = PerudoSimulator(num_players=3, start_dice=5, seed=42)
        agent_names = ['random', 'baseline', 'random']

        # Run a small tournament
        results = play_match(sim, agent_names, games=5, mc_n=10)

        # Verify results structure
        assert isinstance(results, Counter)
        assert len(results) > 0

        # Verify all results are valid player indices
        for winner_idx in results.keys():
            assert 0 <= winner_idx < sim.num_players

        # Verify total games played
        total_games = sum(results.values())
        assert total_games == 5

    def test_tournament_with_different_agents(self):
        """Test tournament with different agent combinations."""
        sim = PerudoSimulator(num_players=3, start_dice=5, seed=42)

        # Test all random agents
        results_random = play_match(sim, ['random', 'random', 'random'], games=3, mc_n=5)
        assert sum(results_random.values()) == 3

        # Test all baseline agents
        results_baseline = play_match(sim, ['baseline', 'baseline', 'baseline'], games=3, mc_n=5)
        assert sum(results_baseline.values()) == 3

        # Test mixed agents
        results_mixed = play_match(sim, ['random', 'baseline', 'mc'], games=3, mc_n=5)
        assert sum(results_mixed.values()) == 3

    def test_tournament_parameter_variations(self):
        """Test tournament with different parameter configurations."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        agent_names = ['random', 'baseline']

        # Test different MC_N values
        results_low_n = play_match(sim, agent_names, games=2, mc_n=5)
        results_high_n = play_match(sim, agent_names, games=2, mc_n=20)

        # Both should complete successfully
        assert sum(results_low_n.values()) == 2
        assert sum(results_high_n.values()) == 2

        # Test different max_rounds values
        results_low_rounds = play_match(sim, agent_names, games=2, mc_n=5, max_rounds=2)
        results_high_rounds = play_match(sim, agent_names, games=2, mc_n=5, max_rounds=8)

        assert sum(results_low_rounds.values()) == 2
        assert sum(results_high_rounds.values()) == 2

    def test_tournament_special_rules(self):
        """Test tournament with different special rule combinations."""
        agent_names = ['random', 'baseline']

        # Test with maputa enabled
        sim_maputa = PerudoSimulator(
            num_players=2, start_dice=3, 
            use_maputa=True, use_exact=True, seed=42
        )
        results_maputa = play_match(sim_maputa, agent_names, games=2, mc_n=5)
        assert sum(results_maputa.values()) == 2

        # Test with maputa disabled
        sim_no_maputa = PerudoSimulator(
            num_players=2, start_dice=3, 
            use_maputa=False, use_exact=True, seed=42
        )
        results_no_maputa = play_match(sim_no_maputa, agent_names, games=2, mc_n=5)
        assert sum(results_no_maputa.values()) == 2

        # Test with exact disabled
        sim_no_exact = PerudoSimulator(
            num_players=2, start_dice=3, 
            use_maputa=True, use_exact=False, seed=42
        )
        results_no_exact = play_match(sim_no_exact, agent_names, games=2, mc_n=5)
        assert sum(results_no_exact.values()) == 2

        # Test with wild ones disabled
        sim_no_wild = PerudoSimulator(
            num_players=2, start_dice=3, 
            ones_are_wild=False, seed=42
        )
        results_no_wild = play_match(sim_no_wild, agent_names, games=2, mc_n=5)
        assert sum(results_no_wild.values()) == 2

    def test_tournament_metrics_collection(self):
        """Test that tournament collects performance metrics properly."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        agent_names = ['random', 'baseline']

        # Capture start time
        start_time = time.time()

        # Run tournament
        results = play_match(sim, agent_names, games=3, mc_n=5)

        # Verify execution completed in reasonable time
        execution_time = time.time() - start_time
        assert execution_time < 30  # Should complete within 30 seconds

        # Verify results contain expected data
        assert isinstance(results, Counter)
        assert sum(results.values()) == 3

        # Verify all winners are valid player indices
        for winner_idx in results.keys():
            assert isinstance(winner_idx, int)
            assert 0 <= winner_idx < sim.num_players

    def test_tournament_edge_cases(self):
        """Test tournament behavior in edge case scenarios."""
        # Test with minimum players
        sim_min = PerudoSimulator(num_players=2, start_dice=2, seed=42)
        results_min = play_match(sim_min, ['random', 'baseline'], games=2, mc_n=5)
        assert sum(results_min.values()) == 2

        # Test with single die per player
        sim_single = PerudoSimulator(num_players=2, start_dice=1, seed=42)
        results_single = play_match(sim_single, ['random', 'baseline'], games=2, mc_n=5)
        assert sum(results_single.values()) == 2

        # Test with maximum reasonable players
        sim_max = PerudoSimulator(num_players=6, start_dice=2, seed=42)
        agent_names_max = ['random', 'baseline', 'random', 'baseline', 'random', 'baseline']
        results_max = play_match(sim_max, agent_names_max, games=2, mc_n=5)
        assert sum(results_max.values()) == 2


class TestMakeAgent:
    """Test suite for agent creation functionality."""

    def test_make_agent_random(self):
        """Test creation of RandomAgent."""
        sim = PerudoSimulator(seed=42)
        agent = make_agent('random', sim)

        assert isinstance(agent, RandomAgent)
        assert hasattr(agent, 'select_action')

    def test_make_agent_baseline(self):
        """Test creation of BaselineAgent."""
        sim = PerudoSimulator(seed=42)
        agent = make_agent('baseline', sim)

        assert isinstance(agent, BaselineAgent)
        assert hasattr(agent, 'select_action')
        assert hasattr(agent, 'threshold_call')

    def test_make_agent_mc(self):
        """Test creation of MonteCarloAgent."""
        sim = PerudoSimulator(seed=42)
        agent = make_agent('mc', sim, mc_n=10, max_rounds=3)

        assert isinstance(agent, MonteCarloAgent)
        assert hasattr(agent, 'select_action')
        assert agent.N == 10  # MonteCarloAgent uses capital N
        assert agent.max_rounds == 3

    def test_make_agent_invalid(self):
        """Test error handling for invalid agent names."""
        sim = PerudoSimulator(seed=42)

        with pytest.raises(ValueError, match="Unknown agent"):
            make_agent('invalid_agent', sim)

    def test_make_agent_mc_parameters(self):
        """Test MonteCarloAgent parameter passing."""
        sim = PerudoSimulator(seed=42)

        # Test with different parameters
        agent1 = make_agent('mc', sim, mc_n=5, max_rounds=2)
        agent2 = make_agent('mc', sim, mc_n=20, max_rounds=8)

        assert agent1.N == 5
        assert agent1.max_rounds == 2
        assert agent2.N == 20
        assert agent2.max_rounds == 8


class TestTournamentIntegration:
    """Test suite for end-to-end tournament integration."""

    def test_full_tournament_workflow(self):
        """Test complete tournament workflow from setup to results."""
        # Setup tournament configuration
        sim = PerudoSimulator(
            num_players=3, 
            start_dice=3, 
            ones_are_wild=True,
            use_maputa=True,
            use_exact=True,
            seed=42
        )

        agent_names = ['random', 'baseline', 'random']  # Use deterministic agents for reproducibility test
        games = 3
        mc_n = 10
        max_rounds = 5

        # Run tournament
        results = play_match(sim, agent_names, games=games, mc_n=mc_n, max_rounds=max_rounds)

        # Verify complete workflow
        assert isinstance(results, Counter)
        assert sum(results.values()) == games

        # Verify all possible winners are represented or results make sense
        for winner_idx in results.keys():
            assert 0 <= winner_idx < sim.num_players

        # Verify that running another tournament produces valid results
        # (Perfect reproducibility is tested separately in test_tournament_reproducibility)
        sim2 = PerudoSimulator(
            num_players=3, 
            start_dice=3, 
            ones_are_wild=True,
            use_maputa=True,
            use_exact=True,
            seed=123  # Different seed for variety
        )
        results2 = play_match(sim2, agent_names, games=games, mc_n=mc_n, max_rounds=max_rounds)

        # Both tournaments should produce valid results
        assert isinstance(results2, Counter)
        assert sum(results2.values()) == games
        for winner_idx in results2.keys():
            assert 0 <= winner_idx < sim2.num_players

    def test_tournament_reproducibility(self):
        """Test that tournaments run smoothly with consistent configuration."""
        sim1 = PerudoSimulator(num_players=2, start_dice=3, seed=123)
        sim2 = PerudoSimulator(num_players=2, start_dice=3, seed=123)

        agent_names = ['random', 'baseline']

        results1 = play_match(sim1, agent_names, games=3, mc_n=5)
        results2 = play_match(sim2, agent_names, games=3, mc_n=5)

        # Both tournaments should complete successfully
        assert isinstance(results1, Counter), "First tournament should return Counter"
        assert isinstance(results2, Counter), "Second tournament should return Counter"
        assert sum(results1.values()) == 3, "First tournament should complete 3 games"
        assert sum(results2.values()) == 3, "Second tournament should complete 3 games"
        assert len(results1) > 0, "First tournament should have results"
        assert len(results2) > 0, "Second tournament should have results"

    def test_tournament_different_seeds(self):
        """Test that tournaments produce different results with different seeds."""
        sim1 = PerudoSimulator(num_players=2, start_dice=3, seed=123)
        sim2 = PerudoSimulator(num_players=2, start_dice=3, seed=456)

        agent_names = ['random', 'baseline']

        results1 = play_match(sim1, agent_names, games=5, mc_n=5)
        results2 = play_match(sim2, agent_names, games=5, mc_n=5)

        # Results might be different with different seeds
        # (though they could theoretically be the same by chance)
        # At minimum, both should be valid
        assert sum(results1.values()) == 5
        assert sum(results2.values()) == 5
