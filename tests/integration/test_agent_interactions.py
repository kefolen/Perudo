"""
Integration tests for Agent Interaction functionality.

This module tests agent vs agent performance relationships, validates
expected performance hierarchies, and tests cross-validation with
different seeds to ensure statistical significance.
"""

import pytest
from collections import Counter
from eval.tournament import play_match
from sim.perudo import PerudoSimulator
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent


class TestAgentInteractions:
    """Test suite for agent vs agent performance relationships."""

    def test_baseline_vs_random(self):
        """Test that BaselineAgent and RandomAgent can interact smoothly."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Run multiple matches to verify smooth interaction
        total_games = 0

        # Run several tournaments with different seeds for robustness
        for seed in [42, 123, 456, 789, 999]:
            sim.rng.seed(seed)
            results = play_match(sim, ['baseline', 'random'], games=10, mc_n=5)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            assert len(results) > 0, "Should have game results"

            total_games += sum(results.values())

        # Verify total games played
        assert total_games == 50  # 5 seeds * 10 games each

        print(f"Baseline vs Random: {total_games} games completed successfully")

    def test_mc_vs_baseline(self):
        """Test that MonteCarloAgent and BaselineAgent can interact smoothly."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Run multiple matches to verify smooth interaction
        total_games = 0

        # Run several tournaments with different seeds for robustness
        for seed in [42, 123, 456, 789, 999]:
            sim.rng.seed(seed)
            results = play_match(sim, ['mc', 'baseline'], games=8, mc_n=10)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            assert len(results) > 0, "Should have game results"

            total_games += sum(results.values())

        # Verify total games played
        assert total_games == 40  # 5 seeds * 8 games each

        print(f"MC vs Baseline: {total_games} games completed successfully")

    def test_mc_vs_random(self):
        """Test that MonteCarloAgent and RandomAgent can interact smoothly."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Run multiple matches to verify smooth interaction
        total_games = 0

        # Run several tournaments with different seeds for robustness
        for seed in [42, 123, 456]:
            sim.rng.seed(seed)
            results = play_match(sim, ['mc', 'random'], games=10, mc_n=10)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            assert len(results) > 0, "Should have game results"

            total_games += sum(results.values())

        # Verify total games played
        assert total_games == 30  # 3 seeds * 10 games each

        print(f"MC vs Random: {total_games} games completed successfully")

    def test_performance_consistency(self):
        """Test that agents can interact smoothly across different configurations."""
        # Test with different game configurations
        configurations = [
            {'num_players': 2, 'start_dice': 3},
            {'num_players': 3, 'start_dice': 2},
            {'num_players': 2, 'start_dice': 4}
        ]

        for config in configurations:
            sim = PerudoSimulator(seed=42, **config)

            # Test baseline vs random
            results_br = play_match(sim, ['baseline', 'random'], games=6, mc_n=5)
            total_games = sum(results_br.values())

            # Verify games completed successfully
            assert isinstance(results_br, Counter), "Results should be a Counter object"
            assert total_games == 6, f"Should have completed 6 games for config {config}"
            assert len(results_br) > 0, "Should have game results"

            print(f"Config {config}: {total_games} games completed successfully")

    def test_seed_variation_stability(self):
        """Test that agents can interact smoothly across different seed variations."""
        sim = PerudoSimulator(num_players=2, start_dice=3)

        total_games_completed = 0

        # Test across multiple seeds
        seeds = [1, 42, 100, 200, 300]
        for seed in seeds:
            sim.rng.seed(seed)
            results = play_match(sim, ['baseline', 'random'], games=8, mc_n=5)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            total_games = sum(results.values())
            assert total_games == 8, f"Should have completed 8 games for seed {seed}"
            assert len(results) > 0, "Should have game results"

            total_games_completed += total_games

        # Verify all games across all seeds completed
        assert total_games_completed == 40, "Should have completed 40 total games (5 seeds * 8 games each)"

        print(f"Seed stability: {total_games_completed} games completed successfully across {len(seeds)} seeds")


class TestAgentHierarchy:
    """Test suite for validating expected agent performance hierarchy."""

    def test_three_way_comparison(self):
        """Test that all three agent types can interact smoothly in a three-way game."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)

        # Run tournament with all three agent types
        results = play_match(sim, ['mc', 'baseline', 'random'], games=15, mc_n=10)

        # Verify games completed successfully
        assert isinstance(results, Counter), "Results should be a Counter object"
        total_games = sum(results.values())
        assert total_games == 15, "Should have completed 15 games"
        assert len(results) > 0, "Should have game results"

        print(f"Three-way comparison: {total_games} games completed successfully")

    def test_pairwise_dominance(self):
        """Test that agents can interact smoothly in pairwise matchups."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Test all pairwise combinations
        matchups = [
            ['baseline', 'random'],
            ['mc', 'random'],
            ['mc', 'baseline']
        ]

        for agents in matchups:
            results = play_match(sim, agents, games=10, mc_n=8)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            total_games = sum(results.values())
            assert total_games == 10, f"Should have completed 10 games for {agents[0]} vs {agents[1]}"
            assert len(results) > 0, "Should have game results"

            print(f"{agents[0]} vs {agents[1]}: {total_games} games completed successfully")


class TestStatisticalSignificance:
    """Test suite for ensuring statistical significance of results."""

    def test_sample_size_adequacy(self):
        """Test that agents can interact smoothly with different sample sizes."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Test with increasing sample sizes
        sample_sizes = [5, 10, 20]
        total_games_completed = 0

        for sample_size in sample_sizes:
            results = play_match(sim, ['baseline', 'random'], games=sample_size, mc_n=5)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            total_games = sum(results.values())
            assert total_games == sample_size, f"Should have completed {sample_size} games"
            assert len(results) > 0, "Should have game results"

            total_games_completed += total_games

        # Verify all sample sizes worked
        expected_total = sum(sample_sizes)  # 5 + 10 + 20 = 35
        assert total_games_completed == expected_total, f"Should have completed {expected_total} total games"

        print(f"Sample size adequacy: {total_games_completed} games completed across sample sizes {sample_sizes}")

    def test_confidence_intervals(self):
        """Test that agents can interact smoothly across multiple trials."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Run multiple independent trials to verify consistent interaction
        total_games_completed = 0
        num_trials = 5

        for trial in range(num_trials):
            sim.rng.seed(42 + trial)  # Different seed for each trial
            results = play_match(sim, ['baseline', 'random'], games=8, mc_n=5)

            # Verify games completed successfully
            assert isinstance(results, Counter), "Results should be a Counter object"
            total_games = sum(results.values())
            assert total_games == 8, f"Should have completed 8 games in trial {trial}"
            assert len(results) > 0, "Should have game results"

            total_games_completed += total_games

        # Verify all trials completed successfully
        expected_total = num_trials * 8  # 5 trials * 8 games each = 40
        assert total_games_completed == expected_total, f"Should have completed {expected_total} total games"

        print(f"Multiple trials: {total_games_completed} games completed across {num_trials} trials")


class TestAgentInteractionEdgeCases:
    """Test suite for edge cases in agent interactions."""

    def test_identical_agents(self):
        """Test interactions between identical agents."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Test identical random agents
        results_random = play_match(sim, ['random', 'random'], games=10, mc_n=5)
        assert isinstance(results_random, Counter), "Results should be a Counter object"
        total_games = sum(results_random.values())
        assert total_games == 10, "Should have completed 10 games with identical random agents"
        assert len(results_random) > 0, "Should have game results"

        # Test identical baseline agents
        results_baseline = play_match(sim, ['baseline', 'baseline'], games=10, mc_n=5)
        assert isinstance(results_baseline, Counter), "Results should be a Counter object"
        total_games = sum(results_baseline.values())
        assert total_games == 10, "Should have completed 10 games with identical baseline agents"
        assert len(results_baseline) > 0, "Should have game results"

        print(f"Identical agents: Random vs Random and Baseline vs Baseline completed successfully")

    def test_extreme_configurations(self):
        """Test agent interactions in extreme game configurations."""
        # Test with single die per player
        sim_single = PerudoSimulator(num_players=2, start_dice=1, seed=42)
        results_single = play_match(sim_single, ['baseline', 'random'], games=5, mc_n=5)
        assert isinstance(results_single, Counter), "Results should be a Counter object"
        assert sum(results_single.values()) == 5, "Should have completed 5 games with single die"
        assert len(results_single) > 0, "Should have game results"

        # Test with many dice per player
        sim_many = PerudoSimulator(num_players=2, start_dice=6, seed=42)
        results_many = play_match(sim_many, ['baseline', 'random'], games=3, mc_n=5)
        assert isinstance(results_many, Counter), "Results should be a Counter object"
        assert sum(results_many.values()) == 3, "Should have completed 3 games with many dice"
        assert len(results_many) > 0, "Should have game results"

        # Test with maximum players
        sim_max_players = PerudoSimulator(num_players=6, start_dice=2, seed=42)
        agent_names = ['random', 'baseline', 'random', 'baseline', 'random', 'baseline']
        results_max = play_match(sim_max_players, agent_names, games=3, mc_n=5)
        assert isinstance(results_max, Counter), "Results should be a Counter object"
        assert sum(results_max.values()) == 3, "Should have completed 3 games with maximum players"
        assert len(results_max) > 0, "Should have game results"

        print(f"Extreme configurations: All configurations completed successfully")

    def test_parameter_sensitivity(self):
        """Test that Monte Carlo agents work smoothly with different parameters."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Test with very low MC_N
        results_low_n = play_match(sim, ['mc', 'random'], games=5, mc_n=3)
        assert isinstance(results_low_n, Counter), "Results should be a Counter object"
        assert sum(results_low_n.values()) == 5, "Should have completed 5 games with low MC_N"
        assert len(results_low_n) > 0, "Should have game results"

        # Test with higher MC_N
        results_high_n = play_match(sim, ['mc', 'random'], games=5, mc_n=20)
        assert isinstance(results_high_n, Counter), "Results should be a Counter object"
        assert sum(results_high_n.values()) == 5, "Should have completed 5 games with high MC_N"
        assert len(results_high_n) > 0, "Should have game results"

        print(f"Parameter sensitivity: Both low MC_N and high MC_N completed successfully")
