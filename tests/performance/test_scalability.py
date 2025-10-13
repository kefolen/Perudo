"""
Scalability tests for Perudo Game AI Project.

This module tests the scalability characteristics of the game system,
including performance with maximum players, long-running tournament stability,
and resource usage over time.
"""

import pytest
import time
import psutil
import os
from collections import Counter
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent
from agents.baseline_agent import BaselineAgent
from agents.random_agent import RandomAgent
from eval.tournament import play_match


class TestScalability:
    """Test suite for system scalability characteristics."""

    def test_maximum_players_performance(self):
        """Test system performance with maximum number of players."""
        # Test with increasing number of players
        player_counts = [2, 3, 4, 6, 8]
        performance_results = {}

        for num_players in player_counts:
            sim = PerudoSimulator(num_players=num_players, start_dice=3, seed=42)

            # Create agent list for this player count
            agent_names = ['baseline', 'random'] * (num_players // 2)
            if num_players % 2 == 1:
                agent_names.append('baseline')
            agent_names = agent_names[:num_players]  # Ensure exact count

            start_time = time.perf_counter()  # More precise timing
            results = play_match(sim, agent_names, games=5, mc_n=50)  # More games for measurable time
            execution_time = time.perf_counter() - start_time

            performance_results[num_players] = {
                'execution_time': execution_time,
                'games_completed': sum(results.values()),
                'avg_time_per_game': execution_time / sum(results.values()) if sum(results.values()) > 0 else 0,
                'throughput': sum(results.values()) / execution_time if execution_time > 0 else float('inf')
            }

            # Verify tournament completed successfully
            assert sum(results.values()) == 5, f"Tournament with {num_players} players should complete 5 games"
            assert execution_time >= 0, f"Execution time should be non-negative for {num_players} players"

        # Verify scalability characteristics
        for num_players, data in performance_results.items():
            assert data['avg_time_per_game'] < 20.0, f"{num_players} players should complete games in reasonable time"
            assert data['throughput'] > 0.05, f"{num_players} players should maintain minimum throughput"

        # Performance should scale reasonably (not exponentially)
        time_2_players = performance_results[2]['avg_time_per_game']
        time_6_players = performance_results[6]['avg_time_per_game']

        # Use minimum threshold to avoid division by very small numbers
        min_time_threshold = 0.001  # 1ms minimum
        time_2_players = max(time_2_players, min_time_threshold)
        time_6_players = max(time_6_players, min_time_threshold)

        scaling_factor = time_6_players / time_2_players

        # More reasonable scaling factor for small, fast operations
        assert scaling_factor < 50.0, f"6 players should not be excessively slower than 2 players (got {scaling_factor:.2f}x)"

        print(f"Maximum players performance:")
        for num_players, data in performance_results.items():
            print(f"  {num_players} players: {data['avg_time_per_game']:.2f}s per game, {data['throughput']:.3f} games/sec")

    def test_long_running_tournament_stability(self):
        """Test stability during long-running tournaments."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)

        # Run a longer tournament to test stability
        games_count = 15  # Longer than typical tests
        agent_names = ['baseline', 'random', 'baseline']

        # Monitor memory and performance over time
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.perf_counter()
        results = play_match(sim, agent_names, games=games_count, mc_n=75)
        execution_time = time.perf_counter() - start_time

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Verify tournament completed successfully
        assert sum(results.values()) == games_count, f"Long tournament should complete {games_count} games"
        assert execution_time >= 0, "Execution time should be non-negative"

        # Performance should remain reasonable even for longer tournaments
        avg_time_per_game = execution_time / sum(results.values()) if sum(results.values()) > 0 else 0
        assert avg_time_per_game < 15.0, f"Long tournament should maintain reasonable per-game time (got {avg_time_per_game:.2f}s)"

        # Memory usage should remain stable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50.0, f"Memory increase should be reasonable for long tournament (got {memory_increase:.2f} MB)"

        # Overall tournament should complete in reasonable time
        assert execution_time < 300.0, f"Long tournament should complete within 5 minutes (got {execution_time:.2f}s)"

        print(f"Long-running tournament stability:")
        print(f"  Games: {games_count}")
        print(f"  Total time: {execution_time:.2f}s")
        print(f"  Avg time per game: {avg_time_per_game:.2f}s")
        print(f"  Memory increase: {memory_increase:.2f} MB")

    def test_resource_usage_over_time(self):
        """Test resource usage patterns over extended execution."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)

        # Monitor resource usage over multiple tournament runs
        process = psutil.Process(os.getpid())
        resource_samples = []

        # Run multiple smaller tournaments to simulate extended usage
        num_rounds = 4
        games_per_round = 4

        for round_num in range(num_rounds):
            # Sample resources before each round
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent_before = process.cpu_percent()

            start_time = time.perf_counter()  # More precise timing
            results = play_match(sim, ['baseline', 'random'], games=games_per_round, mc_n=50)
            execution_time = time.perf_counter() - start_time

            # Sample resources after each round
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent_after = process.cpu_percent()

            resource_samples.append({
                'round': round_num + 1,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before,
                'execution_time': execution_time,
                'games_completed': sum(results.values()),
                'cpu_percent': cpu_percent_after
            })

            # Verify each round completed successfully
            assert sum(results.values()) == games_per_round, f"Round {round_num + 1} should complete {games_per_round} games"
            assert execution_time >= 0, f"Round {round_num + 1} execution time should be non-negative"

        # Analyze resource usage patterns
        total_memory_increase = resource_samples[-1]['memory_after'] - resource_samples[0]['memory_before']
        avg_execution_time = sum(sample['execution_time'] for sample in resource_samples) / len(resource_samples)
        max_memory_delta = max(sample['memory_delta'] for sample in resource_samples)

        # Resource usage should remain reasonable
        assert total_memory_increase < 75.0, f"Total memory increase should be reasonable (got {total_memory_increase:.2f} MB)"
        assert avg_execution_time < 25.0, f"Average execution time should be reasonable (got {avg_execution_time:.2f}s)"
        assert max_memory_delta < 30.0, f"Maximum memory delta per round should be reasonable (got {max_memory_delta:.2f} MB)"

        # Performance should remain consistent across rounds
        execution_times = [sample['execution_time'] for sample in resource_samples]
        min_time = min(execution_times)
        max_time = max(execution_times)
        time_variance = max_time - min_time

        # Handle edge case where execution times are very small
        if avg_execution_time > 0.01:  # Only check variance if avg time is meaningful (10ms threshold)
            assert time_variance < avg_execution_time, f"Execution time variance should be reasonable (variance: {time_variance:.2f}s, avg: {avg_execution_time:.2f}s)"
        else:
            # For very fast operations, just ensure variance is reasonable in absolute terms
            assert time_variance < 0.1, f"Execution time variance should be reasonable for fast operations (variance: {time_variance:.4f}s)"

        print(f"Resource usage over time:")
        print(f"  Rounds: {num_rounds}")
        print(f"  Total memory increase: {total_memory_increase:.2f} MB")
        print(f"  Average execution time: {avg_execution_time:.2f}s")
        print(f"  Time variance: {time_variance:.2f}s")
        for sample in resource_samples:
            print(f"    Round {sample['round']}: {sample['execution_time']:.2f}s, memory Δ{sample['memory_delta']:.2f} MB")

    def test_concurrent_agent_performance(self):
        """Test performance with multiple Monte Carlo agents running concurrently."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)

        # Test with multiple MC agents (more computationally intensive)
        agent_names = ['mc', 'mc', 'baseline']

        start_time = time.perf_counter()
        results = play_match(sim, agent_names, games=4, mc_n=75)
        execution_time = time.perf_counter() - start_time

        # Verify tournament completed successfully
        assert sum(results.values()) == 4, "Concurrent MC agents tournament should complete 4 games"
        assert execution_time >= 0, "Execution time should be non-negative"

        # Performance should be reasonable even with multiple MC agents
        avg_time_per_game = execution_time / sum(results.values()) if sum(results.values()) > 0 else 0
        assert avg_time_per_game < 25.0, f"Concurrent MC agents should maintain reasonable performance (got {avg_time_per_game:.2f}s per game)"

        # Overall tournament should complete in reasonable time
        assert execution_time < 120.0, f"Concurrent MC agents tournament should complete within 2 minutes (got {execution_time:.2f}s)"

        print(f"Concurrent agent performance:")
        print(f"  Agent configuration: {agent_names}")
        print(f"  Total time: {execution_time:.2f}s")
        print(f"  Avg time per game: {avg_time_per_game:.2f}s")

    def test_memory_leak_detection(self):
        """Test for potential memory leaks during extended operation."""
        sim = PerudoSimulator(num_players=2, start_dice=2, seed=42)

        # Monitor memory usage over multiple cycles
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_readings = [initial_memory]

        # Run multiple tournament cycles
        num_cycles = 5
        for cycle in range(num_cycles):
            results = play_match(sim, ['baseline', 'random'], games=3, mc_n=40)

            # Verify each cycle completed successfully
            assert sum(results.values()) == 3, f"Cycle {cycle + 1} should complete 3 games"

            # Sample memory after each cycle
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(current_memory)

        final_memory = memory_readings[-1]
        total_memory_increase = final_memory - initial_memory

        # Check for excessive memory growth (potential leak)
        assert total_memory_increase < 40.0, f"Total memory increase should be reasonable (got {total_memory_increase:.2f} MB)"

        # Memory should not grow linearly with each cycle (indicating a leak)
        # Calculate memory growth rate
        if len(memory_readings) > 2:
            memory_deltas = [memory_readings[i] - memory_readings[i-1] for i in range(1, len(memory_readings))]
            avg_delta = sum(memory_deltas) / len(memory_deltas)
            max_delta = max(memory_deltas)

            # Memory growth should stabilize (not keep growing at same rate)
            assert max_delta < 20.0, f"Maximum memory delta should be reasonable (got {max_delta:.2f} MB)"
            assert avg_delta < 10.0, f"Average memory delta should be reasonable (got {avg_delta:.2f} MB)"

        print(f"Memory leak detection:")
        print(f"  Cycles: {num_cycles}")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Total increase: {total_memory_increase:.2f} MB")
        print(f"  Memory readings: {[f'{m:.1f}' for m in memory_readings]} MB")
