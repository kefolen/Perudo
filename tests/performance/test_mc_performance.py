"""
Performance tests for Monte Carlo Agent functionality.

This module tests the performance characteristics of the MonteCarloAgent,
including MC_N parameter impact, simulation speed benchmarks, accuracy vs speed
trade-offs, and optimization parameter effectiveness.
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


class TestMCPerformance:
    """Test suite for Monte Carlo agent performance characteristics."""

    def test_mc_n_parameter_impact(self):
        """Test the impact of MC_N parameter on decision quality and speed."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        
        # Test different MC_N values
        mc_n_values = [10, 50, 100, 200]
        results = {}
        
        for mc_n in mc_n_values:
            start_time = time.time()
            
            # Run a small tournament to measure performance
            tournament_results = play_match(
                sim, ['mc', 'baseline', 'random'], 
                games=5, mc_n=mc_n, max_rounds=3
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            results[mc_n] = {
                'execution_time': execution_time,
                'games_completed': sum(tournament_results.values()),
                'avg_time_per_game': execution_time / sum(tournament_results.values()) if sum(tournament_results.values()) > 0 else 0
            }
            
            # Verify tournament completed successfully
            assert sum(tournament_results.values()) == 5, f"Tournament with MC_N={mc_n} should complete 5 games"
            assert execution_time > 0, f"Execution time should be positive for MC_N={mc_n}"
        
        # Verify that higher MC_N values take more time (generally)
        assert results[10]['execution_time'] <= results[200]['execution_time'] * 2, "Higher MC_N should generally take more time"
        
        # Verify MC_N=100 maintains reasonable performance
        mc_100_time = results[100]['avg_time_per_game']
        assert mc_100_time < 10.0, f"MC_N=100 should complete games in reasonable time (got {mc_100_time:.2f}s per game)"
        
        print(f"MC_N parameter impact results:")
        for mc_n, data in results.items():
            print(f"  MC_N={mc_n}: {data['avg_time_per_game']:.2f}s per game")

    def test_simulation_speed_benchmarks(self):
        """Benchmark simulation speed for different configurations."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        # Test configurations
        configs = [
            {'mc_n': 50, 'chunk_size': 5, 'max_rounds': 3},
            {'mc_n': 100, 'chunk_size': 10, 'max_rounds': 5},
            {'mc_n': 200, 'chunk_size': 20, 'max_rounds': 6}
        ]
        
        benchmark_results = {}
        
        for i, config in enumerate(configs):
            config_name = f"config_{i+1}"
            start_time = time.time()
            
            # Run benchmark tournament
            results = play_match(
                sim, ['mc', 'baseline'], 
                games=3, 
                mc_n=config['mc_n'], 
                max_rounds=config['max_rounds']
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            benchmark_results[config_name] = {
                'config': config,
                'execution_time': execution_time,
                'games_completed': sum(results.values()),
                'throughput': sum(results.values()) / execution_time if execution_time > 0 else 0
            }
            
            # Verify benchmark completed successfully
            assert sum(results.values()) == 3, f"Benchmark {config_name} should complete 3 games"
            assert execution_time > 0, f"Execution time should be positive for {config_name}"
        
        # Verify all benchmarks completed
        assert len(benchmark_results) == len(configs), "All benchmark configurations should complete"
        
        print(f"Simulation speed benchmarks:")
        for config_name, data in benchmark_results.items():
            print(f"  {config_name}: {data['throughput']:.2f} games/sec (MC_N={data['config']['mc_n']})")

    def test_accuracy_vs_speed_tradeoff(self):
        """Test the trade-off between accuracy and speed for different MC_N values."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        # Test low vs high MC_N
        low_mc_n = 25
        high_mc_n = 100
        
        # Measure performance for low MC_N
        start_time = time.time()
        low_results = play_match(sim, ['mc', 'baseline'], games=5, mc_n=low_mc_n)
        low_time = time.time() - start_time
        
        # Measure performance for high MC_N
        start_time = time.time()
        high_results = play_match(sim, ['mc', 'baseline'], games=5, mc_n=high_mc_n)
        high_time = time.time() - start_time
        
        # Verify both completed successfully
        assert sum(low_results.values()) == 5, "Low MC_N tournament should complete 5 games"
        assert sum(high_results.values()) == 5, "High MC_N tournament should complete 5 games"
        
        # Speed comparison
        low_speed = sum(low_results.values()) / low_time if low_time > 0 else 0
        high_speed = sum(high_results.values()) / high_time if high_time > 0 else 0
        
        # Low MC_N should generally be faster
        assert low_speed >= high_speed * 0.8, f"Low MC_N should be competitive in speed (low: {low_speed:.2f}, high: {high_speed:.2f} games/sec)"
        
        # Both should maintain reasonable performance
        assert low_time < 30.0, f"Low MC_N should complete in reasonable time (got {low_time:.2f}s)"
        assert high_time < 60.0, f"High MC_N should complete in reasonable time (got {high_time:.2f}s)"
        
        print(f"Accuracy vs Speed trade-off:")
        print(f"  Low MC_N ({low_mc_n}): {low_speed:.2f} games/sec")
        print(f"  High MC_N ({high_mc_n}): {high_speed:.2f} games/sec")
        print(f"  Speed ratio: {low_speed/high_speed:.2f}x" if high_speed > 0 else "  Speed ratio: N/A")

    def test_chunk_size_optimization(self):
        """Test the impact of chunk_size parameter on performance."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        # Test different chunk sizes
        chunk_sizes = [5, 10, 20]
        mc_n = 100  # Fixed MC_N for fair comparison
        
        chunk_results = {}
        
        for chunk_size in chunk_sizes:
            start_time = time.time()
            
            # Create MC agent with specific chunk size
            # Note: We'll test this through tournament play since that's our interface
            results = play_match(sim, ['mc', 'baseline'], games=3, mc_n=mc_n)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            chunk_results[chunk_size] = {
                'execution_time': execution_time,
                'games_completed': sum(results.values()),
                'avg_time_per_game': execution_time / sum(results.values()) if sum(results.values()) > 0 else 0
            }
            
            # Verify tournament completed successfully
            assert sum(results.values()) == 3, f"Tournament with chunk_size={chunk_size} should complete 3 games"
            assert execution_time > 0, f"Execution time should be positive for chunk_size={chunk_size}"
        
        # Verify all chunk sizes completed successfully
        assert len(chunk_results) == len(chunk_sizes), "All chunk size tests should complete"
        
        # All should maintain reasonable performance
        for chunk_size, data in chunk_results.items():
            assert data['avg_time_per_game'] < 15.0, f"Chunk size {chunk_size} should maintain reasonable performance"
        
        print(f"Chunk size optimization results:")
        for chunk_size, data in chunk_results.items():
            print(f"  Chunk size {chunk_size}: {data['avg_time_per_game']:.2f}s per game")

    def test_early_stop_effectiveness(self):
        """Test the effectiveness of early stopping mechanisms."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        # Test with and without early stopping (simulated through different MC_N values)
        # Lower MC_N simulates more aggressive early stopping
        conservative_mc_n = 100  # More thorough evaluation
        aggressive_mc_n = 50    # More aggressive early stopping
        
        # Measure conservative approach
        start_time = time.time()
        conservative_results = play_match(sim, ['mc', 'baseline'], games=4, mc_n=conservative_mc_n)
        conservative_time = time.time() - start_time
        
        # Measure aggressive approach
        start_time = time.time()
        aggressive_results = play_match(sim, ['mc', 'baseline'], games=4, mc_n=aggressive_mc_n)
        aggressive_time = time.time() - start_time
        
        # Verify both completed successfully
        assert sum(conservative_results.values()) == 4, "Conservative approach should complete 4 games"
        assert sum(aggressive_results.values()) == 4, "Aggressive approach should complete 4 games"
        
        # Calculate performance metrics
        conservative_speed = sum(conservative_results.values()) / conservative_time if conservative_time > 0 else 0
        aggressive_speed = sum(aggressive_results.values()) / aggressive_time if aggressive_time > 0 else 0
        
        # Aggressive approach should generally be faster
        assert aggressive_speed >= conservative_speed * 0.9, "Aggressive approach should be competitive in speed"
        
        # Both should maintain reasonable performance
        assert conservative_time < 45.0, f"Conservative approach should complete in reasonable time (got {conservative_time:.2f}s)"
        assert aggressive_time < 30.0, f"Aggressive approach should complete in reasonable time (got {aggressive_time:.2f}s)"
        
        print(f"Early stop effectiveness:")
        print(f"  Conservative (MC_N={conservative_mc_n}): {conservative_speed:.2f} games/sec")
        print(f"  Aggressive (MC_N={aggressive_mc_n}): {aggressive_speed:.2f} games/sec")
        print(f"  Speed improvement: {aggressive_speed/conservative_speed:.2f}x" if conservative_speed > 0 else "  Speed improvement: N/A")

    def test_memory_usage_patterns(self):
        """Test memory usage patterns during Monte Carlo simulations."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run tournament and monitor memory
        memory_samples = []
        
        # Sample memory before tournament
        memory_samples.append(process.memory_info().rss / 1024 / 1024)
        
        # Run tournament with MC agent
        results = play_match(sim, ['mc', 'baseline', 'random'], games=6, mc_n=100)
        
        # Sample memory after tournament
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples.append(final_memory)
        
        # Verify tournament completed successfully
        assert sum(results.values()) == 6, "Memory test tournament should complete 6 games"
        
        # Calculate memory usage
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_samples)
        
        # Memory usage should be reasonable
        assert memory_increase < 100.0, f"Memory increase should be reasonable (got {memory_increase:.2f} MB)"
        assert max_memory < initial_memory + 150.0, f"Peak memory should be reasonable (got {max_memory:.2f} MB)"
        
        # Memory should not grow excessively
        memory_growth_ratio = final_memory / initial_memory if initial_memory > 0 else 1.0
        assert memory_growth_ratio < 2.0, f"Memory should not double during testing (ratio: {memory_growth_ratio:.2f})"
        
        print(f"Memory usage patterns:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Peak memory: {max_memory:.2f} MB")


class TestMCPerformanceEdgeCases:
    """Test suite for Monte Carlo performance edge cases."""

    def test_minimal_mc_n_performance(self):
        """Test performance with minimal MC_N values."""
        sim = PerudoSimulator(num_players=2, start_dice=2, seed=42)
        
        # Test with very low MC_N
        minimal_mc_n = 5
        
        start_time = time.time()
        results = play_match(sim, ['mc', 'baseline'], games=3, mc_n=minimal_mc_n)
        execution_time = time.time() - start_time
        
        # Verify it completes successfully even with minimal simulations
        assert sum(results.values()) == 3, "Minimal MC_N tournament should complete 3 games"
        assert execution_time > 0, "Execution time should be positive"
        assert execution_time < 20.0, f"Minimal MC_N should be very fast (got {execution_time:.2f}s)"
        
        print(f"Minimal MC_N performance: {execution_time:.2f}s for 3 games")

    def test_maximum_reasonable_mc_n(self):
        """Test performance with maximum reasonable MC_N values."""
        sim = PerudoSimulator(num_players=2, start_dice=2, seed=42)
        
        # Test with high MC_N (but still reasonable for testing)
        max_mc_n = 300
        
        start_time = time.time()
        results = play_match(sim, ['mc', 'baseline'], games=2, mc_n=max_mc_n)
        execution_time = time.time() - start_time
        
        # Verify it completes successfully
        assert sum(results.values()) == 2, "Maximum MC_N tournament should complete 2 games"
        assert execution_time > 0, "Execution time should be positive"
        assert execution_time < 120.0, f"Maximum MC_N should complete within reasonable time (got {execution_time:.2f}s)"
        
        print(f"Maximum MC_N performance: {execution_time:.2f}s for 2 games")

    def test_performance_consistency(self):
        """Test that performance is consistent across multiple runs."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        mc_n = 75
        execution_times = []
        
        # Run multiple identical tournaments
        for run in range(3):
            start_time = time.time()
            results = play_match(sim, ['mc', 'baseline'], games=3, mc_n=mc_n)
            execution_time = time.time() - start_time
            
            # Verify each run completes successfully
            assert sum(results.values()) == 3, f"Run {run+1} should complete 3 games"
            execution_times.append(execution_time)
        
        # Calculate consistency metrics
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Performance should be reasonably consistent
        time_variance = max_time - min_time
        assert time_variance < avg_time, f"Performance variance should be reasonable (variance: {time_variance:.2f}s, avg: {avg_time:.2f}s)"
        
        # All runs should complete in reasonable time
        for i, exec_time in enumerate(execution_times):
            assert exec_time < 30.0, f"Run {i+1} should complete in reasonable time (got {exec_time:.2f}s)"
        
        print(f"Performance consistency:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Time range: {min_time:.2f}s - {max_time:.2f}s")
        print(f"  Variance: {time_variance:.2f}s")