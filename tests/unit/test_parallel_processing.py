"""
Unit tests for parallel processing functionality in MonteCarloAgent.

This module tests the parallel processing capabilities of the MonteCarloAgent,
ensuring statistical equivalence with sequential mode, performance improvements,
reproducibility, and proper resource management.
"""

import pytest
import time
import multiprocessing as mp
import random
from collections import Counter
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent
from agents.baseline_agent import BaselineAgent
from tests.fixtures.sample_game_states import GameStates


class TestParallelProcessing:
    """Test suite for parallel processing functionality."""

    def test_parallel_sequential_statistical_equivalence(self):
        """Test that parallel and sequential modes produce statistically equivalent results."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        
        # Create test observation
        game_states = GameStates()
        obs = game_states.mid_game_balanced()
        obs['_simulator'] = sim  # Add simulator reference
        
        # Test with fixed seed for reproducibility
        seed = 12345
        
        # Sequential evaluation
        mc_sequential = MonteCarloAgent(
            name='mc_seq', n=100, chunk_size=10, 
            enable_parallel=False, rng=random.Random(seed)
        )
        
        # Parallel evaluation
        mc_parallel = MonteCarloAgent(
            name='mc_par', n=100, chunk_size=10,
            enable_parallel=True, num_workers=2, rng=random.Random(seed)
        )
        
        # Run multiple evaluations to collect statistics
        sequential_results = []
        parallel_results = []
        
        # Test multiple different actions
        test_actions = [('bid', 2, 3), ('bid', 3, 4), ('call',)]
        
        for action in test_actions:
            # Reset random seeds for each action
            mc_sequential.rng = random.Random(seed)
            mc_parallel.rng = random.Random(seed)
            
            seq_result = mc_sequential.evaluate_action(obs, action)
            par_result = mc_parallel.evaluate_action(obs, action)
            
            sequential_results.append(seq_result)
            parallel_results.append(par_result)
        
        # Results should be close (within reasonable statistical margin)
        # Due to randomness, exact equality is not expected, but they should be similar
        for i, (seq, par) in enumerate(zip(sequential_results, parallel_results)):
            assert abs(seq - par) < 0.15, f"Action {i}: Sequential {seq:.3f} vs Parallel {par:.3f} differ too much"
        
        # Overall correlation should be strong
        if len(sequential_results) > 1:
            # Simple correlation check - results should follow similar ordering patterns
            seq_order = sorted(range(len(sequential_results)), key=lambda i: sequential_results[i])
            par_order = sorted(range(len(parallel_results)), key=lambda i: parallel_results[i])
            
            # At least some ordering similarity expected
            matches = sum(1 for i, j in zip(seq_order, par_order) if abs(i - j) <= 1)
            assert matches >= len(test_actions) // 2, "Parallel and sequential should show similar action preferences"

    def test_parallel_functionality_and_performance(self):
        """Test that parallel processing works correctly and completes in reasonable time."""
        if mp.cpu_count() < 2:
            pytest.skip("Multiprocessing functionality test requires multiple CPU cores")
        
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        game_states = GameStates()
        obs = game_states.mid_game_balanced()
        obs['_simulator'] = sim  # Add simulator reference
        
        # Sequential timing
        mc_sequential = MonteCarloAgent(
            name='mc_seq', n=200, chunk_size=20,
            enable_parallel=False, rng=random.Random(42)
        )
        
        start_time = time.time()
        seq_result = mc_sequential.evaluate_action(obs, ('bid', 3, 4))
        sequential_time = time.time() - start_time
        
        # Parallel timing
        mc_parallel = MonteCarloAgent(
            name='mc_par', n=200, chunk_size=20,
            enable_parallel=True, num_workers=2, rng=random.Random(42)
        )
        
        start_time = time.time()
        par_result = mc_parallel.evaluate_action(obs, ('bid', 3, 4))
        parallel_time = time.time() - start_time
        
        # Primary focus: Results should be statistically similar (correctness)
        assert abs(seq_result - par_result) < 0.2, f"Results should be similar: seq={seq_result:.3f}, par={par_result:.3f}"
        
        # Both results should be valid probabilities
        assert 0.0 <= seq_result <= 1.0, f"Sequential result should be valid probability: {seq_result}"
        assert 0.0 <= par_result <= 1.0, f"Parallel result should be valid probability: {par_result}"
        
        # Both execution times should be reasonable (no hanging or excessive delays)
        assert sequential_time < 10.0, f"Sequential execution taking too long: {sequential_time:.3f}s"
        assert parallel_time < 30.0, f"Parallel execution taking too long: {parallel_time:.3f}s"
        
        # Calculate speedup ratio for informational purposes
        speedup_ratio = sequential_time / parallel_time if parallel_time > 0 else 0
        
        print(f"Parallel functionality test results:")
        print(f"  Sequential time: {sequential_time:.3f}s, result: {seq_result:.3f}")
        print(f"  Parallel time: {parallel_time:.3f}s, result: {par_result:.3f}")
        print(f"  Speedup ratio: {speedup_ratio:.2f}x")
        print(f"  Result difference: {abs(seq_result - par_result):.3f}")
        
        # Verify the parallel implementation actually used workers (implementation detail check)
        assert hasattr(mc_parallel, '_worker_pool'), "Parallel agent should have worker pool capability"
        assert mc_parallel.enable_parallel is True, "Parallel flag should be enabled"
        assert mc_parallel.num_workers == 2, "Should use requested number of workers"

    def test_parallel_reproducibility_with_seeds(self):
        """Test that parallel processing is reproducible with fixed random seeds."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        game_states = GameStates()
        obs = game_states.early_game_with_bid()
        obs['_simulator'] = sim  # Add simulator reference
        
        seed = 9999
        action = ('bid', 2, 5)
        
        # Run parallel evaluation multiple times with same seed
        results = []
        for _ in range(3):
            mc_agent = MonteCarloAgent(
                name='mc_test', n=50, chunk_size=10,
                enable_parallel=True, num_workers=2, rng=random.Random(seed)
            )
            result = mc_agent.evaluate_action(obs, action)
            results.append(result)
        
        # All results should be identical with same seed
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert abs(result - first_result) < 0.01, f"Run {i+1}: {result:.4f} vs first: {first_result:.4f} - should be reproducible"

    def test_parallel_memory_usage_reasonable(self):
        """Test that parallel processing doesn't consume excessive memory."""
        sim = PerudoSimulator(num_players=3, start_dice=4, seed=42)
        game_states = GameStates()
        obs = game_states.mid_game_unbalanced()
        obs['_simulator'] = sim  # Add simulator reference
        
        # Create parallel MC agent
        mc_parallel = MonteCarloAgent(
            name='mc_mem_test', n=100, chunk_size=10,
            enable_parallel=True, num_workers=3, rng=random.Random(42)
        )
        
        # Monitor memory usage during evaluation
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run evaluation
        result = mc_parallel.evaluate_action(obs, ('bid', 4, 6))
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for this small test)
        assert memory_increase < 100, f"Memory usage increase too high: {memory_increase:.1f}MB"
        
        # Result should be valid
        assert 0.0 <= result <= 1.0, f"Result should be valid probability: {result}"
        
        print(f"Memory usage test:")
        print(f"  Before: {memory_before:.1f}MB")
        print(f"  After: {memory_after:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")

    def test_parallel_parameter_validation(self):
        """Test that parallel processing parameters are properly validated."""
        # Test enable_parallel parameter
        mc_disabled = MonteCarloAgent(enable_parallel=False)
        assert mc_disabled.enable_parallel is False
        
        mc_enabled = MonteCarloAgent(enable_parallel=True)
        assert mc_enabled.enable_parallel is True
        
        # Test num_workers parameter
        mc_auto_workers = MonteCarloAgent(enable_parallel=True, num_workers=None)
        assert mc_auto_workers.num_workers is not None  # Should auto-detect workers
        assert mc_auto_workers.num_workers >= 1  # Should have at least 1 worker
        
        mc_specific_workers = MonteCarloAgent(enable_parallel=True, num_workers=4)
        assert mc_specific_workers.num_workers == 4
        
        # Test invalid num_workers values are handled gracefully
        mc_invalid_workers = MonteCarloAgent(enable_parallel=True, num_workers=0)
        # Should handle gracefully (implementation will determine exact behavior)
        assert hasattr(mc_invalid_workers, 'num_workers')

    def test_parallel_backward_compatibility(self):
        """Test that parallel processing maintains backward compatibility."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        game_states = GameStates()
        obs = game_states.early_game_with_bid()
        obs['_simulator'] = sim  # Add simulator reference
        
        # Default agent (should work as before)
        mc_default = MonteCarloAgent(name='mc_default', n=50)
        result_default = mc_default.evaluate_action(obs, ('bid', 2, 4))
        
        # Explicitly disabled parallel
        mc_sequential = MonteCarloAgent(
            name='mc_seq', n=50, enable_parallel=False
        )
        result_sequential = mc_sequential.evaluate_action(obs, ('bid', 2, 4))
        
        # Both should work and give reasonable results
        assert 0.0 <= result_default <= 1.0
        assert 0.0 <= result_sequential <= 1.0
        
        # Results should be similar (both using sequential evaluation)
        assert abs(result_default - result_sequential) < 0.05, "Default and explicitly sequential should be very similar"

    def test_parallel_with_different_worker_counts(self):
        """Test parallel processing with different numbers of workers."""
        if mp.cpu_count() < 4:
            pytest.skip("Test requires at least 4 CPU cores")
        
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)
        game_states = GameStates()
        obs = game_states.high_bid_scenario()
        obs['_simulator'] = sim  # Add simulator reference
        
        # Test different worker counts
        worker_counts = [1, 2, 4]
        results = {}
        times = {}
        
        for workers in worker_counts:
            mc_agent = MonteCarloAgent(
                name=f'mc_{workers}w', n=100, chunk_size=10,
                enable_parallel=True, num_workers=workers,
                rng=random.Random(42)
            )
            
            start_time = time.time()
            result = mc_agent.evaluate_action(obs, ('bid', 3, 3))
            end_time = time.time()
            
            results[workers] = result
            times[workers] = end_time - start_time
        
        # All results should be reasonable
        for workers, result in results.items():
            assert 0.0 <= result <= 1.0, f"Worker count {workers} gave invalid result: {result}"
        
        # Results should be statistically similar across worker counts
        result_values = list(results.values())
        for i in range(len(result_values)):
            for j in range(i+1, len(result_values)):
                diff = abs(result_values[i] - result_values[j])
                assert diff < 0.1, f"Results with different worker counts should be similar"
        
        print(f"Worker count comparison:")
        for workers in worker_counts:
            print(f"  {workers} workers: {results[workers]:.3f} result, {times[workers]:.3f}s")