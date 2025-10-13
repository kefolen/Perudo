"""
Regression tests for parameter changes in the Perudo Game AI project.

This module tests the impact of parameter changes, specifically focusing on
MC_N reduction and its effects on performance, win rates, decision quality,
and game completion times. These tests ensure that parameter optimizations
don't negatively impact core functionality.
"""

import pytest
import time
import statistics
from collections import Counter
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent
from agents.baseline_agent import BaselineAgent
from agents.random_agent import RandomAgent
from eval.tournament import play_match


class TestParameterChanges:
    """Test suite for parameter change regression testing."""

    def test_mc_n_reduction_impact(self):
        """Test the impact of MC_N reduction from 1000 to 100 on agent performance."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=42)

        # Test original high MC_N vs reduced MC_N
        original_mc_n = 1000
        reduced_mc_n = 100

        # Run tournaments with both configurations
        print(f"Testing MC_N reduction impact: {original_mc_n} -> {reduced_mc_n}")

        # Test with original MC_N (limited games due to performance)
        start_time = time.perf_counter()
        original_results = play_match(
            sim, ['mc', 'baseline', 'random'], 
            games=3, mc_n=original_mc_n, max_rounds=4
        )
        original_time = time.perf_counter() - start_time

        # Test with reduced MC_N
        start_time = time.perf_counter()
        reduced_results = play_match(
            sim, ['mc', 'baseline', 'random'], 
            games=10, mc_n=reduced_mc_n, max_rounds=4
        )
        reduced_time = time.perf_counter() - start_time

        # Verify both tournaments completed successfully
        assert sum(original_results.values()) == 3, "Original MC_N tournament should complete"
        assert sum(reduced_results.values()) == 10, "Reduced MC_N tournament should complete"

        # Calculate performance metrics
        original_avg_time = original_time / sum(original_results.values())
        reduced_avg_time = reduced_time / sum(reduced_results.values())

        # Verify significant performance improvement
        performance_improvement = original_avg_time / reduced_avg_time
        assert performance_improvement > 2.0, f"MC_N reduction should provide significant speedup (got {performance_improvement:.2f}x)"

        # Verify MC agent still performs reasonably (should beat random)
        # Player 0: mc, Player 1: baseline, Player 2: random
        mc_wins_original = original_results.get(0, 0)
        random_wins_original = original_results.get(2, 0)
        mc_wins_reduced = reduced_results.get(0, 0)
        random_wins_reduced = reduced_results.get(2, 0)

        # MC should outperform random in both cases
        assert mc_wins_original >= random_wins_original, "MC should outperform random with original MC_N"
        assert mc_wins_reduced >= random_wins_reduced, "MC should outperform random with reduced MC_N"

        print(f"MC_N reduction results:")
        print(f"  Original MC_N={original_mc_n}: {original_avg_time:.2f}s per game")
        print(f"  Reduced MC_N={reduced_mc_n}: {reduced_avg_time:.2f}s per game")
        print(f"  Performance improvement: {performance_improvement:.2f}x")

    def test_win_rate_comparison(self):
        """Compare win rates before and after MC_N parameter changes."""
        sim = PerudoSimulator(num_players=3, start_dice=3, seed=123)

        # Test different MC_N values for win rate comparison
        mc_n_values = [50, 100, 200]
        win_rate_results = {}

        for mc_n in mc_n_values:
            # Run multiple tournaments for statistical significance
            all_results = []
            for seed_offset in range(3):  # 3 different seeds
                test_sim = PerudoSimulator(num_players=3, start_dice=3, seed=123 + seed_offset)
                results = play_match(
                    test_sim, ['mc', 'baseline', 'random'], 
                    games=15, mc_n=mc_n, max_rounds=5
                )
                all_results.append(results)

            # Calculate aggregate win rates (results are indexed by player position)
            # Player 0: mc, Player 1: baseline, Player 2: random
            total_mc_wins = sum(r.get(0, 0) for r in all_results)
            total_baseline_wins = sum(r.get(1, 0) for r in all_results)
            total_random_wins = sum(r.get(2, 0) for r in all_results)
            total_games = total_mc_wins + total_baseline_wins + total_random_wins

            win_rate_results[mc_n] = {
                'mc_win_rate': total_mc_wins / total_games if total_games > 0 else 0,
                'baseline_win_rate': total_baseline_wins / total_games if total_games > 0 else 0,
                'random_win_rate': total_random_wins / total_games if total_games > 0 else 0,
                'total_games': total_games
            }

            # Verify tournament completed successfully
            assert total_games == 45, f"Should complete 45 games total for MC_N={mc_n} (got {total_games})"

        # Verify win rate consistency across MC_N values
        mc_win_rates = [results['mc_win_rate'] for results in win_rate_results.values()]
        baseline_win_rates = [results['baseline_win_rate'] for results in win_rate_results.values()]

        # MC should consistently outperform baseline and random
        for mc_n, results in win_rate_results.items():
            assert results['mc_win_rate'] > results['random_win_rate'], f"MC should beat random with MC_N={mc_n}"
            # Allow some flexibility for baseline comparison due to variance
            assert results['mc_win_rate'] >= results['baseline_win_rate'] * 0.8, f"MC should be competitive with baseline with MC_N={mc_n}"

        # Win rates shouldn't vary dramatically with MC_N changes
        mc_win_rate_std = statistics.stdev(mc_win_rates) if len(mc_win_rates) > 1 else 0
        assert mc_win_rate_std < 0.15, f"MC win rates should be relatively stable across MC_N values (std={mc_win_rate_std:.3f})"

        print(f"Win rate comparison results:")
        for mc_n, results in win_rate_results.items():
            print(f"  MC_N={mc_n}: MC={results['mc_win_rate']:.3f}, Baseline={results['baseline_win_rate']:.3f}, Random={results['random_win_rate']:.3f}")

    def test_decision_quality_maintenance(self):
        """Test that decision quality is maintained despite parameter changes."""
        # Test decision consistency across different MC_N values by running tournaments
        # and analyzing the performance consistency
        mc_n_values = [50, 100, 200]
        decision_quality_results = {}

        for mc_n in mc_n_values:
            # Run multiple small tournaments to test decision quality
            tournament_results = []
            performance_metrics = []

            for seed_offset in range(5):  # 5 different scenarios
                sim = PerudoSimulator(num_players=2, start_dice=3, seed=456 + seed_offset)

                # Run a small tournament to measure decision quality
                start_time = time.perf_counter()
                results = play_match(
                    sim, ['mc', 'baseline'], 
                    games=6, mc_n=mc_n, max_rounds=4
                )
                end_time = time.perf_counter()

                tournament_results.append(results)

                # Calculate performance metrics
                total_games = sum(results.values())
                mc_wins = results.get(0, 0)  # MC is player 0
                baseline_wins = results.get(1, 0)  # Baseline is player 1
                mc_win_rate = mc_wins / total_games if total_games > 0 else 0
                avg_time_per_game = (end_time - start_time) / total_games if total_games > 0 else 0

                performance_metrics.append({
                    'mc_win_rate': mc_win_rate,
                    'avg_time_per_game': avg_time_per_game,
                    'total_games': total_games
                })

            # Aggregate results for this MC_N value
            total_mc_wins = sum(r.get(0, 0) for r in tournament_results)
            total_baseline_wins = sum(r.get(1, 0) for r in tournament_results)
            total_games = total_mc_wins + total_baseline_wins
            overall_mc_win_rate = total_mc_wins / total_games if total_games > 0 else 0
            avg_game_time = statistics.mean([m['avg_time_per_game'] for m in performance_metrics])

            decision_quality_results[mc_n] = {
                'overall_mc_win_rate': overall_mc_win_rate,
                'avg_game_time': avg_game_time,
                'total_games': total_games,
                'performance_metrics': performance_metrics
            }

            # Verify tournaments completed successfully
            assert total_games == 30, f"Should complete 30 games total for MC_N={mc_n} (got {total_games})"

        # Verify decision quality is maintained across MC_N values
        win_rates = [results['overall_mc_win_rate'] for results in decision_quality_results.values()]
        game_times = [results['avg_game_time'] for results in decision_quality_results.values()]

        # MC should maintain reasonable performance across different MC_N values
        for mc_n, results in decision_quality_results.items():
            # MC should be competitive (win rate should be reasonable)
            assert results['overall_mc_win_rate'] >= 0.3, f"MC should maintain reasonable performance with MC_N={mc_n} (got {results['overall_mc_win_rate']:.3f})"

            # Game times should be reasonable
            assert results['avg_game_time'] < 10.0, f"Games should complete in reasonable time with MC_N={mc_n} (got {results['avg_game_time']:.2f}s)"

        # Win rates should be relatively consistent across MC_N values
        if len(win_rates) > 1:
            win_rate_std = statistics.stdev(win_rates)
            assert win_rate_std < 0.2, f"Win rates should be relatively consistent across MC_N values (std={win_rate_std:.3f})"

        # Higher MC_N should generally not be significantly worse than lower MC_N
        min_mc_n = min(mc_n_values)
        max_mc_n = max(mc_n_values)
        min_win_rate = decision_quality_results[min_mc_n]['overall_mc_win_rate']
        max_win_rate = decision_quality_results[max_mc_n]['overall_mc_win_rate']

        # Allow some variance but ensure quality is maintained
        assert max_win_rate >= min_win_rate * 0.7, f"Higher MC_N should maintain decision quality (min: {min_win_rate:.3f}, max: {max_win_rate:.3f})"

        print(f"Decision quality maintenance results:")
        for mc_n, results in decision_quality_results.items():
            print(f"  MC_N={mc_n}: Win rate={results['overall_mc_win_rate']:.3f}, Avg time={results['avg_game_time']:.2f}s")

    def test_game_completion_time(self):
        """Test that game completion times remain acceptable after parameter changes."""
        sim = PerudoSimulator(num_players=4, start_dice=3, seed=789)

        # Test completion times for different MC_N values
        mc_n_values = [25, 50, 100, 150]
        completion_times = {}

        for mc_n in mc_n_values:
            times = []

            # Run multiple games to get average completion time
            for game_num in range(5):
                start_time = time.perf_counter()

                results = play_match(
                    sim, ['mc', 'baseline', 'random', 'random'], 
                    games=1, mc_n=mc_n, max_rounds=6
                )

                end_time = time.perf_counter()
                game_time = end_time - start_time
                times.append(game_time)

                # Verify game completed
                assert sum(results.values()) == 1, f"Game should complete with MC_N={mc_n}"

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            completion_times[mc_n] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'times': times
            }

            # Verify reasonable completion times
            assert avg_time < 30.0, f"Average game time should be reasonable with MC_N={mc_n} (got {avg_time:.2f}s)"
            assert all(t > 0 for t in times), f"All game times should be positive with MC_N={mc_n}"

        # Verify completion time scaling is reasonable
        min_mc_n = min(mc_n_values)
        max_mc_n = max(mc_n_values)
        min_time = completion_times[min_mc_n]['avg_time']
        max_time = completion_times[max_mc_n]['avg_time']

        # Higher MC_N should take more time, but not excessively more
        time_ratio = max_time / min_time if min_time > 0 else 1
        assert time_ratio < 10.0, f"Time scaling should be reasonable (got {time_ratio:.2f}x from MC_N={min_mc_n} to MC_N={max_mc_n})"

        # Verify that MC_N=100 (target parameter) has acceptable performance
        if 100 in completion_times:
            mc_100_time = completion_times[100]['avg_time']
            assert mc_100_time < 15.0, f"MC_N=100 should have good performance (got {mc_100_time:.2f}s per game)"

        print(f"Game completion time results:")
        for mc_n, data in completion_times.items():
            print(f"  MC_N={mc_n}: {data['avg_time']:.2f}±{data['std_time']:.2f}s per game")


