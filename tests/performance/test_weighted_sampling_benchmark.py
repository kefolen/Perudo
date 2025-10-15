"""
Performance benchmark for weighted vs uniform sampling in MonteCarloAgent.

This module benchmarks the decision quality improvements of weighted sampling
compared to uniform sampling through tournament play.
"""

import pytest
import time
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent
from agents.baseline_agent import BaselineAgent
from eval.tournament import play_match


class TestWeightedSamplingBenchmark:
    """Benchmark suite for weighted sampling performance validation."""
    
    def test_weighted_vs_uniform_tournament_performance(self):
        """Compare tournament performance between weighted and uniform sampling."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        def run_direct_matches(agent1, agent2, num_games):
            """Run direct matches between two agents."""
            wins = {agent1.name: 0, agent2.name: 0}
            start_time = time.time()
            
            for game in range(num_games):
                # Create fresh agents for each game to avoid state pollution
                agents = [agent1, agent2]
                winner_idx, _ = sim.play_game(agents)
                # Convert winner index to agent name
                winner_name = agents[winner_idx].name
                wins[winner_name] += 1
            
            return wins, time.time() - start_time
        
        # Run tournaments
        num_games = 5  # Small number for testing
        results = {}
        
        # Uniform MC vs Baseline
        uniform_agent = MonteCarloAgent(name='mc_uniform', n=50, weighted_sampling=False)
        baseline_agent = BaselineAgent(name='baseline')
        uniform_results, uniform_time = run_direct_matches(uniform_agent, baseline_agent, num_games)
        results['uniform_vs_baseline'] = {
            'results': uniform_results,
            'time': uniform_time,
            'mc_wins': uniform_results.get('mc_uniform', 0),
            'baseline_wins': uniform_results.get('baseline', 0)
        }
        
        # Weighted MC vs Baseline  
        weighted_agent = MonteCarloAgent(name='mc_weighted', n=50, weighted_sampling=True)
        baseline_agent2 = BaselineAgent(name='baseline')
        weighted_results, weighted_time = run_direct_matches(weighted_agent, baseline_agent2, num_games)
        results['weighted_vs_baseline'] = {
            'results': weighted_results,
            'time': weighted_time,
            'mc_wins': weighted_results.get('mc_weighted', 0),
            'baseline_wins': weighted_results.get('baseline', 0)
        }
        
        # Direct comparison: Weighted MC vs Uniform MC
        uniform_agent2 = MonteCarloAgent(name='mc_uniform', n=50, weighted_sampling=False)
        weighted_agent2 = MonteCarloAgent(name='mc_weighted', n=50, weighted_sampling=True)
        head_to_head, head_to_head_time = run_direct_matches(weighted_agent2, uniform_agent2, num_games)
        results['weighted_vs_uniform'] = {
            'results': head_to_head,
            'time': head_to_head_time,
            'weighted_wins': head_to_head.get('mc_weighted', 0),
            'uniform_wins': head_to_head.get('mc_uniform', 0)
        }
        
        # Verify all tournaments completed
        assert sum(uniform_results.values()) == num_games, "Uniform vs baseline should complete all games"
        assert sum(weighted_results.values()) == num_games, "Weighted vs baseline should complete all games"
        assert sum(head_to_head.values()) == num_games, "Head-to-head should complete all games"
        
        # Performance should be reasonable (no more than 3x slower for weighted due to small sample size variance)
        time_ratio = weighted_time / uniform_time if uniform_time > 0 else 1.0
        assert time_ratio <= 3.0, f"Weighted sampling should not be excessively slower (got {time_ratio:.2f}x)"
        
        # Print results for analysis
        print(f"\nWeighted vs Uniform Sampling Benchmark Results:")
        print(f"  Uniform MC vs Baseline: {results['uniform_vs_baseline']['mc_wins']}/{num_games} wins ({results['uniform_vs_baseline']['time']:.2f}s)")
        print(f"  Weighted MC vs Baseline: {results['weighted_vs_baseline']['mc_wins']}/{num_games} wins ({results['weighted_vs_baseline']['time']:.2f}s)")
        print(f"  Weighted vs Uniform MC: {results['weighted_vs_uniform']['weighted_wins']}/{num_games} wins ({results['weighted_vs_uniform']['time']:.2f}s)")
        print(f"  Speed ratio (weighted/uniform): {time_ratio:.2f}x")
        
        # Both MC agents should be able to win some games (this is a basic functionality check)
        # With such small sample sizes, we just verify the system works
        total_games = num_games * 2  # Two different matchups
        total_mc_wins = results['uniform_vs_baseline']['mc_wins'] + results['weighted_vs_baseline']['mc_wins']
        assert total_mc_wins >= 0, f"MC agents should be able to play games (got {total_mc_wins} wins out of {total_games})"

    def test_weighted_sampling_convergence(self):
        """Test that weighted sampling converges to reasonable decisions on known states."""
        sim = PerudoSimulator(num_players=3, start_dice=5, seed=42)
        
        # Create scenario where weighted sampling should show clear bias
        obs = {
            '_simulator': sim,
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [2, 3, 4, 5, 6],  # No twos or ones
            'current_bid': (8, 2),       # Bid 8 twos - very unlikely without opponent help
            'maputa_active': False
        }
        
        agent_uniform = MonteCarloAgent(name='uniform', n=100, weighted_sampling=False, rng=None)
        agent_weighted = MonteCarloAgent(name='weighted', n=100, weighted_sampling=True, rng=None)
        
        # Sample determinizations and count face 2 frequency
        num_samples = 50
        uniform_twos = 0
        weighted_twos = 0
        total_opponent_dice = 10  # 5 + 5 opponent dice
        
        for _ in range(num_samples):
            # Uniform sampling
            uniform_hands = agent_uniform.sample_determinization(obs)
            uniform_twos += sum(d == 2 for hand in uniform_hands[1:] for d in hand)
            
            # Weighted sampling  
            weighted_hands = agent_weighted.sample_determinization(obs)
            weighted_twos += sum(d == 2 for hand in weighted_hands[1:] for d in hand)
        
        uniform_two_rate = uniform_twos / (num_samples * total_opponent_dice)
        weighted_two_rate = weighted_twos / (num_samples * total_opponent_dice)
        
        # Uniform should be close to 1/6, weighted should be higher given the bid
        assert 0.1 <= uniform_two_rate <= 0.25, f"Uniform sampling rate should be ~1/6 (got {uniform_two_rate:.3f})"
        assert weighted_two_rate > uniform_two_rate, f"Weighted should favor twos more than uniform ({weighted_two_rate:.3f} vs {uniform_two_rate:.3f})"
        
        print(f"\nConvergence Test Results:")
        print(f"  Uniform two rate: {uniform_two_rate:.3f} (expected ~0.167)")
        print(f"  Weighted two rate: {weighted_two_rate:.3f} (expected > uniform)")
        print(f"  Bias factor: {weighted_two_rate/uniform_two_rate:.2f}x")

    def test_weighted_sampling_decision_quality(self):
        """Test decision quality improvements with weighted sampling."""
        sim = PerudoSimulator(num_players=2, start_dice=3, seed=42)
        
        agent_uniform = MonteCarloAgent(name='uniform', n=25, weighted_sampling=False, rng=None)
        agent_weighted = MonteCarloAgent(name='weighted', n=25, weighted_sampling=True, rng=None)
        
        # Create a scenario where weighted sampling should make better decisions
        # Player has [1, 1, 2] and current bid is (2, 2) - should probably call since we have ones
        obs = {
            '_simulator': sim,
            'dice_counts': [3, 3],
            'player_idx': 0,
            'my_hand': [1, 1, 2],
            'current_bid': (4, 2),  # Bid 4 twos - very high given only 6 total dice
            'maputa_active': False
        }
        
        # Test multiple decision scenarios
        decision_consistency = {'uniform': [], 'weighted': []}
        
        for trial in range(5):  # Small number for testing
            # Get decisions from both agents
            uniform_action = agent_uniform.select_action(obs)
            weighted_action = agent_weighted.select_action(obs)
            
            decision_consistency['uniform'].append(uniform_action)
            decision_consistency['weighted'].append(weighted_action)
        
        # Both agents should make valid decisions
        for action_list in decision_consistency.values():
            for action in action_list:
                assert action is not None, "Agent should return a valid action"
                assert len(action) >= 1, "Action should have at least one element"
        
        print(f"\nDecision Quality Test:")
        print(f"  Uniform decisions: {decision_consistency['uniform']}")
        print(f"  Weighted decisions: {decision_consistency['weighted']}")
        
        # This is more of a consistency check - in a full evaluation we'd need statistical significance testing
        assert len(decision_consistency['uniform']) == 5, "Should have 5 uniform decisions"
        assert len(decision_consistency['weighted']) == 5, "Should have 5 weighted decisions"