"""
Tests for Robust Fitness Function - TDD Cycle 3.3: Robust Fitness Function

This module tests the sophisticated fitness function with variance penalties
and robustness metrics for the evolutionary optimization framework.

The robust fitness function balances win rate, variance penalty, and robustness
bonus to discover parameter configurations that perform consistently well
across diverse opponent mixes and game scenarios.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from eval.fitness_evaluation import RobustFitnessEvaluator


class TestRobustFitnessFunction:
    """Test suite for robust fitness function implementation."""
    
    @pytest.fixture
    def fitness_evaluator(self):
        """Create RobustFitnessEvaluator instance for testing."""
        return RobustFitnessEvaluator()
    
    @pytest.fixture
    def mock_tournament_results(self):
        """Create mock tournament results for testing."""
        # Results representing games won by each player across multiple tournaments
        return {
            'candidate': [15, 12, 18, 14, 16],  # Candidate agent results across 5 tournaments
            'baseline_1': [10, 13, 8, 12, 9],   # Baseline agent 1 results  
            'baseline_2': [8, 12, 7, 11, 10],   # Baseline agent 2 results
            'random': [5, 8, 4, 6, 7],          # Random agent results
            'previous_champion': [12, 5, 13, 7, 8]  # Previous generation champion
        }
    
    def test_robust_fitness_evaluator_initialization(self, fitness_evaluator):
        """Test RobustFitnessEvaluator initializes with correct weights."""
        assert hasattr(fitness_evaluator, 'win_rate_weight')
        assert hasattr(fitness_evaluator, 'variance_penalty_weight')  
        assert hasattr(fitness_evaluator, 'robustness_bonus_weight')
        
        # Default weights should sum to reasonable total
        total_weight = (fitness_evaluator.win_rate_weight + 
                       fitness_evaluator.variance_penalty_weight + 
                       fitness_evaluator.robustness_bonus_weight)
        assert 1.0 <= total_weight <= 2.0  # Allow for penalty weights
    
    def test_robust_fitness_evaluator_custom_weights(self):
        """Test RobustFitnessEvaluator with custom weight configuration."""
        custom_weights = {
            'win_rate': 1.0,
            'variance_penalty': 0.4,
            'robustness_bonus': 0.2
        }
        
        evaluator = RobustFitnessEvaluator(**custom_weights)
        
        assert evaluator.win_rate_weight == 1.0
        assert evaluator.variance_penalty_weight == 0.4
        assert evaluator.robustness_bonus_weight == 0.2
    
    def test_calculate_win_rate_simple(self, fitness_evaluator, mock_tournament_results):
        """Test basic win rate calculation."""
        candidate_name = 'candidate'
        
        win_rate = fitness_evaluator.calculate_win_rate(mock_tournament_results, candidate_name)
        
        # Total games = sum of all results across all players and tournaments
        candidate_wins = sum(mock_tournament_results[candidate_name])  # 75 wins
        total_games = sum(sum(results) for results in mock_tournament_results.values())  # 250 total
        expected_win_rate = candidate_wins / total_games  # 0.3
        
        assert abs(win_rate - expected_win_rate) < 0.001
        assert 0.0 <= win_rate <= 1.0
    
    def test_calculate_variance_penalty_low_variance(self, fitness_evaluator):
        """Test variance penalty calculation for consistent performance."""
        # Consistent results (low variance)
        consistent_results = {'candidate': [10, 11, 10, 9, 10]}  # Low variance
        
        penalty = fitness_evaluator.calculate_variance_penalty(consistent_results, 'candidate')
        
        # Low variance should result in low penalty
        assert penalty >= 0  # Penalty should be non-negative
        assert penalty < 0.1  # Should be relatively low for consistent performance
    
    def test_calculate_variance_penalty_high_variance(self, fitness_evaluator):
        """Test variance penalty calculation for inconsistent performance."""
        # Inconsistent results (high variance)
        inconsistent_results = {'candidate': [5, 20, 2, 18, 5]}  # High variance
        
        penalty = fitness_evaluator.calculate_variance_penalty(inconsistent_results, 'candidate')
        
        # High variance should result in higher penalty
        assert penalty >= 0  # Penalty should be non-negative
        assert penalty > 0.05  # Should be higher for inconsistent performance
    
    def test_variance_penalty_comparison(self, fitness_evaluator):
        """Test that high variance results in higher penalty than low variance."""
        low_variance_results = {'candidate': [10, 10, 10, 10, 10]}  # Perfect consistency
        high_variance_results = {'candidate': [0, 20, 0, 20, 0]}   # Maximum inconsistency
        
        low_penalty = fitness_evaluator.calculate_variance_penalty(low_variance_results, 'candidate')
        high_penalty = fitness_evaluator.calculate_variance_penalty(high_variance_results, 'candidate')
        
        assert high_penalty > low_penalty
        assert low_penalty >= 0
        assert high_penalty >= 0
    
    def test_calculate_robustness_bonus_diverse_performance(self, fitness_evaluator, mock_tournament_results):
        """Test robustness bonus for performance across diverse opponents."""
        candidate_name = 'candidate'
        
        bonus = fitness_evaluator.calculate_robustness_bonus(mock_tournament_results, candidate_name)
        
        # Robustness bonus should be non-negative
        assert bonus >= 0
        
        # Should reward agents that perform well against diverse opponent types
        assert isinstance(bonus, (int, float))
    
    def test_calculate_robustness_bonus_comparison(self, fitness_evaluator):
        """Test robustness bonus comparison between different performance patterns."""
        # Agent that performs well against all opponent types
        robust_results = {
            'candidate': [15, 15, 15, 15, 15],      # Consistent across all
            'baseline_1': [10, 10, 10, 10, 10],
            'baseline_2': [8, 8, 8, 8, 8],
            'random': [5, 5, 5, 5, 5],
            'champion': [12, 12, 12, 12, 12]
        }
        
        # Agent that performs well against only some opponents  
        specialized_results = {
            'candidate': [25, 25, 5, 5, 5],         # Good vs baselines, poor vs others
            'baseline_1': [0, 0, 20, 20, 20],
            'baseline_2': [0, 0, 20, 20, 20], 
            'random': [20, 20, 0, 0, 0],
            'champion': [20, 20, 0, 0, 0]
        }
        
        robust_bonus = fitness_evaluator.calculate_robustness_bonus(robust_results, 'candidate')
        specialized_bonus = fitness_evaluator.calculate_robustness_bonus(specialized_results, 'candidate')
        
        # Robust performance should get higher bonus
        assert robust_bonus >= specialized_bonus
    
    def test_calculate_combined_fitness_balanced(self, fitness_evaluator, mock_tournament_results):
        """Test combined fitness calculation with balanced performance."""
        candidate_name = 'candidate'
        
        fitness = fitness_evaluator.calculate_fitness(mock_tournament_results, candidate_name)
        
        # Combined fitness should be a reasonable value
        assert isinstance(fitness, (int, float))
        assert fitness > 0  # Should be positive for reasonable performance
        
        # Should be composed of win rate minus penalty plus bonus
        win_rate = fitness_evaluator.calculate_win_rate(mock_tournament_results, candidate_name)
        variance_penalty = fitness_evaluator.calculate_variance_penalty(mock_tournament_results, candidate_name)
        robustness_bonus = fitness_evaluator.calculate_robustness_bonus(mock_tournament_results, candidate_name)
        
        expected_fitness = (fitness_evaluator.win_rate_weight * win_rate - 
                          fitness_evaluator.variance_penalty_weight * variance_penalty +
                          fitness_evaluator.robustness_bonus_weight * robustness_bonus)
        
        assert abs(fitness - expected_fitness) < 0.001
    
    def test_fitness_comparison_high_vs_low_performance(self, fitness_evaluator):
        """Test that high performance gets higher fitness than low performance."""
        # High performance results
        high_perf_results = {
            'candidate': [18, 17, 19, 18, 18],  # High, consistent wins
            'baseline_1': [5, 6, 4, 5, 5],
            'baseline_2': [4, 5, 3, 4, 4], 
            'random': [2, 1, 3, 2, 2],
            'champion': [6, 6, 6, 6, 6]
        }
        
        # Low performance results
        low_perf_results = {
            'candidate': [5, 4, 6, 5, 5],      # Low wins
            'baseline_1': [12, 13, 11, 12, 12],
            'baseline_2': [11, 12, 10, 11, 11],
            'random': [8, 8, 8, 8, 8],
            'champion': [14, 13, 15, 14, 14]
        }
        
        high_fitness = fitness_evaluator.calculate_fitness(high_perf_results, 'candidate')
        low_fitness = fitness_evaluator.calculate_fitness(low_perf_results, 'candidate')
        
        assert high_fitness > low_fitness
    
    def test_fitness_variance_penalty_effect(self, fitness_evaluator):
        """Test that variance penalty reduces fitness for inconsistent agents."""
        # Same win rate, but different variance
        consistent_results = {
            'candidate': [10, 10, 10, 10, 10],  # Same total, low variance
            'opponent': [10, 10, 10, 10, 10]
        }
        
        inconsistent_results = {
            'candidate': [0, 20, 0, 20, 10],    # Same total, high variance  
            'opponent': [10, 0, 20, 0, 10]
        }
        
        consistent_fitness = fitness_evaluator.calculate_fitness(consistent_results, 'candidate')
        inconsistent_fitness = fitness_evaluator.calculate_fitness(inconsistent_results, 'candidate')
        
        # Consistent performance should have higher fitness due to lower variance penalty
        assert consistent_fitness > inconsistent_fitness
    
    def test_handle_empty_results(self, fitness_evaluator):
        """Test handling of empty or invalid tournament results."""
        empty_results = {}
        
        # Should handle gracefully without crashing
        with pytest.raises((ValueError, KeyError)):
            fitness_evaluator.calculate_fitness(empty_results, 'candidate')
    
    def test_handle_single_tournament_results(self, fitness_evaluator):
        """Test handling results from single tournament (no variance calculation)."""
        single_tournament = {
            'candidate': [15],  # Only one tournament
            'baseline': [10],
            'random': [5]
        }
        
        fitness = fitness_evaluator.calculate_fitness(single_tournament, 'candidate')
        
        # Should still calculate fitness, with minimal/zero variance penalty
        assert isinstance(fitness, (int, float))
        assert fitness > 0
    
    def test_fitness_weight_configuration_effect(self):
        """Test that different weight configurations affect fitness calculation."""
        results = {
            'candidate': [10, 15, 5, 20, 0],    # High variance but decent avg
            'opponent': [10, 10, 10, 10, 10]
        }
        
        # High variance penalty weight
        high_penalty_evaluator = RobustFitnessEvaluator(
            win_rate=1.0, variance_penalty=1.0, robustness_bonus=0.1
        )
        
        # Low variance penalty weight  
        low_penalty_evaluator = RobustFitnessEvaluator(
            win_rate=1.0, variance_penalty=0.1, robustness_bonus=0.1
        )
        
        high_penalty_fitness = high_penalty_evaluator.calculate_fitness(results, 'candidate')
        low_penalty_fitness = low_penalty_evaluator.calculate_fitness(results, 'candidate')
        
        # High variance penalty should result in lower fitness for inconsistent agent
        assert low_penalty_fitness > high_penalty_fitness
    
    def test_robustness_metrics_calculation(self, fitness_evaluator):
        """Test calculation of detailed robustness metrics."""
        results = {
            'candidate': [12, 14, 13, 15, 11],
            'baseline_1': [10, 9, 11, 8, 12], 
            'baseline_2': [8, 11, 9, 10, 7],
            'random': [5, 6, 4, 7, 5],
            'champion': [15, 10, 13, 10, 15]
        }
        
        metrics = fitness_evaluator.calculate_robustness_metrics(results, 'candidate')
        
        # Should return dictionary with detailed metrics
        assert isinstance(metrics, dict)
        assert 'win_rate' in metrics
        assert 'variance_penalty' in metrics
        assert 'robustness_bonus' in metrics
        assert 'consistency_score' in metrics
        assert 'diversity_score' in metrics
        
        # All metrics should be reasonable values
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value), f"Metric {metric_name} is NaN"
    
    def test_fitness_evaluation_integration_with_multifidelity(self, fitness_evaluator):
        """Test integration with multi-fidelity evaluation system."""
        # Mock function that returns fitness with number of games parameter
        def mock_evaluate_with_games(params_dict, num_games):
            base_fitness = fitness_evaluator.calculate_fitness(
                {'candidate': [10] * num_games, 'opponent': [5] * num_games}, 
                'candidate'
            )
            # Scale fitness based on number of games (more games = more reliable)
            reliability_bonus = min(0.1, num_games * 0.001)
            return base_fitness + reliability_bonus
        
        # Test with different game counts
        low_fidelity = mock_evaluate_with_games({'n': 100}, 10)
        high_fidelity = mock_evaluate_with_games({'n': 100}, 200)
        
        # Higher fidelity should give more reliable (slightly higher) fitness
        assert high_fidelity >= low_fidelity
        assert isinstance(low_fidelity, (int, float))
        assert isinstance(high_fidelity, (int, float))