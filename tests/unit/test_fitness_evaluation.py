"""
Unit tests for fitness evaluation using tournament system integration.

Tests fitness function implementation that evaluates parameter configurations
by running tournaments against baseline opponents following TDD principles.
"""

import pytest
import random
from unittest.mock import Mock, patch, MagicMock
from eval.hyperopt_framework import ParameterSpace
from eval.fitness_evaluation import (
    FitnessEvaluator, evaluate_candidate_fitness, create_opponent_mix,
    calculate_win_rate, calculate_variance_penalty, calculate_robustness_bonus
)
from agents.optimizable_mc_agent import OptimizableMCAgent
from agents.baseline_agent import BaselineAgent
from agents.random_agent import RandomAgent


class TestFitnessEvaluator:
    """Test suite for fitness evaluation system."""

    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
                "early_stop_margin": {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15},
                "trust_learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1},
            }
        }
        return ParameterSpace(config)

    @pytest.fixture
    def fitness_evaluator(self, parameter_space):
        """Create fitness evaluator instance."""
        return FitnessEvaluator(
            parameter_space=parameter_space,
            opponent_config={
                'baseline_count': 2,
                'random_count': 1,
                'champion_count': 1
            },
            evaluation_games=20
        )

    def test_fitness_evaluator_initialization(self, fitness_evaluator, parameter_space):
        """Test fitness evaluator initialization."""
        assert fitness_evaluator.parameter_space == parameter_space
        assert fitness_evaluator.opponent_config['baseline_count'] == 2
        assert fitness_evaluator.opponent_config['random_count'] == 1
        assert fitness_evaluator.opponent_config['champion_count'] == 1
        assert fitness_evaluator.evaluation_games == 20

    def test_create_opponent_mix(self, fitness_evaluator):
        """Test opponent mix creation."""
        opponents = fitness_evaluator.create_opponent_mix()
        
        # Should create correct number of each opponent type
        baseline_count = sum(1 for agent in opponents if isinstance(agent, BaselineAgent))
        random_count = sum(1 for agent in opponents if isinstance(agent, RandomAgent))
        
        assert baseline_count == 2
        assert random_count == 1
        # Champion count might be 0 if no champions available yet
        assert len(opponents) >= 3

    def test_create_optimizable_agent_from_params(self, fitness_evaluator):
        """Test creation of optimizable agent from parameters."""
        params = {
            'n': 500,
            'chunk_size': 16,
            'early_stop_margin': 0.2,
            'trust_learning_rate': 0.15
        }
        
        agent = fitness_evaluator.create_optimizable_agent(params)
        
        assert isinstance(agent, OptimizableMCAgent)
        assert agent.N == 500
        assert agent.chunk_size == 16
        assert agent.early_stop_margin == 0.2
        assert agent.trust_learning_rate == 0.15

    @patch('eval.fitness_evaluation.play_tournament')
    def test_evaluate_single_configuration(self, mock_play_tournament, fitness_evaluator):
        """Test evaluation of a single parameter configuration."""
        # Mock tournament results
        mock_results = {
            0: 12,  # Candidate wins 12 games
            1: 3,   # Baseline 1 wins 3 games
            2: 3,   # Baseline 2 wins 3 games
            3: 2    # Random agent wins 2 games
        }
        mock_play_tournament.return_value = mock_results
        
        params = {
            'n': 400,
            'chunk_size': 8,
            'early_stop_margin': 0.15,
            'trust_learning_rate': 0.1
        }
        
        fitness_result = fitness_evaluator.evaluate_configuration(params)
        
        assert isinstance(fitness_result, dict)
        assert 'fitness' in fitness_result
        assert 'win_rate' in fitness_result
        assert 'variance' in fitness_result
        assert 'robustness' in fitness_result
        assert 'detailed_results' in fitness_result
        
        # Win rate should be 12/20 = 0.6
        expected_win_rate = 12 / 20
        assert abs(fitness_result['win_rate'] - expected_win_rate) < 0.01

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        results = {0: 15, 1: 3, 2: 2}  # Candidate index 0
        win_rate = calculate_win_rate(results, candidate_index=0)
        
        expected = 15 / (15 + 3 + 2)
        assert abs(win_rate - expected) < 0.001

    def test_calculate_variance_penalty(self):
        """Test variance penalty calculation."""
        # Low variance results (consistent performance)
        low_variance_results = {0: 18, 1: 1, 2: 1}
        low_penalty = calculate_variance_penalty(low_variance_results)
        
        # High variance results (inconsistent performance)
        high_variance_results = {0: 10, 1: 5, 2: 5}
        high_penalty = calculate_variance_penalty(high_variance_results)
        
        # Higher variance should result in higher penalty
        assert high_penalty > low_penalty
        assert low_penalty >= 0
        assert high_penalty >= 0

    def test_calculate_robustness_bonus(self):
        """Test robustness bonus calculation."""
        # Strong performance against all opponents
        strong_results = {0: 15, 1: 3, 2: 2}
        strong_bonus = calculate_robustness_bonus(strong_results)
        
        # Weak performance against all opponents
        weak_results = {0: 5, 1: 8, 2: 7}
        weak_bonus = calculate_robustness_bonus(weak_results)
        
        # Stronger performance should get higher bonus
        assert strong_bonus > weak_bonus

    @patch('eval.fitness_evaluation.play_tournament')
    def test_fitness_calculation_components(self, mock_play_tournament, fitness_evaluator):
        """Test that fitness calculation includes all components."""
        mock_results = {0: 10, 1: 5, 2: 3, 3: 2}
        mock_play_tournament.return_value = mock_results
        
        params = {'n': 400, 'chunk_size': 8, 'early_stop_margin': 0.15, 'trust_learning_rate': 0.1}
        fitness_result = fitness_evaluator.evaluate_configuration(params)
        
        win_rate = fitness_result['win_rate']
        variance_penalty = fitness_result['variance']
        robustness_bonus = fitness_result['robustness']
        fitness = fitness_result['fitness']
        
        # Fitness should be win_rate - variance_penalty + robustness_bonus
        expected_fitness = win_rate - variance_penalty + robustness_bonus
        assert abs(fitness - expected_fitness) < 0.001

    def test_batch_evaluation(self, fitness_evaluator):
        """Test evaluation of multiple configurations."""
        configurations = [
            {'n': 200, 'chunk_size': 4, 'early_stop_margin': 0.1, 'trust_learning_rate': 0.05},
            {'n': 400, 'chunk_size': 8, 'early_stop_margin': 0.15, 'trust_learning_rate': 0.1},
            {'n': 800, 'chunk_size': 16, 'early_stop_margin': 0.2, 'trust_learning_rate': 0.2},
        ]
        
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            # Mock different results for each configuration
            mock_tournament.side_effect = [
                {0: 8, 1: 6, 2: 4, 3: 2},   # Medium performance
                {0: 12, 1: 4, 2: 3, 3: 1},  # Good performance
                {0: 6, 1: 7, 2: 4, 3: 3},   # Poor performance
            ]
            
            results = fitness_evaluator.evaluate_batch(configurations)
            
            assert len(results) == 3
            assert all(isinstance(result, dict) for result in results)
            assert all('fitness' in result for result in results)
            
            # Second configuration should have highest fitness (best performance)
            assert results[1]['fitness'] > results[0]['fitness']
            assert results[1]['fitness'] > results[2]['fitness']

    def test_champion_integration(self, fitness_evaluator):
        """Test integration with champion agents from previous generations."""
        # Mock champion loading
        mock_champion = Mock(spec=OptimizableMCAgent)
        mock_champion.name = "champion_gen_5"
        
        with patch.object(fitness_evaluator, 'load_champion_agent') as mock_load:
            mock_load.return_value = mock_champion
            
            # Set champion count > 0
            fitness_evaluator.opponent_config['champion_count'] = 2
            
            opponents = fitness_evaluator.create_opponent_mix()
            
            # Should include champion agents
            champion_count = sum(1 for agent in opponents if agent == mock_champion)
            assert champion_count == 2

    def test_error_handling_tournament_failure(self, fitness_evaluator):
        """Test error handling when tournament fails."""
        params = {'n': 400, 'chunk_size': 8, 'early_stop_margin': 0.15, 'trust_learning_rate': 0.1}
        
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            mock_tournament.side_effect = Exception("Tournament simulation failed")
            
            # Should handle error gracefully and return default fitness
            fitness_result = fitness_evaluator.evaluate_configuration(params)
            
            assert fitness_result['fitness'] == 0.0  # Default poor fitness
            assert fitness_result['win_rate'] == 0.0
            assert 'error' in fitness_result

    def test_parameter_validation_before_evaluation(self, fitness_evaluator):
        """Test parameter validation before fitness evaluation."""
        # Invalid parameters (out of bounds)
        invalid_params = {
            'n': 5000,  # Too high
            'chunk_size': 50,  # Too high
            'early_stop_margin': -0.1,  # Negative
            'trust_learning_rate': 1.5  # Too high
        }
        
        with pytest.raises((ValueError, KeyError)):
            fitness_evaluator.evaluate_configuration(invalid_params)

    def test_fitness_normalization(self, fitness_evaluator):
        """Test that fitness values are properly normalized."""
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            # Perfect performance
            mock_tournament.return_value = {0: 20, 1: 0, 2: 0, 3: 0}
            
            params = {'n': 400, 'chunk_size': 8, 'early_stop_margin': 0.15, 'trust_learning_rate': 0.1}
            result = fitness_evaluator.evaluate_configuration(params)
            
            # Win rate should be 1.0
            assert abs(result['win_rate'] - 1.0) < 0.001
            
            # Fitness should be reasonable (between 0 and ~1.5 with bonuses)
            assert 0.0 <= result['fitness'] <= 2.0


class TestFitnessEvaluationFunctions:
    """Test standalone fitness evaluation functions."""

    def test_evaluate_candidate_fitness_integration(self):
        """Test the main evaluate_candidate_fitness function."""
        params_dict = {
            'n': 400,
            'chunk_size': 8,
            'early_stop_margin': 0.15,
            'trust_learning_rate': 0.1
        }
        
        evaluation_config = {
            'num_games': 20,
            'opponent_mix': {
                'baseline_count': 2,
                'random_count': 1
            }
        }
        
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            mock_tournament.return_value = {0: 12, 1: 4, 2: 4}
            
            result = evaluate_candidate_fitness(params_dict, evaluation_config)
            
            assert isinstance(result, dict)
            assert 'fitness' in result
            assert result['win_rate'] == 12 / 20

    def test_create_opponent_mix_function(self):
        """Test standalone opponent mix creation function."""
        opponent_config = {
            'baseline_count': 3,
            'random_count': 2,
            'champion_count': 0
        }
        
        opponents = create_opponent_mix(opponent_config)
        
        assert len(opponents) == 5
        baseline_count = sum(1 for agent in opponents if isinstance(agent, BaselineAgent))
        random_count = sum(1 for agent in opponents if isinstance(agent, RandomAgent))
        
        assert baseline_count == 3
        assert random_count == 2