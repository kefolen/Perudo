"""
Tests for MultiFidelityEvaluator - TDD Cycle 3.2: Multi-Fidelity Evaluation

This module tests the successive halving evaluation system that provides
computational efficiency by evaluating candidates at multiple fidelity levels,
promoting only the best performers to higher (more expensive) evaluations.

The multi-fidelity approach can reduce computational cost by 60-80% compared
to full evaluation of all candidates.
"""
import pytest
from unittest.mock import Mock, MagicMock
from eval.hyperopt_framework import MultiFidelityEvaluator


class TestMultiFidelityEvaluator:
    """Test suite for multi-fidelity evaluation system."""
    
    @pytest.fixture
    def evaluator(self):
        """Create MultiFidelityEvaluator instance for testing."""
        return MultiFidelityEvaluator()
    
    @pytest.fixture
    def mock_fitness_function(self):
        """Create mock fitness evaluation function."""
        def mock_fitness(params_dict, num_games):
            # Simple mock: fitness based on parameter sum + small game bonus
            base_fitness = sum(params_dict.values()) / len(params_dict)
            game_bonus = num_games * 0.001  # Small bonus for more games
            return base_fitness + game_bonus
        return mock_fitness
    
    def test_multifidelity_evaluator_initialization(self, evaluator):
        """Test MultiFidelityEvaluator initializes with correct fidelity levels."""
        assert hasattr(evaluator, 'fidelity_levels')
        assert len(evaluator.fidelity_levels) == 3  # Initial, intermediate, final
        
        # Check fidelity level structure
        for level in evaluator.fidelity_levels:
            assert 'games' in level
            assert 'promotion_rate' in level
            assert 0 < level['promotion_rate'] <= 1.0
            assert level['games'] > 0
        
        # Check levels are increasing in games and decreasing in promotion rate
        games = [level['games'] for level in evaluator.fidelity_levels]
        assert games == sorted(games)  # Should be increasing
        
        promotion_rates = [level['promotion_rate'] for level in evaluator.fidelity_levels[:-1]]
        assert all(rate < 1.0 for rate in promotion_rates)  # Should be less than 1 except last
    
    def test_multifidelity_evaluator_custom_levels(self):
        """Test MultiFidelityEvaluator with custom fidelity levels."""
        custom_levels = [
            {'games': 5, 'promotion_rate': 0.6},
            {'games': 25, 'promotion_rate': 0.4},
            {'games': 100, 'promotion_rate': 1.0}
        ]
        
        evaluator = MultiFidelityEvaluator(fidelity_levels=custom_levels)
        
        assert evaluator.fidelity_levels == custom_levels
        assert len(evaluator.fidelity_levels) == 3
    
    def test_evaluate_candidates_single_level(self, evaluator, mock_fitness_function):
        """Test evaluating candidates at a single fidelity level."""
        # Create test candidates (parameter configurations)
        candidates = [
            {'n': 100, 'chunk_size': 8, 'early_stop_margin': 0.1},
            {'n': 200, 'chunk_size': 16, 'early_stop_margin': 0.2},
            {'n': 300, 'chunk_size': 12, 'early_stop_margin': 0.15},
        ]
        
        # Evaluate at single level
        results = evaluator.evaluate_candidates_at_level(
            candidates, fidelity_level={'games': 10, 'promotion_rate': 0.5}, 
            fitness_function=mock_fitness_function
        )
        
        # Should return fitness results for all candidates
        assert len(results) == 3
        for candidate, fitness in results:
            assert candidate in candidates
            assert isinstance(fitness, (int, float))
            assert fitness > 0  # Mock function returns positive values
    
    def test_promote_candidates_by_fitness(self, evaluator):
        """Test promoting best candidates based on fitness."""
        # Mock results with different fitness values
        candidate_results = [
            ({'n': 100}, 0.3),  # Lower fitness
            ({'n': 200}, 0.7),  # Higher fitness
            ({'n': 150}, 0.5),  # Medium fitness
            ({'n': 250}, 0.8),  # Highest fitness
        ]
        
        # Promote top 50% (2 out of 4)
        promoted = evaluator.promote_candidates(candidate_results, promotion_rate=0.5)
        
        assert len(promoted) == 2
        # Should be the two highest fitness candidates
        promoted_params = [candidate for candidate, fitness in promoted]
        assert {'n': 250} in promoted_params  # Highest
        assert {'n': 200} in promoted_params  # Second highest
    
    def test_promote_candidates_edge_cases(self, evaluator):
        """Test candidate promotion with edge cases."""
        candidate_results = [
            ({'n': 100}, 0.3),
            ({'n': 200}, 0.7),
            ({'n': 150}, 0.5),
        ]
        
        # Test promoting 100% (all candidates)
        promoted_all = evaluator.promote_candidates(candidate_results, promotion_rate=1.0)
        assert len(promoted_all) == 3
        
        # Test promoting with small rate that rounds to 1
        promoted_one = evaluator.promote_candidates(candidate_results, promotion_rate=0.2)
        assert len(promoted_one) == 1  # Should promote at least 1
        assert promoted_one[0][0] == {'n': 200}  # Best candidate
        
        # Test empty candidate list
        promoted_empty = evaluator.promote_candidates([], promotion_rate=0.5)
        assert len(promoted_empty) == 0
    
    def test_successive_halving_full_pipeline(self, evaluator, mock_fitness_function):
        """Test complete successive halving evaluation pipeline."""
        # Create population of candidates
        population = [
            {'n': 100 + 50*i, 'chunk_size': 8, 'early_stop_margin': 0.1 + 0.02*i}
            for i in range(10)  # 10 candidates
        ]
        
        generation = 1
        
        # Run successive halving
        final_survivors = evaluator.evaluate_population(
            population, generation, fitness_function=mock_fitness_function
        )
        
        # Should return only the final survivors
        expected_final_count = max(1, int(len(population) * 0.5 * 0.3))  # Apply both promotion rates
        assert len(final_survivors) <= expected_final_count
        assert len(final_survivors) >= 1  # Should have at least one survivor
        
        # All survivors should be from original population
        for survivor in final_survivors:
            assert survivor in population
    
    def test_computational_efficiency_calculation(self, evaluator):
        """Test that multi-fidelity reduces computational cost."""
        population_size = 20
        
        # Calculate baseline cost (all candidates at highest fidelity)
        baseline_cost = population_size * evaluator.fidelity_levels[-1]['games']
        
        # Calculate multi-fidelity cost
        multifidelity_cost = evaluator.calculate_evaluation_cost(population_size)
        
        # Multi-fidelity should be significantly cheaper
        efficiency_gain = (baseline_cost - multifidelity_cost) / baseline_cost
        assert efficiency_gain > 0.5  # Should save at least 50%
        assert efficiency_gain < 1.0  # Shouldn't save more than 100%
        
        # Check expected cost calculation matches spec (72.5% reduction)
        # baseline: 200 games × 20 candidates = 4000 games
        # multifidelity: (10×20) + (50×10) + (200×2) = 1100 games
        if population_size == 20:
            expected_multifidelity = (10 * 20) + (50 * int(20 * 0.5)) + (200 * int(20 * 0.5 * 0.3))
            assert abs(multifidelity_cost - expected_multifidelity) / expected_multifidelity < 0.1
    
    def test_fidelity_level_validation(self):
        """Test validation of fidelity level configuration."""
        # Valid configuration should work
        valid_levels = [
            {'games': 10, 'promotion_rate': 0.5},
            {'games': 50, 'promotion_rate': 1.0}
        ]
        evaluator = MultiFidelityEvaluator(fidelity_levels=valid_levels)
        assert evaluator.fidelity_levels == valid_levels
        
        # Invalid configurations should raise errors
        with pytest.raises((ValueError, AssertionError)):
            # Negative games
            MultiFidelityEvaluator([{'games': -10, 'promotion_rate': 0.5}])
        
        with pytest.raises((ValueError, AssertionError)):
            # Promotion rate > 1.0
            MultiFidelityEvaluator([{'games': 10, 'promotion_rate': 1.5}])
        
        with pytest.raises((ValueError, AssertionError)):
            # Zero promotion rate
            MultiFidelityEvaluator([{'games': 10, 'promotion_rate': 0.0}])
    
    def test_evaluation_with_identical_fitness(self, evaluator):
        """Test evaluation when all candidates have identical fitness."""
        # Mock function that returns same fitness for all
        def uniform_fitness(params_dict, num_games):
            return 0.5  # Same fitness regardless of parameters
        
        candidates = [
            {'n': 100}, {'n': 200}, {'n': 300}, {'n': 400}
        ]
        
        # Should handle uniform fitness gracefully
        results = evaluator.evaluate_candidates_at_level(
            candidates, {'games': 10, 'promotion_rate': 0.5}, uniform_fitness
        )
        
        assert len(results) == 4
        for candidate, fitness in results:
            assert fitness == 0.5
        
        # Promotion should still work (may depend on ordering)
        promoted = evaluator.promote_candidates(results, promotion_rate=0.5)
        assert len(promoted) == 2  # Should still promote 50%
    
    def test_evaluation_logging_and_tracking(self, evaluator, mock_fitness_function):
        """Test that evaluation tracks and logs fidelity level results."""
        population = [{'n': 100 + 50*i} for i in range(6)]
        generation = 2
        
        # Mock logging functionality
        evaluator.evaluation_log = []
        
        def mock_log_results(level, results, gen):
            evaluator.evaluation_log.append({
                'level': level,
                'generation': gen,
                'num_results': len(results),
                'best_fitness': max(fitness for _, fitness in results)
            })
        
        evaluator.custom_log_function = mock_log_results
        
        # Run evaluation
        survivors = evaluator.evaluate_population(
            population, generation, fitness_function=mock_fitness_function
        )
        
        # Should have logged results for each fidelity level
        assert len(evaluator.evaluation_log) == len(evaluator.fidelity_levels)
        
        # Check log entries are reasonable
        for log_entry in evaluator.evaluation_log:
            assert log_entry['generation'] == generation
            assert log_entry['num_results'] > 0
            assert log_entry['best_fitness'] > 0
    
    def test_early_termination_on_single_survivor(self, evaluator, mock_fitness_function):
        """Test early termination when only one candidate remains."""
        # Start with very few candidates and aggressive promotion
        population = [{'n': 100}, {'n': 200}]
        
        # With aggressive rates, should terminate early
        survivors = evaluator.evaluate_population(
            population, generation=1, fitness_function=mock_fitness_function
        )
        
        # Should have at least one survivor
        assert len(survivors) >= 1
        assert len(survivors) <= len(population)
    
    def test_fitness_function_interface_compatibility(self, evaluator):
        """Test compatibility with different fitness function interfaces."""
        candidates = [{'n': 100}, {'n': 200}]
        
        # Test with function that takes additional parameters
        def extended_fitness(params_dict, num_games, **kwargs):
            return sum(params_dict.values()) / len(params_dict) + kwargs.get('bonus', 0)
        
        # Should work with additional parameters
        results = evaluator.evaluate_candidates_at_level(
            candidates, {'games': 10, 'promotion_rate': 1.0}, 
            extended_fitness, bonus=0.1
        )
        
        assert len(results) == 2
        for candidate, fitness in results:
            assert fitness > 0.1  # Should include bonus