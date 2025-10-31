"""
Integration tests for complete evolutionary optimization pipeline.

Tests the end-to-end optimization cycle combining genetic algorithms,
fitness evaluation, and tournament integration following TDD principles.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from eval.hyperopt_framework import ParameterSpace, GeneticAlgorithm
from eval.fitness_evaluation import FitnessEvaluator
from agents.optimizable_mc_agent import OptimizableMCAgent


class TestOptimizationPipeline:
    """Test suite for end-to-end optimization pipeline."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary configuration file for testing."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 500, "default": 200},
                "chunk_size": {"type": "int", "min": 4, "max": 16, "default": 8},
                "early_stop_margin": {"type": "float", "min": 0.1, "max": 0.3, "default": 0.15},
                "trust_learning_rate": {"type": "float", "min": 0.05, "max": 0.2, "default": 0.1},
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            return f.name

    @pytest.fixture
    def parameter_space(self, temp_config_file):
        """Create parameter space for testing."""
        return ParameterSpace(temp_config_file)

    @pytest.fixture
    def genetic_algorithm(self, parameter_space):
        """Create genetic algorithm instance."""
        return GeneticAlgorithm(
            parameter_space=parameter_space,
            population_size=6,  # Small for testing
            mutation_rate=0.2,
            crossover_rate=0.8
        )

    @pytest.fixture  
    def fitness_evaluator(self, parameter_space):
        """Create fitness evaluator instance."""
        return FitnessEvaluator(
            parameter_space=parameter_space,
            opponent_config={'baseline_count': 1, 'random_count': 1, 'champion_count': 0},
            evaluation_games=5  # Small for testing speed
        )

    def test_parameter_space_genetic_algorithm_integration(self, parameter_space, genetic_algorithm):
        """Test integration between parameter space and genetic algorithm."""
        # Generate random population
        population = genetic_algorithm.initialize_population()
        
        assert len(population) == 6
        
        # Test that all individuals can be decoded to valid parameters
        for individual in population:
            params = parameter_space.decode_genotype(individual)
            
            # Validate parameters
            assert parameter_space.validate_parameters(params)
            
            # Check parameter bounds
            assert 100 <= params['n'] <= 500
            assert 4 <= params['chunk_size'] <= 16
            assert 0.1 <= params['early_stop_margin'] <= 0.3
            assert 0.05 <= params['trust_learning_rate'] <= 0.2

    def test_genetic_algorithm_fitness_evaluator_integration(self, genetic_algorithm, fitness_evaluator):
        """Test integration between genetic algorithm and fitness evaluator."""
        population = genetic_algorithm.initialize_population()
        
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            # Mock consistent results for testing
            mock_tournament.return_value = {0: 3, 1: 1, 2: 1}  # Candidate wins most
            
            # Evaluate population fitness
            fitness_scores = []
            for individual in population:
                params = genetic_algorithm.parameter_space.decode_genotype(individual)
                result = fitness_evaluator.evaluate_configuration(params)
                fitness_scores.append(result['fitness'])
            
            # All fitness scores should be valid
            assert len(fitness_scores) == len(population)
            assert all(isinstance(score, (int, float)) for score in fitness_scores)
            assert all(0.0 <= score <= 2.0 for score in fitness_scores)  # Reasonable range

    def test_complete_optimization_cycle(self, genetic_algorithm, fitness_evaluator):
        """Test complete optimization cycle for multiple generations."""
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            # Mock varying tournament results
            tournament_results = [
                {0: 4, 1: 1, 2: 0},  # Good performance
                {0: 3, 1: 1, 2: 1},  # Medium performance
                {0: 2, 1: 2, 2: 1},  # Poor performance
                {0: 5, 1: 0, 2: 0},  # Excellent performance
                {0: 1, 1: 2, 2: 2},  # Very poor performance
                {0: 3, 1: 1, 2: 1},  # Medium performance (repeated for consistency)
            ]
            mock_tournament.side_effect = tournament_results * 10  # Repeat for multiple calls
            
            # Initialize population
            population = genetic_algorithm.initialize_population()
            best_fitness_history = []
            
            # Run optimization for few generations
            for generation in range(3):
                # Evaluate fitness for current population
                fitness_scores = []
                for individual in population:
                    params = genetic_algorithm.parameter_space.decode_genotype(individual)
                    result = fitness_evaluator.evaluate_configuration(params)
                    fitness_scores.append(result['fitness'])
                
                # Track best fitness
                best_fitness = max(fitness_scores)
                best_fitness_history.append(best_fitness)
                
                # Evolve population
                population = genetic_algorithm.evolve_population(population, fitness_scores)
                
                # Verify population maintained
                assert len(population) == genetic_algorithm.population_size
            
            # Fitness should remain reasonable throughout optimization
            assert len(best_fitness_history) == 3
            assert all(fitness >= 0.0 for fitness in best_fitness_history)

    def test_parameter_optimization_convergence(self, genetic_algorithm, fitness_evaluator):
        """Test that parameter optimization shows improvement tendency."""
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            # Create a fitness landscape where n=400 is optimal
            def mock_tournament_func(*args, **kwargs):
                # Access the candidate agent from the call
                agents = args[1] if len(args) > 1 else []
                if agents and hasattr(agents[0], 'N'):
                    n_value = agents[0].N
                    # Reward n values closer to 400
                    distance_from_optimal = abs(n_value - 400)
                    if distance_from_optimal < 50:
                        return {0: 4, 1: 1, 2: 0}  # Good performance
                    elif distance_from_optimal < 150:
                        return {0: 3, 1: 1, 2: 1}  # Medium performance
                    else:
                        return {0: 1, 1: 2, 2: 2}  # Poor performance
                return {0: 2, 1: 2, 2: 1}  # Default medium performance
            
            mock_tournament.side_effect = mock_tournament_func
            
            # Initialize population and track n parameter distribution
            population = genetic_algorithm.initialize_population()
            initial_n_values = []
            final_n_values = []
            
            # Get initial n distribution
            for individual in population:
                params = genetic_algorithm.parameter_space.decode_genotype(individual)
                initial_n_values.append(params['n'])
            
            # Run optimization
            for generation in range(5):
                fitness_scores = []
                for individual in population:
                    params = genetic_algorithm.parameter_space.decode_genotype(individual)
                    result = fitness_evaluator.evaluate_configuration(params)
                    fitness_scores.append(result['fitness'])
                
                population = genetic_algorithm.evolve_population(population, fitness_scores)
            
            # Get final n distribution
            for individual in population:
                params = genetic_algorithm.parameter_space.decode_genotype(individual)
                final_n_values.append(params['n'])
            
            # Evolution should show some directional change
            initial_mean_n = sum(initial_n_values) / len(initial_n_values)
            final_mean_n = sum(final_n_values) / len(final_n_values)
            
            # Values should be different (evolution occurred)
            assert abs(final_mean_n - initial_mean_n) > 10  # Some change occurred

    def test_optimization_with_champion_preservation(self, genetic_algorithm, fitness_evaluator):
        """Test optimization with champion preservation and logging."""
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            mock_tournament.return_value = {0: 3, 1: 1, 2: 1}
            
            population = genetic_algorithm.initialize_population()
            champions = []
            
            for generation in range(3):
                # Evaluate population
                fitness_scores = []
                evaluated_configs = []
                
                for individual in population:
                    params = genetic_algorithm.parameter_space.decode_genotype(individual)
                    result = fitness_evaluator.evaluate_configuration(params)
                    fitness_scores.append(result['fitness'])
                    evaluated_configs.append((individual, params, result))
                
                # Find champion of this generation
                best_idx = fitness_scores.index(max(fitness_scores))
                champion_individual = population[best_idx]
                champion_params = genetic_algorithm.parameter_space.decode_genotype(champion_individual)
                champion_fitness = fitness_scores[best_idx]
                
                champions.append({
                    'generation': generation,
                    'individual': champion_individual,
                    'parameters': champion_params,
                    'fitness': champion_fitness
                })
                
                # Evolve population
                population = genetic_algorithm.evolve_population(population, fitness_scores)
            
            # Verify champions were tracked
            assert len(champions) == 3
            assert all('generation' in champ for champ in champions)
            assert all('fitness' in champ for champ in champions)
            assert all(isinstance(champ['parameters'], dict) for champ in champions)

    def test_optimization_error_handling(self, genetic_algorithm, fitness_evaluator):
        """Test optimization pipeline error handling."""
        population = genetic_algorithm.initialize_population()
        
        # Test with tournament failure
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            mock_tournament.side_effect = Exception("Tournament failed")
            
            # Should handle tournament failures gracefully
            fitness_scores = []
            for individual in population:
                params = genetic_algorithm.parameter_space.decode_genotype(individual)
                result = fitness_evaluator.evaluate_configuration(params)
                # Should return error result, not crash
                assert 'error' in result
                assert result['fitness'] == 0.0
                fitness_scores.append(result['fitness'])
            
            # Evolution should still work with default fitness scores
            new_population = genetic_algorithm.evolve_population(population, fitness_scores)
            assert len(new_population) == len(population)

    def test_optimization_result_logging(self, genetic_algorithm, fitness_evaluator):
        """Test optimization result logging and persistence."""
        with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
            mock_tournament.return_value = {0: 3, 1: 1, 2: 1}
            
            optimization_log = []
            population = genetic_algorithm.initialize_population()
            
            for generation in range(2):
                generation_results = {
                    'generation': generation,
                    'population_size': len(population),
                    'evaluations': []
                }
                
                fitness_scores = []
                for i, individual in enumerate(population):
                    params = genetic_algorithm.parameter_space.decode_genotype(individual)
                    result = fitness_evaluator.evaluate_configuration(params)
                    fitness_scores.append(result['fitness'])
                    
                    generation_results['evaluations'].append({
                        'individual_id': i,
                        'parameters': params,
                        'fitness': result['fitness'],
                        'win_rate': result['win_rate'],
                        'variance_penalty': result['variance'],
                        'robustness_bonus': result['robustness']
                    })
                
                generation_results['best_fitness'] = max(fitness_scores)
                generation_results['avg_fitness'] = sum(fitness_scores) / len(fitness_scores)
                optimization_log.append(generation_results)
                
                population = genetic_algorithm.evolve_population(population, fitness_scores)
            
            # Verify logging structure
            assert len(optimization_log) == 2
            for gen_log in optimization_log:
                assert 'generation' in gen_log
                assert 'population_size' in gen_log
                assert 'evaluations' in gen_log
                assert 'best_fitness' in gen_log
                assert 'avg_fitness' in gen_log
                assert len(gen_log['evaluations']) == genetic_algorithm.population_size

    def test_optimization_with_different_configurations(self, parameter_space):
        """Test optimization with different genetic algorithm configurations."""
        configurations = [
            {'population_size': 4, 'mutation_rate': 0.1, 'crossover_rate': 0.7},
            {'population_size': 8, 'mutation_rate': 0.3, 'crossover_rate': 0.9},
            {'population_size': 6, 'mutation_rate': 0.2, 'crossover_rate': 0.8},
        ]
        
        for config in configurations:
            ga = GeneticAlgorithm(parameter_space=parameter_space, **config)
            fitness_eval = FitnessEvaluator(
                parameter_space=parameter_space,
                opponent_config={'baseline_count': 1, 'random_count': 1, 'champion_count': 0},
                evaluation_games=3
            )
            
            with patch('eval.fitness_evaluation.play_tournament') as mock_tournament:
                mock_tournament.return_value = {0: 2, 1: 1, 2: 0}
                
                # Test one generation of optimization
                population = ga.initialize_population()
                assert len(population) == config['population_size']
                
                fitness_scores = []
                for individual in population:
                    params = parameter_space.decode_genotype(individual)
                    result = fitness_eval.evaluate_configuration(params)
                    fitness_scores.append(result['fitness'])
                
                new_population = ga.evolve_population(population, fitness_scores)
                assert len(new_population) == config['population_size']

    def tearDown(self):
        """Clean up temporary files."""
        # Note: temp_config_file fixture should be cleaned up automatically
        # But we can add explicit cleanup if needed
        pass