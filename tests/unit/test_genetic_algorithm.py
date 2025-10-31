"""
Unit tests for genetic algorithm operations in evolutionary optimization.

Tests genetic operations including mutation, crossover, selection,
and population management following TDD principles.
"""

import pytest
import random
import numpy as np
from eval.hyperopt_framework import GeneticAlgorithm, ParameterSpace


class TestGeneticAlgorithm:
    """Test suite for basic genetic algorithm operations."""

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
    def genetic_algorithm(self, parameter_space):
        """Create genetic algorithm instance for testing."""
        return GeneticAlgorithm(
            parameter_space=parameter_space,
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

    def test_genetic_algorithm_initialization(self, genetic_algorithm, parameter_space):
        """Test genetic algorithm initialization."""
        assert genetic_algorithm.parameter_space == parameter_space
        assert genetic_algorithm.population_size == 10
        assert genetic_algorithm.mutation_rate == 0.1
        assert genetic_algorithm.crossover_rate == 0.8
        assert genetic_algorithm.generation == 0

    def test_create_random_individual(self, genetic_algorithm):
        """Test creation of random individual (genotype)."""
        individual = genetic_algorithm.create_random_individual()
        
        # Should be a list/array of parameter values
        assert isinstance(individual, (list, np.ndarray))
        assert len(individual) == 4  # We have 4 parameters
        
        # Values should be within bounds
        params = genetic_algorithm.parameter_space.decode_genotype(individual)
        assert 100 <= params['n'] <= 2000
        assert 4 <= params['chunk_size'] <= 32
        assert 0.05 <= params['early_stop_margin'] <= 0.3
        assert 0.01 <= params['trust_learning_rate'] <= 0.5

    def test_initialize_population(self, genetic_algorithm):
        """Test population initialization."""
        population = genetic_algorithm.initialize_population()
        
        assert len(population) == genetic_algorithm.population_size
        for individual in population:
            assert isinstance(individual, (list, np.ndarray))
            assert len(individual) == 4

    def test_mutation_operation(self, genetic_algorithm):
        """Test mutation operation on individual."""
        original = genetic_algorithm.create_random_individual()
        mutated = genetic_algorithm.mutate(original.copy())
        
        # Should be same type and length
        assert type(mutated) == type(original)
        assert len(mutated) == len(original)
        
        # With mutation rate 0.1, not all values should be identical
        # But we can't guarantee mutation occurred due to randomness
        params_original = genetic_algorithm.parameter_space.decode_genotype(original)
        params_mutated = genetic_algorithm.parameter_space.decode_genotype(mutated)
        
        # Mutated parameters should still be within bounds
        assert 100 <= params_mutated['n'] <= 2000
        assert 4 <= params_mutated['chunk_size'] <= 32
        assert 0.05 <= params_mutated['early_stop_margin'] <= 0.3
        assert 0.01 <= params_mutated['trust_learning_rate'] <= 0.5

    def test_crossover_operation(self, genetic_algorithm):
        """Test crossover operation between two parents."""
        parent1 = genetic_algorithm.create_random_individual()
        parent2 = genetic_algorithm.create_random_individual()
        
        child1, child2 = genetic_algorithm.crossover(parent1, parent2)
        
        # Children should be same type and length as parents
        assert type(child1) == type(parent1)
        assert type(child2) == type(parent1)
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent1)
        
        # Children should have valid parameter values
        for child in [child1, child2]:
            params = genetic_algorithm.parameter_space.decode_genotype(child)
            assert 100 <= params['n'] <= 2000
            assert 4 <= params['chunk_size'] <= 32
            assert 0.05 <= params['early_stop_margin'] <= 0.3
            assert 0.01 <= params['trust_learning_rate'] <= 0.5

    def test_tournament_selection(self, genetic_algorithm):
        """Test tournament selection mechanism."""
        population = genetic_algorithm.initialize_population()
        
        # Create mock fitness values (higher is better)
        fitness_values = [random.uniform(0.3, 0.8) for _ in population]
        
        # Perform tournament selection
        selected = genetic_algorithm.tournament_selection(population, fitness_values, tournament_size=3)
        
        # Should return valid individual from population
        assert selected in population
        
        # With sufficient trials, better individuals should be selected more often
        selections = []
        for _ in range(100):
            selected = genetic_algorithm.tournament_selection(population, fitness_values, tournament_size=3)
            selections.append(population.index(selected))
        
        # Best individual should be selected more frequently
        best_index = fitness_values.index(max(fitness_values))
        worst_index = fitness_values.index(min(fitness_values))
        
        best_selections = selections.count(best_index)
        worst_selections = selections.count(worst_index)
        
        # Best should be selected more often than worst (with high probability)
        assert best_selections >= worst_selections

    def test_evolve_population(self, genetic_algorithm):
        """Test population evolution for one generation."""
        population = genetic_algorithm.initialize_population()
        fitness_values = [random.uniform(0.3, 0.8) for _ in population]
        
        new_population = genetic_algorithm.evolve_population(population, fitness_values)
        
        # New population should have same size
        assert len(new_population) == len(population)
        
        # All individuals should be valid
        for individual in new_population:
            params = genetic_algorithm.parameter_space.decode_genotype(individual)
            assert 100 <= params['n'] <= 2000
            assert 4 <= params['chunk_size'] <= 32
            assert 0.05 <= params['early_stop_margin'] <= 0.3
            assert 0.01 <= params['trust_learning_rate'] <= 0.5

    def test_elitism_preservation(self, genetic_algorithm):
        """Test that best individuals are preserved across generations."""
        genetic_algorithm.elitism_count = 2  # Preserve top 2 individuals
        
        population = genetic_algorithm.initialize_population()
        fitness_values = [random.uniform(0.3, 0.8) for _ in population]
        
        # Find the two best individuals
        sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)
        best_two = [population[sorted_indices[0]], population[sorted_indices[1]]]
        
        new_population = genetic_algorithm.evolve_population(population, fitness_values)
        
        # Best individuals should be preserved in new population
        for best_individual in best_two:
            assert any(np.array_equal(best_individual, ind) for ind in new_population)

    def test_mutation_rate_effect(self):
        """Test effect of different mutation rates."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
            }
        }
        param_space = ParameterSpace(config)
        
        # Test with high mutation rate
        high_mutation_ga = GeneticAlgorithm(param_space, population_size=10, mutation_rate=0.9)
        individual = high_mutation_ga.create_random_individual()
        mutated = high_mutation_ga.mutate(individual.copy())
        
        # With high mutation rate, more genes should change (statistically)
        # This is probabilistic, so we just verify it doesn't crash
        assert len(mutated) == len(individual)
        
        # Test with zero mutation rate
        no_mutation_ga = GeneticAlgorithm(param_space, population_size=10, mutation_rate=0.0)
        individual = no_mutation_ga.create_random_individual()
        not_mutated = no_mutation_ga.mutate(individual.copy())
        
        # With zero mutation rate, individual should remain unchanged
        assert np.array_equal(individual, not_mutated)

    def test_genetic_diversity_maintenance(self, genetic_algorithm):
        """Test that genetic algorithm maintains diversity."""
        population = genetic_algorithm.initialize_population()
        
        # Calculate diversity metric (average pairwise distance)
        def calculate_diversity(pop):
            if len(pop) < 2:
                return 0.0
            
            total_distance = 0.0
            comparisons = 0
            
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    # Euclidean distance between individuals
                    distance = np.linalg.norm(np.array(pop[i]) - np.array(pop[j]))
                    total_distance += distance
                    comparisons += 1
            
            return total_distance / comparisons if comparisons > 0 else 0.0
        
        initial_diversity = calculate_diversity(population)
        
        # Run evolution for few generations
        current_population = population
        for _ in range(3):
            fitness_values = [random.uniform(0.3, 0.8) for _ in current_population]
            current_population = genetic_algorithm.evolve_population(current_population, fitness_values)
        
        final_diversity = calculate_diversity(current_population)
        
        # Diversity should not collapse to zero (premature convergence)
        # This is a basic check - in practice, diversity should be maintained for longer
        assert final_diversity > 0.0