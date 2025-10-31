"""
Tests for CMA-ES Optimizer - TDD Cycle 3.1: CMA-ES Implementation

This module tests the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) 
optimizer for continuous parameter optimization in the evolutionary framework.

CMA-ES is a state-of-the-art evolutionary algorithm particularly well-suited 
for continuous optimization problems like MC agent parameter tuning.
"""
import pytest
import numpy as np
from eval.hyperopt_framework import ParameterSpace, CMAESOptimizer


class TestCMAESOptimizer:
    """Test suite for CMA-ES optimizer implementation."""
    
    @pytest.fixture
    def parameter_space(self):
        """Create ParameterSpace for testing CMA-ES."""
        config = {
            "parameters": {
                "core_parameters": {
                    "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                    "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
                    "early_stop_margin": {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15}
                },
                "utils_microparameters": {
                    "decay_factor": {"type": "float", "min": 0.6, "max": 0.95, "default": 0.8},
                    "scaling_factor": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0}
                }
            }
        }
        return ParameterSpace(config)
    
    @pytest.fixture
    def cma_es_optimizer(self, parameter_space):
        """Create CMA-ES optimizer instance for testing."""
        return CMAESOptimizer(parameter_space, population_size=10)
    
    def test_cma_es_optimizer_initialization(self, parameter_space):
        """Test CMA-ES optimizer initializes correctly."""
        optimizer = CMAESOptimizer(parameter_space, population_size=15)
        
        assert optimizer.parameter_space == parameter_space
        assert optimizer.population_size == 15
        assert optimizer.generation == 0
        
        # Should initialize CMA-ES internal state
        assert hasattr(optimizer, 'mean')
        assert hasattr(optimizer, 'sigma')
        assert hasattr(optimizer, 'C')  # Covariance matrix
        assert hasattr(optimizer, 'pc')  # Evolution path for covariance
        assert hasattr(optimizer, 'ps')  # Evolution path for step-size
        
        # Check dimensions match parameter space
        n_params = parameter_space.get_parameter_count()
        assert len(optimizer.mean) == n_params
        assert optimizer.C.shape == (n_params, n_params)
    
    def test_cma_es_initialization_parameters(self, parameter_space):
        """Test CMA-ES initializes with proper algorithm parameters."""
        optimizer = CMAESOptimizer(parameter_space, population_size=20)
        
        # Check CMA-ES algorithm parameters are set correctly
        n = parameter_space.get_parameter_count()
        lam = optimizer.population_size
        
        assert optimizer.mu == lam // 2  # Number of parents
        assert len(optimizer.weights) == optimizer.mu
        assert optimizer.mueff > 0  # Effective number of parents
        
        # Check adaptation parameters
        assert 0 < optimizer.cc < 1  # Covariance path decay
        assert 0 < optimizer.cs < 1  # Step-size path decay  
        assert optimizer.c1 > 0  # Rank-one covariance update
        assert optimizer.cmu > 0  # Rank-mu covariance update
        assert optimizer.damps > 1  # Step-size damping
    
    def test_cma_es_sample_population(self, cma_es_optimizer):
        """Test CMA-ES can sample a population."""
        population = cma_es_optimizer.sample_population()
        
        # Should return correct number of individuals
        assert len(population) == cma_es_optimizer.population_size
        
        # Each individual should have correct dimensions
        n_params = cma_es_optimizer.parameter_space.get_parameter_count()
        for individual in population:
            assert len(individual) == n_params
            # Values should be normalized between 0 and 1
            assert all(0.0 <= val <= 1.0 for val in individual)
    
    def test_cma_es_update_distribution(self, cma_es_optimizer):
        """Test CMA-ES updates its distribution based on fitness."""
        # Create mock population and fitness values
        population = cma_es_optimizer.sample_population()
        fitness_values = [0.1 + 0.1 * i for i in range(len(population))]  # Increasing fitness
        
        # Store initial state
        initial_mean = cma_es_optimizer.mean.copy()
        initial_sigma = cma_es_optimizer.sigma
        initial_C = cma_es_optimizer.C.copy()
        
        # Update distribution
        cma_es_optimizer.update_distribution(population, fitness_values)
        
        # Distribution should change
        assert not np.array_equal(cma_es_optimizer.mean, initial_mean)
        assert cma_es_optimizer.sigma != initial_sigma  # Step-size should adapt
        
        # Generation should increment
        assert cma_es_optimizer.generation == 1
    
    def test_cma_es_convergence_detection(self, cma_es_optimizer):
        """Test CMA-ES can detect convergence."""
        # Initially should not be converged
        assert not cma_es_optimizer.has_converged()
        
        # Manually set very small sigma to simulate convergence
        cma_es_optimizer.sigma = 1e-8
        
        # Should detect convergence
        assert cma_es_optimizer.has_converged()
    
    def test_cma_es_optimization_simple_quadratic(self, parameter_space):
        """Test CMA-ES optimization on simple quadratic function."""
        # Simple quadratic fitness function (higher is better, peak at 0.5)
        def quadratic_fitness(genotype):
            # Fitness decreases with distance from 0.5 for each parameter
            distance = sum((x - 0.5) ** 2 for x in genotype)
            return 1.0 / (1.0 + distance)  # Higher fitness for points closer to 0.5
        
        optimizer = CMAESOptimizer(parameter_space, population_size=20)
        
        best_individual, best_fitness = optimizer.optimize(
            quadratic_fitness, max_generations=20, target_fitness=0.8
        )
        
        # Should find reasonably good solution
        assert best_fitness > 0.5  # Should improve significantly
        assert len(best_individual) == parameter_space.get_parameter_count()
        
        # Best solution should be close to optimal (all values near 0.5)
        for value in best_individual:
            assert abs(value - 0.5) < 0.4  # Allow reasonable tolerance for stochastic optimization
    
    def test_cma_es_optimization_with_early_stopping(self, parameter_space):
        """Test CMA-ES optimization with early stopping."""
        # Fitness function with clear optimum
        def peaked_fitness(genotype):
            # Peak at first parameter = 0.3, others = 0.7
            target = [0.3] + [0.7] * (len(genotype) - 1)
            distance = sum((x - t) ** 2 for x, t in zip(genotype, target))
            return 1.0 / (1.0 + 10 * distance)
        
        optimizer = CMAESOptimizer(parameter_space, population_size=15)
        
        best_individual, best_fitness = optimizer.optimize(
            peaked_fitness, max_generations=50, target_fitness=0.9
        )
        
        # Should terminate early if target fitness reached
        assert optimizer.generation <= 50
        assert best_fitness > 0.1  # Should make progress
    
    def test_cma_es_parameter_bounds_enforcement(self, cma_es_optimizer):
        """Test that CMA-ES respects parameter bounds."""
        # Sample many populations and check bounds
        for _ in range(10):
            population = cma_es_optimizer.sample_population()
            
            for individual in population:
                # All values should be in [0, 1] (normalized bounds)
                assert all(0.0 <= val <= 1.0 for val in individual)
                
                # Decode and check actual parameter bounds
                params = cma_es_optimizer.parameter_space.decode_genotype(individual)
                bounds = cma_es_optimizer.parameter_space.get_parameter_bounds()
                
                for param_name, value in params.items():
                    param_bounds = bounds[param_name]
                    assert param_bounds['min'] <= value <= param_bounds['max']
    
    def test_cma_es_reproducibility_with_seed(self, parameter_space):
        """Test that CMA-ES is reproducible with random seed."""
        seed = 42
        
        # Create two optimizers with same seed
        optimizer1 = CMAESOptimizer(parameter_space, population_size=10, random_seed=seed)
        optimizer2 = CMAESOptimizer(parameter_space, population_size=10, random_seed=seed)
        
        # Sample populations should be identical
        pop1 = optimizer1.sample_population()
        pop2 = optimizer2.sample_population()
        
        for ind1, ind2 in zip(pop1, pop2):
            assert np.allclose(ind1, ind2, rtol=1e-10)
    
    def test_cma_es_different_population_sizes(self, parameter_space):
        """Test CMA-ES works with different population sizes."""
        for pop_size in [6, 10, 20, 50]:
            optimizer = CMAESOptimizer(parameter_space, population_size=pop_size)
            
            # Should initialize correctly
            assert optimizer.population_size == pop_size
            assert optimizer.mu == pop_size // 2
            
            # Should sample correct population size
            population = optimizer.sample_population()
            assert len(population) == pop_size
    
    def test_cma_es_handles_degenerate_fitness(self, cma_es_optimizer):
        """Test CMA-ES handles degenerate fitness cases."""
        population = cma_es_optimizer.sample_population()
        
        # All fitness values identical (no selection pressure)
        uniform_fitness = [0.5] * len(population)
        
        # Should not crash on update
        try:
            cma_es_optimizer.update_distribution(population, uniform_fitness)
            assert True  # Success if no exception
        except Exception as e:
            pytest.fail(f"CMA-ES failed on uniform fitness: {e}")
        
        # Test with NaN fitness values
        nan_fitness = [float('nan')] * len(population)
        
        with pytest.raises((ValueError, RuntimeError)):
            cma_es_optimizer.update_distribution(population, nan_fitness)
    
    def test_cma_es_optimization_progress_tracking(self, parameter_space):
        """Test that CMA-ES tracks optimization progress."""
        def simple_fitness(genotype):
            return sum(genotype)  # Simple linear fitness
        
        optimizer = CMAESOptimizer(parameter_space, population_size=10)
        
        # Track progress
        history = []
        
        def fitness_with_tracking(genotype):
            fitness = simple_fitness(genotype)
            history.append(fitness)
            return fitness
        
        optimizer.optimize(fitness_with_tracking, max_generations=5)
        
        # Should have evaluated multiple individuals
        assert len(history) >= 50  # At least 5 generations * 10 population
        
        # Check that best fitness generally improves
        generation_bests = []
        for gen in range(5):
            gen_start = gen * 10
            gen_end = (gen + 1) * 10
            gen_best = max(history[gen_start:gen_end])
            generation_bests.append(gen_best)
        
        # Later generations should generally be better than earlier ones
        final_best = max(generation_bests[-2:])  # Best of last 2 generations
        initial_best = max(generation_bests[:2])  # Best of first 2 generations
        assert final_best >= initial_best  # Should make progress