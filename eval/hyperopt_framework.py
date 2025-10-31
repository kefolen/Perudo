"""
Evolutionary Optimization Framework for MC Agent Parameters

This module provides the core parameter management and optimization infrastructure
for evolving Monte Carlo agent microparameters using evolutionary algorithms.

Main Components:
- ParameterSpace: Parameter encoding/decoding and validation
- GeneticAlgorithm: Simple genetic algorithm implementation (Phase 2)
- EvolutionaryOptimizer: CMA-ES based parameter optimization (Phase 3)
- MultiFidelityEvaluator: Successive halving evaluation (Phase 3)
"""
import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Union, Tuple


class ParameterSpace:
    """
    Manages parameter encoding/decoding and validation for evolutionary optimization.
    
    Handles conversion between:
    - Parameter dictionaries (human-readable with proper types and bounds)
    - Genotype vectors (normalized 0-1 values for optimization algorithms)
    
    Provides validation against JSON schema configuration.
    """
    
    def __init__(self, config_path_or_dict: Union[str, Dict[str, Any]] = "eval/optimization_config.json"):
        """
        Initialize ParameterSpace with configuration file or dictionary.
        
        Args:
            config_path_or_dict: Path to JSON configuration file or configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is malformed
        """
        if isinstance(config_path_or_dict, dict):
            # Direct configuration dictionary provided (for testing)
            self.config_path = None
            self.config = config_path_or_dict
        else:
            # Configuration file path provided
            self.config_path = config_path_or_dict
            self.config = self._load_config(config_path_or_dict)
        
        self.bounds = self._extract_bounds()
        self.parameter_order = self._get_parameter_order()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Malformed JSON in config file: {config_path}", e.doc, e.pos)
    
    def _extract_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameter bounds and types from configuration."""
        bounds = {}
        
        # Handle nested structure (production config)
        if "parameters" in self.config:
            # Extract from core parameters
            core_params = self.config.get("parameters", {}).get("core_parameters", {})
            for param_name, param_config in core_params.items():
                bounds[param_name] = {
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'type': param_config['type'],
                    'default': param_config['default']
                }
            
            # Extract from utils microparameters
            utils_params = self.config.get("parameters", {}).get("utils_microparameters", {})
            for param_name, param_config in utils_params.items():
                bounds[param_name] = {
                    'min': param_config['min'],
                    'max': param_config['max'],
                    'type': param_config['type'],
                    'default': param_config['default']
                }
            
            # Handle flat structure within parameters (test configs)
            flat_params = self.config.get("parameters", {})
            for param_name, param_config in flat_params.items():
                if param_name not in ["core_parameters", "utils_microparameters"] and isinstance(param_config, dict):
                    if all(key in param_config for key in ['min', 'max', 'type', 'default']):
                        bounds[param_name] = {
                            'min': param_config['min'],
                            'max': param_config['max'],
                            'type': param_config['type'],
                            'default': param_config['default']
                        }
        else:
            # Handle direct flat structure (simple test configs)
            for param_name, param_config in self.config.items():
                if isinstance(param_config, dict) and all(key in param_config for key in ['min', 'max', 'type', 'default']):
                    bounds[param_name] = {
                        'min': param_config['min'],
                        'max': param_config['max'],
                        'type': param_config['type'],
                        'default': param_config['default']
                    }
        
        return bounds
    
    def _get_parameter_order(self) -> List[str]:
        """Get consistent ordering of parameters for genotype encoding/decoding."""
        # Sort parameters alphabetically for consistent ordering
        return sorted(self.bounds.keys())
    
    def encode_genotype(self, params_dict: Dict[str, Union[int, float]]) -> List[float]:
        """
        Convert parameter dictionary to normalized genotype vector.
        
        Args:
            params_dict: Dictionary of parameter names to values
            
        Returns:
            List of normalized values between 0.0 and 1.0
            
        Raises:
            KeyError: If required parameter is missing
            ValueError: If parameter value is out of bounds
        """
        genotype = []
        
        for param_name in self.parameter_order:
            if param_name not in params_dict:
                raise KeyError(f"Required parameter '{param_name}' missing from params_dict")
            
            value = params_dict[param_name]
            bounds = self.bounds[param_name]
            
            # Check bounds
            if value < bounds['min'] or value > bounds['max']:
                raise ValueError(f"Parameter '{param_name}' value {value} out of bounds [{bounds['min']}, {bounds['max']}]")
            
            # Normalize to 0-1 range
            normalized = (value - bounds['min']) / (bounds['max'] - bounds['min'])
            genotype.append(float(normalized))
        
        return genotype
    
    def decode_genotype(self, genotype: List[float]) -> Dict[str, Union[int, float]]:
        """
        Convert normalized genotype vector to parameter dictionary.
        
        Args:
            genotype: List of normalized values between 0.0 and 1.0
            
        Returns:
            Dictionary of parameter names to properly typed values
            
        Raises:
            ValueError: If genotype length doesn't match expected parameters
        """
        if len(genotype) != len(self.parameter_order):
            raise ValueError(f"Genotype length {len(genotype)} doesn't match expected {len(self.parameter_order)}")
        
        params_dict = {}
        
        for i, param_name in enumerate(self.parameter_order):
            normalized_value = genotype[i]
            bounds = self.bounds[param_name]
            
            # Denormalize from 0-1 range to actual bounds
            value = bounds['min'] + normalized_value * (bounds['max'] - bounds['min'])
            
            # Apply proper type conversion
            if bounds['type'] == 'int':
                value = int(round(value))
            elif bounds['type'] == 'float':
                value = float(value)
            
            params_dict[param_name] = value
        
        return params_dict
    
    def validate_parameters(self, params_dict: Dict[str, Union[int, float]]) -> bool:
        """
        Validate parameter dictionary against schema.
        
        Args:
            params_dict: Dictionary of parameter names to values
            
        Returns:
            True if all parameters are valid
            
        Raises:
            KeyError: If required parameter is missing
            ValueError: If parameter value is out of bounds
            TypeError: If parameter has wrong type
        """
        # Check that all required parameters are present
        for param_name in self.parameter_order:
            if param_name not in params_dict:
                raise KeyError(f"Required parameter '{param_name}' missing from params_dict")
        
        # Validate each parameter
        for param_name, value in params_dict.items():
            if param_name not in self.bounds:
                continue  # Skip unknown parameters (could be fixed boolean flags)
            
            bounds = self.bounds[param_name]
            
            # Check type
            expected_type = int if bounds['type'] == 'int' else float
            if not isinstance(value, expected_type):
                # Allow int values for float parameters
                if bounds['type'] == 'float' and isinstance(value, int):
                    pass  # This is acceptable
                else:
                    raise TypeError(f"Parameter '{param_name}' should be {bounds['type']}, got {type(value).__name__}")
            
            # Check bounds
            if value < bounds['min'] or value > bounds['max']:
                raise ValueError(f"Parameter '{param_name}' value {value} out of bounds [{bounds['min']}, {bounds['max']}]")
        
        return True
    
    def get_default_parameters(self) -> Dict[str, Union[int, float]]:
        """
        Get dictionary of default parameter values.
        
        Returns:
            Dictionary of parameter names to default values
        """
        defaults = {}
        for param_name in self.parameter_order:
            bounds = self.bounds[param_name]
            default_value = bounds['default']
            
            # Apply proper type conversion
            if bounds['type'] == 'int':
                default_value = int(default_value)
            elif bounds['type'] == 'float':
                default_value = float(default_value)
            
            defaults[param_name] = default_value
        
        return defaults
    
    def get_parameter_bounds(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get parameter bounds information.
        
        Returns:
            Dictionary mapping parameter names to their bounds info
        """
        return {param: {'min': bounds['min'], 'max': bounds['max'], 'type': bounds['type']} 
                for param, bounds in self.bounds.items()}
    
    def get_parameter_count(self) -> int:
        """Get the total number of optimizable parameters."""
        return len(self.parameter_order)
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names in encoding order."""
        return self.parameter_order.copy()


class EvolutionaryOptimizer:
    """
    CMA-ES based parameter optimization (Phase 2 implementation).
    
    This is a placeholder for Phase 2 - Basic Evolutionary Framework.
    Will implement genetic algorithms and CMA-ES optimization.
    """
    
    def __init__(self, parameter_space: ParameterSpace, population_size: int = 20):
        """
        Initialize evolutionary optimizer.
        
        Args:
            parameter_space: ParameterSpace instance for encoding/decoding
            population_size: Size of the population for evolutionary search
        """
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.generation = 0
    
    def optimize(self, fitness_function, max_generations: int = 50):
        """
        Run evolutionary optimization (placeholder for Phase 2).
        
        Args:
            fitness_function: Function that evaluates parameter configurations
            max_generations: Maximum number of generations to run
            
        Returns:
            Best parameter configuration found
        """
        # Phase 2 implementation - placeholder
        raise NotImplementedError("EvolutionaryOptimizer.optimize() will be implemented in Phase 2")


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
    
    CMA-ES is a state-of-the-art evolutionary algorithm for continuous optimization.
    It adapts the covariance matrix of the search distribution based on the evolution path,
    enabling efficient optimization of continuous parameter spaces.
    
    Key features:
    - Adaptive covariance matrix for correlated parameter optimization
    - Step-size control with evolution path
    - Rank-based selection with weighted recombination
    - Convergence detection and early stopping
    """
    
    def __init__(self, parameter_space: ParameterSpace, population_size: int = 20, 
                 random_seed: int = None):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            parameter_space: ParameterSpace for encoding/decoding parameters
            population_size: Number of offspring per generation (lambda)
            random_seed: Random seed for reproducibility
        """
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.generation = 0
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        
        # Problem dimension
        self.n = parameter_space.get_parameter_count()
        
        # CMA-ES parameters
        self.lam = population_size  # Population size (lambda)
        self.mu = self.lam // 2  # Number of parents
        
        # Selection weights (decreasing)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)  # Normalize weights
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)  # Effective mu
        
        # Adaptation parameters
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)  # Covariance path
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)  # Step-size path
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)  # Rank-one update
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n + 2) ** 2 + self.mueff))  # Rank-mu update
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.n + 1)) - 1) + self.cs  # Step-size damping
        
        # Dynamic state variables
        self.mean = 0.5 * np.ones(self.n)  # Initial mean (center of search space)
        self.sigma = 0.3  # Initial step size
        self.C = np.eye(self.n)  # Covariance matrix (identity initially)
        self.pc = np.zeros(self.n)  # Evolution path for covariance
        self.ps = np.zeros(self.n)  # Evolution path for step-size
        
        # Eigendecomposition cache
        self.eigeneval = 0  # Generation when eigendecomposition was last computed
        self.B = np.eye(self.n)  # Eigenvectors
        self.D = np.ones(self.n)  # Square roots of eigenvalues
        
        # Convergence tolerance
        self.convergence_tolerance = 1e-12
    
    def sample_population(self) -> List[List[float]]:
        """
        Sample a population from the current search distribution.
        
        Returns:
            List of normalized genotype vectors
        """
        # Update eigendecomposition if needed
        if self.generation - self.eigeneval > self.lam / (self.c1 + self.cmu) / self.n / 10:
            self.eigeneval = self.generation
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Enforce symmetry
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 0))  # Handle numerical issues
        
        population = []
        for _ in range(self.population_size):
            # Sample from multivariate normal distribution
            z = self.rng.randn(self.n)
            y = self.B.dot(self.D * z)  # Transform by covariance
            x = self.mean + self.sigma * y  # Scale and shift
            
            # Clamp to [0, 1] bounds
            x = np.clip(x, 0.0, 1.0)
            population.append(x.tolist())
        
        return population
    
    def update_distribution(self, population: List[List[float]], fitness_values: List[float]):
        """
        Update the search distribution based on population fitness.
        
        Args:
            population: List of genotype vectors
            fitness_values: Corresponding fitness values (higher is better)
        """
        # Handle degenerate cases
        if any(np.isnan(fitness_values)):
            raise ValueError("NaN fitness values encountered")
        
        # Convert to numpy arrays
        pop_array = np.array(population)
        fitness_array = np.array(fitness_values)
        
        # Sort by fitness (descending)
        idx = np.argsort(fitness_array)[::-1]
        sorted_pop = pop_array[idx]
        
        # Select best mu individuals for recombination
        selected = sorted_pop[:self.mu]
        
        # Weighted recombination for new mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * selected, axis=0)
        
        # Evolution path for step-size control
        y = (self.mean - old_mean) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.B.dot(y)
        
        # Evolution path for covariance matrix
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation + 2)) 
                < (1.4 + 2 / (self.n + 1)) * np.sqrt(self.n))
        
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        
        # Covariance matrix adaptation
        # Rank-one update
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc)
        
        # Rank-mu update
        for i in range(self.mu):
            yi = (selected[i] - old_mean) / self.sigma
            self.C += self.cmu * self.weights[i] * np.outer(yi, yi)
        
        # Step-size adaptation
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n) - 1))
        
        # Increment generation
        self.generation += 1
    
    def has_converged(self) -> bool:
        """
        Check if the algorithm has converged.
        
        Returns:
            True if converged, False otherwise
        """
        # Check if step-size is too small
        if self.sigma < self.convergence_tolerance:
            return True
        
        # Check if coordinate standard deviations are too small
        if np.max(self.D) * self.sigma < self.convergence_tolerance:
            return True
        
        # Additional convergence check: if sigma is extremely small (e.g., manually set)
        if self.sigma < 1e-7:
            return True
        
        return False
    
    def optimize(self, fitness_function, max_generations: int = 50, 
                 target_fitness: float = None) -> Tuple[List[float], float]:
        """
        Run CMA-ES optimization.
        
        Args:
            fitness_function: Function that takes genotype and returns fitness (higher is better)
            max_generations: Maximum number of generations
            target_fitness: Stop if this fitness is reached
            
        Returns:
            Tuple of (best_genotype, best_fitness)
        """
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(max_generations):
            # Sample population
            population = self.sample_population()
            
            # Evaluate fitness
            fitness_values = [fitness_function(individual) for individual in population]
            
            # Track best individual
            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
            
            # Check for early stopping
            if target_fitness is not None and best_fitness >= target_fitness:
                break
            
            # Check for convergence
            if self.has_converged():
                break
            
            # Update distribution
            self.update_distribution(population, fitness_values)
        
        return best_individual, best_fitness


class GeneticAlgorithm:
    """
    Simple genetic algorithm implementation for parameter optimization.
    
    Implements basic genetic operations including mutation, crossover, selection,
    and population evolution for Monte Carlo agent parameter optimization.
    """
    
    def __init__(self, parameter_space: ParameterSpace, population_size: int = 20,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8, 
                 elitism_count: int = 2):
        """
        Initialize genetic algorithm.
        
        Args:
            parameter_space: ParameterSpace for encoding/decoding parameters
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover between parents
            elitism_count: Number of best individuals to preserve each generation
        """
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.generation = 0
        self.rng = random.Random()
    
    def create_random_individual(self) -> List[float]:
        """
        Create a random individual (genotype) with values in [0, 1].
        
        Returns:
            List of normalized parameter values
        """
        num_params = self.parameter_space.get_parameter_count()
        return [self.rng.random() for _ in range(num_params)]
    
    def initialize_population(self) -> List[List[float]]:
        """
        Initialize population with random individuals.
        
        Returns:
            List of genotype vectors
        """
        return [self.create_random_individual() for _ in range(self.population_size)]
    
    def mutate(self, individual: List[float]) -> List[float]:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Individual to mutate (will be copied)
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if self.rng.random() < self.mutation_rate:
                # Gaussian mutation with small standard deviation
                mutated[i] += self.rng.gauss(0, 0.1)
                # Clamp to [0, 1] bounds
                mutated[i] = max(0.0, min(1.0, mutated[i]))
        
        return mutated
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Apply crossover between two parents.
        
        Args:
            parent1: First parent genotype
            parent2: Second parent genotype
            
        Returns:
            Tuple of two child genotypes
        """
        if self.rng.random() > self.crossover_rate or len(parent1) < 2:
            # No crossover (either due to rate or too few parameters), return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        crossover_point = self.rng.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def tournament_selection(self, population: List[List[float]], 
                           fitness_values: List[float], tournament_size: int = 3) -> List[float]:
        """
        Select individual using tournament selection.
        
        Args:
            population: Current population
            fitness_values: Fitness value for each individual
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual genotype
        """
        # Select random individuals for tournament
        tournament_indices = [self.rng.randint(0, len(population) - 1) 
                            for _ in range(tournament_size)]
        
        # Find best individual in tournament
        best_index = max(tournament_indices, key=lambda i: fitness_values[i])
        
        return population[best_index].copy()
    
    def evolve_population(self, population: List[List[float]], 
                         fitness_values: List[float]) -> List[List[float]]:
        """
        Evolve population for one generation.
        
        Args:
            population: Current population
            fitness_values: Fitness values for current population
            
        Returns:
            New population after evolution
        """
        new_population = []
        
        # Elitism: preserve best individuals
        if self.elitism_count > 0:
            # Sort by fitness (descending)
            sorted_indices = sorted(range(len(fitness_values)), 
                                  key=lambda i: fitness_values[i], reverse=True)
            
            for i in range(min(self.elitism_count, len(population))):
                new_population.append(population[sorted_indices[i]].copy())
        
        # Fill rest of population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add children to new population
            if len(new_population) < self.population_size:
                new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.generation += 1
        return new_population


class MultiFidelityEvaluator:
    """
    Successive halving for computational efficiency.
    
    Implements multi-fidelity evaluation where candidates are evaluated at
    increasing levels of fidelity (number of games), with only the best
    performers promoted to higher (more expensive) evaluation levels.
    
    This approach can reduce computational cost by 60-80% compared to
    evaluating all candidates at the highest fidelity level.
    """
    
    def __init__(self, fidelity_levels: List[Dict[str, Union[int, float]]] = None):
        """
        Initialize multi-fidelity evaluator.
        
        Args:
            fidelity_levels: List of fidelity level configurations.
                           Each level should have 'games' and 'promotion_rate' keys.
        """
        if fidelity_levels is None:
            # Default fidelity levels from specification
            self.fidelity_levels = [
                {'games': 10, 'promotion_rate': 0.5},   # Initial screening
                {'games': 50, 'promotion_rate': 0.3},   # Intermediate  
                {'games': 200, 'promotion_rate': 1.0}   # Final assessment
            ]
        else:
            self.fidelity_levels = fidelity_levels
        
        # Validate fidelity levels
        self._validate_fidelity_levels()
    
    def _validate_fidelity_levels(self):
        """Validate fidelity level configuration."""
        if not self.fidelity_levels:
            raise ValueError("At least one fidelity level must be specified")
        
        for i, level in enumerate(self.fidelity_levels):
            if 'games' not in level or 'promotion_rate' not in level:
                raise ValueError(f"Fidelity level {i} missing required keys 'games' or 'promotion_rate'")
            
            if level['games'] <= 0:
                raise ValueError(f"Fidelity level {i}: games must be positive")
            
            if not (0 < level['promotion_rate'] <= 1.0):
                raise ValueError(f"Fidelity level {i}: promotion_rate must be in (0, 1]")
        
        # Check that games are increasing
        games = [level['games'] for level in self.fidelity_levels]
        if games != sorted(games):
            raise ValueError("Fidelity levels must have increasing number of games")
    
    def evaluate_candidates_at_level(self, candidates: List[Dict], fidelity_level: Dict, 
                                   fitness_function, **kwargs) -> List[Tuple[Dict, float]]:
        """
        Evaluate candidates at a single fidelity level.
        
        Args:
            candidates: List of parameter configurations to evaluate
            fidelity_level: Fidelity level specification with 'games' key
            fitness_function: Function that takes (params_dict, num_games, **kwargs) -> fitness
            **kwargs: Additional arguments passed to fitness function
            
        Returns:
            List of (candidate, fitness) tuples
        """
        results = []
        num_games = fidelity_level['games']
        
        for candidate in candidates:
            try:
                fitness = fitness_function(candidate, num_games, **kwargs)
                results.append((candidate, fitness))
            except Exception as e:
                # Handle evaluation failures gracefully
                print(f"Warning: Evaluation failed for candidate {candidate}: {e}")
                results.append((candidate, 0.0))  # Assign poor fitness
        
        return results
    
    def promote_candidates(self, candidate_results: List[Tuple[Dict, float]], 
                         promotion_rate: float) -> List[Tuple[Dict, float]]:
        """
        Promote best candidates based on fitness.
        
        Args:
            candidate_results: List of (candidate, fitness) tuples
            promotion_rate: Fraction of candidates to promote
            
        Returns:
            List of promoted (candidate, fitness) tuples
        """
        if not candidate_results:
            return []
        
        # Sort by fitness (descending - higher is better)
        sorted_results = sorted(candidate_results, key=lambda x: x[1], reverse=True)
        
        # Calculate number to promote (at least 1 if any candidates exist)
        num_to_promote = max(1, int(len(sorted_results) * promotion_rate))
        
        return sorted_results[:num_to_promote]
    
    def evaluate_population(self, population: List[Dict], generation: int, 
                          fitness_function, **kwargs) -> List[Dict]:
        """
        Evaluate population with successive halving.
        
        Args:
            population: List of parameter configurations to evaluate
            generation: Current generation number (for logging)
            fitness_function: Function that evaluates parameter configurations
            **kwargs: Additional arguments passed to fitness function
            
        Returns:
            List of final surviving parameter configurations
        """
        candidates = population.copy()
        
        for level_idx, level in enumerate(self.fidelity_levels):
            if not candidates:
                break
            
            # Evaluate all candidates at current fidelity level
            results = self.evaluate_candidates_at_level(
                candidates, level, fitness_function, **kwargs
            )
            
            # Log results for this fidelity level
            self._log_fidelity_results(level, results, generation)
            
            # Promote candidates to next level (except for final level)
            if level_idx < len(self.fidelity_levels) - 1:
                promoted_results = self.promote_candidates(results, level['promotion_rate'])
                candidates = [candidate for candidate, fitness in promoted_results]
            else:
                # Final level - return all evaluated candidates
                candidates = [candidate for candidate, fitness in results]
        
        return candidates
    
    def _log_fidelity_results(self, level: Dict, results: List[Tuple[Dict, float]], 
                            generation: int):
        """
        Log results for a fidelity level with enhanced progress information.
        
        Args:
            level: Fidelity level configuration
            results: Evaluation results for this level
            generation: Current generation number
        """
        if hasattr(self, 'custom_log_function') and callable(self.custom_log_function):
            # Custom logging function provided (for testing)
            self.custom_log_function(level, results, generation)
        else:
            # Enhanced default logging with detailed progress information
            if results:
                best_fitness = max(fitness for _, fitness in results)
                avg_fitness = sum(fitness for _, fitness in results) / len(results)
                worst_fitness = min(fitness for _, fitness in results)
                fitness_std = (sum((fitness - avg_fitness) ** 2 for _, fitness in results) / len(results)) ** 0.5
                
                # Log to file with detailed statistics
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Generation {generation} - Fidelity Level {level['games']} games: "
                           f"{len(results)} candidates evaluated - "
                           f"Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f} (±{fitness_std:.4f}), "
                           f"Worst: {worst_fitness:.4f}, Promotion rate: {level['promotion_rate']:.1%}")
                
                # Console output with progress indication
                print(f"    → Fidelity {level['games']} games: {len(results)} candidates - "
                      f"Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}, Range: [{worst_fitness:.4f}-{best_fitness:.4f}]")
                
                # Show fitness distribution for better insight
                fitness_values = [fitness for _, fitness in results]
                fitness_values.sort(reverse=True)
                top_3 = fitness_values[:3] if len(fitness_values) >= 3 else fitness_values
                bottom_3 = fitness_values[-3:] if len(fitness_values) >= 3 else []
                
                if len(results) > 6:
                    print(f"      Top performers: {[f'{f:.4f}' for f in top_3]} | "
                          f"Bottom: {[f'{f:.4f}' for f in bottom_3]}")
                else:
                    print(f"      All fitness scores: {[f'{f:.4f}' for f in fitness_values]}")
                
                # Promotion information
                if level['promotion_rate'] < 1.0:
                    promoted_count = max(1, int(len(results) * level['promotion_rate']))
                    print(f"      → Promoting top {promoted_count}/{len(results)} candidates "
                          f"({level['promotion_rate']:.1%}) to next level")
            else:
                print(f"    → Fidelity {level['games']} games: No results to display")
    
    def calculate_evaluation_cost(self, population_size: int) -> int:
        """
        Calculate total evaluation cost (number of games) for multi-fidelity approach.
        
        Args:
            population_size: Initial population size
            
        Returns:
            Total number of games that will be played
        """
        total_cost = 0
        current_size = population_size
        
        for level in self.fidelity_levels:
            # Cost for this level
            level_cost = current_size * level['games']
            total_cost += level_cost
            
            # Update population size for next level
            if level['promotion_rate'] < 1.0:
                current_size = max(1, int(current_size * level['promotion_rate']))
        
        return total_cost
    
    def get_efficiency_statistics(self, population_size: int) -> Dict[str, float]:
        """
        Calculate efficiency statistics compared to baseline evaluation.
        
        Args:
            population_size: Initial population size
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Calculate costs
        multifidelity_cost = self.calculate_evaluation_cost(population_size)
        baseline_cost = population_size * self.fidelity_levels[-1]['games']  # All at highest fidelity
        
        # Calculate efficiency metrics
        efficiency_gain = (baseline_cost - multifidelity_cost) / baseline_cost
        cost_ratio = multifidelity_cost / baseline_cost
        
        return {
            'multifidelity_cost': multifidelity_cost,
            'baseline_cost': baseline_cost,
            'efficiency_gain': efficiency_gain,
            'cost_ratio': cost_ratio,
            'games_saved': baseline_cost - multifidelity_cost
        }