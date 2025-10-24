# MC Agent Evolutionary Optimization Framework - Specification

This document outlines the implementation of an evolutionary optimization system for Monte Carlo agent microparameters, following Test-Driven Development principles and integrating with the existing tournament framework.

## Design Principles

- **TDD Compliance**: All components developed using Red-Green-Refactor cycle
- **Modular Integration**: Leverage existing tournament infrastructure  
- **Computational Efficiency**: Multi-fidelity evaluation with successive halving
- **Robust Evaluation**: Variance-penalized fitness against diverse opponents
- **Maintainable Architecture**: Clean separation between optimization and game mechanics
- **Parameter Focus**: Optimize only non-boolean, non-deprecated parameters

## Parameter Scope (Updated Requirements)

### Optimizable Parameters

Based on current MC agent implementation, excluding deprecated (`max_rounds`, `simulate_to_round_end`) and boolean parameters (kept as `true`):

```json
{
  "core_parameters": {
    "n": {
      "type": "int", 
      "min": 100, 
      "max": 2000, 
      "default": 400,
      "description": "Number of Monte Carlo simulations per action evaluation"
    },
    "chunk_size": {
      "type": "int", 
      "min": 4, 
      "max": 32, 
      "default": 8,
      "description": "Batch size for simulation processing"
    },
    "early_stop_margin": {
      "type": "float", 
      "min": 0.05, 
      "max": 0.3, 
      "default": 0.15,
      "description": "Threshold for early stopping inferior actions"
    },
    "trust_learning_rate": {
      "type": "float", 
      "min": 0.01, 
      "max": 0.5, 
      "default": 0.1,
      "description": "Learning rate for player trust dynamics"
    },
    "history_memory_rounds": {
      "type": "int", 
      "min": 5, 
      "max": 20, 
      "default": 10,
      "description": "Number of rounds to remember for historical analysis"
    },
    "num_workers": {
      "type": "int", 
      "min": 1, 
      "max": 8, 
      "default": 4,
      "description": "Number of parallel worker processes"
    }
  },
  "utils_microparameters": {
    "decay_factor": {
      "type": "float", 
      "min": 0.6, 
      "max": 0.95, 
      "default": 0.8,
      "description": "Recency weighting decay in calculate_recency_weight"
    },
    "scaling_factor": {
      "type": "float", 
      "min": 1.0, 
      "max": 3.0, 
      "default": 2.0,
      "description": "Bid plausibility scaling in sample_weighted_dice"
    },
    "face_weight_boost": {
      "type": "float", 
      "min": 1.2, 
      "max": 3.0, 
      "default": 2.0,
      "description": "Weight multiplier for targeted face sampling"
    },
    "min_weight_threshold": {
      "type": "float", 
      "min": 0.05, 
      "max": 0.2, 
      "default": 0.1,
      "description": "Minimum weight threshold for historical data"
    }
  },
  "fixed_boolean_flags": {
    "weighted_sampling": true,
    "enable_parallel": true,
    "enhanced_pruning": true,
    "variance_reduction": true,
    "betting_history_enabled": true,
    "player_trust_enabled": true,
    "bayesian_sampling": true,
    "note": "All boolean flags remain true and are not optimized"
  }
}
```

## Technical Architecture

### Core Components Structure

```
eval/
├── hyperopt_framework.py      # Main optimization framework
├── fitness_evaluation.py      # Robust tournament evaluation  
├── optimization_config.json   # Parameter schema and settings
└── results/                   # Optimization results and logs

agents/
└── optimizable_mc_agent.py    # Parameter-injectable MC agent

tests/
├── unit/
│   ├── test_parameter_space.py        # Parameter encoding/decoding
│   ├── test_optimizable_agent.py      # Parameter injection
│   └── test_evolutionary_optimizer.py # Optimization algorithms
├── integration/
│   ├── test_fitness_evaluation.py     # Tournament integration
│   └── test_optimization_pipeline.py  # End-to-end optimization
└── performance/
    └── test_multifidelity_evaluation.py # Evaluation efficiency
```

### Parameter Management System

```python
# eval/hyperopt_framework.py
class ParameterSpace:
    """Manages parameter encoding/decoding and validation."""
    
    def __init__(self, config_path="eval/optimization_config.json"):
        self.config = self._load_config(config_path)
        self.bounds = self._extract_bounds()
    
    def encode_genotype(self, params_dict):
        """Convert parameter dictionary to optimization vector."""
        pass
    
    def decode_genotype(self, genotype):
        """Convert optimization vector to parameter dictionary.""" 
        pass
    
    def validate_parameters(self, params_dict):
        """Validate parameter values against schema."""
        pass

class EvolutionaryOptimizer:
    """CMA-ES based parameter optimization."""
    
    def __init__(self, parameter_space, population_size=20):
        self.param_space = parameter_space
        self.population_size = population_size
        self.cma_es = None  # Initialize CMA-ES
    
    def optimize(self, fitness_function, max_generations=50):
        """Run evolutionary optimization."""
        pass
```

### Agent Parameter Injection

```python
# agents/optimizable_mc_agent.py
class OptimizableMCAgent(MonteCarloAgent):
    """MC Agent with runtime parameter modification."""
    
    def __init__(self, **base_params):
        super().__init__(**base_params)
        self._utils_params = {}
    
    def update_parameters(self, params_dict):
        """Update agent parameters at runtime."""
        # Core MC parameters
        if 'n' in params_dict:
            self.N = max(1, int(params_dict['n']))
        if 'chunk_size' in params_dict:
            self.chunk_size = max(1, int(params_dict['chunk_size']))
        if 'early_stop_margin' in params_dict:
            self.early_stop_margin = float(params_dict['early_stop_margin'])
        # ... other core parameters
        
        # Utils microparameters
        self._utils_params = {
            'decay_factor': params_dict.get('decay_factor', 0.8),
            'scaling_factor': params_dict.get('scaling_factor', 2.0),
            'face_weight_boost': params_dict.get('face_weight_boost', 2.0),
            'min_weight_threshold': params_dict.get('min_weight_threshold', 0.1)
        }
    
    def get_utils_param(self, param_name, default):
        """Get utils parameter for mc_utils functions."""
        return self._utils_params.get(param_name, default)
```

## Fitness Function Design

### Robust Tournament Evaluation

```python
# eval/fitness_evaluation.py
def evaluate_candidate_fitness(params_dict, evaluation_config):
    """
    Evaluate parameter configuration with variance penalty.
    
    Args:
        params_dict: Parameter configuration to evaluate
        evaluation_config: Opponent mix and game settings
    
    Returns:
        fitness_score: Win rate minus variance penalty
    """
    
    # Create optimizable agent with parameters
    candidate = OptimizableMCAgent()
    candidate.update_parameters(params_dict)
    
    # Diverse opponent mix for robustness
    opponents = [
        BaselineAgent("baseline_1"),
        BaselineAgent("baseline_2"), 
        RandomAgent("random_1"),
        # Previous generation champions
        load_champion_agent("prev_gen_1"),
        load_champion_agent("prev_gen_2")
    ]
    
    # Tournament evaluation
    results = run_tournament(candidate, opponents, 
                           num_games=evaluation_config['num_games'])
    
    # Calculate fitness with variance penalty
    win_rate = calculate_win_rate(results, candidate)
    variance_penalty = calculate_variance_penalty(results)
    robustness_bonus = calculate_robustness_bonus(results)
    
    fitness = win_rate - variance_penalty + robustness_bonus
    
    return {
        'fitness': fitness,
        'win_rate': win_rate,
        'variance': variance_penalty,
        'robustness': robustness_bonus,
        'detailed_results': results
    }
```

### Multi-Fidelity Evaluation Strategy

```python
class MultiFidelityEvaluator:
    """Successive halving for computational efficiency."""
    
    def __init__(self):
        self.fidelity_levels = [
            {'games': 10, 'promotion_rate': 0.5},   # Initial screening
            {'games': 50, 'promotion_rate': 0.3},   # Intermediate  
            {'games': 200, 'promotion_rate': 0.1}   # Final assessment
        ]
    
    def evaluate_population(self, population, generation):
        """Evaluate population with successive halving."""
        candidates = population.copy()
        
        for level in self.fidelity_levels:
            # Evaluate all candidates at current fidelity
            results = []
            for candidate in candidates:
                fitness = evaluate_candidate_fitness(
                    candidate, {'num_games': level['games']}
                )
                results.append((candidate, fitness))
            
            # Sort by fitness and promote top fraction
            results.sort(key=lambda x: x[1]['fitness'], reverse=True)
            promotion_count = max(1, int(len(results) * level['promotion_rate']))
            candidates = [r[0] for r in results[:promotion_count]]
            
            # Log fidelity level results
            self._log_fidelity_results(level, results, generation)
        
        return candidates  # Final survivors
```

## Implementation Plan (TDD Approach)

### Phase 1: Parameter Infrastructure (Week 1)

**TDD Cycle 1.1: Parameter Schema**
- **Red**: Write tests for ParameterSpace parameter encoding/decoding
- **Green**: Implement basic ParameterSpace with JSON schema validation
- **Refactor**: Optimize parameter validation and type conversion

**TDD Cycle 1.2: Parameter Injection** 
- **Red**: Write tests for OptimizableMCAgent parameter updates
- **Green**: Implement parameter injection for core MC parameters
- **Refactor**: Clean up parameter update interface

**TDD Cycle 1.3: Utils Integration**
- **Red**: Write tests for mc_utils parameter passing
- **Green**: Modify mc_utils functions to accept dynamic parameters
- **Refactor**: Ensure backward compatibility

### Phase 2: Basic Evolutionary Framework (Week 2)

**TDD Cycle 2.1: Simple Genetic Algorithm**
- **Red**: Write tests for basic genetic operations (mutation, crossover)
- **Green**: Implement simple GA for parameter subset
- **Refactor**: Optimize genetic operators

**TDD Cycle 2.2: Tournament Integration**
- **Red**: Write tests for fitness evaluation with existing tournament system
- **Green**: Implement basic fitness function using tournament results
- **Refactor**: Clean up tournament integration interface

**TDD Cycle 2.3: End-to-End Pipeline**
- **Red**: Write integration tests for complete optimization cycle
- **Green**: Implement basic optimization loop
- **Refactor**: Add logging and result persistence

### Phase 3: Advanced Optimization (Week 3)

**TDD Cycle 3.1: CMA-ES Implementation**
- **Red**: Write tests for CMA-ES algorithm with continuous parameters
- **Green**: Implement CMA-ES optimizer for parameter optimization
- **Refactor**: Tune CMA-ES hyperparameters

**TDD Cycle 3.2: Multi-Fidelity Evaluation**
- **Red**: Write tests for successive halving evaluation
- **Green**: Implement multi-fidelity evaluation system
- **Refactor**: Optimize computational efficiency

**TDD Cycle 3.3: Robust Fitness Function**
- **Red**: Write tests for variance-penalized fitness calculation
- **Green**: Implement sophisticated fitness function with robustness metrics
- **Refactor**: Balance fitness components optimally

### Phase 4: Production Features (Week 4)

**TDD Cycle 4.1: Result Management**
- **Red**: Write tests for result logging, checkpointing, and resume capability
- **Green**: Implement comprehensive result management system
- **Refactor**: Optimize storage and retrieval

**TDD Cycle 4.2: Asynchronous Evaluation**
- **Red**: Write tests for distributed candidate evaluation
- **Green**: Implement worker-based asynchronous evaluation
- **Refactor**: Optimize resource utilization

**TDD Cycle 4.3: Analysis Tools**
- **Red**: Write tests for parameter interaction analysis
- **Green**: Implement parameter sensitivity and interaction analysis
- **Refactor**: Create visualization and reporting tools

## Expected Performance Improvements

### Optimization Targets

- **Win Rate Improvement**: 10-20% over default parameter settings
- **Computational Efficiency**: 60-80% reduction in evaluation time via multi-fidelity
- **Robustness**: Reduced variance in performance across different opponent mixes
- **Parameter Insights**: Discovery of optimal parameter interactions and ranges

### Computational Efficiency

```python
# Multi-fidelity cost reduction
baseline_cost = 200_games × 20_candidates = 4000_games
multifidelity_cost = (10×20) + (50×10) + (200×2) = 1100_games
efficiency_gain = 72.5% reduction
```

### Success Metrics

1. **Performance Gains**: Optimized agents consistently outperform default settings
2. **Parameter Discovery**: Clear optimal ranges identified for each parameter
3. **Interaction Understanding**: Significant parameter interactions documented
4. **Computational Feasibility**: Complete optimization run within 24 hours
5. **Robustness Validation**: Optimized parameters work across diverse scenarios

## Integration with Existing Codebase

### Backward Compatibility

- All existing MC agent functionality preserved
- Default parameter values unchanged for existing code
- Optional optimization framework doesn't affect core game mechanics

### Tournament System Leverage

```python
# Direct integration with existing tournament.py
def run_optimization_tournament(candidate_agent, opponents, games=50):
    """Use existing tournament infrastructure for optimization."""
    sim = PerudoSimulator(num_players=len(opponents)+1, start_dice=5, 
                         ones_are_wild=True, use_maputa=True, use_exact=True)
    
    agents = [candidate_agent] + opponents
    results = play_match(sim, [type(a).__name__.lower() for a in agents], 
                        games=games, mc_n=candidate_agent.N)
    
    return results
```

### Testing Integration

- Utilize existing test fixtures and scenarios
- Extend existing performance benchmarks
- Maintain test coverage standards
- Follow established testing patterns

## Configuration Management

### Optimization Config Example

```json
{
  "optimization": {
    "algorithm": "CMA-ES",
    "population_size": 20,
    "max_generations": 50,
    "convergence_tolerance": 1e-6
  },
  "evaluation": {
    "multi_fidelity": true,
    "fidelity_levels": [
      {"games": 10, "promotion_rate": 0.5},
      {"games": 50, "promotion_rate": 0.3}, 
      {"games": 200, "promotion_rate": 1.0}
    ],
    "opponent_mix": {
      "baseline_agents": 2,
      "random_agents": 1, 
      "previous_champions": 2
    },
    "fitness_weights": {
      "win_rate": 1.0,
      "variance_penalty": 0.3,
      "robustness_bonus": 0.1
    }
  },
  "logging": {
    "log_level": "INFO",
    "save_checkpoints": true,
    "checkpoint_frequency": 5,
    "detailed_results": true
  }
}
```

## Conclusion

This evolutionary optimization framework provides a systematic approach to discovering optimal Monte Carlo agent parameters while maintaining the project's TDD principles and modular architecture. By focusing only on optimizable parameters and leveraging the existing tournament infrastructure, the system should efficiently discover parameter configurations that significantly improve agent performance while remaining computationally feasible and maintainable.

The multi-fidelity evaluation strategy and variance-penalized fitness function ensure robust parameter discovery that generalizes well across diverse game scenarios, making this optimization framework a valuable addition to the Perudo AI development toolkit.