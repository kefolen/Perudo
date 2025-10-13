# Monte Carlo Agent Enhancement Specification

This document outlines the enhancement plan for the Monte Carlo Agent implementation in the Perudo AI project, following Test-Driven Development (TDD) principles.

## Overview

The goal is to improve the current Monte Carlo (MC) Agent to outperform the BaselineAgent by addressing key limitations in simulation realism, computational efficiency, and decision-making accuracy.

### Current Limitations

The existing MC Agent shows suboptimal performance (~25% win rate vs BaselineAgent) due to:
- Uniform determinization sampling that ignores bidding history context
- High variance in rollout outcomes affecting decision quality
- Sequential execution limiting computational efficiency
- Inconsistent rule application between simulation and rollouts

## Enhancement Objectives

### Primary Goals

1. **Weighted Determinization**: Sample hidden dice hands based on bidding history plausibility
2. **Action Pruning**: Use statistical priors to reduce candidate actions before simulation
3. **Parallelized Evaluation**: Utilize multi-core processing for faster simulations
4. **Unified Rule Application**: Centralize game logic to ensure consistency
5. **Variance Reduction**: Implement reward shaping and early cutoff strategies

### Success Metrics

- MC Agent win rate vs BaselineAgent: ≥60%
- Average decision time: <0.3s with MC_N=200
- Simulation variance: <0.1 standard deviation after 500 simulations
- Determinization entropy: Lower than uniform sampling baseline

## Technical Approach

### Weighted Determinization

Replace uniform sampling with history-aware determinization that considers:
- Bidding patterns and their statistical likelihood
- Player behavior consistency
- Dice distribution probabilities given observed actions

**Test Requirements:**
- Verify weighted samples have higher plausibility scores
- Confirm convergence to analytical probabilities on known states
- Validate improved decision quality through tournament results

### Action Pruning Strategy

Implement statistical priors to pre-filter candidate actions:
- Calculate binomial probability for each potential bid
- Retain top-K most promising actions (default K=10)
- Always preserve 'call' and 'exact' actions regardless of ranking

**Test Requirements:**
- Ensure BaselineAgent's chosen actions are rarely pruned
- Measure performance improvement vs full action evaluation
- Validate decision quality maintenance with reduced action space

### Computational Optimization

Enhance performance through:
- Multiprocessing with configurable worker pools
- Batched simulation tasks to reduce overhead
- Early stopping mechanisms for clearly inferior actions

**Test Requirements:**
- Benchmark execution time improvements
- Verify result reproducibility with fixed seeds
- Test scalability across different core counts

### Rule Consistency

Implement centralized `apply_action()` function to ensure:
- Identical rule enforcement across all game contexts
- Proper maputa restriction handling
- Correct exact call mechanics implementation

**Test Requirements:**
- Unit tests for each action type with fixed dice configurations
- Validation of dice count updates and turn transitions
- Consistency checks between simulator and rollout logic

## Implementation Strategy

### Development Phases

1. **Foundation (Week 1)**: Implement weighted determinization with comprehensive tests
2. **Optimization (Week 2)**: Add action pruning and parallelization with performance benchmarks
3. **Integration (Week 3)**: Implement unified rule logic with validation tests
4. **Refinement (Week 4)**: Add variance reduction and conduct tournament evaluation
5. **Documentation (Week 5)**: Finalize documentation and code cleanup

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mc_n` | 200 | Simulations per action evaluation |
| `mc_prune_k` | 10 | Maximum candidate actions |
| `mc_parallel` | True | Enable multiprocessing |
| `num_workers` | Auto | Worker process count |
| `batch_size` | 8 | Rollouts per worker task |
| `max_rounds` | 8 | Simulation round cutoff |

### Testing Requirements

Following TDD principles, all enhancements must include:
- Unit tests for individual component functionality
- Integration tests for agent interaction validation
- Performance benchmarks for optimization verification
- Regression tests to ensure no functionality degradation

## Expected Outcomes

### Performance Improvements

- Significant win rate improvement against BaselineAgent
- Faster decision-making through optimization techniques
- More realistic game simulations through weighted sampling
- Better decision consistency through unified rule application

### Code Quality Benefits

- Modular, testable implementation following TDD principles
- Comprehensive test coverage for all new functionality
- Clear separation of concerns between components
- Maintainable codebase ready for future ML extensions

## Future Extension Opportunities

The enhanced MC Agent will provide a solid foundation for:
- Information Set Monte Carlo Tree Search (ISMCTS) implementation
- Neural network policy and value function integration
- Advanced opponent modeling capabilities
- Multi-agent reinforcement learning experiments

## Validation Approach

Success will be measured through:
- Automated tournament results comparing agent performance
- Performance benchmarks validating optimization goals
- Code quality metrics ensuring maintainability standards
- Comprehensive test suite coverage and execution results

This specification focuses on delivering measurable improvements while maintaining the project's educational accessibility and modular architecture.