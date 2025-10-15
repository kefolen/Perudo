# Monte Carlo Agent Enhancement Specification

This document outlines the enhancement plan for the Monte Carlo Agent implementation in the Perudo AI project, following Test-Driven Development (TDD) principles.

## Overview

The goal is to further optimize the already sophisticated Monte Carlo (MC) Agent to achieve even better performance against the BaselineAgent by implementing advanced techniques that build upon the existing optimized foundation.

### Current State Analysis

The existing MC Agent is already well-optimized with sophisticated features:
- **Action Pruning**: Uses binomial probability priors to keep top-K promising actions (default prune_k=12)
- **Chunked Simulations**: Processes simulations in batches (default chunk_size=8) for efficiency
- **Early Stopping**: Implements per-action early termination based on performance margins (early_stop_margin=0.1)
- **Round Limiting**: Caps simulation rounds (max_rounds=6) to ensure bounded execution time
- **Optimized Determinization**: Fast dice hand sampling with minimal allocations
- **Heuristic Evaluation**: Option to simulate only to round end and use heuristic win probability
- **Comprehensive Testing**: Extensive performance test suite already exists

### Remaining Optimization Opportunities

Despite the existing sophistication, further improvements are possible in:
- **Weighted Determinization**: Current sampling is uniform; could be made history-aware
- **Parallel Processing**: Current implementation is sequential; could leverage multiprocessing
- **Advanced Pruning**: Could implement more sophisticated action filtering
- **Variance Reduction**: Could implement additional simulation variance reduction techniques

## Enhancement Objectives

### Primary Goals

1. **Weighted Determinization**: Enhance current uniform sampling with history-aware probabilistic sampling
2. **Parallel Processing**: Add optional multiprocessing support for simulation batches
3. **Advanced Variance Reduction**: Implement control variates and importance sampling techniques
4. **Enhanced Pruning**: Combine statistical priors with opponent modeling for better action filtering
5. **Code Modularization**: Split mc_agent.py into logical modules while maintaining backward compatibility

### Success Metrics

Based on existing performance baselines and realistic improvement targets:

- **Performance**: Maintain MC agent's expected superiority over baseline while improving efficiency
- **Speed**: Target <0.5s average decision time with MC_N=200 (current baseline: ~1.0s)
- **Scalability**: Support parallel processing with configurable worker pools
- **Code Quality**: Achieve >95% test coverage for new components
- **Backward Compatibility**: All existing interfaces and parameters must remain functional

## Technical Approach

Building upon the existing sophisticated implementation, the following enhancements will be added:

### Enhanced Weighted Determinization

The current uniform determinization sampling will be enhanced with history-aware probabilistic sampling:
- **Current**: Fast uniform dice generation using list comprehension
- **Enhancement**: Weight dice combinations based on bidding history plausibility
- **Implementation**: Add optional weighted sampling mode while preserving current fast path
- **Backward Compatibility**: Default behavior remains unchanged

**Test Requirements:**
- Compare weighted vs uniform sampling convergence on known game states
- Benchmark performance impact of weighted sampling
- Validate that weighted sampling improves decision quality in tournament play

### Enhanced Action Pruning

Build upon the existing statistical prior-based pruning (current: binomial probability ranking with prune_k=12):
- **Current**: Binomial probability calculation with top-K retention
- **Enhancement**: Add opponent modeling and multi-criteria scoring
- **Implementation**: Extend existing pruning logic with additional heuristics
- **Backward Compatibility**: Maintain current prune_k parameter behavior

**Test Requirements:**
- Ensure existing pruning effectiveness is maintained or improved
- Validate that enhanced pruning rarely filters out optimal actions
- Benchmark pruning effectiveness across different game scenarios

### Parallel Processing Implementation

Add optional multiprocessing support to the existing sequential evaluation:
- **Current**: Sequential action evaluation with chunked simulations (chunk_size=8)
- **Enhancement**: Optional parallel worker pools for simulation batches
- **Implementation**: Configurable multiprocessing with fallback to current sequential mode
- **Backward Compatibility**: Default remains sequential; parallel mode is opt-in

**Test Requirements:**
- Verify parallel and sequential modes produce statistically equivalent results
- Benchmark performance improvements across different core counts
- Test reproducibility with fixed random seeds in parallel mode
- Validate memory usage remains reasonable with multiple workers

### Advanced Variance Reduction

Enhance the existing early stopping mechanism with additional techniques:
- **Current**: Early stopping based on performance margins (early_stop_margin=0.1)
- **Enhancement**: Add control variates and importance sampling for variance reduction
- **Implementation**: Extend existing evaluation logic with optional variance reduction methods
- **Backward Compatibility**: Current early stopping behavior is preserved

**Test Requirements:**
- Measure variance reduction effectiveness in simulation outcomes
- Ensure variance reduction techniques don't introduce bias
- Validate computational overhead is acceptable for the variance reduction benefit

## Implementation Strategy

### Code Modularization

The current single-file implementation (343 lines) could benefit from modularization for better maintainability:
1. **mc_agent.py**: Core MonteCarloAgent class with existing interface
2. **mc_utils.py**: Helper functions for determinization and pruning logic
3. **mc_parallel.py**: Optional parallel processing components
4. **Backward Compatibility**: All existing interfaces, parameters, and behavior must remain unchanged

### Development Phases

**Phase 1 (Week 1-2)**: Enhanced Determinization
- Implement weighted sampling as optional enhancement to existing uniform sampling
- Leverage existing comprehensive test infrastructure
- Use existing performance benchmarks to validate improvements

**Phase 2 (Week 2-3)**: Parallel Processing
- Add optional multiprocessing support with configurable worker pools
- Integrate with existing chunked simulation architecture
- Extend existing performance tests to validate parallel execution

**Phase 3 (Week 3-4)**: Advanced Optimizations
- Enhance existing action pruning with additional heuristics
- Implement variance reduction techniques building on current early stopping
- Use existing tournament evaluation framework for validation

**Phase 4 (Week 4-5)**: Code Quality and Modularization
- Refactor into modular structure while maintaining backward compatibility
- Extend existing comprehensive test coverage to new components
- Update documentation and ensure all existing tests continue to pass

### Configuration Parameters

Existing parameters to be preserved, with new optional parameters for enhancements:

| Parameter | Current Default | Description | Enhancement Status |
|-----------|----------------|-------------|-------------------|
| `n` | 200 | Simulations per action evaluation | Existing |
| `chunk_size` | 8 | Rollouts per batch | Existing |
| `max_rounds` | 6 | Simulation round cutoff | Existing |
| `early_stop_margin` | 0.1 | Early termination threshold | Existing |
| `simulate_to_round_end` | True | Heuristic evaluation mode | Existing |
| `prune_k` | 12 | Maximum candidate actions in select_action | Existing |
| `weighted_sampling` | False | Enable history-aware determinization | **New** |
| `enable_parallel` | False | Enable multiprocessing | **New** |
| `num_workers` | Auto | Worker process count | **New** |
| `variance_reduction` | False | Enable advanced variance reduction | **New** |

### Testing Requirements

Following TDD principles, all enhancements must include:
- Unit tests for individual component functionality
- Integration tests for agent interaction validation
- Performance benchmarks for optimization verification
- Regression tests to ensure no functionality degradation

## Expected Outcomes

### Realistic Performance Improvements

- **Speed Optimization**: Target 50%+ improvement in decision time through parallel processing
- **Enhanced Decision Quality**: Modest improvements through weighted determinization and advanced pruning
- **Better Scalability**: Support for multi-core processing with configurable worker pools
- **Maintained Superiority**: Preserve MC agent's expected performance advantage over baseline agent

### Code Quality Benefits

- **Modular Architecture**: Clean separation into focused components while maintaining backward compatibility
- **Enhanced Maintainability**: Well-documented, testable code following established project patterns
- **Comprehensive Testing**: Leverage existing extensive test infrastructure with additions for new features
- **Future-Ready Foundation**: Prepared for advanced techniques like ISMCTS and neural network integration

## Future Extension Opportunities

The enhanced modular MC Agent will facilitate:
- Information Set Monte Carlo Tree Search (ISMCTS) implementation
- Neural network policy and value function integration  
- Advanced opponent modeling and learning capabilities
- Multi-agent reinforcement learning research

## Validation Approach

Success will be measured using existing infrastructure:
- **Performance**: Leverage existing performance test suite and tournament framework
- **Quality**: Use existing comprehensive test coverage as baseline, extend to new components
- **Compatibility**: Ensure all existing tests continue to pass with new implementation
- **Benchmarking**: Utilize existing performance metrics and add new parallel processing benchmarks

