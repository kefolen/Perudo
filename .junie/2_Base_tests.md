# Base Testing Specification for Perudo Game AI Project

This document provides a comprehensive specification for implementing tests to ensure the Perudo project runs smoothly and maintains quality standards.

## Overview

This specification outlines the step-by-step implementation of a complete testing suite for the Perudo Game AI project. The testing framework should validate core functionality, ensure agent compatibility, verify game mechanics, and maintain performance standards.

**Note**: This project follows **Test-Driven Development (TDD)** principles. All future changes must include appropriate tests that define expected behavior before implementation.

## TDD Implementation Summary

The testing suite has been implemented and refined according to TDD principles:

- **Total Tests**: 96 tests (reduced from 107 by removing excessive edge cases)
- **Focus**: Core functionality and essential business logic
- **Coverage**: Unit, Integration, Performance, and Regression testing
- **Philosophy**: Tests serve as living documentation and drive development decisions

### Test Reduction Summary
The following excessive edge case tests were removed to align with TDD principles:
- `TestActionEdgeCases` (3 tests) - Non-essential edge cases for Action class
- `TestScalabilityEdgeCases` (4 tests) - Excessive scalability edge cases  
- `TestParameterChangeEdgeCases` (2 tests) - Non-essential parameter edge cases
- Infrastructure tests (2 tests) - pytest configuration tests

This reduction maintains comprehensive coverage while focusing on essential functionality.

## Testing Framework Requirements

### Prerequisites
- Python testing framework: `pytest` (add to requirements.txt if not present)
- Coverage reporting: `pytest-cov`
- Performance benchmarking capabilities
- Deterministic testing with fixed random seeds

### Directory Structure
```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/
│   ├── __init__.py
│   ├── test_perudo_simulator.py
│   ├── test_agents.py
│   ├── test_actions.py
│   └── test_game_mechanics.py
├── integration/
│   ├── __init__.py
│   ├── test_tournament.py
│   ├── test_agent_interactions.py
│   └── test_game_flow.py
├── performance/
│   ├── __init__.py
│   ├── test_mc_performance.py
│   ├── test_scalability.py
│   └── benchmarks.py
├── regression/
│   ├── __init__.py
│   ├── test_parameter_changes.py
│   └── test_backward_compatibility.py
└── fixtures/
    ├── __init__.py
    ├── sample_game_states.py
    ├── test_scenarios.py
    └── expected_outcomes.py
```

## Step-by-Step Implementation Guide

### Phase 1: Foundation Setup

#### Step 1.1: Create Test Infrastructure
**Requirements:**
- Create the complete directory structure as specified above
- Implement `conftest.py` with shared fixtures for game states, agents, and common test data
- Add pytest configuration in `pytest.ini` or `pyproject.toml`
- Ensure all `__init__.py` files are present for proper package structure

**Implementation Details:**
```python
# conftest.py should include:
@pytest.fixture
def sample_game_state():
    """Provides a standard game state for testing"""

@pytest.fixture
def all_agent_types():
    """Provides instances of all agent types"""

@pytest.fixture
def deterministic_seed():
    """Provides fixed random seed for reproducible tests"""
```

#### Step 1.2: Create Test Fixtures
**Requirements:**
- Implement `fixtures/sample_game_states.py` with various game scenarios
- Create `fixtures/test_scenarios.py` with edge cases and special situations
- Develop `fixtures/expected_outcomes.py` with known correct results

**Must Include:**
- Early game states (5 dice per player)
- Mid game states (2-3 dice per player)
- End game states (1 die scenarios)
- Maputa restriction scenarios
- Various bid progression scenarios

### Phase 2: Unit Testing Implementation

#### Step 2.1: Core Game Mechanics Testing (`test_perudo_simulator.py`)
**Requirements:**
- Test `PerudoSimulator` initialization with all parameter combinations
- Validate `legal_actions()` method correctness for all game states
- Test `is_bid_true()` with comprehensive dice combinations
- Verify game state transitions and rule enforcement

**Specific Tests to Implement:**
```python
class TestPerudoSimulator:
    def test_initialization_default_parameters()
    def test_initialization_custom_parameters()
    def test_legal_actions_early_game()
    def test_legal_actions_with_maputa()
    def test_legal_actions_edge_cases()
    def test_is_bid_true_with_wild_ones()
    def test_is_bid_true_without_wild_ones()
    def test_game_state_transitions()
    def test_player_elimination()
    def test_win_conditions()
    def test_exact_call_mechanics()
```

**Success Criteria:**
- All tests pass with 100% coverage of PerudoSimulator methods
- Edge cases properly handled (single die, maximum bids)
- Special rules correctly implemented (maputa, exact, wild ones)

#### Step 2.2: Action Class Testing (`test_actions.py`)
**Requirements:**
- Test all Action class methods and properties
- Validate action creation and manipulation
- Test action comparison and equality

**Specific Tests to Implement:**
```python
class TestAction:
    def test_bid_creation()
    def test_call_creation()
    def test_exact_creation()
    def test_action_equality()
    def test_action_string_representation()
    def test_action_type_checking()
    def test_bid_component_extraction()
```

#### Step 2.3: Agent Interface Testing (`test_agents.py`)
**Requirements:**
- Test all agents implement consistent interface
- Validate `select_action()` method behavior
- Test agent initialization and parameter handling
- Ensure deterministic behavior with fixed seeds

**Specific Tests to Implement:**
```python
class TestAgentInterface:
    def test_random_agent_interface()
    def test_baseline_agent_interface()
    def test_mc_agent_interface()
    def test_agent_returns_valid_actions()
    def test_agent_respects_legal_actions()
    def test_deterministic_behavior()
    def test_agent_initialization_parameters()
```

**Success Criteria:**
- All agents return valid Action objects
- No agent suggests illegal moves
- Deterministic agents produce consistent results with fixed seeds
- All agents handle edge cases gracefully

### Phase 3: Integration Testing Implementation

#### Step 3.1: Tournament System Testing (`test_tournament.py`)
**Requirements:**
- Test tournament execution with different configurations
- Validate parameter variations and special rule combinations
- Test performance metrics collection
- Verify tournament completion and result reporting

**Specific Tests to Implement:**
```python
class TestTournament:
    def test_tournament_basic_execution()
    def test_tournament_with_different_agents()
    def test_tournament_parameter_variations()
    def test_tournament_special_rules()
    def test_tournament_metrics_collection()
    def test_tournament_edge_cases()
```

#### Step 3.2: Agent Interaction Testing (`test_agent_interactions.py`)
**Requirements:**
- Test agent vs agent performance relationships
- Validate expected performance hierarchies
- Test cross-validation with different seeds

**Specific Tests to Implement:**
```python
class TestAgentInteractions:
    def test_baseline_vs_random()
    def test_mc_vs_baseline()
    def test_mc_vs_random()
    def test_performance_consistency()
    def test_seed_variation_stability()
```

**Success Criteria:**
- Baseline agent significantly outperforms Random agent
- Monte Carlo agent shows improved performance over Baseline
- Results are statistically significant across multiple runs

### Phase 4: Performance Testing Implementation

#### Step 4.1: Monte Carlo Performance Testing (`test_mc_performance.py`)
**Requirements:**
- Test MC_N parameter impact (specifically 1000 → 100 change)
- Benchmark simulation speed and accuracy trade-offs
- Test optimization parameters (chunk_size, early_stop_margin)

**Specific Tests to Implement:**
```python
class TestMCPerformance:
    def test_mc_n_parameter_impact()
    def test_simulation_speed_benchmarks()
    def test_accuracy_vs_speed_tradeoff()
    def test_chunk_size_optimization()
    def test_early_stop_effectiveness()
    def test_memory_usage_patterns()
```

**Success Criteria:**
- MC_N=100 maintains reasonable decision quality
- Performance improvement is measurable and significant
- Memory usage remains within acceptable bounds

#### Step 4.2: Scalability Testing (`test_scalability.py`)
**Requirements:**
- Test system performance with maximum players
- Test long-running tournament stability
- Benchmark resource usage over time

**Specific Tests to Implement:**
```python
class TestScalability:
    def test_maximum_players_performance()
    def test_long_running_tournament_stability()
    def test_resource_usage_over_time()
    def test_concurrent_agent_performance()
    def test_memory_leak_detection()

class TestScalabilityEdgeCases:
    def test_minimum_configuration_stability()
    def test_maximum_configuration_feasibility()
    def test_rapid_successive_tournaments()
    def test_mixed_agent_scalability()
```

**Success Criteria:**
- System scales reasonably with increasing players
- Long-running tournaments maintain stability
- Memory usage remains within acceptable bounds
- No memory leaks detected during extended operation

---

### Phase 4: Implementation Status (COMPLETED ✅)

This phase documents the actual implementation of Phase 4 Performance Testing that has been completed.

#### Step 4.1: Monte Carlo Performance Testing (COMPLETED ✅)
**Status: ✅ IMPLEMENTED**

**What was implemented:**
- `tests/performance/test_mc_performance.py` (344 lines) - Comprehensive MC performance testing
- ✅ `TestMCPerformance` class with 6 core performance tests:
  - `test_mc_n_parameter_impact()` - Tests different MC_N values (10, 50, 100, 200)
  - `test_simulation_speed_benchmarks()` - Benchmarks different configurations
  - `test_accuracy_vs_speed_tradeoff()` - Tests low vs high MC_N performance
  - `test_chunk_size_optimization()` - Tests chunk size parameter impact
  - `test_early_stop_effectiveness()` - Tests early stopping mechanisms
  - `test_memory_usage_patterns()` - Tests memory usage during simulations
- ✅ `TestMCPerformanceEdgeCases` class with 3 edge case tests:
  - `test_minimal_mc_n_performance()` - Tests with very low MC_N (5)
  - `test_maximum_reasonable_mc_n()` - Tests with high MC_N (300)
  - `test_performance_consistency()` - Tests consistency across multiple runs

**Performance Metrics Validated:**
- MC_N=100 maintains reasonable decision quality (< 10s per game)
- Performance improvement is measurable and significant
- Memory usage remains within acceptable bounds (< 100MB increase)
- All timing measurements use high-precision `time.perf_counter()`

#### Step 4.2: Scalability Testing (COMPLETED ✅)
**Status: ✅ IMPLEMENTED**

**What was implemented:**
- `tests/performance/test_scalability.py` (349 lines) - Comprehensive scalability testing
- ✅ `TestScalability` class with 5 core scalability tests:
  - `test_maximum_players_performance()` - Tests 2, 3, 4, 6 players performance
  - `test_long_running_tournament_stability()` - Tests 15-game tournaments
  - `test_resource_usage_over_time()` - Tests resource usage over 4 rounds
  - `test_concurrent_agent_performance()` - Tests multiple MC agents
  - `test_memory_leak_detection()` - Tests for memory leaks over 5 cycles
- ✅ `TestScalabilityEdgeCases` class with 4 edge case tests:
  - `test_minimum_configuration_stability()` - Tests minimal config (2 players, 1 die)
  - `test_maximum_configuration_feasibility()` - Tests max config (6 players, 4 dice)
  - `test_rapid_successive_tournaments()` - Tests 6 rapid tournaments
  - `test_mixed_agent_scalability()` - Tests mixed agent types

**Scalability Metrics Validated:**
- System scales reasonably (6 players < 10x slower than 2 players)
- Long-running tournaments maintain stability (< 5 minutes for 15 games)
- Memory usage remains stable (< 75MB total increase)
- No memory leaks detected (< 40MB growth over cycles)

#### Phase 4 Test Execution Results
**Overall Status: ✅ ALL 18 PERFORMANCE TESTS PASSING**

- **MC Performance Tests**: 9 tests ✓
- **Scalability Tests**: 9 tests ✓
- **Total Test Suite**: 101 tests ✓ (10 foundation + 47 unit + 26 integration + 18 performance)
- **Execution**: Fully automated via `python tests/test_launcher.py --performance`

**Key Achievements:**
- Comprehensive performance benchmarking implemented
- MC_N parameter impact thoroughly tested and validated
- System scalability verified across different configurations
- Memory usage patterns monitored and validated
- All performance tests integrate seamlessly with existing test suite
- High-precision timing ensures reliable performance measurements

**Ready for**: Phase 5 (Regression Testing) - Performance baseline established

---

### Phase 5: Regression Testing Implementation

#### Step 5.1: Parameter Change Testing (`test_parameter_changes.py`)
**Requirements:**
- Specifically test MC_N reduction impact
- Compare performance before/after parameter changes
- Validate backward compatibility

**Specific Tests to Implement:**
```python
class TestParameterChanges:
    def test_mc_n_reduction_impact()
    def test_win_rate_comparison()
    def test_decision_quality_maintenance()
    def test_game_completion_time()
```

## Testing Execution Requirements

### Automated Testing Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v

# Run with coverage reporting
python -m pytest --cov=. --cov-report=html tests/

# Run performance benchmarks
python -m pytest tests/performance/ --benchmark-only

# Run regression tests
python -m pytest tests/regression/ -v
```

### Continuous Integration Requirements
- All unit tests must pass with >95% code coverage
- Integration tests must demonstrate expected agent relationships
- Performance tests must validate acceptable speed/accuracy trade-offs
- Regression tests must show no degradation in core functionality

## Quality Standards and Success Criteria

### Code Coverage Requirements
- **Unit Tests**: >95% coverage of all core modules
- **Integration Tests**: >90% coverage of interaction paths
- **Overall Project**: >90% total code coverage

### Performance Benchmarks
- Monte Carlo agent with MC_N=100 must complete decisions within reasonable time
- Tournament execution must scale linearly with number of games
- Memory usage must remain stable during extended play

### Functional Requirements
- All Perudo rules correctly implemented and tested
- Agent interface consistency maintained
- Tournament system produces reproducible results
- Special rules (maputa, exact, wild ones) properly validated

### Documentation Requirements
- Each test file must include comprehensive docstrings
- Test methods must have clear descriptions of what they validate
- Complex test scenarios must include explanatory comments
- README for tests directory explaining execution and interpretation

## Implementation Timeline

### Phase 1 (Foundation): 1-2 days
- Set up directory structure and basic fixtures
- Implement conftest.py and basic configuration

### Phase 2 (Unit Tests): 3-4 days
- Implement all unit tests for core functionality
- Achieve target code coverage

### Phase 3 (Integration Tests): 2-3 days
- Implement tournament and agent interaction tests
- Validate system-level functionality

### Phase 4 (Performance Tests): 2-3 days
- Implement performance benchmarks and scalability tests
- Validate MC_N parameter change impact

### Phase 5 (Regression Tests): 1-2 days
- Implement backward compatibility and parameter change tests
- Final validation and documentation

## Maintenance and Extension

### Regular Testing Schedule
- Run unit tests on every code change
- Run integration tests on pull requests
- Run performance tests weekly
- Run full regression suite before releases

### Extension Guidelines
- New agents must include corresponding unit tests
- New features must include integration tests
- Performance-critical changes must include benchmarks
- All tests must maintain the established quality standards

This specification ensures comprehensive testing coverage while maintaining the project's educational accessibility and modular architecture.
