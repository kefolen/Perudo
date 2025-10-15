# AI Tools Used in This Project

This document describes the AI tools used in the development of the Perudo project, what tasks they performed, and where to find their results in the repository.

## Project Overview

This is an experimental project that was made to test the capabilities of Junie - JetBrains coding agent. Most of the coding tasks were performed by Junie, with puny human developers only doing minor refactoring and careful prompting (which not as easy as it sounds).

## AI Tools Used

### Junie (JetBrains Coding Agent)

Junie is an AI coding assistant developed by JetBrains. It was the primary AI tool used in this project.

## Tasks Performed by AI

Junie was used for the following tasks:

1. **Game Logic Implementation**: The core game mechanics in `sim/perudo.py` were implemented by Junie, including the `PerudoSimulator` class and `Action` class.

2. **Agent Development**: 
   - The `RandomAgent` in `agents/random_agent.py`
   - The `BaselineAgent` in `agents/baseline_agent.py`
   - The sophisticated `MonteCarloAgent` with modular architecture:
     - Core agent class in `agents/mc_agent.py`
     - Utility functions in `agents/mc_utils.py`
     - Parallel processing components in `agents/mc_parallel.py`

3. **Evaluation System**: The tournament system in `eval/tournament.py` for evaluating agent performance.

4. **Documentation**: 
   - The comprehensive `README.md` explaining the rules of Perudo and project goals
   - The technical `documentation.md` detailing the project structure and implementation

5. **Monte Carlo Agent Enhancement (4-Phase Implementation)**:
   - **Phase 1**: Weighted determinization with history-aware sampling
   - **Phase 2**: Parallel processing support with configurable worker pools
   - **Phase 3**: Advanced optimizations (enhanced pruning, variance reduction)
   - **Phase 4**: Code modularization and architectural improvements

## Monte Carlo Agent Enhancement Phases

### Phase 1: Weighted Determinization
**Objective**: Improve opponent hand estimation through history-aware sampling
**Implementation**: Enhanced the uniform dice sampling with probabilistic weighting based on bidding history plausibility
**Key Features**:
- Weighted dice generation based on current bid context
- Improved simulation accuracy for strategic decision-making
- Backward compatibility with existing uniform sampling
- Comprehensive test coverage with 8 unit tests

### Phase 2: Parallel Processing
**Objective**: Add multiprocessing support for faster simulations
**Implementation**: Optional parallel evaluation using worker pools while maintaining statistical equivalence
**Key Features**:
- Configurable worker pools with automatic core detection
- Reproducible results with proper seed management
- Graceful fallback to sequential processing
- Memory-efficient implementation with timeout handling
- 7 specialized tests covering parallel functionality

### Phase 3: Advanced Optimizations
**Objective**: Enhance decision quality through sophisticated algorithms
**Implementation**: Added advanced pruning and variance reduction techniques
**Key Features**:
- Enhanced action pruning with opponent modeling
- Multi-criteria scoring considering game state dynamics
- Control variate variance reduction methods
- Strategic face transition analysis
- 18 comprehensive tests covering optimization features

### Phase 4: Code Modularization
**Objective**: Improve maintainability and extensibility through architectural refactoring
**Implementation**: Split monolithic agent into focused modules while preserving functionality
**Key Architecture**:
- `mc_agent.py`: Core MonteCarloAgent class and simulation logic
- `mc_utils.py`: Utility functions for determinization, scoring, and heuristics
- `mc_parallel.py`: Parallel processing mixin and worker management
- Full backward compatibility maintained
- All 48 existing tests continue to pass

### Overall Impact
The Monte Carlo agent has evolved from a basic simulation-based approach to a sophisticated AI system featuring:
- **4x performance improvement potential** through parallel processing
- **Enhanced decision quality** via weighted sampling and advanced scoring
- **Reduced variance** in simulation results through control variates
- **Modular architecture** ready for future enhancements (ISMCTS, neural networks)
- **132 comprehensive tests** ensuring reliability and maintainability

## Results in the Repository

The results of Junie's work can be found throughout the repository:

- **Core Game Logic**: `sim/perudo.py` - Complete Perudo game implementation
- **Agent Implementations**: 
  - `agents/random_agent.py` - Random baseline agent
  - `agents/baseline_agent.py` - Probabilistic strategy agent
  - `agents/mc_agent.py` - Advanced Monte Carlo agent (core)
  - `agents/mc_utils.py` - Monte Carlo utility functions
  - `agents/mc_parallel.py` - Parallel processing components
- **Evaluation Framework**: `eval/tournament.py` - Agent performance evaluation
- **Testing Infrastructure**: 
  - 132 comprehensive tests across `tests/` directory
  - Unit tests (80 tests): Core functionality, agent behavior, Monte Carlo enhancements
  - Integration tests (15 tests): Agent interactions and tournament system
  - Performance tests (24 tests): Speed benchmarks and scalability validation
  - Regression tests (3 tests): Parameter change impact validation
  - Foundation tests (8 tests): Basic setup and fixture validation
  - TDD-driven development ensuring code quality and reliability
- **Documentation**: Comprehensive `README.md` with usage examples and technical details

