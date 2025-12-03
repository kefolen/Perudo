# AI Tools Used in This Project

This document describes the AI tools used in the development of the Perudo project, what tasks they performed, and where to find their results in the repository.

TL;DR: This is an experimental project that was made to test the capabilities of Junie. All the coding tasks performed by Junie, with a puny human developer only doing minor refactoring and careful prompting (which not as easy as it sounds).

## AI Tools Used

Junie - JetBrains coding agent

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

### Phase 5: Betting History and Trust Management
**Objective**: Implement sophisticated opponent modeling through historical betting analysis
**Implementation**: Comprehensive betting history tracking with dynamic trust parameters and Bayesian player modeling
**Key Features**:
- **Betting History Tracking**: Complete game history with `BettingHistoryEntry` and `GameBettingHistory` classes
- **Dynamic Player Trust**: Trust parameters that adapt based on bidding accuracy using `PlayerTrustManager`
- **Bayesian Player Modeling**: Individual player dice sampling using `sample_bayesian_player_dice`
- **Collective Plausibility**: Enhanced bid evaluation using `compute_collective_plausibility` with historical patterns
- **Enhanced Simulator**: Extended observation structure with betting history and trust data
- **Backward Compatibility**: All new features are opt-in with sensible defaults
- **Comprehensive Testing**: 29 new tests (18 unit + 11 integration) covering all betting history functionality

**Technical Components**:
- `BettingHistoryEntry`: Individual betting actions with context and results
- `GameBettingHistory`: Complete game history with analysis capabilities (face popularity, player accuracy)
- `PlayerTrustManager`: Dynamic trust parameter management with learning rate adaptation
- `calculate_recency_weight`: Temporal weighting for historical data importance
- `sample_bayesian_player_dice`: History-aware dice sampling with trust-weighted probabilities
- `compute_collective_plausibility`: Multi-factor plausibility analysis using betting patterns

### Overall Impact
The Monte Carlo agent has evolved from a basic simulation-based approach to a sophisticated AI system featuring:
- **4x performance improvement potential** through parallel processing
- **Enhanced decision quality** via weighted sampling, advanced scoring, and opponent modeling
- **Sophisticated betting analysis** through historical pattern recognition and trust management
- **Bayesian player modeling** for improved opponent hand estimation
- **Reduced variance** in simulation results through control variates
- **Modular architecture** ready for future enhancements (ISMCTS, neural networks)
- **161 comprehensive tests** ensuring reliability and maintainability (29 new tests for Phase 5)

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

