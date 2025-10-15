# Perudo Game AI

## What is Perudo?

Perudo (also known as Liar's Dice) is a dice game of strategy, bluffing, and luck. Players roll dice secretly and make increasingly higher bids about the total number of dice showing a particular face value among all players. The game combines probability assessment, bluffing, and strategic decision-making.

## Game Rules

### Basic Rules
- Each player starts with 5 dice
- Players roll their dice secretly and keep them hidden from other players
- Players take turns making bids about the total number of dice with a specific face value across all players' hands
- A bid consists of a quantity and a face value (e.g., "three 4s")
- Each new bid must be higher than the previous (either higher quantity or same quantity but higher face)
- Players can "call" (challenge) the previous bid, claiming it's false
- If a bid is challenged:
  - If the bid is true, the challenger loses a die
  - If the bid is false, the bidder loses a die
- Players with no dice are eliminated
- The last player with dice remaining wins

### Special Rules
- **Ones are wild**: Dice showing 1 count as any face value (unless bidding on 1s). 
- **Maputa**: When a player has only one die left, special rules apply for their turn
- **Exact**: Players can call "exact" if they believe the bid is exactly correct
  - If the bid is exactly true, the challenger steals a die from a bidder (players can have more than five dice as a result of this action)
  - Otherwise, the challenger loses a die

## Project Goals

This project aims to:

1. Develop and evaluate AI algorithms for playing Perudo
2. Provide a platform for testing different strategies against each other
3. Create a framework for online play against bots or friends

The primary focus is on developing a Monte Carlo-based agent that can make intelligent decisions in the game, considering both probability and strategic elements.

## Getting Started

1. Clone the repository
2. Create a virtualenv and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run tournaments to evaluate agents:
   ```
   python eval/tournament.py --agent1 baseline --agent2 mc --games 200 --mc-n 300 --maputa --exact
   ```

## Agent Types

The project includes several sophisticated agent implementations:

### Random Agent
- **Purpose**: Baseline for comparison and testing
- **Strategy**: Makes uniformly random legal moves
- **Implementation**: `agents/random_agent.py`

### Baseline Agent
- **Purpose**: Simple probabilistic strategy reference
- **Strategy**: Uses binomial probability calculations with configurable call threshold
- **Implementation**: `agents/baseline_agent.py`
- **Features**: Fast decision-making, probability-based bidding

### Monte Carlo Agent (Advanced)
- **Purpose**: Sophisticated AI using simulation-based decision making
- **Implementation**: Modular architecture across multiple files:
  - `agents/mc_agent.py`: Core agent class and game simulation
  - `agents/mc_utils.py`: Utility functions for determinization and scoring  
  - `agents/mc_parallel.py`: Parallel processing components

#### Monte Carlo Agent Features

**Phase 1: Weighted Determinization**
- History-aware probabilistic dice sampling
- Improved opponent hand estimation based on bidding patterns
- Configurable via `weighted_sampling=True` parameter

**Phase 2: Parallel Processing**
- Optional multiprocessing support for faster simulations
- Configurable worker pools with automatic core detection
- Maintains statistical equivalence with sequential mode
- Enable with `enable_parallel=True, num_workers=N`

**Phase 3: Advanced Optimizations**
- Enhanced action pruning with opponent modeling
- Multi-criteria bid scoring considering game state
- Variance reduction using control variates
- Configurable via `enhanced_pruning=True, variance_reduction=True`

**Phase 4: Code Modularization**
- Clean separation of concerns across three modules
- Improved maintainability and extensibility
- Full backward compatibility preserved
- Prepared for advanced techniques (ISMCTS, neural networks)

**Phase 5: Betting History and Trust Management**
- Comprehensive betting history tracking throughout games
- Dynamic player trust parameters based on bidding accuracy
- Bayesian player modeling for improved opponent hand sampling
- Collective plausibility analysis using historical betting patterns
- Configurable via `betting_history_enabled=True, bayesian_sampling=True`

#### Monte Carlo Agent Parameters
```python
MonteCarloAgent(
    name='mc',                    # Agent identifier
    n=200,                        # Simulations per action evaluation
    chunk_size=8,                 # Simulations per batch
    max_rounds=6,                 # Maximum simulation rounds
    early_stop_margin=0.1,        # Early termination threshold
    simulate_to_round_end=True,   # Use heuristic evaluation
    weighted_sampling=False,      # Enable history-aware sampling
    enable_parallel=False,        # Enable multiprocessing
    num_workers=None,             # Worker count (auto-detect if None)
    enhanced_pruning=False,       # Enable advanced action filtering
    variance_reduction=False,     # Enable control variate techniques
    betting_history_enabled=False,# Enable betting history tracking
    player_trust_enabled=False,   # Enable dynamic trust parameters
    trust_learning_rate=0.1,      # Trust parameter adaptation rate
    history_memory_rounds=10,     # Rounds of history to remember
    bayesian_sampling=False       # Enable Bayesian player modeling
)
```

## Development Philosophy

This project follows **Test-Driven Development (TDD)** principles:

### TDD Approach
- **Red-Green-Refactor**: Write failing tests first, implement minimal code to pass, then refactor
- **Living Documentation**: Tests serve as executable specifications of expected behavior
- **Quality Assurance**: Comprehensive test coverage ensures reliability and maintainability

### Testing Framework
- **132 comprehensive tests** covering unit, integration, performance, and regression testing
- **Fast execution** to support frequent TDD cycles
- **Core functionality focus** rather than exhaustive edge cases

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v          # Unit tests
python -m pytest tests/integration/ -v   # Integration tests
python -m pytest tests/performance/ -v   # Performance tests
python -m pytest tests/regression/ -v    # Regression tests

# Run with coverage reporting
python -m pytest --cov=. --cov-report=html tests/
```

### Contributing
All new features and changes must follow TDD principles:
1. Write tests that define expected behavior
2. Implement minimal code to make tests pass
3. Refactor while maintaining test coverage

### Pre-Commit Hooks
This project uses pre-commit hooks to automatically run tests before each commit, ensuring code quality and preventing regressions:

#### Setup Pre-Commit Hooks
```bash
# Install pre-commit (included in requirements.txt)
pip install pre-commit

# Install the hooks
pre-commit install
```

#### How It Works
- **Automatic Testing**: All 132 tests run automatically before each commit
- **Commit Prevention**: If tests fail, the commit is blocked until issues are resolved
- **TDD Enforcement**: Ensures all commits maintain the established quality standards

#### Manual Testing
```bash
# Run pre-commit hooks manually on all files
pre-commit run --all-files

# Run tests directly using the test launcher
python tests/tests_launcher.py
```

#### Bypassing Hooks (Use Sparingly)
```bash
# Skip pre-commit hooks for emergency commits
git commit --no-verify -m "Emergency fix"
```

## Performance Notes

- The Monte-Carlo agent is single-machine friendly; for speed, you can parallelize evaluate_action in mc_agent using multiprocessing.
- The modular layout allows for plugging in ISMCTS, opponent modeling, or NN-based rollout policies.
