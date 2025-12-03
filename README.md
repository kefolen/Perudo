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

## Web Interface

The project includes a **Flask-based web interface** that allows users to play Perudo online against AI opponents or with friends. The web interface follows the same TDD principles as the core game engine.

### Features

**Room Management**
- Create game rooms with 4-digit codes for easy sharing
- Join existing rooms using room codes
- Support for 2-8 players per room
- Configurable AI opponents (Random, Baseline, Monte Carlo agents)

**Game Interface**
- Real-time game state updates via HTTP polling
- Interactive dice display and bidding interface
- Action buttons for bidding, calling, and exact calls
- Game history and winner announcement

**AI Integration**
- Direct integration with all existing agent types
- Configurable Monte Carlo agent parameters via JSON configuration
- Mixed human-AI games with automatic AI turn processing

### Getting Started with Web Interface

1. **Start the Flask server**:
   ```bash
   cd web/
   python app.py
   ```
   The server will run on `http://localhost:5000`

2. **Create or join a room**:
   - Visit the home page to create a new room or join an existing one
   - Configure AI opponents when creating a room
   - Share the 4-digit room code with friends

3. **Play Perudo**:
   - The web interface handles turn management and game state
   - AI opponents play automatically on their turns
   - Game follows standard Perudo rules with full support for special rules

### Web Architecture

**Backend**: Flask application (`web/app.py`) with RESTful endpoints
**Frontend**: Plain HTML/CSS/JavaScript templates with polling-based updates
**Game Logic**: `InteractivePerudoGame` wrapper around existing `PerudoSimulator`
**AI Configuration**: JSON-based Monte Carlo agent parameter configuration

### API Endpoints

- `GET /` - Home page with room creation/joining
- `POST /create_room` - Create new game room
- `POST /join_room` - Join existing room
- `GET /room/<code>` - Room lobby page
- `GET /game/<code>` - Game interface page
- `POST /start_game/<code>` - Start game in room
- `POST /action` - Submit game action
- `GET /poll/<code>` - Poll current game state


## Testing Framework
- **140+ comprehensive tests** covering unit, integration, performance, regression, and web interface testing
- **Fast execution** to support frequent TDD cycles
- **Core functionality focus** rather than exhaustive edge cases
- **Web interface tests** covering Flask routes, room management, and game integration

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v          # Unit tests
python -m pytest tests/integration/ -v   # Integration tests
python -m pytest tests/performance/ -v   # Performance tests
python -m pytest tests/regression/ -v    # Regression tests
python -m pytest tests/test_web_*.py -v  # Web interface tests

# Run with coverage reporting
python -m pytest --cov=. --cov-report=html tests/
```

