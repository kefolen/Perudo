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

The project includes several agent implementations:
- **Random Agent**: Makes random legal moves
- **Baseline Agent**: Uses simple probability-based strategy
- **Monte Carlo Agent**: Uses Monte Carlo simulation to evaluate possible actions

## Development Philosophy

This project follows **Test-Driven Development (TDD)** principles:

### TDD Approach
- **Red-Green-Refactor**: Write failing tests first, implement minimal code to pass, then refactor
- **Living Documentation**: Tests serve as executable specifications of expected behavior
- **Quality Assurance**: Comprehensive test coverage ensures reliability and maintainability

### Testing Framework
- **96 comprehensive tests** covering unit, integration, performance, and regression testing
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

## Performance Notes

- The Monte-Carlo agent is single-machine friendly; for speed, you can parallelize evaluate_action in mc_agent using multiprocessing.
- The modular layout allows for plugging in ISMCTS, opponent modeling, or NN-based rollout policies.
