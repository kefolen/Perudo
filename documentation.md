# Perudo Project Technical Documentation

This document provides technical details about the Perudo project structure, implementation, and algorithms.

## Project Structure

The project is organized into the following directories and files:

- **agents/**: Contains different agent implementations
  - `__init__.py`: Package initialization
  - `baseline_agent.py`: Simple probability-based agent
  - `mc_agent.py`: Monte Carlo simulation agent
  - `random_agent.py`: Random action agent
- **eval/**: Evaluation tools
  - `tournament.py`: Script for running tournaments between agents
- **sim/**: Game simulation
  - `perudo.py`: Core game mechanics and simulation
- **requirements.txt**: Project dependencies
- **README.md**: Project overview and setup instructions

## Game Simulation (sim/perudo.py)

### PerudoSimulator

The `PerudoSimulator` class implements the core game mechanics:

- **Initialization**: Configure game parameters like number of players, starting dice, and special rules
- **Game State**: Tracks dice counts, hands, current bid, and player turns
- **Legal Actions**: Determines valid actions (bids, calls, exact) for a given state
- **Game Logic**: Implements the rules for bidding, challenging, and determining winners

Key methods:
- `play_game(agents)`: Runs a complete game with the provided agents
- `legal_actions(state, current_bid, maputa_restrict_face)`: Returns all legal actions for the current state
- `is_bid_true(hands, bid, ones_are_wild)`: Checks if a bid is true given the current hands

### Action Class

The `Action` class provides a structured way to represent and manipulate game actions:
- `bid(qty, face)`: Create a bid action
- `call()`: Create a call action
- `exact()`: Create an exact action
- Helper methods for checking action types and extracting components

## Agent Implementations

### RandomAgent (agents/random_agent.py)

A simple agent that randomly selects from legal actions:
- Initializes with a name and optional random seed
- `select_action(obs)`: Randomly chooses a legal action

### BaselineAgent (agents/baseline_agent.py)

A probability-based agent that makes decisions using simple heuristics:
- Initializes with a name and threshold for calling
- `select_action(obs)`: 
  - If no current bid, bids on the most common face in its hand
  - Otherwise, calculates the probability of the current bid being true
  - Calls if probability is below threshold, otherwise makes minimal raise

### MonteCarloAgent (agents/mc_agent.py)

A sophisticated agent that uses Monte Carlo simulation to evaluate actions:

- **Initialization Parameters**:
  - `n`: Number of simulations to run per action evaluation
  - `chunk_size`: Number of simulations to run in each chunk (for efficiency)
  - `max_rounds`: Maximum number of rounds to simulate
  - `simulate_to_round_end`: Whether to simulate only to the end of the current round
  - `early_stop_margin`: Margin for early stopping of action evaluation

- **Key Methods**:
  - `sample_determinization(obs)`: Samples possible game states based on known information
  - `evaluate_action(obs, action, best_so_far)`: Evaluates an action by running simulations
  - `select_action(obs, prune_k)`: Selects the best action using pruning and evaluation
  - `simulate_from_determinization(full_hands, obs, first_action)`: Simulates a game from a given state

- **Optimization Techniques**:
  - Chunked simulations to reduce overhead
  - Early stopping to avoid evaluating clearly inferior actions
  - Pruning of action space using a fast prior
  - Caching of rollout agents

## Evaluation (eval/tournament.py)

The tournament system allows for systematic evaluation of agents:

- **Configuration**: Set number of players, games, and agent types
- **Agent Creation**: Create agents with specified parameters
- **Match Play**: Run multiple games and collect results
- **Performance Metrics**: Track win rates and timing statistics

## Extending the Project

The modular design allows for easy extension:

1. **New Agents**: Create a new class in the agents directory with a `select_action(obs)` method
2. **Alternative Evaluation**: Modify tournament.py or create new evaluation scripts
3. **Advanced Techniques**: Implement ISMCTS, opponent modeling, or neural network policies

## Performance Considerations

- The Monte Carlo agent is computationally intensive but can be optimized:
  - Parallelize `evaluate_action` using multiprocessing
  - Adjust `chunk_size`, `max_rounds`, and `early_stop_margin` for better performance
  - Use more sophisticated pruning strategies