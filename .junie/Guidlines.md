# Junie Configuration for Perudo Game AI Project

## Overview
You are one of developers of this projects. Strictly follow guidelines provided in this file.

## Development Philosophy: Test-Driven Development (TDD)

This project follows **Test-Driven Development (TDD)** principles as its core development philosophy. All future changes and features must adhere to the TDD cycle:

### TDD Cycle
1. **Red**: Write a failing test that describes the desired functionality
2. **Green**: Write the minimal code necessary to make the test pass
3. **Refactor**: Improve the code while keeping tests passing

### TDD Requirements for All Changes
- **New Features**: Must be preceded by comprehensive tests that define the expected behavior
- **Bug Fixes**: Must include regression tests that reproduce the bug before fixing it
- **Refactoring**: Must maintain all existing tests while improving code structure
- **Performance Optimizations**: Must include performance tests that validate improvements

### Testing Standards
- Focus on **essential functionality** rather than exhaustive edge cases
- Prioritize **core business logic** tests over infrastructure tests
- Maintain **fast test execution** to support frequent TDD cycles
- Write **clear, readable tests** that serve as living documentation

## Project Context

This is a Perudo (Liar's Dice) game project focused on developing and evaluating AI algorithms for playing Perudo, a dice game of strategy, bluffing, and luck. The project provides a platform for testing different strategies against each other and aims to create a framework for online play.

### Game Overview
- Perudo is a dice game where players make bids about the total number of dice showing specific face values
- Players start with 5 dice, roll secretly, and make increasingly higher bids
- Game involves probability assessment, bluffing, and strategic decision-making
- Special rules include: ones are wild, maputa (single die rules), and exact calls

### Project Structure
- **agents/**: Different AI agent implementations (Random, Baseline, Monte Carlo)
- **sim/**: Core game mechanics and simulation (PerudoSimulator, Action classes)
- **eval/**: Tournament system for agent evaluation
- Primary focus on Monte Carlo-based agents with intelligent decision-making

## Code Standards and Style Guidelines

### PEP 8 Compliance
- Follow PEP 8 style guidelines for Python code
- Use snake_case for variable and function names
- Use PascalCase for class names
- Line length should be reasonable (prefer readability over strict 79-character limit)

### Existing Codestyle Patterns (maintain consistency)
- Clean class structure with descriptive method names
- Use clear, descriptive variable names (e.g., `threshold_call`, `dice_counts`, `current_bid`)
- Proper import organization at the top of files
- Snake_case naming conventions throughout
- Clear separation of concerns between classes and methods
- Use existing functions and methods if possible

### Documentation Standards
- Use inline comments to explain complex probability calculations and game logic
- Maintain clear README and technical documentation (update after changes if needed)

### Code Organization
- Keep agent implementations in separate files within agents/ directory
- Use modular design allowing easy extension of new agents
- Implement consistent interface patterns (e.g., `select_action(obs)` method for all agents)
- Maintain clear separation between game simulation and agent logic

## Domain-Specific Knowledge

### Perudo Game Mechanics
- Understand probability calculations for dice combinations
- Consider bluffing and strategic elements in AI decision-making
- Account for special rules: ones are wild, maputa restrictions, exact calls
- Game state includes: dice counts, current bid, player hands, turn order

### AI Agent Types
- **RandomAgent**: Baseline for comparison, makes random legal moves
- **BaselineAgent**: Uses simple probability-based heuristics with threshold calling
- **MonteCarloAgent**: Sophisticated simulation-based decision making with optimization techniques

### Performance Considerations
- Monte Carlo agents are computationally intensive
- Consider chunked simulations, early stopping, and pruning for optimization
- Modular design allows for advanced techniques (ISMCTS, opponent modeling, neural networks)
- Single-machine friendly design with potential for multiprocessing parallelization

## Response Preferences

### Code Suggestions
- Maintain backward compatibility with existing agent interfaces
- Prioritize code readability for educational and research purposes
- Include performance implications of any suggested modifications
- Consider the modular architecture when proposing extensions

## Constraints and Guidelines

- Preserve the existing modular agent architecture
- Maintain compatibility with the tournament evaluation system
- Consider memory and computational constraints for real-time play
- Keep the codebase accessible for educational and research purposes
- Follow the established patterns for game state observation and action selection
