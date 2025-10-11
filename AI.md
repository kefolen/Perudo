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
   - The sophisticated `MonteCarloAgent` in `agents/mc_agent.py`

3. **Evaluation System**: The tournament system in `eval/tournament.py` for evaluating agent performance.

4. **Documentation**: 
   - The comprehensive `README.md` explaining the rules of Perudo and project goals
   - The technical `documentation.md` detailing the project structure and implementation

## Results in the Repository

The results of Junie's work can be found throughout the repository:

- **Code Files**: All the Python files in the `sim/`, `agents/`, and `eval/` directories
- **Documentation**: The `README.md` and `documentation.md` files

