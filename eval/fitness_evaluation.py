"""
Fitness Evaluation Module for Monte Carlo Agent Parameter Optimization

This module provides robust tournament-based fitness evaluation for parameter
configurations, integrating with the existing tournament system and providing
variance-penalized fitness scoring with robustness bonuses.

Components:
- FitnessEvaluator: Main class for evaluating parameter configurations
- Fitness calculation functions: win rate, variance penalty, robustness bonus
- Tournament integration with existing infrastructure
- Champion agent management for evolutionary pressure
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, List, Any, Union, Optional
from collections import Counter
import numpy as np

from sim.perudo import PerudoSimulator
from eval.tournament import play_match
from agents.optimizable_mc_agent import OptimizableMCAgent
from agents.baseline_agent import BaselineAgent
from agents.random_agent import RandomAgent
from eval.hyperopt_framework import ParameterSpace


# Configure logging
logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """
    Robust fitness evaluator for Monte Carlo agent parameter configurations.
    
    Evaluates parameter configurations by running tournaments against diverse
    opponents and calculating variance-penalized fitness scores.
    """
    
    def __init__(self, parameter_space: ParameterSpace, 
                 opponent_config: Dict[str, int] = None,
                 evaluation_games: int = 50,
                 champion_storage_path: str = "eval/results/champions"):
        """
        Initialize fitness evaluator.
        
        Args:
            parameter_space: ParameterSpace for parameter validation
            opponent_config: Configuration for opponent mix
            evaluation_games: Number of games per evaluation
            champion_storage_path: Path to store/load champion agents
        """
        self.parameter_space = parameter_space
        self.evaluation_games = evaluation_games
        self.champion_storage_path = champion_storage_path
        
        # Default opponent configuration
        self.opponent_config = opponent_config or {
            'baseline_count': 2,
            'random_count': 1,
            'champion_count': 1
        }
        
        # Ensure champion storage directory exists
        os.makedirs(champion_storage_path, exist_ok=True)
        
        # Cache for loaded champions
        self._champion_cache = {}
    
    def create_opponent_mix(self) -> List:
        """
        Create diverse opponent mix for robust evaluation.
        
        Returns:
            List of agent instances for tournament
        """
        opponents = []
        
        # Add baseline agents
        for i in range(self.opponent_config.get('baseline_count', 2)):
            opponents.append(BaselineAgent(name=f"baseline_{i}"))
        
        # Add random agents
        for i in range(self.opponent_config.get('random_count', 1)):
            opponents.append(RandomAgent(name=f"random_{i}"))
        
        # Add champion agents from previous generations
        champion_count = self.opponent_config.get('champion_count', 1)
        for i in range(champion_count):
            champion = self.load_champion_agent(f"champion_{i}")
            if champion is not None:
                opponents.append(champion)
        
        return opponents
    
    def create_optimizable_agent(self, params: Dict[str, Union[int, float]]) -> OptimizableMCAgent:
        """
        Create optimizable MC agent with given parameters.
        
        Args:
            params: Parameter configuration dictionary
            
        Returns:
            OptimizableMCAgent instance with parameters applied
        """
        # Validate parameters first
        self.parameter_space.validate_parameters(params)
        
        # Create agent with default boolean flags (all True as per spec)
        agent = OptimizableMCAgent(
            name="candidate",
            # Boolean flags (fixed as True per Phase 2 spec)
            weighted_sampling=True,
            enable_parallel=True,
            enhanced_pruning=True,
            variance_reduction=True,
            betting_history_enabled=True,
            player_trust_enabled=True,
            bayesian_sampling=True
        )
        
        # Apply parameter configuration
        agent.update_parameters(params)
        
        return agent
    
    def evaluate_configuration(self, params: Dict[str, Union[int, float]]) -> Dict[str, Any]:
        """
        Evaluate a single parameter configuration with detailed progress logging.
        
        Args:
            params: Parameter configuration to evaluate
            
        Returns:
            Dictionary with fitness metrics and detailed results
            
        Raises:
            ValueError: If parameters are invalid or out of bounds
            KeyError: If required parameters are missing
        """
        eval_start_time = time.time()
        
        # Log start of evaluation with parameter summary
        param_summary = ", ".join([f"{k}={v}" for k, v in params.items()])
        logger.info(f"Starting evaluation of parameters: {param_summary}")
        print(f"  Starting parameter evaluation: [{param_summary}]")
        
        # Parameter validation - let exceptions propagate for invalid params
        validation_start = time.time()
        self.parameter_space.validate_parameters(params)
        validation_time = time.time() - validation_start
        logger.debug(f"Parameter validation completed in {validation_time:.3f}s")
        
        try:
            # Create candidate agent
            agent_start = time.time()
            logger.info("Creating candidate agent with parameters...")
            print("    • Creating candidate agent...")
            candidate = self.create_optimizable_agent(params)
            agent_time = time.time() - agent_start
            logger.debug(f"Candidate agent created in {agent_time:.3f}s")
            
            # Create opponent mix
            opponent_start = time.time()
            logger.info("Setting up opponent mix...")
            print("    • Setting up opponents...")
            opponents = self.create_opponent_mix()
            opponent_time = time.time() - opponent_start
            logger.info(f"Created {len(opponents)} opponents in {opponent_time:.3f}s")
            print(f"    • {len(opponents)} opponents ready")
            
            # Run tournament
            tournament_start = time.time()
            logger.info(f"Starting tournament: {self.evaluation_games} games vs {len(opponents)} opponents")
            print(f"    • Running tournament: {self.evaluation_games} games vs {len(opponents)} opponents...")
            results = self.run_tournament(candidate, opponents)
            tournament_time = time.time() - tournament_start
            
            # Calculate fitness components
            calc_start = time.time()
            logger.info("Calculating fitness metrics...")
            print("    • Calculating fitness metrics...")
            
            win_rate = calculate_win_rate(results, candidate_index=0)
            variance_penalty = calculate_variance_penalty(results)
            robustness_bonus = calculate_robustness_bonus(results)
            
            # Overall fitness score
            fitness = win_rate - variance_penalty + robustness_bonus
            calc_time = time.time() - calc_start
            
            total_time = time.time() - eval_start_time
            
            # Detailed results logging
            candidate_wins = results.get(0, 0)
            total_games = sum(results.values())
            logger.info(f"Evaluation completed: {candidate_wins}/{total_games} wins "
                       f"(win_rate={win_rate:.3f}, fitness={fitness:.4f}) in {total_time:.1f}s")
            print(f"    • Evaluation complete: {candidate_wins}/{total_games} wins, "
                  f"fitness={fitness:.4f} (took {total_time:.1f}s)")
            
            # Performance breakdown
            logger.debug(f"Timing breakdown - Agent: {agent_time:.3f}s, "
                        f"Opponents: {opponent_time:.3f}s, Tournament: {tournament_time:.1f}s, "
                        f"Calc: {calc_time:.3f}s, Total: {total_time:.1f}s")
            
            return {
                'fitness': fitness,
                'win_rate': win_rate,
                'variance': variance_penalty,
                'robustness': robustness_bonus,
                'detailed_results': results,
                'parameters': params.copy(),
                'timing': {
                    'total_time': total_time,
                    'tournament_time': tournament_time,
                    'agent_setup_time': agent_time,
                    'opponent_setup_time': opponent_time,
                    'calculation_time': calc_time
                }
            }
            
        except (ValueError, KeyError, TypeError):
            # Re-raise parameter-related errors
            raise
        except Exception as e:
            # Handle other errors (tournament failures, etc.)
            eval_time = time.time() - eval_start_time
            logger.error(f"Error evaluating configuration {params} after {eval_time:.1f}s: {e}")
            print(f"    • Evaluation failed after {eval_time:.1f}s: {e}")
            return {
                'fitness': 0.0,
                'win_rate': 0.0,
                'variance': 0.0,
                'robustness': 0.0,
                'detailed_results': {},
                'parameters': params.copy(),
                'error': str(e),
                'timing': {'total_time': eval_time}
            }
    
    def evaluate_batch(self, configurations: List[Dict[str, Union[int, float]]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple parameter configurations.
        
        Args:
            configurations: List of parameter configurations
            
        Returns:
            List of fitness evaluation results
        """
        results = []
        for i, config in enumerate(configurations):
            logger.info(f"Evaluating configuration {i+1}/{len(configurations)}")
            result = self.evaluate_configuration(config)
            results.append(result)
        
        return results
    
    def run_tournament(self, candidate: OptimizableMCAgent, opponents: List) -> Dict[int, int]:
        """
        Run tournament between candidate and opponents.
        
        Args:
            candidate: Candidate agent to evaluate
            opponents: List of opponent agents
            
        Returns:
            Dictionary mapping agent indices to win counts
        """
        # Create simulator
        num_players = len(opponents) + 1
        sim = PerudoSimulator(
            num_players=num_players,
            start_dice=5,
            ones_are_wild=True,
            use_maputa=True,
            use_exact=True
        )
        
        # Create agent list (candidate first)
        all_agents = [candidate] + opponents
        agent_names = [agent.name.lower() if hasattr(agent, 'name') else 'agent' for agent in all_agents]
        
        # Use existing tournament infrastructure
        results = play_tournament(sim, all_agents, games=self.evaluation_games)
        
        return results
    
    def load_champion_agent(self, champion_id: str) -> Optional[OptimizableMCAgent]:
        """
        Load champion agent from previous generations.
        
        Args:
            champion_id: Identifier for champion agent
            
        Returns:
            Champion agent instance or None if not found
        """
        if champion_id in self._champion_cache:
            return self._champion_cache[champion_id]
        
        champion_path = os.path.join(self.champion_storage_path, f"{champion_id}.pkl")
        
        if not os.path.exists(champion_path):
            logger.debug(f"Champion {champion_id} not found at {champion_path}")
            return None
        
        try:
            with open(champion_path, 'rb') as f:
                champion_data = pickle.load(f)
            
            # Reconstruct champion agent
            champion = OptimizableMCAgent(name=champion_id)
            champion.update_parameters(champion_data['parameters'])
            
            self._champion_cache[champion_id] = champion
            return champion
            
        except Exception as e:
            logger.warning(f"Error loading champion {champion_id}: {e}")
            return None
    
    def save_champion_agent(self, agent: OptimizableMCAgent, champion_id: str, 
                          fitness_score: float, generation: int):
        """
        Save champion agent for future use.
        
        Args:
            agent: Champion agent to save
            champion_id: Identifier for champion
            fitness_score: Fitness score achieved
            generation: Generation number
        """
        champion_path = os.path.join(self.champion_storage_path, f"{champion_id}.pkl")
        
        champion_data = {
            'parameters': agent.get_current_parameters(),
            'fitness_score': fitness_score,
            'generation': generation,
            'champion_id': champion_id
        }
        
        try:
            with open(champion_path, 'wb') as f:
                pickle.dump(champion_data, f)
            
            logger.info(f"Saved champion {champion_id} with fitness {fitness_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error saving champion {champion_id}: {e}")


def play_tournament(sim: PerudoSimulator, agents: List, games: int) -> Dict[int, int]:
    """
    Run tournament using existing tournament infrastructure with detailed progress logging.
    
    Args:
        sim: PerudoSimulator instance
        agents: List of agent instances  
        games: Number of games to play
        
    Returns:
        Dictionary mapping agent indices to win counts
    """
    results = Counter()
    start_time = time.time()
    
    # Progress logging intervals
    log_interval = max(1, games // 20)  # Log at least 20 times during tournament
    heartbeat_interval = max(1, min(10, games // 10))  # Heartbeat every 10 games or 10% of games
    
    logger.info(f"Starting tournament: {games} games between {len(agents)} agents")
    
    for g in range(games):
        game_start = time.time()
        
        # Play single game
        winner, _ = sim.play_game(agents)
        results[winner] += 1
        
        game_time = time.time() - game_start
        
        # Detailed progress logging
        if g % log_interval == 0 or g == games - 1:
            elapsed = time.time() - start_time
            progress_pct = (g + 1) / games * 100
            
            if g > 0:
                avg_game_time = elapsed / (g + 1)
                remaining_games = games - (g + 1)
                eta_seconds = remaining_games * avg_game_time
                eta_minutes = eta_seconds / 60
                
                current_standings = dict(results)
                winner_summary = ", ".join([f"Agent{i}:{wins}" for i, wins in sorted(current_standings.items())])
                
                logger.info(f"Tournament progress: {g+1}/{games} games ({progress_pct:.1f}%) - "
                           f"Avg {avg_game_time:.2f}s/game - ETA {eta_minutes:.1f}min - "
                           f"Current: [{winner_summary}]")
                print(f"    Tournament: {g+1}/{games} games ({progress_pct:.1f}%) completed, "
                      f"ETA {eta_minutes:.1f} minutes...")
            else:
                logger.info(f"Tournament progress: {g+1}/{games} games ({progress_pct:.1f}%) - "
                           f"First game: {game_time:.2f}s")
                print(f"    Tournament: {g+1}/{games} games ({progress_pct:.1f}%) - "
                      f"First game took {game_time:.2f}s...")
        
        # Heartbeat logging for very long tournaments
        elif g % heartbeat_interval == 0:
            elapsed = time.time() - start_time
            progress_pct = (g + 1) / games * 100
            avg_game_time = elapsed / (g + 1)
            
            logger.debug(f"Tournament heartbeat: {g+1}/{games} games ({progress_pct:.1f}%) - "
                        f"Running {avg_game_time:.2f}s per game")
            print(f"    • Game {g+1}/{games} ({progress_pct:.1f}%) - {avg_game_time:.2f}s avg per game")
    
    total_time = time.time() - start_time
    avg_game_time = total_time / games if games > 0 else 0
    
    # Convert to dictionary with all agent indices
    result_dict = {}
    for i in range(len(agents)):
        result_dict[i] = results.get(i, 0)
    
    # Final tournament summary
    winner_summary = ", ".join([f"Agent{i}:{wins}" for i, wins in sorted(result_dict.items())])
    logger.info(f"Tournament completed: {games} games in {total_time:.1f}s "
               f"(avg {avg_game_time:.2f}s/game) - Final: [{winner_summary}]")
    print(f"    Tournament completed: {total_time:.1f}s total, {avg_game_time:.2f}s per game")
    
    return result_dict


def calculate_win_rate(results: Dict[int, int], candidate_index: int = 0) -> float:
    """
    Calculate win rate for candidate agent.
    
    Args:
        results: Tournament results mapping agent indices to win counts
        candidate_index: Index of candidate agent (usually 0)
        
    Returns:
        Win rate as fraction between 0.0 and 1.0
    """
    total_games = sum(results.values())
    if total_games == 0:
        return 0.0
    
    candidate_wins = results.get(candidate_index, 0)
    return candidate_wins / total_games


def calculate_variance_penalty(results: Dict[int, int]) -> float:
    """
    Calculate variance penalty for fitness score.
    
    Penalizes more evenly distributed performance (less robust candidate).
    Rewards dominant candidate performance (more robust).
    
    Args:
        results: Tournament results mapping agent indices to win counts
        
    Returns:
        Variance penalty (higher penalty for more evenly distributed performance)
    """
    if len(results) < 2:
        return 0.0
    
    total_games = sum(results.values())
    if total_games == 0:
        return 0.0
    
    # Get candidate win rate (assume candidate is index 0)
    candidate_wins = results.get(0, 0)
    candidate_win_rate = candidate_wins / total_games
    
    # Penalty inversely related to candidate dominance
    # If candidate wins most games, penalty is low
    # If wins are evenly distributed, penalty is high
    
    # Use inverse of candidate win rate as a measure of "evenness"
    # High candidate win rate = low penalty
    # Low candidate win rate = high penalty
    evenness = 1.0 - candidate_win_rate
    
    # Scale penalty (0.0 to ~0.3 range)
    penalty = min(0.3, evenness * 0.6)  # Scale factor to reach ~0.3 max
    
    return penalty


def calculate_robustness_bonus(results: Dict[int, int]) -> float:
    """
    Calculate robustness bonus for fitness score.
    
    Bonus for beating all types of opponents consistently.
    
    Args:
        results: Tournament results mapping agent indices to win counts
        
    Returns:
        Robustness bonus (higher bonus for more robust performance)
    """
    if len(results) < 2:
        return 0.0
    
    total_games = sum(results.values())
    candidate_wins = results.get(0, 0)  # Assume candidate is index 0
    
    if total_games == 0:
        return 0.0
    
    win_rate = candidate_wins / total_games
    
    # Bonus increases with win rate, max bonus ~0.2
    if win_rate > 0.8:
        return 0.2  # Exceptional performance
    elif win_rate > 0.6:
        return 0.1  # Good performance
    elif win_rate > 0.4:
        return 0.05  # Decent performance
    else:
        return 0.0  # Poor performance


def evaluate_candidate_fitness(params_dict: Dict[str, Union[int, float]], 
                              evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function to evaluate candidate fitness.
    
    Args:
        params_dict: Parameter configuration to evaluate
        evaluation_config: Evaluation configuration including opponent mix
        
    Returns:
        Dictionary with fitness metrics
    """
    # Create parameter space from the provided parameters (for flexibility)
    config = {
        "parameters": {}
    }
    
    # Add default parameter specifications for the provided parameters
    param_specs = {
        'n': {"type": "int", "min": 100, "max": 2000, "default": 400},
        'chunk_size': {"type": "int", "min": 4, "max": 32, "default": 8},
        'early_stop_margin': {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15},
        'trust_learning_rate': {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1},
        'history_memory_rounds': {"type": "int", "min": 5, "max": 20, "default": 10},
        'num_workers': {"type": "int", "min": 1, "max": 8, "default": 4},
        # Utils microparameters
        'decay_factor': {"type": "float", "min": 0.6, "max": 0.95, "default": 0.8},
        'scaling_factor': {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0},
        'face_weight_boost': {"type": "float", "min": 1.2, "max": 3.0, "default": 2.0},
        'min_weight_threshold': {"type": "float", "min": 0.05, "max": 0.2, "default": 0.1},
    }
    
    # Only include specifications for parameters that are provided
    for param_name in params_dict.keys():
        if param_name in param_specs:
            config["parameters"][param_name] = param_specs[param_name]
    
    param_space = ParameterSpace(config)
    
    # Create evaluator
    evaluator = FitnessEvaluator(
        parameter_space=param_space,
        opponent_config=evaluation_config.get('opponent_mix', {}),
        evaluation_games=evaluation_config.get('num_games', 50)
    )
    
    # Evaluate configuration
    return evaluator.evaluate_configuration(params_dict)


def create_opponent_mix(opponent_config: Dict[str, int]) -> List:
    """
    Standalone function to create opponent mix.
    
    Args:
        opponent_config: Configuration specifying opponent counts
        
    Returns:
        List of agent instances
    """
    opponents = []
    
    # Add baseline agents
    for i in range(opponent_config.get('baseline_count', 2)):
        opponents.append(BaselineAgent(name=f"baseline_{i}"))
    
    # Add random agents
    for i in range(opponent_config.get('random_count', 1)):
        opponents.append(RandomAgent(name=f"random_{i}"))
    
    # Champions would be loaded from storage in real implementation
    # For now, just return baseline + random agents
    
    return opponents


class RobustFitnessEvaluator:
    """
    Sophisticated fitness function with variance penalties and robustness metrics.
    
    This evaluator balances multiple fitness components:
    - Win rate: Primary performance metric
    - Variance penalty: Penalizes inconsistent performance across tournaments
    - Robustness bonus: Rewards performance across diverse opponent types
    
    The combined fitness helps discover parameter configurations that perform
    consistently well across diverse game scenarios and opponent mixes.
    """
    
    def __init__(self, win_rate: float = 1.0, variance_penalty: float = 0.3, 
                 robustness_bonus: float = 0.1):
        """
        Initialize robust fitness evaluator with component weights.
        
        Args:
            win_rate: Weight for win rate component (higher is better)
            variance_penalty: Weight for variance penalty (penalty for inconsistency)
            robustness_bonus: Weight for robustness bonus (bonus for diverse performance)
        """
        self.win_rate_weight = win_rate
        self.variance_penalty_weight = variance_penalty
        self.robustness_bonus_weight = robustness_bonus
    
    def calculate_win_rate(self, tournament_results: Dict[str, List[int]], 
                          candidate_name: str) -> float:
        """
        Calculate win rate for candidate across all tournaments.
        
        Args:
            tournament_results: Dict mapping agent names to lists of wins per tournament
            candidate_name: Name of the candidate agent
            
        Returns:
            Win rate as fraction of total games won
        """
        if candidate_name not in tournament_results:
            raise KeyError(f"Candidate '{candidate_name}' not found in results")
        
        candidate_wins = sum(tournament_results[candidate_name])
        total_games = sum(sum(results) for results in tournament_results.values())
        
        if total_games == 0:
            return 0.0
        
        return candidate_wins / total_games
    
    def calculate_variance_penalty(self, tournament_results: Dict[str, List[int]], 
                                 candidate_name: str) -> float:
        """
        Calculate variance penalty for inconsistent performance.
        
        Args:
            tournament_results: Dict mapping agent names to lists of wins per tournament
            candidate_name: Name of the candidate agent
            
        Returns:
            Variance penalty (0.0 to ~0.3, higher for more inconsistent performance)
        """
        if candidate_name not in tournament_results:
            raise KeyError(f"Candidate '{candidate_name}' not found in results")
        
        candidate_results = tournament_results[candidate_name]
        
        if len(candidate_results) <= 1:
            return 0.0  # No variance with single tournament
        
        # Calculate coefficient of variation (normalized standard deviation)
        mean_wins = np.mean(candidate_results)
        if mean_wins == 0:
            return 0.3  # Maximum penalty for zero performance
        
        std_wins = np.std(candidate_results)
        cv = std_wins / mean_wins  # Coefficient of variation
        
        # Scale coefficient of variation to penalty range (0.0 to ~0.3)
        penalty = min(0.3, cv * 0.3)  # Scale factor to keep penalty reasonable
        
        return penalty
    
    def calculate_robustness_bonus(self, tournament_results: Dict[str, List[int]], 
                                 candidate_name: str) -> float:
        """
        Calculate robustness bonus for diverse performance across opponents.
        
        Args:
            tournament_results: Dict mapping agent names to lists of wins per tournament
            candidate_name: Name of the candidate agent
            
        Returns:
            Robustness bonus (0.0 to ~0.2, higher for more robust performance)
        """
        if candidate_name not in tournament_results:
            raise KeyError(f"Candidate '{candidate_name}' not found in results")
        
        candidate_results = tournament_results[candidate_name]
        
        # Calculate win rate
        candidate_wins = sum(candidate_results)
        total_games = sum(sum(results) for results in tournament_results.values())
        
        if total_games == 0:
            return 0.0
        
        win_rate = candidate_wins / total_games
        
        # Bonus based on performance tiers
        if win_rate >= 0.8:
            return 0.2  # Exceptional performance
        elif win_rate >= 0.6:
            return 0.15  # Very good performance
        elif win_rate >= 0.4:
            return 0.1   # Good performance
        elif win_rate >= 0.25:
            return 0.05  # Decent performance
        else:
            return 0.0   # Poor performance
    
    def calculate_fitness(self, tournament_results: Dict[str, List[int]], 
                         candidate_name: str) -> float:
        """
        Calculate combined fitness score.
        
        Args:
            tournament_results: Dict mapping agent names to lists of wins per tournament
            candidate_name: Name of the candidate agent
            
        Returns:
            Combined fitness score
        """
        win_rate = self.calculate_win_rate(tournament_results, candidate_name)
        variance_penalty = self.calculate_variance_penalty(tournament_results, candidate_name)
        robustness_bonus = self.calculate_robustness_bonus(tournament_results, candidate_name)
        
        # Combined fitness: win_rate - penalty + bonus
        fitness = (self.win_rate_weight * win_rate - 
                  self.variance_penalty_weight * variance_penalty +
                  self.robustness_bonus_weight * robustness_bonus)
        
        return fitness
    
    def calculate_robustness_metrics(self, tournament_results: Dict[str, List[int]], 
                                   candidate_name: str) -> Dict[str, float]:
        """
        Calculate detailed robustness metrics for analysis.
        
        Args:
            tournament_results: Dict mapping agent names to lists of wins per tournament
            candidate_name: Name of the candidate agent
            
        Returns:
            Dictionary with detailed metrics
        """
        win_rate = self.calculate_win_rate(tournament_results, candidate_name)
        variance_penalty = self.calculate_variance_penalty(tournament_results, candidate_name)
        robustness_bonus = self.calculate_robustness_bonus(tournament_results, candidate_name)
        
        candidate_results = tournament_results[candidate_name]
        
        # Consistency score (inverse of coefficient of variation)
        if len(candidate_results) > 1 and np.mean(candidate_results) > 0:
            cv = np.std(candidate_results) / np.mean(candidate_results)
            consistency_score = max(0.0, 1.0 - cv)  # Higher is more consistent
        else:
            consistency_score = 1.0 if len(candidate_results) == 1 else 0.0
        
        # Diversity score (how well agent performs across different tournament scenarios)
        # For now, use win rate as proxy (in real implementation, would analyze opponent-specific performance)
        diversity_score = min(1.0, win_rate * 2.0)  # Scale win rate to diversity measure
        
        return {
            'win_rate': win_rate,
            'variance_penalty': variance_penalty,
            'robustness_bonus': robustness_bonus,
            'consistency_score': consistency_score,
            'diversity_score': diversity_score,
            'combined_fitness': self.calculate_fitness(tournament_results, candidate_name)
        }