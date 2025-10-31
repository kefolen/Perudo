"""
Result Management System for Monte Carlo Agent Parameter Optimization

This module provides comprehensive result logging, checkpointing, and resume
capability for the evolutionary optimization framework, following TDD principles.

Components:
- ResultLogger: Logging optimization results and progress
- OptimizationCheckpoint: Save/load optimization state for resume capability
- ResultManager: Comprehensive management of optimization runs
- Data structures: EvaluationResult, GenerationResult, OptimizationRun
"""

import json
import os
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import glob


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data structure for individual evaluation results."""
    individual_id: str
    parameters: Dict[str, Union[int, float]]
    fitness_score: float
    win_rate: float
    variance_penalty: float
    robustness_bonus: float
    evaluation_time: float
    games_played: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GenerationResult:
    """Data structure for complete generation results."""
    generation: int
    population_size: int
    evaluations: List[EvaluationResult]
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    total_evaluation_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OptimizationRun:
    """Data structure for optimization run metadata."""
    run_id: str
    config: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    generations: List[GenerationResult] = field(default_factory=list)
    best_fitness_achieved: float = 0.0
    total_evaluations: int = 0


class ResultLogger:
    """
    Comprehensive logging system for optimization results.
    
    Handles structured logging of optimization progress, results, and metadata
    with support for multiple output formats and analysis.
    """
    
    def __init__(self, log_dir: str, run_id: str):
        """
        Initialize result logger.
        
        Args:
            log_dir: Directory for log files
            run_id: Unique identifier for optimization run
        """
        self.log_dir = log_dir
        self.run_id = run_id
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file paths
        self.log_file_path = os.path.join(log_dir, f"{run_id}_optimization.log")
        self.results_file_path = os.path.join(log_dir, f"{run_id}_results.json")
        
        # Initialize results structure
        self._init_results_file()
    
    def _init_results_file(self):
        """Initialize JSON results file with basic structure."""
        if not os.path.exists(self.results_file_path):
            initial_data = {
                'run_id': self.run_id,
                'created_at': datetime.now().isoformat(),
                'config': {},
                'generations': [],
                'summary': {}
            }
            with open(self.results_file_path, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def log_generation_start(self, generation_info: Dict[str, Any]):
        """
        Log start of generation evaluation.
        
        Args:
            generation_info: Information about generation being started
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] GENERATION_START: {json.dumps(generation_info)}\n"
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
        
        logger.info(f"Started generation {generation_info.get('generation', 'unknown')}")
    
    def log_evaluation_result(self, generation: int, evaluation_result: EvaluationResult):
        """
        Log individual evaluation result.
        
        Args:
            generation: Generation number
            evaluation_result: Evaluation result to log
        """
        # Cache results data in memory to avoid repeated file I/O
        if not hasattr(self, '_results_cache'):
            with open(self.results_file_path, 'r') as f:
                self._results_cache = json.load(f)
        
        # Ensure generation exists in results
        while len(self._results_cache['generations']) <= generation:
            self._results_cache['generations'].append({
                'generation': len(self._results_cache['generations']),
                'evaluations': [],
                'summary': {}
            })
        
        # Add evaluation to generation (optimized dict creation)
        eval_dict = {
            'individual_id': evaluation_result.individual_id,
            'parameters': evaluation_result.parameters,
            'fitness_score': evaluation_result.fitness_score,
            'win_rate': evaluation_result.win_rate,
            'variance_penalty': evaluation_result.variance_penalty,
            'robustness_bonus': evaluation_result.robustness_bonus,
            'evaluation_time': evaluation_result.evaluation_time,
            'games_played': evaluation_result.games_played,
            'timestamp': evaluation_result.timestamp
        }
        
        self._results_cache['generations'][generation]['evaluations'].append(eval_dict)
        
        # Batch write: only write to disk every N evaluations or on explicit flush
        self._pending_writes = getattr(self, '_pending_writes', 0) + 1
        if self._pending_writes >= 10:  # Batch size of 10
            self._flush_results_cache()
    
    def _flush_results_cache(self):
        """Flush cached results to disk for optimized I/O."""
        if hasattr(self, '_results_cache'):
            with open(self.results_file_path, 'w') as f:
                json.dump(self._results_cache, f, indent=2)
            self._pending_writes = 0
    
    def __del__(self):
        """Ensure cache is flushed when logger is destroyed."""
        try:
            self._flush_results_cache()
        except:
            pass  # Ignore errors during cleanup
    
    def log_generation_summary(self, generation_summary: Dict[str, Any]):
        """
        Log generation summary statistics.
        
        Args:
            generation_summary: Summary statistics for generation
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] GENERATION_SUMMARY: {json.dumps(generation_summary)}\n"
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
        
        # Update generation summary in cached results
        generation = generation_summary.get('generation')
        if generation is not None:
            # Use cached results data
            if not hasattr(self, '_results_cache'):
                with open(self.results_file_path, 'r') as f:
                    self._results_cache = json.load(f)
            
            # Ensure generation exists
            while len(self._results_cache['generations']) <= generation:
                self._results_cache['generations'].append({
                    'generation': len(self._results_cache['generations']),
                    'evaluations': [],
                    'summary': {}
                })
            
            self._results_cache['generations'][generation]['summary'] = generation_summary
            
            # Force flush on generation summary (important milestone)
            self._flush_results_cache()
    
    def log_optimization_config(self, config: Dict[str, Any]):
        """
        Log optimization configuration.
        
        Args:
            config: Optimization configuration to log
        """
        with open(self.results_file_path, 'r') as f:
            results_data = json.load(f)
        
        results_data['config'] = config
        
        with open(self.results_file_path, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from logged results.
        
        Returns:
            Dictionary with summary statistics
        """
        with open(self.results_file_path, 'r') as f:
            results_data = json.load(f)
        
        generations = results_data.get('generations', [])
        
        # Calculate summary statistics
        total_generations = len(generations)
        best_fitness_achieved = 0.0
        total_evaluation_time = 0.0
        
        for gen in generations:
            gen_summary = gen.get('summary', {})
            if 'best_fitness' in gen_summary:
                best_fitness_achieved = max(best_fitness_achieved, gen_summary['best_fitness'])
            if 'evaluation_time' in gen_summary:
                total_evaluation_time += gen_summary['evaluation_time']
        
        return {
            'run_id': self.run_id,
            'total_generations': total_generations,
            'best_fitness_achieved': best_fitness_achieved,
            'total_evaluation_time': total_evaluation_time,
            'log_file': self.log_file_path,
            'results_file': self.results_file_path
        }


class OptimizationCheckpoint:
    """
    Checkpointing system for optimization state persistence.
    
    Enables saving and loading optimization state to resume interrupted
    optimization runs from specific generations.
    """
    
    def __init__(self, checkpoint_dir: str, run_id: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
            run_id: Unique identifier for optimization run
        """
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> str:
        """
        Save optimization checkpoint.
        
        Args:
            checkpoint_data: Optimization state to save
            
        Returns:
            Path to saved checkpoint file
        """
        generation = checkpoint_data.get('generation', 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.run_id}_gen_{generation}_{timestamp}.pkl"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Add metadata
        checkpoint_data['checkpoint_metadata'] = {
            'run_id': self.run_id,
            'generation': generation,
            'saved_at': datetime.now().isoformat(),
            'checkpoint_path': checkpoint_path
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint for generation {generation} to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load optimization checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded optimization state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the most recent checkpoint for this run.
        
        Returns:
            Path to most recent checkpoint or None if no checkpoints exist
        """
        pattern = os.path.join(self.checkpoint_dir, f"{self.run_id}_gen_*.pkl")
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        return checkpoint_files[0]
    
    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent checkpoints to keep
        """
        pattern = os.path.join(self.checkpoint_dir, f"{self.run_id}_gen_*.pkl")
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= keep_count:
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=os.path.getmtime)
        
        # Remove oldest files
        files_to_remove = checkpoint_files[:-keep_count]
        for file_path in files_to_remove:
            os.remove(file_path)
            logger.info(f"Removed old checkpoint: {file_path}")


class ResultManager:
    """
    Comprehensive result management for optimization runs.
    
    Combines logging and checkpointing with high-level optimization
    run management, statistics, and export capabilities.
    """
    
    def __init__(self, results_dir: str, parameter_space, run_id: str):
        """
        Initialize result manager.
        
        Args:
            results_dir: Base directory for results
            parameter_space: ParameterSpace instance for validation
            run_id: Unique identifier for optimization run
        """
        self.results_dir = results_dir
        self.parameter_space = parameter_space
        self.run_id = run_id
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.logger = ResultLogger(
            log_dir=os.path.join(results_dir, "logs"),
            run_id=run_id
        )
        
        self.checkpoint_manager = OptimizationCheckpoint(
            checkpoint_dir=os.path.join(results_dir, "checkpoints"),
            run_id=run_id
        )
        
        # Current optimization run
        self.current_run: Optional[OptimizationRun] = None
    
    def start_optimization_run(self, config: Dict[str, Any]) -> OptimizationRun:
        """
        Start new optimization run with configuration.
        
        Args:
            config: Optimization configuration
            
        Returns:
            OptimizationRun instance
        """
        self.current_run = OptimizationRun(
            run_id=self.run_id,
            config=config,
            start_time=datetime.now()
        )
        
        # Log configuration
        self.logger.log_optimization_config(config)
        
        logger.info(f"Started optimization run {self.run_id}")
        return self.current_run
    
    def record_generation_results(self, generation_result: GenerationResult):
        """
        Record complete generation results.
        
        Args:
            generation_result: Generation result to record
        """
        if self.current_run is None:
            raise RuntimeError("No active optimization run. Call start_optimization_run first.")
        
        # Add generation to current run
        self.current_run.generations.append(generation_result)
        
        # Update run statistics
        self.current_run.best_fitness_achieved = max(
            self.current_run.best_fitness_achieved,
            generation_result.best_fitness
        )
        self.current_run.total_evaluations += len(generation_result.evaluations)
        
        # Log individual evaluations
        for eval_result in generation_result.evaluations:
            self.logger.log_evaluation_result(generation_result.generation, eval_result)
        
        # Log generation summary
        gen_summary = {
            'generation': generation_result.generation,
            'best_fitness': generation_result.best_fitness,
            'avg_fitness': generation_result.avg_fitness,
            'worst_fitness': generation_result.worst_fitness,
            'evaluation_time': generation_result.total_evaluation_time,
            'population_size': generation_result.population_size
        }
        self.logger.log_generation_summary(gen_summary)
        
        logger.info(f"Recorded results for generation {generation_result.generation}")
    
    def save_checkpoint(self, optimization_state: Dict[str, Any]) -> str:
        """
        Save optimization checkpoint.
        
        Args:
            optimization_state: Current optimization state
            
        Returns:
            Path to saved checkpoint
        """
        return self.checkpoint_manager.save_checkpoint(optimization_state)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load optimization checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded optimization state
        """
        return self.checkpoint_manager.load_checkpoint(checkpoint_path)
    
    def resume_optimization(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Resume optimization from checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to load, or None to use latest
            
        Returns:
            Resumed optimization state or None if no checkpoint found
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.warning(f"No checkpoint found for run {self.run_id}")
            return None
        
        state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed optimization from generation {state.get('generation', 'unknown')}")
        
        return state
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization run summary.
        
        Returns:
            Dictionary with optimization statistics
        """
        if self.current_run is None:
            return {'error': 'No active optimization run'}
        
        total_runtime = (datetime.now() - self.current_run.start_time).total_seconds()
        
        summary = {
            'run_id': self.current_run.run_id,
            'status': self.current_run.status,
            'start_time': self.current_run.start_time.isoformat(),
            'total_runtime': total_runtime,
            'generations_completed': len(self.current_run.generations),
            'best_fitness': self.current_run.best_fitness_achieved,
            'total_evaluations': self.current_run.total_evaluations,
            'config': self.current_run.config
        }
        
        if self.current_run.generations:
            # Calculate additional statistics
            recent_gen = self.current_run.generations[-1]
            summary.update({
                'latest_generation': recent_gen.generation,
                'latest_avg_fitness': recent_gen.avg_fitness,
                'latest_evaluation_time': recent_gen.total_evaluation_time
            })
        
        return summary
    
    def export_results(self, format_type: str = 'json') -> str:
        """
        Export optimization results to file.
        
        Args:
            format_type: Export format ('json', 'csv', 'pkl')
            
        Returns:
            Path to exported file
        """
        if format_type == 'json':
            export_path = os.path.join(self.results_dir, f"{self.run_id}_export.json")
            
            export_data = {
                'run_id': self.run_id,
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_optimization_summary()
            }
            
            if self.current_run:
                export_data.update({
                    'config': self.current_run.config,
                    'generations': []
                })
                
                for gen in self.current_run.generations:
                    gen_data = {
                        'generation': gen.generation,
                        'population_size': gen.population_size,
                        'best_fitness': gen.best_fitness,
                        'avg_fitness': gen.avg_fitness,
                        'worst_fitness': gen.worst_fitness,
                        'total_evaluation_time': gen.total_evaluation_time,
                        'evaluations': [
                            {
                                'individual_id': eval_result.individual_id,
                                'parameters': eval_result.parameters,
                                'fitness_score': eval_result.fitness_score,
                                'win_rate': eval_result.win_rate,
                                'variance_penalty': eval_result.variance_penalty,
                                'robustness_bonus': eval_result.robustness_bonus,
                                'evaluation_time': eval_result.evaluation_time,
                                'games_played': eval_result.games_played
                            }
                            for eval_result in gen.evaluations
                        ]
                    }
                    export_data['generations'].append(gen_data)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return export_path
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")