"""
Tests for result management system - TDD Cycle 4.1: Result Management

This module tests comprehensive result logging, checkpointing, and resume
capability for the evolutionary optimization framework.
"""
import pytest
import json
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from eval.result_management import (
    ResultLogger, OptimizationCheckpoint, ResultManager,
    OptimizationRun, GenerationResult, EvaluationResult
)
from eval.hyperopt_framework import ParameterSpace


class TestResultLogger:
    """Test suite for result logging functionality."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def result_logger(self, temp_log_dir):
        """Create ResultLogger instance for testing."""
        return ResultLogger(log_dir=temp_log_dir, run_id="test_run_001")

    def test_result_logger_initialization(self, result_logger, temp_log_dir):
        """Test ResultLogger initialization and directory creation."""
        assert result_logger.log_dir == temp_log_dir
        assert result_logger.run_id == "test_run_001"
        assert os.path.exists(temp_log_dir)
        
        # Should create log files
        assert hasattr(result_logger, 'log_file_path')
        assert hasattr(result_logger, 'results_file_path')

    def test_log_generation_start(self, result_logger):
        """Test logging generation start."""
        generation_info = {
            'generation': 5,
            'population_size': 20,
            'algorithm': 'CMA-ES',
            'timestamp': datetime.now().isoformat()
        }
        
        result_logger.log_generation_start(generation_info)
        
        # Should create log entry without errors
        assert os.path.exists(result_logger.log_file_path)

    def test_log_evaluation_result(self, result_logger):
        """Test logging individual evaluation results."""
        evaluation_result = EvaluationResult(
            individual_id="ind_001",
            parameters={
                'n': 400,
                'chunk_size': 8,
                'early_stop_margin': 0.15
            },
            fitness_score=0.75,
            win_rate=0.65,
            variance_penalty=0.05,
            robustness_bonus=0.15,
            evaluation_time=2.5,
            games_played=50
        )
        
        result_logger.log_evaluation_result(0, evaluation_result)
        
        # Should log result without errors
        assert os.path.exists(result_logger.results_file_path)

    def test_log_generation_summary(self, result_logger):
        """Test logging generation summary statistics."""
        generation_summary = {
            'generation': 3,
            'best_fitness': 0.82,
            'avg_fitness': 0.65,
            'worst_fitness': 0.41,
            'fitness_std': 0.12,
            'evaluation_time': 45.2,
            'convergence_metric': 0.08
        }
        
        result_logger.log_generation_summary(generation_summary)
        
        # Verify logging completed
        with open(result_logger.log_file_path, 'r') as f:
            log_content = f.read()
            assert 'GENERATION_SUMMARY' in log_content
            assert '0.82' in log_content  # best_fitness

    def test_log_optimization_config(self, result_logger):
        """Test logging optimization configuration."""
        config = {
            'algorithm': 'CMA-ES',
            'population_size': 20,
            'max_generations': 50,
            'parameter_space': {
                'n': {'min': 100, 'max': 2000},
                'chunk_size': {'min': 4, 'max': 32}
            },
            'evaluation': {
                'multi_fidelity': True,
                'games_per_level': [10, 50, 200]
            }
        }
        
        result_logger.log_optimization_config(config)
        
        # Should store config in results file
        with open(result_logger.results_file_path, 'r') as f:
            results_data = json.load(f)
            assert 'config' in results_data
            assert results_data['config']['algorithm'] == 'CMA-ES'

    def test_get_log_summary(self, result_logger):
        """Test retrieving log summary statistics."""
        # Log some sample data
        for gen in range(3):
            gen_summary = {
                'generation': gen,
                'best_fitness': 0.5 + gen * 0.1,
                'avg_fitness': 0.4 + gen * 0.08,
                'evaluation_time': 30.0 + gen * 5.0
            }
            result_logger.log_generation_summary(gen_summary)
        
        summary = result_logger.get_log_summary()
        
        assert 'total_generations' in summary
        assert summary['total_generations'] == 3
        assert 'best_fitness_achieved' in summary
        assert summary['best_fitness_achieved'] == 0.7  # 0.5 + 2*0.1


class TestOptimizationCheckpoint:
    """Test suite for optimization checkpointing."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoint files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        """Create OptimizationCheckpoint instance."""
        return OptimizationCheckpoint(
            checkpoint_dir=temp_checkpoint_dir,
            run_id="test_run_001"
        )

    def test_checkpoint_creation(self, checkpoint_manager):
        """Test creating optimization checkpoint."""
        checkpoint_data = {
            'generation': 10,
            'population': [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 0.1, 0.2, 0.3]
            ],
            'fitness_history': [0.45, 0.52, 0.61, 0.58],
            'best_individual': [0.5, 0.6, 0.7, 0.8],
            'best_fitness': 0.61,
            'optimizer_state': {
                'mean': [0.5, 0.5, 0.5, 0.5],
                'sigma': 0.3,
                'covariance_matrix': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            },
            'config': {'algorithm': 'CMA-ES', 'population_size': 20}
        }
        
        checkpoint_path = checkpoint_manager.save_checkpoint(checkpoint_data)
        
        assert os.path.exists(checkpoint_path)
        assert 'test_run_001' in checkpoint_path
        assert 'gen_10' in checkpoint_path

    def test_checkpoint_loading(self, checkpoint_manager):
        """Test loading optimization checkpoint."""
        # Create checkpoint first
        original_data = {
            'generation': 5,
            'population': [[0.1, 0.2], [0.3, 0.4]],
            'fitness_history': [0.3, 0.4, 0.5],
            'best_individual': [0.3, 0.4],
            'best_fitness': 0.5,
            'config': {'algorithm': 'CMA-ES'}
        }
        
        checkpoint_path = checkpoint_manager.save_checkpoint(original_data)
        
        # Load checkpoint
        loaded_data = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        assert loaded_data['generation'] == 5
        assert loaded_data['population'] == [[0.1, 0.2], [0.3, 0.4]]
        assert loaded_data['best_fitness'] == 0.5

    def test_find_latest_checkpoint(self, checkpoint_manager):
        """Test finding most recent checkpoint."""
        # Create multiple checkpoints
        for gen in [2, 5, 8]:
            data = {
                'generation': gen,
                'population': [[0.1, 0.2]],
                'best_fitness': 0.4 + gen * 0.05
            }
            checkpoint_manager.save_checkpoint(data)
        
        latest_path = checkpoint_manager.find_latest_checkpoint()
        
        assert latest_path is not None
        assert 'gen_8' in latest_path

    def test_checkpoint_cleanup(self, checkpoint_manager):
        """Test automatic checkpoint cleanup (keep only N most recent)."""
        # Create many checkpoints
        for gen in range(10):
            data = {
                'generation': gen,
                'population': [[0.1]],
                'best_fitness': 0.3 + gen * 0.02
            }
            checkpoint_manager.save_checkpoint(data)
        
        # Cleanup keeping only 3 most recent
        checkpoint_manager.cleanup_old_checkpoints(keep_count=3)
        
        remaining_files = os.listdir(checkpoint_manager.checkpoint_dir)
        checkpoint_files = [f for f in remaining_files if f.endswith('.pkl')]
        
        assert len(checkpoint_files) == 3


class TestResultManager:
    """Test suite for comprehensive result management."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for result management."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8}
            }
        }
        return ParameterSpace(config)

    @pytest.fixture
    def result_manager(self, temp_results_dir, parameter_space):
        """Create ResultManager instance."""
        return ResultManager(
            results_dir=temp_results_dir,
            parameter_space=parameter_space,
            run_id="test_optimization_001"
        )

    def test_result_manager_initialization(self, result_manager, temp_results_dir):
        """Test ResultManager initialization."""
        assert result_manager.results_dir == temp_results_dir
        assert result_manager.run_id == "test_optimization_001"
        assert hasattr(result_manager, 'logger')
        assert hasattr(result_manager, 'checkpoint_manager')

    def test_start_optimization_run(self, result_manager):
        """Test starting new optimization run."""
        config = {
            'algorithm': 'CMA-ES',
            'population_size': 20,
            'max_generations': 50
        }
        
        optimization_run = result_manager.start_optimization_run(config)
        
        assert isinstance(optimization_run, OptimizationRun)
        assert optimization_run.run_id == "test_optimization_001"
        assert optimization_run.config == config
        assert optimization_run.status == "running"

    def test_record_generation_results(self, result_manager):
        """Test recording complete generation results."""
        # Start optimization run
        config = {'algorithm': 'CMA-ES', 'population_size': 4}
        optimization_run = result_manager.start_optimization_run(config)
        
        # Create generation result
        generation_result = GenerationResult(
            generation=3,
            population_size=4,
            evaluations=[
                EvaluationResult("ind_1", {'n': 200, 'chunk_size': 8}, 0.6, 0.55, 0.05, 0.10, 2.1, 50),
                EvaluationResult("ind_2", {'n': 400, 'chunk_size': 12}, 0.72, 0.68, 0.04, 0.08, 2.3, 50),
                EvaluationResult("ind_3", {'n': 600, 'chunk_size': 16}, 0.58, 0.53, 0.07, 0.12, 2.5, 50),
                EvaluationResult("ind_4", {'n': 300, 'chunk_size': 10}, 0.65, 0.60, 0.05, 0.10, 2.2, 50)
            ],
            best_fitness=0.72,
            avg_fitness=0.64,
            worst_fitness=0.58,
            total_evaluation_time=9.1
        )
        
        result_manager.record_generation_results(generation_result)
        
        # Verify results were recorded
        summary = result_manager.get_optimization_summary()
        assert summary['generations_completed'] == 1
        assert summary['best_fitness'] == 0.72

    def test_save_and_load_checkpoint(self, result_manager):
        """Test checkpoint save/load functionality."""
        # Create sample optimization state
        state = {
            'generation': 7,
            'population': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            'best_individual': [0.3, 0.4],
            'best_fitness': 0.78,
            'optimizer_state': {'sigma': 0.25}
        }
        
        # Save checkpoint
        checkpoint_path = result_manager.save_checkpoint(state)
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        loaded_state = result_manager.load_checkpoint(checkpoint_path)
        assert loaded_state['generation'] == 7
        assert loaded_state['best_fitness'] == 0.78

    def test_resume_optimization(self, result_manager):
        """Test resuming optimization from checkpoint."""
        # Create and save checkpoint
        state = {
            'generation': 10,
            'population': [[0.2, 0.3], [0.4, 0.5]],
            'best_fitness': 0.82,
            'fitness_history': [0.6, 0.7, 0.75, 0.82]
        }
        checkpoint_path = result_manager.save_checkpoint(state)
        
        # Resume optimization
        resumed_state = result_manager.resume_optimization(checkpoint_path)
        
        assert resumed_state is not None
        assert resumed_state['generation'] == 10
        assert resumed_state['best_fitness'] == 0.82
        assert len(resumed_state['fitness_history']) == 4

    def test_get_optimization_summary(self, result_manager):
        """Test getting optimization summary statistics."""
        # Start run and record some results
        config = {'algorithm': 'CMA-ES'}
        result_manager.start_optimization_run(config)
        
        # Record multiple generations
        for gen in range(3):
            gen_result = GenerationResult(
                generation=gen,
                population_size=2,
                evaluations=[
                    EvaluationResult(f"ind_{gen}_1", {'n': 200}, 0.5 + gen*0.1, 0.45 + gen*0.1, 0.05, 0.10, 2.0, 20),
                    EvaluationResult(f"ind_{gen}_2", {'n': 400}, 0.6 + gen*0.08, 0.55 + gen*0.08, 0.05, 0.10, 2.1, 20)
                ],
                best_fitness=0.6 + gen*0.08,
                avg_fitness=0.55 + gen*0.09,
                worst_fitness=0.5 + gen*0.1,
                total_evaluation_time=4.1
            )
            result_manager.record_generation_results(gen_result)
        
        summary = result_manager.get_optimization_summary()
        
        assert summary['run_id'] == "test_optimization_001"
        assert summary['generations_completed'] == 3
        assert summary['best_fitness'] == 0.76  # 0.6 + 2*0.08
        assert 'total_evaluations' in summary
        assert 'total_runtime' in summary

    def test_export_results(self, result_manager):
        """Test exporting results to different formats."""
        # Start run and add some data
        config = {'algorithm': 'CMA-ES'}
        optimization_run = result_manager.start_optimization_run(config)
        
        gen_result = GenerationResult(
            generation=0,
            population_size=1,
            evaluations=[
                EvaluationResult("ind_1", {'n': 300, 'chunk_size': 10}, 0.7, 0.65, 0.05, 0.10, 2.2, 30)
            ],
            best_fitness=0.7,
            avg_fitness=0.7,
            worst_fitness=0.7,
            total_evaluation_time=2.2
        )
        result_manager.record_generation_results(gen_result)
        
        # Export to JSON
        json_path = result_manager.export_results('json')
        assert os.path.exists(json_path)
        assert json_path.endswith('.json')
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            exported_data = json.load(f)
            assert 'run_id' in exported_data
            assert 'config' in exported_data
            assert 'generations' in exported_data


class TestDataStructures:
    """Test suite for result management data structures."""

    def test_evaluation_result_creation(self):
        """Test EvaluationResult data structure."""
        eval_result = EvaluationResult(
            individual_id="test_001",
            parameters={'n': 500, 'chunk_size': 16},
            fitness_score=0.75,
            win_rate=0.68,
            variance_penalty=0.07,
            robustness_bonus=0.14,
            evaluation_time=3.2,
            games_played=100
        )
        
        assert eval_result.individual_id == "test_001"
        assert eval_result.parameters['n'] == 500
        assert eval_result.fitness_score == 0.75
        assert eval_result.evaluation_time == 3.2

    def test_generation_result_creation(self):
        """Test GenerationResult data structure."""
        evaluations = [
            EvaluationResult("ind_1", {'n': 200}, 0.6, 0.55, 0.05, 0.10, 2.0, 50),
            EvaluationResult("ind_2", {'n': 400}, 0.7, 0.65, 0.05, 0.10, 2.1, 50)
        ]
        
        gen_result = GenerationResult(
            generation=5,
            population_size=2,
            evaluations=evaluations,
            best_fitness=0.7,
            avg_fitness=0.65,
            worst_fitness=0.6,
            total_evaluation_time=4.1
        )
        
        assert gen_result.generation == 5
        assert gen_result.population_size == 2
        assert len(gen_result.evaluations) == 2
        assert gen_result.best_fitness == 0.7

    def test_optimization_run_creation(self):
        """Test OptimizationRun data structure."""
        config = {'algorithm': 'CMA-ES', 'population_size': 20}
        
        opt_run = OptimizationRun(
            run_id="run_001",
            config=config,
            start_time=datetime.now()
        )
        
        assert opt_run.run_id == "run_001"
        assert opt_run.config == config
        assert opt_run.status == "running"
        assert opt_run.generations == []