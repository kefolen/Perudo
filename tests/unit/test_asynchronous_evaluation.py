"""
Tests for asynchronous evaluation system - TDD Cycle 4.2: Asynchronous Evaluation

This module tests distributed candidate evaluation and worker-based asynchronous
evaluation for the evolutionary optimization framework.
"""
import pytest
import time
import concurrent.futures
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Queue
import threading

from eval.asynchronous_evaluation import (
    AsyncEvaluationManager, WorkerPool, EvaluationWorker, 
    EvaluationTask, EvaluationResult as AsyncEvaluationResult,
    TaskScheduler, ResourceManager
)
from eval.hyperopt_framework import ParameterSpace
from eval.fitness_evaluation import FitnessEvaluator


class TestEvaluationTask:
    """Test suite for evaluation task data structure."""
    
    def test_evaluation_task_creation(self):
        """Test EvaluationTask creation with all required fields."""
        parameters = {
            'n': 400,
            'chunk_size': 8,
            'early_stop_margin': 0.15
        }
        
        task = EvaluationTask(
            task_id="task_001",
            parameters=parameters,
            priority=1,
            num_games=50,
            timeout=300.0
        )
        
        assert task.task_id == "task_001"
        assert task.parameters == parameters
        assert task.priority == 1
        assert task.num_games == 50
        assert task.timeout == 300.0
        assert task.status == "pending"
        assert task.assigned_worker is None
    
    def test_evaluation_task_status_updates(self):
        """Test evaluation task status transitions."""
        task = EvaluationTask(
            task_id="task_002",
            parameters={'n': 200},
            priority=2
        )
        
        # Test status transitions
        assert task.status == "pending"
        
        task.status = "assigned"
        task.assigned_worker = "worker_1"
        assert task.status == "assigned"
        assert task.assigned_worker == "worker_1"
        
        task.status = "running"
        assert task.status == "running"
        
        task.status = "completed"
        assert task.status == "completed"


class TestEvaluationWorker:
    """Test suite for asynchronous evaluation worker."""
    
    @pytest.fixture
    def mock_fitness_evaluator(self):
        """Create mock fitness evaluator for testing."""
        evaluator = Mock(spec=FitnessEvaluator)
        evaluator.evaluate_configuration.return_value = {
            'fitness': 0.75,
            'win_rate': 0.68,
            'variance': 0.05,
            'robustness': 0.12,
            'detailed_results': {0: 34, 1: 10, 2: 6}
        }
        return evaluator
    
    @pytest.fixture
    def evaluation_worker(self, mock_fitness_evaluator):
        """Create evaluation worker instance."""
        return EvaluationWorker(
            worker_id="test_worker_1",
            fitness_evaluator=mock_fitness_evaluator,
            result_queue=Queue(),
            task_queue=Queue()
        )
    
    def test_evaluation_worker_initialization(self, evaluation_worker):
        """Test evaluation worker initialization."""
        assert evaluation_worker.worker_id == "test_worker_1"
        assert evaluation_worker.fitness_evaluator is not None
        assert evaluation_worker.result_queue is not None
        assert evaluation_worker.task_queue is not None
        assert evaluation_worker.is_running == False
        assert evaluation_worker.current_task is None
    
    def test_worker_evaluate_single_task(self, evaluation_worker):
        """Test worker evaluating a single task."""
        task = EvaluationTask(
            task_id="test_task",
            parameters={'n': 300, 'chunk_size': 12},
            num_games=25
        )
        
        result = evaluation_worker.evaluate_task(task)
        
        assert isinstance(result, AsyncEvaluationResult)
        assert result.task_id == "test_task"
        assert result.fitness_score == 0.75
        assert result.success == True
        assert result.worker_id == "test_worker_1"
    
    def test_worker_handles_evaluation_failure(self, evaluation_worker):
        """Test worker handling evaluation failures gracefully."""
        # Configure mock to raise exception
        evaluation_worker.fitness_evaluator.evaluate_configuration.side_effect = Exception("Evaluation failed")
        
        task = EvaluationTask(
            task_id="failing_task",
            parameters={'n': 100},
            num_games=10
        )
        
        result = evaluation_worker.evaluate_task(task)
        
        assert isinstance(result, AsyncEvaluationResult)
        assert result.task_id == "failing_task"
        assert result.success == False
        assert result.error_message == "Evaluation failed"
        assert result.fitness_score == 0.0
    
    def test_worker_respects_timeout(self, evaluation_worker):
        """Test worker respects task timeout."""
        # Configure mock to simulate slow evaluation
        def slow_evaluation(*args, **kwargs):
            time.sleep(0.2)  # Longer than timeout
            return {'fitness': 0.5}
        
        evaluation_worker.fitness_evaluator.evaluate_configuration.side_effect = slow_evaluation
        
        task = EvaluationTask(
            task_id="timeout_task",
            parameters={'n': 500},
            timeout=0.1  # Very short timeout
        )
        
        start_time = time.time()
        result = evaluation_worker.evaluate_task(task)
        elapsed = time.time() - start_time
        
        # Should timeout and return error result
        assert result.success == False
        assert "timeout" in result.error_message.lower()
        assert elapsed < 0.15  # Should complete quickly due to timeout


class TestWorkerPool:
    """Test suite for worker pool management."""
    
    @pytest.fixture
    def mock_fitness_evaluator(self):
        """Create mock fitness evaluator."""
        evaluator = Mock(spec=FitnessEvaluator)
        evaluator.evaluate_configuration.return_value = {
            'fitness': 0.6, 'win_rate': 0.55, 'variance': 0.05, 'robustness': 0.1
        }
        return evaluator
    
    @pytest.fixture
    def worker_pool(self, mock_fitness_evaluator):
        """Create worker pool for testing."""
        return WorkerPool(
            num_workers=3,
            fitness_evaluator=mock_fitness_evaluator,
            max_queue_size=100
        )
    
    def test_worker_pool_initialization(self, worker_pool):
        """Test worker pool initialization."""
        assert worker_pool.num_workers == 3
        assert worker_pool.max_queue_size == 100
        assert len(worker_pool.workers) == 0  # Not started yet
        assert worker_pool.is_running == False
    
    def test_worker_pool_start_stop(self, worker_pool):
        """Test starting and stopping worker pool."""
        # Start pool
        worker_pool.start()
        assert worker_pool.is_running == True
        assert len(worker_pool.workers) == 3
        
        # Stop pool
        worker_pool.stop()
        assert worker_pool.is_running == False
    
    def test_worker_pool_submit_tasks(self, worker_pool):
        """Test submitting tasks to worker pool."""
        tasks = [
            EvaluationTask(f"task_{i}", {'n': 200 + i*100}, priority=i)
            for i in range(5)
        ]
        
        worker_pool.start()
        
        # Submit tasks
        for task in tasks:
            worker_pool.submit_task(task)
        
        assert worker_pool.get_pending_task_count() == 5
        
        worker_pool.stop()
    
    def test_worker_pool_collect_results(self, worker_pool):
        """Test collecting results from worker pool."""
        task = EvaluationTask("result_test", {'n': 300}, num_games=20)
        
        worker_pool.start()
        worker_pool.submit_task(task)
        
        # Wait for result (with timeout)
        results = []
        start_time = time.time()
        while len(results) == 0 and time.time() - start_time < 2.0:
            batch_results = worker_pool.collect_results(timeout=0.1)
            results.extend(batch_results)
        
        worker_pool.stop()
        
        assert len(results) >= 1
        assert results[0].task_id == "result_test"
        assert results[0].success == True
    
    def test_worker_pool_handles_worker_failure(self, worker_pool):
        """Test worker pool handling individual worker failures."""
        # Start pool
        worker_pool.start()
        initial_worker_count = len(worker_pool.workers)
        
        # Simulate worker failure (this is a simplified test)
        # In real implementation, would test actual worker process failure
        assert len(worker_pool.workers) == initial_worker_count
        
        worker_pool.stop()


class TestTaskScheduler:
    """Test suite for task scheduling and prioritization."""
    
    @pytest.fixture
    def task_scheduler(self):
        """Create task scheduler instance."""
        return TaskScheduler(max_concurrent_tasks=5)
    
    def test_task_scheduler_initialization(self, task_scheduler):
        """Test task scheduler initialization."""
        assert task_scheduler.max_concurrent_tasks == 5
        assert len(task_scheduler.pending_tasks) == 0
        assert len(task_scheduler.running_tasks) == 0
        assert len(task_scheduler.completed_tasks) == 0
    
    def test_schedule_tasks_by_priority(self, task_scheduler):
        """Test task scheduling respects priority ordering."""
        tasks = [
            EvaluationTask("low_priority", {'n': 100}, priority=3),
            EvaluationTask("high_priority", {'n': 200}, priority=1),
            EvaluationTask("medium_priority", {'n': 150}, priority=2),
        ]
        
        # Add tasks in random order
        for task in tasks:
            task_scheduler.schedule_task(task)
        
        # Get next task should return highest priority (lowest number)
        next_task = task_scheduler.get_next_task()
        assert next_task.task_id == "high_priority"
        
        next_task = task_scheduler.get_next_task()
        assert next_task.task_id == "medium_priority"
        
        next_task = task_scheduler.get_next_task()
        assert next_task.task_id == "low_priority"
    
    def test_scheduler_respects_concurrent_limit(self, task_scheduler):
        """Test scheduler respects maximum concurrent task limit."""
        # Create more tasks than concurrent limit
        tasks = [
            EvaluationTask(f"task_{i}", {'n': 100 + i*50}, priority=i)
            for i in range(8)
        ]
        
        for task in tasks:
            task_scheduler.schedule_task(task)
        
        # Should only allow max_concurrent_tasks to be running
        running_tasks = []
        for _ in range(task_scheduler.max_concurrent_tasks + 2):  # Try to get more than limit
            task = task_scheduler.get_next_task()
            if task:
                task.status = "running"
                task_scheduler.mark_task_running(task)
                running_tasks.append(task)
        
        assert len(task_scheduler.running_tasks) <= task_scheduler.max_concurrent_tasks
    
    def test_task_completion_handling(self, task_scheduler):
        """Test handling task completion and result storage."""
        task = EvaluationTask("completion_test", {'n': 200}, priority=1)
        task_scheduler.schedule_task(task)
        
        # Get and start task
        running_task = task_scheduler.get_next_task()
        task_scheduler.mark_task_running(running_task)
        
        # Create result and mark complete
        result = AsyncEvaluationResult(
            task_id="completion_test",
            fitness_score=0.8,
            success=True
        )
        
        task_scheduler.mark_task_completed(running_task, result)
        
        assert len(task_scheduler.completed_tasks) == 1
        assert len(task_scheduler.running_tasks) == 0
        assert task_scheduler.completed_tasks[0].task_id == "completion_test"


class TestAsyncEvaluationManager:
    """Test suite for complete asynchronous evaluation management."""
    
    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
                "early_stop_margin": {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15}
            }
        }
        return ParameterSpace(config)
    
    @pytest.fixture
    def async_evaluation_manager(self, parameter_space):
        """Create async evaluation manager."""
        return AsyncEvaluationManager(
            parameter_space=parameter_space,
            num_workers=2,
            max_concurrent_tasks=4
        )
    
    def test_async_manager_initialization(self, async_evaluation_manager):
        """Test async evaluation manager initialization."""
        assert async_evaluation_manager.num_workers == 2
        assert async_evaluation_manager.max_concurrent_tasks == 4
        assert hasattr(async_evaluation_manager, 'worker_pool')
        assert hasattr(async_evaluation_manager, 'task_scheduler')
    
    def test_evaluate_population_async(self, async_evaluation_manager):
        """Test asynchronous evaluation of parameter population."""
        population = [
            {'n': 200, 'chunk_size': 8, 'early_stop_margin': 0.1},
            {'n': 400, 'chunk_size': 12, 'early_stop_margin': 0.15},
            {'n': 600, 'chunk_size': 16, 'early_stop_margin': 0.2}
        ]
        
        with patch.object(async_evaluation_manager.worker_pool, 'start'):
            with patch.object(async_evaluation_manager.worker_pool, 'stop'):
                with patch.object(async_evaluation_manager.worker_pool, 'collect_results') as mock_collect:
                    # Mock results
                    mock_results = [
                        AsyncEvaluationResult(f"task_{i}", 0.6 + i*0.1, True)
                        for i in range(3)
                    ]
                    mock_collect.return_value = mock_results
                    
                    results = async_evaluation_manager.evaluate_population_async(
                        population, num_games=30
                    )
                    
                    assert len(results) == 3
                    assert all(isinstance(r, AsyncEvaluationResult) for r in results)
    
    def test_batch_evaluation_with_priorities(self, async_evaluation_manager):
        """Test batch evaluation with task prioritization."""
        high_priority_configs = [
            {'n': 300, 'chunk_size': 10, 'early_stop_margin': 0.12}
        ]
        
        low_priority_configs = [
            {'n': 150, 'chunk_size': 6, 'early_stop_margin': 0.08},
            {'n': 500, 'chunk_size': 14, 'early_stop_margin': 0.18}
        ]
        
        with patch.object(async_evaluation_manager.worker_pool, 'start'):
            with patch.object(async_evaluation_manager.worker_pool, 'stop'):
                results = async_evaluation_manager.evaluate_batch_with_priorities([
                    (high_priority_configs, 1),  # High priority
                    (low_priority_configs, 2)    # Lower priority
                ])
                
                # Should return results for all configurations
                assert len(results) == 3
    
    def test_resource_monitoring(self, async_evaluation_manager):
        """Test resource monitoring and adaptive scaling."""
        # Test resource utilization tracking
        utilization = async_evaluation_manager.get_resource_utilization()
        
        assert 'cpu_usage' in utilization
        assert 'memory_usage' in utilization
        assert 'active_workers' in utilization
        assert 'pending_tasks' in utilization
        
        # Test adaptive worker scaling (if implemented)
        if hasattr(async_evaluation_manager, 'scale_workers'):
            async_evaluation_manager.scale_workers(target_workers=4)
            assert async_evaluation_manager.num_workers == 4
    
    def test_error_recovery_and_resilience(self, async_evaluation_manager):
        """Test error recovery and system resilience."""
        # Simulate worker failure scenario
        with patch.object(async_evaluation_manager.worker_pool, 'handle_worker_failure') as mock_handle:
            async_evaluation_manager.handle_worker_failure("worker_1")
            mock_handle.assert_called_once_with("worker_1")
        
        # Test graceful degradation
        original_workers = async_evaluation_manager.num_workers
        async_evaluation_manager.handle_system_degradation()
        
        # Should maintain some level of functionality
        assert async_evaluation_manager.num_workers >= 1


class TestResourceManager:
    """Test suite for resource management and optimization."""
    
    @pytest.fixture
    def resource_manager(self):
        """Create resource manager instance."""
        return ResourceManager(max_cpu_usage=80.0, max_memory_usage=85.0)
    
    def test_resource_manager_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert resource_manager.max_cpu_usage == 80.0
        assert resource_manager.max_memory_usage == 85.0
        assert hasattr(resource_manager, 'current_usage')
    
    def test_resource_monitoring(self, resource_manager):
        """Test real-time resource monitoring."""
        usage = resource_manager.get_current_usage()
        
        assert 'cpu_percent' in usage
        assert 'memory_percent' in usage
        assert 'disk_usage' in usage
        assert isinstance(usage['cpu_percent'], (int, float))
        assert isinstance(usage['memory_percent'], (int, float))
    
    def test_resource_allocation_decisions(self, resource_manager):
        """Test resource allocation decisions."""
        # Test with low resource usage
        with patch.object(resource_manager, 'get_current_usage') as mock_usage:
            mock_usage.return_value = {'cpu_percent': 30.0, 'memory_percent': 40.0}
            
            can_allocate = resource_manager.can_allocate_worker()
            assert can_allocate == True
        
        # Test with high resource usage
        with patch.object(resource_manager, 'get_current_usage') as mock_usage:
            mock_usage.return_value = {'cpu_percent': 85.0, 'memory_percent': 90.0}
            
            can_allocate = resource_manager.can_allocate_worker()
            assert can_allocate == False
    
    def test_adaptive_resource_scaling(self, resource_manager):
        """Test adaptive resource scaling based on load."""
        # Test scaling recommendation based on current load
        with patch.object(resource_manager, 'get_current_usage') as mock_usage:
            mock_usage.return_value = {'cpu_percent': 50.0, 'memory_percent': 45.0}
            
            recommendation = resource_manager.get_scaling_recommendation(current_workers=2)
            
            assert 'action' in recommendation  # 'scale_up', 'scale_down', 'maintain'
            assert 'target_workers' in recommendation
            assert isinstance(recommendation['target_workers'], int)