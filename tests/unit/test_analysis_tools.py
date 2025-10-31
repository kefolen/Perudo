"""
Tests for analysis tools - TDD Cycle 4.3: Analysis Tools

This module tests parameter interaction analysis, parameter sensitivity analysis,
and visualization tools for the evolutionary optimization framework.
"""
import pytest
import numpy as np
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from eval.analysis_tools import (
    ParameterAnalyzer, SensitivityAnalyzer, InteractionAnalyzer,
    VisualizationGenerator, OptimizationReportGenerator,
    ParameterImportance, InteractionStrength, SensitivityResult
)
from eval.hyperopt_framework import ParameterSpace
from eval.result_management import EvaluationResult, GenerationResult, OptimizationRun


class TestParameterImportance:
    """Test suite for parameter importance data structure."""
    
    def test_parameter_importance_creation(self):
        """Test ParameterImportance data structure creation."""
        importance = ParameterImportance(
            parameter_name="n",
            importance_score=0.75,
            variance_explained=0.65,
            confidence_interval=(0.70, 0.80),
            rank=1
        )
        
        assert importance.parameter_name == "n"
        assert importance.importance_score == 0.75
        assert importance.variance_explained == 0.65
        assert importance.confidence_interval == (0.70, 0.80)
        assert importance.rank == 1


class TestSensitivityAnalyzer:
    """Test suite for parameter sensitivity analysis."""
    
    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
                "early_stop_margin": {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15},
                "trust_learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1}
            }
        }
        return ParameterSpace(config)
    
    @pytest.fixture
    def sensitivity_analyzer(self, parameter_space):
        """Create SensitivityAnalyzer instance."""
        return SensitivityAnalyzer(parameter_space=parameter_space)
    
    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results for testing."""
        results = []
        for i in range(20):
            result = EvaluationResult(
                individual_id=f"ind_{i}",
                parameters={
                    'n': 200 + i * 50,
                    'chunk_size': 8 + (i % 4) * 2,
                    'early_stop_margin': 0.1 + (i % 3) * 0.05,
                    'trust_learning_rate': 0.05 + (i % 4) * 0.025
                },
                fitness_score=0.5 + (i % 10) * 0.05,
                win_rate=0.45 + (i % 10) * 0.05,
                variance_penalty=0.02 + (i % 5) * 0.01,
                robustness_bonus=0.08 + (i % 3) * 0.02,
                evaluation_time=2.0 + (i % 6) * 0.5,
                games_played=50
            )
            results.append(result)
        return results
    
    def test_sensitivity_analyzer_initialization(self, sensitivity_analyzer, parameter_space):
        """Test SensitivityAnalyzer initialization."""
        assert sensitivity_analyzer.parameter_space == parameter_space
        assert hasattr(sensitivity_analyzer, 'parameter_names')
        assert len(sensitivity_analyzer.parameter_names) == 4
    
    def test_sobol_sensitivity_analysis(self, sensitivity_analyzer, mock_evaluation_results):
        """Test Sobol sensitivity analysis."""
        sensitivity_results = sensitivity_analyzer.analyze_sobol_sensitivity(
            evaluation_results=mock_evaluation_results,
            n_samples=100
        )
        
        assert isinstance(sensitivity_results, dict)
        assert len(sensitivity_results) == 4  # One for each parameter
        
        for param_name, result in sensitivity_results.items():
            assert isinstance(result, SensitivityResult)
            assert param_name in ['n', 'chunk_size', 'early_stop_margin', 'trust_learning_rate']
            assert 0.0 <= result.first_order_index <= 1.0
            assert 0.0 <= result.total_order_index <= 1.0
            assert result.total_order_index >= result.first_order_index
    
    def test_morris_sensitivity_analysis(self, sensitivity_analyzer, mock_evaluation_results):
        """Test Morris sensitivity analysis."""
        sensitivity_results = sensitivity_analyzer.analyze_morris_sensitivity(
            evaluation_results=mock_evaluation_results,
            n_trajectories=10
        )
        
        assert isinstance(sensitivity_results, dict)
        assert len(sensitivity_results) == 4
        
        for param_name, result in sensitivity_results.items():
            assert isinstance(result, SensitivityResult)
            assert hasattr(result, 'mu_star')
            assert hasattr(result, 'sigma')
            assert result.mu_star >= 0.0  # Absolute mean effect
    
    def test_parameter_importance_ranking(self, sensitivity_analyzer, mock_evaluation_results):
        """Test parameter importance ranking."""
        importance_ranking = sensitivity_analyzer.rank_parameter_importance(
            evaluation_results=mock_evaluation_results
        )
        
        assert isinstance(importance_ranking, list)
        assert len(importance_ranking) == 4
        
        # Check that ranking is sorted by importance (highest first)
        for i in range(len(importance_ranking) - 1):
            assert importance_ranking[i].importance_score >= importance_ranking[i + 1].importance_score
            assert importance_ranking[i].rank == i + 1
        
        # Verify all parameters are included
        param_names = [imp.parameter_name for imp in importance_ranking]
        expected_params = ['n', 'chunk_size', 'early_stop_margin', 'trust_learning_rate']
        assert set(param_names) == set(expected_params)
    
    def test_sensitivity_confidence_intervals(self, sensitivity_analyzer, mock_evaluation_results):
        """Test confidence interval calculation for sensitivity indices."""
        confidence_intervals = sensitivity_analyzer.calculate_confidence_intervals(
            evaluation_results=mock_evaluation_results,
            confidence_level=0.95
        )
        
        assert isinstance(confidence_intervals, dict)
        
        for param_name, intervals in confidence_intervals.items():
            assert 'first_order' in intervals
            assert 'total_order' in intervals
            
            # Check confidence interval structure
            first_order_ci = intervals['first_order']
            total_order_ci = intervals['total_order']
            
            assert len(first_order_ci) == 2  # Lower and upper bounds
            assert len(total_order_ci) == 2
            assert first_order_ci[0] <= first_order_ci[1]  # Lower <= Upper
            assert total_order_ci[0] <= total_order_ci[1]


class TestInteractionAnalyzer:
    """Test suite for parameter interaction analysis."""
    
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
    def interaction_analyzer(self, parameter_space):
        """Create InteractionAnalyzer instance."""
        return InteractionAnalyzer(parameter_space=parameter_space)
    
    @pytest.fixture
    def mock_evaluation_data(self):
        """Create mock evaluation data with known interactions."""
        results = []
        for i in range(50):
            # Create artificial interaction between n and chunk_size
            n_val = 200 + i * 20
            chunk_val = 4 + (i % 8) * 2
            margin_val = 0.1 + (i % 5) * 0.04
            
            # Simulate interaction: performance improves when n and chunk_size are balanced
            interaction_effect = 0.1 if (n_val / chunk_val) > 40 and (n_val / chunk_val) < 80 else 0.0
            base_fitness = 0.5 + (i % 10) * 0.02
            
            result = EvaluationResult(
                individual_id=f"interaction_test_{i}",
                parameters={
                    'n': n_val,
                    'chunk_size': chunk_val,
                    'early_stop_margin': margin_val
                },
                fitness_score=base_fitness + interaction_effect,
                win_rate=0.45 + (base_fitness + interaction_effect) * 0.5,
                variance_penalty=0.02,
                robustness_bonus=0.08,
                evaluation_time=2.0,
                games_played=50
            )
            results.append(result)
        return results
    
    def test_interaction_analyzer_initialization(self, interaction_analyzer, parameter_space):
        """Test InteractionAnalyzer initialization."""
        assert interaction_analyzer.parameter_space == parameter_space
        assert hasattr(interaction_analyzer, 'parameter_names')
        assert len(interaction_analyzer.parameter_names) == 3
    
    def test_pairwise_interaction_analysis(self, interaction_analyzer, mock_evaluation_data):
        """Test pairwise parameter interaction analysis."""
        interactions = interaction_analyzer.analyze_pairwise_interactions(
            evaluation_results=mock_evaluation_data
        )
        
        assert isinstance(interactions, dict)
        
        # Should have interactions for all parameter pairs
        expected_pairs = [('n', 'chunk_size'), ('n', 'early_stop_margin'), ('chunk_size', 'early_stop_margin')]
        for pair in expected_pairs:
            # Check both orderings of the pair
            assert pair in interactions or (pair[1], pair[0]) in interactions
        
        # Check interaction strength data structure
        for pair, strength in interactions.items():
            assert isinstance(strength, InteractionStrength)
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert strength.strength_score >= 0.0
            assert strength.strength_score <= 1.0
    
    def test_higher_order_interaction_detection(self, interaction_analyzer, mock_evaluation_data):
        """Test detection of higher-order interactions (3-way, 4-way, etc.)."""
        higher_order_interactions = interaction_analyzer.analyze_higher_order_interactions(
            evaluation_results=mock_evaluation_data,
            max_order=3
        )
        
        assert isinstance(higher_order_interactions, dict)
        
        # Should include 3-way interaction
        three_way_key = ('n', 'chunk_size', 'early_stop_margin')
        assert three_way_key in higher_order_interactions or any(
            set(key) == set(three_way_key) for key in higher_order_interactions.keys()
        )
        
        for interaction_tuple, strength in higher_order_interactions.items():
            assert isinstance(strength, InteractionStrength)
            assert len(interaction_tuple) <= 3  # Max order constraint
    
    def test_interaction_strength_ranking(self, interaction_analyzer, mock_evaluation_data):
        """Test ranking of interaction strengths."""
        interaction_ranking = interaction_analyzer.rank_interactions(
            evaluation_results=mock_evaluation_data
        )
        
        assert isinstance(interaction_ranking, list)
        assert len(interaction_ranking) > 0
        
        # Check ranking is sorted (strongest first)
        for i in range(len(interaction_ranking) - 1):
            current_strength = interaction_ranking[i][1].strength_score
            next_strength = interaction_ranking[i + 1][1].strength_score
            assert current_strength >= next_strength
    
    def test_statistical_significance_testing(self, interaction_analyzer, mock_evaluation_data):
        """Test statistical significance of detected interactions."""
        significance_results = interaction_analyzer.test_interaction_significance(
            evaluation_results=mock_evaluation_data,
            alpha=0.05
        )
        
        assert isinstance(significance_results, dict)
        
        for interaction_key, result in significance_results.items():
            assert 'p_value' in result
            assert 'is_significant' in result
            assert 'test_statistic' in result
            
            assert 0.0 <= result['p_value'] <= 1.0
            assert isinstance(result['is_significant'], bool)
            
            # Significance should match p-value threshold
            expected_significant = result['p_value'] < 0.05
            assert result['is_significant'] == expected_significant


class TestParameterAnalyzer:
    """Test suite for comprehensive parameter analysis."""
    
    @pytest.fixture
    def parameter_space(self):
        """Create parameter space for testing."""
        config = {
            "parameters": {
                "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
                "early_stop_margin": {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15},
                "trust_learning_rate": {"type": "float", "min": 0.01, "max": 0.5, "default": 0.1}
            }
        }
        return ParameterSpace(config)
    
    @pytest.fixture
    def parameter_analyzer(self, parameter_space):
        """Create ParameterAnalyzer instance."""
        return ParameterAnalyzer(parameter_space=parameter_space)
    
    @pytest.fixture
    def mock_optimization_results(self):
        """Create mock complete optimization results."""
        generations = []
        for gen in range(5):
            evaluations = []
            for i in range(10):
                eval_result = EvaluationResult(
                    individual_id=f"gen_{gen}_ind_{i}",
                    parameters={
                        'n': 300 + gen * 100 + i * 20,
                        'chunk_size': 6 + gen + (i % 4),
                        'early_stop_margin': 0.1 + gen * 0.02 + (i % 3) * 0.02,
                        'trust_learning_rate': 0.08 + gen * 0.01 + (i % 4) * 0.01
                    },
                    fitness_score=0.4 + gen * 0.1 + (i % 5) * 0.02,
                    win_rate=0.35 + gen * 0.1 + (i % 5) * 0.02,
                    variance_penalty=0.03 - gen * 0.005,
                    robustness_bonus=0.05 + gen * 0.01,
                    evaluation_time=2.0 + (i % 3) * 0.5,
                    games_played=50
                )
                evaluations.append(eval_result)
            
            gen_result = GenerationResult(
                generation=gen,
                population_size=10,
                evaluations=evaluations,
                best_fitness=max(eval_result.fitness_score for eval_result in evaluations),
                avg_fitness=sum(eval_result.fitness_score for eval_result in evaluations) / len(evaluations),
                worst_fitness=min(eval_result.fitness_score for eval_result in evaluations),
                total_evaluation_time=sum(eval_result.evaluation_time for eval_result in evaluations)
            )
            generations.append(gen_result)
        
        return generations
    
    def test_parameter_analyzer_initialization(self, parameter_analyzer, parameter_space):
        """Test ParameterAnalyzer initialization."""
        assert parameter_analyzer.parameter_space == parameter_space
        assert hasattr(parameter_analyzer, 'sensitivity_analyzer')
        assert hasattr(parameter_analyzer, 'interaction_analyzer')
    
    def test_comprehensive_parameter_analysis(self, parameter_analyzer, mock_optimization_results):
        """Test comprehensive analysis of optimization results."""
        analysis_results = parameter_analyzer.analyze_optimization_results(
            generation_results=mock_optimization_results
        )
        
        assert isinstance(analysis_results, dict)
        assert 'parameter_importance' in analysis_results
        assert 'parameter_interactions' in analysis_results
        assert 'sensitivity_analysis' in analysis_results
        assert 'optimization_trends' in analysis_results
        
        # Check parameter importance results
        importance = analysis_results['parameter_importance']
        assert isinstance(importance, list)
        assert len(importance) == 4  # One for each parameter
        
        # Check interaction analysis results
        interactions = analysis_results['parameter_interactions']
        assert isinstance(interactions, dict)
        assert len(interactions) > 0
    
    def test_convergence_analysis(self, parameter_analyzer, mock_optimization_results):
        """Test analysis of parameter convergence over generations."""
        convergence_analysis = parameter_analyzer.analyze_convergence(
            generation_results=mock_optimization_results
        )
        
        assert isinstance(convergence_analysis, dict)
        
        for param_name in parameter_analyzer.parameter_space.get_parameter_names():
            assert param_name in convergence_analysis
            param_convergence = convergence_analysis[param_name]
            
            assert 'mean_trajectory' in param_convergence
            assert 'std_trajectory' in param_convergence
            assert 'convergence_rate' in param_convergence
            assert 'final_distribution' in param_convergence
            
            # Check trajectories have correct length
            assert len(param_convergence['mean_trajectory']) == len(mock_optimization_results)
            assert len(param_convergence['std_trajectory']) == len(mock_optimization_results)
    
    def test_parameter_correlation_analysis(self, parameter_analyzer, mock_optimization_results):
        """Test parameter correlation analysis."""
        correlation_matrix = parameter_analyzer.calculate_parameter_correlations(
            generation_results=mock_optimization_results
        )
        
        param_names = parameter_analyzer.parameter_space.get_parameter_names()
        
        # Should be symmetric matrix
        assert correlation_matrix.shape == (len(param_names), len(param_names))
        
        # Diagonal should be 1.0 (self-correlation)
        for i in range(len(param_names)):
            assert abs(correlation_matrix[i, i] - 1.0) < 0.001
        
        # Matrix should be symmetric
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                assert abs(correlation_matrix[i, j] - correlation_matrix[j, i]) < 0.001
                # Correlation values should be in [-1, 1]
                assert -1.0 <= correlation_matrix[i, j] <= 1.0
    
    def test_optimal_region_identification(self, parameter_analyzer, mock_optimization_results):
        """Test identification of optimal parameter regions."""
        optimal_regions = parameter_analyzer.identify_optimal_regions(
            generation_results=mock_optimization_results,
            top_percentile=0.2  # Top 20% of results
        )
        
        assert isinstance(optimal_regions, dict)
        
        for param_name in parameter_analyzer.parameter_space.get_parameter_names():
            assert param_name in optimal_regions
            region = optimal_regions[param_name]
            
            assert 'mean' in region
            assert 'std' in region
            assert 'min' in region
            assert 'max' in region
            assert 'confidence_interval' in region
            
            # Verify confidence interval structure
            ci = region['confidence_interval']
            assert len(ci) == 2
            assert ci[0] <= ci[1]  # Lower bound <= Upper bound


class TestVisualizationGenerator:
    """Test suite for visualization and reporting tools."""
    
    @pytest.fixture
    def visualization_generator(self):
        """Create VisualizationGenerator instance."""
        return VisualizationGenerator(output_dir="test_output")
    
    def test_visualization_generator_initialization(self, visualization_generator):
        """Test VisualizationGenerator initialization."""
        assert visualization_generator.output_dir == "test_output"
        assert hasattr(visualization_generator, 'plot_configs')
    
    def test_parameter_importance_plot(self, visualization_generator):
        """Test parameter importance visualization."""
        # Mock parameter importance data
        importance_data = [
            ParameterImportance("n", 0.75, 0.65, (0.70, 0.80), 1),
            ParameterImportance("chunk_size", 0.45, 0.35, (0.40, 0.50), 2),
            ParameterImportance("early_stop_margin", 0.25, 0.15, (0.20, 0.30), 3),
        ]
        
        plot_path = visualization_generator.generate_importance_plot(
            importance_data=importance_data,
            title="Parameter Importance Analysis"
        )
        
        assert isinstance(plot_path, str)
        assert "importance" in plot_path.lower()
    
    def test_interaction_heatmap(self, visualization_generator):
        """Test parameter interaction heatmap generation."""
        # Mock interaction matrix
        interaction_matrix = np.array([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        parameter_names = ["n", "chunk_size", "early_stop_margin"]
        
        plot_path = visualization_generator.generate_interaction_heatmap(
            interaction_matrix=interaction_matrix,
            parameter_names=parameter_names,
            title="Parameter Interaction Strengths"
        )
        
        assert isinstance(plot_path, str)
        assert "interaction" in plot_path.lower()
    
    def test_convergence_plots(self, visualization_generator):
        """Test convergence trajectory visualization."""
        # Mock convergence data
        convergence_data = {
            'n': {
                'mean_trajectory': [400, 420, 380, 350, 360],
                'std_trajectory': [100, 80, 60, 40, 30],
                'generations': list(range(5))
            },
            'chunk_size': {
                'mean_trajectory': [8, 10, 12, 11, 10],
                'std_trajectory': [3, 2.5, 2, 1.5, 1],
                'generations': list(range(5))
            }
        }
        
        plot_path = visualization_generator.generate_convergence_plots(
            convergence_data=convergence_data,
            title="Parameter Convergence Analysis"
        )
        
        assert isinstance(plot_path, str)
        assert "convergence" in plot_path.lower()


class TestOptimizationReportGenerator:
    """Test suite for comprehensive optimization report generation."""
    
    @pytest.fixture
    def report_generator(self):
        """Create OptimizationReportGenerator instance."""
        return OptimizationReportGenerator(template_dir="templates")
    
    def test_report_generator_initialization(self, report_generator):
        """Test OptimizationReportGenerator initialization."""
        assert report_generator.template_dir == "templates"
        assert hasattr(report_generator, 'supported_formats')
    
    def test_comprehensive_report_generation(self, report_generator):
        """Test generation of comprehensive optimization report."""
        # Mock analysis results
        analysis_results = {
            'parameter_importance': [
                ParameterImportance("n", 0.75, 0.65, (0.70, 0.80), 1),
                ParameterImportance("chunk_size", 0.45, 0.35, (0.40, 0.50), 2)
            ],
            'parameter_interactions': {
                ('n', 'chunk_size'): InteractionStrength(0.6, 0.05, True)
            },
            'sensitivity_analysis': {
                'n': SensitivityResult(0.7, 0.8, 0.1, 0.15),
                'chunk_size': SensitivityResult(0.4, 0.5, 0.08, 0.12)
            },
            'optimization_summary': {
                'best_fitness': 0.85,
                'generations': 10,
                'total_evaluations': 200,
                'convergence_achieved': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            report_path = report_generator.generate_comprehensive_report(
                analysis_results=analysis_results,
                output_path=tmp_file.name,
                format='html'
            )
            
            assert report_path == tmp_file.name
    
    def test_json_report_generation(self, report_generator):
        """Test JSON format report generation."""
        analysis_data = {
            'parameter_importance': "mock_data",
            'optimization_summary': {'best_fitness': 0.8}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            report_path = report_generator.generate_json_report(
                analysis_results=analysis_data,
                output_path=tmp_file.name
            )
            
            assert report_path == tmp_file.name
            
            # Verify JSON is valid
            with open(tmp_file.name, 'r') as f:
                loaded_data = json.load(f)
                assert 'parameter_importance' in loaded_data
                assert 'optimization_summary' in loaded_data