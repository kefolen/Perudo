"""
Tests for ParameterSpace class - TDD Cycle 1.1: Parameter Schema

This module tests parameter encoding/decoding and JSON schema validation
for the evolutionary optimization framework.
"""
import pytest
import json
import os
from eval.hyperopt_framework import ParameterSpace


class TestParameterSpace:
    """Test suite for ParameterSpace parameter management."""
    
    @pytest.fixture
    def config_path(self, tmp_path):
        """Create a temporary config file for testing."""
        config_data = {
            "parameters": {
                "core_parameters": {
                    "n": {"type": "int", "min": 100, "max": 2000, "default": 400},
                    "chunk_size": {"type": "int", "min": 4, "max": 32, "default": 8},
                    "early_stop_margin": {"type": "float", "min": 0.05, "max": 0.3, "default": 0.15}
                },
                "utils_microparameters": {
                    "decay_factor": {"type": "float", "min": 0.6, "max": 0.95, "default": 0.8},
                    "scaling_factor": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0}
                }
            }
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        return str(config_file)
    
    @pytest.fixture
    def parameter_space(self, config_path):
        """Create ParameterSpace instance for testing."""
        return ParameterSpace(config_path)
    
    def test_parameter_space_initialization(self, parameter_space):
        """Test ParameterSpace loads config correctly."""
        assert parameter_space.config is not None
        assert "parameters" in parameter_space.config
        assert "core_parameters" in parameter_space.config["parameters"]
        assert "utils_microparameters" in parameter_space.config["parameters"]
    
    def test_parameter_bounds_extraction(self, parameter_space):
        """Test bounds extraction from config."""
        bounds = parameter_space.bounds
        
        # Check that bounds are extracted for all parameters
        expected_params = ['n', 'chunk_size', 'early_stop_margin', 'decay_factor', 'scaling_factor']
        for param in expected_params:
            assert param in bounds
            assert 'min' in bounds[param]
            assert 'max' in bounds[param]
            assert 'type' in bounds[param]
    
    def test_encode_genotype_valid_params(self, parameter_space):
        """Test encoding valid parameter dictionary to genotype vector."""
        params_dict = {
            'n': 500,
            'chunk_size': 16,
            'early_stop_margin': 0.2,
            'decay_factor': 0.85,
            'scaling_factor': 2.5
        }
        
        genotype = parameter_space.encode_genotype(params_dict)
        
        # Should return a list/array of normalized values
        assert isinstance(genotype, (list, tuple))
        assert len(genotype) == 5  # Number of parameters
        
        # All values should be normalized between 0 and 1
        for value in genotype:
            assert 0.0 <= value <= 1.0
    
    def test_decode_genotype_valid_vector(self, parameter_space):
        """Test decoding genotype vector to parameter dictionary."""
        # Normalized genotype vector (all values between 0 and 1)
        genotype = [0.5, 0.5, 0.5, 0.5, 0.5]  # Should give middle values
        
        params_dict = parameter_space.decode_genotype(genotype)
        
        # Check that all expected parameters are present
        expected_params = ['n', 'chunk_size', 'early_stop_margin', 'decay_factor', 'scaling_factor']
        for param in expected_params:
            assert param in params_dict
        
        # Check that values are within bounds and proper types
        assert isinstance(params_dict['n'], int)
        assert 100 <= params_dict['n'] <= 2000
        assert isinstance(params_dict['chunk_size'], int)
        assert 4 <= params_dict['chunk_size'] <= 32
        assert isinstance(params_dict['early_stop_margin'], float)
        assert 0.05 <= params_dict['early_stop_margin'] <= 0.3
    
    def test_encode_decode_roundtrip(self, parameter_space):
        """Test that encode->decode preserves parameter values."""
        original_params = {
            'n': 300,
            'chunk_size': 12,
            'early_stop_margin': 0.1,
            'decay_factor': 0.75,
            'scaling_factor': 1.8
        }
        
        # Encode then decode
        genotype = parameter_space.encode_genotype(original_params)
        decoded_params = parameter_space.decode_genotype(genotype)
        
        # Values should be approximately equal (within type conversion tolerance)
        assert decoded_params['n'] == original_params['n']  # Exact for int
        assert decoded_params['chunk_size'] == original_params['chunk_size']  # Exact for int
        assert abs(decoded_params['early_stop_margin'] - original_params['early_stop_margin']) < 0.001
        assert abs(decoded_params['decay_factor'] - original_params['decay_factor']) < 0.001
        assert abs(decoded_params['scaling_factor'] - original_params['scaling_factor']) < 0.001
    
    def test_validate_parameters_valid_dict(self, parameter_space):
        """Test parameter validation with valid parameter dictionary."""
        valid_params = {
            'n': 500,
            'chunk_size': 16,
            'early_stop_margin': 0.2,
            'decay_factor': 0.85,
            'scaling_factor': 2.5
        }
        
        # Should not raise any exceptions
        result = parameter_space.validate_parameters(valid_params)
        assert result is True  # or returns validated params
    
    def test_validate_parameters_out_of_bounds(self, parameter_space):
        """Test parameter validation with out-of-bounds values."""
        invalid_params = {
            'n': 50,  # Below minimum of 100
            'chunk_size': 16,
            'early_stop_margin': 0.5,  # Above maximum of 0.3
            'decay_factor': 0.85,
            'scaling_factor': 2.5
        }
        
        with pytest.raises((ValueError, AssertionError)):
            parameter_space.validate_parameters(invalid_params)
    
    def test_validate_parameters_wrong_type(self, parameter_space):
        """Test parameter validation with wrong parameter types."""
        invalid_params = {
            'n': 500.5,  # Should be int, not float
            'chunk_size': 16,
            'early_stop_margin': 0.2,
            'decay_factor': 0.85,
            'scaling_factor': 2.5
        }
        
        with pytest.raises((TypeError, ValueError)):
            parameter_space.validate_parameters(invalid_params)
    
    def test_validate_parameters_missing_required(self, parameter_space):
        """Test parameter validation with missing required parameters."""
        incomplete_params = {
            'n': 500,
            'chunk_size': 16,
            # Missing other required parameters
        }
        
        with pytest.raises((KeyError, ValueError)):
            parameter_space.validate_parameters(incomplete_params)
    
    def test_get_default_parameters(self, parameter_space):
        """Test getting default parameter values."""
        defaults = parameter_space.get_default_parameters()
        
        # Should contain all parameters with default values
        expected_defaults = {
            'n': 400,
            'chunk_size': 8, 
            'early_stop_margin': 0.15,
            'decay_factor': 0.8,
            'scaling_factor': 2.0
        }
        
        for param, expected_value in expected_defaults.items():
            assert param in defaults
            assert defaults[param] == expected_value
    
    def test_get_parameter_bounds(self, parameter_space):
        """Test getting parameter bounds."""
        bounds = parameter_space.get_parameter_bounds()
        
        # Should return min/max for each parameter
        for param in ['n', 'chunk_size', 'early_stop_margin', 'decay_factor', 'scaling_factor']:
            assert param in bounds
            assert 'min' in bounds[param]
            assert 'max' in bounds[param]
            assert bounds[param]['min'] < bounds[param]['max']
    
    def test_config_file_not_found(self):
        """Test behavior when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ParameterSpace("nonexistent_config.json")
    
    def test_malformed_config_file(self, tmp_path):
        """Test behavior with malformed JSON config."""
        bad_config = tmp_path / "bad_config.json"
        with open(bad_config, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            ParameterSpace(str(bad_config))