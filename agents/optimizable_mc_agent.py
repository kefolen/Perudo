"""
OptimizableMCAgent - Parameter-Injectable Monte Carlo Agent

This module provides a Monte Carlo agent with runtime parameter modification
capabilities for evolutionary optimization. Extends the base MonteCarloAgent
with dynamic parameter updates.

Key Features:
- Runtime parameter injection for core MC parameters
- Utils microparameter storage and retrieval
- Type conversion and bounds enforcement
- Backward compatibility with existing MonteCarloAgent interface
"""
from agents.mc_agent import MonteCarloAgent
from typing import Dict, Any, Union


class OptimizableMCAgent(MonteCarloAgent):
    """
    MC Agent with runtime parameter modification capabilities.
    
    Extends MonteCarloAgent to support dynamic parameter updates during
    evolutionary optimization. Maintains full backward compatibility while
    adding parameter injection functionality.
    """
    
    def __init__(self, **base_params):
        """
        Initialize OptimizableMCAgent with base parameters.
        
        Args:
            **base_params: All parameters accepted by MonteCarloAgent
        """
        super().__init__(**base_params)
        
        # Storage for utils microparameters
        self._utils_params = {}
    
    def update_parameters(self, params_dict: Dict[str, Union[int, float]]):
        """
        Update agent parameters at runtime.
        
        Handles both core MC agent parameters and utils microparameters.
        Provides type conversion and basic bounds enforcement.
        
        Args:
            params_dict: Dictionary of parameter names to values
        """
        if not params_dict:
            return  # Handle empty parameter updates gracefully
        
        # Update core MC parameters
        self._update_core_parameters(params_dict)
        
        # Update utils microparameters
        self._update_utils_parameters(params_dict)
    
    def _update_core_parameters(self, params_dict: Dict[str, Union[int, float]]):
        """Update core MonteCarloAgent parameters."""
        
        # Number of simulations (n -> N)
        if 'n' in params_dict:
            self.N = max(1, int(params_dict['n']))  # Enforce minimum of 1
        
        # Chunk size for batch processing
        if 'chunk_size' in params_dict:
            self.chunk_size = max(1, int(params_dict['chunk_size']))  # Enforce minimum of 1
        
        # Early stopping margin
        if 'early_stop_margin' in params_dict:
            self.early_stop_margin = max(0.0, float(params_dict['early_stop_margin']))  # Enforce non-negative
        
        # Trust learning rate
        if 'trust_learning_rate' in params_dict:
            self.trust_learning_rate = float(params_dict['trust_learning_rate'])
        
        # History memory rounds
        if 'history_memory_rounds' in params_dict:
            self.history_memory_rounds = max(1, int(params_dict['history_memory_rounds']))  # Enforce minimum of 1
        
        # Number of workers (if parallel processing is enabled)
        if 'num_workers' in params_dict:
            self.num_workers = max(1, int(params_dict['num_workers']))  # Enforce minimum of 1
    
    def _update_utils_parameters(self, params_dict: Dict[str, Union[int, float]]):
        """Update utils microparameters storage."""
        
        # Known utils microparameters
        utils_param_names = [
            'decay_factor',
            'scaling_factor', 
            'face_weight_boost',
            'min_weight_threshold'
        ]
        
        for param_name in utils_param_names:
            if param_name in params_dict:
                self._utils_params[param_name] = float(params_dict[param_name])
    
    def get_utils_param(self, param_name: str, default: Union[int, float]) -> Union[int, float]:
        """
        Get utils parameter for mc_utils functions.
        
        This method is used by modified mc_utils functions to retrieve
        dynamic parameter values with fallback to defaults.
        
        Args:
            param_name: Name of the parameter to retrieve
            default: Default value if parameter is not set
            
        Returns:
            Parameter value if set, otherwise default value
        """
        return self._utils_params.get(param_name, default)