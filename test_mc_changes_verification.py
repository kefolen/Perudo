#!/usr/bin/env python3
"""
Verification script for the MonteCarloAgent changes.
This demonstrates that max_rounds and simulate_to_round_end parameters have been removed
and the agent now always simulates until the game ends naturally.
"""

import random
import time
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent

def test_parameters_removed():
    """Test that the removed parameters are no longer accepted."""
    print("Testing that removed parameters are no longer accepted...")
    
    # This should work (no obsolete parameters)
    try:
        agent_valid = MonteCarloAgent(
            name='test_valid',
            n=10,
            chunk_size=4,
            early_stop_margin=0.1,
            rng=random.Random(42)
        )
        print("  ✓ Agent creation without obsolete parameters: SUCCESS")
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False
    
    # This should fail (max_rounds parameter)
    try:
        agent_invalid = MonteCarloAgent(
            name='test_invalid',
            n=10,
            max_rounds=5,  # This parameter should no longer exist
            rng=random.Random(42)
        )
        print("  ✗ Agent creation with max_rounds should have failed but didn't")
        return False
    except TypeError as e:
        print(f"  ✓ Agent creation with max_rounds correctly failed: {e}")
    
    # This should fail (simulate_to_round_end parameter)
    try:
        agent_invalid2 = MonteCarloAgent(
            name='test_invalid2',
            n=10,
            simulate_to_round_end=True,  # This parameter should no longer exist
            rng=random.Random(42)
        )
        print("  ✗ Agent creation with simulate_to_round_end should have failed but didn't")
        return False
    except TypeError as e:
        print(f"  ✓ Agent creation with simulate_to_round_end correctly failed: {e}")
    
    return True


def test_agent_functionality():
    """Test that the agent still functions correctly after parameter removal."""
    print("\nTesting agent functionality...")
    
    simulator = PerudoSimulator(seed=42)
    
    # Create test observation
    obs = {
        'dice_counts': [2, 2, 2],
        'player_idx': 0,
        'my_hand': [1, 4],
        'current_bid': None,
        'maputa_active': False,
        'maputa_restrict_face': None,
        '_simulator': simulator
    }
    
    agent = MonteCarloAgent(
        name='test_functionality',
        n=5,  # Small n for faster testing
        chunk_size=2,
        rng=random.Random(42)
    )
    
    # Test that agent can select actions
    try:
        start_time = time.time()
        action = agent.select_action(obs)
        duration = time.time() - start_time
        
        print(f"  ✓ Agent selected action: {action} in {duration:.3f}s")
        
        # Action should be a valid tuple
        assert isinstance(action, tuple), f"Expected tuple, got {type(action)}"
        assert len(action) >= 1, f"Action tuple should have at least 1 element"
        
        print("  ✓ Action format is valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Agent functionality test failed: {e}")
        return False


def test_simulation_always_completes():
    """Test that simulations now always run to completion."""
    print("\nTesting that simulations run to completion...")
    
    simulator = PerudoSimulator(seed=42)
    
    obs = {
        'dice_counts': [1, 1, 1],  # Small game for faster completion
        'player_idx': 0,
        'my_hand': [3],
        'current_bid': None,
        'maputa_active': False,
        'maputa_restrict_face': None,
        '_simulator': simulator
    }
    
    agent = MonteCarloAgent(
        name='test_completion',
        n=3,  # Very small n for fast testing
        chunk_size=1,
        rng=random.Random(42)
    )
    
    try:
        # Generate determinization
        full_hands = agent.sample_determinization(obs)
        
        # Test direct simulation
        bid_action = ('bid', 1, 1)
        result = agent.simulate_from_determinization(full_hands, obs, bid_action)
        
        print(f"  ✓ Simulation completed with result: {result}")
        
        # Result should be a probability between 0 and 1
        assert 0.0 <= result <= 1.0, f"Result should be in [0,1], got {result}"
        
        print("  ✓ Simulation result is valid probability")
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation completion test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=== MonteCarloAgent Parameter Removal Verification ===")
    print("Verifying that max_rounds and simulate_to_round_end have been successfully removed\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_parameters_removed():
        tests_passed += 1
    
    if test_agent_functionality():
        tests_passed += 1
    
    if test_simulation_always_completes():
        tests_passed += 1
    
    print(f"\n=== Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All verification tests passed!")
        print("✓ max_rounds and simulate_to_round_end parameters successfully removed")
        print("✓ Agent now always simulates until game ends naturally")
    else:
        print("✗ Some verification tests failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)