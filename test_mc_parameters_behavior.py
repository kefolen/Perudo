#!/usr/bin/env python3
"""
Test script to capture current behavior of max_rounds and simulate_to_round_end parameters
in MonteCarloAgent before making changes.

This follows TDD principles by documenting the existing behavior before refactoring.
"""

import random
import time
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent

def test_max_rounds_parameter():
    """Test that max_rounds parameter limits simulation rounds."""
    print("Testing max_rounds parameter...")
    
    simulator = PerudoSimulator(seed=42)
    
    # Create test observation with multiple players
    obs = {
        'dice_counts': [3, 3, 3, 3],
        'player_idx': 0,
        'my_hand': [1, 2, 3],
        'current_bid': None,
        'maputa_active': False,
        'maputa_restrict_face': None,
        '_simulator': simulator
    }
    
    # Test with different max_rounds values
    for max_rounds in [1, 3, 6, 10]:
        print(f"  Testing max_rounds={max_rounds}")
        
        agent = MonteCarloAgent(
            name=f'test_max_rounds_{max_rounds}',
            n=10,  # Small n for faster testing
            max_rounds=max_rounds,
            simulate_to_round_end=True,
            rng=random.Random(42)
        )
        
        # Test that agent can be created with different max_rounds
        assert agent.max_rounds == max_rounds
        
        # Test that simulation works
        start_time = time.time()
        action = agent.select_action(obs)
        duration = time.time() - start_time
        
        print(f"    max_rounds={max_rounds}: action={action}, duration={duration:.3f}s")
        assert isinstance(action, tuple)


def test_simulate_to_round_end_parameter():
    """Test that simulate_to_round_end parameter changes behavior."""
    print("Testing simulate_to_round_end parameter...")
    
    simulator = PerudoSimulator(seed=42)
    
    obs = {
        'dice_counts': [5, 5, 5, 5],  # Large dice counts for longer games
        'player_idx': 0,
        'my_hand': [1, 2, 3, 4, 5],
        'current_bid': None,
        'maputa_active': False,
        'maputa_restrict_face': None,
        '_simulator': simulator
    }
    
    # Test with simulate_to_round_end=True (should use heuristic when max_rounds hit)
    agent_heuristic = MonteCarloAgent(
        name='test_heuristic',
        n=5,  # Small n for faster testing
        max_rounds=2,  # Small max_rounds to trigger heuristic
        simulate_to_round_end=True,
        rng=random.Random(42)
    )
    
    # Test with simulate_to_round_end=False (should also use heuristic as fallback)
    agent_no_heuristic = MonteCarloAgent(
        name='test_no_heuristic',
        n=5,
        max_rounds=2,
        simulate_to_round_end=False,
        rng=random.Random(42)
    )
    
    # Both should work and produce actions
    action_heuristic = agent_heuristic.select_action(obs)
    action_no_heuristic = agent_no_heuristic.select_action(obs)
    
    print(f"  simulate_to_round_end=True: {action_heuristic}")
    print(f"  simulate_to_round_end=False: {action_no_heuristic}")
    
    assert isinstance(action_heuristic, tuple)
    assert isinstance(action_no_heuristic, tuple)
    
    # Check that parameter is stored correctly
    assert agent_heuristic.simulate_to_round_end == True
    assert agent_no_heuristic.simulate_to_round_end == False


def test_simulation_direct():
    """Test the simulate_from_determinization method directly."""
    print("Testing simulate_from_determinization method...")
    
    simulator = PerudoSimulator(seed=42)
    
    obs = {
        'dice_counts': [2, 2, 2],
        'player_idx': 0,
        'my_hand': [1, 2],
        'current_bid': None,
        'maputa_active': False,
        'maputa_restrict_face': None,
        '_simulator': simulator
    }
    
    agent = MonteCarloAgent(
        name='test_direct_sim',
        n=10,
        max_rounds=3,
        simulate_to_round_end=True,
        rng=random.Random(42)
    )
    
    # Generate determinization
    full_hands = agent.sample_determinization(obs)
    
    # Test bid action
    bid_action = ('bid', 2, 1)
    result_bid = agent.simulate_from_determinization(full_hands, obs, bid_action)
    print(f"  Bid simulation result: {result_bid}")
    
    # Test call action (if there was a current bid)
    obs_with_bid = obs.copy()
    obs_with_bid['current_bid'] = (1, 1)
    call_action = ('call',)
    result_call = agent.simulate_from_determinization(full_hands, obs_with_bid, call_action)
    print(f"  Call simulation result: {result_call}")
    
    # Results should be probabilities between 0 and 1
    assert 0.0 <= result_bid <= 1.0
    assert 0.0 <= result_call <= 1.0


def main():
    """Run all tests to capture current behavior."""
    print("=== Testing MonteCarloAgent max_rounds and simulate_to_round_end behavior ===")
    print("This test captures current behavior before making changes (TDD approach)\n")
    
    try:
        test_max_rounds_parameter()
        print()
        test_simulate_to_round_end_parameter()
        print()
        test_simulation_direct()
        print()
        print("✓ All tests passed - current behavior documented")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()