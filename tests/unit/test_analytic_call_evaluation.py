import math
import pytest
from sim.perudo import Action, binom_cdf_ge_fast
from agents.mc_agent import MonteCarloAgent


def compute_expected_call_reward(obs):
    sim = obs['_simulator']
    current_bid = obs['current_bid']
    assert current_bid is not None
    qty, face = current_bid
    my_hand = obs['my_hand']
    total_dice = sum(obs['dice_counts'])
    unknown = total_dice - len(my_hand)
    x_f = sum(1 for d in my_hand if d == face)
    x_1 = sum(1 for d in my_hand if d == 1)
    ones_wild_effective = sim.ones_are_wild and not obs.get('maputa_active', False)
    if face != 1 and ones_wild_effective:
        p = 1/3
        need = max(0, qty - (x_f + x_1))
    else:
        p = 1/6
        need = max(0, qty - (x_f if face != 1 else x_1))
    P_true = binom_cdf_ge_fast(unknown, need, p) if unknown >= 0 else 0.0
    # analytic-leaf reward scale (E[Δdice])
    return 1 - 2 * P_true


def test_analytic_leaf_call_matches_formula(sample_simulator, sample_game_state):
    # Construct observation with a specific current bid
    obs = sample_game_state.copy()
    obs['_simulator'] = sample_simulator
    obs['current_bid'] = (3, 2)  # 3 twos
    # Ensure not in maputa for this test
    obs['maputa_active'] = False

    # Build agent with analytic leaf mode
    import random
    agent = MonteCarloAgent(n=5, rng=random.Random(123), chunk_size=5, early_stop_margin=0.0)
    # Inject experimental mode via attribute if available later
    # We expect the project to support call_eval_mode; fallback to attribute set for test
    setattr(agent, 'call_eval_mode', 'analytic_leaf')

    action = Action.call()
    result = agent.evaluate_action(obs, action)

    expected = compute_expected_call_reward(obs)
    assert math.isfinite(result)
    # Analytic leaf returns E[Δdice] in [-1,1]
    assert -1.0 <= result <= 1.0
    # Match the analytic expectation closely
    assert pytest.approx(expected, rel=1e-12, abs=1e-12) == result


def test_default_mode_remains_rollout(mc_agent, observation_with_simulator, sample_actions):
    # By default, MonteCarloAgent should use rollout mode, not analytic leaf.
    obs = observation_with_simulator
    # Ensure there is a current bid to make 'call' legal
    obs = obs.copy()
    obs['current_bid'] = (2, 3)

    action = sample_actions['call']
    result = mc_agent.evaluate_action(obs, action)

    # In rollout mode, return should be a probability in [0,1]
    assert 0.0 <= result <= 1.0
