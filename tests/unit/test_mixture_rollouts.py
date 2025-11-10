import math
import pytest
from agents.mc_agent import MonteCarloAgent


def test_policy_sampler_respects_weights(deterministic_seed):
    # Configure mixture with two policies
    import random
    rng = random.Random(deterministic_seed)
    agent = MonteCarloAgent(
        n=1,
        rng=rng,
        mixture_enabled=True,
        mixture_policies=["baseline", "random"],
        mixture_weights=[0.7, 0.3],
    )
    # Sample many times and check frequencies
    counts = {"baseline": 0, "random": 0}
    trials = 5000
    for _ in range(trials):
        pid = agent._sample_policy_id(depth=0, rng=agent.rng)
        counts[pid] += 1
    freq_baseline = counts["baseline"] / trials
    freq_random = counts["random"] / trials
    # Allow tolerance due to randomness
    assert abs(freq_baseline - 0.7) < 0.05
    assert abs(freq_random - 0.3) < 0.05


def test_mixture_single_policy_evaluates_with_baseline(sample_simulator, sample_game_state):
    # Prepare observation
    obs = sample_game_state.copy()
    obs['_simulator'] = sample_simulator
    # Make sure there is an initial bid to avoid call/exact
    obs['current_bid'] = None

    import random
    # Agent A: baseline (mixture disabled)
    agent_base = MonteCarloAgent(n=6, rng=random.Random(123), chunk_size=3)
    # Agent B: mixture enabled but only baseline policy
    agent_mix = MonteCarloAgent(
        n=6,
        rng=random.Random(123),
        chunk_size=3,
        mixture_enabled=True,
        mixture_policies=["baseline"],
        mixture_weights=[1.0],
    )
    # Evaluate a simple action chosen by baseline rollout policy
    # Use the agent's own policy to generate a legal bid
    from sim.perudo import Action
    action = Action.bid(1, 2)

    v_base = agent_base.evaluate_action(obs, action)
    v_mix = agent_mix.evaluate_action(obs, action)

    # Both must return valid probabilities within [0,1]
    assert 0.0 <= v_base <= 1.0
    assert 0.0 <= v_mix <= 1.0
    # They should be reasonably close even if RNG streams differ slightly
    assert abs(v_base - v_mix) <= 0.25
