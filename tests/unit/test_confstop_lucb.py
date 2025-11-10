import math
import random
import pytest

from agents.mc_agent import MonteCarloAgent
from sim.perudo import Action


class DeterministicValueAgent(MonteCarloAgent):
    """
    Test double that returns deterministic values for two actions:
    - Action.bid(1,2): reward ~ 0.7
    - Action.bid(1,3): reward ~ 0.5
    It bypasses full simulation to keep tests fast and deterministic.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value_map = {
            ('bid', 1, 2): 0.7,
            ('bid', 1, 3): 0.5,
        }

    def _run_single_rollout_value(self, obs, action):
        key = (action[0],) + tuple(action[1:])
        base = self._value_map.get(key, 0.5)
        # Add tiny noise for variance
        return base


def test_empirical_bernstein_radius_monotonicity(deterministic_seed):
    rng = random.Random(deterministic_seed)
    agent = MonteCarloAgent(rng=rng)
    s = MonteCarloAgent.ActionStats()

    # Initially infinite radius
    assert math.isinf(s.radius(agent.confstop_delta, agent.confstop_bound_b))

    # Feed constant values -> zero variance, radius should decrease with n
    for n in range(1, 51):
        s.update(0.5)
        r = s.radius(agent.confstop_delta, agent.confstop_bound_b)
        assert r >= 0
        if n > 1:
            # radius should be non-increasing
            assert r <= prev + 1e-9
        prev = r


def test_lucb_selects_best_action_with_small_budget(sample_simulator, sample_game_state):
    obs = sample_game_state.copy()
    obs['_simulator'] = sample_simulator
    obs['current_bid'] = None

    rng = random.Random(123)
    agent = DeterministicValueAgent(
        n=200,  # would be used by baseline path
        rng=rng,
        confstop_enabled=True,
        confstop_delta=0.05,
        confstop_min_init=4,
        confstop_batch_size=4,
        confstop_max_total_rollouts=40,
        confstop_max_per_action=40,
    )

    # Two candidate actions
    a_good = Action.bid(1, 2)
    a_bad = Action.bid(1, 3)

    # Use LUCB controller directly
    choice = agent._select_action_confstop(obs, [a_good, a_bad])
    assert choice == a_good
