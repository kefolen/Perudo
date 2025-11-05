from eval.experiment_config import ExperimentConfig, ExperimentCondition
from eval.runner import run_experiment


def _run_and_assert(condition_layout: str):
    cond = ExperimentCondition(mc_n=1, baseline_profile="aggressive", layout=condition_layout, player_count=6)
    cfg = ExperimentConfig(name="smoke", seeds=[1], num_matches_per_seed=1, conditions=[cond])
    matches, players = run_experiment(cfg, out_dir=None)

    assert len(matches) == 1
    assert len(players) == 6

    m = matches[0]
    assert m["winner_player"] in range(6)
    assert m["layout"] == condition_layout
    assert m["mc_n"] == 1
    assert m["baseline_profile"] == "aggressive"

    mc_cnt = sum(1 for p in players if p["agent_type"] == "MC")
    base_cnt = sum(1 for p in players if p["agent_type"] == "BASE")
    if condition_layout == "3v3":
        assert mc_cnt == 3 and base_cnt == 3
    else:
        assert mc_cnt == 1 and base_cnt == 5


def test_runner_smoke_3v3():
    _run_and_assert("3v3")


def test_runner_smoke_1v5():
    _run_and_assert("1v5")
