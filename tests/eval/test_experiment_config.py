import json
import pytest

from eval.experiment_config import ExperimentConfig, ExperimentCondition


def test_config_round_trip():
    cond = ExperimentCondition(mc_n=100, baseline_profile="aggressive", layout="3v3", player_count=6)
    cfg = ExperimentConfig(
        name="demo",
        seeds=[1, 2],
        num_matches_per_seed=1,
        conditions=[cond],
        log_decisions=False,
        shuffle_seats=False,
        out_dir=None,
    )
    s = cfg.to_json()
    cfg2 = ExperimentConfig.from_json(s)
    assert cfg2.name == cfg.name
    assert cfg2.seeds == cfg.seeds
    assert cfg2.num_matches_per_seed == cfg.num_matches_per_seed
    assert len(cfg2.conditions) == 1
    c = cfg2.conditions[0]
    assert c.mc_n == 100
    assert c.baseline_profile == "aggressive"
    assert c.layout == "3v3"
    assert c.player_count == 6


def test_config_validation_errors():
    with pytest.raises(ValueError):
        ExperimentCondition(mc_n=100, baseline_profile="aggressive", layout="2v4", player_count=6).validate()
    with pytest.raises(ValueError):
        ExperimentCondition(mc_n=100, baseline_profile="weird", layout="3v3", player_count=6).validate()
    with pytest.raises(ValueError):
        ExperimentCondition(mc_n=-1, baseline_profile="aggressive", layout="3v3", player_count=6).validate()
    with pytest.raises(ValueError):
        ExperimentCondition(mc_n=100, baseline_profile="aggressive", layout="3v3", player_count=5).validate()

    cond = ExperimentCondition(mc_n=100, baseline_profile="aggressive", layout="3v3", player_count=6)
    with pytest.raises(ValueError):
        ExperimentConfig(name="", seeds=[1], num_matches_per_seed=1, conditions=[cond]).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", seeds="not_list", num_matches_per_seed=1, conditions=[cond]).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", seeds=[], num_matches_per_seed=0, conditions=[cond]).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", seeds=[1], num_matches_per_seed=1, conditions=[]).validate()
