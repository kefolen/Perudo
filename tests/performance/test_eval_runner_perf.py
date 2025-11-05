import os
import sys
import time
import pytest

from eval.experiment_config import ExperimentConfig, ExperimentCondition
from eval.runner import run_experiment


@pytest.mark.skipif(True, reason="Performance test is environment-dependent; skip by default")
def test_run_match_perf():
    cond = ExperimentCondition(mc_n=50, baseline_profile="aggressive", layout="3v3", player_count=6)
    cfg = ExperimentConfig(name="perf", seeds=[1], num_matches_per_seed=1, conditions=[cond])
    t0 = time.time()
    run_experiment(cfg, out_dir=None)
    dt = time.time() - t0
    assert dt < 2.0
