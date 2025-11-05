import os
import tempfile

from eval.plots import save_winrate_vs_mc


def test_save_winrate_vs_mc_smoke():
    aggregates = [
        {"layout": "3v3", "baseline_profile": "aggressive", "mc_n": 100, "win_rate_mc_team": 0.6},
        {"layout": "3v3", "baseline_profile": "aggressive", "mc_n": 200, "win_rate_mc_team": 0.62},
        {"layout": "1v5", "baseline_profile": "passive", "mc_n": 100, "win_rate_mc_team": 0.55},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        ok = save_winrate_vs_mc(aggregates, tmp, title="demo")
        if ok:
            assert os.path.exists(os.path.join(tmp, "winrate_vs_mc.png"))
        else:
            # matplotlib missing: should return False gracefully
            assert not os.path.exists(os.path.join(tmp, "winrate_vs_mc.png"))
