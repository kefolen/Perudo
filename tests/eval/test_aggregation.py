from eval.aggregate import aggregate_win_rates, wilson_ci


def test_wilson_basic():
    low, high = wilson_ci(50, 100)
    assert 0.4 < low < 0.6
    assert 0.5 < high < 0.7


def test_aggregate_win_rates_groups():
    matches = [
        {"layout": "3v3", "baseline_profile": "aggressive", "mc_n": 100, "mc_team_win": True},
        {"layout": "3v3", "baseline_profile": "aggressive", "mc_n": 100, "mc_team_win": False},
        {"layout": "3v3", "baseline_profile": "aggressive", "mc_n": 200, "mc_team_win": True},
        {"layout": "1v5", "baseline_profile": "passive", "mc_n": 100, "mc_team_win": True},
    ]
    rows = aggregate_win_rates(matches)
    # Should have three groups
    assert len(rows) == 3
    # Find 3v3,aggressive,100
    r = next(r for r in rows if r["layout"] == "3v3" and r["baseline_profile"] == "aggressive" and r["mc_n"] == 100)
    assert r["matches"] == 2
    assert r["mc_team_wins"] == 1
    assert abs(r["win_rate_mc_team"] - 0.5) < 1e-9
