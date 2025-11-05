import pytest

from eval.profiles import get_baseline_kwargs, get_mc_kwargs


def test_baseline_profiles_mapping():
    aggr = get_baseline_kwargs("aggressive")
    passv = get_baseline_kwargs("passive")
    assert isinstance(aggr, dict) and isinstance(passv, dict)
    assert aggr.get("threshold_call") == pytest.approx(0.25, rel=1e-6)
    assert passv.get("threshold_call") == pytest.approx(0.55, rel=1e-6)


def test_mc_kwargs_validation():
    assert get_mc_kwargs(100).get("n") == 100
    with pytest.raises(ValueError):
        get_mc_kwargs(0)
    with pytest.raises(ValueError):
        get_mc_kwargs(-1)
    with pytest.raises(ValueError):
        get_mc_kwargs(12.5)  # type: ignore[arg-type]
