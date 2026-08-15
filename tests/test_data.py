"""Case and profile loader tests, including convergence on all three cases."""

import numpy as np
import pytest

from psa import loadflow
from psa.data import load_case, load_profile

# name, expected losses MW, expected slack P MW
CASES = [
    ("ieee14", 13.39, 232.4),
    ("ieee30", 17.56, 261.0),
    ("ieee57", 27.86, 478.7),
]


@pytest.mark.parametrize("name,losses,slack", CASES)
def test_case_converges_with_expected_losses(name, losses, slack):
    case = load_case(name)
    res = loadflow.solve(case, "nr")
    assert res.converged
    assert res.iterations <= 6
    assert abs(res.losses_mw - losses) < 0.05
    assert abs(res.Pg[0] - slack) < 0.5
    assert np.all(res.Vm > 0.9) and np.all(res.Vm < 1.1)


def test_case_structure():
    case = load_case("ieee57")
    assert case["base_mva"] == 100.0
    assert "University of Washington" in case["source"]
    assert len(case["buses"]) == 57
    assert len(case["branches"]) == 80
    assert sum(1 for b in case["buses"] if b["type"] == 3) == 1


def test_case_loader_rejects_missing_and_invalid():
    with pytest.raises(FileNotFoundError):
        load_case("ieee999")
    case = load_case("ieee14")
    import json
    import tempfile
    bad = dict(case)
    bad["gens"] = [dict(case["gens"][0], bus=99)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        path = f.name
    with pytest.raises(ValueError):
        load_case(path)


def test_load_profile_shapes_and_ranges():
    load = np.array(load_profile("load_8760", "load_mw"))
    pv = np.array(load_profile("pv_cf_8760", "cf"))
    wind = np.array(load_profile("wind_cf_8760", "cf"))
    for series in (load, pv, wind):
        assert len(series) == 8760
    assert load.min() > 0
    assert 100 < load.max() < 200
    assert pv.min() >= 0 and pv.max() <= 1
    assert wind.min() >= 0 and wind.max() <= 1
    # PV produces nothing at night: hour 0 of every day is dark
    assert pv[0::24].max() == 0.0
    # annual capacity factors in a plausible band
    assert 0.10 < pv.mean() < 0.25
    assert 0.20 < wind.mean() < 0.45


def test_load_profile_rejects_unknown_column():
    with pytest.raises(ValueError):
        load_profile("load_8760", "nope")
    with pytest.raises(FileNotFoundError):
        load_profile("missing_8760", "cf")
