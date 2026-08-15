"""Load flow validation against the IEEE 14-bus published solution.

Reference values are the IEEE Common Data Format printed solution from the
University of Washington archive, Q limits not enforced, stored in
psa.data.ieee14 as REFERENCE_VM / REFERENCE_VA_DEG:
    Vm  = 1.060 1.045 1.010 1.019 1.020 1.070 1.062 1.090 1.056 1.051 1.057 1.055 1.050 1.036
    Va  = 0 -4.98 -12.72 -10.33 -8.78 -14.22 -13.37 -13.36 -14.94 -15.10 -14.79 -15.08 -15.16 -16.04
    slack P = 232.4 MW, losses = 13.39 MW
"""

import numpy as np
import pytest

from psa import loadflow
from psa.data import ieee14
from psa.data.ieee14 import (REFERENCE_LOSS_MW, REFERENCE_SLACK_P_MW, REFERENCE_VA_DEG,
                             REFERENCE_VM)


@pytest.mark.parametrize("method", ["nr", "gs", "fdlf"])
def test_converges_and_matches_reference(method):
    res = loadflow.solve(ieee14(), method)
    assert res.converged
    ref_vm = np.array([REFERENCE_VM[i] for i in range(1, 15)])
    ref_va = np.array([REFERENCE_VA_DEG[i] for i in range(1, 15)])
    # achieved: 0.13 percent Vm at bus 4 and 0.017 deg, both against the CDF
    # printout, which itself is rounded to three decimals in Vm
    assert np.max(np.abs(res.Vm - ref_vm) / ref_vm) < 0.002
    assert np.max(np.abs(res.Va_deg - ref_va)) < 0.025
    assert abs(res.losses_mw - REFERENCE_LOSS_MW) < 0.05
    assert abs(res.Pg[0] - REFERENCE_SLACK_P_MW) < 0.1


def test_nr_iterations_small():
    res = loadflow.newton_raphson(ieee14(), tol=1e-8)
    assert res.iterations <= 6


def test_power_balance_mismatch_below_tolerance():
    tol = 1e-8
    res = loadflow.newton_raphson(ieee14(), tol=tol)
    assert loadflow.power_balance_mismatch(ieee14(), res) < tol
    # generation equals load plus losses
    case = ieee14()
    pd = sum(b["Pd"] for b in case["buses"])
    assert abs(res.Pg.sum() - pd - res.losses_mw) < 1e-6


def test_q_limits_keep_generators_inside_limits():
    case = ieee14()
    # tighten bus 2 limit so it must switch to PQ
    case["gens"][1]["Qmax"] = 30.0
    res = loadflow.newton_raphson(case, enforce_q_limits=True)
    assert res.converged
    assert res.Qg[1] <= 30.0 + 1e-6
    assert res.bus_types[1] == loadflow.PQ


def test_fdlf_honors_branch_status():
    case = ieee14()
    case["branches"][0]["status"] = 0
    a = loadflow.solve(case, "nr")
    b = loadflow.solve(case, "fdlf")
    assert a.converged and b.converged
    assert np.allclose(a.Vm, b.Vm, atol=1e-5)
    assert np.allclose(a.Va_deg, b.Va_deg, atol=1e-3)


def test_q_limits_kwarg_rejected_outside_nr():
    for method in ("gs", "fdlf"):
        with pytest.raises(ValueError):
            loadflow.solve(ieee14(), method, enforce_q_limits=True)


def test_methods_agree():
    a = loadflow.solve(ieee14(), "nr")
    b = loadflow.solve(ieee14(), "gs", tol=1e-8)
    c = loadflow.solve(ieee14(), "fdlf")
    assert np.allclose(a.Vm, b.Vm, atol=1e-4)
    assert np.allclose(a.Vm, c.Vm, atol=1e-5)
