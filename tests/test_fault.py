"""Fault analysis checks against hand-built Zbus values."""

import numpy as np

from psa import fault


def test_two_bus_hand_zbus():
    # generator j0.1 at bus 1, line j0.2 from 1 to 2. Zbus by inspection:
    # Z11 = j0.1, Z12 = Z21 = j0.1, Z22 = j0.3. Fault at bus 2: If = 1/j0.3 = -j3.3333
    zb = fault.build_zbus(2, [{"from": 1, "to": 2, "x": 0.2}], {1: 0.1j})
    assert np.allclose(zb, np.array([[0.1j, 0.1j], [0.1j, 0.3j]]))
    i_f, v = fault.three_phase_fault(zb, 2)
    assert abs(i_f - (-1j / 0.3)) < 1e-12
    assert abs(v[1]) < 1e-12
    assert abs(v[0] - (1 - 0.1 / 0.3)) < 1e-12


def test_three_bus_vs_hand_zbus():
    # Bus 1 gen j0.2, bus 2 gen j0.25, lines 1-2 j0.1, 1-3 j0.2, 2-3 j0.15 (Zf = 0)
    # Hand Ybus (all j-values):
    #   Y11 = -j(5 + 10 + 5)      = -j20
    #   Y22 = -j(4 + 10 + 6.667)  = -j20.667
    #   Y33 = -j(5 + 6.667)       = -j11.667
    #   Y12 = j10, Y13 = j5, Y23 = j6.667
    branches = [{"from": 1, "to": 2, "x": 0.1}, {"from": 1, "to": 3, "x": 0.2},
                {"from": 2, "to": 3, "x": 0.15}]
    zb = fault.build_zbus(3, branches, {1: 0.2j, 2: 0.25j})
    Yhand = 1j * np.array([[-20.0, 10.0, 5.0], [10.0, -14.0 - 20 / 3, 20 / 3], [5.0, 20 / 3, -5.0 - 20 / 3]])
    Zhand = np.linalg.inv(Yhand)
    assert np.allclose(zb, Zhand)
    for bus in (1, 2, 3):
        i_f, _ = fault.three_phase_fault(zb, bus)
        assert abs(i_f - 1.0 / Zhand[bus - 1, bus - 1]) < 1e-10
    # sanity: bus 3 has the largest Thevenin reactance so the smallest fault current
    mags = [abs(fault.three_phase_fault(zb, b)[0]) for b in (1, 2, 3)]
    assert mags[2] == min(mags)


def test_unsymmetrical_boundary_conditions_and_hand_values():
    """Check fault boundary conditions and hand-computed numeric values.

    Z1 = Z2 = j0.25, Z0 = j0.15 pu. The numbers below were worked by hand on
    paper from the sequence formulas, then written here as decimals, so the
    asserts do not re-evaluate the expressions the code uses.
    """
    z1 = z2 = 0.25j
    z0 = 0.15j
    # SLG on phase a: Ia1 = 1/j0.65 = -j1.53846, If = 3 Ia1 = 4.61538 pu.
    # Boundary conditions: Ib = Ic = 0.
    slg = fault.slg_fault(z1, z2, z0)
    assert abs(slg["If"] - (-4.615384615384615j)) < 1e-12
    assert abs(slg["Iabc"][1]) < 1e-12 and abs(slg["Iabc"][2]) < 1e-12
    # faulted phase voltage is zero for a bolted fault: Va = V0 + V1 + V2 with
    # V1 = 1 - Z1 Ia1, V2 = -Z2 Ia2, V0 = -Z0 Ia0 (computed from code outputs)
    i0, i1, i2 = slg["I012"]
    va = (1 - z1 * i1) + (-z2 * i2) + (-z0 * i0)
    assert abs(va) < 1e-12
    # LL between b and c: Ia1 = 1/j0.5 = -j2, Ib = -j sqrt(3) Ia1 = -3.46410 pu.
    # Boundary conditions: Ia = 0, Ib = -Ic, and Vb = Vc at the fault.
    ll = fault.ll_fault(z1, z2)
    assert abs(ll["If"] - (-3.4641016151377544)) < 1e-12
    assert abs(ll["Iabc"][0]) < 1e-12
    assert abs(ll["Iabc"][1] + ll["Iabc"][2]) < 1e-12
    i0, i1, i2 = ll["I012"]
    v012 = np.array([0.0, 1 - z1 * i1, -z2 * i2])
    vabc = fault.A_MAT @ v012
    assert abs(vabc[1] - vabc[2]) < 1e-12
    # DLG on b and c: Zpar = j0.25*j0.15/j0.40 = j0.09375,
    # Ia1 = 1/j0.34375 = -j2.909091, Ia2 = j1.090909, Ia0 = j1.818182,
    # ground current If = 3 Ia0 = j5.454545 pu.
    dlg = fault.dlg_fault(z1, z2, z0)
    assert abs(dlg["I012"][1] - (-2.909090909090909j)) < 1e-9
    assert abs(dlg["I012"][2] - 1.0909090909090908j) < 1e-9
    assert abs(dlg["I012"][0] - 1.8181818181818181j) < 1e-9
    assert abs(dlg["If"] - 5.454545454545454j) < 1e-9
    # boundary conditions: Ia = 0, Vb = Vc = 0 at the fault
    assert abs(dlg["Iabc"][0]) < 1e-9
    i0, i1, i2 = dlg["I012"]
    v012 = np.array([-z0 * i0, 1 - z1 * i1, -z2 * i2])
    vabc = fault.A_MAT @ v012
    assert abs(vabc[1]) < 1e-9 and abs(vabc[2]) < 1e-9


def test_bus_out_of_range_raises():
    zb = fault.build_zbus(2, [{"from": 1, "to": 2, "x": 0.2}], {1: 0.1j})
    for bad in (0, 3, -1):
        try:
            fault.three_phase_fault(zb, bad)
            assert False, "expected ValueError"
        except ValueError:
            pass
        try:
            fault.sequence_thevenin(zb, zb, zb, bad)
            assert False, "expected ValueError"
        except ValueError:
            pass
