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


def test_unsymmetrical_textbook():
    # Grainger and Stevenson style check with Z1 = Z2 = j0.25, Z0 = j0.15 (per unit)
    z1 = z2 = 0.25j
    z0 = 0.15j
    slg = fault.slg_fault(z1, z2, z0)
    assert abs(abs(slg["If"]) - 3 / 0.65) < 1e-9
    ll = fault.ll_fault(z1, z2)
    assert abs(abs(ll["If"]) - np.sqrt(3) / 0.5) < 1e-9
    dlg = fault.dlg_fault(z1, z2, z0)
    # phase a current must be zero for a b-c-ground fault
    assert abs(dlg["Iabc"][0]) < 1e-9
    # ground return current equals 3 Ia0
    assert abs(dlg["If"] - 3 * dlg["I012"][0]) < 1e-9
