"""IEEE 14-bus test system.

Source: the IEEE 14-bus data as distributed in the University of Washington
power systems test case archive (Christie, 1993), in the form used by
MATPOWER case14. Base 100 MVA. Bus 9 carries a 19 MVAr shunt capacitor.
Transformers 4-7, 4-9 and 5-6 have off-nominal taps on the from side.

Reference solution below: the solved voltages printed in the IEEE Common
Data Format file from the same archive, given there to three decimals in Vm
and two in Va. This is the CDF printout, not a MATPOWER run. MATPOWER
runpf on case14 agrees with it to the printed precision except at bus 4,
where MATPOWER gives Vm 1.018 and Va -10.31. This toolkit reproduces the
CDF table to a maximum Vm error of 0.13 percent, at bus 4, and a maximum
angle error of 0.02 deg. Q limits not enforced.
    bus  Vm      Va(deg)
    1    1.060    0.00
    2    1.045   -4.98
    3    1.010  -12.72
    4    1.019  -10.33
    5    1.020   -8.78
    6    1.070  -14.22
    7    1.062  -13.37
    8    1.090  -13.36
    9    1.056  -14.94
    10   1.051  -15.10
    11   1.057  -14.79
    12   1.055  -15.08
    13   1.050  -15.16
    14   1.036  -16.04
Slack generation 232.4 MW, total losses 13.39 MW.
"""

from __future__ import annotations

import copy

# bus types: 3 slack, 2 PV, 1 PQ. Loads in MW / MVAr, shunts in MW / MVAr at 1 pu.
_BUSES = [
    # bus type  Pd    Qd    Gs  Bs   Vm     Va
    (1, 3, 0.0, 0.0, 0.0, 0.0, 1.060, 0.0),
    (2, 2, 21.7, 12.7, 0.0, 0.0, 1.045, 0.0),
    (3, 2, 94.2, 19.0, 0.0, 0.0, 1.010, 0.0),
    (4, 1, 47.8, -3.9, 0.0, 0.0, 1.0, 0.0),
    (5, 1, 7.6, 1.6, 0.0, 0.0, 1.0, 0.0),
    (6, 2, 11.2, 7.5, 0.0, 0.0, 1.070, 0.0),
    (7, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    (8, 2, 0.0, 0.0, 0.0, 0.0, 1.090, 0.0),
    (9, 1, 29.5, 16.6, 0.0, 19.0, 1.0, 0.0),
    (10, 1, 9.0, 5.8, 0.0, 0.0, 1.0, 0.0),
    (11, 1, 3.5, 1.8, 0.0, 0.0, 1.0, 0.0),
    (12, 1, 6.1, 1.6, 0.0, 0.0, 1.0, 0.0),
    (13, 1, 13.5, 5.8, 0.0, 0.0, 1.0, 0.0),
    (14, 1, 14.9, 5.0, 0.0, 0.0, 1.0, 0.0),
]

# bus, Pg, Qg, Qmax, Qmin, Vset, Pmax, Pmin
_GENS = [
    (1, 232.4, -16.9, 10.0, 0.0, 1.060, 332.4, 0.0),
    (2, 40.0, 42.4, 50.0, -40.0, 1.045, 140.0, 0.0),
    (3, 0.0, 23.4, 40.0, 0.0, 1.010, 100.0, 0.0),
    (6, 0.0, 12.2, 24.0, -6.0, 1.070, 100.0, 0.0),
    (8, 0.0, 17.4, 24.0, -6.0, 1.090, 100.0, 0.0),
]

# from, to, r, x, b, tap
_BRANCHES = [
    (1, 2, 0.01938, 0.05917, 0.0528, 0.0),
    (1, 5, 0.05403, 0.22304, 0.0492, 0.0),
    (2, 3, 0.04699, 0.19797, 0.0438, 0.0),
    (2, 4, 0.05811, 0.17632, 0.0340, 0.0),
    (2, 5, 0.05695, 0.17388, 0.0346, 0.0),
    (3, 4, 0.06701, 0.17103, 0.0128, 0.0),
    (4, 5, 0.01335, 0.04211, 0.0, 0.0),
    (4, 7, 0.0, 0.20912, 0.0, 0.978),
    (4, 9, 0.0, 0.55618, 0.0, 0.969),
    (5, 6, 0.0, 0.25202, 0.0, 0.932),
    (6, 11, 0.09498, 0.19890, 0.0, 0.0),
    (6, 12, 0.12291, 0.25581, 0.0, 0.0),
    (6, 13, 0.06615, 0.13027, 0.0, 0.0),
    (7, 8, 0.0, 0.17615, 0.0, 0.0),
    (7, 9, 0.0, 0.11001, 0.0, 0.0),
    (9, 10, 0.03181, 0.08450, 0.0, 0.0),
    (9, 14, 0.12711, 0.27038, 0.0, 0.0),
    (10, 11, 0.08205, 0.19207, 0.0, 0.0),
    (12, 13, 0.22092, 0.19988, 0.0, 0.0),
    (13, 14, 0.17093, 0.34802, 0.0, 0.0),
]

REFERENCE_VM = {
    1: 1.060, 2: 1.045, 3: 1.010, 4: 1.019, 5: 1.020, 6: 1.070, 7: 1.062,
    8: 1.090, 9: 1.056, 10: 1.051, 11: 1.057, 12: 1.055, 13: 1.050, 14: 1.036,
}
REFERENCE_VA_DEG = {
    1: 0.0, 2: -4.98, 3: -12.72, 4: -10.33, 5: -8.78, 6: -14.22, 7: -13.37,
    8: -13.36, 9: -14.94, 10: -15.10, 11: -14.79, 12: -15.08, 13: -15.16, 14: -16.04,
}
REFERENCE_LOSS_MW = 13.39
REFERENCE_SLACK_P_MW = 232.4


def ieee14() -> dict:
    """Return a fresh deep copy of the IEEE 14-bus case as a dict.

    Keys: 'base_mva', 'buses' (list of dicts), 'gens' (list of dicts),
    'branches' (list of dicts).
    """
    buses = [
        dict(bus=b, type=t, Pd=pd, Qd=qd, Gs=gs, Bs=bs, Vm=vm, Va=va)
        for (b, t, pd, qd, gs, bs, vm, va) in _BUSES
    ]
    gens = [
        dict(bus=b, Pg=pg, Qg=qg, Qmax=qmx, Qmin=qmn, Vset=vs, Pmax=pmx, Pmin=pmn)
        for (b, pg, qg, qmx, qmn, vs, pmx, pmn) in _GENS
    ]
    branches = [
        dict(**{"from": f, "to": t}, r=r, x=x, b=b, tap=tap)
        for (f, t, r, x, b, tap) in _BRANCHES
    ]
    return copy.deepcopy(dict(base_mva=100.0, buses=buses, gens=gens, branches=branches))
