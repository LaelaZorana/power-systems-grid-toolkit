"""Bus admittance matrix construction.

Branch model: series impedance r + jx, total line charging b (split half at
each end), and an off-nominal tap t with phase shift on the from side. The
pi-model with the tap on the from side gives (MATPOWER convention)

    Yff = (ys + j b/2) / |t|^2
    Yft = -ys / conj(t)
    Ytf = -ys / t
    Ytt =  ys + j b/2

with ys = 1 / (r + jx) and t = tap * exp(j shift). Bus shunts Gs + jBs (pu)
are added on the diagonal.
"""

from __future__ import annotations

import numpy as np


def build_ybus(n_bus: int, branches: list[dict], bus_shunts: dict | None = None) -> np.ndarray:
    """Return the complex n x n bus admittance matrix.

    Parameters
    ----------
    n_bus : number of buses (indexed 1..n_bus in the branch dicts)
    branches : list of dicts with keys 'from', 'to', 'r', 'x', 'b'
        and optional 'tap' (default 1.0, 0 treated as 1.0), 'shift' (degrees),
        'status' (default 1).
    bus_shunts : optional {bus: (Gs, Bs)} in pu on system base
    """
    Y = np.zeros((n_bus, n_bus), dtype=complex)
    for br in branches:
        if br.get("status", 1) == 0:
            continue
        i = br["from"] - 1
        k = br["to"] - 1
        ys = 1.0 / complex(br["r"], br["x"])
        bc = 1j * br.get("b", 0.0) / 2.0
        tap = br.get("tap", 1.0) or 1.0
        shift = np.deg2rad(br.get("shift", 0.0))
        t = tap * np.exp(1j * shift)
        Y[i, i] += (ys + bc) / (abs(t) ** 2)
        Y[k, k] += ys + bc
        Y[i, k] += -ys / np.conj(t)
        Y[k, i] += -ys / t
    if bus_shunts:
        for bus, (gs, bs) in bus_shunts.items():
            Y[bus - 1, bus - 1] += complex(gs, bs)
    return Y


def branch_admittances(br: dict) -> tuple[complex, complex, complex, complex]:
    """Return (Yff, Yft, Ytf, Ytt) for one branch, used for line flow calcs."""
    ys = 1.0 / complex(br["r"], br["x"])
    bc = 1j * br.get("b", 0.0) / 2.0
    tap = br.get("tap", 1.0) or 1.0
    t = tap * np.exp(1j * np.deg2rad(br.get("shift", 0.0)))
    return (ys + bc) / abs(t) ** 2, -ys / np.conj(t), -ys / t, ys + bc
