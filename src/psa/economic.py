"""Economic dispatch by lambda iteration.

Each unit has a quadratic cost C_i(P) = a_i + b_i P + c_i P^2 with limits
[Pmin, Pmax]. Without losses the optimum satisfies equal incremental cost
dC_i/dP = b_i + 2 c_i P_i = lambda for all units inside their limits.

With losses expressed through B-coefficients, P_L = P^T B P + B0^T P + B00,
the condition becomes (dC_i/dP) / (1 - dP_L/dP_i) = lambda (penalty factors),
solved by an outer lambda bisection with an inner fixed point on P.

Unit commitment note: which units to switch on over a horizon is a
combinatorial problem (start-up costs, min up/down times). A simple
priority-list heuristic is provided in `priority_list_commitment`; a proper
solution would use dynamic programming or mixed integer programming.
"""

from __future__ import annotations

import numpy as np


def unit_cost(units, P):
    return sum(u["a"] + u["b"] * p + u["c"] * p**2 for u, p in zip(units, P))


def _dispatch_for_lambda(units, lam, penalty=None):
    P = np.zeros(len(units))
    for i, u in enumerate(units):
        pf = 1.0 if penalty is None else penalty[i]
        p = (lam / pf - u["b"]) / (2 * u["c"])
        P[i] = min(max(p, u["Pmin"]), u["Pmax"])
    return P


def economic_dispatch(units, demand, B=None, B0=None, B00=0.0, tol=1e-6, max_iter=200):
    """Return dict with P (array), lam, losses, cost, iterations.

    units: list of dicts a, b, c, Pmin, Pmax (MW and $/h units)
    demand: MW
    B, B0, B00: optional loss coefficients in MW-consistent units
    """
    units = list(units)
    lam_lo = min(u["b"] + 2 * u["c"] * u["Pmin"] for u in units)
    lam_hi = max(u["b"] + 2 * u["c"] * u["Pmax"] for u in units) * 3 + 1
    P = np.array([u["Pmin"] for u in units], dtype=float)
    losses = 0.0
    it = 0
    for it in range(1, max_iter + 1):
        lam = 0.5 * (lam_lo + lam_hi)
        if B is None:
            P = _dispatch_for_lambda(units, lam)
            losses = 0.0
        else:
            B = np.asarray(B, dtype=float)
            b0 = np.zeros(len(units)) if B0 is None else np.asarray(B0, dtype=float)
            for _ in range(50):
                dPL = 2 * B @ P + b0
                penalty = 1.0 / (1.0 - dPL)
                Pn = _dispatch_for_lambda(units, lam, penalty)
                if np.max(np.abs(Pn - P)) < 1e-9:
                    P = Pn
                    break
                P = Pn
            losses = float(P @ B @ P + b0 @ P + B00)
        err = P.sum() - demand - losses
        if abs(err) < tol:
            break
        if err > 0:
            lam_hi = lam
        else:
            lam_lo = lam
    return dict(P=P, lam=lam, losses=losses, cost=unit_cost(units, P), iterations=it,
                mismatch=P.sum() - demand - losses)


def priority_list_commitment(units, demand):
    """Commit units in order of full-load average cost until capacity >= demand."""
    def flac(u):
        return (u["a"] + u["b"] * u["Pmax"] + u["c"] * u["Pmax"] ** 2) / u["Pmax"]
    order = sorted(range(len(units)), key=lambda i: flac(units[i]))
    on, cap = [], 0.0
    for i in order:
        on.append(i)
        cap += units[i]["Pmax"]
        if cap >= demand:
            break
    return sorted(on)
