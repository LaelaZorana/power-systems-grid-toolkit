"""Single machine infinite bus transient stability.

Swing equation (per unit power, H in seconds, delta in rad):
    (2H / omega_s) d2delta/dt2 = Pm - Pe(delta) - D domega
with Pe = Pmax sin(delta) for the prefault, during-fault and postfault
networks (Pmax1, Pmax2, Pmax3).

Equal area criterion: with delta0 = asin(Pm/Pmax1), delta_max = pi -
asin(Pm/Pmax3), the critical clearing angle is
    cos(delta_cr) = [Pm (delta_max - delta0) + Pmax3 cos(delta_max)
                     - Pmax2 cos(delta0)] / (Pmax3 - Pmax2)
For Pmax2 = 0 (fault at the machine terminal) the critical clearing time is
    t_cr = sqrt(2 H (delta_cr - delta0) / (pi f Pm)).
Otherwise the swing curve is integrated numerically to reach delta_cr.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def critical_clearing_angle(pm, pmax1, pmax2, pmax3):
    if pm > pmax1:
        raise ValueError("no prefault equilibrium, Pm exceeds Pmax1")
    if pm > pmax3:
        raise ValueError("no postfault equilibrium, Pm exceeds Pmax3")
    if pmax3 <= pmax2:
        raise ValueError("postfault Pmax3 must exceed during-fault Pmax2")
    d0 = np.arcsin(pm / pmax1)
    dmax = np.pi - np.arcsin(pm / pmax3)
    cosd = (pm * (dmax - d0) + pmax3 * np.cos(dmax) - pmax2 * np.cos(d0)) / (pmax3 - pmax2)
    return float(np.arccos(np.clip(cosd, -1, 1))), float(d0), float(dmax)


def critical_clearing_time_equal_area(pm, pmax1, pmax2, pmax3, H, f=60.0):
    """CCT from equal-area. Closed form if pmax2 == 0, else time at which the
    during-fault swing reaches delta_cr (found by integration)."""
    dcr, d0, _ = critical_clearing_angle(pm, pmax1, pmax2, pmax3)
    if pmax2 == 0.0:
        return float(np.sqrt(2 * H * (dcr - d0) / (np.pi * f * pm))), dcr
    ws = 2 * np.pi * f

    def rhs(t, y):
        return [y[1], ws / (2 * H) * (pm - pmax2 * np.sin(y[0]))]

    def hit(t, y):
        return y[0] - dcr
    hit.terminal = True
    hit.direction = 1
    sol = solve_ivp(rhs, (0, 10), [d0, 0.0], events=hit, max_step=1e-3, rtol=1e-9, atol=1e-11)
    return float(sol.t_events[0][0]), dcr


def swing_curve(pm, pmax1, pmax2, pmax3, H, t_clear, f=60.0, t_end=3.0, D=0.0, dt=1e-3):
    """Integrate the swing equation with the fault cleared at t_clear.

    Returns (t, delta_deg, omega_dev). D is a per unit damping coefficient on
    the speed deviation (rad/s).
    """
    ws = 2 * np.pi * f
    d0 = np.arcsin(pm / pmax1)

    def make_rhs(pmax):
        def rhs(t, y):
            return [y[1], ws / (2 * H) * (pm - pmax * np.sin(y[0]) - D * y[1])]
        return rhs

    if t_clear <= 0.0:
        # fault cleared instantly, integrate the postfault network only
        t2 = np.linspace(0.0, t_end, max(int(round(t_end / dt)), 2))
        s2 = solve_ivp(make_rhs(pmax3), (0.0, t_end), [d0, 0.0],
                       t_eval=t2, max_step=dt, rtol=1e-8)
        return s2.t, np.rad2deg(s2.y[0]), s2.y[1]
    t1 = np.linspace(0, t_clear, max(int(round(t_clear / dt)), 2))
    s1 = solve_ivp(make_rhs(pmax2), (0, t_clear), [d0, 0.0], t_eval=t1, max_step=dt, rtol=1e-8)
    t2 = np.linspace(t_clear, t_end, max(int(round((t_end - t_clear) / dt)), 2))
    s2 = solve_ivp(make_rhs(pmax3), (t_clear, t_end), s1.y[:, -1],
                   t_eval=t2, max_step=dt, rtol=1e-8)
    t = np.concatenate([s1.t, s2.t])
    delta = np.concatenate([s1.y[0], s2.y[0]])
    omega = np.concatenate([s1.y[1], s2.y[1]])
    return t, np.rad2deg(delta), omega


def is_stable(delta_deg, limit_deg=180.0):
    """Stable if the rotor angle stays below the limit for the simulated window."""
    return bool(np.max(delta_deg) < limit_deg)


def critical_clearing_time_numeric(pm, pmax1, pmax2, pmax3, H, f=60.0, t_end=3.0, tol=1e-4):
    """Bisection on clearing time using the swing-curve integration."""
    lo, hi = 0.0, 1.0
    _, d, _ = swing_curve(pm, pmax1, pmax2, pmax3, H, hi, f, t_end)
    while is_stable(d):
        hi *= 2
        _, d, _ = swing_curve(pm, pmax1, pmax2, pmax3, H, hi, f, t_end)
        if hi > 20:
            return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        _, d, _ = swing_curve(pm, pmax1, pmax2, pmax3, H, mid, f, t_end)
        if is_stable(d):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
