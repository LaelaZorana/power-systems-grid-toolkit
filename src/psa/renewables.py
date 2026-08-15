"""Renewable generation models and grid integration studies.

PV array, after the PVWatts v5 DC power equation (Dobos, PVWatts Version 5
Manual, NREL/TP-6A20-62641, 2014) with an NOCT cell temperature model in
place of the Fuentes model PVWatts itself uses:
    T_cell = T_amb + (NOCT - 20) / 800 * G
    P_dc   = P_dc0 * (G / 1000) * (1 + gamma (T_cell - 25))
    P_ac   = min(P_dc * eta_inv, P_ac_max)
gamma defaults to -0.0047 per degree C, the PVWatts v5 default for standard
crystalline silicon modules. At standard test conditions, G = 1000 W/m2 and
T_cell = 25 C, the DC output equals the rated P_dc0.

Wind turbine: idealised piecewise power curve, cubic between cut-in and rated
speed, flat at rated between rated and cut-out, zero otherwise. This is the
standard textbook idealisation, for example Patel, Wind and Solar Power
Systems, ch. 4. There is no hysteresis at cut-out.

Hosting capacity: add PV injection at a bus in steps, run a power flow, and
record the voltage at the monitored buses. The hosting capacity is the largest
injection that keeps them at or below v_max.

Battery peak shaving: given a daily load profile, find by bisection the lowest
flat cap the battery can hold given its energy and power limits, then
discharge above the cap and recharge below a fill level.
"""

from __future__ import annotations

import copy

import numpy as np

from . import loadflow


# ---------------------------------------------------------------- PV
def pv_cell_temperature(irradiance, t_ambient, noct=45.0):
    return np.asarray(t_ambient, dtype=float) + (noct - 20.0) / 800.0 * np.asarray(irradiance, dtype=float)


def pv_dc_power(irradiance, t_cell, p_dc0, gamma=-0.0047):
    """DC output in same units as p_dc0. gamma is the power temperature coefficient (1/C)."""
    g = np.asarray(irradiance, dtype=float)
    return p_dc0 * (g / 1000.0) * (1.0 + gamma * (np.asarray(t_cell, dtype=float) - 25.0))


def pv_ac_power(irradiance, t_ambient, p_dc0, eta_inv=0.96, p_ac_max=None, gamma=-0.0047, noct=45.0):
    tc = pv_cell_temperature(irradiance, t_ambient, noct)
    p = pv_dc_power(irradiance, tc, p_dc0, gamma) * eta_inv
    if p_ac_max is not None:
        p = np.minimum(p, p_ac_max)
    return np.maximum(p, 0.0)


def clear_sky_irradiance(hours, g_peak=1000.0, sunrise=6.0, sunset=18.0):
    """Simple half-sine irradiance profile for a day (W/m2)."""
    h = np.asarray(hours, dtype=float)
    frac = (h - sunrise) / (sunset - sunrise)
    g = g_peak * np.sin(np.pi * np.clip(frac, 0, 1))
    return np.where((h >= sunrise) & (h <= sunset), g, 0.0)


# ---------------------------------------------------------------- Wind
def wind_power(speed, p_rated, v_cut_in=3.0, v_rated=12.0, v_cut_out=25.0):
    v = np.asarray(speed, dtype=float)
    p = np.zeros_like(v)
    ramp = (v >= v_cut_in) & (v < v_rated)
    p[ramp] = p_rated * ((v[ramp] ** 3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    p[(v >= v_rated) & (v <= v_cut_out)] = p_rated
    return p


# ---------------------------------------------------------------- Hosting capacity
def hosting_capacity_sweep(case, bus, pv_mw_values, v_max=1.05, pf=1.0, method="nr", monitor_buses=None):
    """Add PV at 'bus' as negative load and return (rows, hosting_capacity_mw).

    rows is a list of dicts with pv_mw, vmax, v_bus, losses_mw and converged,
    where vmax is the highest voltage among monitor_buses. The default monitors
    the injection bus only, because other buses may already sit above the limit
    in the base case. Bus 7 next to the 1.09 pu generator in the IEEE 14-bus
    system is one example. Hosting capacity is the largest pv_mw for which
    vmax <= v_max.

    Sign convention for pf below 1.0: the inverter consumes reactive power, so
    Qd at the bus increases. In generator terms this is underexcited or leading
    operation, which lowers the voltage rise from the active injection.
    """
    n = len(case["buses"])
    if not 1 <= bus <= n:
        raise ValueError(f"bus {bus} out of range 1..{n}")
    monitor = np.array(monitor_buses or [bus]) - 1
    rows = []
    for pv in pv_mw_values:
        c = copy.deepcopy(case)
        b = c["buses"][bus - 1]
        b["Pd"] -= pv
        if pf < 1.0:
            b["Qd"] += pv * np.tan(np.arccos(pf))
        res = loadflow.solve(c, method)
        rows.append(dict(pv_mw=pv, vmax=float(res.Vm[monitor].max()), v_bus=float(res.Vm[bus - 1]),
                         losses_mw=res.losses_mw, converged=res.converged))
    ok = [r["pv_mw"] for r in rows if r["converged"] and r["vmax"] <= v_max]
    hc = max(ok) if ok else 0.0
    return rows, hc


# ---------------------------------------------------------------- Storage
def battery_peak_shave(load, e_max_mwh, p_max_mw, dt_h=1.0, eta=0.95, soc0=1.0):
    """Peak-shave a load profile in MW per step with a battery.

    Returns dict with net_load, battery_power with discharge positive, soc in
    MWh, and cap, the achieved peak cap in MW.

    eta is a one-way efficiency applied on both charge and discharge, so the
    round-trip efficiency is eta squared, 0.9025 for the default 0.95. soc0 is
    the starting state of charge as a fraction of e_max_mwh and defaults to a
    full battery. There is no end-of-horizon state of charge constraint, so on
    a profile that ends below the cap the battery recharges toward full.
    """
    load = np.asarray(load, dtype=float)
    if load.size == 0:
        raise ValueError("load profile is empty")
    if e_max_mwh <= 0 or p_max_mw <= 0:
        raise ValueError("e_max_mwh and p_max_mw must be positive")

    def simulate(cap):
        soc = soc0 * e_max_mwh
        p_b = np.zeros_like(load)
        s = np.zeros_like(load)
        fill = cap  # charge whenever load below cap without exceeding cap
        for t, L in enumerate(load):
            if L > cap:
                want = min(L - cap, p_max_mw, soc * eta / dt_h)
                soc -= want * dt_h / eta
                p_b[t] = want
            else:
                room = (e_max_mwh - soc) / (eta * dt_h)
                want = min(fill - L, p_max_mw, room)
                want = max(want, 0.0)
                soc += want * dt_h * eta
                p_b[t] = -want
            s[t] = soc
        return p_b, s

    lo, hi = load.min(), load.max()
    for _ in range(60):
        cap = 0.5 * (lo + hi)
        p_b, _ = simulate(cap)
        if np.max(load - p_b) <= cap + 1e-9:
            hi = cap
        else:
            lo = cap
    cap = hi
    p_b, s = simulate(cap)
    return dict(net_load=load - p_b, battery_power=p_b, soc=s, cap=cap,
                peak_before=float(load.max()), peak_after=float((load - p_b).max()))
