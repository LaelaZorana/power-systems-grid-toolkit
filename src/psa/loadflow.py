"""AC power flow: Newton-Raphson, Gauss-Seidel and fast-decoupled.

Case format (see data/ieee14.py): dict with 'base_mva', 'buses', 'gens',
'branches'. Bus type codes: 3 slack, 2 PV, 1 PQ.

Sign convention: injections positive into the bus, S_i = V_i conj(sum Y_ik V_k).

Newton-Raphson uses the full polar Jacobian
    [dP/dtheta  dP/dV] [dtheta]   [dP]
    [dQ/dtheta  dQ/dV] [dV    ] = [dQ]
with rows for P at all non-slack buses and Q at PQ buses. Generator reactive
limits are enforced by switching a PV bus to PQ when Qg leaves [Qmin, Qmax]
(and back when the voltage error would drive it inside again).

Fast-decoupled uses the XB variant: B' from series reactances only (no
shunts, no taps), B'' from the imaginary part of Ybus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ybus import branch_admittances, build_ybus

SLACK, PV, PQ = 3, 2, 1


@dataclass
class PowerFlowResult:
    converged: bool
    iterations: int
    V: np.ndarray            # complex bus voltages
    Vm: np.ndarray
    Va_deg: np.ndarray
    Pg: np.ndarray           # MW per bus (net generation incl. slack)
    Qg: np.ndarray           # MVAr per bus
    mismatch_history: list
    bus_types: np.ndarray    # final types after Q-limit switching
    line_flows: list = field(default_factory=list)  # dicts per branch
    losses_mw: float = 0.0
    losses_mvar: float = 0.0
    method: str = ""


def _unpack(case):
    buses = case["buses"]
    n = len(buses)
    base = case["base_mva"]
    types = np.array([b["type"] for b in buses])
    Pd = np.array([b["Pd"] for b in buses]) / base
    Qd = np.array([b["Qd"] for b in buses]) / base
    Vm = np.array([b["Vm"] if b["type"] != PQ else 1.0 for b in buses], dtype=float)
    Va = np.deg2rad(np.array([b["Va"] for b in buses], dtype=float))
    Pg = np.zeros(n)
    Qg = np.zeros(n)
    Qmin = np.full(n, -np.inf)
    Qmax = np.full(n, np.inf)
    for g in case["gens"]:
        i = g["bus"] - 1
        Pg[i] += g["Pg"] / base
        Qg[i] += g.get("Qg", 0.0) / base
        Qmin[i] = (0 if np.isinf(Qmin[i]) else Qmin[i]) + g["Qmin"] / base
        Qmax[i] = (0 if np.isinf(Qmax[i]) else Qmax[i]) + g["Qmax"] / base
        if types[i] != SLACK:
            Vm[i] = g["Vset"]
        elif "Vset" in g:
            Vm[i] = g["Vset"]
    shunts = {b["bus"]: (b["Gs"] / base, b["Bs"] / base) for b in buses if b["Gs"] or b["Bs"]}
    Y = build_ybus(n, case["branches"], shunts)
    return n, base, types.copy(), Pd, Qd, Pg, Qg, Vm, Va, Qmin, Qmax, Y


def _injections(Y, V):
    return V * np.conj(Y @ V)


def _jacobian(Y, V, pv_pq, pq):
    """Full polar Jacobian (Saadat / Grainger form)."""
    n = len(V)
    Vm = np.abs(V)
    Ibus = Y @ V
    diagV = np.diag(V)
    diagI = np.diag(Ibus)
    diagVnorm = np.diag(V / Vm)
    dS_dVa = 1j * diagV @ np.conj(diagI - Y @ diagV)
    dS_dVm = diagV @ np.conj(Y @ diagVnorm) + np.conj(diagI) @ diagVnorm
    J11 = dS_dVa[np.ix_(pv_pq, pv_pq)].real
    J12 = dS_dVm[np.ix_(pv_pq, pq)].real
    J21 = dS_dVa[np.ix_(pq, pv_pq)].imag
    J22 = dS_dVm[np.ix_(pq, pq)].imag
    return np.block([[J11, J12], [J21, J22]])


def _line_flows(case, V, base):
    flows = []
    ploss = qloss = 0.0
    for br in case["branches"]:
        if br.get("status", 1) == 0:
            continue
        i, k = br["from"] - 1, br["to"] - 1
        yff, yft, ytf, ytt = branch_admittances(br)
        Sf = V[i] * np.conj(yff * V[i] + yft * V[k]) * base
        St = V[k] * np.conj(ytf * V[i] + ytt * V[k]) * base
        loss = Sf + St
        ploss += loss.real
        qloss += loss.imag
        flows.append(dict(**{"from": br["from"], "to": br["to"]},
                          Pf=Sf.real, Qf=Sf.imag, Pt=St.real, Qt=St.imag,
                          Ploss=loss.real, Qloss=loss.imag))
    return flows, ploss, qloss


def _apply_q_limits(types, Q, Qg, Qd, Qmin, Qmax, Vm, Vset, original_types):
    """Switch PV to PQ if Q outside limits; PQ back to PV if voltage recovers."""
    changed = False
    for i in range(len(types)):
        if original_types[i] != PV:
            continue
        if types[i] == PV:
            qg = Q[i] + Qd[i]
            if qg > Qmax[i]:
                types[i] = PQ
                Qg[i] = Qmax[i]
                changed = True
            elif qg < Qmin[i]:
                types[i] = PQ
                Qg[i] = Qmin[i]
                changed = True
        else:
            # at Qmax and voltage above setpoint (or at Qmin and below): release
            if (Qg[i] >= Qmax[i] and Vm[i] > Vset[i]) or (Qg[i] <= Qmin[i] and Vm[i] < Vset[i]):
                types[i] = PV
                Vm[i] = Vset[i]
                changed = True
    return changed


def newton_raphson(case, tol=1e-8, max_iter=30, enforce_q_limits=False, flat_start=False):
    n, base, types, Pd, Qd, Pg, Qg, Vm, Va, Qmin, Qmax, Y = _unpack(case)
    original_types = types.copy()
    Vset = Vm.copy()
    if flat_start:
        Va[:] = 0.0
    Psp = Pg - Pd
    history = []
    it = 0
    converged = False
    while it < max_iter:
        pv_pq = np.where(types != SLACK)[0]
        pq = np.where(types == PQ)[0]
        Qsp = Qg - Qd
        V = Vm * np.exp(1j * Va)
        S = _injections(Y, V)
        dP = Psp[pv_pq] - S.real[pv_pq]
        dQ = Qsp[pq] - S.imag[pq]
        mis = np.concatenate([dP, dQ])
        norm = np.max(np.abs(mis)) if mis.size else 0.0
        history.append(norm)
        if norm < tol:
            if enforce_q_limits and _apply_q_limits(types, S.imag, Qg, Qd, Qmin, Qmax, Vm, Vset, original_types):
                it += 1
                continue
            converged = True
            break
        J = _jacobian(Y, V, pv_pq, pq)
        dx = np.linalg.solve(J, mis)
        Va[pv_pq] += dx[: len(pv_pq)]
        Vm[pq] += dx[len(pv_pq):]
        it += 1
    return _finish(case, "newton-raphson", converged, it, Vm, Va, Y, Pd, Qd, types, history, base)


def gauss_seidel(case, tol=1e-6, max_iter=2000, accel=1.6):
    n, base, types, Pd, Qd, Pg, Qg, Vm, Va, Qmin, Qmax, Y = _unpack(case)
    V = Vm * np.exp(1j * Va)
    Psp = Pg - Pd
    Qsp = Qg - Qd
    history = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Vold = V.copy()
        for i in range(n):
            if types[i] == SLACK:
                continue
            if types[i] == PV:
                Qi = -np.imag(np.conj(V[i]) * (Y[i] @ V))
                Qsp[i] = Qi
            s = Y[i] @ V - Y[i, i] * V[i]
            Vnew = (np.conj((Psp[i] + 1j * Qsp[i]) / V[i]) - s) / Y[i, i]
            Vnew = V[i] + accel * (Vnew - V[i])
            if types[i] == PV:
                Vnew = Vm[i] * Vnew / abs(Vnew)
            V[i] = Vnew
        S = _injections(Y, V)
        pv_pq = types != SLACK
        pq = types == PQ
        mis = np.concatenate([(Psp - S.real)[pv_pq], (Qsp - S.imag)[pq]])
        norm = np.max(np.abs(mis))
        history.append(norm)
        if norm < tol:
            converged = True
            break
    Vm = np.abs(V)
    Va = np.angle(V)
    return _finish(case, "gauss-seidel", converged, it, Vm, Va, Y, Pd, Qd, types, history, base)


def fast_decoupled(case, tol=1e-8, max_iter=100):
    n, base, types, Pd, Qd, Pg, Qg, Vm, Va, Qmin, Qmax, Y = _unpack(case)
    # B' : series susceptance only, no shunts, no taps (XB scheme)
    Bp = np.zeros((n, n))
    for br in case["branches"]:
        i, k = br["from"] - 1, br["to"] - 1
        bx = 1.0 / br["x"]
        Bp[i, i] += bx
        Bp[k, k] += bx
        Bp[i, k] -= bx
        Bp[k, i] -= bx
    Bpp = -Y.imag
    pv_pq = np.where(types != SLACK)[0]
    pq = np.where(types == PQ)[0]
    Bp_inv = np.linalg.inv(Bp[np.ix_(pv_pq, pv_pq)])
    Bpp_inv = np.linalg.inv(Bpp[np.ix_(pq, pq)])
    Psp = Pg - Pd
    Qsp = Qg - Qd
    history = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        V = Vm * np.exp(1j * Va)
        S = _injections(Y, V)
        dP = (Psp - S.real)[pv_pq] / Vm[pv_pq]
        Va[pv_pq] += Bp_inv @ dP
        V = Vm * np.exp(1j * Va)
        S = _injections(Y, V)
        dQ = (Qsp - S.imag)[pq] / Vm[pq]
        Vm[pq] += Bpp_inv @ dQ
        V = Vm * np.exp(1j * Va)
        S = _injections(Y, V)
        mis = np.concatenate([(Psp - S.real)[pv_pq], (Qsp - S.imag)[pq]])
        norm = np.max(np.abs(mis))
        history.append(norm)
        if norm < tol:
            converged = True
            break
    return _finish(case, "fast-decoupled", converged, it, Vm, Va, Y, Pd, Qd, types, history, base)


def _finish(case, method, converged, it, Vm, Va, Y, Pd, Qd, types, history, base):
    V = Vm * np.exp(1j * Va)
    S = _injections(Y, V)
    Pg = (S.real + Pd) * base
    Qg = (S.imag + Qd) * base
    flows, pl, ql = _line_flows(case, V, base)
    return PowerFlowResult(converged, it, V, Vm, np.rad2deg(Va), Pg, Qg, history,
                           types, flows, pl, ql, method)


def solve(case, method="nr", **kw):
    if method == "nr":
        return newton_raphson(case, **kw)
    if method == "gs":
        return gauss_seidel(case, **kw)
    if method == "fdlf":
        return fast_decoupled(case, **kw)
    raise ValueError(method)


def power_balance_mismatch(case, result):
    """Largest |S_specified - S_calculated| in pu over non-slack P and PQ Q."""
    n, base, types, Pd, Qd, Pg, Qg, Vm, Va, Qmin, Qmax, Y = _unpack(case)
    S = _injections(Y, result.V)
    dP = (Pg - Pd - S.real)[types != SLACK]
    dQ = (Qg - Qd - S.imag)[types == PQ]   # original PQ buses only
    return float(np.max(np.abs(np.concatenate([dP, dQ]))))
