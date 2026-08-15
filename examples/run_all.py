"""Generate every figure in figures/ and print the key numbers.

Run from the repo root:
    python examples/run_all.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

from psa import economic, fault, loadflow, renewables, stability  # noqa: E402
from psa.data import ieee14  # noqa: E402
from psa.data.ieee14 import REFERENCE_VM  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def fig_voltage_profile():
    res = loadflow.newton_raphson(ieee14())
    buses = np.arange(1, 15)
    ref = [REFERENCE_VM[i] for i in buses]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(buses - 0.2, res.Vm, 0.4, label="Newton-Raphson (this toolkit)")
    ax.bar(buses + 0.2, ref, 0.4, label="MATPOWER case14 reference")
    ax.set_ylim(0.95, 1.10)
    ax.set_xlabel("bus")
    ax.set_ylabel("|V| (pu)")
    ax.set_title("IEEE 14-bus voltage profile")
    ax.set_xticks(buses)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ieee14_voltage_profile.png"))
    print(f"NR: {res.iterations} iterations, losses {res.losses_mw:.2f} MW, slack {res.Pg[0]:.1f} MW")
    return res


def fig_convergence():
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, label in [("nr", "Newton-Raphson"), ("fdlf", "fast decoupled"), ("gs", "Gauss-Seidel")]:
        r = loadflow.solve(ieee14(), m, tol=1e-8) if m != "gs" else loadflow.solve(ieee14(), m, tol=1e-8, max_iter=500)
        h = np.array(r.mismatch_history)
        ax.semilogy(np.arange(len(h)), np.maximum(h, 1e-16), marker="o", ms=3, label=f"{label} ({r.iterations} it)")
        print(f"{label}: {r.iterations} iterations, converged={r.converged}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("max |mismatch| (pu)")
    ax.set_title("IEEE 14-bus power flow convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.png"))


def fig_fault_currents():
    case = ieee14()
    n = 14
    branches = [dict(**{"from": b["from"], "to": b["to"]}, x=b["x"], r=b["r"]) for b in case["branches"]]
    # subtransient reactances for the five machines (assumed, typical values)
    xd2 = {1: 0.25j, 2: 0.25j, 3: 0.30j, 6: 0.30j, 8: 0.30j}
    z1 = fault.build_zbus(n, branches, xd2)
    z2 = z1
    # zero sequence: lines 3x positive sequence, transformers 4-7, 4-9, 5-6 grounded wye-delta
    # modelled as open on the delta side (bus 7, 9, 6 side); machines grounded through xd0 = 0.1
    z0_branches = []
    for b in branches:
        pair = (b["from"], b["to"])
        if pair in [(4, 7), (4, 9), (5, 6)]:
            continue
        z0_branches.append(dict(b, x=3 * b["x"], r=3 * b["r"]))
    x0 = {k: 0.10j for k in xd2}
    x0.update({7: 0.2j, 9: 0.2j, 6: 0.2j})  # delta side ground reference for the study
    z0 = fault.build_zbus(n, z0_branches, x0)
    labels = ["3ph", "SLG", "LL", "DLG"]
    data = {lab: [] for lab in labels}
    for bus in range(1, n + 1):
        m = fault.all_faults_at_bus(z1, z2, z0, bus)
        for lab in labels:
            data[lab].append(m[lab])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(1, n + 1)
    w = 0.2
    for i, lab in enumerate(labels):
        ax.bar(x + (i - 1.5) * w, data[lab], w, label=lab)
    ax.set_xlabel("faulted bus")
    ax.set_ylabel("fault current (pu, 100 MVA base)")
    ax.set_title("IEEE 14-bus fault currents (assumed machine reactances)")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fault_currents.png"))
    print(f"3ph fault at bus 1: {data['3ph'][0]:.2f} pu, at bus 14: {data['3ph'][13]:.2f} pu")


def fig_hosting_capacity():
    pv = np.arange(0, 121, 5)
    bus = 14
    # generator setpoints in this system already put several buses above 1.05, so the
    # limit is applied to the injection bus and its neighbour bus 13 with a 1.06 pu ceiling
    v_lim = 1.06
    rows, hc = renewables.hosting_capacity_sweep(ieee14(), bus, pv, v_max=v_lim,
                                                 monitor_buses=[13, 14])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pv, [r["v_bus"] for r in rows], marker="o", ms=3, label=f"bus {bus} voltage")
    ax.plot(pv, [r["vmax"] for r in rows], ls="--", label="max of buses 13, 14")
    ax.axhline(v_lim, color="k", lw=0.8, label=f"{v_lim} pu limit")
    ax.axvline(hc, color="tab:red", lw=0.8, ls=":", label=f"hosting capacity {hc:.0f} MW")
    ax.set_xlabel(f"PV injected at bus {bus} (MW)")
    ax.set_ylabel("|V| (pu)")
    ax.set_title("PV hosting capacity sweep, IEEE 14-bus")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pv_hosting_capacity.png"))
    print(f"hosting capacity at bus {bus} (Vmax {v_lim}): {hc:.0f} MW")


def fig_peak_shaving():
    hours = np.arange(24)
    load = np.array([50, 45, 42, 40, 40, 45, 55, 70, 85, 90, 92, 95,
                     100, 105, 110, 115, 120, 125, 118, 105, 90, 75, 65, 55.0])
    pv = renewables.pv_ac_power(renewables.clear_sky_irradiance(hours), 25.0, 30.0)
    net = load - pv
    r = renewables.battery_peak_shave(net, e_max_mwh=60, p_max_mw=20)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(hours, load, label="load")
    ax[0].plot(hours, net, label="load minus 30 MW PV")
    ax[0].plot(hours, r["net_load"], label="after 60 MWh / 20 MW battery")
    ax[0].axhline(r["cap"], color="k", lw=0.8, ls="--", label=f"cap {r['cap']:.1f} MW")
    ax[0].set_ylabel("MW")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Battery peak shaving on a daily profile")
    ax[1].bar(hours, r["battery_power"], label="battery power (discharge +)")
    ax[1].plot(hours, r["soc"], color="tab:green", label="state of charge (MWh)")
    ax[1].set_xlabel("hour")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "storage_peak_shaving.png"))
    print(f"peak {r['peak_before']:.1f} MW reduced to {r['peak_after']:.1f} MW")


def fig_swing_curves():
    pm, p1, p2, p3, H = 0.8, 2.0, 0.0, 1.5, 5.0
    tcr, dcr = stability.critical_clearing_time_equal_area(pm, p1, p2, p3, H)
    tnum = stability.critical_clearing_time_numeric(pm, p1, p2, p3, H)
    fig, ax = plt.subplots(figsize=(7, 4))
    for tc, lab in [(0.8 * tcr, "cleared before CCT (stable)"), (1.15 * tcr, "cleared after CCT (unstable)")]:
        t, d, _ = stability.swing_curve(pm, p1, p2, p3, H, tc, t_end=2.0)
        ax.plot(t, d, label=f"{lab}, t_c = {tc:.3f} s")
    ax.axhline(np.rad2deg(dcr), color="k", lw=0.8, ls="--", label=f"critical angle {np.rad2deg(dcr):.1f} deg")
    ax.set_ylim(-60, 200)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotor angle (deg)")
    ax.set_title(f"SMIB swing curves, CCT equal-area {tcr:.3f} s, numeric {tnum:.3f} s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "swing_curves.png"))
    print(f"CCT equal-area {tcr:.4f} s, numeric {tnum:.4f} s, delta_cr {np.rad2deg(dcr):.1f} deg")


def print_dispatch():
    units = [dict(a=500, b=5.3, c=0.004, Pmin=200, Pmax=450),
             dict(a=400, b=5.5, c=0.006, Pmin=150, Pmax=350),
             dict(a=200, b=5.8, c=0.009, Pmin=100, Pmax=225)]
    r = economic.economic_dispatch(units, 975)
    print(f"economic dispatch 975 MW: P = {np.round(r['P'], 1)}, lambda = {r['lam']:.3f} $/MWh, cost {r['cost']:.1f} $/h")


if __name__ == "__main__":
    fig_voltage_profile()
    fig_convergence()
    fig_fault_currents()
    fig_hosting_capacity()
    fig_peak_shaving()
    fig_swing_curves()
    print_dispatch()
    print("figures written to", FIG)
