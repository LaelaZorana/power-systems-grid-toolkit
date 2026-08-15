"""Yearly battery dispatch on the synthetic utility profiles.

Runs the peak shaving battery day by day over the 8760 hour synthetic load
net of PV and wind, then reports peak reduction and energy throughput for the
year and plots the peak week plus a monthly summary. All input series are
synthetic, see data/README.md.

Run from the repo root:
    python examples/storage_dispatch_year.py
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

from psa import renewables  # noqa: E402
from psa.data import load_profile  # noqa: E402

PV_MW = 30.0
WIND_MW = 20.0
E_MAX_MWH = 60.0
P_MAX_MW = 20.0


def main():
    load = np.array(load_profile("load_8760", "load_mw"))
    pv = PV_MW * np.array(load_profile("pv_cf_8760", "cf"))
    wind = WIND_MW * np.array(load_profile("wind_cf_8760", "cf"))
    net = load - pv - wind

    # dispatch one day at a time so each day gets its own peak cap, carrying
    # the state of charge across midnight
    battery = np.zeros(8760)
    soc = np.zeros(8760)
    soc_frac = 1.0
    for d in range(365):
        s = slice(24 * d, 24 * (d + 1))
        r = renewables.battery_peak_shave(net[s], E_MAX_MWH, P_MAX_MW, soc0=soc_frac)
        battery[s] = r["battery_power"]
        soc[s] = r["soc"]
        soc_frac = r["soc"][-1] / E_MAX_MWH
    after = net - battery

    daily_peaks_before = net.reshape(365, 24).max(axis=1)
    daily_peaks_after = after.reshape(365, 24).max(axis=1)
    discharge = battery[battery > 0].sum()
    charge = -battery[battery < 0].sum()
    cycles = discharge / E_MAX_MWH

    print(f"annual peak before {net.max():.1f} MW, after {after.max():.1f} MW, "
          f"reduction {net.max() - after.max():.1f} MW")
    print(f"mean daily peak reduction {np.mean(daily_peaks_before - daily_peaks_after):.1f} MW")
    print(f"energy throughput: discharge {discharge:.0f} MWh, charge {charge:.0f} MWh, "
          f"about {cycles:.0f} full cycle equivalents")
    print(f"round-trip energy ratio {discharge / charge:.3f}")

    peak_day = int(np.argmax(daily_peaks_before))
    wk = slice(24 * max(peak_day - 3, 0), 24 * min(peak_day + 4, 365))
    t = np.arange(8760)[wk] / 24.0
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, net[wk], label="net load, synthetic")
    ax[0].plot(t, after[wk], label=f"after {E_MAX_MWH:.0f} MWh / {P_MAX_MW:.0f} MW battery")
    ax[0].set_ylabel("MW")
    ax[0].set_title("Peak week of the synthetic year, battery peak shaving")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, soc[wk], color="tab:green")
    ax[1].set_ylabel("state of charge (MWh)")
    ax[1].set_xlabel("day of year")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "storage_dispatch_peak_week.png"))

    month_edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    mb = [daily_peaks_before[month_edges[m]:month_edges[m + 1]].max() for m in range(12)]
    ma = [daily_peaks_after[month_edges[m]:month_edges[m + 1]].max() for m in range(12)]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, 13)
    ax.bar(x - 0.2, mb, 0.4, label="monthly peak before")
    ax.bar(x + 0.2, ma, 0.4, label="monthly peak after")
    ax.set_xlabel("month")
    ax.set_ylabel("MW")
    ax.set_title("Monthly net load peaks, synthetic year, before and after storage")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "storage_dispatch_monthly_peaks.png"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
