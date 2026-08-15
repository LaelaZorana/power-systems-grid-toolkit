"""PV hosting capacity sweep on the IEEE 30-bus case.

Sweeps PV injection at three load buses and reports the hosting capacity at
each for a 1.06 pu voltage ceiling on the injection bus. The 30-bus data is
the classic IEEE test case, see data/README.md.

Run from the repo root:
    python examples/hosting_capacity_30bus.py
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
from psa.data import load_case  # noqa: E402

V_LIM = 1.06
BUSES = [26, 29, 30]  # weak buses at the end of the 30-bus feeders


def main():
    case = load_case("ieee30")
    pv = np.arange(0, 61, 2.5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for bus in BUSES:
        rows, hc = renewables.hosting_capacity_sweep(case, bus, pv, v_max=V_LIM)
        ax.plot(pv, [r["v_bus"] for r in rows], marker="o", ms=3,
                label=f"bus {bus}, hosting capacity {hc:.1f} MW")
        print(f"bus {bus}: hosting capacity {hc:.1f} MW at {V_LIM} pu, "
              f"base voltage {rows[0]['v_bus']:.4f} pu, "
              f"losses at limit {[r['losses_mw'] for r in rows if r['pv_mw'] == hc][0]:.2f} MW")
    ax.axhline(V_LIM, color="k", lw=0.8, ls="--", label=f"{V_LIM} pu ceiling")
    ax.set_xlabel("PV injection (MW)")
    ax.set_ylabel("|V| at injection bus (pu)")
    ax.set_title("PV hosting capacity sweep, IEEE 30-bus")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "hosting_capacity_30bus.png"))
    print("figure written to", FIG)


if __name__ == "__main__":
    main()
