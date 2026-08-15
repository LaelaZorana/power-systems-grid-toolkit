# Power systems grid toolkit

A compact, tested Python library for steady-state and dynamic power system
analysis, with a set of renewables integration studies on top: PV and wind
models, hosting capacity sweeps, and battery peak shaving. Everything is
written from the equations up with numpy and scipy, and validated against
published results for the IEEE 14-bus system and textbook examples.

## Contents

```
src/psa/
  ybus.py         bus admittance matrix with taps, phase shift, line charging, bus shunts
  loadflow.py     Newton-Raphson (full Jacobian), Gauss-Seidel, fast-decoupled (XB)
                  PV/PQ/slack buses, generator Q limits, line flows and losses
  fault.py        Zbus, three-phase faults, sequence networks, SLG / LL / DLG faults
  economic.py     economic dispatch by lambda iteration, optional B-coefficient losses,
                  priority-list unit commitment heuristic
  renewables.py   PVWatts-style PV array, wind power curve, hosting capacity sweep,
                  battery peak shaving on a daily profile
  stability.py    single machine infinite bus swing equation, equal-area critical
                  clearing angle and time, numeric swing curves
  data/ieee14.py  IEEE 14-bus case as Python dicts with the reference solution
examples/run_all.py   regenerates every figure in figures/
tests/                pytest suite (18 tests)
figures/              output figures
```

## Theory summary

Power flow. The network is described by the bus admittance matrix
Y = G + jB. Injected complex power at each bus is S_i = V_i conj(sum_k Y_ik V_k).
Newton-Raphson linearises the P and Q mismatch equations in polar form and
solves the full Jacobian each iteration, converging quadratically (4
iterations on the 14-bus case to 1e-8). Gauss-Seidel updates one voltage at
a time with an acceleration factor and needs tens of iterations. The
fast-decoupled method exploits the weak P-V and Q-theta coupling of
transmission networks and uses two constant matrices B' (series reactances)
and B'' (imaginary part of Y). PV buses hold voltage magnitude; if a
generator reaches a reactive limit the bus is switched to PQ at that limit.

Faults. Zbus is the inverse of Ybus with machine reactances included. A
bolted three-phase fault at bus k draws I_f = V_f / Z_kk. Unsymmetrical
faults are solved with symmetrical components: the sequence Thevenin
impedances at the faulted bus are connected in series (SLG), in parallel
between positive and negative (LL), or with negative and zero in parallel
(DLG).

Economic dispatch. With quadratic cost curves the optimum satisfies equal
incremental cost across units inside their limits. Lambda is found by
bisection; when losses are modelled by B-coefficients the incremental cost is
scaled by penalty factors 1 / (1 - dP_L/dP_i).

Renewables. PV output follows a PVWatts style model, DC power proportional
to irradiance and corrected linearly for cell temperature. Wind power uses a
cubic curve between cut-in and rated speed. Hosting capacity is estimated by
injecting PV at a bus in steps and running a full power flow at each step.
The battery model finds by bisection the lowest flat peak cap that a given
energy and power rating can hold on a daily load profile.

Transient stability. The single machine infinite bus swing equation
(2H / omega_s) d2delta/dt2 = P_m - P_max sin(delta) is integrated with
scipy for the during-fault and postfault networks. The equal-area criterion
gives the critical clearing angle in closed form and, for a fault at the
machine terminal, the critical clearing time in closed form; the numeric
result is obtained by bisection on the clearing time.

## API

```python
from psa.data import ieee14
from psa import loadflow, fault, economic, renewables, stability

case = ieee14()
res = loadflow.solve(case, "nr")            # or "gs", "fdlf"
res.Vm, res.Va_deg, res.Pg, res.Qg, res.losses_mw, res.line_flows, res.iterations
res = loadflow.newton_raphson(case, enforce_q_limits=True)

zb = fault.build_zbus(n, branches, {bus: x_source_j})
i_f, v = fault.three_phase_fault(zb, bus)
fault.slg_fault(z1, z2, z0); fault.ll_fault(z1, z2); fault.dlg_fault(z1, z2, z0)
fault.all_faults_at_bus(z1bus, z2bus, z0bus, bus)

economic.economic_dispatch(units, demand, B=None)   # units: a, b, c, Pmin, Pmax

renewables.pv_ac_power(G, t_amb, p_dc0)
renewables.wind_power(v, p_rated)
rows, hc = renewables.hosting_capacity_sweep(case, bus, pv_mw_values, v_max=1.05)
renewables.battery_peak_shave(load, e_max_mwh, p_max_mw)

stability.critical_clearing_time_equal_area(pm, pmax1, pmax2, pmax3, H)
stability.critical_clearing_time_numeric(pm, pmax1, pmax2, pmax3, H)
stability.swing_curve(pm, pmax1, pmax2, pmax3, H, t_clear)
```

Case format: a dict with `base_mva`, `buses` (bus, type 3/2/1 for slack/PV/PQ,
Pd, Qd, Gs, Bs, Vm, Va), `gens` (bus, Pg, Qg, Qmax, Qmin, Vset, Pmax, Pmin)
and `branches` (from, to, r, x, b, tap).

## Validation

| Check | Reference | Result | Test |
|---|---|---|---|
| IEEE 14-bus NR bus voltages | MATPOWER case14 Vm and Va (listed in `data/ieee14.py`) | max Vm error 0.1 percent, max angle error 0.02 deg | `test_converges_and_matches_reference` |
| IEEE 14-bus losses and slack | 13.39 MW loss, 232.4 MW slack | 13.39 MW, 232.4 MW | same |
| NR iteration count | quadratic convergence | 4 iterations to 1e-8 pu | `test_nr_iterations_small` |
| Power balance | mismatch below tolerance | 4e-15 pu | `test_power_balance_mismatch_below_tolerance` |
| GS and FDLF agree with NR | same solution | Vm within 1e-5 | `test_methods_agree` |
| Q limit enforcement | Qg clamped, bus switched to PQ | passes | `test_q_limits_keep_generators_inside_limits` |
| Two-bus Zbus by inspection | Z22 = j0.3, If = 3.333 pu | exact | `test_two_bus_hand_zbus` |
| Three-bus Zbus vs hand Ybus | inverse of hand-assembled Ybus | exact | `test_three_bus_vs_hand_zbus` |
| SLG, LL, DLG formulas | 3/(Z1+Z2+Z0), sqrt3/(Z1+Z2), Ia = 0 for DLG | exact | `test_unsymmetrical_textbook` |
| Equal-area CCT vs numeric (terminal fault) | closed form 0.2366 s | numeric 0.2366 s (under 1 percent) | `test_equal_area_cct_matches_numeric_terminal_fault` |
| Equal-area CCT vs numeric (partial fault) | 0.3433 s | 0.3433 s | `..._partial_fault` |
| PV at standard test conditions | rated power | exact | `test_pv_output_at_stc_equals_rated` |
| Economic dispatch | Saadat ex. 7.4: 450 / 325 / 200 MW, lambda 9.4 | matches to 1e-3 | `test_economic_dispatch_saadat_example` |

Key example numbers (from `examples/run_all.py`):
three-phase fault 10.5 pu at bus 1 and 3.3 pu at bus 14 (assumed machine
reactances), PV hosting capacity at bus 14 about 25 MW for a 1.06 pu ceiling,
a 60 MWh / 20 MW battery cuts the 118 MW net peak to 98 MW.

## Figures

`figures/ieee14_voltage_profile.png`, `convergence.png`, `fault_currents.png`,
`pv_hosting_capacity.png`, `storage_peak_shaving.png`, `swing_curves.png`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas pytest
pip install -e .            # optional; tests also work via pyproject pythonpath
pytest -q
python examples/run_all.py  # regenerates figures/
```

## Data source

IEEE 14-bus data from the University of Washington power systems test case
archive as distributed in MATPOWER `case14`. Machine reactances used in the
fault example are assumed typical values, not part of the published case.
