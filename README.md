# Power systems grid toolkit

Grid analysis tools usually come as sealed commercial packages, but the core algorithms are just linear algebra, so this library builds them from the equations up in numpy and scipy: steady state and dynamic power system analysis with renewables integration studies on top, meaning PV and wind models, hosting capacity sweeps and battery peak shaving. Everything is validated against published results for the IEEE 14-bus system and textbook examples. The 14-bus voltages match MATPOWER to 0.1 percent.

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

Power flow starts from the bus admittance matrix Y = G + jB, and the injected complex power at each bus is S_i = V_i conj(sum_k Y_ik V_k). Newton Raphson linearises the P and Q mismatch equations in polar form and solves the full Jacobian each iteration, so it converges quadratically and takes 4 iterations on the 14-bus case to reach 1e-8. Gauss Seidel updates one voltage at a time with an acceleration factor and needs tens of iterations to do the same job. The fast decoupled method exploits the weak coupling between P and V and between Q and theta in transmission networks, so it gets away with two constant matrices, B prime from the series reactances and B double prime from the imaginary part of Y. PV buses hold voltage magnitude, and when a generator hits a reactive limit the bus is switched to PQ at that limit.

Faults work through Zbus, the inverse of Ybus with machine reactances included. A bolted three phase fault at bus k draws I_f = V_f / Z_kk. Unsymmetrical faults use symmetrical components: the sequence Thevenin impedances at the faulted bus connect in series for SLG, in parallel between positive and negative for LL, or with negative and zero in parallel for DLG.

Economic dispatch with quadratic cost curves reduces to one condition, equal incremental cost across all units inside their limits. Lambda is found by bisection, and when losses are modelled with B coefficients the incremental cost is scaled by penalty factors of 1 over 1 minus dP_L/dP_i.

Renewables follow simple physical models. PV output uses a PVWatts style model where DC power is proportional to irradiance and corrected linearly for cell temperature, and wind power follows a cubic curve between cut in and rated speed. Hosting capacity is estimated by injecting PV at a bus in steps and running a full power flow at each step until a voltage ceiling is hit. The battery model finds by bisection the lowest flat peak cap that a given energy and power rating can hold on a daily load profile.

Transient stability integrates the single machine infinite bus swing equation, 2H over omega_s times d2delta/dt2 = P_m - P_max sin delta, with scipy for the during fault and postfault networks. The equal area criterion gives the critical clearing angle in closed form, and for a fault at the machine terminal it gives the critical clearing time in closed form too, while the numeric answer comes from bisection on the clearing time. The two agree to under 1 percent.

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

The case format is a dict with `base_mva`, `buses` holding bus, type 3/2/1 for slack, PV and PQ, Pd, Qd, Gs, Bs, Vm and Va, `gens` holding bus, Pg, Qg, Qmax, Qmin, Vset, Pmax and Pmin, and `branches` holding from, to, r, x, b and tap.

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

Key numbers from `examples/run_all.py`: a three phase fault draws 10.5 pu at bus 1 and 3.3 pu at bus 14 with the assumed machine reactances, PV hosting capacity at bus 14 is about 25 MW for a 1.06 pu ceiling, and a 60 MWh, 20 MW battery cuts the 118 MW net peak to 98 MW.

## Figures

`figures/ieee14_voltage_profile.png`, `convergence.png`, `fault_currents.png`,
`pv_hosting_capacity.png`, `storage_peak_shaving.png`, `swing_curves.png`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas pytest
pip install -e .            # optional, tests also work through pyproject pythonpath
pytest -q
python examples/run_all.py  # regenerates figures/
```

## Data source

The IEEE 14-bus data comes from the University of Washington power systems test case archive as distributed in MATPOWER `case14`. Machine reactances used in the fault example are assumed typical values and are not part of the published case.
