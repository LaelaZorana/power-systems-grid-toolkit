# Power systems grid toolkit

Grid analysis tools usually come as sealed commercial packages, but the core algorithms are just linear algebra, so this library builds them from the equations up in numpy and scipy: steady state and dynamic power system analysis with renewables integration studies on top, meaning PV and wind models, hosting capacity sweeps and battery storage dispatch. The load flow, fault, dispatch and stability modules are checked against published numbers from the IEEE test systems and textbook worked examples, while the renewables and storage models are checked for internal consistency, since there is no published benchmark to pin them to. The 14-bus voltages match the IEEE Common Data Format printed solution to 0.13 percent in magnitude and 0.02 degrees in angle.

## Contents

```
src/psa/
  ybus.py         bus admittance matrix with taps, phase shift, line charging, bus shunts
  loadflow.py     Newton-Raphson with the full Jacobian, Gauss-Seidel, fast-decoupled XB
                  PV, PQ and slack buses, generator Q limits, line flows and losses
  fault.py        Zbus, three-phase faults, sequence networks, SLG, LL and DLG faults
  economic.py     economic dispatch by lambda iteration, optional B-coefficient losses,
                  priority-list unit commitment heuristic
  renewables.py   PVWatts based PV model, wind power curve, hosting capacity sweep,
                  battery peak shaving
  stability.py    single machine infinite bus swing equation, equal-area critical
                  clearing angle and time, numeric swing curves
  data/ieee14.py  IEEE 14-bus case as Python dicts with the reference solution
  data/loader.py  JSON case loader and CSV profile loader with validation
data/cases/       IEEE 14, 30 and 57 bus cases in JSON, sources in data/README.md
data/profiles/    synthetic 8760 hour load, PV and wind series plus their generator
examples/         run_all.py, storage_dispatch_year.py, hosting_capacity_30bus.py
tests/            pytest suite
figures/          output figures
```

## Theory summary

Power flow starts from the bus admittance matrix Y = G + jB, and the injected complex power at each bus is S_i = V_i conj(sum_k Y_ik V_k). Newton Raphson linearises the P and Q mismatch equations in polar form and solves the full Jacobian each iteration, so it converges quadratically and takes 4 iterations on the 14-bus case to reach 1e-8. Gauss Seidel updates one voltage at a time with an acceleration factor and needs tens of iterations to do the same job. The fast decoupled method exploits the weak coupling between P and V and between Q and theta in transmission networks, so it gets away with two constant matrices, B prime from the series reactances of in-service branches and B double prime from the imaginary part of Y. PV buses hold voltage magnitude, and when a generator hits a reactive limit the bus is switched to PQ at that limit. Reactive limit enforcement is implemented in the Newton-Raphson solver only, and the other methods refuse the option rather than ignoring it.

Faults work through Zbus, the inverse of Ybus with machine reactances included. A bolted three phase fault at bus k draws I_f = V_f / Z_kk. Unsymmetrical faults use symmetrical components: the sequence Thevenin impedances at the faulted bus connect in series for SLG, in parallel between positive and negative for LL, or with negative and zero in parallel for DLG.

Economic dispatch with quadratic cost curves reduces to one condition, equal incremental cost across all units inside their limits. Lambda is found by bisection, and when losses are modelled with B coefficients the incremental cost is scaled by penalty factors of 1 over 1 minus dP_L/dP_i. Infeasible demand raises an error instead of returning a silent mismatch.

Renewables follow simple physical models. PV output uses the PVWatts v5 DC power equation from Dobos 2014 with an NOCT cell temperature model, so DC power is proportional to irradiance and corrected linearly for cell temperature, while wind power follows the standard idealised cubic curve between cut in and rated speed. Hosting capacity is estimated by injecting PV at a bus in steps and running a full power flow at each step until a voltage ceiling is hit, and the battery model finds by bisection the lowest flat peak cap that a given energy and power rating can hold on a load profile.

Transient stability integrates the single machine infinite bus swing equation, 2H over omega_s times d2delta/dt2 = P_m - P_max sin delta, with scipy for the during fault and postfault networks. The equal area criterion gives the critical clearing angle in closed form, and for a fault at the machine terminal it gives the critical clearing time in closed form too, while the numeric answer comes from bisection on the clearing time. The two agree to under 1 percent.

## API

```python
from psa.data import ieee14, load_case, load_profile
from psa import loadflow, fault, economic, renewables, stability

case = ieee14()                             # embedded 14-bus case
case = load_case("ieee30")                  # or "ieee14", "ieee57" from data/cases
load = load_profile("load_8760", "load_mw") # 8760 hour series from data/profiles

res = loadflow.solve(case, "nr")            # or "gs", "fdlf"
res.Vm, res.Va_deg, res.Pg, res.Qg, res.losses_mw, res.line_flows, res.iterations
res = loadflow.newton_raphson(case, enforce_q_limits=True)   # NR only

zb = fault.build_zbus(n, branches, {bus: x_source_j})
i_f, v = fault.three_phase_fault(zb, bus)
fault.slg_fault(z1, z2, z0)
fault.ll_fault(z1, z2)
fault.dlg_fault(z1, z2, z0)
fault.all_faults_at_bus(z1bus, z2bus, z0bus, bus)

economic.economic_dispatch(units, demand, B=None)   # units: a, b, c, Pmin, Pmax

renewables.pv_ac_power(G, t_amb, p_dc0)
renewables.wind_power(v, p_rated)
rows, hc = renewables.hosting_capacity_sweep(case, bus, pv_mw_values, v_max=1.05)
renewables.battery_peak_shave(load, e_max_mwh, p_max_mw)

stability.critical_clearing_time_equal_area(pm, pmax1, pmax2, pmax3, H)  # (t_cr, delta_cr)
stability.critical_clearing_time_numeric(pm, pmax1, pmax2, pmax3, H)
stability.swing_curve(pm, pmax1, pmax2, pmax3, H, t_clear)
```

The case format is a dict with `base_mva`, `buses` holding bus, type 3, 2 or 1 for slack, PV and PQ, Pd, Qd, Gs, Bs, Vm and Va, `gens` holding bus, Pg, Qg, Qmax, Qmin, Vset, Pmax and Pmin, and `branches` holding from, to, r, x, b, tap and optionally shift and status. The battery model applies its efficiency one way on each of charge and discharge, so the default 0.95 means a 0.9025 round trip, and there is no end-of-horizon state of charge constraint.

## Validation

| Check | Reference | Result | Test |
|---|---|---|---|
| IEEE 14-bus NR bus voltages | IEEE CDF printed solution, listed in `data/ieee14.py` | max Vm error 0.13 percent, max angle error 0.02 deg, test bounds 0.2 percent and 0.025 deg | `test_converges_and_matches_reference` |
| IEEE 14-bus losses and slack | 13.39 MW loss, 232.4 MW slack | 13.39 MW, 232.4 MW | same |
| IEEE 30 and 57 bus cases | published loss ballparks 17.5 to 17.6 MW and 27.9 MW | 17.56 MW and 27.86 MW, NR in 3 and 4 iterations | `test_case_converges_with_expected_losses` |
| NR iteration count | quadratic convergence | 4 iterations to 1e-8 pu | `test_nr_iterations_small` |
| Power balance | mismatch below tolerance | 4e-15 pu | `test_power_balance_mismatch_below_tolerance` |
| GS and FDLF agree with NR | same solution | Vm within 1e-5 | `test_methods_agree` |
| FDLF honors branch outages | NR on the same outage case | Vm within 1e-5 | `test_fdlf_honors_branch_status` |
| Q limit enforcement | Qg clamped, bus switched to PQ | passes | `test_q_limits_keep_generators_inside_limits` |
| Two-bus Zbus by inspection | Z22 = j0.3, If = 3.333 pu | exact | `test_two_bus_hand_zbus` |
| Three-bus Zbus vs hand Ybus | inverse of hand-assembled Ybus | exact | `test_three_bus_vs_hand_zbus` |
| SLG, LL, DLG faults | hand-worked currents and boundary conditions, faulted phase voltages zero | exact | `test_unsymmetrical_boundary_conditions_and_hand_values` |
| Equal-area CCT vs numeric, terminal fault | closed form 0.2366 s | numeric 0.2366 s, under 1 percent | `test_equal_area_cct_matches_numeric_terminal_fault` |
| Equal-area CCT vs numeric, partial fault | 0.3433 s | 0.3433 s | `..._partial_fault` |
| PV at standard test conditions | rated power | exact | `test_pv_output_at_stc_equals_rated` |
| Economic dispatch | Saadat 2nd ed. example 7.5, 450, 325 and 200 MW at lambda 9.4 | matches to 1e-3 | `test_economic_dispatch_saadat_example` |
| Loss dispatch optimality | equal penalised incremental cost across units | equal to 1e-6 | `test_economic_dispatch_with_losses_balances_and_is_optimal` |

Key numbers from the examples: a three phase fault draws 10.5 pu at bus 1 and 3.3 pu at bus 14 with the assumed machine reactances, PV hosting capacity is about 25 MW at 14-bus bus 14 and 17.5 to 27.5 MW at the weak 30-bus buses 26, 29 and 30 for a 1.06 pu ceiling, and over the synthetic year a 60 MWh, 20 MW battery cuts the 147 MW annual net peak to 127 MW with about 20 GWh discharged, roughly 337 full cycle equivalents.

## Fault study assumptions

The published IEEE 14-bus case carries no sequence data, so the fault example in `examples/run_all.py` assumes all of it, and the SLG and DLG results are illustrative rather than reproducible from a published source. The assumptions, also stated in the code: machine subtransient reactances are typical values, the negative sequence network equals the positive sequence one with X2 = X''d, line zero sequence reactance is 3 times positive sequence, the three transformers 4-7, 4-9 and 5-6 are grounded wye on buses 4 and 5 and delta on buses 7, 9 and 6, so each contributes its leakage reactance as a shunt to ground at the wye bus and leaves the delta side open, and every machine is assumed grounded through j0.1 pu in the zero sequence, which is what gives the 6 to 14 region a ground source. Real machine grounding varies from solid to none.

## Data

`data/cases/` holds the IEEE 14, 30 and 57 bus systems in a documented JSON format, transcribed mechanically from the MATPOWER distribution of the University of Washington archive files, with NR convergence and known loss figures checked in the test suite. `data/profiles/` holds a synthetic 8760 hour utility load profile, a synthetic PV capacity factor year and a synthetic Weibull driven wind capacity factor year, all generated by the committed script `make_profiles.py` with a fixed seed and labeled synthetic in their headers. Formats, sources and the validation numbers are documented in `data/README.md`.

## Figures

`figures/ieee14_voltage_profile.png`, `convergence.png`, `fault_currents.png`,
`pv_hosting_capacity.png`, `storage_peak_shaving.png`, `swing_curves.png`,
`storage_dispatch_peak_week.png`, `storage_dispatch_monthly_peaks.png`,
`hosting_capacity_30bus.png`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pytest
pip install -e .            # optional, tests also work through pyproject pythonpath
pytest -q
python examples/run_all.py                 # regenerates the core figures
python examples/storage_dispatch_year.py   # yearly storage dispatch on the profiles
python examples/hosting_capacity_30bus.py  # hosting capacity on the 30-bus case
```

## License

MIT, see LICENSE.
