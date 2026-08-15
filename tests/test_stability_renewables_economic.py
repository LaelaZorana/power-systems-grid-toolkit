import numpy as np

from psa import economic, renewables, stability


def test_equal_area_cct_matches_numeric_terminal_fault():
    # Pm 0.8, Pmax pre 2.0, during 0 (terminal fault), post 1.5, H 5 s
    t_ea, dcr = stability.critical_clearing_time_equal_area(0.8, 2.0, 0.0, 1.5, 5.0)
    t_num = stability.critical_clearing_time_numeric(0.8, 2.0, 0.0, 1.5, 5.0, tol=1e-4)
    assert abs(t_ea - t_num) / t_ea < 0.01
    assert 0 < dcr < np.pi


def test_equal_area_cct_matches_numeric_partial_fault():
    t_ea, _ = stability.critical_clearing_time_equal_area(0.8, 2.0, 0.5, 1.5, 5.0)
    t_num = stability.critical_clearing_time_numeric(0.8, 2.0, 0.5, 1.5, 5.0, tol=1e-4)
    assert abs(t_ea - t_num) / t_ea < 0.01


def test_stable_and_unstable_swing():
    t_cr, _ = stability.critical_clearing_time_equal_area(0.8, 2.0, 0.0, 1.5, 5.0)
    _, d_ok, _ = stability.swing_curve(0.8, 2.0, 0.0, 1.5, 5.0, 0.8 * t_cr)
    _, d_bad, _ = stability.swing_curve(0.8, 2.0, 0.0, 1.5, 5.0, 1.2 * t_cr)
    assert stability.is_stable(d_ok)
    assert not stability.is_stable(d_bad)


def test_pv_output_at_stc_equals_rated():
    assert renewables.pv_dc_power(1000.0, 25.0, 250.0) == 250.0
    # ambient chosen so cell temperature is exactly 25 C with NOCT 45
    t_amb = 25.0 - 25.0 / 800.0 * 1000.0
    p = renewables.pv_ac_power(1000.0, t_amb, 250.0, eta_inv=1.0)
    assert abs(p - 250.0) < 1e-9
    # half irradiance gives half power at STC cell temperature
    assert abs(renewables.pv_dc_power(500.0, 25.0, 250.0) - 125.0) < 1e-9


def test_wind_curve():
    p = renewables.wind_power([2, 3, 12, 20, 26], 2.0)
    assert p[0] == 0 and p[1] == 0 and p[2] == 2.0 and p[3] == 2.0 and p[4] == 0


def test_battery_reduces_peak_and_respects_limits():
    load = np.array([50, 45, 42, 40, 40, 45, 55, 70, 85, 90, 92, 95,
                     100, 105, 110, 115, 120, 125, 118, 105, 90, 75, 65, 55.0])
    r = renewables.battery_peak_shave(load, e_max_mwh=60, p_max_mw=20)
    assert r["peak_after"] < r["peak_before"]
    assert np.all(np.abs(r["battery_power"]) <= 20 + 1e-9)
    assert np.all(r["soc"] >= -1e-9) and np.all(r["soc"] <= 60 + 1e-9)


def test_economic_dispatch_saadat_example():
    # Saadat, Power System Analysis, example 7.4: lambda 9.4 $/MWh, P = 450, 325, 200 MW
    units = [dict(a=500, b=5.3, c=0.004, Pmin=200, Pmax=450),
             dict(a=400, b=5.5, c=0.006, Pmin=150, Pmax=350),
             dict(a=200, b=5.8, c=0.009, Pmin=100, Pmax=225)]
    r = economic.economic_dispatch(units, 975)
    assert np.allclose(r["P"], [450, 325, 200], atol=1e-3)
    assert abs(r["lam"] - 9.4) < 1e-3


def test_economic_dispatch_with_losses_balances():
    units = [dict(a=200, b=7.0, c=0.008, Pmin=10, Pmax=85),
             dict(a=180, b=6.3, c=0.009, Pmin=10, Pmax=80),
             dict(a=140, b=6.8, c=0.007, Pmin=10, Pmax=70)]
    B = np.array([[0.0218, 0.0093, 0.0028], [0.0093, 0.0228, 0.0017], [0.0028, 0.0017, 0.0179]]) / 100
    r = economic.economic_dispatch(units, 150, B=B)
    assert abs(r["P"].sum() - 150 - r["losses"]) < 1e-4
    assert r["losses"] > 0
