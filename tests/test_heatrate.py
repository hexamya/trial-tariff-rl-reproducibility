"""Tests for the temperature-dependent heat-rate and derating models (S4)."""
import numpy as np
import pandas as pd
import pytest

from ttrl.supply import dispatch as D
from ttrl.supply import heatrate as H
from ttrl.supply import units as U
from ttrl.supply.fleet import cems_year_hours


# --------------------------------------------------------------------------
# Heat-input model
# --------------------------------------------------------------------------
def _synthetic_unit(n=3000, seed=0, d=-2.0, e=0.01, outliers=True):
    rng = np.random.default_rng(seed)
    p = rng.uniform(150, 300, n)
    t = rng.uniform(0, 40, n)
    h = 400 + 6.0 * p + 0.003 * p ** 2 + (t - 15) * (d + e * p) + rng.normal(0, 4, n)
    if outliers:
        h[:100] += 600
    day = np.repeat(np.arange(n // 24 + 1), 24)[:n]
    return p, h, t, day


def test_fit_temp_heat_curve_recovers_terms():
    p, h, t, day = _synthetic_unit()
    fit = H.fit_temp_heat_curve(p, h, t, pmin=150, pmax=300, day=day)
    assert fit["method"] == "quadratic"
    assert fit["d"] == pytest.approx(-2.0, abs=0.3)
    assert fit["e"] == pytest.approx(0.01, abs=0.0015)
    assert fit["b"] == pytest.approx(6.0, rel=0.03)
    assert np.isfinite(fit["se_d"]) and np.isfinite(fit["se_e"]) and fit["se_e"] > 0


def test_fit_temp_heat_curve_drops_concave_term():
    rng = np.random.default_rng(1)
    p = rng.uniform(50, 150, 800)
    t = rng.uniform(5, 35, 800)
    h = 300 + 12.0 * p - 0.02 * p ** 2 + 0.5 * (t - 15) + rng.normal(0, 2, 800)
    fit = H.fit_temp_heat_curve(p, h, t, pmin=50, pmax=150)
    assert fit["method"] == "linear" and fit["c"] == 0.0


def test_hr_change_per_c_matches_definition():
    par = {"a": 400.0, "b": 6.0, "c": 0.0, "d": -2.0, "e": 0.02}
    p = 200.0
    h15 = 400 + 6 * 200
    assert H.hr_change_per_c(par, p) == pytest.approx(100 * (-2 + 0.02 * 200) / h15)
    # Heat input at 25 C is the 15 C value plus 10 x (d + e P).
    assert H.heat_input(par, p, 25.0) == pytest.approx(h15 + 10 * (-2 + 4))


def test_within_day_estimate_ignores_seasonal_drift():
    """A slow drift correlated with seasonal temperature biases the pooled
    fit but not the day-fixed-effect estimate."""
    rng = np.random.default_rng(2)
    days = 200
    hours = np.arange(days * 24)
    day = hours // 24
    t = 10 + 15 * np.sin(np.pi * day / days) + 5 * np.sin(2 * np.pi * (hours % 24) / 24)
    p = rng.uniform(150, 300, hours.size)
    drift = 0.04 * (t - t.mean()) * 30      # heat drifts with the seasonal cycle only
    drift = pd.Series(drift).groupby(day).transform("mean").to_numpy()
    h = 400 + 6 * p + (t - 15) * 1.0 + drift + rng.normal(0, 2, hours.size)
    d_wd, e_wd = H.within_day_temp_effect(p, h, t, day, pref=300)
    assert d_wd + e_wd * 225 == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------
# Derating
# --------------------------------------------------------------------------
def _derating_unit(kappa=0.5, n=20000, seed=3, mild_cap=None):
    """Hours of a unit with capability 100 (1 - kappa/100 (T - 15)) above 15 C.

    20% of hours run at capability, the rest below it. With ``mild_cap`` the
    unit is never run above that output when T < 25 C (dispatch-limited).
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(5, 40, n)
    cap = 100 * H.derate_factor(kappa, t)
    at_max = rng.random(n) < 0.2
    p = np.where(at_max, cap, cap * rng.uniform(0.5, 0.95, n))
    if mild_cap is not None:
        p = np.where(t < 25, np.minimum(p, mild_cap), p)
    month = rng.integers(1, 13, n)
    return p, t, month


def test_envelope_derating_recovers_kappa():
    p, t, month = _derating_unit(kappa=0.5)
    out = H.unit_derating(p, t, month)
    assert out["kappa"] == pytest.approx(0.5, abs=0.05)
    assert out["p15"] == pytest.approx(100, rel=0.02)
    assert out["n_bins"] >= 10 and out["months_dropped"] == 0


def test_monotone_envelope_removes_dispatch_limited_dip():
    """Flat capability, but mild hours are capped at 85 MW by dispatch: the raw
    envelope shows capability rising with temperature, the monotone one not."""
    p, t, month = _derating_unit(kappa=0.0, mild_cap=85.0)
    out = H.unit_derating(p, t, month)
    assert out["raw_kappa"] < -0.2
    assert out["kappa"] == pytest.approx(0.0, abs=0.02)


def test_limited_months_flags_partial_outage_months():
    p, t, month = _derating_unit(kappa=0.3)
    p = np.where(month == 6, 0.5 * p, p)
    assert H.limited_months(month, p, 0.7) == [6]
    out = H.unit_derating(p, t, month)
    assert out["months_dropped"] == 1
    assert out["kappa"] == pytest.approx(0.3, abs=0.06)


def test_fit_derating_needs_enough_bins():
    bins = pd.DataFrame({"t_mid": [17.0, 19.0, 21.0], "n": [50, 50, 50], "p_env": [100, 99, 98]})
    assert np.isnan(H.fit_derating(bins)["kappa"])


# --------------------------------------------------------------------------
# Unit-hour preparation
# --------------------------------------------------------------------------
def test_steady_rows_excludes_starts_stops_and_ramps():
    ts = cems_year_hours(2024)[:10]
    load = [50, 100, 100, 100, 100, 130, 130, 130, 100, 100]
    op = [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    uh = pd.DataFrame({"unit_id": "1_A", "operating_datetime_utc": ts,
                       "operating_time_hours": op, "gross_load_mw": load,
                       "heat": np.array(load) * 8.0, "clean": np.array(op) >= 1})
    s = H.steady_rows(uh).tolist()
    # Row 0 partial; rows 1-2 lack two full previous hours; row 5 ramps by
    # 30 MW (> 10% of Pref = 130); row 8 ramps down; row 9 has no next hour.
    assert s == [False, False, False, True, True, False, True, True, False, False]


def test_add_temperature_uses_hour_beginning_mean():
    ts = cems_year_hours(2024)[:3]
    temp = pd.DataFrame({"plant_id_eia": 7, "ts_utc": list(ts) + [ts[-1] + pd.Timedelta(hours=1)],
                         "temp_c": [10.0, 12.0, 20.0, 22.0]})
    units = pd.DataFrame({"plant_id_epa": [70], "emissions_unit_id_epa": ["A"], "plant_id_eia": [7]})
    uh = pd.DataFrame({"unit_id": "70_A", "operating_datetime_utc": ts})
    out = H.add_temperature(uh, units, temp)
    assert out["temp_c"].tolist() == pytest.approx([11.0, 16.0, 21.0])
    # A missing next observation leaves the hour without a temperature.
    out2 = H.add_temperature(uh, units, temp.iloc[[0, 1, 3]])
    assert np.isnan(out2["temp_c"].iloc[1])


# --------------------------------------------------------------------------
# Curve model for dispatch
# --------------------------------------------------------------------------
def _table():
    return pd.DataFrame({"unit_id": ["u1", "u2"], "a": [100.0, 50.0], "b": [7.0, 10.0],
                         "c": [0.002, 0.0], "d": [-1.0, 0.0], "e": [0.01, 0.0],
                         "pmin": [50.0, 20.0], "p15": [300.0, 100.0], "kappa": [0.2, 0.5],
                         "t_fit_lo": [0.0, np.nan], "t_fit_hi": [35.0, np.nan],
                         "co2_per_mmbtu": [0.0585, 0.0585]})


def test_temperature_curves_arrays():
    m = H.TemperatureCurves(_table())
    temp = np.array([[15.0, 15.0], [40.0, 40.0], [5.0, 5.0]])
    cv = m.arrays(temp)
    assert cv.a.shape == (3, 2) and cv.pmax.shape == (3, 2)
    assert cv.pmax[0] == pytest.approx([300, 100])
    assert cv.pmax[1] == pytest.approx([300 * (1 - 0.002 * 25), 100 * (1 - 0.005 * 25)])
    assert cv.pmax[2] == pytest.approx([300, 100])          # flat below 15 C
    # Heat terms of u1 clipped at its fit range (35 C); derating is not.
    assert cv.a[1, 0] == pytest.approx(100 - 1.0 * 20)
    assert cv.b[1, 0] == pytest.approx(7 + 0.01 * 20)
    assert m.arrays(None).pmax.shape == (1, 2)
    off = H.TemperatureCurves(_table(), use_heat=False, use_derate=False).arrays(temp)
    assert off.a[1] == pytest.approx([100, 50]) and off.pmax[1] == pytest.approx([300, 100])


def test_dispatch_respects_derated_capacity():
    m = H.TemperatureCurves(_table())
    temp = np.array([[15.0, 15.0], [40.0, 40.0]])
    cv = m.arrays(temp)
    target = np.array([350.0, 350.0])
    res = D.economic_dispatch(target, cv, np.ones((2, 2), bool))
    assert res.p.sum(axis=1) == pytest.approx(target)
    assert (res.p <= cv.pmax + 1e-9).all()
    # Hot hour: u1 is derated, so the more expensive u2 must carry more.
    assert res.p[1, 1] > res.p[0, 1]


# --------------------------------------------------------------------------
# Whole-table fit and summaries
# --------------------------------------------------------------------------
def test_fit_temperature_curves_end_to_end():
    rng = np.random.default_rng(4)
    ts = cems_year_hours(2024)[:3000]
    tday = 20 + 12 * np.sin(2 * np.pi * np.arange(3000) / 3000) \
        + 5 * np.sin(2 * np.pi * (np.arange(3000) % 24) / 24)
    rows = []
    for unit, cap, n_on in [("A", 200.0, 3000), ("B", 100.0, 150)]:
        p = np.clip(rng.uniform(0.6, 1.0, 3000) * cap * H.derate_factor(0.4, tday), 1, None)
        h = 0.3 * cap + 9.0 * p + (tday - 15) * 0.02 * p + rng.normal(0, 1, 3000)
        op = np.where(np.arange(3000) < n_on, 1.0, 0.0)
        rows.append(pd.DataFrame({
            "plant_id_epa": 1, "emissions_unit_id_epa": unit, "plant_id_eia": 1,
            "operating_datetime_utc": ts, "operating_time_hours": op,
            "gross_load_mw": np.where(op > 0, p, np.nan),
            "heat_content_mmbtu": np.where(op > 0, h, np.nan),
            "co2_mass_tons": np.where(op > 0, 0.0585 * h, np.nan), "tech": "CT", "chp": False}))
    cems = pd.concat(rows, ignore_index=True)
    units = pd.DataFrame({"plant_id_epa": 1, "emissions_unit_id_epa": ["A", "B"],
                          "plant_id_eia": 1, "plant_name_eia": "P", "tech": "CT",
                          "chp": False, "generator_ids": ["G1", "G2"]})
    gens = pd.DataFrame({"plant_id_eia": 1, "generator_id": ["G1", "G2"],
                         "capacity_mw": [200.0, 100.0], "summer_capacity_mw": [180.0, 90.0],
                         "winter_capacity_mw": [200.0, 100.0], "minimum_load_mw": [80.0, 40.0]})
    temp = pd.DataFrame({"plant_id_eia": 1, "ts_utc": list(ts) + [ts[-1] + pd.Timedelta(hours=1)],
                         "temp_c": np.r_[tday, tday[-1]]})
    uh = H.add_temperature(U.prepare_unit_hours(cems), units, temp)
    uh["steady"] = H.steady_rows(uh)
    tab = H.fit_temperature_curves(uh, units, gens, H.TempConfig(min_steady_hours=100),
                                   U.FitConfig(min_hours=50)).set_index("unit_id")
    assert tab.loc["1_A", "hr_method"] in ("quadratic", "linear")
    assert tab.loc["1_A", "kappa_method"] == "own"
    assert tab.loc["1_A", "kappa"] == pytest.approx(0.4, abs=0.1)
    assert tab.loc["1_A", "e"] == pytest.approx(0.02, abs=0.004)
    # Unit B has too few hours: group terms and group kappa.
    assert tab.loc["1_B", "hr_method"] == "group"
    assert tab.loc["1_B", "kappa_method"] == "group"
    assert tab.loc["1_B", "kappa"] == pytest.approx(tab.loc["1_A", "kappa"])
    assert tab.loc["1_B", "e"] == pytest.approx(tab.loc["1_A", "e"])
    assert tab.loc["1_A", "summer_mw"] == 180.0


def test_limited_months_left_out_of_heat_fit():
    """A CC that runs in a reduced mode (half output, much higher heat rate)
    through the warm months must not get that mode's heat rate as a
    temperature effect: those capability-limited months are left out."""
    rng = np.random.default_rng(6)
    ts = cems_year_hours(2024)
    n = len(ts)
    month = (ts - U.LST_OFFSET).month.to_numpy()
    tday = 20 + 12 * np.sin(2 * np.pi * (np.arange(n) / n - 0.3)) \
        + 5 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24)
    reduced = np.isin(month, [5, 6, 7, 8, 9, 10])
    p = np.where(reduced, rng.uniform(140, 150, n), rng.uniform(220, 300, n))
    h = np.where(reduced, 12.5 * p, 300 + 6.5 * p) + rng.normal(0, 2, n)
    cems = pd.DataFrame({"plant_id_epa": 1, "emissions_unit_id_epa": "A", "plant_id_eia": 1,
                         "operating_datetime_utc": ts, "operating_time_hours": 1.0,
                         "gross_load_mw": p, "heat_content_mmbtu": h,
                         "co2_mass_tons": 0.0585 * h, "tech": "CC", "chp": False})
    units = pd.DataFrame({"plant_id_epa": [1], "emissions_unit_id_epa": ["A"], "plant_id_eia": [1],
                          "plant_name_eia": ["P"], "tech": ["CC"], "chp": [False],
                          "generator_ids": ["G1"]})
    gens = pd.DataFrame({"plant_id_eia": [1], "generator_id": ["G1"], "capacity_mw": [300.0],
                         "summer_capacity_mw": [280.0], "winter_capacity_mw": [300.0],
                         "minimum_load_mw": [120.0]})
    temp = pd.DataFrame({"plant_id_eia": 1, "ts_utc": list(ts) + [ts[-1] + pd.Timedelta(hours=1)],
                         "temp_c": np.r_[tday, tday[-1]]})
    uh = H.add_temperature(U.prepare_unit_hours(cems), units, temp)
    uh["steady"] = H.steady_rows(uh, H.TempConfig(ramp_frac=1.0))
    tab = H.fit_temperature_curves(uh, units, gens, H.TempConfig(min_temp_span=5.0))
    r = tab.iloc[0]
    assert r["limited_months"] == "5,6,7,8,9,10"
    assert r["hr_method"] in ("quadratic", "linear")
    # No temperature effect in the normal mode: d + e P stays near zero.
    assert abs(r["d"] + r["e"] * 260) < 0.5
    mask = H.in_limited_month(uh, tab)
    assert mask.sum() == reduced.sum()


def test_group_stats_and_stratified_bootstrap():
    rng = np.random.default_rng(5)
    x = rng.normal(0.4, 0.1, 200)
    s = H.group_stats(x, np.ones_like(x), n_boot=500)
    assert s["median_lo"] < 0.4 < s["median_hi"]
    assert s["wmean"] == pytest.approx(x.mean())
    # A constant per-unit ratio is reproduced exactly by every draw.
    den = rng.uniform(1, 10, 50)
    draws = H.stratified_ratio_bootstrap(0.3 * den, den, np.repeat(["a", "b"], 25), n_boot=100)
    assert draws == pytest.approx(0.3)
