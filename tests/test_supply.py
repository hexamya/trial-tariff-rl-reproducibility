"""Tests for the gas-fleet unit curves and the merit-order dispatch (S4)."""
import numpy as np
import pandas as pd
import pytest

from ttrl.supply import dispatch as D
from ttrl.supply import units as U
from ttrl.supply.fleet import build_fleet_hours, cems_year_hours


def curves(a, b, c, pmin, pmax) -> U.CurveArrays:
    """(1, N) curve arrays from per-unit lists."""
    return U.CurveArrays(*(np.asarray(x, float)[None, :] for x in (a, b, c, pmin, pmax)))


# --------------------------------------------------------------------------
# Curve fitting
# --------------------------------------------------------------------------
def test_fit_heat_curve_recovers_quadratic_despite_outliers():
    rng = np.random.default_rng(1)
    p = rng.uniform(100, 300, 2000)
    h = 500 + 6.0 * p + 0.004 * p ** 2 + rng.normal(0, 5, p.size)
    h[:100] += 800  # 5% gross outliers
    fit = U.fit_heat_curve(p, h, pmin=100, pmax=300)
    assert fit["method"] == "quadratic"
    assert fit["a"] == pytest.approx(500, rel=0.05)
    assert fit["b"] == pytest.approx(6.0, rel=0.05)
    assert fit["c"] == pytest.approx(0.004, rel=0.1)


def test_fit_heat_curve_concave_data_gives_linear_fit():
    rng = np.random.default_rng(2)
    p = rng.uniform(50, 150, 500)
    h = 300 + 12.0 * p - 0.02 * p ** 2 + rng.normal(0, 2, p.size)
    fit = U.fit_heat_curve(p, h, pmin=50, pmax=150)
    assert fit["method"] == "linear" and fit["c"] == 0.0 and fit["b"] > 0


def _synthetic_fleet(n_hours: int = 400):
    """CEMS-like rows for three CT units; unit 3 has only 10 operating hours."""
    rng = np.random.default_rng(3)
    t = cems_year_hours(2024)[:n_hours]
    rows = []
    for unit, cap, n_on in [("A", 100.0, n_hours), ("B", 200.0, n_hours), ("C", 150.0, 10)]:
        p = rng.uniform(0.5 * cap, cap, n_hours)
        h = 0.5 * cap + 9.0 * p + rng.normal(0, 3, n_hours)
        op = np.where(np.arange(n_hours) < n_on, 1.0, 0.0)
        rows.append(pd.DataFrame({
            "plant_id_epa": 1, "emissions_unit_id_epa": unit, "plant_id_eia": 1,
            "operating_datetime_utc": t, "operating_time_hours": op,
            "gross_load_mw": np.where(op > 0, p, np.nan),
            "heat_content_mmbtu": np.where(op > 0, h, np.nan),
            "co2_mass_tons": np.where(op > 0, 0.0594 * h, np.nan),
            "tech": "CT", "chp": False}))
    cems = pd.concat(rows, ignore_index=True)
    units = pd.DataFrame({"plant_id_epa": 1, "emissions_unit_id_epa": ["A", "B", "C"],
                          "plant_id_eia": 1, "plant_name_eia": "P", "tech": "CT",
                          "chp": False, "generator_ids": ["G1", "G2", "G3"]})
    gens = pd.DataFrame({"plant_id_eia": 1, "generator_id": ["G1", "G2", "G3"],
                         "capacity_mw": [100.0, 200.0, 150.0],
                         "summer_capacity_mw": [90.0, 180.0, 140.0],
                         "winter_capacity_mw": [100.0, 200.0, 150.0],
                         "minimum_load_mw": [40.0, 80.0, 60.0]})
    return cems, units, gens


def test_fit_unit_curves_own_fits_pooled_fallback_and_capacity():
    cems, units, gens = _synthetic_fleet()
    uh = U.prepare_unit_hours(cems)
    tab = U.fit_unit_curves(uh, units, gens, U.FitConfig(min_hours=50)).set_index("unit_id")
    assert list(tab.index) == ["1_A", "1_B", "1_C"]
    assert tab.loc["1_C", "fit_method"] == "pooled"
    assert tab.loc["1_A", "fit_method"] != "pooled"
    assert tab.loc["1_A", "b"] == pytest.approx(9.0, rel=0.05)
    # Pooled unit: Pmax scaled from nameplate, incremental heat rate ~ 9.
    assert tab.loc["1_C", "pmax"] == pytest.approx(150 * np.median([
        tab.loc["1_A", "pmax"] / 100, tab.loc["1_B", "pmax"] / 200]), rel=1e-6)
    assert tab.loc["1_C", "b"] == pytest.approx(9.0, rel=0.1)
    assert tab["summer_mw"].tolist() == [90.0, 180.0, 140.0]
    assert tab["co2_per_mmbtu"].to_numpy() == pytest.approx(0.0594)


def test_fit_drops_physically_impossible_heat_rates():
    cems, units, gens = _synthetic_fleet()
    # Unit A reports a heat rate of 4 MMBtu/MWh in its first 150 hours (data error).
    bad = (cems["emissions_unit_id_epa"] == "A") & (cems.index < 150)
    cems.loc[bad, "heat_content_mmbtu"] = 4.0 * cems.loc[bad, "gross_load_mw"]
    uh = U.prepare_unit_hours(cems)
    rows = U.fit_rows(uh)
    assert int((uh["clean"] & ~rows).sum()) == 150
    tab = U.fit_unit_curves(uh, units, gens, U.FitConfig(min_hours=50)).set_index("unit_id")
    assert tab.loc["1_A", "n_fit"] == 400 - 150
    assert tab.loc["1_A", "b"] == pytest.approx(9.0, rel=0.05)


def test_time_labels_use_local_standard_time():
    lab = U.time_labels(pd.to_datetime(["2024-01-01 05:00", "2024-01-01 06:00"]))
    assert lab["month"].tolist() == [12, 1]
    assert lab["hour_lst"].tolist() == [23, 0]
    assert lab["season"].tolist() == ["winter", "winter"]


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def test_economic_dispatch_equalises_incremental_heat_rate():
    cv = curves([0, 0], [5.0, 6.0], [0.01, 0.005], [50, 50], [300, 300])
    res = D.economic_dispatch(np.array([300.0]), cv, np.ones((1, 2), bool))
    assert res.p[0] == pytest.approx([400 / 3, 500 / 3], abs=1e-6)
    assert res.lam[0] == pytest.approx(5 + 0.02 * 400 / 3, abs=1e-6)
    assert res.status[0] == D.STATUS_OK


def test_economic_dispatch_linear_units_fill_in_merit_order():
    cv = curves([10, 10], [7.0, 9.0], [0, 0], [10, 10], [100, 100])
    res = D.economic_dispatch(np.array([150.0, 30.0]), cv, np.ones((2, 2), bool))
    assert res.p[0] == pytest.approx([100, 50])
    assert res.p[1] == pytest.approx([20, 10])
    assert res.p.sum(axis=1) == pytest.approx([150, 30])


def test_economic_dispatch_flags_infeasible_hours():
    cv = curves([0, 0], [5.0, 6.0], [0.01, 0.01], [50, 50], [100, 100])
    res = D.economic_dispatch(np.array([60.0, 250.0]), cv, np.ones((2, 2), bool))
    assert res.status.tolist() == [D.STATUS_BELOW_MIN, D.STATUS_SHORT]
    assert res.p[0] == pytest.approx([30, 30])
    assert res.p[1] == pytest.approx([100, 100])
    assert res.unserved_mw == pytest.approx([0, 50])


def test_priority_commitment_order_headroom_and_trim():
    # Full-load heat rates: unit0 7.0, unit1 8.0, unit2 10.0 (a = 0, linear).
    cv = curves([0, 0, 0], [7.0, 8.0, 10.0], [0, 0, 0], [40, 40, 40], [100, 100, 100])
    avail = np.ones((3, 3), bool)
    target = np.array([90.0, 150.0, 90.0])
    com = D.priority_commitment(target, cv, avail, headroom_mw=np.array([0.0, 0.0, 100.0]))
    assert com[0].tolist() == [True, False, False]
    assert com[1].tolist() == [True, True, False]
    # Headroom asks for 190 MW (two units, Pmin 80 <= 90 is feasible).
    assert com[2].tolist() == [True, True, False]
    # Unavailable cheapest unit is skipped.
    avail[0, 0] = False
    assert D.priority_commitment(target[:1], cv, avail[:1]).tolist() == [[False, True, False]]
    # Trim: headroom wants three units but their Pmin (120) exceeds the target (90).
    com = D.priority_commitment(np.array([90.0]), cv, np.ones((1, 3), bool), headroom_mw=200.0)
    assert com.tolist() == [[True, True, False]]


class _DeratedCurves:
    """Toy temperature model: Pmax falls 1% per degree C above 15 C."""

    def __init__(self, table: pd.DataFrame):
        self.base = U.StaticCurves(table)
        self.unit_ids = self.base.unit_ids

    def arrays(self, temp_c=None) -> U.CurveArrays:
        cv = self.base.arrays()
        if temp_c is None:
            return cv
        factor = 1.0 - 0.01 * np.clip(temp_c - 15.0, 0, None)
        return U.CurveArrays(cv.a, cv.b, cv.c, cv.pmin, cv.pmax * factor)


def test_dispatch_accepts_hour_varying_curves():
    table = pd.DataFrame({"unit_id": ["x", "y"], "a": [0.0, 0.0], "b": [7.0, 10.0],
                          "c": [0.0, 0.0], "pmin": [10.0, 10.0], "pmax": [100.0, 100.0],
                          "co2_per_mmbtu": [0.0594, 0.0594]})
    model: U.CurveModel = _DeratedCurves(table)
    temp = np.array([[15.0, 15.0], [35.0, 35.0]])
    cv = model.arrays(temp)
    res = D.dispatch(np.array([120.0, 120.0]), cv, np.ones((2, 2), bool))
    assert res.p[0] == pytest.approx([100, 20])
    assert res.p[1] == pytest.approx([80, 40])   # cheap unit derated to 80 MW
    heat = res.fleet_heat(cv)
    assert heat == pytest.approx([7 * 100 + 10 * 20, 7 * 80 + 10 * 40])


def test_finite_difference_mhr_equals_incremental_heat_rate():
    cv = curves([100.0], [6.0], [0.01], [50], [300])
    target = np.array([150.0, 200.0])
    mhr = D.finite_difference_mhr(target, cv, np.ones((2, 1), bool), delta_mw=10.0)
    assert mhr == pytest.approx(6.0 + 0.02 * target)


def test_fit_headroom_recovers_line():
    d = np.linspace(5000, 40000, 50)
    r0, r1 = D.fit_headroom(d, d + 3000 + 0.1 * d)
    assert (r0, r1) == pytest.approx((3000, 0.1))


# --------------------------------------------------------------------------
# Fleet arrays
# --------------------------------------------------------------------------
def test_build_fleet_hours_and_availability():
    cems, _, _ = _synthetic_fleet(n_hours=48)
    fleet = build_fleet_hours(cems, ["1_A", "1_C"], 2024)
    assert fleet.energy.shape == (8784, 2)
    assert fleet.hours[0] == pd.Timestamp("2024-01-01 06:00")
    assert fleet.on[:48, 0].all() and fleet.on[:10, 1].all() and not fleet.on[10:, 1].any()
    day = fleet.availability("day")
    # Unit C ran on LST day 1 (hours 0-9) only: available that day while CEMS has rows.
    assert day[:24, 1].all() and not day[24:48, 1].any()
    month = fleet.availability("month")
    assert month[:48, 1].all() and not month[48:, 1].any()   # no CEMS rows after hour 48
