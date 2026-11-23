"""Tests for the S4 integration: temperature-aware fleet model, error band and
the impact function the tariff environment calls."""
import numpy as np
import pandas as pd
import pytest

from ttrl.supply import errorband as EB
from ttrl.supply import fleetmodel as FM
from ttrl.supply.fleet import cems_year_hours
from ttrl.supply.impact import (TONNE_PER_SHORT_TON, S4Model, compare, local_hours,
                                shift_profile)
from ttrl.supply.units import time_labels


# --------------------------------------------------------------------------
# Toy fleet: A = cheap CC, B = simple-cycle CT, C = suspect CC (CT output only)
# --------------------------------------------------------------------------
def toy_table(kappa_a: float = 0.0, d_a: float = 0.0) -> pd.DataFrame:
    t = pd.DataFrame({
        "unit_id": ["1_A", "1_B", "1_C"], "tech": ["CC", "CT", "CC"], "group": ["CC", "CT", "CC*"],
        "suspect_cc": [False, False, True], "const_heat": False,
        "a": [100.0, 50.0, 80.0], "b": [7.0, 10.0, 9.0], "c": [0.001, 0.0, 0.0],
        "d": [d_a, 0.0, 0.0], "e": 0.0, "pmin": [50.0, 20.0, 30.0], "pmax": [300.0, 100.0, 150.0],
        "p15": [300.0, 100.0, 150.0], "kappa": [kappa_a, 0.0, 0.0],
        "a0": [100.0, 50.0, 80.0], "b0": [7.0, 10.0, 9.0], "c0": [0.001, 0.0, 0.0],
        "t_fit_lo": np.nan, "t_fit_hi": np.nan, "hr_method": "quadratic", "kappa_method": "own",
        "nameplate_mw": [320.0, 110.0, 160.0], "source": "temperature fit",
        "co2_per_mmbtu": [0.0585, 0.0585, 0.0585]})
    return t


def toy_data(temp_c: float = 20.0) -> FM.FleetData:
    """A year in which only A produces (300 MW); every unit counts as available."""
    hours = cems_year_hours(2024)
    t_n = len(hours)
    energy = np.tile([300.0, 0.0, 0.0], (t_n, 1))
    heat = np.tile([2290.0, 0.0, 0.0], (t_n, 1))
    lab = time_labels(hours)
    return FM.FleetData(year=2024, hours=hours, unit_ids=["1_A", "1_B", "1_C"],
                        plant_id_eia=np.array([1, 2, 3]), tech=np.array(["CC", "CT", "CC"]),
                        energy=energy, heat=heat, co2=0.0585 * heat, on=np.ones((t_n, 3), bool),
                        avail=np.ones((t_n, 3), bool), temp=np.full((t_n, 3), temp_c),
                        month=lab["month"].to_numpy(), season=lab["season"].to_numpy(),
                        hour_lst=lab["hour_lst"].to_numpy(), day=FM.lst_day(hours))


def toy_band(err_fuel: float = 0.0, beta: float = 1.0, m_bar: float = 8.0, n_draws: int = 50,
             band_id: str = "toy") -> EB.ErrorBand:
    load_edges = (250.0, 500.0)
    k = len(EB.stratum_labels(3))
    err = np.zeros((n_draws, k, 3))
    err[:, :, 0] = err_fuel
    mb = np.zeros((k, 3))
    mb[:, 0] = m_bar
    return EB.ErrorBand(load_edges, EB.HOT_EDGE, err, err[0].copy(), mb,
                        np.full((n_draws, 3, 3), beta), np.full((3, 3), beta),
                        np.full(k, 500), np.full(k, 20), [], {"id": band_id})


def toy_model(config: str = "C", mode: str = "temperature", **band_kw) -> S4Model:
    fm = FM.FleetModel(toy_table(), mode=mode, config=config)
    return S4Model(toy_data(), fm, toy_band(**band_kw), meta={"config": config, "mode": mode})


# --------------------------------------------------------------------------
# Fleet model
# --------------------------------------------------------------------------
def test_static_table_drops_temperature_terms():
    t = toy_table(kappa_a=0.5, d_a=-2.0).assign(a0=[90.0, 50.0, 80.0], pmax=[310.0, 100.0, 150.0])
    s = FM.static_table(t)
    assert (s["d"] == 0).all() and (s["e"] == 0).all() and (s["kappa"] == 0).all()
    assert s["a"].tolist() == [90.0, 50.0, 80.0]
    assert s["p15"].tolist() == [310.0, 100.0, 150.0]


def test_fleet_model_modes_and_configs():
    t = toy_table(kappa_a=1.0, d_a=2.0)
    temp = np.full((1, 3), 35.0)
    full = FM.FleetModel(t, "temperature", "C").arrays(temp)
    assert full.pmax[0, 0] == pytest.approx(300 * (1 - 0.01 * 20))     # derated
    assert full.a[0, 0] == pytest.approx(100 + 2.0 * 20)                # heat shift
    assert FM.FleetModel(t, "heat", "C").arrays(temp).pmax[0, 0] == pytest.approx(300)
    assert FM.FleetModel(t, "derate", "C").arrays(temp).a[0, 0] == pytest.approx(100)
    assert FM.FleetModel(t, "static", "C").arrays(temp).pmax[0, 0] == pytest.approx(300)
    sx = FM.FleetModel(t, "temperature", "C-sx")
    assert sx.disp.tolist() == [True, True, False]
    assert sx.split(np.array([[300.0, 0.0, 100.0]]))["exo_energy"].tolist() == [100.0]
    with pytest.raises(ValueError):
        FM.FleetModel(t, "hot", "C")


def test_fleet_model_run_marginal_and_ct_count():
    m = FM.FleetModel(toy_table(), "static", "C")
    temp, avail = np.full((3, 3), 20.0), np.ones((3, 3), bool)
    r = m.run(np.array([300.0, 400.0, 600.0]), temp, avail)
    # 300 MW: A alone at Pmax; 400 MW: A + C (C at 100 MW); 600 MW: all three.
    assert r.heat[0] == pytest.approx(100 + 7 * 300 + 0.001 * 300 ** 2)
    assert r.heat[1] == pytest.approx(2290 + 80 + 9 * 100)
    assert r.n_ct.tolist() == [0, 0, 1]
    # Inside one committed set the derivative is the incremental heat rate of the marginal unit.
    mg = m.marginal(np.array([380.0]), temp[:1], avail[:1], delta=10.0)
    assert mg["fuel"][0] == pytest.approx(9.0)
    assert mg["co2"][0] == pytest.approx(9.0 * 0.0585)


def test_complete_curve_table_uses_static_fallback():
    tt = toy_table().iloc[:2].assign(plant_id_epa=1, emissions_unit_id_epa=["A", "B"])
    tt.loc[1, "a"] = np.nan                          # unit B has no usable temperature fit
    st = pd.DataFrame({"unit_id": ["1_A", "1_B", "1_C"], "tech": ["CC", "CT", "CC"],
                       "suspect_cc": [False, False, True], "a": [1.0, 2.0, 3.0], "b": [7.0, 8.0, 9.0],
                       "c": 0.0, "pmin": 10.0, "pmax": [300.0, 100.0, 150.0],
                       "nameplate_mw": [320.0, 110.0, 160.0], "co2_per_mmbtu": [0.05, 0.06, 0.07]})
    out = FM.complete_curve_table(tt, st, ["1_A", "1_B", "1_C"]).set_index("unit_id")
    assert out.loc["1_A", "source"] == "temperature fit"
    assert out.loc["1_B", "source"] == out.loc["1_C", "source"] == "static fallback"
    assert out.loc["1_C", "group"] == "CC*" and bool(out.loc["1_C", "suspect_cc"])
    assert out.loc["1_B", "a"] == 2.0 and out.loc["1_B", "p15"] == 100.0
    assert out["co2_per_mmbtu"].tolist() == [0.05, 0.06, 0.07]


def test_plant_temperature_matrix_is_hour_beginning_mean():
    ts = pd.date_range("2024-01-01 06:00", periods=4, freq="h")
    temp = pd.DataFrame({"plant_id_eia": [7] * 4, "ts_utc": ts, "temp_c": [10.0, 12.0, 14.0, 20.0]})
    m = FM.plant_temperature_matrix(temp, ts[:3], np.array([7, 7]))
    assert m[:, 0].tolist() == [11.0, 13.0, 17.0]
    assert m.shape == (3, 2)


# --------------------------------------------------------------------------
# Error band
# --------------------------------------------------------------------------
def _pairs(n=4000, seed=0, beta=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 500, n)
    m = rng.normal(8.0, 1.0, n)                       # model marginal per hour
    stratum = rng.integers(0, 3, n)
    a = np.array([7.0, 8.0, 9.0])[stratum]
    mbar = np.array([m[stratum == s].mean() for s in range(3)])
    y = (a + beta * (m - mbar[stratum])) * x + rng.normal(0, 50, n)
    p = EB.Pairs(x=x, dy={"fuel": y, "co2": 0.0585 * y, "ct": np.zeros(n)},
                 m={"fuel": m, "co2": 0.0585 * m, "ct": np.zeros(n)},
                 block=np.arange(n) // 40, t=np.arange(n))
    return p, stratum, mbar


def test_block_stats_slopes_match_ols():
    p, st, _ = _pairs()
    _, stats = EB.block_stats(p, st, 3)
    est = EB.slopes_from_stats(stats.sum(axis=0))
    for s in range(3):
        k = st == s
        assert est["a_fuel"][s] == pytest.approx(np.polyfit(p.x[k], p.dy["fuel"][k], 1)[0])
        assert est["m_fuel"][s] == pytest.approx(p.m["fuel"][k].mean())


def test_bootstrap_weights_resample_whole_blocks():
    w = EB.bootstrap_weights(12, 100, seed=3)
    assert w.shape == (100, 12) and (w.sum(axis=1) == 12).all()
    assert np.array_equal(w, EB.bootstrap_weights(12, 100, seed=3))


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_within_beta_recovers_confirmed_share(beta):
    p, st, mbar = _pairs(beta=beta, n=20000, seed=1)
    m_bar = np.column_stack([mbar, 0.0585 * mbar, np.zeros(3)])
    ws = EB.within_stats(p, st, 3, m_bar)
    b = EB.within_beta(ws.sum(axis=0), np.zeros(3, int), 1)
    assert b[0, 0] == pytest.approx(beta, abs=0.03)
    assert b[0, 2] == 1.0                              # no model variation -> beta 1 by definition


def _step_pairs(beta1: float, beta2: tuple[float, ...], n: int = 40000, seed: int = 0):
    """Pairs whose actual change confirms share beta1 of the model's hour-level variation and
    share beta2[c] of its step-size part in size class c."""
    rng = np.random.default_rng(seed)
    stratum = rng.integers(0, 3, n)
    x = rng.normal(0, 1200, n)
    mbar = np.array([7.0, 8.0, 9.0])
    m = mbar[stratum] + rng.normal(0, 0.8, n)
    z2 = rng.normal(0, 400, n) + 1e-4 * x ** 2           # curvature plus commitment jumps
    b2 = np.asarray(beta2)[EB.size_class(x)]
    y = mbar[stratum] * x - 0.3 * x + beta1 * (m - mbar[stratum]) * x + b2 * z2 + rng.normal(0, 150, n)
    step = m * x + z2
    p = EB.Pairs(x=x, dy={"fuel": y, "co2": 0.0585 * y, "ct": np.zeros(n)},
                 m={"fuel": m, "co2": 0.0585 * m, "ct": np.zeros(n)},
                 block=np.arange(n) // 40, t=np.arange(n),
                 step={"fuel": step, "co2": 0.0585 * step, "ct": np.zeros(n)})
    return p, stratum, mbar


def test_joint_beta_recovers_hour_level_and_step_size_shares():
    beta2 = (0.0, 0.3, 0.6, 0.9)
    p, st, mbar = _step_pairs(0.7, beta2, seed=4)
    m_bar = np.column_stack([mbar, 0.0585 * mbar, np.zeros(3)])
    js = EB.joint_stats(p, st, 3, m_bar)
    b1, b2 = EB.joint_beta(js.sum(axis=0), np.zeros(3, int), 1)
    assert b1.shape == (1, 3) and b2.shape == (4, 3)
    assert b1[0, 0] == pytest.approx(0.7, abs=0.03)
    assert b2[:, 0] == pytest.approx(beta2, abs=0.05)
    assert b2[:, 1] == pytest.approx(beta2, abs=0.05)
    assert b1[0, 2] == 1.0 and (b2[:, 2] == 1.0).all()       # CT: no model variation
    # without the step-size part the joint estimate equals within_beta
    p0 = EB.Pairs(p.x, p.dy, p.m, p.block, p.t)
    wb = EB.within_beta(EB.within_stats(p0, st, 3, m_bar).sum(axis=0), np.zeros(3, int), 1)
    p_nostep = EB.Pairs(p.x, p.dy, p.m, p.block, p.t, step={k: p.m[k] * p.x for k in p.m})
    jb, _ = EB.joint_beta(EB.joint_stats(p_nostep, st, 3, m_bar).sum(axis=0), np.zeros(3, int), 1)
    assert jb[0, :2] == pytest.approx(wb[0, :2])


def test_correct_splits_hour_level_and_step_size_parts():
    band = toy_band(err_fuel=0.5, beta=1.0, m_bar=8.0)
    band.beta2 = np.zeros((band.n_draws, 4, 3))
    band.beta2_point = np.zeros((4, 3))
    d_mwh = np.array([300.0, -1500.0])
    m_hour = np.array([[8.4, 0.0, 0.0], [7.6, 0.0, 0.0]])
    d_model = np.column_stack([d_mwh * m_hour[:, 0] + np.array([200.0, -350.0]), np.zeros(2), np.zeros(2)])
    st = np.array([2, 0])
    out = band.correct(d_model, d_mwh, st, m_hour=m_hour)
    # beta = 1, beta2 = 0: per hour dE x a_s + dE (m_t - m_bar), the step-size part removed
    a = 8.0 - 0.5
    expect = (300 * a + 300 * 0.4) + (-1500 * a + -1500 * -0.4)
    assert out[:, 0] == pytest.approx(expect)
    band.beta2[:, 1, 0] = 1.0                               # 0.5-1 GW class: no hour in it
    assert band.correct(d_model, d_mwh, st, m_hour=m_hour)[:, 0] == pytest.approx(expect)
    band.beta2[:, 3, 0] = 1.0                               # >= 2 GW class: no hour in it either
    band.beta2[:, 2, 0] = 1.0                               # 1-2 GW: the second hour keeps its -350
    assert band.correct(d_model, d_mwh, st, m_hour=m_hour)[:, 0] == pytest.approx(expect - 350)
    # NaN marginal: the hour falls back to the stratum mean (all deviation is step-size part)
    m_nan = m_hour.copy()
    m_nan[0, 0] = np.nan
    band.beta2[:] = 0.0
    out = band.correct(d_model, d_mwh, st, m_hour=m_nan)
    assert out[:, 0] == pytest.approx(300 * a + (-1500 * a + -1500 * -0.4))
    # without m_hour the old formula applies
    assert band.correct(d_model, d_mwh, st)[:, 0] == pytest.approx(d_model[:, 0].sum() - 0.5 * (300 - 1500))


def test_strata_labels_and_parents():
    labels = EB.stratum_labels(3)
    s = EB.strata(np.array([35.0, 20.0]), np.array([600.0, 100.0]), np.array([7, 1]), (250.0, 500.0))
    assert [labels[i] for i in s] == ["summer T3 hot", "winter T1"]
    par = EB.parent_of(3)
    assert par[s[0]] == par[labels.index("summer T3")] == 2 * 3 + 2


def test_correct_bias_and_noise_terms():
    band = toy_band(err_fuel=0.5, beta=1.0, m_bar=8.0)
    d_model = np.array([[900.0, 52.65, 0.0], [-850.0, -49.7, 0.0]])
    d_mwh = np.array([100.0, -100.0])
    st = np.array([2, 0])
    out = band.correct(d_model, d_mwh, st)
    # beta = 1: only the stratum bias is removed: 50 - (100 - 100) * 0.5
    assert out[:, 0] == pytest.approx(50.0)
    band0 = toy_band(err_fuel=0.5, beta=0.0, m_bar=8.0)
    # beta = 0: each hour becomes dE x (m_bar - err) = dE x CEMS marginal (7.5)
    assert band0.correct(d_model, d_mwh, st)[:, 0] == pytest.approx(7.5 * 100 - 7.5 * 100)
    assert band0.correct(d_model[:1], d_mwh[:1], st[:1])[:, 0] == pytest.approx(750.0)


def test_from_pairs_pools_thin_cells_and_roundtrips(tmp_path):
    rng = np.random.default_rng(2)
    n = 3000
    t_idx = np.arange(1, n + 1)
    ft = np.where(t_idx % 50 == 0, 35.0, 20.0)        # few hot hours -> thin hot cells
    load = rng.uniform(0, 900, n + 1)
    month = np.full(n + 1, 7)
    x = rng.normal(0, 300, n)
    y = 8.0 * x + rng.normal(0, 20, n)
    p = EB.Pairs(x=x, dy={"fuel": y, "co2": 0.06 * y, "ct": 0.001 * x},
                 m={"fuel": np.full(n, 8.4), "co2": np.full(n, 0.06 * 8.4), "ct": np.full(n, 0.001)},
                 block=t_idx // 100, t=t_idx)
    ft = np.r_[20.0, ft]
    band = EB.ErrorBand.from_pairs(p, ft, load, month, (300.0, 600.0), n_draws=200, min_pairs=100)
    labels = band.labels
    assert labels.index("summer T3 hot") in band.pooled
    assert band.meta["pooled_to"][str(labels.index("summer T3 hot"))] == "summer T3"
    own = labels.index("summer T2")
    assert own not in band.pooled
    assert band.point[own, 0] == pytest.approx(0.4, abs=0.05)
    path = tmp_path / "band.parquet"
    band.save(path)
    back = EB.ErrorBand.load(path)
    assert np.array_equal(back.err, band.err) and np.array_equal(back.beta, band.beta)
    assert back.labels == labels and back.pooled == band.pooled
    assert back.beta2 is None


def test_from_pairs_with_step_part_roundtrips(tmp_path):
    p, st, _ = _step_pairs(0.8, (0.1, 0.4, 0.7, 0.9), n=12000, seed=5)
    n = len(p.x)
    ft = np.full(n + 1, 20.0)
    load = np.r_[0.0, np.array([100.0, 400.0, 800.0])[st]]
    p.t = np.arange(1, n + 1)
    band = EB.ErrorBand.from_pairs(p, ft, load, np.full(n + 1, 7), (300.0, 600.0), n_draws=100)
    assert band.beta2.shape == (100, 4, 3) and band.beta2_point.shape == (4, 3)
    assert band.beta2_point[:, 0] == pytest.approx([0.1, 0.4, 0.7, 0.9], abs=0.1)
    assert sum(band.meta["pairs_by_size"]) == n
    path = tmp_path / "band2.parquet"
    band.save(path)
    back = EB.ErrorBand.load(path)
    assert np.array_equal(back.beta2, band.beta2) and back.size_edges == band.size_edges


# --------------------------------------------------------------------------
# Impact function
# --------------------------------------------------------------------------
def test_impact_zero_change_is_zero():
    m = toy_model(err_fuel=0.3)
    hrs = m.data.hours[100:104]
    res = m.impact(np.zeros(4), hrs)
    assert np.allclose(res.totals["estimate"].astype(float), 0.0)
    assert np.allclose(res.draws.to_numpy(), 0.0)


def test_impact_values_units_and_ct_hours():
    m = toy_model()
    hrs = m.data.hours[100:102]
    res = m.impact(np.array([100.0, 200.0]), hrs)
    # Base 300 MW = A at Pmax. +100 MW brings C on (no-load 80 + 9 x 100);
    # +200 MW needs C at Pmax (80 + 9 x 150) and the CT B at 50 MW (50 + 10 x 50).
    h = res.hourly
    assert h["d_fuel_mmbtu"].tolist() == pytest.approx([80 + 900, 80 + 1350 + 50 + 500])
    assert h["d_ct_units"].tolist() == [0.0, 1.0]
    assert h["status_new"].tolist() == ["ok", "ok"]
    assert res["ct_unit_hours"] == 1.0
    assert res["co2_tonnes"] == pytest.approx(res["co2_short_tons"] * TONNE_PER_SHORT_TON)
    assert res["co2_short_tons"] == pytest.approx(0.0585 * res["fuel_mmbtu"])
    # Zero error and beta = 1: the band collapses on the model estimate.
    assert res.interval("fuel_mmbtu") == pytest.approx((res["fuel_mmbtu"],) * 2)
    # Fleet heat rate: (F + dF) / (E + dE) - F / E.
    f0, e0 = 2 * 2290.0, 2 * 300.0
    assert res["fleet_heat_rate_mmbtu_per_mwh"] == pytest.approx(
        (f0 + res["fuel_mmbtu"]) / (e0 + 300.0) - f0 / e0)


def test_impact_band_applies_validated_error_and_gas_share():
    m = toy_model(err_fuel=0.5)
    hrs = m.data.hours[100:101]
    res = m.impact(np.array([100.0]), hrs)
    assert res.totals.at["fuel_mmbtu", "corrected"] == pytest.approx(res["fuel_mmbtu"] - 0.5 * 100)
    half = m.impact(np.array([200.0]), hrs, gas_share=0.5)
    assert half["fuel_mmbtu"] == pytest.approx(res["fuel_mmbtu"])
    assert half.hourly["d_gas_mwh"].iloc[0] == 100.0


def test_impact_with_step_size_part():
    m = toy_model()
    hrs = m.data.hours[100:101]
    m.band.beta2 = np.ones((m.band.n_draws, 4, 3))
    m.band.beta2_point = np.ones((4, 3))
    res = m.impact(np.array([100.0]), hrs)
    # +/- 500 MW is short of capacity in the toy fleet, so the hour falls back to the stratum mean;
    # beta = beta2 = 1 and no error: the band collapses on the model estimate (C on: 80 + 9 x 100)
    assert np.isnan(res.hourly["model_marginal_fuel"].iloc[0])
    assert res.totals.at["fuel_mmbtu", "corrected"] == pytest.approx(980.0)
    m.band.beta2[:] = 0.0
    # beta2 = 0: the whole step-size part (980 - 100 x 8) is removed
    assert m.impact(np.array([100.0]), hrs).totals.at["fuel_mmbtu", "corrected"] == pytest.approx(800.0)


def test_impact_config_c_sx_holds_suspect_units():
    m = toy_model(config="C-sx")
    hrs = m.data.hours[100:101]
    res = m.impact(np.array([50.0]), hrs)
    # Only A and B are dispatched; A is at Pmax with 300 MW, so B (CT) takes the 50 MW.
    assert res["fuel_mmbtu"] == pytest.approx(50 + 10 * 50)
    assert res["ct_unit_hours"] == 1.0


def test_compare_is_draw_aligned_and_checks_band():
    m = toy_model(err_fuel=0.2)
    hrs = m.data.hours[200:202]
    a = m.impact(np.array([100.0, 0.0]), hrs)
    b = m.impact(np.array([0.0, 100.0]), hrs)
    d = compare(a, a)
    assert np.allclose(d[["estimate", "lo", "hi"]].astype(float), 0.0)
    assert compare(a, b).at["fuel_mmbtu", "estimate"] == pytest.approx(a["fuel_mmbtu"] - b["fuel_mmbtu"])
    other = toy_model(err_fuel=0.2, band_id="other").impact(np.array([100.0, 0.0]), hrs)
    with pytest.raises(ValueError):
        compare(a, other)


def test_impact_temperatures_and_hours_outside_model_year():
    fm = FM.FleetModel(toy_table(kappa_a=1.0), "temperature", "C")
    m = S4Model(toy_data(), fm, toy_band())
    hrs = m.data.hours[300:301]
    hot = m.impact(np.array([0.0]), hrs, temp_c=np.array([35.0]))
    assert hot["fuel_mmbtu"] == 0.0
    # At 35 C A is derated to 240 MW, so the 300 MW base needs C for 60 MW.
    assert hot.meta["base_fuel_mmbtu"] == pytest.approx(100 + 7 * 240 + 0.001 * 240 ** 2 + 80 + 9 * 60)
    # At the toy year's 20 C A is derated 5% to 285 MW: C runs at its 30 MW minimum, A at 270 MW.
    mild = m.impact(np.array([0.0]), hrs)
    assert mild.meta["base_fuel_mmbtu"] == pytest.approx(100 + 7 * 270 + 0.001 * 270 ** 2 + 80 + 9 * 30)
    assert m.plant_temperatures(hrs, np.array([35.0])) == pytest.approx(np.full((1, 3), 35.0))
    future = pd.date_range("2030-07-01 18:00", periods=2, freq="h")
    with pytest.raises(ValueError):
        m.impact(np.array([100.0, 100.0]), future)
    res = m.impact(np.array([100.0, 100.0]), future, temp_c=np.array([20.0, 20.0]),
                   base_mw=np.array([300.0, 300.0]))
    assert res.meta["hours_outside_model_year"] == 2
    base = 100 + 7 * 270 + 0.001 * 270 ** 2 + 80 + 9 * 30          # A 270 MW, C 30 MW
    new = 100 + 7 * 285 + 0.001 * 285 ** 2 + 80 + 9 * 115          # A at its 285 MW limit, C 115 MW
    assert res["fuel_mmbtu"] == pytest.approx(2 * (new - base))


def test_local_hours_and_shift_profile_handle_dst_and_midnight():
    summer = local_hours(["2024-08-20"], 16, 19)
    assert list(summer) == list(pd.to_datetime(["2024-08-20 21:00", "2024-08-20 22:00",
                                                "2024-08-20 23:00"]))
    winter = local_hours(["2024-01-10"], 22, 1)
    assert list(winter) == list(pd.to_datetime(["2024-01-11 04:00", "2024-01-11 05:00",
                                                "2024-01-11 06:00"]))
    prof = shift_profile(["2024-08-20", "2024-08-21"], 500.0)
    assert len(prof) == 12 and prof.sum() == 0.0
    assert (prof < 0).sum() == 6 and prof.max() == 500.0
