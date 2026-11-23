"""Tests for the ERCOT environment on synthetic inputs."""
import numpy as np
import pandas as pd
import pytest

from ttrl.env import assumed as am
from ttrl.env import ercot as er
from ttrl.env import ercot_env as ee
from ttrl.env import fastkernel as fk
from ttrl.env import tariff as tf
from ttrl.env.responses import AssumedResponse, TrialResponse


def _arrays(members=3, seed=0):
    rng = np.random.default_rng(seed)
    ie = rng.normal(0, 1, size=(members, 2, 8))
    ie[:, 0] -= 8
    ie[:, 1] += 8
    return fk.KernelArrays(ie, np.zeros((members, 2, 3)), rng.normal(0, 0.5, size=(members, 2, 3)))


def _days(year=2024, months=(1, 2), hot=False, curtailed=()):
    hours = er.local_year_hours(year)
    loc = hours.tz_localize("UTC").tz_convert(er.TZ).tz_localize(None)
    keep = np.isin(loc.month, months) | ((loc.month == months[-1] + 1) & (loc.day <= 2)) | \
        ((loc.month == 12) & (loc.day == 31))
    hours = hours[keep]
    loc = loc[keep]
    demand = 50000 + 15000 * np.sin(2 * np.pi * (loc.hour - 6) / 24)
    curt = np.isin(loc.normalize(), pd.DatetimeIndex(list(curtailed)))
    y = er.ErcotYear(year, hours, demand.to_numpy(float), 0.35 * demand.to_numpy(float), 0.5 * demand.to_numpy(float),
                     None, 1.0, curt)
    sur = er.Surrogate(hours, np.full(len(hours), 8.0), np.full(len(hours), 0.45), np.full(len(hours), 8.5),
                       np.zeros(len(hours), int))
    temp = np.full(len(hours), 30.0 if hot else 10.0)
    return ee.ErcotDays.build(y, sur, temp)


def test_hour_key_and_grid():
    days = _days()
    assert (days.row >= 0).sum() == 24 * len(days.tdays) - (days.row < 0).sum()
    # every tariff day starts at 02:00 local: hour 0 of a day is 02:00
    first = days.year.hours[days.row[0, 0]]
    assert first.tz_localize("UTC").tz_convert(er.TZ).hour == 2
    spring = _days(months=(3, 4))
    d = list(spring.tdays).index(pd.Timestamp("2024-03-10"))
    assert (spring.row[d] < 0).sum() == 1                      # one missing hour on the spring night


def test_episode_costs_and_reward():
    days = _days()
    cfg = ee.ErcotConfig(months=["2024-01"], members=np.arange(3), gas_price=3.0, co2_price=50.0,
                         capacity_value=80000.0, peak_month_weight={1: 0.25, 8: 0.75})
    env = ee.ErcotTariffEnv(days, TrialResponse(_arrays()), cfg, seed=0)
    obs, _ = env.reset(options={"month": "2024-01", "member": 1})
    assert obs.shape == (env.obs_size,)
    rng = np.random.default_rng(0)
    pats = tf.valid_day_patterns()
    total, done = 0.0, False
    while not done:
        a = pats[rng.integers(len(pats))] if rng.random() < 0.3 else np.zeros(8, np.int8)
        obs, r, done, _, info = env.step(a)
        total += r
    o = pd.DataFrame(env.outcomes)
    assert len(o) == env.n
    # surrogate is linear: fuel = 8 MMBtu per MWh of load change
    dl = []
    for k, d in enumerate(env.day_idx):
        eh = ee.hourly_effect(env.effect[k])
        for rows in (days.row[d], days.row2[d]):
            ok = rows >= 0
            dl.append(days.year.residential[rows[ok]] * eh[ok] / 100)
    dl = np.concatenate(dl)
    assert info["metrics"]["d_fuel_mmbtu"] == pytest.approx(8 * dl.sum())
    cap = 80000.0 * 0.25 * (o["net_peak_mw"].max() - o["base_net_peak_mw"].max())
    assert total == pytest.approx(-(3 * 8 * dl.sum() + 50 * 0.45 * dl.sum() + cap))
    assert info["metrics"]["capacity_cost"] == pytest.approx(cap)
    assert info["metrics"]["fuel_cost"] == pytest.approx(3 * 8 * dl.sum())
    assert set(info["cost"]) == {"revenue", "high_hours", "low_hours"}


def test_flat_normal_revenue_and_assumed_response():
    days = _days()
    cat = np.array([0] * 9 + [1] * 10 + [2] * 5)
    resp = AssumedResponse(am.aalami_2010(cat))
    cfg = ee.ErcotConfig(months=["2024-01"], members=np.array([0]), gas_price=3.0, co2_price=0.0,
                         capacity_value=0.0, peak_month_weight={})
    env = ee.ErcotTariffEnv(days, resp, cfg, seed=0)
    env.reset(options={"month": "2024-01", "member": 0})
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(8, np.int8))
    # all-Normal: residential revenue ratio = Normal / flat on the responded load
    eff = resp.effect(np.zeros((1, 8), np.int8))[0].mean()
    assert info["metrics"]["revenue_ratio"] == pytest.approx(tf.PRICE[0] / tf.PRICE_FLAT * (1 + eff / 100), rel=1e-3)


def test_hot_multiplier_scales_flagged_days():
    days = _days(hot=True)
    arr = _arrays()
    out = {}
    for m in (1.0, 2.0):
        cfg = ee.ErcotConfig(months=["2024-01"], members=np.array([0]), gas_price=1.0, co2_price=0.0,
                             capacity_value=0.0, peak_month_weight={}, hot_multiplier=m)
        env = ee.ErcotTariffEnv(days, TrialResponse(arr), cfg, seed=0)
        env.reset(options={"month": "2024-01", "member": 0})
        day = np.zeros(8, np.int8)
        day[5] = tf.HIGH
        done = False
        while not done:
            _, _, done, _, info = env.step(day)
        out[m] = info["metrics"]["d_fuel_mmbtu"]
        assert info["metrics"]["out_of_support_days"] == env.n
    assert out[2.0] == pytest.approx(2 * out[1.0])


def test_curtailed_days_are_masked_in_training_only():
    days = _days(curtailed=[pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-11")])
    arr = _arrays()
    day = np.zeros(8, np.int8)
    day[5] = tf.HIGH
    res = {}
    for mask in (True, False):
        cfg = ee.ErcotConfig(months=["2024-01"], members=np.array([0]), gas_price=1.0, co2_price=0.0,
                             capacity_value=0.0, peak_month_weight={}, mask_curtailed=mask)
        env = ee.ErcotTariffEnv(days, TrialResponse(arr), cfg, seed=0)
        env.reset(options={"month": "2024-01", "member": 0})
        total, done, masked = 0.0, False, []
        while not done:
            _, r, done, _, info = env.step(day)
            total += r
            masked.append(info["masked"])
        res[mask] = (env, total, info["metrics"])
        assert sum(masked) == (3 if mask else 0)        # tariff days 9 (ends 02:00 on 10 Jan), 10 and 11
    env, total, met = res[True]
    m = env.masked
    assert (env.blocks[m] == tf.NORMAL).all() and (env.blocks[~m][:, 5] == tf.HIGH).all()
    o = pd.DataFrame(env.outcomes)
    assert met["masked_days"] == 3
    assert met["d_fuel_mmbtu"] == pytest.approx(o.loc[~o["masked"], "d_fuel_mmbtu"].sum())
    assert total == pytest.approx(-met["d_fuel_mmbtu"])
    assert res[False][2]["masked_days"] == 0


def test_surrogate_curvature_enters_day_outcome():
    days = _days()
    n = len(days.year.hours)
    days.surrogate.fuel_curv = np.full(n, 2e-4)
    days.surrogate.co2_curv = np.zeros(n)
    eff = np.full(tf.HH_PER_DAY, -10.0)
    out = ee.day_outcome(days, 3, eff)
    r = days.rows(3)
    dl = days.year.residential[r] * -0.10
    assert out["d_fuel_mmbtu"] == pytest.approx((8.0 * dl + 0.5 * 2e-4 * dl ** 2).sum())
    assert out["d_co2_t"] == pytest.approx((0.45 * dl).sum())


def test_gas_generation_restores_misreported_hours(monkeypatch):
    idx = pd.date_range("2020-01-08", periods=4, freq="h")
    df = pd.DataFrame({"datetime_utc": idx, "gas": [12000.0, 500.0, 510.0, 11000.0],
                       "storage_other": [20.0, 9000.0, 9300.0, 2500.0]})
    monkeypatch.setattr(er.eia930, "load_ba_hourly", lambda year: df)
    g, n = er.gas_generation(2020)
    assert n == 3 and g.tolist() == [12000.0, 9500.0, 9810.0, 13500.0]
    g, n = er.gas_generation(2024)                  # batteries: no correction from 2023 on
    assert n == 0 and g.tolist() == df["gas"].tolist()


def test_surrogate_step_share_interpolates_and_roundtrips(tmp_path, monkeypatch):
    days = _days()
    n = len(days.year.hours)
    sur = er.Surrogate(days.year.hours, np.full(n, 8.0), np.full(n, 0.45), np.full(n, 8.5), np.zeros(n, int),
                       np.full(n, 2e-4), np.full(n, 1e-5), np.array([[0.0, 0.0], [0.5, 0.2], [0.5, 0.2], [1.0, 0.4]]))
    rows = np.array([0, 1, 2])
    dl = np.array([-250.0, -2250.0, 4000.0])
    b2 = np.array([0.0, 0.75, 1.0])                          # interpolated between the class midpoints
    assert sur.fuel_change(rows, dl) == pytest.approx(8.0 * dl + 0.5 * b2 * 2e-4 * dl ** 2)
    assert sur.co2_change(rows, dl) == pytest.approx(0.45 * dl + 0.5 * np.array([0.0, 0.3, 0.4]) * 1e-5 * dl ** 2)
    monkeypatch.setattr(er, "inputs_path", lambda year: tmp_path / f"env_{year}.parquet")
    er.save_inputs(days.year, sur, np.full(n, 10.0))
    _, back, temp = er.load_inputs(days.year.year)
    assert np.array_equal(back.curv_beta, sur.curv_beta) and back.fuel_change(rows, dl) == pytest.approx(sur.fuel_change(rows, dl))


def test_capacity_charge_follows_the_monthly_net_load_peak():
    days = _days()
    n = len(days.year.hours)
    days.year.renewables = np.full(n, 5000.0)
    arr = _arrays()
    cfg = ee.ErcotConfig(months=["2024-01"], members=np.array([0]), gas_price={1: 2.0, 2: 9.0}, co2_price=0.0,
                         capacity_value=100000.0, peak_month_weight={1: 0.4})
    env = ee.ErcotTariffEnv(days, TrialResponse(arr), cfg, seed=0)
    env.reset(options={"month": "2024-01", "member": 0})
    rng = np.random.default_rng(3)
    pats = tf.valid_day_patterns()
    rewards, done = [], False
    while not done:
        _, r, done, _, info = env.step(pats[rng.integers(len(pats))])
        rewards.append(r)
    o = pd.DataFrame(env.outcomes)
    # net load = demand - 5,000 MW; the month's charge is value x weight x change of the month's net-load peak,
    # paid day by day as the running peak moves
    gap = o["net_peak_mw"].max() - o["base_net_peak_mw"].max()
    assert o["capacity_cost"].sum() == pytest.approx(100000.0 * 0.4 * gap)
    assert (o["base_net_peak_mw"] == o["base_peak_mw"] - 5000.0).all()
    assert info["metrics"]["fuel_cost"] == pytest.approx(2.0 * o["d_fuel_mmbtu"].sum())     # January price
    assert sum(rewards) == pytest.approx(-(o["fuel_cost"].sum() + o["capacity_cost"].sum()))
    obs, _ = env.reset(options={"month": "2024-01", "member": 0})
    assert obs.shape == (env.obs_size,) and np.isfinite(obs).all()


def test_config_from_settings():
    from ttrl.env.rewards import S2Settings
    st = S2Settings(2024, {m: 2.0 + m / 10 for m in range(1, 13)}, 220.0, 95000.0, {7: 0.3, 8: 0.7}, 0.03, 0.03)
    cfg = ee.ErcotConfig.from_settings(st, ["2024-08"], np.array([0]), hot_multiplier=1.5)
    assert cfg.gas_price_in(8) == pytest.approx(2.8) and cfg.co2_price == 220.0 and cfg.capacity_value == 95000.0
    assert cfg.peak_month_weight == {7: 0.3, 8: 0.7} and cfg.hot_multiplier == 1.5
