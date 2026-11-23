"""Tests for the London environment on synthetic inputs."""
import numpy as np
import pandas as pd
import pytest

from ttrl.analysis import lcl_did as did
from ttrl.data import lcl_features as lf
from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env import tariff as tf


def _arrays(members=4, seed=0):
    rng = np.random.default_rng(seed)
    ie = rng.normal(0, 1, size=(members, 2, 8))
    ie[:, 0] -= 8                      # High lowers load
    ie[:, 1] += 8                      # Low raises it
    return fk.KernelArrays(ie, rng.normal(0, 1, size=(members, 2, 3)) * [0, 1, 1], rng.normal(0, 1, size=(members, 2, 3)))


def _inputs(months=("2013-01", "2013-02"), households=20, seed=0):
    rng = np.random.default_rng(seed)
    tdays = pd.date_range(f"{months[0]}-01", pd.Period(months[-1]).end_time.normalize(), freq="D")
    days = ld.day_table(tdays, pd.Series(8.0, index=tdays))
    slot = np.arange(48)
    shape = 0.15 + 0.12 * np.exp(-((slot - 34) / 4.0) ** 2) + 0.05 * np.exp(-((slot - 12) / 3.0) ** 2)
    base = shape[None, :] * rng.uniform(0.9, 1.1, size=(len(tdays), 1))
    hh = base[None] * rng.gamma(5, 0.2, size=(households, 1, 1))
    return ld.LondonInputs(days, base, hh, np.arange(households))


def test_grid_handles_dst_nights(monkeypatch):
    local_times = pd.date_range("2013-01-01", "2013-12-31 23:30", freq="30min")
    prices = pd.DataFrame({"ts_local": local_times, "price_gbp_kwh": lf.PRICE_NORMAL, "band": "Normal"})
    monkeypatch.setattr(lf, "price_grid", lambda: prices)

    cal = did.calendar("2013-03-29 00:30", "2013-04-02 00:00")
    gi = ld.grid_index(cal)
    tdays = ld.full_tariff_days(cal)
    vals = np.arange(len(cal), dtype=float)
    g = ld.to_grid(vals, gi, tdays)
    assert g.shape == (len(tdays), 48) and not np.isnan(g).any()
    spring = list(tdays).index(pd.Timestamp("2013-03-30"))
    assert g[spring, 46] == g[spring, 45] and g[spring, 47] == g[spring, 45]   # missing hour filled
    cal2 = did.calendar("2013-10-25 00:30", "2013-10-29 00:00")
    gi2, td2 = ld.grid_index(cal2), ld.full_tariff_days(cal2)
    g2 = ld.to_grid(np.ones(len(cal2)), gi2, td2)
    assert np.allclose(g2, 1)                                                   # repeated hour averaged
    # cells outside the data stay NaN (e.g. a 2013-only series on a longer grid)
    short = cal.iloc[96:]
    g3 = ld.to_grid(np.ones(len(short)), ld.grid_index(short), tdays)
    assert np.isnan(g3[0]).all() and not np.isnan(g3[-1]).any()


def test_revenue_and_bills():
    inp = _inputs()
    n = len(inp.days)
    flat = np.zeros((n, 8), np.int8)
    zero = np.zeros((n, 48))
    # all-Normal days earn Normal / flat of the flat revenue
    assert ld.revenue_ratio(flat, inp.baseline, zero) == pytest.approx(tf.PRICE[0] / tf.PRICE_FLAT)
    bills = ld.bill_changes(flat, inp.households, zero)
    assert np.allclose(bills, tf.PRICE[0] / tf.PRICE_FLAT - 1)
    x = np.arange(100.0)
    assert ld.cvar(x, 0.95) == pytest.approx(np.mean(x[95:]))
    hh = inp.households.copy()
    hh[3] = 0                                              # an empty home
    b = ld.bill_changes(flat, hh, zero)
    assert np.isnan(b[3]) and np.isfinite(np.delete(b, 3)).all()
    assert np.isfinite(ld.cvar(b)) and np.isfinite(ld.percentile(b, 95))


def test_episode_settles_like_evaluate():
    inp = _inputs()
    arrays = _arrays()
    months = ["2013-01", "2013-02"]
    cfg = le.LondonConfig(months=months, members=np.arange(4))
    env = le.LondonTariffEnv(inp, arrays, cfg, seed=0)
    obs, info = env.reset(options={"month": "2013-01", "member": 2})
    assert obs.shape == (env.obs_size,)
    rng = np.random.default_rng(1)
    pats = tf.valid_day_patterns()
    total, done, invalid = 0.0, False, 0
    while not done:
        a = pats[rng.integers(len(pats))] if rng.random() < 0.5 else np.zeros(8, np.int8)
        obs, r, done, trunc, info = env.step(a)
        total += r
        invalid += info["invalid"]
    assert set(info["cost"]) == {"revenue", "bill_segment", "high_hours", "low_hours", "double_days"}
    # the episode's settled effect equals settling the whole month at once
    eff = le.settle(env.blocks, arrays, 2)
    assert np.allclose(env.effect, eff)
    peak_cut = 100 * (1 - ld.daily_peak_kw(ld.responded(inp.baseline[env.day_idx], eff)) / ld.daily_peak_kw(inp.baseline[env.day_idx]))
    assert total == pytest.approx(peak_cut.sum())
    ev = le.evaluate(env.blocks, inp, env.day_idx, arrays, np.array([2]))
    assert ev.loc[0, "revenue_ratio"] == pytest.approx(info["metrics"]["revenue_ratio"])
    assert ev.loc[0, "bill_cvar95"] == pytest.approx(info["metrics"]["bill_cvar95"])


def test_invalid_action_becomes_normal_day():
    inp = _inputs()
    env = le.LondonTariffEnv(inp, _arrays(), le.LondonConfig(months=["2013-01"], members=np.arange(4)), seed=0)
    env.reset(options={"month": "2013-01", "member": 0})
    _, _, _, _, info = env.step(np.array([1, 0, 1, 0, 0, 0, 0, 0]))      # two High events in a day
    assert info["invalid"] and (env.blocks[0] == 0).all()


def test_member_split_is_fixed():
    a, b = le.member_split(200)
    a2, b2 = le.member_split(200)
    assert len(a) == 150 and len(b) == 50 and not set(a) & set(b)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def test_day_ahead_forecast_uses_lags_known_a_day_ahead_and_pretrial_fit():
    rng = np.random.default_rng(2)
    tdays = pd.date_range("2012-07-01", "2013-03-31", freq="D")
    temp = pd.Series(12 + 6 * np.sin(np.arange(len(tdays)) / 58.0), index=tdays)
    days = ld.day_table(tdays, temp)
    n = len(tdays)
    base = np.zeros((n, 48))
    base[:7] = rng.uniform(0.2, 0.3, size=(7, 48))
    t1 = np.r_[temp.iloc[0], temp.to_numpy()[:-1]]
    for d in range(7, n):                      # an exact process in the forecast's own regressors
        base[d] = 0.02 + 0.5 * base[d - 7] + 0.3 * base[d - 2] + 0.004 * np.maximum(15.5 - t1[d], 0)
    fc = ld.day_ahead_forecast(base, days)
    trial = (days["period"] == "trial").to_numpy()
    assert np.isnan(fc[:7]).all()
    assert np.allclose(fc[trial], base[trial], atol=1e-8)
    # changing a trial day does not change the forecast of any earlier day or of the fit
    b2 = base.copy()
    j = int(np.flatnonzero(trial)[40])
    b2[j] += 1.0
    fc2 = ld.day_ahead_forecast(b2, days)
    assert np.allclose(fc2[:j + 2], fc[:j + 2], equal_nan=True)


def test_bill_constraint_is_on_households_that_do_not_respond():
    inp = _inputs()
    arrays = _arrays()
    cfg = le.LondonConfig(months=["2013-01"], members=np.arange(4), segment_bill_cap=0.0)
    env = le.LondonTariffEnv(inp, arrays, cfg, seed=0)
    env.reset(options={"month": "2013-01", "member": 1})
    day = np.zeros(8, np.int8)
    day[5] = tf.HIGH
    done = False
    while not done:
        _, _, done, _, info = env.step(day)
    m = info["metrics"]
    bills0 = ld.bill_changes(env.blocks, inp.households[:, env.day_idx], np.zeros_like(env.effect))
    assert m["bill_cvar95_nonresponding"] == pytest.approx(ld.cvar(bills0))
    seg = ld.exposure_segments(inp.households, inp.days)
    means = [np.nanmean(bills0[seg == s]) for s in range(4)]
    assert m["bill_segment_max_nonresponding"] == pytest.approx(max(means))
    assert info["cost"]["bill_segment"] == pytest.approx(max(0.0, max(means)))
    assert set(np.unique(seg)) <= {0, 1, 2, 3}
    assert m["bill_cvar95"] < m["bill_cvar95_nonresponding"]          # responding to High lowers bills
    from ttrl.env.rewards import S2Settings
    st = S2Settings(2024, {}, 0.0, 0.0, {}, 0.03, 0.03)
    c2 = le.LondonConfig.from_settings(st, ["2013-01"], np.arange(4))
    assert c2.segment_bill_cap == 0.03 and c2.revenue_tol == 0.03


def test_exposure_segments_rank_evening_share_on_pretrial_days():
    tdays = pd.date_range("2012-12-20", "2013-01-10", freq="D")
    days = ld.day_table(tdays, pd.Series(8.0, index=tdays))
    hh = np.ones((8, len(tdays), 48))
    share = np.array([0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8])
    pre = (days["period"] == "pre").to_numpy()
    for i, sh in enumerate(share):          # evening (12 of 48 slots) gets the household's share on pre-trial days
        hh[i, pre, 30:42] = sh * 36 / (1 - sh) / 12
        hh[i, ~pre, 30:42] = 5.0            # trial days must not matter
    seg = ld.exposure_segments(hh, days)
    assert seg.tolist() == [0, 2, 0, 2, 1, 3, 1, 3]
    bills = np.arange(8, dtype=float)
    assert ld.segment_bill_means(bills, seg).tolist() == [1.0, 5.0, 2.0, 6.0]
