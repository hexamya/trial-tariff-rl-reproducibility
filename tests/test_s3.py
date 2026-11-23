"""Tests for the S3 policies: patterns, MIP planner, baselines and the evaluation harness."""
import numpy as np
import pandas as pd
import pytest

from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env import tariff as tf
from ttrl.s3 import baselines as bl
from ttrl.s3 import harness as hs
from ttrl.s3 import mip
from ttrl.s3 import patterns as pt
from ttrl.s3 import planners as pl


def _arrays(members=4, seed=0):
    rng = np.random.default_rng(seed)
    ie = rng.normal(0, 1, size=(members, 2, 8))
    ie[:, 0] -= 8
    ie[:, 1] += 8
    return fk.KernelArrays(ie, rng.normal(0, 1, size=(members, 2, 3)) * [0, 1, 1], rng.normal(0, 1, size=(members, 2, 3)))


def _inputs(households=12, seed=0):
    rng = np.random.default_rng(seed)
    tdays = pd.date_range("2012-12-01", "2013-02-28", freq="D")
    days = ld.day_table(tdays, pd.Series(8.0, index=tdays))
    slot = np.arange(48)
    shape = 0.15 + 0.12 * np.exp(-((slot - 34) / 4.0) ** 2) + 0.05 * np.exp(-((slot - 12) / 3.0) ** 2)
    base = shape[None, :] * rng.uniform(0.9, 1.1, size=(len(tdays), 1))
    tilt = rng.uniform(0.7, 1.3, size=(households, 1, 48)) ** (slot >= 30)
    hh = base[None] * rng.gamma(5, 0.2, size=(households, 1, 1)) * tilt
    return ld.LondonInputs(days, base, hh, np.arange(households))


def test_patterns_states_and_compatibility():
    P = pt.candidate_patterns()
    assert (P[0] == 0).all() and len(P) == len(np.unique(P, axis=0))
    assert pt.trailing_state(np.array([0, 0, 0, 0, 0, 0, 1, 1])) == 2
    assert pt.trailing_state(np.array([0, 0, 0, 0, 0, -1, -1, -1])) == 8 + 3
    assert pt.trailing_state(np.array([1, 0, 0, 0, 0, 0, 0, 0])) == 0
    C = pt.compatibility(P)
    rules = tf.TrialRules()
    rng = np.random.default_rng(0)
    for _ in range(200):
        prev, q = P[rng.integers(len(P))], rng.integers(len(P))
        assert C[pt.trailing_state(prev), q] == rules.day_ok(prev[None], P[q])


def test_pattern_effects_match_fast_kernel():
    arr = _arrays()
    P = pt.candidate_patterns()[:20]
    E = pt.pattern_effects(P, arr)
    assert np.allclose(E[:, 0], 0.0)
    j = 7
    bands = np.r_[np.zeros(48, np.int8), tf.slot_bands(P[j]), np.zeros(48, np.int8)]
    assert np.allclose(E[:, j], fk.effect(bands, arr)[:, 48:96])


def test_mip_respects_constraints_and_rules():
    P = pt.candidate_patterns()
    rng = np.random.default_rng(1)
    D = 6
    value = rng.normal(0, 1, (D, len(P)))
    value[:, 0] = 0.0
    hh = (P == tf.HIGH).sum(1)
    rev = np.tile(1.0 + 0.4 * hh - 0.1 * (P == tf.LOW).sum(1), (D, 1)) / D
    m = mip.MonthModel(value, rev, 1.0, 0.05, P, pt.compatibility(P), week=np.zeros(D, int),
                       rules=tf.TrialRules(max_high_hh_per_month=24))
    choice, info = mip.solve(m, time_limit=30)
    assert info["status"] == 0
    plan = P[choice]
    r = float(rev[np.arange(D), choice].sum())
    assert 0.95 - 1e-6 <= r <= 1.05 + 1e-6
    assert (plan == tf.HIGH).sum() * tf.HH_PER_BLOCK <= 24
    both = ((plan == tf.HIGH).any(1) & (plan == tf.LOW).any(1)).sum()
    assert both <= tf.TrialRules().max_double_days_per_week
    for d in range(1, D):
        assert tf.TrialRules().day_ok(plan[d - 1][None], plan[d])
    m.fixed = {0: 0, 3: int(choice[3])}
    c2, _ = mip.solve(m, time_limit=30)
    assert c2[0] == 0 and c2[3] == choice[3]


def test_harness_baselines_and_planner_in_london_env():
    inp = _inputs()
    arr = _arrays()
    env = le.LondonTariffEnv(inp, arr, le.LondonConfig(months=["2013-01", "2013-02"], members=np.arange(4)), seed=0)
    tdays = pd.DatetimeIndex(inp.days["tday"])
    sched = {t: np.array([0, 0, 0, 0, 0, 1, 0, 0], np.int8) for t in tdays if t.dayofweek == 2}
    r = hs.evaluate(env, bl.ScheduleReplay(sched), ["2013-01"], [0, 1])
    assert len(r) == 2 and (r["high_hh"] == 6 * sum(1 for t in sched if t.month == 1 and t.year == 2013)).all()
    q = hs.evaluate(env, bl.QuotaCPP(1.2), ["2013-01"], [0])
    assert q["invalid_days"].iloc[0] == 0 and 0.85 < q["revenue_ratio"].iloc[0] < 1.2
    P = pt.candidate_patterns()
    E = pt.pattern_effects(P, arr)
    mipp = pl.MonthlyMIP(E, P, time_limit=60)
    m = hs.evaluate(env, mipp, ["2013-01"], [0, 1, 2, 3])
    assert mipp.log[-1]["status"] == 0 and (m["invalid_days"] == 0).all()
    # the plan meets the revenue band for the members it planned with, and the segment cap
    assert (m["revenue_ratio"] >= 0.97 - 2e-3).all() and (m["revenue_ratio"] <= 1.03 + 2e-3).all()
    assert (m["bill_segment_max_nonresponding"] <= 0.03 + 1e-6).all()
    assert (m["daily_peak_cut_pct"] > q["daily_peak_cut_pct"].iloc[0] - 1e-9).all()


def test_known_load_uses_only_past_and_forecast():
    inp = _inputs()
    env = le.LondonTariffEnv(inp, _arrays(), le.LondonConfig(months=["2013-01"], members=np.arange(4)), seed=0)
    env.reset(options={"month": "2013-01", "member": 0})
    t = 10
    k = pl.known_load(env, t)
    days = env.day_idx
    assert np.allclose(k[int(days[t - 3])], inp.baseline[days[t - 3]])          # complete past day: actual
    assert np.allclose(k[int(days[t])], env.forecast[days[t]])                  # tomorrow: forecast
    later = days[t + 5]
    src = later - 7 * int(np.ceil((t + 5 - (t - 2)) / 7))
    assert src <= days[t - 2] and np.allclose(k[int(later)], inp.baseline[src])  # later: latest complete same weekday


def test_soft_band_gets_as_close_as_possible():
    P = pt.candidate_patterns()
    D = 4
    rev = np.tile(1.0 + 0.1 * (P == tf.HIGH).sum(1), (D, 1)) / D       # at most +80% per day with 8 High blocks...
    value = np.zeros((D, len(P)))
    m = mip.MonthModel(value, rev, 1.0, 0.0, P, pt.compatibility(P), rules=tf.TrialRules(max_high_hh_per_month=12))
    m.rev_den = 2.0                                                          # ...so the ratio stays below 0.53: infeasible
    choice, info = mip.solve(m, time_limit=30)
    assert info["objective"] is None
    m.soft_penalty = 10.0
    choice, info = mip.solve(m, time_limit=30)
    assert info["objective"] is not None and info["slack_revenue"] > 0
    hh = (P[choice] == tf.HIGH).sum() * tf.HH_PER_BLOCK
    assert hh == 12                                                          # uses the whole High budget to get closer




def test_ercot_known_load_hides_future_renewables_and_demand():
    from ttrl.env import ercot as er
    from ttrl.env import ercot_env as ee
    from ttrl.env.responses import TrialResponse
    hours = er.local_year_hours(2024)
    loc = hours.tz_localize("UTC").tz_convert(er.TZ).tz_localize(None)
    keep = (loc.month <= 2) | ((loc.month == 3) & (loc.day <= 2)) | ((loc.month == 12) & (loc.day == 31))
    hours, loc = hours[keep], loc[keep]
    n = len(hours)
    demand = (50000 + 15000 * np.sin(2 * np.pi * (loc.hour - 6) / 24)).to_numpy(float)
    ren = 1000.0 * np.arange(n) / 24                      # grows day by day: future values are distinguishable
    y = er.ErcotYear(2024, hours, demand, 0.35 * demand, 0.5 * demand, None, 1.0, np.zeros(n, bool), ren, demand * 1.01)
    sur = er.Surrogate(hours, np.full(n, 8.0), np.full(n, 0.45), np.full(n, 8.5), np.zeros(n, int))
    days = ee.ErcotDays.build(y, sur, np.full(n, 10.0))
    cfg = ee.ErcotConfig(["2024-01"], np.array([0]), 2.0, 0.0, 1e5, {1: 1.0})
    env = ee.ErcotTariffEnv(days, TrialResponse(_arrays()), cfg, seed=0)
    env.reset(options={"month": "2024-01", "member": 0})
    t = 10
    k = pl.known_load(env, t)
    d_now, d_later, d_last = int(env.day_idx[t]), int(env.day_idx[t + 6]), int(env.day_idx[t - 2])
    rows_last = days.row[d_last]
    assert np.allclose(k["ren"][d_later], ren[rows_last])                    # persistence of the latest complete day
    assert not np.allclose(k["ren"][d_later], ren[days.row[d_later]])
    assert np.allclose(k[d_now], y.demand_forecast[days.row[d_now]])         # tomorrow: day-ahead forecast
    assert int(env.day_idx[t - 3]) not in k                                  # complete past days: actual (no override)


def test_revenue_scenarios_span_the_response_range():
    arr = _arrays(members=30, seed=3)
    P = pt.candidate_patterns()[:40]
    E = pt.pattern_effects(P, arr)
    pick = pl.revenue_scenarios(E, P, k=5)
    stat = np.einsum("mph,ph->m", E, tf.slot_prices(P))
    assert len(pick) == 5 and stat.argmin() in pick and stat.argmax() in pick
