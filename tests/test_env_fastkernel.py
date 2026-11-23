"""The fast kernel must reproduce ttrl.analysis.lcl_kernel.kernel_effect on block schedules."""
import numpy as np
import pandas as pd
import pytest

from ttrl.analysis import lcl_kernel as kn
from ttrl.env import fastkernel as fk
from ttrl.env import tariff as tf


def _spec_and_coef(rng, members=3):
    cal = _calendar(np.zeros((2, 8), int))
    temp = _temp(cal)
    X = kn.kernel_design(cal, temp, ("hour8", "dur"))
    terms = list(X.columns)
    coef = pd.DataFrame(rng.normal(0, 5, size=(members, len(terms))), columns=terms)
    return kn.KernelSpec(("hour8", "dur"), "month", terms), coef


def _calendar(days: np.ndarray) -> pd.DataFrame:
    """Tariff days from 2013-11-02 02:00 (no DST change), 48 half-hours each."""
    n = len(days)
    ts = pd.date_range("2013-11-02 02:00", periods=n * 48, freq="30min")
    cal = pd.DataFrame({"ts_local": ts})
    cal["date"] = cal["ts_local"].dt.normalize()
    cal["hh"] = cal["ts_local"].dt.hour * 2 + cal["ts_local"].dt.minute // 30
    cal["weekend"] = cal["ts_local"].dt.dayofweek >= 5
    names = {tf.LOW: "Low", tf.NORMAL: "Normal", tf.HIGH: "High"}
    cal["band"] = pd.array([names[int(x)] for x in tf.slot_bands(days).reshape(-1)], dtype="string")
    return cal


def _temp(cal):
    d = pd.date_range(cal["date"].min() - pd.Timedelta(days=1), cal["date"].max() + pd.Timedelta(days=1))
    return pd.Series(10.0, index=d)


def _random_days(rng, n):
    pats = tf.valid_day_patterns()
    days = pats[rng.integers(0, len(pats), n)]
    days[rng.random(n) < 0.4] = 0  # plenty of Normal days, so windows show up
    return days


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_kernel_matches_kernel_effect(seed):
    rng = np.random.default_rng(seed)
    spec, coef = _spec_and_coef(rng)
    days = _random_days(rng, 20)
    cal = _calendar(days)
    temp = _temp(cal)
    ref = kn.kernel_effect(spec, coef, cal, temp)                 # (M, T)
    arr = fk.from_coefficients(spec, coef)
    fast = fk.effect(tf.slot_bands(days).reshape(-1), arr)
    assert fast.shape == ref.shape
    assert np.allclose(fast, ref, atol=1e-9)
    one = fk.effect(tf.slot_bands(days).reshape(-1), arr.member(1))
    assert np.allclose(one[0], ref[1])


def test_runs_cross_day_boundary_and_duration_class():
    days = np.zeros((2, 8), int)
    days[0, 7] = tf.LOW
    days[1, 0] = tf.LOW            # one 6-hour Low event 23:00-05:00 across the boundary
    runs = fk.event_runs(tf.slot_bands(days).reshape(-1))
    assert runs == [(tf.LOW, 42, 53)]
    assert fk.duration_class(12) == 1 and fk.duration_class(6) == 0 and fk.duration_class(14) == 2


def test_rules_and_patterns():
    rules = tf.TrialRules()
    pats = tf.valid_day_patterns(rules)
    assert len(pats) > 100 and (pats == 0).all(1).any()
    two_high = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    assert not rules.day_ok(np.empty((0, 8)), two_high)
    long_low = np.full(8, tf.LOW)
    assert rules.day_ok(np.empty((0, 8)), long_low)
    assert not rules.day_ok(long_low[None, :], np.r_[tf.LOW, tf.LOW, tf.LOW, np.zeros(5, int)])  # 11 blocks
    c = rules.counts(np.array([two_high, long_low]))
    assert c == {"high_hh": 12, "low_hh": 48, "double_days": 0}
    assert np.allclose(tf.slot_prices(np.array([1, 0, -1, 0, 0, 0, 0, 0]))[[0, 6, 12]],
                       [tf.PRICE[tf.HIGH], tf.PRICE[tf.NORMAL], tf.PRICE[tf.LOW]])
