"""Tests for the control-group checks (Economy 7-like night profiles, synthetic control)."""
import numpy as np
import pandas as pd
import pytest

from ttrl.analysis import lcl_controls as ctl
from ttrl.analysis import lcl_did as did


def _pre_calendar() -> pd.DataFrame:
    ts_label = pd.date_range("2012-11-01 00:30", "2013-01-01 00:00", freq="30min")
    cal = pd.DataFrame({"ts_label": ts_label, "ts_local": ts_label - did.HH})
    cal["date"] = cal["ts_local"].dt.normalize()
    cal["hh"] = cal["ts_local"].dt.hour * 2 + cal["ts_local"].dt.minute // 30
    cal["weekend"] = cal["ts_local"].dt.dayofweek >= 5
    cal["period"] = "pre"
    return cal


def _profiles(rng, n_tou=400, n_std=200, n_e7=12):
    """Households x (month, hh, weekend) mean kWh; the first n_e7 Std households have a night step."""
    cols = pd.MultiIndex.from_product([[11, 12], range(48), [False, True]], names=["month", "hh", "weekend"])
    hh = cols.get_level_values("hh").to_numpy()
    shape = 0.15 + 0.1 * np.exp(-((hh - 38) / 5.0) ** 2) + 0.05 * (hh >= 14)
    n = n_tou + n_std
    P = rng.gamma(8, 1 / 8, size=(n, 1)) * shape[None, :] * rng.normal(1, 0.05, size=(n, len(cols)))
    e7 = np.zeros(n, bool)
    e7[n_tou:n_tou + n_e7] = True
    P[e7] += 0.6 * ((hh >= 1) & (hh < 14))[None, :]
    tou = np.r_[np.ones(n_tou, bool), np.zeros(n_std, bool)]
    return pd.DataFrame(P, index=np.arange(1, n + 1), columns=cols), tou, e7


def test_night_share_flags_night_step():
    rng = np.random.default_rng(0)
    P, tou, e7 = _profiles(rng)
    share = ctl.night_share(P, _pre_calendar())
    thr = ctl.night_threshold(share, pd.Series(tou), 0.99)
    flagged = (~tou) & (share.to_numpy() > thr)
    assert flagged[e7].all()                                  # every night-step household
    assert flagged[~tou & ~e7].mean() < 0.05                  # few others above the dToU 99th pct


def test_fill_profiles():
    rng = np.random.default_rng(0)
    P, _, _ = _profiles(rng, 5, 5, 0)
    Q = P.copy()
    Q.iloc[0, 0] = np.nan                      # month 11, hh 0, weekday
    Q.iloc[1, :] = np.nan                      # a household with no readings at all
    F = ctl.fill_profiles(Q)
    assert not F.isna().any().any()
    assert F.iloc[0, 0] == pytest.approx(P.iloc[0][(12, 0, False)])
    assert F.iloc[1, 0] == pytest.approx(F.drop(index=F.index[1]).iloc[:, 0].mean())


def test_synthetic_weights_recover_mixture():
    rng = np.random.default_rng(0)
    donors = rng.gamma(4, 0.05, size=(200, 30))
    w_true = np.zeros(30)
    w_true[[3, 17]] = [0.3, 0.7]
    w = ctl.synthetic_weights(donors, donors @ w_true)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, w_true, atol=1e-3)
    s = ctl.sc_fit_stats(donors, donors @ w_true, w)
    assert s["rmse_pct"] < 0.1 and s["n_donors"] <= 3


def test_weight_rows():
    keep = np.array([True, False, True, True, False])
    W = ctl.subset_weights(4, keep, n_boot=20, seed=0)
    assert W[True].shape == (21, 4) and W[False].shape == (21, 5)
    assert (W[False][:, ~keep] == 0).all()
    assert (W[False][1:].sum(1) == keep.sum()).all() and (W[True][1:].sum(1) == 4).all()
    keep_t = np.array([True, True, False, True])
    Wt = ctl.subset_weights(4, keep, n_boot=10, seed=0, keep_tou=keep_t)
    assert (Wt[True][:, ~keep_t] == 0).all() and (Wt[True][1:].sum(1) == 3).all()
    W0 = ctl.subset_weights(4, keep, n_boot=20, seed=0, keep_tou=np.ones(4, bool))
    assert np.array_equal(W0[True], W[True]) and np.array_equal(W0[False], W[False])  # same draws

    rng = np.random.default_rng(1)
    M_std = rng.gamma(4, 0.05, size=(40, 50))
    M_tou = M_std[:10] * 1.0
    S, stats = ctl.sc_weights(M_tou, M_std, n_boot=3, seed=0)
    assert S[True].shape == (4, 10) and S[False].shape == (4, 40)
    assert np.allclose(S[False].sum(1), 1, atol=1e-5)
    assert len(stats) == 4
    stacked, spans = ctl.stack_weights([W, {True: W[True][:2], False: W[False][:2]}])
    assert stacked[True].shape[0] == 23 and spans[1] == slice(21, 23)


def test_pretrial_profiles(tmp_path):
    cal = _pre_calendar()
    rng = np.random.default_rng(0)
    hh_id = np.array([5, 9])
    panel = pd.DataFrame({
        "hh_id": np.repeat(hh_id, len(cal)).astype(np.int32),
        "tou": np.repeat([True, False], len(cal)),
        "ts_label": np.tile(cal["ts_label"].to_numpy(), 2),
        "kwh": rng.gamma(4, 0.05, size=2 * len(cal)).astype(np.float32),
    })
    panel.to_parquet(tmp_path / "p.parquet")
    sample = pd.DataFrame({"hh_id": [9, 5], "tou": [False, True]})
    P = ctl.pretrial_profiles(cal, sample, panel=tmp_path / "p.parquet")
    assert list(P.index) == [9, 5] and P.shape[1] == 2 * 48 * 2
    m = panel.merge(cal[["ts_label", "ts_local", "hh", "weekend"]], on="ts_label")
    exp = m[(m["hh_id"] == 5) & (m["hh"] == 0) & (m["ts_local"].dt.month == 11) & ~m["weekend"]]["kwh"].mean()
    assert P.loc[5, (11, 0, False)] == pytest.approx(exp, rel=1e-5)
