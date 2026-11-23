"""Tests for the LCL event-study DiD on a simulated panel with known effects."""
import numpy as np
import pandas as pd
import pytest

from ttrl.analysis import lcl_did as did
from ttrl.data import lcl_features as lf

TRUE = {"trial": 0.02, "High": -0.10, "Low": 0.08, "High_post_0_2h": 0.03}


def _calendar() -> pd.DataFrame:
    """Dec 2012 (pre) and Jan 2013 (trial), GMT, with High 17-20 and Low 01-05 events."""
    ts_label = pd.date_range("2012-12-01 00:30", "2013-02-01 00:00", freq="30min")
    cal = pd.DataFrame({"ts_label": ts_label, "ts_local": ts_label - did.HH})
    cal["date"] = cal["ts_local"].dt.normalize()
    cal["hh"] = cal["ts_local"].dt.hour * 2 + cal["ts_local"].dt.minute // 30
    cal["weekend"] = cal["ts_local"].dt.dayofweek >= 5
    cal["period"] = np.where(cal["ts_local"] < did.TRIAL_START, "pre", "trial")
    band = np.where(cal["period"] == "trial", "Normal", None).astype(object)
    day, hour = cal["ts_local"].dt.day, cal["ts_local"].dt.hour
    trial = cal["period"] == "trial"
    band[trial & (day % 5 == 0) & (hour >= 17) & (hour < 20)] = "High"
    band[trial & (day % 5 == 2) & (hour >= 1) & (hour < 5)] = "Low"
    cal["band"] = pd.array(band, dtype="string")
    cal["t_idx"] = np.arange(len(cal))
    return cal


def _panel(cal: pd.DataFrame, X: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_tou, n_std = 30, 60
    hh_id = np.arange(1, n_tou + n_std + 1)
    tou = hh_id <= n_tou
    a = rng.gamma(4, 0.05, size=len(hh_id))                       # household level
    prof = 0.1 + 0.08 * np.sin(2 * np.pi * cal["hh"].to_numpy() / 48)  # daily shape
    lam = np.repeat(rng.normal(0, 0.05, size=len(cal) // 48 + 1), 48)[:len(cal)]  # common daily shocks
    effect = sum(TRUE[k] * X[k].to_numpy() for k in TRUE)
    y = a[:, None] + prof[None, :] + lam[None, :] + tou[:, None] * effect[None, :]
    y = y + rng.normal(0, 0.05, size=y.shape)
    panel = pd.DataFrame({
        "hh_id": np.repeat(hh_id, len(cal)).astype(np.int32),
        "tou": np.repeat(tou, len(cal)),
        "ts_label": np.tile(cal["ts_label"].to_numpy(), len(hh_id)),
        "kwh": y.ravel().astype(np.float32),
    })
    panel.loc[rng.random(len(panel)) < 0.02, "kwh"] = np.nan  # scattered missing readings
    sample = pd.DataFrame({"hh_id": hh_id, "tou": tou})
    sample["h_idx"] = sample.groupby("tou").cumcount()
    return panel, sample


def test_event_design_windows():
    cal = _calendar()
    X = did.event_design(cal)
    assert X["High"].sum() == (cal["band"] == "High").sum()
    # 2 h before a High event that starts at 17:00 is 15:00-16:30 local.
    pre = cal.loc[X["High_pre_2h"] == 1, "ts_local"]
    assert set(pre.dt.strftime("%H:%M")) == {"15:00", "15:30", "16:00", "16:30"}
    post = cal.loc[X["High_post_2_4h"] == 1, "ts_local"]
    assert set(post.dt.strftime("%H:%M")) == {"22:00", "22:30", "23:00", "23:30"}
    assert (X.loc[cal["period"] == "pre"].drop(columns=["trial", "post"]) == 0).all().all()
    Xp = did.event_design(cal, profile=True)
    assert Xp["High_in_0"].sum() == (X["High"].sum() / 6)  # one first half-hour per 3-hour event
    assert Xp["High_post_1"].sum() == Xp["High_in_0"].sum()


def test_did_recovers_simulated_effects(tmp_path):
    rng = np.random.default_rng(0)
    cal = _calendar()
    X = did.event_design(cal)
    panel, sample = _panel(cal, X, rng)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path)

    W = did.bootstrap_weights(sample, n_boot=30, seed=1)
    sums = did.group_sums(cal, sample, W, panel=panel_path, chunk=500)
    assert sums.n_readings == panel["kwh"].notna().sum()
    assert sums.n[True][0].sum() + sums.n[False][0].sum() == sums.n_readings
    res = did.estimate(cal, sums, X)
    tab = res.table.set_index("term")
    for k, v in TRUE.items():
        assert tab.loc[k, "coef_kwh"] == pytest.approx(v, abs=0.012), k
    assert abs(tab.loc["Low_post_2_4h", "coef_kwh"]) < 0.015
    assert (tab["se_boot_household"] > 0).all() and (tab["se_cluster_date"] > 0).all()
    assert res.boot.shape == (30, len(tab))


def test_estimate_rejects_collinear_design(tmp_path):
    rng = np.random.default_rng(0)
    cal = _calendar()
    X = did.event_design(cal)
    panel, sample = _panel(cal, X, rng)
    panel.to_parquet(tmp_path / "panel.parquet")
    sums = did.group_sums(cal, sample, did.bootstrap_weights(sample, 2, 0), panel=tmp_path / "panel.parquet")
    with pytest.raises(ValueError, match="rank deficient"):
        did.estimate(cal, sums, X, fe="month")  # 'trial' is a sum of month dummies
    for fe in ("month", "moy"):  # without the level terms the month FE are identified
        assert "High" in did.estimate(cal, sums, X.drop(columns=["trial", "post"]), fe=fe).table["term"].tolist()


@pytest.mark.skipif(not lf.XLSX_PATH.exists(), reason="Data_features.xlsx not present")
def test_calendar_real_grid():
    cal = did.calendar("2012-07-01 00:30", "2014-02-28 00:00")
    trial = cal[cal["period"] == "trial"]
    assert len(trial) == lf.TRIAL_HH and trial["band"].notna().all()
    assert trial.groupby("date").size().loc["2013-03-31"] == 46
    assert (trial["band"] == "High").sum() == 788 and (trial["band"] == "Low").sum() == 1660
    start = did.calendar("2013-01-01 00:00", "2013-01-02 00:00", label="start")
    assert start["ts_local"].iloc[0] == pd.Timestamp("2013-01-01 00:00")
