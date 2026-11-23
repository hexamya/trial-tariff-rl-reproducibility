"""Tests for the S1 response kernel (design, month-fold CV, ensemble, S2 interface)."""
import numpy as np
import pandas as pd
import pytest

from ttrl.analysis import lcl_did as did
from ttrl.analysis import lcl_kernel as kn


def _calendar() -> pd.DataFrame:
    """Nov-Dec 2012 (pre) and Jan-Apr 2013 (trial, one month per CV fold), GMT clock."""
    ts_label = pd.date_range("2012-11-01 00:30", "2013-05-01 00:00", freq="30min")
    cal = pd.DataFrame({"ts_label": ts_label, "ts_local": ts_label - did.HH})
    cal["date"] = cal["ts_local"].dt.normalize()
    cal["hh"] = cal["ts_local"].dt.hour * 2 + cal["ts_local"].dt.minute // 30
    cal["weekend"] = cal["ts_local"].dt.dayofweek >= 5
    cal["period"] = np.where(cal["ts_local"] < did.TRIAL_START, "pre", "trial")
    trial = (cal["period"] == "trial").to_numpy()
    band = np.where(trial, "Normal", None).astype(object)
    doy, hour = cal["ts_local"].dt.dayofyear.to_numpy(), cal["ts_local"].dt.hour.to_numpy()
    band[trial & (doy % 4 == 0) & (hour >= 17) & (hour < 20)] = "High"      # 3 h evening
    band[trial & (doy % 6 == 1) & (hour >= 5) & (hour < 11)] = "High"       # 6 h morning
    band[trial & (doy % 4 == 2) & (hour >= 1) & (hour < 5)] = "Low"         # 4 h night
    band[trial & (doy % 9 == 5) & (hour >= 11) & (hour < 23)] = "Low"       # 12 h day
    cal["band"] = pd.array(band, dtype="string")
    cal["t_idx"] = np.arange(len(cal))
    return cal


def _temp(cal: pd.DataFrame, rng) -> pd.Series:
    dates = pd.date_range(cal["date"].min(), cal["date"].max(), freq="D")
    t = 8 + 5 * np.sin(np.arange(len(dates)) / 20) + rng.normal(0, 2, len(dates))
    return pd.Series(t, index=dates, name="temp_c")


def _simulate(cal, temp, rng, noise=1.0):
    """y in percent: seasonal x block baseline, High stronger in the evening, constant Low."""
    rows = kn.event_rows(cal, temp)
    hh, month = cal["hh"].to_numpy(), cal["ts_local"].dt.month.to_numpy()
    base = 0.5 * np.sin(2 * np.pi * hh / 48) + 0.8 * np.cos(month) * (hh // 8 == 3)
    high = (rows["band"] == "High").to_numpy()
    evening = rows["hour4"].to_numpy() == 3
    eff = np.where(high & evening, -10.0, np.where(high, -3.0, 0.0))
    eff = eff + 6.0 * (rows["band"] == "Low").to_numpy()
    X = kn.kernel_design(cal, temp, ())
    eff = eff - 2.0 * X["Low_pre_2h"].to_numpy()
    return base + eff + rng.normal(0, noise, len(cal))


def test_event_rows_features():
    rng = np.random.default_rng(0)
    cal = _calendar()
    temp = _temp(cal, rng)
    rows = kn.event_rows(cal, temp)
    high = rows[rows["band"] == "High"]
    assert set(high["duration_h"]) == {3.0, 6.0}
    assert set(cal.loc[high.index, "ts_local"].dt.hour[high["hour4"] == 3]) == {17, 18, 19}
    assert high.loc[high["hour4"] == 1, "duration_h"].eq(6.0).all()
    # one first half-hour per event
    from ttrl.data import lcl_features as lf
    assert rows["first"].sum() == len(lf.find_events(cal))
    assert rows.loc[rows["band"] == "", "event_id"].eq(-1).all()
    assert rows.loc[rows["band"] != "", "fatigue"].between(0, kn.FATIGUE_DAYS).all()
    # temperature is the row's own local date
    r = rows.index[rows["band"] == "Low"][0]
    assert rows.loc[r, "temp_c"] == pytest.approx(temp[cal.loc[r, "date"]])


def test_kernel_design_columns():
    rng = np.random.default_rng(0)
    cal = _calendar()
    temp = _temp(cal, rng)
    X = kn.kernel_design(cal, temp, ("hour4", "dur", "temp"))
    hcols = [c for c in X.columns if c.startswith("High|hour4=")]
    assert len(hcols) == 4
    # hour4 levels partition the High half-hours; dur drops its first level
    assert X[hcols].to_numpy().sum() == (cal["band"] == "High").sum()
    assert "High|dur=<=3h" not in X.columns and "High|dur=3-6h" in X.columns
    assert "Low_pre_2h" in X.columns and "trial" not in X.columns
    assert kn.kernel_design(cal, temp, None).shape[1] == 0
    with pytest.raises(ValueError):
        kn.kernel_design(cal, temp, ("weekday",))


def test_cv_selects_the_true_factor():
    rng = np.random.default_rng(1)
    cal = _calendar()
    temp = _temp(cal, rng)
    y = _simulate(cal, temp, rng)
    cid, clusters = kn.affected_clusters(cal)
    assert set(clusters["fold"]) == {"A", "B", "C", "D"}
    null = kn.cross_validate(cal, y, temp, None, "month", cid, clusters)
    const = kn.cross_validate(cal, y, temp, (), "month", cid, clusters)
    d, se = kn.paired_diff(const, null, cid)
    assert d < -3 * se
    chosen, path, results = kn.forward_select(cal, y, temp, "month", cid, clusters)
    assert len(chosen) == 1 and chosen[0] in ("hour4", "hour8")
    assert path["accepted"].sum() == 1
    assert not path.loc[path["model"] == chosen[0] + "+dur", "identified"].any()  # dur = hour4 for High here
    calib = kn.event_calibration(results["+".join(chosen)], cal, y, clusters, cid)
    slope = np.polyfit(calib["predicted"], calib["observed"], 1)[0]
    assert 0.8 < slope < 1.2


def test_fit_ensemble_and_kernel_effect():
    rng = np.random.default_rng(2)
    cal = _calendar()
    temp = _temp(cal, rng)
    draws = np.vstack([_simulate(cal, temp, rng) for _ in range(6)])
    X = kn.kernel_design(cal, temp, ("hour4",))
    F = kn.fe_matrix(cal, "month")
    f = kn.fit(cal, draws, X, F, ("hour4",), "month")
    assert (f.table["se_boot_household"] > 0).all() and (f.table["hi"] > f.table["lo"]).all()
    assert kn.fit(cal, draws[0], X, F).table["se_boot_household"].eq(0).all()
    assert f.coef["High|hour4=17-23"] == pytest.approx(-10, abs=0.6)
    assert f.coef["Low|hour4=23-05"] == pytest.approx(6, abs=0.5)
    assert "Low|hour4=05-11" not in f.coef.index          # no Low event in that block
    members = kn.ensemble(cal, draws, X, F, seed=0)
    assert members.shape == (5, len(f.table))
    assert list(members.columns) == list(f.table["term"])
    assert members["High|hour4=17-23"].std() > 0

    terms = [c for c in X.columns if c in f.coef.index]
    spec = kn.KernelSpec(("hour4",), "month", terms, support=kn.support(cal, temp))
    spec2 = kn.KernelSpec.from_json(spec.to_json())
    assert spec2.factors == ("hour4",) and spec2.terms == spec.terms
    eff = kn.kernel_effect(spec2, f.coef, cal, temp)
    assert np.allclose(eff, X[terms].to_numpy() @ f.coef[terms].to_numpy())
    assert kn.kernel_effect(spec2, members, cal, temp).shape == (5, len(cal))
    # a hypothetical schedule without a period column: one 3-hour evening High event
    day = pd.DataFrame({"ts_local": pd.date_range("2013-02-11", periods=48, freq="30min")})
    day["date"], day["hh"] = day["ts_local"].dt.normalize(), np.arange(48)
    day["weekend"] = False
    day["band"] = pd.array(np.where((day["hh"] >= 34) & (day["hh"] < 40), "High", "Normal"), dtype="string")
    e = kn.kernel_effect(spec2, f.coef, day, temp)
    assert np.allclose(e[34:40], f.coef["High|hour4=17-23"])
    assert np.allclose(e[30:34], f.coef["High_pre_2h"]) and np.allclose(e[40:44], f.coef["High_post_0_2h"])
    day.loc[10:15, "band"] = "Low"                         # 05:00-08:00 Low: a cell never estimated
    with pytest.raises(ValueError, match="without an estimate"):
        kn.kernel_effect(spec2, f.coef, day, temp)


def test_pct_outcome():
    n = 5
    ones = np.ones((1, n))
    tou_kwh, tou_dev, std_dev = 0.30 * ones, 0.02 * ones, 0.05 * ones
    sums = did.GroupSums(dev={True: tou_dev * 10, False: std_dev * 20}, kwh={True: tou_kwh * 10, False: ones},
                         n={True: 10 * ones, False: 20 * ones})
    y = kn.pct_outcome(sums)
    # P = (0.30 - 0.02) + 0.05 = 0.33; Delta = 0.02 - 0.05
    assert np.allclose(y, 100 * (-0.03) / 0.33)
    sub = sums.select([0])
    assert sub.delta().shape == (1, n)


def test_ensemble_spread_matches_combined_se():
    rng = np.random.default_rng(3)
    cal = _calendar()
    temp = _temp(cal, rng)
    draws = np.vstack([_simulate(cal, temp, rng) for _ in range(41)])
    X = kn.kernel_design(cal, temp, ())
    F = kn.fe_matrix(cal, "month")
    tab = kn.fit(cal, draws, X, F).table.set_index("term")
    members = kn.ensemble(cal, draws, X, F, seed=1)
    ratio = members.std(ddof=1) / tab["se_combined"]
    assert ratio[["High", "Low"]].between(0.7, 1.4).all()


def test_baseline_ignores_event_rows():
    rng = np.random.default_rng(4)
    cal = _calendar()
    temp = _temp(cal, rng)
    y = _simulate(cal, temp, rng, noise=0.5)
    cid, _ = kn.affected_clusters(cal)
    base = kn.baseline(cal, y, "month", cid)
    rows = kn.event_rows(cal, temp)
    high_evening = ((rows["band"] == "High") & (rows["hour4"] == 3)).to_numpy()
    # the baseline is not pulled down by the -10% evening High effect it was not fitted on
    assert np.mean(y[high_evening] - base[high_evening]) == pytest.approx(-10, abs=0.5)
    assert abs(np.mean(y[cid < 0] - base[cid < 0])) < 1e-8
