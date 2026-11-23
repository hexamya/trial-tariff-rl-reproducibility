"""Tests for the LCL features loader and event tools.

Synthetic tests always run; tests on the real file are skipped when it is absent.
"""
import numpy as np
import pandas as pd
import pytest

from ttrl.analysis.lcl_matched import window_times
from ttrl.data import lcl_features as lf


def _frame(bands: list[str], start: str = "2013-01-07 00:00") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(bands), freq="30min")
    return pd.DataFrame({"ts_local": ts, "band": bands})


def test_label_bands_maps_three_levels_and_rejects_others():
    out = lf.label_bands(pd.Series([0.672, 0.1176, 0.0399, np.nan]))
    assert out.iloc[:3].tolist() == ["High", "Normal", "Low"]
    assert pd.isna(out.iloc[3])
    with pytest.raises(ValueError):
        lf.label_bands(pd.Series([lf.PRICE_FLAT_CONTROL]))


def test_find_events_groups_runs_and_handles_midnight():
    bands = ["Normal"] * 46 + ["High"] * 2 + ["High"] * 2 + ["Low"] * 3 + ["Normal"] * 43
    ev = lf.find_events(_frame(bands))
    assert ev["band"].tolist() == ["High", "Low"]
    hi = ev.iloc[0]
    assert hi["n_hh"] == 4 and hi["duration_h"] == 2.0
    assert hi["start"] == pd.Timestamp("2013-01-07 23:00") and hi["end"] == pd.Timestamp("2013-01-08 01:00")
    assert bool(hi["crosses_midnight"]) and hi["start_hour"] == 23.0 and hi["weekday"] == "Monday"
    assert ev.iloc[1]["start"] == pd.Timestamp("2013-01-08 01:00")


def test_find_events_ignores_unpriced_rows():
    df = _frame(["Normal"] * 2 + ["High"] * 2 + ["Normal"] * 2)
    df["band"] = pd.array([pd.NA, pd.NA, "High", "High", "Normal", "Normal"], dtype="string")
    ev = lf.find_events(df)
    assert ev["band"].tolist() == ["High"] and ev.iloc[0]["n_hh"] == 2


def test_window_times():
    s, e = pd.Timestamp("2013-01-07 17:00"), pd.Timestamp("2013-01-07 20:00")
    assert len(window_times(s, e, "event")) == 6
    assert window_times(s, e, "pre_2h")[0] == pd.Timestamp("2013-01-07 15:00")
    assert window_times(s, e, "post_2_4h")[-1] == pd.Timestamp("2013-01-07 23:30")


needs_file = pytest.mark.skipif(not lf.XLSX_PATH.exists(), reason="Data_features.xlsx not present")


@needs_file
def test_real_file_alignment_and_shape():
    df = lf.load_lcl_features()  # validate=True re-estimates the lag
    assert len(df) == lf.TRIAL_HH and df["ts_utc"].is_unique
    per_day = df.groupby("date").size()
    assert per_day.loc["2013-03-31"] == 46 and per_day.loc["2013-10-27"] == 50
    assert (per_day.drop(pd.to_datetime(["2013-03-31", "2013-10-27"])) == 48).all()
    ev = lf.find_events(df)
    assert ev.groupby("band")["n_hh"].sum().to_dict() == {"High": 788, "Low": 1660}
