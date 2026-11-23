"""Tests for ISD-lite parsing and plant temperature building.

Synthetic tests always run; checks of the built file are skipped when it is
absent (data/ is not in git).
"""
import gzip

import numpy as np
import pandas as pd
import pytest

from ttrl.data import isd
from ttrl.data import weather as w


def _line(ts: str, temp: int, dew: int = -9999) -> str:
    """One ISD-lite line in the fixed-width layout."""
    t = pd.Timestamp(ts)
    vals = [temp, dew, 10181, 170, 21, 6, -9999, -9999]
    return f"{t.year:4d} {t.month:02d} {t.day:02d} {t.hour:02d}" + "".join(f"{v:6d}" for v in vals)


def _hourly(values, start="2024-07-01 00:00") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h").astype("datetime64[us]")
    return pd.Series(values, index=idx, dtype="float64")


# --------------------------------------------------------------------------
# Parsing and single-station checks
# --------------------------------------------------------------------------
def test_parse_isd_lite_scales_tenths_and_missing():
    text = "\n".join([_line("2024-01-01 00:00", 200, 106),
                      _line("2024-01-01 01:00", -56),
                      _line("2024-01-01 02:00", -9999)]) + "\n"
    assert len(text.splitlines()[0]) == 61
    for source in (text, gzip.compress(text.encode())):
        df = isd.parse_isd_lite(source)
        assert df["ts_utc"].tolist() == list(pd.date_range("2024-01-01", periods=3, freq="h"))
        assert df["temp_c"].iloc[:2].tolist() == [20.0, -5.6]
        assert np.isnan(df["temp_c"].iloc[2])
        assert df["dewpoint_c"].iloc[0] == 10.6 and np.isnan(df["dewpoint_c"].iloc[1])
        assert df["slp_hpa"].iloc[0] == 1018.1 and df["wind_speed_ms"].iloc[0] == 2.1


def test_hour_grid_covers_utc_year_and_cems_year():
    g = w.hour_grid(2024)
    assert g[0] == pd.Timestamp("2024-01-01 00:00")
    assert g[-1] == pd.Timestamp("2025-01-01 06:00")
    assert len(g) == 8784 + 7
    assert (g.year == 2024).sum() == 8784


def test_to_grid_keeps_first_valid_duplicate_and_reindexes():
    obs = pd.DataFrame({"ts_utc": pd.to_datetime(["2024-07-01 00:00", "2024-07-01 00:00",
                                                  "2024-07-01 01:00", "2024-07-01 02:00"]),
                        "temp_c": [np.nan, 25.0, 26.0, 27.0]})
    grid = pd.date_range("2024-07-01", periods=4, freq="h")
    s = w.to_grid(obs, grid)
    assert s.iloc[:3].tolist() == [25.0, 26.0, 27.0] and np.isnan(s.iloc[3])


def test_qc_temperature_removes_range_and_isolated_spikes_but_not_steps():
    s = _hourly([20, 21, 60, 22, 22, 35, 22, 22, 22, 10, 10, 10])
    clean, n = w.qc_temperature(s)
    assert np.isnan(clean.iloc[2])          # out of range
    assert np.isnan(clean.iloc[5])          # +13 C spike vs both neighbours
    assert clean.iloc[9:].tolist() == [10, 10, 10]  # a 12 C step stays
    assert n == {"range": 1, "spike": 1}


def test_coverage_counts_calendar_year_only():
    s = pd.Series(1.0, index=w.hour_grid(2024))
    s.iloc[:84] = np.nan            # 84 hours missing in 2024
    s.iloc[-7:] = np.nan            # hours in 2025 do not count
    assert w.coverage(s, 2024) == pytest.approx(1 - 84 / 8784)


def test_haversine_km():
    assert isd.haversine_km(30, -97, 31, -97) == pytest.approx(111.19, abs=0.01)
    assert isd.haversine_km(30, -97, 30, -97) == 0
    d = isd.haversine_km(29.98, -95.36, np.array([32.90, 29.98]), np.array([-97.04, -95.36]))
    assert d.shape == (2,) and d[1] == 0 and 360 < d[0] < 370  # Houston to DFW area


# --------------------------------------------------------------------------
# Gap filling
# --------------------------------------------------------------------------
def test_gap_lengths_and_interior_missing():
    s = _hourly([np.nan, 1, np.nan, np.nan, 4, np.nan])
    assert w.gap_lengths(s.isna()).tolist() == [1, 0, 2, 2, 0, 1]
    assert w.interior_missing(s).tolist() == [False, False, True, True, False, False]


def test_interpolate_short_gaps_fills_only_short_interior_gaps():
    s = _hourly([0, np.nan, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, np.nan])
    out = w.interpolate_short_gaps(s, max_gap=3)
    assert out.iloc[:5].tolist() == [0, 1, 2, 3, 4]
    assert out.iloc[5:9].isna().all()       # 4-hour gap left whole
    assert np.isnan(out.iloc[10])           # end gap left
    assert out.iloc[9] == 9


def test_donor_offset_by_month_and_hour():
    idx = pd.date_range("2024-07-01", periods=24 * 30, freq="h")
    target = pd.Series(30.0 + (idx.hour >= 12), index=idx)  # +1 C in the afternoon
    donor = pd.Series(28.0, index=idx)
    off = w.donor_offset(target, donor)
    assert off[idx.hour < 12].eq(2.0).all() and off[idx.hour >= 12].eq(3.0).all()
    # Few pairs: falls back to the overall mean.
    few = w.donor_offset(target.iloc[:5], donor.iloc[:5], min_count=10)
    assert few.eq(2.0).all()


def test_fill_plant_series_order_and_labels():
    n = 48
    primary = _hourly(np.full(n, 30.0))
    primary.iloc[0] = np.nan                 # edge gap -> donor
    primary.iloc[5:7] = np.nan               # 2 h -> interpolated
    primary.iloc[20:30] = np.nan             # 10 h -> donor
    donor = _hourly(np.full(n, 28.0))
    donor.iloc[25] = np.nan                  # short donor gap: its own interpolation
    out = w.fill_plant_series(primary, "P", 5.0, [("D", 40.0, donor)])
    m = out["fill_method"]
    assert m.iloc[5:7].eq("interpolated").all() and out["temp_c"].iloc[5:7].eq(30.0).all()
    assert m.iloc[[0, *range(20, 30)]].eq("donor").all()
    assert out["temp_c"].iloc[20:30].eq(30.0).all()        # donor + 2 C offset
    assert out.loc[m == "donor", "station_id"].eq("D").all()
    assert out.loc[m == "donor", "distance_km"].eq(40.0).all()
    assert out.loc[m == "observed", "station_id"].eq("P").all()
    assert out["filled"].sum() == 1 + 2 + 10


def test_fill_plant_series_interpolates_rest_but_not_edges():
    primary = _hourly([np.nan, 10, np.nan, np.nan, np.nan, np.nan, np.nan, 16])
    out = w.fill_plant_series(primary, "P", 5.0, [])
    assert out["fill_method"].tolist() == ["missing", "observed"] + ["interpolated_long"] * 5 + ["observed"]
    assert out["temp_c"].iloc[2:7].tolist() == [11, 12, 13, 14, 15]
    assert out["station_id"].iloc[2:7].isna().all()
    kept = w.fill_plant_series(primary, "P", 5.0, [], interpolate_rest=False)
    assert kept["fill_method"].iloc[2:7].eq("missing").all()


# --------------------------------------------------------------------------
# Station choice
# --------------------------------------------------------------------------
def test_good_stations_is_lazy_and_skips_low_coverage():
    stations = pd.DataFrame({"station_id": ["a", "b", "c", "d", "far"],
                             "lat": [30.01, 30.02, 30.03, 30.04, 35.0],
                             "lon": [-97.0] * 5})
    cov = {"a": 0.90, "b": 0.99, "c": 0.97, "d": 0.99, "far": 1.0}
    calls = []

    def coverage_of(s):
        calls.append(s)
        return cov[s]

    plants = pd.DataFrame({"plant_id_eia": [1], "latitude": [30.0], "longitude": [-97.0]})
    sel = w.select_stations(plants, stations, coverage_of, n_donors=1)
    assert sel["station_id"].tolist() == ["b", "c"] and sel["rank"].tolist() == [0, 1]
    assert calls == ["a", "b", "c"]          # "d" never read
    gen = list(w.good_stations(30.0, -97.0, stations, cov.get, search_km=100))
    assert [g[0] for g in gen] == ["b", "c", "d"]   # "far" is beyond 100 km


# --------------------------------------------------------------------------
# Buddy check
# --------------------------------------------------------------------------
def test_persistent_runs_needs_consecutive_days_and_extends():
    idx = pd.date_range("2024-07-01", periods=24 * 20, freq="h")
    a = pd.Series(0.0, index=idx)
    a["2024-07-03"] = 2.5                    # extension day
    a["2024-07-04":"2024-07-07"] = 5.0       # 4-day run
    a["2024-07-12":"2024-07-13"] = -6.0      # only 2 days: not a run
    runs = w.persistent_runs(a, threshold=4, min_days=3, extend=2)
    assert len(runs) == 1
    first, last, sign, _ = runs[0]
    assert (first, last, sign) == (pd.Timestamp("2024-07-03"), pd.Timestamp("2024-07-07"), 1)


def test_buddy_check_flags_faulty_station_only():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-06-01", periods=24 * 60, freq="h")
    base = 28 + 6 * np.sin(2 * np.pi * (idx.hour - 15) / 24)
    temps = pd.DataFrame({s: base + rng.normal(0, 0.5, len(idx)) for s in "abcde"}, index=idx)
    temps.loc["2024-07-10":"2024-07-14", "c"] += 7.0       # 5-day warm bias
    lat = [30.0, 30.1, 30.2, 30.3, 30.4]
    lon = [-97.0] * 5
    flags, episodes, n_used, anomaly = w.buddy_check(temps, lat, lon)
    assert episodes["station_id"].tolist() == ["c"]
    assert flags["c"].sum() == 5 * 24 and flags.drop(columns="c").sum().sum() == 0
    assert (n_used == 3).all() and anomaly.shape == temps.shape


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def test_mean_diurnal_cycle_uses_local_clock_time():
    idx = pd.date_range("2024-07-01", periods=24 * 10, freq="h")
    temp = pd.DataFrame({"plant_id_eia": 1, "ts_utc": idx,
                         "temp_c": np.cos(2 * np.pi * (idx.hour - 21) / 24)})  # peak 21 UTC
    cyc = w.mean_diurnal_cycle(temp)
    assert cyc.idxmax(axis=1).iloc[0] == 16  # 21 UTC = 16 CDT


def test_hour_beginning_mean_within_plant():
    temp = pd.DataFrame({"plant_id_eia": [1, 1, 1, 2, 2], "temp_c": [10.0, 12, 14, 20, 22]})
    out = w.hour_beginning_mean(temp)
    assert out.iloc[[0, 1, 3]].tolist() == [11, 13, 21]
    assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[4])


# --------------------------------------------------------------------------
# Built file (skipped when absent)
# --------------------------------------------------------------------------
BUILT = w.output_paths(2024)["temperature"]


@pytest.mark.skipif(not BUILT.exists(), reason="plant_temperature_2024.parquet not built")
def test_built_file_schema_and_cems_hours_complete():
    t = pd.read_parquet(BUILT)
    assert list(t.columns) == ["plant_id_eia", "ts_utc", "temp_c", "station_id", "distance_km",
                               "filled", "fill_method"]
    assert not t.duplicated(["plant_id_eia", "ts_utc"]).any()
    assert t.groupby("plant_id_eia").size().eq(len(w.hour_grid(2024))).all()
    cems_year = (t["ts_utc"] >= "2024-01-01 06:00") & (t["ts_utc"] <= "2025-01-01 05:00")
    assert t.loc[cems_year, "temp_c"].notna().all()
    assert t["temp_c"].dropna().between(w.TEMP_MIN_C, w.TEMP_MAX_C).all()
    assert (t["filled"] == (t["fill_method"] != "observed")).all()
