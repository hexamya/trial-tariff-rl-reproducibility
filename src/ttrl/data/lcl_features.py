"""Loader and price-event tools for the LCL dToU aggregate file ``Data_features.xlsx``.

The sheet ``DTOU_2013`` holds two series of different length side by side:

* ``DateTime``, ``Price`` and ``Temperature``: a regular half-hourly 2013 grid
  (17,520 rows, 48 per day, no DST gaps or repeats). Labels are interval starts
  on the local (Europe/London) clock.
* ``mean_*`` / ``sum_*`` consumption of the dToU group: 29,137 undated rows.

The consumption rows are NOT aligned with the 2013 grid row by row. Cross-
correlating consumption residuals with the High/Low price indicators gives a
sharp optimum at an offset of 8,833 rows in GMT months and 8,831 rows in BST
months (see :func:`estimate_alignment_lag`). This is what one gets if the
consumption rows are a regular UTC series whose row ``t`` covers the UTC
interval starting ``2012-06-30 23:30 + 30 min * t``. Equivalently, the original
labels ran 2012-07-01 00:00 ... 2014-02-28 00:00 and marked interval ends,
which matches the last timestamp of the public LCL smart-meter release. So the
file carries six months before the trial (Jul-Dec 2012), the trial year 2013,
and two months after it (Jan-Feb 2014).

:func:`load_lcl_features` rebuilds that timeline and attaches price and
temperature by local clock time. Only the trial year has a price.

DST handling in the tidy frame: ``ts_utc`` is unique and regular. ``ts_local``
is timezone-naive local time, so 2013-03-31 has 46 half-hours (01:00 and 01:30
do not exist) and 2013-10-27 has 50 (01:00 and 01:30 occur twice, told apart
by ``is_dst``). The price grid's 01:00/01:30 rows on 2013-03-31 are not used,
and the single 01:00/01:30 grid rows on 2013-10-27 apply to both repeats. All
four are Normal-price rows, which the loader checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ttrl.paths import RAW

XLSX_PATH = RAW / "lcl" / "Data_features.xlsx"
SHEET = "DTOU_2013"

# dToU 2013 tariff levels (GBP/kWh) and the control group's flat rate (14.228 p/kWh per the
# London Datastore and Low Carbon London Report A3).
PRICE_HIGH = 0.6720
PRICE_NORMAL = 0.1176
PRICE_LOW = 0.0399
PRICE_FLAT_CONTROL = 0.14228
BANDS = {"High": PRICE_HIGH, "Normal": PRICE_NORMAL, "Low": PRICE_LOW}

TZ = "Europe/London"
HALF_HOUR = pd.Timedelta(minutes=30)
TRIAL_START_UTC = pd.Timestamp("2013-01-01 00:00")  # local midnight = UTC in January
TRIAL_HH = 17_520

# UTC start of the interval covered by consumption row 0, for each alignment.
ALIGNMENTS = {
    # Inferred: original labels are interval ends (2012-07-01 00:00 + 30 min * t).
    "utc_end_label": pd.Timestamp("2012-06-30 23:30"),
    # Alternative: original labels are interval starts (a half-hour later).
    "utc_start_label": pd.Timestamp("2012-07-01 00:00"),
}
EXPECTED_LAG = {"GMT": 8833, "BST": 8831}  # consumption row = price row + lag

CONSUMPTION_COLS = ["mean_flex", "mean_noflex", "mean_all", "sum_flex", "sum_noflex", "sum_all"]


def read_raw(path: Path | str = XLSX_PATH) -> pd.DataFrame:
    """Read the ``DTOU_2013`` sheet as-is (29,137 rows; the last 11,617 have no date)."""
    return pd.read_excel(path, sheet_name=SHEET)


def _price_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the dated 2013 rows (price, temperature) and check that the grid is regular."""
    grid = raw.loc[raw["DateTime"].notna(), ["DateTime", "Price", "Temperature"]].copy()
    grid = grid.rename(columns={"DateTime": "ts_local", "Price": "price_gbp_kwh", "Temperature": "temp_c"})
    expected = pd.date_range("2013-01-01 00:00", periods=TRIAL_HH, freq="30min")
    if len(grid) != TRIAL_HH or not (grid["ts_local"].to_numpy() == expected.to_numpy()).all():
        raise ValueError("DateTime column is not the regular 2013 half-hourly grid")
    if grid[["price_gbp_kwh", "temp_c"]].isna().any().any():
        raise ValueError("missing price or temperature in the 2013 grid")
    return grid.reset_index(drop=True)


def price_grid(path: Path | str = XLSX_PATH) -> pd.DataFrame:
    """The 2013 local-time price and temperature grid (ts_local, price_gbp_kwh, temp_c, band)."""
    grid = _price_grid(read_raw(path))
    _check_dst_rows(grid)
    grid["band"] = label_bands(grid["price_gbp_kwh"])
    return grid


def label_bands(price: pd.Series, tol: float = 1e-4) -> pd.Series:
    """Map GBP/kWh prices to 'High' / 'Normal' / 'Low'; NaN stays NaN, other values raise."""
    out = pd.Series(pd.NA, index=price.index, dtype="string")
    for name, level in BANDS.items():
        out[(price - level).abs() < tol] = name
    unknown = price.notna() & out.isna()
    if unknown.any():
        raise ValueError(f"unexpected price values: {sorted(price[unknown].unique())}")
    return out


def residual_log_load(x: np.ndarray, window_days: int = 15) -> np.ndarray:
    """Log load minus its rolling median at the same row position mod 48 (alignment helper)."""
    s = np.log(x)
    res = np.full(len(s), np.nan)
    pos = np.arange(len(s)) % 48
    for k in range(48):
        idx = np.flatnonzero(pos == k)
        v = pd.Series(s[idx])
        res[idx] = (v - v.rolling(window_days, center=True, min_periods=5).median()).to_numpy()
    return res


def estimate_alignment_lag(raw: pd.DataFrame, search: range = range(8800, 8870)) -> pd.DataFrame:
    """Score row offsets between the 2013 price grid and the consumption rows.

    For each lag ``L`` and each clock regime (GMT / BST half-hours of the price
    grid) it reports the mean consumption residual (in %) at High and at Low
    half-hours when price row ``s`` is paired with consumption row ``s + L``.
    The true alignment has the most negative High and most positive Low
    residual. ``L = 0`` is how the file is laid out.
    """
    grid = _price_grid(raw)
    res = residual_log_load(raw["mean_all"].to_numpy(float))
    band = label_bands(grid["price_gbp_kwh"]).to_numpy()
    ts = grid["ts_local"]
    bst = ((ts >= "2013-03-31 01:00") & (ts < "2013-10-27 01:00")).to_numpy()
    rows = []
    for lag in [0, *search]:
        for clock, mask in (("GMT", ~bst), ("BST", bst)):
            rec = {"lag": lag, "clock": clock}
            for b in ("High", "Low"):
                idx = np.flatnonzero((band == b) & mask) + lag
                idx = idx[idx < len(res)]
                rec[f"resid_{b}_pct"] = 100 * np.nanmean(res[idx])
            rec["score"] = rec["resid_Low_pct"] - rec["resid_High_pct"]
            rows.append(rec)
    return pd.DataFrame(rows)


def best_lags(scores: pd.DataFrame) -> dict[str, int]:
    """Return the highest-scoring lag per clock regime from :func:`estimate_alignment_lag`."""
    best = scores.loc[scores.groupby("clock")["score"].idxmax()]
    return dict(zip(best["clock"], best["lag"].astype(int)))


def load_lcl_features(
    path: Path | str = XLSX_PATH,
    alignment: str = "utc_end_label",
    period: str = "trial",
    validate: bool = True,
) -> pd.DataFrame:
    """Return a tidy half-hourly frame of dToU-group consumption with price and temperature.

    Parameters
    ----------
    alignment : 'utc_end_label' (inferred, default), 'utc_start_label' (one half-hour
        later), or 'file_order' (the sheet's row order, rows 0..17519 paired with the
        2013 grid; kept only to show how much the misalignment matters).
    period : 'trial' (2013 local year, 17,520 rows) or 'all' (Jul 2012 - Feb 2014).
    validate : re-estimate the alignment lag and check it matches the expected values.

    Columns: ts_utc, ts_local, is_dst, date, hh (local half-hour of day 0-47),
    period, price_gbp_kwh, band, temp_c, the six consumption columns, n_flex,
    n_noflex, n_all (households reporting = sum / mean) and sum_dup_flag (rows
    where the sums double-count meters; the means look unaffected).
    """
    raw = read_raw(path)
    grid = _price_grid(raw)
    cons = raw[CONSUMPTION_COLS].astype(float).reset_index(drop=True)

    if alignment == "file_order":
        df = pd.concat([grid, cons.iloc[:TRIAL_HH].reset_index(drop=True)], axis=1)
        df["ts_utc"] = pd.NaT
        df["is_dst"] = pd.NA
        df["period"] = "trial"
    elif alignment in ALIGNMENTS:
        if validate:
            lags = best_lags(estimate_alignment_lag(raw))
            if lags != EXPECTED_LAG:
                raise ValueError(f"alignment check failed: best lags {lags}, expected {EXPECTED_LAG}")
        start = ALIGNMENTS[alignment]
        cons["ts_utc"] = pd.date_range(start, periods=len(cons), freq="30min")
        local = cons["ts_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ)
        cons["ts_local"] = local.dt.tz_localize(None)
        cons["is_dst"] = local.map(lambda t: bool(t.dst()))
        cons["period"] = np.select(
            [cons["ts_local"] < "2013-01-01", cons["ts_local"] >= "2014-01-01"], ["pre", "post"], "trial"
        )
        _check_dst_rows(grid)
        df = cons.merge(grid, on="ts_local", how="left", validate="many_to_one")
        if period == "trial":
            df = df[df["period"] == "trial"].reset_index(drop=True)
            if len(df) != TRIAL_HH or df["price_gbp_kwh"].isna().any():
                raise ValueError("trial year does not have 17,520 priced half-hours")
    else:
        raise ValueError(f"unknown alignment {alignment!r}")

    df["band"] = label_bands(df["price_gbp_kwh"])
    df["date"] = df["ts_local"].dt.normalize()
    df["hh"] = df["ts_local"].dt.hour * 2 + df["ts_local"].dt.minute // 30
    for g in ("flex", "noflex", "all"):
        df[f"n_{g}"] = (df[f"sum_{g}"] / df[f"mean_{g}"]).round()
    typical = df["n_all"].rolling(97, center=True, min_periods=1).median()
    df["sum_dup_flag"] = df["n_all"] > 1.5 * typical
    cols = ["ts_utc", "ts_local", "is_dst", "date", "hh", "period", "price_gbp_kwh", "band", "temp_c",
            *CONSUMPTION_COLS, "n_flex", "n_noflex", "n_all", "sum_dup_flag"]
    return df[cols]


def _check_dst_rows(grid: pd.DataFrame) -> None:
    """Grid rows that do not exist (31 Mar) or are reused twice (27 Oct) must be Normal price."""
    special = grid["ts_local"].isin(pd.to_datetime(
        ["2013-03-31 01:00", "2013-03-31 01:30", "2013-10-27 01:00", "2013-10-27 01:30"]))
    if not (label_bands(grid.loc[special, "price_gbp_kwh"]) == "Normal").all():
        raise ValueError("a DST-affected grid half-hour is not Normal price")


def find_events(df: pd.DataFrame, bands: tuple[str, ...] = ("High", "Low")) -> pd.DataFrame:
    """Group consecutive half-hours of the same price band into events.

    ``df`` must be in time order (as returned by :func:`load_lcl_features`).
    Rows without a price (``period='all'``: Jul-Dec 2012, Jan-Feb 2014) have a
    missing band and never form events.
    Returns one row per event with start / end (local, end exclusive), n_hh,
    duration_h, band, date (local date of the start), weekday, month,
    start_hour (local, e.g. 17.5) and the row positions ``i0``/``i1`` (inclusive).
    """
    band = df["band"].astype(object).fillna("").to_numpy()
    new_run = np.r_[True, band[1:] != band[:-1]]
    run_id = np.cumsum(new_run) - 1
    pos = np.arange(len(df))
    g = pd.DataFrame({"run": run_id, "band": band, "pos": pos, "ts": df["ts_local"].to_numpy()})
    ev = g.groupby("run").agg(band=("band", "first"), i0=("pos", "min"), i1=("pos", "max"),
                              start=("ts", "first"), last=("ts", "last"))
    ev = ev[ev["band"].isin(bands)].reset_index(drop=True)
    ev["n_hh"] = ev["i1"] - ev["i0"] + 1
    ev["duration_h"] = ev["n_hh"] / 2
    ev["end"] = ev["last"] + HALF_HOUR
    ev["date"] = ev["start"].dt.normalize()
    ev["weekday"] = ev["start"].dt.day_name()
    ev["month"] = ev["start"].dt.month
    ev["start_hour"] = ev["start"].dt.hour + ev["start"].dt.minute / 60
    ev["crosses_midnight"] = ev["last"].dt.normalize() > ev["date"]
    ev.insert(0, "event_id", np.arange(len(ev)))
    return ev[["event_id", "band", "start", "end", "n_hh", "duration_h", "date", "weekday", "month",
               "start_hour", "crosses_midnight", "i0", "i1"]]


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per local date: mean temperature, High/Low half-hour counts and day type."""
    d = df.groupby("date").agg(
        temp_mean=("temp_c", "mean"),
        n_high=("band", lambda b: int((b == "High").sum())),
        n_low=("band", lambda b: int((b == "Low").sum())),
    )
    d["day_type"] = np.select(
        [(d.n_high > 0) & (d.n_low > 0), d.n_high > 0, d.n_low > 0], ["High+Low", "High", "Low"], "Normal-only")
    d["weekend"] = d.index.dayofweek >= 5
    return d
