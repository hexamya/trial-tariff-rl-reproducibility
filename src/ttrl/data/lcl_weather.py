"""Daily London temperature for the LCL analysis: NOAA ISD-lite, London Heathrow.

The LCL workbook (``Data_features.xlsx``) carries temperature for 2013 only, and its clock is
unresolved (``docs/reports/lcl_schedule_summary.md``). The response kernel and the control-group
checks also need the pre-trial months (Jul-Dec 2012) and Jan-Feb 2014, so they use Heathrow
(ISD station 037720-99999, ICAO EGLL) for the whole window. Hourly UTC observations are
converted to local (Europe/London) time and averaged per local calendar date. In 2013 the
daily means correlate at 0.993 with the workbook's (mean difference -0.2 C).
"""
from __future__ import annotations

import pandas as pd

from ttrl.data import isd
from ttrl.data.lcl_features import TZ

STATION = "037720-99999"  # LONDON HEATHROW (isd-history.csv)
MIN_HOURS = 12            # fewest hourly readings accepted for a daily mean


def hourly(years: tuple[int, ...] = (2012, 2013, 2014)) -> pd.DataFrame:
    """Hourly Heathrow temperature: ts_utc, ts_local (naive local time), temp_c."""
    df = pd.concat([isd.parse_isd_lite(isd.isd_lite_path(STATION, y)) for y in years], ignore_index=True)
    df["ts_local"] = df["ts_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
    return df[["ts_utc", "ts_local", "temp_c"]]


def daily_mean(first: str = "2012-06-30", last: str = "2014-02-28") -> pd.Series:
    """Mean temperature per local date (index: date), name ``temp_c``. Raises on gaps."""
    years = tuple(range(pd.Timestamp(first).year, pd.Timestamp(last).year + 1))
    h = hourly(years)
    g = h.dropna(subset=["temp_c"]).groupby(h["ts_local"].dt.normalize())["temp_c"]
    d = g.mean().where(g.count() >= MIN_HOURS)
    dates = pd.date_range(first, last, freq="D")
    d = d.reindex(dates)
    if d.isna().any():
        raise ValueError(f"no Heathrow daily mean for {d.index[d.isna()].strftime('%Y-%m-%d').tolist()}")
    d.index.name = "date"
    return d.rename("temp_c")
