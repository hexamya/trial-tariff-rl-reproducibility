"""Hourly residential load and residential share of ERCOT demand, 2019-2024 (S2 input).

Run:  python scripts/build_ercot_residential.py   (needs scripts/build_ercot_weather_years.py first)

The household-response kernel acts on the residential part of ERCOT load. The residential load is
constructed from two public sources:

1. **Shape:** NREL ResStock 2024 release 2 (AMY2018), the modelled 15-minute consumption of the
   whole ERCOT residential stock under 2018 weather (ttrl.data.resstock). A regression of hourly
   residential load on calendar and temperature is fitted to 2018 and applied to each year's actual
   temperatures, so every heat wave falls on its own days. Temperature: mean of four ISD stations
   at the load centres (Dallas-Fort Worth, Houston, San Antonio, Austin). Specification (``SPECS``,
   chosen by month-fold cross-validation on 2018, all candidates reported): local hour x day type
   dummies, two annual harmonics, cooling and heating degree-hours (base 18.3 C) by hour of day, their
   squares by 4-hour block, and degree-hours of the 3- and 24-hour lagged mean temperature with squares.
   Temperatures are clipped to the 2018 range before prediction.
2. **Level:** EIA-861 residential retail sales in the ERCOT balancing authority for each year
   (PUDL), grossed up by the year's ratio of EIA-930 demand to total EIA-861 retail sales (losses
   and unbilled use), so residential MWh and system demand are on the same footing.

Residential share = residential MWh / EIA-930 demand, per hour. Residential load is capped at the
hour's demand (``capped`` column): the cap binds only where reported demand was itself curtailed
(rotating outages), and those days are masked in training (ttrl.env.ercot.curtailed_days).

Writes data/processed/ercot_residential_<year>.parquet and outputs/ercot_residential.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ttrl.data import eia930, isd, pudl
from ttrl.data import resstock as rs
from ttrl.env.ercot import curtailed_days
from ttrl.paths import OUTPUTS, PROCESSED
from ttrl.reporting import md_table

YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
FIT_YEAR = 2018
STATIONS = {"KDFW": "722590-03927", "KIAH": "722430-12960", "KSAT": "722530-12921", "KAUS": "722540-13904"}
TZ = "America/Chicago"
LST_OFFSET = pd.Timedelta(hours=6)       # US Central standard time = UTC - 6
BASE_C = 18.3
FOLDS = {m: "ABCD"[(m - 1) % 4] for m in range(1, 13)}


def station_temps(year: int) -> pd.Series:
    """Hourly mean temperature of the four stations (UTC hour beginning), short gaps interpolated."""
    cols = {}
    for name, sid in STATIONS.items():
        d = isd.parse_isd_lite(isd.isd_lite_path(sid, year)).set_index("ts_utc")["temp_c"]
        cols[name] = d[~d.index.duplicated()]
    t = pd.DataFrame(cols)
    grid = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    t = t.reindex(grid).interpolate(limit=6, limit_area="inside")
    return t.mean(axis=1).rename("temp_c")


SPECS = {                                   # (temperature by hour, squared terms, lagged temperature)
    "4-h blocks, linear": dict(hourly_t=False, quad=False, lags=False),
    "hourly, linear": dict(hourly_t=True, quad=False, lags=False),
    "hourly + squared": dict(hourly_t=True, quad=True, lags=False),
    "hourly + squared + lags": dict(hourly_t=True, quad=True, lags=True),
}
CHOSEN = "hourly + squared + lags"


def design(index_utc: pd.DatetimeIndex, temp: pd.Series, hourly_t: bool = True, quad: bool = True,
           lags: bool = True, t_range: tuple[float, float] | None = None) -> pd.DataFrame:
    """Calendar and temperature regressors for UTC hours. ``temp`` must start at least 24 hours before
    the first hour (lags). ``t_range`` clips temperatures to the fitting year's range (no extrapolation
    of the squared terms)."""
    tmp = temp.clip(*t_range) if t_range else temp
    local = index_utc.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    t = tmp.reindex(index_utc).to_numpy()
    lag3 = tmp.rolling(3).mean().shift(1).reindex(index_utc).to_numpy()
    lag24 = tmp.rolling(24).mean().shift(1).reindex(index_utc).to_numpy()
    df = {}
    hod, wkend = local.hour, local.dayofweek >= 5
    for h in range(24):
        for w in (False, True):
            df[f"h{h}_{'we' if w else 'wd'}"] = ((hod == h) & (wkend == w)).astype(float)
    doy = 2 * np.pi * local.dayofyear.to_numpy() / 365.25    # smooth season (month dummies cannot be
    for k in (1, 2):                                           # estimated for a held-out month)
        df[f"sin{k}"], df[f"cos{k}"] = np.sin(k * doy), np.cos(k * doy)
    cdh, hdh = np.maximum(t - BASE_C, 0), np.maximum(BASE_C - t, 0)
    blk = hod // 4
    if hourly_t:
        for h in range(24):
            df[f"cdh_h{h}"], df[f"hdh_h{h}"] = cdh * (hod == h), hdh * (hod == h)
    else:
        for b in range(6):
            df[f"cdh_b{b}"], df[f"hdh_b{b}"] = cdh * (blk == b), hdh * (blk == b)
    if quad:
        for b in range(6):
            df[f"cdh2_b{b}"], df[f"hdh2_b{b}"] = cdh ** 2 * (blk == b), hdh ** 2 * (blk == b)
    if lags:
        for nm, v in (("l3", lag3), ("l24", lag24)):
            c, h_ = np.maximum(v - BASE_C, 0), np.maximum(BASE_C - v, 0)
            df[f"cdh_{nm}"], df[f"hdh_{nm}"], df[f"cdh2_{nm}"], df[f"hdh2_{nm}"] = c, h_, c ** 2, h_ ** 2
    return pd.DataFrame(df, index=index_utc).fillna(0.0)


def temps_with_lead(year: int, lead_hours: int = 48) -> pd.Series:
    """The year's station temperatures plus the last hours of the previous year (for lags) and the
    first hours of the next (local years run past 1 January 00:00 UTC)."""
    parts = [station_temps(year - 1).iloc[-lead_hours:], station_temps(year), station_temps(year + 1).iloc[:lead_hours]]
    return pd.concat(parts)


def resstock_hourly() -> pd.Series:
    """ResStock 2018 ERCOT residential MWh per UTC hour (interval-ending local standard time)."""
    r = rs.ercot_residential()
    start_lst = r["timestamp"] - pd.Timedelta(minutes=15)
    utc_hour = (start_lst + LST_OFFSET).dt.floor("h")
    s = r.groupby(utc_hour)["mwh"].sum()
    return s[(s.index >= f"{FIT_YEAR}-01-01") & (s.index < f"{FIT_YEAR + 1}-01-01")]


def fit(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    return np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)[0]


def main() -> None:
    y = resstock_hourly()
    full = y.index
    local = full.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    prof = y.groupby(local.hour).mean()
    summer = (local.month >= 6) & (local.month <= 8)
    prof_s = y[summer].groupby(local[summer].hour).mean()
    T18 = temps_with_lead(FIT_YEAR)
    t_range = (float(station_temps(FIT_YEAR).min()), float(station_temps(FIT_YEAR).max()))
    fold = pd.Series(local.month, index=full).map(FOLDS).to_numpy()
    spec_rows = []
    for name, kw in SPECS.items():                      # month-fold cross-validation on 2018
        X = design(full, T18, **kw)
        pred_cv = np.full(len(y), np.nan)
        for f in "ABCD":
            tr = fold != f
            b = fit(X[tr], y[tr])
            pred_cv[~tr] = X[~tr].to_numpy() @ b
        resid = y.to_numpy() - pred_cv
        top = np.argsort(y.to_numpy())[-100:]
        spec_rows.append({"specification": name, "columns": X.shape[1],
                          "CV R2": 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2),
                          "CV MAPE %": float(np.mean(np.abs(resid) / y.to_numpy()) * 100),
                          "error at 2018 peak hour %": float(100 * (pred_cv[np.argmax(y.to_numpy())] / y.max() - 1)),
                          "error, top 100 hours %": float(100 * (pred_cv[top].mean() / y.to_numpy()[top].mean() - 1)),
                          "chosen": name == CHOSEN})
    spec_tab = pd.DataFrame(spec_rows)
    beta = fit(design(full, T18, **SPECS[CHOSEN]), y)

    sales = pd.read_parquet(pudl.download_table("core_eia861__yearly_sales"))
    sales["year"] = pd.to_datetime(sales["report_date"]).dt.year
    ercot = sales[sales["balancing_authority_code_eia"] == "ERCO"]
    by = ercot.groupby(["year", "customer_class"])["sales_mwh"].sum().unstack()
    rows = []
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for yr in YEARS:
        d = eia930.load_ba_hourly(yr)
        d = d[(d["datetime_utc"] >= f"{yr}-01-01 06:00") & (d["datetime_utc"] < f"{yr + 1}-01-01 06:00")]
        idx = pd.DatetimeIndex(d["datetime_utc"])
        T = temps_with_lead(yr)
        clipped = int(((T.reindex(idx) < t_range[0]) | (T.reindex(idx) > t_range[1])).sum())
        pred = design(idx, T, **SPECS[CHOSEN], t_range=t_range).to_numpy() @ beta
        total_sales = by.loc[yr].sum()
        loss = d["demand"].sum() / total_sales
        res_target = by.loc[yr, "residential"] * loss
        res = pred * res_target / pred.sum()
        dem = d["demand"].to_numpy()
        capped = res > dem
        res = np.minimum(res, dem)
        share = res / dem
        loc = idx.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
        curt = np.isin(loc.normalize(), pd.DatetimeIndex(curtailed_days(yr)["date"]))
        out = pd.DataFrame({"datetime_utc": idx, "demand_mwh": dem, "residential_mwh": res, "share": share,
                            "capped": capped, "temp_c": T.reindex(idx).to_numpy()})
        out.to_parquet(PROCESSED / f"ercot_residential_{yr}.parquet", index=False)
        top = np.argsort(dem)[-100:]
        sa = (loc.month >= 6) & (loc.month <= 9) & (loc.hour >= 16) & (loc.hour < 20)
        rows.append({"year": yr, "EIA-861 residential share of sales %": 100 * by.loc[yr, "residential"] / total_sales,
                     "demand / sales": loss, "mean hourly share %": 100 * share.mean(),
                     "share at annual peak hour %": 100 * share[np.argmax(d["demand"].to_numpy())],
                     "share, top 100 hours %": 100 * share[top].mean(),
                     "share, Jun-Sep 16-20 h %": 100 * share[sa].mean(),
                     "min / max hourly share %": f"{100 * share.min():.0f} / {100 * share.max():.0f}",
                     "max share off curtailed days %": 100 * share[~curt].max(),
                     "at (local)": str(loc[~curt][np.argmax(share[~curt])])[:13] + "h",
                     "hours capped (on curtailed days)": f"{int(capped.sum())} ({int((capped & curt).sum())})",
                     "residential peak GW": res.max() / 1000,
                     "hours with temperature clipped": clipped})
    tab = pd.DataFrame(rows)
    report = "\n".join([
        "# ERCOT residential load and share of demand, 2019-2024 (S2 input)",
        "",
        "Generated by `python scripts/build_ercot_residential.py`. Shape: NREL ResStock 2024 release 2 (AMY2018) "
        "ERCOT aggregates, refitted on calendar and temperature (four ISD stations: DFW, IAH, SAT, AUS) and applied "
        "to each year's temperatures. Level: EIA-861 residential sales in BA ERCO, grossed up by EIA-930 demand / "
        "total EIA-861 sales.",
        "",
        "## Fit to ResStock 2018",
        "",
        f"* ResStock 2018 ERCOT residential total: {y.sum() / 1e6:.1f} TWh over {len(y):,} hours; peak {y.max() / 1000:.1f} GW.",
        f"* Timestamp check (taken as interval-ending local standard time): the mean daily profile has its minimum at "
        f"{int(prof.idxmin())}:00 local and its summer maximum at {int(prof_s.idxmax())}:00 local.",
        f"* Temperatures outside the 2018 range ({t_range[0]:.1f} to {t_range[1]:.1f} C) are clipped to it before "
        "prediction, so the squared terms are not extrapolated (count per year in the table below).",
        "",
        "Specifications compared by month-fold cross-validation on 2018 (four folds of interleaved months). All four "
        "were tried after a first version with month dummies, which cannot be estimated for a held-out month; the "
        "comparison is on modelled ResStock load, not on the paper's outcomes.",
        "",
        md_table(spec_tab, floatfmt="{:.3f}"),
        "",
        "## Residential share by year",
        "",
        md_table(tab, floatfmt="{:.1f}"),
        "",
        "Residential load is capped at the hour's demand; the cap binds only on curtailed-demand days (rotating "
        "outages, reported demand below what was consumed), which are masked in training. The highest share off "
        "those days, and the hour it falls in, is given per year.",
        "",
        "Limits: the hourly shape is a regression on modelled (ResStock) load, not metered residential load; the level "
        "assumes residential losses in proportion to sales; industrial load growth (data centres, crypto mining) "
        "enters only through total demand. The ERCOT environment reports results for residential shares scaled by "
        "0.8 and 1.2 as a sensitivity.",
    ])
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "ercot_residential.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
