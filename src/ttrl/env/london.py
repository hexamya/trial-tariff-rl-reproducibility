"""London case of the S2 environment: inputs on the tariff-day grid, and its metrics.

Inputs (``build_inputs``; written by ``scripts/build_london_env.py`` to ``data/processed/``):

* **baseline**: mean kWh per household and half-hour of the Economy 7-screened Std training
  households (the S1 kernel's control group). They paid a flat rate throughout, so this is load
  without any price signal, Jul 2012 - Feb 2014.
* **households**: a fixed random sample of those households, for bill distributions, with missing
  readings filled by the household's own mean for the same slot and day type.
* **days**: one row per tariff day (02:00-02:00 local): date, weekend, month, S0 month fold,
  period (pre / trial / post) and Heathrow daily mean temperature of the date.

Grid: slot s of tariff day d starts at d 02:00 + 30 min x s on the local clock. On the autumn
change night the repeated 01:00-02:00 hour is averaged into slots 46-47; on the spring night the
missing hour is filled with slot 45.

Metrics: the response turns baseline load ``b`` into ``b * (1 + e / 100)`` with ``e`` from the
kernel; revenue and household bills are priced at the trial's levels against the flat rate on the
unresponsive load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ttrl.data.lcl_meters import PANEL_PATH, connect
from ttrl.data.splits import MONTH_FOLDS
from ttrl.env.tariff import HH_PER_DAY, PRICE_FLAT, slot_prices
from ttrl.paths import PROCESSED

INPUTS_PATH = PROCESSED / "env_london.npz"
DAYS_PATH = PROCESSED / "env_london_days.parquet"
DAY_START = pd.Timedelta(hours=2)


# ---------------------------------------------------------------- grid
def grid_index(cal: pd.DataFrame) -> pd.DataFrame:
    """Tariff day and slot of every calendar row (``ts_local`` = interval start, local naive)."""
    shifted = cal["ts_local"] - DAY_START
    tday = shifted.dt.normalize()
    slot = ((shifted - tday) // pd.Timedelta(minutes=30)).astype(int)
    return pd.DataFrame({"tday": tday, "slot": slot}, index=cal.index)


def grid_sources(gi: pd.DataFrame, tdays: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """For every grid cell (day x slot, flattened) the first and last calendar row that maps to it
    (equal for ordinary cells, the two repeats on the autumn night); cells without a row (the spring
    night) take the previous cell's rows."""
    day_pos = pd.Index(tdays).get_indexer(gi["tday"])
    ok = np.flatnonzero(day_pos >= 0)
    cell = day_pos[ok] * HH_PER_DAY + gi["slot"].to_numpy()[ok]
    n = len(tdays) * HH_PER_DAY
    first, last = np.full(n, -1), np.full(n, -1)
    g = pd.Series(ok).groupby(cell)
    first[g.min().index.to_numpy()] = g.min().to_numpy()
    last[g.max().index.to_numpy()] = g.max().to_numpy()
    missing = first < 0
    edge = np.flatnonzero(np.diff(np.r_[0, missing.astype(int), 0]))
    for s0, s1 in zip(edge[::2], edge[1::2]):           # runs of missing cells [s0, s1)
        if s0 > 0 and s1 - s0 <= 2 and s1 < n:           # a DST gap inside the data
            first[s0:s1], last[s0:s1] = first[s0 - 1], last[s0 - 1]
    return first, last                                   # -1 outside the data: NaN in to_grid


def to_grid(values: np.ndarray, gi: pd.DataFrame, tdays: pd.DatetimeIndex) -> np.ndarray:
    """(..., T) calendar-row values -> (..., n_days, 48) on the grid; the two autumn repeats are
    averaged (ignoring NaN), the missing spring hour copies the previous slot. Keeps the input dtype."""
    first, last = grid_sources(gi, tdays)
    v = np.atleast_2d(values)
    if not np.issubdtype(v.dtype, np.floating):
        v = v.astype(float)
    A, B = v[:, first], v[:, last]
    A[:, first < 0] = np.nan
    B[:, last < 0] = np.nan
    with np.errstate(invalid="ignore"):
        out = np.where(np.isnan(A), B, np.where(np.isnan(B), A, (A + B) / 2))
    out = out.reshape(v.shape[0], len(tdays), HH_PER_DAY)
    return out[0] if np.ndim(values) == 1 else out


def full_tariff_days(cal: pd.DataFrame) -> pd.DatetimeIndex:
    """Tariff days whose 48 slots all lie inside the calendar (allowing for the DST nights)."""
    gi = grid_index(cal)
    first, last = cal["ts_local"].min(), cal["ts_local"].max()
    days = pd.DatetimeIndex(sorted(gi["tday"].unique()))
    return days[(days + DAY_START >= first) & (days + DAY_START + pd.Timedelta(hours=23, minutes=30) <= last)]


# ---------------------------------------------------------------- inputs
@dataclass
class LondonInputs:
    days: pd.DataFrame          # one row per tariff day
    baseline: np.ndarray        # (n_days, 48) kWh per household and half-hour
    households: np.ndarray      # (H, n_days, 48) kWh, sample for bills
    household_ids: np.ndarray   # (H,)

    def save(self, npz: Path = INPUTS_PATH, days_path: Path = DAYS_PATH) -> None:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, baseline=self.baseline.astype(np.float32),
                            households=self.households.astype(np.float32), household_ids=self.household_ids)
        self.days.to_parquet(days_path, index=False)

    @classmethod
    def load(cls, npz: Path = INPUTS_PATH, days_path: Path = DAYS_PATH) -> "LondonInputs":
        z = np.load(npz)
        return cls(pd.read_parquet(days_path), z["baseline"].astype(float), z["households"], z["household_ids"])


def day_table(tdays: pd.DatetimeIndex, temp: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"tday": tdays})
    d["weekend"] = d["tday"].dt.dayofweek >= 5
    d["month"] = d["tday"].dt.to_period("M").astype(str)
    d["period"] = np.select([d["tday"] < "2013-01-01", d["tday"] >= "2014-01-01"], ["pre", "post"], "trial")
    d["fold"] = d["tday"].dt.month.map(MONTH_FOLDS)
    d["temp_c"] = temp.reindex(tdays).to_numpy()
    return d


EVENING_SLOTS = slice(30, 42)        # 17:00-23:00 (blocks 17-20 and 20-23)
N_SEGMENTS = 4


def exposure_segments(households: np.ndarray, days: pd.DataFrame, n: int = N_SEGMENTS) -> np.ndarray:
    """Quartile (0 = lowest) of each household's share of consumption at 17:00-23:00 on the pre-trial
    days (all days if there are none): the households most exposed to evening High prices."""
    pre = (days["period"] == "pre").to_numpy()
    use = pre if pre.any() else np.ones(len(days), bool)
    x = households[:, use]
    share = np.nansum(x[:, :, EVENING_SLOTS], axis=(1, 2)) / np.maximum(np.nansum(x, axis=(1, 2)), 1e-12)
    return pd.qcut(share, n, labels=False, duplicates="drop").astype(int)


def segment_bill_means(bills: np.ndarray, segments: np.ndarray, n: int = N_SEGMENTS) -> np.ndarray:
    """Mean bill change per segment (NaN households left out)."""
    return np.array([np.nanmean(bills[segments == s]) for s in range(n)])


FORECAST_LAGS = (7, 2)       # same day last week; the latest full day known when the tariff is posted
HDD_BASE_C = 15.5
MIN_FIT_DAYS = 60


def day_ahead_forecast(baseline: np.ndarray, days: pd.DataFrame, fit: np.ndarray | None = None) -> np.ndarray:
    """Day-ahead forecast of the baseline (n_days, 48), from what is known when a day's tariff is
    posted the day before: per half-hour OLS of the load on the same half-hour 7 and 2 days earlier,
    the previous day's mean temperature and its heating degrees below 15.5 C, and weekend dummies of
    the day and of the day two days earlier. Fitted on the ``fit`` days (default: the pre-trial days,
    July-December 2012, so no trial day enters the fit). NaN for days without both lags; with fewer than
    ``MIN_FIT_DAYS`` fit days the forecast is the same half-hour a week earlier."""
    tday = pd.DatetimeIndex(days["tday"])
    pos = pd.Index(tday)
    lag = {k: pos.get_indexer(tday - pd.Timedelta(days=k)) for k in (*FORECAST_LAGS, 1)}
    ok = (lag[7] >= 0) & (lag[2] >= 0) & (lag[1] >= 0)
    out = np.full(baseline.shape, np.nan)
    fit = (days["period"] == "pre").to_numpy() if fit is None else np.asarray(fit, bool)
    fit = fit & ok
    if fit.sum() < MIN_FIT_DAYS:
        out[lag[7] >= 0] = baseline[lag[7][lag[7] >= 0]]
        return out
    t1 = days["temp_c"].to_numpy(float)[np.maximum(lag[1], 0)]
    wk = days["weekend"].to_numpy(float)
    common = np.column_stack([np.ones(len(days)), t1, np.maximum(HDD_BASE_C - t1, 0.0), wk, wk[np.maximum(lag[2], 0)]])
    for h in range(baseline.shape[1]):
        X = np.column_stack([common, baseline[np.maximum(lag[7], 0), h], baseline[np.maximum(lag[2], 0), h]])
        b = np.linalg.lstsq(X[fit], baseline[fit, h], rcond=None)[0]
        out[ok, h] = X[ok] @ b
    return out


def household_matrix(cal: pd.DataFrame, hh_ids: np.ndarray, panel: Path = PANEL_PATH, chunk: int = 100) -> np.ndarray:
    """kWh per household (rows, in ``hh_ids`` order) and calendar row (NaN if missing)."""
    con = connect()
    con.register("cal_df", cal[["ts_label"]].assign(t_idx=np.arange(len(cal))))
    out = np.full((len(hh_ids), len(cal)), np.nan, np.float32)
    pos = {int(h): i for i, h in enumerate(hh_ids)}
    for c0 in range(0, len(hh_ids), chunk):
        ids = pd.DataFrame({"hh_id": np.asarray(hh_ids[c0:c0 + chunk], dtype=np.int32)})
        con.register("ids_df", ids)
        df = con.execute(f"""SELECT m.hh_id, c.t_idx, m.kwh FROM read_parquet('{panel.as_posix()}') m
                             JOIN ids_df USING (hh_id) JOIN cal_df c USING (ts_label)
                             WHERE m.kwh IS NOT NULL""").df()
        rows = df["hh_id"].map(pos).to_numpy()
        out[rows, df["t_idx"].to_numpy()] = df["kwh"].to_numpy(np.float32)
    return out


def fill_by_slot(x: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Fill NaN in (H, n_days, 48), in place, with each household's mean for the same slot and day type."""
    out = x
    for wk in (False, True):
        sel = weekend == wk
        block = out[:, sel, :]
        mean = np.nanmean(block, axis=1, keepdims=True)
        out[:, sel, :] = np.where(np.isnan(block), mean, block)
    overall = np.nanmean(out, axis=(1, 2), keepdims=True)
    return np.where(np.isnan(out), overall, out)


# ---------------------------------------------------------------- metrics
def responded(base: np.ndarray, effect_pct: np.ndarray) -> np.ndarray:
    return base * (1 + effect_pct / 100)


def revenue_ratio(blocks: np.ndarray, base: np.ndarray, effect_pct: np.ndarray) -> float:
    """Tariff revenue on responded load over flat revenue on the baseline (same days)."""
    price = slot_prices(blocks)
    return float((price * responded(base, effect_pct)).sum() / (PRICE_FLAT * base.sum()))


def bill_changes(blocks: np.ndarray, households: np.ndarray, effect_pct: np.ndarray) -> np.ndarray:
    """Per-household bill change (fraction) over the given days: tariff bill on responded load vs
    flat bill on the household's own load. ``households`` (H, n_days, 48), ``effect_pct`` (n_days, 48).
    Households with no consumption over the days (empty homes; 6 of the 500 in some month) get NaN
    and are left out of the distribution by :func:`cvar` and :func:`percentile`."""
    price = slot_prices(blocks)
    tariff = (price[None] * households * (1 + effect_pct[None] / 100)).sum(axis=(1, 2))
    flat = PRICE_FLAT * households.sum(axis=(1, 2))
    out = np.full(len(flat), np.nan)
    ok = flat > 0
    out[ok] = tariff[ok] / flat[ok] - 1
    return out


def cvar(x: np.ndarray, q: float = 0.95) -> float:
    """Mean of the upper (1 - q) tail, ignoring NaN."""
    x = np.sort(np.asarray(x, float)[~np.isnan(np.asarray(x, float))])
    if len(x) == 0:
        raise ValueError("no finite values")
    k = int(np.floor(q * len(x)))
    return float(x[k:].mean())


def percentile(x: np.ndarray, q: float) -> float:
    """Percentile ignoring NaN (q in 0-100)."""
    return float(np.nanpercentile(np.asarray(x, float), q))


def daily_peak_kw(load_kwh: np.ndarray) -> np.ndarray:
    """Daily maximum of half-hourly kWh, as average kW over the half-hour."""
    return 2 * np.asarray(load_kwh).max(axis=-1)
