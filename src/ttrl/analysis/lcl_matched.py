"""Descriptive matched-day comparison of dToU consumption around price events.

This is NOT a causal estimate. For each High or Low event, consumption in a
clock window on the event day is compared with the same local clock window on
"Normal-only" days (no High or Low half-hour on the local calendar date) that
fall within +/- ``days`` days and whose daily mean temperature is within
+/- ``temp_tol`` degrees C of the event day's. There is no control group in the
source file, so these comparisons cannot separate the price response from
whatever else differs between event and comparison days.

Windows, in half-hours relative to the event:

* ``pre_2h``    : [start - 2 h, start)
* ``event``     : [start, end)
* ``post_0_2h`` : [end, end + 2 h)
* ``post_2_4h`` : [end + 2 h, end + 4 h)

On the event side, the pre/post windows are used only if every half-hour in
them is Normal price; otherwise they would include another event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ttrl.data.lcl_features import HALF_HOUR, daily_table

WINDOWS = ("pre_2h", "event", "post_0_2h", "post_2_4h")


def window_times(start: pd.Timestamp, end: pd.Timestamp, window: str) -> pd.DatetimeIndex:
    """Local half-hour start times making up ``window`` for an event [start, end)."""
    if window == "event":
        a, b = start, end
    elif window == "pre_2h":
        a, b = start - pd.Timedelta(hours=2), start
    elif window == "post_0_2h":
        a, b = end, end + pd.Timedelta(hours=2)
    elif window == "post_2_4h":
        a, b = end + pd.Timedelta(hours=2), end + pd.Timedelta(hours=4)
    else:
        raise ValueError(window)
    return pd.date_range(a, b - HALF_HOUR, freq="30min")


def matched_window_table(
    df: pd.DataFrame,
    events: pd.DataFrame,
    col: str = "mean_all",
    days: int = 14,
    temp_tol: float = 2.0,
    same_daytype: bool = False,
    windows: tuple[str, ...] = WINDOWS,
) -> pd.DataFrame:
    """Return one row per (event, window) with event and matched-comparison consumption.

    Columns: event_id, band, window, n_hh, event_kwh (mean kWh per household per
    half-hour on the event day), ctrl_kwh (same, averaged over matched days),
    n_ctrl_days, pct (100 * (event_kwh / ctrl_kwh - 1)) and ``reason`` (why a
    row has no estimate, if any). Repeated local half-hours on the autumn DST
    day are averaged; half-hours that do not exist (spring DST) make a window
    unusable.
    """
    load = df.groupby("ts_local")[col].mean()
    band = df.groupby("ts_local")["band"].first()
    daily = daily_table(df)
    normal_days = set(daily.index[daily["day_type"] == "Normal-only"])

    rows = []
    for ev in events.itertuples(index=False):
        for w in windows:
            t = window_times(ev.start, ev.end, w)
            rec = {"event_id": ev.event_id, "band": ev.band, "window": w, "n_hh": len(t),
                   "event_kwh": np.nan, "ctrl_kwh": np.nan, "n_ctrl_days": 0, "reason": ""}
            if not t.isin(load.index).all():
                rec["reason"] = "outside data or DST gap"
            elif w != "event" and not (band.reindex(t) == "Normal").all():
                rec["reason"] = "overlaps another event"
            else:
                rec["event_kwh"] = load.reindex(t).mean()
                ctrl = _matched_controls(t, ev.date, load, daily, normal_days, days, temp_tol, same_daytype)
                rec["n_ctrl_days"] = len(ctrl)
                if ctrl:
                    rec["ctrl_kwh"] = float(np.mean(ctrl))
                else:
                    rec["reason"] = "no matched day"
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["pct"] = 100 * (out["event_kwh"] / out["ctrl_kwh"] - 1)
    return out


def _matched_controls(t, ev_date, load, daily, normal_days, days, temp_tol, same_daytype) -> list[float]:
    """Mean load over window ``t`` shifted by each admissible day offset."""
    vals = []
    t_ev = daily.loc[ev_date, "temp_mean"]
    wk_ev = daily.loc[ev_date, "weekend"]
    for k in range(-days, days + 1):
        if k == 0:
            continue
        d = ev_date + pd.Timedelta(days=k)
        if d not in daily.index or abs(daily.loc[d, "temp_mean"] - t_ev) > temp_tol:
            continue
        if same_daytype and daily.loc[d, "weekend"] != wk_ev:
            continue
        t2 = t + pd.Timedelta(days=k)
        if not set(t2.normalize()).issubset(normal_days) or not t2.isin(load.index).all():
            continue
        vals.append(load.reindex(t2).mean())
    return vals


def bootstrap_summary(tab: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                      by: tuple[str, ...] = ("band", "window")) -> pd.DataFrame:
    """Summarise ``matched_window_table`` output with event-level bootstrap 95% intervals.

    ``pooled_pct`` is the half-hour-weighted ratio sum(event) / sum(control) - 1;
    ``mean_pct`` is the unweighted mean of event-level percentages. Both are in %.
    Intervals are percentile intervals from resampling events with replacement. Each group
    gets its own generator seeded with ``seed``, so a group's interval does not depend on
    which other groups are in ``tab``.
    """
    rows = []
    total = tab.groupby(list(by)).size()
    ok = tab.dropna(subset=["pct"])
    for key, g in ok.groupby(list(by), sort=False):
        e = (g["event_kwh"] * g["n_hh"]).to_numpy()
        c = (g["ctrl_kwh"] * g["n_hh"]).to_numpy()
        p = g["pct"].to_numpy()
        idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
        pooled_b = 100 * (e[idx].sum(1) / c[idx].sum(1) - 1)
        mean_b = p[idx].mean(1)
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(by, key))
        rec.update(
            n_events=len(g),
            n_dropped=int(total[key if len(by) > 1 else key[0]] - len(g)),
            pooled_pct=100 * (e.sum() / c.sum() - 1),
            pooled_lo=np.percentile(pooled_b, 2.5), pooled_hi=np.percentile(pooled_b, 97.5),
            mean_pct=p.mean(),
            mean_lo=np.percentile(mean_b, 2.5), mean_hi=np.percentile(mean_b, 97.5),
            median_ctrl_days=float(g["n_ctrl_days"].median()),
        )
        rows.append(rec)
    return pd.DataFrame(rows)
