"""Event-study difference-in-differences for the LCL dToU trial (S1 response kernel).

Design
------
Every dToU (ToU) household faced the same 2013 price calendar, so event effects are
identified only against the flat-rate control group (Std). With household x
half-hour-of-day x weekend fixed effects and one fixed effect per half-hour (date x
half-hour), the treatment coefficients ``b`` of

    y_it = a_{i,h(t),w(t)} + l_t + D_i * x_t'b + e_it

depend on the data only through the difference of group means

    Delta_t = mean_{i in ToU} y~_it - mean_{i in Std} y~_it,

where ``y~`` is load minus the household's own half-hour-of-day x weekend mean. On a
balanced panel, OLS of ``Delta_t`` on ``x_t`` and half-hour-of-day x weekend dummies gives
exactly the two-way FE estimate. The panel here is close to balanced (coverage filter);
demeaning within household keeps households with a few missing readings from moving the
group means.

Regressors ``x_t`` (local time; all zero before 2013):

* ``trial``: 1 in 2013. Level shift of dToU households on Normal half-hours relative to the
  pre-trial months: Normal 11.76 p vs the flat 14.228 p, plus anything else that changed for
  dToU households when the trial began.
* ``post``: 1 in Jan-Feb 2014, after the trial.
* ``High``, ``Low``: event half-hours, relative to Normal 2013 half-hours.
* ``<band>_pre_2h``, ``<band>_post_0_2h``, ``<band>_post_2_4h``: Normal half-hours within
  2 h before an event or 0-2 h / 2-4 h after it (load moved ahead of or after events).
  With ``profile=True`` these become one dummy per half-hour of event time, and the event
  itself is split into its first six half-hours and the rest.

Time convention: the meter ``ts_label`` is taken as the UTC end of the half-hour (checked
against the price response in ``scripts/build_lcl_panel.py``); ``label='start'`` shifts
everything by 30 minutes for the sensitivity check.

Inference: (a) household bootstrap, resampling households within each group and
recomputing ``Delta``; (b) cluster-robust standard errors by local date on the ``Delta``
regression, for common shocks and serial correlation within a day. They cover different
sources of noise and both are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ttrl.data import lcl_features as lf
from ttrl.data.lcl_meters import PANEL_PATH, connect

HH = pd.Timedelta(minutes=30)
TRIAL_START = pd.Timestamp("2013-01-01")
TRIAL_END = pd.Timestamp("2014-01-01")
WINDOWS = {"pre_2h": ("pre", 1, 4), "post_0_2h": ("post", 1, 4), "post_2_4h": ("post", 5, 8)}
PROFILE_PRE, PROFILE_IN, PROFILE_POST = 4, 6, 8


# ---------------------------------------------------------------- calendar and design
def calendar(first_label: str, last_label: str, label: str = "end") -> pd.DataFrame:
    """One row per half-hour: ts_label (as in the meter files), ts_utc / ts_local (interval
    start), date, hh (local half-hour of day), weekend, period, price and band.

    ``label='end'``: ts_label marks the end of a UTC half-hour; ``'start'``: its start.
    """
    if label not in ("end", "start"):
        raise ValueError(label)
    ts_label = pd.date_range(first_label, last_label, freq="30min")
    cal = pd.DataFrame({"ts_label": ts_label})
    cal["ts_utc"] = ts_label - HH if label == "end" else ts_label
    local = cal["ts_utc"].dt.tz_localize("UTC").dt.tz_convert(lf.TZ)
    cal["ts_local"] = local.dt.tz_localize(None)
    cal["date"] = cal["ts_local"].dt.normalize()
    cal["hh"] = cal["ts_local"].dt.hour * 2 + cal["ts_local"].dt.minute // 30
    cal["weekend"] = cal["ts_local"].dt.dayofweek >= 5
    cal["period"] = np.select([cal["ts_local"] < TRIAL_START, cal["ts_local"] >= TRIAL_END],
                              ["pre", "post"], "trial")
    grid = lf.price_grid()[["ts_local", "price_gbp_kwh", "band"]]
    cal = cal.merge(grid, on="ts_local", how="left", validate="many_to_one")
    trial = cal["period"] == "trial"
    if cal.loc[trial, "band"].isna().any():
        raise ValueError("trial half-hour without a price")
    cal.loc[~trial, ["price_gbp_kwh", "band"]] = np.nan
    cal["t_idx"] = np.arange(len(cal))
    return cal


def event_design(cal: pd.DataFrame, profile: bool = False) -> pd.DataFrame:
    """Treatment regressors x_t (see module docstring), one row per calendar row."""
    n = len(cal)
    band = cal["band"].astype(object).fillna("").to_numpy()
    normal = band == "Normal"
    X: dict[str, np.ndarray] = {
        "trial": (cal["period"] == "trial").to_numpy(float),
        "post": (cal["period"] == "post").to_numpy(float),
    }
    ev = lf.find_events(cal)

    def put(name: str, idx: np.ndarray) -> None:
        idx = idx[(idx >= 0) & (idx < n)]
        X.setdefault(name, np.zeros(n))[idx] = 1.0

    for b in ("High", "Low"):
        if not profile:
            X[b] = (band == b).astype(float)
        for e in ev[ev["band"] == b].itertuples():
            pos = np.arange(e.i0, e.i1 + 1)
            if profile:
                for j in range(PROFILE_IN):
                    put(f"{b}_in_{j}", pos[j:j + 1])
                put(f"{b}_in_late", pos[PROFILE_IN:])
                pre = {f"{b}_pre_{k}": e.i0 - k for k in range(1, PROFILE_PRE + 1)}
                post = {f"{b}_post_{k}": e.i1 + k for k in range(1, PROFILE_POST + 1)}
                for name, i in {**pre, **post}.items():
                    if 0 <= i < n and normal[i]:
                        put(name, np.array([i]))
            else:
                for w, (side, k0, k1) in WINDOWS.items():
                    ks = np.arange(k0, k1 + 1)
                    idx = e.i0 - ks if side == "pre" else e.i1 + ks
                    idx = idx[(idx >= 0) & (idx < n)]
                    put(f"{b}_{w}", idx[normal[idx]])
    return pd.DataFrame(X, index=cal.index)


def fe_dummies(cal: pd.DataFrame, extra: str | None = None) -> pd.DataFrame:
    """Group-level half-hour-of-day x weekend dummies, optionally crossed with ``extra``
    ('month': calendar month of the sample, 'moy': month of year) in 4-hour blocks."""
    key = cal["hh"].astype(str) + "_" + cal["weekend"].astype(int).astype(str)
    D = pd.get_dummies(key, prefix="fe", dtype=float)
    if extra:
        per = cal["ts_local"].dt.to_period("M").astype(str) if extra == "month" else cal["ts_local"].dt.month
        k2 = per.astype(str) + "_b" + (cal["hh"] // 8).astype(str)
        D2 = pd.get_dummies(k2, prefix=extra, dtype=float)
        # Within each block the period dummies sum to that block's half-hour dummies: drop one period.
        ref = f"{extra}_{per.iloc[0]}_b"
        D = pd.concat([D, D2.loc[:, ~D2.columns.str.startswith(ref)]], axis=1)
    return D


# ---------------------------------------------------------------- sample and data pass
def coverage(cal: pd.DataFrame, panel: Path = PANEL_PATH) -> pd.DataFrame:
    """Share of calendar half-hours with a non-null reading, per household and period."""
    con = connect()
    con.register("cal_df", cal[["ts_label", "period"]])
    out = con.execute(f"""
        SELECT m.hh_id, any_value(m.tou) AS tou, c.period, count(*) AS n_obs
        FROM read_parquet('{panel.as_posix()}') m JOIN cal_df c USING (ts_label)
        WHERE m.kwh IS NOT NULL GROUP BY m.hh_id, c.period""").df()
    wide = out.pivot_table(index=["hh_id", "tou"], columns="period", values="n_obs", fill_value=0)
    expected = cal["period"].value_counts()
    for p in expected.index:
        wide[p] = wide.get(p, 0) / expected[p]
    return wide.reset_index().rename_axis(columns=None)


def select_households(cov: pd.DataFrame, min_pre: float = 0.8, min_trial: float = 0.9) -> pd.DataFrame:
    """Households with at least ``min_pre`` of pre-trial and ``min_trial`` of trial half-hours."""
    keep = (cov["pre"] >= min_pre) & (cov["trial"] >= min_trial)
    s = cov.loc[keep, ["hh_id", "tou"]].sort_values(["tou", "hh_id"]).reset_index(drop=True)
    s["h_idx"] = s.groupby("tou").cumcount()
    return s


@dataclass
class GroupSums:
    """Weighted sums per draw (row 0 = full sample, rows 1.. = bootstrap) and half-hour."""
    dev: dict[bool, np.ndarray]   # sum of w * dev
    kwh: dict[bool, np.ndarray]   # sum of w * kwh
    n: dict[bool, np.ndarray]     # sum of w over households with a reading
    n_readings: int = 0           # readings used (full sample)

    def mean(self, field: str, tou: bool) -> np.ndarray:
        num = getattr(self, field)[tou].astype(np.float64)
        n = self.n[tou].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, num / n, np.nan)

    def delta(self) -> np.ndarray:
        return self.mean("dev", True) - self.mean("dev", False)

    def select(self, rows) -> "GroupSums":
        """The same sums for a subset of weight rows (row 0 of the result is the estimate)."""
        pick = lambda d: {g: v[rows] for g, v in d.items()}  # noqa: E731
        return GroupSums(pick(self.dev), pick(self.kwh), pick(self.n), self.n_readings)


def bootstrap_weights(sample: pd.DataFrame, n_boot: int, seed: int) -> dict[bool, np.ndarray]:
    """Row 0 all ones; rows 1..n_boot multinomial resampling counts within each group."""
    rng = np.random.default_rng(seed)
    W = {}
    for tou in (False, True):
        h = int((sample["tou"] == tou).sum())
        draws = rng.multinomial(h, np.full(h, 1.0 / h), size=n_boot)
        W[tou] = np.vstack([np.ones((1, h)), draws]).astype(np.float32)
    return W


def group_sums(cal: pd.DataFrame, sample: pd.DataFrame, weights: dict[bool, np.ndarray],
               panel: Path = PANEL_PATH, chunk: int = 1488, dtype=np.float64) -> GroupSums:
    """Weighted group sums of load and demeaned load per half-hour, for every weight row.

    ``dev`` is kwh minus the household's half-hour-of-day x weekend mean over the calendar
    window. The panel is read one time chunk at a time (nothing is sorted or materialised),
    each chunk becomes dense household x half-hour blocks, and the weight matrices multiply
    them. Assumes one reading per household and label, which
    :func:`ttrl.data.lcl_meters.build_panel` reports (0 conflicting duplicates in the release).
    The products are float32 either way; ``dtype=np.float32`` halves the memory of the result
    without changing its values.
    """
    con = connect()
    con.register("cal_df", cal[["ts_label", "t_idx", "hh", "weekend"]])
    con.register("smp_df", sample[["hh_id", "tou", "h_idx"]])
    src = f"read_parquet('{panel.as_posix()}') m JOIN smp_df s USING (hh_id) JOIN cal_df c USING (ts_label)"
    con.execute(f"""CREATE TEMP TABLE mu AS SELECT m.hh_id, c.hh, c.weekend, avg(m.kwh) AS mu
                    FROM {src} WHERE m.kwh IS NOT NULL GROUP BY ALL""")
    n_t = len(cal)
    labels = cal["ts_label"].to_numpy()
    n_draw = next(iter(weights.values())).shape[0]
    res = {f: {g: np.zeros((n_draw, n_t), dtype) for g in (False, True)} for f in ("dev", "kwh", "n")}
    n_readings = 0
    for t0 in range(0, n_t, chunk):
        t1 = min(n_t, t0 + chunk)
        lo, hi = pd.Timestamp(labels[t0]), pd.Timestamp(labels[t1 - 1])
        df = con.execute(f"""
            SELECT s.tou, s.h_idx, c.t_idx, m.kwh, m.kwh - mu.mu AS dev
            FROM {src} JOIN mu ON mu.hh_id = m.hh_id AND mu.hh = c.hh AND mu.weekend = c.weekend
            WHERE m.kwh IS NOT NULL AND m.ts_label BETWEEN TIMESTAMP '{lo}' AND TIMESTAMP '{hi}'""").df()
        n_readings += len(df)
        for g in (False, True):
            d = df[df["tou"] == g]
            h = weights[g].shape[1]
            rows, cols = d["h_idx"].to_numpy(), d["t_idx"].to_numpy() - t0
            for field, vals in (("dev", d["dev"].to_numpy(np.float32)), ("kwh", d["kwh"].to_numpy(np.float32)),
                                ("n", np.ones(len(d), np.float32))):
                M = np.zeros((h, t1 - t0), np.float32)
                M[rows, cols] = vals
                res[field][g][:, t0:t1] = weights[g] @ M
    return GroupSums(res["dev"], res["kwh"], res["n"], n_readings)


# ---------------------------------------------------------------- estimation
@dataclass
class DidResult:
    table: pd.DataFrame          # one row per treatment regressor
    boot: pd.DataFrame           # bootstrap draws of the coefficients (kWh per household-half-hour)
    n_obs: int
    fe: str


def estimate(cal: pd.DataFrame, sums: GroupSums, X: pd.DataFrame, fe: str | None = None) -> DidResult:
    """OLS of Delta_t on treatment regressors and FE dummies, for the full sample and each
    bootstrap draw. Percent effects are relative to the dToU counterfactual on the
    regressor's own half-hours (dToU mean load there minus the effect)."""
    delta = sums.delta()
    ok = np.isfinite(delta[0])
    F = fe_dummies(cal, fe)
    Z = pd.concat([X, F], axis=1).loc[ok]
    Z = Z.loc[:, Z.abs().sum() > 0]
    names = [c for c in X.columns if c in Z.columns]
    rank = np.linalg.matrix_rank(Z.to_numpy())
    if rank < Z.shape[1]:
        raise ValueError(f"design is rank deficient ({rank} < {Z.shape[1]}); drop terms collinear with the FE")
    y = delta[0, ok]

    fit = sm.OLS(y, Z.to_numpy()).fit(cov_type="cluster",
                                      cov_kwds={"groups": cal.loc[ok, "date"].factorize()[0]})
    coef = pd.Series(fit.params, index=Z.columns)[names]
    se_cl = pd.Series(fit.bse, index=Z.columns)[names]

    P = np.linalg.pinv(Z.to_numpy())
    B = delta[1:, ok] @ P.T
    boot = pd.DataFrame(B, columns=Z.columns)[names]
    if boot.isna().any().any():
        raise ValueError("bootstrap draw with missing half-hours")

    tou_mean = sums.mean("kwh", True)
    rows = []
    for c in names:
        mask = Z[c].to_numpy() == 1
        if c == "trial":  # Normal 2013 half-hours outside all event windows
            others = [k for k in names if k not in ("trial", "post")]
            mask = mask & (Z[others].to_numpy().sum(1) == 0)
        level = tou_mean[0, ok][mask].mean()
        level_b = np.nanmean(tou_mean[1:, ok][:, mask], axis=1)
        pct_b = 100 * boot[c].to_numpy() / (level_b - boot[c].to_numpy())
        cf = level - coef[c]
        rows.append({
            "term": c, "n_hh": int(mask.sum()), "coef_kwh": coef[c],
            "se_cluster_date": se_cl[c], "se_boot_household": boot[c].std(ddof=1),
            "tou_mean_kwh": level, "pct": 100 * coef[c] / cf,
            "pct_lo_cluster": 100 * (coef[c] - 1.96 * se_cl[c]) / cf,
            "pct_hi_cluster": 100 * (coef[c] + 1.96 * se_cl[c]) / cf,
            "pct_lo_boot": np.percentile(pct_b, 2.5), "pct_hi_boot": np.percentile(pct_b, 97.5),
        })
    tab = pd.DataFrame(rows)
    tab["se_combined"] = np.hypot(tab["se_cluster_date"], tab["se_boot_household"])
    cf = tab["tou_mean_kwh"] - tab["coef_kwh"]
    tab["pct_lo"] = 100 * (tab["coef_kwh"] - 1.96 * tab["se_combined"]) / cf
    tab["pct_hi"] = 100 * (tab["coef_kwh"] + 1.96 * tab["se_combined"]) / cf
    return DidResult(tab, boot, int(ok.sum()), fe or "hh x weekend")
