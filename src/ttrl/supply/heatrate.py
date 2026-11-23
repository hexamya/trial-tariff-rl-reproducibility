"""Ambient-temperature dependence of heat input and available capacity.

Stage S4, ERCOT gas fleet, EPA CEMS. Two per-unit models, both referenced to
``T_REF`` = 15 C (ISO ambient):

Heat input (MMBtu/h) at gross output P (MW) and ambient temperature T (C)::

    H(P, T) = a + b P + c P^2 + d (T - 15) + e (T - 15) P

- ``a, b, c`` are the unit's curve at 15 C (same form as
  :mod:`ttrl.supply.units`); ``d`` shifts the whole curve (MMBtu/h per C) and
  ``e`` tilts it (change of the incremental heat rate, MMBtu/MWh per C).
  So the heat-rate change at equal output is (d + e P) / H(P, 15) per C, and
  the change of the incremental heat rate is e per C.
- Why this form: it is the static curve of the dispatch step with its
  intercept and slope made linear in temperature. It stays a convex quadratic
  in P at every temperature, so a temperature-aware model plugs into
  :func:`ttrl.supply.dispatch.economic_dispatch` unchanged, and it needs only
  two extra parameters per unit. A quadratic in T was not used; residuals by
  temperature bin are checked instead (see the script's report).
- Fit: Huber robust regression on *steady* full hours (:func:`steady_rows`:
  the unit also ran full hours in the two previous and the next hour and its
  output changed by at most ``ramp_frac`` of its reference output since the
  previous hour), so start-up and ramping fuel do not load on temperature.
  The unit's capability-limited months (:func:`limited_months`, the same
  screen as for derating) are left out as well: a block running with part
  of its equipment out (e.g. a CC with its steam turbine out from May to
  October) is a different operating mode, and because such outages are
  seasonal their heat rate would otherwise load on temperature.
  If the quadratic term is concave or the incremental heat rate is not
  positive at Pmin, P^2 is dropped (as in the dispatch step). Standard errors
  of ``d, e`` are cluster-robust by local day (weighted least squares with
  the Huber weights).

Available capacity (MW) at temperature T::

    Pavail(T) = P15 * (1 - kappa / 100 * (T - 15))   for T >= 15
    Pavail(T) = P15                                  for T < 15

- ``kappa`` (% of 15 C capacity per C) and ``P15`` come from the upper
  envelope of output (:func:`envelope_bins`, :func:`fit_derating`): the
  ``env_quantile`` (p99) of clean full-hour gross output in 2 C bins of
  temperature, bins with at least ``env_min_hours`` hours, and an unweighted
  straight line through the bins whose centre is at least ``env_t_min``.
  Before the fit, months whose output never comes near the unit's maximum
  are dropped (:func:`limited_months`: outages or caps that temperature
  cannot explain), and the envelope is made non-increasing in temperature
  (``env_monotone``, see :func:`fit_derating`). :func:`unit_derating` also
  returns the raw, >= 21 C and high-dispatch-hour variants.
- Below 15 C the capacity is held at P15: the cold side is not estimated
  (units are rarely pushed to their limit in mild winter hours, and gas
  turbines often hit generator or other limits in the cold), so the model is
  conservative there.
- Selection bias: in mild hours a unit may not run at its limit because
  demand is low, which pulls the mild-bin envelope down and biases kappa
  toward zero. The script checks this with the same estimator on
  high-dispatch hours only and against EIA-860 summer/winter ratings.

Temperature is the hour-beginning mean of the plant's hourly observations
(same definition as :func:`ttrl.data.weather.hour_beginning_mean`, see
:func:`add_temperature`), so it covers the CEMS hour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ttrl.supply.units import (LST_OFFSET, UNIT_KEY, CurveArrays, FitConfig,
                               allocated_capacity, fit_heat_curve, fit_rows,
                               make_unit_id)

T_REF = 15.0
KG_PER_SHORT_TON = 907.18474
GROUPS = ["CT", "CC", "CC*", "ST"]   # CC* = suspect CC (CT output only); CHP excluded


@dataclass(frozen=True)
class TempConfig:
    """Settings of the temperature fits (see module docstring)."""
    t_ref: float = T_REF
    ramp_frac: float = 0.10          # steady hours: |dP| <= ramp_frac * Pref
    min_steady_hours: int = 200      # own heat-input temperature fit
    min_temp_span: float = 15.0      # P95 - P5 of temperature over fit hours, C
    bin_width: float = 2.0           # envelope bins, C
    env_quantile: float = 0.99
    env_min_hours: int = 30
    env_min_bins: int = 4
    env_min_span: float = 10.0       # between the outer bin centres used, C
    env_t_min: float = 15.0          # lowest bin centre in the slope fit, C
    env_monotone: bool = True        # capability may not rise with T above t_min
    limited_month_ratio: float = 0.7  # month p99 below this x unit p99: capability-limited month
    hi_quantile: float = 0.95        # envelope in high-dispatch hours (fewer hours per bin)
    hi_min_hours: int = 20
    hi_min_bins: int = 3
    hi_min_span: float = 6.0
    load_fraction: float = 0.8       # common load fraction for heat-rate results
    constant_heat_share: float = 0.9  # modal heat value share flagging substituted data
    huber_t: float = 1.345


# --------------------------------------------------------------------------
# Unit-hour preparation
# --------------------------------------------------------------------------
def unit_groups(table: pd.DataFrame) -> pd.Series:
    """Analysis group: CHP, CC* (suspect CC), else the technology."""
    g = np.where(table["chp"], "CHP", np.where(table["suspect_cc"], "CC*", table["tech"]))
    return pd.Series(g, index=table.index)


def add_temperature(unit_hours: pd.DataFrame, units: pd.DataFrame,
                    temp: pd.DataFrame) -> pd.DataFrame:
    """Unit hours with ``temp_c``: the plant's hour-beginning mean temperature.

    ``unit_hours`` comes from :func:`ttrl.supply.units.prepare_unit_hours`,
    ``units`` is the CEMS-unit table (gives ``plant_id_eia``) and ``temp``
    is ``data/interim/plant_temperature_<year>.parquet`` (``plant_id_eia,
    ts_utc, temp_c``). The temperature of the CEMS hour starting at ts is
    the mean of the observations at ts and ts + 1 h. Rows without a plant
    temperature get NaN.
    """
    t = temp[["plant_id_eia", "ts_utc", "temp_c"]].sort_values(["plant_id_eia", "ts_utc"])
    nxt = t.groupby("plant_id_eia", sort=False)["temp_c"].shift(-1)
    gap = t.groupby("plant_id_eia", sort=False)["ts_utc"].shift(-1) - t["ts_utc"]
    t = t.assign(temp_hb=((t["temp_c"] + nxt) / 2).where(gap == pd.Timedelta(hours=1)))
    t = t[["plant_id_eia", "ts_utc", "temp_hb"]].astype({"plant_id_eia": "int64"})
    link = units[UNIT_KEY + ["plant_id_eia"]].copy()
    link["unit_id"] = make_unit_id(link["plant_id_epa"], link["emissions_unit_id_epa"])
    link = link[["unit_id", "plant_id_eia"]].astype({"plant_id_eia": "int64"})
    out = unit_hours.drop(columns=[c for c in ("plant_id_eia", "temp_c") if c in unit_hours])
    out = out.merge(link, on="unit_id", how="left")
    out = out.merge(t, left_on=["plant_id_eia", "operating_datetime_utc"],
                    right_on=["plant_id_eia", "ts_utc"], how="left")
    return out.drop(columns="ts_utc").rename(columns={"temp_hb": "temp_c"})


def steady_rows(unit_hours: pd.DataFrame, cfg: TempConfig = TempConfig(),
                fit_cfg: FitConfig = FitConfig()) -> pd.Series:
    """Fit rows (:func:`ttrl.supply.units.fit_rows`) in steady operation.

    A row is steady when the unit also ran full operating hours in the two
    previous and in the next clock hour, and its gross output changed by at
    most ``cfg.ramp_frac`` of its reference output (99th percentile of its
    fit-row output in ``unit_hours``) since the previous hour. Returns a
    boolean Series aligned with ``unit_hours``.
    """
    d = unit_hours[["unit_id", "operating_datetime_utc", "operating_time_hours",
                    "gross_load_mw"]].copy()
    ok = fit_rows(unit_hours, fit_cfg)
    pref = unit_hours.loc[ok].groupby("unit_id")["gross_load_mw"].quantile(0.99)
    d = d.sort_values(["unit_id", "operating_datetime_utc"])
    g = d.groupby("unit_id", sort=False)
    hour = pd.Timedelta(hours=1)
    full = d["operating_time_hours"] >= 1

    def lag(k: int) -> pd.Series:
        """Full hour exactly k clock hours away (k < 0: earlier)."""
        same = (d["operating_datetime_utc"] - g["operating_datetime_utc"].shift(-k)) == -k * hour
        return same & full.groupby(d["unit_id"], sort=False).shift(-k, fill_value=False)

    dp = (d["gross_load_mw"] - g["gross_load_mw"].shift(1)).abs()
    ramp_ok = dp <= cfg.ramp_frac * d["unit_id"].map(pref)
    steady = lag(-1) & lag(-2) & lag(1) & ramp_ok
    return (steady.reindex(unit_hours.index).fillna(False).astype(bool) & ok)


def local_day(ts_utc: pd.Series) -> np.ndarray:
    """Local-standard-time calendar day (int64 days since epoch) of UTC stamps."""
    return ((ts_utc - LST_OFFSET).dt.floor("D").astype("int64") // 86_400_000_000).to_numpy()


# --------------------------------------------------------------------------
# Heat-input model
# --------------------------------------------------------------------------
def _design(x: np.ndarray, z: np.ndarray, quad: bool) -> np.ndarray:
    cols = [np.ones_like(x), x] + ([x * x] if quad else []) + [z, z * x]
    return np.column_stack(cols)


def fit_temp_heat_curve(p: np.ndarray, h: np.ndarray, t: np.ndarray, pmin: float,
                        pmax: float, day: np.ndarray | None = None,
                        cfg: TempConfig = TempConfig()) -> dict:
    """Robust fit of H = a + bP + cP^2 + d(T - Tref) + e(T - Tref)P for one unit.

    Returns ``a, b, c, d, e`` in natural units (MMBtu/h, MW, C), ``method``
    ("quadratic" or "linear" when the P^2 term is dropped), in-sample ``r2``
    and ``rmse`` (MMBtu/h), ``n`` and, when ``day`` labels are given,
    cluster-robust standard errors ``se_d, se_e`` and covariance ``cov_de``.
    P is scaled by ``pmax`` and T - Tref by 10 C inside the regression.
    """
    p, h, t = (np.asarray(v, float) for v in (p, h, t))
    s = max(float(pmax), 1.0)
    x, z = p / s, (t - cfg.t_ref) / 10.0
    norm = sm.robust.norms.HuberT(t=cfg.huber_t)

    def run(quad: bool):
        xm = _design(x, z, quad)
        return xm, sm.RLM(h, xm, M=norm).fit()

    xm, res = run(True)
    beta = res.params
    a, b, c = beta[0], beta[1] / s, beta[2] / s ** 2
    method = "quadratic"
    if c < 0 or b + 2 * c * pmin <= 0:
        xm, res = run(False)
        beta = np.insert(res.params, 2, 0.0)
        a, b, c = beta[0], beta[1] / s, 0.0
        method = "linear"
    d, e = beta[3] / 10.0, beta[4] / (10.0 * s)
    fitted = a + b * p + c * p * p + (t - cfg.t_ref) * (d + e * p)
    resid = h - fitted
    ss_tot = float(((h - h.mean()) ** 2).sum())
    out = {"a": float(a), "b": float(b), "c": float(c), "d": float(d), "e": float(e),
           "method": method, "n": len(p), "rmse": float(np.sqrt((resid ** 2).mean())),
           "r2": 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan,
           "se_d": np.nan, "se_e": np.nan, "cov_de": np.nan}
    if day is not None and len(np.unique(day)) > 10:
        w = np.asarray(res.weights, float)
        wls = sm.WLS(h, xm, weights=w).fit(cov_type="cluster",
                                           cov_kwds={"groups": np.asarray(day)})
        k = xm.shape[1]
        cov = np.asarray(wls.cov_params())[k - 2:, k - 2:]
        scale = np.array([1 / 10.0, 1 / (10.0 * s)])
        cov = cov * np.outer(scale, scale)
        out.update(se_d=float(np.sqrt(cov[0, 0])), se_e=float(np.sqrt(cov[1, 1])),
                   cov_de=float(cov[0, 1]))
    return out


def heat_input(params: pd.DataFrame | dict, p, t, t_ref: float = T_REF):
    """H(P, T) in MMBtu/h from columns/keys ``a, b, c, d, e`` (broadcasts)."""
    a, b, c, d, e = (np.asarray(params[k], float) for k in "abcde")
    p, t = np.asarray(p, float), np.asarray(t, float)
    return a + b * p + c * p * p + (t - t_ref) * (d + e * p)


def hr_change_per_c(params, p, t_ref: float = T_REF):
    """Heat-rate change at equal output P, % of the 15 C heat rate per C."""
    d, e = np.asarray(params["d"], float), np.asarray(params["e"], float)
    return 100.0 * (d + e * np.asarray(p, float)) / heat_input(params, p, t_ref, t_ref)


def within_day_temp_effect(p: np.ndarray, h: np.ndarray, t: np.ndarray, day: np.ndarray,
                           pref: float, t_ref: float = T_REF) -> tuple[float, float]:
    """``d, e`` estimated from within-day variation only (robustness check).

    OLS of H on [P, P^2, (T - Tref), (T - Tref) P] after removing each local
    day's mean from every variable (day fixed effects), so slow seasonal
    changes (degradation, washing, maintenance) cannot load on temperature.
    """
    x = np.asarray(p, float) / pref
    z = (np.asarray(t, float) - t_ref) / 10.0
    m = np.column_stack([x, x * x, z, z * x, np.asarray(h, float)])
    df = pd.DataFrame(m)
    dm = (df - df.groupby(np.asarray(day)).transform("mean")).to_numpy()
    beta = np.linalg.lstsq(dm[:, :4], dm[:, 4], rcond=None)[0]
    return float(beta[2] / 10.0), float(beta[3] / (10.0 * pref))


# --------------------------------------------------------------------------
# Derating
# --------------------------------------------------------------------------
def envelope_bins(p: np.ndarray, t: np.ndarray, cfg: TempConfig = TempConfig(),
                  quantile: float | None = None) -> pd.DataFrame:
    """Upper envelope of output by temperature bin.

    Columns ``t_mid`` (bin centre, C), ``n`` (hours) and ``p_env`` (the
    ``quantile`` of output, default ``cfg.env_quantile``). Bins are
    ``cfg.bin_width`` wide with edges at multiples of the width.
    """
    q = cfg.env_quantile if quantile is None else quantile
    t = np.asarray(t, float)
    w = cfg.bin_width
    mid = np.floor(t / w) * w + w / 2
    df = pd.DataFrame({"t_mid": mid, "p": np.asarray(p, float)}).dropna()
    g = df.groupby("t_mid")["p"]
    return pd.DataFrame({"n": g.size(), "p_env": g.quantile(q)}).reset_index()


def fit_derating(bins: pd.DataFrame, cfg: TempConfig = TempConfig(),
                 min_hours: int | None = None, t_min: float | None = None,
                 min_bins: int | None = None, min_span: float | None = None,
                 monotone: bool | None = None) -> dict:
    """Straight line through envelope bins: P15 and kappa (% of P15 per C).

    Uses bins with at least ``min_hours`` hours and centre >= ``t_min``.
    Needs ``min_bins`` such bins spanning ``min_span`` C, else returns NaNs.
    With ``monotone`` (default ``cfg.env_monotone``) each bin's envelope is
    first raised to the highest envelope of any hotter bin used: a unit that
    produced X MW at a hotter temperature could produce X MW at a cooler
    one, so a lower mild-bin value shows that the unit was not pushed to its
    limit there (the selection bias described in the module docstring).
    ``kappa_se`` is the classical standard error from the bin residuals
    (delta method for the ratio); it ignores the sampling error of each bin's
    quantile.
    """
    min_hours = cfg.env_min_hours if min_hours is None else min_hours
    t_min = cfg.env_t_min if t_min is None else t_min
    min_bins = cfg.env_min_bins if min_bins is None else min_bins
    min_span = cfg.env_min_span if min_span is None else min_span
    monotone = cfg.env_monotone if monotone is None else monotone
    b = bins[(bins["n"] >= min_hours) & (bins["t_mid"] >= t_min)].sort_values("t_mid")
    if monotone and len(b):
        b = b.assign(p_env=np.maximum.accumulate(b["p_env"].to_numpy()[::-1])[::-1])
    out = {"p15": np.nan, "kappa": np.nan, "kappa_se": np.nan, "n_bins": len(b),
           "t_lo": b["t_mid"].min() if len(b) else np.nan,
           "t_hi": b["t_mid"].max() if len(b) else np.nan}
    if len(b) < min_bins or b["t_mid"].max() - b["t_mid"].min() < min_span:
        return out
    x = b["t_mid"].to_numpy() - cfg.t_ref
    y = b["p_env"].to_numpy()
    xm = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(xm, y, rcond=None)
    p15, slope = beta
    kappa = -100.0 * slope / p15
    se = np.nan
    if len(b) > 2:
        resid = y - xm @ beta
        cov = (resid @ resid / (len(b) - 2)) * np.linalg.inv(xm.T @ xm)
        grad = np.array([100.0 * slope / p15 ** 2, -100.0 / p15])
        se = float(np.sqrt(grad @ cov @ grad))
    out.update(p15=float(p15), kappa=float(kappa), kappa_se=se)
    return out


def limited_months(month: np.ndarray, p: np.ndarray, ratio: float) -> list[int]:
    """Months whose 99th-percentile output is below ``ratio`` x the unit's.

    Such months show a capability limit that temperature cannot explain
    (e.g. one gas turbine of a block out of service for months); their
    hours are left out of the envelope.
    """
    df = pd.DataFrame({"m": np.asarray(month), "p": np.asarray(p, float)})
    top = df["p"].quantile(0.99)
    mq = df.groupby("m")["p"].quantile(0.99)
    return sorted(int(m) for m in mq.index[mq < ratio * top])


def in_limited_month(unit_hours: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """Rows of ``unit_hours`` in one of their unit's ``limited_months``.

    ``table`` is :func:`fit_temperature_curves` output, whose
    ``limited_months`` column lists each unit's capability-limited months as
    a comma-separated string. Returns a boolean Series aligned with
    ``unit_hours`` (False for units missing from ``table``).
    """
    pairs = [(u, int(m)) for u, s in zip(table["unit_id"], table["limited_months"].fillna(""))
             for m in str(s).split(",") if m]
    if not pairs:
        return pd.Series(False, index=unit_hours.index)
    key = pd.MultiIndex.from_arrays([unit_hours["unit_id"], unit_hours["month"].astype(int)])
    return pd.Series(key.isin(pd.MultiIndex.from_tuples(pairs)), index=unit_hours.index)


def unit_derating(p: np.ndarray, t: np.ndarray, month: np.ndarray,
                  high: np.ndarray | None = None, cfg: TempConfig = TempConfig()) -> dict:
    """Derating estimates for one unit from its clean full hours.

    Drops :func:`limited_months`, then fits (keys prefixed accordingly):

    - primary (no prefix): all hours, ``cfg.env_quantile`` envelope,
      monotone, bins from ``cfg.env_t_min``;
    - ``raw_``: as primary without the monotone step;
    - ``b21_``: as primary on bins from 21 C (same range as ``hi_``);
    - ``hi_`` (if ``high`` is given): only high-dispatch hours, the
      ``cfg.hi_quantile`` envelope, monotone, looser bin requirements.

    Also returns ``months_dropped`` (count).
    """
    p, t, month = np.asarray(p, float), np.asarray(t, float), np.asarray(month)
    drop = limited_months(month, p, cfg.limited_month_ratio)
    keep = ~np.isin(month, drop)
    bins = envelope_bins(p[keep], t[keep], cfg)
    out = {"months_dropped": len(drop)}
    out.update(fit_derating(bins, cfg))
    out.update({f"raw_{k}": v for k, v in fit_derating(bins, cfg, monotone=False).items()})
    out.update({f"b21_{k}": v for k, v in fit_derating(bins, cfg, t_min=21.0).items()})
    if high is not None:
        sel = keep & np.asarray(high, bool)
        hb = envelope_bins(p[sel], t[sel], cfg, quantile=cfg.hi_quantile)
        hi = fit_derating(hb, cfg, min_hours=cfg.hi_min_hours, min_bins=cfg.hi_min_bins,
                          min_span=cfg.hi_min_span)
        out.update({f"hi_{k}": v for k, v in hi.items()})
    return out


def derate_factor(kappa, t, t_ref: float = T_REF):
    """Available-capacity multiplier 1 - kappa/100 * max(T - Tref, 0)."""
    return 1.0 - np.asarray(kappa, float) / 100.0 * np.maximum(np.asarray(t, float) - t_ref, 0.0)


# --------------------------------------------------------------------------
# All units
# --------------------------------------------------------------------------
def fit_temperature_curves(unit_hours: pd.DataFrame, units: pd.DataFrame,
                           gens: pd.DataFrame, cfg: TempConfig = TempConfig(),
                           fit_cfg: FitConfig = FitConfig(),
                           group_of: pd.Series | None = None) -> pd.DataFrame:
    """Per-unit temperature-dependent heat-input curve and derating.

    ``unit_hours`` is :func:`add_temperature` output restricted to the fit
    window, with a boolean ``steady`` column (:func:`steady_rows`) and,
    optionally, a boolean ``high_dispatch`` column (hours of high fleet
    output, for the ``hi_`` derating variant of :func:`unit_derating`).
    ``group_of`` maps unit_id to the analysis group (default: computed from
    the static flags of this window, see :func:`unit_groups`).

    Returns one row per unit with: identifiers, ``tech, chp, suspect_cc,
    group``; EIA-860 ``nameplate_mw, summer_mw, winter_mw``; static curve of
    the same window (``a0, b0, c0`` from :func:`ttrl.supply.units.fit_heat_curve`
    on the same steady hours, ``pmin, pmax`` = 5th/99th percentile of fit-row
    output); the temperature curve ``a, b, c, d, e`` with ``hr_method`` (own
    fit "quadratic"/"linear", "group" = group median temperature terms on the
    unit's static curve, "none"), fit statistics and cluster-robust SEs;
    the within-day robustness estimate ``d_wd, e_wd``; the envelope fit
    ``p15, kappa, kappa_se, n_bins, t_lo, t_hi`` with ``kappa_method``
    ("own"/"group"/"none") and the variants ``raw_*, b21_*, hi_*`` and
    ``months_dropped`` of :func:`unit_derating`; ``limited_months`` (those
    months, comma-separated; they are left out of the heat fit too, so
    ``n_steady`` counts steady hours outside them); ``const_heat``
    (substituted heat input); the temperature range of the fit-row hours
    ``t_p05, t_p95`` and, outside the limited months, ``t_fit_lo, t_fit_hi``
    (min, max); heat-rate results at the common load fraction ``cfg.load_fraction`` x pmax: ``hr15_lf`` (MMBtu/MWh at 15 C),
    ``hr_pct_per_c`` (own or group terms), ``hr_pct_per_c_wd`` (within-day
    terms) and ``lf_in_range`` (that load lies within [pmin, pmax]);
    ``hr_full_pct_per_c``: change of the heat rate at full available load
    between 15 C and 35 C, % per C (derating and heat terms together).
    """
    ok = fit_rows(unit_hours, fit_cfg) & unit_hours["temp_c"].notna()
    clean = unit_hours.loc[ok]
    meta = units[UNIT_KEY + ["plant_id_eia", "plant_name_eia", "tech", "chp"]].copy()
    meta["unit_id"] = make_unit_id(meta["plant_id_epa"], meta["emissions_unit_id_epa"])
    meta["chp"] = meta["chp"].fillna(False).astype(bool)
    meta = meta.merge(allocated_capacity(units, gens), on=UNIT_KEY, how="left")
    meta = meta[meta["unit_id"].isin(clean["unit_id"].unique())]

    rows = []
    for uid, d in clean.groupby("unit_id"):
        p_all = d["gross_load_mw"].to_numpy()
        pmin = float(np.quantile(p_all, fit_cfg.pmin_quantile))
        pmax = float(np.quantile(p_all, fit_cfg.pmax_quantile))
        heat_round = d["heat"].round(1)
        row = {"unit_id": uid, "n_clean": len(d), "pmin": pmin, "pmax": pmax,
               "median_hr": float(np.median(d["heat"] / d["gross_load_mw"])),
               "const_heat": bool(heat_round.value_counts(normalize=True).iloc[0]
                                  >= cfg.constant_heat_share),
               "t_p05": float(d["temp_c"].quantile(0.05)),
               "t_p95": float(d["temp_c"].quantile(0.95))}
        high = d["high_dispatch"].to_numpy() if "high_dispatch" in d else None
        row.update(unit_derating(p_all, d["temp_c"].to_numpy(), d["month"].to_numpy(),
                                 high, cfg))
        # Capability-limited months are a different operating mode: left out
        # of the heat fit as well as the envelope (see module docstring), and
        # the heat terms' temperature range is that of the remaining hours.
        drop = limited_months(d["month"].to_numpy(), p_all, cfg.limited_month_ratio)
        row["limited_months"] = ",".join(str(m) for m in drop)
        t_keep = d.loc[~d["month"].isin(drop), "temp_c"]
        row.update(t_fit_lo=float(t_keep.min()), t_fit_hi=float(t_keep.max()))
        st = d[d["steady"] & ~d["month"].isin(drop)]
        row["n_steady"] = len(st)
        span = (st["temp_c"].quantile(0.95) - st["temp_c"].quantile(0.05)) if len(st) else 0.0
        if len(st) >= cfg.min_steady_hours and pmax > pmin and span >= cfg.min_temp_span:
            p, h, t = (st[c].to_numpy() for c in ("gross_load_mw", "heat", "temp_c"))
            day = local_day(st["operating_datetime_utc"])
            base = fit_heat_curve(p, h, pmin, pmax, cfg.huber_t)
            fit = fit_temp_heat_curve(p, h, t, pmin, pmax, day, cfg)
            d_wd, e_wd = within_day_temp_effect(p, h, t, day, pmax, cfg.t_ref)
            row.update(a0=base["a"], b0=base["b"], c0=base["c"], rmse0=base["rmse"],
                       a=fit["a"], b=fit["b"], c=fit["c"], d=fit["d"], e=fit["e"],
                       hr_method=fit["method"], r2=fit["r2"], rmse=fit["rmse"],
                       se_d=fit["se_d"], se_e=fit["se_e"], cov_de=fit["cov_de"],
                       d_wd=d_wd, e_wd=e_wd)
        elif len(d) >= fit_cfg.min_hours and pmax > pmin:
            base = fit_heat_curve(p_all, d["heat"].to_numpy(), pmin, pmax, cfg.huber_t)
            row.update(a0=base["a"], b0=base["b"], c0=base["c"], rmse0=base["rmse"])
        rows.append(row)
    fits = pd.DataFrame(rows)
    table = meta.merge(fits, on="unit_id", how="inner")
    table["suspect_cc"] = ((table["tech"] == "CC") & ~table["chp"]
                           & (table["median_hr"] >= fit_cfg.suspect_cc_heat_rate))
    table["group"] = (unit_groups(table) if group_of is None
                      else table["unit_id"].map(group_of))
    for col in ("a", "b", "c", "d", "e", "a0", "b0", "c0", "rmse0", "r2", "rmse",
                "se_d", "se_e", "cov_de", "d_wd", "e_wd", "t_fit_lo", "t_fit_hi"):
        if col not in table:
            table[col] = np.nan
    if "hr_method" not in table:
        table["hr_method"] = None
    table["hr_method"] = table["hr_method"].fillna("none")
    # Substituted (constant) heat input carries no information on temperature.
    bad = table["const_heat"]
    table.loc[bad, ["d", "e", "se_d", "se_e", "cov_de", "d_wd", "e_wd"]] = np.nan
    table.loc[bad, "hr_method"] = "none"

    # Group fallback for heat terms: median d/pmax and median e of own fits.
    own = table["hr_method"].isin(["quadratic", "linear"])
    med = (table[own].assign(d_rel=lambda x: x["d"] / x["pmax"])
           .groupby("group")[["d_rel", "e"]].median())
    need = ~own & table["a0"].notna() & table["group"].isin(med.index) & ~bad
    for i in table.index[need]:
        g = table.at[i, "group"]
        table.loc[i, ["a", "b", "c"]] = table.loc[i, ["a0", "b0", "c0"]].to_numpy(float)
        table.loc[i, "d"] = med.at[g, "d_rel"] * table.at[i, "pmax"]
        table.loc[i, "e"] = med.at[g, "e"]
        table.loc[i, "hr_method"] = "group"
    # Constant-heat and unfitted units keep their static curve, no temperature terms.
    rest = table["a"].isna() & table["a0"].notna()
    table.loc[rest, ["a", "b", "c"]] = table.loc[rest, ["a0", "b0", "c0"]].to_numpy(float)
    table.loc[rest | bad, ["d", "e"]] = 0.0

    # Group fallback for derating: median kappa of own fits; P15 = pmax.
    table["kappa_method"] = np.where(table["kappa"].notna(), "own", "group")
    kmed = table[table["kappa_method"] == "own"].groupby("group")["kappa"].median()
    fb = table["kappa_method"] == "group"
    table.loc[fb, "kappa"] = table.loc[fb, "group"].map(kmed)
    table.loc[fb, "p15"] = table.loc[fb, "pmax"]
    table.loc[table["kappa"].isna(), ["kappa", "kappa_method"]] = [0.0, "none"]
    table["p15"] = table["p15"].fillna(table["pmax"])

    lf = cfg.load_fraction * table["pmax"]
    table["hr15_lf"] = heat_input(table, lf, cfg.t_ref, cfg.t_ref) / lf
    table["hr_pct_per_c"] = hr_change_per_c(table, lf, cfg.t_ref)
    table["hr_pct_per_c_wd"] = hr_change_per_c(table.assign(d=table["d_wd"], e=table["e_wd"]),
                                               lf, cfg.t_ref)
    table["lf_in_range"] = (lf >= table["pmin"]) & (lf <= table["pmax"])
    # Heat rate at full available load, 15 C vs 35 C (derating and heat terms).
    t_hot = cfg.t_ref + 20.0
    p_hot = table["p15"] * derate_factor(table["kappa"], t_hot, cfg.t_ref)
    hr_full15 = heat_input(table, table["p15"], cfg.t_ref, cfg.t_ref) / table["p15"]
    hr_full_hot = heat_input(table, p_hot, t_hot, cfg.t_ref) / p_hot
    table["hr_full_pct_per_c"] = 100.0 * (hr_full_hot / hr_full15 - 1.0) / (t_hot - cfg.t_ref)
    first = ["unit_id", *UNIT_KEY, "plant_id_eia", "plant_name_eia", "tech", "group",
             "chp", "suspect_cc", "const_heat", "nameplate_mw", "summer_mw", "winter_mw"]
    return table[first + [c for c in table.columns if c not in first]].sort_values(
        "unit_id", ignore_index=True)


# --------------------------------------------------------------------------
# Curve model for dispatch
# --------------------------------------------------------------------------
class TemperatureCurves:
    """Temperature-dependent curve model for :mod:`ttrl.supply.dispatch`.

    Implements the ``CurveModel`` protocol of :mod:`ttrl.supply.units`:
    ``arrays(temp_c)`` with ``temp_c`` of shape (hours, units) returns
    (hours, units) :class:`CurveArrays` with a(T) = a + d (T - Tref),
    b(T) = b + e (T - Tref), c unchanged, Pmax(T) = P15 x
    :func:`derate_factor` (never below Pmin) and Pmin unchanged.

    ``use_heat`` / ``use_derate`` switch the two effects on and off. For
    the heat terms, temperatures are clipped to the table columns
    ``t_fit_lo, t_fit_hi`` when present (the unit's observed range), so the
    heat curve is not extrapolated; derating uses the unclipped temperature
    (a linear extrapolation a few degrees past the hottest bin).
    With ``temp_c=None`` the 15 C curves are returned as (1, units).
    """

    def __init__(self, table: pd.DataFrame, use_heat: bool = True,
                 use_derate: bool = True, t_ref: float = T_REF):
        self.table = table.reset_index(drop=True)
        self.unit_ids: list[str] = self.table["unit_id"].tolist()
        self.use_heat, self.use_derate, self.t_ref = use_heat, use_derate, t_ref

    def _col(self, name: str) -> np.ndarray:
        return self.table[name].to_numpy(float)[None, :]

    def arrays(self, temp_c: np.ndarray | None = None) -> CurveArrays:
        """Curve arrays at ``temp_c`` (hours, units); (1, units) at 15 C if None."""
        a, b, c, pmin = (self._col(k) for k in ("a", "b", "c", "pmin"))
        p15 = self._col("p15")
        if temp_c is None:
            return CurveArrays(a, b, c, pmin, np.maximum(p15, pmin))
        t = np.asarray(temp_c, float)
        th = t
        if {"t_fit_lo", "t_fit_hi"} <= set(self.table.columns):
            lo, hi = self._col("t_fit_lo"), self._col("t_fit_hi")
            th = np.clip(t, np.where(np.isnan(lo), -np.inf, lo),
                         np.where(np.isnan(hi), np.inf, hi))
        dt = th - self.t_ref
        if self.use_heat:
            a = a + self._col("d") * dt
            b = b + self._col("e") * dt
        pmax = p15 * (derate_factor(self._col("kappa"), t, self.t_ref)
                      if self.use_derate else 1.0)
        n = t.shape[0]
        bc = lambda x: np.broadcast_to(x, (n, x.shape[-1]))  # noqa: E731
        return CurveArrays(bc(a), bc(b), bc(c), bc(pmin), np.maximum(bc(pmax), pmin))

    @property
    def co2_per_mmbtu(self) -> np.ndarray:
        """CO2 factor per unit (short tons per MMBtu), shape (units,)."""
        return self.table["co2_per_mmbtu"].to_numpy(float)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean ignoring NaN values."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w)
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else np.nan


def group_stats(x: np.ndarray, w: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Median, quartiles, weighted mean and bootstrap 95% CIs over units.

    The bootstrap resamples units with replacement (percentile intervals).
    """
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    out = {"n": len(x), "median": np.nan, "p25": np.nan, "p75": np.nan,
           "median_lo": np.nan, "median_hi": np.nan, "wmean": np.nan,
           "wmean_lo": np.nan, "wmean_hi": np.nan}
    if len(x) == 0:
        return out
    out.update(median=float(np.median(x)), p25=float(np.quantile(x, 0.25)),
               p75=float(np.quantile(x, 0.75)), wmean=weighted_mean(x, w))
    if len(x) < 3:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    med = np.median(x[idx], axis=1)
    wm = (x[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    out.update(median_lo=float(np.quantile(med, 0.025)), median_hi=float(np.quantile(med, 0.975)),
               wmean_lo=float(np.quantile(wm, 0.025)), wmean_hi=float(np.quantile(wm, 0.975)))
    return out


def stratified_ratio_bootstrap(num: np.ndarray, den: np.ndarray, strata: np.ndarray,
                               n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap draws of sum(num) / sum(den) resampling units within strata.

    ``num`` and ``den`` are per-unit totals (e.g. extra fuel and output).
    Each stratum is resampled with replacement to its own size, so the mix
    of technologies is held fixed.
    """
    num, den, strata = np.asarray(num, float), np.asarray(den, float), np.asarray(strata)
    rng = np.random.default_rng(seed)
    tot_n = np.zeros(n_boot)
    tot_d = np.zeros(n_boot)
    for s in np.unique(strata):
        k = np.flatnonzero(strata == s)
        idx = k[rng.integers(0, len(k), size=(n_boot, len(k)))]
        tot_n += num[idx].sum(axis=1)
        tot_d += den[idx].sum(axis=1)
    return tot_n / tot_d
