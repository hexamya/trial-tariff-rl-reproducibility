"""Monthly mixed-integer planning and day-ahead model predictive control.

Both plan with the same response model as the agent they are compared with: per-pattern effects
(``patterns.pattern_effects``) for the training ensemble members. The objective averages over all of them
(sample-average approximation, as Kale et al.'s stochastic MIP); the revenue band must hold for each of
``robust_k`` representative members (``revenue_scenarios``: the members at evenly spaced quantiles of the
revenue response, from the least to the most responsive), which carries the response uncertainty into
the constraint at a fraction of the cost of one row per member.

* ``MonthlyMIP`` (oracle): plans the whole month once with the actual baseline load of every day; an
  upper bound on what a planner with this response model can reach, not a feasible operating rule.
* ``RollingMPC``: before each day it re-plans the rest of the month and posts tomorrow's pattern, using
  only what is known then: past days' load (complete up to two days before), the day-ahead forecast for
  tomorrow (and the not yet complete day before), and for later days the load of the same weekday in the
  latest complete week; ERCOT wind and solar of the latest complete day for every later day (EIA-930
  publishes no renewables forecast). Tomorrow's choice is integer and later days are relaxed (relax-and-fix), posted
  days are fixed, and the revenue band and segment cap are soft (penalty of 100 times the month's largest
  daily objective term per unit of violation), so the planner gets as close as it can when earlier
  choices made on forecasts no longer allow the band. The plan does not use the realised response, so
  one plan per month serves every evaluation member; so does the MIP's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrl.env import london as ld
from ttrl.env.ercot_env import PRICE_RATIO
from ttrl.env.tariff import PRICE_FLAT, slot_bands, slot_prices
from ttrl.s3 import harness as hs
from ttrl.s3 import mip
from ttrl.s3 import patterns as pt


# ------------------------------------------------------------------ coefficient tables
def london_tables(base: np.ndarray, effects: np.ndarray, patterns: np.ndarray, households: np.ndarray | None = None,
                  segments: np.ndarray | None = None) -> dict:
    """``base`` (D, 48) baseline, ``effects`` (M, P, 48), ``households`` (H, D, 48)."""
    price = slot_prices(patterns)                                            # (P, 48)
    M = len(effects)
    peak_cut = np.zeros((base.shape[0], len(patterns)))
    rev = np.zeros((M,) + peak_cut.shape)
    for m, e in enumerate(effects):
        resp = base[:, None, :] * (1 + e[None] / 100)                        # (D, P, 48)
        peak_cut += 100 * (1 - resp.max(-1) / base.max(-1)[:, None])
        rev[m] = np.einsum("dph,ph->dp", resp, price)
    out = {"value": peak_cut / M, "rev_num": rev, "rev_den": PRICE_FLAT * base.sum()}
    if households is not None:
        q = np.nan_to_num(households)
        tot = q.sum(axis=(1, 2))
        ok = tot > 0
        contrib = np.einsum("idh,ph->idp", q[ok], price) / (PRICE_FLAT * tot[ok])[:, None, None]
        segs = segments[ok]
        out["seg"] = np.stack([contrib[segs == s].mean(axis=0) for s in range(ld.N_SEGMENTS)])
    return out


def ercot_tables(env, days_idx: np.ndarray, demand_override: dict | None, effects: np.ndarray, patterns: np.ndarray) -> dict:
    """Coefficients for ERCOT tariff days ``days_idx`` (global). ``demand_override``: day -> hourly demand
    (24,) used instead of the actual (forecast or proxy), residential load scaling with it; its entry
    "ren" (day -> hourly wind + solar) replaces the actual renewables."""
    y, s, ds, cfg = env.days.year, env.days.surrogate, env.days, env.cfg
    M, P, D = effects.shape[0], len(patterns), len(days_idx)
    eh = effects.reshape(M, P, 24, 2).mean(-1)                                       # (M, P, 24)
    bands = slot_bands(patterns)
    ratio = np.select([bands == 1, bands == -1], [PRICE_RATIO[1], PRICE_RATIO[-1]], PRICE_RATIO[0]).reshape(P, 24, 2).mean(-1)
    cost, peak, rev = np.zeros((D, P)), np.full((D, P), -np.inf), np.zeros((M, D, P))
    den = 0.0
    month = int(str(ds.month[days_idx[0]])[5:7])
    gas = cfg.gas_price_in(month)
    for k, d in enumerate(days_idx):
        for rows in (ds.row[d], ds.row2[d]):
            ok = rows >= 0
            r = rows[ok]
            if not len(r):
                continue
            dem, res, ren = y.demand[r].copy(), y.residential[r].copy(), y.renewables[r].copy()
            if demand_override is not None and int(d) in demand_override:
                f = demand_override[int(d)][ok]
                res = res * f / np.maximum(dem, 1e-9)
                dem = f
            if demand_override is not None and int(d) in demand_override.get("ren", {}):
                ren = demand_override["ren"][int(d)][ok]
            dl = res[None, None, :] * eh[:, :, ok] / 100                                # (M, P, h)
            fuel = s.fuel[r] * dl + 0.5 * s._b2(dl, 0) * s.fuel_curv[r] * dl ** 2
            co2 = s.co2_t[r] * dl + 0.5 * s._b2(dl, 1) * s.co2_curv[r] * dl ** 2
            cost[k] += (gas * fuel.sum(-1) + cfg.co2_price * co2.sum(-1)).mean(0)
            net = dem - ren
            peak[k] = np.maximum(peak[k], (net[None, None, :] + dl).max(-1).mean(0))
            rev[:, k] += (ratio[None, :, ok] * res[None, None, :] * (1 + eh[:, :, ok] / 100)).sum(-1)
            den += float(res.sum())
    return {"value": -cost, "rev_num": rev, "rev_den": den, "peak": peak,
            "peak_coef": cfg.capacity_value * float(cfg.peak_month_weight.get(month, 0.0))}


def revenue_scenarios(effects: np.ndarray, patterns: np.ndarray, k: int = 11) -> np.ndarray:
    """Members at evenly spaced quantiles (0, 1/(k-1), ..., 1) of their revenue response: the mean over the
    candidate patterns of sum_h price x effect (flat load profile)."""
    stat = np.einsum("mph,ph->m", effects, slot_prices(patterns)) / len(patterns)
    order = np.argsort(stat)
    pick = np.unique(np.round(np.linspace(0, len(stat) - 1, min(k, len(stat)))).astype(int))
    return np.sort(order[pick])


def iso_weeks(tdays) -> np.ndarray:
    iso = pd.DatetimeIndex(tdays).isocalendar()
    return iso["year"].to_numpy() * 100 + iso["week"].to_numpy()


# ------------------------------------------------------------------ policies
@dataclass
class _Planner:
    effects: np.ndarray                  # (M, P, 48) planning members
    patterns: np.ndarray
    time_limit: float = 60.0
    mip_gap: float = 1e-3
    robust_k: int = 11
    name: str = ""
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.compat = pt.compatibility(self.patterns)
        self.index = {tuple(p): j for j, p in enumerate(self.patterns)}
        self.robust = revenue_scenarios(self.effects, self.patterns, self.robust_k)

    def _model(self, env, known: dict | None, fixed: dict) -> mip.MonthModel:
        days = np.asarray(env.day_idx)
        if hs.is_london(env):
            base = np.stack([known[int(d)] if known else env.inputs.baseline[d] for d in days])
            hh = known["households"] if known else env.inputs.households[:, days]
            t = london_tables(base, self.effects, self.patterns, hh, env.segments)
            return mip.MonthModel(t["value"], t["rev_num"][self.robust], t["rev_den"], env.cfg.revenue_tol, self.patterns, self.compat,
                                  seg=t["seg"], seg_cap=env.cfg.segment_bill_cap,
                                  week=iso_weeks(env.inputs.days["tday"].to_numpy()[days]), fixed=fixed, rules=env.cfg.rules)
        t = ercot_tables(env, days, known, self.effects, self.patterns)
        masked = np.flatnonzero(env.masked)
        num, den = t["rev_num"][self.robust], t["rev_den"]
        if len(masked):                                   # masked days leave the revenue ratio (environment rule)
            num = num.copy()
            num[:, masked] = 0.0
            den -= sum(float(env.days.year.residential[env.days.rows(days[k])].sum()) for k in masked)
        return mip.MonthModel(t["value"], num, den, env.cfg.revenue_tol, self.patterns, self.compat,
                              peak=t["peak"], peak_coef=t["peak_coef"], fixed={**{int(k): 0 for k in masked}, **fixed},
                              rules=env.cfg.rules)


@dataclass
class MonthlyMIP(_Planner):
    def __post_init__(self):
        super().__post_init__()
        self.name = self.name or "MIP (oracle month)"

    cache: dict = field(default_factory=dict)

    def reset(self, env) -> None:
        if env.month not in self.cache:
            choice, info = mip.solve(self._model(env, None, {}), self.time_limit, self.mip_gap)
            self.cache[env.month] = self.patterns[choice]
            self.log.append({"month": env.month, **info})

    def act(self, env, obs) -> np.ndarray:
        return self.cache[env.month][env.t].copy()


def known_load(env, t: int) -> dict:
    """Load the planner may use for every day of the month when posting day ``t`` (see module docstring)."""
    days = np.asarray(env.day_idx)
    back = lambda k: int(np.ceil((k - (t - 2)) / 7)) * 7      # noqa: E731  days back to the latest complete same weekday
    out = {}
    if hs.is_london(env):
        base, hh = env.inputs.baseline, env.inputs.households
        cols = []
        for k, d in enumerate(days):
            if k <= t - 2:
                out[int(d)], src = base[d], d
            elif k in (t - 1, t):
                out[int(d)], src = env.forecast[d], max(0, d - 7)
            else:
                src = max(0, d - back(k))
                out[int(d)] = base[src]
            cols.append(hh[:, d] if k <= t - 2 else hh[:, src])
        out["households"] = np.stack(cols, axis=1)
        return out
    y, ds = env.days.year, env.days

    def profile(vals, src):
        srows = ds.row[src]
        f = np.where(srows >= 0, vals[np.maximum(srows, 0)], np.nan)
        return np.where(np.isnan(f), np.nanmean(f), f)
    out["ren"] = {}
    last = int(days[t - 2]) if t >= 2 else max(0, int(days[0]) - 2)   # latest complete day
    for k, d in enumerate(days):
        if k <= t - 2:
            continue                                      # actual load and renewables
        src, vals = (d, y.demand_forecast) if k in (t - 1, t) else (max(0, d - back(k)), y.demand)
        out[int(d)] = profile(vals, src)
        out["ren"][int(d)] = profile(y.renewables, last)  # no renewables forecast in EIA-930: persistence
    return out


@dataclass
class RollingMPC(_Planner):
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.name = self.name or "MPC (day-ahead)"

    def plan_month(self, env) -> np.ndarray:
        posted = {}
        for t in range(env.n):
            if hs.is_london(env) is False and env.masked[t]:
                posted[t] = 0
                continue
            model = self._model(env, known_load(env, t), dict(posted))
            model.integer_days = np.array([t])
            model.soft_penalty = 100.0 * float(np.abs(model.value).max(axis=1).sum()) + 1.0
            choice, info = mip.solve(model, self.time_limit, self.mip_gap)
            self.log.append({"month": env.month, "day": t, **info})
            if info["objective"] is None:                  # infeasible: post Normal
                posted[t] = 0
                continue
            posted[t] = int(choice[t])
        return self.patterns[[posted[t] for t in range(env.n)]]

    def reset(self, env) -> None:
        if env.month not in self.cache:
            self.cache[env.month] = self.plan_month(env)

    def act(self, env, obs) -> np.ndarray:
        return self.cache[env.month][env.t].copy()
