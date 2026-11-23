"""Rule-based tariff design baselines.

* No tariff (flat rate): the reference; every change is zero by definition and revenue equals the flat
  rate, so it is not run in the environment (all-Normal days are not the flat rate: the trial's
  Normal price is 17% below it).
* ``ScheduleReplay``: a fixed schedule by date, here the 2013 LCL schedule (London), the only policy
  whose outcome was observed.
* ``ForecastCPP``: critical-peak pricing called a day ahead from the forecast. A High event of
  ``length`` blocks is posted on the block(s) with the highest forecast load when tomorrow's forecast
  daily peak is at least the k-th highest forecast daily peak of the previous ``window`` days (a
  causal, season-following threshold), as long as the month's High budget allows it. One parameter,
  k, set on training months by ``calibrate``. Following the season badly (no events while peaks fall,
  events almost daily while they rise), it misses monthly revenue neutrality; kept as the naive rule.
* ``QuotaCPP``: CPP that tracks the month's revenue quota from the forecasts (the operator's rule).

A static time-of-use tariff is not among them: with the trial's prices (Normal 17% below the flat
rate) revenue can only be recovered with High blocks, and a daily High block exceeds the trial's
monthly High limit, so a revenue-neutral static TOU is outside the trial's treatment support.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrl.env.tariff import BLOCKS_PER_DAY, HH_PER_BLOCK, HIGH, NORMAL, PRICE, PRICE_FLAT, TrialRules
from ttrl.s3 import harness as hs


@dataclass
class ScheduleReplay:
    blocks_by_day: dict                    # tariff day (Timestamp) -> (8,) codes; missing days Normal
    name: str = "2013 schedule"

    def reset(self, env) -> None:
        pass

    def act(self, env, obs) -> np.ndarray:
        d = hs.next_day(env)
        tday = pd.Timestamp(env.inputs.days["tday"].iloc[d]) if hs.is_london(env) else pd.Timestamp(env.days.tdays[d])
        return np.asarray(self.blocks_by_day.get(tday, np.zeros(BLOCKS_PER_DAY)), np.int8)


@dataclass
class ForecastCPP:
    k: int
    window: int = 28
    length: int = 1
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            self.name = f"CPP k={self.k}" + (f" ({self.length} blocks)" if self.length > 1 else "")

    def reset(self, env) -> None:
        self._peaks = {}

    def _peak(self, env, d: int) -> float:
        if d not in self._peaks:
            self._peaks[d] = float(np.max(hs.forecast_profile(env, d)))
        return self._peaks[d]

    def act(self, env, obs) -> np.ndarray:
        d = hs.next_day(env)
        day = np.full(BLOCKS_PER_DAY, NORMAL, np.int8)
        past = [self._peak(env, j) for j in range(max(0, d - self.window), d)]
        if len(past) < self.k:
            return day
        if self._peak(env, d) < np.sort(past)[::-1][self.k - 1]:
            return day
        used = int((env.blocks[:env.t] == HIGH).sum()) * HH_PER_BLOCK
        if used + self.length * HH_PER_BLOCK > env.cfg.rules.max_high_hh_per_month:
            return day
        prof = hs.block_profile(hs.forecast_profile(env, d))
        start = int(np.argmax(prof))
        if self.length > 1:                # widen towards the higher neighbour, inside the day
            lo, hi = start, start
            while hi - lo + 1 < self.length:
                left = prof[lo - 1] if lo > 0 else -np.inf
                right = prof[hi + 1] if hi < BLOCKS_PER_DAY - 1 else -np.inf
                lo, hi = (lo - 1, hi) if left >= right else (lo, hi + 1)
            start = lo
        day[start:start + self.length] = HIGH
        if not env.cfg.rules.day_ok(env.blocks[max(0, env.t - 1):env.t], day):
            return np.full(BLOCKS_PER_DAY, NORMAL, np.int8)
        return day


@dataclass
class QuotaCPP:
    """CPP with a revenue quota: each day, the High events still needed this month to bring revenue to
    the flat rate are estimated from the forecasts (zero-response estimate: an event on block b of day d
    adds (p_High - p_Normal) / p_flat x forecast energy of the block / forecast energy of the month), and
    tomorrow gets an event on its highest forecast block when its forecast daily peak ranks in the top
    ``alpha`` x needed / days-left share of the previous ``window`` days. ``alpha`` (front-loading of the
    quota) is set on training months by ``calibrate``."""
    alpha: float
    window: int = 14
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            self.name = f"CPP quota a={self.alpha:g}"

    def reset(self, env) -> None:
        self._fc = {}

    def _profile(self, env, d: int) -> np.ndarray:
        if d not in self._fc:
            self._fc[d] = hs.forecast_profile(env, d)
        return self._fc[d]

    def act(self, env, obs) -> np.ndarray:
        day = np.full(BLOCKS_PER_DAY, NORMAL, np.int8)
        d = hs.next_day(env)
        month_days = env.day_idx[:env.t + 1]
        energy = np.mean([self._profile(env, j).sum() for j in month_days]) * env.n
        uplift = (PRICE[HIGH] - PRICE[NORMAL]) / PRICE_FLAT
        done = sum(uplift * self._profile(env, j)[hs.slot_mask(env.blocks[k] == HIGH)].sum()
                   for k, j in enumerate(env.day_idx[:env.t]))
        gap = 1.0 - PRICE[NORMAL] / PRICE_FLAT - done / energy
        prof = hs.block_profile(self._profile(env, d))
        b = int(np.argmax(prof))
        per_event = uplift * prof[b] * HH_PER_BLOCK / energy
        if gap <= 0 or per_event <= 0:
            return day
        need = gap / per_event
        share = min(1.0, self.alpha * need / (env.n - env.t))
        past = [float(np.max(self._profile(env, j))) for j in range(max(0, d - self.window), d)]
        rank = float(np.mean(np.asarray(past) < np.max(self._profile(env, d)))) if past else 1.0
        if rank < 1.0 - share:
            return day
        used = int((env.blocks[:env.t] == HIGH).sum()) * HH_PER_BLOCK
        if used + HH_PER_BLOCK > env.cfg.rules.max_high_hh_per_month:
            return day
        day[b] = HIGH
        if not env.cfg.rules.day_ok(env.blocks[max(0, env.t - 1):env.t], day):
            return np.full(BLOCKS_PER_DAY, NORMAL, np.int8)
        return day


def constraint_excess(r: pd.DataFrame) -> pd.Series:
    """Per episode: revenue and bill excess (fractions) plus count excesses as fractions of their limits."""
    out = r["cost_revenue"].copy()
    if "cost_bill_segment" in r:
        out = out + r["cost_bill_segment"]
    rules = TrialRules()
    out = out + r["cost_high_hours"] / rules.max_high_hh_per_month + r["cost_low_hours"] / rules.max_low_hh_per_month
    if "cost_double_days" in r:
        out = out + r["cost_double_days"] / rules.max_double_days_per_week
    return out


def calibrate(env, make, grid, months: list[str], members) -> tuple[object, pd.DataFrame]:
    """Pick the parameter with the lowest training violation rate, then the lowest mean constraint
    excess, then the highest mean reward."""
    rows = []
    for g in grid:
        r = hs.evaluate(env, make(g), months, members)
        rows.append({"param": g, "violation_rate": float(r["violated"].mean()),
                     "constraint_excess": float(constraint_excess(r).mean()), "reward": float(r["reward"].mean()),
                     "revenue_ratio": float(r["revenue_ratio"].mean()),
                     "revenue_in_band": float((r["cost_revenue"] <= 1e-12).mean())})
    tab = pd.DataFrame(rows)
    best = tab.sort_values(["violation_rate", "constraint_excess", "reward"], ascending=[True, True, False]).iloc[0]["param"]
    return best, tab
