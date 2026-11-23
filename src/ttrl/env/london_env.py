"""London case of the S2 environment: a monthly constrained MDP over day-ahead block tariffs.

One episode is one calendar month of tariff days (02:00-02:00). At each step the agent posts the
next tariff day's eight blocks (Low / Normal / High, :mod:`ttrl.env.tariff`). The response is a
:mod:`ttrl.env.responses` object: the S1 kernel ensemble, one member drawn per episode (domain
randomisation), or an assumed model (agent B); load is the control-group baseline scaled by the
response's percent effect.

Timing: the kernel's windows reach 2 h before and 4 h after an event, and an event that runs
across midnight has a duration that depends on the next day, so day t is settled when day t+1 is
posted; the day after the last day of the episode is taken as Normal.

Reward per settled day (``"peak"``): percent reduction of the day's peak of the aggregate household
load against the baseline peak, the objective of Kale et al. on the same trial (mean daily peak
reduction), so results can be compared with theirs. Constraint costs, at the end of the episode
(``info["cost"]``): revenue outside ``revenue_tol`` of the flat-rate revenue (Kale et al.: +/-3%);
the mean bill change of the most affected household segment above ``segment_bill_cap`` (Kale et al.:
+3%), with segments the quartiles of the households' pre-trial evening consumption share
(``london.exposure_segments``) and bills those of households that do not respond (response-independent,
so the constraint is the same whatever response an agent trains on); High half-hours above the monthly
limit; days with both bands above the weekly limit. Household P95 and CVaR95 are reported as metrics. Schedules that break the per-day rules
(:class:`TrialRules`) are replaced by an all-Normal day and flagged in ``info``.

Observation: the day-ahead forecast of the next day's baseline (``london.day_ahead_forecast``,
fitted on pre-trial days), calendar, the previous day's mean temperature, the previous day's blocks
and month-to-date constraint state.

The API follows Gymnasium (``reset`` / ``step`` returning observation, reward, terminated,
truncated, info) without importing it; S3 wraps it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env.responses import TrialResponse
from ttrl.env.tariff import BLOCKS_PER_DAY, HH_PER_DAY, NORMAL, TrialRules, slot_bands

MEMBER_SPLIT_SEED = 20261010
LOOKBACK_DAYS = 3  # runs are at most 30 h, windows 4 h: three earlier days settle a day exactly


def member_split(n_members: int, n_eval: int = 50, seed: int = MEMBER_SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Fixed split of ensemble members into training and evaluation members."""
    perm = np.random.default_rng(seed).permutation(n_members)
    return np.sort(perm[n_eval:]), np.sort(perm[:n_eval])


@dataclass
class LondonConfig:
    months: list[str]                       # months episodes are drawn from, e.g. ['2013-01', ...]
    members: np.ndarray                     # ensemble members episodes are drawn from
    reward: str = "peak"
    revenue_tol: float = 0.03
    segment_bill_cap: float = 0.03          # S2Settings.london_segment_bill_cap (Kale et al.)
    rules: TrialRules = field(default_factory=TrialRules)

    @classmethod
    def from_settings(cls, settings, months: list[str], members: np.ndarray, **kw) -> "LondonConfig":
        return cls(months, members, revenue_tol=settings.london_revenue_tol,
                   segment_bill_cap=settings.london_segment_bill_cap, **kw)


def settle(blocks: np.ndarray, arrays: fk.KernelArrays, member: int) -> np.ndarray:
    """Percent effect (n_days, 48) of a block schedule (n_days, 8) for one member, treating the
    days before and after as Normal."""
    eff = fk.effect(slot_bands(blocks).reshape(-1), arrays.member(member))[0]
    return eff.reshape(-1, HH_PER_DAY)


def evaluate(blocks: np.ndarray, inputs: ld.LondonInputs, day_idx: np.ndarray, arrays: fk.KernelArrays,
             members: np.ndarray) -> pd.DataFrame:
    """Metrics of a fixed schedule over days ``day_idx`` (consecutive), one row per member and month:
    revenue ratio, peak reduction (mean of daily %, and of the monthly peak), bill change P95 and
    CVaR95 (and CVaR95 for a household that does not respond), High / Low half-hours, days with both
    bands."""
    rows = []
    base = inputs.baseline[day_idx]
    months = inputs.days["month"].to_numpy()[day_idx]
    for m in members:
        eff = settle(blocks, arrays, int(m))
        load = ld.responded(base, eff)
        for mon in pd.unique(months):
            s = months == mon
            pk_b, pk_r = ld.daily_peak_kw(base[s]), ld.daily_peak_kw(load[s])
            hh = inputs.households[:, day_idx[s]]
            bills = ld.bill_changes(blocks[s], hh, eff[s])
            bills0 = ld.bill_changes(blocks[s], hh, np.zeros_like(eff[s]))
            rules = TrialRules().counts(blocks[s])
            rows.append({"member": int(m), "month": mon, "days": int(s.sum()),
                         "revenue_ratio": ld.revenue_ratio(blocks[s], base[s], eff[s]),
                         "daily_peak_cut_pct": float(np.mean(100 * (1 - pk_r / pk_b))),
                         "monthly_peak_cut_pct": float(100 * (1 - pk_r.max() / pk_b.max())),
                         "bill_p95": ld.percentile(bills, 95), "bill_cvar95": ld.cvar(bills),
                         "bill_cvar95_nonresponding": ld.cvar(bills0), **rules})
    return pd.DataFrame(rows)


class LondonTariffEnv:
    """Gymnasium-style environment (see module docstring)."""

    def __init__(self, inputs: ld.LondonInputs, response, config: LondonConfig, seed: int | None = None):
        """``response``: a :mod:`ttrl.env.responses` object (trial kernel or assumed model); kernel
        arrays are wrapped as the trial response."""
        if isinstance(response, fk.KernelArrays):
            response = TrialResponse(response)
        self.inputs, self.response, self.cfg = inputs, response, config
        self.rng = np.random.default_rng(seed)
        months = inputs.days["month"].to_numpy()
        self._month_days = {m: np.flatnonzero(months == m) for m in config.months}
        missing = [m for m, d in self._month_days.items() if len(d) == 0]
        if missing:
            raise ValueError(f"no tariff days for months {missing}")
        self.obs_size = HH_PER_DAY + 4 + BLOCKS_PER_DAY + 5
        self.segments = ld.exposure_segments(inputs.households, inputs.days)
        fc = ld.day_ahead_forecast(inputs.baseline, inputs.days)
        self.forecast = np.where(np.isnan(fc), inputs.baseline, fc)   # no lags: only before the panel's second week

    # ------------------------------------------------------------ episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.month = options.get("month") or self.rng.choice(self.cfg.months)
        self.member = int(options.get("member", self.rng.choice(self.cfg.members)))
        self.day_idx = self._month_days[self.month]
        self.n = len(self.day_idx)
        self.blocks = np.zeros((self.n, BLOCKS_PER_DAY), np.int8)
        self.t = 0
        self.settled = 0
        self.effect = np.zeros((self.n, HH_PER_DAY))
        return self._obs(), {"month": self.month, "member": self.member}

    def step(self, action):
        action = np.asarray(action, dtype=np.int8).reshape(BLOCKS_PER_DAY)
        hist = self.blocks[max(0, self.t - 1):self.t]
        invalid = not self.cfg.rules.day_ok(hist, action)
        if invalid:
            action = np.full(BLOCKS_PER_DAY, NORMAL, np.int8)
        self.blocks[self.t] = action
        self.t += 1
        reward = 0.0
        if self.t >= 2:                      # day t-2 (0-based) can now be settled
            reward += self._settle_through(self.t - 2)
        terminated = self.t >= self.n
        info = {"invalid": invalid, "cost": {}}
        if terminated:
            reward += self._settle_through(self.n - 1)
            info["cost"], info["metrics"] = self._episode_costs()
        return self._obs(), reward, terminated, False, info

    # ------------------------------------------------------------ internals
    def _settle_through(self, last: int) -> float:
        """Settle days self.settled..last; returns the summed reward of those days."""
        r = 0.0
        while self.settled <= last:
            d = self.settled
            lo = max(0, d - LOOKBACK_DAYS)
            hi = min(self.n, d + 2)                         # day d+1 posted (or Normal past the end)
            eff = self.response.effect(self.blocks[lo:hi], self.member)
            self.effect[d] = eff[d - lo]
            base = self.inputs.baseline[self.day_idx[d]]
            if self.cfg.reward == "peak":
                r += float(100 * (1 - ld.daily_peak_kw(ld.responded(base, self.effect[d])) / ld.daily_peak_kw(base)))
            else:
                raise ValueError(self.cfg.reward)
            self.settled += 1
        return r

    def _episode_costs(self) -> tuple[dict, dict]:
        base = self.inputs.baseline[self.day_idx]
        rev = ld.revenue_ratio(self.blocks, base, self.effect)
        hh = self.inputs.households[:, self.day_idx]
        bills = ld.bill_changes(self.blocks, hh, self.effect)
        bills0 = ld.bill_changes(self.blocks, hh, np.zeros_like(self.effect))      # household that does not respond
        seg0 = ld.segment_bill_means(bills0, self.segments)
        counts = self.cfg.rules.counts(self.blocks)
        weeks = pd.DatetimeIndex(self.inputs.days["tday"].to_numpy()[self.day_idx]).isocalendar()
        both = ((self.blocks == 1).any(1) & (self.blocks == -1).any(1))
        double_week_max = int(pd.Series(both, index=weeks.index).groupby([weeks.year.to_numpy(), weeks.week.to_numpy()]).sum().max())
        cost = {"revenue": max(0.0, abs(rev - 1) - self.cfg.revenue_tol),
                "bill_segment": max(0.0, float(np.max(seg0)) - self.cfg.segment_bill_cap),
                "high_hours": max(0, counts["high_hh"] - self.cfg.rules.max_high_hh_per_month),
                "low_hours": max(0, counts["low_hh"] - self.cfg.rules.max_low_hh_per_month),
                "double_days": max(0, double_week_max - self.cfg.rules.max_double_days_per_week)}
        metrics = {"revenue_ratio": rev, "bill_cvar95": ld.cvar(bills), "bill_p95": ld.percentile(bills, 95),
                   "bill_cvar95_nonresponding": ld.cvar(bills0), "bill_segment_max_nonresponding": float(np.max(seg0)),
                   **{f"bill_segment{s + 1}_nonresponding": float(v) for s, v in enumerate(seg0)},
                   **counts, "double_days_max_week": double_week_max}
        return cost, metrics

    def _obs(self) -> np.ndarray:
        """Day-ahead forecast of the next day's baseline, calendar, the previous day's temperature, the
        previous day's blocks, and month-to-date state."""
        d = min(self.t, self.n - 1)
        idx = self.day_idx[d]
        row = self.inputs.days.iloc[idx]
        base = self.forecast[idx] / max(self.inputs.baseline.mean(), 1e-9)
        mon = pd.Timestamp(row["tday"]).month
        t_prev = float(self.inputs.days["temp_c"].iloc[max(idx - 1, 0)])
        cal = [float(row["weekend"]), np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12), t_prev / 10]
        prev = self.blocks[self.t - 1] if self.t > 0 else np.zeros(BLOCKS_PER_DAY)
        done = self.blocks[:self.t]
        c = self.cfg.rules.counts(done) if self.t else {"high_hh": 0, "low_hh": 0, "double_days": 0}
        settled = slice(0, self.settled)
        rev = (ld.revenue_ratio(self.blocks[settled], self.inputs.baseline[self.day_idx[settled]], self.effect[settled])
               - 1) if self.settled else 0.0
        state = [rev, c["high_hh"] / self.cfg.rules.max_high_hh_per_month, c["low_hh"] / self.cfg.rules.max_low_hh_per_month,
                 c["double_days"] / 4, (self.n - self.t) / 31]
        return np.concatenate([base, cal, prev, state]).astype(np.float32)
