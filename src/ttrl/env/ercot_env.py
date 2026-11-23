"""ERCOT case of the S2 environment: a monthly constrained MDP over day-ahead block tariffs.

The agent posts each tariff day's eight blocks (as in London, :mod:`ttrl.env.tariff`); the
response (:mod:`ttrl.env.responses`) changes the residential part of ERCOT load by its percent
effect, averaged from half-hours to hours; the load change is met by the non-CHP gas fleet, costed
with the band-corrected training surrogate (:class:`ttrl.env.ercot.Surrogate`). Final evaluation
replaces the surrogate with ``S4Model.impact`` (see ``evaluate_with_s4``).

Per settled day the environment reports the change in gas fuel (MMBtu), CO2 (t), the day's peaks of
demand and of net load (demand minus wind and solar, MW) and the residential energy moved.

Reward (2024 US$, ``ttrl.env.rewards``, fixed before S3): minus the cost of the fuel, CO2 and
capacity that the tariff changes. Each settled day pays gas price (by calendar month) x fuel change +
CO2 price x CO2 change; capacity is charged on the month's peak of net load: capacity value ($/MW-yr)
x the calendar month's weight x the change of the month-to-date net-load peak, paid as the running
peak moves, so the month's capacity charge is the value x weight x (peak with the tariff - peak
without). ``ErcotConfig.from_settings`` builds the configuration from ``S2Settings``.

Observation (what is known when the next day's tariff is posted): ERCOT's day-ahead demand forecast
for the 24 hours of that day (EIA-930), calendar, the previous day's mean temperature, residential
share and renewable output, the month-to-date net-load peak, the previous day's blocks and
month-to-date constraint state.

Constraint costs at the end of the episode: residential revenue outside ``revenue_tol`` of the flat
rate (the trial's price ratios to the flat rate applied to Texas residential load), and the trial
schedule counts (:class:`TrialRules`). Bill distributions are a London-case metric (no Texas
household data).

Days whose mean temperature (four-station mean) exceeds the kernel's 24.7 C are flagged
(``info["out_of_support"]``) and the response on them can be scaled by ``hot_multiplier``.

Curtailed-demand days (rotating outages or large forecast misses, ``ercot.curtailed_days``) are masked
when ``mask_curtailed`` is set (training and validation years): the posted blocks are replaced by
all-Normal and the day is left out of the reward, the revenue ratio and the episode metrics. In the
test year they are kept and reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrl.env import ercot as er
from ttrl.env.responses import with_hot_multiplier
from ttrl.env.tariff import BLOCKS_PER_DAY, HH_PER_DAY, NORMAL, PRICE, PRICE_FLAT, TrialRules, slot_bands

KERNEL_TEMP_MAX = 24.7
LOOKBACK_DAYS = 3
PRICE_RATIO = {k: v / PRICE_FLAT for k, v in PRICE.items()}


@dataclass
class ErcotDays:
    """One weather year on the tariff-day grid: per day and hour of the tariff day (24), the UTC hour
    rows that belong to it (-1 if none: spring night) and a second row for the autumn repeat."""
    year: er.ErcotYear
    surrogate: er.Surrogate
    tdays: pd.DatetimeIndex
    row: np.ndarray              # (n_days, 24) first UTC-hour row, -1 if none
    row2: np.ndarray             # (n_days, 24) second row on the autumn night, else -1
    temp_day: np.ndarray         # (n_days,) mean temperature of the tariff day
    month: np.ndarray            # (n_days,) 'YYYY-MM' of the tariff day
    curtailed: np.ndarray        # (n_days,) any curtailed hour

    @classmethod
    def build(cls, year: er.ErcotYear, surrogate: er.Surrogate, temp_c: np.ndarray) -> "ErcotDays":
        key = year.key
        tdays = pd.DatetimeIndex(sorted(key["tday"].unique()))
        first_full, last_full = tdays[1], tdays[-2]         # the edges are partial days
        tdays = tdays[(tdays >= first_full) & (tdays <= last_full)]
        pos = pd.Index(tdays).get_indexer(key["tday"])
        row = np.full((len(tdays), 24), -1)
        row2 = np.full((len(tdays), 24), -1)
        for i, (p, h) in enumerate(zip(pos, key["hour"].to_numpy())):
            if p < 0:
                continue
            if row[p, h] < 0:
                row[p, h] = i
            else:
                row2[p, h] = i
        temp = np.array([np.nanmean(temp_c[np.r_[row[d][row[d] >= 0], row2[d][row2[d] >= 0]]]) for d in range(len(tdays))])
        curt = np.array([year.curtailed[row[d][row[d] >= 0]].any() for d in range(len(tdays))])
        return cls(year, surrogate, tdays, row, row2, temp, tdays.to_period("M").astype(str).to_numpy(), curt)

    @classmethod
    def load(cls, year: int) -> "ErcotDays":
        """Saved inputs of a weather year (scripts/build_ercot_env.py)."""
        return cls.build(*er.load_inputs(year))

    def rows(self, d: int) -> np.ndarray:
        r = np.r_[self.row[d], self.row2[d]]
        return r[r >= 0]


def hourly_effect(effect_pct_day: np.ndarray) -> np.ndarray:
    """(48,) half-hourly percent effect -> (24,) hourly mean."""
    return np.asarray(effect_pct_day).reshape(24, 2).mean(axis=1)


def day_outcome(days: ErcotDays, d: int, effect_pct_day: np.ndarray) -> dict:
    """Load change and its costs for tariff day ``d`` given the day's half-hourly effect."""
    y, s = days.year, days.surrogate
    eh = hourly_effect(effect_pct_day)
    out = {"d_fuel_mmbtu": 0.0, "d_co2_t": 0.0, "moved_mwh": 0.0}
    loads, base_loads = [], []
    for rows in (days.row[d], days.row2[d]):
        ok = rows >= 0
        r = rows[ok]
        dl = y.residential[r] * eh[ok] / 100
        out["d_fuel_mmbtu"] += float(s.fuel_change(r, dl).sum())
        out["d_co2_t"] += float(s.co2_change(r, dl).sum())
        out["moved_mwh"] += float(np.abs(dl).sum() / 2)
        loads.append(y.demand[r] + dl)
        base_loads.append(y.demand[r])
    out["peak_mw"] = float(np.concatenate(loads).max())
    out["base_peak_mw"] = float(np.concatenate(base_loads).max())
    out["d_peak_mw"] = out["peak_mw"] - out["base_peak_mw"]
    r = days.rows(d)                      # same hour order as ``loads``
    out["net_peak_mw"] = float((np.concatenate(loads) - y.renewables[r]).max())
    out["base_net_peak_mw"] = float((y.demand[r] - y.renewables[r]).max())
    return out


@dataclass
class ErcotConfig:
    months: list[str]
    members: np.ndarray
    gas_price: float | dict          # $/MMBtu, or {calendar month: $/MMBtu}
    co2_price: float                 # $/t
    capacity_value: float            # $ per MW-yr of capacity-relevant net-load peak
    peak_month_weight: dict          # {calendar month: share of the capacity value}
    revenue_tol: float = 0.03
    hot_multiplier: float = 1.0
    mask_curtailed: bool = True      # False in the test year
    rules: TrialRules = field(default_factory=TrialRules)

    @classmethod
    def from_settings(cls, settings, months: list[str], members: np.ndarray, **kw) -> "ErcotConfig":
        """Configuration with the reward weights of ``ttrl.env.rewards.S2Settings``."""
        return cls(months, members, dict(settings.gas_price_by_month), settings.co2_price,
                   settings.capacity_value, dict(settings.peak_month_weight), **kw)

    def gas_price_in(self, month: int) -> float:
        return float(self.gas_price[month]) if isinstance(self.gas_price, dict) else float(self.gas_price)


class ErcotTariffEnv:
    """Gymnasium-style environment (see module docstring)."""

    def __init__(self, days: ErcotDays, response, config: ErcotConfig, seed: int | None = None):
        self.days, self.response, self.cfg = days, response, config
        self.rng = np.random.default_rng(seed)
        self._month_days = {m: np.flatnonzero(days.month == m) for m in config.months}
        missing = [m for m, d in self._month_days.items() if len(d) == 0]
        if missing:
            raise ValueError(f"no tariff days for months {missing}")
        self.obs_size = 24 + 7 + BLOCKS_PER_DAY + 5
        self.demand_scale = float(np.nanmean(days.year.demand))

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.month = options.get("month") or self.rng.choice(self.cfg.months)
        self.member = int(options.get("member", self.rng.choice(self.cfg.members)))
        self.day_idx = self._month_days[self.month]
        self.n = len(self.day_idx)
        self.masked = self.days.curtailed[self.day_idx] & self.cfg.mask_curtailed
        cal_month = int(self.month[5:7])
        self.gas = self.cfg.gas_price_in(cal_month)
        self.peak_w = float(self.cfg.peak_month_weight.get(cal_month, 0.0))
        self.run_peak = self.run_peak_base = -np.inf
        self.blocks = np.zeros((self.n, BLOCKS_PER_DAY), np.int8)
        self.effect = np.zeros((self.n, HH_PER_DAY))
        self.outcomes: list[dict] = []
        self.t = self.settled = 0
        return self._obs(), {"month": self.month, "member": self.member}

    def step(self, action):
        action = np.asarray(action, dtype=np.int8).reshape(BLOCKS_PER_DAY)
        masked = bool(self.masked[self.t])
        invalid = not masked and not self.cfg.rules.day_ok(self.blocks[max(0, self.t - 1):self.t], action)
        if invalid or masked:
            action = np.full(BLOCKS_PER_DAY, NORMAL, np.int8)
        self.blocks[self.t] = action
        self.t += 1
        reward = self._settle_through(self.t - 2) if self.t >= 2 else 0.0
        terminated = self.t >= self.n
        info = {"invalid": invalid, "masked": masked, "cost": {}}
        if terminated:
            reward += self._settle_through(self.n - 1)
            info["cost"], info["metrics"] = self._episode_costs()
        return self._obs(), reward, terminated, False, info

    def _hot(self, d_local: np.ndarray) -> np.ndarray:
        return self.days.temp_day[self.day_idx[d_local]] > KERNEL_TEMP_MAX

    def _settle_through(self, last: int) -> float:
        r = 0.0
        while self.settled <= last:
            d = self.settled
            lo, hi = max(0, d - LOOKBACK_DAYS), min(self.n, d + 2)
            eff = self.response.effect(self.blocks[lo:hi], self.member)
            eff = with_hot_multiplier(eff, self._hot(np.arange(lo, hi)), self.cfg.hot_multiplier)
            self.effect[d] = eff[d - lo]
            o = day_outcome(self.days, self.day_idx[d], self.effect[d])
            o["out_of_support"] = bool(self._hot(np.array([d]))[0])
            o["masked"] = bool(self.masked[d])
            self.outcomes.append(o)
            self.settled += 1
            if o["masked"]:
                continue
            gap0 = self.run_peak - self.run_peak_base if np.isfinite(self.run_peak) else 0.0
            self.run_peak = max(self.run_peak, o["net_peak_mw"])
            self.run_peak_base = max(self.run_peak_base, o["base_net_peak_mw"])
            o["capacity_cost"] = self.cfg.capacity_value * self.peak_w * (self.run_peak - self.run_peak_base - gap0)
            o["fuel_cost"] = self.gas * o["d_fuel_mmbtu"]
            o["co2_cost"] = self.cfg.co2_price * o["d_co2_t"]
            r -= o["fuel_cost"] + o["co2_cost"] + o["capacity_cost"]
        return r

    def revenue_ratio(self) -> float:
        """Residential revenue at the trial's price ratios over flat revenue, settled unmasked days."""
        num = den = 0.0
        y = self.days.year
        for k in range(self.settled):
            if self.masked[k]:
                continue
            d = self.day_idx[k]
            ratio_hh = np.select([slot_bands(self.blocks[k]) == 1, slot_bands(self.blocks[k]) == -1],
                                 [PRICE_RATIO[1], PRICE_RATIO[-1]], PRICE_RATIO[0])
            rh = ratio_hh.reshape(24, 2).mean(axis=1)
            eh = hourly_effect(self.effect[k])
            for rows in (self.days.row[d], self.days.row2[d]):
                ok = rows >= 0
                res = y.residential[rows[ok]]
                num += float((rh[ok] * res * (1 + eh[ok] / 100)).sum())
                den += float(res.sum())
        return num / den if den else 1.0

    def _episode_costs(self) -> tuple[dict, dict]:
        rev = self.revenue_ratio()
        counts = self.cfg.rules.counts(self.blocks)
        o_all = pd.DataFrame(self.outcomes)
        o = o_all[~o_all["masked"]]
        cost = {"revenue": max(0.0, abs(rev - 1) - self.cfg.revenue_tol),
                "high_hours": max(0, counts["high_hh"] - self.cfg.rules.max_high_hh_per_month),
                "low_hours": max(0, counts["low_hh"] - self.cfg.rules.max_low_hh_per_month)}
        metrics = {"revenue_ratio": rev, **counts, "d_fuel_mmbtu": float(o["d_fuel_mmbtu"].sum()),
                   "d_co2_t": float(o["d_co2_t"].sum()), "mean_d_peak_mw": float(o["d_peak_mw"].mean()),
                   "monthly_peak_change_mw": float(o["peak_mw"].max() - o["base_peak_mw"].max()),
                   "monthly_net_peak_change_mw": float(o["net_peak_mw"].max() - o["base_net_peak_mw"].max()),
                   "fuel_cost": float(o["fuel_cost"].sum()), "co2_cost": float(o["co2_cost"].sum()),
                   "capacity_cost": float(o["capacity_cost"].sum()),
                   "out_of_support_days": int(o["out_of_support"].sum()), "masked_days": int(o_all["masked"].sum())}
        return cost, metrics

    def _obs(self) -> np.ndarray:
        """Known when day d's tariff is posted: day-ahead forecast of day d, the previous day's
        observations (temperature, residential share, renewables), calendar and month-to-date state."""
        d = min(self.t, self.n - 1)
        idx = self.day_idx[d]
        y, ds = self.days.year, self.days
        rows = ds.row[idx]
        fc = np.where(rows >= 0, y.demand_forecast[np.maximum(rows, 0)], np.nan)
        fc = np.nan_to_num(fc / self.demand_scale, nan=float(np.nanmean(fc / self.demand_scale)))
        prev_idx = idx - 1 if idx > 0 else idx
        pr = ds.rows(prev_idx)
        share = float(np.mean(y.residential[pr] / y.demand[pr]))
        ren = float(np.mean(y.renewables[pr]) / self.demand_scale)
        tday = pd.Timestamp(ds.tdays[idx])
        mon = tday.month
        cal = [float(tday.dayofweek >= 5), np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
               ds.temp_day[prev_idx] / 10, share, ren, self.peak_w]
        prev = self.blocks[self.t - 1] if self.t > 0 else np.zeros(BLOCKS_PER_DAY)
        c = self.cfg.rules.counts(self.blocks[:self.t]) if self.t else {"high_hh": 0, "low_hh": 0}
        mtd_peak = self.run_peak / self.demand_scale if np.isfinite(self.run_peak) else 0.0
        state = [self.revenue_ratio() - 1 if self.settled else 0.0, c["high_hh"] / self.cfg.rules.max_high_hh_per_month,
                 c["low_hh"] / self.cfg.rules.max_low_hh_per_month, (self.n - self.t) / 31, mtd_peak]
        return np.concatenate([fc, cal, prev, state]).astype(np.float32)
