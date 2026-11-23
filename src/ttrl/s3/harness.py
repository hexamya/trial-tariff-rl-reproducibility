"""Run tariff policies in the S2 environments and collect episode results.

A policy has ``reset(env)`` and ``act(env, obs) -> (8,) block codes``; it may read from the
environment only what is known when the next day's tariff is posted (``next_forecast``,
``past_forecasts``, the posted blocks and month-to-date state). The environment settles every day
itself, so all policies are judged by the same code.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from ttrl.env.london_env import LondonTariffEnv
from ttrl.env.tariff import BLOCKS_PER_DAY, HH_PER_BLOCK


class Policy(Protocol):
    name: str

    def reset(self, env) -> None: ...

    def act(self, env, obs: np.ndarray) -> np.ndarray: ...


def is_london(env) -> bool:
    return isinstance(env, LondonTariffEnv)


def n_days_total(env) -> int:
    return len(env.inputs.days) if is_london(env) else len(env.days.tdays)


def forecast_profile(env, day: int) -> np.ndarray:
    """Day-ahead forecast of day ``day`` (global day index) on the half-hour grid (48,): London, the
    household baseline forecast (kWh per household); ERCOT, ERCOT's day-ahead demand forecast (MW,
    each hour repeated)."""
    if is_london(env):
        return np.asarray(env.forecast[day], float)
    rows = env.days.row[day]
    fc = np.where(rows >= 0, env.days.year.demand_forecast[np.maximum(rows, 0)], np.nan)
    fc = np.where(np.isnan(fc), np.nanmean(fc), fc)
    return np.repeat(fc, 2)


def next_day(env) -> int:
    """Global index of the day whose tariff is posted next."""
    return int(env.day_idx[min(env.t, env.n - 1)])


def slot_mask(block_mask: np.ndarray) -> np.ndarray:
    """(8,) block mask -> (48,) half-hour mask."""
    return np.repeat(np.asarray(block_mask, bool), HH_PER_BLOCK)


def block_profile(profile48: np.ndarray) -> np.ndarray:
    return np.asarray(profile48, float).reshape(BLOCKS_PER_DAY, HH_PER_BLOCK).mean(axis=1)


def run_month(env, policy: Policy, month: str, member: int) -> dict:
    obs, _ = env.reset(options={"month": month, "member": int(member)})
    policy.reset(env)
    total, done, invalid = 0.0, False, 0
    while not done:
        obs, r, done, _, info = env.step(policy.act(env, obs))
        total += r
        invalid += int(info.get("invalid", False))
    out = {"policy": policy.name, "month": month, "member": int(member), "days": int(env.n), "reward": total,
           "invalid_days": invalid}
    out.update(info["metrics"])
    out.update({f"cost_{k}": v for k, v in info["cost"].items()})
    out["violated"] = any(v > 1e-12 for v in info["cost"].values())
    if is_london(env):
        out["daily_peak_cut_pct"] = total / env.n
    out["blocks"] = env.blocks.copy()
    out["effect"] = env.effect.copy()                   # percent effect per settled day (for the S4 evaluation)
    out["day_idx"] = np.asarray(env.day_idx).copy()
    return out


def evaluate(env, policy: Policy, months: list[str], members) -> pd.DataFrame:
    """One row per (month, member); ``blocks`` holds the posted schedule."""
    return pd.DataFrame([run_month(env, policy, m, k) for m in months for k in np.atleast_1d(members)])


class MonthRouter:
    """Several ERCOT weather-year environments behind one interface: ``reset`` picks the environment of
    the month's year; every other attribute is the current environment's (training on 2019-2022)."""

    def __init__(self, envs: dict):
        self.envs = dict(envs)
        self.current = next(iter(self.envs.values()))

    def reset(self, seed=None, options=None):
        month = (options or {}).get("month")
        if month is None:
            raise ValueError("MonthRouter needs options['month']")
        self.current = self.envs[int(str(month)[:4])]
        return self.current.reset(seed=seed, options=options)

    def __getattr__(self, name):
        return getattr(self.__dict__["current"], name)
