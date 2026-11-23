"""Price responses the environments can run on: the trial-estimated kernel or an assumed model.

Both return the percent change of counterfactual load per half-hour of a block schedule
(``effect(blocks, member) -> (n_days, 48)``), so the same environment judges agent A (trained on
the trial-estimated response) and agent B (trained on an assumed response), which is the paper's
comparison.

* :class:`TrialResponse`: the S1 kernel ensemble (:mod:`ttrl.env.fastkernel`); events and windows
  run across day boundaries; ``member`` picks the ensemble member.
* :class:`AssumedResponse`: an assumed model from :mod:`ttrl.env.assumed`; it acts within each
  tariff day and has no members. Effects below -100% (possible for large elasticities at the trial's
  High price) are clipped and counted.
* ``hot_multiplier``: factor applied to the effect on days flagged as outside the kernel's
  temperature support. The default factor is 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.env import assumed as am
from ttrl.env import fastkernel as fk
from ttrl.env.tariff import HH_PER_DAY, slot_bands


@dataclass
class TrialResponse:
    arrays: fk.KernelArrays
    name: str = "trial"

    @property
    def n_members(self) -> int:
        return self.arrays.n_members

    def effect(self, blocks: np.ndarray, member: int = 0) -> np.ndarray:
        eff = fk.effect(slot_bands(blocks).reshape(-1), self.arrays.member(int(member)))[0]
        return eff.reshape(-1, HH_PER_DAY)


@dataclass
class AssumedResponse:
    model: am.ElasticityMatrix | am.LinearOwnPrice
    name: str = "assumed"
    clipped: int = 0

    @property
    def n_members(self) -> int:
        return 1

    def effect(self, blocks: np.ndarray, member: int = 0) -> np.ndarray:
        raw = am.per_day(self.model.effect(np.asarray(blocks).reshape(-1, 8)))
        self.clipped += int((raw < -100).sum())
        return am.clip_effect(raw)


def with_hot_multiplier(effect_pct: np.ndarray, hot_days: np.ndarray, m: float) -> np.ndarray:
    """Scale the effect on flagged days (rows) by ``m``."""
    if m == 1.0:
        return effect_pct
    out = np.array(effect_pct, float, copy=True)
    out[np.asarray(hot_days, bool)] *= m
    return out
