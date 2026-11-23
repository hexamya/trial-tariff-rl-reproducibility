"""Assumed price-response models for agent B (the paper's comparison case).

Two families used in the tariff-design literature this paper responds to:

* **Elasticity matrix** (Aalami-style): hourly load responds to the relative price change of its
  own hour (self-elasticity) and of every other hour of the day (cross-elasticities, by period
  category):

      d_i = d0_i * (1 + own[c_i] * r_i + sum_{j != i} cross[c_i, c_j] * r_j),   r_j = (p_j - p0) / p0

  with hour categories c (valley / off-peak / peak) and the reference price p0 (the flat rate).
* **Linear own-price demand** (Nejad & Dehghani 2026): d_i = theta1_i - theta2 * p_i, written here
  through the own-price elasticity at the reference point, eps = theta2 * p0 / d0_i, so that
  d_i = d0_i * (1 + eps * r_i) with no shifting between hours.

:func:`aalami_2010` holds
Aalami's Table 1 values; Nejad & Dehghani give no parameter values, so the linear case takes a cited
elasticity. Hours are local clock hours of the tariff day (24 per day; the half-hour grid is
averaged to hours and the hourly effect applied to both half-hours). Aalami's periods are clock
hours of the Iranian peak day; here they are mapped onto the local load shape by rank with the same
period lengths (:func:`categories_by_load`).

Effects are percent changes of the counterfactual load, like the trial-estimated kernel, so both
plug into the same environments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.env.tariff import HH_PER_DAY, PRICE_FLAT, slot_prices

HOURS = 24
CATEGORIES = ("valley", "off-peak", "peak")


AALAMI_PERIOD_HOURS = (9, 10, 5)   # valley 00-09, off-peak 09-19, peak 19-24 on the Iranian peak day


def categories_by_load(baseline: np.ndarray, counts: tuple[int, int, int] = AALAMI_PERIOD_HOURS) -> np.ndarray:
    """Hour-of-tariff-day categories (0 valley, 1 off-peak, 2 peak) by rank of the mean baseline
    profile: the ``counts[0]`` lowest-load hours are valley, the ``counts[2]`` highest peak.
    ``baseline`` (n_days, 48)."""
    if sum(counts) != HOURS:
        raise ValueError(counts)
    hourly = np.asarray(baseline).reshape(-1, HOURS, 2).mean(axis=(0, 2))
    order = np.argsort(hourly)
    cat = np.empty(HOURS, int)
    v, o = counts[0], counts[0] + counts[1]
    cat[order[:v]], cat[order[v:o]], cat[order[o:]] = 0, 1, 2
    return cat


def aalami_2010(categories: np.ndarray, same_period_cross: float = 0.0, p0: float = PRICE_FLAT) -> "ElasticityMatrix":
    """Aalami et al. (2010), Table 1: self-elasticity -0.10 in every period; cross 0.016 peak /
    off-peak, 0.012 peak / valley, 0.01 off-peak / valley (Eq. 16 with A = pen = 0). The paper does
    not give the cross-elasticity between different hours of the same period; 0 by default
    (cross-elasticities are positive in the paper; the alternative reading is a sensitivity)."""
    own = np.array([-0.10, -0.10, -0.10])
    cross = np.array([[same_period_cross, 0.010, 0.012],
                      [0.010, same_period_cross, 0.016],
                      [0.012, 0.016, same_period_cross]])
    return ElasticityMatrix(own, cross, np.asarray(categories), p0=p0,
                            source="Aalami, Parsa Moghaddam, Yousefi 2010, EPSR 80:426-435, Table 1 and Eq. 16")


@dataclass(frozen=True)
class ElasticityMatrix:
    """Aalami-style hourly response. ``own[a]``: self-elasticity of a category-a hour; ``cross[a, b]``:
    response of a category-a hour to the relative price change of a *different* category-b hour
    (same-category pairs included; separate from the self-elasticity)."""
    own: np.ndarray               # (3,)
    cross: np.ndarray             # (3, 3)
    categories: np.ndarray        # (24,) hour-of-tariff-day categories
    p0: float = PRICE_FLAT
    source: str = "unverified"

    def matrix(self) -> np.ndarray:
        """The implied 24 x 24 hourly elasticity matrix (row: responding hour)."""
        c = self.categories
        M = np.asarray(self.cross, float)[c][:, c].copy()
        np.fill_diagonal(M, np.asarray(self.own, float)[c])
        return M

    def effect(self, blocks: np.ndarray) -> np.ndarray:
        """Percent effect (n_days, 48) for a block schedule (n_days, 8); days are independent
        (the matrix acts within the tariff day)."""
        p = slot_prices(blocks).reshape(-1, HOURS, 2).mean(axis=2)           # (n_days, 24)
        r = (p - self.p0) / self.p0
        hourly = r @ self.matrix().T                                         # (n_days, 24)
        return 100 * np.repeat(hourly, 2, axis=1)


@dataclass(frozen=True)
class LinearOwnPrice:
    """Linear demand with own-price response only: percent effect = 100 * eps * r per hour."""
    eps: float                    # own-price elasticity at the reference point (negative)
    p0: float = PRICE_FLAT
    source: str = "unverified"

    def effect(self, blocks: np.ndarray) -> np.ndarray:
        p = slot_prices(blocks)
        return 100 * self.eps * (p - self.p0) / self.p0


def clip_effect(effect_pct: np.ndarray, floor: float = -100.0) -> np.ndarray:
    """Load cannot fall below zero: assumed linear models can predict more than -100% at the trial's
    High price for large elasticities; such values are clipped and should be reported."""
    return np.maximum(effect_pct, floor)


def per_day(effect_pct: np.ndarray) -> np.ndarray:
    return np.asarray(effect_pct).reshape(-1, HH_PER_DAY)
