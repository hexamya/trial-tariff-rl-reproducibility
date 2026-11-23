"""Error band of the S4 model's marginal fuel, CO2 and CT hours per MWh.

A tariff run reads the dispatch model at the margin: how much fuel, CO2 and
simple-cycle CT running time one more (or one less) MWh of non-CHP gas output
costs in a given hour. This module measures how far that response is from
what CEMS shows and how uncertain the difference is. Tariff effects are only
meaningful where they are larger than this band.

Estimator, per stratum s of hours, on hour pairs (t-1, t) inside one
evaluation window (both hours in the window and in the same ``group``, e.g.
the same cross-validation fold), the pair belonging to the stratum and the
block of its later hour t:

- CEMS marginal a_s: OLS slope (with intercept) of the hour-to-hour change of
  the actual quantity (non-CHP heat input, CO2, number of CT units on) on the
  change of actual non-CHP gross output; the estimator of the dispatch step.
- model marginal m_s: mean over the same later hours t of the model's central
  difference m_t (target +/- 500 MW, commitment re-solved).
- error err_s = m_s - a_s (model minus CEMS; relative error m_s / a_s - 1).
- confirmed share beta_g of the model's hour-level variation inside a stratum,
  per load tercile g: the coefficient on (m_t - m_s) x dx when the actual
  change is regressed, stratum by stratum, on [1, dx, (m_t - m_s) dx]
  (pooled within g). beta = 1: the hour-to-hour differences of the model's
  marginal are real; beta = 0: they are noise (e.g. commitment steps).
- step-size part (when the pairs carry the model's response to the actual
  change, ``Pairs.step``): the model's change for the pair's own step,
  S_t = X_t(target_t) - X_t(target_t - dx), differs from m_t dx by curvature
  (large steps) and commitment jumps (small steps). Its confirmed share
  beta2_c, per class c of |dx| (``SIZE_EDGES``), is estimated jointly with
  beta_g: the actual change is regressed, within strata, on [1, dx,
  (m_t - m_s) dx by load tercile, (S_t - m_t dx) by size class].

Block bootstrap (:func:`bootstrap_weights`): blocks (local weeks by default
in the S4 scripts) are resampled with replacement and every statistic is
recomputed from per-block sufficient statistics with the same weights, so a
draw keeps the correlation between strata, metrics and beta.

Correction of a profile (:meth:`ErrorBand.correct`): with model changes
dX_t and gas-output changes dE_t, draw b gives

    sum_t [ dX_t - dE_t err_b(s_t) - (1 - beta_b(g_t)) (dX_t - dE_t m(s_t)) ],

i.e. per hour the CEMS stratum marginal plus the confirmed share of the
model's deviation from its stratum mean. With the step-size part and the
model's +/- 500 MW marginal m_t of each profile hour, the deviation is split:

    sum_t [ dE_t a_b(s_t) + beta_b(g_t) dE_t (m_t - m(s_t))
            + beta2_b(c_t) (dX_t - dE_t m_t) ],   a_b = m(s) - err_b,

with c_t the size class of |dE_t|. The band mixes the model's
systematic error, the noise in its hour-level answers and the sampling error
of the CEMS slope; it does not cover errors that hour-to-hour data cannot
show (e.g. a planned, repeated load shift met with different commitment).

Strata of the band (:func:`strata`): season of the local-standard-time month
x load tercile of non-CHP gross output (2024 edges) x hot hour (capacity-
weighted fleet temperature >= 30 C). Season and load are where the dispatch
step found the model's marginal error to vary; the earlier steps found the
temperature-aware model to overstate the CEMS slope in the hottest hours. A
cell with fewer than ``min_pairs`` pairs or ``min_blocks`` blocks takes the
error (and model mean) of its season x tercile parent; if that parent is also
too thin, those of all pairs. Reporting classes of fleet temperature
(:func:`temp_class`): cold < 10 C, mild 10-25 C, warm 25-30 C, hot >= 30 C.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

METRICS = ("fuel", "co2", "ct")
METRIC_UNITS = {"fuel": "MMBtu/MWh", "co2": "short tons/MWh", "ct": "CT units per MW"}
SEASONS = ("winter", "spring", "summer", "fall")
SEASON_OF_MONTH = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3}
HOT_EDGE = 30.0
CLASS_EDGES = (10.0, 25.0, 30.0)
CLASS_NAMES = ("cold", "mild", "warm", "hot")
SIZE_EDGES = (500.0, 1000.0, 2000.0)          # MW classes of |change| for the step-size part


# --------------------------------------------------------------------------
# Strata and pairs
# --------------------------------------------------------------------------
def temp_class(fleet_temp: np.ndarray, edges=CLASS_EDGES) -> np.ndarray:
    """Reporting class of fleet temperature: 0 cold, 1 mild, 2 warm, 3 hot."""
    return np.searchsorted(np.asarray(edges, float), np.asarray(fleet_temp, float), side="right")


def season_index(month_lst: np.ndarray) -> np.ndarray:
    """0 winter (Dec-Feb), 1 spring (Mar-May), 2 summer (Jun-Sep), 3 fall (Oct-Nov)."""
    return np.vectorize(SEASON_OF_MONTH.get, otypes=[int])(np.asarray(month_lst, int))


def strata(fleet_temp: np.ndarray, load: np.ndarray, month_lst: np.ndarray,
           load_edges, hot_edge: float = HOT_EDGE) -> np.ndarray:
    """Stratum index ((season x 2 + hot) x n_load + load bin), see module docstring."""
    n_load = len(load_edges) + 1
    lb = np.searchsorted(np.asarray(load_edges, float), np.asarray(load, float), side="right")
    hot = (np.asarray(fleet_temp, float) >= hot_edge).astype(int)
    return (season_index(month_lst) * 2 + hot) * n_load + lb


def parent_of(n_load: int) -> np.ndarray:
    """Season x load-bin parent (index season x n_load + bin) of every stratum."""
    k = np.arange(len(SEASONS) * 2 * n_load)
    return (k // (2 * n_load)) * n_load + k % n_load


def size_class(d_mwh: np.ndarray, edges=SIZE_EDGES) -> np.ndarray:
    """Class of |change| (MW): 0 below the first edge, ..., len(edges) above the last."""
    return np.searchsorted(np.asarray(edges, float), np.abs(np.asarray(d_mwh, float)), side="right")


def size_labels(edges=SIZE_EDGES) -> list[str]:
    e = [f"{v / 1000:g}" for v in edges]
    return [f"< {e[0]} GW"] + [f"{a}-{b} GW" for a, b in zip(e[:-1], e[1:])] + [f">= {e[-1]} GW"]


def stratum_labels(n_load: int) -> list[str]:
    """Labels of :func:`strata`, e.g. 'summer T3 hot' (hot = fleet >= 30 C)."""
    return [f"{s} T{k + 1}{' hot' if h else ''}" for s in SEASONS for h in (0, 1)
            for k in range(n_load)]


@dataclass
class Pairs:
    """Hour pairs of an evaluation window (one row per pair, later hour t).

    ``x`` change of actual non-CHP output (MWh); ``dy`` metric -> change of
    the actual quantity; ``m`` metric -> model marginal at hour t; ``block``
    bootstrap block id; ``t`` index of the later hour in the hour grid;
    ``step`` (optional) metric -> the model's change at hour t for the pair's
    own step, X_t(target_t) - X_t(target_t - x).
    """
    x: np.ndarray
    dy: dict[str, np.ndarray]
    m: dict[str, np.ndarray]
    block: np.ndarray
    t: np.ndarray
    step: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.x)


def make_pairs(target: np.ndarray, actual: dict[str, np.ndarray], model: dict[str, np.ndarray],
               window: np.ndarray, block: np.ndarray, group: np.ndarray | None = None,
               step: dict[str, np.ndarray] | None = None) -> Pairs:
    """Pairs (t-1, t) with both hours in ``window`` (and the same ``group``).

    ``actual`` and ``model`` map each metric to (T,) series: the actual level
    (heat, CO2, CT units on) and the model's marginal per MW; ``step`` (optional)
    the model's change for the actual change into each hour (see :class:`Pairs`).
    Pairs where any model value is NaN are dropped.
    """
    target = np.asarray(target, float)
    window = np.asarray(window, bool)
    ok = window[1:] & window[:-1]
    if group is not None:
        g = np.asarray(group)
        ok &= g[1:] == g[:-1]
    for k in model:
        ok &= np.isfinite(model[k][1:])
    for k in step or {}:
        ok &= np.isfinite(step[k][1:])
    t = np.flatnonzero(ok) + 1
    return Pairs(x=np.diff(target)[t - 1], dy={k: np.diff(np.asarray(v, float))[t - 1]
                                               for k, v in actual.items()},
                 m={k: np.asarray(v, float)[t] for k, v in model.items()},
                 block=np.asarray(block)[t], t=t,
                 step=None if step is None else {k: np.asarray(v, float)[t] for k, v in step.items()})


# --------------------------------------------------------------------------
# Sufficient statistics and bootstrap
# --------------------------------------------------------------------------
def _sums(idx: np.ndarray, size: int, cols: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.bincount(idx, weights=c, minlength=size) for c in cols], axis=-1)


def block_stats(pairs: Pairs, stratum: np.ndarray, n_strata: int,
                metrics=METRICS) -> tuple[np.ndarray, np.ndarray]:
    """Per (block, stratum) sums for slopes and model means.

    ``stratum`` (n_pairs,). Returns (block ids, stats) with stats of shape
    (n_blocks, n_strata, 3 + 3 M): n, Sx, Sxx, then per metric Sy, Sxy, Sm.
    """
    blocks, bi = np.unique(pairs.block, return_inverse=True)
    idx = bi * n_strata + np.asarray(stratum, int)
    cols = [np.ones_like(pairs.x), pairs.x, pairs.x ** 2]
    for k in metrics:
        cols += [pairs.dy[k], pairs.x * pairs.dy[k], pairs.m[k]]
    stats = _sums(idx, len(blocks) * n_strata, cols)
    return blocks, stats.reshape(len(blocks), n_strata, len(cols))


def slopes_from_stats(s: np.ndarray, metrics=METRICS) -> dict[str, np.ndarray]:
    """CEMS slope ``a_<k>`` and model mean ``m_<k>`` from summed statistics (..., F)."""
    n, sx, sxx = s[..., 0], s[..., 1], s[..., 2]
    den = n * sxx - sx ** 2
    out = {"n": n}
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, k in enumerate(metrics):
            sy, sxy, sm = s[..., 3 + 3 * j], s[..., 4 + 3 * j], s[..., 5 + 3 * j]
            out[f"a_{k}"] = np.where(den > 0, (n * sxy - sx * sy) / den, np.nan)
            out[f"m_{k}"] = np.where(n > 0, sm / n, np.nan)
    return out


def bootstrap_weights(n_blocks: int, n_draws: int, seed: int = 0) -> np.ndarray:
    """(n_draws, n_blocks) multinomial counts: blocks resampled with replacement."""
    if n_blocks == 0:
        raise ValueError("no blocks to bootstrap")
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_draws).astype(float)


def bootstrap(stats: np.ndarray, n_draws: int = 2000, seed: int = 0,
              metrics=METRICS) -> tuple[dict, dict]:
    """Point estimates and block-bootstrap draws from :func:`block_stats` output.

    Returns (point, draws): dicts of ``n``, ``a_<k>``, ``m_<k>`` with shapes
    (S,) and (n_draws, S).
    """
    point = slopes_from_stats(stats.sum(axis=0), metrics)
    w = bootstrap_weights(stats.shape[0], n_draws, seed)
    draws = slopes_from_stats(np.einsum("bn,nsf->bsf", w, stats), metrics)
    return point, draws


def within_stats(pairs: Pairs, stratum: np.ndarray, n_strata: int, m_bar: np.ndarray,
                 metrics=METRICS) -> np.ndarray:
    """Per (block, stratum) sums for the within-stratum regression of the
    actual change on [1, dx, z], z = (m_t - m_bar_s) dx.

    ``m_bar`` (n_strata, M): the model mean per stratum. Returns
    (n_blocks, n_strata, 3 + 6 M): n, Sx, Sxx, then per metric Sz, Sxz, Szz,
    Sy, Sxy, Szy.
    """
    blocks, bi = np.unique(pairs.block, return_inverse=True)
    st = np.asarray(stratum, int)
    idx = bi * n_strata + st
    cols = [np.ones_like(pairs.x), pairs.x, pairs.x ** 2]
    for j, k in enumerate(metrics):
        z = (pairs.m[k] - m_bar[st, j]) * pairs.x
        y = pairs.dy[k]
        cols += [z, pairs.x * z, z * z, y, pairs.x * y, z * y]
    return _sums(idx, len(blocks) * n_strata, cols).reshape(len(blocks), n_strata, len(cols))


def within_beta(s: np.ndarray, group: np.ndarray, n_groups: int, metrics=METRICS) -> np.ndarray:
    """Confirmed share beta per group (..., G, M) from summed :func:`within_stats` (..., S, F).

    Frisch-Waugh within each stratum: z and y are residualised on [1, dx],
    then beta = sum(z~ y~) / sum(z~^2) over the strata of a group. A group
    without within-stratum variation of the model marginal gets beta = 1.
    """
    n, sx, sxx = s[..., 0], s[..., 1], s[..., 2]
    det = n * sxx - sx ** 2
    ok = det > 0
    safe = np.where(ok, det, 1.0)
    out = []
    for j in range(len(metrics)):
        sz, sxz, szz, sy, sxy, szy = (s[..., 3 + 6 * j + i] for i in range(6))
        quad = lambda u1, u2, v1, v2: (sxx * u1 * v1 - sx * u1 * v2 - sx * u2 * v1 + n * u2 * v2) / safe  # noqa: E731
        zz = np.where(ok, szz - quad(sz, sxz, sz, sxz), 0.0)
        zy = np.where(ok, szy - quad(sz, sxz, sy, sxy), 0.0)
        num = np.stack([zy[..., group == g].sum(axis=-1) for g in range(n_groups)], axis=-1)
        den = np.stack([zz[..., group == g].sum(axis=-1) for g in range(n_groups)], axis=-1)
        scale = np.abs(den).max() if np.size(den) else 0.0
        out.append(np.where(den > 1e-12 * max(scale, 1e-300), num / np.where(den > 0, den, 1.0), 1.0))
    return np.stack(out, axis=-1)


def joint_stats(pairs: Pairs, stratum: np.ndarray, n_strata: int, m_bar: np.ndarray,
                size_edges=SIZE_EDGES, metrics=METRICS) -> np.ndarray:
    """Per (block, stratum, metric) Gram matrices of [1, dx, z1, z2_0 .. z2_{C-1}, y] with
    z1 = (m_t - m_bar_s) dx and z2_c = (S_t - m_t dx) in size class c (0 elsewhere).

    Returns (n_blocks, n_strata, M, P, P), P = 4 + C.
    """
    blocks, bi = np.unique(pairs.block, return_inverse=True)
    st = np.asarray(stratum, int)
    idx = bi * n_strata + st
    cls = size_class(pairs.x, size_edges)
    n_cls = len(size_edges) + 1
    out = []
    for j, k in enumerate(metrics):
        z1 = (pairs.m[k] - m_bar[st, j]) * pairs.x
        z2 = pairs.step[k] - pairs.m[k] * pairs.x
        v = np.column_stack([np.ones_like(pairs.x), pairs.x, z1]
                            + [np.where(cls == c, z2, 0.0) for c in range(n_cls)] + [pairs.dy[k]])
        g = np.zeros((len(blocks) * n_strata, v.shape[1], v.shape[1]))
        np.add.at(g, idx, v[:, :, None] * v[:, None, :])
        out.append(g.reshape(len(blocks), n_strata, v.shape[1], v.shape[1]))
    return np.stack(out, axis=2)


def joint_beta(g: np.ndarray, group: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Confirmed shares from summed :func:`joint_stats` (..., S, M, P, P).

    Within each stratum the regressors and y are residualised on [1, dx]
    (Schur complement); the residual cross-products are pooled over strata
    with z1 in its stratum's load group, and y is regressed on [z1 by group,
    z2 by size class]. Returns beta (..., G, M) and beta2 (..., C, M). A
    regressor without residual variation gets 1 (as in :func:`within_beta`).
    """
    g = np.asarray(g, float)
    n_reg = g.shape[-1] - 3                         # z1 + C size classes
    n_cls = n_reg - 1
    a, b = slice(0, 2), slice(2, g.shape[-1])
    gaa = g[..., a, a]
    det = gaa[..., 0, 0] * gaa[..., 1, 1] - gaa[..., 0, 1] ** 2
    ok = det > 1e-9 * np.maximum(gaa[..., 0, 0] * gaa[..., 1, 1], 1e-300)
    inv = np.stack([np.stack([gaa[..., 1, 1], -gaa[..., 0, 1]], -1),
                    np.stack([-gaa[..., 1, 0], gaa[..., 0, 0]], -1)], -2) / np.where(ok, det, 1.0)[..., None, None]
    res = g[..., b, b] - g[..., b, a] @ inv @ g[..., a, b]
    res = np.where(ok[..., None, None], res, 0.0)   # (..., S, M, R+1, R+1)
    k = n_groups + n_cls
    lead = res.shape[:-4]
    n_m = res.shape[-3]
    zz = np.zeros(lead + (n_m, k, k))
    zy = np.zeros(lead + (n_m, k))
    for s in range(res.shape[-4]):
        cols = np.r_[group[s], n_groups + np.arange(n_cls)]
        r = res[..., s, :, :, :]
        zz[..., cols[:, None], cols[None, :]] += r[..., :n_reg, :n_reg]
        zy[..., cols] += r[..., :n_reg, n_reg]
    diag = np.diagonal(zz, axis1=-2, axis2=-1)                  # (..., M, k)
    scale = np.moveaxis(np.abs(diag), -2, 0).reshape(n_m, -1).max(axis=1)   # per metric
    dead = diag <= 1e-12 * np.maximum(scale, 1e-300)[:, None]
    eye = np.eye(k, dtype=bool)
    zz = np.where(dead[..., :, None] | dead[..., None, :], np.where(eye, 1.0, 0.0), zz)
    zy = np.where(dead, 1.0, zy)
    beta = np.linalg.solve(zz, zy[..., None])[..., 0]          # (..., M, k)
    beta = np.moveaxis(beta, -2, -1)                            # (..., k, M)
    return beta[..., :n_groups, :], beta[..., n_groups:, :]


def band_table(point: dict, draws: dict, labels: list[str], level: float = 0.95,
               metrics=METRICS) -> pd.DataFrame:
    """Per stratum: pairs, model and CEMS marginal (with its interval), error
    (model - CEMS) with its bootstrap interval, and the relative error m/a - 1."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2

    def q(v, p):
        return np.nanquantile(v, p) if np.isfinite(v).any() else np.nan
    rows = []
    for s, lab in enumerate(labels):
        row = {"stratum": lab, "pairs": int(point["n"][s])}
        for k in metrics:
            a, m = point[f"a_{k}"][s], point[f"m_{k}"][s]
            aa, mm = draws[f"a_{k}"][:, s], draws[f"m_{k}"][:, s]
            e = mm - aa
            with np.errstate(divide="ignore", invalid="ignore"):
                r = mm / aa - 1
            row.update({f"{k}_model": m, f"{k}_cems": a, f"{k}_err": m - a,
                        f"{k}_cems_lo": q(aa, lo_q), f"{k}_cems_hi": q(aa, hi_q),
                        f"{k}_err_lo": q(e, lo_q), f"{k}_err_hi": q(e, hi_q),
                        f"{k}_rel": 100 * (m / a - 1) if a else np.nan,
                        f"{k}_rel_lo": 100 * q(r, lo_q), f"{k}_rel_hi": 100 * q(r, hi_q)})
        rows.append(row)
    return pd.DataFrame(rows)


def shift_table(point: dict, draws: dict, labels: list[str], moves: list[tuple[int, int]],
                level: float = 0.95, metric: str = "fuel") -> pd.DataFrame:
    """Fuel (or other metric) saved per MWh moved from stratum i to stratum j.

    Saving = marginal(i) - marginal(j), model and CEMS; error = model - CEMS
    with its bootstrap interval (draw-aligned, so the two strata's errors are
    combined with their correlation).
    """
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    a, m = draws[f"a_{metric}"], draws[f"m_{metric}"]
    rows = []
    for i, j in moves:
        sm = point[f"m_{metric}"][i] - point[f"m_{metric}"][j]
        sa = point[f"a_{metric}"][i] - point[f"a_{metric}"][j]
        e = (m[:, i] - m[:, j]) - (a[:, i] - a[:, j])
        rows.append({"from": labels[i], "to": labels[j], "model saving": sm, "CEMS saving": sa,
                     "error": sm - sa, "error lo": np.nanquantile(e, lo_q),
                     "error hi": np.nanquantile(e, hi_q),
                     "error %": 100 * (sm / sa - 1) if sa else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The band used by the tariff function
# --------------------------------------------------------------------------
def _stack(d: dict) -> np.ndarray:
    return np.stack([d[f"m_{m}"] - d[f"a_{m}"] for m in METRICS], axis=-1)


def _means(d: dict) -> np.ndarray:
    return np.stack([d[f"m_{m}"] for m in METRICS], axis=-1)


@dataclass
class ErrorBand:
    """Bootstrap draws of the model's marginal error by stratum, and the
    confirmed share of its hour-level variation.

    ``err`` (n_draws, n_strata, 3): model minus CEMS marginal for
    :data:`METRICS` (fuel MMBtu/MWh, CO2 short tons/MWh, CT units per MW);
    ``point`` (n_strata, 3) the same at the full sample; ``m_bar`` (n_strata, 3)
    the model's mean marginal per stratum. ``beta`` (n_draws, n_load, 3) and
    ``beta_point`` (n_load, 3): confirmed share per load tercile. ``beta2``
    (n_draws, n_size, 3) and ``beta2_point``: confirmed share of the step-size
    part per class of |change| (``size_edges``); None for a band without it. Strata follow
    :func:`strata` with ``load_edges`` and ``hot_edge``; ``n_pairs`` and
    ``n_blocks`` count each cell's data; ``pooled`` lists cells that took
    their season x tercile parent's error and model mean (or, if the parent
    is too thin, those of all pairs). ``meta`` holds provenance.
    """
    load_edges: tuple[float, ...]
    hot_edge: float
    err: np.ndarray
    point: np.ndarray
    m_bar: np.ndarray
    beta: np.ndarray
    beta_point: np.ndarray
    n_pairs: np.ndarray
    n_blocks: np.ndarray
    pooled: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    beta2: np.ndarray | None = None
    beta2_point: np.ndarray | None = None
    size_edges: tuple[float, ...] = SIZE_EDGES

    @property
    def labels(self) -> list[str]:
        return stratum_labels(self.n_load)

    @property
    def n_load(self) -> int:
        return len(self.load_edges) + 1

    @property
    def n_draws(self) -> int:
        return self.err.shape[0]

    def stratum(self, fleet_temp: np.ndarray, load: np.ndarray, month_lst: np.ndarray) -> np.ndarray:
        """Stratum of hours with fleet temperature (C), non-CHP output (MW) and LST month."""
        return strata(fleet_temp, load, month_lst, self.load_edges, self.hot_edge)

    def correct(self, d_model: np.ndarray, d_mwh: np.ndarray, stratum: np.ndarray,
                m_hour: np.ndarray | None = None) -> np.ndarray:
        """(n_draws, 3) corrected totals of a profile (see module docstring).

        ``d_model`` (H, 3): the model's change per hour of fuel (MMBtu), CO2
        (short tons) and CT units on; ``d_mwh`` (H,): change of non-CHP gas
        output (MWh); ``stratum`` (H,); ``m_hour`` (H, 3): the model's +/- 500 MW
        marginal in each hour (NaN: use the stratum mean), needed for the
        step-size part; without it (or without ``beta2``) the whole deviation
        from the stratum mean is scaled by beta.
        """
        dx = np.asarray(d_model, float)
        de = np.asarray(d_mwh, float)
        st = np.asarray(stratum, int)
        g = st % self.n_load
        k = len(self.labels)
        e_per = np.bincount(st, weights=de, minlength=k)                       # MWh per stratum
        bias = np.einsum("s,bsm->bm", e_per, self.err)

        def by(idx, v, n):
            return np.stack([np.bincount(idx, weights=v[:, j], minlength=n) for j in range(v.shape[1])], axis=-1)
        if self.beta2 is None or m_hour is None:
            dev = dx - de[:, None] * self.m_bar[st]                            # (H, 3)
            noise = np.einsum("gm,bgm->bm", by(g, dev, self.n_load), 1.0 - self.beta)
            return dx.sum(axis=0)[None, :] - bias - noise
        mh = np.asarray(m_hour, float)
        mh = np.where(np.isfinite(mh), mh, self.m_bar[st])
        dev1 = de[:, None] * (mh - self.m_bar[st])                             # hour-level part
        dev2 = dx - de[:, None] * mh                                           # step-size part
        c = size_class(de, self.size_edges)
        noise = (np.einsum("gm,bgm->bm", by(g, dev1, self.n_load), 1.0 - self.beta)
                 + np.einsum("cm,bcm->bm", by(c, dev2, len(self.size_edges) + 1), 1.0 - self.beta2))
        return dx.sum(axis=0)[None, :] - bias - noise

    @classmethod
    def from_pairs(cls, pairs: Pairs, fleet_temp: np.ndarray, load: np.ndarray,
                   month_lst: np.ndarray, load_edges, hot_edge: float = HOT_EDGE,
                   n_draws: int = 2000, seed: int = 0, min_pairs: int = 100,
                   min_blocks: int = 8, meta: dict | None = None,
                   size_edges=SIZE_EDGES) -> "ErrorBand":
        """Build the band from validation pairs (strata of the later hour); with ``pairs.step``
        the confirmed shares are estimated jointly with the step-size part."""
        n_load = len(load_edges) + 1
        st = strata(fleet_temp[pairs.t], load[pairs.t], month_lst[pairs.t], load_edges, hot_edge)
        k = len(SEASONS) * 2 * n_load
        par = parent_of(n_load)
        _, stats = block_stats(pairs, st, k)
        _, pstats = block_stats(pairs, par[st], len(SEASONS) * n_load)
        _, astats = block_stats(pairs, np.zeros(len(st), int), 1)
        w = bootstrap_weights(stats.shape[0], n_draws, seed)
        agg = lambda s: (slopes_from_stats(s.sum(axis=0)),                          # noqa: E731
                         slopes_from_stats(np.einsum("bn,nsf->bsf", w, s)))
        (point, draws), (ppoint, pdraws), (apoint, adraws) = agg(stats), agg(pstats), agg(astats)
        err, pt, mb = _stack(draws), _stack(point), _means(point)
        n_blocks = (stats[:, :, 0] > 0).sum(axis=0)
        pn_blocks = (pstats[:, :, 0] > 0).sum(axis=0)
        thin = lambda n, nb, e: n < min_pairs or nb < min_blocks or not np.isfinite(e).all()  # noqa: E731
        pooled = [s for s in range(k) if thin(point["n"][s], n_blocks[s], err[:, s])]
        plabels = [f"{s} T{j + 1}" for s in SEASONS for j in range(n_load)]
        pooled_to = {}
        for s in pooled:
            q = par[s]
            if thin(ppoint["n"][q], pn_blocks[q], _stack(pdraws)[:, q]):   # parent too thin: all pairs
                err[:, s], pt[s], mb[s] = _stack(adraws)[:, 0], _stack(apoint)[0], _means(apoint)[0]
                pooled_to[str(s)] = "all pairs"
            else:
                err[:, s], pt[s], mb[s] = _stack(pdraws)[:, q], _stack(ppoint)[q], _means(ppoint)[q]
                pooled_to[str(s)] = plabels[q]
        # Confirmed share of the hour-level variation, per load tercile, same draws.
        group = np.arange(k) % n_load
        beta2 = beta2_point = None
        if pairs.step is None:
            wstats = within_stats(pairs, st, k, mb)
            beta_point = within_beta(wstats.sum(axis=0), group, n_load)
            beta = within_beta(np.einsum("bn,nsf->bsf", w, wstats), group, n_load)
        else:
            js = joint_stats(pairs, st, k, mb, size_edges)
            beta_point, beta2_point = joint_beta(js.sum(axis=0), group, n_load)
            beta, beta2 = joint_beta(np.einsum("bn,nsmpq->bsmpq", w, js), group, n_load)
        n_size = np.bincount(size_class(pairs.x, size_edges), minlength=len(size_edges) + 1)
        return cls(tuple(map(float, load_edges)), float(hot_edge), err, pt, mb, beta, beta_point,
                   point["n"].astype(int), n_blocks.astype(int), pooled,
                   dict(meta or {}, min_pairs=min_pairs, min_blocks=min_blocks, seed=seed,
                        n_draws=n_draws, pooled_to=pooled_to, pairs_by_size=n_size.tolist()),
                   beta2, beta2_point, tuple(map(float, size_edges)))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the draws to Parquet (long format) with the strata in the metadata."""
        b, s = np.meshgrid(np.arange(self.n_draws), np.arange(len(self.labels)), indexing="ij")
        df = pd.DataFrame({"draw": b.ravel(), "stratum": s.ravel()})
        for j, m in enumerate(METRICS):
            df[f"err_{m}"] = self.err[:, :, j].ravel()
        meta = {"load_edges": list(self.load_edges), "hot_edge": self.hot_edge,
                "point": self.point.tolist(), "m_bar": self.m_bar.tolist(),
                "beta": self.beta.tolist(), "beta_point": self.beta_point.tolist(),
                "n_pairs": self.n_pairs.tolist(), "n_blocks": self.n_blocks.tolist(),
                "pooled": list(map(int, self.pooled)), "labels": self.labels,
                "metric_units": METRIC_UNITS, "meta": self.meta, "size_edges": list(self.size_edges),
                "beta2": None if self.beta2 is None else self.beta2.tolist(),
                "beta2_point": None if self.beta2_point is None else self.beta2_point.tolist()}
        tab = pa.Table.from_pandas(df, preserve_index=False)
        tab = tab.replace_schema_metadata({**(tab.schema.metadata or {}),
                                           b"ttrl_error_band": json.dumps(meta).encode()})
        pq.write_table(tab, path)

    @classmethod
    def load(cls, path: str | Path) -> "ErrorBand":
        """Read a band written by :meth:`save`."""
        tab = pq.read_table(path)
        meta = json.loads(tab.schema.metadata[b"ttrl_error_band"])
        df = tab.to_pandas().sort_values(["draw", "stratum"])
        n_d, n_s = df["draw"].nunique(), df["stratum"].nunique()
        err = np.stack([df[f"err_{m}"].to_numpy().reshape(n_d, n_s) for m in METRICS], axis=-1)
        return cls(tuple(meta["load_edges"]), float(meta["hot_edge"]), err,
                   np.asarray(meta["point"], float), np.asarray(meta["m_bar"], float),
                   np.asarray(meta["beta"], float), np.asarray(meta["beta_point"], float),
                   np.asarray(meta["n_pairs"], int), np.asarray(meta["n_blocks"], int),
                   meta["pooled"], meta["meta"],
                   None if meta.get("beta2") is None else np.asarray(meta["beta2"], float),
                   None if meta.get("beta2_point") is None else np.asarray(meta["beta2_point"], float),
                   tuple(meta.get("size_edges", SIZE_EDGES)))
