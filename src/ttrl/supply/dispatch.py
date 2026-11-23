"""Merit-order commitment and economic dispatch of the ERCOT gas fleet.

Problem solved for every hour t independently (vectorised over hours):

    minimise  sum_i H_i(P_i)            (one fuel, one price: cost = heat input)
    s.t.      sum_i P_i = D_t,  Pmin_i <= P_i <= Pmax_i for committed units,
              P_i = 0 for units that are not committed.

1. Commitment (:func:`priority_commitment`, a priority list). Available units
   are ranked by their average heat rate at full load, H(Pmax)/Pmax. The
   cheapest units are committed until their summed Pmax covers the target
   plus a headroom (``headroom_mw``, MW; zero gives the textbook priority
   list). If the committed units' summed Pmin then exceeds the target, the
   most expensive committed units are dropped again, as long as the remaining
   capacity still covers the target. There are no start-up costs, minimum
   up/down times or ramp limits: each hour is committed from scratch.
   Alternatively the caller passes a committed set directly (e.g. the units
   that actually ran, to isolate the dispatch step).
2. Dispatch (:func:`economic_dispatch`): equal incremental heat rate. For a
   common marginal heat rate lambda, a convex quadratic unit runs at
   clip((lambda - b) / 2c, Pmin, Pmax) and a linear unit (c = 0) at Pmin
   below b and Pmax above it. Lambda is found by bisection, and the last
   bracket is closed by linear interpolation so that the energy balance holds
   exactly even when linear units make the supply curve step.

Curve parameters come as :class:`ttrl.supply.units.CurveArrays` that
broadcast to (hours, units), so hour-specific (e.g. temperature-dependent)
curves and limits need no change here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.supply.units import CurveArrays

STATUS_OK, STATUS_BELOW_MIN, STATUS_SHORT, STATUS_EMPTY = 0, 1, 2, 3
STATUS_NAMES = {STATUS_OK: "ok", STATUS_BELOW_MIN: "below committed Pmin",
                STATUS_SHORT: "short of capacity", STATUS_EMPTY: "nothing committed"}


@dataclass
class DispatchResult:
    """Hourly dispatch of N units over T hours.

    ``p`` (T, N) output in MW; ``committed`` (T, N); ``lam`` (T,) marginal
    incremental heat rate in MMBtu/MWh (NaN unless status is ok);
    ``status`` (T,) codes in :data:`STATUS_NAMES`; ``unserved_mw`` (T,) target
    minus dispatched output (positive only when short of capacity).
    """
    p: np.ndarray
    committed: np.ndarray
    lam: np.ndarray
    status: np.ndarray
    unserved_mw: np.ndarray

    def unit_heat(self, curves: CurveArrays) -> np.ndarray:
        """Heat input per unit and hour (T, N), MMBtu."""
        return curves.heat(self.p)

    def fleet_heat(self, curves: CurveArrays) -> np.ndarray:
        """Fleet heat input per hour (T,), MMBtu."""
        return self.unit_heat(curves).sum(axis=1)


def _unit_output(lam: np.ndarray, b: np.ndarray, inv2c: np.ndarray,
                 quad: np.ndarray, pmin: np.ndarray, pmax: np.ndarray) -> np.ndarray:
    """Output of every unit at marginal heat rate ``lam`` (T, 1)."""
    p = np.where(quad, (lam - b) * inv2c, np.where(lam > b, np.inf, -np.inf))
    return np.clip(p, pmin, pmax)


def economic_dispatch(target: np.ndarray, curves: CurveArrays,
                      committed: np.ndarray, n_iter: int = 50) -> DispatchResult:
    """Equal-incremental-heat-rate dispatch of committed units.

    Parameters
    ----------
    target : (T,) output to supply, MW (= MWh in the hour).
    curves : curve parameters broadcastable to (T, N).
    committed : (T, N) boolean mask of committed units.
    n_iter : bisection steps on lambda.

    Hours whose target is below the committed units' summed Pmin run every
    committed unit at Pmin scaled down pro rata (status ``below_min``); hours
    whose target exceeds summed Pmax run all at Pmax (status ``short``).
    """
    target = np.asarray(target, float)
    committed = np.asarray(committed, bool)
    t_n, u_n = committed.shape
    cv = curves.broadcast(t_n)
    b, c = cv.b, cv.c
    pmin = np.where(committed, cv.pmin, 0.0)
    pmax = np.where(committed, np.maximum(cv.pmax, cv.pmin), 0.0)
    quad = c > 0
    with np.errstate(divide="ignore"):
        inv2c = np.where(quad, 0.5 / np.where(quad, c, 1.0), 0.0)

    smin, smax = pmin.sum(axis=1), pmax.sum(axis=1)
    inc_lo = np.where(committed, cv.incremental(pmin), np.inf).min(axis=1)
    inc_hi = np.where(committed, cv.incremental(pmax), -np.inf).max(axis=1)
    has_units = committed.any(axis=1)
    lo = np.where(has_units, inc_lo - 1.0, 0.0)[:, None]
    hi = np.where(has_units, inc_hi + 1.0, 1.0)[:, None]

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        s = _unit_output(mid, b, inv2c, quad, pmin, pmax).sum(axis=1, keepdims=True)
        above = s > target[:, None]
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)

    p_lo = _unit_output(lo, b, inv2c, quad, pmin, pmax)
    p_hi = _unit_output(hi, b, inv2c, quad, pmin, pmax)
    s_lo, s_hi = p_lo.sum(axis=1), p_hi.sum(axis=1)
    gap = s_hi - s_lo
    w = np.where(gap > 0, (target - s_lo) / np.where(gap > 0, gap, 1.0), 0.0)
    p = p_lo + np.clip(w, 0.0, 1.0)[:, None] * (p_hi - p_lo)
    lam = 0.5 * (lo + hi)[:, 0]

    status = np.full(t_n, STATUS_OK, dtype=np.int8)
    below = has_units & (target < smin)
    short = has_units & (target > smax)
    if below.any():
        scale = np.where(smin > 0, target / np.where(smin > 0, smin, 1.0), 0.0)
        p[below] = pmin[below] * scale[below, None]
    if short.any():
        p[short] = pmax[short]
    status[below] = STATUS_BELOW_MIN
    status[short] = STATUS_SHORT
    status[~has_units] = STATUS_EMPTY
    p[~has_units] = 0.0
    lam = np.where(status == STATUS_OK, lam, np.nan)
    unserved = target - p.sum(axis=1)
    return DispatchResult(p=p, committed=committed & (p > 0), lam=lam,
                          status=status, unserved_mw=unserved)


def priority_commitment(target: np.ndarray, curves: CurveArrays,
                        available: np.ndarray,
                        headroom_mw: float | np.ndarray = 0.0) -> np.ndarray:
    """Committed units per hour from a full-load heat-rate priority list.

    Commits available units, cheapest first, until summed Pmax reaches
    ``target + headroom_mw``; then trims the most expensive ones while the
    committed Pmin exceeds the target and the remaining Pmax still covers it.
    Returns a (T, N) boolean mask.
    """
    target = np.asarray(target, float)
    available = np.asarray(available, bool)
    t_n, u_n = available.shape
    cv = curves.broadcast(t_n)
    key = np.where(available, cv.full_load_heat_rate(), np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    pmax_s = np.take_along_axis(np.where(available, cv.pmax, 0.0), order, axis=1)
    pmin_s = np.take_along_axis(np.where(available, cv.pmin, 0.0), order, axis=1)
    cum_max = np.cumsum(pmax_s, axis=1)
    cum_min = np.cumsum(pmin_s, axis=1)
    n_avail = available.sum(axis=1)

    def first_reaching(level: np.ndarray) -> np.ndarray:
        reach = cum_max >= level[:, None]
        return np.where(reach.any(axis=1), reach.argmax(axis=1) + 1, n_avail)

    need = target + np.broadcast_to(np.asarray(headroom_mw, float), target.shape)
    k_need = first_reaching(need)            # covers target + headroom
    k_cap = first_reaching(target)           # smallest set covering the target
    k_min = np.minimum((cum_min <= target[:, None]).sum(axis=1), n_avail)
    k = np.maximum(np.minimum(k_need, k_min), k_cap)

    committed_sorted = np.arange(u_n)[None, :] < k[:, None]
    committed = np.zeros_like(available)
    np.put_along_axis(committed, order, committed_sorted, axis=1)
    return committed & available


def dispatch(target: np.ndarray, curves: CurveArrays, available: np.ndarray,
             committed: np.ndarray | None = None,
             headroom_mw: float | np.ndarray = 0.0) -> DispatchResult:
    """Commit (priority list unless ``committed`` is given) and dispatch."""
    if committed is None:
        committed = priority_commitment(target, curves, available, headroom_mw)
    return economic_dispatch(target, curves, committed & available)


def fit_headroom(target: np.ndarray, committed_pmax: np.ndarray) -> tuple[float, float]:
    """OLS of committed capacity minus output on output: headroom = r0 + r1 * D.

    ``committed_pmax`` (T,) is the summed Pmax of the units that actually ran
    in each hour. Returns (r0 in MW, r1 dimensionless).
    """
    y = np.asarray(committed_pmax, float) - np.asarray(target, float)
    x = np.column_stack([np.ones_like(y), np.asarray(target, float)])
    ok = np.isfinite(y)
    r0, r1 = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
    return float(r0), float(r1)


def finite_difference_mhr(target: np.ndarray, curves: CurveArrays,
                          available: np.ndarray, delta_mw: float,
                          headroom: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Model marginal heat rate (MMBtu/MWh) by central difference in the target.

    Re-runs commitment and dispatch at ``target +/- delta_mw`` (headroom
    recomputed as r0 + r1 * target for each) and returns
    (H(D + delta) - H(D - delta)) / (2 delta). Hours where either run is not
    status ``ok`` are NaN.
    """
    out = []
    for sign in (+1.0, -1.0):
        d = np.asarray(target, float) + sign * delta_mw
        res = dispatch(d, curves, available, headroom_mw=headroom[0] + headroom[1] * d)
        out.append((res.fleet_heat(curves), res.status))
    (h_up, s_up), (h_dn, s_dn) = out
    ok = (s_up == STATUS_OK) & (s_dn == STATUS_OK)
    return np.where(ok, (h_up - h_dn) / (2.0 * delta_mw), np.nan)
