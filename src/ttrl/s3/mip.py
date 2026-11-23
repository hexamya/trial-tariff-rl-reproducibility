"""Monthly tariff planning as a mixed-integer programme (HiGHS via scipy.optimize.milp).

One binary per (day, candidate pattern), one pattern per day. With per-pattern effects averaged over a
sample of ensemble members (sample-average approximation, as Kale et al.'s stochastic MIP), every term is
linear in the binaries:

* objective: London, the mean daily peak cut (%); ERCOT, minus fuel and CO2 cost, minus capacity value x
  month weight x the month's net-load peak (an auxiliary variable above every day's peak);
* revenue within the band of the flat rate (fixed denominator), for every planning member separately
  (robust to the response uncertainty the members carry) when per-member numerators are given;
* London: each exposure segment's mean bill change (households that do not respond) at most the cap;
* High and Low half-hours per month, days with both bands per ISO week (London);
* day-to-day rules through the previous day's trailing run (``patterns.compatibility``);
* fixed days (already posted, or masked curtailed days in ERCOT training years).

With ``soft_penalty`` set, the revenue band and the segment cap become soft: one slack per family,
charged at that penalty per unit of violation (used by the rolling MPC, where the rest of the month
may no longer allow the band after earlier decisions made on forecasts).

``MonthModel`` holds the coefficients; ``solve`` returns the pattern index of every day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from ttrl.env.tariff import HH_PER_BLOCK, HIGH, LOW, TrialRules
from ttrl.s3 import patterns as pt


@dataclass
class MonthModel:
    value: np.ndarray                    # (D, P) objective to maximise
    rev_num: np.ndarray                  # (D, P) revenue numerator, or (M, D, P) per planning member
    rev_den: float
    rev_tol: float
    patterns: np.ndarray                 # (P, 8)
    compat: np.ndarray                   # (N_STATES, P)
    prev_state: int = 0                  # trailing state of the day before day 0
    rev_fixed: float = 0.0               # numerator already earned (settled days outside the model)
    high_used: int = 0                   # half-hours already used this month
    low_used: int = 0
    seg: np.ndarray | None = None        # (S, D, P) segment bill ratio contributions
    seg_fixed: np.ndarray | None = None  # (S,) contributions of days outside the model
    seg_cap: float = 0.03
    week: np.ndarray | None = None       # (D,) ISO week id, for the double-day limit
    double_used: dict = field(default_factory=dict)
    peak: np.ndarray | None = None       # (D, P) day peak (ERCOT net load)
    peak_coef: float = 0.0               # $ per MW of the month's peak (value x month weight)
    peak_floor: float = 0.0              # month peak already reached on days outside the model
    fixed: dict = field(default_factory=dict)   # day -> pattern index
    rules: TrialRules = field(default_factory=TrialRules)
    integer_days: np.ndarray | None = None       # days whose binaries are integer (default all; others relaxed)
    soft_penalty: float | None = None            # objective units per unit of revenue / segment violation


def solve(m: MonthModel, time_limit: float = 60.0, mip_gap: float = 1e-3) -> tuple[np.ndarray, dict]:
    D, P = m.value.shape
    n_x = D * P
    use_peak = m.peak is not None and m.peak_coef > 0
    soft = m.soft_penalty is not None
    i_rev, i_seg = n_x + int(use_peak), n_x + int(use_peak) + 1          # slack columns when soft
    n = n_x + int(use_peak) + 2 * int(soft)
    c = np.zeros(n)
    c[:n_x] = -m.value.ravel()
    if use_peak:
        c[n_x] = m.peak_coef
    if soft:
        c[i_rev] = c[i_seg] = m.soft_penalty
    rows, lo, hi = [], [], []

    def add(row, a, b):
        rows.append(row)
        lo.append(a)
        hi.append(b)

    def xrow(coef_dp):
        r = np.zeros(n)
        r[:n_x] = np.asarray(coef_dp, float).ravel()
        return r

    one_day = sp.kron(sp.eye(D), np.ones((1, P)), format="csr")             # one pattern per day
    A = [sp.hstack([one_day, sp.csr_matrix((D, n - n_x))])]
    alo, ahi = [np.ones(D)], [np.ones(D)]
    for rv in (m.rev_num if m.rev_num.ndim == 3 else [m.rev_num]):
        lo_r, hi_r = 1 - m.rev_tol - m.rev_fixed / m.rev_den, 1 + m.rev_tol - m.rev_fixed / m.rev_den
        if soft:
            r = xrow(rv / m.rev_den)
            r[i_rev] = 1.0
            add(r, lo_r, np.inf)
            r = xrow(rv / m.rev_den)
            r[i_rev] = -1.0
            add(r, -np.inf, hi_r)
        else:
            add(xrow(rv / m.rev_den), lo_r, hi_r)
    hh = (m.patterns == HIGH).sum(1) * HH_PER_BLOCK
    lh = (m.patterns == LOW).sum(1) * HH_PER_BLOCK
    add(xrow(np.tile(hh, (D, 1))), -np.inf, m.rules.max_high_hh_per_month - m.high_used)
    add(xrow(np.tile(lh, (D, 1))), -np.inf, m.rules.max_low_hh_per_month - m.low_used)
    if m.seg is not None:
        fixed = np.zeros(len(m.seg)) if m.seg_fixed is None else m.seg_fixed
        for s in range(len(m.seg)):
            r = xrow(m.seg[s])
            if soft:
                r[i_seg] = -1.0
            add(r, -np.inf, 1 + m.seg_cap - fixed[s])
    if m.week is not None:
        both = ((m.patterns == HIGH).any(1) & (m.patterns == LOW).any(1)).astype(float)
        for w in np.unique(m.week):
            coef = np.zeros((D, P))
            coef[m.week == w] = both
            add(xrow(coef), -np.inf, m.rules.max_double_days_per_week - m.double_used.get(int(w), 0))
    trail = np.array([pt.trailing_state(p) for p in m.patterns])
    for d in range(D):                                                         # day-to-day rules
        for s in range(1, pt.N_STATES):
            bad = ~m.compat[s]
            if not bad.any():
                continue
            coef = np.zeros((D, P))
            coef[d, bad] = 1.0
            if d == 0:
                if m.prev_state == s:
                    add(xrow(coef), -np.inf, 0.0)
                continue
            if not (trail == s).any():
                continue
            coef[d - 1, trail == s] = 1.0
            add(xrow(coef), -np.inf, 1.0)
    if use_peak:
        for d in range(D):
            r = xrow(np.where(np.arange(D)[:, None] == d, m.peak, 0.0))
            r[n_x] = -1.0
            add(r, -np.inf, 0.0)
    A.append(sp.csr_matrix(np.array(rows)))
    alo.append(np.array(lo))
    ahi.append(np.array(hi))
    lb, ub = np.zeros(n), np.ones(n)
    for d, j in m.fixed.items():
        lb[d * P:(d + 1) * P] = 0.0
        ub[d * P:(d + 1) * P] = 0.0
        lb[d * P + j] = ub[d * P + j] = 1.0
    if use_peak:
        lb[n_x], ub[n_x] = m.peak_floor, np.inf
    if soft:
        ub[i_rev] = ub[i_seg] = np.inf
    integer = np.ones((D, P))
    if m.integer_days is not None:
        integer[:] = 0.0
        integer[np.asarray(m.integer_days, int)] = 1.0
    integrality = np.r_[integer.ravel(), np.zeros(n - n_x)]
    res = milp(c, constraints=LinearConstraint(sp.vstack(A, format="csr"), np.concatenate(alo), np.concatenate(ahi)),
               integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False})
    info = {"status": int(res.status), "message": res.message, "objective": None if res.x is None else -float(res.fun),
            "mip_gap": getattr(res, "mip_gap", None),
            "slack_revenue": float(res.x[i_rev]) if soft and res.x is not None else 0.0,
            "slack_segment": float(res.x[i_seg]) if soft and res.x is not None else 0.0}
    if res.x is None:
        return np.zeros(D, int), info
    choice = res.x[:n_x].reshape(D, P).argmax(axis=1)
    return choice, info
