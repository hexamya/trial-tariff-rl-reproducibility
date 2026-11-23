"""Day patterns for the optimisation baselines (MIP, MPC): candidate set, day-to-day compatibility and
per-pattern response effects.

A tariff day is one of the patterns allowed by the trial's per-day rules (``tariff.valid_day_patterns``).
The optimisation works with the patterns that have at most one run of each band in a day (as the 2013
schedule's days), which keeps the problem small without leaving the trial's support.

Whether a pattern may follow the previous day depends only on the previous day's trailing run (band and
length, ``trailing_state``), so day-to-day rules become one compatibility table (states x patterns).

``pattern_effects`` gives each pattern's percent effect on its own day with Normal days around it, for
every ensemble member; effects that spill into the neighbouring days (windows after a late event, runs
across midnight) are left out of the planner's model. The environment settles every schedule exactly.
"""
from __future__ import annotations

import numpy as np

from ttrl.env import fastkernel as fk
from ttrl.env.tariff import BLOCKS_PER_DAY, HH_PER_DAY, HIGH, LOW, NORMAL, TrialRules, runs, slot_bands, valid_day_patterns

N_STATES = 1 + 2 * BLOCKS_PER_DAY        # none, High run 1..8, Low run 1..8


def candidate_patterns(rules: TrialRules = TrialRules()) -> np.ndarray:
    """Valid day patterns with at most one High run and at most one Low run; row 0 is all-Normal."""
    out = []
    for p in valid_day_patterns(rules):
        r = runs(p)
        if sum(1 for b, _, _ in r if b == HIGH) <= 1 and sum(1 for b, _, _ in r if b == LOW) <= 1:
            out.append(p)
    out = np.array(out, np.int8)
    normal = np.flatnonzero((out == NORMAL).all(axis=1))[0]
    return np.r_[out[normal:normal + 1], np.delete(out, normal, axis=0)]


def trailing_state(p: np.ndarray) -> int:
    """0 if the day ends Normal, else 1..8 for a High run of that length at the end, 9..16 for Low."""
    p = np.asarray(p)
    if p[-1] == NORMAL:
        return 0
    n = 1
    while n < BLOCKS_PER_DAY and p[-1 - n] == p[-1]:
        n += 1
    return n if p[-1] == HIGH else BLOCKS_PER_DAY + n


def state_day(state: int) -> np.ndarray:
    """A representative day with the given trailing state."""
    d = np.zeros(BLOCKS_PER_DAY, np.int8)
    if state == 0:
        return d
    band, n = (HIGH, state) if state <= BLOCKS_PER_DAY else (LOW, state - BLOCKS_PER_DAY)
    d[BLOCKS_PER_DAY - n:] = band
    return d


def compatibility(patterns: np.ndarray, rules: TrialRules = TrialRules()) -> np.ndarray:
    """(N_STATES, P) bool: may the pattern follow a day with that trailing state?"""
    return np.array([[rules.day_ok(state_day(s)[None], q) for q in patterns] for s in range(N_STATES)])


def pattern_effects(patterns: np.ndarray, arrays: fk.KernelArrays) -> np.ndarray:
    """(M, P, 48) percent effect of each pattern on its own day, Normal days before and after."""
    out = np.zeros((arrays.n_members, len(patterns), HH_PER_DAY))
    pad = np.zeros(HH_PER_DAY, np.int8)
    for j, p in enumerate(patterns):
        if not (p != NORMAL).any():
            continue
        bands = np.r_[pad, slot_bands(p), pad]
        out[:, j] = fk.effect(bands, arrays)[:, HH_PER_DAY:2 * HH_PER_DAY]
    return out
