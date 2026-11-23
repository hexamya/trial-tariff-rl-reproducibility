"""Fast evaluation of the S1 response kernel on block schedules (numpy, no pandas design matrix).

:func:`ttrl.analysis.lcl_kernel.kernel_effect` builds a full design matrix through pandas; an RL
agent calls the response many thousands of times, so this module evaluates the same kernel
directly from a flat sequence of half-hour band codes. It supports the kernel S1 selected
(factors ``hour8`` and ``dur``) and reproduces ``kernel_effect`` exactly on the tariff-day grid
(``tests/test_env_fastkernel.py``):

* events are runs of consecutive half-hours in the same non-Normal band;
* an event half-hour gets the cell of its band and 3-hour block plus the deviation of the event's
  duration class (<= 3 h, 3-6 h, > 6 h; the first class is the reference);
* Normal half-hours 1-4 slots before an event, 1-4 and 5-8 slots after it get the window terms of
  that event's band (a half-hour in two windows of the same type and band counts once, as in
  ``lcl_did.event_design``).

Effects are percent changes of the counterfactual load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttrl.analysis import lcl_kernel as kn
from ttrl.env.tariff import HH_PER_BLOCK, HH_PER_DAY, HIGH, LOW, NORMAL

BAND_INDEX = {HIGH: 0, LOW: 1}
BAND_NAME = ("High", "Low")
WINDOW_NAMES = ("pre_2h", "post_0_2h", "post_2_4h")
WINDOW_SLOTS = {"pre_2h": (-4, -1), "post_0_2h": (1, 4), "post_2_4h": (5, 8)}  # relative to start / end


@dataclass
class KernelArrays:
    """Coefficients of one or more kernel members: in_event (M, 2, 8), dur_dev (M, 2, 3), window (M, 2, 3);
    band axis 0 = High, 1 = Low."""
    in_event: np.ndarray
    dur_dev: np.ndarray
    window: np.ndarray

    @property
    def n_members(self) -> int:
        return self.in_event.shape[0]

    def member(self, m: int) -> "KernelArrays":
        return KernelArrays(self.in_event[m:m + 1], self.dur_dev[m:m + 1], self.window[m:m + 1])


def from_coefficients(spec: kn.KernelSpec, coef: pd.Series | pd.DataFrame) -> KernelArrays:
    """Arrays from kernel coefficients (a Series for one member, a members x terms frame for many)."""
    if tuple(spec.factors) != ("hour8", "dur"):
        raise ValueError(f"fast kernel supports factors ('hour8', 'dur'), not {spec.factors}")
    C = coef.to_frame().T if isinstance(coef, pd.Series) else coef
    m = len(C)
    ie, dd, wd = np.zeros((m, 2, 8)), np.zeros((m, 2, 3)), np.zeros((m, 2, 3))
    for b, name in enumerate(BAND_NAME):
        for k, lab in enumerate(kn.HOUR8):
            ie[:, b, k] = C[f"{name}|hour8={lab}"].to_numpy(float)
        for k, lab in enumerate(kn.DUR[1:], start=1):
            dd[:, b, k] = C[f"{name}|dur={lab}"].to_numpy(float)
        for k, w in enumerate(WINDOW_NAMES):
            wd[:, b, k] = C[f"{name}_{w}"].to_numpy(float)
    return KernelArrays(ie, dd, wd)


def event_runs(bands: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal non-Normal codes in a flat half-hour sequence: (band, first, last)."""
    b = np.asarray(bands)
    edge = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    ends = np.r_[edge[1:] - 1, len(b) - 1]
    return [(int(b[s]), int(s), int(e)) for s, e in zip(edge, ends) if b[s] != NORMAL]


def duration_class(n_slots: int) -> int:
    """0: <= 3 h, 1: 3-6 h, 2: > 6 h (as lcl_kernel, cuts inclusive on the right)."""
    return int(np.digitize(n_slots / 2, kn.DUR_CUTS, right=True))


def effect(bands: np.ndarray, arrays: KernelArrays, slot0: int = 0) -> np.ndarray:
    """Percent effect per half-hour for every member: (M, T).

    ``bands``: flat half-hour band codes on the tariff-day grid; ``slot0``: slot of day of the
    first element (so the hour block of element t is ((slot0 + t) % 48) // 6).
    """
    b = np.asarray(bands)
    T = len(b)
    block = ((slot0 + np.arange(T)) % HH_PER_DAY) // HH_PER_BLOCK
    out = np.zeros((arrays.n_members, T))
    normal = b == NORMAL
    masks = np.zeros((2, 3, T), bool)
    for band, s, e in event_runs(b):
        bi = BAND_INDEX[band]
        idx = np.arange(s, e + 1)
        out[:, idx] += arrays.in_event[:, bi, block[idx]] + arrays.dur_dev[:, bi, duration_class(e - s + 1)][:, None]
        for k, w in enumerate(WINDOW_NAMES):
            lo, hi = WINDOW_SLOTS[w]
            ref = s if w == "pre_2h" else e
            j = np.arange(ref + lo, ref + hi + 1)
            j = j[(j >= 0) & (j < T)]
            masks[bi, k, j[normal[j]]] = True
    for bi in range(2):
        for k in range(3):
            if masks[bi, k].any():
                out[:, masks[bi, k]] += arrays.window[:, bi, k][:, None]
    return out
