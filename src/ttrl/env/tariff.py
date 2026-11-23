"""Day-ahead tariff schedules on the LCL 2013 block grid, and the trial's schedule rules.

A tariff day runs from 02:00 to 02:00 local time and holds eight 3-hour blocks (02-05, 05-08, ...,
23-02), the grid 153 of the 161 events of the 2013 schedule were built on
(``docs/reports/lcl_schedule_summary.md``). Each block is Low, Normal or High (codes -1, 0, 1) at
the trial's prices. Half-hour slot ``s`` of a tariff day starts at 02:00 + 30 min x s, so slot s
lies in block ``s // 6``, which is also the kernel's ``hour8`` block.

:class:`TrialRules` keeps an RL agent inside what the trial actually did, so the response kernel
is never asked about a schedule it has not seen. Defaults come from the 2013 schedule
(``scripts/build_london_env.py`` prints them): at most one High event starting per day, High runs
of at most 24 h and Low runs of at most 30 h, at most 120 High and 306 Low half-hours per calendar
month, at most two days with both bands per ISO week.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttrl.data import lcl_features as lf

LOW, NORMAL, HIGH = -1, 0, 1
BLOCKS_PER_DAY = 8
HH_PER_BLOCK = 6
HH_PER_DAY = BLOCKS_PER_DAY * HH_PER_BLOCK
BLOCK_LABELS = ("02-05", "05-08", "08-11", "11-14", "14-17", "17-20", "20-23", "23-02")
PRICE = {LOW: lf.PRICE_LOW, NORMAL: lf.PRICE_NORMAL, HIGH: lf.PRICE_HIGH}  # GBP/kWh
PRICE_FLAT = lf.PRICE_FLAT_CONTROL


def slot_bands(blocks: np.ndarray) -> np.ndarray:
    """Band code per half-hour: (..., 8) blocks -> (..., 48) slots."""
    return np.repeat(np.asarray(blocks, dtype=np.int8), HH_PER_BLOCK, axis=-1)


def slot_prices(blocks: np.ndarray) -> np.ndarray:
    """Price per half-hour (GBP/kWh) for (..., 8) blocks."""
    b = slot_bands(blocks)
    return np.select([b == HIGH, b == LOW], [PRICE[HIGH], PRICE[LOW]], PRICE[NORMAL])


def runs(seq: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal non-Normal codes in a flat sequence: (band, start, length)."""
    seq = np.asarray(seq)
    out = []
    i, n = 0, len(seq)
    while i < n:
        if seq[i] == NORMAL:
            i += 1
            continue
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]:
            j += 1
        out.append((int(seq[i]), i, j - i + 1))
        i = j + 1
    return out


@dataclass(frozen=True)
class TrialRules:
    max_high_starts_per_day: int = 1
    max_high_run_blocks: int = 8      # 24 h
    max_low_run_blocks: int = 10      # 30 h
    max_high_hh_per_month: int = 120
    max_low_hh_per_month: int = 306
    max_double_days_per_week: int = 2

    def day_ok(self, history: np.ndarray, day: np.ndarray) -> bool:
        """Can ``day`` (8 codes) follow ``history`` ((n, 8) earlier days, may be empty)?

        Checks the per-day rules and the run lengths across the day boundary. Monthly and weekly
        counts are constraints of the CMDP, not hard rules (:meth:`counts`).
        """
        day = np.asarray(day)
        if day.shape != (BLOCKS_PER_DAY,) or not np.isin(day, (LOW, NORMAL, HIGH)).all():
            return False
        hist = np.asarray(history).reshape(-1)
        seq = np.r_[hist, day]
        starts_today = [r for r in runs(seq) if r[1] >= len(hist)]
        if sum(1 for r in starts_today if r[0] == HIGH) > self.max_high_starts_per_day:
            return False
        for band, _, length in runs(seq):
            lim = self.max_high_run_blocks if band == HIGH else self.max_low_run_blocks
            if length > lim:
                return False
        return True

    def counts(self, days: np.ndarray) -> dict[str, int]:
        """High / Low half-hours and days with both bands in ``days`` ((n, 8))."""
        d = np.asarray(days).reshape(-1, BLOCKS_PER_DAY)
        both = ((d == HIGH).any(1) & (d == LOW).any(1)).sum()
        return {"high_hh": int((d == HIGH).sum() * HH_PER_BLOCK), "low_hh": int((d == LOW).sum() * HH_PER_BLOCK),
                "double_days": int(both)}


def valid_day_patterns(rules: TrialRules = TrialRules()) -> np.ndarray:
    """Every 8-block day that satisfies the per-day rules on its own (no history): (n, 8)."""
    grid = np.array(np.meshgrid(*[[LOW, NORMAL, HIGH]] * BLOCKS_PER_DAY, indexing="ij")).reshape(BLOCKS_PER_DAY, -1).T
    keep = [rules.day_ok(np.empty((0, BLOCKS_PER_DAY)), d) for d in grid]
    return grid[np.array(keep)].astype(np.int8)
