"""Fixed household and calendar splits for the Low Carbon London response model.

Households: 20% of each group (dToU and Std) are held out, drawn once with a fixed seed from
the household table of the meter panel. Held-out households are never used to fit or select
the response model; they are used once, for the final out-of-sample check (Fig. 4).

Months: the twelve trial months form four folds of three (every fourth month, so each fold
spans three seasons) for block cross-validation on the training households: fit on three
folds, predict the events of the fourth.

The assignment is regenerated from code, not stored in git (it is derived from the data).
:func:`split_digest` pins it: a test fails if the split ever changes.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SPLIT_SEED = 20261005
HOLDOUT_SHARE = 0.20
MONTH_FOLDS = {1: "A", 5: "A", 9: "A",
               2: "B", 6: "B", 10: "B",
               3: "C", 7: "C", 11: "C",
               4: "D", 8: "D", 12: "D"}
FOLDS = ("A", "B", "C", "D")


def household_split(hh: pd.DataFrame, share: float = HOLDOUT_SHARE, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Assign each household to 'train' or 'holdout', ``share`` of each group held out.

    ``hh`` needs columns hh_id and tou. The result does not depend on the row order of ``hh``.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for tou in (False, True):
        ids = np.sort(hh.loc[hh["tou"] == tou, "hh_id"].to_numpy())
        n_hold = int(round(share * len(ids)))
        hold = set(rng.permutation(ids)[:n_hold].tolist())
        parts.append(pd.DataFrame({"hh_id": ids, "tou": tou,
                                   "split": ["holdout" if i in hold else "train" for i in ids]}))
    return pd.concat(parts).sort_values("hh_id").reset_index(drop=True)


def split_digest(split: pd.DataFrame) -> str:
    """sha256 prefix of the sorted 'hh_id,split' lines."""
    lines = "\n".join(f"{i},{s}" for i, s in split.sort_values("hh_id")[["hh_id", "split"]].itertuples(index=False))
    return hashlib.sha256(lines.encode()).hexdigest()[:16]


def month_fold(ts_local: pd.Series) -> pd.Series:
    """Fold label (A-D) of each local timestamp's calendar month."""
    return ts_local.dt.month.map(MONTH_FOLDS)


def restrict(sample: pd.DataFrame, split: pd.DataFrame, which: str = "train") -> pd.DataFrame:
    """Keep the households of ``sample`` in split ``which`` and renumber h_idx within group."""
    keep = sample.merge(split.loc[split["split"] == which, ["hh_id"]], on="hh_id")
    keep = keep.sort_values(["tou", "hh_id"]).reset_index(drop=True)
    keep["h_idx"] = keep.groupby("tou").cumcount()
    return keep
