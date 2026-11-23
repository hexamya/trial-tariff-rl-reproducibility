"""Tests for the fixed S0 splits."""
import pandas as pd
import pytest

from ttrl.data import lcl_meters as lm
from ttrl.data import splits as sp

PINNED_DIGEST = "ba0bf981f2bf1333"  # the release's 5,561 households; must never change


def _households(n_std: int = 500, n_tou: int = 120) -> pd.DataFrame:
    return pd.DataFrame({"hh_id": range(1, n_std + n_tou + 1), "tou": [False] * n_std + [True] * n_tou})


def test_household_split_shares_and_determinism():
    hh = _households()
    s = sp.household_split(hh)
    counts = s.groupby(["tou", "split"]).size()
    assert counts[(False, "holdout")] == 100 and counts[(True, "holdout")] == 24
    assert sp.split_digest(sp.household_split(hh.sample(frac=1, random_state=3))) == sp.split_digest(s)
    assert sp.split_digest(sp.household_split(hh, seed=1)) != sp.split_digest(s)


def test_restrict_renumbers_within_group():
    hh = _households(10, 5)
    s = sp.household_split(hh)
    sample = hh.assign(h_idx=hh.groupby("tou").cumcount())
    tr = sp.restrict(sample, s, "train")
    assert set(tr["hh_id"]) == set(s.loc[s["split"] == "train", "hh_id"])
    for _, g in tr.groupby("tou"):
        assert g["h_idx"].tolist() == list(range(len(g)))


def test_month_folds_span_three_seasons():
    seasons = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
               6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    for f in sp.FOLDS:
        months = [m for m, k in sp.MONTH_FOLDS.items() if k == f]
        assert len(months) == 3 and len({seasons[m] for m in months}) == 3
    ts = pd.Series(pd.to_datetime(["2013-01-15", "2013-08-01"]))
    assert sp.month_fold(ts).tolist() == ["A", "D"]


@pytest.mark.skipif(not lm.HOUSEHOLDS_PATH.exists(), reason="meter panel not built")
def test_real_split_is_pinned():
    assert sp.split_digest(sp.household_split(lm.households())) == PINNED_DIGEST
