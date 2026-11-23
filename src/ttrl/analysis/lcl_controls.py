"""Control-group checks for the S1 response: Economy 7-like controls and a synthetic control.

Both groups had smart meters and in-home displays, but only the dToU offer excluded Economy 7
(E7) customers (Low Carbon London Report A3). E7 households, often with storage
heating, would sit only in the Std control group and could create a seasonal, weather-driven
dToU-Std gap, and with it part of the negative Normal-price level term.

Two checks, both built from pre-trial (Jul-Dec 2012) readings only:

* **Night profile.** Share of Nov-Dec 2012 consumption between 00:00 and 07:00 (local time =
  GMT in those months), from household x month x half-hour x weekend mean profiles weighted by
  day counts. The dToU group, from which E7 was excluded, gives the reference distribution; Std
  households above its ``q`` quantile are flagged as E7-like and dropped from the control group.
* **Synthetic control.** Non-negative Std weights summing to one that reproduce the dToU group's
  mean pre-trial profile (month x half-hour x weekend, 576 moments). Solved as non-negative least
  squares with a heavily weighted sum-to-one row, then normalised. The weights replace the equal
  weights of the Std group in the group means of :mod:`ttrl.analysis.lcl_did`.

Bootstrap: households are resampled within group; the synthetic weights are re-solved on every
draw (a donor drawn twice is the same column, so the donor pool is the set of drawn households).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ttrl.data.lcl_meters import PANEL_PATH, connect

NIGHT_HH = range(0, 14)          # 00:00-07:00 local
NIGHT_MONTHS = (11, 12)          # Nov-Dec 2012, GMT
SUM_PENALTY = 1e3
HDD_BASE = 15.5                  # heating degree-day base, C


# ---------------------------------------------------------------- pre-trial profiles
def pretrial_profiles(cal: pd.DataFrame, sample: pd.DataFrame, panel: Path = PANEL_PATH) -> pd.DataFrame:
    """Mean kWh per household (rows, ordered as ``sample``) and (month, hh, weekend) cell
    (columns) over the pre-trial rows of ``cal``. Cells without a reading are NaN."""
    pre = cal.loc[cal["period"] == "pre", ["ts_label", "hh", "weekend"]].copy()
    pre["month"] = cal.loc[pre.index, "ts_local"].dt.month
    con = connect()
    con.register("pre_df", pre)
    con.register("smp_df", sample[["hh_id"]])
    df = con.execute(f"""
        SELECT m.hh_id, c.month, c.hh, c.weekend, avg(m.kwh) AS kwh
        FROM read_parquet('{panel.as_posix()}') m JOIN smp_df USING (hh_id) JOIN pre_df c USING (ts_label)
        WHERE m.kwh IS NOT NULL GROUP BY ALL""").df()
    wide = df.pivot_table(index="hh_id", columns=["month", "hh", "weekend"], values="kwh")
    cols = pd.MultiIndex.from_product([sorted(pre["month"].unique()), range(48), [False, True]],
                                      names=["month", "hh", "weekend"])
    return wide.reindex(index=sample["hh_id"].to_numpy(), columns=cols)


def day_counts(cal: pd.DataFrame) -> pd.Series:
    """Local days per (month, weekend) in the pre-trial rows."""
    pre = cal.loc[cal["period"] == "pre"]
    d = pre.drop_duplicates("date")
    return d.groupby([d["ts_local"].dt.month, d["weekend"]]).size().rename_axis(["month", "weekend"])


def fill_profiles(P: pd.DataFrame) -> pd.DataFrame:
    """Fill empty cells with the household's mean for the same (hh, weekend) over the other
    months, then with the cell's mean over households."""
    filled = P.copy()
    own = P.T.groupby(level=["hh", "weekend"]).mean().T          # households x (hh, weekend)
    for m in P.columns.get_level_values("month").unique():
        block = filled.loc[:, m]
        filled.loc[:, m] = block.fillna(own.reindex(columns=block.columns)).to_numpy()
    return filled.fillna(filled.mean())


def night_share(P: pd.DataFrame, cal: pd.DataFrame, months: tuple[int, ...] = NIGHT_MONTHS,
                night=NIGHT_HH) -> pd.Series:
    """Share of consumption in the night half-hours, from profiles weighted by day counts."""
    days = day_counts(cal)
    Pm = P.loc[:, P.columns.get_level_values("month").isin(months)]
    w = np.array([days.get((m, wk), 0) for m, _, wk in Pm.columns], float)
    is_night = np.isin(Pm.columns.get_level_values("hh"), list(night))
    V = Pm.to_numpy() * w
    with np.errstate(invalid="ignore"):  # households without a reading in those months -> NaN
        return pd.Series(np.nansum(V[:, is_night], 1) / np.nansum(V, 1), index=P.index, name="night_share")


def night_threshold(share: pd.Series, tou: pd.Series, q: float) -> float:
    """``q`` quantile of the night share among dToU households (E7 was excluded from dToU)."""
    return float(share[tou.to_numpy()].quantile(q))


# ---------------------------------------------------------------- synthetic control
def synthetic_weights(donors: np.ndarray, target: np.ndarray, penalty: float = SUM_PENALTY) -> np.ndarray:
    """argmin ||donors @ w - target|| with w >= 0 and sum(w) = 1 (penalised, then normalised).

    ``donors``: moments x donors; ``target``: moments.
    """
    A = np.vstack([donors, np.full((1, donors.shape[1]), penalty)])
    b = np.r_[target, penalty]
    w, _ = nnls(A, b, maxiter=50 * A.shape[1])
    if w.sum() <= 0:
        raise ValueError("synthetic control has no positive weight")
    return w / w.sum()


def sc_fit_stats(donors: np.ndarray, target: np.ndarray, w: np.ndarray) -> dict:
    """Fit of the weighted donors to the target: RMSE in kWh and % of the target mean, and the
    number of donors carrying weight."""
    r = donors @ w - target
    return {"rmse_kwh": float(np.sqrt(np.mean(r ** 2))), "rmse_pct": float(100 * np.sqrt(np.mean(r ** 2)) / target.mean()),
            "n_donors": int((w > 1e-6).sum()), "max_weight": float(w.max()),
            "eff_donors": float(1 / np.sum(w ** 2))}


# ---------------------------------------------------------------- weight rows for group_sums
def subset_weights(n_tou: int, keep_std: np.ndarray, n_boot: int, seed: int,
                   keep_tou: np.ndarray | None = None) -> dict[bool, np.ndarray]:
    """Row 0: the kept households of each group (all dToU unless ``keep_tou`` is given); rows 1..:
    resampling within group among the kept."""
    rng = np.random.default_rng(seed)
    keep = {False: np.asarray(keep_std, bool),
            True: np.ones(n_tou, bool) if keep_tou is None else np.asarray(keep_tou, bool)}
    out = {}
    for g in (True, False):  # dToU first, so the draws match the earlier all-dToU version
        k = int(keep[g].sum())
        w = np.zeros((n_boot + 1, len(keep[g])))
        w[0, keep[g]] = 1
        w[1:, keep[g]] = rng.multinomial(k, np.full(k, 1 / k), size=n_boot)
        out[g] = w.astype(np.float32)
    return out


def hdd_fit(cal: pd.DataFrame, sums, X: pd.DataFrame, temp: pd.Series, fe: str | None,
            level_row: pd.DataFrame, base: float = HDD_BASE) -> pd.DataFrame:
    """Add heating degree-days x 4-hour block to the DiD of :mod:`ttrl.analysis.lcl_did`.

    The HDD terms measure the dToU-minus-control difference in temperature sensitivity (kWh per
    household-half-hour per degree-day); a control group with extra heating load (Economy 7
    storage heaters) shows up as negative night-time terms. Returns the trial / post level terms
    (with percent, using ``level_row`` = the matching ``lcl_did.estimate`` table indexed by term)
    and the six HDD terms, with date-clustered, household-bootstrap and combined standard errors.
    """
    import statsmodels.api as sm

    from ttrl.analysis import lcl_did as did

    hdd = np.maximum(base - temp.reindex(pd.DatetimeIndex(cal["date"])).to_numpy(), 0)
    blk = (cal["hh"] // 8).to_numpy()
    H = pd.DataFrame({f"hdd_b{b}": hdd * (blk == b) for b in range(6)}, index=cal.index)
    delta = sums.delta()
    ok = np.isfinite(delta[0])
    Z = pd.concat([X, H, did.fe_dummies(cal, fe)], axis=1).loc[ok]
    Z = Z.loc[:, Z.abs().sum() > 0]
    res = sm.OLS(delta[0, ok], Z.to_numpy()).fit(cov_type="cluster",
                                                 cov_kwds={"groups": cal.loc[ok, "date"].factorize()[0]})
    B = delta[1:, ok] @ np.linalg.pinv(Z.to_numpy()).T
    names = [c for c in ("trial", "post") if c in Z.columns] + list(H.columns)
    idx = [list(Z.columns).index(c) for c in names]
    out = pd.DataFrame({"term": names, "coef_kwh": res.params[idx], "se_cluster_date": res.bse[idx],
                        "se_boot_household": B[:, idx].std(0, ddof=1)})
    out["se"] = np.hypot(out["se_cluster_date"], out["se_boot_household"])
    for t in ("trial", "post"):
        if t not in names:
            continue
        lv = level_row.loc[t, "tou_mean_kwh"]
        r = out["term"] == t
        c, s = out.loc[r, "coef_kwh"].iloc[0], out.loc[r, "se"].iloc[0]
        out.loc[r, "pct"], out.loc[r, "pct_lo"], out.loc[r, "pct_hi"] = (
            100 * c / (lv - c), 100 * (c - 1.96 * s) / (lv - c), 100 * (c + 1.96 * s) / (lv - c))
    return out


def sc_weights(M_tou: np.ndarray, M_std: np.ndarray, n_boot: int, seed: int,
               moments: np.ndarray | None = None) -> tuple[dict[bool, np.ndarray], list[dict]]:
    """Synthetic-control weight rows: row 0 on the full sample, rows 1.. on household resamples.

    ``M_tou``/``M_std``: households x moments (filled profiles). ``moments`` selects the columns
    used for fitting (default all). Returns the weight rows and the fit statistics of each row.
    """
    rng = np.random.default_rng(seed)
    cols = slice(None) if moments is None else moments
    n_t, n_s = len(M_tou), len(M_std)
    tou_rows, std_rows, stats = [], [], []
    for b in range(n_boot + 1):
        ct = np.ones(n_t) if b == 0 else rng.multinomial(n_t, np.full(n_t, 1 / n_t)).astype(float)
        cs = np.ones(n_s) if b == 0 else rng.multinomial(n_s, np.full(n_s, 1 / n_s)).astype(float)
        target = (ct @ M_tou[:, cols]) / ct.sum()
        pool = np.flatnonzero(cs > 0)
        w = synthetic_weights(M_std[pool][:, cols].T, target)
        row = np.zeros(n_s)
        row[pool] = w
        tou_rows.append(ct)
        std_rows.append(row)
        stats.append(sc_fit_stats(M_std[pool][:, cols].T, target, w))
    return {True: np.array(tou_rows, np.float32), False: np.array(std_rows, np.float32)}, stats


def stack_weights(parts: list[dict[bool, np.ndarray]]) -> tuple[dict[bool, np.ndarray], list[slice]]:
    """Concatenate weight rows of several variants; returns the stack and each variant's rows."""
    out = {g: np.vstack([p[g] for p in parts]).astype(np.float32) for g in (False, True)}
    spans, start = [], 0
    for p in parts:
        n = p[True].shape[0]
        spans.append(slice(start, start + n))
        start += n
    return out, spans
