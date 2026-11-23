"""S1 response kernel: heterogeneous event effects on a percent scale, chosen by month-fold CV.

Outcome
-------
The kernel is estimated on the same dToU-minus-Std difference of household-demeaned group means
as :mod:`ttrl.analysis.lcl_did`, divided by a proxy of the dToU counterfactual:

    y_t = 100 * Delta_t / P_t,   P_t = (dToU mean load - dToU mean deviation) + Std mean deviation,

i.e. the dToU households' own half-hour-of-day x weekend means plus the Std group's deviation from
its means at t. Coefficients are then percent of the counterfactual, so heterogeneity (by hour,
temperature, ...) describes a multiplicative response that S2 can apply to another baseline
load. For the constant kernel this is the percentage the DiD module reports, to first order.

Kernel
------
For each band b in (High, Low) the in-event effect is built from factors (``FACTORS``):

* ``hour4`` / ``hour8``: block of the half-hour's local time (6-hour blocks starting 23:00,
  3-hour blocks starting 02:00, the grid the 2013 schedule was built on);
* ``temp``: daily mean Heathrow temperature of the half-hour's local date, linear (percent per
  degree C, centred at ``TEMP_CENTER``); ``temp3``: three bins (cuts ``TEMP_CUTS``);
* ``dur``: duration class of the event (``DUR_CUTS``);
* ``first``: first half-hour of the event;
* ``fatigue``: local dates with any event half-hour in the ``FATIGUE_DAYS`` days before the
  event's start date (centred at ``FATIGUE_CENTER``).

The first categorical factor gets one dummy per level, further categorical factors drop their
first level, continuous factors enter as centred slopes. Windows of Normal half-hours 2 h before
and 0-2 h / 2-4 h after events (as in ``lcl_did``) are in every non-null kernel, constant per band.
The cuts were fixed from the 2013 schedule and temperatures before any outcome was looked at:
temperature cuts near the terciles of event half-hours, duration classes from the schedule audit.

Cross-validation
----------------
Rows touched by an event (event half-hours and the window half-hours) form contiguous clusters;
each cluster belongs to the S0 month fold (``ttrl.data.splits``) of its first event half-hour.
For each fold, the model is fitted on every row except that fold's clusters and predicts them.
Normal half-hours of the test months stay in the fit, so the baseline (fixed effects) of a test
month comes from its own Normal days and the loss measures how well the kernel predicts unseen
events. Loss is the mean squared error on all test rows; paired differences between models get
standard errors from the clusters. Selection is forward and one-SE: a factor is added only if it
lowers the loss by more than one standard error of the paired difference.

Ensemble
--------
Each member adds two independent deviations to the estimate: that of one household-bootstrap draw
(households resampled within group) and that of one date-bootstrap fit (local dates resampled).
The members' spread is then the combined uncertainty that the intervals report (household
sampling plus common daily shocks), as in ``lcl_did``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ttrl.analysis import lcl_did as did
from ttrl.data import lcl_features as lf
from ttrl.data.splits import FOLDS, MONTH_FOLDS

BANDS = ("High", "Low")
HOUR4 = ("23-05", "05-11", "11-17", "17-23")
HOUR8 = ("02-05", "05-08", "08-11", "11-14", "14-17", "17-20", "20-23", "23-02")
TEMP_CUTS = (6.0, 12.0)
TEMP3 = ("<6C", "6-12C", ">12C")
TEMP_CENTER = 10.0
DUR_CUTS = (3.0, 6.0)
DUR = ("<=3h", "3-6h", ">6h")
FATIGUE_DAYS = 7
FATIGUE_CENTER = 2.0
FACTORS = {"hour4": "cat", "hour8": "cat", "temp3": "cat", "dur": "cat",
           "temp": "slope", "first": "slope", "fatigue": "slope"}
FAMILIES = {"hour": ("hour4", "hour8"), "temp": ("temp", "temp3"), "dur": ("dur",),
            "first": ("first",), "fatigue": ("fatigue",)}
FE_CHOICES = ("hhw", "moy", "month")
LEVEL_TERMS = ("trial", "post")


# ---------------------------------------------------------------- outcome
def pct_outcome(sums: did.GroupSums) -> np.ndarray:
    """y = 100 * Delta / P for every draw (rows) and half-hour (columns); see module docstring."""
    tou_typical = sums.mean("kwh", True) - sums.mean("dev", True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * sums.delta() / (tou_typical + sums.mean("dev", False))


# ---------------------------------------------------------------- event features
def _ensure_period(cal: pd.DataFrame) -> pd.DataFrame:
    if "period" in cal:
        return cal
    return cal.assign(period=np.where(cal["band"].notna(), "trial", "pre"))


def event_rows(cal: pd.DataFrame, temp: pd.Series) -> pd.DataFrame:
    """Per calendar row: band ('' if Normal or unpriced), event_id (-1), first, duration_h,
    temp_c (daily mean of the row's local date), fatigue, hour4 and hour8 block indices.

    ``cal`` needs ts_local, date, hh and band (a hypothetical schedule works: S2 uses this);
    ``temp`` is indexed by local date.
    """
    n = len(cal)
    ev = lf.find_events(cal)
    band = np.full(n, "", dtype=object)
    eid = np.full(n, -1)
    first, dur, fat = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    event_dates = set(cal.loc[cal["band"].isin(BANDS).fillna(False).to_numpy(bool), "date"])
    for e in ev.itertuples():
        pos = np.arange(e.i0, e.i1 + 1)
        band[pos], eid[pos], dur[pos] = e.band, e.event_id, e.duration_h
        first[e.i0] = 1.0
        fat[pos] = sum((e.date - pd.Timedelta(days=k)) in event_dates for k in range(1, FATIGUE_DAYS + 1))
    t = temp.reindex(pd.DatetimeIndex(cal["date"])).to_numpy(float)
    if np.isnan(t[band != ""]).any():
        raise ValueError("event half-hour without a daily temperature")
    hh = cal["hh"].to_numpy()
    return pd.DataFrame({"band": band, "event_id": eid, "first": first, "duration_h": dur, "fatigue": fat,
                         "temp_c": t, "hour4": ((hh - 46) % 48) // 12, "hour8": ((hh - 4) % 48) // 6},
                        index=cal.index)


def _levels(rows: pd.DataFrame, f: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if f == "hour4":
        return rows["hour4"].to_numpy(), HOUR4
    if f == "hour8":
        return rows["hour8"].to_numpy(), HOUR8
    if f == "temp3":
        return np.digitize(rows["temp_c"].to_numpy(), TEMP_CUTS), TEMP3
    if f == "dur":
        d = rows["duration_h"].to_numpy()
        return np.digitize(d, DUR_CUTS, right=True), DUR
    raise ValueError(f)


def _slope(rows: pd.DataFrame, f: str) -> np.ndarray:
    if f == "temp":
        return rows["temp_c"].to_numpy() - TEMP_CENTER
    if f == "first":
        return rows["first"].to_numpy()
    if f == "fatigue":
        return rows["fatigue"].to_numpy() - FATIGUE_CENTER
    raise ValueError(f)


def kernel_design(cal: pd.DataFrame, temp: pd.Series, factors: tuple[str, ...] | None = ()) -> pd.DataFrame:
    """Kernel regressors for ``factors`` (``None``: the null kernel, no columns).

    Column names: ``High`` (no categorical factor) or ``High|hour4=17-23`` (levels of the first
    categorical factor), ``High|dur=>6h`` (further categorical factors, first level dropped),
    ``High|temp`` (slopes), and the window terms ``High_pre_2h`` etc.
    """
    if factors is None:
        return pd.DataFrame(index=cal.index)
    unknown = set(factors) - set(FACTORS)
    if unknown:
        raise ValueError(f"unknown factors {sorted(unknown)}")
    cal = _ensure_period(cal)
    windows = did.event_design(cal).drop(columns=["trial", "post", *BANDS])
    names = [f"{b}_{w}" for b in BANDS for w in did.WINDOWS]
    windows = windows.reindex(columns=names, fill_value=0.0)  # a schedule may lack a band
    rows = event_rows(cal, temp)
    cats = [f for f in factors if FACTORS[f] == "cat"]
    slopes = [f for f in factors if FACTORS[f] == "slope"]
    cols: dict[str, np.ndarray] = {}
    for b in BANDS:
        on = (rows["band"] == b).to_numpy(float)
        if cats:
            lev, names = _levels(rows, cats[0])
            for k, nm in enumerate(names):
                cols[f"{b}|{cats[0]}={nm}"] = on * (lev == k)
        else:
            cols[b] = on
        for f in cats[1:]:
            lev, names = _levels(rows, f)
            for k, nm in enumerate(names[1:], start=1):
                cols[f"{b}|{f}={nm}"] = on * (lev == k)
        for f in slopes:
            cols[f"{b}|{f}"] = on * np.nan_to_num(_slope(rows, f))
    return pd.concat([pd.DataFrame(cols, index=cal.index), windows], axis=1)


def fe_matrix(cal: pd.DataFrame, fe: str) -> pd.DataFrame:
    """Group-level fixed effects: 'hhw' (half-hour-of-day x weekend, plus trial and post level
    terms), 'moy' (+ month-of-year x 4-hour block), 'month' (+ calendar month x 4-hour block,
    which absorbs the level terms)."""
    if fe not in FE_CHOICES:
        raise ValueError(fe)
    F = did.fe_dummies(cal, None if fe == "hhw" else fe)
    if fe != "month":
        lv = did.event_design(cal)[list(LEVEL_TERMS)]
        F = pd.concat([lv, F], axis=1)
    return F


# ---------------------------------------------------------------- least squares
def _lstsq(Z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """OLS (or WLS) coefficients; columns that are all zero on the used rows get 0."""
    live = np.abs(Z).sum(0) > 0 if w is None else (np.abs(Z) * w[:, None]).sum(0) > 0
    coef = np.zeros(Z.shape[1])
    Zl = Z[:, live]
    if w is not None:
        s = np.sqrt(w)
        Zl, y = Zl * s[:, None], y * s
    rank = np.linalg.matrix_rank(Zl)
    if rank < Zl.shape[1]:
        raise np.linalg.LinAlgError(f"rank deficient ({rank} < {Zl.shape[1]})")
    coef[live] = np.linalg.lstsq(Zl, y, rcond=None)[0]
    return coef


@dataclass
class KernelFit:
    factors: tuple[str, ...]
    fe: str
    table: pd.DataFrame       # term, coef (percent), standard errors, 95% interval
    n_obs: int

    @property
    def coef(self) -> pd.Series:
        return self.table.set_index("term")["coef"]


def fit(cal: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, F: pd.DataFrame,
        factors: tuple[str, ...] = (), fe: str = "hhw") -> KernelFit:
    """Fit on ``y`` (1-D, or 2-D with row 0 the estimate and rows 1.. household-bootstrap draws).

    Reports kernel and level terms with date-clustered and (2-D ``y``) household-bootstrap
    standard errors; the 95% interval uses their combination, as in ``lcl_did``.
    """
    import statsmodels.api as sm

    Y = np.atleast_2d(y)
    ok = np.isfinite(Y[0])
    Z = pd.concat([X, F], axis=1).loc[ok]
    Z = Z.loc[:, Z.abs().sum() > 0]
    res = sm.OLS(Y[0, ok], Z.to_numpy()).fit(cov_type="cluster",
                                             cov_kwds={"groups": cal.loc[ok, "date"].factorize()[0]})
    keep = [i for i, c in enumerate(Z.columns) if c in X.columns or c in LEVEL_TERMS]
    tab = pd.DataFrame({"term": Z.columns[keep], "coef": res.params[keep], "se_cluster_date": res.bse[keep]})
    if Y.shape[0] > 1:
        if not np.isfinite(Y[1:, ok]).all():
            raise ValueError("bootstrap draw with missing half-hours")
        B = Y[1:, ok] @ np.linalg.pinv(Z.to_numpy()).T
        tab["se_boot_household"] = B[:, keep].std(0, ddof=1)
    else:
        tab["se_boot_household"] = 0.0
    tab["se_combined"] = np.hypot(tab["se_cluster_date"], tab["se_boot_household"])
    tab["lo"] = tab["coef"] - 1.96 * tab["se_combined"]
    tab["hi"] = tab["coef"] + 1.96 * tab["se_combined"]
    return KernelFit(tuple(factors), fe, tab.reset_index(drop=True), int(ok.sum()))


# ---------------------------------------------------------------- cross-validation
def affected_clusters(cal: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster id per row (-1 if no event or window touches it) and a cluster table with its
    fold (S0 month fold of the cluster's first event half-hour), rows and first event start."""
    cal = _ensure_period(cal)
    windows = did.event_design(cal).drop(columns=["trial", "post"])
    hit = windows.to_numpy().sum(1) > 0
    start = hit & ~np.r_[False, hit[:-1]]
    cid = np.where(hit, np.cumsum(start) - 1, -1)
    band = cal["band"].astype(object).fillna("").to_numpy()
    ts = cal["ts_local"].to_numpy()
    rows = []
    for c in range(int(cid.max()) + 1):
        idx = np.flatnonzero(cid == c)
        ev = idx[np.isin(band[idx], BANDS)]
        if len(ev) == 0:
            raise ValueError(f"cluster {c} has no event half-hour")
        t0 = pd.Timestamp(ts[ev[0]])
        rows.append({"cluster": c, "fold": MONTH_FOLDS[t0.month], "n_rows": len(idx), "first_event": t0})
    return cid, pd.DataFrame(rows)


@dataclass
class CVResult:
    name: str
    factors: tuple[str, ...] | None
    fe: str
    sq_err: np.ndarray        # per row; NaN outside the test rows
    pred_kernel: np.ndarray   # kernel part of the prediction on test rows
    pred_base: np.ndarray     # fixed-effect part on test rows
    n_params: int

    def loss(self) -> float:
        return float(np.nanmean(self.sq_err))


def cross_validate(cal: pd.DataFrame, y: np.ndarray, temp: pd.Series, factors: tuple[str, ...] | None,
                   fe: str, cid: np.ndarray, clusters: pd.DataFrame, name: str | None = None) -> CVResult:
    """Month-fold CV of one kernel (see module docstring). ``factors=None`` is the null kernel:
    fixed effects only, fitted on rows untouched by any event."""
    X = kernel_design(cal, temp, factors)
    F = fe_matrix(cal, fe)
    Z = pd.concat([X, F], axis=1).to_numpy(float)
    nk = X.shape[1]
    ok = np.isfinite(y)
    fold_of = clusters.set_index("cluster")["fold"]
    row_fold = np.where(cid >= 0, fold_of.reindex(cid.clip(0)).to_numpy(), "")
    sq = np.full(len(y), np.nan)
    pk = np.full(len(y), np.nan)
    pb = np.full(len(y), np.nan)
    for f in FOLDS:
        test = ok & (row_fold == f)
        train = ok & ~test
        if factors is None:
            train &= cid < 0
        coef = _lstsq(Z[train], y[train])
        pk[test] = Z[test][:, :nk] @ coef[:nk]
        pb[test] = Z[test][:, nk:] @ coef[nk:]
        sq[test] = (y[test] - pk[test] - pb[test]) ** 2
    label = name or ("null" if factors is None else "+".join(factors) or "const")
    return CVResult(label, factors, fe, sq, pk, pb, Z.shape[1])


def paired_diff(a: CVResult, b: CVResult, cid: np.ndarray) -> tuple[float, float]:
    """Loss of ``a`` minus loss of ``b`` (mean over test rows) and its cluster standard error."""
    test = np.isfinite(a.sq_err)
    d = pd.Series(a.sq_err[test] - b.sq_err[test]).groupby(cid[test]).sum()
    n = int(test.sum())
    return float(d.sum() / n), float(np.sqrt(len(d)) * d.std(ddof=1) / n)


def forward_select(cal: pd.DataFrame, y: np.ndarray, temp: pd.Series, fe: str, cid: np.ndarray,
                   clusters: pd.DataFrame, families: dict[str, tuple[str, ...]] = FAMILIES
                   ) -> tuple[tuple[str, ...], pd.DataFrame, dict[str, CVResult]]:
    """Forward selection from the constant kernel with the one-SE rule (module docstring).

    Returns the selected factors, a table of every candidate tried at every step, and the CV
    results by name.
    """
    results: dict[str, CVResult] = {}

    def cv(factors):
        r = cross_validate(cal, y, temp, factors, fe, cid, clusters)
        results[r.name] = r
        return r

    current = cv(())
    chosen: tuple[str, ...] = ()
    left = dict(families)
    rows = []
    step = 0
    while left:
        step += 1
        tried = []
        for fam, members in left.items():
            for f in members:
                cats = [g for g in chosen if FACTORS[g] == "cat"]
                # a categorical factor after another categorical one enters as deviations; order the
                # tuple so the first categorical factor keeps its full set of levels
                new = chosen + (f,)
                if FACTORS[f] == "cat" and not cats:
                    new = (f,) + chosen
                try:
                    r = cv(new)
                except np.linalg.LinAlgError:  # collinear with factors already in, on some fold
                    rows.append({"step": step, "base": current.name, "add": f, "family": fam,
                                 "model": "+".join(new), "loss": np.nan, "change": np.nan, "se": np.nan,
                                 "params": np.nan, "identified": False})
                    continue
                d, se = paired_diff(r, current, cid)
                tried.append((fam, f, r, d, se))
                rows.append({"step": step, "base": current.name, "add": f, "family": fam, "model": r.name,
                             "loss": r.loss(), "change": d, "se": se, "params": r.n_params, "identified": True})
        step_rows = [row for row in rows if row["step"] == step]
        if not tried:
            for row in step_rows:
                row["best_in_step"] = row["accepted"] = False
            break
        fam, f, best, d, se = min(tried, key=lambda t: t[3])
        accept = d < -se
        for row in step_rows:
            row["best_in_step"] = row["model"] == best.name
            row["accepted"] = accept and row["model"] == best.name
        if not accept:
            break
        chosen, current = best.factors, best
        left.pop(fam)
    return chosen, pd.DataFrame(rows), results


def baseline(cal: pd.DataFrame, y: np.ndarray, fe: str, cid: np.ndarray) -> np.ndarray:
    """Fixed-effect baseline fitted on the rows no event touches (``cid < 0``) and predicted for
    every row. The held-out check measures observed event effects as ``y`` minus this baseline."""
    F = fe_matrix(cal, fe).to_numpy(float)
    train = np.isfinite(y) & (cid < 0)
    return F @ _lstsq(F[train], y[train])


def event_calibration(cv: CVResult, cal: pd.DataFrame, y: np.ndarray, clusters: pd.DataFrame,
                      cid: np.ndarray) -> pd.DataFrame:
    """Per held-out event: observed effect (mean of y minus the CV baseline over the event's
    half-hours) and the kernel's prediction, in percent."""
    ev = lf.find_events(_ensure_period(cal))
    fold_of = clusters.set_index("cluster")["fold"]
    rows = []
    for e in ev.itertuples():
        pos = np.arange(e.i0, e.i1 + 1)
        pos = pos[np.isfinite(cv.sq_err[pos])]
        if len(pos) == 0:
            continue
        rows.append({"event_id": e.event_id, "band": e.band, "start": e.start, "duration_h": e.duration_h,
                     "fold": fold_of[cid[pos[0]]],
                     "observed": float(np.mean(y[pos] - cv.pred_base[pos])),
                     "predicted": float(np.mean(cv.pred_kernel[pos]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ensemble
def date_weights(cal: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resampling counts of local dates, expanded to rows."""
    codes, uniq = pd.factorize(cal["date"])
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)))
    return counts[codes].astype(float)


def ensemble(cal: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, F: pd.DataFrame, seed: int,
             max_tries: int = 20) -> pd.DataFrame:
    """Ensemble members from ``y`` (row 0 the estimate, rows 1.. household-bootstrap draws).

    Member b = beta(y_b) + [beta_date_b(y_0) - beta(y_0)]: the household-bootstrap fit of draw b
    plus the deviation of a date-bootstrap fit (local dates resampled, WLS on the estimate). The
    two deviations are independent, so the members' covariance is the household-bootstrap plus the
    date-bootstrap covariance, which is what the combined intervals of ``fit`` use. Returns
    members x (kernel and level terms with an estimate), percent.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y[0])
    if not np.isfinite(y[1:, ok]).all():
        raise ValueError("bootstrap draw with missing half-hours")
    Z = pd.concat([X, F], axis=1).loc[ok]
    Z = Z.loc[:, Z.abs().sum() > 0]
    keep = [i for i, c in enumerate(Z.columns) if c in X.columns or c in LEVEL_TERMS]
    Zn = Z.to_numpy(float)
    beta0 = _lstsq(Zn, y[0, ok])
    B_hh = y[1:, ok] @ np.linalg.pinv(Zn).T
    cal_ok = cal.loc[ok]
    B_date = np.empty_like(B_hh)
    for b in range(B_hh.shape[0]):
        for _ in range(max_tries):
            w = date_weights(cal_ok, rng)
            if not ((np.abs(Zn) * w[:, None]).sum(0) > 0).all():
                continue  # a term lost all its dates in this resample
            try:
                B_date[b] = _lstsq(Zn, y[0, ok], w)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError(f"member {b}: no identified date resample in {max_tries} tries")
    members = B_hh + (B_date - beta0)
    return pd.DataFrame(members[:, keep], columns=Z.columns[keep])


# ---------------------------------------------------------------- the kernel for S2
@dataclass
class KernelSpec:
    """What S2 needs to rebuild the kernel regressors for any price schedule."""
    factors: tuple[str, ...]
    fe: str
    terms: list[str]
    temp_center: float = TEMP_CENTER
    temp_cuts: tuple[float, ...] = TEMP_CUTS
    dur_cuts: tuple[float, ...] = DUR_CUTS
    fatigue_days: int = FATIGUE_DAYS
    fatigue_center: float = FATIGUE_CENTER
    support: dict = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> "KernelSpec":
        d = json.loads(text)
        for k in ("factors", "temp_cuts", "dur_cuts"):
            d[k] = tuple(d[k])
        return cls(**d)


def support(cal: pd.DataFrame, temp: pd.Series) -> dict:
    """Range of the 2013 schedule the kernel was estimated on (S2 must stay inside it)."""
    rows = event_rows(_ensure_period(cal), temp)
    ev = rows[rows["band"] != ""]
    out = {}
    for b in BANDS:
        e = ev[ev["band"] == b]
        out[b] = {"temp_c": [round(float(e["temp_c"].min()), 1), round(float(e["temp_c"].max()), 1)],
                  "duration_h": sorted(float(x) for x in e["duration_h"].unique()),
                  "fatigue": [int(e["fatigue"].min()), int(e["fatigue"].max())],
                  "half_hours": int(len(e))}
    return out


def kernel_effect(spec: KernelSpec, coef: pd.Series | pd.DataFrame, cal: pd.DataFrame,
                  temp: pd.Series) -> np.ndarray:
    """Percent change of load per half-hour for a schedule ``cal`` (columns ts_local, date, hh,
    band) under kernel coefficients ``coef`` (a Series, or a members x terms frame -> members x
    rows). Level terms (trial, post) are not part of the price response and are ignored."""
    X = kernel_design(cal, temp, spec.factors)
    missing = [t for t in spec.terms if t not in X.columns]
    if missing:
        raise ValueError(f"design lacks kernel terms {missing}")
    extra = [c for c in X.columns if c not in spec.terms and X[c].abs().sum() > 0]
    if extra:
        raise ValueError(f"schedule uses kernel cells without an estimate: {extra}")
    A = X[spec.terms].to_numpy(float)
    C = coef[spec.terms].to_numpy(float)
    return A @ C if C.ndim == 1 else C @ A.T
