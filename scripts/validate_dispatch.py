"""Validate the merit-order dispatch harness for the ERCOT gas fleet (stage S4).

Temperature-free first version. Fits one heat-input curve per CEMS unit
(``ttrl.supply.units``), dispatches the non-CHP gas fleet hour by hour to the
ACTUAL non-CHP gross output reported in CEMS (``ttrl.supply.dispatch``) and
compares fuel, CO2 and, above all, the marginal heat rate with CEMS 2024.
It also uses EIA-930 hourly generation by source to check how often gas is
the resource that follows load in ERCOT.

Time split (local standard time, the CEMS clock): unit curves and the
commitment headroom are fitted on January-August 2024 and validated on
September-December 2024; in-sample results use the same or full-year fits.

Writes outputs/dispatch_validation.md (copied to docs/reports/), a figure,
the fitted curve tables (data/processed/unit_curves_<year>_{train,full}.parquet,
outputs/dispatch_unit_curves_<year>.csv) and the hourly series
(data/processed/dispatch_hourly_<year>.parquet).

Usage:
    python scripts/validate_dispatch.py [--year 2024] [--delta 500]
"""
from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass, field
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from ttrl.data import eia930, pudl  # noqa: E402
from ttrl.paths import OUTPUTS, PROCESSED, ROOT  # noqa: E402
from ttrl.supply import dispatch as D  # noqa: E402
from ttrl.supply import units as U  # noqa: E402
from ttrl.supply.fleet import FleetHours, build_fleet_hours  # noqa: E402

TRAIN_MONTHS = range(1, 9)
TEST_MONTHS = range(9, 13)
HAC_LAGS = 24
SEASONS = ["winter", "spring", "summer", "fall"]
DOCS_REPORTS = ROOT / "docs" / "reports"
FIG_NAME = "dispatch_marginal_hr.png"

# Priority-list configurations. All dispatch the non-CHP fleet; CHP units are
# exogenous at their actual output in every configuration.
CONFIGS = {
    "C0": dict(label="priority list, no headroom, monthly availability",
               availability="month", headroom=False, suspect_exo=False),
    "C": dict(label="priority list + headroom, monthly availability",
              availability="month", headroom=True, suspect_exo=False),
    "C-sx": dict(label="as C, suspect CC units exogenous (actual output)",
                 availability="month", headroom=True, suspect_exo=True),
    "C-day": dict(label="as C, daily availability",
                  availability="day", headroom=True, suspect_exo=False),
}
BASE_LABELS = {
    "A": "curves only: unit curves at each unit's actual output",
    "B": "actual commitment: units that ran, equal-incremental dispatch",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}",
             formats: dict[str, str] | None = None) -> str:
    """GitHub markdown table with per-column float formats."""
    formats = formats or {}

    def fmt(col, v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else formats.get(col, floatfmt).format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, r)) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def pct(model: float, actual: float) -> float:
    """Percent error of ``model`` relative to ``actual``."""
    return 100.0 * (model / actual - 1.0)


def hac_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float, int]:
    """OLS slope of y on x (with intercept) and its HAC standard error."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    res = sm.OLS(y[ok], sm.add_constant(x[ok])).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(res.params[1]), float(res.bse[1]), int(ok.sum())


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_inputs(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CEMS extract, CEMS-unit table and EIA-860 generators (built if missing)."""
    paths = pudl.build_ercot_gas_cems(year)
    return (pd.read_parquet(paths["cems"]), pd.read_parquet(paths["units"]),
            pd.read_parquet(paths["generators"]))


@dataclass
class Setup:
    """Everything the experiments share."""
    year: int
    fleet: FleetHours            # non-CHP units, columns = curve-table order
    chp_heat: np.ndarray         # (T,) exogenous CHP heat input
    chp_co2: np.ndarray
    chp_energy: np.ndarray
    co2_actual: np.ndarray       # (T, N) non-CHP CO2 with gaps filled
    co2_filled_share: float
    curves: dict[str, pd.DataFrame]          # "train" / "full" -> non-CHP curve table
    curves_all: dict[str, pd.DataFrame]      # incl. CHP, for the report
    train: np.ndarray            # (T,) bool
    test: np.ndarray
    month: np.ndarray
    season: np.ndarray
    hour_lst: np.ndarray
    target: np.ndarray           # (T,) actual non-CHP gross output, MWh
    heat: np.ndarray             # (T,) actual non-CHP heat input, MMBtu
    co2: np.ndarray              # (T,) actual non-CHP CO2, short tons
    load_bin: np.ndarray         # (T,) quintile 0..4 of target over the year
    bin_labels: list[str] = field(default_factory=list)
    low_hr: dict = field(default_factory=dict)   # rows dropped by the heat-rate floor


def build_setup(year: int, fit_cfg: U.FitConfig) -> Setup:
    """Fit curves (Jan-Aug and full year) and build hourly arrays."""
    cems, units, gens = load_inputs(year)
    uh = U.prepare_unit_hours(cems)
    all_ids = sorted(uh["unit_id"].unique())
    curves_all = {
        "train": U.fit_unit_curves(uh[uh["month"].isin(TRAIN_MONTHS)], units, gens,
                                   fit_cfg, unit_ids=all_ids),
        "full": U.fit_unit_curves(uh, units, gens, fit_cfg, unit_ids=all_ids),
    }
    chp_ids = set(curves_all["full"].loc[curves_all["full"]["chp"], "unit_id"])
    nc_ids = [u for u in all_ids if u not in chp_ids]
    curves = {k: v.set_index("unit_id").loc[nc_ids].reset_index()
              for k, v in curves_all.items()}

    fleet_all = build_fleet_hours(cems, all_ids, year)
    is_chp = np.array([u in chp_ids for u in all_ids])
    fleet = fleet_all.columns(~is_chp)
    assert fleet.unit_ids == nc_ids

    # CO2 gap fill with each unit's full-year CO2/heat ratio.
    def co2_factor(f: FleetHours) -> np.ndarray:
        both = (f.heat > 0) & ~np.isnan(f.co2)
        num = np.where(both, f.co2, 0.0).sum(axis=0)
        den = np.where(both, f.heat, 0.0).sum(axis=0)
        fac = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        return np.where(np.isnan(fac), np.nanmedian(fac), fac)
    co2_nc = fleet.co2_filled(co2_factor(fleet))
    chp = fleet_all.columns(is_chp)
    co2_chp = chp.co2_filled(co2_factor(chp))
    filled = (np.isnan(fleet.co2) & (fleet.heat > 0))
    filled_share = float(np.where(filled, co2_nc, 0).sum() / co2_nc.sum())

    # Clean full hours dropped from the Jan-Aug fits by the heat-rate floor.
    tr = uh[uh["month"].isin(TRAIN_MONTHS) & uh["clean"]]
    low = tr.loc[~U.fit_rows(tr, fit_cfg)]
    low_hr = {"rows": len(low), "share": 100 * len(low) / len(tr),
              "units": low["unit_id"].nunique(),
              "top": low["unit_id"].value_counts().head(2).to_dict()}

    lab = fleet.labels
    month = lab["month"].to_numpy()
    target = fleet.energy.sum(axis=1)
    edges = np.quantile(target, [0.2, 0.4, 0.6, 0.8])
    load_bin = np.searchsorted(edges, target, side="right")
    lo = np.concatenate([[target.min()], edges])
    hi = np.concatenate([edges, [target.max()]])
    bin_labels = [f"Q{i + 1} ({a / 1e3:.1f}-{b / 1e3:.1f} GW)"
                  for i, (a, b) in enumerate(zip(lo, hi))]
    return Setup(
        year=year, fleet=fleet, chp_heat=chp.heat.sum(axis=1),
        chp_co2=co2_chp.sum(axis=1), chp_energy=chp.energy.sum(axis=1),
        co2_actual=co2_nc, co2_filled_share=filled_share,
        curves=curves, curves_all=curves_all,
        train=np.isin(month, TRAIN_MONTHS), test=np.isin(month, TEST_MONTHS),
        month=month, season=lab["season"].to_numpy(), hour_lst=lab["hour_lst"].to_numpy(),
        target=target, heat=fleet.heat.sum(axis=1), co2=co2_nc.sum(axis=1),
        load_bin=load_bin, bin_labels=bin_labels, low_hr=low_hr)


# --------------------------------------------------------------------------
# Model runs
# --------------------------------------------------------------------------
@dataclass
class Run:
    """Hourly result of one model variant with one curve set."""
    heat: np.ndarray                     # (T,) non-CHP heat input, MMBtu
    co2: np.ndarray                      # (T,) short tons
    energy_by_group: dict[str, np.ndarray]   # group -> (T,) model gross MWh
    heat_by_group: dict[str, np.ndarray]     # group -> (T,) model MMBtu
    status: np.ndarray | None = None
    mhr: dict[float, np.ndarray] = field(default_factory=dict)  # delta -> (T,)
    headroom: tuple[float, float] = (0.0, 0.0)


def unit_groups(table: pd.DataFrame) -> np.ndarray:
    """Tech label with suspect CC units separated ("CC*")."""
    return np.where(table["suspect_cc"], "CC*", table["tech"])


def by_group(values: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Hourly sums of a (T, N) array per unit group."""
    return {k: values[:, groups == k].sum(axis=1) for k in np.unique(groups)}


def run_curves_only(s: Setup, table: pd.DataFrame) -> Run:
    """Variant A: each unit's curve evaluated at its actual output."""
    cv = U.StaticCurves(table).arrays()
    uh = s.fleet.optime * cv.heat(s.fleet.load)
    f = table["co2_per_mmbtu"].to_numpy()
    g = unit_groups(table)
    return Run(heat=uh.sum(axis=1), co2=(uh * f).sum(axis=1),
               energy_by_group=by_group(s.fleet.energy, g), heat_by_group=by_group(uh, g))


def run_actual_commitment(s: Setup, table: pd.DataFrame) -> Run:
    """Variant B: dispatch the units that actually produced in each hour."""
    cv = U.StaticCurves(table).arrays()
    on = s.fleet.on
    res = D.dispatch(s.target, cv, on, committed=on)
    uh = res.unit_heat(cv)
    f = table["co2_per_mmbtu"].to_numpy()
    g = unit_groups(table)
    return Run(heat=uh.sum(axis=1), co2=(uh * f).sum(axis=1), status=res.status,
               energy_by_group=by_group(res.p, g), heat_by_group=by_group(uh, g))


def run_priority(s: Setup, table: pd.DataFrame, cfg: dict, fit_hours: np.ndarray,
                 deltas: list[float]) -> Run:
    """Priority-list commitment + economic dispatch (configurations C*)."""
    disp = ~table["suspect_cc"].to_numpy() if cfg["suspect_exo"] else np.ones(len(table), bool)
    model = U.StaticCurves(table.loc[disp])
    cv = model.arrays()
    f = model.co2_per_mmbtu
    target = s.fleet.energy[:, disp].sum(axis=1)
    avail = s.fleet.availability(cfg["availability"])[:, disp]
    headroom = (0.0, 0.0)
    if cfg["headroom"]:
        committed_pmax = (s.fleet.on[:, disp] * cv.pmax).sum(axis=1)
        headroom = D.fit_headroom(target[fit_hours], committed_pmax[fit_hours])
    res = D.dispatch(target, cv, avail, headroom_mw=headroom[0] + headroom[1] * target)
    uh = res.unit_heat(cv)
    exo_heat = s.fleet.heat[:, ~disp].sum(axis=1)
    exo_co2 = s.co2_actual[:, ~disp].sum(axis=1)
    g_all = unit_groups(table)
    p_all = s.fleet.energy.copy()          # exogenous units keep actual output
    h_all = s.fleet.heat.copy()
    p_all[:, disp] = res.p
    h_all[:, disp] = uh
    energy, heat = by_group(p_all, g_all), by_group(h_all, g_all)
    run = Run(heat=uh.sum(axis=1) + exo_heat, co2=(uh * f).sum(axis=1) + exo_co2,
              status=res.status, energy_by_group=energy, heat_by_group=heat,
              headroom=headroom)
    for d in deltas:
        run.mhr[d] = D.finite_difference_mhr(target, cv, avail, d, headroom)
    return run


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def level_metrics(s: Setup, run: Run, window: np.ndarray) -> dict:
    """Period fuel and CO2 error (%), hourly MAPE of fuel (%)."""
    h, m = s.heat[window], run.heat[window]
    return {"fuel error %": pct(m.sum(), h.sum()),
            "CO2 error %": pct(run.co2[window].sum(), s.co2[window].sum()),
            "hourly fuel MAPE %": 100 * float(np.mean(np.abs(m / h - 1))),
            "fuel error incl. CHP %": pct(m.sum() + s.chp_heat[window].sum(),
                                          h.sum() + s.chp_heat[window].sum())}


def first_differences(s: Setup, series: np.ndarray, window: np.ndarray):
    """Hour-to-hour differences inside ``window`` and the midpoint load bin."""
    dx = np.diff(s.target)
    dy = np.diff(series)
    both = window[1:] & window[:-1]
    mid = 0.5 * (s.target[1:] + s.target[:-1])
    return dx, dy, both, mid


def marginal_table(s: Setup, runs: dict[str, Run], window: np.ndarray,
                   groups: np.ndarray, group_labels: list[str], main: str,
                   delta: float) -> pd.DataFrame:
    """Empirical vs model marginal heat rate by group of hours.

    ``groups`` (T,) labels each hour; a first difference belongs to the group
    of its later hour. Actual and model slopes: OLS of the hour-to-hour change
    in non-CHP heat input on the change in non-CHP gross output (HAC SE).
    ``diff`` is the slope of (model change - actual change) on the output
    change, i.e. model minus actual marginal heat rate, with its own HAC SE.
    ``FD`` is the mean of the model's central-difference marginal heat rate
    (target +/- delta) over the group's hours.
    """
    dx, dy_act, both, _ = first_differences(s, s.heat, window)
    g_next = groups[1:]
    rows = []
    for gi, label in enumerate(group_labels):
        sel = both & (g_next == gi)
        if sel.sum() < 30:
            continue
        b_act, se_act, n = hac_slope(dy_act[sel], dx[sel])
        row = {"hours": label, "n": n,
               "mean output GW": float(s.target[1:][sel].mean() / 1e3),
               "actual": b_act, "actual 95% CI": f"{b_act - 1.96 * se_act:.2f} to {b_act + 1.96 * se_act:.2f}",
               "actual SE": se_act}
        for name, run in runs.items():
            dy = np.diff(run.heat)
            row[name] = hac_slope(dy[sel], dx[sel])[0]
        dy_main = np.diff(runs[main].heat)
        b_d, se_d, _ = hac_slope(dy_main[sel] - dy_act[sel], dx[sel])
        row[f"{main} - actual"] = b_d
        row[f"{main} - actual 95% CI"] = f"{b_d - 1.96 * se_d:+.2f} to {b_d + 1.96 * se_d:+.2f}"
        row[f"{main} error %"] = 100 * b_d / b_act
        fd = runs[main].mhr[delta][1:][sel]
        row[f"{main} FD"] = float(np.nanmean(fd))
        row[f"{main} FD error %"] = 100 * (row[f"{main} FD"] / b_act - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def binned_level_slopes(s: Setup, runs: dict[str, Run], window: np.ndarray) -> pd.DataFrame:
    """Cross-sectional marginal heat rate between adjacent load quintiles.

    (mean heat of quintile k+1 - mean heat of quintile k) / (difference in
    mean output), for actual CEMS and each model run, over ``window`` hours.
    """
    rows = []
    for k in range(4):
        a, b = window & (s.load_bin == k), window & (s.load_bin == k + 1)
        if a.sum() < 30 or b.sum() < 30:
            continue
        dp = s.target[b].mean() - s.target[a].mean()
        row = {"between": f"Q{k + 1} -> Q{k + 2}",
               "actual": (s.heat[b].mean() - s.heat[a].mean()) / dp}
        for name, run in runs.items():
            row[name] = (run.heat[b].mean() - run.heat[a].mean()) / dp
        rows.append(row)
    return pd.DataFrame(rows)


def spread_table(tab: pd.DataFrame, main: str) -> pd.DataFrame:
    """Marginal heat-rate difference between the top load quintile and each lower one.

    This is the fuel saved per MWh moved from top-quintile hours to hours of
    the lower quintile. The CI treats the two quintile slopes as independent.
    """
    top = tab.iloc[-1]
    rows = []
    for _, low in tab.iloc[:-1].iterrows():
        a = top["actual"] - low["actual"]
        se = float(np.hypot(top["actual SE"], low["actual SE"]))
        m = top[main] - low[main]
        fd = top[f"{main} FD"] - low[f"{main} FD"]
        rows.append({"from top quintile to": low["hours"], "actual": a,
                     "actual 95% CI": f"{a - 1.96 * se:.2f} to {a + 1.96 * se:.2f}",
                     f"{main} slope": m, f"{main} slope error %": 100 * (m / a - 1),
                     f"{main} FD": fd, f"{main} FD error %": 100 * (fd / a - 1)})
    return pd.DataFrame(rows)


def monthly_errors(s: Setup, runs: dict[str, Run], main: str) -> pd.DataFrame:
    """Monthly non-CHP fuel error (%) per run and CO2 error of the main run."""
    rows = []
    for m in range(1, 13):
        w = s.month == m
        row = {"month": pd.Timestamp(2000, m, 1).strftime("%b"),
               "window": "fit" if m in TRAIN_MONTHS else "validation",
               "actual fuel (million MMBtu)": s.heat[w].sum() / 1e6}
        for name, run in runs.items():
            row[f"{name} fuel %"] = pct(run.heat[w].sum(), s.heat[w].sum())
        row[f"{main} CO2 %"] = pct(runs[main].co2[w].sum(), s.co2[w].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# EIA-930: does gas follow load?
# --------------------------------------------------------------------------
def eia930_analysis(s: Setup) -> dict:
    """Alignment check and gas-response statistics from EIA-930 (ERCO).

    Returns tables for the report: lag correlations, regression shares of
    each source in demand changes (holding wind and solar changes fixed), and
    shares of hours in which gas moves with net load.
    """
    e = eia930.load_ba_hourly(s.year)
    e = e.set_index("datetime_utc").sort_index()
    cems_gas = pd.Series(s.target + s.chp_energy, index=s.fleet.hours)

    lag_rows = []
    for lag in range(-2, 3):
        shifted = e["gas"].reindex(s.fleet.hours + pd.Timedelta(hours=lag))
        shifted.index = s.fleet.hours
        j = pd.concat([cems_gas.rename("cems"), shifted.rename("e930")], axis=1).dropna()
        lag_rows.append({"EIA-930 timestamp minus CEMS hour start (h)": lag,
                         "corr (levels)": j.corr().iloc[0, 1],
                         "corr (first differences)": j.diff().corr().iloc[0, 1]})
    lags = pd.DataFrame(lag_rows)
    best = int(lags.loc[lags["corr (first differences)"].idxmax(),
                        "EIA-930 timestamp minus CEMS hour start (h)"])
    h = e.reindex(s.fleet.hours + pd.Timedelta(hours=best))
    h.index = s.fleet.hours
    h["cems_nonchp"] = s.target
    h["cems_all"] = s.target + s.chp_energy
    h["netload"] = h["demand"] - h["wind"] - h["solar"]
    h["season"] = s.season
    h["block"] = pd.cut(s.hour_lst, [-1, 5, 11, 17, 23],
                        labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)
    d = h.drop(columns=["season", "block"]).diff()
    d["season"], d["block"] = h["season"], h["block"]
    d = d.dropna(subset=["demand", "gas", "wind", "solar"])

    sources = ["gas", "coal", "storage_other", "nuclear", "hydro", "interchange",
               "cems_all", "cems_nonchp"]
    names = {"gas": "gas (EIA-930 net)", "coal": "coal", "storage_other":
             "storage + other", "nuclear": "nuclear", "hydro": "hydro",
             "interchange": "interchange (exports +)",
             "cems_all": "CEMS gas fleet gross, all units",
             "cems_nonchp": "CEMS gas fleet gross, non-CHP (dispatch target)"}

    def coef(frame: pd.DataFrame, y: str) -> tuple[float, float]:
        f = frame[[y, "demand", "wind", "solar"]].dropna()
        x = sm.add_constant(f[["demand", "wind", "solar"]])
        r = sm.OLS(f[y], x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        return float(r.params["demand"]), float(r.bse["demand"])

    share_rows = []
    for y in sources:
        b, se = coef(d, y)
        row = {"response to +1 MWh demand": names[y], "all hours": b,
               "95% CI": f"{b - 1.96 * se:.3f} to {b + 1.96 * se:.3f}"}
        for sea in SEASONS:
            row[sea] = coef(d[d["season"] == sea], y)[0]
        share_rows.append(row)
    shares = pd.DataFrame(share_rows)

    block_rows = []
    for blk in ["00-05", "06-11", "12-17", "18-23"]:
        f = d[d["block"] == blk]
        block_rows.append({"LST hours": blk, "gas": coef(f, "gas")[0],
                           "coal": coef(f, "coal")[0],
                           "storage + other": coef(f, "storage_other")[0],
                           "CEMS non-CHP gross": coef(f, "cems_nonchp")[0]})
    blocks = pd.DataFrame(block_rows)

    thr = 250.0
    big = d["netload"].abs() >= thr
    ratio = d["gas"] / d["netload"]
    other_max = d[["coal", "storage_other"]].mul(np.sign(d["netload"]), axis=0).max(axis=1)
    gas_dir = d["gas"] * np.sign(d["netload"])
    hour_rows = []
    for label, sel in [("all", big)] + [(sea, big & (d["season"] == sea)) for sea in SEASONS] \
            + [(f"LST {b}", big & (d["block"] == b)) for b in ["00-05", "06-11", "12-17", "18-23"]]:
        hour_rows.append({
            "hours": label, "n": int(sel.sum()),
            "gas moves with net load %": 100 * float((ratio[sel] > 0).mean()),
            "gas covers >= 50% %": 100 * float((ratio[sel] >= 0.5).mean()),
            "gas covers >= 80% %": 100 * float((ratio[sel] >= 0.8).mean()),
            "gas largest mover %": 100 * float((gas_dir[sel] > other_max[sel]).mean()),
        })
    hours_tab = pd.DataFrame(hour_rows)
    fit = sm.OLS(h["gas"], sm.add_constant(h["cems_all"]), missing="drop").fit()
    return {"lags": lags, "best_lag": best, "shares": shares, "blocks": blocks,
            "hours": hours_tab, "threshold": thr,
            "small_share": 100 * float((~big).mean()),
            "level_fit": (float(fit.params.iloc[1]), float(fit.rsquared)),
            "energy_ratio": float(h["gas"].sum() / h["cems_all"].sum()),
            "storage_note_from": e.index[e["storage_other"].lt(0)].min()}


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def marginal_figure(tabs: dict[str, pd.DataFrame], main: str, delta: float, path) -> None:
    """Actual vs model marginal heat rate by load quintile, two windows."""
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, len(tabs), figsize=(10, 4.2), sharey=True)
    for ax, (title, t) in zip(np.atleast_1d(axes), tabs.items()):
        x = t["mean output GW"].to_numpy()
        half = 1.96 * t["actual SE"].to_numpy()
        ax.fill_between(x, t["actual"] - half, t["actual"] + half, color=blue, alpha=0.12, lw=0)
        ax.plot(x, t["actual"], color=blue, lw=2, marker="o", ms=8, mec="white",
                mew=2, label="CEMS actual (95% CI band)")
        ax.plot(x, t[main], color=orange, lw=2, marker="s", ms=8, mec="white",
                mew=2, label=f"model {main}, hour-to-hour slope")
        ax.plot(x, t[f"{main} FD"], color=aqua, lw=2, marker="^", ms=8, mec="white",
                mew=2, label=f"model {main}, +/-{delta:,.0f} MW derivative")
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("non-CHP gas fleet gross output, quintile mean (GW)", color=muted)
        ax.grid(color=grid, lw=1)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=muted)
    np.atleast_1d(axes)[0].set_ylabel("marginal heat rate (MMBtu per MWh gross)", color=muted)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def curve_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Fit methods and quality by unit group (CHP and suspect CC separate)."""
    t = table.assign(group=np.where(table["chp"], table["tech"] + " CHP",
                                    np.where(table["suspect_cc"], "CC suspect", table["tech"])))
    rows = []
    for g, x in t.groupby("group"):
        own = x[x["fit_method"] != "pooled"]
        rows.append({
            "group": g, "units": len(x),
            "quadratic": int((x["fit_method"] == "quadratic").sum()),
            "linear (c<0 refit)": int((x["fit_method"] == "linear").sum()),
            "pooled": int((x["fit_method"] == "pooled").sum()),
            "median R2": own["r2"].median(), "median RMSE (MMBtu/h)": own["rmse"].median(),
            "median hours": own["n_fit"].median(),
            "median Pmin/Pmax": (x["pmin"] / x["pmax"]).median(),
            "median Pmax/nameplate": (x["pmax"] / x["nameplate_mw"]).median(),
            "median Pmax/summer cap": (x["pmax"] / x["summer_mw"]).median(),
            "median full-load HR": ((x["a"] + x["b"] * x["pmax"] + x["c"] * x["pmax"] ** 2)
                                    / x["pmax"]).median(),
            "sum Pmax (MW)": x["pmax"].sum(),
        })
    return pd.DataFrame(rows)


def build_report(s: Setup, res: dict, main: str, delta: float,
                 e930: dict, runtime: dict, fit_cfg: U.FitConfig) -> str:
    """Assemble the markdown report from computed tables."""
    L = []
    w = L.append
    mt = res["marg"]
    band = mt["test"]
    w(f"# Dispatch harness validation: ERCOT gas fleet vs EPA CEMS {s.year}\n")
    w(f"Generated by `scripts/validate_dispatch.py` on {date.today().isoformat()} from PUDL "
      f"release `{pudl.DEFAULT_RELEASE}` (EPA CEMS hourly, EIA-860, EIA-930). Temperature-free "
      "first version of the S4 dispatch model. Every number below was computed by that script.\n")

    # ---- Summary
    abs_err = band[f"{main} error %"].abs()
    fd_err = band[f"{main} FD error %"].abs()
    wts = band["n"] / band["n"].sum()
    lv = res["levels"]
    w("## Summary\n")
    w(f"- **Model.** Per-unit robust quadratic heat-input curves; CHP units exogenous at actual output; "
      f"the rest dispatched each hour to the actual CEMS non-CHP gross output by a full-load heat-rate "
      f"priority list with a headroom rule and equal-incremental-heat-rate dispatch. Main configuration "
      f"**{main}** ({CONFIGS[main]['label']}), selected by the lowest Jan-Aug marginal-heat-rate error "
      f"among the priority-list configurations (rule in section 1).")
    w(f"- **Curves alone are accurate.** With each unit at its actual output (variant A), curves fitted "
      f"on Jan-Aug miss Sep-Dec non-CHP fuel by {lv.loc[('A', 'out-of-sample'), 'fuel error %']:+.2f}% "
      f"(hourly MAPE {lv.loc[('A', 'out-of-sample'), 'hourly fuel MAPE %']:.2f}%).")
    w(f"- **Dispatch is biased low in level.** Out of sample (Sep-Dec) the main model's non-CHP fuel "
      f"error is {lv.loc[(main, 'out-of-sample'), 'fuel error %']:+.1f}% and CO2 "
      f"{lv.loc[(main, 'out-of-sample'), 'CO2 error %']:+.1f}% (hourly MAPE "
      f"{lv.loc[(main, 'out-of-sample'), 'hourly fuel MAPE %']:.1f}%); with actual commitment "
      f"(variant B) it is {lv.loc[('B', 'out-of-sample'), 'fuel error %']:+.1f}%. The model runs "
      f"too little steam-turbine and simple-cycle capacity (section 3).")
    w(f"- **Marginal error band (the number tariff effects must beat).** Out of sample, the model's "
      f"marginal heat rate (hour-to-hour slope) differs from the CEMS slope by "
      f"{band[f'{main} error %'].min():+.1f}% to {band[f'{main} error %'].max():+.1f}% across load "
      f"quintiles (hours-weighted mean absolute error {float((abs_err * wts).sum()):.1f}%, max "
      f"{abs_err.max():.1f}%). The model's +/-{delta:,.0f} MW derivative, which a tariff run would use, "
      f"differs by {band[f'{main} FD error %'].min():+.1f}% to {band[f'{main} FD error %'].max():+.1f}% "
      f"(weighted mean absolute {float((fd_err * wts).sum()):.1f}%, max {fd_err.max():.1f}%). "
      f"A tariff's fuel effect per MWh shifted is only distinguishable from model error if it exceeds "
      f"this band. The band depends on the configuration: with the suspect CC units dispatched (C) the "
      f"derivative's weighted mean absolute error is "
      f"{res['cfg_band'].set_index('configuration').at['C', 'Sep-Dec FD MAE %']:.1f}% (section 4).")
    sp = res["spread"]
    st = mt["season_test"].set_index("hours")
    fd_col = f"{main} FD error %"
    worst = st[fd_col].abs().idxmax()
    w(f"- **By season and load tercile** (out of sample) the derivative error ranges from "
      f"{st[fd_col].min():+.1f}% to {st[fd_col].max():+.1f}%; the largest is {st.at[worst, fd_col]:+.1f}% "
      f"in {worst.split(' (')[0]} hours. In the top summer tercile (the hot hours a tariff targets; "
      f"September only in the validation window) it is "
      f"{st.filter(like='summer T3', axis=0)[fd_col].iloc[0]:+.1f}%.")
    w(f"- **Load-shifting spread.** The CEMS marginal heat rate rises from "
      f"{sp['actual_lo']:.2f} (Q1) to {sp['actual_hi']:.2f} MMBtu/MWh (Q5) out of sample, a spread of "
      f"{sp['actual']:.2f} MMBtu/MWh: roughly the fuel saved per MWh of non-CHP gas output moved from "
      f"Q5-type to Q1-type hours. "
      f"The model's spread is {sp['model']:.2f} (slope) and {sp['model_fd']:.2f} (derivative), i.e. "
      f"{100 * (sp['model_fd'] / sp['actual'] - 1):+.0f}%: because the per-quintile errors do not "
      f"cancel, the model overstates load-shifting fuel savings more than it misstates any single "
      f"quintile (section 4, spread table).")
    gs = e930["shares"].set_index("response to +1 MWh demand")
    w(f"- **Gas follows load, but not alone.** In EIA-930 for ERCO, holding wind and solar changes "
      f"fixed, a 1 MWh hour-to-hour demand change comes with {gs.loc['gas (EIA-930 net)', 'all hours']:.2f} "
      f"MWh of gas, {gs.loc['coal', 'all hours']:.2f} of coal and "
      f"{gs.loc['storage + other', 'all hours']:.2f} of storage + other; the CEMS non-CHP gas fleet "
      f"(the dispatch target) moves {gs.loc['CEMS gas fleet gross, non-CHP (dispatch target)', 'all hours']:.2f} "
      f"MWh gross. Gas moves with net load in "
      f"{e930['hours'].iloc[0]['gas moves with net load %']:.0f}% of hours with a net-load change of "
      f"at least {e930['threshold']:.0f} MW and covers at least half of it in "
      f"{e930['hours'].iloc[0]['gas covers >= 50% %']:.0f}%.\n")

    # ---- 1. Setup
    w("## 1. Data, model and validation design\n")
    nc = s.curves["train"]
    w(f"- Fleet: ERCOT gas CC/CT/ST CEMS units from `data/interim`, matched with the PUDL crosswalk. "
      f"{len(nc)} non-CHP units are dispatched; "
      f"{int(s.curves_all['full']['chp'].sum())} CHP units are exogenous: their actual hourly heat input "
      f"and CO2 are added unchanged, so they carry no model error and are left out of the error metrics "
      f"unless stated. CHP share of the fleet's {s.year} heat input: "
      f"{100 * s.chp_heat.sum() / (s.chp_heat.sum() + s.heat.sum()):.1f}%.")
    w(f"- Target: each hour's actual non-CHP gross output from CEMS (sum of gross load x operating time), "
      f"so the validation isolates curve, commitment and dispatch error from demand error. Hours: the "
      f"{len(s.target):,} UTC hours of the CEMS {s.year} year (local standard time, UTC-6). Months and "
      "seasons use local standard time (winter Dec-Feb, spring Mar-May, summer Jun-Sep, fall Oct-Nov).")
    w(f"- Actual fuel includes partial operating hours and hours with zero gross load. Their share of "
      f"non-CHP heat input: partial hours {res['partial_share']:.2f}%, zero-load operating hours "
      f"{res['zero_share']:.2f}%. The model has no start-up fuel, so this part of actual fuel is "
      "represented only through the no-load term of committed units.")
    w(f"- CO2: CEMS CO2 for non-CHP units, with gaps (heat input reported, CO2 missing; "
      f"{100 * s.co2_filled_share:.2f}% of the non-CHP CO2 total after filling) filled by the unit's "
      "full-year CO2/heat ratio. Model CO2 = model heat x the unit's CO2/heat ratio from the fit window.")
    w(f"- Unit curves (`ttrl.supply.units`): Huber robust fit (t = {fit_cfg.huber_t}) of H = a + bP + cP^2 "
      f"on full operating hours with P > 0; refit as a robust line when c < 0 or the incremental heat rate "
      f"is not positive at Pmin (this is the c >= 0 constrained fit). Pmin / Pmax = "
      f"{100 * fit_cfg.pmin_quantile:.0f}th / {100 * fit_cfg.pmax_quantile:.0f}th percentile of full-hour "
      f"output. Hours with a heat rate below {fit_cfg.min_heat_rate} MMBtu/MWh gross (about 62% HHV "
      f"efficiency, physically impossible for a gas unit) are reporting errors and are left out of the "
      f"fits: {s.low_hr['rows']:,} of the clean Jan-Aug hours ({s.low_hr['share']:.2f}%) at "
      f"{s.low_hr['units']} units, mostly "
      + " and ".join(f"{u} ({n:,} h)" for u, n in s.low_hr["top"].items())
      + ". Their CEMS heat input stays in the actual series. Units with fewer than "
      f"{fit_cfg.min_hours} clean hours in the fit window get their "
      "technology's pooled curve scaled by EIA-860 nameplate capacity.")
    w(f"- Suspect CC units (non-CHP CC with median fit-window heat rate >= {fit_cfg.suspect_cc_heat_rate} "
      f"MMBtu/MWh gross, probably reporting combustion-turbine output only): "
      f"{int(nc['suspect_cc'].sum())} units, {nc.loc[nc['suspect_cc'], 'pmax'].sum():,.0f} MW of Pmax "
      "(Jan-Aug fit). They are dispatched with their own (CT-only) curves in C0, C and C-day, which keeps "
      "the target and the curves on the same CEMS accounting basis, and held at actual output in C-sx.")
    w("- Availability: a unit can be committed in hour t if CEMS has a row for it and it produced in at "
      "least one hour of the same local month (C0, C, C-sx) or day (C-day); otherwise it is on outage. "
      "This mask is taken from actual operation and is the same for any target, so it is an exogenous "
      "input, but C-day uses more information about actual operation than the monthly rule.")
    hr = res["headroom"]
    w(f"- Commitment (`ttrl.supply.dispatch.priority_commitment`): available units in order of full-load "
      f"heat rate H(Pmax)/Pmax until summed Pmax >= target + headroom, then the most expensive committed "
      f"units are dropped while committed Pmin exceeds the target and capacity still covers it. Headroom "
      f"= r0 + r1 x target, an OLS fit of (summed Pmax of units that actually ran) - output on output "
      f"over the fit window; Jan-Aug fit: r0 = {hr['train'][0]:,.0f} MW, r1 = {hr['train'][1]:.3f} "
      f"(full-year fit: {hr['full'][0]:,.0f} MW, {hr['full'][1]:.3f}). It stands in for reserves, "
      "minimum up times and units kept on for reasons the model does not see. C0 has no headroom. "
      "Every hour is committed independently (no start-up costs, minimum up/down times or ramp limits).")
    w("- Dispatch: equal incremental heat rate across committed units (one gas price, so the merit order "
      "is by heat rate), solved by vectorised bisection on lambda with exact energy balance.")
    w("- Time split: curves and headroom fitted on Jan-Aug (\"fit\"), validated on Sep-Dec "
      "(\"validation\", out of sample). In-sample results: Jan-Aug fit on Jan-Aug; full-year fit on "
      "Sep-Dec (same hours as the validation, so the two Sep-Dec rows show the cost of fitting out of "
      "sample) and on the full year.")
    w(f"- Configuration choice: among {', '.join(CONFIGS)} the main configuration is the one with the "
      f"lowest hours-weighted mean absolute error of the hour-to-hour marginal heat rate by load quintile "
      f"on the Jan-Aug fit window (Jan-Aug fit). Scores (%): "
      + ", ".join(f"{k} {v:.2f}" for k, v in res["selection"].items())
      + ". All configurations are reported on the validation window. The rule was written after "
      "prototype runs had been inspected, so the choice is not blind, but it uses fit-window data only.")
    w(f"- Runtime (4 CPUs): curve fits {runtime['fit']:.0f} s; one year of priority-list dispatch "
      f"(8,784 hours x {len(nc)} units) {runtime['dispatch']:.1f} s; whole script {runtime['total']:.0f} s.\n")

    # ---- 2. Curves
    w("## 2. Unit curves\n")
    w("Jan-Aug fit, all units (CHP units are fitted for reference but never dispatched):\n")
    w(md_table(curve_summary(s.curves_all["train"]),
               formats={"median R2": "{:.3f}", "median RMSE (MMBtu/h)": "{:.1f}",
                        "median hours": "{:,.0f}", "sum Pmax (MW)": "{:,.0f}"}))
    w("\nR2 and RMSE are in-sample, on the clean full hours of each unit's own fit. 'Pmax/summer cap' "
      "compares the 99th-percentile CEMS gross output with the EIA-860 summer capacity allocated to the "
      "unit (net, generator-level), which is one input for the temperature step. "
      f"{int((s.curves_all['train']['fit_method'].ne('pooled') & s.curves_all['train']['r2'].isna()).sum())} "
      "units report the same heat input in every fit hour whatever their output (probably substituted "
      "values); their fitted curves are flat (incremental heat rate about 0), so they never respond at "
      "the margin. Full curve table: "
      f"`outputs/dispatch_unit_curves_{s.year}.csv`; Parquet copies in `data/processed/`.\n")

    # ---- 3. Levels
    w("## 3. Fuel and CO2 levels\n")
    w("Non-CHP fleet, model minus actual. 'fuel error incl. CHP' adds the exogenous CHP fuel to both sides.\n")
    lt = lv.reset_index().rename(columns={"variant": "variant", "window": "window"})
    w(md_table(lt, formats={c: "{:+.2f}" for c in lt.columns if "%" in c and "MAPE" not in c}))
    w("\nWindows: in-sample (Jan-Aug fit, Jan-Aug hours); out-of-sample (Jan-Aug fit, Sep-Dec hours); "
      "in-sample Sep-Dec (full-year fit, Sep-Dec hours); in-sample year (full-year fit, all hours).")
    def status_text(counts: dict) -> str:
        bad = [f"{D.STATUS_NAMES[k]} {v}" for k, v in counts.items() if k != D.STATUS_OK]
        return ", ".join(bad) if bad else "none"
    w(f"Hours not solved to equal lambda on the validation window: main model {status_text(res['status'])}; "
      f"variant B {status_text(res['status_b'])}. In variant B these are hours whose units that actually "
      "ran were below their 5th-percentile output (start-up or shut-down); they run pro rata below Pmin.\n")
    w("Monthly non-CHP fuel error (%), Jan-Aug fit for all months:\n")
    mo = res["monthly"]
    w(md_table(mo, formats={c: "{:+.1f}" for c in mo.columns if "%" in c}
               | {"actual fuel (million MMBtu)": "{:,.1f}"}))
    w("\nWhere the level error comes from (Sep-Dec, Jan-Aug fit): gross energy and fuel by unit group, "
      "actual vs model. CC* = suspect CC units.\n")
    w(md_table(res["by_group"], formats={c: "{:,.2f}" for c in res["by_group"].columns if c != "group"}))
    w("\nThe priority list loads the efficient CC units harder than they ran and replaces most "
      "steam-turbine (and part of the simple-cycle) output with CC output, so modelled fuel is low. Even "
      "with the actual committed set (B), equal-incremental dispatch moves output from ST to CC units, so, "
      "judged by these fitted curves and limits, actual operation is not heat-rate-economic hour by hour. "
      "Possible reasons, none tested here: "
      "multi-day commitment of slow units, transmission and local reliability constraints, gas-price "
      "differences between Texas locations, outages inside the month, and lower CC capacity in hot "
      "hours. The level bias matters less for tariff effects than the marginal response, which section 4 "
      "tests directly.\n")

    # ---- 4. Marginal
    w("## 4. Marginal heat rate: model vs CEMS (the error band)\n")
    w("Empirical marginal heat rate = OLS slope of the hour-to-hour change in actual non-CHP heat input "
      "on the change in non-CHP gross output (MMBtu per MWh gross), with HAC (Newey-West, 24 lags) "
      "standard errors. Model columns: the same estimator applied to each model's hourly heat on the same "
      f"output path. '{main} - actual' is the slope of (model change - actual change), with its own CI. "
      f"'{main} FD' is the mean over the group's hours of the model's central difference "
      f"(H(D + {delta:,.0f} MW) - H(D - {delta:,.0f} MW)) / {2 * delta:,.0f} MW with commitment re-solved, "
      "which is how a tariff run would read the model. A difference is assigned to the load quintile of "
      "its later hour. Load quintiles are fixed on the full-year non-CHP output.\n")
    fm = {c: "{:.2f}" for c in band.columns if c not in ("hours", "n")}
    fm.update({f"{main} error %": "{:+.1f}", f"{main} FD error %": "{:+.1f}",
               f"{main} - actual": "{:+.2f}", "mean output GW": "{:.1f}"})
    for key, title in [("test", "Out of sample: Jan-Aug fit, Sep-Dec hours"),
                       ("train", "In sample: Jan-Aug fit, Jan-Aug hours"),
                       ("full", "In sample: full-year fit, all hours")]:
        w(f"**{title}**\n")
        w(md_table(mt[key].drop(columns="actual SE"), formats=fm))
        w("")
    w(f"![marginal heat rate]({FIG_NAME})\n")
    w("By season and load tercile (terciles of the full-year output; A, B and priority-list columns as "
      "above):\n")
    for key, title in [("season_test", "Out of sample, Sep-Dec (Sep = summer, Oct-Nov = fall, Dec = winter)"),
                       ("season_full", "In sample, full-year fit, all hours")]:
        w(f"**{title}**\n")
        w(md_table(mt[key].drop(columns="actual SE"), formats=fm))
        w("")
    w("Cross-sectional slope between adjacent load quintiles (difference in mean heat input over "
      "difference in mean output; includes commitment changes across load levels):\n")
    for key, title in [("bins_test", "Out of sample, Sep-Dec"), ("bins_full", "In sample, full year")]:
        w(f"**{title}**\n")
        w(md_table(res[key], formats={c: "{:.2f}" for c in res[key].columns if c != "between"}))
        w("")
    w("Sensitivity of the model derivative to the step size (out of sample, mean over hours by quintile, "
      "MMBtu/MWh):\n")
    w(md_table(res["delta_sens"], formats={c: "{:.2f}" for c in res["delta_sens"].columns if c != "hours"}))
    w("\nMarginal heat-rate spread between the top load quintile and each lower quintile (MMBtu per "
      "MWh moved out of top-quintile hours into hours of the lower quintile; the CI treats the two "
      "quintile slopes as independent):\n")
    for key, title in [("spread_test", "Out of sample, Sep-Dec"), ("spread_full", "In sample, full year")]:
        w(f"**{title}**\n")
        t = res[key]
        w(md_table(t, formats={c: ("{:+.1f}" if "%" in c else "{:.2f}") for c in t.columns
                               if c not in ("from top quintile to", "actual 95% CI")}))
        w("")
    w("**Error band to use downstream.** For a tariff that changes non-CHP gas output by dE MWh in "
      "hours of load quintile q, the fuel change is about MHR_q x dE. The model's out-of-sample MHR "
      f"error by quintile (main model, +/-{delta:,.0f} MW derivative vs CEMS hour-to-hour slope) is the "
      f"'{main} FD error %' column of the quintile table; the same estimator on both sides gives the "
      f"'{main} error %' column. For a pure load shift (energy conserved), the error is that of the "
      "spread table. A tariff's estimated fuel or CO2 effect should be reported as distinguishable from "
      "model error only if it exceeds the model error for the quintiles (or spread) it touches and the "
      "CEMS sampling interval (actual 95% CI).\n")
    w("Out-of-sample marginal error of every priority-list configuration (hours-weighted mean absolute "
      "error over load quintiles; FD = the model's derivative as above):\n")
    cb = res["cfg_band"]
    w(md_table(cb, formats={c: "{:.2f}" for c in cb.columns if "MAE" in c}
               | {"Sep-Dec FD Q5-Q1 spread error %": "{:+.1f}"}))
    w(f"\nThe error band depends on the configuration. In C-sx the suspect CC units stay at their "
      f"actual output, and their actual heat input is part of the model's hourly heat. Its "
      f"hour-to-hour slope column therefore copies their actual response: they carry "
      f"{res['sus_share']:.3f} MWh of each MWh of non-CHP output change in Sep-Dec. This "
      "mechanically favours C-sx in the slope-based selection score. The FD column holds those units "
      "fixed and has no such advantage, so the FD band is the one to carry downstream.\n")
    w("Pattern: the model's marginal heat rate is too low at low and middle load, most of all in the "
      "shoulder seasons (spring and fall), where the CEMS hour-to-hour slope at a given load is higher "
      "than in winter or summer. One candidate explanation, not tested here, is maintenance outages "
      "inside a month, which the monthly availability rule cannot see, so the model meets load changes "
      "with efficient CC units that were not actually available.\n")

    # ---- 5. EIA-930
    w("## 5. Is gas the marginal resource? (EIA-930, ERCO)\n")
    w("Source: PUDL `core_eia930__hourly_net_generation_by_energy_source` and "
      "`core_eia930__hourly_operations`, balancing authority ERCO, adjusted series. EIA-930 categories "
      "changed on 2024-07-01 (solar, wind and hydro were split into variants) and are summed back into "
      "stable groups (`ttrl.data.eia930.SOURCE_GROUPS`). Battery storage is reported separately only "
      f"from {e930['storage_note_from']:%Y-%m-%d}; before that it is presumably inside 'other', so it is "
      "grouped with other.\n")
    w("Time alignment: correlation of hourly CEMS gas gross output (all units) with EIA-930 gas net "
      "generation at different timestamp offsets:\n")
    w(md_table(e930["lags"], formats={"corr (levels)": "{:.4f}", "corr (first differences)": "{:.4f}"}))
    w(f"\nThe best offset ({e930['best_lag']:+d} h) is used: the EIA-930 `datetime_utc` stamp marks the "
      "end of the hour. Level fit EIA-930 gas on CEMS gas gross: slope "
      f"{e930['level_fit'][0]:.3f}, R2 {e930['level_fit'][1]:.3f}; energy ratio EIA-930 gas net / CEMS "
      f"gas gross {e930['energy_ratio']:.3f} (EIA-930 net covers all gas units incl. those outside our "
      "CEMS set, CEMS is gross of auxiliary load).\n")
    w("Response of each source to a 1 MWh hour-to-hour change in ERCO demand, holding the changes in "
      "wind and solar fixed (OLS of the source's first difference on the first differences of demand, "
      "wind and solar; HAC 95% CI; by season in the right columns):\n")
    w(md_table(e930["shares"], formats={c: "{:.3f}" for c in ["all hours"] + SEASONS}))
    w("\nSame coefficient by local-standard-time block of the day:\n")
    w(md_table(e930["blocks"], formats={c: "{:.3f}" for c in e930["blocks"].columns if c != "LST hours"}))
    w(f"\nHour counts. Net load = demand - wind - solar. Among hours whose net load changed by at least "
      f"{e930['threshold']:.0f} MW from the previous hour ({100 - e930['small_share']:.1f}% of hours): share "
      "in which gas moved in the same direction, covered at least 50% / 80% of the change, and moved more "
      "in that direction than coal and than storage + other:\n")
    w(md_table(e930["hours"], formats={c: "{:.1f}" for c in e930["hours"].columns if "%" in c}))
    w("\nReading: gas is the main load-following resource in ERCOT in every season and time block, "
      "so treating a demand change as met mostly by gas is supported, but not entirely by gas: coal and "
      "storage take a measurable share, which a tariff run should carry through (use the non-CHP CEMS "
      "coefficient above to convert an ERCO demand change into a change of the dispatch target). These "
      "are associations in hour-to-hour data, not the response to a tariff; EIA-930 does not show "
      "curtailment or which unit set the price, so hours where curtailed wind or solar were marginal "
      "cannot be identified here.\n")

    # ---- 6. Limitations
    w("## 6. Assumptions and limitations\n")
    w("- No ambient temperature: curves and Pmax are constant through the year. Summer CC and CT capacity "
      "derating and heat-rate change are the subject of the next step; `CurveModel.arrays(temp_c)` "
      "already accepts (hours, units) temperature and dispatch takes (hours, units) curve arrays.")
    w("- One fuel price for all units: merit order by heat rate only. Gas price differences between Texas "
      "hubs, variable O&M, emission costs and bids are ignored.")
    w("- Commitment is hourly and memoryless (priority list + headroom); no start-up fuel, minimum up/down "
      "times or ramp limits. This is the main reason the fuel level is biased low.")
    w("- Availability comes from actual operation (unit ran at least once that month or day).")
    if CONFIGS[main]["suspect_exo"]:
        w(f"- In the main configuration ({main}) the suspect CC units are held at their actual output. In a "
          "tariff run they would stay at baseline output and every change would be met by the other "
          "units; their own response is not modelled.")
    w("- Gross output: CEMS gross load at the generator terminals, not net. For suspect CC units the "
      "steam turbine's output is probably missing from CEMS, so their curves are not plant heat rates and "
      "the dispatch target omits that steam-turbine output.")
    w("- CHP units are exogenous; their process-steam fuel is not modelled.")
    w("- The marginal comparison uses hour-to-hour variation. A tariff moves load in a planned, "
      "repeated pattern, which the real fleet could meet with different commitment than it uses for "
      "ordinary hour-to-hour changes; the error band therefore describes the model's fidelity to observed "
      "short-run responses.")
    w("- HAC standard errors are computed within groups of non-contiguous hours; treat the CIs as "
      "approximate.")
    w("- EIA-930 shares are associations, not causal effects; storage is only reported separately "
      "from late October 2024.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--delta", type=float, default=500.0,
                    help="step (MW) of the model's central-difference marginal heat rate")
    args = ap.parse_args()
    t_start = time.time()
    delta = args.delta
    deltas = sorted({delta, 100.0, 2000.0})
    fit_cfg = U.FitConfig()

    t0 = time.time()
    s = build_setup(args.year, fit_cfg)
    t_fit = time.time() - t0
    print(f"setup + fits: {t_fit:.1f} s; {len(s.fleet.unit_ids)} non-CHP units")

    # Model runs: curve set "train" (Jan-Aug) and "full" (year).
    fit_hours = {"train": s.train, "full": np.ones_like(s.train)}
    runs: dict[str, dict[str, Run]] = {}
    t_disp = np.nan
    for cs, table in s.curves.items():
        runs[cs] = {"A": run_curves_only(s, table), "B": run_actual_commitment(s, table)}
        for name, cfg in CONFIGS.items():
            t0 = time.time()
            runs[cs][name] = run_priority(s, table, cfg, fit_hours[cs], deltas)
            if name == "C" and cs == "train":
                t_disp = (time.time() - t0) / (1 + 2 * len(deltas))
            print(f"  {cs} {name}: {time.time() - t0:.1f} s")

    # Choose the main configuration on the fit window only.
    def weighted_mae(tab: pd.DataFrame, col: str) -> float:
        return float((tab[col].abs() * tab["n"]).sum() / tab["n"].sum())
    selection = {}
    for name in CONFIGS:
        tab = marginal_table(s, {name: runs["train"][name]}, s.train, s.load_bin,
                             s.bin_labels, name, delta)
        selection[name] = weighted_mae(tab, f"{name} error %")
    main_cfg = min(selection, key=selection.get)
    print("selection (Jan-Aug marginal MAE %):", {k: round(v, 2) for k, v in selection.items()})

    # Level metrics.
    windows = {"in-sample": ("train", s.train), "out-of-sample": ("train", s.test),
               "in-sample Sep-Dec": ("full", s.test),
               "in-sample year": ("full", np.ones_like(s.train))}
    rows = []
    for name in ["A", "B", *CONFIGS]:
        for wname, (cs, wmask) in windows.items():
            rows.append({"variant": name, "window": wname,
                         **level_metrics(s, runs[cs][name], wmask)})
    levels = pd.DataFrame(rows).set_index(["variant", "window"])

    order = ["A", "B", *CONFIGS]
    tr_runs = {k: runs["train"][k] for k in order}
    full_runs = {k: runs["full"][k] for k in order}
    terc_edges = np.quantile(s.target, [1 / 3, 2 / 3])
    terc = np.searchsorted(terc_edges, s.target, side="right")
    seas_groups = np.array([f"{se} T{t + 1}" for se, t in zip(s.season, terc)])
    seas_labels = [f"{se} T{t}" for se in SEASONS for t in (1, 2, 3)]
    seas_idx = np.array([seas_labels.index(g) for g in seas_groups])
    tl = [f"{lab} ({lo / 1e3:.1f}-{hi / 1e3:.1f} GW)" for lab, lo, hi in
          [(f"{se} T{t + 1}", np.r_[s.target.min(), terc_edges][t], np.r_[terc_edges, s.target.max()][t])
           for se in SEASONS for t in range(3)]]
    marg = {
        "test": marginal_table(s, tr_runs, s.test, s.load_bin, s.bin_labels, main_cfg, delta),
        "train": marginal_table(s, tr_runs, s.train, s.load_bin, s.bin_labels, main_cfg, delta),
        "full": marginal_table(s, full_runs, np.ones_like(s.train), s.load_bin, s.bin_labels,
                               main_cfg, delta),
        "season_test": marginal_table(s, tr_runs, s.test, seas_idx, tl, main_cfg, delta),
        "season_full": marginal_table(s, full_runs, np.ones_like(s.train), seas_idx, tl,
                                      main_cfg, delta),
    }
    band = marg["test"]

    # Out-of-sample marginal error of every priority-list configuration, so
    # the error band can be read for a configuration other than the main one.
    cfg_rows = []
    for name in CONFIGS:
        t = marginal_table(s, {name: runs["train"][name]}, s.test, s.load_bin,
                           s.bin_labels, name, delta)
        a_sp = t["actual"].iloc[-1] - t["actual"].iloc[0]
        fd_sp = t[f"{name} FD"].iloc[-1] - t[f"{name} FD"].iloc[0]
        cfg_rows.append({
            "configuration": name,
            "Jan-Aug slope MAE % (selection)": selection[name],
            "Sep-Dec slope MAE %": weighted_mae(t, f"{name} error %"),
            "Sep-Dec FD MAE %": weighted_mae(t, f"{name} FD error %"),
            "Sep-Dec FD error range %": f"{t[f'{name} FD error %'].min():+.1f} to "
                                        f"{t[f'{name} FD error %'].max():+.1f}",
            "Sep-Dec FD Q5-Q1 spread error %": 100 * (fd_sp / a_sp - 1)})
    # Share of the non-CHP output change carried by suspect CC units (Sep-Dec).
    sus = s.curves["train"]["suspect_cc"].to_numpy()
    both = s.test[1:] & s.test[:-1]
    sus_share = hac_slope(np.diff(s.fleet.energy[:, sus].sum(axis=1))[both],
                          np.diff(s.target)[both])[0]

    spread = {"actual_lo": band["actual"].iloc[0], "actual_hi": band["actual"].iloc[-1],
              "actual": band["actual"].iloc[-1] - band["actual"].iloc[0],
              "model": band[main_cfg].iloc[-1] - band[main_cfg].iloc[0],
              "model_fd": band[f"{main_cfg} FD"].iloc[-1] - band[f"{main_cfg} FD"].iloc[0]}

    delta_rows = []
    for gi, lab in enumerate(s.bin_labels):
        sel = s.test & (s.load_bin == gi)
        delta_rows.append({"hours": lab, **{f"+/-{d:,.0f} MW": float(np.nanmean(
            runs["train"][main_cfg].mhr[d][sel])) for d in deltas}})

    group_rows = []
    g_train = unit_groups(s.curves["train"])
    for k in sorted(np.unique(g_train)):
        g = g_train == k
        r_main, r_b = runs["train"][main_cfg], runs["train"]["B"]
        group_rows.append({
            "group": k,
            "actual TWh": s.fleet.energy[s.test][:, g].sum() / 1e6,
            f"{main_cfg} TWh": r_main.energy_by_group[k][s.test].sum() / 1e6,
            "B TWh": r_b.energy_by_group[k][s.test].sum() / 1e6,
            "actual fuel (million MMBtu)": s.fleet.heat[s.test][:, g].sum() / 1e6,
            f"{main_cfg} fuel (million MMBtu)": r_main.heat_by_group[k][s.test].sum() / 1e6,
            "B fuel (million MMBtu)": r_b.heat_by_group[k][s.test].sum() / 1e6,
        })

    t0 = time.time()
    e930 = eia930_analysis(s)
    print(f"EIA-930: {time.time() - t0:.1f} s")

    nc_heat = s.heat.sum()
    partial = (s.fleet.optime > 0) & (s.fleet.optime < 1)
    zero = (s.fleet.optime > 0) & (s.fleet.load <= 0)
    status_counts = dict(zip(*np.unique(runs["train"][main_cfg].status[s.test],
                                        return_counts=True)))
    res = {
        "levels": levels, "monthly": monthly_errors(s, tr_runs, main_cfg), "marg": marg,
        "bins_test": binned_level_slopes(s, tr_runs, s.test),
        "bins_full": binned_level_slopes(s, full_runs, np.ones_like(s.train)),
        "spread": spread, "delta_sens": pd.DataFrame(delta_rows),
        "by_group": pd.DataFrame(group_rows), "selection": selection,
        "headroom": {cs: runs[cs]["C"].headroom for cs in runs},
        "partial_share": 100 * s.fleet.heat[partial].sum() / nc_heat,
        "zero_share": 100 * s.fleet.heat[zero].sum() / nc_heat,
        "status": {int(k): int(v) for k, v in status_counts.items()},
        "status_b": {int(k): int(v) for k, v in zip(*np.unique(
            runs["train"]["B"].status[s.test], return_counts=True))},
        "spread_test": spread_table(marg["test"], main_cfg),
        "spread_full": spread_table(marg["full"], main_cfg),
        "cfg_band": pd.DataFrame(cfg_rows), "sus_share": sus_share,
    }

    # Outputs.
    OUTPUTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    DOCS_REPORTS.mkdir(parents=True, exist_ok=True)
    for cs, tab in s.curves_all.items():
        tab.to_parquet(PROCESSED / f"unit_curves_{s.year}_{cs}.parquet", index=False)
    s.curves_all["train"].to_csv(OUTPUTS / f"dispatch_unit_curves_{s.year}.csv", index=False)
    hourly = pd.DataFrame({"hour_utc": s.fleet.hours, "month_lst": s.month,
                           "target_mwh": s.target, "heat_actual": s.heat, "co2_actual": s.co2,
                           "chp_heat": s.chp_heat, "chp_energy": s.chp_energy})
    for cs in runs:
        for name in order:
            hourly[f"heat_{name}_{cs}"] = runs[cs][name].heat
        hourly[f"mhr_{main_cfg}_{cs}"] = runs[cs][main_cfg].mhr[delta]
    hourly.to_parquet(PROCESSED / f"dispatch_hourly_{s.year}.parquet", index=False)
    marginal_figure({"In sample: Jan-Aug fit, Jan-Aug hours": marg["train"],
                     "Out of sample: Jan-Aug fit, Sep-Dec hours": marg["test"]},
                    main_cfg, delta, OUTPUTS / FIG_NAME)

    runtime = {"fit": t_fit, "dispatch": t_disp, "total": time.time() - t_start}
    text = build_report(s, res, main_cfg, delta, e930, runtime, fit_cfg)
    out = OUTPUTS / "dispatch_validation.md"
    out.write_text(text)
    shutil.copy(out, DOCS_REPORTS / out.name)
    shutil.copy(OUTPUTS / FIG_NAME, DOCS_REPORTS / FIG_NAME)
    print(f"wrote {out} and docs/reports copy ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
