"""Temperature-aware S4 dispatch model: validation and error band (ERCOT gas, 2024).

Plugs the temperature-dependent heat-input curves and capacity derating of
``scripts/fit_heatrate_derating.py`` into the dispatch model of
``scripts/validate_dispatch.py`` and validates both versions against CEMS:

1. Fit windows. For each window the temperature table
   (``ttrl.supply.heatrate.fit_temperature_curves``, steady hours recomputed
   on the window) and static fallback curves (``ttrl.supply.units``) use only
   the window's months: Jan-Aug, the full year, and the 12 folds of a
   month-blocked cross-validation (fit on 11 months, predict the 12th).
2. Models (``ttrl.supply.fleetmodel.FleetModel``): "static" (the same units'
   curves fitted on the same hours without temperature terms, Pmax = 99th
   percentile of output) and "temperature" (H(P, T), Pavail(T)); for the
   Jan-Aug fit also "heat" and "derate" (one effect only). Configurations C
   (all non-CHP units dispatched) and C-sx (suspect CC units held at actual
   output). Headroom refitted per window and model.
3. Main configuration: lower Jan-Aug in-sample marginal error of the
   temperature model (fit-window data only).
4. Validation: fuel, CO2 and CT-hour levels (annual, monthly, hourly MAPE)
   and the marginal fuel per MWh (model +/-500 MW derivative vs CEMS
   hour-to-hour slope) by load quintile, season x load tercile and fleet
   temperature class; windows Sep-Dec (out of sample), in sample, and CV.
5. S4 error band: block bootstrap over local-standard-time weeks (day blocks
   as a sensitivity) of the out-of-sample pairs (CV and Sep-Dec), saved for
   ``ttrl.supply.impact``; with the step-size part (the model's response to
   each pair's own change, its confirmed share by size class of the change).
6. Temperature part of the marginal heat rate in the hottest 1% of hours.

Writes outputs/s4_integration.md (+ docs/reports copy), a figure,
outputs/s4_integration.json (numbers for scripts/s4_summary.py) and, in
data/processed/: s4_curves_<year>.parquet, s4_error_band_<year>_<mode>_<band>.parquet,
s4_model_<year>.json, s4_validation_hourly_<year>.parquet.

Usage:
    python scripts/integrate_s4.py [--year 2024] [--draws 2000] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttrl.data import pudl  # noqa: E402
from ttrl.paths import INTERIM, OUTPUTS, PROCESSED, ROOT  # noqa: E402
from ttrl.supply import errorband as EB  # noqa: E402
from ttrl.supply import fleetmodel as FM  # noqa: E402
from ttrl.supply import heatrate as H  # noqa: E402
from ttrl.supply import units as U  # noqa: E402
from ttrl.supply.dispatch import STATUS_OK as D_OK  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_MONTHS = list(range(1, 9))
TEST_MONTHS = list(range(9, 13))
CONFIG_NAMES = ["C", "C-sx"]
DELTA_MW = 500.0
HOT_SHARE = 0.01
SEASONS = ["winter", "spring", "summer", "fall"]
DOCS_REPORTS = ROOT / "docs" / "reports"
REPORT = "s4_integration.md"
FIG = "s4_integration.png"
KG = H.KG_PER_SHORT_TON
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PARAM_COLS = ["a", "b", "c", "d", "e", "pmin", "pmax", "p15", "kappa", "a0", "b0", "c0",
              "t_fit_lo", "t_fit_hi"]

# Shared with forked fit workers.
_G: dict = {}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def md_table(df: pd.DataFrame, formats: dict[str, str] | None = None,
             floatfmt: str = "{:,.2f}") -> str:
    """GitHub markdown table with per-column float formats."""
    formats = formats or {}

    def fmt(col, v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else formats.get(col, floatfmt).format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, r)) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def ci(v: float, lo: float, hi: float, f: str = "{:+.1f}") -> str:
    """'v [lo, hi]'."""
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def pct(model: float, actual: float) -> float:
    return 100.0 * (model / actual - 1.0)


def wmae(rel: np.ndarray, n: np.ndarray) -> float:
    """Pairs-weighted mean absolute value."""
    rel, n = np.asarray(rel, float), np.asarray(n, float)
    ok = np.isfinite(rel)
    return float((np.abs(rel[ok]) * n[ok]).sum() / n[ok].sum())


# --------------------------------------------------------------------------
# Inputs and fits
# --------------------------------------------------------------------------
def load_inputs(year: int) -> dict:
    """CEMS, unit tables, plant temperature, unit-hours and fleet arrays."""
    paths = pudl.build_ercot_gas_cems(year)
    cems = pd.read_parquet(paths["cems"])
    units = pd.read_parquet(paths["units"])
    gens = pd.read_parquet(paths["generators"])
    tpath = INTERIM / f"plant_temperature_{year}.parquet"
    if not tpath.exists():
        raise FileNotFoundError(f"{tpath} missing: run scripts/build_plant_weather.py first")
    temp = pd.read_parquet(tpath)
    uh = H.add_temperature(U.prepare_unit_hours(cems), units, temp)
    fd = FM.build_fleet_data(cems, units, temp, year)
    return {"year": year, "units": units, "gens": gens, "uh": uh, "fd": fd,
            "all_ids": sorted(uh["unit_id"].unique())}


def fit_window(months: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temperature table and completed curve table for a fit window (worker)."""
    g = _G
    w = g["uh"][g["uh"]["month"].isin(months)].copy()
    w["steady"] = H.steady_rows(w)                     # Pref from the window only
    full_year = len(months) == 12
    tt = H.fit_temperature_curves(w, g["units"], g["gens"],
                                  group_of=None if full_year else g["groups"])
    st = U.fit_unit_curves(w, g["units"], g["gens"], unit_ids=g["all_ids"])
    return tt, FM.complete_curve_table(tt, st, g["fd"].unit_ids)


def cached_fit(name: str, months: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fit_window with its result kept in data/interim, so an interrupted sequential run resumes.
    The cache belongs to the inputs of this year's run; delete s4_fit_<year>_* to refit."""
    p_tt = INTERIM / f"s4_fit_{_G['year']}_{name}_temperature.parquet"
    p_tab = INTERIM / f"s4_fit_{_G['year']}_{name}_curves.parquet"
    if p_tt.exists() and p_tab.exists():
        return pd.read_parquet(p_tt), pd.read_parquet(p_tab)
    t0 = time.time()
    tt, tab = fit_window(months)
    tt.to_parquet(p_tt)
    tab.to_parquet(p_tab)
    print(f"  fit {name}: {time.time() - t0:.0f} s", flush=True)
    return tt, tab


def fit_all_windows(inp: dict, workers: int) -> tuple[dict, dict]:
    """Full year first (it sets the group labels), then Jan-Aug and the CV folds."""
    _G.update(inp)
    tt_full, full = cached_fit("full", tuple(range(1, 13)))
    _G["groups"] = tt_full.set_index("unit_id")["group"]
    windows = {"janaug": tuple(TRAIN_MONTHS)}
    windows.update({f"cv{m:02d}": tuple(k for k in range(1, 13) if k != m) for m in range(1, 13)})
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            res = dict(zip(windows, ex.map(fit_window, windows.values())))
    else:
        # No fork (Windows): spawned workers would each need a copy of the inputs, which does not
        # fit in memory on a small machine, so the windows are fitted one after another (same fits),
        # each kept on disk as it finishes.
        res = {k: cached_fit(k, v) for k, v in windows.items()}
    raw = {"full": tt_full, **{k: v[0] for k, v in res.items()}}
    tables = {"full": full, **{k: v[1] for k, v in res.items()}}
    return raw, tables


def table_check(raw: dict) -> dict:
    """Largest parameter difference vs the curves step's files (same fits expected)."""
    out = {}
    for name, f in {"full": "unit_curves_{y}.parquet", "janaug": "unit_curves_{y}_janaug.parquet"}.items():
        p = PROCESSED / f.format(y=_G["year"])
        if not p.exists():
            out[name] = np.nan
            continue
        ref = pd.read_parquet(p).set_index("unit_id")
        mine = raw[name].set_index("unit_id").loc[ref.index, PARAM_COLS].to_numpy(float)
        out[name] = float(np.nanmax(np.abs(mine - ref[PARAM_COLS].to_numpy(float))))
    return out


# --------------------------------------------------------------------------
# Model runs
# --------------------------------------------------------------------------
def run_model(fd: FM.FleetData, table: pd.DataFrame, config: str, mode: str,
              fit: np.ndarray, ev: np.ndarray) -> dict:
    """Fit headroom on ``fit`` hours, dispatch and differentiate on ``ev`` hours.

    ``s_<metric>``: the model's change at hour t for the actual change of
    non-CHP output into t, X_t(target_t) - X_t(target_t - dx_t) (the step-size
    part of the error band); NaN where a run is not solved or the earlier
    level would be negative.
    """
    m = FM.FleetModel(table, mode, config)
    m.fit_headroom(fd.energy[fit], fd.on[fit], fd.temp[fit])
    sp = m.split(fd.energy[ev], fd.heat[ev], fd.co2[ev])
    r = m.run(sp["target"], fd.temp[ev], fd.avail[ev])
    mg = m.marginal(sp["target"], fd.temp[ev], fd.avail[ev], DELTA_MW)
    dx = np.r_[np.nan, np.diff(fd.target)][ev]
    prev = sp["target"] - np.nan_to_num(dx)
    rp = m.run(np.maximum(prev, 0.0), fd.temp[ev], fd.avail[ev])
    ok = np.isfinite(dx) & (prev >= 0) & (r.status == D_OK) & (rp.status == D_OK)
    step = lambda a, b: np.where(ok, a - b, np.nan)  # noqa: E731
    return {"heat": r.heat + sp["exo_heat"], "co2": r.co2 + sp["exo_co2"],
            "n_ct": r.n_ct.astype(float), "status": r.status, "m_fuel": mg["fuel"],
            "m_co2": mg["co2"], "m_ct": mg["ct"], "s_fuel": step(r.heat, rp.heat),
            "s_co2": step(r.co2, rp.co2), "s_ct": step(r.n_ct.astype(float), rp.n_ct.astype(float)),
            "headroom": m.headroom}


def all_runs(fd: FM.FleetData, tables: dict) -> dict:
    """(window, config, mode) -> hourly series over all T hours (NaN outside)."""
    t_n = len(fd.hours)
    allh = np.ones(t_n, bool)
    train = np.isin(fd.month, TRAIN_MONTHS)
    runs = {}
    plan = [("janaug", m) for m in ("static", "temperature", "heat", "derate")] + \
           [("full", m) for m in ("static", "temperature")]
    for win, mode in plan:
        fit = train if win == "janaug" else allh
        for cfg in CONFIG_NAMES:
            runs[(win, cfg, mode)] = run_model(fd, tables[win], cfg, mode, fit, allh)
    for cfg in CONFIG_NAMES:
        for mode in ("static", "temperature"):
            out = {k: np.full(t_n, np.nan) for k in ("heat", "co2", "n_ct", "m_fuel", "m_co2", "m_ct",
                                                     "s_fuel", "s_co2", "s_ct")}
            out["status"] = np.zeros(t_n, np.int8)
            for m in range(1, 13):
                ev = fd.month == m
                r = run_model(fd, tables[f"cv{m:02d}"], cfg, mode, ~ev, ev)
                for k in out:
                    out[k][ev] = r[k]
            runs[("cv", cfg, mode)] = out
    return runs


# --------------------------------------------------------------------------
# Validation tables
# --------------------------------------------------------------------------
BLOCK = "week"            # main bootstrap block (local-standard-time weeks); "day" = sensitivity
CLASS_LABELS = [f"{n} ({r} C)" for n, r in zip(EB.CLASS_NAMES, ["< 10", "10-25", "25-30", ">= 30"])]


class Ctx:
    """Hour labels and actual series shared by the tables."""

    def __init__(self, fd: FM.FleetData, weights: np.ndarray):
        self.fd = fd
        self.target = fd.target
        self.heat = fd.heat.sum(axis=1)
        self.co2 = fd.co2.sum(axis=1)
        self.n_ct = fd.on[:, fd.tech == "CT"].sum(axis=1).astype(float)
        self.ft = fd.fleet_temp(weights)
        self.q_edges = np.quantile(self.target, [0.2, 0.4, 0.6, 0.8])
        self.t_edges = np.quantile(self.target, [1 / 3, 2 / 3])
        self.qbin = np.searchsorted(self.q_edges, self.target, side="right")
        self.tbin = np.searchsorted(self.t_edges, self.target, side="right")
        lo, hi = np.r_[self.target.min(), self.q_edges], np.r_[self.q_edges, self.target.max()]
        self.q_labels = [f"Q{i + 1} ({a / 1e3:.1f}-{b / 1e3:.1f} GW)" for i, (a, b) in enumerate(zip(lo, hi))]
        self.cls = EB.temp_class(self.ft)
        self.train = np.isin(fd.month, TRAIN_MONTHS)
        self.test = np.isin(fd.month, TEST_MONTHS)
        self.allh = np.ones(len(self.target), bool)
        self.blocks = {"day": fd.day, "week": fd.day // 7}
        n_hot = int(round(HOT_SHARE * len(self.target)))
        self.hottest = np.zeros(len(self.target), bool)
        self.hottest[np.argsort(self.ft)[::-1][:n_hot]] = True

    def window(self, name: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluation hours and pair group of a named evaluation window."""
        return {"Sep-Dec (Jan-Aug fit)": (self.test, None),
                "Jan-Aug in sample": (self.train, None),
                "year in sample (full fit)": (self.allh, None),
                "CV (month folds)": (self.allh, self.fd.month)}[name]


RUN_OF = {"Sep-Dec (Jan-Aug fit)": "janaug", "Jan-Aug in sample": "janaug",
          "year in sample (full fit)": "full", "CV (month folds)": "cv"}
WINDOWS = list(RUN_OF)
CV, HOLDOUT = "CV (month folds)", "Sep-Dec (Jan-Aug fit)"


def pairs_for(c: Ctx, runs: dict, win: str, models: list[tuple[str, str]],
              metrics=("fuel",), block: str = BLOCK, with_step: bool = False) -> EB.Pairs:
    """Pairs of a window with marginals of several (config, mode) models; ``with_step`` adds
    the model's response to each pair's own change (pairs without it are dropped).

    Metric keys are '<metric>|<config>|<mode>'.
    """
    ev, group = c.window(win)
    actual_of = {"fuel": c.heat, "co2": c.co2, "ct": c.n_ct}
    actual, model, step = {}, {}, {}
    for cfg, mode in models:
        for k in metrics:
            key = f"{k}|{cfg}|{mode}"
            actual[key] = actual_of[k]
            model[key] = runs[(RUN_OF[win], cfg, mode)][f"m_{k}"]
            step[key] = runs[(RUN_OF[win], cfg, mode)][f"s_{k}"]
    return EB.make_pairs(c.target, actual, model, ev, c.blocks[block], group,
                         step=step if with_step else None)


def grouped(pairs: EB.Pairs, labels_per_hour: np.ndarray, names: list[str], n_draws: int,
            seed: int = 0) -> pd.DataFrame:
    """band_table for a grouping of the later hours (labels index into names; -1 = none)."""
    g = np.asarray(labels_per_hour)[pairs.t]
    keep = g >= 0
    sub = EB.Pairs(pairs.x[keep], {k: v[keep] for k, v in pairs.dy.items()},
                   {k: v[keep] for k, v in pairs.m.items()}, pairs.block[keep], pairs.t[keep])
    metrics = list(pairs.m)
    _, stats = EB.block_stats(sub, g[keep], len(names), metrics)
    point, draws = EB.bootstrap(stats, n_draws, seed, metrics)
    out = EB.band_table(point, draws, names, metrics=metrics)
    out["blocks"] = (stats[:, :, 0] > 0).sum(axis=0)
    return out


def flag(mask: np.ndarray) -> np.ndarray:
    """0 inside the mask, -1 outside (a one-group labelling for :func:`grouped`)."""
    return np.where(mask, 0, -1)


# --------------------------------------------------------------------------
# Validation tables
# --------------------------------------------------------------------------
def level_table(c: Ctx, runs: dict) -> pd.DataFrame:
    """Fuel, CO2, hourly MAPE and CT unit-hour errors by model and window."""
    rows = []
    for cfg in CONFIG_NAMES:
        for mode in ("static", "temperature"):
            row = {"config": cfg, "model": mode}
            for win, short in [(HOLDOUT, "Sep-Dec"), (CV, "CV"), ("year in sample (full fit)", "in-sample")]:
                ev, _ = c.window(win)
                r = runs[(RUN_OF[win], cfg, mode)]
                row[f"{short} fuel %"] = pct(r["heat"][ev].sum(), c.heat[ev].sum())
                row[f"{short} CO2 %"] = pct(r["co2"][ev].sum(), c.co2[ev].sum())
                row[f"{short} MAPE %"] = 100 * float(np.mean(np.abs(r["heat"][ev] / c.heat[ev] - 1)))
                row[f"{short} CT hours %"] = pct(r["n_ct"][ev].sum(), c.n_ct[ev].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_table(c: Ctx, runs: dict, cfg: str) -> pd.DataFrame:
    """Monthly fuel, CO2 and CT-hour errors of the main configuration."""
    rows = []
    for m in range(1, 13):
        w = c.fd.month == m
        row = {"month": pd.Timestamp(2000, m, 1).strftime("%b"),
               "actual fuel (million MMBtu)": c.heat[w].sum() / 1e6}
        for win, short in [("janaug", "Jan-Aug fit"), ("cv", "CV")]:
            for mode in ("static", "temperature"):
                row[f"{short} {mode} fuel %"] = pct(runs[(win, cfg, mode)]["heat"][w].sum(), c.heat[w].sum())
        row["CV temperature CO2 %"] = pct(runs[("cv", cfg, "temperature")]["co2"][w].sum(), c.co2[w].sum())
        row["actual CT unit-hours"] = int(c.n_ct[w].sum())
        row["CV temperature CT hours %"] = pct(runs[("cv", cfg, "temperature")]["n_ct"][w].sum(),
                                               c.n_ct[w].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def marginal_compare(c: Ctx, runs: dict, cfg: str, labels: np.ndarray, names: list[str],
                     n_draws: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Static vs temperature FD error (%) by group of hours, in every window.

    Returns the report table and, per window, the full band table.
    """
    out, full = pd.DataFrame({"hours": names, "CEMS (CV)": np.nan}), {}
    s, tm = f"fuel|{cfg}|static", f"fuel|{cfg}|temperature"
    for win in WINDOWS:
        p = pairs_for(c, runs, win, [(cfg, "static"), (cfg, "temperature")])
        t = grouped(p, labels, names, n_draws)
        out[f"{win}: static / temp %"] = [
            f"{a:+.1f} / {b:+.1f}" if np.isfinite(a) and n >= 30 else ""
            for a, b, n in zip(t[f"{s}_rel"], t[f"{tm}_rel"], t["pairs"])]
        if win == CV:
            out["CEMS (CV)"] = t[f"{tm}_cems"]
        full[win] = t
    return out, full


# --------------------------------------------------------------------------
# Error band
# --------------------------------------------------------------------------
def build_band(c: Ctx, runs: dict, cfg: str, mode: str, win: str, n_draws: int,
               block: str = BLOCK) -> EB.ErrorBand:
    """ErrorBand of one model from a window's out-of-sample pairs, with the step-size part."""
    p = pairs_for(c, runs, win, [(cfg, mode)], metrics=EB.METRICS, block=block, with_step=True)
    n_all = len(pairs_for(c, runs, win, [(cfg, mode)], metrics=EB.METRICS, block=block))
    p.dy = {k.split("|")[0]: v for k, v in p.dy.items()}
    p.m = {k.split("|")[0]: v for k, v in p.m.items()}
    p.step = {k.split("|")[0]: v for k, v in p.step.items()}
    short = {CV: "cv", HOLDOUT: "holdout"}[win]
    return EB.ErrorBand.from_pairs(
        p, c.ft, c.target, c.fd.month, c.t_edges, n_draws=n_draws, seed=0,
        meta={"id": f"{c.fd.year}-{short}-{cfg}-{mode}-{block}-step", "window": win, "config": cfg,
              "mode": mode, "delta_mw": DELTA_MW, "block": f"local {block}",
              "pairs_without_step_run": n_all - len(p)})


def summary_sets(c: Ctx) -> dict[str, tuple[np.ndarray, int]]:
    """Hour sets of the season / temperature summary: (mask, minimum pairs to show)."""
    sets = {"all hours": (c.allh, 30)}
    for s in SEASONS:
        sets[s] = (c.fd.season == s, 30)
    for i, lab in enumerate(CLASS_LABELS):
        sets[lab] = (c.cls == i, 30)
    for s in SEASONS:
        for i, k in enumerate(EB.CLASS_NAMES):
            sets[f"{s} {k}"] = ((c.fd.season == s) & (c.cls == i), 100)
    sets[f"hottest {HOT_SHARE:.0%} of hours"] = (c.hottest, 30)
    return sets


def season_class_band(c: Ctx, runs: dict, cfg: str, win: str, n_draws: int) -> pd.DataFrame:
    """Band of the marginal error by season and temperature class (both modes)."""
    p = pairs_for(c, runs, win, [(cfg, "temperature"), (cfg, "static")], metrics=("fuel", "co2"))
    f, s, co = (f"fuel|{cfg}|temperature", f"fuel|{cfg}|static", f"co2|{cfg}|temperature")
    rows = []
    for name, (mask, min_n) in summary_sets(c).items():
        if mask[p.t].sum() < min_n:
            continue
        t = grouped(p, flag(mask), [name], n_draws).iloc[0]
        rows.append({"hours": name, "pairs": int(t["pairs"]), "weeks": int(t["blocks"]),
                     "CEMS MMBtu/MWh": t[f"{f}_cems"],
                     "temperature model error %": ci(t[f"{f}_rel"], t[f"{f}_rel_lo"], t[f"{f}_rel_hi"]),
                     "error MMBtu/MWh": ci(t[f"{f}_err"], t[f"{f}_err_lo"], t[f"{f}_err_hi"], "{:+.2f}"),
                     "CO2 error kg/MWh": ci(KG * t[f"{co}_err"], KG * t[f"{co}_err_lo"],
                                             KG * t[f"{co}_err_hi"], "{:+.0f}"),
                     "temperature-free model error %": ci(t[f"{s}_rel"], t[f"{s}_rel_lo"], t[f"{s}_rel_hi"]),
                     "_rel": t[f"{f}_rel"], "_lo": t[f"{f}_rel_lo"], "_hi": t[f"{f}_rel_hi"],
                     "_srel": t[f"{s}_rel"], "_slo": t[f"{s}_rel_lo"], "_shi": t[f"{s}_rel_hi"],
                     "_cems": t[f"{f}_cems"], "_cems_lo": t[f"{f}_cems_lo"],
                     "_cems_hi": t[f"{f}_cems_hi"], "_model": t[f"{f}_model"],
                     "_smodel": t[f"{s}_model"]})
    return pd.DataFrame(rows)


def stratum_table(band: EB.ErrorBand, c: Ctx, runs: dict, cfg: str, mode: str, win: str,
                  n_draws: int) -> pd.DataFrame:
    """Per stratum of the band: CEMS and model marginal, error and interval, which error
    the impact function uses, and the width of the day-block interval relative to weeks."""
    st = band.stratum(c.ft, c.target, c.fd.month)
    t = grouped(pairs_for(c, runs, win, [(cfg, mode)], metrics=EB.METRICS), st, band.labels, n_draws)
    td = grouped(pairs_for(c, runs, win, [(cfg, mode)], metrics=("fuel",), block="day"),
                 st, band.labels, n_draws)
    f, co, ct = (f"{k}|{cfg}|{mode}" for k in EB.METRICS)
    pooled_to = band.meta.get("pooled_to", {})
    rows = []
    for i, r in t.iterrows():
        if r["pairs"] == 0:
            continue
        used = f"pooled: {pooled_to[str(i)]}" if str(i) in pooled_to else "own"
        rows.append({"stratum": r["stratum"], "pairs": int(r["pairs"]), "weeks": int(r["blocks"]),
                     "CEMS MMBtu/MWh": r[f"{f}_cems"], "model MMBtu/MWh": r[f"{f}_model"],
                     "fuel error %": ci(r[f"{f}_rel"], r[f"{f}_rel_lo"], r[f"{f}_rel_hi"]),
                     "CO2 error kg/MWh": ci(KG * r[f"{co}_err"], KG * r[f"{co}_err_lo"],
                                             KG * r[f"{co}_err_hi"], "{:+.0f}"),
                     "CEMS CT units per GW": 1e3 * r[f"{ct}_cems"],
                     "model CT units per GW": 1e3 * r[f"{ct}_model"],
                     "CT error per GW": ci(1e3 * r[f"{ct}_err"], 1e3 * r[f"{ct}_err_lo"],
                                            1e3 * r[f"{ct}_err_hi"], "{:+.2f}"),
                     "day/week band width": (td.at[i, f"{f}_rel_hi"] - td.at[i, f"{f}_rel_lo"])
                     / (r[f"{f}_rel_hi"] - r[f"{f}_rel_lo"]),
                     "impact function uses": used,
                     "_rel": r[f"{f}_rel"], "_lo": r[f"{f}_rel_lo"], "_hi": r[f"{f}_rel_hi"]})
    return pd.DataFrame(rows)


def beta_table(bands: dict) -> pd.DataFrame:
    """Confirmed share of the model's hour-level variation per load tercile (point [95% band])."""
    rows = []
    for (mode, short), b in bands.items():
        if short != "cv":
            continue
        for g in range(b.n_load):
            row = {"model": {"temperature": "temperature-aware", "static": "temperature-free"}[mode],
                   "load tercile": f"T{g + 1}"}
            for j, k in enumerate(EB.METRICS):
                lo, hi = np.quantile(b.beta[:, g, j], [0.025, 0.975])
                row[f"{k} beta"] = ci(b.beta_point[g, j], lo, hi, "{:.2f}")
            rows.append(row)
    return pd.DataFrame(rows)


def beta2_table(bands: dict) -> pd.DataFrame:
    """Confirmed share of the step-size part per class of |change| (point [95% band])."""
    rows = []
    for (mode, short), b in bands.items():
        if short != "cv" or b.beta2 is None:
            continue
        for c, lab in enumerate(EB.size_labels(b.size_edges)):
            row = {"model": {"temperature": "temperature-aware", "static": "temperature-free"}[mode],
                   "|change|": lab, "pairs": int(b.meta["pairs_by_size"][c])}
            for j, k in enumerate(EB.METRICS):
                lo, hi = np.quantile(b.beta2[:, c, j], [0.025, 0.975])
                row[f"{k} beta2"] = ci(b.beta2_point[c, j], lo, hi, "{:.2f}")
            rows.append(row)
    return pd.DataFrame(rows)


MOVES = [("summer T3 hot", "summer T2"), ("summer T3 hot", "summer T1"),
         ("summer T3 hot", "summer T3"), ("summer T3", "summer T1"), ("winter T3", "winter T1"),
         ("spring T3", "spring T1"), ("fall T3", "fall T1")]


def shift_rows(c: Ctx, runs: dict, cfg: str, mode: str, win: str, n_draws: int,
               band: EB.ErrorBand) -> pd.DataFrame:
    """Fuel and CO2 saved per MWh moved between strata: model vs CEMS with the band."""
    p = pairs_for(c, runs, win, [(cfg, mode)], metrics=("fuel", "co2"))
    labels = band.labels
    s_of = band.stratum(c.ft, c.target, c.fd.month)[p.t]
    _, stats = EB.block_stats(p, s_of, len(labels), list(p.m))
    point, draws = EB.bootstrap(stats, n_draws, 0, list(p.m))
    ix = {lab: i for i, lab in enumerate(labels)}
    moves = [(ix[a], ix[b]) for a, b in MOVES if point["n"][ix[a]] >= 30 and point["n"][ix[b]] >= 30]
    f = EB.shift_table(point, draws, labels, moves, metric=f"fuel|{cfg}|{mode}")
    co = EB.shift_table(point, draws, labels, moves, metric=f"co2|{cfg}|{mode}")
    f["CO2 error kg/MWh"] = [ci(KG * e, KG * lo, KG * hi, "{:+.0f}")
                             for e, lo, hi in zip(co["error"], co["error lo"], co["error hi"])]
    return f


# --------------------------------------------------------------------------
# Temperature part of the marginal heat rate in the hottest hours
# --------------------------------------------------------------------------
def hot_hour_value(fd: FM.FleetData, table: pd.DataFrame, cfg: str, hot: np.ndarray) -> dict:
    """FD at actual temperatures minus FD with every plant at 15 C (full-year fit).

    Same output target, availability and headroom (fitted with the
    temperature-aware Pmax on all hours) in every run.
    """
    allh = np.ones(len(fd.hours), bool)
    base = FM.FleetModel(table, "temperature", cfg)
    hr = base.fit_headroom(fd.energy[allh], fd.on[allh], fd.temp[allh])
    sp = base.split(fd.energy[hot])
    t15 = np.full_like(fd.temp[hot], H.T_REF)
    m15 = base.marginal(sp["target"], t15, fd.avail[hot], DELTA_MW)
    out = {"fd15": float(np.nanmean(m15["fuel"])), "co2_15": KG * float(np.nanmean(m15["co2"])),
           "ct15": 1e3 * float(np.nanmean(m15["ct"]))}
    for mode in ("temperature", "heat", "derate"):
        m = FM.FleetModel(table, mode, cfg, headroom=hr)
        mg = m.marginal(sp["target"], fd.temp[hot], fd.avail[hot], DELTA_MW)
        out[mode] = {"fuel": float(np.nanmean(mg["fuel"] - m15["fuel"])),
                     "co2_kg": KG * float(np.nanmean(mg["co2"] - m15["co2"])),
                     "ct_per_gw": 1e3 * float(np.nanmean(mg["ct"] - m15["ct"])),
                     "n_ok": int(np.isfinite(mg["fuel"] - m15["fuel"]).sum())}
    out["pct"] = 100 * out["temperature"]["fuel"] / out["fd15"]
    out["target_gw"] = float(sp["target"].mean() / 1e3)
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors=INK2)


def figure(quint: pd.DataFrame, strat: pd.DataFrame, strat_static: pd.DataFrame,
           cfg: str, path) -> None:
    """Left: marginal fuel by load quintile (CV). Right: band of the error by stratum."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
    t = quint
    x = np.arange(len(t))
    s, tm = f"fuel|{cfg}|static", f"fuel|{cfg}|temperature"
    a1.fill_between(x, t[f"{tm}_cems_lo"], t[f"{tm}_cems_hi"], color=BLUE, alpha=0.12, lw=0)
    a1.plot(x, t[f"{tm}_cems"], color=BLUE, lw=2, marker="o", ms=8, mec="white", mew=2,
            label="CEMS hour-to-hour slope (95% band)")
    a1.plot(x, t[f"{s}_model"], color=ORANGE, lw=2, marker="s", ms=8, mec="white", mew=2,
            label="model, temperature-free")
    a1.plot(x, t[f"{tm}_model"], color=AQUA, lw=2, marker="^", ms=8, mec="white", mew=2,
            label="model, temperature-aware")
    a1.set_xticks(x, [lab.split(" ")[0] for lab in t["stratum"]])
    a1.set_xlabel("load quintile of non-CHP gas output", color=INK2)
    a1.set_ylabel("marginal fuel (MMBtu per MWh gross)", color=INK2)
    a1.set_title("Marginal fuel per MWh, 12-fold month CV", color=INK, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    _style(a1)
    st = strat[strat["impact function uses"] == "own"].set_index("stratum")
    ss = strat_static.set_index("stratum").reindex(st.index)
    y = np.arange(len(st))[::-1]
    a2.axvline(0, color="#b9b8b2", lw=1)
    a2.errorbar(ss["_rel"], y + 0.17, xerr=[ss["_rel"] - ss["_lo"], ss["_hi"] - ss["_rel"]],
                fmt="s", color=ORANGE, ms=6, mec="white", capsize=0, lw=2, label="temperature-free")
    a2.errorbar(st["_rel"], y - 0.17, xerr=[st["_rel"] - st["_lo"], st["_hi"] - st["_rel"]],
                fmt="^", color=AQUA, ms=7, mec="white", capsize=0, lw=2, label="temperature-aware")
    a2.set_yticks(y, st.index)
    a2.set_xlabel("model minus CEMS marginal fuel, % (95% week-block bootstrap)", color=INK2)
    a2.set_title("S4 error band by stratum (CV; hot = fleet >= 30 C)", color=INK, fontsize=11, loc="left")
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    _style(a2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(r: dict, path) -> None:
    L = []
    w = L.append
    cfg = r["cfg"]
    band = r["band_cv"]
    bt = r["strat_cv"]
    sc = r["season_cv"].set_index("hours")
    lv = r["levels"].set_index(["config", "model"])
    hv = r["hot_value"]
    hot_lab = CLASS_LABELS[3]
    hh = sc.loc[f"hottest {HOT_SHARE:.0%} of hours"]
    own = bt["impact function uses"] == "own"
    w(f"# S4 integration: temperature-aware dispatch, validation and error band (ERCOT gas, {r['year']})\n")
    w(f"Generated by `scripts/integrate_s4.py` on {date.today().isoformat()} from EPA CEMS (PUDL "
      f"`{pudl.DEFAULT_RELEASE}`), EIA-860 and the ISD-lite plant temperatures. Every number below was "
      "computed by that script. Whole S4 pipeline: `python scripts/run_s4.py` (section 7).\n")

    # ---- Summary
    w("## Summary\n")
    w(f"- **Error band first.** The S4 error band is the uncertainty of the model's marginal fuel and CO2 "
      f"per MWh of gas output moved, measured out of sample against the CEMS hour-to-hour slope "
      f"(model +/-{DELTA_MW:,.0f} MW derivative vs slope; {r['n_draws']:,} bootstrap draws of whole local "
      f"weeks; section 1). Over all hours of the 12-fold month cross-validation the temperature-aware "
      f"model (configuration {cfg}) is off by {sc.at['all hours', 'temperature model error %']}% "
      f"(95% band). By season: " + "; ".join(f"{s} {sc.at[s, 'temperature model error %']}%" for s in SEASONS)
      + ". By fleet temperature: " + "; ".join(f"{k.split(' (')[0]} {sc.at[k, 'temperature model error %']}%"
                                               for k in CLASS_LABELS if k in sc.index)
      + f". Across the {int(own.sum())} strata of the band (season x load tercile x hot hour) that use their "
      f"own error, it runs from {bt.loc[own, '_rel'].min():+.1f}% to {bt.loc[own, '_rel'].max():+.1f}% "
      f"(pairs-weighted mean absolute {r['band_wmae']:.1f}%). Inside a stratum CEMS confirms only "
      f"{band.beta_point[0, 0]:.2f} (bottom load tercile) to {band.beta_point[-1, 0]:.2f} (top tercile) of "
      "the model's hour-to-hour variation in marginal fuel; the band removes the rest. A tariff's fuel or "
      "CO2 effect is distinguishable from model error only "
      "where the impact function's band, which applies these corrections over the hours a profile "
      "touches, excludes zero.")
    cv_t, cv_s = lv.loc[(cfg, "temperature")], lv.loc[(cfg, "static")]
    w(f"- **Temperature in the model.** Unit heat input H(P, T) and available capacity Pavail(T) now follow "
      f"each plant's hourly temperature; the temperature-free model stays available (`mode=\"static\"`). "
      f"Fuel level: CV {cv_s['CV fuel %']:+.1f}% temperature-free vs {cv_t['CV fuel %']:+.1f}% "
      f"temperature-aware; Sep-Dec out of sample {cv_s['Sep-Dec fuel %']:+.1f}% vs "
      f"{cv_t['Sep-Dec fuel %']:+.1f}%; hourly fuel MAPE (CV) {cv_s['CV MAPE %']:.1f}% vs "
      f"{cv_t['CV MAPE %']:.1f}%. The level stays biased low, as in the dispatch step (which attributes it to "
      "the missing start-up fuel and the hourly priority-list commitment); tariff effects depend on the "
      "marginal response, not the level.")
    qm = r["quint_mae"]
    w(f"- **Marginal fuel per MWh: temperature {'improves' if r['better'] else 'does not improve'} the "
      f"fit.** Pairs-weighted mean absolute error over load quintiles, CV {qm[CV][0]:.1f}% temperature-free vs {qm[CV][1]:.1f}% "
      f"temperature-aware; Sep-Dec {qm[HOLDOUT][0]:.1f}% vs {qm[HOLDOUT][1]:.1f}%. In hot hours (fleet "
      f">= 30 C, CV) the error is {sc.at[hot_lab, 'temperature-free model error %']}% temperature-free and "
      f"{sc.at[hot_lab, 'temperature model error %']}% temperature-aware. {r['verdict']}")
    w(f"- **Temperature part at the margin: {'confirmed' if r['resolved'] else 'not confirmed'} by the "
      f"data.** In the hottest 1% of hours "
      f"({hv['n_hours']} h, mean fleet temperature {hv['mean_temp']:.1f} C) the full-year model's marginal "
      f"fuel is {hv['fd15']:.2f} MMBtu/MWh with every plant at 15 C and "
      f"{hv['fd15'] + hv['temperature']['fuel']:.2f} at the actual temperatures: "
      f"{hv['temperature']['fuel']:+.3f} MMBtu/MWh ({hv['pct']:+.1f}%), {hv['temperature']['co2_kg']:+.1f} kg "
      f"CO2/MWh (derating {hv['derate']['fuel']:+.3f}, heat-rate terms {hv['heat']['fuel']:+.3f}). "
      f"{r['hot_verdict']}")
    w(f"- **CT hours.** CV level error of simple-cycle CT unit-hours: {cv_s['CV CT hours %']:+.0f}% "
      f"temperature-free, {cv_t['CV CT hours %']:+.0f}% temperature-aware, and from "
      f"{r['monthly']['CV temperature CT hours %'].min():+.0f}% to "
      f"{r['monthly']['CV temperature CT hours %'].max():+.0f}% by month. At low load the model starts no "
      "CTs at the margin while ERCOT did; CT-hour effects are the least reliable output (section 1).\n")

    # ---- 1. Error band
    w("## 1. The S4 error band (read this first)\n")
    w("**Definition.** For hours of a stratum s, the CEMS marginal a_s is the OLS slope of the "
      "hour-to-hour change in actual non-CHP heat input (CO2; number of simple-cycle CT units producing) on "
      "the change in actual non-CHP gross output; the model marginal m_s is the mean of the model's "
      f"central difference (target +/- {DELTA_MW:,.0f} MW, commitment re-solved) over the same hours. "
      "Error = m_s - a_s (relative: m_s / a_s - 1). Inside a stratum the model's marginal also varies from "
      "hour to hour; beta, the share of that variation which CEMS confirms (coefficient on (m_t - m_s) x dx "
      "when the actual change is regressed stratum by stratum on [1, dx, (m_t - m_s) dx], pooled by load "
      "tercile), says how far the model's hour-level answers can be trusted. A pair (t-1, t) lies inside one "
      "evaluation window and, "
      "for the cross-validation, inside one fold (month); it belongs to the stratum and block of its later "
      f"hour. Band: {r['n_draws']:,} block-bootstrap draws, whole local-standard-time weeks resampled with "
      "replacement, 2.5th-97.5th percentiles. It contains the model's systematic error in the stratum and "
      "the sampling error of the CEMS slope. Blocks are 7-day periods because the errors persist across "
      f"days: day blocks, the first choice, give intervals {r['day_week_all']:.2f} times as wide over all "
      f"hours and a median {r['day_week_median']:.2f} times as wide per stratum.\n")
    w("**Evaluation windows.** (i) 12-fold month-blocked cross-validation (CV): each month is predicted by "
      "curves, CO2 factors and headroom fitted on the other 11 months, so every season is out of sample, "
      "including the July-August heat. (ii) The fixed split of the dispatch step: Jan-Aug fit, Sep-Dec "
      "validation (summer = September only, no spring). The impact function uses the CV band; the Sep-Dec "
      "band is saved too.\n")
    w(f"**Strata of the band** (used by `ttrl.supply.impact`): season x tercile of non-CHP gross output "
      f"(2024 edges {' and '.join(f'{e / 1e3:.1f}' for e in r['t_edges'])} GW) x hot hour (P15-weighted "
      "fleet temperature >= 30 C). The first version used temperature classes only (< 10, 10-25, >= 25 C, "
      "fixed before computing any error). Those cells mixed errors of opposite sign (season x class table "
      f"below: spring mild {sc.at['spring mild', 'temperature model error %']}%, fall mild "
      f"{sc.at['fall mild', 'temperature model error %']}%, winter mild "
      f"{sc.at['winter mild', 'temperature model error %']}%; hot >= 30 C "
      f"{sc.at[hot_lab, 'temperature model error %']}% vs warm 25-30 C "
      f"{sc.at[CLASS_LABELS[2], 'temperature model error %']}%), i.e. the pattern by season and load that "
      "the dispatch step reported and the hot-hour overstatement that the heat-rate step reported. The "
      "strata were changed to those dimensions, so this choice used the validation data once. Cells with "
      f"fewer than {band.meta['min_pairs']} pairs or {band.meta['min_blocks']} weeks take the error of "
      f"their season x tercile parent. Temperature-aware model, configuration {cfg}, CV (strata with >= 30 "
      "pairs):\n")
    show = ["stratum", "pairs", "weeks", "CEMS MMBtu/MWh", "model MMBtu/MWh", "fuel error %",
            "CO2 error kg/MWh", "CEMS CT units per GW", "model CT units per GW", "CT error per GW",
            "impact function uses"]
    big = bt["pairs"] >= 30
    w(md_table(bt.loc[big, show], {"CEMS MMBtu/MWh": "{:.2f}", "model MMBtu/MWh": "{:.2f}",
                                   "CEMS CT units per GW": "{:.2f}", "model CT units per GW": "{:.2f}"}))
    small = ", ".join(f"{a} ({n} pairs; {u})" for a, n, u in
                      zip(bt.loc[~big, "stratum"], bt.loc[~big, "pairs"], bt.loc[~big, "impact function uses"]))
    w("\nCT units per GW: change in the number of simple-cycle CT units producing per GW change of "
      f"non-CHP output. Strata with fewer than 30 pairs: {small or 'none'}; strata without pairs (e.g. "
      "hot winter hours) take their parent's error.\n")
    w("Confirmed share beta of the model's within-stratum hour-level variation (CV, configuration "
      f"{cfg}; 1 = fully confirmed, 0 = noise):\n")
    w(md_table(r["beta_tab"]))
    w("\n**Step-size part.** The model's change for a step of a given size differs from its +/- "
      f"{DELTA_MW:,.0f} MW marginal times the step: by curvature for large steps and by commitment jumps for "
      "small ones. For every pair the model is also run at the earlier hour's output level (same hour's "
      "temperatures and availability), giving its change for the pair's own step, S_t = X_t(target_t) - "
      "X_t(target_t - dx). The share of S_t - m_t dx that CEMS confirms, beta2, is estimated per class of "
      "|dx| jointly with beta (within-stratum regression of the actual change on [1, dx, (m_t - m_s) dx by "
      "load tercile, (S_t - m_t dx) by size class]); beta above comes from this joint fit. "
      f"{r['pairs_without_step_run']} CV pairs without a solved step run are left out of the band.\n")
    w(md_table(r["beta2_tab"]))
    w("\nThe impact function uses, for each hour and draw, the CEMS stratum marginal, plus beta times the "
      "model's hour-level deviation from its stratum mean (its +/- 500 MW marginal minus the stratum mean, "
      "times the change), plus beta2 of the change's size class times the rest of the model's answer "
      "(curvature and commitment jumps). The noisy parts of the model's answer are removed and their "
      "uncertainty enters the band. The beta step was added after a first version of the worked example "
      "showed the model's mild-day result changing sign from day to day; the step-size part after the "
      "worked example showed a 250 MW shift looking significant where 500 MW did not, and after the ERCOT "
      "environment check showed hourly changes of several GW. The size classes were "
      "set after one diagnostic run on these pairs.\n")
    w(f"By season and fleet temperature class (CV; configuration {cfg}; season x class cells with >= 100 "
      "pairs). 'error MMBtu/MWh' and 'CO2 error' are for the temperature-aware model:\n")
    cols = ["hours", "pairs", "weeks", "CEMS MMBtu/MWh", "temperature model error %", "error MMBtu/MWh",
            "CO2 error kg/MWh", "temperature-free model error %"]
    w(md_table(r["season_cv"][cols], {"CEMS MMBtu/MWh": "{:.2f}"}))
    w("\nSame, Sep-Dec out of sample (Jan-Aug fit):\n")
    w(md_table(r["season_ho"][cols], {"CEMS MMBtu/MWh": "{:.2f}"}))
    w("\n**Per MWh shifted.** Fuel saved per MWh moved from one stratum to another = marginal(from) - "
      "marginal(to). Model vs CEMS, CV, temperature-aware model; the interval combines both strata's "
      "errors draw by draw:\n")
    sh = r["shifts"]
    sh = sh.assign(**{"error MMBtu/MWh": [ci(e, lo, hi, "{:+.2f}") for e, lo, hi in
                                          zip(sh["error"], sh["error lo"], sh["error hi"])]})
    sh["error %"] = sh["error %"].where(sh["CEMS saving"].abs() >= 0.5)
    w(md_table(sh[["from", "to", "model saving", "CEMS saving", "error MMBtu/MWh", "error %",
                   "CO2 error kg/MWh"]], {"model saving": "{:.2f}", "CEMS saving": "{:.2f}",
                                         "error %": "{:+.0f}"}))
    w("\nSavings in MMBtu per MWh moved; 'error %' is left empty where the CEMS saving is below 0.5 "
      "MMBtu/MWh in absolute value.")
    w(f"\n**Threshold rule.** A modelled tariff effect (fuel, CO2 or CT hours) counts as larger than model "
      "error only if the 95% band that `S4Model.impact` returns excludes zero; a difference between two "
      "tariffs only if the band from `compare` excludes zero. Per MWh shifted, the error column above is "
      f"the bias to remove and its interval the uncertainty left. Configuration {r['other_cfg']} instead of "
      f"{cfg} (all hours, CV): {r['other_cfg_band']}%. The band does not cover errors that hour-to-hour "
      "data cannot show, e.g. a planned, repeated load shift that the fleet meets with different "
      "commitment.\n")
    w("![S4 integration](s4_integration.png)\n")
    w("Figure: left, marginal fuel per MWh by load quintile (CV); right, the band of the strata that use "
      "their own error (thinner strata take their parent's).\n")

    # ---- 2. Model
    w("## 2. What changed in the dispatch model\n")
    w("- `ttrl.supply.fleetmodel.FleetModel(table, mode, config)`: mode \"temperature\" uses, per unit and "
      "hour, H(P, T) = a + bP + cP^2 + d(T-15) + e(T-15)P (heat terms not extrapolated beyond the unit's "
      "fitted temperature range) and Pmax(T) = P15 (1 - kappa/100 max(T-15, 0)), at the hour-beginning "
      "temperature of the unit's plant. The priority order (full-load heat rate at Pmax(T)), the headroom "
      "rule and equal-incremental dispatch then follow the hour's curves. Mode \"static\" keeps the "
      "temperature-free curves of the same units fitted on the same hours (Pmax = 99th percentile of "
      "output); \"heat\" and \"derate\" switch on one effect only.")
    w(f"- Curves per fit window: temperature table of `ttrl.supply.heatrate` on the window's steady hours; "
      f"units without a usable fit take the static (possibly pooled) curve of `ttrl.supply.units` with "
      f"group-median temperature terms and kappa ({r['n_fallback']} of {r['n_units']} non-CHP units in the "
      f"full-year table). CO2 per MMBtu from the window's own hours. Check: the full-year and Jan-Aug "
      f"temperature tables refitted here differ from the curves step's files by at most "
      f"{r['table_check']['full']:.1e} and {r['table_check']['janaug']:.1e} in any curve parameter.")
    w("- Configuration: C dispatches every non-CHP unit; C-sx holds the suspect CC units (probably CT "
      "output only) at actual output. Main configuration: the lower Jan-Aug in-sample marginal error of the "
      "temperature-aware model (pairs-weighted MAE of the derivative over load quintiles, fit-window data "
      "only; unlike the dispatch step's slope-based score it gives no credit for copied actual heat): "
      + ", ".join(f"{k} {v:.2f}%" for k, v in r["selection"].items())
      + f" -> **{cfg}**. Headroom refitted per window and model; full-year temperature-aware fit: "
      f"r0 = {r['headroom']['temperature'][0]:,.0f} MW, r1 = {r['headroom']['temperature'][1]:.3f} "
      f"(temperature-free: {r['headroom']['static'][0]:,.0f} MW, {r['headroom']['static'][1]:.3f}).")
    w("- CHP units (their fuel also makes process steam) are outside the model in every run and every "
      "metric; all fuel, CO2 and output numbers here are non-CHP.\n")

    # ---- 3. Validation
    w("## 3. Validation: temperature-free vs temperature-aware\n")
    w("### 3.1 Fuel, CO2 and CT-hour levels\n")
    w("Model minus actual over the window's hours (non-CHP). MAPE = mean absolute % error of hourly fuel. "
      "CT hours = unit-hours of simple-cycle CT units with output > 0. Windows: Sep-Dec (Jan-Aug fit, out "
      "of sample), CV (each month from the other 11), in-sample (full-year fit, all hours).\n")
    lt = r["levels"]
    w(md_table(lt, {c: ("{:.2f}" if "MAPE" in c else "{:+.1f}") for c in lt.columns if "%" in c}))
    w(f"\nMonthly (configuration {cfg}):\n")
    mt = r["monthly"]
    w(md_table(mt, {c: "{:+.1f}" for c in mt.columns if "%" in c} | {"actual fuel (million MMBtu)": "{:.1f}"}))
    w("\n### 3.2 Marginal fuel per MWh by load level, season and temperature\n")
    w(f"Model +/-{DELTA_MW:,.0f} MW derivative minus CEMS hour-to-hour slope, % of the CEMS slope, "
      f"'temperature-free / temperature-aware', configuration {cfg}. 'CEMS (CV)' = CEMS slope over the "
      "CV pairs (MMBtu/MWh). Load quintiles and terciles of the 2024 non-CHP output; empty where a window "
      "has fewer than 30 pairs.\n")
    w(md_table(r["quint"], {"CEMS (CV)": "{:.2f}"}))
    w("\nPairs-weighted mean absolute error over the quintiles (temperature-free / temperature-aware): "
      + "; ".join(f"{k} {a:.1f}% / {b:.1f}%" for k, (a, b) in qm.items()) + ".\n")
    w("By season and load tercile:\n")
    w(md_table(r["season_terc"], {"CEMS (CV)": "{:.2f}"}))
    w("\nBy fleet temperature class:\n")
    w(md_table(r["class_tab"], {"CEMS (CV)": "{:.2f}"}))
    w(f"\nOne effect at a time (Sep-Dec, Jan-Aug fit, {cfg}; pairs-weighted MAE over load quintiles): "
      + ", ".join(f"{k} {v:.1f}%" for k, v in r["decomp"].items()) + ".\n")

    # ---- 4. Hot hours
    w("## 4. Temperature part of the marginal fuel in the hottest hours\n")
    w(f"Hottest 1% of hours by fleet temperature ({hv['n_hours']} h, mean {hv['mean_temp']:.1f} C, "
      f"mean dispatched target {hv['target_gw']:.1f} GW), full-year fit. +/-{DELTA_MW:,.0f} MW derivative "
      "at the actual plant temperatures minus the same with every plant at 15 C (same target, "
      "availability and headroom).\n")
    rows = [{"configuration": k, "MMBtu/MWh at 15 C": v["fd15"],
             "temperature part MMBtu/MWh": v["temperature"]["fuel"], "%": v["pct"],
             "kg CO2/MWh": v["temperature"]["co2_kg"], "derating only": v["derate"]["fuel"],
             "heat-rate terms only": v["heat"]["fuel"],
             "CT units per GW, temperature part": v["temperature"]["ct_per_gw"],
             "hours solved": v["temperature"]["n_ok"]} for k, v in r["hot_values"].items()]
    w(md_table(pd.DataFrame(rows), {"MMBtu/MWh at 15 C": "{:.2f}", "temperature part MMBtu/MWh": "{:+.3f}",
                                    "%": "{:+.1f}", "kg CO2/MWh": "{:+.1f}", "derating only": "{:+.3f}",
                                    "heat-rate terms only": "{:+.3f}",
                                    "CT units per GW, temperature part": "{:+.2f}"}))
    w(f"\nIn the same hours, out of sample (CV): CEMS slope {hh['_cems']:.2f} MMBtu/MWh (95% band "
      f"{hh['_cems_lo']:.2f} to {hh['_cems_hi']:.2f}); temperature-free model {hh['_smodel']:.2f} "
      f"({hh['temperature-free model error %']}%); temperature-aware model {hh['_model']:.2f} "
      f"({hh['temperature model error %']}%). {r['hot_verdict']}\n")

    # ---- 5. Function
    w("## 5. The function for the RL environment\n")
    w("```python\nfrom ttrl.supply.impact import S4Model, shift_profile, compare\n"
      "model = S4Model.load()            # temperature-aware, CV error band, 2024 fleet\n"
      "res = model.impact(delta_mw, hours, temp_c=None, base_mw=None, gas_share=1.0)\n"
      "res.totals   # fuel_mmbtu, co2_short_tons, co2_tonnes, ct_unit_hours,\n"
      "             # fleet_heat_rate_mmbtu_per_mwh, fleet_heat_rate_pct: estimate, corrected, lo, hi\n```\n")
    w("- Input: an hourly profile of load changes (MW, + or -) and its hours (UTC or tz-aware); "
      "temperatures as (hours,) fleet or (hours, units) plant values, defaulting to the actual 2024 plant "
      "temperatures; the baseline non-CHP gas output defaults to 2024 actual. `gas_share` is the share of "
      "the load change met by the non-CHP gas fleet (default 1, the gas-marginal assumption).")
    w("- Method: dispatch at baseline and at baseline + gas_share x change; deltas of fuel, CO2, CT "
      "unit-hours and the fleet heat rate (non-CHP fuel / gross output over the profile's hours). Band: "
      "for each bootstrap draw of the CV error band, each hour's change becomes dE_t x CEMS stratum "
      "marginal + beta x the model's hour-level deviation + beta2 x its step-size part (section 1; two "
      "extra dispatch runs give each hour's +/- 500 MW marginal); 'corrected' is the median draw, "
      "lo/hi the 2.5th/97.5th percentiles. `compare(a, b)` gives the band "
      "of a difference between two profiles from the same draws.")
    w(f"- Speed: one call for a 24-hour profile takes about {r['speed_ms']:.0f} ms after `S4Model.load()` "
      f"({r['load_s']:.0f} s). Worked example: `scripts/s4_worked_example.py`.\n")

    # ---- 6. Limitations
    w("## 6. Assumptions and limitations\n")
    for item in r["limitations"]:
        w(f"- {item}")
    w("\n## 7. Reproduce\n")
    w("`python scripts/run_s4.py` runs the whole S4 pipeline in order (PUDL extract, plant weather, "
      "dispatch validation, heat-rate and derating fits, this integration, the worked example and the "
      f"summary). This script alone: `python scripts/integrate_s4.py` ({r['runtime']:.0f} s on 4 CPUs).")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def compute(year: int, workers: int) -> dict:
    """Expensive part: inputs, window fits and all model runs."""
    t0 = time.time()
    inp = load_inputs(year)
    print(f"inputs {time.time() - t0:.0f} s: {len(inp['fd'].unit_ids)} non-CHP units")
    raw, tables = fit_all_windows(inp, workers)
    check = table_check(raw)
    print(f"fits {time.time() - t0:.0f} s; check vs curves step {check}")
    runs = all_runs(inp["fd"], tables)
    print(f"runs {time.time() - t0:.0f} s")
    return {"fd": inp["fd"], "tables": tables, "runs": runs, "check": check, "t0": t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--draws", type=int, default=2000, help="block-bootstrap draws")
    ap.add_argument("--workers", type=int, default=4, help="processes for the window fits")
    args = ap.parse_args()
    analyse(compute(args.year, args.workers), args.year, args.draws)


def analyse(st: dict, year: int, nd: int) -> None:
    """Tables, error band, model spec, report and figure from :func:`compute` output."""
    t0, fd, tables, runs, check = st["t0"], st["fd"], st["tables"], st["runs"], st["check"]
    full = tables["full"]
    c = Ctx(fd, full["p15"].to_numpy(float))

    # Main configuration from the fit window only (Jan-Aug, in sample).
    selection = {}
    for cfg in CONFIG_NAMES:
        t = grouped(pairs_for(c, runs, "Jan-Aug in sample", [(cfg, "temperature")]),
                    c.qbin, c.q_labels, 200)
        selection[cfg] = wmae(t[f"fuel|{cfg}|temperature_rel"], t["pairs"])
    cfg = min(selection, key=selection.get)
    other = [k for k in CONFIG_NAMES if k != cfg][0]
    print("selection", selection, "->", cfg)

    # Validation tables.
    levels = level_table(c, runs)
    monthly = monthly_table(c, runs, cfg)
    quint, quint_full = marginal_compare(c, runs, cfg, c.qbin, c.q_labels, nd)
    quint_mae = {win: (wmae(t[f"fuel|{cfg}|static_rel"], t["pairs"]),
                       wmae(t[f"fuel|{cfg}|temperature_rel"], t["pairs"])) for win, t in quint_full.items()}
    seas_idx = np.array([SEASONS.index(s) for s in fd.season]) * 3 + c.tbin
    season_terc, _ = marginal_compare(c, runs, cfg, seas_idx,
                                      [f"{s} T{k + 1}" for s in SEASONS for k in range(3)], nd)
    class_tab, _ = marginal_compare(c, runs, cfg, c.cls, CLASS_LABELS, nd)
    decomp = {}
    for mode, lab in [("static", "temperature-free"), ("heat", "heat-rate terms only"),
                      ("derate", "derating only"), ("temperature", "both")]:
        t = grouped(pairs_for(c, runs, HOLDOUT, [(cfg, mode)]), c.qbin, c.q_labels, 200)
        decomp[lab] = wmae(t[f"fuel|{cfg}|{mode}_rel"], t["pairs"])
    print(f"validation {time.time() - t0:.0f} s")

    # Error band (CV and Sep-Dec), both modes; saved for the impact function.
    PROCESSED.mkdir(parents=True, exist_ok=True)
    bands, band_files = {}, {"temperature": {}, "static": {}}
    for mode in ("temperature", "static"):
        for win, short in [(CV, "cv"), (HOLDOUT, "holdout")]:
            b = build_band(c, runs, cfg, mode, win, nd)
            name = f"s4_error_band_{year}_{mode}_{short}.parquet"
            b.save(PROCESSED / name)
            bands[(mode, short)] = b
            band_files[mode][short] = name
    band = bands[("temperature", "cv")]
    strat_cv = stratum_table(band, c, runs, cfg, "temperature", CV, nd)
    strat_cv_static = stratum_table(band, c, runs, cfg, "static", CV, nd)
    beta_tab = beta_table(bands)
    beta2_tab = beta2_table(bands)
    own_rows = strat_cv["impact function uses"] == "own"
    band_wmae = wmae(strat_cv.loc[own_rows, "_rel"], strat_cv.loc[own_rows, "pairs"])
    season_cv = season_class_band(c, runs, cfg, CV, nd)
    season_ho = season_class_band(c, runs, cfg, HOLDOUT, nd)
    shifts = shift_rows(c, runs, cfg, "temperature", CV, nd, band)
    k = f"fuel|{cfg}|temperature"
    t_w = grouped(pairs_for(c, runs, CV, [(cfg, "temperature")]), flag(c.allh), ["all"], nd).iloc[0]
    t_d = grouped(pairs_for(c, runs, CV, [(cfg, "temperature")], block="day"), flag(c.allh), ["all"],
                  nd).iloc[0]
    day_week_all = (t_d[f"{k}_rel_hi"] - t_d[f"{k}_rel_lo"]) / (t_w[f"{k}_rel_hi"] - t_w[f"{k}_rel_lo"])
    day_week_median = float(np.nanmedian(strat_cv["day/week band width"]))
    ko = f"fuel|{other}|temperature"
    to = grouped(pairs_for(c, runs, CV, [(other, "temperature")]), flag(c.allh), ["all"], nd).iloc[0]
    other_band = ci(to[f"{ko}_rel"], to[f"{ko}_rel_lo"], to[f"{ko}_rel_hi"])
    print(f"band {time.time() - t0:.0f} s")

    # Temperature part in the hottest hours (full-year fit).
    hot_values = {}
    for k_cfg in (cfg, other):
        v = hot_hour_value(fd, full, k_cfg, c.hottest)
        v.update(n_hours=int(c.hottest.sum()), mean_temp=float(c.ft[c.hottest].mean()))
        hot_values[k_cfg] = v
    hv = hot_values[cfg]

    # Verdicts computed from the tables.
    sci = season_cv.set_index("hours")
    hot_row = sci.loc[CLASS_LABELS[3]]
    better = quint_mae[CV][1] < quint_mae[CV][0]
    hot_better = abs(hot_row["_rel"]) < abs(hot_row["_srel"])
    verdict = ("Over the load quintiles the temperature-aware model's marginal error is "
               + ("lower" if better else "higher") + " than the temperature-free model's in the CV, and "
               + ("lower" if hot_better else "higher") + " in hours at >= 30 C.")
    hh = sci.loc[f"hottest {HOT_SHARE:.0%} of hours"]
    resolved = abs(hv["pct"]) > max(abs(hh["_lo"]), abs(hh["_hi"]))
    nearer_15 = abs(hh["_cems"] - hv["fd15"]) < abs(hh["_cems"] - (hv["fd15"] + hv["temperature"]["fuel"]))
    hot_verdict = (
        f"Out of sample in the same hours the temperature-aware model overstates the CEMS slope by "
        f"{hh['temperature model error %']}% and the temperature-free model by "
        f"{hh['temperature-free model error %']}%; the temperature part ({hv['pct']:+.1f}%) is "
        + ("larger" if resolved else "not larger") + " than the model's error band there. "
        + (f"The CEMS slope ({hh['_cems']:.2f} MMBtu/MWh) lies nearer the model's 15 C value "
           f"({hv['fd15']:.2f}) than its value at the actual temperatures, so " if nearer_15 else
           f"The CEMS slope ({hh['_cems']:.2f} MMBtu/MWh) lies nearer the model's value at the actual "
           f"temperatures than its 15 C value ({hv['fd15']:.2f}), but ")
        + "the hour-to-hour data do not confirm a temperature increase of the marginal fuel per MWh. The "
        "physical results of the heat-rate step (capacity derating, heat rate at equal output) are not in "
        "question; what is unconfirmed is how they change the fleet's marginal response.")

    # Model spec for ttrl.supply.impact.
    curves_name = f"s4_curves_{year}.parquet"
    full.to_parquet(PROCESSED / curves_name, index=False)
    headroom = {mode: list(runs[("full", cfg, mode)]["headroom"]) for mode in ("temperature", "static")}
    spec = {"year": year, "config": cfg, "curves": curves_name, "headroom": headroom,
            "bands": band_files, "delta_mw": DELTA_MW, "hot_edge": EB.HOT_EDGE,
            "load_edges": list(map(float, c.t_edges)), "block": BLOCK,
            "created": date.today().isoformat(), "created_by": "scripts/integrate_s4.py"}
    (PROCESSED / f"s4_model_{year}.json").write_text(json.dumps(spec, indent=2))

    # Speed of the impact function.
    from ttrl.supply.impact import S4Model
    t1 = time.time()
    model = S4Model.load(year)
    load_s = time.time() - t1
    prof = pd.Series(100.0, index=fd.hours[4000:4024])
    t1 = time.time()
    for _ in range(5):
        model.impact(prof.to_numpy(), prof.index)
    speed_ms = (time.time() - t1) / 5 * 1e3

    # Hourly validation file.
    hourly = pd.DataFrame({"hour_utc": fd.hours, "month_lst": fd.month, "fleet_temp_c": c.ft,
                           "target_mwh": c.target, "heat_actual": c.heat, "co2_actual": c.co2,
                           "n_ct_actual": c.n_ct})
    cols = {f"{s}_{win}_{k_cfg}_{mode}": rr[s] for (win, k_cfg, mode), rr in runs.items()
            for s in ("heat", "co2", "n_ct", "m_fuel", "m_co2", "m_ct")}
    hourly = pd.concat([hourly, pd.DataFrame(cols)], axis=1)
    hourly.to_parquet(PROCESSED / f"s4_validation_hourly_{year}.parquet", index=False)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    figure(quint_full[CV], strat_cv, strat_cv_static, cfg, OUTPUTS / FIG)
    stn = FM.station_distance(np.unique(fd.plant_id_eia), year)
    limitations = [
        "CHP units are outside the model: their fuel also makes process steam, so they have no electric "
        "heat rate. All results are non-CHP.",
        f"Suspect CC units ({int(full['suspect_cc'].sum())} units, full-year flag) probably report the "
        "combustion turbine's output only; their steam turbine's output is missing from CEMS. In C they "
        "respond with CT-only curves; in C-sx they are held at actual output and never respond.",
        "Gas-marginal assumption: a load change is met by the non-CHP gas fleet (share `gas_share`, default "
        "1). EIA-930 shows that coal and storage also follow load in ERCOT; their fuel is not modelled.",
        "One year (2024), one region. The band's strata were revised once after seeing the first band table, "
        "the beta step was added after the first worked example and the step-size part (beta2, size classes) "
        "before the RL stage after the worked example and the ERCOT environment check (section 1); the "
        "+/-500 MW step and the CV design were fixed beforehand. Unit availability comes from actual operation in each month "
        "(exogenous) in every window.",
        "Temperatures are dry-bulb readings at the nearest good ISD station (median "
        f"{stn['median']:.0f} km, at most {stn['max']:.0f} km from the {stn['plants']} plants), not "
        "compressor-inlet conditions; humidity and inlet cooling are not modelled.",
        "Dispatch is hourly and memoryless: no start-up fuel, minimum up/down times or ramp limits, one gas "
        "price. The fuel level is biased low (section 3.1) and CT hours are badly placed by season.",
        "The CEMS benchmark is the hour-to-hour slope of observed operation. A tariff moves load in a "
        "planned, repeated pattern that the fleet could meet with different commitment; the band does not "
        "cover that.",
        "Derating below 15 C is not modelled (capacity held at P15); heat terms are not extrapolated beyond "
        "each unit's fitted temperature range.",
    ]
    res = {"year": year, "cfg": cfg, "other_cfg": other, "other_cfg_band": other_band,
           "selection": selection, "levels": levels, "monthly": monthly, "quint": quint,
           "quint_mae": quint_mae, "season_terc": season_terc, "class_tab": class_tab, "decomp": decomp,
           "band_cv": band, "strat_cv": strat_cv, "band_wmae": band_wmae, "season_cv": season_cv,
           "season_ho": season_ho, "shifts": shifts, "day_week_all": day_week_all,
           "day_week_median": day_week_median, "n_draws": nd, "hot_value": hv,
           "hot_values": hot_values, "verdict": verdict, "hot_verdict": hot_verdict,
           "better": better, "resolved": resolved, "headroom": headroom, "beta_tab": beta_tab, "beta2_tab": beta2_tab,
           "pairs_without_step_run": band.meta["pairs_without_step_run"], "table_check": check, "t_edges": c.t_edges,
           "n_fallback": int((full["source"] == "static fallback").sum()), "n_units": len(full),
           "speed_ms": speed_ms, "load_s": load_s, "limitations": limitations,
           "runtime": time.time() - t0}
    write_report(res, OUTPUTS / REPORT)
    DOCS_REPORTS.mkdir(parents=True, exist_ok=True)
    for name in (REPORT, FIG):
        shutil.copyfile(OUTPUTS / name, DOCS_REPORTS / name)

    # Numbers for scripts/s4_summary.py.
    lv = levels.set_index(["config", "model"])
    pub = lambda df: df[[c_ for c_ in df.columns if not c_.startswith("_")]]  # noqa: E731
    summary = {
        "year": year, "config": cfg, "other_config": other, "selection": selection,
        "n_units": len(full), "n_suspect": int(full["suspect_cc"].sum()), "draws": nd, "block": BLOCK,
        "levels": {f"{a}|{b}": v for (a, b), v in lv.to_dict(orient="index").items()},
        "quint_mae": quint_mae, "decomp": decomp,
        "band_season_cv": pub(season_cv).to_dict(orient="records"),
        "band_season_ho": pub(season_ho).to_dict(orient="records"),
        "band_strata": pub(strat_cv).to_dict(orient="records"),
        "band_range": [float(strat_cv.loc[own_rows, "_rel"].min()),
                       float(strat_cv.loc[own_rows, "_rel"].max())],
        "beta": beta_tab.to_dict(orient="records"), "beta2": beta2_tab.to_dict(orient="records"), "n_own_strata": int(own_rows.sum()), "better": bool(better), "resolved": bool(resolved),
        "band_wmae": band_wmae, "day_week_all": day_week_all, "day_week_median": day_week_median,
        "other_cfg_band": other_band, "hot_value": hot_values,
        "hottest_cv": {k_: (float(v) if isinstance(v, (float, np.floating)) else v)
                       for k_, v in hh.items()},
        "verdict": verdict, "hot_verdict": hot_verdict,
        "shifts": shifts.to_dict(orient="records"), "runtime": time.time() - t0,
    }
    (OUTPUTS / "s4_integration.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {OUTPUTS / REPORT}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
