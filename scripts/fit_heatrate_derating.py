"""Temperature-dependent heat-rate curves and capacity derating, ERCOT gas fleet.

Stage S4, the paper's standalone physical finding: how much ambient
temperature reduces the available capacity and the efficiency of ERCOT gas
units (CT, CC, ST) in 2024, and what that means for the fuel value of moving
load out of hot hours. Models are in :mod:`ttrl.supply.heatrate`.

Steps (numbers in the report come from this script only):

0. The dispatch model's own error band, recomputed from the hourly output of
   ``scripts/validate_dispatch.py`` (``data/processed/dispatch_hourly_<year>.parquet``).
1. Per-unit heat input H(P, T) on steady full hours, residual checks.
2. Per-unit derating kappa from the output envelope, with selection-bias
   checks (raw vs monotone envelope, high-dispatch hours) and EIA-860
   summer/winter capacity as an external check.
3. Technology results (median, IQR, capacity-weighted mean, bootstrap CIs).
4. Fleet-level capacity loss in the hottest hours and the extra fuel per
   MWh at equal output (average at actual unit outputs; marginal through the
   dispatch model), in MMBtu and kg CO2.
5. Hold-out: fit on Jan-Aug, predict Sep-Dec heat input with and without
   the temperature terms; same check for the dispatch model's marginal heat
   rate.

Writes ``outputs/heatrate_derating.md`` (copied to ``docs/reports/``), two
figures, ``data/processed/unit_curves_<year>.parquet`` (full-year fit, one
row per unit) and ``data/processed/unit_curves_<year>_janaug.parquet``
(Jan-Aug fit used in the hold-out).

Usage:
    python scripts/fit_heatrate_derating.py [--year 2024] [--boot 2000] [--draws 200]
"""
from __future__ import annotations

import argparse
import shutil
import time
import warnings
from dataclasses import dataclass, replace
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from ttrl.data import pudl  # noqa: E402
from ttrl.paths import INTERIM, OUTPUTS, PROCESSED, ROOT  # noqa: E402
from ttrl.supply import dispatch as D  # noqa: E402
from ttrl.supply import heatrate as H  # noqa: E402
from ttrl.supply import units as U  # noqa: E402
from ttrl.supply.fleet import build_fleet_hours  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_MONTHS = range(1, 9)
TEST_MONTHS = range(9, 13)
HIGH_DISPATCH_Q = 0.80      # high-dispatch hours: fleet output >= this quantile
HOT_SHARE = 0.01            # hottest 1% of CEMS hours
PEAK_SHARE = 0.05           # seasonal peak hours for the EIA-860 comparison
DELTA_MW = 500.0
HAC_LAGS = 24
DOCS_REPORTS = ROOT / "docs" / "reports"
REPORT = "heatrate_derating.md"
FIG_CAP = "heatrate_derating_capacity.png"
FIG_HR = "heatrate_derating_heatrate.png"
GROUP_LABEL = {"CT": "CT (simple cycle)", "CC": "CC", "CC*": "CC* (suspect, CT output only)",
               "ST": "ST (steam)"}
# Categorical slots 1-4 of the reference palette (validated, light mode).
COLORS = {"CT": "#2a78d6", "CC": "#eb6834", "CC*": "#1baf7a", "ST": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}",
             formats: dict[str, str] | None = None) -> str:
    """GitHub markdown table with per-column float formats."""
    formats = formats or {}

    def fmt(col, v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else formats.get(col, floatfmt).format(v)
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, r)) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def ci_text(est: float, lo: float, hi: float, f: str = "{:.2f}") -> str:
    """'est [lo, hi]'."""
    return f"{f.format(est)} [{f.format(lo)}, {f.format(hi)}]"


def hac_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float, int]:
    """OLS slope of y on x (with intercept) and its HAC standard error."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    res = sm.OLS(y[ok], sm.add_constant(x[ok])).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(res.params[1]), float(res.bse[1]), int(ok.sum())


def weighted_mae(df: pd.DataFrame, col: str, w: str = "n") -> float:
    """Hours-weighted mean absolute value of a column."""
    return float((df[col].abs() * df[w]).sum() / df[w].sum())


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass
class Inputs:
    """Data shared by all steps."""
    year: int
    cems: pd.DataFrame
    units: pd.DataFrame
    gens: pd.DataFrame
    temp: pd.DataFrame
    uh: pd.DataFrame          # operating unit-hours with temp_c, steady, high_dispatch
    fleet_out: pd.Series      # hourly CEMS gross output of all linked gas units (MWh)


def load_inputs(year: int) -> Inputs:
    """CEMS extract, unit tables, plant temperature and the unit-hour table."""
    paths = pudl.build_ercot_gas_cems(year)
    cems = pd.read_parquet(paths["cems"])
    units = pd.read_parquet(paths["units"])
    gens = pd.read_parquet(paths["generators"])
    tpath = INTERIM / f"plant_temperature_{year}.parquet"
    if not tpath.exists():
        raise FileNotFoundError(f"{tpath} missing: run scripts/build_plant_weather.py first")
    temp = pd.read_parquet(tpath)
    uh = U.prepare_unit_hours(cems)
    uh = H.add_temperature(uh, units, temp)
    uh["steady"] = H.steady_rows(uh)
    fleet_out = uh.groupby("operating_datetime_utc")["energy_mwh"].sum()
    hi = fleet_out >= fleet_out.quantile(HIGH_DISPATCH_Q)
    uh["high_dispatch"] = uh["operating_datetime_utc"].map(hi).fillna(False).astype(bool)
    return Inputs(year, cems, units, gens, temp, uh, fleet_out)


def co2_factors(uh: pd.DataFrame) -> pd.Series:
    """CO2 per heat input by unit (short tons per MMBtu) over hours with both."""
    both = uh[(uh["heat"] > 0) & uh["co2"].notna()]
    f = both.groupby("unit_id")[["co2", "heat"]].sum()
    return f["co2"] / f["heat"]


def fit_tables(inp: Inputs, cfg: H.TempConfig) -> dict[str, pd.DataFrame]:
    """Full-year and Jan-Aug temperature tables (groups fixed by the full year)."""
    full = H.fit_temperature_curves(inp.uh, inp.units, inp.gens, cfg)
    groups = full.set_index("unit_id")["group"]
    tr = inp.uh[inp.uh["month"].isin(TRAIN_MONTHS)].copy()
    tr["steady"] = H.steady_rows(tr)     # Pref from the fit window only (no Sep-Dec data)
    train = H.fit_temperature_curves(tr, inp.units, inp.gens, cfg, group_of=groups)
    co2 = co2_factors(inp.uh)
    out = {}
    for name, t in {"full": full, "train": train}.items():
        t = t.copy()
        t["co2_per_mmbtu"] = t["unit_id"].map(co2)
        t["co2_per_mmbtu"] = t["co2_per_mmbtu"].fillna(
            t.groupby("group")["co2_per_mmbtu"].transform("median"))
        out[name] = t
    return out


def own_hr(t: pd.DataFrame) -> pd.Series:
    """Units with their own heat-input temperature fit (non-CHP)."""
    return t["hr_method"].isin(["quadratic", "linear"]) & ~t["const_heat"] & (t["group"] != "CHP")


def own_kappa(t: pd.DataFrame) -> pd.Series:
    """Units with their own derating fit (non-CHP)."""
    return (t["kappa_method"] == "own") & (t["group"] != "CHP")


# --------------------------------------------------------------------------
# 0. Dispatch error band (recomputed from the dispatch step's hourly output)
# --------------------------------------------------------------------------
def dispatch_band(year: int) -> dict:
    """Out-of-sample marginal-heat-rate error of the dispatch model (C-sx).

    Same estimator as scripts/validate_dispatch.py: CEMS hour-to-hour slope
    of non-CHP heat input on non-CHP gross output by load quintile (Sep-Dec,
    Jan-Aug fit), vs the mean of the model's +/-500 MW derivative.
    """
    path = PROCESSED / f"dispatch_hourly_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/validate_dispatch.py first")
    d = pd.read_parquet(path)
    target, heat = d["target_mwh"].to_numpy(), d["heat_actual"].to_numpy()
    edges = np.quantile(target, [0.2, 0.4, 0.6, 0.8])
    qbin = np.searchsorted(edges, target, side="right")
    lo = np.r_[target.min(), edges]
    hi = np.r_[edges, target.max()]
    labels = [f"Q{i + 1} ({a / 1e3:.1f}-{b / 1e3:.1f} GW)" for i, (a, b) in enumerate(zip(lo, hi))]
    test = d["month_lst"].isin(TEST_MONTHS).to_numpy()
    both = test[1:] & test[:-1]
    dx, dy = np.diff(target), np.diff(heat)
    fd = d["mhr_C-sx_train"].to_numpy()[1:]
    rows = []
    for q, lab in enumerate(labels):
        sel = both & (qbin[1:] == q)
        b, se, n = hac_slope(dy[sel], dx[sel])
        f = float(np.nanmean(fd[sel]))
        rows.append({"hours": lab, "n": n, "actual": b, "actual 95% CI":
                     f"{b - 1.96 * se:.2f} to {b + 1.96 * se:.2f}",
                     "model FD": f, "FD error %": 100 * (f / b - 1)})
    tab = pd.DataFrame(rows)
    return {"table": tab, "mae": weighted_mae(tab, "FD error %"),
            "lo": tab["FD error %"].min(), "hi": tab["FD error %"].max(),
            "edges": edges, "labels": labels}


# --------------------------------------------------------------------------
# 1. Heat-input model: residual checks
# --------------------------------------------------------------------------
def residual_checks(uh: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Mean residual (% of fitted heat) by group and temperature band.

    Steady fit rows of own-fit units; columns for the static curve (a0, b0,
    c0) and the temperature curve, both fitted on the same rows.
    """
    tt = t[own_hr(t)].set_index("unit_id")
    rows = uh["steady"] & U.fit_rows(uh) & uh["unit_id"].isin(tt.index) & ~H.in_limited_month(uh, t)
    d = uh[rows].copy()
    par = tt.loc[d["unit_id"]]
    p, temp = d["gross_load_mw"].to_numpy(), d["temp_c"].to_numpy()
    d["fit0"] = par["a0"].to_numpy() + par["b0"].to_numpy() * p + par["c0"].to_numpy() * p * p
    d["fitT"] = H.heat_input(par, p, temp)
    d["group"] = par["group"].to_numpy()
    bands = [-np.inf, 5, 15, 25, 30, 35, np.inf]
    names = ["< 5", "5-15", "15-25", "25-30", "30-35", ">= 35"]
    d["band"] = pd.cut(d["temp_c"], bands, labels=names, right=False)
    rows = []
    for (g, b), x in d.groupby(["group", "band"], observed=True):
        rows.append({"group": g, "temperature C": b, "unit-hours": len(x),
                     "static %": 100 * (x["heat"].sum() / x["fit0"].sum() - 1),
                     "temperature model %": 100 * (x["heat"].sum() / x["fitT"].sum() - 1)})
    out = pd.DataFrame(rows)
    out["group"] = pd.Categorical(out["group"], H.GROUPS, ordered=True)
    out["temperature C"] = pd.Categorical(out["temperature C"], names, ordered=True)
    out = out.sort_values(["group", "temperature C"]).reset_index(drop=True)
    return out.astype({"group": str, "temperature C": str})


# --------------------------------------------------------------------------
# 2-3. Technology summaries and EIA-860 comparison
# --------------------------------------------------------------------------
def tech_table(t: pd.DataFrame, col: str, mask: pd.Series, n_boot: int,
               fmt: str = "{:.3f}") -> pd.DataFrame:
    """Per-group median [IQR], bootstrap CI and capacity-weighted mean of ``col``."""
    rows = []
    for g in H.GROUPS:
        x = t[mask & (t["group"] == g)]
        s = H.group_stats(x[col], x["nameplate_mw"], n_boot=n_boot)
        rows.append({"group": GROUP_LABEL[g], "units": s["n"],
                     "MW (nameplate)": float(x["nameplate_mw"].sum()),
                     "median": s["median"], "IQR": f"{fmt.format(s['p25'])} to {fmt.format(s['p75'])}",
                     "median 95% CI": f"{fmt.format(s['median_lo'])} to {fmt.format(s['median_hi'])}",
                     "cap-weighted mean": s["wmean"],
                     "cap-weighted 95% CI": f"{fmt.format(s['wmean_lo'])} to {fmt.format(s['wmean_hi'])}"})
    return pd.DataFrame(rows)


def seasonal_peak_temps(inp: Inputs, t: pd.DataFrame) -> pd.DataFrame:
    """Plant temperature in the seasonal peak hours of fleet output, per unit.

    Peak hours: the ``PEAK_SHARE`` of June-September (summer) and
    December-February (winter) hours with the highest CEMS fleet gross output
    (local standard time months). Returns unit_id, t_summer, t_winter.
    """
    fo = inp.fleet_out
    month = (fo.index - U.LST_OFFSET).month
    sets = {}
    for name, months in {"summer": [6, 7, 8, 9], "winter": [12, 1, 2]}.items():
        x = fo[np.isin(month, months)]
        sets[name] = x[x >= x.quantile(1 - PEAK_SHARE)].index
    temp = inp.temp.sort_values(["plant_id_eia", "ts_utc"]).copy()
    nxt = temp.groupby("plant_id_eia")["temp_c"].shift(-1)
    temp["t_hb"] = (temp["temp_c"] + nxt) / 2
    out = t[["unit_id", "plant_id_eia"]].copy()
    for name, idx in sets.items():
        m = temp[temp["ts_utc"].isin(idx)].groupby("plant_id_eia")["t_hb"].mean()
        out[f"t_{name}"] = out["plant_id_eia"].astype("int64").map(m)
    return out.drop(columns="plant_id_eia")


def eia_comparison(inp: Inputs, t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """CEMS derating vs EIA-860 summer/winter net capacity, per unit and group.

    Per unit: EIA ratio S/W; CEMS seasonal ratio (99th percentile of clean
    full-hour output in Jun-Sep over Dec-Feb, >= 100 hours each); model
    ratio Pavail(T_summer)/P15 (the model is flat below 15 C, so this is the
    model's summer/winter ratio whenever the winter rating temperature is
    <= 15 C); EIA-implied kappa = 100 (1 - S/W) / (T_summer - 15) under the
    same flat-below-15 C assumption, and the lower value
    100 (1 - S/W) / (T_summer - T_winter) if capacity kept rising below 15 C.
    """
    pk = seasonal_peak_temps(inp, t)
    c = inp.uh[U.fit_rows(inp.uh)]
    m = c["month"]
    s = c[m.isin([6, 7, 8, 9])].groupby("unit_id")["gross_load_mw"].agg(["size", lambda x: x.quantile(0.99)])
    w = c[m.isin([12, 1, 2])].groupby("unit_id")["gross_load_mw"].agg(["size", lambda x: x.quantile(0.99)])
    s.columns, w.columns = ["n_s", "p_s"], ["n_w", "p_w"]
    u = t.merge(pk, on="unit_id").set_index("unit_id").join(s).join(w)
    u = u[(u["group"] != "CHP") & u["summer_mw"].gt(0) & u["winter_mw"].gt(0)]
    u["eia_ratio"] = u["summer_mw"] / u["winter_mw"]
    ok_s = (u["n_s"] >= 100) & (u["n_w"] >= 100)
    u["cems_season_ratio"] = (u["p_s"] / u["p_w"]).where(ok_s)
    u["model_ratio"] = H.derate_factor(u["kappa"], u["t_summer"])
    u["kappa_eia_flat"] = 100 * (1 - u["eia_ratio"]) / (u["t_summer"] - H.T_REF)
    u["kappa_eia_lin"] = 100 * (1 - u["eia_ratio"]) / (u["t_summer"] - u["t_winter"])
    rows = []
    for g in H.GROUPS:
        x = u[u["group"] == g]
        own = x[x["kappa_method"] == "own"]
        cs = x["cems_season_ratio"].dropna()
        rows.append({
            "group": GROUP_LABEL[g], "units": len(x),
            "EIA S = W %": 100 * float((x["summer_mw"] == x["winter_mw"]).mean()),
            "EIA S/W median": float(x["eia_ratio"].median()),
            "CEMS Jun-Sep/Dec-Feb p99 median": float(cs.median()) if len(cs) else np.nan,
            "units with CEMS ratio": len(cs),
            "model P(T_summer)/P15 median": float(own["model_ratio"].median()),
            "T_summer C median": float(x["t_summer"].median()),
            "T_winter C median": float(x["t_winter"].median()),
            "kappa EIA (flat < 15 C)": float(x["kappa_eia_flat"].median()),
            "kappa EIA (linear to T_winter)": float(x["kappa_eia_lin"].median()),
            "kappa CEMS (own fits)": float(own["kappa"].median()),
        })
    both = u[(u["kappa_method"] == "own")].dropna(subset=["kappa_eia_flat"])
    stats = {
        "corr_kappa": float(np.corrcoef(both["kappa"], both["kappa_eia_flat"])[0, 1]),
        "n_corr": len(both),
        "corr_season": float(u[["cems_season_ratio", "eia_ratio"]].dropna().corr().iloc[0, 1]),
        "n_season": int(u["cems_season_ratio"].notna().sum()),
        "fleet_w_minus_s": float((u["winter_mw"] - u["summer_mw"]).sum()),
        "fleet_w": float(u["winter_mw"].sum()),
    }
    return pd.DataFrame(rows), u, stats


def kappa_variants(t: pd.DataFrame) -> pd.DataFrame:
    """Medians of the derating variants on the same units (own fits)."""
    rows = []
    for g in H.GROUPS:
        x = t[own_kappa(t) & (t["group"] == g)]
        pair = x.dropna(subset=["b21_kappa", "hi_kappa"])
        diff = (pair["hi_kappa"] - pair["b21_kappa"])
        rows.append({"group": GROUP_LABEL[g], "units": len(x),
                     "raw envelope >= 15 C": float(x["raw_kappa"].median()),
                     "monotone >= 15 C (primary)": float(x["kappa"].median()),
                     "monotone >= 21 C": float(pair["b21_kappa"].median()),
                     "high-dispatch hours >= 15 C": float(pair["hi_kappa"].median()),
                     "paired units": len(pair),
                     "high-dispatch minus >= 21 C, median": float(diff.median()),
                     "share of units higher in high-dispatch %": 100 * float((diff > 0).mean())
                     if len(pair) else np.nan,
                     "raw < 0 %": 100 * float((x["raw_kappa"] < 0).mean()),
                     "months dropped": int(x["months_dropped"].sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Fleet level
# --------------------------------------------------------------------------
@dataclass
class FleetSetup:
    """Hourly arrays for the non-CHP units of a curve table."""
    hours: pd.DatetimeIndex
    table: pd.DataFrame       # rows aligned with the array columns
    temp: np.ndarray          # (T, N) hour-beginning plant temperature
    load: np.ndarray          # (T, N) actual gross MW
    energy: np.ndarray        # (T, N) actual gross MWh
    heat: np.ndarray          # (T, N) actual heat MMBtu
    on: np.ndarray            # (T, N) produced
    avail: np.ndarray         # (T, N) monthly availability
    clean: np.ndarray         # (T, N) clean full hour
    limited: np.ndarray       # (T, N) hour in one of the unit's capability-limited months
    month: np.ndarray         # (T,) LST month
    fleet_temp: np.ndarray    # (T,) P15-weighted mean temperature


def complete_table(t: pd.DataFrame, inp: Inputs, ids: list[str], months) -> pd.DataFrame:
    """Curve table for ``ids``; units missing from ``t`` get the dispatch
    step's static (possibly pooled) curve with group temperature terms."""
    have = t.set_index("unit_id")
    miss = [u for u in ids if u not in have.index or pd.isna(have.loc[u, "a"])]
    if miss:
        uh = inp.uh[inp.uh["month"].isin(months)]
        static = U.fit_unit_curves(uh, inp.units, inp.gens, unit_ids=ids).set_index("unit_id")
        own = t[own_hr(t)].assign(d_rel=lambda x: x["d"] / x["pmax"])
        med = own.groupby("group")[["d_rel", "e"]].median()
        kmed = t[own_kappa(t)].groupby("group")["kappa"].median()
        add = []
        for u in miss:
            r = static.loc[u]
            g = ("CC*" if r["suspect_cc"] else r["tech"]) if not r["chp"] else "CHP"
            add.append({"unit_id": u, "group": g, "tech": r["tech"], "chp": r["chp"],
                        "suspect_cc": r["suspect_cc"], "a": r["a"], "b": r["b"], "c": r["c"],
                        "pmin": r["pmin"], "pmax": r["pmax"], "p15": r["pmax"],
                        "d": med.at[g, "d_rel"] * r["pmax"] if g in med.index else 0.0,
                        "e": med.at[g, "e"] if g in med.index else 0.0,
                        "kappa": kmed.get(g, 0.0), "hr_method": "group (static curve)",
                        "kappa_method": "group", "co2_per_mmbtu": r["co2_per_mmbtu"],
                        "a0": r["a"], "b0": r["b"], "c0": r["c"],
                        "nameplate_mw": r["nameplate_mw"]})
        have = pd.concat([have.drop(index=[u for u in miss if u in have.index]),
                          pd.DataFrame(add).set_index("unit_id")])
    return have.loc[ids].reset_index()


def build_fleet(inp: Inputs, t: pd.DataFrame, months=range(1, 13)) -> FleetSetup:
    """Arrays for every non-CHP CEMS unit, in a fixed column order."""
    chp = set(inp.units.loc[inp.units["chp"].fillna(False).astype(bool)].pipe(
        lambda x: U.make_unit_id(x["plant_id_epa"], x["emissions_unit_id_epa"])))
    ids = sorted(u for u in inp.uh["unit_id"].unique() if u not in chp)
    table = complete_table(t, inp, ids, months)
    fh = build_fleet_hours(inp.cems, ids, inp.year)
    plant = inp.units.assign(unit_id=U.make_unit_id(inp.units["plant_id_epa"],
                                                    inp.units["emissions_unit_id_epa"]))
    plant = plant.set_index("unit_id").loc[ids, "plant_id_eia"].astype("int64").to_numpy()
    tmp = inp.temp.sort_values(["plant_id_eia", "ts_utc"]).copy()
    tmp["t_hb"] = (tmp["temp_c"] + tmp.groupby("plant_id_eia")["temp_c"].shift(-1)) / 2
    piv = tmp.pivot(index="ts_utc", columns="plant_id_eia", values="t_hb").reindex(fh.hours)
    temp = piv[plant].to_numpy(float)
    clean_rows = inp.uh[U.fit_rows(inp.uh) & inp.uh["unit_id"].isin(ids)]
    clean = np.zeros_like(fh.present)
    ti = ((clean_rows["operating_datetime_utc"] - fh.hours[0]) / pd.Timedelta(hours=1)).astype(int)
    ui = pd.Categorical(clean_rows["unit_id"], categories=ids).codes
    ok = (ui >= 0) & (ti >= 0) & (ti < len(fh.hours))
    clean[ti[ok], ui[ok]] = True
    w = table["p15"].to_numpy(float)
    fleet_temp = (temp * w).sum(axis=1) / w.sum()
    month = fh.labels["month"].to_numpy()
    lim_m = table.get("limited_months", pd.Series("", index=table.index)).fillna("")
    limited = np.column_stack([np.isin(month, [int(m) for m in str(s).split(",") if m])
                               for s in lim_m])
    return FleetSetup(fh.hours, table, temp, fh.load, fh.energy, fh.heat, fh.on,
                      fh.availability("month"), clean, limited, month, fleet_temp)


def capacity_loss(fs: FleetSetup, hot: np.ndarray, n_boot: int) -> dict:
    """Capacity lost vs 15 C in the hot hours, non-CHP units (incl. CC*).

    Lost MW = sum_i P15_i kappa_i/100 max(T_it - 15, 0). Bootstrap: units
    resampled within groups; the ratio lost/P15 is applied to the fleet P15.
    """
    t = fs.table
    p15 = t["p15"].to_numpy(float)
    kap = t["kappa"].to_numpy(float)
    groups = t["group"].to_numpy()
    out = {"p15_total": float(p15.sum())}
    dt = np.maximum(fs.temp[hot] - H.T_REF, 0.0)           # (h, N)
    lost_h = (p15 * kap / 100 * dt).sum(axis=1)             # (h,)
    out["lost_mean"] = float(lost_h.mean())
    out["lost_max_hour"] = float(lost_h[np.argmax(fs.fleet_temp[hot])])
    num = (p15 * kap / 100 * dt).mean(axis=0)
    draws = H.stratified_ratio_bootstrap(num, p15, groups, n_boot) * p15.sum()
    out["lost_lo"], out["lost_hi"] = np.quantile(draws, [0.025, 0.975])
    by = {}
    for g in H.GROUPS:
        k = groups == g
        by[g] = {"P15 MW": float(p15[k].sum()), "lost MW": float(num[k].sum()),
                 "lost %": 100 * float(num[k].sum() / p15[k].sum()),
                 "mean T C": float((fs.temp[hot][:, k] * p15[k]).sum(axis=1).mean() / p15[k].sum())}
    out["by_group"] = by
    # Same with the high-dispatch-hours kappa where available.
    kap_hi = t.get("hi_kappa", pd.Series(np.nan, index=t.index)).to_numpy(float)
    kap_hi = np.where(np.isfinite(kap_hi) & (t["kappa_method"] == "own"), kap_hi, kap)
    out["lost_mean_hi"] = float((p15 * kap_hi / 100 * dt).sum(axis=1).mean())
    return out


def average_penalty(fs: FleetSetup, hot: np.ndarray, n_boot: int) -> dict:
    """Extra heat per MWh at the actual unit outputs of the hot hours vs 15 C.

    For clean full hours of non-CHP units: dH = (d + e P)(T - 15). Hours in a
    unit's capability-limited months are left out (a different operating
    mode that the heat fit excludes; ``excluded_pct`` is their share of the
    hot hours' clean output). Returns MMBtu/MWh and kg CO2/MWh with bootstrap
    CIs over units (within groups).
    """
    t = fs.table
    d, e = t["d"].to_numpy(float), t["e"].to_numpy(float)
    excl = np.where(fs.clean[hot] & fs.limited[hot], fs.load[hot], 0.0).sum()
    all_mwh = np.where(fs.clean[hot], fs.load[hot], 0.0).sum()
    m = fs.clean[hot] & ~fs.limited[hot]
    p = np.where(m, fs.load[hot], 0.0)
    dT = fs.temp[hot] - H.T_REF
    dh = np.where(m, (d + e * p) * dT, 0.0).sum(axis=0)
    mwh = p.sum(axis=0)
    co2 = dh * t["co2_per_mmbtu"].to_numpy(float) * H.KG_PER_SHORT_TON
    groups = t["group"].to_numpy()
    use = mwh > 0
    est = dh[use].sum() / mwh[use].sum()
    dr = H.stratified_ratio_bootstrap(dh[use], mwh[use], groups[use], n_boot)
    est_c = co2[use].sum() / mwh[use].sum()
    drc = H.stratified_ratio_bootstrap(co2[use], mwh[use], groups[use], n_boot, seed=1)
    base_hr = np.where(m, H.heat_input(t, p, H.T_REF), 0.0).sum() / mwh.sum()
    return {"mmbtu": est, "lo": np.quantile(dr, 0.025), "hi": np.quantile(dr, 0.975),
            "kg": est_c, "kg_lo": np.quantile(drc, 0.025), "kg_hi": np.quantile(drc, 0.975),
            "pct": 100 * est / base_hr, "mean_dT": float((dT * p).sum() / p.sum()),
            "mwh": float(mwh.sum()), "hr15": base_hr, "excluded_pct": 100 * excl / all_mwh}


def priority_setup(fs: FleetSetup, model: H.TemperatureCurves, fit_hours: np.ndarray,
                   temp: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
    """C-sx dispatch inputs: dispatched mask, target, headroom fit, availability."""
    disp = ~fs.table["suspect_cc"].astype(bool).to_numpy()
    target = fs.energy[:, disp].sum(axis=1)
    cv = model.arrays(temp)
    pmax = np.broadcast_to(cv.pmax, fs.energy.shape)[:, disp]
    committed_pmax = (fs.on[:, disp] * pmax).sum(axis=1)
    headroom = D.fit_headroom(target[fit_hours], committed_pmax[fit_hours])
    return disp, target, headroom, fs.avail[:, disp]


def fd_heat_co2(target: np.ndarray, cv, avail: np.ndarray, headroom: tuple,
                co2f: np.ndarray, delta: float = DELTA_MW) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference marginal heat rate and CO2 rate (re-commitment)."""
    out = []
    for sign in (+1.0, -1.0):
        dd = target + sign * delta
        res = D.dispatch(dd, cv, avail, headroom_mw=headroom[0] + headroom[1] * dd)
        uh = res.unit_heat(cv)
        out.append((uh.sum(axis=1), (uh * co2f).sum(axis=1), res.status))
    (h1, c1, s1), (h2, c2, s2) = out
    ok = (s1 == D.STATUS_OK) & (s2 == D.STATUS_OK)
    mhr = np.where(ok, (h1 - h2) / (2 * delta), np.nan)
    mco2 = np.where(ok, (c1 - c2) / (2 * delta) * H.KG_PER_SHORT_TON, np.nan)
    return mhr, mco2


def sub_model(table: pd.DataFrame, mask: np.ndarray, **kw) -> H.TemperatureCurves:
    """TemperatureCurves on the rows of ``table`` where ``mask``."""
    return H.TemperatureCurves(table.loc[mask].reset_index(drop=True), **kw)


def table_variant(t: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Curve table with alternative temperature parameters for sensitivities.

    ``within-day``: own-fit units take d, e from day-fixed-effect estimates;
    ``high-dispatch kappa``: own-envelope units take their high-dispatch kappa.
    Other units keep their parameters.
    """
    x = t.copy()
    if kind == "within-day":
        own = x["hr_method"].isin(["quadratic", "linear"]) & x["d_wd"].notna()
        x.loc[own, "d"], x.loc[own, "e"] = x.loc[own, "d_wd"], x.loc[own, "e_wd"]
    elif kind == "high-dispatch kappa":
        own = (x["kappa_method"] == "own") & x["hi_kappa"].notna()
        x.loc[own, "kappa"] = x.loc[own, "hi_kappa"]
    return x


def marginal_value(fs: FleetSetup, hot: np.ndarray, n_draws: int, seed: int = 0) -> dict:
    """Temperature-attributable marginal heat rate in the hot hours.

    Main dispatch configuration (C-sx: suspect CC units held at actual
    output; monthly availability; headroom rule fitted on the year with the
    temperature-aware Pmax). For each hot hour, the +/-500 MW derivative at
    the actual plant temperatures minus the derivative with every plant at
    15 C, same output target, same availability and headroom. Uncertainty:
    ``n_draws`` parametric draws of the temperature parameters (own fits:
    normal with the cluster-robust covariance of d, e and the envelope SE of
    kappa; group-assigned units: the parameters of a random own-fit unit of
    the group, scaled to the unit's size). Sensitivities: heat terms only,
    derating only, within-day heat terms, high-dispatch kappa.
    """
    t = fs.table
    disp = ~t["suspect_cc"].astype(bool).to_numpy()
    model = sub_model(t, disp)
    allh = np.ones(len(fs.hours), bool)
    _, target, headroom, avail = priority_setup(fs, H.TemperatureCurves(t), allh, fs.temp)
    target_h, avail_h = target[hot], avail[hot]
    temp_h = fs.temp[hot][:, disp]
    co2f = model.co2_per_mmbtu
    cv15 = model.arrays(np.full_like(temp_h, H.T_REF))
    m15, c15 = fd_heat_co2(target_h, cv15, avail_h, headroom, co2f)
    variants = {"both effects": (t, {}), "heat-rate terms only": (t, {"use_derate": False}),
                "derating only": (t, {"use_heat": False}),
                "both, within-day heat terms": (table_variant(t, "within-day"), {}),
                "both, high-dispatch kappa": (table_variant(t, "high-dispatch kappa"), {})}
    diffs = {}
    for name, (tab, kw) in variants.items():
        cv = sub_model(tab, disp, **kw).arrays(temp_h)
        m, c = fd_heat_co2(target_h, cv, avail_h, headroom, co2f)
        diffs[name] = (float(np.nanmean(m - m15)), float(np.nanmean(c - c15)),
                       int(np.isfinite(m - m15).sum()))
    # Parametric draws.
    rng = np.random.default_rng(seed)
    td = t.loc[disp].reset_index(drop=True)
    const = td["const_heat"].fillna(False).astype(bool).to_numpy() if "const_heat" in td else \
        np.zeros(len(td), bool)
    own = td["hr_method"].isin(["quadratic", "linear"]).to_numpy() & ~const
    ownk = (td["kappa_method"] == "own").to_numpy()
    grp = td["group"].to_numpy()
    pmax = td["pmax"].to_numpy(float)
    d0, e0, k0 = (td[c].to_numpy(float) for c in ("d", "e", "kappa"))
    se_d, se_e, cde, kse = (td[c].to_numpy(float) for c in ("se_d", "se_e", "cov_de", "kappa_se"))
    d_rel = d0 / pmax
    vals = []
    for _ in range(n_draws):
        dd, ee, kk = d0.copy(), e0.copy(), k0.copy()
        for i in range(len(td)):
            if own[i] and np.isfinite(se_d[i]):
                cov = np.array([[se_d[i] ** 2, cde[i]], [cde[i], se_e[i] ** 2]])
                dd[i], ee[i] = rng.multivariate_normal([d0[i], e0[i]], cov)
            elif not own[i] and not const[i]:
                pool = np.flatnonzero(own & (grp == grp[i]))
                if len(pool):
                    j = rng.choice(pool)
                    dd[i], ee[i] = d_rel[j] * pmax[i], e0[j]
            if ownk[i] and np.isfinite(kse[i]):
                kk[i] = rng.normal(k0[i], kse[i])
            elif not ownk[i]:
                pool = np.flatnonzero(ownk & (grp == grp[i]))
                if len(pool):
                    kk[i] = k0[rng.choice(pool)]
        cv = H.TemperatureCurves(td.assign(d=dd, e=ee, kappa=kk)).arrays(temp_h)
        m, c = fd_heat_co2(target_h, cv, avail_h, headroom, co2f)
        vals.append((np.nanmean(m - m15), np.nanmean(c - c15)))
    vals = np.array(vals)
    both = diffs["both effects"]
    out = {"fd15": float(np.nanmean(m15)), "fd_hot": float(np.nanmean(m15)) + both[0],
           "value": both[0], "co2": both[1], "co2_15": float(np.nanmean(c15)),
           "diffs": diffs,
           "lo": float(np.quantile(vals[:, 0], 0.025)), "hi": float(np.quantile(vals[:, 0], 0.975)),
           "co2_lo": float(np.quantile(vals[:, 1], 0.025)),
           "co2_hi": float(np.quantile(vals[:, 1], 0.975)),
           "n_ok": both[2], "n_hours": int(hot.sum()),
           "target_gw": float(target_h.mean() / 1e3),
           "mean_temp": float(fs.fleet_temp[hot].mean())}
    out["pct"] = 100 * out["value"] / out["fd15"]
    out["per_c"] = out["value"] / (out["mean_temp"] - H.T_REF)
    return out


# --------------------------------------------------------------------------
# 5. Hold-out
# --------------------------------------------------------------------------
def holdout(uh: pd.DataFrame, tr: pd.DataFrame, n_boot: int) -> dict:
    """Jan-Aug fits, Sep-Dec heat input with and without temperature terms.

    Units with their own Jan-Aug temperature fit (non-CHP, not constant
    heat). Rows: Sep-Dec clean full hours (fit rows) and the steady subset.
    """
    tt = tr[own_hr(tr)].set_index("unit_id")
    test = uh[uh["month"].isin(TEST_MONTHS) & U.fit_rows(uh) & uh["unit_id"].isin(tt.index)].copy()
    par = tt.loc[test["unit_id"]]
    p, temp = test["gross_load_mw"].to_numpy(), test["temp_c"].to_numpy()
    test["pred0"] = par["a0"].to_numpy() + par["b0"].to_numpy() * p + par["c0"].to_numpy() * p * p
    test["predT"] = H.heat_input(par, p, temp)
    test["group"] = par["group"].to_numpy()

    def metrics(x: pd.DataFrame) -> dict:
        h = x["heat"].sum()
        return {"unit-hours": len(x),
                "bias static %": 100 * (x["pred0"].sum() / h - 1),
                "bias temp %": 100 * (x["predT"].sum() / h - 1),
                "MAE static %": 100 * (x["pred0"] - x["heat"]).abs().sum() / h,
                "MAE temp %": 100 * (x["predT"] - x["heat"]).abs().sum() / h}

    rows = []
    unit_rel = {}
    for sub_name, sub in {"all clean hours": test, "steady hours": test[test["steady"]]}.items():
        for g in H.GROUPS + ["all"]:
            x = sub if g == "all" else sub[sub["group"] == g]
            if not len(x):
                continue
            per = x.groupby("unit_id").apply(
                lambda y: pd.Series({"m0": (y["pred0"] - y["heat"]).abs().mean(),
                                     "mT": (y["predT"] - y["heat"]).abs().mean()}))
            rel = 100 * (per["mT"] / per["m0"] - 1)
            if sub_name == "all clean hours":
                unit_rel[g] = rel
            st = H.group_stats(rel, np.ones(len(rel)), n_boot=n_boot)
            rows.append({"rows": sub_name, "group": GROUP_LABEL.get(g, g), "units": len(per),
                         **metrics(x), "units improved %": 100 * float((rel < 0).mean()),
                         "median unit MAE change %": st["median"],
                         "95% CI": f"{st['median_lo']:+.2f} to {st['median_hi']:+.2f}"})
    by_t = []
    bands = [-np.inf, 10, 20, 25, 30, np.inf]
    names = ["< 10", "10-20", "20-25", "25-30", ">= 30"]
    test["band"] = pd.cut(test["temp_c"], bands, labels=names, right=False)
    for b, x in test.groupby("band", observed=True):
        by_t.append({"temperature C": b, **metrics(x)})
    # Fleet hourly totals of the covered units.
    hourly = test.groupby("operating_datetime_utc")[["heat", "pred0", "predT"]].sum()
    fleet = {"MAPE static %": 100 * float((hourly["pred0"] / hourly["heat"] - 1).abs().mean()),
             "MAPE temp %": 100 * float((hourly["predT"] / hourly["heat"] - 1).abs().mean()),
             "hours": len(hourly),
             "heat share %": 100 * float(test["heat"].sum() / uh.loc[
                 uh["month"].isin(TEST_MONTHS) & U.fit_rows(uh) & (uh["chp"] == False),  # noqa: E712
                 "heat"].sum())}
    return {"table": pd.DataFrame(rows), "by_temp": pd.DataFrame(by_t), "fleet": fleet,
            "test": test}


def dispatch_holdout(inp: Inputs, tr: pd.DataFrame, band: dict) -> pd.DataFrame:
    """Sep-Dec marginal heat-rate error of C-sx with Jan-Aug curves, static vs
    temperature-aware, by load quintile (same estimator as the error band).

    'static' = the same steady-hour curves without temperature terms
    (a0, b0, c0, Pmax = 99th percentile of output); 'temperature' = H(P, T)
    and Pavail(T) at the actual plant temperatures.
    """
    fs = build_fleet(inp, tr, TRAIN_MONTHS)
    t = fs.table
    train = np.isin(fs.month, TRAIN_MONTHS)
    test = np.isin(fs.month, TEST_MONTHS)
    disp = ~t["suspect_cc"].astype(bool).to_numpy()
    static_tab = t.assign(a=t["a0"].fillna(t["a"]), b=t["b0"].fillna(t["b"]),
                          c=t["c0"].fillna(t["c"]), p15=t["pmax"], d=0.0, e=0.0, kappa=0.0)
    heat_actual = fs.heat.sum(axis=1)
    target_all = fs.energy.sum(axis=1)
    edges = band["edges"]
    qbin = np.searchsorted(edges, target_all, side="right")
    both = test[1:] & test[:-1]
    dx, dy = np.diff(target_all), np.diff(heat_actual)
    rows = []
    fds = {}
    disp_hourly = pd.read_parquet(PROCESSED / f"dispatch_hourly_{inp.year}.parquet")
    assert (pd.DatetimeIndex(disp_hourly["hour_utc"]) == fs.hours).all()
    fds["dispatch step C-sx"] = (disp_hourly["mhr_C-sx_train"].to_numpy(), None)
    for name, tab in {"static": static_tab, "temperature": t}.items():
        full_model = H.TemperatureCurves(tab)
        _, target, headroom, avail = priority_setup(fs, full_model, train, fs.temp)
        cv = sub_model(tab, disp).arrays(fs.temp[:, disp])
        mhr, _ = fd_heat_co2(target, cv, avail, headroom, sub_model(tab, disp).co2_per_mmbtu)
        res = D.dispatch(target, cv, avail, headroom_mw=headroom[0] + headroom[1] * target)
        model_heat = res.fleet_heat(cv) + fs.heat[:, ~disp].sum(axis=1)
        fds[name] = (mhr, model_heat)
    for q, lab in enumerate(band["labels"]):
        sel = both & (qbin[1:] == q)
        b, se, n = hac_slope(dy[sel], dx[sel])
        row = {"hours": lab, "n": n, "actual": b}
        for name, (mhr, _) in fds.items():
            f = float(np.nanmean(mhr[1:][sel]))
            row[f"{name} FD"] = f
            row[f"{name} FD error %"] = 100 * (f / b - 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    hot = test & (fs.fleet_temp >= np.quantile(fs.fleet_temp[test], 0.9))
    sel = both & hot[1:]
    b, se, n = hac_slope(dy[sel], dx[sel])
    row = {"hours": f"hottest 10% of Sep-Dec hours (>= {np.quantile(fs.fleet_temp[test], 0.9):.1f} C)",
           "n": n, "actual": b}
    for name, (mhr, _) in fds.items():
        f = float(np.nanmean(mhr[1:][sel]))
        row[f"{name} FD"], row[f"{name} FD error %"] = f, 100 * (f / b - 1)
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    out.attrs["level"] = {name: 100 * (mh[test].sum() / heat_actual[test].sum() - 1)
                          for name, (_, mh) in fds.items() if mh is not None}
    out.attrs["actual_ci"] = (b - 1.96 * se, b + 1.96 * se)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors=INK2, labelsize=9)


def envelope_curves(uh: pd.DataFrame, t: pd.DataFrame, cfg: H.TempConfig) -> pd.DataFrame:
    """Per-unit monotone envelope / P15 by bin (own derating fits)."""
    tt = t[own_kappa(t)].set_index("unit_id")
    c = uh[U.fit_rows(uh) & uh["unit_id"].isin(tt.index)]
    rows = []
    for uid, d in c.groupby("unit_id"):
        drop = H.limited_months(d["month"].to_numpy(), d["gross_load_mw"].to_numpy(),
                                cfg.limited_month_ratio)
        d = d[~d["month"].isin(drop)]
        b = H.envelope_bins(d["gross_load_mw"], d["temp_c"], cfg)
        b = b[(b["n"] >= cfg.env_min_hours) & (b["t_mid"] >= cfg.env_t_min)].sort_values("t_mid")
        if not len(b):
            continue
        b["p_mono"] = np.maximum.accumulate(b["p_env"].to_numpy()[::-1])[::-1]
        b["rel"] = b["p_mono"] / tt.at[uid, "p15"]
        b["group"] = tt.at[uid, "group"]
        b["unit_id"] = uid
        rows.append(b)
    return pd.concat(rows, ignore_index=True)


def figure_capacity(env: pd.DataFrame, kt: pd.DataFrame, eia_units: pd.DataFrame, path) -> None:
    """Left: normalised capability vs temperature; right: kappa estimates by group."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for g in H.GROUPS:
        x = env[env["group"] == g].groupby("t_mid")["rel"].agg(
            ["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "size"])
        x.columns = ["med", "q1", "q3", "n"]
        n_units = env.loc[env["group"] == g, "unit_id"].nunique()
        x = x[x["n"] >= max(5, 0.3 * n_units)]
        a1.fill_between(x.index, 100 * x["q1"], 100 * x["q3"], color=COLORS[g], alpha=0.10, lw=0)
        a1.plot(x.index, 100 * x["med"], color=COLORS[g], lw=2, solid_capstyle="round",
                label=GROUP_LABEL[g])
    a1.set_xlabel("plant temperature (C, hour mean)", color=INK2, fontsize=9)
    a1.set_ylabel("output envelope, % of 15 C capability", color=INK2, fontsize=9)
    a1.set_title("Upper envelope of output vs temperature (monotone)\n(median unit, band = IQR; bins with >= 30% of units)",
                 fontsize=10, color=INK, loc="left")
    a1.legend(fontsize=8, frameon=False, loc="lower left")
    _style(a1)
    a1.set_xlim(14, 40)
    # Right: kappa by group, three estimates side by side.
    est = {"CEMS envelope (primary)": "kappa", "CEMS high-dispatch hours": "hi_kappa",
           "EIA-860 S/W (flat below 15 C)": "kappa_eia_flat"}
    shades = {"CEMS envelope (primary)": 1.0, "CEMS high-dispatch hours": 0.55,
              "EIA-860 S/W (flat below 15 C)": 0.25}
    markers = {"CEMS envelope (primary)": "o", "CEMS high-dispatch hours": "s",
               "EIA-860 S/W (flat below 15 C)": "D"}
    for gi, g in enumerate(H.GROUPS):
        x = eia_units[eia_units["group"] == g]
        own = x[x["kappa_method"] == "own"]
        for k, (lab, col) in enumerate(est.items()):
            v = (own if col != "kappa_eia_flat" else x)[col].dropna()
            if not len(v):
                continue
            s = H.group_stats(v, np.ones(len(v)), n_boot=1000)
            y = gi + (k - 1) * 0.22
            a2.plot([s["p25"], s["p75"]], [y, y], color=COLORS[g], lw=2, alpha=shades[lab])
            a2.plot(s["median"], y, markers[lab], ms=7, color=COLORS[g], mec="white", mew=1.5,
                    alpha=max(shades[lab], 0.6))
    a2.set_yticks(range(len(H.GROUPS)))
    a2.set_yticklabels([GROUP_LABEL[g] for g in H.GROUPS], fontsize=9, color=INK)
    a2.invert_yaxis()
    a2.axvline(0, color="#b9b8b2", lw=1)
    a2.set_xlabel("kappa, % of 15 C capacity lost per C (median, bar = IQR)", color=INK2, fontsize=9)
    a2.set_title("Derating slope by technology", fontsize=10, color=INK, loc="left")
    handles = [plt.Line2D([], [], marker=markers[l], ls="", color=INK2, mec="white", label=l)
               for l in est]
    a2.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    _style(a2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_heatrate(t: pd.DataFrame, by_temp: pd.DataFrame, path) -> None:
    """Heat-rate change vs temperature (equal output; full available load) and
    the hold-out bias by temperature band."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.2),
                                     gridspec_kw={"width_ratios": [1, 1, 1.05]})
    temps = np.linspace(15, 40, 26)
    m = own_hr(t)
    for ax, col, title in [(a1, "hr_pct_per_c", "At equal output (80% of Pref)"),
                           (a2, "hr_full_pct_per_c", "At full available load")]:
        for g in H.GROUPS:
            x = t[m & (t["group"] == g)]
            if col == "hr_full_pct_per_c":
                x = x[own_kappa(t).loc[x.index]]
            curves = np.outer(x[col].to_numpy(), temps - H.T_REF)
            med, q1, q3 = (np.nanquantile(curves, q, axis=0) for q in (0.5, 0.25, 0.75))
            ax.fill_between(temps, q1, q3, color=COLORS[g], alpha=0.10, lw=0)
            ax.plot(temps, med, color=COLORS[g], lw=2, label=GROUP_LABEL[g])
        ax.axhline(0, color="#b9b8b2", lw=1)
        ax.set_xlabel("plant temperature (C)", color=INK2, fontsize=9)
        ax.set_title(title + "\n(median unit, band = IQR)", fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.set_xlim(15, 40)
    a1.set_ylabel("heat-rate change vs 15 C, %", color=INK2, fontsize=9)
    lo = min(a1.get_ylim()[0], a2.get_ylim()[0])
    hi = max(a1.get_ylim()[1], a2.get_ylim()[1])
    a1.set_ylim(lo, hi)
    a2.set_ylim(lo, hi)
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    x = np.arange(len(by_temp))
    a3.plot(x, by_temp["bias static %"], color=INK2, lw=2, marker="s", ms=7, mec="white", mew=1.5,
            label="static curve")
    a3.plot(x, by_temp["bias temp %"], color=COLORS["CT"], lw=2, marker="o", ms=7, mec="white",
            mew=1.5, label="temperature curve")
    a3.set_xticks(x)
    a3.set_xticklabels(by_temp["temperature C"].astype(str), fontsize=9)
    a3.axhline(0, color="#b9b8b2", lw=1)
    a3.set_xlabel("Sep-Dec plant temperature band (C)", color=INK2, fontsize=9)
    a3.set_ylabel("out-of-sample heat-input bias, %", color=INK2, fontsize=9)
    a3.set_title("Hold-out: Jan-Aug fit, Sep-Dec\n(units with own fits, all clean hours)",
                 fontsize=10, color=INK, loc="left")
    a3.legend(fontsize=8, frameon=False)
    _style(a3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
SCHEMA = [
    ("unit_id, plant_id_epa, emissions_unit_id_epa, plant_id_eia, plant_name_eia", "identifiers"),
    ("tech, group", "EIA technology (CC/CT/ST); analysis group CT, CC, CC* (suspect CC: median heat rate >= 8.5, probably CT output only), ST, CHP"),
    ("chp, suspect_cc, const_heat", "flags; const_heat = >= 90% of fit hours report the same heat input (substituted data; no temperature terms)"),
    ("nameplate_mw, summer_mw, winter_mw", "EIA-860 capacity allocated to the CEMS unit (net, MW)"),
    ("n_clean, n_steady", "clean full hours (heat-rate floor applied) and the steady subset outside capability-limited months, used in the heat fit"),
    ("pmin, pmax", "5th / 99th percentile of clean full-hour gross output (MW); pmax is Pref for load fractions"),
    ("a, b, c, d, e", "H = a + bP + cP^2 + d(T-15) + e(T-15)P, MMBtu/h, P gross MW, T C"),
    ("hr_method", "quadratic / linear (own fit), group (static curve + group-median d/pmax and e), none (no temperature terms)"),
    ("a0, b0, c0, rmse0", "static curve H = a0 + b0 P + c0 P^2 on the same rows (no temperature)"),
    ("r2, rmse", "in-sample fit of the temperature curve (MMBtu/h)"),
    ("se_d, se_e, cov_de", "cluster-robust (local day) standard errors and covariance of d, e"),
    ("d_wd, e_wd", "d, e from within-day variation only (day fixed effects, OLS)"),
    ("hr15_lf, hr_pct_per_c, hr_pct_per_c_wd, lf_in_range", "heat rate at 0.8 pmax and 15 C (MMBtu/MWh); its change % per C (main and within-day); 0.8 pmax inside [pmin, pmax]"),
    ("hr_full_pct_per_c", "heat rate at full available load, change % per C between 15 and 35 C"),
    ("p15, kappa, kappa_se, kappa_method", "Pavail(T) = p15 (1 - kappa/100 (T-15)) above 15 C; kappa in % of p15 per C; own / group (group median, p15 = pmax) / none"),
    ("n_bins, t_lo, t_hi, months_dropped, limited_months", "envelope bins used, their centre range (C), number and list of capability-limited months (left out of the envelope and the heat fit)"),
    ("raw_*, b21_*, hi_*", "derating variants: no monotone step; bins >= 21 C; high-dispatch hours only"),
    ("t_p05, t_p95, t_fit_lo, t_fit_hi", "temperature range of the unit's clean hours: P5/P95 (all months) and min/max outside the capability-limited months (the dispatch curve model does not extrapolate the heat terms beyond t_fit_lo/hi)"),
    ("median_hr, co2_per_mmbtu", "median hourly heat rate; CO2 short tons per MMBtu (CEMS)"),
]


def write_report(ctx: dict, path) -> None:
    """Assemble the markdown report from computed results."""
    L = []
    a = L.append
    t = ctx["full"]
    band = ctx["band"]
    mv = ctx["marginal"]
    ap = ctx["avg"]
    apw = ctx["avg_wd"]
    cl = ctx["cap"]
    ho = ctx["holdout"]
    dh = ctx["disp_hold"]
    yr = ctx["year"]
    a(f"# Ambient temperature, heat rate and capacity of the ERCOT gas fleet, {yr}")
    a("")
    a(f"Generated by `scripts/fit_heatrate_derating.py` on {date.today().isoformat()} from EPA CEMS "
      f"(PUDL `{pudl.DEFAULT_RELEASE}`), EIA-860 and the plant temperatures of "
      "`scripts/build_plant_weather.py`. Every number below was computed by that script; the "
      "dispatch error band is recomputed from the hourly output of `scripts/validate_dispatch.py`.")
    a("")
    # ---------------- Summary
    tk = ctx["kappa_tab"].set_index("group")
    vv = ctx["variants"].set_index("group")
    th = ctx["hr_tab"].set_index("group")
    tw = ctx["hr_wd_tab"].set_index("group")
    tf = ctx["hr_full_tab"].set_index("group")
    a("## Summary")
    a("")
    a(f"- **Dispatch model error band first.** Out of sample (Sep-Dec, Jan-Aug fit) the dispatch "
      f"model's marginal heat rate (+/-500 MW derivative, main configuration C-sx) differs from the CEMS "
      f"hour-to-hour slope by {band['lo']:+.1f}% to {band['hi']:+.1f}% across load quintiles "
      f"(hours-weighted mean absolute error {band['mae']:.1f}%). In the hottest 10% of Sep-Dec hours the same "
      f"model overstates the CEMS slope by {dh.iloc[5]['dispatch step C-sx FD error %']:+.1f}% (section 6). "
      "A temperature effect on the marginal heat rate is only distinguishable from model error if it is "
      "larger than this.")
    a(f"- **Capacity falls with temperature.** Above 15 C, the upper envelope of output declines by "
      f"a median {tk.at[GROUP_LABEL['CT'], 'median']:.2f}% of the 15 C capability per C for simple-cycle CTs "
      f"(95% CI of the median {tk.at[GROUP_LABEL['CT'], 'median 95% CI']}), "
      f"{tk.at[GROUP_LABEL['CC'], 'median']:.2f}% for CC units ({tk.at[GROUP_LABEL['CC'], 'median 95% CI']}), "
      f"{tk.at[GROUP_LABEL['CC*'], 'median']:.2f}% for CC units that report CT output only "
      f"and {tk.at[GROUP_LABEL['ST'], 'median']:.2f}% for steam units ("
      f"{vv.at[GROUP_LABEL['ST'], 'raw envelope >= 15 C']:.2f} with the raw envelope and "
      f"{vv.at[GROUP_LABEL['ST'], 'high-dispatch hours >= 15 C']:.2f} in high-dispatch hours, so ST derating "
      "is small and uncertain). Dispatch selection still biases the CT estimate toward zero: in "
      f"high-dispatch hours the CT median is {vv.at[GROUP_LABEL['CT'], 'high-dispatch hours >= 15 C']:.2f}. "
      "The technology medians lie inside the range implied by EIA-860 summer/winter ratings for "
      f"{', '.join(ctx['eia_inside'])} (section 4.3).")
    a(f"- **Hottest hours.** In the hottest 1% of 2024 hours ({cl['n_hot']} h, capacity-weighted "
      f"fleet temperature {cl['t_hot']:.1f} C on average) the non-CHP gas units had "
      f"{cl['lost_mean']:,.0f} MW less capability than at 15 C (95% CI {cl['lost_lo']:,.0f} to "
      f"{cl['lost_hi']:,.0f} MW), {100 * cl['lost_mean'] / cl['p15_total']:.1f}% of their "
      f"{cl['p15_total']:,.0f} MW 15 C capability; {cl['lost_max_hour']:,.0f} MW in the single "
      f"hottest hour ({cl['t_max']:.1f} C).")
    a(f"- **Heat rate at equal output barely moves.** At 80% of reference output, the median unit's "
      f"heat rate changes by {th.at[GROUP_LABEL['CT'], 'median']:+.3f}% per C (CT), "
      f"{th.at[GROUP_LABEL['CC'], 'median']:+.3f}% (CC), {th.at[GROUP_LABEL['CC*'], 'median']:+.3f}% (CC*) "
      f"and {th.at[GROUP_LABEL['ST'], 'median']:+.3f}% (ST). Estimated from within-day variation alone "
      "(day fixed effects), the median lies inside the 95% CI of the main median for "
      + (", ".join(f"{g} ({tw.at[GROUP_LABEL[g], 'median']:+.3f}%)" for g in ctx["wd_stable"]) or "no group")
      + " but not for "
      + (", ".join(f"{g} ({tw.at[GROUP_LABEL[g], 'median']:+.3f}%)" for g in H.GROUPS
                   if g not in ctx["wd_stable"]) or "any group")
      + ". At full available load, where derating moves units down their curves, the median heat rate rises "
      f"{tf.at[GROUP_LABEL['CT'], 'median']:+.2f}% per C (CT) and {tf.at[GROUP_LABEL['CC'], 'median']:+.2f}% (CC). "
      "At the actual unit outputs of the hottest "
      f"1% of hours this adds {ap['mmbtu']:.3f} MMBtu/MWh (95% CI {ap['lo']:.3f} to {ap['hi']:.3f}, units "
      f"resampled; {ap['pct']:.2f}% of the heat rate), {ap['kg']:.1f} kg CO2/MWh, relative to the same outputs "
      f"at 15 C; with the within-day heat terms {apw['mmbtu']:.3f} MMBtu/MWh ({apw['kg']:.1f} kg CO2/MWh).")
    dv = mv["diffs"]
    a(f"- **Value of moving 1 MWh out of a hot hour (temperature part, equal output).** Unit level, "
      f"without the dispatch model: {ap['mmbtu']:.3f} MMBtu and {ap['kg']:.1f} kg CO2 per MWh "
      f"(95% CI {ap['lo']:.3f} to {ap['hi']:.3f} MMBtu; {ap['kg_lo']:.1f} to {ap['kg_hi']:.1f} kg). At the margin, "
      f"in the dispatch model: the marginal heat rate in the hottest 1% of hours is {mv['fd_hot']:.2f} "
      f"MMBtu/MWh at the actual temperatures and {mv['fd15']:.2f} with every plant at 15 C (same output, "
      f"availability and commitment rule), so temperature adds {mv['value']:+.3f} MMBtu/MWh "
      f"({mv['pct']:+.1f}%; 95% range over parameter draws {mv['lo']:+.3f} to {mv['hi']:+.3f}) and "
      f"{mv['co2']:+.1f} kg CO2/MWh ({mv['co2_lo']:+.1f} to {mv['co2_hi']:+.1f}); derating alone "
      f"{dv['derating only'][0]:+.3f}, heat-rate terms alone {dv['heat-rate terms only'][0]:+.3f}. "
      + ctx["verdict"])
    hall = ho["table"]
    ha = hall[(hall["rows"] == "all clean hours") & (hall["group"] == "all")].iloc[0]
    bt = ho["by_temp"]
    a(f"- **Hold-out (fit Jan-Aug, predict Sep-Dec).** Without temperature terms the out-of-sample bias of "
      f"unit heat input runs from {bt['bias static %'].iloc[0]:+.2f}% (coldest band) to "
      f"{bt['bias static %'].iloc[-1]:+.2f}% (hottest band); with them it stays within "
      f"{bt['bias temp %'].min():+.2f}% to {bt['bias temp %'].max():+.2f}%, and the hourly error of the "
      f"summed heat input falls from {ho['fleet']['MAPE static %']:.2f}% to {ho['fleet']['MAPE temp %']:.2f}% "
      f"(MAPE). Unit-hour scatter does not shrink (heat-weighted MAE {ha['MAE static %']:.2f}% vs "
      f"{ha['MAE temp %']:.2f}%; {ha['units improved %']:.0f}% of {int(ha['units'])} units improve; "
      f"the median unit gets worse for {ctx['holdout_worse_text']}, section 6). In the temperature-aware "
      "dispatch the Sep-Dec "
      f"derivative error is {dh.attrs['mae_temp']:.1f}% vs {dh.attrs['mae_static']:.1f}% without temperature "
      "(weighted MAE over quintiles).")
    a("")
    # ---------------- 1. Error band
    a("## 1. The dispatch model's error band (read this first)")
    a("")
    a("Recomputed from `data/processed/dispatch_hourly_{0}.parquet` with the estimator of "
      "`scripts/validate_dispatch.py`: CEMS hour-to-hour slope of non-CHP heat input on non-CHP gross "
      "output (OLS, HAC 24 lags) by load quintile of the later hour, Sep-Dec, against the mean of the main "
      "model's (C-sx, Jan-Aug curves) +/-500 MW derivative over the same hours.".format(yr))
    a("")
    a(md_table(band["table"], formats={"FD error %": "{:+.1f}"}))
    a("")
    a(f"Error band: {band['lo']:+.1f}% to {band['hi']:+.1f}%, hours-weighted MAE {band['mae']:.1f}%. "
      "The dispatch report (`docs/reports/dispatch_validation.md`) gives the same band and shows that it "
      "depends on the configuration. Everything below that runs through the dispatch model is read "
      "against it.")
    a("")
    # ---------------- 2. Data and method
    cfg = ctx["cfg"]
    a("## 2. Data and method")
    a("")
    a(f"- Units: {len(t)} CEMS units with clean full hours in {yr}; CHP units ({int((t['group'] == 'CHP').sum())}) are fitted "
      "but left out of every result (their fuel also makes process steam). Groups: CT (simple cycle), CC, "
      "CC* and ST. **CC\\* = suspect CC units** (non-CHP CC with a median hourly heat rate >= 8.5 MMBtu/MWh "
      "gross, full year), which probably report the combustion turbine's output without the steam "
      f"turbine's: {int((t['group'] == 'CC*').sum())} units, {t.loc[t['group'] == 'CC*', 'nameplate_mw'].sum():,.0f} MW nameplate. "
      "They are reported as a separate group throughout; their numbers describe a CT inside a combined "
      "cycle, not a CC plant.")
    a(f"- Hours: CEMS full operating hours with positive load and heat and a heat rate >= 5.5 MMBtu/MWh "
      f"(`ttrl.supply.units.fit_rows`). Heat-input fits use the steady subset: the unit also ran full "
      f"hours in the two previous and the next hour, and its output changed by at most "
      f"{100 * cfg.ramp_frac:.0f}% of Pref since the previous hour; the unit's capability-limited "
      "months (see Derating) are left out too, because a block running with part of its equipment out "
      "(e.g. a CC with the steam turbine out for the summer) is a different operating mode whose seasonal "
      f"heat rate would otherwise load on temperature ({ctx['n_steady']:,} of {ctx['n_clean']:,} clean "
      "unit-hours).")
    a("- Temperature: the plant's ISD-lite temperature (`data/interim/plant_temperature_{0}.parquet`), "
      "averaged over the observations at the start and end of each CEMS hour (the reading at ts is taken "
      "about 7 minutes before the hour starts). Dry-bulb only.".format(yr))
    a(f"- Units with substituted heat input (>= 90% of hours report the same value): {int(t['const_heat'].sum())} "
      f"({', '.join(t.loc[t['const_heat'], 'unit_id'])}); they get no temperature terms.")
    a("")
    a("**Heat-input model.** Per unit, H(P, T) = a + bP + cP^2 + d(T - 15) + e(T - 15)P (MMBtu/h, P gross MW, "
      "T in C), Huber robust regression. d shifts the curve, e tilts it, so the heat rate at equal output "
      "changes by (d + eP)/H(P, 15) per C and the incremental heat rate by e per C. Reasons for this form: "
      "(i) it is the dispatch step's convex quadratic with intercept and slope linear in temperature, so it "
      "plugs into the existing equal-incremental dispatch for every hour; (ii) two extra parameters per unit "
      "keep it estimable for peakers with a few hundred steady hours; (iii) a heat rate at one fixed load "
      "fraction would throw away the load information and could not feed dispatch. Linearity in T is "
      "checked with residuals by temperature band (section 3.1). Units need >= "
      f"{cfg.min_steady_hours} steady hours spanning >= {cfg.min_temp_span:.0f} C (P5-P95) for their own fit; "
      "others get their static curve plus the group-median d/Pref and e. Standard errors are cluster-robust "
      "by local day. Robustness: the same terms estimated from within-day variation only (day fixed effects), "
      "which removes slow seasonal changes such as degradation, washing or maintenance.")
    a("")
    a("**Common load fraction.** Heat-rate results are at 80% of each unit's reference output Pref "
      "(99th percentile of its clean full-hour output). This load lies inside the unit's 5th-99th percentile "
      "output range for " + ", ".join(
          f"{100 * t.loc[own_hr(t) & (t['group'] == g), 'lf_in_range'].mean():.0f}% of {g}"
          for g in H.GROUPS) + " units with own fits (many peakers rarely run below 85-90%). The "
      "full-available-load heat rate (Pavail(T), 15 vs 35 C) is reported as well.")
    a("")
    a(f"**Derating.** Per unit, the {100 * cfg.env_quantile:.0f}th percentile of clean full-hour gross output "
      f"in {cfg.bin_width:.0f} C temperature bins (>= {cfg.env_min_hours} hours per bin), and a straight line through "
      f"the bins with centre >= {cfg.env_t_min:.0f} C (>= {cfg.env_min_bins} bins spanning >= {cfg.env_min_span:.0f} C): "
      "Pavail(T) = P15 (1 - kappa/100 (T - 15)), kappa in % of the 15 C capability per C. Below 15 C the "
      "model holds P15 (cold side not estimated). Two screens before the fit: (1) months whose 99th-percentile "
      f"output is below {cfg.limited_month_ratio:.0%} of the unit's are left out (capability limits that "
      "temperature cannot explain, e.g. Tenaska Gateway's CC units ran at about half output from May to October); "
      "(2) **monotone envelope**: a bin's value is raised to the highest value of any hotter bin, because a unit "
      "that produced X MW at a hotter temperature could produce X MW at a cooler one.")
    a("")
    a("**Selection bias.** In mild hours demand is lower and units are often not pushed to their limit, so the "
      "mild-bin envelope understates capability and kappa is biased toward zero (in a summer-peaking system "
      "this bias is one-sided for T > 15 C). The monotone step removes the visible part of it (mild bins below "
      "hotter bins). Checks: (a) the raw envelope without that step; (b) the same estimator on **high-dispatch "
      f"hours** only (fleet CEMS gross output in the top {100 * (1 - HIGH_DISPATCH_Q):.0f}% of {yr} hours, "
      f"{100 * cfg.hi_quantile:.0f}th percentile, >= {cfg.hi_min_hours} hours per bin), when most available units run hard, "
      "compared with the all-hours estimate on the same temperature range (bins >= 21 C, because high-dispatch hours "
      "below 21 C are rare); (c) EIA-860 summer and winter net capacity, an external rating that does not depend "
      "on dispatch. Units without an own envelope fit get their group's median kappa and P15 = Pref.")
    a("")
    # ---------------- 3. Heat rate
    a("## 3. Heat input and temperature")
    a("")
    a("Fit coverage (full-year fit):")
    a("")
    a(md_table(ctx["coverage"], formats={"median steady hours (own fits)": "{:,.0f}",
                                         "median R2": "{:.3f}"}))
    a("")
    a("### 3.1 Residual check")
    a("")
    a("Steady hours of own-fit units, actual minus fitted heat input over fitted, % (positive = the model "
      "under-predicts). 'static' is the same unit's curve without temperature terms fitted on the same hours.")
    a("")
    a(md_table(ctx["resid"], formats={"static %": "{:+.2f}", "temperature model %": "{:+.2f}"}))
    a("")
    a(ctx["resid_text"])
    a("")
    a("### 3.2 Heat-rate change per C, by technology")
    a("")
    a("At equal output (80% of Pref), % of the 15 C heat rate per C; own-fit units; capacity weights = "
      "EIA-860 nameplate; CIs are bootstrap percentiles over units (2,000 resamples).")
    a("")
    a(md_table(ctx["hr_tab"], floatfmt="{:+.3f}", formats={"MW (nameplate)": "{:,.0f}"}))
    a("")
    a("Same, estimated from within-day variation only (day fixed effects):")
    a("")
    a(md_table(ctx["hr_wd_tab"], floatfmt="{:+.3f}", formats={"MW (nameplate)": "{:,.0f}"}))
    a("")
    a("At full available load, 15 C vs 35 C (includes the move to a lower output on the 15 C curve because "
      "of derating), % per C:")
    a("")
    a(md_table(ctx["hr_full_tab"], floatfmt="{:+.3f}", formats={"MW (nameplate)": "{:,.0f}"}))
    a("")
    a("Incremental heat rate: the median e (change of dH/dP) is " + ", ".join(
        f"{g} {t.loc[own_hr(t) & (t['group'] == g), 'e'].median():+.4f}" for g in H.GROUPS)
      + " MMBtu/MWh per C.")
    a("")
    a("![heat rate](heatrate_derating_heatrate.png)")
    a("")
    # ---------------- 4. Derating
    a("## 4. Available capacity and temperature")
    a("")
    a("### 4.1 kappa by technology")
    a("")
    a("Own envelope fits (primary estimator); % of 15 C capability lost per C above 15 C.")
    a("")
    a(md_table(ctx["kappa_tab"], floatfmt="{:.3f}", formats={"MW (nameplate)": "{:,.0f}"}))
    a("")
    cov_k = ctx["coverage_k"]
    a("Units per group with an own envelope fit / group-median fallback: " + "; ".join(
        f"{g} {cov_k.get((g, 'own'), 0)} / {cov_k.get((g, 'group'), 0)}" for g in H.GROUPS) + ".")
    a("")
    a("![capacity](heatrate_derating_capacity.png)")
    a("")
    a("### 4.2 Selection-bias checks")
    a("")
    a("Median kappa (% per C) of the same own-fit units under each estimator:")
    a("")
    a(md_table(ctx["variants"], floatfmt="{:.3f}", formats={
        "share of units higher in high-dispatch %": "{:.0f}", "raw < 0 %": "{:.0f}",
        "high-dispatch minus >= 21 C, median": "{:+.3f}"}))
    a("")
    a(ctx["variants_text"])
    a("")
    a("### 4.3 External check: EIA-860 summer and winter capacity")
    a("")
    e = ctx["eia_stats"]
    a(f"EIA-860 reports net summer (June-September peak) and winter (December-February) capacity but no "
      f"rating temperature. T_summer / T_winter are each plant's mean temperature in the {100 * PEAK_SHARE:.0f}% "
      "of June-September / December-February hours with the highest CEMS fleet output. The model ratio "
      "Pavail(T_summer)/P15 equals the model's summer/winter ratio if winter ratings refer to 15 C or colder "
      "(the model is flat below 15 C). The EIA-implied kappa uses the same assumption; the 'linear to T_winter' "
      "column assumes instead that capacity keeps rising at the same slope down to T_winter (a lower bound).")
    a("")
    a(md_table(ctx["eia_tab"], floatfmt="{:.3f}", formats={
        "EIA S = W %": "{:.0f}", "T_summer C median": "{:.1f}", "T_winter C median": "{:.1f}"}))
    a("")
    a(f"Across units, the CEMS seasonal p99 ratio and the EIA ratio correlate at {e['corr_season']:.2f} "
      f"(n = {e['n_season']}); per-unit kappa (own fits) and EIA-implied kappa at {e['corr_kappa']:.2f} "
      f"(n = {e['n_corr']}). The fleet's EIA winter-minus-summer capacity for these units is "
      f"{e['fleet_w_minus_s']:,.0f} MW ({100 * e['fleet_w_minus_s'] / e['fleet_w']:.1f}% of winter "
      "capacity). EIA capacity is net and CEMS output gross; auxiliary load is somewhat higher in summer, "
      "so a net ratio should be slightly lower than a gross one.")
    a("")
    a(ctx["eia_text"])
    a("")
    # ---------------- 5. Fleet
    a("## 5. Fleet level: hot hours")
    a("")
    a(f"Hot hours: the {cl['n_hot']} hottest CEMS hours of {yr} ({100 * HOT_SHARE:.0f}%) by the "
      "capacity-weighted (P15) mean temperature of the non-CHP units; mean "
      f"{cl['t_hot']:.1f} C, all between {cl['hot_first']} and {cl['hot_last']} (local standard time). "
      "Non-CHP units only (CT, CC, CC*, ST).")
    a("")
    a("### 5.1 Capacity lost vs 15 C")
    a("")
    a(md_table(ctx["cap_tab"], formats={"P15 MW": "{:,.0f}", "lost MW": "{:,.0f}", "lost %": "{:.1f}",
                                        "mean T C": "{:.1f}"}))
    a("")
    a(f"Total {cl['lost_mean']:,.0f} MW (95% CI {cl['lost_lo']:,.0f} to {cl['lost_hi']:,.0f} MW, units resampled "
      f"within groups); single hottest hour ({cl['t_max_when']}, {cl['t_max']:.1f} C): {cl['lost_max_hour']:,.0f} MW. "
      f"With the high-dispatch-hour kappa (where a unit has one): {cl['lost_mean_hi']:,.0f} MW. The raw "
      "envelope is not used at fleet level because a few dispatch-limited units get non-physical slopes "
      "(section 4.2). For comparison, EIA-860 winter minus summer capacity of the same units is "
      f"{e['fleet_w_minus_s']:,.0f} MW (winter ratings refer to colder conditions than 15 C, which this "
      "model does not cover).")
    a("")
    a("### 5.2 Extra fuel per MWh at equal output")
    a("")
    a(f"**Average (unit level).** At the actual output of every unit in the hot hours "
      f"({ap['mwh'] / 1e6:.2f} TWh gross), the temperature terms add {ap['mmbtu']:.3f} MMBtu/MWh "
      f"(95% CI {ap['lo']:.3f} to {ap['hi']:.3f}) to a 15 C heat rate of {ap['hr15']:.2f} MMBtu/MWh "
      f"({ap['pct']:.2f}%), i.e. {ap['kg']:.1f} kg CO2/MWh (95% CI {ap['kg_lo']:.1f} to {ap['kg_hi']:.1f}); the "
      f"output-weighted temperature excess over 15 C is {ap['mean_dT']:.1f} C. Unit-hours in a unit's "
      f"capability-limited months ({ap['excluded_pct']:.1f}% of the hot hours' clean output, e.g. Tenaska "
      "Gateway's reduced summer mode) are left out, because the heat model does not describe that mode. "
      "The CIs resample units within "
      "groups with each unit's d and e held at their estimates, so they do not include the estimation error "
      f"of d and e. With the within-day (day fixed effects) heat terms the same quantity is {apw['mmbtu']:.3f} "
      f"MMBtu/MWh (95% CI {apw['lo']:.3f} to {apw['hi']:.3f}), {apw['kg']:.1f} kg CO2/MWh.")
    a("")
    a(f"**Marginal (dispatch model).** Main configuration C-sx (non-CHP, suspect CC units at actual output), "
      "monthly availability, headroom rule refitted with the temperature-aware capacity, full-year curves. "
      f"For each hot hour, the +/-{DELTA_MW:.0f} MW derivative at the actual temperatures minus the same "
      f"derivative with every plant at 15 C, at the same output target (mean {mv['target_gw']:.1f} GW) and "
      f"availability ({mv['n_ok']} of {mv['n_hours']} hours solved in both runs).")
    a("")
    a(md_table(ctx["mv_tab"], formats={"MMBtu/MWh": "{:+.3f}", "kg CO2/MWh": "{:+.1f}"}))
    a("")
    a(f"Uncertainty: {ctx['n_draws']} parametric draws of every unit's temperature parameters (own fits: "
      "normal with their cluster-robust covariance and the envelope SE of kappa; group-assigned units: the "
      "parameters of a random own-fit unit of the same group). It covers parameter uncertainty only, not "
      "the dispatch model's structural error (section 1).")
    a("")
    a("**Value of moving 1 MWh out of a hot hour** (temperature part only, at equal output, vs the same hour "
      f"at 15 C): {ci_text(mv['value'], mv['lo'], mv['hi'], '{:+.3f}')} MMBtu and "
      f"{ci_text(mv['co2'], mv['co2_lo'], mv['co2_hi'], '{:+.1f}')} kg CO2 at the margin "
      f"({mv['pct']:+.1f}% of the 15 C marginal heat rate; {mv['per_c']:.4f} MMBtu/MWh per C of fleet "
      f"temperature above 15 C); {ap['mmbtu']:.3f} MMBtu and {ap['kg']:.1f} kg CO2 on average over the units "
      "that run. " + ctx["value_text"])
    a("")
    # ---------------- 6. Hold-out
    a("## 6. Hold-out: Jan-Aug fit, Sep-Dec prediction")
    a("")
    a("Units with their own Jan-Aug temperature fit (non-CHP, no substituted heat). 'static' = the same "
      "unit's curve without temperature terms fitted on the same Jan-Aug steady hours. Bias = predicted/actual "
      "- 1; MAE = sum |error| / sum actual; 'median unit MAE change' = median over units of "
      "MAE(temperature)/MAE(static) - 1 with a bootstrap CI over units.")
    a("")
    a(md_table(ho["table"], formats={"bias static %": "{:+.2f}", "bias temp %": "{:+.2f}",
                                     "MAE static %": "{:.2f}", "MAE temp %": "{:.2f}",
                                     "units improved %": "{:.0f}", "median unit MAE change %": "{:+.2f}"}))
    a("")
    a("By Sep-Dec temperature band (all clean hours of the same units):")
    a("")
    a(md_table(ho["by_temp"], formats={"bias static %": "{:+.2f}", "bias temp %": "{:+.2f}",
                                       "MAE static %": "{:.2f}", "MAE temp %": "{:.2f}"}))
    a("")
    f = ho["fleet"]
    a(f"Hourly sum over these units ({f['hours']:,} hours, {f['heat share %']:.0f}% of non-CHP clean-hour heat "
      f"input in Sep-Dec): MAPE {f['MAPE static %']:.2f}% static vs {f['MAPE temp %']:.2f}% with temperature.")
    a("")
    a(ctx["holdout_text"])
    a("")
    a("**Dispatch model with and without temperature (Sep-Dec, Jan-Aug curves).** 'dispatch step C-sx' is "
      "the model of section 1 (its curves). 'static' and 'temperature' are the same configuration with this "
      "step's steady-hour curves: without temperature terms (Pmax = Pref), and with H(P, T) and Pavail(T) at "
      "the actual plant temperatures; the headroom rule is refitted for each. The last row uses the 10% "
      "hottest Sep-Dec hours (capacity-weighted fleet temperature).")
    a("")
    a(md_table(dh, formats={"static FD error %": "{:+.1f}", "temperature FD error %": "{:+.1f}",
                            "dispatch step C-sx FD error %": "{:+.1f}"}))
    a("")
    a(f"Weighted MAE over the five quintiles: static {dh.attrs['mae_static']:.1f}%, temperature "
      f"{dh.attrs['mae_temp']:.1f}%. Sep-Dec fuel level error: static {dh.attrs['level']['static']:+.1f}%, "
      f"temperature {dh.attrs['level']['temperature']:+.1f}%. Hot-hour row: CEMS slope 95% CI "
      f"{dh.attrs['actual_ci'][0]:.2f} to {dh.attrs['actual_ci'][1]:.2f}. " + ctx["disp_hold_text"])
    a("")
    # ---------------- 7. Outputs
    a("## 7. Output: `data/processed/unit_curves_{0}.parquet`".format(yr))
    a("")
    a(f"One row per CEMS unit with clean hours ({len(t)} rows, full-year fit; the Jan-Aug fit used in "
      f"section 6 is `unit_curves_{yr}_janaug.parquet` with the same columns). "
      "`ttrl.supply.heatrate.TemperatureCurves(table)` turns it into a dispatch curve model "
      "(`arrays(temp_c)` with temp_c of shape (hours, units)).")
    a("")
    a(md_table(pd.DataFrame(SCHEMA, columns=["columns", "meaning"])))
    a("")
    # ---------------- 8. Limitations
    a("## 8. Assumptions and limitations")
    a("")
    for x in ctx["limitations"]:
        a(f"- {x}")
    a("")
    a(f"Runtime: {ctx['runtime']:.0f} s on 4 CPUs.")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples over units")
    ap.add_argument("--draws", type=int, default=200, help="parametric draws for the dispatch value")
    args = ap.parse_args()
    t_start = time.time()
    cfg = H.TempConfig()

    band = dispatch_band(args.year)
    print(f"dispatch band {band['lo']:+.1f} to {band['hi']:+.1f}%, MAE {band['mae']:.1f}%")
    inp = load_inputs(args.year)
    print(f"inputs: {len(inp.uh):,} unit-hours, {time.time() - t_start:.0f} s")
    tabs = fit_tables(inp, cfg)
    full, train = tabs["full"], tabs["train"]
    print(f"fits: {time.time() - t_start:.0f} s")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    full.to_parquet(PROCESSED / f"unit_curves_{args.year}.parquet", index=False)
    train.to_parquet(PROCESSED / f"unit_curves_{args.year}_janaug.parquet", index=False)

    ctx: dict = {"year": args.year, "cfg": cfg, "band": band, "full": full,
                 "n_draws": args.draws}
    ctx["n_clean"], ctx["n_steady"] = int(full["n_clean"].sum()), int(full["n_steady"].sum())

    # Coverage.
    cov = []
    for g in H.GROUPS:
        x = full[full["group"] == g]
        cov.append({"group": GROUP_LABEL[g], "units": len(x),
                    "own heat fit (quadratic)": int((x["hr_method"] == "quadratic").sum()),
                    "own heat fit (linear)": int((x["hr_method"] == "linear").sum()),
                    "group terms": int((x["hr_method"] == "group").sum()),
                    "no terms": int((x["hr_method"] == "none").sum()),
                    "median steady hours (own fits)": float(x.loc[own_hr(x), "n_steady"].median()),
                    "median R2": float(x.loc[own_hr(x), "r2"].median()),
                    "median RMSE static (MMBtu/h)": float(x.loc[own_hr(x), "rmse0"].median()),
                    "median RMSE temp (MMBtu/h)": float(x.loc[own_hr(x), "rmse"].median())})
    ctx["coverage"] = pd.DataFrame(cov)
    ctx["coverage_k"] = full.groupby(["group", "kappa_method"]).size().to_dict()

    # Residual check.
    resid = residual_checks(inp.uh, full)
    resid["group"] = resid["group"].astype(str)
    ctx["resid"] = resid
    warm = resid[resid["temperature C"] != "< 5"]
    cold = resid[resid["temperature C"] == "< 5"].set_index("group")
    spread = warm.groupby("group")[["static %", "temperature model %"]].agg(lambda x: x.max() - x.min())
    ctx["resid_text"] = (
        "Reading: from 5 C up, the spread of the band means (max - min, percentage points) is "
        + "; ".join(f"{g} {spread.at[g, 'static %']:.2f} static vs {spread.at[g, 'temperature model %']:.2f} "
                    "with temperature" for g in H.GROUPS)
        + ". Where the static residual rises steadily with temperature (a sign that heat input at equal "
        "output grows with temperature), the linear terms remove the trend; where it does not, they change "
        "little. Below 5 C the temperature model's mean residual is " + ", ".join(
            f"{cold.at[g, 'temperature model %']:+.2f}% ({g})" for g in H.GROUPS if g in cold.index)
        + ": a straight line fitted mostly on warmer hours does not hold in the few cold hours, so the "
        "terms should not be extrapolated far below 5 C. No band other than < 5 C has a mean residual "
        f"beyond +/-{warm['temperature model %'].abs().max():.2f}% with the temperature model.")

    # Technology tables.
    mh, mk = own_hr(full), own_kappa(full)
    ctx["hr_tab"] = tech_table(full, "hr_pct_per_c", mh, args.boot)
    ctx["hr_wd_tab"] = tech_table(full, "hr_pct_per_c_wd", mh, args.boot)
    ctx["hr_full_tab"] = tech_table(full, "hr_full_pct_per_c", mh & mk, args.boot)
    # Groups whose within-day median lies inside the bootstrap CI of the main median.
    ctx["wd_stable"] = []
    for g in H.GROUPS:
        x = full[mh & (full["group"] == g)]
        s_main = H.group_stats(x["hr_pct_per_c"], x["nameplate_mw"], n_boot=args.boot)
        s_wd = H.group_stats(x["hr_pct_per_c_wd"], x["nameplate_mw"], n_boot=args.boot)
        if s_main["median_lo"] <= s_wd["median"] <= s_main["median_hi"]:
            ctx["wd_stable"].append(g)
    ctx["kappa_tab"] = tech_table(full, "kappa", mk, args.boot)
    var = kappa_variants(full)
    ctx["variants"] = var
    v = var.set_index("group")
    ctx["variants_text"] = (
        "Reading: the raw envelope gives negative kappa (capability rising with temperature, which is not "
        "physical) for " + ", ".join(f"{v.at[GROUP_LABEL[g], 'raw < 0 %']:.0f}% of {g}" for g in H.GROUPS)
        + " units, the signature of mild hours in which the unit was not run to its limit. In high-dispatch "
        "hours, on the same temperature range, the median paired difference is " + ", ".join(
            f"{v.at[GROUP_LABEL[g], 'high-dispatch minus >= 21 C, median']:+.3f} ({g})" for g in H.GROUPS)
        + " %/C. A positive difference means the all-hours estimate still understates derating "
        "because of dispatch selection; a difference near zero means selection does not matter on that "
        "range. High-dispatch hours are themselves not guaranteed to be at full output (reserves, outages "
        "inside the hour), so neither estimator is a clean upper bound.")
    eia_tab, eia_units, eia_stats = eia_comparison(inp, full)
    ctx["eia_tab"], ctx["eia_stats"] = eia_tab, eia_stats
    et = eia_tab.set_index("group")
    inside = [g for g in H.GROUPS
              if et.at[GROUP_LABEL[g], "kappa EIA (linear to T_winter)"] - 1e-9
              <= et.at[GROUP_LABEL[g], "kappa CEMS (own fits)"]
              <= et.at[GROUP_LABEL[g], "kappa EIA (flat < 15 C)"] + 1e-9]
    higher = [g for g in H.GROUPS if et.at[GROUP_LABEL[g], "model P(T_summer)/P15 median"]
              > et.at[GROUP_LABEL[g], "EIA S/W median"]]
    vst = v.loc[GROUP_LABEL["ST"]]
    ctx["eia_inside"] = inside
    ctx["eia_text"] = (
        "Reading: by group, the median EIA summer/winter ratio and the model ratio at summer-peak "
        "temperatures are " + "; ".join(
            f"{g} {et.at[GROUP_LABEL[g], 'EIA S/W median']:.3f} vs "
            f"{et.at[GROUP_LABEL[g], 'model P(T_summer)/P15 median']:.3f}" for g in H.GROUPS)
        + (f". For {', '.join(higher)} the model ratio is higher (less seasonal loss), as expected if "
           "capacity keeps rising below 15 C, which the model does not include while winter ratings refer "
           "to colder peak conditions" if higher else "")
        + ". The CEMS median kappa lies inside the EIA-implied bracket (linear to T_winter, flat below "
        "15 C) for " + (", ".join(inside) if inside else "no group") + ". "
        + f"Steam units: {et.at[GROUP_LABEL['ST'], 'EIA S = W %']:.0f}% report the same summer and winter "
        f"capacity, and the raw CEMS envelope gives a median slope of {vst['raw envelope >= 15 C']:.3f} %/C "
        f"(high-dispatch hours {vst['high-dispatch hours >= 15 C']:.3f}); the primary (monotone) median of "
        f"{vst['monotone >= 15 C (primary)']:.3f} %/C is partly an artefact of raising noisy mild-temperature "
        "bins, so ST derating is best read as the range between these values. The weak unit-level "
        "correlations show that EIA seasonal ratings are coarse (often rounded or equal in both seasons) and "
        "that a single unit's kappa is noisy; the technology medians are the robust result.")

    # Fleet level.
    fs = build_fleet(inp, full)
    n_hot = int(round(HOT_SHARE * len(fs.hours)))
    order = np.argsort(fs.fleet_temp)[::-1]
    hot = np.zeros(len(fs.hours), bool)
    hot[order[:n_hot]] = True
    cl = capacity_loss(fs, hot, args.boot)
    lst = fs.hours - U.LST_OFFSET
    cl.update(n_hot=n_hot, t_hot=float(fs.fleet_temp[hot].mean()), t_max=float(fs.fleet_temp.max()),
              t_max_when=f"{lst[order[0]]:%Y-%m-%d %H:%M} LST",
              hot_first=f"{lst[hot].min():%Y-%m-%d}", hot_last=f"{lst[hot].max():%Y-%m-%d}")
    ctx["cap"] = cl
    cap_rows = [{"group": GROUP_LABEL[g], **cl["by_group"][g]} for g in H.GROUPS]
    cap_rows.append({"group": "all non-CHP", "P15 MW": cl["p15_total"], "lost MW": cl["lost_mean"],
                     "lost %": 100 * cl["lost_mean"] / cl["p15_total"], "mean T C": cl["t_hot"]})
    ctx["cap_tab"] = pd.DataFrame(cap_rows)
    ctx["avg"] = average_penalty(fs, hot, args.boot)
    ctx["avg_wd"] = average_penalty(replace(fs, table=table_variant(fs.table, "within-day")),
                                    hot, args.boot)
    print(f"fleet: {time.time() - t_start:.0f} s")
    mv = marginal_value(fs, hot, args.draws)
    ctx["marginal"] = mv
    mv_rows = [{"quantity (hot-hour mean)": "marginal heat rate at 15 C (MMBtu/MWh)",
                "MMBtu/MWh": mv["fd15"], "kg CO2/MWh": mv["co2_15"]},
               {"quantity (hot-hour mean)": "marginal heat rate at actual temperature",
                "MMBtu/MWh": mv["fd_hot"], "kg CO2/MWh": mv["co2_15"] + mv["co2"]}]
    for name, (dv, dc, _) in mv["diffs"].items():
        mv_rows.append({"quantity (hot-hour mean)": f"difference: {name}",
                        "MMBtu/MWh": dv, "kg CO2/MWh": dc})
    ctx["mv_tab"] = pd.DataFrame(mv_rows)
    print(f"marginal: {time.time() - t_start:.0f} s")

    # Hold-out.
    ho = holdout(inp.uh, train, args.boot)
    ctx["holdout"] = ho
    ha = ho["table"]
    a_all = ha[(ha["rows"] == "all clean hours") & (ha["group"] == "all")].iloc[0]
    improves = a_all["MAE temp %"] < a_all["MAE static %"]
    bt = ho["by_temp"]
    span0 = bt["bias static %"].max() - bt["bias static %"].min()
    spanT = bt["bias temp %"].max() - bt["bias temp %"].min()
    per_g = ha[(ha["rows"] == "all clean hours") & (ha["group"] != "all")]
    worse = [g for g in H.GROUPS
             if (per_g.loc[per_g["group"] == GROUP_LABEL[g], "median unit MAE change %"] > 0).any()]
    ctx["holdout_worse_text"] = (", ".join(f"{g} units" for g in worse) or "no group")
    worse_text = (f"For {', '.join(worse)} units the extra terms add noise (the median unit's MAE gets "
                  "worse). " if worse else "")
    ctx["holdout_text"] = (
        "Reading: the temperature terms "
        + ("reduce" if improves else "do not reduce")
        + f" the pooled unit-hour error ({a_all['MAE static %']:.2f}% -> {a_all['MAE temp %']:.2f}%), which "
        "is dominated by hour-to-hour scatter that temperature does not explain. "
        + ("What they do reduce is the " if spanT < span0 else "They do not reduce the ")
        + f"temperature-dependent bias: across the Sep-Dec temperature bands the static curve's bias spans "
        f"{span0:.2f} percentage points, the temperature curve's {spanT:.2f}. That is the part that matters "
        "for summing fuel over hot or cold hours"
        + (", and it shows up in the lower MAPE of the hourly sum. "
           if ho["fleet"]["MAPE temp %"] < ho["fleet"]["MAPE static %"]
           else "; it does not lower the MAPE of the hourly sum. ")
        + worse_text
        + "Sep-Dec spans hot September and cold December, so the check exercises the terms on both sides "
        "of the fit-window mean.")
    dh = dispatch_holdout(inp, train, band)
    q = dh.iloc[:5]
    dh.attrs["mae_static"] = weighted_mae(q, "static FD error %")
    dh.attrs["mae_temp"] = weighted_mae(q, "temperature FD error %")
    ctx["disp_hold"] = dh
    hot_row = dh.iloc[5]
    hs, ht = hot_row["static FD error %"], hot_row["temperature FD error %"]
    pct = mv["pct"]
    gt_band = abs(pct) > band["mae"]
    gt_max = abs(pct) > max(abs(band["lo"]), abs(band["hi"]))
    closer = abs(ht) < abs(hs)
    verdict = (f"The temperature part ({pct:+.1f}%) is {'larger' if gt_band else 'smaller'} than the error "
               f"band's weighted MAE ({band['mae']:.1f}%) and {'larger' if gt_max else 'not larger'} than its "
               f"largest quintile error ({max(abs(band['lo']), abs(band['hi'])):.1f}%). ")
    if closer:
        verdict += (f"Out of sample, in the hottest Sep-Dec hours, adding the temperature terms moves the "
                    f"model's derivative closer to the CEMS slope ({hs:+.1f}% -> {ht:+.1f}%), which supports "
                    "the model's temperature response at the margin.")
    else:
        verdict += (f"Out of sample, in the hottest Sep-Dec hours, the model already overstates the CEMS "
                    f"slope ({hs:+.1f}% without temperature terms) and adding them moves it further away "
                    f"({ht:+.1f}%). So the data do not confirm the marginal number: it is a model estimate "
                    "whose error is of the same size as the effect.")
    ctx["verdict"] = verdict
    ctx["value_text"] = (
        verdict + " The unit-level average does not go through the dispatch model; it rests only on the "
        "per-unit heat-input fits at observed outputs and temperatures (a fitted counterfactual, not a direct "
        "measurement). "
        "Both numbers are counterfactuals at equal output; the full fuel value of moving 1 MWh out of a hot "
        "hour also includes the load-level effect (hot hours are high-load hours, so the marginal unit is "
        "less efficient), which is the Q5-Q1 marginal heat-rate spread of the dispatch report.")
    ctx["disp_hold_text"] = (
        "Temperature terms "
        + ("lower" if dh.attrs["mae_temp"] < dh.attrs["mae_static"] else "do not lower")
        + " the dispatch model's out-of-sample marginal error over the quintiles. These static curves differ "
        "from the dispatch step's (steady hours only, Pmax = Pref), so the static row is not identical to "
        "section 1.")
    print(f"hold-out: {time.time() - t_start:.0f} s")

    # Figures.
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    env = envelope_curves(inp.uh, full, cfg)
    figure_capacity(env, ctx["kappa_tab"], eia_units, OUTPUTS / FIG_CAP)
    figure_heatrate(full, ho["by_temp"], OUTPUTS / FIG_HR)

    ctx["limitations"] = [
        "Dry-bulb airport temperature stands in for compressor-inlet conditions. Humidity is not used, so "
        "evaporative inlet cooling or chillers (which respond to wet-bulb temperature) are folded into each "
        "unit's average temperature response; units with inlet cooling will show a smaller slope.",
        "kappa describes the upper envelope of reported gross output. It mixes ambient physics with "
        "operating practice: duct firing in CC units can hold output up in hot weather at a higher heat rate, "
        "and units capped below their physical limit (interconnection, contracts, permits) show kappa near zero "
        "up to the temperature at which the physical limit falls below the cap.",
        "Selection bias in the envelope is reduced (monotone step, limited-month screen) but not removed; for "
        "a summer-peaking system it biases kappa toward zero. The high-dispatch check and EIA-860 bracket it.",
        "Below 15 C capacity is held at P15. The cold side is not estimated, so winter capacity is understated "
        "for units whose capability keeps rising below 15 C.",
        "The heat-input temperature terms are identified mainly from seasonal and diurnal temperature "
        "variation at similar output; slow changes such as degradation, compressor washing and maintenance "
        "can load on them. The within-day estimate (day fixed effects) removes these and is reported beside "
        "the main one.",
        "CC* units report combustion-turbine output only; their derating and heat-rate numbers are those of a "
        "gas turbine inside a combined cycle, and the plant-level effect (with the steam turbine) is not "
        "observed. CHP units are excluded from all results.",
        "Units without enough hours get group-median parameters (kappa for "
        + ", ".join(f"{cov_k_count} {g}" for g, cov_k_count in
                    full[full["kappa_method"] == "group"].groupby("group").size().items())
        + " units). Group-assigned heat terms are added to a static curve fitted at the unit's own mean "
        "temperature, which slightly double counts the temperature effect for those units (negligible given "
        "the size of d and e).",
        "The marginal value comes from the dispatch model (hourly priority list, no start-up costs, monthly "
        "availability, one gas price). It is a model estimate, not a measurement, and must be read against "
        "the model's own error (sections 1 and 6); the average (unit-level) number does not depend on dispatch.",
        "CEMS gross output is measured before auxiliary load; EIA capacity is net. Auxiliary load (fans, "
        "pumps, chillers) also rises in hot weather, so net derating is somewhat larger than the gross "
        "derating estimated here.",
        f"Capability-limited months (monthly 99th-percentile output below {cfg.limited_month_ratio:.0%} of the "
        f"unit's) are left out of both the envelope and the heat fit, for "
        f"{int(((full['limited_months'] != '') & (full['group'] != 'CHP')).sum())} non-CHP units. Such a unit's "
        "curves describe its normal operating mode only (e.g. Tenaska Gateway's CC units, which ran at about "
        "half output from May to October); left in, a seasonal mode change like that loads on the temperature "
        "terms. Its hours in those months are also left out of the unit-level hot-hour average (section 5.2), "
        "while the dispatch model still uses its normal-mode curve whenever it is available.",
        "One year (2024) and one region; the thresholds of the screens (70% limited-month ratio, 2 C bins, "
        ">= 30 hours, steady-hour rule) were set after looking at the 2024 data and were not tuned on another year.",
    ]
    ctx["runtime"] = time.time() - t_start
    out = OUTPUTS / REPORT
    write_report(ctx, out)
    DOCS_REPORTS.mkdir(parents=True, exist_ok=True)
    for name in (REPORT, FIG_CAP, FIG_HR):
        shutil.copyfile(OUTPUTS / name, DOCS_REPORTS / name)
    print(f"wrote {out} and docs/reports copy; {ctx['runtime']:.0f} s")


if __name__ == "__main__":
    main()
