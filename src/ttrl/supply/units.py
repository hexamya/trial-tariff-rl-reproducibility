"""Per-unit heat-input curves for the ERCOT gas fleet, fitted on EPA CEMS.

Unit model (gross output P in MW, heat input H in MMBtu per hour):

    H_i(P) = a_i + b_i P + c_i P^2     while the unit is on, Pmin_i <= P <= Pmax_i
    H_i    = 0                         while it is off

- Data: full operating hours (``operating_time_hours == 1``) with positive
  gross load and heat input. Partial hours are left out of the fit because
  their heat input is a total for the operating fraction of the hour and is
  dominated by start-up and shut-down. Hours whose heat rate is below
  ``FitConfig.min_heat_rate`` (5.5 MMBtu/MWh gross, about 62% HHV
  efficiency, which no gas unit reaches) are reporting errors and are also
  left out (e.g. plant 3612 reports 3.7-5.3 MMBtu/MWh in Jan-Apr 2024).
- Fit: robust least squares (Huber M-estimator, ``statsmodels`` RLM) of H on
  [1, P, P^2]. The quadratic is kept only if it is convex (c >= 0) and its
  incremental heat rate b + 2 c P is positive at Pmin; otherwise the unit
  gets a robust straight line (c = 0). Convexity makes equal-incremental
  dispatch well defined (see :mod:`ttrl.supply.dispatch`).
- Limits: Pmin and Pmax are low and high quantiles of the unit's full-hour
  gross load in the fit window (defaults 5th and 99th percentiles).
- Units with fewer than ``FitConfig.min_hours`` clean hours in the fit window
  get the pooled, capacity-normalised curve of their technology (non-CHP,
  non-suspect units only), scaled by their EIA-860 capacity.
- CO2 factor: sum of reported CO2 over sum of heat input (short tons per
  MMBtu) on the unit's fit-window hours; technology median if none.

Flags carried in the curve table:

- ``chp``: EIA-860 combined-heat-and-power unit. Its fuel also makes process
  steam, so H(P) is not an electric heat rate; dispatch treats CHP units as
  must-run at their actual output.
- ``suspect_cc``: non-CHP combined-cycle unit whose median hourly heat rate
  in the fit window is at or above ``FitConfig.suspect_cc_heat_rate``
  (8.5 MMBtu/MWh gross). Such units probably report the combustion turbine's
  output without the steam turbine's share in the EPA and EIA crosswalk.

Temperature hook: dispatch never reads the table directly. It asks a curve
model for :class:`CurveArrays` through ``model.arrays(temp_c)``, whose
fields broadcast to (hours, units). :class:`StaticCurves` ignores
``temp_c`` and returns (1, units) arrays; a temperature-dependent model only
needs to return (hours, units) arrays from the same method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm

UNIT_KEY = ["plant_id_epa", "emissions_unit_id_epa"]

# CEMS reports Texas units in local standard time (CST = UTC-6). Month and
# season labels follow that clock so that "2024" means the CEMS 2024 year.
LST_OFFSET = pd.Timedelta(hours=6)

SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "summer",
                   10: "fall", 11: "fall"}


# --------------------------------------------------------------------------
# Unit-hour table
# --------------------------------------------------------------------------
def make_unit_id(plant_id_epa: pd.Series, unit: pd.Series) -> pd.Series:
    """String unit identifier ``"<plant_id_epa>_<emissions_unit_id_epa>"``."""
    return plant_id_epa.astype(str) + "_" + unit.astype(str)


def time_labels(ts_utc: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Local-standard-time month, hour of day and season for UTC timestamps."""
    ts = pd.Series(pd.DatetimeIndex(ts_utc))
    lst = ts - LST_OFFSET
    month = lst.dt.month
    return pd.DataFrame({"ts_lst": lst, "month": month,
                         "hour_lst": lst.dt.hour,
                         "season": month.map(SEASON_OF_MONTH)})


def prepare_unit_hours(cems: pd.DataFrame) -> pd.DataFrame:
    """Operating unit-hours of a CEMS extract, with derived columns.

    Keeps rows with ``operating_time_hours > 0`` and adds ``unit_id``,
    ``energy_mwh`` (gross load times operating time), ``heat`` (MMBtu),
    ``co2`` (short tons, may be missing), the LST ``month`` and ``clean``
    (full hour, positive load and heat input: the rows used in fits).
    """
    op = cems.loc[cems["operating_time_hours"] > 0].copy()
    op["unit_id"] = make_unit_id(op["plant_id_epa"], op["emissions_unit_id_epa"])
    p = op["gross_load_mw"].fillna(0.0)
    op["energy_mwh"] = p * op["operating_time_hours"]
    op["heat"] = op["heat_content_mmbtu"]
    op["co2"] = op["co2_mass_tons"]
    op["month"] = (op["operating_datetime_utc"] - LST_OFFSET).dt.month
    op["clean"] = ((op["operating_time_hours"] >= 1) & (p > 0)
                   & (op["heat"] > 0))
    cols = UNIT_KEY + ["unit_id", "operating_datetime_utc", "month", "tech",
                       "chp", "operating_time_hours", "gross_load_mw",
                       "energy_mwh", "heat", "co2", "clean"]
    return op[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Curve fitting
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FitConfig:
    """Settings of :func:`fit_unit_curves`."""
    pmin_quantile: float = 0.05
    pmax_quantile: float = 0.99
    min_hours: int = 100            # clean hours needed for a unit's own fit
    huber_t: float = 1.345          # Huber tuning constant (statsmodels default)
    suspect_cc_heat_rate: float = 8.5   # MMBtu/MWh gross, see module docstring
    min_heat_rate: float = 5.5      # MMBtu/MWh gross; lower hourly values are data errors
    pooled_max_rows: int = 200_000  # subsample size for the pooled fallback fit
    seed: int = 0


def fit_rows(unit_hours: pd.DataFrame, cfg: FitConfig = FitConfig()) -> pd.Series:
    """Rows of :func:`prepare_unit_hours` used in curve fits.

    Clean full hours whose heat rate is at least ``cfg.min_heat_rate``
    (lower values are physically impossible for a gas unit).
    """
    hr = unit_hours["heat"] / unit_hours["gross_load_mw"].where(unit_hours["clean"])
    return unit_hours["clean"] & (hr >= cfg.min_heat_rate)


def _rlm(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Huber robust regression coefficients of y on the columns of x."""
    return sm.RLM(y, x, M=sm.robust.norms.HuberT(t=t)).fit().params


def fit_heat_curve(p: np.ndarray, h: np.ndarray, pmin: float, pmax: float,
                   huber_t: float = 1.345) -> dict:
    """Robust convex fit of heat input ``h`` on gross load ``p`` for one unit.

    Returns ``a, b, c`` (H = a + b P + c P^2), ``method`` ("quadratic" or
    "linear"), in-sample ``r2`` and ``rmse`` (MMBtu/h) and ``n``. Load is
    scaled by ``pmax`` inside the regression for numerical conditioning.
    """
    p = np.asarray(p, float)
    h = np.asarray(h, float)
    s = max(float(pmax), 1.0)
    x = p / s
    beta = _rlm(np.column_stack([np.ones_like(x), x, x * x]), h, huber_t)
    a, b, c = beta[0], beta[1] / s, beta[2] / s ** 2
    method = "quadratic"
    if c < 0 or b + 2 * c * pmin <= 0:
        beta = _rlm(np.column_stack([np.ones_like(x), x]), h, huber_t)
        a, b, c = beta[0], beta[1] / s, 0.0
        method = "linear"
    fitted = a + b * p + c * p * p
    resid = h - fitted
    ss_tot = float(((h - h.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return {"a": float(a), "b": float(b), "c": float(c), "method": method,
            "r2": r2, "rmse": float(np.sqrt((resid ** 2).mean())), "n": len(p)}


def allocated_capacity(units: pd.DataFrame, gens: pd.DataFrame) -> pd.DataFrame:
    """EIA-860 nameplate, summer and winter capacity allocated to CEMS units.

    Each generator's capacity is split evenly over the CEMS units that list it
    in ``generator_ids`` (same rule as ``ttrl.data.pudl``), then summed per unit.
    """
    links = units[UNIT_KEY + ["plant_id_eia", "generator_ids"]].copy()
    links["generator_id"] = links["generator_ids"].str.split(",")
    links = links.explode("generator_id")
    gk = ["plant_id_eia", "generator_id"]
    links["n_share"] = links.groupby(gk)["emissions_unit_id_epa"].transform("size")
    g = gens[gk + ["capacity_mw", "summer_capacity_mw", "winter_capacity_mw",
                   "minimum_load_mw"]].copy()
    g["plant_id_eia"] = g["plant_id_eia"].astype(links["plant_id_eia"].dtype)
    m = links.merge(g, on=gk, how="left")
    cap_cols = ["capacity_mw", "summer_capacity_mw", "winter_capacity_mw",
                "minimum_load_mw"]
    for col in cap_cols:
        m[col] = m[col] / m["n_share"]
    out = m.groupby(UNIT_KEY, as_index=False)[cap_cols].sum(min_count=1)
    return out.rename(columns={"capacity_mw": "nameplate_mw",
                               "summer_capacity_mw": "summer_mw",
                               "winter_capacity_mw": "winter_mw",
                               "minimum_load_mw": "eia_min_load_mw"})


def _pooled_curve(hours: pd.DataFrame, table: pd.DataFrame,
                  cfg: FitConfig) -> pd.DataFrame:
    """Capacity-normalised Huber fit per technology for fallback units.

    Uses clean hours of units with their own fit that are neither CHP nor
    suspect CC: y = H / Pmax on x = P / Pmax. Returns one row per tech with
    ``alpha, beta, gamma`` (y = alpha + beta x + gamma x^2) and the median
    ratios ``pmax_over_nameplate`` and ``pmin_over_pmax`` of those units.
    """
    ok = table[(table["fit_method"] != "pooled") & ~table["chp"] & ~table["suspect_cc"]]
    rows = []
    rng = np.random.default_rng(cfg.seed)
    for tech, t in ok.groupby("tech"):
        d = hours.merge(t[["unit_id", "pmax"]], on="unit_id")
        if len(d) > cfg.pooled_max_rows:
            d = d.iloc[rng.choice(len(d), cfg.pooled_max_rows, replace=False)]
        x = d["gross_load_mw"].to_numpy() / d["pmax"].to_numpy()
        y = d["heat"].to_numpy() / d["pmax"].to_numpy()
        pmin_ratio = float((t["pmin"] / t["pmax"]).median())
        fit = fit_heat_curve(x, y, pmin_ratio, 1.0, cfg.huber_t)
        rows.append({"tech": tech, "alpha": fit["a"], "beta": fit["b"],
                     "gamma": fit["c"], "pmin_over_pmax": pmin_ratio,
                     "pmax_over_nameplate": float((t["pmax"] / t["nameplate_mw"]).median())})
    return pd.DataFrame(rows)


def fit_unit_curves(unit_hours: pd.DataFrame, units: pd.DataFrame,
                    gens: pd.DataFrame, cfg: FitConfig = FitConfig(),
                    unit_ids: list[str] | None = None) -> pd.DataFrame:
    """Fit one heat-input curve per CEMS unit.

    Parameters
    ----------
    unit_hours : output of :func:`prepare_unit_hours`, already restricted to
        the fit window (e.g. January to August).
    units : ``data/interim/ercot_gas_units_<year>.parquet``.
    gens : ``data/interim/ercot_gas_generators_<year>.parquet``.
    unit_ids : units that need a curve (default: those in ``unit_hours``).
        Units without enough clean hours in ``unit_hours`` get the pooled
        technology curve, so a curve exists for every unit that may run in
        a later validation window.

    Returns
    -------
    One row per unit in ``unit_ids`` with columns
    ``unit_id, plant_id_epa, emissions_unit_id_epa, plant_name_eia, tech,
    chp, suspect_cc, a, b, c, pmin, pmax, fit_method (quadratic | linear |
    pooled), n_fit, r2, rmse, median_hr, co2_per_mmbtu`` and the allocated
    EIA-860 capacities ``nameplate_mw, summer_mw, winter_mw``.
    """
    hours = unit_hours.loc[fit_rows(unit_hours, cfg)]
    meta = units[UNIT_KEY + ["plant_name_eia", "tech", "chp"]].copy()
    meta["unit_id"] = make_unit_id(meta["plant_id_epa"], meta["emissions_unit_id_epa"])
    meta["chp"] = meta["chp"].fillna(False).astype(bool)
    meta = meta.merge(allocated_capacity(units, gens), on=UNIT_KEY, how="left")
    wanted = unit_hours["unit_id"].unique() if unit_ids is None else unit_ids
    meta = meta[meta["unit_id"].isin(list(wanted))]

    rows = []
    for uid, d in hours.groupby("unit_id"):
        p = d["gross_load_mw"].to_numpy()
        h = d["heat"].to_numpy()
        pmin = float(np.quantile(p, cfg.pmin_quantile))
        pmax = float(np.quantile(p, cfg.pmax_quantile))
        row = {"unit_id": uid, "n_fit": len(d), "pmin": pmin, "pmax": pmax,
               "median_hr": float(np.median(h / p))}
        if len(d) >= cfg.min_hours and pmax > pmin:
            fit = fit_heat_curve(p, h, pmin, pmax, cfg.huber_t)
            row.update(a=fit["a"], b=fit["b"], c=fit["c"], fit_method=fit["method"],
                       r2=fit["r2"], rmse=fit["rmse"])
        rows.append(row)
    fits = pd.DataFrame(rows, columns=["unit_id", "n_fit", "pmin", "pmax",
                                       "median_hr", "a", "b", "c", "fit_method",
                                       "r2", "rmse"])
    table = meta.merge(fits, on="unit_id", how="left")
    table["n_fit"] = table["n_fit"].fillna(0).astype(int)
    table["suspect_cc"] = ((table["tech"] == "CC") & ~table["chp"]
                           & (table["median_hr"] >= cfg.suspect_cc_heat_rate))
    table["fit_method"] = table["fit_method"].fillna("pooled")

    # Pooled fallback for units with too few clean hours.
    pooled = _pooled_curve(hours, table, cfg).set_index("tech")
    fb = table["fit_method"] == "pooled"
    for i in table.index[fb]:
        q = pooled.loc[table.at[i, "tech"]]
        pmax = q["pmax_over_nameplate"] * table.at[i, "nameplate_mw"]
        table.loc[i, ["pmax", "pmin", "a", "b", "c"]] = [
            pmax, q["pmin_over_pmax"] * pmax, q["alpha"] * pmax, q["beta"],
            q["gamma"] / pmax]

    # CO2 per MMBtu from the fit-window hours that report both.
    both = unit_hours[(unit_hours["heat"] > 0) & unit_hours["co2"].notna()]
    f = both.groupby("unit_id")[["co2", "heat"]].sum()
    table["co2_per_mmbtu"] = table["unit_id"].map(f["co2"] / f["heat"])
    tech_med = table.groupby("tech")["co2_per_mmbtu"].transform("median")
    table["co2_per_mmbtu"] = table["co2_per_mmbtu"].fillna(tech_med)

    cols = ["unit_id", *UNIT_KEY, "plant_name_eia", "tech", "chp", "suspect_cc",
            "a", "b", "c", "pmin", "pmax", "fit_method", "n_fit", "r2", "rmse",
            "median_hr", "co2_per_mmbtu", "nameplate_mw", "summer_mw", "winter_mw",
            "eia_min_load_mw"]
    return table[cols].sort_values("unit_id", ignore_index=True)


# --------------------------------------------------------------------------
# Curve interface used by dispatch
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CurveArrays:
    """Curve parameters that broadcast to (hours, units).

    ``a, b, c`` give H = a + b P + c P^2 (MMBtu/h, P in MW gross);
    ``pmin, pmax`` are the output limits of a committed unit (MW).
    """
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    pmin: np.ndarray
    pmax: np.ndarray

    def heat(self, p: np.ndarray) -> np.ndarray:
        """Heat input (MMBtu/h) at output ``p``; zero where ``p <= 0`` (off)."""
        return np.where(p > 0, self.a + self.b * p + self.c * p * p, 0.0)

    def incremental(self, p: np.ndarray) -> np.ndarray:
        """Incremental heat rate dH/dP (MMBtu/MWh) at output ``p``."""
        return self.b + 2.0 * self.c * p

    def full_load_heat_rate(self) -> np.ndarray:
        """Average heat rate at Pmax (MMBtu/MWh), used for the priority list."""
        return (self.a + self.b * self.pmax + self.c * self.pmax ** 2) / self.pmax

    def broadcast(self, n_hours: int) -> "CurveArrays":
        """Copy with every field expanded to shape (n_hours, units)."""
        def f(x):
            x = np.atleast_2d(x)
            return np.broadcast_to(x, (n_hours, x.shape[-1]))
        return CurveArrays(*(f(x) for x in (self.a, self.b, self.c, self.pmin, self.pmax)))


class CurveModel(Protocol):
    """What dispatch needs from a unit-curve model."""
    unit_ids: list[str]

    def arrays(self, temp_c: np.ndarray | None = None) -> CurveArrays:
        """Curve parameters, broadcastable to (hours, units).

        ``temp_c`` (hours, units) is ambient temperature at each unit; a
        temperature-free model ignores it.
        """
        ...


class StaticCurves:
    """Temperature-free curve model built from a :func:`fit_unit_curves` table."""

    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True)
        self.unit_ids: list[str] = self.table["unit_id"].tolist()

    def arrays(self, temp_c: np.ndarray | None = None) -> CurveArrays:
        """(1, units) arrays; ``temp_c`` is accepted and ignored."""
        t = self.table
        return CurveArrays(*(t[c].to_numpy(float)[None, :]
                             for c in ("a", "b", "c", "pmin", "pmax")))

    def subset(self, unit_ids: list[str]) -> "StaticCurves":
        """Model restricted to ``unit_ids`` (in that order)."""
        t = self.table.set_index("unit_id").loc[list(unit_ids)].reset_index()
        return StaticCurves(t)

    @property
    def co2_per_mmbtu(self) -> np.ndarray:
        """CO2 factor per unit (short tons per MMBtu), shape (units,)."""
        return self.table["co2_per_mmbtu"].to_numpy(float)
