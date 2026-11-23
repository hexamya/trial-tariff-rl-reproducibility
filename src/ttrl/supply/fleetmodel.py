"""Temperature-aware dispatch of the ERCOT gas fleet (stage S4 integration).

Joins the three earlier pieces into one model that a validation script or the
tariff environment can run:

- :mod:`ttrl.supply.units`: static heat-input curves H(P) fitted on CEMS;
- :mod:`ttrl.supply.heatrate`: H(P, T) = a + bP + cP^2 + d(T - 15) + e(T - 15)P
  and available capacity Pavail(T) = P15 (1 - kappa/100 max(T - 15, 0));
- :mod:`ttrl.supply.dispatch`: priority-list commitment with a headroom rule
  and equal-incremental-heat-rate dispatch, hour by hour.

Pieces:

- :class:`FleetData`: actual 2024 operation of the non-CHP gas units as
  (hours, units) arrays, with the hour-beginning plant temperature of every
  unit (:func:`build_fleet_data`).
- :func:`complete_curve_table`: one temperature curve per non-CHP unit from a
  :func:`ttrl.supply.heatrate.fit_temperature_curves` table; units without a
  usable fit get their static curve from :func:`ttrl.supply.units.fit_unit_curves`
  (possibly the pooled technology curve) plus the group-median temperature
  terms and kappa.
- :func:`static_table`: the temperature-free version of such a table (the
  same units' curves fitted on the same hours without temperature terms,
  Pmax = 99th percentile of output). It is the comparison model.
- :class:`FleetModel`: dispatch with a ``mode`` ("static", "temperature",
  or "heat" / "derate" for one effect only) and a ``config`` ("C": every
  non-CHP unit dispatched; "C-sx": suspect CC units, which probably report
  combustion-turbine output only, held at their actual output). CHP units
  are never in the model: their fuel also makes process steam, so they are
  exogenous everywhere in S4.

Units: output P in MW gross (= MWh in an hour), heat in MMBtu, CO2 in CEMS
short tons, temperatures in C. Hours are tz-naive UTC, hour-beginning; month,
season and day labels use local standard time (UTC-6), as in CEMS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttrl.supply import dispatch as D
from ttrl.supply import heatrate as H
from ttrl.supply.fleet import build_fleet_hours
from ttrl.supply.units import LST_OFFSET, make_unit_id, time_labels

MODES = {"static": (False, False), "temperature": (True, True),
         "heat": (True, False), "derate": (False, True)}
CONFIGS = {"C": "priority list + headroom, every non-CHP unit dispatched",
           "C-sx": "as C, suspect CC units held at their actual output"}
TABLE_COLS = ["unit_id", "tech", "group", "suspect_cc", "const_heat", "a", "b", "c", "d", "e",
              "pmin", "pmax", "p15", "kappa", "a0", "b0", "c0", "t_fit_lo", "t_fit_hi",
              "hr_method", "kappa_method", "nameplate_mw"]


# --------------------------------------------------------------------------
# Actual operation and plant temperature
# --------------------------------------------------------------------------
@dataclass
class FleetData:
    """Actual hourly operation of the N non-CHP units over the T hours of a CEMS year.

    (T, N) arrays: ``energy`` gross MWh, ``heat`` MMBtu, ``co2`` short tons
    (gaps where CEMS reports heat but no CO2 filled with the unit's full-year
    CO2/heat ratio), ``on`` (produced), ``avail`` (monthly availability:
    CEMS row present and the unit produced in that local month), ``temp``
    (plant temperature in C, mean of the observations at the start and end of
    the hour). (T,) labels: ``month``, ``season``, ``hour_lst`` and ``day``
    (local-standard-time day, days since 1970-01-01). (N,) ``plant_id_eia``,
    ``tech`` (CC/CT/ST from EIA-860).
    """
    year: int
    hours: pd.DatetimeIndex
    unit_ids: list[str]
    plant_id_eia: np.ndarray
    tech: np.ndarray
    energy: np.ndarray
    heat: np.ndarray
    co2: np.ndarray
    on: np.ndarray
    avail: np.ndarray
    temp: np.ndarray
    month: np.ndarray
    season: np.ndarray
    hour_lst: np.ndarray
    day: np.ndarray

    @property
    def target(self) -> np.ndarray:
        """Actual non-CHP gross output per hour (MWh), the dispatch target."""
        return self.energy.sum(axis=1)

    def fleet_temp(self, weights: np.ndarray, temp: np.ndarray | None = None) -> np.ndarray:
        """Weighted mean plant temperature per hour (weights per unit, e.g. P15)."""
        t = self.temp if temp is None else temp
        w = np.asarray(weights, float)
        return (t * w).sum(axis=1) / w.sum()


def plant_temperature_matrix(temp: pd.DataFrame, hours: pd.DatetimeIndex,
                             plant_ids: np.ndarray) -> np.ndarray:
    """(hours, units) hour-beginning plant temperature.

    ``temp`` is ``data/interim/plant_temperature_<year>.parquet`` (``plant_id_eia,
    ts_utc, temp_c``). The value for the hour starting at ts is the mean of
    the observations at ts and ts + 1 h (the definition of
    :func:`ttrl.supply.heatrate.add_temperature`); NaN if either is missing.
    """
    t = temp[["plant_id_eia", "ts_utc", "temp_c"]].sort_values(["plant_id_eia", "ts_utc"])
    g = t.groupby("plant_id_eia", sort=False)
    nxt = g["temp_c"].shift(-1)
    gap = g["ts_utc"].shift(-1) - t["ts_utc"]
    t = t.assign(t_hb=((t["temp_c"] + nxt) / 2).where(gap == pd.Timedelta(hours=1)))
    piv = t.pivot(index="ts_utc", columns="plant_id_eia", values="t_hb").reindex(hours)
    piv.columns = piv.columns.astype("int64")
    return piv.reindex(columns=np.asarray(plant_ids, "int64")).to_numpy(float)


def non_chp_unit_ids(cems: pd.DataFrame, units: pd.DataFrame) -> list[str]:
    """Sorted ids of units with at least one operating hour, CHP units removed."""
    op = cems[cems["operating_time_hours"] > 0]
    ids = set(make_unit_id(op["plant_id_epa"], op["emissions_unit_id_epa"]))
    u = units.assign(unit_id=make_unit_id(units["plant_id_epa"], units["emissions_unit_id_epa"]))
    chp = set(u.loc[u["chp"].fillna(False).astype(bool), "unit_id"])
    return sorted(ids - chp)


def build_fleet_data(cems: pd.DataFrame, units: pd.DataFrame, temp: pd.DataFrame,
                     year: int) -> FleetData:
    """:class:`FleetData` for the non-CHP units of a CEMS extract.

    ``cems`` and ``units`` are the interim files of
    :func:`ttrl.data.pudl.build_ercot_gas_cems`, ``temp`` the plant
    temperature file of ``scripts/build_plant_weather.py``.
    """
    ids = non_chp_unit_ids(cems, units)
    fh = build_fleet_hours(cems, ids, year)
    both = (fh.heat > 0) & ~np.isnan(fh.co2)
    num = np.where(both, fh.co2, 0.0).sum(axis=0)
    den = np.where(both, fh.heat, 0.0).sum(axis=0)
    fac = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    fac = np.where(np.isnan(fac), np.nanmedian(fac), fac)
    u = units.assign(unit_id=make_unit_id(units["plant_id_epa"], units["emissions_unit_id_epa"]))
    u = u.drop_duplicates("unit_id").set_index("unit_id").loc[ids]
    plant = u["plant_id_eia"].astype("int64").to_numpy()
    lab = time_labels(fh.hours)
    return FleetData(year=year, hours=fh.hours, unit_ids=ids, plant_id_eia=plant,
                     tech=u["tech"].astype(str).to_numpy(), energy=fh.energy, heat=fh.heat,
                     co2=fh.co2_filled(fac), on=fh.on, avail=fh.availability("month"),
                     temp=plant_temperature_matrix(temp, fh.hours, plant),
                     month=lab["month"].to_numpy(), season=lab["season"].to_numpy(),
                     hour_lst=lab["hour_lst"].to_numpy(), day=lst_day(fh.hours))


def lst_day(hours_utc: pd.DatetimeIndex) -> np.ndarray:
    """Local-standard-time calendar day of UTC hours, as days since 1970-01-01."""
    d = (pd.DatetimeIndex(hours_utc) - LST_OFFSET).normalize() - pd.Timestamp("1970-01-01")
    return np.asarray(d // pd.Timedelta(days=1), dtype=np.int64)


# --------------------------------------------------------------------------
# Curve tables
# --------------------------------------------------------------------------
def complete_curve_table(temp_table: pd.DataFrame, static_table: pd.DataFrame,
                         unit_ids: list[str]) -> pd.DataFrame:
    """One temperature curve per unit of ``unit_ids``, in that order.

    ``temp_table``: :func:`ttrl.supply.heatrate.fit_temperature_curves` output
    for a fit window. ``static_table``: :func:`ttrl.supply.units.fit_unit_curves`
    output for the same window with ``unit_ids`` requested, so it has a curve
    (own or pooled) for every unit. Units whose temperature fit has no curve
    (fewer than ``FitConfig.min_hours`` clean hours) take the static curve,
    the group median of d/Pmax and e of own heat fits, the group median kappa
    of own envelope fits and P15 = Pmax (``source`` = "static fallback").
    ``co2_per_mmbtu`` always comes from ``static_table`` (CO2/heat of the
    window's hours). Columns: :data:`TABLE_COLS` + ``co2_per_mmbtu, source``.
    """
    tt = temp_table.set_index("unit_id")
    st = static_table.set_index("unit_id")
    const = tt["const_heat"].fillna(False).astype(bool)
    own = tt["hr_method"].isin(["quadratic", "linear"]) & ~const & (tt["group"] != "CHP")
    med = (tt[own].assign(d_rel=tt.loc[own, "d"] / tt.loc[own, "pmax"])
           .groupby("group")[["d_rel", "e"]].median())
    kmed = tt[(tt["kappa_method"] == "own") & (tt["group"] != "CHP")].groupby("group")["kappa"].median()
    rows = []
    for u in unit_ids:
        if u in tt.index and pd.notna(tt.at[u, "a"]):
            r = {c: tt.at[u, c] for c in TABLE_COLS if c != "unit_id"}
            r["source"] = "temperature fit"
        else:
            s = st.loc[u]
            g = "CC*" if bool(s["suspect_cc"]) else str(s["tech"])
            pmax = float(s["pmax"])
            r = {"tech": s["tech"], "group": g, "suspect_cc": bool(s["suspect_cc"]),
                 "const_heat": False, "a": s["a"], "b": s["b"], "c": s["c"],
                 "d": med.at[g, "d_rel"] * pmax if g in med.index else 0.0,
                 "e": med.at[g, "e"] if g in med.index else 0.0,
                 "pmin": s["pmin"], "pmax": pmax, "p15": pmax,
                 "kappa": float(kmed.get(g, 0.0)), "a0": s["a"], "b0": s["b"], "c0": s["c"],
                 "t_fit_lo": np.nan, "t_fit_hi": np.nan,
                 "hr_method": "group (static curve)", "kappa_method": "group",
                 "nameplate_mw": s["nameplate_mw"], "source": "static fallback"}
        r["unit_id"] = u
        rows.append(r)
    out = pd.DataFrame(rows)[TABLE_COLS + ["source"]]
    out["co2_per_mmbtu"] = st.loc[unit_ids, "co2_per_mmbtu"].to_numpy(float)
    out["suspect_cc"] = out["suspect_cc"].astype(bool)
    out["const_heat"] = out["const_heat"].fillna(False).astype(bool)
    return out


def static_table(table: pd.DataFrame) -> pd.DataFrame:
    """Temperature-free version of a :func:`complete_curve_table` table.

    H = a0 + b0 P + c0 P^2 (fitted on the same hours without temperature
    terms), Pmax = ``pmax`` (99th percentile of output) in every hour.
    """
    return table.assign(a=table["a0"].fillna(table["a"]), b=table["b0"].fillna(table["b"]),
                        c=table["c0"].fillna(table["c"]), p15=table["pmax"],
                        d=0.0, e=0.0, kappa=0.0)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class FleetRun:
    """One dispatch of the model's dispatched units over T hours.

    ``p`` (T, Nd) MW; ``heat`` MMBtu, ``co2`` short tons, ``n_ct`` simple-cycle
    CT units producing, ``status`` (codes of :mod:`ttrl.supply.dispatch`), all (T,).
    """
    p: np.ndarray
    heat: np.ndarray
    co2: np.ndarray
    n_ct: np.ndarray
    status: np.ndarray


class FleetModel:
    """Commitment and dispatch of the non-CHP gas fleet, with or without temperature.

    ``table`` is a :func:`complete_curve_table` table whose rows follow the
    columns of the (T, N) arrays passed to the methods (all non-CHP units).
    ``mode``: "static" (:func:`static_table`), "temperature" (H(P, T) and
    Pavail(T)), "heat" (H(P, T), Pmax = P15) or "derate" (15 C curve,
    Pavail(T)); see :class:`ttrl.supply.heatrate.TemperatureCurves` for how
    temperatures enter (heat terms not extrapolated beyond each unit's fitted
    range, derating linear above 15 C). ``config``: "C" or "C-sx" (see
    :data:`CONFIGS`). ``headroom`` = (r0 MW, r1): committed capacity covers
    target + r0 + r1 x target; set it with :meth:`fit_headroom`.
    """

    def __init__(self, table: pd.DataFrame, mode: str = "temperature", config: str = "C-sx",
                 headroom: tuple[float, float] = (0.0, 0.0)):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {list(CONFIGS)}")
        self.table = table.reset_index(drop=True)
        self.mode, self.config = mode, config
        tab = static_table(self.table) if mode == "static" else self.table
        use_heat, use_derate = MODES[mode]
        sus = self.table["suspect_cc"].astype(bool).to_numpy()
        self.disp = ~sus if config == "C-sx" else np.ones(len(self.table), bool)
        self.curves = H.TemperatureCurves(tab.loc[self.disp].reset_index(drop=True),
                                          use_heat=use_heat, use_derate=use_derate)
        self.co2f = self.curves.co2_per_mmbtu
        self.is_ct = (tab["tech"].astype(str).to_numpy() == "CT")[self.disp]
        self.headroom = (float(headroom[0]), float(headroom[1]))

    # -- inputs ------------------------------------------------------------
    def arrays(self, temp: np.ndarray):
        """Curve arrays of the dispatched units at plant temperatures ``temp`` (T, N)."""
        return self.curves.arrays(np.asarray(temp, float)[:, self.disp])

    def split(self, energy: np.ndarray, heat: np.ndarray | None = None,
              co2: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Dispatched target and the exogenous (held) units' energy, heat and CO2 per hour."""
        out = {"target": energy[:, self.disp].sum(axis=1),
               "exo_energy": energy[:, ~self.disp].sum(axis=1)}
        if heat is not None:
            out["exo_heat"] = heat[:, ~self.disp].sum(axis=1)
        if co2 is not None:
            out["exo_co2"] = co2[:, ~self.disp].sum(axis=1)
        return out

    def fit_headroom(self, energy: np.ndarray, on: np.ndarray, temp: np.ndarray) -> tuple[float, float]:
        """Fit the headroom rule on actual hours (rows of the (T, N) arrays given).

        OLS of (summed model Pmax of the dispatched units that actually ran)
        minus their output on that output, with Pmax at the hour's
        temperatures for temperature-aware modes.
        """
        cv = self.arrays(temp)
        target = energy[:, self.disp].sum(axis=1)
        pmax = np.broadcast_to(cv.pmax, (len(target), int(self.disp.sum())))
        committed = (on[:, self.disp] * pmax).sum(axis=1)
        self.headroom = D.fit_headroom(target, committed)
        return self.headroom

    # -- runs --------------------------------------------------------------
    def _run(self, cv, target: np.ndarray, avail: np.ndarray) -> FleetRun:
        hr = self.headroom[0] + self.headroom[1] * target
        res = D.dispatch(target, cv, avail, headroom_mw=hr)
        uh = res.unit_heat(cv)
        return FleetRun(p=res.p, heat=uh.sum(axis=1), co2=(uh * self.co2f).sum(axis=1),
                        n_ct=(res.p[:, self.is_ct] > 0).sum(axis=1), status=res.status)

    def run(self, target: np.ndarray, temp: np.ndarray, avail: np.ndarray) -> FleetRun:
        """Dispatch the dispatched units to ``target`` (T,) MW.

        ``temp`` and ``avail`` are (T, N) over all non-CHP units (the
        exogenous columns are ignored).
        """
        return self._run(self.arrays(temp), np.asarray(target, float),
                         np.asarray(avail, bool)[:, self.disp])

    def marginal(self, target: np.ndarray, temp: np.ndarray, avail: np.ndarray,
                 delta: float = 500.0) -> dict[str, np.ndarray]:
        """Central-difference marginal response per MW of target, commitment re-solved.

        Returns (T,) arrays ``fuel`` (MMBtu/MWh), ``co2`` (short tons/MWh),
        ``ct`` (CT units on per MW) from runs at target +/- ``delta``; NaN
        where either run is not solved to equal lambda (status not ok).
        """
        cv = self.arrays(temp)
        av = np.asarray(avail, bool)[:, self.disp]
        target = np.asarray(target, float)
        up = self._run(cv, target + delta, av)
        dn = self._run(cv, target - delta, av)
        ok = (up.status == D.STATUS_OK) & (dn.status == D.STATUS_OK)
        f = lambda a, b: np.where(ok, (a - b) / (2.0 * delta), np.nan)  # noqa: E731
        return {"fuel": f(up.heat, dn.heat), "co2": f(up.co2, dn.co2),
                "ct": f(up.n_ct.astype(float), dn.n_ct.astype(float))}


def station_distance(plant_ids, year: int) -> dict[str, float]:
    """Distance (km) from plants to their primary ISD station: median and max over ``plant_ids``.

    Reads ``data/interim/plant_weather_stations_<year>.parquet`` written by
    ``scripts/build_plant_weather.py``.
    """
    from ttrl.paths import INTERIM
    st = pd.read_parquet(INTERIM / f"plant_weather_stations_{year}.parquet",
                         columns=["plant_id_eia", "distance_km"])
    d = st[st["plant_id_eia"].astype("int64").isin(np.asarray(plant_ids, "int64"))]["distance_km"]
    return {"median": float(d.median()), "max": float(d.max()), "plants": int(len(d))}
