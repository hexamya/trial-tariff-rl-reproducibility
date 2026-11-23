"""ERCOT case of the S2 environment: hourly inputs per weather year and the training surrogate.

Inputs for one year (``build_year``): UTC hours of the local (US Central) year; EIA-930 demand, wind
and solar (net load = demand - wind - solar) and ERCOT's day-ahead demand forecast as published in
EIA-930 (what the agent sees when it posts the next day's tariff);
residential MWh (``scripts/build_ercot_residential.py``); the non-CHP gas output the S4 model
starts from (``base_mw``) and plant temperatures (2024 fleet). Training years are scaled to the
2023 annual demand. The scale applies to demand and
residential load, not to gas output, which keeps each year's own renewable and gas pattern.

Gas base for years other than 2024: EIA-930 gas net generation of the year (``gas_generation``, with
the misreported hours of 2019-2020 corrected) times the 2024 ratio of S4 non-CHP gross output to
EIA-930 gas generation, by local-standard-time month and hour.

Training surrogate (``Surrogate``): fuel and CO2 change for a load change x_t (MWh) in hour t,

    f_t(x) = c_t * x + b2(|x|) * q_t * x^2 / 2

the smooth counterpart of ``ErrorBand.correct`` at the band's point estimates. c_t is the corrected
+/- 500 MW marginal (hour-level part),

    c_t = beta(g_t) * m_t + (1 - beta(g_t)) * m_bar(s_t) - err(s_t)

(m_t model marginal, s_t stratum, g_t load tercile); q_t the curvature of the model's response, fitted
per hour to its even part at +/- 1, 2 and 4 GW, F(+s) + F(-s) = q s^2 (steps truncated at zero output
or short of capacity left out), floored at zero; it stands for the step-size part of the model's answer
(F(x) - m_t x), which the band scales by the share beta2 that CEMS confirms for changes of that size.
b2 interpolates beta2 linearly between the size classes' midpoints (``CURV_X``; flat outside), so f is
smooth in x and an agent cannot exploit the dispatch model's lumpy response to step size. Without the
curvature the linear form overstated the fuel saving of multi-GW summer reductions by several percent
against ``S4Model.impact`` (``docs/reports/ercot_env_summary.md``). Final evaluation uses
``S4Model.impact`` with the band.

Tariff days run 02:00-02:00 local time; ``hour_key`` maps every UTC hour to (tariff day, hour of
the tariff day), and the kernel's half-hourly effect is averaged to hours. On the autumn change night
two UTC hours share a key; on the spring night one key has no hour.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ttrl.data import eia930
from ttrl.paths import INTERIM, PROCESSED

TZ = "America/Chicago"
LST_OFFSET = pd.Timedelta(hours=6)
FLEET_YEAR = 2024
SCALE_TO_YEAR = 2023
DELTA_MW = 500.0
CURVATURE_STEPS_MW = (1000.0, 2000.0, 4000.0)
CURV_X = (250.0, 750.0, 1500.0, 3000.0)     # MW: midpoints of the band's size classes (3 GW for >= 2 GW)
DAY_START_HOUR = 2
SHED_GAP, SHED_HOURS = 0.10, 4
MISREPORTED_OTHER_MW, MISREPORTED_BEFORE = 2000.0, 2023


def local_year_hours(year: int) -> pd.DatetimeIndex:
    """UTC hour-beginning timestamps (naive) of the local Central-time calendar year."""
    start = pd.Timestamp(f"{year}-01-01", tz=TZ).tz_convert("UTC").tz_localize(None)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=TZ).tz_convert("UTC").tz_localize(None)
    return pd.date_range(start, end, freq="h", inclusive="left")


def hour_key(hours_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Tariff day (local date of its 02:00 start) and hour of the tariff day (0-23) per UTC hour."""
    local = hours_utc.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    shifted = local - pd.Timedelta(hours=DAY_START_HOUR)
    return pd.DataFrame({"tday": shifted.normalize(), "hour": shifted.hour}, index=hours_utc)


def gas_generation(year: int) -> tuple[pd.Series, int]:
    """EIA-930 ERCOT gas net generation by UTC hour, and the number of hours corrected. Before 2023,
    when ERCOT batteries never exceeded 0.7 GW in EIA-930, hours with more than 2 GW under
    'storage_other' are gas reported in the wrong category (December 2019 and January-February 2020:
    gas drops to about 0.5 GW while 'storage_other' jumps to 9-27 GW); there it is added back to gas."""
    df = eia930.load_ba_hourly(year).set_index("datetime_utc")
    gas = df["gas"].copy()
    bad = (df["storage_other"] > MISREPORTED_OTHER_MW) if year < MISREPORTED_BEFORE else pd.Series(False, df.index)
    gas[bad] += df.loc[bad, "storage_other"]
    return gas, int(bad.sum())


def curtailed_days(year: int) -> pd.DataFrame:
    """Local days of ``year`` with at least SHED_HOURS hours of reported demand more than SHED_GAP below
    the EIA-930 day-ahead forecast (the curtailed-demand rule, fixed before any RL training; listed in
    docs/reports/ercot_weather_years.md). Columns: date, hours, max_gap."""
    fc = eia930.load_ba_forecast(year)
    local = fc["datetime_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)
    fc = fc[local.dt.year == year].assign(date=local[local.dt.year == year].dt.normalize())
    fc["gap"] = (fc["demand_forecast"] - fc["demand_reported"]) / fc["demand_forecast"]
    d = fc.groupby("date").agg(hours=("gap", lambda g: int((g > SHED_GAP).sum())), max_gap=("gap", "max"))
    return d[d["hours"] >= SHED_HOURS].reset_index()


@dataclass
class ErcotYear:
    year: int
    hours: pd.DatetimeIndex     # UTC, naive
    demand: np.ndarray          # MWh (scaled for training years)
    residential: np.ndarray     # MWh (scaled likewise)
    base_mw: np.ndarray         # non-CHP gas gross output the S4 model starts from
    temp_path: Path             # plant temperatures (2024 fleet)
    scale: float                # demand scale applied (1 for 2023 and 2024)
    curtailed: np.ndarray       # bool per hour: day flagged as curtailed demand
    renewables: np.ndarray | None = None        # EIA-930 wind + solar MWh (not scaled); None: zero
    demand_forecast: np.ndarray | None = None   # EIA-930 day-ahead demand forecast (scaled like demand)

    def __post_init__(self):
        if self.renewables is None:
            self.renewables = np.zeros(len(self.hours))
        if self.demand_forecast is None:
            self.demand_forecast = np.asarray(self.demand, float).copy()

    @property
    def net_load(self) -> np.ndarray:
        """Demand minus wind and solar (MWh)."""
        return self.demand - self.renewables

    @property
    def key(self) -> pd.DataFrame:
        return hour_key(self.hours)


def gas_ratio_2024(s4_target: pd.Series) -> pd.Series:
    """S4 non-CHP gross output over EIA-930 gas generation in 2024, by LST month x hour."""
    g = eia930.load_ba_hourly(FLEET_YEAR).set_index("datetime_utc")["gas"]
    df = pd.DataFrame({"target": s4_target, "gas": g.reindex(s4_target.index)}).dropna()
    lst = df.index - LST_OFFSET
    key = (lst.month - 1) * 24 + lst.hour
    r = df.groupby(key)["target"].sum() / df.groupby(key)["gas"].sum()
    return r.reindex(range(288))


def build_year(year: int, s4_target: pd.Series, curtailed_days: set[pd.Timestamp]) -> ErcotYear:
    """Inputs of one weather year (see module docstring). ``s4_target``: S4 non-CHP gross output in
    2024 by UTC hour; ``curtailed_days``: local dates flagged by build_ercot_weather_years.py."""
    hours = local_year_hours(year)
    res = pd.read_parquet(PROCESSED / f"ercot_residential_{year}.parquet").set_index("datetime_utc")
    res = res.reindex(hours)
    if res[["demand_mwh", "residential_mwh"]].isna().any().any():
        raise ValueError(f"{year}: missing demand or residential hours")
    scale = 1.0
    if year < SCALE_TO_YEAR:
        ref = pd.read_parquet(PROCESSED / f"ercot_residential_{SCALE_TO_YEAR}.parquet")["demand_mwh"].sum()
        scale = float(ref / res["demand_mwh"].sum())
    if year == FLEET_YEAR:
        base = s4_target.reindex(hours).to_numpy(float)
    else:
        ratio = gas_ratio_2024(s4_target)
        g = gas_generation(year)[0].reindex(hours).to_numpy(float)
        lst = hours - LST_OFFSET
        base = g * ratio.to_numpy()[(lst.month - 1) * 24 + lst.hour]
    if np.isnan(base).any():
        raise ValueError(f"{year}: missing gas base hours")
    local_day = hours.tz_localize("UTC").tz_convert(TZ).tz_localize(None).normalize()
    curtailed = np.isin(local_day, pd.DatetimeIndex(sorted(curtailed_days)))
    h = eia930.load_ba_hourly(year).set_index("datetime_utc").reindex(hours)
    ren = (h["wind"].fillna(0.0) + h["solar"].fillna(0.0)).to_numpy(float)
    fc = eia930.load_ba_forecast(year).set_index("datetime_utc")["demand_forecast"].reindex(hours)
    if fc.isna().any():
        raise ValueError(f"{year}: missing day-ahead forecast hours")
    return ErcotYear(year, hours, res["demand_mwh"].to_numpy() * scale, res["residential_mwh"].to_numpy() * scale,
                     base, plant_temperature_path(year), scale, curtailed, ren, fc.to_numpy(float) * scale)


def plant_temperature_path(year: int) -> Path:
    """Plant temperatures of the 2024 fleet's plants in ``year`` (scripts/build_ercot_weather_years.py)."""
    if year == FLEET_YEAR:
        return INTERIM / f"plant_temperature_{year}.parquet"
    return INTERIM / f"plant_temperature_{year}_fleet2024.parquet"


def plant_temperature_matrix(path: Path, hours: pd.DatetimeIndex, plant_order: np.ndarray) -> tuple[np.ndarray, int]:
    """(H, N) hour-beginning temperatures for the S4 units (the S4 definition: mean of the observations
    at the start and end of the hour, ``ttrl.supply.fleetmodel.plant_temperature_matrix``) from a
    plant-temperature file; ``plant_order`` is the plant id of each S4 unit. Remaining gaps (hours at
    the edge of the file's year) take the nearest value; their count is returned."""
    from ttrl.supply.fleetmodel import plant_temperature_matrix as hour_beginning

    t = pd.read_parquet(path, columns=["plant_id_eia", "ts_utc", "temp_c"])
    wide = pd.DataFrame(hour_beginning(t, hours, plant_order), index=hours)
    n_gap = int(wide.isna().any(axis=1).sum())
    wide = wide.interpolate(limit=6, limit_area="inside").ffill().bfill()
    return wide.to_numpy(float), n_gap


@dataclass
class Surrogate:
    """Band-corrected marginal fuel (MMBtu/MWh) and CO2 (t/MWh) per hour of one year, with the
    model's curvature (per MWh^2; zero if not given) and the confirmed share of the step-size part by
    size (``curv_beta`` (len(CURV_X), 2) for fuel and CO2; 1 if not given)."""
    hours: pd.DatetimeIndex
    fuel: np.ndarray
    co2_t: np.ndarray
    model_fuel: np.ndarray      # uncorrected central difference, for reference
    stratum: np.ndarray
    fuel_curv: np.ndarray | None = None
    co2_curv: np.ndarray | None = None
    curv_beta: np.ndarray | None = None

    def __post_init__(self):
        if self.fuel_curv is None:
            self.fuel_curv = np.zeros(len(self.hours))
        if self.co2_curv is None:
            self.co2_curv = np.zeros(len(self.hours))

    def _b2(self, dl: np.ndarray, j: int) -> np.ndarray | float:
        if self.curv_beta is None:
            return 1.0
        return np.interp(np.abs(dl), CURV_X, self.curv_beta[:, j])

    def fuel_change(self, rows: np.ndarray, dl: np.ndarray) -> np.ndarray:
        """Fuel change (MMBtu) per hour for load changes ``dl`` (MWh) in hour rows ``rows``."""
        return self.fuel[rows] * dl + 0.5 * self._b2(dl, 0) * self.fuel_curv[rows] * dl ** 2

    def co2_change(self, rows: np.ndarray, dl: np.ndarray) -> np.ndarray:
        """CO2 change (t) per hour."""
        return self.co2_t[rows] * dl + 0.5 * self._b2(dl, 1) * self.co2_curv[rows] * dl ** 2


def build_surrogate(model, year: ErcotYear, temps: np.ndarray, chunk: int = 24 * 7) -> Surrogate:
    """Central difference of the S4 model at +/- DELTA_MW and curvature from CURVATURE_STEPS_MW in
    every hour, band-corrected per hour (module docstring)."""
    from ttrl.supply.impact import TONNE_PER_SHORT_TON

    H = len(year.hours)
    m_fuel, m_co2 = np.zeros(H), np.zeros(H)
    num = np.zeros((H, 2))
    den = np.zeros(H)
    st = np.zeros(H, int)
    for a in range(0, H, chunk):
        sl = slice(a, min(H, a + chunk))
        hrs = year.hours[sl]
        kw = dict(temp_c=temps[sl], base_mw=year.base_mw[sl])
        up = model.impact(np.full(len(hrs), DELTA_MW), hrs, **kw).hourly
        dn = model.impact(np.full(len(hrs), -DELTA_MW), hrs, **kw).hourly
        de = (up["d_gas_mwh"] - dn["d_gas_mwh"]).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            m_fuel[sl] = np.where(de > 0, (up["d_fuel_mmbtu"] - dn["d_fuel_mmbtu"]).to_numpy() / de, np.nan)
            m_co2[sl] = np.where(de > 0, (up["d_co2_short_tons"] - dn["d_co2_short_tons"]).to_numpy() / de, np.nan)
        labels = list(model.band.labels)
        st[sl] = [labels.index(x) for x in up["stratum"]]
        for step in CURVATURE_STEPS_MW:
            u = model.impact(np.full(len(hrs), step), hrs, **kw).hourly
            d = model.impact(np.full(len(hrs), -step), hrs, **kw).hourly
            ok = ((np.abs(u["d_gas_mwh"] - step) < 1e-6) & (np.abs(d["d_gas_mwh"] + step) < 1e-6)
                  & (u["status_new"] == "ok") & (d["status_new"] == "ok") & (u["status_base"] == "ok")).to_numpy()
            even = np.column_stack([(u["d_fuel_mmbtu"] + d["d_fuel_mmbtu"]).to_numpy(),
                                    (u["d_co2_short_tons"] + d["d_co2_short_tons"]).to_numpy()])
            num[sl] += np.where(ok[:, None], even * step ** 2, 0.0)
            den[sl] += np.where(ok, step ** 4, 0.0)
    band = model.band
    g = st % band.n_load
    beta_f, beta_c = band.beta_point[g, 0], band.beta_point[g, 1]
    fuel = beta_f * m_fuel + (1 - beta_f) * band.m_bar[st, 0] - band.point[st, 0]
    co2 = beta_c * m_co2 + (1 - beta_c) * band.m_bar[st, 1] - band.point[st, 1]
    # hours where the model could not move (floor at zero output): fall back to the stratum CEMS marginal
    fuel = np.where(np.isnan(fuel), band.m_bar[st, 0] - band.point[st, 0], fuel)
    co2 = np.where(np.isnan(co2), band.m_bar[st, 1] - band.point[st, 1], co2)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(den[:, None] > 0, num / den[:, None], 0.0)         # least squares of F(s) + F(-s) = q s^2
    q = np.maximum(q, 0.0)
    if band.beta2_point is None:            # band without a step-size part: beta scales the curvature
        return Surrogate(year.hours, fuel, co2 * TONNE_PER_SHORT_TON, m_fuel, st,
                         beta_f * q[:, 0], beta_c * q[:, 1] * TONNE_PER_SHORT_TON)
    if tuple(band.size_edges) != (500.0, 1000.0, 2000.0):
        raise ValueError("CURV_X assumes the band's size classes 0.5 / 1 / 2 GW")
    return Surrogate(year.hours, fuel, co2 * TONNE_PER_SHORT_TON, m_fuel, st, q[:, 0],
                     q[:, 1] * TONNE_PER_SHORT_TON, np.asarray(band.beta2_point[:, :2], float))


def inputs_path(year: int) -> Path:
    return PROCESSED / f"ercot_env_{year}.parquet"


def save_inputs(year: ErcotYear, sur: Surrogate, temp_c: np.ndarray) -> Path:
    """One weather year's environment inputs (scripts/build_ercot_env.py); ``temp_c`` is the
    four-station mean temperature that flags days outside the kernel's support."""
    df = pd.DataFrame({"datetime_utc": year.hours, "demand_mwh": year.demand, "residential_mwh": year.residential,
                       "base_mw": year.base_mw, "curtailed": year.curtailed, "temp_c": temp_c,
                       "fuel_mmbtu_per_mwh": sur.fuel, "co2_t_per_mwh": sur.co2_t,
                       "model_fuel_mmbtu_per_mwh": sur.model_fuel, "stratum": sur.stratum,
                       "fuel_curv_mmbtu_per_mwh2": sur.fuel_curv, "co2_curv_t_per_mwh2": sur.co2_curv, "scale": year.scale,
                       "renewables_mwh": year.renewables, "demand_forecast_mwh": year.demand_forecast})
    path = inputs_path(year.year)
    tab = pa.Table.from_pandas(df, preserve_index=False)
    meta = {"curv_beta": None if sur.curv_beta is None else np.asarray(sur.curv_beta).tolist(), "curv_x": list(CURV_X)}
    tab = tab.replace_schema_metadata({**(tab.schema.metadata or {}), b"ttrl_surrogate": json.dumps(meta).encode()})
    pq.write_table(tab, path)
    return path


def load_inputs(year: int) -> tuple[ErcotYear, Surrogate, np.ndarray]:
    """(ErcotYear, Surrogate, four-station temperature) saved by ``save_inputs``."""
    tab = pq.read_table(inputs_path(year))
    meta = json.loads((tab.schema.metadata or {}).get(b"ttrl_surrogate", b"{}"))
    df = tab.to_pandas()
    hours = pd.DatetimeIndex(df["datetime_utc"])
    y = ErcotYear(year, hours, df["demand_mwh"].to_numpy(), df["residential_mwh"].to_numpy(), df["base_mw"].to_numpy(),
                  plant_temperature_path(year), float(df["scale"].iloc[0]), df["curtailed"].to_numpy(bool),
                  df["renewables_mwh"].to_numpy(), df["demand_forecast_mwh"].to_numpy())
    sur = Surrogate(hours, df["fuel_mmbtu_per_mwh"].to_numpy(), df["co2_t_per_mwh"].to_numpy(),
                    df["model_fuel_mmbtu_per_mwh"].to_numpy(), df["stratum"].to_numpy(),
                    df["fuel_curv_mmbtu_per_mwh2"].to_numpy(), df["co2_curv_t_per_mwh2"].to_numpy(),
                    None if meta.get("curv_beta") is None else np.asarray(meta["curv_beta"], float))
    return y, sur, df["temp_c"].to_numpy()
