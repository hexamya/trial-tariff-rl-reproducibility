"""Hour-by-unit matrices of actual CEMS operation for the gas fleet.

Dispatch and validation work on dense (hours, units) arrays. This module
turns the long CEMS unit-hour table into those arrays on the hour grid of a
CEMS year (local standard time, so 2024 runs from 2024-01-01 06:00 UTC to
2025-01-01 05:00 UTC; timestamps are tz-naive UTC, hour-beginning) and
derives the unit availability masks used by the dispatch model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttrl.supply.units import LST_OFFSET, make_unit_id, time_labels


def cems_year_hours(year: int) -> pd.DatetimeIndex:
    """UTC hour-beginning timestamps of a CEMS (local standard time) year."""
    start = pd.Timestamp(year, 1, 1) + LST_OFFSET
    end = pd.Timestamp(year + 1, 1, 1) + LST_OFFSET
    return pd.date_range(start, end, freq="h", inclusive="left")


@dataclass
class FleetHours:
    """Actual hourly operation of N units over T hours.

    Arrays are (T, N): ``energy`` gross MWh, ``load`` average gross MW over
    the operating time, ``optime`` operating fraction of the hour, ``heat``
    MMBtu, ``co2`` short tons (NaN where CEMS has no value while heat > 0),
    ``present`` (a CEMS row exists). ``labels`` has one row per hour with
    the LST ``month``, ``hour_lst`` and ``season``.
    """
    hours: pd.DatetimeIndex
    unit_ids: list[str]
    energy: np.ndarray
    load: np.ndarray
    optime: np.ndarray
    heat: np.ndarray
    co2: np.ndarray
    present: np.ndarray
    labels: pd.DataFrame

    @property
    def on(self) -> np.ndarray:
        """Units producing (gross energy > 0) in each hour."""
        return self.energy > 0

    def availability(self, period: str = "month") -> np.ndarray:
        """Units available to the dispatch model in each hour.

        A unit is available in hour t if CEMS has a row for it in that hour
        and it produced in at least one hour of the same LST ``period``
        ("month" or "day"); otherwise it is treated as on outage. "always"
        returns ``present``. The mask uses actual operation, so it is an
        exogenous input of the dispatch, the same for any target.
        """
        if period == "always":
            return self.present.copy()
        lst = self.labels["ts_lst"]
        key = {"month": lst.dt.month, "day": lst.dt.dayofyear}[period].to_numpy()
        avail = np.zeros_like(self.present)
        on = self.on
        for k in np.unique(key):
            rows = key == k
            avail[rows] = on[rows].any(axis=0)
        return avail & self.present

    def co2_filled(self, co2_per_mmbtu: np.ndarray) -> np.ndarray:
        """CO2 with gaps (heat > 0, CO2 missing) filled as heat x unit factor."""
        return np.where(np.isnan(self.co2), self.heat * co2_per_mmbtu[None, :], self.co2)

    def columns(self, mask: np.ndarray) -> "FleetHours":
        """Copy restricted to the units where ``mask`` (N,) is True."""
        mask = np.asarray(mask, bool)
        ids = [u for u, m in zip(self.unit_ids, mask) if m]
        return FleetHours(self.hours, ids, self.energy[:, mask], self.load[:, mask],
                          self.optime[:, mask], self.heat[:, mask], self.co2[:, mask],
                          self.present[:, mask], self.labels)


def build_fleet_hours(cems: pd.DataFrame, unit_ids: list[str], year: int) -> FleetHours:
    """Dense hourly arrays for ``unit_ids`` from a CEMS extract of ``year``.

    ``cems`` is the interim extract (all rows, including non-operating ones),
    e.g. ``data/interim/cems_ercot_gas_2024.parquet``.
    """
    hours = cems_year_hours(year)
    t_n, u_n = len(hours), len(unit_ids)
    uid = make_unit_id(cems["plant_id_epa"], cems["emissions_unit_id_epa"])
    keep = uid.isin(unit_ids).to_numpy()
    c = cems.loc[keep]
    ti = ((c["operating_datetime_utc"] - hours[0]) / pd.Timedelta(hours=1)).to_numpy()
    ti = ti.astype(int)
    ui = pd.Categorical(uid[keep], categories=unit_ids).codes
    inside = (ti >= 0) & (ti < t_n)
    ti, ui, c = ti[inside], ui[inside], c.loc[inside]

    def mat(values, fill=0.0):
        m = np.full((t_n, u_n), fill)
        m[ti, ui] = values
        return m

    optime = mat(c["operating_time_hours"].fillna(0.0).to_numpy())
    load = mat(c["gross_load_mw"].fillna(0.0).to_numpy())
    load = np.where(optime > 0, load, 0.0)
    heat = mat(c["heat_content_mmbtu"].fillna(0.0).to_numpy())
    heat = np.where(optime > 0, heat, 0.0)
    co2 = mat(c["co2_mass_tons"].to_numpy(dtype=float), fill=np.nan)
    co2 = np.where(heat > 0, co2, 0.0)
    present = np.zeros((t_n, u_n), bool)
    present[ti, ui] = True
    labels = time_labels(hours)
    labels.index = hours
    return FleetHours(hours=hours, unit_ids=list(unit_ids), energy=load * optime,
                      load=load, optime=optime, heat=heat, co2=co2,
                      present=present, labels=labels)
