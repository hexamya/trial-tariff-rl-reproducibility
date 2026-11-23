"""EIA-930 hourly balancing-authority data for ERCOT from PUDL on S3.

Two PUDL tables (release pinned in :mod:`ttrl.data.pudl`):

- ``core_eia930__hourly_operations``: demand, net generation, interchange.
- ``core_eia930__hourly_net_generation_by_energy_source``: net generation by
  energy source. In 2024 EIA changed the ERCO categories on 2024-07-01
  06:00 UTC (``solar`` -> ``solar_wo_integrated_battery_storage``, ``wind`` ->
  ``wind_wo_integrated_battery_storage``, ``hydro`` ->
  ``hydro_excluding_pumped_storage``) and added ``battery_storage`` (discharge)
  and ``unknown_energy_storage`` (charging, negative) from 2024-10-23. Before
  that date storage is presumably inside ``other``. :data:`SOURCE_GROUPS`
  sums the variants into stable groups.

Both files are sorted by ``datetime_utc``, so a time filter prunes row groups
and one year for one BA costs a few seconds. ``datetime_utc`` is used as
PUDL publishes it; how it lines up with CEMS hour-beginning timestamps is
checked empirically in ``scripts/validate_dispatch.py``.
"""
from __future__ import annotations

import pandas as pd
import pyarrow.dataset as ds

from ttrl.data import pudl
from ttrl.paths import INTERIM

OPS_TABLE = "core_eia930__hourly_operations"
BY_SOURCE_TABLE = "core_eia930__hourly_net_generation_by_energy_source"

SOURCE_GROUPS = {
    "gas": ["gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
    "wind": ["wind", "wind_wo_integrated_battery_storage",
             "wind_w_integrated_battery_storage"],
    "solar": ["solar", "solar_wo_integrated_battery_storage",
              "solar_w_integrated_battery_storage"],
    "hydro": ["hydro", "hydro_excluding_pumped_storage", "pumped_storage"],
    "storage_other": ["battery_storage", "unknown_energy_storage",
                      "other_energy_storage", "other", "oil", "geothermal",
                      "unknown"],
}


def _read(table: str, ba: str, start: pd.Timestamp, end: pd.Timestamp,
          release: str) -> pd.DataFrame:
    flt = ((ds.field("balancing_authority_code_eia") == ba)
           & (ds.field("datetime_utc") >= start) & (ds.field("datetime_utc") < end))
    return pudl.remote_dataset(table, release).to_table(filter=flt).to_pandas()


def load_ba_hourly(year: int, ba: str = pudl.ERCOT_BA_CODE,
                   release: str = pudl.DEFAULT_RELEASE,
                   overwrite: bool = False) -> pd.DataFrame:
    """Hourly demand, interchange and net generation by source group (MWh).

    Covers ``datetime_utc`` from 30 December of ``year - 1`` to 3 January of
    ``year + 1`` (a margin for time shifts). Columns: ``datetime_utc``,
    ``demand``, ``net_generation``, ``interchange`` (adjusted series) and
    one column per key of :data:`SOURCE_GROUPS` (adjusted net generation,
    missing categories counted as zero). Cached in ``data/interim``.
    """
    path = INTERIM / f"eia930_{ba.lower()}_{year}.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    start = pd.Timestamp(year - 1, 12, 30)
    end = pd.Timestamp(year + 1, 1, 3)
    ops = _read(OPS_TABLE, ba, start, end, release)
    ops = ops.rename(columns={"demand_adjusted_mwh": "demand",
                              "net_generation_adjusted_mwh": "net_generation",
                              "interchange_adjusted_mwh": "interchange"})
    ops = ops[["datetime_utc", "demand", "net_generation", "interchange"]]

    src = _read(BY_SOURCE_TABLE, ba, start, end, release)
    src["generation_energy_source"] = src["generation_energy_source"].astype(str)
    wide = src.pivot_table(index="datetime_utc", columns="generation_energy_source",
                           values="net_generation_adjusted_mwh", aggfunc="sum")
    groups = pd.DataFrame(index=wide.index)
    for name, cols in SOURCE_GROUPS.items():
        present = [c for c in cols if c in wide.columns]
        groups[name] = wide[present].sum(axis=1, min_count=1).fillna(0.0)
    out = ops.merge(groups.reset_index(), on="datetime_utc", how="left")
    out = out.sort_values("datetime_utc", ignore_index=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def load_ba_forecast(year: int, ba: str = pudl.ERCOT_BA_CODE,
                     release: str = pudl.DEFAULT_RELEASE, overwrite: bool = False) -> pd.DataFrame:
    """Hourly day-ahead demand forecast next to reported and adjusted demand (MWh), same window as
    :func:`load_ba_hourly`. Used to find hours of curtailed demand (load shed), when actual demand
    falls far below the forecast. Cached in ``data/interim``."""
    path = INTERIM / f"eia930_{ba.lower()}_{year}_forecast.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    ops = _read(OPS_TABLE, ba, pd.Timestamp(year - 1, 12, 30), pd.Timestamp(year + 1, 1, 3), release)
    out = ops.rename(columns={"demand_forecast_mwh": "demand_forecast", "demand_reported_mwh": "demand_reported",
                              "demand_adjusted_mwh": "demand"})[["datetime_utc", "demand_forecast", "demand_reported", "demand"]]
    out = out.sort_values("datetime_utc", ignore_index=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
