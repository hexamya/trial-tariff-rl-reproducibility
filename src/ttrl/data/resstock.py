"""NREL ResStock end-use load profiles for the ERCOT region (OEDI data lake on AWS S3).

Source: ``s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/
2024/resstock_amy2018_release_2/timeseries_aggregates/by_iso_rto_region/upgrade=0/
iso_rto_region=ERCOT/up00-ercot-<building type>.csv`` (anonymous access, region us-west-2, routed
through ``HTTPS_PROXY`` when set). Five building types; 15-minute modelled consumption of the whole
residential stock of the region under actual 2018 weather (AMY2018), baseline (upgrade 0).
Columns kept: ``timestamp`` (as published), ``units_represented`` and
``out.electricity.total.energy_consumption.kwh``.

The published timestamps are taken as the end of each 15-minute interval in local standard time
(the ResStock convention; checked in ``scripts/build_ercot_residential.py`` against the daily
cycle). Files are cached as Parquet under ``data/raw/resstock``.
"""
from __future__ import annotations

import io
import os
import time
from functools import lru_cache

import pandas as pd
import pyarrow.fs as pafs

from ttrl.paths import RAW

BUCKET_PATH = ("oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/"
               "resstock_amy2018_release_2/timeseries_aggregates/by_iso_rto_region/upgrade=0/iso_rto_region=ERCOT")
BUILDING_TYPES = ("mobile_home", "multi-family_with_2_-_4_units", "multi-family_with_5plus_units",
                  "single-family_attached", "single-family_detached")
COLUMNS = ["timestamp", "units_represented", "out.electricity.total.energy_consumption.kwh"]
RAW_DIR = RAW / "resstock"
ATTEMPTS = 4


@lru_cache(maxsize=1)
def s3_filesystem() -> pafs.S3FileSystem:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return pafs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=proxy or None)


def load_building_type(btype: str, overwrite: bool = False) -> pd.DataFrame:
    """15-minute total electricity (kWh) of one ERCOT building type, cached as Parquet."""
    path = RAW_DIR / f"ercot_{btype}.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    key = f"{BUCKET_PATH}/up00-ercot-{btype}.csv"
    for attempt in range(1, ATTEMPTS + 1):          # the proxied link drops now and then
        try:
            with s3_filesystem().open_input_stream(key) as f:
                raw = f.read()
            break
        except OSError:
            if attempt == ATTEMPTS:
                raise
            time.sleep(10 * attempt)
    df = pd.read_csv(io.BytesIO(raw), usecols=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"out.electricity.total.energy_consumption.kwh": "kwh"})
    df["building_type"] = btype
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def ercot_residential(overwrite: bool = False) -> pd.DataFrame:
    """15-minute total residential electricity of the ERCOT stock (MWh) with units represented."""
    parts = [load_building_type(b, overwrite) for b in BUILDING_TYPES]
    df = pd.concat(parts, ignore_index=True)
    out = df.groupby("timestamp").agg(kwh=("kwh", "sum"), units=("units_represented", "sum")).reset_index()
    out["mwh"] = out.pop("kwh") / 1000
    return out
