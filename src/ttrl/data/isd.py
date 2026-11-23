"""NOAA Integrated Surface Database (ISD-lite) on the AWS Open Data bucket.

Bucket ``s3://noaa-isd-pds`` (region us-east-1, anonymous access, routed
through ``HTTPS_PROXY`` when set):

- ``isd-history.csv``: station list (USAF, WBAN, name, LAT, LON, BEGIN, END).
- ``isd-lite/data/<year>/<USAF>-<WBAN>-<year>.gz``: one gzip text file per
  station-year, one fixed-width line per hour (see :func:`parse_isd_lite`).
  Times are UTC; each value is the observation closest to the top of the
  hour within the 10 minutes before it (so an ASOS report at 15:53 UTC is
  stamped 16:00 UTC). Air temperature is in tenths of a degree C;
  ``-9999`` marks a missing value.

Downloads are cached under ``data/raw/noaa_isd/`` and fetched only when a
caller asks for a station-year (about 60-100 KB each).
"""
from __future__ import annotations

import gzip
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.fs as pafs

from ttrl.paths import RAW

BUCKET = "noaa-isd-pds"
REGION = "us-east-1"
RAW_ISD = RAW / "noaa_isd"
MISSING = -9999
EARTH_RADIUS_KM = 6371.0088  # mean Earth radius (IUGG)

# ISD-lite fixed-width layout (0-based, end-exclusive), checked against the
# files: lines are 61 characters, "YYYY MM DD HH" followed by eight
# right-aligned 6-character integer fields.
ISD_LITE_COLSPECS = [(0, 4), (5, 7), (8, 10), (11, 13), (13, 19), (19, 25),
                     (25, 31), (31, 37), (37, 43), (43, 49), (49, 55), (55, 61)]
ISD_LITE_NAMES = ["year", "month", "day", "hour", "temp", "dewpoint", "slp",
                  "wind_dir", "wind_speed", "sky", "precip_1h", "precip_6h"]
# (output column, raw column, divisor): raw integers are scaled by 10 for
# temperatures, pressure, wind speed and precipitation.
ISD_LITE_FIELDS = [
    ("temp_c", "temp", 10),
    ("dewpoint_c", "dewpoint", 10),
    ("slp_hpa", "slp", 10),
    ("wind_dir_deg", "wind_dir", 1),
    ("wind_speed_ms", "wind_speed", 10),
    ("sky_code", "sky", 1),
    ("precip_1h_mm", "precip_1h", 10),
    ("precip_6h_mm", "precip_6h", 10),
]


# --------------------------------------------------------------------------
# S3 access
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def s3_filesystem() -> pafs.S3FileSystem:
    """Anonymous pyarrow S3 filesystem for the NOAA ISD bucket."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return pafs.S3FileSystem(anonymous=True, region=REGION,
                             proxy_options=proxy or None)


def _download(key: str, dest: Path) -> Path:
    """Copy ``s3://noaa-isd-pds/<key>`` to ``dest`` byte for byte."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    # compression=None keeps .gz files compressed (pyarrow would otherwise
    # decompress them on the fly because of the extension).
    with s3_filesystem().open_input_stream(f"{BUCKET}/{key}", compression=None) as fin, \
            open(tmp, "wb") as fout:
        while chunk := fin.read(8 << 20):
            fout.write(chunk)
    tmp.replace(dest)
    return dest


def station_history(overwrite: bool = False) -> pd.DataFrame:
    """ISD station list with numeric coordinates.

    Columns: ``station_id`` ("USAF-WBAN"), ``name``, ``state``, ``icao``,
    ``lat``, ``lon``, ``elev_m``, ``begin``, ``end``. Stations without
    coordinates (or at 0, 0) are dropped.
    """
    path = RAW_ISD / "isd-history.csv"
    if overwrite or not path.exists():
        _download("isd-history.csv", path)
    h = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = pd.DataFrame({
        "station_id": h["USAF"] + "-" + h["WBAN"],
        "name": h["STATION NAME"].str.strip(),
        "state": h["STATE"],
        "icao": h["ICAO"],
        "lat": pd.to_numeric(h["LAT"], errors="coerce"),
        "lon": pd.to_numeric(h["LON"], errors="coerce"),
        "elev_m": pd.to_numeric(h["ELEV(M)"], errors="coerce"),
        "begin": pd.to_datetime(h["BEGIN"], format="%Y%m%d", errors="coerce"),
        "end": pd.to_datetime(h["END"], format="%Y%m%d", errors="coerce"),
    })
    ok = out["lat"].notna() & out["lon"].notna() & ~((out["lat"] == 0) & (out["lon"] == 0))
    return out.loc[ok].drop_duplicates("station_id").reset_index(drop=True)


def isd_lite_inventory(year: int, overwrite: bool = False) -> pd.DataFrame:
    """Station files present under ``isd-lite/data/<year>/`` (id and bytes).

    The listing (about 13,000 keys for 2024) is cached as CSV.
    """
    path = RAW_ISD / f"isd-lite-{year}-files.csv"
    if overwrite or not path.exists():
        infos = s3_filesystem().get_file_info(
            pafs.FileSelector(f"{BUCKET}/isd-lite/data/{year}"))
        rows = [{"station_id": Path(i.path).name.removesuffix(f"-{year}.gz"),
                 "size_bytes": i.size}
                for i in infos if i.type == pafs.FileType.File and i.path.endswith(".gz")]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["station_id", "size_bytes"]).to_csv(path, index=False)
    return pd.read_csv(path, dtype={"station_id": str})


def isd_lite_path(station_id: str, year: int) -> Path:
    """Local copy of one ISD-lite station-year file (downloaded if absent)."""
    dest = RAW_ISD / "isd-lite" / str(year) / f"{station_id}-{year}.gz"
    if not dest.exists():
        _download(f"isd-lite/data/{year}/{station_id}-{year}.gz", dest)
    return dest


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_isd_lite(source: Path | str | bytes) -> pd.DataFrame:
    """Parse an ISD-lite file (gzip path, raw bytes, or decoded text).

    Returns ``ts_utc`` (tz-naive UTC) and the eight data fields in physical
    units (``temp_c``, ``dewpoint_c``, ``slp_hpa``, ``wind_dir_deg``,
    ``wind_speed_ms``, ``sky_code``, ``precip_1h_mm``, ``precip_6h_mm``)
    with ``-9999`` turned into NaN (trace precipitation, coded -1, becomes
    -0.1 mm). Rows keep file order.
    """
    if isinstance(source, Path):
        source = source.read_bytes()
    if isinstance(source, bytes):
        source = gzip.decompress(source) if source[:2] == b"\x1f\x8b" else source
        source = source.decode("ascii")
    raw = pd.read_fwf(io.StringIO(source), colspecs=ISD_LITE_COLSPECS,
                      names=ISD_LITE_NAMES, header=None, dtype="int64")
    ts = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    out = pd.DataFrame({"ts_utc": ts.astype("datetime64[us]")})
    for name, col, divisor in ISD_LITE_FIELDS:
        out[name] = raw[col].where(raw[col] != MISSING).astype("float64") / divisor
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (inputs in degrees, broadcastable)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64"))
                              for x in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
