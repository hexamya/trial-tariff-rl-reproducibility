"""Read PUDL (Catalyst Cooperative) tables from the public S3 bucket.

PUDL publishes each data release as flat Parquet files under
``s3://pudl.catalyst.coop/<release>/<table>.parquet`` (anonymous access,
region us-west-2). Releases are named ``vYYYY.M.P``; ``stable/`` and
``nightly/`` are moving aliases, so we pin a versioned release.

Read method: PyArrow's built-in S3 filesystem with
anonymous credentials and the ``HTTPS_PROXY`` of the environment. Parquet
row-group statistics let pyarrow skip most of the 4.9 GB CEMS file: one
year of hourly data for selected plants took about 150 MB of range requests.
DuckDB can query :func:`remote_dataset` objects directly.

Small tables (EIA-860 generators, EPA-EIA crosswalk) are downloaded once to
``data/raw/pudl/<release>/``; filtered CEMS extracts go to ``data/interim/``.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from ttrl.paths import INTERIM, RAW

BUCKET = "pudl.catalyst.coop"
REGION = "us-west-2"
HTTPS_BASE = f"https://s3.{REGION}.amazonaws.com/{BUCKET}"

# Pinned for reproducibility. On 2026-10-03 this was the newest versioned
# release and stable/ held byte-identical files.
DEFAULT_RELEASE = "v2026.9.0"

ERCOT_BA_CODE = "ERCO"

# EIA-860 technology_description -> our short technology label.
GAS_TECHNOLOGIES = {
    "Natural Gas Fired Combined Cycle": "CC",
    "Natural Gas Fired Combustion Turbine": "CT",
    "Natural Gas Steam Turbine": "ST",
}
# When one CEMS unit maps to generators of several technologies, the first
# label in this order wins (a CT feeding a heat-recovery boiler is "CC").
TECH_PRIORITY = ("CC", "CT", "ST")

GENERATORS_TABLE = "out_eia__yearly_generators"
CROSSWALK_TABLE = "core_epa__assn_eia_epacamd"
CEMS_TABLE = "core_epacems__hourly_emissions"

GENERATOR_COLUMNS = [
    "report_date", "plant_id_eia", "generator_id", "plant_name_eia",
    "state", "county", "latitude", "longitude",
    "balancing_authority_code_eia", "technology_description",
    "prime_mover_code", "energy_source_code_1", "fuel_type_code_pudl",
    "capacity_mw", "summer_capacity_mw", "winter_capacity_mw",
    "minimum_load_mw", "generator_operating_date",
    "generator_retirement_date", "operational_status",
    "unit_heat_rate_mmbtu_per_mwh", "associated_combined_heat_power",
    "data_maturity",
]

CEMS_COLUMNS = [
    "plant_id_epa", "emissions_unit_id_epa", "plant_id_eia",
    "operating_datetime_utc", "year", "state", "operating_time_hours",
    "gross_load_mw", "heat_content_mmbtu", "co2_mass_tons",
    "co2_mass_measurement_code", "nox_mass_lbs", "steam_load_lbs",
]


# --------------------------------------------------------------------------
# Low-level access
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def s3_filesystem() -> pafs.S3FileSystem:
    """Anonymous pyarrow S3 filesystem for the PUDL bucket.

    Uses ``HTTPS_PROXY`` when configured and a direct connection otherwise.
    """
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return pafs.S3FileSystem(anonymous=True, region=REGION,
                             proxy_options=proxy or None)


def _version_key(name: str) -> tuple[int, ...]:
    """Sort key for release names such as ``v2026.9.0``."""
    return tuple(int(x) for x in name.lstrip("v").split("."))


def list_releases() -> list[str]:
    """Versioned releases (``vYYYY.M.P``) in the bucket, oldest first."""
    infos = s3_filesystem().get_file_info(pafs.FileSelector(BUCKET))
    names = [Path(i.path).name for i in infos if i.type == pafs.FileType.Directory]
    versions = [n for n in names if re.fullmatch(r"v\d+\.\d+\.\d+", n)]
    return sorted(versions, key=_version_key)


def latest_release() -> str:
    """Newest versioned release name."""
    return list_releases()[-1]


def list_release_files(release: str = DEFAULT_RELEASE,
                       suffix: str = ".parquet") -> pd.DataFrame:
    """Files directly under one release folder, with size and timestamp."""
    infos = s3_filesystem().get_file_info(pafs.FileSelector(f"{BUCKET}/{release}"))
    rows = [
        {
            "table": Path(i.path).name.removesuffix(".parquet"),
            "key": i.path.removeprefix(f"{BUCKET}/"),
            "size_bytes": i.size,
            "last_modified": i.mtime,
        }
        for i in infos
        if i.type == pafs.FileType.File and i.path.endswith(suffix)
    ]
    return pd.DataFrame(rows).sort_values("key", ignore_index=True)


def table_url(table: str, release: str = DEFAULT_RELEASE) -> str:
    """Plain HTTPS URL of a table (usable with curl or DuckDB httpfs)."""
    return f"{HTTPS_BASE}/{release}/{table}.parquet"


def download_table(table: str, release: str = DEFAULT_RELEASE,
                   overwrite: bool = False) -> Path:
    """Download a whole (small) table to ``data/raw/pudl/<release>/``."""
    dest = RAW / "pudl" / release / f"{table}.parquet"
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    src = f"{BUCKET}/{release}/{table}.parquet"
    with s3_filesystem().open_input_stream(src) as fin, open(tmp, "wb") as fout:
        while chunk := fin.read(8 << 20):
            fout.write(chunk)
    tmp.replace(dest)
    return dest


def remote_dataset(table: str, release: str = DEFAULT_RELEASE) -> ds.Dataset:
    """pyarrow dataset over the remote Parquet file (lazy, range reads)."""
    return ds.dataset(f"{BUCKET}/{release}/{table}.parquet",
                      filesystem=s3_filesystem(), format="parquet")


# --------------------------------------------------------------------------
# EIA-860 metadata for ERCOT gas units
# --------------------------------------------------------------------------
def load_generators_in_service(year: int,
                               release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """All EIA-860 generators (any fuel, any region) in service during ``year``.

    "In service" means status existing, or retired with a retirement date on
    or after 1 January of ``year``.
    """
    path = download_table(GENERATORS_TABLE, release)
    gens = pd.read_parquet(path, columns=GENERATOR_COLUMNS)
    gens["report_date"] = pd.to_datetime(gens["report_date"])
    for col in ("generator_operating_date", "generator_retirement_date"):
        gens[col] = pd.to_datetime(gens[col])
    year_start = pd.Timestamp(year=year, month=1, day=1)
    in_service = (gens["operational_status"] == "existing") | (
        (gens["operational_status"] == "retired")
        & (gens["generator_retirement_date"] >= year_start)
    )
    keep = (gens["report_date"].dt.year == year) & in_service
    return gens.loc[keep].reset_index(drop=True)


def load_ercot_gas_generators(year: int,
                              release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """ERCOT gas-fired CC / CT / steam generators in service during ``year``.

    ERCOT membership is EIA-860 ``balancing_authority_code_eia == "ERCO"``
    for that report year; ``tech`` is the CC / CT / ST label.
    """
    gens = load_generators_in_service(year, release)
    keep = ((gens["balancing_authority_code_eia"] == ERCOT_BA_CODE)
            & gens["technology_description"].isin(list(GAS_TECHNOLOGIES)))
    out = gens.loc[keep].copy()
    out["tech"] = out["technology_description"].map(GAS_TECHNOLOGIES)
    return out.reset_index(drop=True)


def load_epa_eia_crosswalk(year: int,
                           release: str = DEFAULT_RELEASE) -> pd.DataFrame:
    """EPA CAMD unit <-> EIA generator links valid for ``year``.

    The crosswalk is published per ``report_year``; when ``year`` is newer
    than the last published year we fall back to the latest one.
    """
    xw = pd.read_parquet(download_table(CROSSWALK_TABLE, release))
    xw = xw.dropna(subset=["plant_id_epa", "emissions_unit_id_epa",
                           "plant_id_eia", "generator_id"])
    ref_year = min(year, int(xw["report_year"].max()))
    xw = xw[xw["report_year"] == ref_year]
    cols = ["plant_id_epa", "emissions_unit_id_epa", "plant_id_eia", "generator_id"]
    return xw[cols + ["report_year"]].drop_duplicates(cols, ignore_index=True)


def _plant_level_links(gens: pd.DataFrame, xw: pd.DataFrame,
                       cems_units: pd.DataFrame, year: int,
                       release: str) -> pd.DataFrame:
    """Fallback links for CEMS units the crosswalk cannot place in ``year``.

    A CEMS unit needs a fallback when no crosswalk link points to a
    generator in service that year (unit missing from the crosswalk, or
    linked only to retired / renamed generators). It is linked to every
    ERCOT gas generator of its EIA plant that the crosswalk leaves
    unlinked, provided those generators share one technology; their
    capacity is split evenly over the units linked this way.
    """
    key = ["plant_id_epa", "emissions_unit_id_epa"]
    gen_key = ["plant_id_eia", "generator_id"]
    live = load_generators_in_service(year, release)[gen_key]
    placed = xw.merge(live, on=gen_key)[key].drop_duplicates()
    missing = cems_units.merge(placed, on=key, how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", key + ["plant_id_eia"]]

    free = gens.merge(xw[gen_key].drop_duplicates(), on=gen_key,
                      how="left", indicator=True)
    free = free[free["_merge"] == "left_only"].drop(columns="_merge")
    free = free.groupby("plant_id_eia").filter(lambda g: g["tech"].nunique() == 1)

    links = missing.merge(free, on="plant_id_eia")
    links["n_units_sharing"] = links.groupby(gen_key)["emissions_unit_id_epa"].transform("nunique")
    links["report_year"] = pd.NA
    links["link_method"] = "plant"
    return links


def load_ercot_gas_units(year: int, release: str = DEFAULT_RELEASE,
                         cems_units: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per CEMS emissions unit linked to an ERCOT gas generator.

    Links come from the EPA-EIA crosswalk (``link_method == "crosswalk"``).
    If ``cems_units`` (distinct ``plant_id_epa``, ``emissions_unit_id_epa``,
    ``plant_id_eia`` seen in CEMS) is given, units
    that the crosswalk cannot place get plant-level links
    (``link_method == "plant"``, see :func:`_plant_level_links`).

    ``capacity_mw`` is the nameplate of the linked generators, with each
    generator's capacity split evenly over the CEMS units it is linked to
    (so summing over units does not double count a shared steam turbine).
    ``chp`` is True when any linked generator is flagged by EIA-860 as
    associated with combined heat and power (its fuel also makes process
    steam, so heat input per MWh is not an electric heat rate).
    """
    gens = load_ercot_gas_generators(year, release)
    xw = load_epa_eia_crosswalk(year, release)

    gen_key = ["plant_id_eia", "generator_id"]
    units_per_gen = (
        xw.groupby(gen_key)["emissions_unit_id_epa"]
        .nunique().rename("n_units_sharing").reset_index()
    )
    links = (
        xw.merge(gens, on=gen_key, how="inner")
        .merge(units_per_gen, on=gen_key)
        .assign(link_method="crosswalk")
    )
    if cems_units is not None:
        fallback = _plant_level_links(gens, xw, cems_units, year, release)
        links = pd.concat([links, fallback], ignore_index=True)
    links["capacity_alloc_mw"] = links["capacity_mw"] / links["n_units_sharing"]

    def _tech(labels: pd.Series) -> str:
        present = set(labels)
        return next(t for t in TECH_PRIORITY if t in present)

    key = ["plant_id_epa", "emissions_unit_id_epa"]
    units = links.groupby(key, as_index=False).agg(
        plant_id_eia=("plant_id_eia", "first"),
        plant_name_eia=("plant_name_eia", "first"),
        state=("state", "first"),
        county=("county", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        tech=("tech", _tech),
        prime_movers=("prime_mover_code", lambda s: "+".join(sorted(set(s)))),
        generator_ids=("generator_id", lambda s: ",".join(sorted(set(s)))),
        capacity_mw=("capacity_alloc_mw", "sum"),
        chp=("associated_combined_heat_power", lambda s: bool(s.fillna(False).any())),
        link_method=("link_method", "first"),
        crosswalk_year=("report_year", "first"),
    )
    return units


# --------------------------------------------------------------------------
# EPA CEMS hourly
# --------------------------------------------------------------------------
def load_cems_hourly(year: int, plant_ids_epa: Iterable[int] = (),
                     plant_ids_eia: Iterable[int] = (),
                     release: str = DEFAULT_RELEASE,
                     columns: list[str] | None = None) -> pd.DataFrame:
    """CEMS unit-hours in ``year`` for the given EPA or EIA plant IDs.

    Filtering happens at read time: the ``year`` predicate prunes row groups
    by their min/max statistics, so only about one year of the file is
    fetched, and only the requested columns.
    """
    dset = remote_dataset(CEMS_TABLE, release)
    plants = (ds.field("plant_id_epa").isin(sorted({int(p) for p in plant_ids_epa}))
              | ds.field("plant_id_eia").isin(sorted({int(p) for p in plant_ids_eia})))
    flt = (ds.field("year") == year) & plants
    table = dset.to_table(columns=columns or CEMS_COLUMNS, filter=flt)
    return table.to_pandas()


def interim_paths(year: int) -> dict[str, Path]:
    """Where :func:`build_ercot_gas_cems` writes its outputs."""
    return {
        "generators": INTERIM / f"ercot_gas_generators_{year}.parquet",
        "units": INTERIM / f"ercot_gas_units_{year}.parquet",
        "cems": INTERIM / f"cems_ercot_gas_{year}.parquet",
        "unlinked": INTERIM / f"cems_ercot_unlinked_units_{year}.parquet",
    }


def build_ercot_gas_cems(year: int, release: str = DEFAULT_RELEASE,
                         overwrite: bool = False) -> dict[str, Path]:
    """Extract ERCOT gas generators, CEMS units and hourly CEMS for a year.

    Writes Parquet files to ``data/interim/`` (see :func:`interim_paths`).
    CEMS is read for every plant that has an ERCOT gas generator or a
    crosswalk-linked unit. Rows are kept only for units linked to an ERCOT
    gas generator and carry the unit's ``tech`` and ``chp`` labels; other
    CEMS units at those plants are summarised in the ``unlinked`` file.
    Existing outputs are reused unless ``overwrite`` is set.
    """
    paths = interim_paths(year)
    if not overwrite and all(p.exists() for p in paths.values()):
        return paths
    INTERIM.mkdir(parents=True, exist_ok=True)

    key = ["plant_id_epa", "emissions_unit_id_epa"]
    gens = load_ercot_gas_generators(year, release)
    xw_units = load_ercot_gas_units(year, release)
    cems = load_cems_hourly(year, xw_units["plant_id_epa"].tolist(),
                            gens["plant_id_eia"].tolist(), release)
    cems["emissions_unit_id_epa"] = cems["emissions_unit_id_epa"].astype("string")
    units = load_ercot_gas_units(
        year, release, cems_units=cems[key + ["plant_id_eia"]].drop_duplicates())
    cems = cems.merge(units[key + ["tech", "chp"]], on=key, how="left")

    # CEMS units at these plants that are not linked to an ERCOT gas
    # generator (e.g. coal units at a plant that also has gas units).
    cems[cems["tech"].isna()].groupby(key, as_index=False).agg(
        plant_id_eia=("plant_id_eia", "first"),
        n_rows=("operating_datetime_utc", "size"),
        heat_content_mmbtu=("heat_content_mmbtu", "sum"),
    ).to_parquet(paths["unlinked"], index=False)

    cems = cems.dropna(subset=["tech"]).astype({"chp": bool})
    cems = cems.sort_values(key + ["operating_datetime_utc"], ignore_index=True)
    gens.to_parquet(paths["generators"], index=False)
    units.to_parquet(paths["units"], index=False)
    cems.to_parquet(paths["cems"], index=False)
    return paths
