"""Loader for the LCL smart-meter release (UK Power Networks, Low Carbon London).

Source: ``data/raw/lcl/meters/partitioned_lcl_data.zip`` (``scripts/fetch_lcl_meters.py``),
168 CSV pieces ``Small LCL Data/LCL-June2015v2_<k>.csv`` (8.5 GB uncompressed) with columns
``LCLid, stdorToU, DateTime, KWH/hh (per half hour)``. ``stdorToU`` is ``Std`` (flat-rate
control group) or ``ToU`` (dToU trial group). kWh values are padded with spaces and some
are ``Null``. Rows are sorted by household and time, so a household can span two pieces.

:func:`build_panel` streams the pieces through DuckDB into one parquet file with one row
per household and half-hour:

* ``hh_id`` (int32): the digits of ``LCLid`` (``MAC000002`` -> 2)
* ``tou`` (bool): ``stdorToU == 'ToU'``
* ``ts_label`` (timestamp): ``DateTime`` as written in the source, without time zone
* ``kwh`` (float32): NULL where the source says ``Null`` (none left after dropping the
  off-grid rows, see :func:`build_panel`)

The panel keeps the source's time labels. What they mean (clock, and whether a label marks
the start or the end of its half-hour) is checked separately against the dToU price
schedule, see ``scripts/build_lcl_panel.py``.

The full panel has ~167M rows and does not fit in pandas on a small machine; query it
with DuckDB (:func:`connect`).
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from ttrl.paths import INTERIM, PROCESSED, RAW

ZIP_PATH = RAW / "lcl" / "meters" / "partitioned_lcl_data.zip"
PIECES_DIR = INTERIM / "lcl_meters"
PANEL_PATH = PROCESSED / "lcl_meters.parquet"
HOUSEHOLDS_PATH = PROCESSED / "lcl_households.parquet"
N_PIECES = 168


def connect(threads: int = 4, memory_limit: str = "2GB") -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a memory cap and a spill directory under data/interim."""
    tmp = INTERIM / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order = true")
    return con


def piece_number(name: str) -> int:
    m = re.search(r"_(\d+)\.csv$", name)
    if not m:
        raise ValueError(f"unexpected member name {name!r}")
    return int(m.group(1))


PIECE_SQL = """
SELECT CAST(substr(lclid, 4) AS INTEGER)              AS hh_id,
       tariff = 'ToU'                                 AS tou,
       CAST(substr(dt, 1, 19) AS TIMESTAMP)           AS ts_label,
       CAST(TRY_CAST(trim(kwh) AS DOUBLE) AS FLOAT)   AS kwh,
       lclid, tariff, trim(kwh) AS kwh_raw, dt
FROM read_csv('{csv}', header = true, all_varchar = true, quote = '',
              names = ['lclid', 'tariff', 'dt', 'kwh'])
"""


def convert_piece(con: duckdb.DuckDBPyConnection, csv: Path, out: Path) -> dict:
    """Parse one CSV piece into parquet and return row counts and format problems."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW piece AS {PIECE_SQL.format(csv=csv.as_posix())}")
    stats = con.execute("""
        SELECT count(*) AS n_rows,
               count(*) FILTER (WHERE kwh_raw = 'Null') AS n_null,
               count(*) FILTER (WHERE kwh IS NULL AND kwh_raw <> 'Null') AS n_bad_kwh,
               count(*) FILTER (WHERE tariff NOT IN ('Std', 'ToU')) AS n_bad_tariff,
               count(*) FILTER (WHERE NOT regexp_full_match(lclid, 'MAC\\d{6}')) AS n_bad_id,
               count(*) FILTER (WHERE substr(dt, 20) <> '.0000000'
                                OR minute(ts_label) NOT IN (0, 30) OR second(ts_label) <> 0) AS n_bad_ts
        FROM piece""").df().iloc[0].to_dict()
    con.execute(f"COPY (SELECT hh_id, tou, ts_label, kwh FROM piece) TO '{out.as_posix()}' "
                "(FORMAT parquet, COMPRESSION zstd)")
    return {k: int(v) for k, v in stats.items()}


def convert_zip(zip_path: Path = ZIP_PATH, pieces_dir: Path = PIECES_DIR, n_pieces: int = N_PIECES,
                verbose: bool = True) -> pd.DataFrame:
    """Convert every CSV piece in the zip to ``pieces_dir/piece_<k>.parquet`` (resumable)."""
    pieces_dir.mkdir(parents=True, exist_ok=True)
    con = connect()
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        members = sorted((m for m in z.namelist() if m.endswith(".csv")), key=piece_number)
        if len(members) != n_pieces:
            raise ValueError(f"expected {n_pieces} CSV pieces, found {len(members)}")
        for m in members:
            k = piece_number(m)
            out = pieces_dir / f"piece_{k:03d}.parquet"
            stats_path = out.with_suffix(".stats.json")
            if out.exists() and stats_path.exists():
                rows.append({"piece": k, **pd.read_json(stats_path, typ="series").to_dict()})
                continue
            csv = pieces_dir / "piece.csv"
            with z.open(m) as src, open(csv, "wb") as dst:
                shutil.copyfileobj(src, dst)
            stats = convert_piece(con, csv, out)
            pd.Series(stats).to_json(stats_path)
            rows.append({"piece": k, **stats})
            if verbose and (k % 20 == 0 or k == n_pieces - 1):
                print(f"piece {k}: {stats['n_rows']:,} rows", flush=True)
        (pieces_dir / "piece.csv").unlink(missing_ok=True)
    return pd.DataFrame(rows).sort_values("piece").reset_index(drop=True)


def build_panel(pieces_dir: Path = PIECES_DIR, out: Path = PANEL_PATH,
                households_out: Path = HOUSEHOLDS_PATH) -> dict:
    """Join the pieces into one panel parquet plus a household table.

    Dropped: rows whose label is off the half-hour grid (in the release these are one ``Null``
    row per household, mostly stamped 2012-12-18 15:1x-15:2x, not readings) and exact
    duplicate rows. Returns the counts, plus conflicting duplicates (same household and label,
    different kWh), which are kept and must be handled by the analysis.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    con = connect()
    glob = (pieces_dir / "piece_*.parquet").as_posix()
    con.execute(f"CREATE TEMP VIEW all_rows AS SELECT * FROM read_parquet('{glob}')")
    n_raw, n_off, n_off_null = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE NOT on_grid), count(*) FILTER (WHERE NOT on_grid AND kwh IS NULL)
        FROM (SELECT kwh, minute(ts_label) IN (0, 30) AND second(ts_label) = 0 AS on_grid FROM all_rows)
        """).fetchone()
    con.execute("""CREATE TEMP VIEW raw AS SELECT * FROM all_rows
                   WHERE minute(ts_label) IN (0, 30) AND second(ts_label) = 0""")
    tou_conflict = con.execute(
        "SELECT count(*) FROM (SELECT hh_id FROM raw GROUP BY hh_id HAVING count(DISTINCT tou) > 1)").fetchone()[0]
    if tou_conflict:
        raise ValueError(f"{tou_conflict} households switch between Std and ToU")
    con.execute(f"""
        COPY (SELECT DISTINCT hh_id, tou, ts_label, kwh FROM raw ORDER BY hh_id, ts_label)
        TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
    con.execute(f"CREATE TEMP VIEW panel AS SELECT * FROM read_parquet('{out.as_posix()}')")
    n_panel, n_conflict = con.execute("""
        SELECT (SELECT count(*) FROM panel),
               (SELECT coalesce(sum(n - 1), 0) FROM
                   (SELECT count(*) AS n FROM panel GROUP BY hh_id, ts_label HAVING count(*) > 1))
        """).fetchone()
    con.execute(f"""
        COPY (SELECT hh_id, printf('MAC%06d', hh_id) AS lclid, any_value(tou) AS tou,
                     min(ts_label) AS first_label, max(ts_label) AS last_label,
                     count(*) AS n_rows, count(*) FILTER (WHERE kwh IS NULL) AS n_null
              FROM panel GROUP BY hh_id ORDER BY hh_id)
        TO '{households_out.as_posix()}' (FORMAT parquet)""")
    return {"n_raw": int(n_raw), "n_off_grid": int(n_off), "n_off_grid_null": int(n_off_null),
            "n_exact_dup": int(n_raw - n_off - n_panel), "n_panel": int(n_panel),
            "n_conflicting_dup": int(n_conflict)}


def households(path: Path = HOUSEHOLDS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
