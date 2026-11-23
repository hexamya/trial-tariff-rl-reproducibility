"""Hourly ambient temperature at ERCOT gas plants from NOAA ISD-lite stations.

Station files come from :mod:`ttrl.data.isd`. Method (implemented by
:func:`build_plant_temperature`):

1. Plant coordinates come from the PUDL EIA-860 plants table
   (``out_eia__yearly_plants``, report year ``year``).
2. Station temperatures are placed on a regular UTC hour grid
   (:func:`hour_grid`) and checked three ways: a plausible range and an
   isolated one-hour spike test (:func:`qc_temperature`), and a buddy check
   against neighbouring stations that removes whole days of a persistent
   multi-day bias (:func:`buddy_check`). A station is *good* when it has a
   valid temperature in at least ``MIN_COVERAGE`` of the calendar-year
   hours after these checks.
3. Each plant gets the nearest good station as primary and the next
   ``N_DONORS`` nearest good stations as donors (:func:`good_stations`).
   Station files are downloaded lazily, nearest first, so only the files
   that are needed are fetched.
4. Gaps of at most ``MAX_GAP_HOURS`` consecutive hours in the primary
   series are filled by linear interpolation; longer gaps are filled from
   the donors, nearest first, after adding the mean primary-minus-donor
   difference for the same calendar month and UTC hour of day
   (:func:`donor_offset`, :func:`fill_plant_series`). If interior hours
   are still missing, up to ``MAX_EXTRA_DONORS`` further good stations are
   tried, and any gap left after that is interpolated linearly.

The hour grid runs from 1 January 00:00 UTC of ``year`` through
1 January 06:00 UTC of the next year. It therefore covers the UTC calendar
year and every hour of the EPA CEMS reporting year, which is the local
standard-time year (UTC-6 for ERCOT): CEMS 2024 hours start at
2024-01-01 06:00 UTC and the last one begins at 2025-01-01 05:00 UTC.
Timestamps are tz-naive UTC, like ``operating_datetime_utc`` in the CEMS
extracts. ``temp_c`` is a point observation at ``ts_utc``; CEMS hours are
hour-beginning, so :func:`hour_beginning_mean` gives the average of the
observations at both ends of each hour if that alignment is wanted.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from ttrl.data import isd
from ttrl.data.isd import haversine_km
from ttrl.paths import INTERIM

LOCAL_TZ = "America/Chicago"  # every ERCOT gas plant in EIA-860 2024

# Station choice and gap filling (see module docstring).
MIN_COVERAGE = 0.95
MAX_GAP_HOURS = 3
FAR_KM = 50.0
N_DONORS = 2
MAX_EXTRA_DONORS = 5
SEARCH_KM = 250.0
EXTRA_HOURS = 7  # grid runs to 06:00 UTC on 1 January of the next year
FILL_METHODS = ["observed", "interpolated", "donor", "interpolated_long", "missing"]

# Single-station plausibility checks for hourly dry-bulb temperature.
TEMP_MIN_C = -35.0
TEMP_MAX_C = 55.0
SPIKE_C = 10.0  # isolated jump vs. both neighbours, same sign

# Buddy check (see buddy_check).
BUDDY_NEIGHBOURS = 3
BUDDY_KM = 100.0
BUDDY_MIN_COVERAGE = 0.80
BUDDY_THRESHOLD_C = 4.0
BUDDY_MIN_DAYS = 3
BUDDY_EXTEND_C = 2.0
BUDDY_MIN_HOURS = 12


# --------------------------------------------------------------------------
# Hour grid and single-station checks
# --------------------------------------------------------------------------
def hour_grid(year: int, extra_hours: int = EXTRA_HOURS) -> pd.DatetimeIndex:
    """UTC hours from 1 January ``year`` 00:00 to ``extra_hours - 1`` h
    into the next year (tz-naive)."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1) + pd.Timedelta(hours=extra_hours - 1)
    return pd.date_range(start, end, freq="h", name="ts_utc").astype("datetime64[us]")


def to_grid(obs: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """Hourly temperature on ``grid`` from parsed ISD-lite rows.

    Rows without temperature are dropped first; if an hour still appears
    twice (unmerged duplicates), the first row in file order is kept.
    """
    obs = obs.dropna(subset=["temp_c"]).drop_duplicates("ts_utc", keep="first")
    return obs.set_index("ts_utc")["temp_c"].reindex(grid)


def qc_temperature(temp: pd.Series, tmin: float = TEMP_MIN_C, tmax: float = TEMP_MAX_C,
                   spike: float = SPIKE_C) -> tuple[pd.Series, dict[str, int]]:
    """Blank implausible hourly temperatures on a regular hourly series.

    Two checks: values outside ``[tmin, tmax]``, and isolated spikes, i.e.
    a value that differs from both its previous and next hour by more than
    ``spike`` degrees in the same direction (a front makes a step, not a
    one-hour spike). Returns the cleaned series and removal counts.
    """
    s = temp.astype("float64")
    out_of_range = s.notna() & ((s < tmin) | (s > tmax))
    s = s.mask(out_of_range)
    d_prev = s - s.shift(1)
    d_next = s - s.shift(-1)
    is_spike = ((d_prev > spike) & (d_next > spike)) | ((d_prev < -spike) & (d_next < -spike))
    s = s.mask(is_spike)
    return s, {"range": int(out_of_range.sum()), "spike": int(is_spike.sum())}


def coverage(temp: pd.Series, year: int) -> float:
    """Share of the calendar-year UTC hours with a valid temperature."""
    in_year = temp.index.year == year
    return float(temp[in_year].notna().mean())


# --------------------------------------------------------------------------
# Buddy check
# --------------------------------------------------------------------------
def neighbour_anomaly(target: pd.Series, neighbours: pd.DataFrame,
                      min_present: int = 2) -> pd.Series:
    """Target minus the neighbours' median, less its usual value.

    The reference is the hourly median of the neighbour columns (NaN when
    fewer than ``min_present`` report). The usual difference is the median
    of ``target - reference`` for the same calendar month and UTC hour of
    day, so a station that is always warmer at night (coast, city) has no
    anomaly from that alone.
    """
    present = neighbours.notna().sum(axis=1)
    ref = neighbours.median(axis=1).where(present >= min_present)
    diff = target - ref
    key = diff.index.month * 100 + diff.index.hour
    return diff - diff.groupby(key).transform("median")


def persistent_runs(anomaly: pd.Series, threshold: float = BUDDY_THRESHOLD_C,
                    min_days: int = BUDDY_MIN_DAYS, extend: float = BUDDY_EXTEND_C,
                    min_hours: int = BUDDY_MIN_HOURS) -> list[tuple]:
    """Runs of days whose mean anomaly stays beyond ``threshold``.

    Days are UTC calendar days. A run is at least ``min_days`` consecutive
    days, each with at least ``min_hours`` anomaly values and a daily mean
    above ``threshold`` (or below ``-threshold``). Each run is then
    extended over adjacent days whose mean has the same sign and exceeds
    ``extend`` in size (any number of hours). Returns tuples
    ``(first_day, last_day, sign, mean_anomaly)``.
    """
    day = anomaly.index.floor("D")
    daily = anomaly.groupby(day).agg(["mean", "count"])
    days = pd.date_range(day.min(), day.max(), freq="D")
    daily = daily.reindex(days)
    mean, count = daily["mean"].to_numpy(), daily["count"].fillna(0).to_numpy()
    runs = []
    for sign in (1, -1):
        core = (sign * np.nan_to_num(mean) > threshold) & (count >= min_hours)
        i = 0
        while i < len(days):
            if not core[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(days) and core[j + 1]:
                j += 1
            if j - i + 1 >= min_days:
                a, b = i, j
                while a > 0 and sign * np.nan_to_num(mean[a - 1]) > extend:
                    a -= 1
                while b + 1 < len(days) and sign * np.nan_to_num(mean[b + 1]) > extend:
                    b += 1
                runs.append((days[a], days[b], sign, float(np.nanmean(mean[a:b + 1]))))
            i = j + 1
    return sorted(runs)


def buddy_check(temps: pd.DataFrame, lat: Sequence[float], lon: Sequence[float],
                n_neighbours: int = BUDDY_NEIGHBOURS, max_km: float = BUDDY_KM,
                min_coverage: float = BUDDY_MIN_COVERAGE, iterations: int = 2,
                **run_kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Flag days on which a station departs persistently from its neighbours.

    ``temps`` has one column per station (hourly, after the single-station
    checks); ``lat``/``lon`` give the station coordinates in column order.
    Each station is compared with the median of its ``n_neighbours``
    nearest other stations within ``max_km`` that have at least
    ``min_coverage`` valid hours (:func:`neighbour_anomaly`); stations with
    fewer than two such neighbours are not checked. All hours of the days in
    :func:`persistent_runs` are flagged. The check is repeated
    ``iterations`` times with flagged hours removed from the neighbours, so
    a faulty station does not make its neighbours look faulty.

    Returns a boolean flag frame like ``temps``, a table of flagged
    episodes, the number of neighbours used per station and the hourly
    anomalies of the last iteration (NaN for unchecked stations).
    """
    ids = list(temps.columns)
    dist = haversine_km(np.asarray(lat)[:, None], np.asarray(lon)[:, None],
                        np.asarray(lat)[None, :], np.asarray(lon)[None, :])
    flags = pd.DataFrame(False, index=temps.index, columns=ids)
    episodes, n_used = [], pd.Series(0, index=ids, dtype="int64")
    anomalies = pd.DataFrame(np.nan, index=temps.index, columns=ids)
    for _ in range(iterations):
        masked = temps.mask(flags)
        eligible = (masked.notna().mean() >= min_coverage).to_numpy()
        new = pd.DataFrame(False, index=temps.index, columns=ids)
        episodes = []
        for i, sid in enumerate(ids):
            d = np.where(eligible, dist[i], np.inf)
            d[i] = np.inf
            nb = [j for j in np.argsort(d, kind="stable")[:n_neighbours] if d[j] <= max_km]
            n_used[sid] = len(nb)
            if len(nb) < 2:
                continue
            anomaly = neighbour_anomaly(temps[sid], masked.iloc[:, nb])
            anomalies[sid] = anomaly
            for first, last, sign, mean in persistent_runs(anomaly, **run_kwargs):
                new.loc[first:last + pd.Timedelta(hours=23), sid] = True
                episodes.append({"station_id": sid, "first_day": first, "last_day": last,
                                 "days": (last - first).days + 1, "sign": sign,
                                 "mean_anomaly_c": mean,
                                 "neighbours": ",".join(ids[j] for j in nb)})
        flags = new
    cols = ["station_id", "first_day", "last_day", "days", "sign", "mean_anomaly_c", "neighbours"]
    return flags, pd.DataFrame(episodes, columns=cols), n_used, anomalies


# --------------------------------------------------------------------------
# Station store
# --------------------------------------------------------------------------
class StationStore:
    """Downloads, parses, checks and caches station series for one year."""

    def __init__(self, year: int, grid: pd.DatetimeIndex, available_next: set[str]):
        self.year, self.grid = year, grid
        self.available_next = available_next
        self.with_next: set[str] = set()  # stations whose next-year file was read
        self.clean: dict[str, pd.Series] = {}  # after range and spike checks
        self.flags: dict[str, pd.Series] = {}  # buddy-check flags (calendar year)
        self.qc: dict[str, dict] = {}
        self.anomaly = pd.DataFrame()  # buddy-check anomalies (last run)

    def load(self, station_id: str, with_next: bool = False) -> pd.Series:
        """Series after the single-station checks (downloaded on first use).

        The next year's file, needed only for the grid hours after
        31 December, is read when ``with_next`` is set; until then those
        hours are NaN. Station choice and the buddy check use calendar-year
        hours only, so they do not depend on it.
        """
        want_next = (with_next and self.grid[-1].year > self.year
                     and station_id in self.available_next)
        if station_id not in self.clean or (want_next and station_id not in self.with_next):
            frames = [isd.parse_isd_lite(isd.isd_lite_path(station_id, self.year))]
            if want_next:
                frames.append(isd.parse_isd_lite(isd.isd_lite_path(station_id, self.year + 1)))
                self.with_next.add(station_id)
            raw = to_grid(pd.concat(frames, ignore_index=True), self.grid)
            clean, removed = qc_temperature(raw)
            self.clean[station_id] = clean
            old = self.qc.get(station_id, {})
            self.qc[station_id] = {"station_id": station_id,
                                   "raw_coverage": coverage(raw, self.year),
                                   "removed_range": removed["range"],
                                   "removed_spike": removed["spike"],
                                   "removed_buddy": old.get("removed_buddy", 0),
                                   "buddy_neighbours": old.get("buddy_neighbours", np.nan)}
        return self.clean[station_id]

    def _flagged(self, station_id: str, s: pd.Series) -> pd.Series:
        f = self.flags.get(station_id)
        return s if f is None else s.mask(f.reindex(s.index, fill_value=False))

    def get(self, station_id: str) -> pd.Series:
        """Full-grid series after all checks (buddy-flagged hours blanked)."""
        return self._flagged(station_id, self.load(station_id, with_next=True))

    def coverage_of(self, station_id: str) -> float:
        """Calendar-year coverage after all checks."""
        return coverage(self._flagged(station_id, self.load(station_id)), self.year)

    def run_buddy_check(self, stations: pd.DataFrame) -> pd.DataFrame:
        """Buddy-check every loaded station (calendar-year hours); returns
        the flagged episodes.

        ``stations`` maps ``station_id`` to ``lat`` and ``lon``.
        """
        ids = sorted(self.clean)
        coords = stations.set_index("station_id").loc[ids]
        in_year = self.grid.year == self.year
        temps = pd.DataFrame({s: self.clean[s][in_year] for s in ids})
        flags, episodes, n_used, self.anomaly = buddy_check(temps, coords["lat"], coords["lon"])
        self.flags = {s: flags[s] for s in ids}
        for s in ids:
            self.qc[s]["removed_buddy"] = int((flags[s] & temps[s].notna()).sum())
            self.qc[s]["buddy_neighbours"] = int(n_used[s])
        return episodes

    def qc_table(self) -> pd.DataFrame:
        """One row per loaded station: coverage before/after checks, removals."""
        rows = [{**self.qc[s], "coverage": self.coverage_of(s), "read_next_year": s in self.with_next}
                for s in sorted(self.qc)]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Station choice
# --------------------------------------------------------------------------
def good_stations(lat: float, lon: float, stations: pd.DataFrame,
                  coverage_of: Callable[[str], float], min_coverage: float = MIN_COVERAGE,
                  search_km: float = SEARCH_KM) -> Iterator[tuple[str, float, float]]:
    """Yield ``(station_id, distance_km, coverage)`` of good stations,
    nearest first, up to ``search_km`` from (``lat``, ``lon``).

    ``stations`` needs ``station_id``, ``lat``, ``lon``. ``coverage_of`` is
    called only as far as the caller iterates, so a caller that downloads
    on demand fetches only what it uses.
    """
    dist = haversine_km(lat, lon, stations["lat"].to_numpy(), stations["lon"].to_numpy())
    sid = stations["station_id"].to_numpy()
    for j in np.argsort(dist, kind="stable"):
        if dist[j] > search_km:
            return
        cov = coverage_of(sid[j])
        if cov >= min_coverage:
            yield sid[j], float(dist[j]), cov


def select_stations(plants: pd.DataFrame, stations: pd.DataFrame,
                    coverage_of: Callable[[str], float], n_donors: int = N_DONORS,
                    **kwargs) -> pd.DataFrame:
    """Primary (rank 0) and ``n_donors`` donor stations for each plant.

    ``plants`` needs ``plant_id_eia``, ``latitude``, ``longitude``. Returns
    ``plant_id_eia``, ``rank``, ``station_id``, ``distance_km``,
    ``coverage``; see :func:`good_stations`.
    """
    rows = []
    for p in plants.itertuples(index=False):
        found = islice(good_stations(p.latitude, p.longitude, stations, coverage_of, **kwargs),
                       1 + n_donors)
        rows += [{"plant_id_eia": p.plant_id_eia, "rank": r, "station_id": s,
                  "distance_km": d, "coverage": c} for r, (s, d, c) in enumerate(found)]
    return pd.DataFrame(rows, columns=["plant_id_eia", "rank", "station_id",
                                       "distance_km", "coverage"])


# --------------------------------------------------------------------------
# Gap filling
# --------------------------------------------------------------------------
def gap_lengths(isna: pd.Series) -> pd.Series:
    """Length of the run of missing values each position belongs to (0 if
    not missing)."""
    isna = isna.astype(bool)
    run_id = (isna != isna.shift()).cumsum()
    lengths = isna.groupby(run_id).transform("sum")
    return lengths.where(isna, 0).astype("int64")


def interpolate_short_gaps(s: pd.Series, max_gap: int = MAX_GAP_HOURS) -> pd.Series:
    """Linearly interpolate interior gaps of at most ``max_gap`` values.

    Longer gaps are left fully missing (not partly filled), and so are
    gaps at the start or end of the series. Assumes a regular index.
    """
    short = (gap_lengths(s.isna()) <= max_gap) & s.isna()
    interp = s.interpolate(method="linear", limit_area="inside")
    return s.where(~short, interp)


def donor_offset(target: pd.Series, donor: pd.Series, min_count: int = 10) -> pd.Series:
    """Mean ``target - donor`` by calendar month and UTC hour of day.

    Estimated on hours where both are observed and returned on the
    target's index. Cells with fewer than ``min_count`` pairs use the
    month mean, then the overall mean, then zero.
    """
    idx = target.index
    diff = (target - donor.reindex(idx)).dropna()
    cell = diff.groupby(diff.index.month * 100 + diff.index.hour).agg(["mean", "count"])
    month = diff.groupby(diff.index.month).agg(["mean", "count"])
    overall = float(diff.mean()) if len(diff) else 0.0
    cell_mean = cell.loc[cell["count"] >= min_count, "mean"]
    month_mean = month.loc[month["count"] >= min_count, "mean"]
    off = (pd.Series(idx.month * 100 + idx.hour).map(cell_mean)
           .fillna(pd.Series(idx.month).map(month_mean))
           .fillna(overall))
    return pd.Series(off.to_numpy(), index=idx, name="offset")


def interior_missing(temp: pd.Series) -> pd.Series:
    """True for missing values that have a valid value before and after."""
    valid = temp.notna()
    return ~valid & valid.cummax() & valid[::-1].cummax()[::-1]


def fill_plant_series(primary: pd.Series, primary_id: str, primary_km: float,
                      donors: Sequence[tuple[str, float, pd.Series]] = (),
                      max_gap: int = MAX_GAP_HOURS,
                      interpolate_rest: bool = True) -> pd.DataFrame:
    """Gap-filled temperature for one plant on the primary series' index.

    Order: observed primary value; linear interpolation over primary gaps
    of at most ``max_gap`` hours; then each donor in turn (its own short
    gaps interpolated) plus :func:`donor_offset`. If ``interpolate_rest``,
    interior gaps that no donor can fill are interpolated linearly
    whatever their length ("interpolated_long", no station). Hours still
    without a value (series ends) are "missing".

    Returns ``temp_c``, ``station_id`` and ``distance_km`` (of the station
    that supplied the value), ``fill_method`` ("observed", "interpolated",
    "donor", "interpolated_long", "missing") and ``filled`` (True unless
    observed).
    """
    temp = interpolate_short_gaps(primary, max_gap)
    method = pd.Series(np.where(primary.notna(), "observed",
                                np.where(temp.notna(), "interpolated", "missing")),
                       index=primary.index, dtype=object)
    station = pd.Series(primary_id, index=primary.index, dtype=object)
    distance = pd.Series(primary_km, index=primary.index, dtype="float64")
    for donor_id, donor_km, donor in donors:
        need = temp.isna()
        if not need.any():
            break
        donor = donor.reindex(primary.index)
        values = interpolate_short_gaps(donor, max_gap) + donor_offset(primary, donor)
        take = need & values.notna()
        temp = temp.where(~take, values)
        method[take] = "donor"
        station[take] = donor_id
        distance[take] = donor_km
    if interpolate_rest:
        rest = interior_missing(temp)
        temp = temp.where(~rest, temp.interpolate(method="linear", limit_area="inside"))
        method[rest] = "interpolated_long"
    no_station = method.isin(["interpolated_long", "missing"])
    station[no_station] = None
    distance[no_station] = np.nan
    return pd.DataFrame({"temp_c": temp, "station_id": station, "distance_km": distance,
                         "fill_method": method, "filled": method != "observed"})


# --------------------------------------------------------------------------
# Helpers for users of the hourly file
# --------------------------------------------------------------------------
def mean_diurnal_cycle(temp: pd.DataFrame, value: str = "temp_c", by: str = "plant_id_eia",
                       ts: str = "ts_utc", tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Mean ``value`` by local clock hour (rows: ``by``; columns: hour 0-23).

    ``ts`` is tz-naive UTC; local time follows ``tz`` including daylight
    saving time, i.e. the clock time at the plant.
    """
    local_hour = pd.DatetimeIndex(temp[ts]).tz_localize("UTC").tz_convert(tz).hour
    keys = [temp[by], pd.Series(local_hour, index=temp.index, name="local_hour")]
    return temp.groupby(keys)[value].mean().unstack("local_hour")


def hour_beginning_mean(temp: pd.DataFrame, value: str = "temp_c",
                        by: str = "plant_id_eia") -> pd.Series:
    """Mean of the observations at ``ts`` and ``ts + 1 h`` within each plant.

    Approximates the average over the hour that begins at ``ts`` (the CEMS
    convention). Rows must be sorted by plant and hour on a regular grid;
    the last hour of each plant gets NaN.
    """
    nxt = temp.groupby(by, sort=False)[value].shift(-1)
    return (temp[value] + nxt) / 2


# --------------------------------------------------------------------------
# Plants and build
# --------------------------------------------------------------------------
def plant_locations(year: int, plant_ids: Sequence[int]) -> pd.DataFrame:
    """EIA-860 coordinates and names for ``plant_ids`` in report ``year``."""
    from ttrl.data import pudl  # local import: keeps tests free of PUDL I/O

    path = pudl.download_table("out_eia__yearly_plants")
    cols = ["plant_id_eia", "plant_name_eia", "state", "county", "latitude",
            "longitude", "timezone", "report_date"]
    p = pd.read_parquet(path, columns=cols)
    p = p[pd.to_datetime(p["report_date"]).dt.year == year]
    p = p[p["plant_id_eia"].isin(list(plant_ids))].drop(columns="report_date")
    p = p.astype({"latitude": "float64", "longitude": "float64"})
    missing = set(plant_ids) - set(p["plant_id_eia"])
    if missing:
        raise ValueError(f"No EIA-860 {year} coordinates for plants {sorted(missing)}")
    return p.sort_values("plant_id_eia", ignore_index=True)


def output_paths(year: int, tag: str = "") -> dict[str, Path]:
    """Where :func:`build_plant_temperature` writes its outputs (``tag`` e.g. ``"_fleet2024"``)."""
    return {
        "temperature": INTERIM / f"plant_temperature_{year}{tag}.parquet",
        "stations": INTERIM / f"plant_weather_stations_{year}{tag}.parquet",
        "station_qc": INTERIM / f"weather_station_qc_{year}{tag}.parquet",
        "buddy_flags": INTERIM / f"weather_buddy_flags_{year}{tag}.parquet",
    }


def assign_and_fill(plants: pd.DataFrame, stations: pd.DataFrame,
                    store: StationStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose stations and fill the hourly series of every plant.

    Returns the hourly frame (all plants) and one summary row per plant.
    """
    frames, summary = [], []
    max_sources = 1 + N_DONORS + MAX_EXTRA_DONORS
    for plant in plants.itertuples(index=False):
        found = good_stations(plant.latitude, plant.longitude, stations, store.coverage_of)
        chosen = list(islice(found, 1 + N_DONORS))
        if not chosen:
            raise RuntimeError(f"No station with coverage >= {MIN_COVERAGE} within "
                               f"{SEARCH_KM} km of plant {plant.plant_id_eia}")

        def fill(chosen, rest):
            (pid, pkm, _), donors = chosen[0], chosen[1:]
            return fill_plant_series(store.get(pid), pid, pkm,
                                     [(s, d, store.get(s)) for s, d, _ in donors],
                                     interpolate_rest=rest)

        # Further donors only for interior gaps: the first grid hour
        # (1 January 00:00 UTC) is absent from most ISD-lite files, and
        # searching donors for that hour alone would fetch many files.
        f = fill(chosen, rest=False)
        while interior_missing(f["temp_c"]).any() and len(chosen) < max_sources:
            nxt = next(found, None)
            if nxt is None:
                break
            chosen.append(nxt)
            f = fill(chosen, rest=False)
        f = fill(chosen, rest=True)
        f.insert(0, "plant_id_eia", plant.plant_id_eia)
        frames.append(f.reset_index())

        counts = f["fill_method"].value_counts()
        row = {**plant._asdict(), "station_id": chosen[0][0], "distance_km": chosen[0][1],
               "coverage": chosen[0][2], "far": chosen[0][1] > FAR_KM}
        for k in range(1, N_DONORS + 1):
            s, d, _ = chosen[k] if k < len(chosen) else (None, np.nan, np.nan)
            row[f"donor{k}_id"], row[f"donor{k}_km"] = s, d
        row["extra_donors"] = ",".join(s for s, _, _ in chosen[1 + N_DONORS:])
        for m in FILL_METHODS:
            row[f"hours_{m}"] = int(counts.get(m, 0))
        summary.append(row)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summary)


def build_plant_temperature(year: int, plant_ids: Sequence[int], overwrite: bool = False,
                            max_passes: int = 4, verbose: bool = True, locations_year: int | None = None,
                            tag: str = "") -> dict[str, Path]:
    """Hourly gap-filled temperature for each plant; writes Parquet files.

    Passes: the first pass chooses stations with the single-station checks
    only; each later pass buddy-checks every station read so far and
    chooses and fills again, until a pass reads no new station file.

    Outputs (see :func:`output_paths`):

    - ``temperature``: ``plant_id_eia``, ``ts_utc``, ``temp_c``,
      ``station_id``, ``distance_km``, ``filled``, ``fill_method``.
    - ``stations``: one row per plant with coordinates, primary and donor
      stations, distances, coverage, the ``far`` flag (primary farther than
      ``FAR_KM``) and hour counts by fill method.
    - ``station_qc``: one row per station read (coverage, removals).
    - ``buddy_flags``: one row per flagged multi-day episode.

    ``locations_year`` takes plant coordinates from another EIA-860 year (weather years: the 2024
    fleet with an earlier year's weather); ``tag`` keeps those outputs apart.
    """
    paths = output_paths(year, tag)
    if not overwrite and all(p.exists() for p in paths.values()):
        return paths
    grid = hour_grid(year)
    plants = plant_locations(locations_year or year, plant_ids)
    stations = isd.station_history()
    avail = set(isd.isd_lite_inventory(year)["station_id"])
    avail_next = set(isd.isd_lite_inventory(year + 1)["station_id"])
    stations = stations[stations["station_id"].isin(avail)].reset_index(drop=True)
    store = StationStore(year, grid, avail_next)

    episodes = pd.DataFrame()
    for n in range(max_passes):
        n_before = len(store.clean)
        if n > 0:
            episodes = store.run_buddy_check(stations)
        temp, summary = assign_and_fill(plants, stations, store)
        if verbose:
            print(f"pass {n + 1}: {len(store.clean)} station files read, "
                  f"{len(episodes)} buddy-flagged episodes")
        if n > 0 and len(store.clean) == n_before:
            break
    else:
        print(f"warning: stations read in the last of {max_passes} passes were not buddy-checked")

    temp = temp[["plant_id_eia", "ts_utc", "temp_c", "station_id", "distance_km",
                 "filled", "fill_method"]].astype({
                     "plant_id_eia": "int64", "temp_c": "float32", "station_id": "string",
                     "distance_km": "float32", "filled": "bool", "fill_method": "category"})
    meta = stations.set_index("station_id")[["name", "lat", "lon", "elev_m"]]
    summary = summary.join(meta, on="station_id").rename(columns={
        "name": "station_name", "lat": "station_lat", "lon": "station_lon",
        "elev_m": "station_elev_m"})
    qc = store.qc_table().join(meta, on="station_id")

    INTERIM.mkdir(parents=True, exist_ok=True)
    temp.to_parquet(paths["temperature"], index=False)
    summary.to_parquet(paths["stations"], index=False)
    qc.to_parquet(paths["station_qc"], index=False)
    episodes.to_parquet(paths["buddy_flags"], index=False)
    return paths
