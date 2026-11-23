"""Hourly ambient temperature for every ERCOT gas plant (NOAA ISD-lite).

Builds data/interim/plant_temperature_<year>.parquet, plus a per-plant
station table, a per-station QC table and the buddy-check episodes, with
ttrl.data.weather; then writes outputs/weather_report.md and a copy in
docs/reports/.

Plants: every EIA plant in data/interim/ercot_gas_generators_<year>.parquet
(ERCOT gas CC / CT / ST generators in service), which includes all plants of
data/interim/ercot_gas_units_<year>.parquet (plants with CEMS units).

Usage:
    python scripts/build_plant_weather.py [--year 2024] [--overwrite]
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date

import numpy as np
import pandas as pd

from ttrl.data import isd
from ttrl.data import weather as w
from ttrl.paths import INTERIM, OUTPUTS, ROOT
from ttrl.reporting import md_table

REPORT_NAME = "weather_report.md"
SUMMER = [6, 7, 8, 9]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def plant_capacity(year: int) -> pd.DataFrame:
    """Nameplate MW of ERCOT gas generators per plant, and CEMS-unit flag."""
    gens = pd.read_parquet(INTERIM / f"ercot_gas_generators_{year}.parquet",
                           columns=["plant_id_eia", "capacity_mw", "tech"])
    units = pd.read_parquet(INTERIM / f"ercot_gas_units_{year}.parquet",
                            columns=["plant_id_eia"])
    cap = gens.groupby("plant_id_eia").agg(
        capacity_mw=("capacity_mw", "sum"),
        techs=("tech", lambda s: "+".join(sorted(set(s))))).reset_index()
    cap["has_cems_units"] = cap["plant_id_eia"].isin(set(units["plant_id_eia"]))
    return cap


def fleet_gross_output(year: int) -> pd.Series:
    """Hourly CEMS gross output of the linked ERCOT gas units (MWh per UTC hour)."""
    c = pd.read_parquet(INTERIM / f"cems_ercot_gas_{year}.parquet",
                        columns=["operating_datetime_utc", "gross_load_mw",
                                 "operating_time_hours"])
    mwh = c["gross_load_mw"].fillna(0) * c["operating_time_hours"].fillna(0)
    return mwh.groupby(c["operating_datetime_utc"].astype("datetime64[us]")).sum()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def quantile_row(label: str, x: pd.Series, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) -> dict:
    """One table row with selected quantiles of ``x``."""
    names = {0: "min", 1: "max"}
    return {"": label, "n": f"{int(x.notna().sum()):,}",
            **{names.get(q, f"P{int(q * 100)}"): float(x.quantile(q)) for q in qs}}


def as_text(df: pd.DataFrame, cols: list[str], fmt: str = "{:,}") -> pd.DataFrame:
    """Format integer-like columns as text so md_table prints them as such."""
    df = df.copy()
    for c in cols:
        df[c] = [fmt.format(int(v)) if pd.notna(v) else "" for v in df[c]]
    return df


def to_local(ts: pd.Series | pd.DatetimeIndex):
    """tz-naive UTC timestamps -> plant local clock time."""
    if isinstance(ts, pd.Series):
        return ts.dt.tz_localize("UTC").dt.tz_convert(w.LOCAL_TZ)
    return ts.tz_localize("UTC").tz_convert(w.LOCAL_TZ)


# --------------------------------------------------------------------------
# Checks of the gap filling
# --------------------------------------------------------------------------
def interpolation_holdout(series: dict[str, pd.Series], year: int, gap: int,
                          step: int = 97) -> pd.Series:
    """Errors of linear interpolation over artificial gaps of ``gap`` hours.

    In each station series, blocks of ``gap`` observed hours that start
    every ``step`` hours (97, so that blocks visit all hours of the day) and
    have observed neighbours are blanked and re-filled with
    :func:`ttrl.data.weather.interpolate_short_gaps`. Returns the
    interpolated-minus-observed differences pooled over stations.
    """
    errs = []
    for s in series.values():
        s = s[s.index.year == year]
        valid = s.notna().to_numpy()
        keep = [i for i in range(1, len(s) - gap - 1, step) if valid[i - 1:i + gap + 1].all()]
        idx = np.concatenate([np.arange(i, i + gap) for i in keep])
        masked = s.copy()
        masked.iloc[idx] = np.nan
        filled = w.interpolate_short_gaps(masked, max_gap=gap)
        errs.append((filled.iloc[idx] - s.iloc[idx]).to_numpy())
    return pd.Series(np.concatenate(errs))


def donor_check(stations: pd.DataFrame, series: dict[str, pd.Series],
                coords: dict[str, tuple[float, float]], year: int) -> pd.DataFrame:
    """First donor vs. primary on hours where both are observed (in-sample).

    Per plant: distance between the two stations and the MAE / RMSE of the
    raw donor and of the donor plus the month-by-hour offset used for gap
    filling.
    """
    rows = []
    for r in stations.dropna(subset=["donor1_id"]).itertuples():
        p = series[r.station_id]
        p = p[p.index.year == year]
        d = series[r.donor1_id].reindex(p.index)
        adj = d + w.donor_offset(p, d)
        both = p.notna() & d.notna()
        e_raw, e_adj = (d - p)[both], (adj - p)[both]
        rows.append({"plant_id_eia": r.plant_id_eia,
                     "pair_km": float(isd.haversine_km(*coords[r.station_id], *coords[r.donor1_id])),
                     "mae_raw": e_raw.abs().mean(), "mae_adj": e_adj.abs().mean(),
                     "rmse_adj": np.sqrt((e_adj ** 2).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(year: int, paths: dict, n_plants: int) -> str:
    temp = pd.read_parquet(paths["temperature"])
    st = pd.read_parquet(paths["stations"]).merge(plant_capacity(year), on="plant_id_eia", how="left")
    qc = pd.read_parquet(paths["station_qc"])
    saved_episodes = pd.read_parquet(paths["buddy_flags"])
    grid = w.hour_grid(year)
    in_year = temp["ts_utc"].dt.year == year
    total_mw = st["capacity_mw"].sum()

    # Re-create the checked station series from the cached files (no
    # download) and re-run the buddy check on the same station set.
    stations_meta = isd.station_history()
    store = w.StationStore(year, grid, set(isd.isd_lite_inventory(year + 1)["station_id"]))
    for s in qc["station_id"]:
        store.load(s)
    episodes = store.run_buddy_check(stations_meta)
    assert len(episodes) == len(saved_episodes), "buddy check does not reproduce the build"
    coords = dict(zip(qc["station_id"], zip(qc["lat"], qc["lon"])))
    primary_ids = set(st["station_id"])
    donor_ids = set(st[[f"donor{k}_id" for k in range(1, w.N_DONORS + 1)]].stack().dropna())
    extra_ids = {s for x in st["extra_donors"] for s in x.split(",") if s}
    used = sorted(primary_ids | donor_ids | extra_ids)
    series = {s: store.get(s) for s in used}

    # Station counts
    inv = isd.isd_lite_inventory(year)
    with_file = stations_meta[stations_meta["station_id"].isin(set(inv["station_id"]))]
    d_all = isd.haversine_km(st["latitude"].to_numpy()[:, None], st["longitude"].to_numpy()[:, None],
                             with_file["lat"].to_numpy()[None, :], with_file["lon"].to_numpy()[None, :])
    n_first = sum(int(pd.notna(store.load(sid).iloc[0])) for sid in qc["station_id"])
    files_y = [isd.isd_lite_path(sid, year) for sid in qc["station_id"]]
    files_n = [isd.isd_lite_path(sid, year + 1) for sid in qc.loc[qc["read_next_year"], "station_id"]]
    n_mb = sum(f.stat().st_size for f in files_y + files_n) / 1e6
    last_next = max((isd.parse_isd_lite(f)["ts_utc"].max() for f in files_n), default=None)
    good = qc["coverage"] >= w.MIN_COVERAGE
    checked = qc["buddy_neighbours"] >= 2

    # Buddy-check episodes
    ep = episodes.merge(qc[["station_id", "name"]], on="station_id", how="left")
    ep["role"] = ["primary" if s in primary_ids else "donor" if s in donor_ids | extra_ids
                  else "not used" for s in ep["station_id"]]
    ep["first_day"] = ep["first_day"].dt.strftime("%Y-%m-%d")
    ep = ep[["station_id", "name", "first_day", "days", "mean_anomaly_c", "role"]]
    ep = as_text(ep, ["days"])
    daily = store.anomaly[store.anomaly.index.year == year].resample("D").mean()
    flagged_days = pd.DataFrame({s: store.flags[s] for s in daily.columns}).resample("D").max()
    clean_daily = daily.where(~flagged_days.reindex(daily.index).fillna(False).astype(bool))
    abs_daily = clean_daily.stack().dropna().abs()

    # Fill shares
    fm, fm_all = temp.loc[in_year, "fill_method"].value_counts(), temp["fill_method"].value_counts()
    fill_tbl = pd.DataFrame({"fill_method": w.FILL_METHODS,
                             "plant-hours (UTC year)": [fm.get(k, 0) for k in w.FILL_METHODS],
                             "share %": [100 * fm.get(k, 0) / in_year.sum() for k in w.FILL_METHODS],
                             "plant-hours (full grid)": [fm_all.get(k, 0) for k in w.FILL_METHODS]})
    fill_tbl = as_text(fill_tbl, ["plant-hours (UTC year)", "plant-hours (full grid)"])
    st["filled_share"] = 1 - st["hours_observed"] / len(grid)
    n_extra = int((st["extra_donors"] != "").sum())
    miss = temp.loc[temp["fill_method"] == "missing", "ts_utc"].value_counts().sort_index()
    missing_hours = "; ".join(f"{ts:%Y-%m-%d %H:%M} x {n} plants" for ts, n in miss.items())
    donor_km = temp.loc[temp["fill_method"] == "donor", ["plant_id_eia", "distance_km"]]
    donor_far = donor_km[donor_km["distance_km"] > 100].groupby("plant_id_eia").size()
    # One-hour changes where the source of the value switches (seams of the fill).
    ts_sorted = temp.sort_values(["plant_id_eia", "ts_utc"])
    step = ts_sorted.groupby("plant_id_eia")["temp_c"].diff().abs()
    prev_m = ts_sorted.groupby("plant_id_eia")["fill_method"].shift().astype("string")
    switch = prev_m.notna() & (ts_sorted["fill_method"].astype("string") != prev_m)

    # Distances
    labels = ["0-10 km", "10-25 km", "25-50 km", "> 50 km"]
    st["dist_bin"] = pd.cut(st["distance_km"], [0, 10, 25, 50, np.inf], labels=labels)
    dist_tbl = st.groupby("dist_bin", observed=False).agg(
        plants=("plant_id_eia", "size"), plants_with_cems_units=("has_cems_units", "sum"),
        nameplate_MW=("capacity_mw", "sum")).reset_index().rename(columns={"dist_bin": "primary distance"})
    dist_tbl["share of MW %"] = 100 * dist_tbl["nameplate_MW"] / total_mw
    dist_tbl = as_text(dist_tbl, ["plants", "plants_with_cems_units", "nameplate_MW"])
    dist_q = pd.DataFrame([
        quantile_row("primary station (km)", st["distance_km"]),
        quantile_row("donor 1 (km)", st["donor1_km"]),
        quantile_row("donor 2 (km)", st["donor2_km"]),
        quantile_row("primary coverage after QC (%)", 100 * st["coverage"]),
    ])
    far = st[st["far"]].sort_values("distance_km", ascending=False)
    far_tbl = far[["plant_id_eia", "plant_name_eia", "techs", "capacity_mw", "station_id",
                   "station_name", "distance_km"]].assign(plant_id_eia=lambda d: d["plant_id_eia"].astype(str))

    # Gap-filling error checks
    holdout = []
    for gap in (1, 3):
        e = interpolation_holdout(series, year, gap)
        holdout.append({"check": f"linear interpolation, {gap}-h gap", "n hours": f"{len(e):,}",
                        "bias": e.mean(), "MAE": e.abs().mean(),
                        "RMSE": float(np.sqrt((e ** 2).mean())), "P95 abs error": e.abs().quantile(0.95)})
    dc = donor_check(st, series, coords, year)
    donor_tbl = pd.DataFrame([
        quantile_row("primary-to-donor distance (km)", dc["pair_km"]),
        quantile_row("MAE raw donor (C)", dc["mae_raw"]),
        quantile_row("MAE donor + offset (C)", dc["mae_adj"]),
        quantile_row("RMSE donor + offset (C)", dc["rmse_adj"]),
    ])

    # Summer temperatures (local time)
    t = temp.merge(st[["plant_id_eia", "capacity_mw"]], on="plant_id_eia")
    t["ts_local"] = to_local(t["ts_utc"])
    t_year = t[t["ts_local"].dt.year == year]
    summer = t_year[t_year["ts_local"].dt.month.isin(SUMMER)]
    per_plant = summer.groupby("plant_id_eia")["temp_c"].agg(
        max_c="max", p99_c=lambda s: s.quantile(0.99),
        h35=lambda s: (s >= 35).sum(), h40=lambda s: (s >= 40).sum())
    per_plant["mean_daily_max_c"] = (summer.groupby(["plant_id_eia", summer["ts_local"].dt.date])
                                     ["temp_c"].max().groupby("plant_id_eia").mean())
    summer_q = pd.DataFrame([
        quantile_row("summer maximum (C)", per_plant["max_c"]),
        quantile_row("summer P99 (C)", per_plant["p99_c"]),
        quantile_row("mean daily maximum (C)", per_plant["mean_daily_max_c"]),
        quantile_row("hours >= 35 C", per_plant["h35"].astype(float)),
        quantile_row("hours >= 40 C", per_plant["h40"].astype(float)),
    ])
    hottest = (per_plant.join(st.set_index("plant_id_eia")[["plant_name_eia", "techs", "capacity_mw",
                                                           "station_id", "distance_km"]])
               .sort_values(["max_c", "p99_c"], ascending=False).head(8).reset_index())
    hottest = as_text(hottest[["plant_id_eia", "plant_name_eia", "techs", "capacity_mw", "max_c", "p99_c",
                               "mean_daily_max_c", "h40", "station_id", "distance_km"]],
                      ["plant_id_eia", "h40"], fmt="{}")

    # Nameplate-weighted fleet temperature, and at the CEMS gas-output peaks
    ok = t["temp_c"].notna()
    fleet = ((t.loc[ok, "temp_c"] * t.loc[ok, "capacity_mw"]).groupby(t.loc[ok, "ts_utc"]).sum()
             / t.loc[ok, "capacity_mw"].groupby(t.loc[ok, "ts_utc"]).sum())
    fleet_y = fleet[fleet.index.year == year]
    fleet_local = pd.Series(fleet_y.to_numpy(), index=to_local(fleet_y.index))
    out = fleet_gross_output(year)
    top = out.sort_values(ascending=False).head(10).rename("gas gross MWh/h").to_frame()
    top["fleet temp (C)"] = fleet.reindex(top.index)
    top.insert(0, "local time", to_local(top.index).strftime("%Y-%m-%d %H:%M %Z"))
    top = top.reset_index(drop=True)
    both = pd.concat([out.rename("out"), fleet.rename("temp")], axis=1, sort=True).dropna()
    in_summer = to_local(both.index).month.isin(SUMMER)
    r_all = both["out"].corr(both["temp"])
    r_summer = both.loc[in_summer, "out"].corr(both.loc[in_summer, "temp"])

    # Diurnal sanity check
    peak_rows = []
    for label, months in [("all year", range(1, 13)), ("Jun-Sep", SUMMER), ("Dec-Feb", [12, 1, 2])]:
        cyc = w.mean_diurnal_cycle(t_year[t_year["ts_local"].dt.month.isin(list(months))])
        pk, mn = cyc.idxmax(axis=1), cyc.idxmin(axis=1)
        peak_rows.append({"period": label, "plants": str(len(cyc)),
                          "peak hour median": float(pk.median()),
                          "peak hour range": f"{int(pk.min())}-{int(pk.max())}",
                          "plants peaking 13-18 h": str(int(pk.between(13, 18).sum())),
                          "min hour median": float(mn.median()),
                          "mean amplitude (C)": float((cyc.max(axis=1) - cyc.min(axis=1)).mean())})
    obs = t_year[t_year["temp_c"].notna()]
    imax = obs.groupby(["plant_id_eia", obs["ts_local"].dt.date])["temp_c"].idxmax()
    share_afternoon = 100 * obs.loc[imax, "ts_local"].dt.hour.between(12, 18).mean()
    fleet_cycle = w.mean_diurnal_cycle(
        summer.assign(fleet=1), by="fleet").iloc[0]

    # ------------------------------------------------------------------ text
    n_cems = int(st["has_cems_units"].sum())
    L = [f"# Plant ambient temperature, ERCOT gas fleet, {year}\n",
         f"Generated by `scripts/build_plant_weather.py` on {date.today().isoformat()}. "
         "Every number below was computed by that script from the files it names.\n",
         "## 1. Data and method\n",
         f"- Plants: the {n_plants} EIA plants with an ERCOT gas CC / CT / ST generator in service in {year} "
         f"(`data/interim/ercot_gas_generators_{year}.parquet`, {total_mw:,.0f} MW nameplate); {n_cems} of them "
         f"have CEMS units in `ercot_gas_units_{year}.parquet`. Coordinates: PUDL `out_eia__yearly_plants`, "
         f"report year {year}. EIA-860 time zone of all {len(st)} plants: "
         f"{', '.join(sorted(st['timezone'].astype(str).unique()))}.",
         f"- Weather: NOAA ISD-lite on `s3://{isd.BUCKET}` (anonymous): station list `isd-history.csv` and "
         "`isd-lite/data/<year>/<USAF>-<WBAN>-<year>.gz`. Fixed-width hourly lines; temperature in tenths "
         "of a degree C; -9999 = missing; times UTC, each value the observation within the 10 minutes "
         "before the hour.",
         f"- Hour grid: {grid[0]:%Y-%m-%d %H:%M} to {grid[-1]:%Y-%m-%d %H:%M} UTC ({len(grid):,} hours): the "
         f"UTC calendar year plus the first {w.EXTRA_HOURS} hours of {year + 1} (from the stations' "
         f"{year + 1} files), so that every CEMS hour of the local-standard-time year (CEMS {year}: "
         f"{year}-01-01 06:00 to {year + 1}-01-01 05:00 UTC, hour-beginning) has an observation at both "
         "ends. `ts_utc` is tz-naive UTC like CEMS `operating_datetime_utc`.",
         f"- Single-station QC: values outside [{w.TEMP_MIN_C:g}, {w.TEMP_MAX_C:g}] C, and isolated one-hour "
         f"spikes (more than {w.SPIKE_C:g} C from both neighbouring hours, same sign), are removed. "
         "Duplicate hours keep the first row.",
         f"- Buddy check: each station is compared with the hourly median of its {w.BUDDY_NEIGHBOURS} nearest "
         f"other stations read (within {w.BUDDY_KM:g} km, >= {100 * w.BUDDY_MIN_COVERAGE:g}% valid hours; "
         "at least 2 needed), after removing the usual difference for that month and UTC hour (median). "
         f"Runs of >= {w.BUDDY_MIN_DAYS} consecutive UTC days whose mean anomaly exceeds "
         f"{w.BUDDY_THRESHOLD_C:g} C in the same direction (days with >= {w.BUDDY_MIN_HOURS} hours), "
         f"extended over adjacent days beyond {w.BUDDY_EXTEND_C:g} C, are removed whole. The check is run "
         "twice, the second time with flagged data left out of the neighbour medians.",
         f"- Station choice: stations with a {year} ISD-lite file are tried nearest first (great-circle "
         f"distance, up to {w.SEARCH_KM:g} km). A station is good if it has a valid temperature in "
         f">= {100 * w.MIN_COVERAGE:g}% of the {year} UTC hours after all QC. Primary = nearest good station; "
         f"donors = next {w.N_DONORS} nearest good stations; if hours are still missing, up to "
         f"{w.MAX_EXTRA_DONORS} further good stations are tried. Plants whose primary station is more than "
         f"{w.FAR_KM:g} km away are flagged (`far`). Selection and buddy check alternate until no new "
         "station file is needed.",
         f"- Gap filling: interior gaps of <= {w.MAX_GAP_HOURS} h in the primary series are linearly "
         "interpolated; other gaps take the first donor with a value, plus the mean primary-minus-donor "
         "difference for that calendar month and UTC hour (month or overall mean when a cell has fewer "
         "than 10 paired hours). Interior hours that no donor covers are interpolated linearly "
         "(`interpolated_long`); hours at the ends of the grid that no station covers stay `missing`. "
         "In the hourly file, `station_id` and `distance_km` name the station that supplied each value "
         "(empty for `interpolated_long` and `missing`); `filled` is True unless the value was observed "
         "at the primary station.",
         "- Outputs: " + ", ".join(f"`{p.relative_to(ROOT).as_posix()}`" for p in paths.values())
         + " (hourly values; one row per plant; one row per station read; buddy-check episodes).\n"]

    L += ["## 2. Stations\n",
          f"- ISD stations with a {year} ISD-lite file: {len(inv):,} ({len(with_file):,} with coordinates); "
          f"within {w.SEARCH_KM:g} km of a plant: {int((d_all <= w.SEARCH_KM).any(axis=0).sum())}; within "
          f"{w.FAR_KM:g} km: {int((d_all <= w.FAR_KM).any(axis=0).sum())}.",
          f"- Files downloaded: {len(files_y)} for {year} and {len(files_n)} for {year + 1} "
          f"({n_mb:.1f} MB gzip). Of the {len(qc)} stations read, {int(good.sum())} are good after QC; the "
          f"others have {100 * qc.loc[~good, 'coverage'].median():.1f}% coverage at the median "
          f"(max {100 * qc.loc[~good, 'coverage'].max():.1f}%).",
          f"- Stations used: {len(primary_ids)} as primary, {len(donor_ids)} as donor 1 or 2, "
          f"{len(extra_ids)} as extra donor; {len(used)} distinct.",
          f"- Single-station QC removed {int(qc['removed_range'].sum())} out-of-range values and "
          f"{int(qc['removed_spike'].sum())} spikes.",
          f"- Buddy check: {int(checked.sum())} of {len(qc)} stations had at least 2 neighbours and were "
          f"checked; {len(episodes)} episodes flagged at {episodes['station_id'].nunique()} stations, "
          f"{int(qc['removed_buddy'].sum()):,} hourly values removed. Outside flagged days, the absolute "
          f"daily-mean anomaly of checked stations has median {abs_daily.median():.2f} C, P99 "
          f"{abs_daily.quantile(0.99):.2f} C and P99.9 {abs_daily.quantile(0.999):.2f} C "
          f"({len(abs_daily):,} station-days). Primary stations not checked (fewer than 2 neighbours): "
          + (", ".join(f"{r.name} `{r.station_id}`" for r in qc[~checked & qc["station_id"].isin(primary_ids)]
                       .itertuples()) or "none") + ".\n"]
    if len(ep):
        L += [md_table(ep.rename(columns={"mean_anomaly_c": "mean anomaly (C)"}), floatfmt="{:+.1f}"), "",
              "Role is the station's role after the check. The data show the departures but not their cause; "
              "the flagged days are filled from donors.\n"]

    L += ["## 3. Distance and coverage\n",
          md_table(dist_q, floatfmt="{:.1f}"), "",
          md_table(dist_tbl, floatfmt="{:.0f}"), "",
          f"Nameplate is EIA-860 nameplate of the plant's ERCOT gas generators. Plants flagged `far` "
          f"(primary station > {w.FAR_KM:g} km): {len(far)}" + (":\n" if len(far) else ".\n")]
    if len(far):
        L += [md_table(far_tbl, floatfmt="{:.1f}"), ""]
    L += [f"Plant-hours by source ({len(st)} plants; UTC year = {int(in_year.sum() / len(st)):,} h per plant, "
          f"full grid = {len(grid):,} h):\n",
          md_table(fill_tbl, floatfmt="{:.2f}"), "",
          f"Per-plant share of non-observed hours (full grid): median {100 * st['filled_share'].median():.2f}%, "
          f"max {100 * st['filled_share'].max():.2f}%. Plants that needed extra donors: {n_extra}. "
          f"Plant-hours left missing: {int(fm_all.get('missing', 0))}"
          + (f" (UTC hours: {missing_hours})" if missing_hours else "") + ". "
          f"The first grid hour, {grid[0]:%Y-%m-%d %H:%M} UTC, has a value in {n_first} of the "
          f"{len(qc)} station files read; it precedes the first CEMS hour of {year}.\n",
          f"Distance of the station that supplied donor-filled plant-hours: median "
          f"{donor_km['distance_km'].median():.1f} km, P99 {donor_km['distance_km'].quantile(0.99):.1f} km, "
          f"max {donor_km['distance_km'].max():.1f} km; {int(donor_far.sum())} plant-hours came from stations "
          f"more than 100 km away"
          + (" (" + ", ".join(f"plant {p}: {n}" for p, n in donor_far.items()) + ")" if len(donor_far) else "")
          + ".\n"]

    L += ["## 4. Error of the gap filling\n",
          "Interpolation hold-out: in every station series used, blocks of 1 or 3 observed hours (starting "
          "every 97 h, neighbours observed) were blanked and re-interpolated. Error = interpolated minus "
          "observed, C.\n",
          md_table(pd.DataFrame(holdout), floatfmt="{:.2f}"), "",
          "Donor fill: first donor vs. primary station on hours where both are observed, one value per plant. "
          "In-sample: the offsets are estimated on the same hours, so the adjusted error is optimistic.\n",
          md_table(donor_tbl, floatfmt="{:.2f}"), "",
          f"Seams: at the {int(switch.sum()):,} plant-hours where the source of the value changes (e.g. "
          f"observed to donor), the absolute one-hour temperature change has P99 {step[switch].quantile(0.99):.1f} C "
          f"and maximum {step[switch].max():.1f} C; {int((step[switch] > 7).sum())} exceed 7 C. Over all "
          f"plant-hours the P99 is {step.quantile(0.99):.1f} C and P99.9 {step.quantile(0.999):.1f} C. A large "
          "seam means the month-by-hour donor offset did not match that hour's weather at the primary site. "
          f"Largest: plant {ts_sorted.loc[step[switch].idxmax(), 'plant_id_eia']} at "
          f"{ts_sorted.loc[step[switch].idxmax(), 'ts_utc']:%Y-%m-%d %H:%M} UTC "
          f"({ts_sorted.loc[step[switch].idxmax(), 'fill_method']}, station "
          f"{ts_sorted.loc[step[switch].idxmax(), 'station_id']}, "
          f"{ts_sorted.loc[step[switch].idxmax(), 'distance_km']:.0f} km).\n"]

    pk_ts = fleet_local.idxmax()
    L += [f"## 5. Summer temperatures (June-September {year}, local time)\n",
          f"Per-plant statistics of the gap-filled series ({len(per_plant)} plants):\n",
          md_table(summer_q, floatfmt="{:.1f}"), "",
          "Hottest plants by summer maximum:\n",
          md_table(hottest, floatfmt="{:.1f}"), "",
          f"Nameplate-weighted fleet temperature ({len(st)} plants): {year} maximum {fleet_local.max():.1f} C "
          f"at {pk_ts:%Y-%m-%d %H:%M %Z}; June-September mean "
          f"{fleet_local[fleet_local.index.month.isin(SUMMER)].mean():.1f} C; hours >= 35 C: "
          f"{int((fleet_y >= 35).sum())}, >= 38 C: {int((fleet_y >= 38).sum())}.\n",
          "Fleet temperature in the 10 hours of highest CEMS gross output of the linked gas units "
          "(sum of `gross_load_mw * operating_time_hours`; gross generation of these units, not ERCOT load):\n",
          md_table(top, floatfmt="{:,.1f}"), "",
          f"Correlation of hourly CEMS gas gross output with fleet temperature: {r_all:.2f} (all hours), "
          f"{r_summer:.2f} (June-September).\n"]

    L += ["## 6. Sanity check: diurnal cycle in local time\n",
          f"Mean temperature by local clock hour ({w.LOCAL_TZ}: CDT = UTC-5 in summer, CST = UTC-6 in winter), "
          "per plant; the peak hour is the hour with the highest mean.\n",
          md_table(pd.DataFrame(peak_rows), floatfmt="{:.1f}"), "",
          f"Share of plant-days whose daily maximum falls between 12:00 and 18:59 local time: "
          f"{share_afternoon:.1f}%. Fleet-mean June-September cycle: minimum {fleet_cycle.min():.1f} C at "
          f"{int(fleet_cycle.idxmin())}:00, maximum {fleet_cycle.max():.1f} C at "
          f"{int(fleet_cycle.idxmax())}:00 local.\n"]

    L += ["## 7. Assumptions and limitations\n",
          "- Airport (mostly ASOS/AWOS) dry-bulb temperature stands in for compressor-inlet air at the plant. "
          "EIA-860 gives no plant elevation, so no lapse-rate correction is applied; station elevation is in "
          "the station table.",
          "- Distances use EIA-860 plant coordinates and isd-history station coordinates (3 decimals). "
          "Coastal, urban and elevation contrasts between plant and station are not modelled.",
          "- Humidity is not used. Evaporative inlet cooling and cooling towers respond to wet-bulb "
          "temperature; the cached ISD-lite files also hold dew point (`isd.parse_isd_lite` returns it), so "
          "it can be added without new downloads.",
          "- `temp_c` is a point observation at `ts_utc`. CEMS hours are hour-beginning; "
          "`weather.hour_beginning_mean` averages the observations at both ends of an hour if that "
          "alignment is preferred.",
          "- ISD-lite has no quality flags. The QC here catches implausible values, one-hour spikes and "
          "multi-day biases against neighbours; shorter faults or slow drifts shared by neighbours would pass.",
          f"- The buddy-check thresholds ({w.BUDDY_THRESHOLD_C:g} C, {w.BUDDY_MIN_DAYS} days, "
          f"{w.BUDDY_EXTEND_C:g} C extension) were chosen after inspecting the {year} data (the July "
          "episode in section 2). They have not been tested on another year.",
          "- The donor offset is a climatological correction (month x hour of day); on a given day a donor "
          "can still differ from the primary site, e.g. when a front lies between them.",
          "- ISD-lite on S3 has complete years through 2024 only (checked October 2026)"
          + (f": the latest observation in the {len(files_n)} {year + 1} files read is "
             f"{last_next:%Y-%m-%d %H:%M} UTC" if last_next is not None else "")
          + ". Later years need another source (e.g. ERA5)."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--overwrite", action="store_true", help="rebuild the Parquet outputs")
    args = ap.parse_args()

    plant_ids = sorted(plant_capacity(args.year)["plant_id_eia"].unique())
    paths = w.build_plant_temperature(args.year, plant_ids, overwrite=args.overwrite)
    for k, p in paths.items():
        print(f"{k}: {p}")
    report = write_report(args.year, paths, len(plant_ids))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / REPORT_NAME
    out.write_text(report)
    docs = ROOT / "docs" / "reports" / REPORT_NAME
    shutil.copyfile(out, docs)
    print(f"report: {out}\ncopy:   {docs}")


if __name__ == "__main__":
    main()
