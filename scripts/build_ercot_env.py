"""ERCOT environment inputs per weather year, 2019-2024: hourly load, gas base and training surrogate.

Run:  python scripts/build_ercot_env.py   (~6 min; needs run_s4.py, build_ercot_weather_years.py and
      build_ercot_residential.py first)

For every weather year (ttrl.env.ercot): EIA-930 demand and the residential load, scaled to 2023
energy for the training years; the non-CHP gas output the 2024 S4 fleet starts from (EIA-930 gas hours
misreported under 'storage_other' corrected); plant temperatures at the 2024 fleet's plants (the S4
hour-beginning definition); the curtailed-demand days; and the band-corrected training surrogate
(central difference of the S4 model at +/- 500 MW plus the curvature of its response). Inputs are
saved to data/processed/ercot_env_<year>.parquet (ttrl.env.ercot.load_inputs).

Checks written to outputs/ercot_env_summary.md:

* per year: energy, gas base, the surrogate's marginal fuel, curvature and CO2, hours where the model
  cannot move (zero output floor), tariff days, masked (curtailed) days and days outside the kernel's
  support;
* surrogate against the full S4 evaluation (``S4Model.impact`` with its error band) on two fixed
  schedules applied with the kernel estimate to 14 tariff days from the 8th of every month of every
  year: weekday High 17-20 h and weekday High 14-20 h, at the kernel's size and doubled. The surrogate
  uses the band's point estimates, so it should fall inside the 95% band and close to its median. The
  first run (linear surrogate) overstated multi-GW summer savings by 2-5%; the curvature term was added
  after it, and both versions are reported.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ttrl.analysis import lcl_kernel as kn
from ttrl.env import ercot as er
from ttrl.env import ercot_env as ee
from ttrl.env import fastkernel as fk
from ttrl.env import tariff as tf
from ttrl.env.responses import TrialResponse
from ttrl.paths import OUTPUTS
from ttrl.reporting import md_table
from ttrl.supply.impact import S4Model

YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
ROLE = {2019: "training", 2020: "training", 2021: "training", 2022: "training", 2023: "validation", 2024: "test"}
CHECK_DAYS, CHECK_START = 14, 8
SCHEDULES = {"weekday High 17-20 h": (5,), "weekday High 14-20 h": (4, 5)}
SCALES = (1.0, 2.0)          # x2: step-size stress (as with the hot multiplier m = 2)


def schedule(days: ee.ErcotDays, idx: np.ndarray, high_blocks: tuple[int, ...]) -> np.ndarray:
    b = np.zeros((len(idx), tf.BLOCKS_PER_DAY), np.int8)
    weekday = days.tdays[idx].dayofweek < 5
    for k in high_blocks:
        b[weekday, k] = tf.HIGH
    return b


def window_profile(days: ee.ErcotDays, idx: np.ndarray, effect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UTC-hour rows of the window's tariff days (sorted) and the residential load change (MWh) in each."""
    rows, dl = [], []
    for k, d in enumerate(idx):
        eh = ee.hourly_effect(effect[k])
        for r in (days.row[d], days.row2[d]):
            ok = r >= 0
            rows.append(r[ok])
            dl.append(days.year.residential[r[ok]] * eh[ok] / 100)
    rows, dl = np.concatenate(rows), np.concatenate(dl)
    o = np.argsort(rows)
    return rows[o], dl[o]


def surrogate_check(model: S4Model, days: ee.ErcotDays, temps: np.ndarray, response: TrialResponse) -> list[dict]:
    y, s = days.year, days.surrogate
    out = []
    for mon in sorted(set(days.month)):
        idx = np.flatnonzero(days.month == mon)
        idx = idx[days.tdays[idx].day >= CHECK_START][:CHECK_DAYS]
        for name, blocks in SCHEDULES.items():
            rows, dl1 = window_profile(days, idx, response.effect(schedule(days, idx, blocks), 0))
            for scale in SCALES:
                dl = scale * dl1
                res = model.impact(dl, y.hours[rows], temp_c=temps[rows], base_mw=y.base_mw[rows])
                t = res.totals
                out.append({"year": y.year, "month": mon, "schedule": name, "scale": scale,
                            "max_abs_mw": float(np.abs(dl).max()), "net_mwh": float(dl.sum()),
                            "surrogate_fuel": float(s.fuel_change(rows, dl).sum()),
                            "linear_fuel": float((s.fuel[rows] * dl).sum()),
                            "s4_model_fuel": t.at["fuel_mmbtu", "estimate"],
                            "s4_corrected_fuel": t.at["fuel_mmbtu", "corrected"],
                            "s4_lo_fuel": t.at["fuel_mmbtu", "lo"], "s4_hi_fuel": t.at["fuel_mmbtu", "hi"],
                            "surrogate_co2_t": float(s.co2_change(rows, dl).sum()),
                            "s4_corrected_co2_t": t.at["co2_tonnes", "corrected"],
                            "s4_lo_co2_t": t.at["co2_tonnes", "lo"], "s4_hi_co2_t": t.at["co2_tonnes", "hi"],
                            "hot_days": int((days.temp_day[idx] > ee.KERNEL_TEMP_MAX).sum()),
                            "not_solved": res.meta["hours_not_solved"], "clipped": res.meta["hours_clipped_at_zero"]})
    return out


def year_row(year: int, y: er.ErcotYear, sur: er.Surrogate, days: ee.ErcotDays, ft: np.ndarray, n_gap: int) -> dict:
    hot = ft >= 30
    top = y.demand >= np.quantile(y.demand, 0.9)
    return {"year": year, "role": ROLE[year], "demand scale": y.scale, "demand TWh": y.demand.sum() / 1e6,
            "residential TWh": y.residential.sum() / 1e6, "gas base TWh": y.base_mw.sum() / 1e6,
            "gas hours corrected": er.gas_generation(year)[1],
            "fuel MMBtu/MWh": sur.fuel.mean(), "fuel, fleet >= 30 C": sur.fuel[hot].mean(),
            "fuel, top-10% demand h": sur.fuel[top].mean(), "model fuel (uncorrected)": np.nanmean(sur.model_fuel),
            "model curvature per GW, fleet >= 30 C": 1000 * sur.fuel_curv[hot].mean(),
            "model curvature per GW, other hours": 1000 * sur.fuel_curv[~hot].mean(),
            "hours curvature = 0": int((sur.fuel_curv == 0).sum()),
            "CO2 t/MWh": sur.co2_t.mean(), "hours model cannot move": int(np.isnan(sur.model_fuel).sum()),
            "temperature gap hours": n_gap, "tariff days": len(days.tdays), "masked days": int(days.curtailed.sum()),
            "days > 24.7 C": int((days.temp_day > ee.KERNEL_TEMP_MAX).sum())}


def summarise(ch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    def inside(v, lo, hi):
        return (ch[v] >= ch[lo]) & (ch[v] <= ch[hi])
    ch["inside_fuel"] = inside("surrogate_fuel", "s4_lo_fuel", "s4_hi_fuel")
    ch["inside_linear"] = inside("linear_fuel", "s4_lo_fuel", "s4_hi_fuel")
    ch["inside_model"] = inside("s4_model_fuel", "s4_lo_fuel", "s4_hi_fuel")
    ch["inside_co2"] = inside("surrogate_co2_t", "s4_lo_co2_t", "s4_hi_co2_t")
    ch["gap %"] = 100 * (ch["surrogate_fuel"] / ch["s4_corrected_fuel"] - 1)
    ch["linear gap %"] = 100 * (ch["linear_fuel"] / ch["s4_corrected_fuel"] - 1)
    ch["band width %"] = 100 * (ch["s4_hi_fuel"] - ch["s4_lo_fuel"]) / ch["s4_corrected_fuel"].abs()
    ch["summer"] = ch["month"].str[5:].isin(["06", "07", "08", "09"])

    def worst(g):
        return g.loc[g.abs().idxmax()]
    by_year = ch.groupby(["scale", "year"]).agg(
        windows=("month", "size"), max_hourly_MW=("max_abs_mw", "max"),
        inside=("inside_fuel", "mean"), inside_linear=("inside_linear", "mean"), inside_model=("inside_model", "mean"),
        co2_inside=("inside_co2", "mean"), median_gap_pct=("gap %", "median"),
        median_linear_gap_pct=("linear gap %", "median"), worst_gap_pct=("gap %", worst),
        median_band_width_pct=("band width %", "median")).reset_index()
    by_season = ch.groupby(["scale", "summer"]).agg(
        windows=("month", "size"), inside=("inside_fuel", "mean"), inside_linear=("inside_linear", "mean"),
        median_gap_pct=("gap %", "median"), median_linear_gap_pct=("linear gap %", "median"),
        worst_gap_pct=("gap %", worst), worst_linear_gap_pct=("linear gap %", worst)).reset_index()
    corr = float(np.corrcoef(ch["surrogate_fuel"], ch["s4_corrected_fuel"])[0, 1])
    return by_year, by_season, corr


def main() -> None:
    t0 = time.time()
    model = S4Model.load()
    s4_target = pd.Series(model.data.target, index=model.data.hours)
    spec = kn.KernelSpec.from_json((OUTPUTS / "lcl_kernel_spec.json").read_text(encoding="utf-8"))
    terms = pd.read_csv(OUTPUTS / "lcl_kernel_terms.csv").set_index("term")["coef"]
    response = TrialResponse(fk.from_coefficients(spec, terms))
    rows, checks = [], []
    for year in YEARS:
        y = er.build_year(year, s4_target, set(er.curtailed_days(year)["date"]))
        temps, n_gap = er.plant_temperature_matrix(y.temp_path, y.hours, model.data.plant_id_eia)
        sur = er.build_surrogate(model, y, temps)
        temp_c = pd.read_parquet(er.PROCESSED / f"ercot_residential_{year}.parquet").set_index("datetime_utc")[
            "temp_c"].reindex(y.hours).to_numpy()
        er.save_inputs(y, sur, temp_c)
        days = ee.ErcotDays.build(y, sur, temp_c)
        ft = (temps * model.weights).sum(axis=1) / model.weights.sum()
        rows.append(year_row(year, y, sur, days, ft, n_gap))
        checks += surrogate_check(model, days, temps, response)
        print(f"{year}: done ({time.time() - t0:.0f} s)", flush=True)
    tab = pd.DataFrame(rows)
    ch = pd.DataFrame(checks)
    by_year, by_season, corr = summarise(ch)
    ch.to_csv(OUTPUTS / "ercot_env_surrogate_check.csv", index=False)
    report = "\n".join([
        "# ERCOT environment inputs, 2019-2024 (S2)",
        "",
        "Generated by `python scripts/build_ercot_env.py`. The 2024 gas fleet is run against each weather year. "
        "Training years' demand and residential load are scaled to "
        "2023 energy; gas base = EIA-930 gas generation x the 2024 ratio of S4 non-CHP output to it (month x hour). "
        "Surrogate = S4 central difference at +/- 500 MW, band-corrected per hour with the band's point estimates, "
        "plus the curvature of the model's response fitted at +/- 1, 2 and 4 GW, scaled by the band's confirmed share "
        "of the step-size part (beta2, interpolated by the size of the change; `ttrl.env.ercot`). Gas "
        "hours misreported under 'storage_other' in EIA-930 (December 2019, early 2020) are corrected "
        "(`ercot.gas_generation`).",
        "",
        "## Per year",
        "",
        md_table(tab, floatfmt="{:,.2f}"),
        "",
        "'fuel' columns: mean over hours of the surrogate's marginal fuel per MWh of load change (fleet temperature "
        "is the P15-weighted plant mean); 'model fuel': the uncorrected central difference. 'model curvature per GW': "
        "q x 1000 before the beta2 scaling, the fall of the model's marginal fuel (MMBtu/MWh) per GW of load removed, "
        "mean over hours. "
        "'hours model cannot move': the fleet sits at zero dispatchable output, so the surrogate takes the stratum's "
        "CEMS marginal. 'masked days': tariff days touching a curtailed-demand day (masked in training and "
        "validation, kept in the test year). 'days > 24.7 C': four-station daily mean above the kernel's support.",
        "",
        "## Surrogate against the full S4 evaluation",
        "",
        f"Kernel estimate (no ensemble draw, no hot multiplier) on 14 tariff days from the 8th of each month, two "
        f"schedules, at the kernel's size (scale 1) and doubled (scale 2, a step-size stress); {len(ch)} windows. "
        "'inside' = share of windows where the surrogate's window fuel lies inside the S4 95% band; 'inside_linear' = "
        "the same for the surrogate without its curvature term; 'inside_model' = the uncorrected S4 model estimate. "
        "Gaps are surrogate / S4 corrected - 1 (fuel; for load reductions a positive gap overstates the saving). "
        f"Correlation of surrogate and S4 corrected fuel over all windows: {corr:.4f}.",
        "",
        md_table(by_year, floatfmt="{:,.2f}"),
        "",
        "By season (June-September vs other months):",
        "",
        md_table(by_season, floatfmt="{:,.2f}"),
        "",
        f"Hours not solved by the dispatch in the check windows: {int(ch['not_solved'].sum())}; hours clipped at zero "
        f"output: {int(ch['clipped'].sum())}. Window-level results: `outputs/ercot_env_surrogate_check.csv`.",
    ])
    (OUTPUTS / "ercot_env_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
