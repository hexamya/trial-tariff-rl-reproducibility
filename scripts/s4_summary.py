"""One-page summary of stage S4 for the paper (ERCOT gas fleet, 2024).

Collects the S4 results: the fleet, the unit curves, the temperature finding
(capacity derating and heat rate), the validation of the dispatch model, the
S4 error band and the worked example. The physical numbers are recomputed
here from the curve tables and CEMS with the estimators and seeds of
``scripts/fit_heatrate_derating.py`` (so they also check that step); the rest
is read from the JSON files written by ``scripts/integrate_s4.py`` and
``scripts/s4_worked_example.py``.

Writes outputs/s4_summary.md and a copy in docs/reports/.

Usage:
    python scripts/s4_summary.py [--year 2024] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import date

import numpy as np
import pandas as pd

from ttrl.data import pudl
from ttrl.paths import INTERIM, OUTPUTS, PROCESSED, ROOT
from ttrl.supply import heatrate as H
from ttrl.supply import units as U
from ttrl.supply.fleetmodel import build_fleet_data, station_distance

DOCS_REPORTS = ROOT / "docs" / "reports"
REPORT = "s4_summary.md"
HOT_SHARE = 0.01
GROUP_LABEL = {"CT": "CT (simple cycle)", "CC": "CC", "CC*": "CC* (CT output only)", "ST": "ST (steam)"}


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def ci(est: float, lo: float, hi: float, f: str = "{:.2f}") -> str:
    return f"{f.format(est)} [{f.format(lo)}, {f.format(hi)}]"


def physical(year: int, n_boot: int) -> dict:
    """Fleet, derating and heat-rate numbers recomputed from the curve tables and CEMS."""
    paths = pudl.interim_paths(year)
    cems, units = pd.read_parquet(paths["cems"]), pd.read_parquet(paths["units"])
    temp = pd.read_parquet(INTERIM / f"plant_temperature_{year}.parquet")
    fd = build_fleet_data(cems, units, temp, year)
    tab = pd.read_parquet(PROCESSED / f"s4_curves_{year}.parquet")        # aligned with fd.unit_ids
    raw = pd.read_parquet(PROCESSED / f"unit_curves_{year}.parquet")      # curves step, incl. CHP
    assert list(tab["unit_id"]) == fd.unit_ids
    out = {"n_units": len(tab), "n_chp": int(raw["chp"].sum()),
           "stations": station_distance(np.unique(fd.plant_id_eia), year)}

    # Fleet by group (nameplate MW, 2024 fuel, CO2, gross output).
    g = tab["group"].to_numpy()
    rows = []
    for grp in H.GROUPS:
        k = g == grp
        rows.append({"group": GROUP_LABEL[grp], "units": int(k.sum()),
                     "nameplate MW": f"{tab.loc[k, 'nameplate_mw'].sum():,.0f}",
                     "gross TWh": f"{fd.energy[:, k].sum() / 1e6:.1f}",
                     "fuel million MMBtu": f"{fd.heat[:, k].sum() / 1e6:.0f}"})
    out["fleet"] = pd.DataFrame(rows)
    out["totals"] = {"twh": fd.energy.sum() / 1e6, "fuel": fd.heat.sum() / 1e6,
                     "co2_mt": fd.co2.sum() * H.KG_PER_SHORT_TON / 1e9}

    # Technology medians (own fits, non-CHP), bootstrap over units, as the curves step.
    own_k = (raw["kappa_method"] == "own") & (raw["group"] != "CHP")
    own_h = raw["hr_method"].isin(["quadratic", "linear"]) & ~raw["const_heat"] & (raw["group"] != "CHP")
    rows = []
    for grp in H.GROUPS:
        xk = raw[own_k & (raw["group"] == grp)]
        xh = raw[own_h & (raw["group"] == grp)]
        sk = H.group_stats(xk["kappa"], xk["nameplate_mw"], n_boot=n_boot)
        sh = H.group_stats(xh["hr_pct_per_c"], xh["nameplate_mw"], n_boot=n_boot)
        rows.append({"group": GROUP_LABEL[grp],
                     "capacity loss, % of 15 C capability per C above 15 C (units)":
                         f"{ci(sk['median'], sk['median_lo'], sk['median_hi'])} ({sk['n']})",
                     "heat rate at equal output, % per C (units)":
                         f"{ci(sh['median'], sh['median_lo'], sh['median_hi'], '{:+.3f}')} ({sh['n']})"})
    out["tech"] = pd.DataFrame(rows)

    # Capacity lost in the hottest 1% of hours (capability-weighted fleet temperature).
    p15, kap = tab["p15"].to_numpy(float), tab["kappa"].to_numpy(float)
    ft = fd.fleet_temp(p15)
    n_hot = int(round(HOT_SHARE * len(ft)))
    hot = np.zeros(len(ft), bool)
    hot[np.argsort(ft)[::-1][:n_hot]] = True
    dt = np.maximum(fd.temp[hot] - H.T_REF, 0.0)
    per_unit = (p15 * kap / 100 * dt).mean(axis=0)
    draws = H.stratified_ratio_bootstrap(per_unit, p15, g, n_boot) * p15.sum()
    out["cap"] = {"lost": float(per_unit.sum()), "lo": float(np.quantile(draws, 0.025)),
                  "hi": float(np.quantile(draws, 0.975)), "p15": float(p15.sum()),
                  "n_hot": n_hot, "t_hot": float(ft[hot].mean())}

    # Extra fuel per MWh at the actual unit outputs of those hours vs 15 C (clean full hours,
    # capability-limited months left out), as in the curves step.
    uh = U.prepare_unit_hours(cems)
    rows_ok = U.fit_rows(uh) & uh["unit_id"].isin(fd.unit_ids)
    c = uh.loc[rows_ok, ["unit_id", "operating_datetime_utc", "month"]]
    ti = ((c["operating_datetime_utc"] - fd.hours[0]) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    ui = pd.Categorical(c["unit_id"], categories=fd.unit_ids).codes
    keep = (ti >= 0) & (ti < len(fd.hours))
    clean = np.zeros(fd.energy.shape, bool)
    clean[ti[keep], ui[keep]] = True
    lim = raw.set_index("unit_id")["limited_months"].reindex(fd.unit_ids).fillna("")
    limited = np.column_stack([np.isin(fd.month, [int(m) for m in str(s).split(",") if m]) for s in lim])
    m = clean[hot] & ~limited[hot]
    p = np.where(m, fd.energy[hot], 0.0)                 # clean full hours: energy = load
    d, e = tab["d"].to_numpy(float), tab["e"].to_numpy(float)
    dh = np.where(m, (d + e * p) * (fd.temp[hot] - H.T_REF), 0.0).sum(axis=0)
    mwh = p.sum(axis=0)
    co2 = dh * tab["co2_per_mmbtu"].to_numpy(float) * H.KG_PER_SHORT_TON
    use = mwh > 0
    dr = H.stratified_ratio_bootstrap(dh[use], mwh[use], g[use], n_boot)
    drc = H.stratified_ratio_bootstrap(co2[use], mwh[use], g[use], n_boot, seed=1)
    base = np.where(m, H.heat_input(tab, p, H.T_REF), 0.0).sum() / mwh.sum()
    out["avg"] = {"mmbtu": dh[use].sum() / mwh[use].sum(), "lo": float(np.quantile(dr, 0.025)),
                  "hi": float(np.quantile(dr, 0.975)), "kg": co2[use].sum() / mwh[use].sum(),
                  "kg_lo": float(np.quantile(drc, 0.025)), "kg_hi": float(np.quantile(drc, 0.975)),
                  "hr15": base}

    # Curves alone (dispatch step, variant A): Sep-Dec fuel with Jan-Aug curves at actual output.
    hd = pd.read_parquet(PROCESSED / f"dispatch_hourly_{year}.parquet")
    sd = hd["month_lst"].isin(range(9, 13))
    out["curves_only"] = {"fuel": 100 * (hd.loc[sd, "heat_A_train"].sum() / hd.loc[sd, "heat_actual"].sum() - 1),
                          "mape": 100 * float((hd.loc[sd, "heat_A_train"] / hd.loc[sd, "heat_actual"] - 1)
                                              .abs().mean())}
    out["fit_methods"] = raw[raw["group"] != "CHP"]["hr_method"].value_counts().to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    ph = physical(args.year, args.boot)
    ig = json.loads((OUTPUTS / "s4_integration.json").read_text())
    ex = json.loads((OUTPUTS / "s4_worked_example.json").read_text())
    cfg = ig["config"]
    lv = ig["levels"]
    sc = {r["hours"]: r for r in ig["band_season_cv"]}
    beta = {r["load tercile"]: r["fuel beta"] for r in ig["beta"] if r["model"] == "temperature-aware"}
    hv = ig["hot_value"][cfg]
    hh = ig["hottest_cv"]
    hot = ex["results"]["10 hottest days"]
    mild = ex["results"]["10 mild days"]
    share_key = next(k for k in hot if k.startswith("temperature-aware, gas share"))

    def eff(r: dict, key: str, f: str) -> str:
        x = r[key]
        return f"model {f.format(x['estimate'])}, corrected {ci(x['corrected'], x['lo'], x['hi'], f)}"

    L = []
    w = L.append
    w(f"# S4 summary: ERCOT gas fleet, ambient temperature and the fuel value of load shifting ({args.year})\n")
    w(f"Generated by `scripts/s4_summary.py` on {date.today().isoformat()}. Every number was produced by the "
      "S4 code (EPA CEMS via PUDL `" + pudl.DEFAULT_RELEASE + "`, EIA-860, EIA-930, NOAA ISD-lite); the "
      "physical numbers are recomputed here from the curve tables, the rest is read from the outputs of "
      "`scripts/integrate_s4.py` and `scripts/s4_worked_example.py`. Details: `docs/reports/` "
      "(`dispatch_validation.md`, `heatrate_derating.md`, `s4_integration.md`, `s4_worked_example.md`).\n")

    w("**1. Error band first.** The S4 model's marginal fuel per MWh of gas output, compared out of sample "
      "with the CEMS hour-to-hour slope (12-fold month cross-validation, week-block bootstrap, 95% bands), "
      f"is off by {sc['all hours']['temperature model error %']}% over all hours, "
      f"{sc['spring']['temperature model error %']}% in spring, {sc['fall']['temperature model error %']}% in "
      f"fall, {sc['winter']['temperature model error %']}% in winter, {sc['summer']['temperature model error %']}% "
      f"in summer; {sc['mild (10-25 C)']['temperature model error %']}% in mild hours (fleet 10-25 C) and "
      f"{sc['hot (>= 30 C)']['temperature model error %']}% in hot hours (>= 30 C). CEMS confirms "
      f"{beta['T1']} (low load) to {beta['T3']} (high load) of the model's hour-to-hour variation within a "
      "stratum. The impact function applies these errors hour by hour; a tariff effect counts only where "
      "its band excludes zero.\n")

    w(f"**2. Fleet.** {ph['n_units']} non-CHP ERCOT gas units with CEMS data ({ph['totals']['twh']:.0f} TWh "
      f"gross, {ph['totals']['fuel']:,.0f} million MMBtu, {ph['totals']['co2_mt']:.0f} Mt CO2 in {args.year}); "
      f"{ph['n_chp']} CHP units are left out because their fuel also makes process steam.\n")
    w(md_table(ph["fleet"]))
    w(f"\n**3. Curves.** Per-unit robust heat-input curves; temperature terms are own fits for "
      f"{ph['fit_methods'].get('quadratic', 0) + ph['fit_methods'].get('linear', 0)} units, group terms for "
      f"{ph['fit_methods'].get('group', 0)}, none for {ph['fit_methods'].get('none', 0)}. Evaluated at each "
      f"unit's actual output, Jan-Aug curves miss Sep-Dec non-CHP fuel by {ph['curves_only']['fuel']:+.2f}% "
      f"(hourly MAPE {ph['curves_only']['mape']:.2f}%).\n")

    a = ph["avg"]
    cp = ph["cap"]
    w("**4. Temperature finding (physical).** Above 15 C, technology medians over units with own fits "
      "(95% CI of the median, units resampled):\n")
    w(md_table(ph["tech"]))
    w(f"\nIn the hottest 1% of {args.year} hours ({cp['n_hot']} h, capability-weighted fleet temperature "
      f"{cp['t_hot']:.1f} C) the non-CHP units had {cp['lost']:,.0f} MW less capability than at 15 C "
      f"({cp['lo']:,.0f} to {cp['hi']:,.0f}), {100 * cp['lost'] / cp['p15']:.1f}% of {cp['p15']:,.0f} MW. At "
      f"their actual outputs the same hours needed {a['mmbtu']:.3f} MMBtu more fuel per MWh than at 15 C "
      f"({a['lo']:.3f} to {a['hi']:.3f}; {100 * a['mmbtu'] / a['hr15']:.1f}% of the heat rate), "
      f"{a['kg']:.1f} kg CO2/MWh ({a['kg_lo']:.1f} to {a['kg_hi']:.1f}); a fitted counterfactual whose CI "
      "resamples units only. At the margin the dispatch model adds "
      f"{hv['temperature']['fuel']:+.2f} MMBtu/MWh ({hv['pct']:+.1f}%) in those hours (derating alone "
      f"{hv['derate']['fuel']:+.2f}, heat-rate terms alone {hv['heat']['fuel']:+.2f}); out of sample it "
      "overstates the CEMS slope there by "
      f"{hh['temperature model error %']}% (temperature-free model {hh['temperature-free model error %']}%), "
      + ("so the hour-to-hour data do not confirm that temperature raises the fleet's marginal fuel per MWh.\n"
         if not ig["resolved"] else "and the temperature part exceeds that error band.\n"))

    t, s = lv[f"{cfg}|temperature"], lv[f"{cfg}|static"]
    qm = ig["quint_mae"]
    w(f"**5. Validation** (configuration {cfg}, chosen on Jan-Aug data; temperature-free / temperature-aware). "
      f"Fuel level: Sep-Dec out of sample {s['Sep-Dec fuel %']:+.1f}% / {t['Sep-Dec fuel %']:+.1f}%, CV "
      f"{s['CV fuel %']:+.1f}% / {t['CV fuel %']:+.1f}% (CO2 {s['CV CO2 %']:+.1f}% / {t['CV CO2 %']:+.1f}%); "
      f"hourly fuel MAPE (CV) {s['CV MAPE %']:.1f}% / {t['CV MAPE %']:.1f}%; in sample "
      f"{s['in-sample fuel %']:+.1f}% / {t['in-sample fuel %']:+.1f}%. Marginal fuel per MWh, pairs-weighted "
      f"MAE over load quintiles: Sep-Dec {qm['Sep-Dec (Jan-Aug fit)'][0]:.1f}% / "
      f"{qm['Sep-Dec (Jan-Aug fit)'][1]:.1f}%, CV {qm['CV (month folds)'][0]:.1f}% / "
      f"{qm['CV (month folds)'][1]:.1f}%. Simple-cycle CT unit-hours (CV): {s['CV CT hours %']:+.0f}% / "
      f"{t['CV CT hours %']:+.0f}%. The temperature terms "
      + ("improve" if ig["better"] else "do not improve")
      + " the model's fit to CEMS at the margin; the level is biased low, as in the dispatch step.\n")

    w(f"**6. Worked example (illustrative).** Moving {ex['mw']:,.0f} MW from 16-19 h to 22-01 h local time on "
      f"the 10 hottest days of {args.year} ({hot['mwh_shifted']:,.0f} MWh): fuel saved per MWh moved "
      f"{eff(hot['temperature-aware'], 'fuel_saved_per_mwh', '{:.2f}')} MMBtu; CO2 "
      f"{eff(hot['temperature-aware'], 'co2_kg_saved_per_mwh', '{:.0f}')} kg/MWh; simple-cycle CT unit-hours "
      f"{eff(hot['temperature-aware'], 'ct_unit_hours', '{:+.0f}')}; fleet heat rate "
      f"{eff(hot['temperature-aware'], 'fleet_heat_rate_pct', '{:+.3f}')}%. On 10 mild days: fuel saved "
      f"{eff(mild['temperature-aware'], 'fuel_saved_per_mwh', '{:+.2f}')} MMBtu/MWh. With the EIA-930 gas "
      f"share ({ex['gas_share_eia930']:.2f}) the hot-day saving is "
      f"{eff(hot[share_key], 'fuel_saved_per_mwh', '{:.2f}')} MMBtu per MWh of load moved. The corrected "
      "values remove the model's validated bias and the unconfirmed parts of its answers. On the hottest days "
      "the corrected fuel change splits (point values) into the CEMS slopes of the strata touched "
      f"({ex['correction_split_mmbtu']['10 hottest days']['cems']:+,.0f} MMBtu), the confirmed share beta of the "
      f"model's hour-level deviation ({ex['correction_split_mmbtu']['10 hottest days']['beta']:+,.0f} MMBtu) and "
      f"the confirmed share beta2 of its step-size part ({ex['correction_split_mmbtu']['10 hottest days']['beta2']:+,.0f} MMBtu). "
      "CT unit-hours are the least reliable output: the corrected CT change is set almost entirely by the CEMS "
      "hour-to-hour CT slopes.\n")

    w("**7. Limitations.**\n")
    for item in [
        "CHP units are excluded: their fuel also makes process steam, so they have no electric heat rate.",
        f"{ig['n_suspect']} CC units probably report only the combustion turbine's output (median heat rate >= "
        f"{U.FitConfig().suspect_cc_heat_rate} MMBtu/MWh gross; steam-turbine output missing from CEMS); their "
        "curves and derating are those of a gas turbine inside a combined cycle.",
        "Gas-marginal assumption: the load change is met by the non-CHP gas fleet (gas share 1 by default); "
        "in ERCOT coal and storage also follow load, and their fuel is not modelled.",
        "2024 only, one region; the band's strata, its beta step and the size classes of its step-size part "
        "were set after a first look at the validation results.",
        "Temperatures are dry-bulb readings at the nearest good ISD station (median "
        f"{ph['stations']['median']:.0f} km, at most {ph['stations']['max']:.0f} km from the "
        f"{ph['stations']['plants']} plants), not compressor-inlet conditions; humidity and inlet cooling "
        "are not modelled.",
        "Hourly, memoryless dispatch with one gas price; the CEMS benchmark is the hour-to-hour response, not "
        "the response to a planned, repeated tariff shift.",
        "The band scales the step-size part of the model's answer (curvature, commitment jumps) by the share "
        f"CEMS confirms for changes of that size; with {ex['half_step']['mw']:,.0f} MW on the same mild days the "
        "fuel saved per MWh moved is "
        f"{ci(*(ex['half_step']['results']['10 mild days']['fuel_saved_per_mwh'][k] for k in ('corrected', 'lo', 'hi')), '{:+.2f}')} "
        f"(at {ex['mw']:,.0f} MW: "
        f"{ci(*(mild['temperature-aware']['fuel_saved_per_mwh'][k] for k in ('corrected', 'lo', 'hi')), '{:+.2f}')}).",
    ]:
        w(f"- {item}")
    timing = OUTPUTS / "s4_pipeline_runtime.json"
    last = ""
    if timing.exists():
        tj = json.loads(timing.read_text())
        done = {k: v for k, v in tj.get("steps", {}).items() if k != "s4_summary"}
        if done:
            last = (f" The latest timings of its {len(done)} steps before this page add up to "
                    f"{sum(done.values()) / 60:.1f} min on 4 CPUs (downloads cached; "
                    "outputs/s4_pipeline_runtime.json).")
    w(f"\n**Reproduce.** `python scripts/run_s4.py` runs the whole S4 pipeline in order.{last} This page "
      f"alone: `python scripts/s4_summary.py` ({time.time() - t0:.0f} s).")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / REPORT).write_text("\n".join(L) + "\n")
    DOCS_REPORTS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUTS / REPORT, DOCS_REPORTS / REPORT)
    print(f"wrote {OUTPUTS / REPORT}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
