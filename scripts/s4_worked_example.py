"""Worked example (illustrative) of the S4 impact function for the RL environment.

Shifts 500 MW of load from 16:00-19:00 to 22:00-01:00 local time
(America/Chicago) on the 10 hottest days of 2024 and on 10 mild days, and
reports the change of non-CHP gas fuel, CO2, simple-cycle CT hours and fleet
heat rate with the S4 error band. Uses only the public API
(``ttrl.supply.impact``), the way the tariff environment will.

Day selection (local calendar days whose shift hours all lie in the CEMS
2024 year): fleet temperature = P15-weighted mean of the plant temperatures
(as in the error band); hottest days = the 10 highest daily maxima; mild
days = the 10 days whose daily maximum is closest to 20 C.

Writes outputs/s4_worked_example.md (+ docs/reports copy) and
outputs/s4_worked_example.json. Needs ``scripts/integrate_s4.py`` first.

Usage:
    python scripts/s4_worked_example.py [--mw 500] [--days 10]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import date

import numpy as np
import pandas as pd

from ttrl.paths import OUTPUTS, ROOT
from ttrl.supply.impact import (TZ, S4Model, compare, gas_response_share, local_hours,
                                shift_profile)

DOCS_REPORTS = ROOT / "docs" / "reports"
REPORT = "s4_worked_example.md"
FROM_H, TO_H = (16, 19), (22, 1)
MILD_TARGET_C = 20.0


def md_table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    """GitHub markdown table with per-column float formats."""
    formats = formats or {}

    def fmt(col, v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else formats.get(col, "{:,.2f}").format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, r)) + " |"
                                    for r in df.itertuples(index=False)])


def daily_max_temperature(model: S4Model) -> pd.Series:
    """Daily maximum fleet temperature (C) by local calendar day."""
    ft = model.data.fleet_temp(model.weights)
    local = model.data.hours.tz_localize("UTC").tz_convert(TZ)
    return pd.Series(ft, index=local).groupby(local.date).max()


def eligible(days, model: S4Model) -> list:
    """Days whose shift hours are all inside the model year."""
    grid = model.data.hours
    out = []
    for d in days:
        try:
            hrs = local_hours([d], *FROM_H).append(local_hours([d], *TO_H))
        except Exception:      # ambiguous or missing local hour (DST change)
            continue
        if grid.get_indexer(hrs).min() >= 0:
            out.append(d)
    return out


def summarise(res, mwh_shifted: float) -> dict:
    """Totals and per-MWh-shifted values (savings positive) with bands."""
    t = res.totals
    out = {}
    for m in t.index:
        out[m] = {k: float(t.at[m, k]) for k in ("estimate", "corrected", "lo", "hi")}
    f = t.loc["fuel_mmbtu"]
    c = t.loc["co2_tonnes"]
    out["fuel_saved_per_mwh"] = {k: -float(f[k]) / mwh_shifted for k in ("estimate", "corrected")}
    out["fuel_saved_per_mwh"].update(lo=-float(f["hi"]) / mwh_shifted, hi=-float(f["lo"]) / mwh_shifted)
    out["co2_kg_saved_per_mwh"] = {k: -1000 * float(c[k]) / mwh_shifted for k in ("estimate", "corrected")}
    out["co2_kg_saved_per_mwh"].update(lo=-1000 * float(c["hi"]) / mwh_shifted,
                                       hi=-1000 * float(c["lo"]) / mwh_shifted)
    out["excludes_zero"] = {m: bool(t.at[m, "lo"] > 0 or t.at[m, "hi"] < 0) for m in t.index}
    out["meta"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                   for k, v in res.meta.items()}
    return out


def decompose(band, hourly: pd.DataFrame) -> dict:
    """Point-value split of the corrected fuel change (MMBtu) of a profile.

    ``cems``: sum of dE_t x CEMS stratum marginal; ``beta``: confirmed share
    beta (point) x the model's hour-level deviation (its +/- 500 MW marginal
    minus the stratum mean, times dE_t); ``beta2``: beta2 (point) of each
    hour's size class x the rest of the model's answer (step-size part). Their
    sum is the corrected value at the point estimates (the reported
    'corrected' is the median over draws).
    """
    from ttrl.supply.errorband import size_class
    st = np.array([band.labels.index(s) for s in hourly["stratum"]])
    de = hourly["d_gas_mwh"].to_numpy(float)
    dx = hourly["d_fuel_mmbtu"].to_numpy(float)
    cems_slope = band.m_bar[:, 0] - band.point[:, 0]
    mh = hourly["model_marginal_fuel"].to_numpy(float)
    mh = np.where(np.isfinite(mh), mh, band.m_bar[st, 0])
    dev1, dev2 = de * (mh - band.m_bar[st, 0]), dx - de * mh
    g, c = st % band.n_load, size_class(de, band.size_edges)
    beta = sum(float(band.beta_point[k, 0]) * float(dev1[g == k].sum()) for k in range(band.n_load))
    beta2 = sum(float(band.beta2_point[k, 0]) * float(dev2[c == k].sum()) for k in range(len(band.size_edges) + 1))
    return {"cems": float((de * cems_slope[st]).sum()), "beta": beta, "beta2": beta2}


def ci(d: dict, f: str = "{:+,.0f}") -> str:
    """'corrected [lo, hi]'."""
    return f"{f.format(d['corrected'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


METRIC_ROWS = [("fuel_mmbtu", "fuel, MMBtu", "{:+,.0f}"),
               ("fuel_saved_per_mwh", "fuel saved per MWh moved, MMBtu/MWh", "{:.2f}"),
               ("co2_tonnes", "CO2, t", "{:+,.0f}"), ("co2_short_tons", "CO2, short tons", "{:+,.0f}"),
               ("co2_kg_saved_per_mwh", "CO2 saved per MWh moved, kg/MWh", "{:.0f}"),
               ("ct_unit_hours", "simple-cycle CT unit-hours", "{:+.0f}"),
               ("fleet_heat_rate_mmbtu_per_mwh", "fleet heat rate, MMBtu/MWh", "{:+.4f}"),
               ("fleet_heat_rate_pct", "fleet heat rate, %", "{:+.3f}")]


def direction(d: dict) -> str:
    """Sign of an effect whose band excludes zero, else 'no effect beyond the band'."""
    if d["hi"] < 0:
        return "a decrease"
    if d["lo"] > 0:
        return "an increase"
    return "no change distinguishable from model error"


def set_table(res: dict) -> pd.DataFrame:
    """Rows = metrics; columns: model estimate and corrected [band] per model run."""
    rows = []
    for key, label, f in METRIC_ROWS:
        row = {"quantity": label}
        for run, s in res.items():
            if not isinstance(s, dict) or key not in s:
                continue
            row[f"{run}: model"] = f.format(s[key]["estimate"])
            row[f"{run}: corrected [95% band]"] = ci(s[key], f)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mw", type=float, default=500.0)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()
    t0 = time.time()
    model = S4Model.load(args.year)                       # temperature-aware, CV band
    static = S4Model.load(args.year, mode="static")       # temperature-free comparison
    print(f"models loaded {time.time() - t0:.0f} s")

    tmax = daily_max_temperature(model)
    ok = eligible(tmax.index, model)
    tmax = tmax.loc[ok]
    hot_days = list(tmax.sort_values(ascending=False).index[:args.days])
    mild_days = list((tmax - MILD_TARGET_C).abs().sort_values(kind="stable").index[:args.days])
    sets = {"10 hottest days": sorted(hot_days), "10 mild days": sorted(mild_days)}

    share, share_se = gas_response_share(model.data)
    results, per_day, strata, hourly_main = {}, [], [], {}
    for name, days in sets.items():
        prof = shift_profile(days, args.mw, FROM_H, TO_H)
        mwh = float(-prof[prof < 0].sum())
        runs = {"temperature-aware": model.impact(prof.to_numpy(), prof.index),
                "temperature-free": static.impact(prof.to_numpy(), prof.index),
                f"temperature-aware, gas share {share:.2f}":
                    model.impact(prof.to_numpy(), prof.index, gas_share=share)}
        results[name] = {k: summarise(v, mwh) for k, v in runs.items()}
        results[name]["mwh_shifted"] = mwh
        results[name]["days"] = [str(d) for d in days]
        main_res = runs["temperature-aware"]
        h = main_res.hourly
        local = pd.DatetimeIndex(h["hour_utc"]).tz_localize("UTC").tz_convert(TZ)
        h = h.assign(date=[(t - pd.Timedelta(hours=2)).date() for t in local],   # 00:00 belongs to the evening before
                     window=np.where(h["delta_mw"] < 0, "peak", "evening"))
        for d, x in h.groupby("date"):
            pk, ev = x[x["window"] == "peak"], x[x["window"] == "evening"]
            per_day.append({"days": name, "date": str(d), "daily max C": float(tmax.loc[d]),
                            "16-19 h C": pk["fleet_temp_c"].mean(), "22-01 h C": ev["fleet_temp_c"].mean(),
                            "16-19 h GW": pk["base_mw"].mean() / 1e3, "22-01 h GW": ev["base_mw"].mean() / 1e3,
                            "fuel MMBtu (model)": x["d_fuel_mmbtu"].sum(),
                            "CT unit-hours (model)": x["d_ct_units"].sum()})
        st = h.groupby(["window", "stratum"])["d_gas_mwh"].sum().reset_index()
        for _, r in st.iterrows():
            strata.append({"days": name, "window": r["window"], "stratum": r["stratum"], "MWh": r["d_gas_mwh"]})
        results[name]["_res"] = main_res
        hourly_main[name] = h

    diff = compare(results["10 hottest days"]["_res"], results["10 mild days"]["_res"])
    dec = {name: decompose(model.band, hourly_main[name]) for name in sets}
    # Step-size check: the same days with half the shift.
    half = {}
    for name, days in sets.items():
        prof = shift_profile(days, args.mw / 2, FROM_H, TO_H)
        half[name] = summarise(model.impact(prof.to_numpy(), prof.index), float(-prof[prof < 0].sum()))
    for name in sets:
        results[name].pop("_res")
    hot, mild = results["10 hottest days"], results["10 mild days"]
    temp_part = (hot["temperature-aware"]["fuel_saved_per_mwh"]["estimate"]
                 - hot["temperature-free"]["fuel_saved_per_mwh"]["estimate"])
    temp_part_corr = (hot["temperature-aware"]["fuel_saved_per_mwh"]["corrected"]
                      - hot["temperature-free"]["fuel_saved_per_mwh"]["corrected"])
    mwh = hot["mwh_shifted"]

    pdf = pd.DataFrame(per_day)

    # Report.
    L = []
    w = L.append
    w(f"# S4 worked example (illustrative): shifting {args.mw:,.0f} MW out of the evening peak, ERCOT 2024\n")
    w(f"Generated by `scripts/s4_worked_example.py` on {date.today().isoformat()} with "
      "`ttrl.supply.impact.S4Model`. **Illustrative only**: a fixed, made-up load shift on 2024 days, not a "
      "tariff result. Every number below was computed by that script.\n")
    w("## Set-up\n")
    w(f"- Shift: -{args.mw:,.0f} MW in the local hours {FROM_H[0]}:00-{FROM_H[1]}:00 and +{args.mw:,.0f} MW in "
      f"{TO_H[0]}:00-{TO_H[1]:02d}:00 ({TZ}, so CDT in summer) on each day: {mwh / len(hot['days']):,.0f} MWh "
      f"moved per day, {mwh:,.0f} MWh per set of {args.days} days; energy is conserved.")
    w("- Days: local calendar days of 2024 whose shift hours lie in the CEMS 2024 year. Fleet temperature = "
      "P15-weighted mean plant temperature. Hottest = the "
      f"{args.days} highest daily maxima; mild = the {args.days} days whose daily maximum is closest to "
      f"{MILD_TARGET_C:.0f} C.")
    w(f"- Hottest days: {', '.join(hot['days'])}. Mild days: {', '.join(mild['days'])}.")
    w("- Baseline: actual 2024 non-CHP gas output and plant temperatures. Model: full-year fit, "
      f"configuration {model.meta['config']}, temperature-aware (main) and temperature-free. Gas share 1 "
      f"(the whole shift moves non-CHP gas output; the gas-marginal assumption) and, as a sensitivity, "
      f"{share:.3f} (HAC SE {share_se:.3f}): the change of non-CHP CEMS gas output per MWh change of ERCO "
      "demand in 2024 hour-to-hour data (EIA-930, wind and solar held fixed).")
    w("- Band: 95% interval of the effect after the CV error band of the model's marginal fuel, CO2 and CT "
      "response (`docs/reports/s4_integration.md`, section 1) is applied hour by hour; draws of whole "
      "weeks, so it carries the model's validated bias in the strata touched and the sampling error of "
      "the CEMS benchmark. The temperature-free model uses its own band.\n")
    w("## Results\n")
    w("Change over each set of days. 'model' = the dispatch model's estimate; 'corrected [95% band]' = the "
      "median and 2.5th-97.5th percentiles after the model's validated marginal error is removed hour by "
      "hour (the band can exclude the model estimate where the model is biased in the strata touched). "
      "Negative fuel and CO2 = less; 'saved per MWh moved' is positive for a saving. CT unit-hours = "
      "change in simple-cycle CT unit-hours with output > 0. Fleet heat rate = non-CHP fuel / gross output "
      "over the shifted hours. Gas-share column: temperature-aware model with the EIA-930 share.\n")
    for name in sets:
        w(f"**{name}**\n")
        w(md_table(set_table(results[name])))
        w("")
    def dci(m: str, f: str) -> str:
        """'model X, corrected Y [lo, hi]' of the hot-minus-mild difference."""
        return (f"model {f.format(diff.at[m, 'estimate'])}, corrected {f.format(diff.at[m, 'corrected'])} "
                f"[{f.format(diff.at[m, 'lo'])}, {f.format(diff.at[m, 'hi'])}]")
    w(f"\nHottest minus mild days, temperature-aware model, corrected values and band from the same band "
      f"draws (`compare`): fuel {dci('fuel_mmbtu', '{:+,.0f}')} MMBtu; CO2 {dci('co2_tonnes', '{:+,.0f}')} t; "
      f"CT unit-hours {dci('ct_unit_hours', '{:+.0f}')}.\n")
    hs, ms = hot["temperature-aware"], mild["temperature-aware"]
    w("## Reading\n")
    pk_h, ev_h = (pdf.loc[pdf["days"] == "10 hottest days", c].mean() for c in ("16-19 h GW", "22-01 h GW"))
    pk_m, ev_m = (pdf.loc[pdf["days"] == "10 mild days", c].mean() for c in ("16-19 h GW", "22-01 h GW"))
    w(f"- Hottest days: the model saves {hs['fuel_saved_per_mwh']['estimate']:.2f} MMBtu per MWh moved; after "
      f"the validated error it is {hs['fuel_saved_per_mwh']['corrected']:.2f} "
      f"[{hs['fuel_saved_per_mwh']['lo']:.2f}, {hs['fuel_saved_per_mwh']['hi']:.2f}], so the fuel effect is "
      f"{direction(hs['fuel_mmbtu'])} beyond model error. Baseline gas output averages {pk_h:.1f} GW at "
      f"16-19 h and {ev_h:.1f} GW at 22-01 h, so load moves down the merit order.")
    lab_ = model.band.labels
    a_slope = model.band.m_bar[:, 0] - model.band.point[:, 0]
    dh = dec["10 hottest days"]
    parts = {"CEMS slopes of the strata touched": dh["cems"], "beta x the model's hour-level deviation": dh["beta"],
             "beta2 x its step-size part": dh["beta2"]}
    w(f"- Where the hot-day correction comes from (point values): the CEMS slopes of the strata touched give "
      f"{dh['cems']:+,.0f} MMBtu ({-dh['cems'] / mwh:+.2f} MMBtu saved per MWh moved; CEMS slope "
      f"{a_slope[lab_.index('summer T3')]:.2f} MMBtu/MWh in summer top-tercile hours below 30 C, "
      f"{a_slope[lab_.index('summer T3 hot')]:.2f} at >= 30 C); beta x the model's hour-level deviation from "
      f"its stratum means (beta {model.band.beta_point[-1, 0]:.2f} in the top tercile) gives {dh['beta']:+,.0f} "
      f"MMBtu; beta2 x the step-size part of its answer gives {dh['beta2']:+,.0f} MMBtu. The largest part is "
      f"'{max(parts, key=lambda k: abs(parts[k]))}'.")
    n_more = int((pdf.loc[pdf["days"] == "10 mild days", "fuel MMBtu (model)"] > 0).sum())
    hm = hourly_main["10 mild days"]
    commit_share = 100 * float((hm["d_units_on"] != 0).mean())
    per_sd = float((hm["d_fuel_mmbtu"] / hm["d_gas_mwh"]).std())
    beta_t1 = float(model.band.beta_point[0, 0])
    w(f"- Mild days: model {ms['fuel_saved_per_mwh']['estimate']:+.2f} MMBtu per MWh moved, corrected "
      f"{ms['fuel_saved_per_mwh']['corrected']:+.2f} [{ms['fuel_saved_per_mwh']['lo']:+.2f}, "
      f"{ms['fuel_saved_per_mwh']['hi']:+.2f}]: {direction(ms['fuel_mmbtu'])}. Baseline gas output averages "
      f"{pk_m:.1f} GW at 16-19 h and {ev_m:.1f} GW at 22-01 h, so the shift barely changes the load level; "
      f"the model's per-day results change sign ({n_more} of {len(mild['days'])} days with more fuel). Its "
      f"fuel per MWh varies from hour to hour with a standard deviation of {per_sd:.2f} MMBtu/MWh, and in "
      f"{commit_share:.0f}% of the shifted hours the 500 MW change also changes the set of committed units; "
      f"CEMS confirms only {beta_t1:.2f} of the model's hour-level variation in bottom-tercile hours, and "
      "the band removes the rest.")
    ct_h = hs["ct_unit_hours"]
    lab = model.band.labels
    cems_ct = 1e3 * (model.band.m_bar[:, 2] - model.band.point[:, 2])      # CT units per GW, CEMS
    i_c, i_h = lab.index("summer T3"), lab.index("summer T3 hot")
    beta_ct = float(model.band.beta_point[-1, 2])
    w(f"- Simple-cycle CT unit-hours on the hottest days: model {ct_h['estimate']:+.0f}, corrected "
      f"{ct_h['corrected']:+.0f} [{ct_h['lo']:+.0f}, {ct_h['hi']:+.0f}] ({direction(ct_h)}). In CEMS the "
      f"number of CTs running changes by {cems_ct[i_c]:.1f} units per GW of gas output in summer top-tercile "
      f"hours below 30 C (where most of the evening hours fall) and by {cems_ct[i_h]:.1f} in those at >= 30 C "
      "(the afternoon peak hours); CEMS confirms only "
      f"{beta_ct:.2f} of the model's hour-level variation in CT units at high load.")
    w(f"- Temperature terms raise the hot-day saving by {temp_part:+.2f} MMBtu/MWh in the model estimates "
      f"(temperature-aware minus temperature-free) and by {temp_part_corr:+.2f} after each model's own "
      "validated error is removed. The integration report finds that the hour-to-hour CEMS data do not "
      "confirm the temperature increase of the marginal fuel in the hottest hours; the corrected values "
      "carry that finding.")
    w(f"- With the EIA-930 gas share ({share:.2f}) the gas fleet moves only that share of the {args.mw:,.0f} "
      "MW; the rest falls on coal, storage and other resources, whose fuel the model does not cover.")
    w("- CT unit-hours come from an hourly priority list that places peakers poorly by season (integration "
      "report, section 3.1); read them as the least reliable output.\n")
    w("## By day (temperature-aware model, model estimates)\n")
    w(md_table(pdf, {"daily max C": "{:.1f}", "16-19 h C": "{:.1f}", "22-01 h C": "{:.1f}",
                     "16-19 h GW": "{:.1f}", "22-01 h GW": "{:.1f}", "fuel MMBtu (model)": "{:+,.0f}",
                     "CT unit-hours (model)": "{:+.0f}"}))
    w("\nGW = baseline non-CHP gas gross output (mean over the window's hours); 22-01 h covers 22:00, "
      "23:00 and 00:00 of the next day.\n")
    w("## Strata touched (MWh of gas output change, temperature-aware model)\n")
    w(md_table(pd.DataFrame(strata), {"MWh": "{:+,.0f}"}))
    w("\n## Limitations\n")
    w("- Illustrative: a fixed shift on 2024 days with 2024 baseline output and weather, not a tariff "
      "response; ERCOT's actual response to such a shift is not observed.")
    w("- Gas-marginal assumption (gas share 1) in the main rows; the EIA-930 share is an hour-to-hour "
      "association, not a causal response.")
    w("- Each hour is dispatched independently (no start-up fuel, ramp limits or multi-hour commitment), "
      "so the evening ramp and any extra starts are not costed.")
    w("- CHP units, the steam turbines of suspect CC units and units outside CEMS are not in the model; "
      "temperatures are nearest-station dry-bulb readings.")
    hh_, hm_ = half["10 hottest days"]["fuel_saved_per_mwh"], half["10 mild days"]["fuel_saved_per_mwh"]
    h500 = hot["temperature-aware"]["fuel_saved_per_mwh"]
    m500 = mild["temperature-aware"]["fuel_saved_per_mwh"]
    w(f"- Step size: the band carries the step-size part of the model's answer, scaled by the share CEMS "
      "confirms for changes of that size (beta2; `outputs/s4_integration.md`, section 1). With "
      f"{args.mw / 2:,.0f} MW instead of {args.mw:,.0f} MW on the same days, fuel saved per MWh moved is "
      f"{hh_['corrected']:.2f} [{hh_['lo']:.2f}, {hh_['hi']:.2f}] on the hottest days "
      f"(fuel: {direction(half['10 hottest days']['fuel_mmbtu'])}; at {args.mw:,.0f} MW {h500['corrected']:.2f} "
      f"[{h500['lo']:.2f}, {h500['hi']:.2f}]) and {hm_['corrected']:+.2f} [{hm_['lo']:+.2f}, {hm_['hi']:+.2f}] on "
      f"the mild days (fuel: {direction(half['10 mild days']['fuel_mmbtu'])}; at {args.mw:,.0f} MW "
      f"{m500['corrected']:+.2f} [{m500['lo']:+.2f}, {m500['hi']:+.2f}]).")
    w(f"\nRuntime: {time.time() - t0:.0f} s. Re-run: `python scripts/s4_worked_example.py` (after "
      "`scripts/integrate_s4.py`), or the whole S4 pipeline with `python scripts/run_s4.py`.")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / REPORT).write_text("\n".join(L) + "\n")
    DOCS_REPORTS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUTS / REPORT, DOCS_REPORTS / REPORT)
    out = {"mw": args.mw, "from_hours": FROM_H, "to_hours": TO_H, "tz": TZ, "gas_share_eia930": share,
           "gas_share_se": share_se, "config": model.meta["config"], "results": results,
           "hot_minus_mild": {m: {k: float(diff.at[m, k]) for k in ("estimate", "lo", "hi")}
                              for m in diff.index},
           "temperature_part_per_mwh": temp_part, "temperature_part_per_mwh_corrected": temp_part_corr,
           "correction_split_mmbtu": dec, "half_step": {"mw": args.mw / 2, "results": half},
           "per_day": pdf.to_dict(orient="records")}
    (OUTPUTS / "s4_worked_example.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUTPUTS / REPORT}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
