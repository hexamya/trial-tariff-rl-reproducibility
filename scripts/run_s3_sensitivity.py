"""Sensitivity of the overstatement result (S3), with the monthly MIP planner (oracle).

Run:  python scripts/run_s3_sensitivity.py --variants E_co2_0 E_res08 ...   (each variant cached)
      python scripts/run_s3_sensitivity.py --report

The planner-based measure is used because re-training the agents for every case is out of reach, and the
main results show the planners' and the agents' overstatement point the same way. MIP_B plans with an
assumed response and is judged in the trial-estimated environment (realised, 50 evaluation members) and in
its own (self-assessed); MIP_A plans with the trial-estimated response (150 training members) and is
judged in the trial-estimated environment. London: the 12 months of 2013. ERCOT: the test year 2024
(curtailed days kept); fuel and CO2 through ``S4Model.impact``. The variants are:

* assumed model: Aalami with the period diagonal (-0.10) also between different hours of the same period
  (``samecross``, the alternative reading of Table 1); linear own-price demand, eps -0.10 and -1;
* ERCOT reward weights from data/processed/s2_settings.json ``sensitivity``: CO2 0, 52 and 492 $/t, the
  2022 Texas power-sector gas price, capacity with a 20-year recovery and with an aeroderivative CT, month
  weights from demand instead of net load;
* ERCOT residential load x0.8 and x1.2;
* the all-controls kernel as the trial-estimated response (the pre-declared kernel sensitivity).
Writes outputs/s3_sensitivity.md (copied to docs/reports).
"""
from __future__ import annotations

import argparse
import copy
import shutil
import time

import numpy as np
import pandas as pd

from evaluate_s3 import S4Eval, boot_ratio, with_s4
from run_s3_baselines import TRAIN_YEARS
from ttrl.analysis import lcl_kernel as kn
from ttrl.env import assumed as am
from ttrl.env import ercot_env as ee
from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env.responses import AssumedResponse, TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import harness as hs
from ttrl.s3 import patterns as pt
from ttrl.s3 import planners as pl

OUT = OUTPUTS / "sensitivity"
SAMECROSS = -0.10
VARIANTS = {
    "L_main": {}, "L_samecross": {"assumed": "samecross"}, "L_lin010": {"assumed": "lin010"},
    "L_lin1": {"assumed": "lin1"}, "L_allctl": {"kernel": "allctl", "with_A": True},
    "E_main": {}, "E_samecross": {"assumed": "samecross"}, "E_lin010": {"assumed": "lin010"}, "E_lin1": {"assumed": "lin1"},
    "E_co2_0": {"co2": "co2_price_zero", "with_A": True}, "E_co2_lo": {"co2": "co2_price_rennert_5pct", "with_A": True},
    "E_co2_hi": {"co2": "co2_price_rennert_95pct", "with_A": True}, "E_gas2022": {"gas": "gas_price_texas_power_2022", "with_A": True},
    "E_cap20": {"cap": "capacity_value_20yr", "with_A": True}, "E_capaero": {"cap": "capacity_value_aeroderivative", "with_A": True},
    "E_wdemand": {"weights": "peak_month_weight_demand", "with_A": True},
    "E_res08": {"res": 0.8, "with_A": True}, "E_res12": {"res": 1.2, "with_A": True},
    "E_allctl": {"kernel": "allctl", "with_A": True},
}


def kernel(name: str) -> fk.KernelArrays:
    prefix = "lcl_kernel_allctl_" if name == "allctl" else "lcl_kernel_"
    spec = kn.KernelSpec.from_json((OUTPUTS / f"{prefix}spec.json").read_text(encoding="utf-8"))
    return fk.from_coefficients(spec, pd.read_parquet(OUTPUTS / f"{prefix}ensemble.parquet"))


def assumed_model(kind: str, categories: np.ndarray):
    if kind == "samecross":
        return am.aalami_2010(categories, same_period_cross=SAMECROSS)
    if kind == "lin010":
        return am.LinearOwnPrice(-0.10, source="Aalami et al. 2010 self-elasticity")
    if kind == "lin1":
        return am.LinearOwnPrice(-1.0, source="unit elasticity implied by revenue-optimal pricing (Nejad & Dehghani 2026)")
    return am.aalami_2010(categories)


def settings_for(spec: dict) -> S2Settings:
    st = S2Settings.load()
    s = copy.deepcopy(st)
    sens = st.sensitivity
    if "co2" in spec:
        s.co2_price = float(sens[spec["co2"]])
    if "gas" in spec:
        s.gas_price_by_month = {m: float(sens[spec["gas"]]) for m in range(1, 13)}
    if "cap" in spec:
        s.capacity_value = float(sens[spec["cap"]])
    if "weights" in spec:
        s.peak_month_weight = {int(k): float(v) for k, v in sens[spec["weights"]].items()}
    return s


def run_variant(name: str) -> None:
    path = OUT / f"{name}.parquet"
    if path.exists():
        return
    spec = VARIANTS[name]
    arr = kernel(spec.get("kernel", "main"))
    tr, ev = le.member_split(arr.n_members)
    st = settings_for(spec)
    P = pt.candidate_patterns()
    if name.startswith("L_"):
        inp = ld.LondonInputs.load()
        months = sorted(inp.days.loc[inp.days["period"] == "trial", "month"].unique())
        cats = am.categories_by_load(inp.baseline[(inp.days["period"] == "pre").to_numpy()])
        cfg = lambda: le.LondonConfig.from_settings(st, months, tr)            # noqa: E731
        trial = le.LondonTariffEnv(inp, TrialResponse(arr), cfg(), seed=0)
        model = assumed_model(spec.get("assumed", "aalami"), cats)
        own = le.LondonTariffEnv(inp, AssumedResponse(model), cfg(), seed=0)
        s4 = None
    else:
        months = [f"2024-{m:02d}" for m in range(1, 13)]
        days = ee.ErcotDays.load(2024)
        if "res" in spec:
            days.year.residential = days.year.residential * spec["res"]
        train = [ee.ErcotDays.load(y) for y in TRAIN_YEARS]
        prof = np.concatenate([np.repeat(x.year.residential[x.row[np.all(x.row >= 0, axis=1)]], 2, axis=1) for x in train])
        model = assumed_model(spec.get("assumed", "aalami"), am.categories_by_load(prof))
        cfg = lambda: ee.ErcotConfig.from_settings(st, months, tr, mask_curtailed=False)   # noqa: E731
        trial = ee.ErcotTariffEnv(days, TrialResponse(arr), cfg(), seed=0)
        own = ee.ErcotTariffEnv(days, AssumedResponse(model), cfg(), seed=0)
        s4 = S4Eval(days)
    parts = []
    EB = am.clip_effect(model.effect(P))[None]
    mip_b = pl.MonthlyMIP(EB, P, time_limit=120.0)
    jobs = [("MIP_B", mip_b, own, [0], "self-assessed"), ("MIP_B", mip_b, trial, ev, "realised")]
    if spec.get("with_A"):
        EA = pt.pattern_effects(P, fk.KernelArrays(arr.in_event[tr], arr.dur_dev[tr], arr.window[tr]))
        jobs.append(("MIP_A", pl.MonthlyMIP(EA, P, time_limit=120.0), trial, ev, "realised"))
    for planner, pol, env, members, world in jobs:
        r = hs.evaluate(env, pol, months, members)
        if s4 is not None:
            r = with_s4(r, env, s4)
        parts.append(r.assign(planner=planner, world=world))
    res = pd.concat(parts, ignore_index=True).drop(columns=["blocks", "effect", "day_idx"]).assign(variant=name, seed=0)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(path, index=False)


def report() -> None:
    res = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))], ignore_index=True)
    rng = np.random.default_rng(20261017)
    rows = []
    for v, g in res.groupby("variant"):
        b = g[g["planner"] == "MIP_B"]
        s, r = b[b["world"] == "self-assessed"], b[b["world"] == "realised"]
        metrics = [("daily_peak_cut_pct", 1.0, "daily peak cut %")] if v.startswith("L_") else \
            [("s4_fuel", -1.0, "fuel saved (S4, MMBtu)"), ("s4_co2_t", -1.0, "CO2 saved (S4, t)"),
             ("monthly_net_peak_change_mw", -1.0, "net-load peak cut (MW)"), ("reward", 1.0, "reward ($)")]
        a = g[(g["planner"] == "MIP_A") & (g["world"] == "realised")]
        # months the MIP found no plan within the limits (mip.solve falls back to Normal on every day):
        # no High or Low half-hour posted, and a limit broken in the planner's own environment
        m = s.groupby("month").agg(hh=("high_hh", "sum"), lh=("low_hh", "sum"), viol=("violated", "any"))
        no_plan = int(((m["hh"] + m["lh"] == 0) & m["viol"]).sum())
        for col, sign, label in metrics:
            pnt, lo, hi = boot_ratio(s, r, col, sign, rng)        # drawn for every case to keep the random stream
            if no_plan:
                pnt = lo = hi = np.nan
            rows.append({"variant": v, "outcome": label, "MIP_B self-assessed": sign * s[col].mean(),
                         "MIP_B realised": sign * r[col].mean(), "ratio": pnt, "ratio lo": lo, "ratio hi": hi,
                         "MIP_A realised": sign * a[col].mean() if len(a) else np.nan,
                         "MIP_B realised violation": float(r["violated"].mean()),
                         "MIP_B months without a plan": no_plan})
    lines = ["# Sensitivity of the overstatement result (monthly MIP planner)", "",
             "Generated by `python scripts/run_s3_sensitivity.py`. MIP_B plans with the assumed response; self-assessed = "
             "its own environment, realised = the trial-estimated environment with the 50 evaluation members. MIP_A plans "
             "with the trial-estimated response (realised only). London: 2013; ERCOT: test year 2024, fuel and CO2 through "
             "S4Model.impact. The script documents the variants. Intervals use a bootstrap over months.", "",
             "MIP_B months without a plan: months in which no plan meets the revenue band and the other limits in the "
             "assumed environment, so the MIP posts Normal on every day. Under the trial's prices a planner that assumes "
             "this response cannot build a revenue-neutral plan; these cases are reported with a missing ratio.", "",
             md_table(pd.DataFrame(rows), floatfmt="{:,.3f}")]
    out = OUTPUTS / "s3_sensitivity.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy(out, ROOT / "docs" / "reports" / out.name)
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    t0 = time.time()
    for v in args.variants:
        run_variant(v)
        print(f"{v} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
