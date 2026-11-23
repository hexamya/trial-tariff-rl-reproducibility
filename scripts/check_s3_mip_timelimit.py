"""Does the 120 s MIP time limit explain the July 2013 difference between the two London MIP_B runs? (S3)

Run:  python scripts/check_s3_mip_timelimit.py [--long 3600]

The monthly MIP (oracle) plans with the assumed response (Aalami et al. 2010, hour categories from the
pre-trial days' load shape), as in scripts/run_s3_planners_assumed.py and the sensitivity case L_main of
scripts/run_s3_sensitivity.py. The two runs agree in every 2013 month except July. Here every 2013
month is solved once with the original 120 s limit, and July a second time
with 120 s and once with a long limit. Each solve's HiGHS status, gap, objective and wall time are
logged, each plan is evaluated in the assumed environment (self-assessed) and in the trial-estimated one
with the 50 evaluation members (realised), and July is compared with the two earlier runs. Nothing
earlier is rerun or replaced. Writes outputs/s3_mip_timelimit_check.md (copied to docs/reports) and
outputs/s3_mip_timelimit_check.csv.
"""
from __future__ import annotations

import argparse
import shutil
import time

import pandas as pd

from run_s3_baselines import kernel_arrays
from ttrl.env import assumed as am
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env.responses import AssumedResponse, TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import harness as hs
from ttrl.s3 import patterns as pt
from ttrl.s3 import planners as pl

MONTH = "2013-07"
BASE_LIMIT = 120.0                                   # the limit both earlier runs used
EARLIER = {"run_s3_planners_assumed.py (2026-10-06)": ("s3_planners_assumed_london_episodes.parquet", "MIP (oracle month)"),
           "run_s3_sensitivity.py L_main (2026-10-07)": ("sensitivity/L_main.parquet", "MIP_B")}
STATUS = {0: "optimal", 1: "time limit", 2: "infeasible", 3: "unbounded", 4: "other"}


def solve(planner, env, month: str) -> dict:
    env.reset(options={"month": month, "member": 0})     # as harness.run_month before policy.reset
    t0 = time.time()
    planner.reset(env)
    return {**planner.log[-1], "seconds": time.time() - t0}


def outcome(planner, own, trial, month: str, ev) -> dict:
    s = hs.evaluate(own, planner, [month], [0])           # the plan is cached, so both worlds see the same plan
    r = hs.evaluate(trial, planner, [month], ev)
    return {"self_peak_cut_pct": float(s["daily_peak_cut_pct"].mean()), "self_violated": bool(s["violated"].any()),
            "realised_peak_cut_pct": float(r["daily_peak_cut_pct"].mean()),
            "realised_violation_rate": float(r["violated"].mean()),
            "high_hh": int(s["high_hh"].iloc[0]), "low_hh": int(s["low_hh"].iloc[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=float, default=3600.0, help="time limit (s) of the long July solve")
    args = ap.parse_args()
    t0 = time.time()
    arr = kernel_arrays()
    tr, ev = le.member_split(arr.n_members)
    st = S2Settings.load()
    P = pt.candidate_patterns()
    inp = ld.LondonInputs.load()
    months = sorted(inp.days.loc[inp.days["period"] == "trial", "month"].unique())
    model = am.aalami_2010(am.categories_by_load(inp.baseline[(inp.days["period"] == "pre").to_numpy()]))
    own = le.LondonTariffEnv(inp, AssumedResponse(model), le.LondonConfig.from_settings(st, months, tr), seed=0)
    trial = le.LondonTariffEnv(inp, TrialResponse(arr), le.LondonConfig.from_settings(st, months, tr), seed=0)
    E = am.clip_effect(model.effect(P))[None]             # (1, P, 48), as in both earlier runs
    runs = [(m, BASE_LIMIT, "each month, 120 s") for m in months] + \
        [(MONTH, BASE_LIMIT, "July repeated, 120 s"), (MONTH, args.long, f"July, {args.long:.0f} s")]
    rows = []
    for month, limit, label in runs:
        planner = pl.MonthlyMIP(E, P, time_limit=limit)
        info = solve(planner, own, month)
        rows.append({"run": label, "month": month, "time_limit_s": limit, "status": STATUS.get(info["status"], info["status"]),
                     "mip_gap": info["mip_gap"], "objective": info["objective"], "solve_s": info["seconds"],
                     **outcome(planner, own, trial, month, ev)})
        print(f"{label} {month}: {rows[-1]['status']}, gap {info['mip_gap']}, {info['seconds']:.0f} s "
              f"({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "s3_mip_timelimit_check.csv", index=False)
    earlier = []
    for name, (rel, planner_name) in EARLIER.items():
        d = pd.read_parquet(OUTPUTS / rel)
        d = d[(d["planner"] == planner_name) & (d["month"] == MONTH)]
        s, r = d[d["world"] == "self-assessed"], d[d["world"] == "realised"]
        earlier.append({"run": name, "self_peak_cut_pct": float(s["daily_peak_cut_pct"].mean()),
                        "realised_peak_cut_pct": float(r["daily_peak_cut_pct"].mean()),
                        "realised_violation_rate": float(r["violated"].mean()),
                        "high_hh": int(s["high_hh"].iloc[0]), "low_hh": int(s["low_hh"].iloc[0])})
    base = res[res["time_limit_s"] == BASE_LIMIT]
    hit = base[base["status"] == "time limit"]
    lines = ["# MIP time-limit check: London MIP_B, assumed response (S3)", "",
             "Generated by `python scripts/check_s3_mip_timelimit.py`. Monthly MIP (oracle) with the Aalami et al. (2010) "
             "response, set up as in `run_s3_planners_assumed.py` and the sensitivity case L_main. Self-assessed = the plan in the "
             "assumed environment; realised = the same plan in the trial-estimated environment, 50 evaluation members. "
             "Objective = the MIP's objective (sum over days); peak cut = mean daily peak cut (%).", "",
             f"Solves at {BASE_LIMIT:.0f} s that stopped at the time limit: {len(hit)} of {len(base)}"
             + (f" ({', '.join(f'{m}' for m in hit['month'])})." if len(hit) else "."), "",
             "## Solves", "", md_table(res, floatfmt="{:,.4f}"), "",
             f"## {MONTH} in the two earlier runs", "", md_table(pd.DataFrame(earlier), floatfmt="{:,.4f}")]
    out = OUTPUTS / "s3_mip_timelimit_check.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy(out, ROOT / "docs" / "reports" / out.name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
