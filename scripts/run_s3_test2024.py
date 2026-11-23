"""Evaluate the fixed policy designs on the ERCOT 2024 test loads.

Run:  python scripts/run_s3_test2024.py --group g1|g2|g3    (parallel workers; each method cached)
      python scripts/run_s3_test2024.py --report            (assemble outputs/s3_test2024.md)

Nothing is trained or calibrated here: rule parameters come from scripts/run_s3_baselines.py (ERCOT,
2019-2022), agents from outputs/rl/final/ (settings chosen on 2023), planners need no calibration. 2024 is
the S4 fleet's own year; curtailed days are kept and reported (mask off). Realised = trial-estimated
environment with the 50 evaluation members; self-assessed (agents and planners on the assumed response)
= the assumed-response environment. Fuel and CO2 through ``S4Model.impact`` with its band. Days hotter
than the kernel's 24.7 C are counted, and the realised surrogate outcomes are repeated with the hot-day
multiplier m = 0.5, 1.5 and 2.
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import pandas as pd

from evaluate_s3 import S4Eval, boot_ratio, with_s4
from run_s3_baselines import TRAIN_YEARS, kernel_arrays
from run_s3_rl import FINAL_DIR
from ttrl.env import assumed as am
from ttrl.env import ercot_env as ee
from ttrl.env import fastkernel as fk
from ttrl.env import london_env as le
from ttrl.env.responses import AssumedResponse, TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import baselines as bl
from ttrl.s3 import harness as hs
from ttrl.s3 import patterns as pt
from ttrl.s3 import planners as pl
from ttrl.s3 import rl

YEAR = 2024
OUT = OUTPUTS / "test2024"
QUOTA_ALPHA, THRESHOLD_K = 0.6, 5            # calibrated on 2019-2022 (docs/reports/s3_baselines.md)
HOT_M = (0.5, 1.5, 2.0)
GROUPS = {"g1": ["cpp_quota", "cpp_threshold", "rl_A_s0", "rl_A_s1", "rl_A_s2", "ppo_A_s0"],
          "g2": ["rl_B_s0", "rl_B_s1", "rl_B_s2", "mip_B", "mpc_B"],
          "g3": ["mip_A", "mpc_A"]}


def setup():
    arr = kernel_arrays()
    tr, ev = le.member_split(arr.n_members)
    st = S2Settings.load()
    months = [f"{YEAR}-{m:02d}" for m in range(1, 13)]
    days = ee.ErcotDays.load(YEAR)
    train = [ee.ErcotDays.load(y) for y in TRAIN_YEARS]
    prof = np.concatenate([np.repeat(x.year.residential[x.row[np.all(x.row >= 0, axis=1)]], 2, axis=1) for x in train])
    assumed = am.aalami_2010(am.categories_by_load(prof))

    def env(response, m=1.0):
        cfg = ee.ErcotConfig.from_settings(st, months, tr, mask_curtailed=False, hot_multiplier=m)
        return ee.ErcotTariffEnv(days, response, cfg, seed=0)
    return arr, tr, ev, months, env, assumed, days


def policy(name, arr, tr, assumed, trial_env):
    P = pt.candidate_patterns()
    if name == "cpp_quota":
        return bl.QuotaCPP(QUOTA_ALPHA), False
    if name == "cpp_threshold":
        return bl.ForecastCPP(THRESHOLD_K), False
    if name.startswith(("rl_", "ppo_")):
        agent, seed = name.split("_")[1], name.split("_s")[-1]
        f = FINAL_DIR / (f"ercot_{agent}_ppo_all_s{seed}.pt" if name.startswith("ppo") else f"ercot_{agent}_all_s{seed}.pt")
        return rl.load_agent(f, trial_env), agent == "B"
    if name.endswith("_A"):
        E = pt.pattern_effects(P, fk.KernelArrays(arr.in_event[tr], arr.dur_dev[tr], arr.window[tr]))
    else:
        E = am.clip_effect(assumed.effect(P))[None]
    cls = pl.MonthlyMIP if name.startswith("mip") else pl.RollingMPC
    kw = {"time_limit": 120.0} if cls is pl.MonthlyMIP else {"time_limit": 30.0, "mip_gap": 5e-3}
    return cls(E, P, **kw), name.endswith("_B")


def run_method(name: str) -> None:
    path = OUT / f"{name}.parquet"
    if path.exists():
        return
    arr, tr, ev, months, env, assumed, days = setup()
    trial = env(TrialResponse(arr))
    pol, uses_assumed = policy(name, arr, tr, assumed, trial)
    s4 = S4Eval(days)
    parts = []
    if uses_assumed or name.startswith("rl_A"):           # self-assessed: the method's own model of the world
        own_env = env(AssumedResponse(assumed)) if uses_assumed else trial
        own_pol = rl.load_agent(FINAL_DIR / f"ercot_{name.split('_')[1]}_all_s{name[-1]}.pt", own_env) \
            if name.startswith("rl_") else pol
        own_members = [0] if uses_assumed else np.random.default_rng(20261013).choice(tr, len(ev), replace=False)
        parts.append(with_s4(hs.evaluate(own_env, own_pol, months, own_members), own_env, s4).assign(world="self-assessed", m=1.0))
    parts.append(with_s4(hs.evaluate(trial, pol, months, ev), trial, s4).assign(world="realised", m=1.0))
    for m in HOT_M:                                        # hot-day multiplier band (surrogate outcomes only)
        e_m = env(TrialResponse(arr), m)
        p_m = rl.load_agent(FINAL_DIR / (f"ercot_{name.split('_')[1]}_ppo_all_s0.pt" if name.startswith("ppo") else
                                         f"ercot_{name.split('_')[1]}_all_s{name[-1]}.pt"), e_m) \
            if name.startswith(("rl_", "ppo_")) else pol
        parts.append(hs.evaluate(e_m, p_m, months, ev).assign(world="realised", m=m))
    hot = days.temp_day > ee.KERNEL_TEMP_MAX
    res = pd.concat(parts, ignore_index=True).drop(columns=["blocks", "effect", "day_idx"]).assign(method=name, seed=0)
    res.attrs["hot_days"] = int(hot.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(path, index=False)


def report() -> None:
    res = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))], ignore_index=True)
    base = res[(res["world"] == "realised") & (res["m"] == 1.0)]
    coverage = base.groupby("month")[["days", "out_of_support_days"]]
    if coverage.ngroups != 12 or not coverage.nunique().eq(1).all().all():
        raise ValueError("Archived reference episodes must agree on coverage for all twelve months.")
    tariff_days, hot = coverage.first().sum().astype(int)
    cols = {"reward": "mean", "fuel_cost": "mean", "co2_cost": "mean", "capacity_cost": "mean", "s4_fuel": "mean",
            "s4_fuel_lo": "mean", "s4_fuel_hi": "mean", "s4_co2_t": "mean", "monthly_net_peak_change_mw": "mean",
            "revenue_ratio": "mean", "violated": "mean"}
    method = base["method"].str.replace(r"_s\d$", "", regex=True)
    tab = base.assign(method=method).groupby("method").agg(cols).reset_index()
    hotm = res[res["world"] == "realised"].assign(method=lambda d: d["method"].str.replace(r"_s\d$", "", regex=True)) \
        .pivot_table(index="method", columns="m", values="reward", aggfunc="mean").reset_index()
    hotm.columns = ["method"] + [f"reward m={c:g}" for c in hotm.columns[1:]]
    rng = np.random.default_rng(20261016)
    over = []
    for name, g in res[res["m"] == 1.0].assign(group=lambda d: d["method"].str.replace(r"_s\d$", "", regex=True)) \
            .assign(seed=lambda d: d["method"].str.extract(r"_s(\d)$")[0].fillna("0").astype(int)).groupby("group"):
        s, r = g[g["world"] == "self-assessed"], g[g["world"] == "realised"]
        if not len(s):
            continue
        for col, sign, label in [("reward", 1.0, "reward ($)"), ("s4_fuel", -1.0, "fuel saved (S4)"),
                                 ("s4_co2_t", -1.0, "CO2 saved (S4)"), ("monthly_net_peak_change_mw", -1.0, "net-load peak cut")]:
            pnt, lo, hi = boot_ratio(s, r, col, sign, rng)
            over.append({"method": name, "outcome": label, "self-assessed": sign * s[col].mean(),
                         "realised": sign * r[col].mean(), "ratio": pnt, "ratio lo": lo, "ratio hi": hi})
    lines = ["# ERCOT test year 2024 (run once, every method fixed)", "",
             "Generated by `python scripts/run_s3_test2024.py`. Realised = trial-estimated environment, 50 evaluation "
             "members, curtailed days kept; costs in 2024 US$ per month (surrogate) and S4 fuel (MMBtu, corrected with "
             f"95% band) and CO2 (t). {hot} of {tariff_days} tariff days are hotter than the kernel's 24.7 C "
             "(outside its support); the hot-day multiplier rows scale the response on those days.", "",
             "## Realised outcomes (m = 1)", "", md_table(tab, floatfmt="{:,.3f}"), "",
             "## Reward under the hot-day multiplier", "", md_table(hotm, floatfmt="{:,.0f}"), "",
             "## Overstatement (self-assessed / realised benefit)", "", md_table(pd.DataFrame(over), floatfmt="{:,.3f}"), "",
             "Methods: cpp_* rule baselines; mip/mpc_A planners with the trial-estimated response (oracle MIP knows the "
             "month's load); mip/mpc_B planners with the assumed response; rl_A/rl_B PPO-Lagrangian agents (three seeds "
             "pooled); ppo_A unconstrained PPO."]
    out = OUTPUTS / "s3_test2024.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy(out, ROOT / "docs" / "reports" / out.name)
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    t0 = time.time()
    for name in GROUPS[args.group]:
        run_method(name)
        print(f"{name} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
