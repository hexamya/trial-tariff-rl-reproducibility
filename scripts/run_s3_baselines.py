"""S3 rule-based baselines in both environments, with the trial-estimated response (agent A's world).

Run:  python scripts/run_s3_baselines.py   (~10 min; needs build_s2_settings.py)

London: policies are cross-fitted over four month folds. The one
parameter of each rule is calibrated (``baselines.calibrate``) on the other three folds' months with 10
training ensemble members and evaluated on the fold's months with all 50 evaluation members. The 2013
schedule needs no calibration. ERCOT: calibrated on the training years 2019-2022 (10 training members),
reported on the validation year 2023 (50 evaluation members); the test year 2024 is not touched.

Policies: the 2013 schedule (London), ``ForecastCPP`` (naive day-ahead threshold rule) and ``QuotaCPP``
(revenue-quota CPP). Writes outputs/s3_baselines.md (copied to docs/reports) and
outputs/s3_baselines_episodes.parquet.
"""
from __future__ import annotations

import shutil
import time

import numpy as np
import pandas as pd

from build_london_env import schedule_on_grid
from ttrl.analysis import lcl_kernel as kn
from ttrl.env import ercot_env as ee
from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env.responses import TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import baselines as bl
from ttrl.s3 import harness as hs

CAL_MEMBERS, SEED = 10, 20261012
K_GRID = list(range(2, 15))
ALPHA_GRID = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
OLD_CVAR_CAP = 0.0441          # first London bill cap (household CVaR95), replaced; kept for the diagnostic
TRAIN_YEARS, VAL_YEAR = (2019, 2020, 2021, 2022), 2023


def kernel_arrays():
    spec = kn.KernelSpec.from_json((OUTPUTS / "lcl_kernel_spec.json").read_text(encoding="utf-8"))
    return fk.from_coefficients(spec, pd.read_parquet(OUTPUTS / "lcl_kernel_ensemble.parquet"))


def london(arr, tr, ev, st) -> tuple[pd.DataFrame, pd.DataFrame]:
    inp = ld.LondonInputs.load()
    days = inp.days
    trial = days[days["period"] == "trial"]
    months = sorted(trial["month"].unique())
    fold_of = trial.groupby("month")["fold"].first().to_dict()
    env = le.LondonTariffEnv(inp, arr, le.LondonConfig.from_settings(st, months, tr), seed=0)
    cal = np.random.default_rng(SEED).choice(tr, CAL_MEMBERS, replace=False)
    tdays = pd.DatetimeIndex(days["tday"])
    blocks, _ = schedule_on_grid(tdays)
    rows, cal_rows = [hs.evaluate(env, bl.ScheduleReplay({t: blocks[i] for i, t in enumerate(tdays)}), months, ev)], []
    for fold in sorted(set(fold_of.values())):
        test_m = [m for m in months if fold_of[m] == fold]
        train_m = [m for m in months if fold_of[m] != fold]
        for fam, make, grid in (("CPP (threshold)", lambda k: bl.ForecastCPP(int(k)), K_GRID),
                                ("CPP (revenue quota)", lambda a: bl.QuotaCPP(float(a)), ALPHA_GRID)):
            best, tab = bl.calibrate(env, make, grid, train_m, cal)
            cal_rows.append(tab.assign(case="London", family=fam, fold=fold, chosen=tab["param"] == best))
            r = hs.evaluate(env, make(best), test_m, ev)
            rows.append(r.assign(family=fam, param=best, fold=fold))
    res = pd.concat(rows, ignore_index=True)
    res["family"] = res["family"].fillna("2013 schedule")
    return res.assign(case="London"), pd.concat(cal_rows, ignore_index=True)


def ercot(arr, tr, ev, st) -> tuple[pd.DataFrame, pd.DataFrame]:
    resp = TrialResponse(arr)
    envs = {y: ee.ErcotTariffEnv(ee.ErcotDays.load(y), resp,
                                 ee.ErcotConfig.from_settings(st, [f"{y}-{m:02d}" for m in range(1, 13)], tr), seed=0)
            for y in (*TRAIN_YEARS, VAL_YEAR)}
    cal = np.random.default_rng(SEED).choice(tr, CAL_MEMBERS, replace=False)
    rows, cal_rows = [], []
    for fam, make, grid in (("CPP (threshold)", lambda k: bl.ForecastCPP(int(k)), K_GRID),
                            ("CPP (revenue quota)", lambda a: bl.QuotaCPP(float(a)), ALPHA_GRID)):
        tabs = []
        for y in TRAIN_YEARS:
            _, t = bl.calibrate(envs[y], make, grid, [f"{y}-{m:02d}" for m in range(1, 13)], cal)
            tabs.append(t)
        tab = pd.concat(tabs).groupby("param", as_index=False).mean()
        best = tab.sort_values(["violation_rate", "constraint_excess", "reward"], ascending=[True, True, False]).iloc[0]["param"]
        cal_rows.append(tab.assign(case="ERCOT", family=fam, chosen=tab["param"] == best))
        r = hs.evaluate(envs[VAL_YEAR], make(best), [f"{VAL_YEAR}-{m:02d}" for m in range(1, 13)], ev)
        rows.append(r.assign(family=fam, param=best))
    return pd.concat(rows, ignore_index=True).assign(case="ERCOT"), pd.concat(cal_rows, ignore_index=True)


def summary(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["case", "family"])
    out = g.agg(episodes=("reward", "size"), reward=("reward", "mean"), revenue_ratio=("revenue_ratio", "mean"),
                revenue_in_band=("cost_revenue", lambda c: float((c <= 1e-12).mean())),
                violation_rate=("violated", "mean"), high_hh=("high_hh", "mean")).reset_index()
    lon = res[res["case"] == "London"].groupby("family")
    if len(res[res["case"] == "London"]):
        out = out.merge(lon.agg(daily_peak_cut_pct=("daily_peak_cut_pct", "mean"),
                                segment_cap_met=("cost_bill_segment", lambda c: float((c <= 1e-12).mean())),
                                worst_segment_pct=("bill_segment_max_nonresponding", lambda v: 100 * v.mean()),
                                household_p95_pct=("bill_p95", lambda v: 100 * v.mean()),
                                household_cvar95_nonresp_pct=("bill_cvar95_nonresponding", lambda v: 100 * v.mean()),
                                old_cvar_cap_met=("bill_cvar95_nonresponding", lambda v: float((v <= OLD_CVAR_CAP).mean())))
                        .reset_index().assign(case="London"), on=["case", "family"], how="left")
    erc = res[res["case"] == "ERCOT"].groupby("family")
    if len(res[res["case"] == "ERCOT"]):
        out = out.merge(erc.agg(fuel_cost=("fuel_cost", "mean"), co2_cost=("co2_cost", "mean"),
                                capacity_cost=("capacity_cost", "mean"), d_fuel_mmbtu=("d_fuel_mmbtu", "mean"),
                                net_peak_change_mw=("monthly_net_peak_change_mw", "mean"))
                        .reset_index().assign(case="ERCOT"), on=["case", "family"], how="left")
    return out


def main() -> None:
    t0 = time.time()
    arr = kernel_arrays()
    tr, ev = le.member_split(arr.n_members)
    st = S2Settings.load()
    lres, lcal = london(arr, tr, ev, st)
    print(f"London done ({time.time() - t0:.0f} s)", flush=True)
    eres, ecal = ercot(arr, tr, ev, st)
    print(f"ERCOT done ({time.time() - t0:.0f} s)", flush=True)
    res = pd.concat([lres, eres], ignore_index=True)
    res.drop(columns=["blocks", "effect", "day_idx"], errors="ignore").to_parquet(OUTPUTS / "s3_baselines_episodes.parquet", index=False)
    summ = summary(res)
    cal = pd.concat([lcal, ecal], ignore_index=True)
    chosen = cal[cal["chosen"]][["case", "family", "fold", "param", "violation_rate", "constraint_excess", "reward"]]
    lon_cols = ["family", "episodes", "daily_peak_cut_pct", "revenue_ratio", "revenue_in_band", "segment_cap_met",
                "worst_segment_pct", "household_p95_pct", "household_cvar95_nonresp_pct", "old_cvar_cap_met", "violation_rate", "high_hh"]
    erc_cols = ["family", "episodes", "reward", "fuel_cost", "co2_cost", "capacity_cost", "d_fuel_mmbtu", "net_peak_change_mw",
                "revenue_ratio", "revenue_in_band", "violation_rate", "high_hh"]
    lon_month = lres.groupby(["family", "month"])["bill_cvar95_nonresponding"].mean().unstack(0).mul(100)
    lines = [
        "# S3 rule-based baselines (trial-estimated response)",
        "",
        "Generated by `python scripts/run_s3_baselines.py`. London: cross-fitted over the four month folds (rule "
        "parameter calibrated on the other folds with 10 training members, evaluated with the 50 evaluation members). "
        f"ERCOT: calibrated on {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}, reported on the validation year {VAL_YEAR} (50 evaluation "
        "members); 2024 untouched. Means over episodes (one month x one member).",
        "",
        "## London",
        "",
        md_table(summ[summ["case"] == "London"][lon_cols], floatfmt="{:,.3f}"),
        "",
        "'segment_cap_met': share of episodes where every exposure segment's mean bill change (households that do not "
        "respond) is at most +3%. 'old_cvar_cap_met': share of episodes that would meet the first bill cap (household "
        f"CVaR95 <= {100 * OLD_CVAR_CAP:.2f}%), shown because it could not be met together with monthly revenue "
        "neutrality. Household CVaR95 (no response) by month, %:",
        "",
        md_table(lon_month.reset_index(), floatfmt="{:.1f}"),
        "",
        "## ERCOT (validation year)",
        "",
        md_table(summ[summ["case"] == "ERCOT"][erc_cols], floatfmt="{:,.2f}"),
        "",
        "Costs in 2024 US$ per month (negative = saving; reward = minus the sum of fuel, CO2 and capacity costs).",
        "",
        "## Calibrated parameters",
        "",
        md_table(chosen, floatfmt="{:,.3f}"),
    ]
    out = OUTPUTS / "s3_baselines.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy(out, ROOT / "docs" / "reports" / out.name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
