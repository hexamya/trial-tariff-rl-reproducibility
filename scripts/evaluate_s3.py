"""S3 evaluation: trained agents against the baselines, and the overstatement measurement (the paper's
core result), on held-out data.

Run:  python scripts/evaluate_s3.py   (after the RL batches; ~1 h)

For every trained agent (outputs/rl/*.pt):

* realised outcome: the deterministic policy in the trial-estimated environment with the 50 evaluation
  members, at the evaluated limits (London: the held-out fold's months; ERCOT: the validation year 2023);
* self-assessed outcome: the same policy in the environment it was trained in (agent A: trial-estimated
  response, 50 training members; agent B: the assumed response), same months and limits.

Overstatement = self-assessed benefit / realised benefit (London: mean daily peak cut; ERCOT: monthly
reward, fuel and CO2 saved, net-load peak cut), with 95% intervals from a bootstrap over months and seeds.
ERCOT fuel and CO2 are computed with ``S4Model.impact`` and its error band. The training reward uses the reduced model.
The test year 2024 is
not touched. Each agent's episodes are cached in outputs/rl/eval/ (agents already evaluated are not run
again). Writes outputs/s3_rl.md (copied to docs/reports) and outputs/s3_rl_episodes.parquet.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time

import numpy as np
import pandas as pd

from run_s3_baselines import VAL_YEAR, kernel_arrays
from run_s3_rl import RL_DIR, TRAIN_YEARS
from ttrl.env import assumed as am
from ttrl.env import ercot as er
from ttrl.env import ercot_env as ee
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env.responses import AssumedResponse, TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import harness as hs
from ttrl.s3 import rl

N_BOOT, SEED = 2000, 20261013
NAME = re.compile(r"(?P<case>london|ercot)_(?P<agent>[AB])(?P<ppo>_ppo)?_(?P<fold>[A-D]|all)_s(?P<seed>\d+)\.pt$")


def london_envs(arr, tr, st):
    inp = ld.LondonInputs.load()
    trial = inp.days[inp.days["period"] == "trial"]
    months = sorted(trial["month"].unique())
    fold_of = trial.groupby("month")["fold"].first().to_dict()
    cfg = lambda: le.LondonConfig.from_settings(st, months, tr)          # noqa: E731
    trial_env = le.LondonTariffEnv(inp, TrialResponse(arr), cfg(), seed=0)

    def assumed_env(fold):
        train_days = inp.days["month"].isin([m for m in months if fold_of[m] != fold]).to_numpy()
        resp = AssumedResponse(am.aalami_2010(am.categories_by_load(inp.baseline[train_days])))
        return le.LondonTariffEnv(inp, resp, cfg(), seed=0)
    return trial_env, assumed_env, months, fold_of


def ercot_envs(arr, tr, st):
    months = [f"{VAL_YEAR}-{m:02d}" for m in range(1, 13)]
    d = ee.ErcotDays.load(VAL_YEAR)
    train = [ee.ErcotDays.load(y) for y in TRAIN_YEARS]
    prof = np.concatenate([np.repeat(x.year.residential[x.row[np.all(x.row >= 0, axis=1)]], 2, axis=1) for x in train])
    cfg = lambda: ee.ErcotConfig.from_settings(st, months, tr)            # noqa: E731
    trial_env = ee.ErcotTariffEnv(d, TrialResponse(arr), cfg(), seed=0)
    assumed_env = ee.ErcotTariffEnv(d, AssumedResponse(am.aalami_2010(am.categories_by_load(prof))), cfg(), seed=0)
    return trial_env, assumed_env, months


class S4Eval:
    """Fuel and CO2 of an ERCOT episode's residential load change with ``S4Model.impact`` (2023 weather)."""

    def __init__(self, days: ee.ErcotDays):
        from ttrl.supply.impact import S4Model
        self.model = S4Model.load()
        self.days = days
        self.temps, _ = er.plant_temperature_matrix(days.year.temp_path, days.year.hours, self.model.data.plant_id_eia)

    def __call__(self, effect: np.ndarray, day_idx: np.ndarray, masked: np.ndarray) -> dict:
        y, ds = self.days.year, self.days
        rows, dl = [], []
        for k, d in enumerate(day_idx):
            if masked[k]:
                continue
            eh = ee.hourly_effect(effect[k])
            for r in (ds.row[d], ds.row2[d]):
                ok = r >= 0
                rows.append(r[ok])
                dl.append(y.residential[r[ok]] * eh[ok] / 100)
        rows, dl = np.concatenate(rows), np.concatenate(dl)
        o = np.argsort(rows)
        rows, dl = rows[o], dl[o]
        t = self.model.impact(dl, y.hours[rows], temp_c=self.temps[rows], base_mw=y.base_mw[rows]).totals
        return {"s4_fuel": t.at["fuel_mmbtu", "corrected"], "s4_fuel_lo": t.at["fuel_mmbtu", "lo"],
                "s4_fuel_hi": t.at["fuel_mmbtu", "hi"], "s4_co2_t": t.at["co2_tonnes", "corrected"],
                "s4_co2_lo": t.at["co2_tonnes", "lo"], "s4_co2_hi": t.at["co2_tonnes", "hi"]}


def with_s4(r: pd.DataFrame, env, s4: S4Eval) -> pd.DataFrame:
    extra = []
    for _, row in r.iterrows():
        env.reset(options={"month": row["month"], "member": int(row["member"])})
        extra.append(s4(row["effect"], row["day_idx"], env.masked))
    return pd.concat([r.reset_index(drop=True), pd.DataFrame(extra)], axis=1)


def boot_ratio(self_df: pd.DataFrame, real_df: pd.DataFrame, col: str, sign: float, rng) -> tuple[float, float, float]:
    """Ratio of mean self-assessed to mean realised benefit (benefit = sign x col), bootstrap over
    (seed, month) clusters."""
    s = self_df.groupby(["seed", "month"])[col].mean() * sign
    r = real_df.groupby(["seed", "month"])[col].mean() * sign
    keys = s.index.intersection(r.index)
    s, r = s.loc[keys].to_numpy(), r.loc[keys].to_numpy()
    point = s.mean() / r.mean() if r.mean() != 0 else np.nan
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(keys), len(keys))
        draws.append(s[i].mean() / r[i].mean() if r[i].mean() != 0 else np.nan)
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["london", "ercot"])
    ap.add_argument("--settings", choices=["frozen", "final"], default="final",
                    help="which set of agents: final (outputs/rl/final/, reported as s3_rl.md) or the frozen "
                         "design (outputs/rl/, reported as s3_rl_frozen.md)")
    ap.add_argument("--preliminary", action="store_true", help="partial run: outputs/s3_rl_preliminary.md only")
    ap.add_argument("--only", nargs="+", help="evaluate only these agents (file stems) into the cache; no report "
                    "(lets several processes share the work)")
    args = ap.parse_args()
    t0 = time.time()
    arr = kernel_arrays()
    tr, ev = le.member_split(arr.n_members)
    st = S2Settings.load()
    own_members = np.random.default_rng(SEED).choice(tr, len(ev), replace=False)
    rl_dir = RL_DIR / "final" if args.settings == "final" else RL_DIR
    files = sorted(p for p in rl_dir.glob("*.pt") if NAME.match(p.name) and NAME.match(p.name)["case"] in args.cases)
    if args.only:                                         # in the order given
        by_stem = {p.stem: p for p in files}
        files = [by_stem[k] for k in args.only if k in by_stem]
    rows = []
    if "london" in args.cases:
        lt, lassumed, lmonths, fold_of = london_envs(arr, tr, st)
    if "ercot" in args.cases:
        et, eassumed, emonths = ercot_envs(arr, tr, st)
        s4 = S4Eval(et.days)
    cache_dir = rl_dir / "eval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        g = NAME.match(p.name).groupdict()
        meta = {"case": g["case"], "agent": g["agent"], "ppo": bool(g["ppo"]), "fold": g["fold"], "seed": int(g["seed"])}
        cache = cache_dir / f"{p.stem}.parquet"
        if cache.exists() and cache.stat().st_mtime > p.stat().st_mtime:
            rows.append(pd.read_parquet(cache))
            continue
        if g["case"] == "london":
            test_m = [m for m in lmonths if fold_of[m] == g["fold"]]
            agent = rl.load_agent(p, lt)
            real = hs.evaluate(lt, agent, test_m, ev).assign(world="realised")
            own_env = lt if g["agent"] == "A" else lassumed(g["fold"])
            agent_own = rl.load_agent(p, own_env)
            own = hs.evaluate(own_env, agent_own, test_m, own_members if g["agent"] == "A" else [0]).assign(world="self-assessed")
        else:
            agent = rl.load_agent(p, et)
            real = with_s4(hs.evaluate(et, agent, emonths, ev), et, s4).assign(world="realised")
            own_env = et if g["agent"] == "A" else eassumed
            agent_own = rl.load_agent(p, own_env)
            own = with_s4(hs.evaluate(own_env, agent_own, emonths, own_members if g["agent"] == "A" else [0]), own_env, s4) \
                .assign(world="self-assessed")
        one = pd.concat([real, own], ignore_index=True).assign(**meta).drop(columns=["blocks", "effect", "day_idx"])
        one.to_parquet(cache, index=False)
        rows.append(one)
        print(f"{p.name} ({time.time() - t0:.0f} s)", flush=True)
    if args.only:
        return
    res = pd.concat(rows, ignore_index=True)
    tag = "_preliminary" if args.preliminary else ("" if args.settings == "final" else "_frozen")
    res.to_parquet(OUTPUTS / f"s3_rl_episodes{tag}.parquet", index=False)
    write_report(res, tag)


def write_report(res: pd.DataFrame, tag: str = "") -> None:
    rng = np.random.default_rng(SEED)
    lines = ["# S3 reinforcement learning: agents, baselines and overstatement", "",
             "Generated by `python scripts/evaluate_s3.py`. Deterministic policies; realised = trial-estimated environment, "
             "50 evaluation members, held-out months (London: the fold's months, cross-fitted; ERCOT: validation year "
             f"{VAL_YEAR}); self-assessed = the agent's own training environment (A: trial-estimated, 50 training members; "
             "B: Aalami et al. 2010 elasticity matrix). Evaluated limits (revenue +/-3%, segment cap +3%).", ""]
    real = res[res["world"] == "realised"]
    cols_l = {"daily_peak_cut_pct": "mean", "revenue_ratio": "mean", "violated": "mean",
              "bill_segment_max_nonresponding": "mean"}
    cols_e = {"reward": "mean", "fuel_cost": "mean", "co2_cost": "mean", "capacity_cost": "mean",
              "monthly_net_peak_change_mw": "mean", "s4_fuel": "mean", "s4_co2_t": "mean", "revenue_ratio": "mean",
              "violated": "mean"}
    for case, cols in (("london", cols_l), ("ercot", cols_e)):
        sub = real[real["case"] == case]
        if not len(sub):
            continue
        tab = sub.assign(policy=np.where(sub["ppo"], "PPO (unconstrained)", "PPO-Lagrangian agent " + sub["agent"])) \
            .groupby("policy").agg({**cols, "seed": "nunique", "month": "size"}).reset_index() \
            .rename(columns={"seed": "seeds", "month": "episodes"})
        lines += [f"## {case.upper() if case == 'ercot' else 'London'}: realised outcomes", "", md_table(tab, floatfmt="{:,.3f}"), ""]
    lines += ["## Overstatement (self-assessed / realised benefit)", ""]
    over = []
    main = res[~res["ppo"]]
    for case, metrics in (("london", [("daily_peak_cut_pct", 1.0, "daily peak cut")]),
                          ("ercot", [("reward", 1.0, "reward ($)"), ("s4_fuel", -1.0, "fuel saved (S4)"),
                                     ("s4_co2_t", -1.0, "CO2 saved (S4)"), ("monthly_net_peak_change_mw", -1.0, "net-load peak cut")])):
        for agent in ("A", "B"):
            sub = main[(main["case"] == case) & (main["agent"] == agent)]
            if not len(sub):
                continue
            s, r = sub[sub["world"] == "self-assessed"], sub[sub["world"] == "realised"]
            for col, sign, label in metrics:
                pnt, lo, hi = boot_ratio(s, r, col, sign, rng)
                over.append({"case": case, "agent": agent, "outcome": label, "self-assessed": sign * s[col].mean(),
                             "realised": sign * r[col].mean(), "ratio": pnt, "ratio lo": lo, "ratio hi": hi,
                             "realised violation rate": float(r["violated"].mean())})
    lines += [md_table(pd.DataFrame(over), floatfmt="{:,.3f}"), "",
              "Ratio > 1: the agent expects more than it delivers. Intervals: bootstrap over (seed, month) clusters. "
              "ERCOT fuel and CO2 are S4Model.impact corrected values (negative = saving before the sign flip)."]
    if tag == "_preliminary":
        lines.insert(1, "\nPRELIMINARY: partial set of trained agents; not for reporting.")
    elif tag == "_frozen":
        lines.insert(1, "\nFrozen-design agents (first runs, before settings were selected on the ERCOT validation year).")
    out = OUTPUTS / f"s3_rl{tag}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    if tag != "_preliminary":
        shutil.copy(out, ROOT / "docs" / "reports" / out.name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
