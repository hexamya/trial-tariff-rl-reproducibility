"""Train one PPO-Lagrangian agent (S3) and save it with its learning curve.

Run:  python scripts/run_s3_rl.py --case london --agent A --fold A --seed 0
      python scripts/run_s3_rl.py --case ercot --agent B --seed 1

``--agent A``: trained on the trial-estimated response (the 150 training ensemble members);
``--agent B``: trained on the assumed response (Aalami et al. 2010 elasticity matrix, hour categories by
the training months' load shape). London: trained on the months outside ``--fold`` (cross-fitting);
ERCOT: trained on the 2019-2022 weather years (curtailed days masked). ``--unconstrained``: PPO without
the Lagrangian (baseline 8). Training uses constraints tightened by ``TRAIN_BACKOFF`` (revenue band and
segment cap 0.5 points inside the evaluated limits), because the multipliers control the stochastic
policy's costs while the deterministic policy is evaluated; the margin was set from the deterministic
policy's excess on the training months in the pilot runs. Writes outputs/rl/<run>.pt and
<run>_history.csv; evaluation is scripts/evaluate_s3.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from run_s3_baselines import TRAIN_YEARS, kernel_arrays
from ttrl.env import assumed as am
from ttrl.env import ercot_env as ee
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env.responses import AssumedResponse, TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS
from ttrl.s3 import harness as hs
from ttrl.s3 import patterns as pt
from ttrl.s3 import rl

RL_DIR = OUTPUTS / "rl"
# Settings chosen on the ERCOT validation year (scripts/tune_s3_rl.py, candidate c4); runs with
# --settings final are saved under outputs/rl/final/.
FINAL_SETTINGS = {"entropy_final": 0.0, "ki": 0.005}
FINAL_DIR = RL_DIR / "final"
TRAIN_BACKOFF = 0.005
ITERATIONS = 800
ERCOT_REWARD_SCALE = 1e-5          # $ -> 100 k$ units


def run_name(case: str, agent: str, fold: str | None, seed: int, unconstrained: bool) -> str:
    return f"{case}_{agent}{'_ppo' if unconstrained else ''}_{fold or 'all'}_s{seed}"


def london_setup(agent: str, fold: str, arr, tr, st):
    inp = ld.LondonInputs.load()
    trial = inp.days[inp.days["period"] == "trial"]
    months = sorted(trial["month"].unique())
    fold_of = trial.groupby("month")["fold"].first().to_dict()
    train_m = [m for m in months if fold_of[m] != fold]
    if agent == "A":
        response = TrialResponse(arr)
    else:
        train_days = inp.days["month"].isin(train_m).to_numpy()
        response = AssumedResponse(am.aalami_2010(am.categories_by_load(inp.baseline[train_days])))
    cfg = le.LondonConfig.from_settings(st, months, tr)
    cfg.revenue_tol -= TRAIN_BACKOFF
    cfg.segment_bill_cap -= TRAIN_BACKOFF
    env = le.LondonTariffEnv(inp, response, cfg, seed=0)
    return env, train_m


def ercot_setup(agent: str, arr, tr, st):
    envs = {}
    days = {y: ee.ErcotDays.load(y) for y in TRAIN_YEARS}
    if agent == "A":
        response = TrialResponse(arr)
    else:
        prof = np.concatenate([np.repeat(d.year.residential[d.row[np.all(d.row >= 0, axis=1)]], 2, axis=1)
                               for d in days.values()])
        response = AssumedResponse(am.aalami_2010(am.categories_by_load(prof)))
    for y, d in days.items():
        months = [f"{y}-{m:02d}" for m in range(1, 13)]
        cfg = ee.ErcotConfig.from_settings(st, months, tr)
        cfg.revenue_tol -= TRAIN_BACKOFF
        envs[y] = ee.ErcotTariffEnv(d, response, cfg, seed=0)
    train_m = [f"{y}-{m:02d}" for y in TRAIN_YEARS for m in range(1, 13)]
    return hs.MonthRouter(envs), train_m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["london", "ercot"], required=True)
    ap.add_argument("--agent", choices=["A", "B"], required=True)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--unconstrained", action="store_true")
    ap.add_argument("--entropy-final", type=float, default=None)
    ap.add_argument("--lambda-max", type=float, default=None)
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--tag", default="", help="tuning run: saved under outputs/rl/tune/<tag>/")
    ap.add_argument("--settings", choices=["frozen", "final"], default="frozen",
                    help="final: the settings chosen on the ERCOT validation year (FINAL_SETTINGS)")
    args = ap.parse_args()
    if args.settings == "final":
        args.entropy_final, args.ki = FINAL_SETTINGS["entropy_final"], FINAL_SETTINGS["ki"]
    torch.set_num_threads(args.threads)
    arr = kernel_arrays()
    tr, _ = le.member_split(arr.n_members)
    st = S2Settings.load()
    if args.case == "london":
        if args.fold is None:
            raise SystemExit("--fold is required for London")
        env, train_m = london_setup(args.agent, args.fold, arr, tr, st)
        scale = 1.0
    else:
        env, train_m = ercot_setup(args.agent, arr, tr, st)
        scale = ERCOT_REWARD_SCALE
    cfg = rl.PPOConfig(iterations=args.iterations, seed=args.seed, reward_scale=scale, lagrangian=not args.unconstrained,
                       entropy_final=args.entropy_final, lambda_max=args.lambda_max)
    if args.ki is not None:
        cfg.pid = (cfg.pid[0], args.ki, cfg.pid[2])
    name = run_name(args.case, args.agent, args.fold, args.seed, args.unconstrained)
    agent, hist = rl.train(env, train_m, tr, pt.candidate_patterns(), cfg, log_every=50, label=name,
                           eval_every=50, eval_members=np.random.default_rng(args.seed).choice(tr, 5, replace=False))
    out_dir = RL_DIR / "tune" / args.tag if args.tag else (FINAL_DIR if args.settings == "final" else RL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rl.save_agent(agent, cfg, out_dir / f"{name}.pt")
    hist.to_csv(out_dir / f"{name}_history.csv", index=False)
    print(f"saved {out_dir / f'{name}.pt'}")


if __name__ == "__main__":
    main()
