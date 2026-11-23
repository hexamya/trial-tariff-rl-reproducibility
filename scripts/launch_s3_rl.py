"""Run a batch of RL trainings (scripts/run_s3_rl.py) in parallel processes.

Run:  python scripts/launch_s3_rl.py --batch seed0 [--parallel 6]

Batches: ``seed<k>`` = London agents A and B on the four folds plus ERCOT A and B, seed k (10 runs);
``ppo`` = the unconstrained PPO baseline (agent A; London four folds and ERCOT, seed 0); ``all`` = the
three seeds and ``ppo`` in that order. Runs whose model file exists are skipped. Logs go to
outputs/rl/logs/. A full batch takes hours; start it detached from the shell (e.g. PowerShell
Start-Process) so that it is not bound to a terminal session.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ttrl.paths import OUTPUTS

RL_DIR = OUTPUTS / "rl"
SCRIPT = Path(__file__).resolve().parent / "run_s3_rl.py"
FOLDS = ("A", "B", "C", "D")


def batch(name: str) -> list[dict]:
    if name == "all":                                   # every run of the S3 design, in order
        return [r for b in ("seed0", "seed1", "seed2", "ppo") for r in batch(b)]
    if name.startswith("seed"):
        k = int(name[4:])
        runs = [{"case": "london", "agent": a, "fold": f, "seed": k} for a in ("A", "B") for f in FOLDS]
        return runs + [{"case": "ercot", "agent": a, "fold": None, "seed": k} for a in ("A", "B")]
    if name == "ppo":
        runs = [{"case": "london", "agent": "A", "fold": f, "seed": 0, "unconstrained": True} for f in FOLDS]
        return runs + [{"case": "ercot", "agent": "A", "fold": None, "seed": 0, "unconstrained": True}]
    raise SystemExit(f"unknown batch {name}")


def name_of(r: dict) -> str:
    return f"{r['case']}_{r['agent']}{'_ppo' if r.get('unconstrained') else ''}_{r['fold'] or 'all'}_s{r['seed']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--settings", choices=["frozen", "final"], default="frozen")
    args = ap.parse_args()
    out_dir = RL_DIR / "final" if args.settings == "final" else RL_DIR
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    todo = [r for r in batch(args.batch) if not (out_dir / f"{name_of(r)}.pt").exists()]
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < args.parallel:
            r = todo.pop(0)
            cmd = [sys.executable, str(SCRIPT), "--case", r["case"], "--agent", r["agent"], "--seed", str(r["seed"]),
                   "--threads", "1", "--settings", args.settings] + (["--fold", r["fold"]] if r["fold"] else []) \
                + (["--unconstrained"] if r.get("unconstrained") else [])
            log = open(out_dir / "logs" / f"{name_of(r)}.log", "w")
            running.append((name_of(r), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"started {name_of(r)} ({time.time() - t0:.0f} s)", flush=True)
        time.sleep(10)
        for item in list(running):
            n, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print(f"finished {n} exit {proc.returncode} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
