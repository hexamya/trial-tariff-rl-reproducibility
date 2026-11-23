"""Run the whole S4 pipeline (ERCOT gas fleet) in order.

Steps (each is also runnable on its own):

1. ``pudl``: ERCOT gas generators, CEMS units and hourly CEMS from PUDL on S3
   (``ttrl.data.pudl.build_ercot_gas_cems``; cached in data/interim).
2. ``weather``: plant temperatures from NOAA ISD-lite (scripts/build_plant_weather.py).
3. ``dispatch``: temperature-free dispatch validation (scripts/validate_dispatch.py).
4. ``heatrate``: temperature-dependent heat rate and derating (scripts/fit_heatrate_derating.py).
5. ``integrate``: temperature-aware dispatch, validation, S4 error band (scripts/integrate_s4.py).
6. ``example``: worked example of the impact function (scripts/s4_worked_example.py).
7. ``summary``: one-page S4 summary (scripts/s4_summary.py).

Downloads are cached, so a re-run reads only local files. The runtime of
every step run is written to outputs/s4_pipeline_runtime.json (the latest
timing of each step is kept).

Usage:
    python scripts/run_s4.py [--year 2024] [--from STEP] [--only STEP]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from ttrl.paths import OUTPUTS

SCRIPTS = Path(__file__).resolve().parent
STEPS = [("pudl", None), ("weather", "build_plant_weather.py"), ("dispatch", "validate_dispatch.py"),
         ("heatrate", "fit_heatrate_derating.py"), ("integrate", "integrate_s4.py"),
         ("example", "s4_worked_example.py"), ("summary", "s4_summary.py")]
TIMING_KEYS = {"example": "s4_worked_example", "summary": "s4_summary"}


def run_step(name: str, script: str | None, year: int) -> None:
    """Run one step in a fresh Python process (fails loudly on error)."""
    if script is None:
        code = f"from ttrl.data import pudl; print(pudl.build_ercot_gas_cems({year}))"
        cmd = [sys.executable, "-c", code]
    else:
        cmd = [sys.executable, str(SCRIPTS / script), "--year", str(year)]
    print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=2024)
    names = [n for n, _ in STEPS]
    ap.add_argument("--from", dest="start", choices=names, default=names[0],
                    help="first step to run (earlier outputs must exist)")
    ap.add_argument("--only", choices=names, help="run a single step")
    args = ap.parse_args()
    todo = [s for s in STEPS if (s[0] == args.only if args.only else
                                 names.index(s[0]) >= names.index(args.start))]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    timing_path = OUTPUTS / "s4_pipeline_runtime.json"
    timing = {"year": args.year, "steps": {}, "dates": {}}
    if timing_path.exists():            # keep the latest timing of steps not run now
        old = json.loads(timing_path.read_text())
        if old.get("year") == args.year:
            timing["steps"].update(old.get("steps", {}))
            timing["dates"].update(old.get("dates", {}))
    t_all = time.time()
    for name, script in todo:
        t0 = time.time()
        run_step(name, script, args.year)
        key = TIMING_KEYS.get(name, name)
        timing["steps"][key] = round(time.time() - t0, 1)
        timing["dates"][key] = date.today().isoformat()
        timing_path.write_text(json.dumps(timing, indent=2))
    print(f"\nS4 pipeline done in {(time.time() - t_all) / 60:.1f} min; step times in {timing_path}")


if __name__ == "__main__":
    main()
