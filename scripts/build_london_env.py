"""Build the inputs of the S2 London environment and check them.

Run:  python scripts/build_london_env.py [--households 500]
      (needs scripts/build_lcl_panel.py and scripts/estimate_lcl_kernel.py first)

Control group as the S1 kernel's: Std training households of the main DiD sample, Economy 7-like
ones (night share above the q99 of the training dToU households) dropped. Baseline = their mean
load per half-hour; bill sample = a fixed random subset. Everything on the tariff-day grid
(ttrl.env.london).

Checks written to the report: the trial's prices were set so that an average customer who does not
react pays the flat rate over the year (LCL Report A3); the 2013 schedule, put on the block grid,
priced on the baseline with zero response should give a revenue ratio near 1. The schedule rules
(ttrl.env.tariff.TrialRules) are recomputed from the 2013 schedule, and the ensemble member split
is recorded.

Writes data/processed/env_london.npz, env_london_days.parquet; outputs/env_london_summary.md
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ttrl.analysis import lcl_controls as ctl
from ttrl.analysis import lcl_did as did
from ttrl.analysis import lcl_kernel as kn
from ttrl.data import lcl_features as lf
from ttrl.data import lcl_meters as lm
from ttrl.data import lcl_weather as lw
from ttrl.data import splits as sp
from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env import tariff as tf
from ttrl.paths import OUTPUTS
from ttrl.reporting import md_table

FIRST, LAST = "2012-07-01 00:30", "2014-02-28 00:00"
SAMPLE_SEED = 20261011


def schedule_on_grid(tdays: pd.DatetimeIndex) -> tuple[np.ndarray, int]:
    """The 2013 schedule as blocks (majority band of each block's six half-hours) and the number of
    half-hours whose band differs from the block's."""
    g = lf.price_grid()
    gi = ld.grid_index(g)
    codes = g["band"].map({"Low": tf.LOW, "Normal": tf.NORMAL, "High": tf.HIGH}).to_numpy(float)
    slots = ld.to_grid(codes, gi, tdays)                    # (n_days, 48), NaN outside 2013
    blocks = np.zeros((len(tdays), tf.BLOCKS_PER_DAY), np.int8)
    off = 0
    for k in range(tf.BLOCKS_PER_DAY):
        seg = slots[:, 6 * k:6 * k + 6]
        for d in range(len(tdays)):
            v = seg[d][~np.isnan(seg[d])]
            if len(v) == 0:
                continue
            vals, cnt = np.unique(np.round(v), return_counts=True)
            blocks[d, k] = int(vals[np.argmax(cnt)])
            off += int((np.round(v) != blocks[d, k]).sum())
    return blocks, off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=500)
    args = ap.parse_args()

    cal = did.calendar(FIRST, LAST)
    split = sp.household_split(lm.households())
    train = sp.restrict(did.select_households(did.coverage(cal)), split, "train")
    tou = train["tou"].to_numpy()
    share = ctl.night_share(ctl.pretrial_profiles(cal, train), cal)
    thr = ctl.night_threshold(share, train["tou"], 0.99)
    std = train[~tou].reset_index(drop=True)
    controls = std.loc[~(share[~tou].to_numpy() > thr), "hh_id"].to_numpy()
    print(f"controls: {len(controls)} of {len(std)} Std training households (threshold {thr:.4f})", flush=True)

    con = lm.connect()
    con.register("ctrl_df", pd.DataFrame({"hh_id": controls.astype(np.int32)}))
    con.register("cal_df", cal[["ts_label"]])
    m = con.execute(f"""SELECT c.ts_label, avg(m.kwh) AS kwh, count(*) AS n
                        FROM read_parquet('{lm.PANEL_PATH.as_posix()}') m JOIN ctrl_df USING (hh_id)
                        JOIN cal_df c USING (ts_label) WHERE m.kwh IS NOT NULL GROUP BY c.ts_label""").df()
    m = cal[["ts_label"]].merge(m, on="ts_label", how="left")
    tdays = ld.full_tariff_days(cal)
    gi = ld.grid_index(cal)
    baseline = ld.to_grid(m["kwh"].to_numpy(float), gi, tdays)
    days = ld.day_table(tdays, lw.daily_mean())
    if np.isnan(baseline).any():
        raise ValueError("baseline has missing half-hours")

    rng = np.random.default_rng(SAMPLE_SEED)
    hh_ids = np.sort(rng.choice(controls, size=min(args.households, len(controls)), replace=False))
    raw = ld.household_matrix(cal, hh_ids)
    hh = ld.to_grid(raw, gi, tdays)
    del raw
    missing_share = float(np.isnan(hh).mean())
    hh = ld.fill_by_slot(hh, days["weekend"].to_numpy())
    inputs = ld.LondonInputs(days, baseline, hh, hh_ids)
    inputs.save()
    print(f"saved {ld.INPUTS_PATH}", flush=True)

    # ---- checks
    blocks, off = schedule_on_grid(tdays)
    trial = (days["period"] == "trial").to_numpy()
    zero = np.zeros((int(trial.sum()), tf.HH_PER_DAY))
    rev0 = ld.revenue_ratio(blocks[trial], baseline[trial], zero)
    bills0 = ld.bill_changes(blocks[trial], hh[:, trial], zero)
    spec = kn.KernelSpec.from_json((OUTPUTS / "lcl_kernel_spec.json").read_text(encoding="utf-8"))
    terms = pd.read_csv(OUTPUTS / "lcl_kernel_terms.csv").set_index("term")["coef"]
    arrays_est = fk.from_coefficients(spec, terms)
    members = pd.read_parquet(OUTPUTS / "lcl_kernel_ensemble.parquet")
    arrays = fk.from_coefficients(spec, members)
    tr_m, ev_m = le.member_split(arrays.n_members)
    eff = le.settle(blocks[trial], arrays_est, 0)
    rev1 = ld.revenue_ratio(blocks[trial], baseline[trial], eff)
    pk0 = ld.daily_peak_kw(baseline[trial])
    pk1 = ld.daily_peak_kw(ld.responded(baseline[trial], eff))
    rules = tf.TrialRules()
    ok_days = sum(rules.day_ok(blocks[trial][max(0, i - 1):i], blocks[trial][i]) for i in range(int(trial.sum())))
    counts = days.groupby("period").size()

    report = "\n".join([
        "# S2 London environment: inputs and checks",
        "",
        "Generated by `python scripts/build_london_env.py`. Method: `src/ttrl/env/london.py`, `london_env.py`, "
        "`fastkernel.py`, `tariff.py`.",
        "",
        "## Inputs",
        "",
        f"* Control group (baseline): {len(controls)} Economy 7-screened Std training households (threshold {thr:.4f}).",
        f"* Tariff days (02:00-02:00): {len(tdays)} from {tdays[0].date()} to {tdays[-1].date()} "
        f"({', '.join(f'{k} {v}' for k, v in counts.items())}).",
        f"* Baseline: mean {baseline.mean():.4f} kWh per household-half-hour; mean daily peak {ld.daily_peak_kw(baseline).mean():.3f} kW.",
        f"* Bill sample: {len(hh_ids)} households (seed {SAMPLE_SEED}); {100 * missing_share:.2f}% of their half-hours "
        "were missing and filled with the household's own slot x day-type mean.",
        f"* Ensemble split (seed {le.MEMBER_SPLIT_SEED}): {len(tr_m)} training and {len(ev_m)} evaluation members.",
        "",
        "## Checks",
        "",
        f"* 2013 schedule on the block grid: {off} of 17,520 half-hours differ from their block's majority band "
        "(the eight events off the 3-hour grid). Per-day rules hold on "
        f"{ok_days} of {int(trial.sum())} days.",
        f"* Revenue ratio of the 2013 schedule on the baseline with zero response: {rev0:.4f} (the trial's prices were "
        "set for a ratio of 1 for an average non-responding customer). Household bill change with zero response: "
        f"median {100 * np.nanmedian(bills0):+.2f}%, P95 {100 * ld.percentile(bills0, 95):+.2f}%.",
        f"* With the kernel estimate: revenue ratio {rev1:.4f}; mean daily peak change "
        f"{np.mean(100 * (pk1 / pk0 - 1)):+.2f}% over 2013 days.",
        "",
        "## Schedule rules from the 2013 schedule",
        "",
        md_table(pd.DataFrame([{"rule": k, "value": v} for k, v in rules.__dict__.items()])),
    ])
    (OUTPUTS / "env_london_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
