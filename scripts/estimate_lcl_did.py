"""Event-study DiD estimate of the dToU price response (S1) on the S0 training households.

Run:  python scripts/estimate_lcl_did.py [--boot 200] [--split train|all]
      (needs scripts/build_lcl_panel.py first)

Sample: dToU (ToU) and flat-rate (Std) households with >= 80% of pre-trial and >= 90% of
trial-year half-hours, from the training split (ttrl.data.splits; ``--split all`` reproduces
the exploratory analysis on every household). The reserved households are used for
the final out-of-sample check. Window: Jul 2012 - Feb 2014 (labels 2012-07-01 00:30 ... 2014-02-28 00:00).
Method and regressors: see ttrl.analysis.lcl_did.

Specifications:
  main         half-hour-of-day x weekend FE; trial, post, High, Low, pre/post windows
  month FE     + calendar month x 4-hour block FE (absorbs trial/post)
  moy FE       + month-of-year x 4-hour block FE (trial level from Jul-Dec 2012 vs Jul-Dec 2013)
  pre-trend    month-of-sample dummies (Dec 2012 = 0) with the event terms
  profile      one dummy per half-hour of event time
  start labels meter labels read as interval starts (everything 30 min later)
  loose sample >= 50% coverage in both periods

Writes to outputs/: lcl_did_summary.md, lcl_did_terms.csv, lcl_did_boot_main.parquet,
lcl_did_boot_profile.parquet, lcl_did_profile.png, lcl_did_pretrend.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttrl.analysis import lcl_did as did  # noqa: E402
from ttrl.data import lcl_meters as lm  # noqa: E402
from ttrl.data import splits as sp  # noqa: E402
from ttrl.paths import OUTPUTS  # noqa: E402
from ttrl.reporting import ci, md_table  # noqa: E402

FIRST, LAST = "2012-07-01 00:30", "2014-02-28 00:00"
SEED = 20130101
COLORS = {"High": "#eb6834", "Low": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEY_TERMS = ["High", "Low", "trial", "post", "High_pre_2h", "High_post_0_2h", "High_post_2_4h",
             "Low_pre_2h", "Low_post_0_2h", "Low_post_2_4h"]


def run(cal, sample, tag, n_boot, specs):
    """One pass over the panel for the group sums, then estimate each (name, X, fe) spec."""
    sums = did.group_sums(cal, sample, did.bootstrap_weights(sample, n_boot, SEED))
    print(f"[{tag}] {sums.n_readings:,} readings", flush=True)
    return {name: did.estimate(cal, sums, X, fe) for name, X, fe in specs}, sums


def fmt_terms(tab: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "term": tab["term"], "half-hours": tab["n_hh"],
        "effect kWh/hh": tab["coef_kwh"].map("{:+.4f}".format),
        "dToU mean kWh": tab["tou_mean_kwh"].map("{:.3f}".format),
        "% [95% CI combined]": [ci(r.pct, r.pct_lo, r.pct_hi) for r in tab.itertuples()],
        "% CI date-cluster": [f"[{r.pct_lo_cluster:+.1f}, {r.pct_hi_cluster:+.1f}]" for r in tab.itertuples()],
        "% CI household boot": [f"[{r.pct_lo_boot:+.1f}, {r.pct_hi_boot:+.1f}]" for r in tab.itertuples()],
    })


def compare(results: dict[str, did.DidResult]) -> pd.DataFrame:
    rows = []
    for t in KEY_TERMS:
        rec = {"term": t}
        for name, r in results.items():
            row = r.table.set_index("term")
            rec[name] = ci(*row.loc[t, ["pct", "pct_lo", "pct_hi"]]) if t in row.index else "-"
        rows.append(rec)
    return pd.DataFrame(rows)


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.axhline(0, color=INK2, lw=0.8)


def profile_frame(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in tab.itertuples():
        parts = r.term.split("_")
        if parts[0] not in ("High", "Low"):
            continue
        b, kind, k = parts[0], parts[1], parts[2]
        if kind == "pre":
            x = -int(k)
        elif kind == "in":
            x = did.PROFILE_IN if k == "late" else int(k)
        else:
            x = did.PROFILE_IN + int(k)
        rows.append({"band": b, "kind": kind, "x": x, "pct": r.pct, "lo": r.pct_lo, "hi": r.pct_hi})
    return pd.DataFrame(rows)


def fig_profile(tab: pd.DataFrame, path) -> None:
    pf = profile_frame(tab)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    for ax, b in zip(axes, ("High", "Low")):
        d = pf[pf["band"] == b].sort_values("x")
        ax.axvspan(-0.5, did.PROFILE_IN + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.errorbar(d["x"], d["pct"], yerr=[d["pct"] - d["lo"], d["hi"] - d["pct"]], fmt="o", ms=3.5,
                    color=COLORS[b], ecolor=COLORS[b], elinewidth=1, capsize=0)
        ticks = list(range(-did.PROFILE_PRE, did.PROFILE_IN + did.PROFILE_POST + 1))
        labels = [f"{t}" for t in range(-did.PROFILE_PRE, 0)] + [f"e{j}" for j in range(did.PROFILE_IN)] + \
                 ["late"] + [f"+{k}" for k in range(1, did.PROFILE_POST + 1)]
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_title(f"{b} events: dToU vs Std, % of counterfactual", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("half-hours: before start | in event (shaded) | after end", fontsize=8, color=INK2)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pretrend(tab: pd.DataFrame, path) -> None:
    m = tab[tab["term"].str.startswith("m_")].copy()
    m["month"] = pd.PeriodIndex(m["term"].str[2:], freq="M")
    m = pd.concat([m, pd.DataFrame({"month": [pd.Period("2012-12", "M")], "pct": [0.0], "pct_lo": [0.0],
                                    "pct_hi": [0.0]})]).sort_values("month")
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.errorbar(x, m["pct"], yerr=[m["pct"] - m["pct_lo"], m["pct_hi"] - m["pct"]], fmt="o", ms=3.5,
                color=INK, elinewidth=1)
    ax.axvline(x[list(m["month"].astype(str)).index("2012-12")] + 0.5, color=INK2, ls="--", lw=0.8)
    ax.set_xticks(x, m["month"].astype(str), rotation=60, fontsize=7)
    ax.set_title("dToU minus Std on Normal half-hours by month, % (Dec 2012 = 0)", fontsize=9, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--split", choices=["train", "all"], default="train",
                    help="households to use; the S0 holdout is never available here")
    args = ap.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    cal = did.calendar(FIRST, LAST)
    cov = did.coverage(cal)
    sample = did.select_households(cov)
    loose = did.select_households(cov, 0.5, 0.5)
    split = sp.household_split(lm.households())
    if args.split == "train":
        sample, loose = sp.restrict(sample, split, "train"), sp.restrict(loose, split, "train")
        split_txt = (f"S0 training split only ({1 - sp.HOLDOUT_SHARE:.0%} of each group, split digest "
                     f"`{sp.split_digest(split)}`); held-out households are not used.")
    else:
        split_txt = "Every household, as in the exploratory analysis before the household split."
    n_grp = lambda s: s["tou"].map({True: "ToU", False: "Std"}).value_counts().to_dict()  # noqa: E731

    X = did.event_design(cal)
    Xp = did.event_design(cal, profile=True)
    month = cal["ts_local"].dt.to_period("M").astype(str)
    Xm = pd.get_dummies("m_" + month, dtype=float).drop(columns="m_2012-12")
    Xm = pd.concat([Xm, X.drop(columns=["trial", "post"])], axis=1)
    no_level = X.drop(columns=["trial", "post"])
    specs = [("main", X, None), ("month FE", no_level, "month"), ("moy FE", X, "moy"),
             ("pre-trend", Xm, None), ("profile", Xp, None)]
    res, _ = run(cal, sample, "main", args.boot, specs)

    cal_s = did.calendar(FIRST, LAST, label="start")
    res_s, _ = run(cal_s, sample, "start", args.boot, [("start labels", did.event_design(cal_s), None)])
    res_l, _ = run(cal, loose, "loose", args.boot, [("loose sample", X, None)])
    main_r = res["main"]

    allres = {"main": main_r, "month FE": res["month FE"], "moy FE": res["moy FE"],
              "start labels": res_s["start labels"], "loose sample": res_l["loose sample"]}
    terms = pd.concat([r.table.assign(spec=k) for k, r in {**res, **res_s, **res_l}.items()])
    terms.to_csv(OUTPUTS / "lcl_did_terms.csv", index=False)
    main_r.boot.to_parquet(OUTPUTS / "lcl_did_boot_main.parquet")
    res["profile"].boot.to_parquet(OUTPUTS / "lcl_did_boot_profile.parquet")
    fig_profile(res["profile"].table, OUTPUTS / "lcl_did_profile.png")
    fig_pretrend(res["pre-trend"].table, OUTPUTS / "lcl_did_pretrend.png")

    pt = res["pre-trend"].table
    pm = pt[pt["term"].str.startswith("m_")].copy()
    pm.index = pm["term"].str[2:].to_numpy()
    pm.loc["2012-12", ["pct", "pct_lo", "pct_hi"]] = 0.0
    monthly = pd.DataFrame({"month": sorted(pm.index)})
    monthly["gap % [95% CI combined]"] = [
        "0 (reference)" if m == "2012-12" else ci(*pm.loc[m, ["pct", "pct_lo", "pct_hi"]]) for m in monthly["month"]]
    yoy = pd.DataFrame([{"month of year": m[5:], "earlier": m, "later": f"{int(m[:4]) + 1}{m[4:]}",
                         "change, points": pm.loc[f"{int(m[:4]) + 1}{m[4:]}", "pct"] - pm.loc[m, "pct"]}
                        for m in sorted(pm.index) if f"{int(m[:4]) + 1}{m[4:]}" in pm.index])
    t = main_r.table.set_index("term")
    tm = res["moy FE"].table.set_index("term")

    report = "\n".join([
        "# LCL dToU 2013: event-study DiD estimate (S1)",
        "",
        f"Generated by `python scripts/estimate_lcl_did.py --split {args.split}`. Causal design: dToU households against the flat-rate "
        "(Std) control group, household x half-hour-of-day x weekend and half-hour (date x time) fixed effects. "
        "Method details: `src/ttrl/analysis/lcl_did.py`.",
        "",
        "## Key results (main specification)",
        "",
        f"* High-price half-hours: **{ci(t.loc['High', 'pct'], t.loc['High', 'pct_lo'], t.loc['High', 'pct_hi'])}%** "
        "relative to Normal-price half-hours of the trial.",
        f"* Low-price half-hours: **{ci(t.loc['Low', 'pct'], t.loc['Low', 'pct_lo'], t.loc['Low', 'pct_hi'])}%**.",
        f"* Normal-price level vs pre-trial (Normal 11.76 p vs flat 14.228 p): "
        f"{ci(t.loc['trial', 'pct'], t.loc['trial', 'pct_lo'], t.loc['trial', 'pct_hi'])}% in the main "
        f"specification and {ci(tm.loc['trial', 'pct'], tm.loc['trial', 'pct_lo'], tm.loc['trial', 'pct_hi'])}% "
        "comparing the same calendar months (moy FE, preferred: the dToU-Std gap is seasonal, see Pre-trends). "
        "Negative although the Normal price is below the flat rate. "
        f"After the trial (Jan-Feb 2014): {ci(t.loc['post', 'pct'], t.loc['post', 'pct_lo'], t.loc['post', 'pct_hi'])}%.",
        "* Intervals are 95% from the combined standard error (date-clustered and household-bootstrap variances "
        "added); the separate intervals are in the table.",
        f"* Descriptive matched-day numbers from the workbook (no control group) were High -5.8 [-7.2, -4.4]% and "
        "Low +5.4 [+3.2, +7.8]%.",
        "",
        "## Sample",
        "",
        f"Window {FIRST} to {LAST} (meter labels, UTC half-hour ends), {main_r.n_obs:,} half-hours with both "
        f"groups observed. Households: main sample {n_grp(sample)} (>= 80% of pre-trial and >= 90% of trial "
        f"half-hours); loose sample {n_grp(loose)} (>= 50% of both); all households with any reading in the "
        f"window {n_grp(cov)}. {split_txt} Bootstrap: {args.boot} household resamples within group.",
        "",
        "## Main specification",
        "",
        md_table(fmt_terms(main_r.table)),
        "",
        "`trial` is measured on Normal 2013 half-hours outside all event windows; `High`/`Low` are relative to "
        "Normal 2013 half-hours; window terms are Normal half-hours 2 h before or 0-2 h / 2-4 h after an event. "
        "Percentages are relative to the dToU counterfactual on the same half-hours (dToU mean minus the effect).",
        "",
        "## Robustness (% [95% CI combined])",
        "",
        md_table(compare(allres)),
        "",
        "`month FE`: calendar month x 4-hour block FE, so event effects come from event vs non-event days within "
        "the month and the trial/post levels are absorbed. `moy FE`: month-of-year x 4-hour block FE, so the "
        "trial level compares Jul-Dec 2013 with Jul-Dec 2012. `start labels`: meter labels read as interval starts.",
        "",
        "## Pre-trends",
        "",
        "Monthly dToU-minus-Std gap relative to Dec 2012 (month-of-sample dummies with the event terms, so 2013 "
        "months are Normal-price levels). See `lcl_did_pretrend.png`.",
        "",
        md_table(monthly),
        "",
        "Same calendar month one year later (change in the gap, percentage points):",
        "",
        md_table(yoy, floatfmt="{:+.1f}"),
        "",
        "The gap is seasonal in both years (higher in summer, lower in winter), so the fall from Jul to Dec 2012 "
        "need not be a pre-trend; with six pre-trial months a trend cannot be ruled out either. The pre-trial "
        "window holds Jul-Dec only, so the main specification's `trial` term mixes this seasonality in; the moy-FE "
        "version compares like months.",
        "",
        "## Event-time profile",
        "",
        md_table(fmt_terms(res["profile"].table)),
        "",
        "See `lcl_did_profile.png`. The bootstrap draws of the main and profile coefficients "
        "(`lcl_did_boot_*.parquet`) are the first version of the response-kernel ensemble.",
    ])
    (OUTPUTS / "lcl_did_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
