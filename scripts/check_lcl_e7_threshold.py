"""How the S1 DiD depends on the Economy 7 screening threshold (robustness curve for review).

Run:  python scripts/check_lcl_e7_threshold.py [--boot 20]   (needs scripts/build_lcl_panel.py first)

Training households of the main DiD sample only. Std households whose Nov-Dec 2012 night share
(00:00-07:00) exceeds the q quantile of the dToU distribution are dropped, for q from none to 0.90
(ttrl.analysis.lcl_controls). Symmetric variants drop households above the threshold in both
groups (common-support trimming on a pre-treatment covariate). For each variant: the main event
terms, Low and High with calendar-month FE, the level term with month-of-year FE, and the
dToU-minus-control temperature sensitivity at night (heating degree-days x 4-hour block).

The question for review: does the estimate settle once the Economy 7-like households are out, or
does it keep moving as more households with high night use are dropped?

Writes to outputs/: lcl_e7_threshold.md, lcl_e7_threshold.csv, lcl_e7_threshold.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttrl.analysis import lcl_controls as ctl  # noqa: E402
from ttrl.analysis import lcl_did as did  # noqa: E402
from ttrl.data import lcl_meters as lm  # noqa: E402
from ttrl.data import lcl_weather as lw  # noqa: E402
from ttrl.data import splits as sp  # noqa: E402
from ttrl.paths import OUTPUTS  # noqa: E402
from ttrl.reporting import ci, md_table  # noqa: E402

FIRST, LAST = "2012-07-01 00:30", "2014-02-28 00:00"
SEED = 20261008
ASYM_Q = (0.999, 0.995, 0.99, 0.98, 0.975, 0.95, 0.90)
SYM_Q = (0.99, 0.95)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_ASYM, C_SYM = "#eb6834", "#2a78d6"


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20)
    args = ap.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    cal = did.calendar(FIRST, LAST)
    split = sp.household_split(lm.households())
    sample = sp.restrict(did.select_households(did.coverage(cal)), split, "train")
    tou = sample["tou"].to_numpy()
    n_tou, n_std = int(tou.sum()), int((~tou).sum())
    temp = lw.daily_mean()

    share = ctl.night_share(ctl.pretrial_profiles(cal, sample), cal)
    s_tou, s_std = share[tou].to_numpy(), share[~tou].to_numpy()
    variants = {"all controls": (None, ctl.subset_weights(n_tou, np.ones(n_std, bool), args.boot, SEED))}
    meta = {"all controls": {"kind": "none", "q": np.nan, "threshold": np.nan, "std_dropped": 0, "tou_dropped": 0}}
    for k, q in enumerate(ASYM_Q):
        thr = ctl.night_threshold(share, sample["tou"], q)
        keep = ~(s_std > thr)
        name = f"drop Std > q{100 * q:g}"
        variants[name] = (thr, ctl.subset_weights(n_tou, keep, args.boot, SEED + 1 + k))
        meta[name] = {"kind": "Std only", "q": q, "threshold": thr, "std_dropped": int((~keep).sum()), "tou_dropped": 0}
    for k, q in enumerate(SYM_Q):
        thr = ctl.night_threshold(share, sample["tou"], q)
        keep_s, keep_t = ~(s_std > thr), ~(s_tou > thr)
        name = f"drop both > q{100 * q:g}"
        variants[name] = (thr, ctl.subset_weights(n_tou, keep_s, args.boot, SEED + 20 + k, keep_tou=keep_t))
        meta[name] = {"kind": "both groups", "q": q, "threshold": thr, "std_dropped": int((~keep_s).sum()),
                      "tou_dropped": int((~keep_t).sum())}
    W, spans = ctl.stack_weights([w for _, w in variants.values()])
    span = dict(zip(variants, spans))
    sums = did.group_sums(cal, sample, W, dtype=np.float32)
    print(f"{sums.n_readings:,} readings, {len(variants)} variants", flush=True)

    X = did.event_design(cal)
    no_level = X.drop(columns=["trial", "post"])
    rows = []
    for v in variants:
        sv = sums.select(span[v])
        main_t = did.estimate(cal, sv, X).table.set_index("term")
        month_t = did.estimate(cal, sv, no_level, "month").table.set_index("term")
        moy_t = did.estimate(cal, sv, X, "moy").table.set_index("term")
        hdd = ctl.hdd_fit(cal, sv, X, temp, None, main_t).set_index("term")
        rec = {"variant": v, **meta[v]}
        for t in ("High", "Low", "Low_pre_2h", "Low_post_0_2h", "High_post_0_2h"):
            rec.update({f"{t}": main_t.loc[t, "pct"], f"{t}_lo": main_t.loc[t, "pct_lo"], f"{t}_hi": main_t.loc[t, "pct_hi"]})
        for t in ("High", "Low"):
            rec.update({f"{t}_month": month_t.loc[t, "pct"], f"{t}_month_lo": month_t.loc[t, "pct_lo"],
                        f"{t}_month_hi": month_t.loc[t, "pct_hi"]})
        rec.update({"trial_moy": moy_t.loc["trial", "pct"], "trial_moy_lo": moy_t.loc["trial", "pct_lo"],
                    "trial_moy_hi": moy_t.loc["trial", "pct_hi"]})
        for b in (0, 1):
            rec[f"hdd_b{b}_wh"] = 1000 * hdd.loc[f"hdd_b{b}", "coef_kwh"]
            rec[f"hdd_b{b}_se_wh"] = 1000 * hdd.loc[f"hdd_b{b}", "se"]
        rows.append(rec)
        print(f"[{v}] Low {rec['Low']:+.2f}  HDD00-04 {rec['hdd_b0_wh']:+.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "lcl_e7_threshold.csv", index=False)

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    asym = res[res["kind"] != "both groups"].sort_values("std_dropped")
    sym = res[res["kind"] == "both groups"]
    panels = [("Low", "Low event, % (main spec)"), ("Low_pre_2h", "2 h before Low, %"),
              ("hdd_b0_wh", "Night (00-04 h) HDD term, Wh per degree-day")]
    for ax, (col, title) in zip(axes, panels):
        for d, c, lab in ((asym, C_ASYM, "drop Std above threshold"), (sym, C_SYM, "drop both groups")):
            if col.startswith("hdd"):
                lo, hi = d[col] - 1.96 * d[col.replace("_wh", "_se_wh")], d[col] + 1.96 * d[col.replace("_wh", "_se_wh")]
            else:
                lo, hi = d[f"{col}_lo"], d[f"{col}_hi"]
            ax.errorbar(d["std_dropped"], d[col], yerr=[d[col] - lo, hi - d[col]], fmt="o-" if c == C_ASYM else "s",
                        ms=4, color=c, elinewidth=0.9, lw=1, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        q99 = res.loc[res["variant"] == "drop Std > q99", "std_dropped"].iloc[0]
        ax.axvline(q99, color=INK2, ls="--", lw=0.8)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlabel("Std households dropped (dashed: q99 rule)", fontsize=8, color=INK2)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "lcl_e7_threshold.png", dpi=150)
    plt.close(fig)

    # ---- report
    tab = pd.DataFrame({
        "variant": res["variant"],
        "threshold": res["threshold"].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}"),
        "Std / dToU dropped": [f"{a} / {b}" for a, b in zip(res["std_dropped"], res["tou_dropped"])],
        "High": [ci(r["High"], r["High_lo"], r["High_hi"]) for _, r in res.iterrows()],
        "Low": [ci(r["Low"], r["Low_lo"], r["Low_hi"]) for _, r in res.iterrows()],
        "Low (month FE)": [ci(r["Low_month"], r["Low_month_lo"], r["Low_month_hi"]) for _, r in res.iterrows()],
        "2 h before Low": [ci(r["Low_pre_2h"], r["Low_pre_2h_lo"], r["Low_pre_2h_hi"]) for _, r in res.iterrows()],
        "level (moy FE)": [ci(r["trial_moy"], r["trial_moy_lo"], r["trial_moy_hi"]) for _, r in res.iterrows()],
        "HDD 00-04 h, Wh": [f"{r['hdd_b0_wh']:+.2f} ({r['hdd_b0_se_wh']:.2f})" for _, r in res.iterrows()],
        "HDD 04-08 h, Wh": [f"{r['hdd_b1_wh']:+.2f} ({r['hdd_b1_se_wh']:.2f})" for _, r in res.iterrows()],
    })
    report = "\n".join([
        "# LCL S1: Economy 7 screening threshold (robustness curve)",
        "",
        "Generated by `python scripts/check_lcl_e7_threshold.py`. S0 training households of the main DiD sample "
        f"({n_std} Std, {n_tou} dToU); held-out households are not used. Night share: Nov-Dec 2012 consumption "
        "between 00:00 and 07:00 (pre-trial data only). 'drop Std > qX' drops Std households above the X-th "
        "percentile of the dToU distribution (the dToU offer excluded Economy 7); 'drop both' applies the same "
        "threshold to both groups. Percent effects as in `lcl_did`; 95% intervals from the combined standard error "
        f"with only {args.boot} household-bootstrap draws per variant (for the curve, not for headline numbers). "
        "HDD terms: dToU-minus-control load per heating degree-day (base 15.5 C), Wh per household-half-hour "
        "(combined SE); negative = controls heat more at night.",
        "",
        md_table(tab),
        "",
        "See `lcl_e7_threshold.png`.",
    ])
    (OUTPUTS / "lcl_e7_threshold.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
