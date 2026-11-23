"""Control-group checks for the S1 DiD: Economy 7-like controls, temperature, synthetic control.

Run:  python scripts/check_lcl_controls.py [--boot 60]   (needs scripts/build_lcl_panel.py first)

Training households of the main DiD sample only (S0 split; the holdout is refused). Methods in
ttrl.analysis.lcl_controls. Variants of the control group, each with its own household bootstrap:

  all controls          every Std household (the main DiD)
  no E7-like (q99)      Std households whose Nov-Dec 2012 night share (00:00-07:00) exceeds the
                        99th percentile of the dToU group are dropped
  no E7-like (q97.5)    the same at the 97.5th percentile
  synthetic control     Std weights that reproduce the dToU mean pre-trial profile
                        (month x half-hour x weekend), re-solved on every bootstrap draw

Also: differential temperature sensitivity (dToU minus Std load on heating degree-days by
4-hour block) and what it does to the Normal-price level term; an in-time placebo for the
synthetic control (fitted on Jul-Oct 2012, checked on Nov-Dec 2012).

Writes to outputs/: lcl_controls_summary.md, lcl_controls_terms.csv, lcl_controls_night.png,
lcl_controls_gap.png
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
SEED = 20261006
HDD_BASE = 15.5
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VAR_COLORS = {"all controls": "#0b0b0b", "no E7-like (q99)": "#eb6834", "synthetic control": "#2a78d6"}
TERMS = ["High", "Low", "trial", "post", "High_pre_2h", "High_post_0_2h", "Low_pre_2h", "Low_post_0_2h"]


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)


def monthly_gap(sums_level, rows_tou, rows_std, cal):
    """Monthly percent gap of dToU over a control from level means (full sample row)."""
    t = sums_level.mean("kwh", True)[rows_tou]
    s = sums_level.mean("kwh", False)[rows_std]
    month = cal["ts_local"].dt.to_period("M").astype(str).to_numpy()
    df = pd.DataFrame({"month": month, "tou": t, "ctrl": s}).dropna()
    g = df.groupby("month")[["tou", "ctrl"]].mean()
    return 100 * (g["tou"] / g["ctrl"] - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=60)
    args = ap.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    cal = did.calendar(FIRST, LAST)
    split = sp.household_split(lm.households())
    sample = sp.restrict(did.select_households(did.coverage(cal)), split, "train")
    tou = sample["tou"].to_numpy()
    n_tou, n_std = int(tou.sum()), int((~tou).sum())
    temp = lw.daily_mean()

    # ---- night profile (pre-trial only)
    P = ctl.pretrial_profiles(cal, sample)
    share = ctl.night_share(P, cal)
    thr = {q: ctl.night_threshold(share, sample["tou"], q) for q in (0.95, 0.975, 0.99)}
    s_std = share[~tou]
    flags = {q: (s_std > thr[q]).to_numpy() for q in thr}  # NaN share (no Nov-Dec reading) -> not flagged
    print(f"night share: thresholds {thr}; flagged {[int(f.sum()) for f in flags.values()]}", flush=True)

    # ---- synthetic control weights
    M = ctl.fill_profiles(P)
    month_lv = M.columns.get_level_values("month").to_numpy()
    Mn = M.to_numpy()
    W_sc, sc_stats = ctl.sc_weights(Mn[tou], Mn[~tou], args.boot, SEED + 3)
    W_pl, pl_stats = ctl.sc_weights(Mn[tou], Mn[~tou], 0, SEED + 4, moments=np.flatnonzero(month_lv <= 10))
    print(f"synthetic control: {sc_stats[0]}", flush=True)

    variants = {
        "all controls": ctl.subset_weights(n_tou, np.ones(n_std, bool), args.boot, SEED),
        "no E7-like (q99)": ctl.subset_weights(n_tou, ~flags[0.99], args.boot, SEED + 1),
        "no E7-like (q97.5)": ctl.subset_weights(n_tou, ~flags[0.975], args.boot, SEED + 2),
        "synthetic control": W_sc,
        "SC placebo (Jul-Oct fit)": W_pl,
    }
    W, spans = ctl.stack_weights(list(variants.values()))
    span = dict(zip(variants, spans))
    sums = did.group_sums(cal, sample, W, dtype=np.float32)
    print(f"{sums.n_readings:,} readings", flush=True)

    X = did.event_design(cal)
    no_level = X.drop(columns=["trial", "post"])
    month = cal["ts_local"].dt.to_period("M").astype(str)
    Xm = pd.concat([pd.get_dummies("m_" + month, dtype=float).drop(columns="m_2012-12"), no_level], axis=1)
    res, terms = {}, []
    for v in list(variants)[:4]:
        sv = sums.select(span[v])
        res[v] = {"main": did.estimate(cal, sv, X), "moy FE": did.estimate(cal, sv, X, "moy"),
                  "month FE": did.estimate(cal, sv, no_level, "month"), "pre-trend": did.estimate(cal, sv, Xm)}
        for spec, r in res[v].items():
            terms.append(r.table.assign(variant=v, spec=spec))
        print(f"[{v}] done", flush=True)
    pd.concat(terms).to_csv(OUTPUTS / "lcl_controls_terms.csv", index=False)

    # ---- temperature sensitivity
    hdd_rows = []
    for v in ("all controls", "no E7-like (q99)"):
        sv = sums.select(span[v])
        for fe, base in ((None, "main"), ("moy", "moy FE")):
            h = ctl.hdd_fit(cal, sv, X, temp, fe, res[v][base].table.set_index("term"), HDD_BASE)
            hdd_rows.append(h.assign(variant=v, spec=base))
    hdd = pd.concat(hdd_rows)

    # ---- tables
    def cell(v, spec, t):
        tab = res[v][spec].table.set_index("term")
        return ci(*tab.loc[t, ["pct", "pct_lo", "pct_hi"]]) if t in tab.index else "-"

    main_tab = []
    for t in TERMS:
        for spec in ("main", "moy FE", "month FE"):
            if t in ("trial", "post") and spec == "month FE":
                continue
            if t not in ("trial", "post") and spec != "main":
                continue
            main_tab.append({"term": t if spec == "main" else f"{t} ({spec})",
                             **{v: cell(v, spec, t) for v in res}})
    main_tab = pd.DataFrame(main_tab)
    ev_tab = pd.DataFrame([{"term": t, **{v: cell(v, "month FE", t) for v in res}}
                           for t in ("High", "Low", "Low_pre_2h", "Low_post_0_2h")])

    gaps = {}
    for v in ("all controls", "no E7-like (q99)", "synthetic control"):
        pt = res[v]["pre-trend"].table
        pm = pt[pt["term"].str.startswith("m_")].copy()
        pm.index = pm["term"].str[2:].to_numpy()
        pm.loc["2012-12", ["pct", "pct_lo", "pct_hi"]] = 0.0
        gaps[v] = pm.sort_index()
    gap_tab = pd.DataFrame({"month": gaps["all controls"].index})
    for v, g in gaps.items():
        gap_tab[v] = ["0 (ref.)" if m == "2012-12" else ci(*g.loc[m, ["pct", "pct_lo", "pct_hi"]]) for m in g.index]

    tm = temp.groupby(temp.index.to_period("M")).mean()
    yoy = []
    for m in ["07", "08", "09", "10", "11", "12", "01", "02"]:
        a = f"2012-{m}" if m >= "07" else f"2013-{m}"
        b = f"{int(a[:4]) + 1}-{m}"
        rec = {"month": m, "T earlier C": tm[pd.Period(a)], "T later C": tm[pd.Period(b)]}
        rec["dT"] = rec["T later C"] - rec["T earlier C"]
        for v, g in gaps.items():
            rec[f"gap change: {v}"] = g.loc[b, "pct"] - g.loc[a, "pct"]
        yoy.append(rec)
    yoy = pd.DataFrame(yoy)

    q_tab = pd.DataFrame({
        "statistic": ["households", "median", "p90", "p95", "p97.5", "p99", "max"],
        "dToU": [int(tou.sum()), *share[tou].quantile([.5, .9, .95, .975, .99, 1]).round(3)],
        "Std": [int(s_std.notna().sum()), *s_std.quantile([.5, .9, .95, .975, .99, 1]).round(3)]})
    above = pd.DataFrame([{"dToU quantile": f"p{100 * q:g}", "threshold": f"{thr[q]:.3f}", "Std above": int(flags[q].sum()),
                           "Std share above %": 100 * flags[q].mean(), "expected %": 100 * (1 - q),
                           "excess households": flags[q].sum() - (1 - q) * s_std.notna().sum()} for q in thr])

    days = ctl.day_counts(cal)
    wk = P.loc[:, (P.columns.get_level_values("month") >= 11) & ~P.columns.get_level_values("weekend")]
    prof = pd.DataFrame({"hh": np.arange(48)})
    groups = {"dToU": tou, "Std flagged (q99)": np.zeros(len(tou), bool), "Std other": np.zeros(len(tou), bool)}
    groups["Std flagged (q99)"][np.flatnonzero(~tou)[flags[0.99]]] = True
    groups["Std other"][np.flatnonzero(~tou)[~flags[0.99]]] = True
    for g, m in groups.items():
        prof[g] = wk[m].T.groupby(level="hh").mean().mean(axis=1).to_numpy()
    prof2 = prof.groupby(prof["hh"] // 4).mean().drop(columns="hh")
    prof2.insert(0, "local time", [f"{2 * k:02d}-{2 * k + 2:02d}" for k in range(12)])

    sc0 = sc_stats[0]
    sc_boot = pd.DataFrame(sc_stats[1:])
    gap_all = monthly_gap(sums, span["all controls"].start, span["all controls"].start, cal)
    gap_pl = monthly_gap(sums, span["all controls"].start, span["SC placebo (Jul-Oct fit)"].start, cal)
    placebo = pd.DataFrame({"month": [f"2012-{m:02d}" for m in range(7, 13)]})
    placebo["all controls, gap %"] = [gap_all[m] for m in placebo["month"]]
    base_all = gap_all[[f"2012-{m:02d}" for m in range(7, 11)]].mean()
    placebo["all controls, minus Jul-Oct mean"] = placebo["all controls, gap %"] - base_all
    placebo["SC fitted on Jul-Oct, gap %"] = [gap_pl[m] for m in placebo["month"]]

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.linspace(0, 0.9, 46)
    axes[0].hist(share[tou], bins=bins, density=True, color="#2a78d6", alpha=0.55, label="dToU")
    axes[0].hist(s_std.dropna(), bins=bins, density=True, histtype="step", color=INK, lw=1.2, label="Std")
    axes[0].axvline(thr[0.99], color="#eb6834", ls="--", lw=1, label="dToU 99th pct")
    axes[0].set_title("Night share 00-07 h, Nov-Dec 2012", fontsize=9, loc="left", color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    for g, c in (("dToU", "#2a78d6"), ("Std flagged (q99)", "#eb6834"), ("Std other", INK)):
        axes[1].plot(prof["hh"] / 2, prof[g], color=c, lw=1.3, label=g)
    axes[1].set_title("Mean weekday profile, Nov-Dec 2012 (kWh/half-hour)", fontsize=9, loc="left", color=INK)
    axes[1].set_xlabel("local time, h", fontsize=8, color=INK2)
    axes[1].legend(fontsize=7, frameon=False)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "lcl_controls_night.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    for k, (v, g) in enumerate(gaps.items()):
        x = np.arange(len(g)) + (k - 1) * 0.22
        ax.errorbar(x, g["pct"], yerr=[g["pct"] - g["pct_lo"], g["pct_hi"] - g["pct"]], fmt="o", ms=3,
                    color=VAR_COLORS[v], elinewidth=0.9, label=v)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(list(gaps["all controls"].index).index("2012-12") + 0.5, color=INK2, ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(gaps["all controls"])), gaps["all controls"].index, rotation=60, fontsize=7)
    ax.set_title("dToU minus control on Normal half-hours by month, % (Dec 2012 = 0)", fontsize=9, loc="left",
                 color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "lcl_controls_gap.png", dpi=150)
    plt.close(fig)

    # ---- report
    def lv(v, spec="moy FE", t="trial"):
        return cell(v, spec, t)

    hd = hdd.set_index(["variant", "spec", "term"])
    hdd_tab = []
    for v in ("all controls", "no E7-like (q99)"):
        for spec in ("main", "moy FE"):
            r = {"control group": v, "FE": spec, "trial without HDD": lv(v, spec),
                 "trial with HDD x block": ci(*hd.loc[(v, spec, "trial"), ["pct", "pct_lo", "pct_hi"]])}
            for b in range(6):
                c, s = hd.loc[(v, spec, f"hdd_b{b}"), ["coef_kwh", "se"]]
                r[f"HDD {4 * b:02d}-{4 * b + 4:02d} h, Wh"] = f"{1000 * c:+.2f} ({1000 * s:.2f})"
            hdd_tab.append(r)
    hdd_tab = pd.DataFrame(hdd_tab)

    report = "\n".join([
        "# LCL S1: control-group checks (Economy 7, temperature, synthetic control)",
        "",
        "Generated by `python scripts/check_lcl_controls.py`. S0 training households of the main DiD sample "
        f"({n_std} Std, {n_tou} dToU); held-out households are not used. Methods: "
        "`src/ttrl/analysis/lcl_controls.py`; DiD as in `src/ttrl/analysis/lcl_did.py`. Intervals are 95% "
        f"from the combined (date-clustered and household-bootstrap, {args.boot} draws) standard error.",
        "",
        "## Key results",
        "",
        f"* Normal-price level term (moy FE): all controls {lv('all controls')}%, without E7-like controls (q99) "
        f"{lv('no E7-like (q99)')}%, (q97.5) {lv('no E7-like (q97.5)')}%, synthetic control "
        f"{lv('synthetic control')}%.",
        f"* High: all controls {cell('all controls', 'main', 'High')}%, synthetic control "
        f"{cell('synthetic control', 'main', 'High')}%; Low: {cell('all controls', 'main', 'Low')}% and "
        f"{cell('synthetic control', 'main', 'Low')}%.",
        f"* {int(flags[0.99].sum())} Std households ({100 * flags[0.99].mean():.1f}%) exceed the dToU 99th "
        f"percentile of the Nov-Dec 2012 night share ({thr[0.99]:.3f}); 1% would be expected without "
        "E7-like households.",
        "",
        "## 1. Night profile (Economy 7 signature, pre-trial data only)",
        "",
        "Share of Nov-Dec 2012 consumption between 00:00 and 07:00 local time (= GMT), from household x month x "
        "half-hour x weekend mean profiles weighted by day counts. The dToU offer excluded Economy 7 customers, so "
        "the dToU distribution is the reference. See `lcl_controls_night.png`.",
        "",
        md_table(q_tab, floatfmt="{:.3f}"),
        "",
        md_table(above, floatfmt="{:.1f}"),
        "",
        "Mean weekday profile, Nov-Dec 2012, kWh per half-hour averaged over 2-hour blocks:",
        "",
        md_table(prof2, floatfmt="{:.3f}"),
        "",
        "## 2. DiD with each control group (% [95% CI])",
        "",
        md_table(main_tab),
        "",
        "Event terms with calendar month x 4-hour block FE (event effects within the month):",
        "",
        md_table(ev_tab),
        "",
        "## 3. Temperature sensitivity",
        "",
        f"Heating degree-days (base {HDD_BASE} C, Heathrow daily mean) times 4-hour block, added to the DiD. "
        "Coefficients are the dToU-minus-Std load difference per degree-day, Wh per household-half-hour (SE). "
        "A negative value means Std load rises more in the cold than dToU load.",
        "",
        md_table(hdd_tab),
        "",
        "Monthly mean temperature and the change in the dToU-control gap one year later (points; Normal "
        "half-hours, month-of-sample dummies):",
        "",
        md_table(yoy, floatfmt="{:+.1f}"),
        "",
        "## 4. Monthly gap by control group (% [95% CI], Dec 2012 = 0)",
        "",
        md_table(gap_tab),
        "",
        "See `lcl_controls_gap.png`.",
        "",
        "## 5. Synthetic control",
        "",
        f"Full sample: {sc0['n_donors']} of {n_std} Std donors carry weight (effective number "
        f"{sc0['eff_donors']:.0f}, largest weight {sc0['max_weight']:.4f}); pre-trial RMSE {sc0['rmse_pct']:.3f}% "
        f"of the mean over the 576 moments. Bootstrap draws: median RMSE {sc_boot['rmse_pct'].median():.3f}%, "
        f"median effective donors {sc_boot['eff_donors'].median():.0f}.",
        "",
        f"In-time placebo: weights fitted on Jul-Oct 2012 moments only (RMSE {pl_stats[0]['rmse_pct']:.3f}%), "
        "then the monthly level gap (dToU over the weighted controls, % of the controls). The all-controls column "
        "minus its Jul-Oct mean is the corresponding DiD-style placebo.",
        "",
        md_table(placebo, floatfmt="{:+.2f}"),
    ])
    (OUTPUTS / "lcl_controls_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
