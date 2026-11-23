"""S1 response kernel on the S0 training households: heterogeneity chosen by month-fold
cross-validation, and the household x date bootstrap ensemble that S2 samples from.

Run:  python scripts/estimate_lcl_kernel.py [--boot 200] [--controls no_e7|all]
      (needs scripts/build_lcl_panel.py first)

Sample as in scripts/estimate_lcl_did.py (main coverage rule, training split; the holdout is
reserved for the separate household check). By default Std households with an Economy 7-like night
profile are dropped (q99 of the dToU night share, scripts/check_lcl_controls.py), since the dToU
offer excluded Economy 7; ``--controls all`` keeps every Std household and writes its outputs
with the prefix ``lcl_kernel_allctl_``. Method: ttrl.analysis.lcl_kernel.

Steps:
  1. percent-scale outcome y = 100 * Delta / P for the estimate and every household draw
  2. fixed effects for the baseline, chosen by CV of the constant kernel (one-SE rule, simplest first)
  3. forward selection of heterogeneity factors by CV (one-SE rule)
  4. final fit with combined intervals; event-level calibration of the CV predictions
  5. ensemble: estimate + one household-bootstrap deviation + one date-bootstrap deviation per member

Writes to outputs/: lcl_kernel_summary.md, lcl_kernel_cv.csv, lcl_kernel_events.csv,
lcl_kernel_terms.csv, lcl_kernel_ensemble.parquet, lcl_kernel_spec.json, lcl_kernel_effects.png,
lcl_kernel_calibration.png (prefix lcl_kernel_allctl_ with --controls all)
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
from ttrl.analysis import lcl_kernel as kn  # noqa: E402
from ttrl.data import lcl_meters as lm  # noqa: E402
from ttrl.data import lcl_weather as lw  # noqa: E402
from ttrl.data import splits as sp  # noqa: E402
from ttrl.paths import OUTPUTS  # noqa: E402
from ttrl.reporting import ci, md_table  # noqa: E402

FIRST, LAST = "2012-07-01 00:30", "2014-02-28 00:00"
SEED = 20130101          # household draws, as in estimate_lcl_did.py
ENSEMBLE_SEED = 20261007  # date resampling of the members
COLORS = {"High": "#eb6834", "Low": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FE_ORDER = ("hhw", "moy", "month")  # simplest first


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.axhline(0, color=INK2, lw=0.8)


def one_se_choice(results: dict, order, cid) -> tuple[str, pd.DataFrame]:
    """Simplest entry of ``order`` whose loss is within one SE of the best."""
    best = min(results, key=lambda k: results[k].loss())
    rows = []
    for k in order:
        d, se = (0.0, 0.0) if k == best else kn.paired_diff(results[k], results[best], cid)
        rows.append({"choice": k, "loss": results[k].loss(), "minus best": d, "se": se,
                     "within 1 SE": d <= se})
    tab = pd.DataFrame(rows)
    return tab.loc[tab["within 1 SE"], "choice"].iloc[0], tab


def calib_stats(ev: pd.DataFrame) -> dict:
    out = {}
    for b in ("High", "Low", "all"):
        e = ev if b == "all" else ev[ev["band"] == b]
        slope, icpt = np.polyfit(e["predicted"], e["observed"], 1)
        r = np.corrcoef(e["predicted"], e["observed"])[0, 1]
        out[b] = {"events": len(e), "mean observed": e["observed"].mean(), "mean predicted": e["predicted"].mean(),
                  "slope": slope, "corr": r, "rmse": float(np.sqrt(np.mean((e["observed"] - e["predicted"]) ** 2)))}
    return out


def term_label(t: str) -> str:
    return t.replace("|", " | ")


def fig_effects(tab: pd.DataFrame, factors, path) -> None:
    kern = tab[~tab["term"].isin(kn.LEVEL_TERMS)].copy()
    kern["band"] = kern["term"].str.extract(r"^(High|Low)")[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, b in zip(axes, ("High", "Low")):
        d = kern[kern["band"] == b].reset_index(drop=True)
        x = np.arange(len(d))
        ax.errorbar(x, d["coef"], yerr=1.96 * d["se_combined"], fmt="o", ms=4, color=COLORS[b],
                    elinewidth=1, capsize=0)
        ax.set_xticks(x, [t.replace(f"{b}|", "").replace(f"{b}_", "") or b for t in d["term"]],
                      rotation=55, ha="right", fontsize=7)
        ax.set_title(f"{b}: kernel terms, % of counterfactual (95% CI)", fontsize=9, loc="left", color=INK)
        style(ax)
    fig.suptitle(f"Selected kernel: factors {', '.join(factors) or 'none (constant)'}", fontsize=9, color=INK2,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(ev_sel: pd.DataFrame, ev_const: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    lim = np.nanpercentile(np.abs(np.r_[ev_sel["observed"], ev_sel["predicted"]]), 99) * 1.1
    for ax, (name, ev) in zip(axes, (("constant kernel", ev_const), ("selected kernel", ev_sel))):
        for b in ("High", "Low"):
            e = ev[ev["band"] == b]
            ax.scatter(e["predicted"], e["observed"], s=10, color=COLORS[b], alpha=0.7, label=b, lw=0)
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"Held-out events, {name}", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("predicted effect, %", fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel("observed effect, %", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--controls", choices=["no_e7", "all"], default="no_e7")
    args = ap.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    pre = "lcl_kernel" if args.controls == "no_e7" else "lcl_kernel_allctl"

    cal = did.calendar(FIRST, LAST)
    split = sp.household_split(lm.households())
    sample = sp.restrict(did.select_households(did.coverage(cal)), split, "train")
    tou = sample["tou"].to_numpy()
    temp = lw.daily_mean()
    if args.controls == "no_e7":
        P = ctl.pretrial_profiles(cal, sample)
        share = ctl.night_share(P, cal)
        thr = ctl.night_threshold(share, sample["tou"], 0.99)
        keep = ~(share[~tou] > thr).to_numpy()
        W = ctl.subset_weights(int(tou.sum()), keep, args.boot, SEED)
        ctrl_txt = f"Std households with an Economy 7-like night profile dropped ({int((~keep).sum())}, q99)."
    else:
        W = did.bootstrap_weights(sample, args.boot, SEED)
        ctrl_txt = "All Std households as controls."
    n_grp = {"Std": int((~tou).sum()), "dToU": int(tou.sum())}

    sums = did.group_sums(cal, sample, W, dtype=np.float32)
    print(f"{sums.n_readings:,} readings", flush=True)
    y = kn.pct_outcome(sums)

    # constant kernel on both scales (consistency of the percent scale)
    X0 = kn.kernel_design(cal, temp, ())
    kwh = did.estimate(cal, sums, did.event_design(cal)).table.set_index("term")
    pct0 = kn.fit(cal, y, X0, kn.fe_matrix(cal, "hhw")).table.set_index("term")
    scale_tab = pd.DataFrame([{"term": t, "kWh-scale DiD, %": ci(*kwh.loc[t, ["pct", "pct_lo", "pct_hi"]]),
                               "percent-scale fit, %": ci(*pct0.loc[t, ["coef", "lo", "hi"]])}
                              for t in ("High", "Low", "High_pre_2h", "High_post_0_2h", "Low_pre_2h",
                                        "Low_post_0_2h", "trial")])

    # CV: fixed effects, then factors
    cid, clusters = kn.affected_clusters(cal)
    null = {fe: kn.cross_validate(cal, y[0], temp, None, fe, cid, clusters) for fe in FE_ORDER}
    const = {fe: kn.cross_validate(cal, y[0], temp, (), fe, cid, clusters) for fe in FE_ORDER}
    fe, fe_tab = one_se_choice(const, FE_ORDER, cid)
    fe_tab["null loss"] = [null[k].loss() for k in FE_ORDER]
    print(f"fixed effects: {fe}", flush=True)
    chosen, path, cvres = kn.forward_select(cal, y[0], temp, fe, cid, clusters)
    print(f"selected factors: {chosen}", flush=True)
    sel_name = "+".join(chosen) or "const"
    L_null = null[fe].loss()
    path["explained %"] = 100 * (1 - path["loss"] / L_null)
    path.to_csv(OUTPUTS / f"{pre}_cv.csv", index=False)

    # final fit and calibration
    X = kn.kernel_design(cal, temp, chosen)
    F = kn.fe_matrix(cal, fe)
    final = kn.fit(cal, y, X, F, chosen, fe)
    final.table.to_csv(OUTPUTS / f"{pre}_terms.csv", index=False)
    ev_sel = kn.event_calibration(cvres[sel_name], cal, y[0], clusters, cid)
    ev_const = kn.event_calibration(cvres["const"], cal, y[0], clusters, cid)
    pd.concat([ev_sel.assign(model=sel_name), ev_const.assign(model="const")]).to_csv(
        OUTPUTS / f"{pre}_events.csv", index=False)
    cs_sel, cs_const = calib_stats(ev_sel), calib_stats(ev_const)

    # ensemble
    members = kn.ensemble(cal, y, X, F, ENSEMBLE_SEED)
    terms = [t for t in final.table["term"] if t not in kn.LEVEL_TERMS]
    members = members[[c for c in final.table["term"]]]
    members.insert(0, "member", np.arange(len(members)))
    members.to_parquet(OUTPUTS / f"{pre}_ensemble.parquet", index=False)
    spec = kn.KernelSpec(chosen, fe, terms, support=kn.support(cal, temp),
                         notes=(f"S1 kernel on S0 training households ({n_grp}); {ctrl_txt} Percent of the "
                                "counterfactual load; level terms (trial, post) are not part of the price response. "
                                "Member = estimate + household-bootstrap deviation + local-date-bootstrap deviation."))
    (OUTPUTS / f"{pre}_spec.json").write_text(spec.to_json(), encoding="utf-8")
    spread = pd.DataFrame({"term": final.table["term"], "combined SE": final.table["se_combined"],
                           "member SD": members[final.table["term"]].std(ddof=1).to_numpy()})
    in_sample = kn.kernel_effect(spec, final.coef, cal, temp)
    X_fit = X[terms].to_numpy() @ final.coef[terms].to_numpy()
    assert np.allclose(in_sample, X_fit)

    fig_effects(final.table, chosen, OUTPUTS / f"{pre}_effects.png")
    fig_calibration(ev_sel, ev_const, OUTPUTS / f"{pre}_calibration.png")

    # ---- report
    def fmt_path(p):
        out = p[["step", "model", "loss", "change", "se", "explained %", "params", "identified", "accepted"]].copy()
        out["change / SE"] = out["change"] / out["se"]
        return out

    term_tab = pd.DataFrame({
        "term": final.table["term"].map(term_label),
        "% [95% CI combined]": [ci(r.coef, r.lo, r.hi) for r in final.table.itertuples()],
        "SE date-cluster": final.table["se_cluster_date"].map("{:.2f}".format),
        "SE household boot": final.table["se_boot_household"].map("{:.2f}".format)})
    calib = pd.DataFrame([{"kernel": name, "band": b, **v} for name, cs in (("const", cs_const), (sel_name, cs_sel))
                          for b, v in cs.items()])
    by_fold = (ev_sel.assign(err=ev_sel["observed"] - ev_sel["predicted"])
               .groupby(["fold", "band"]).agg(events=("err", "size"), observed=("observed", "mean"),
                                              predicted=("predicted", "mean"), rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))))
               .reset_index())
    sup_rows = [{"band": b, "temp C": f"{v['temp_c'][0]} to {v['temp_c'][1]}",
                 "durations h": ", ".join(f"{d:g}" for d in v["duration_h"]),
                 "fatigue (event days in prior 7)": f"{v['fatigue'][0]}-{v['fatigue'][1]}",
                 "half-hours": v["half_hours"]} for b, v in spec.support.items()]

    report = "\n".join([
        "# LCL S1: response kernel (heterogeneity by month-fold CV, bootstrap ensemble)",
        "",
        f"Generated by `python scripts/estimate_lcl_kernel.py --controls {args.controls}`. S0 training households "
        f"of the main DiD sample ({n_grp}); held-out households are not used. {ctrl_txt} Method: "
        "`src/ttrl/analysis/lcl_kernel.py`.",
        "",
        "## Key results",
        "",
        f"* Baseline fixed effects chosen by CV: **{fe}**. Selected heterogeneity: "
        f"**{', '.join(chosen) or 'none (constant kernel)'}**.",
        f"* On held-out events the selected kernel explains {100 * (1 - cvres[sel_name].loss() / L_null):.1f}% of "
        f"the squared deviation from the baseline on event-affected half-hours (constant kernel "
        f"{100 * (1 - cvres['const'].loss() / L_null):.1f}%).",
        f"* Event-level calibration of the CV predictions (observed on predicted): slope "
        f"{cs_sel['all']['slope']:.2f}, correlation {cs_sel['all']['corr']:.2f} over {cs_sel['all']['events']} events "
        f"(constant kernel: {cs_const['all']['slope']:.2f}, {cs_const['all']['corr']:.2f}).",
        f"* Ensemble: {len(members)} members (`{pre}_ensemble.parquet`), spec in `{pre}_spec.json`; "
        "`ttrl.analysis.lcl_kernel.kernel_effect` applies a member to any price schedule.",
        "",
        "## Percent scale",
        "",
        "Outcome y = 100 x Delta / P (module docstring). Constant kernel, half-hour-of-day x weekend FE, against the "
        "kWh-scale DiD of `lcl_did` on the same draws:",
        "",
        md_table(scale_tab),
        "",
        "## Cross-validation",
        "",
        f"Event-affected half-hours (event plus 2 h before and 4 h after, Normal only) form {len(clusters)} "
        "clusters; each is held out with its S0 month fold "
        f"({', '.join(f'{f}: {n}' for f, n in clusters['fold'].value_counts().sort_index().items())} clusters). "
        "Loss: mean squared error of y on held-out half-hours (percent squared). 'explained %' is relative to the "
        "fixed-effects baseline fitted on unaffected half-hours (no kernel). Standard errors of loss differences "
        "come from the clusters.",
        "",
        "Baseline fixed effects (constant kernel):",
        "",
        md_table(fe_tab, floatfmt="{:.3f}"),
        "",
        "Forward selection (each step adds the factor with the lowest loss; accepted only if it lowers the loss by "
        "more than one SE):",
        "",
        md_table(fmt_path(path), floatfmt="{:.3f}"),
        "",
        "## Selected kernel",
        "",
        f"Fixed effects `{fe}`, factors {list(chosen)}. Percent of the counterfactual load on the term's half-hours; "
        "categorical levels are cell effects (first factor) or deviations from the first level (further factors); "
        f"slopes are per degree C or per event day. See `{pre}_effects.png`.",
        "",
        md_table(term_tab),
        "",
        "Ensemble spread against the combined standard error:",
        "",
        md_table(spread, floatfmt="{:.2f}"),
        "",
        "## Event-level calibration (held-out folds)",
        "",
        "Observed: mean of y minus the CV baseline over the event's half-hours; predicted: the kernel's mean over the "
        f"same half-hours, from the fit without the event's fold. See `{pre}_calibration.png`.",
        "",
        md_table(calib, floatfmt="{:.2f}"),
        "",
        "Selected kernel by fold:",
        "",
        md_table(by_fold, floatfmt="{:.2f}"),
        "",
        "## Support (S2 must stay inside it)",
        "",
        md_table(pd.DataFrame(sup_rows)),
        "",
        "Hour blocks and durations outside the 2013 schedule have no estimate; `kernel_effect` refuses schedules "
        "that use a cell without one. Temperatures outside the range above are extrapolation (the ERCOT case needs "
        "a stated rule in S2).",
    ])
    (OUTPUTS / f"{pre}_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
