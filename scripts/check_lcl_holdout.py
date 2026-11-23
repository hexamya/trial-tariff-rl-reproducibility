"""Held-out household check of the frozen S1 response kernel (Fig. 4). Run once.

Run:  python scripts/check_lcl_holdout.py [--boot 100]
      (needs scripts/build_lcl_panel.py and scripts/estimate_lcl_kernel.py, both control groups)

The protocol was fixed before the household check. The kernels estimated on the training households are frozen (outputs/
lcl_kernel_{spec.json,terms.csv,ensemble.parquet} and the lcl_kernel_allctl_ ones) and applied to
the held-out households. The preliminary aggregate analysis used all households before the split.
Held-out Std households are screened with the Economy 7 threshold fixed on the
training dToU households (q99 of the night share, 0.354); nothing is re-estimated. Observed effect of
an event: mean of the percent-scale outcome minus the fixed-effect baseline (fitted on half-hours no
event touches) over the event's half-hours. Predicted: the frozen kernel on the same half-hours.
The same kernel specification is also fitted on the held-out households, as their measurement of
each cell. Everything is reported; nothing is refitted afterwards.

Writes to outputs/: lcl_holdout_summary.md, lcl_holdout_events.csv, lcl_holdout_cells.csv,
lcl_holdout_calibration.png
"""
from __future__ import annotations

import argparse
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttrl.analysis import lcl_controls as ctl  # noqa: E402
from ttrl.analysis import lcl_did as did  # noqa: E402
from ttrl.analysis import lcl_kernel as kn  # noqa: E402
from ttrl.data import lcl_features as lf  # noqa: E402
from ttrl.data import lcl_meters as lm  # noqa: E402
from ttrl.data import lcl_weather as lw  # noqa: E402
from ttrl.data import splits as sp  # noqa: E402
from ttrl.paths import OUTPUTS  # noqa: E402
from ttrl.reporting import ci, md_table  # noqa: E402

FIRST, LAST = "2012-07-01 00:30", "2014-02-28 00:00"
SEED = 20261009
PROTOCOL_THRESHOLD = 0.354
KERNELS = {"screened controls (main)": "lcl_kernel", "all controls (sensitivity)": "lcl_kernel_allctl"}
COLORS = {"High": "#eb6834", "Low": "#2a78d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_kernel(prefix: str):
    files = [OUTPUTS / f"{prefix}_{x}" for x in ("spec.json", "terms.csv", "ensemble.parquet")]
    spec = kn.KernelSpec.from_json(files[0].read_text(encoding="utf-8"))
    terms = pd.read_csv(files[1]).set_index("term")
    members = pd.read_parquet(files[2])
    return spec, terms, members, {f.name: sha(f) for f in files}


def event_table(cal, y, base, eff, eff_members) -> pd.DataFrame:
    rows = []
    for e in lf.find_events(cal).itertuples():
        pos = np.arange(e.i0, e.i1 + 1)
        pos = pos[np.isfinite(y[pos])]
        pm = eff_members[:, pos].mean(1)
        rows.append({"event_id": e.event_id, "band": e.band, "start": e.start, "duration_h": e.duration_h,
                     "observed": float(np.mean(y[pos] - base[pos])), "predicted": float(np.mean(eff[pos])),
                     "pred_lo": float(np.percentile(pm, 2.5)), "pred_hi": float(np.percentile(pm, 97.5))})
    return pd.DataFrame(rows)


def calib(ev: pd.DataFrame) -> pd.DataFrame:
    out = []
    for b in ("High", "Low", "all"):
        e = ev if b == "all" else ev[ev["band"] == b]
        slope = np.polyfit(e["predicted"], e["observed"], 1)[0]
        out.append({"band": b, "events": len(e), "mean observed %": e["observed"].mean(),
                    "mean predicted %": e["predicted"].mean(), "slope": slope,
                    "corr": np.corrcoef(e["predicted"], e["observed"])[0, 1],
                    "RMSE": float(np.sqrt(np.mean((e["observed"] - e["predicted"]) ** 2)))})
    return pd.DataFrame(out)


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()

    cal = did.calendar(FIRST, LAST)
    split = sp.household_split(lm.households())
    base_sample = did.select_households(did.coverage(cal))
    train = sp.restrict(base_sample, split, "train")
    hold = sp.restrict(base_sample, split, "holdout")
    tou_h = hold["tou"].to_numpy()
    n_grp = {"Std": int((~tou_h).sum()), "dToU": int(tou_h.sum())}
    temp = lw.daily_mean()

    share_tr = ctl.night_share(ctl.pretrial_profiles(cal, train), cal)
    thr = ctl.night_threshold(share_tr, train["tou"], 0.99)
    if round(thr, 3) != PROTOCOL_THRESHOLD:
        raise RuntimeError(f"training threshold {thr:.4f} differs from the protocol value {PROTOCOL_THRESHOLD}")
    share_h = ctl.night_share(ctl.pretrial_profiles(cal, hold), cal)
    keep = ~(share_h[~tou_h] > thr).to_numpy()
    print(f"held-out sample {n_grp}; Std above threshold: {int((~keep).sum())}", flush=True)

    weights = {"screened controls (main)": ctl.subset_weights(n_grp["dToU"], keep, args.boot, SEED),
               "all controls (sensitivity)": ctl.subset_weights(n_grp["dToU"], np.ones(n_grp["Std"], bool),
                                                                args.boot, SEED + 1)}
    W, spans = ctl.stack_weights(list(weights.values()))
    span = dict(zip(weights, spans))
    sums = did.group_sums(cal, hold, W, dtype=np.float32)
    print(f"{sums.n_readings:,} readings", flush=True)
    cid, _ = kn.affected_clusters(cal)

    results = {}
    for name, prefix in KERNELS.items():
        spec, terms, members, digests = load_kernel(prefix)
        y = kn.pct_outcome(sums.select(span[name]))
        base = kn.baseline(cal, y[0], spec.fe, cid)
        eff = kn.kernel_effect(spec, terms["coef"], cal, temp)
        effm = kn.kernel_effect(spec, members, cal, temp)
        ev = event_table(cal, y[0], base, eff, effm)
        X = kn.kernel_design(cal, temp, spec.factors)
        F = kn.fe_matrix(cal, spec.fe)
        fit_h = kn.fit(cal, y, X, F, spec.factors, spec.fe).table.set_index("term")
        cells = pd.DataFrame({"term": spec.terms})
        cells["training"] = terms.loc[spec.terms, "coef"].to_numpy()
        cells["held-out"] = fit_h.loc[spec.terms, "coef"].to_numpy()
        cells["difference"] = cells["held-out"] - cells["training"]
        cells["se"] = np.hypot(fit_h.loc[spec.terms, "se_combined"].to_numpy(),
                               terms.loc[spec.terms, "se_combined"].to_numpy())
        cells["held-out lo"] = fit_h.loc[spec.terms, "lo"].to_numpy()
        cells["held-out hi"] = fit_h.loc[spec.terms, "hi"].to_numpy()
        ref = pd.read_csv(OUTPUTS / f"{prefix}_events.csv")
        ref = ref[ref["model"] == "+".join(spec.factors)]
        results[name] = {"spec": spec, "digests": digests, "events": ev, "calib": calib(ev), "cells": cells,
                         "train_cv": calib(ref)}
        print(f"[{name}] slope {results[name]['calib'].iloc[-1]['slope']:.2f}", flush=True)

    pd.concat([r["events"].assign(kernel=k) for k, r in results.items()]).to_csv(
        OUTPUTS / "lcl_holdout_events.csv", index=False)
    pd.concat([r["cells"].assign(kernel=k) for k, r in results.items()]).to_csv(
        OUTPUTS / "lcl_holdout_cells.csv", index=False)

    # ---- Fig. 4 draft
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)
    allv = np.concatenate([np.r_[r["events"]["observed"], r["events"]["predicted"]] for r in results.values()])
    lim = np.nanpercentile(np.abs(allv), 99) * 1.1
    for ax, (name, r) in zip(axes, results.items()):
        for b in ("High", "Low"):
            e = r["events"][r["events"]["band"] == b]
            ax.scatter(e["predicted"], e["observed"], s=12, color=COLORS[b], alpha=0.75, lw=0, label=b)
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        c = r["calib"].set_index("band").loc["all"]
        ax.set_title(f"{name}\nslope {c['slope']:.2f}, corr {c['corr']:.2f}", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("predicted by the frozen kernel, %", fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel("observed in held-out households, %", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "lcl_holdout_calibration.png", dpi=150)
    plt.close(fig)

    # ---- report
    parts = [
        "# LCL S1: held-out household check of the frozen kernel (Fig. 4)",
        "",
        "Generated by `python scripts/check_lcl_holdout.py` under the prespecified protocol. "
        "Held-out households meeting the main coverage rule: "
        f"{n_grp}. Economy 7 screening of held-out Std households with the threshold fixed on the training "
        f"dToU households ({thr:.4f}): {int((~keep).sum())} dropped for the main kernel. Household bootstrap: "
        f"{args.boot} draws. The kernels are the frozen training estimates; nothing was refitted after this run. "
        "The exploratory DiD used all households before the split, including the households reserved for this check.",
        "",
        "Observed effect of an event: mean of the percent-scale outcome minus the fixed-effect baseline (fitted on "
        "held-out half-hours that no event touches) over the event's half-hours. Predicted: the frozen kernel on the "
        "same half-hours. The slope is from regressing observed on predicted across events (1 = calibrated). "
        "The training-CV rows repeat the month-fold cross-validation on the training households for comparison.",
    ]
    for name, r in results.items():
        cal_tab = pd.concat([r["calib"].assign(sample="held-out households"),
                             r["train_cv"].assign(sample="training, month-fold CV")])
        cal_tab = cal_tab[["sample", "band", "events", "mean observed %", "mean predicted %", "slope", "corr", "RMSE"]]
        cells = r["cells"]
        cell_tab = pd.DataFrame({
            "term": cells["term"].str.replace("|", " | ", regex=False),
            "training %": cells["training"].map("{:+.1f}".format),
            "held-out % [95% CI]": [ci(h, lo, hi) for h, lo, hi in zip(cells["held-out"], cells["held-out lo"],
                                                                      cells["held-out hi"])],
            "difference": [f"{d:+.1f} ({s:.1f})" for d, s in zip(cells["difference"], cells["se"])],
        })
        within = (np.abs(cells["difference"]) <= 1.96 * cells["se"]).mean()
        parts += [
            "",
            f"## {name}",
            "",
            f"Frozen files (sha256 prefix): " + ", ".join(f"`{k}` {v}" for k, v in r["digests"].items()) + ".",
            f"Fixed effects `{r['spec'].fe}`, factors {list(r['spec'].factors)}.",
            "",
            "Event-level calibration:",
            "",
            md_table(cal_tab, floatfmt="{:.2f}"),
            "",
            "Held-out measurement of each kernel cell (same specification fitted on the held-out households) against "
            "the training estimate. Difference: held-out minus training (SE combining both; conservative, because "
            f"the two samples share the same dates). {100 * within:.0f}% of cells differ by less than 1.96 SE.",
            "",
            md_table(cell_tab),
        ]
    parts += ["", "See `lcl_holdout_calibration.png` (draft of Fig. 4)."]
    report = "\n".join(parts)
    (OUTPUTS / "lcl_holdout_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
