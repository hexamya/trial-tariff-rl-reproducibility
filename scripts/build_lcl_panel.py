"""Build the LCL half-hourly household panel and check its time labels.

Run:  python scripts/build_lcl_panel.py      (needs scripts/fetch_lcl_meters.py first)

Steps:
  1. CSV pieces in the zip -> data/interim/lcl_meters/piece_<k>.parquet (resumable)
  2. one panel data/processed/lcl_meters.parquet (+ lcl_households.parquet)
  3. time-label checks:
     - clock: UTC or local? (households reporting at each label on the DST change days)
     - same labels as Data_features.xlsx? (dToU group mean vs the workbook's mean_all at
       the alignment inferred in ttrl.data.lcl_features)
     - where the price response steps at event edges, now against the Std control group

Writes outputs/lcl_meters_summary.md (also printed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ttrl.analysis import lcl_did as did
from ttrl.data import lcl_features as lf
from ttrl.data import lcl_meters as lm
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table

DST_DAYS = ["2012-03-25", "2012-10-28", "2013-03-31", "2013-10-27"]
FEATURES_LABEL0 = pd.Timestamp("2012-07-01 00:00")  # original label of workbook row 0 (utc_end_label)


def section_build(stats: pd.DataFrame, counts: dict, hh: pd.DataFrame) -> str:
    tot = stats[["n_rows", "n_null", "n_bad_kwh", "n_bad_tariff", "n_bad_id", "n_bad_ts"]].sum()
    by = hh.groupby("tou").agg(households=("hh_id", "size"), rows=("n_rows", "sum"), null=("n_null", "sum"),
                               first=("first_label", "min"), last=("last_label", "max"),
                               median_first=("first_label", "median"), median_last=("last_label", "median"))
    by.index = by.index.map({False: "Std", True: "ToU"})
    tod_first = hh["first_label"].dt.strftime("%H:%M").value_counts().head(5)
    tod_last = hh["last_label"].dt.strftime("%H:%M").value_counts().head(5)
    return "\n".join([
        "## 1. Build",
        "",
        f"Source: `{lm.ZIP_PATH.relative_to(ROOT).as_posix()}`, {len(stats)} CSV pieces, "
        f"{tot.n_rows:,} rows. `Null` kWh: {tot.n_null:,}. Unparseable kWh: {tot.n_bad_kwh}, "
        f"tariff codes other than Std/ToU: {tot.n_bad_tariff}, malformed ids: {tot.n_bad_id}, labels off the "
        f":00/:30 grid: {tot.n_bad_ts}.",
        "",
        f"Dropped: {counts['n_off_grid']:,} rows with a label off the half-hour grid ({counts['n_off_grid_null']:,} "
        "of them `Null`; one per household, mostly stamped 2012-12-18 15:1x-15:2x, so not readings) and "
        f"{counts['n_exact_dup']:,} exact duplicate rows. Same household and label with different kWh: "
        f"{counts['n_conflicting_dup']:,} (averaged in the analysis). Panel rows: {counts['n_panel']:,}. "
        "No household switches between Std and ToU.",
        "",
        md_table(by.reset_index(names="group")),
        "",
        "Time of day of each household's first and last label (top 5):",
        "",
        md_table(pd.DataFrame({"first label": tod_first.index, "households": tod_first.to_numpy()})),
        "",
        md_table(pd.DataFrame({"last label": tod_last.index, "households": tod_last.to_numpy()})),
    ])


def dst_check(con) -> tuple[str, pd.DataFrame]:
    """Households reporting at each label on DST change days, relative to the same label a week earlier."""
    days = ", ".join(f"DATE '{d}'" for d in DST_DAYS)
    weeks = ", ".join(f"DATE '{(pd.Timestamp(d) - pd.Timedelta(days=7)).date()}'" for d in DST_DAYS)
    df = con.execute(f"""
        SELECT CAST(ts_label AS DATE) AS day, strftime(ts_label, '%H:%M') AS tod,
               count(*) AS n_rows, count(DISTINCT hh_id) AS n_hh
        FROM read_parquet('{lm.PANEL_PATH.as_posix()}')
        WHERE CAST(ts_label AS DATE) IN ({days}, {weeks})
        GROUP BY ALL""").df()
    df["day"] = pd.to_datetime(df["day"])
    rows = []
    for d in DST_DAYS:
        d0, d1 = pd.Timestamp(d), pd.Timestamp(d) - pd.Timedelta(days=7)
        a = df[df["day"] == d0].set_index("tod")
        b = df[df["day"] == d1].set_index("tod")
        ratio = (a["n_hh"] / b["n_hh"]).reindex(b.index)
        dup = (a["n_rows"] - a["n_hh"]).reindex(b.index).fillna(0)
        rows.append({"day": d, "labels": len(a), "labels week before": len(b),
                     "min ratio": ratio.min(), "label at min": ratio.idxmin(),
                     "max ratio": ratio.max(), "repeated readings": int(dup.sum())})
    tab = pd.DataFrame(rows)
    utc = (tab["labels"] == 48).all() and (tab["min ratio"] > 0.9).all() and (tab["repeated readings"] == 0).all()
    text = ("All four DST change days have 48 labels, no label loses households relative to the same label a week "
            "earlier, and no household has a repeated label. **The meter labels are on a clock without DST (UTC/GMT).**"
            if utc else "The DST days do NOT look like a UTC series; see the table.")
    return text, tab


def tou_series(con) -> pd.DataFrame:
    """dToU and Std group mean kWh and household counts per label, Jul 2012 - Feb 2014."""
    df = con.execute(f"""
        SELECT ts_label, tou, avg(kwh) AS mean_kwh, count(*) AS n
        FROM read_parquet('{lm.PANEL_PATH.as_posix()}')
        WHERE kwh IS NOT NULL AND ts_label >= TIMESTAMP '2012-06-30' AND ts_label <= TIMESTAMP '2014-03-01'
        GROUP BY ALL""").df()
    out = df.pivot_table(index="ts_label", columns="tou", values=["mean_kwh", "n"])
    out.columns = [f"{a}_{'tou' if b else 'std'}" for a, b in out.columns]
    return out.sort_index()


def features_check(series: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Correlate the meter dToU mean with the workbook's mean_all at shifted label offsets."""
    raw = lf.read_raw()
    feat = pd.DataFrame({"mean_all": raw["mean_all"].astype(float).to_numpy(),
                         "n_all": (raw["sum_all"] / raw["mean_all"]).round().to_numpy()})
    feat.index = FEATURES_LABEL0 + pd.to_timedelta(np.arange(len(feat)) * 30, unit="min")
    rows = []
    for off in range(-4, 5):
        m = series["mean_kwh_tou"].copy()
        m.index = m.index + pd.Timedelta(minutes=30 * off)
        j = feat.join(m, how="inner").dropna()
        dlog_f, dlog_m = np.diff(np.log(j["mean_all"])), np.diff(np.log(j["mean_kwh_tou"]))
        rows.append({"offset (half-hours)": off, "rows": len(j),
                     "corr level": np.corrcoef(j["mean_all"], j["mean_kwh_tou"])[0, 1],
                     "corr diff log": np.corrcoef(dlog_f, dlog_m)[0, 1],
                     "mean ratio": (j["mean_all"] / j["mean_kwh_tou"]).mean()})
    tab = pd.DataFrame(rows)
    best = int(tab.loc[tab["corr diff log"].idxmax(), "offset (half-hours)"])
    j = feat.join(series, how="inner")
    n_ratio = (j["n_all"] / j["n_tou"]).median()
    text = (f"Best match at offset {best:+d} half-hours. "
            + ("**The meter labels are the same labels the workbook rows had before they lost their dates**, so "
               "the alignment inferred in `lcl_features.py` (row t = UTC half-hour ending 2012-07-01 00:00 + 30 min x t) "
               "carries over: meter label = UTC end of the half-hour, under that inference."
               if best == 0 else "**The meter labels do NOT match the inferred workbook alignment.**")
            + f" The workbook's households are a subset: its n_all is a median {n_ratio:.2f} of the meter dToU count "
            "at the same label.")
    return text, tab


def edge_check(series: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """dToU minus Std (% of Std) at half-hours around event edges, under the end-label convention."""
    out = []
    for label in ("end", "start"):
        cal = did.calendar("2012-07-01 00:30", "2014-02-28 00:00", label=label).set_index("ts_label")
        s = series.reindex(cal.index)
        gap = 100 * (s["mean_kwh_tou"] / s["mean_kwh_std"] - 1)
        # residual gap: minus the same half-hour-of-day x weekend mean over Normal 2013 half-hours
        base = gap[cal["band"] == "Normal"].groupby([cal["hh"], cal["weekend"]]).mean()
        res = gap - base.reindex(pd.MultiIndex.from_arrays([cal["hh"], cal["weekend"]])).to_numpy()
        ev = lf.find_events(cal.reset_index())
        r = res.to_numpy()
        for b in ("High", "Low"):
            e = ev[ev["band"] == b]
            for edge, pos in (("start", e["i0"].to_numpy()), ("end", e["i1"].to_numpy() + 1)):
                rec = {"labels mark": label, "band": b, "edge": edge}
                for k in range(-2, 3):
                    idx = pos + k
                    rec[f"k={k:+d}"] = np.nanmean(r[idx[(idx >= 0) & (idx < len(r))]])
                out.append(rec)
    tab = pd.DataFrame(out)
    text = ("Residual dToU-minus-Std gap (% of Std, minus its Normal-2013 mean at the same half-hour of day and "
            "weekend flag), averaged over events. k = 0 is the first event half-hour (start) or the first "
            "half-hour after the event (end).")
    return text, tab


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stats = lm.convert_zip()
    counts = lm.build_panel()
    hh = lm.households()
    con = lm.connect()
    dst_text, dst_tab = dst_check(con)
    series = tou_series(con)
    feat_text, feat_tab = features_check(series)
    edge_text, edge_tab = edge_check(series)

    report = "\n".join([
        "# LCL smart-meter panel: build and time-label checks",
        "",
        "Generated by `python scripts/build_lcl_panel.py`. Descriptive data checks; no estimates.",
        "",
        section_build(stats, counts, hh),
        "",
        "## 2. Clock: UTC or local?",
        "",
        "Households reporting at each label on the four DST change days, divided by the count at the same "
        "label a week earlier. A local-clock series would lose two labels in spring and repeat two in autumn.",
        "",
        md_table(dst_tab),
        "",
        dst_text,
        "",
        "## 3. Same labels as `Data_features.xlsx`?",
        "",
        "Workbook `mean_all` (row t given the label 2012-07-01 00:00 + 30 min x t) against the meter dToU group "
        "mean with its labels shifted by the offset:",
        "",
        md_table(feat_tab, floatfmt="{:.4f}"),
        "",
        feat_text,
        "",
        "## 4. Response step at event edges, against the control group",
        "",
        edge_text,
        "",
        md_table(edge_tab),
        "",
        "Under the end-label reading the step should fall between k = -1 and k = 0 at both edges. The same data "
        "fit start labels with a 30-minute reaction lag, so this cannot separate the two; the DiD reports both.",
    ])
    (OUTPUTS / "lcl_meters_summary.md").write_text(report, encoding="utf-8")
    series.to_parquet(OUTPUTS / "lcl_group_means.parquet")
    print(report)


if __name__ == "__main__":
    main()
