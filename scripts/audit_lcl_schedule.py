"""Audit the LCL dToU 2013 price schedule and take a first, descriptive look at response size.

Run:  python scripts/audit_lcl_schedule.py

Writes to outputs/:
  lcl_schedule_summary.md     the report (also printed)
  lcl_events.csv              one row per High/Low event
  lcl_matched_windows.csv     event-level matched-day comparisons (mean_all)
  lcl_alignment_lag.png       evidence for the consumption/price row alignment
  lcl_matched_windows.png     matched-day % differences by window
Nothing here is causal: the source file has no control group.
"""
from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from ttrl.analysis.lcl_matched import WINDOWS, bootstrap_summary, matched_window_table  # noqa: E402
from ttrl.data import lcl_features as lf  # noqa: E402
from ttrl.paths import OUTPUTS  # noqa: E402
from ttrl.reporting import ci, md_table  # noqa: E402

N_BOOT = 2000
SEED = 20130101
COLORS = {"High": "#eb6834", "Low": "#2a78d6"}  # reference palette slots 2 and 1
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLOCKS = {"night 23-05": (23, 5), "morning 05-11": (5, 11), "midday 11-17": (11, 17), "evening 17-23": (17, 23)}
WINDOW_LABELS = {"pre_2h": "2 h before", "event": "during event", "post_0_2h": "0-2 h after",
                 "post_2_4h": "2-4 h after"}


# ---------------------------------------------------------------- helpers
def start_block(h: float) -> str:
    """Name of the 6-hour block that contains local start hour ``h``."""
    for name, (a, b) in BLOCKS.items():
        if (a < b and a <= h < b) or (a > b and (h >= a or h < b)):
            return name
    raise ValueError(h)


def duration_bucket(hours: float) -> str:
    return "<=3 h" if hours <= 3 else ("4-6 h" if hours <= 6 else ">6 h")


def describe_day(bands: list[str]) -> str:
    """Compress a 48-slot day of bands into runs like '05:00-17:00 Low, 17:00-23:00 High'."""
    runs, start = [], 0
    for i in range(1, len(bands) + 1):
        if i == len(bands) or bands[i] != bands[start]:
            if bands[start] != "Normal":
                runs.append(f"{start // 2:02d}:{30 * (start % 2):02d}-{i // 2:02d}:{30 * (i % 2):02d} {bands[start]}")
            start = i
    return ", ".join(runs) or "all Normal"


def temp_phase(grid: pd.DataFrame) -> float:
    """Clock hour of the first-harmonic peak of the mean diurnal temperature profile."""
    prof = grid.groupby(grid["ts_local"].dt.hour * 2 + grid["ts_local"].dt.minute // 30)["temp_c"].mean()
    prof = prof - prof.mean()
    ang = 2 * np.pi * prof.index.to_numpy() / 48
    return float(np.arctan2((prof * np.sin(ang)).sum(), (prof * np.cos(ang)).sum()) / (2 * np.pi) * 24 % 24)


def boot_mean(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def hours_label(h: float) -> str:
    return f"{h:g}"


def edge_profile(df_all: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Mean load residual (%) around event starts and ends at the adopted alignment.

    k = 0 is the first event half-hour (start) or the first half-hour after the event (end).
    """
    res = 100 * lf.residual_log_load(df_all["mean_all"].to_numpy(float))
    off = int((df_all["period"] == "pre").sum())
    rows = []
    for b in ("High", "Low"):
        e = ev[ev["band"] == b]
        for edge, pos in (("start", e["i0"].to_numpy()), ("end", e["i1"].to_numpy() + 1)):
            rec = {"band": b, "edge": edge}
            for k in range(-2, 3):
                rec[f"k={k:+d}"] = np.nanmean(res[pos + off + k])
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sections
def section_data(raw, df, df_all, scores, ev, K) -> str:
    sha = hashlib.sha256(lf.XLSX_PATH.read_bytes()).hexdigest()[:16]
    lags = lf.best_lags(scores)
    K["lags"] = lags
    show = scores[(scores["lag"] == 0) | scores.apply(lambda r: abs(r["lag"] - lags[r["clock"]]) <= 1, axis=1)]
    show = show.sort_values(["clock", "lag"])[["clock", "lag", "resid_High_pct", "resid_Low_pct"]]
    show.columns = ["clock", "L", "High residual %", "Low residual %"]
    lag0 = scores[scores["lag"] == 0].set_index("clock")
    edges = edge_profile(df_all, ev)

    grid = raw.loc[raw["DateTime"].notna(), ["DateTime", "Temperature"]].set_axis(["ts_local", "temp_c"], axis=1)
    phases = []
    for day in ("2013-03-31", "2013-10-27"):
        t = pd.Timestamp(day)
        before = grid[(grid.ts_local < t) & (grid.ts_local >= t - pd.Timedelta(days=21))]
        after = grid[(grid.ts_local >= t + pd.Timedelta(days=1)) & (grid.ts_local < t + pd.Timedelta(days=22))]
        phases.append(temp_phase(after) - temp_phase(before))

    ok = df[~df["sum_dup_flag"]]
    n = ok[["n_all", "n_flex", "n_noflex"]].describe().T[["min", "50%", "max"]].rename(columns={"50%": "median"})
    K["n_all_median"] = float(ok["n_all"].median())
    near = (ok["n_all"] - ok["n_all"].median()).abs().le(2).mean() * 100
    monthly = (df_all[~df_all["sum_dup_flag"]].set_index("ts_local")["n_all"].resample("MS").median()
               .rename(lambda t: t.strftime("%Y-%m")))
    ev_hh = df["band"].isin(["High", "Low"])
    sum_gap = (df["sum_all"] - df["sum_flex"] - df["sum_noflex"]).abs().max()
    dup_all = int(df_all["sum_dup_flag"].sum())
    dup_gaps = np.unique(np.diff(np.flatnonzero(df_all["sum_dup_flag"].to_numpy())))
    dup_spacing = (f"every {dup_gaps[0]:,} rows ({dup_gaps[0] / 48:g} days)" if len(dup_gaps) == 1
                   else "at irregular spacing")

    return f"""## 1. Data audit

Source: `data/raw/lcl/Data_features.xlsx`, sheet `DTOU_2013` (sha256 prefix `{sha}`). The sheet holds
two series of different length. `DateTime`/`Price`/`Temperature` sit on a regular 2013 half-hourly grid
({raw['DateTime'].notna().sum():,} rows, 48 per day, no DST gaps or repeats, labels 00:00-23:30, so
interval starts). The consumption columns have {len(raw):,} rows, and the last
{raw['DateTime'].isna().sum():,} have no date, price or temperature.

**The consumption rows are not aligned with the 2013 grid row by row.** Residual = log `mean_all`
minus its rolling 15-day median at the same row position mod 48, averaged over High and Low
half-hours when price row *s* is paired with consumption row *s + L*:

{md_table(show)}

As laid out in the file (L = 0) there is almost no price signal ({lag0.loc['GMT', 'resid_High_pct']:+.2f}% /
{lag0.loc['GMT', 'resid_Low_pct']:+.2f}% in GMT months). The response is sharpest at L = {lags['GMT']}
for GMT half-hours and L = {lags['BST']} for BST half-hours. The two differ by exactly one hour, so
consumption is on a clock without DST (UTC/GMT) and the price grid on local time. The loader treats
consumption row *t* as the UTC interval starting 2012-06-30 23:30 + 30 min x *t*. That is the same as
original labels 2012-07-01 00:00 ... 2014-02-28 00:00 marking interval ends. The file therefore holds
{(df_all.period == 'pre').sum():,} pre-trial half-hours (from 2012-07-01 00:30 local),
{(df_all.period == 'trial').sum():,} trial half-hours (2013) and {(df_all.period == 'post').sum():,}
post-trial half-hours (to 2014-02-27 23:30 GMT). **Pairing the first 17,520 consumption rows with
the 2013 prices, as the sheet layout invites, compares Jul 2012 - Jun 2013 load with 2013 prices.**

Residual (%) around event edges at the adopted alignment (k = 0 is the first event half-hour for
`start`, and the first half-hour after the event for `end`):

{md_table(edges)}

The step falls between k = -1 and k = 0 at both edges. Another reading of the same data is that
labels mark interval starts and households react 30 minutes after each price change. The data
cannot separate the two, so Section 4 also reports the event window under that reading.

DST handling: `ts_utc` is regular and unique. `ts_local` is timezone-naive local time, so 2013-03-31
has 46 half-hours and 2013-10-27 has 50 (01:00 and 01:30 twice, told apart by `is_dst`). The grid's
01:00/01:30 rows on 31 Mar go unused, and those on 27 Oct apply to both repeats. All four are
Normal price. Temperature is joined by local clock. Its diurnal peak (first harmonic, 21 days either
side) shifts by {phases[0]:+.1f} h at the spring change and {phases[1]:+.1f} h at the autumn change.
That lies between the 0 h expected for a UTC series and the +/-1 h expected for a local one, so the
temperature clock is unresolved. It does not matter here, because temperature is used only as a
daily mean.

Households behind the aggregates (`sum / mean`) in 2013. The table excludes {int(df['sum_dup_flag'].sum())}
half-hours ({dup_all} over Jul 2012 - Feb 2014, {dup_spacing}, each at original label 00:00) where `n_all`
is about 860, which means the sums double-count meters. The means in those rows are in line with
their neighbours.

{md_table(n.reset_index().rename(columns={'index': 'group'}), floatfmt='{:.0f}')}

In 2013, {near:.1f}% of half-hours have `n_all` within 2 of the median, so the count is stable; the
dips are scattered missing readings. Mean `n_all` is {df.loc[ev_hh, 'n_all'].mean():.1f} on event
half-hours and {df.loc[~ev_hh, 'n_all'].mean():.1f} on Normal ones. Monthly medians before the trial
are {', '.join(f'{k} {v:.0f}' for k, v in monthly.items() if k < '2013-01')}, and after it
{', '.join(f'{k} {v:.0f}' for k, v in monthly.items() if k >= '2014-01')}.
`sum_all` = `sum_flex + sum_noflex` to within {sum_gap:.1e} kWh.
"""


def section_schedule(df, ev, daily, K) -> str:
    rows = []
    for b in ("High", "Low"):
        e = ev[ev["band"] == b]
        rows.append({"band": b, "events": len(e), "half-hours": int(e["n_hh"].sum()),
                     "% of year": 100 * e["n_hh"].sum() / len(df),
                     "dates touched": int(daily[f"n_{b.lower()}"].gt(0).sum()),
                     "median h": e["duration_h"].median(), "mean h": e["duration_h"].mean(),
                     "max h": e["duration_h"].max(), "cross midnight": int(e["crosses_midnight"].sum())})
    counts = pd.DataFrame(rows)

    def tab(col: str, name: str) -> pd.DataFrame:
        t = pd.crosstab(ev[col], ev["band"]).rename_axis(index=name, columns=None).reset_index()
        t[name] = t[name].map(hours_label)
        return t

    by_month = pd.crosstab(ev["band"], ev["month"]).rename_axis(index="band", columns=None)
    by_month.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_wd = pd.crosstab(ev["band"], ev["weekday"]).reindex(columns=days).rename_axis(index="band", columns=None)
    by_wd.columns = [d[:3] for d in days]
    dtype = daily["day_type"].value_counts().reindex(["Normal-only", "High", "Low", "High+Low"])

    full = df.groupby("date")["band"].apply(list)
    full = full[full.map(len) == 48]
    profiles = full.map(describe_day).value_counts().drop("all Normal", errors="ignore")
    top = profiles.head(6).rename_axis("non-Normal runs within the local day").reset_index(name="days")
    hl_days = daily.index[daily["day_type"] == "High+Low"]
    hl = df[df["date"].isin(hl_days) & (df["band"] == "High")].groupby("date")["hh"].agg(["min", "max"])
    hl_evening = int(((hl["min"] == 34) & (hl["max"] == 45)).sum())

    blk = (df["ts_local"] - pd.Timedelta(hours=2)).dt.floor("3h")
    mixed = int(df.groupby(blk)["band"].nunique().gt(1).sum())
    K.update(
        n_events={b: int((ev["band"] == b).sum()) for b in ("High", "Low")},
        n_combos=int(ev.groupby(["band", "start_hour", "duration_h"]).ngroups),
        n_profiles=len(profiles),
        grid_starts=100 * (ev["start_hour"] % 3 == 2).mean(),
        grid_durs=100 * (ev["duration_h"] % 3 == 0).mean(),
        n_blocks=int(blk.nunique()), mixed_blocks=mixed,
        winter=100 * ev["month"].isin([12, 1, 2, 3]).mean(),
        both=int(dtype["High+Low"]), hl_evening=hl_evening,
        high17=int(((ev["band"] == "High") & (ev["start_hour"] == 17)).sum()),
        low5_23=int(((ev["band"] == "Low") & ev["start_hour"].isin([5, 23])).sum()),
        rare_high=pd.crosstab(ev["band"], ev["start_hour"]).loc["High"].loc[lambda c: c <= 3],
        dtype=dtype,
    )

    return f"""## 2. Price schedule (local time)

Prices take exactly three values: High 67.20, Normal 11.76 and Low 3.99 p/kWh. An event is a run of
consecutive half-hours in one non-Normal band. A run that crosses midnight counts as one event, dated
by its start.

{md_table(counts, floatfmt='{:.1f}')}

Day types (local calendar dates): {', '.join(f'{k} {v}' for k, v in dtype.items())}.
**{K['both']} days have both High and Low**, and on {hl_evening} of them High is exactly 17:00-23:00.

Duration and start-hour distributions (events):

{md_table(tab('duration_h', 'duration h'))}

{md_table(tab('start_hour', 'start hour'))}

{K['high17']} of {K['n_events']['High']} High events start at 17:00. {K['low5_23']} of
{K['n_events']['Low']} Low events start at 05:00 or 23:00. {K['grid_starts']:.0f}% of events start
on a 3-hour grid anchored at 02:00 (02, 05, 08, ..., 23), and {K['grid_durs']:.0f}% last a multiple
of 3 h. Only {mixed} of the {K['n_blocks']:,} 3-hour blocks on that grid mix two price bands.

Events by month and weekday:

{md_table(by_month, index=True)}

{md_table(by_wd, index=True)}

{K['winter']:.0f}% of events fall in December-March, which is a third of the year. There are {K['n_combos']}
distinct (band, start hour, duration) combinations and {K['n_profiles']} distinct within-day patterns
on 48-slot days that contain an event. The most common (overnight events show up as two calendar-day
pieces):

{md_table(top)}
"""


def section_weather(ev, daily, rng, K) -> str:
    t = daily.groupby("day_type")["temp_mean"].agg(days="size", mean_C="mean", sd_C="std")
    t = t.reindex(["Normal-only", "High", "Low", "High+Low"]).reset_index()
    d = daily.assign(month=daily.index.month, high=daily["n_high"].gt(0).astype(int),
                     low=daily["n_low"].gt(0).astype(int))
    fits = {"raw": smf.ols("temp_mean ~ high + low", data=d).fit(cov_type="HC1"),
            "month FE": smf.ols("temp_mean ~ C(month) + high + low", data=d).fit(cov_type="HC1")}
    reg = []
    for name, f in fits.items():
        cis = f.conf_int()
        reg.append({"model": name, **{f"{v} day, C [95% CI]": ci(f.params[v], *cis.loc[v], "{:+.2f}")
                                       for v in ("high", "low")}})
    raw_hi = fits["raw"].params["high"]
    fe_cover = all(fits["month FE"].conf_int().loc[v].prod() < 0 for v in ("high", "low"))

    normal = daily[daily["day_type"] == "Normal-only"]
    local = []
    for b in ("High", "Low"):
        diffs = []
        for day in ev.loc[ev["band"] == b, "date"].unique():
            near = normal[np.abs((normal.index - day).days) <= 14]
            if len(near):
                diffs.append(daily.loc[day, "temp_mean"] - near["temp_mean"].mean())
        m, lo, hi = boot_mean(np.array(diffs), rng)
        local.append(f"{b} {ci(m, lo, hi, '{:+.2f}')} C ({len(diffs)} dates)")
    wk = {k: 100 * daily.loc[daily["day_type"] == k, "weekend"].mean() for k in t["day_type"]}
    K["temp_raw_high"] = raw_hi

    verdict = ("Within month, event days are not detectably colder or warmer (both month-FE intervals "
               "include zero)" if fe_cover else "A temperature gap remains within month")
    return f"""## 3. Is event timing confounded with weather or season?

{md_table(t)}

OLS of daily mean temperature on "any High" / "any Low" day indicators (HC1 95% CI):

{md_table(pd.DataFrame(reg))}

Event date minus the mean of Normal-only dates within +/-14 days (bootstrap 95% CI over dates):
{'; '.join(local)}. Weekend share: {', '.join(f'{k} {v:.0f}%' for k, v in wk.items())}.

Event days are colder on a raw comparison because events cluster in winter. {verdict}, and within
+/-14 days the gap is small. Event timing is therefore confounded with **season**, which local
matching (Section 4) and, later, the control group must absorb. Within-season weather confounding
is not visible in daily mean temperature. That does not rule out other drivers of event scheduling
(wind output, network constraints) that may also move demand; check the trial documentation for the
scheduling rule.
"""


def run_matched(df, ev, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    tab = matched_window_table(df, ev, **kw)
    return tab, bootstrap_summary(tab, n_boot=N_BOOT, seed=SEED)


def fmt_summary(s: pd.DataFrame) -> pd.DataFrame:
    order = {w: i for i, w in enumerate(WINDOWS)}
    s = s.assign(o=s["window"].map(order)).sort_values(["band", "o"])
    return pd.DataFrame({
        "band": s["band"], "window": s["window"].map(WINDOW_LABELS), "events used": s["n_events"],
        "dropped": s["n_dropped"], "matched days (median)": s["median_ctrl_days"].map("{:.0f}".format),
        "pooled % [95% CI]": [ci(*r) for r in s[["pooled_pct", "pooled_lo", "pooled_hi"]].to_numpy()],
        "mean of event % [95% CI]": [ci(*r) for r in s[["mean_pct", "mean_lo", "mean_hi"]].to_numpy()],
    })


def section_response(df, ev, K) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    tab, summ = run_matched(df, ev)
    drops = (tab[tab["pct"].isna()].pivot_table(index=["band", "window"], columns="reason", values="event_id",
                                                aggfunc="size", fill_value=0).reset_index())
    drops["window"] = drops["window"].map(WINDOW_LABELS)

    groups = []
    for col in ("mean_all", "mean_flex", "mean_noflex"):
        for r in run_matched(df, ev, col=col, windows=("event",))[1].itertuples():
            groups.append({"series": col, "band": r.band, "events": r.n_events,
                           "event window pooled % [95% CI]": ci(r.pooled_pct, r.pooled_lo, r.pooled_hi)})
    groups = pd.DataFrame(groups).sort_values(["band", "series"])
    K["flex"] = {g["band"]: g["event window pooled % [95% CI]"] for g in groups.to_dict("records")
                 if g["series"] == "mean_flex"}

    t_ev = tab[tab["window"] == "event"].merge(ev[["event_id", "start_hour", "duration_h"]], on="event_id")
    t_ev["start block"] = t_ev["start_hour"].map(start_block)
    t_ev["duration"] = t_ev["duration_h"].map(duration_bucket)
    het = []
    for by, levels in (("start block", list(BLOCKS)), ("duration", ["<=3 h", "4-6 h", ">6 h"])):
        s = bootstrap_summary(t_ev, n_boot=N_BOOT, seed=SEED, by=("band", by))
        for r in s.to_dict("records"):
            het.append({"band": r["band"], "split": by, "level": r[by], "o": levels.index(r[by]),
                        "events": r["n_events"], "pooled % [95% CI]": ci(r["pooled_pct"], r["pooled_lo"], r["pooled_hi"])})
    het = pd.DataFrame(het).sort_values(["band", "split", "o"]).drop(columns="o")

    variants = [
        ("adopted (UTC, interval-end labels; +/-14 d, +/-2 C)", "utc_end_label", {}),
        ("+ same weekday/weekend type", "utc_end_label", {"same_daytype": True}),
        ("temperature +/-1 C", "utc_end_label", {"temp_tol": 1.0}),
        ("+/-7 days", "utc_end_label", {"days": 7}),
        ("alt. alignment: interval-start labels (+30 min)", "utc_start_label", {}),
        ("file row order as shipped (misaligned)", "file_order", {}),
    ]
    sens = []
    for label, align, kw in variants:
        d = df if align == "utc_end_label" else lf.load_lcl_features(alignment=align, validate=False)
        e = ev if align == "utc_end_label" else lf.find_events(d)
        rec = {"variant": label}
        for r in run_matched(d, e, windows=("event",), **kw)[1].itertuples():
            rec[f"{r.band}: events, pooled % [95% CI]"] = f"{r.n_events}, {ci(r.pooled_pct, r.pooled_lo, r.pooled_hi)}"
        sens.append(rec)
    sens = pd.DataFrame(sens)
    K["sens"] = sens

    s = summ.set_index(["band", "window"])
    K["resp"] = {k: ci(*s.loc[k, ["pooled_pct", "pooled_lo", "pooled_hi"]]) for k in s.index}
    K["resp_pct"] = s["pooled_pct"].to_dict()
    def _signif(b: str, w: str) -> str:
        lo, hi = s.loc[(b, w), ["pooled_lo", "pooled_hi"]]
        edge = " - borderline, bound within 0.25 points of zero" if min(abs(lo), abs(hi)) < 0.25 else ""
        return f"{WINDOW_LABELS[w]} {b} events ({K['resp'][(b, w)]}%{edge})"

    K["signif"] = [_signif(b, w) for (b, w) in s.index
                   if w != "event" and s.loc[(b, w), "pooled_lo"] * s.loc[(b, w), "pooled_hi"] > 0]
    K["null"] = [f"{WINDOW_LABELS[w]} {b} events" for (b, w) in s.index
                 if w != "event" and s.loc[(b, w), "pooled_lo"] * s.loc[(b, w), "pooled_hi"] <= 0]

    text = f"""## 4. Descriptive first look at response size (NOT causal)

Method: for each event, compare `mean_all` (kWh per household per half-hour) in a window on the event
day with the same local clock window on Normal-only dates (no High or Low half-hour on any date the
window touches). Those dates must lie within +/-14 days and have a daily mean temperature within
+/-2 C of the event date's. Windows: 2 h before the start, the event itself, 0-2 h and 2-4 h after
the end. A pre or post window is used only if it is all Normal price on the event day. "Pooled %" =
sum(event kWh) / sum(matched kWh) - 1, weighted by window length. "Mean of event %" weights events
equally. 95% CIs are a percentile bootstrap over events ({N_BOOT} draws). Events share matched days
and some share dates, so these intervals are, if anything, too narrow.

{md_table(fmt_summary(summ))}

Windows without an estimate:

{md_table(drops)}

Post-High windows exist only for the High events that are not directly followed by a Low event, and
pre-Low windows only for Low events not directly preceded by a High event. Those are selected subsets.

By series (event window). `flex`/`noflex` is the mirror authors' ex-post cluster split:

{md_table(groups)}

By start block and duration (event window, `mean_all`):

{md_table(het)}

Sensitivity of the event-window result:

{md_table(sens)}

Limitations:

* No control group. The comparison is the same households on other days, so anything else that
  differs between event and Normal-only days enters these numbers: weather beyond the daily mean,
  holidays, whatever drives the scheduling rule, and notification effects.
* Day-ahead notice lets households move load into the hours around an event. Habits formed on event
  days can also carry over to the matched days. Both bias a same-household comparison.
* The `flex`/`noflex` split was made ex post by the mirror authors, and the file does not document
  its features. If it used 2013 load, it selects on the response, so the flex/noflex gap is not
  evidence of heterogeneity.
* The ~{K['n_all_median']:.0f} households are whatever subset the mirror aggregated. Selection into
  the trial and into this file is not documented here.
* The alignment is inferred from the data (Section 1), and the half-hour convention is not settled.
"""
    return text, summ, tab


def section_key(K) -> str:
    r = K["resp"]
    return f"""## Key findings

* **The file's consumption rows are offset from its 2013 price rows.** The best alignment is
  {K['lags']['GMT']} rows (GMT) / {K['lags']['BST']} rows (BST), so consumption runs Jul 2012 -
  Feb 2014 on a UTC clock. As shipped, the sheet shows no price response (sensitivity table,
  Section 4). The loader fixes this, and the fix also gives 6 pre-trial and 2 post-trial months.
* 2013 schedule: {K['n_events']['High']} High and {K['n_events']['Low']} Low events, built almost
  entirely from 3-hour blocks on a grid anchored at 02:00 ({K['grid_starts']:.0f}% of starts,
  {K['grid_durs']:.0f}% of durations). {K['winter']:.0f}% of events fall in Dec-Mar.
  {K['both']} days have both High and Low.
* Event timing is confounded with season (events cluster in winter). Once month or a +/-14-day
  window is held fixed, event days show no detectable gap in daily mean temperature.
* Descriptive matched-day differences in `mean_all` (pooled, 95% CI): High events {r[('High', 'event')]}%,
  Low events {r[('Low', 'event')]}%. Windows whose CI excludes zero: {'; '.join(K['signif']) or 'none'}.
  These are not causal estimates.
"""


def section_implications(K) -> str:
    return f"""## 5. Implications

**Identification in the full DiD (meter-level data with the flat-rate control group):**

* Every dToU household faces the same event calendar. Within the treated group, event effects are
  therefore collinear with date x half-hour effects. Only the control group identifies those time
  effects, so the contrast is (dToU - control) on event half-hours vs on Normal half-hours, with
  household x half-hour-of-day fixed effects.
* Events cluster in winter ({K['winter']:.0f}% in Dec-Mar), so the seasonal counterfactual has to come
  from the control group. Group x temperature (or group x month) interactions are a needed robustness
  check.
* Load moves outside the event window: {'; '.join(K['signif']) or 'none detected'}. No clear change:
  {', '.join(K['null']) or 'none'}. A DiD that uses every Normal half-hour as the baseline would put
  these spillovers into the reference period. It should include event-time leads and lags (about
  -2 h to +4 h) and a baseline that excludes them.
* With the alignment fixed, the file includes Jul-Dec 2012 for the same aggregate group, although its
  household count still changes during 2012 (Section 1), so the composition is not fixed. If the
  meter-level data confirm this, a pre-period is available to test (dToU - control) for level gaps and
  pre-trends before any event, on a balanced panel.
* Normal (11.76 p) is below the control's flat 14.228 p, so dToU and control also differ in price on
  Normal half-hours. State event effects relative to Normal, or model both price gaps.

**RL action design:**

* The observed support is a 3-hour block grid anchored at 02:00, eight blocks a day, each High, Normal
  or Low. {K['grid_starts']:.0f}% of event starts and {K['grid_durs']:.0f}% of durations fit it, and only
  {K['mixed_blocks']} of {K['n_blocks']:,} blocks mix bands. There are {K['n_combos']} distinct
  (band, start, duration) combinations and {K['n_profiles']} distinct within-day patterns on event
  days. A discrete action space of 8 blocks x 3 levels keeps each block's timing and price level
  inside the observed support, but it allows 3^8 = 6,561 day patterns, far more than the
  {K['n_profiles']} observed. Day patterns outside the observed set, finer actions (30-min or 1-h
  events) and placements the trial rarely used all mean extrapolating. Rarely used placements: High
  events starting at {', '.join(f'{h:02.0f}:00 (n={n})' for h, n in K['rare_high'].items())}.
* Use the event-window sizes here (High {K['resp'][('High', 'event')]}%, Low {K['resp'][('Low', 'event')]}%)
  only as sanity bounds. The response model has to come from the DiD and needs pre- and post-event
  terms, because the reward depends on net shifted energy.
* Of the {K['both']} High+Low days, {K['hl_evening']} have High exactly 17:00-23:00, so interactions
  between adjacent High and Low blocks are identified mostly from that one pattern.
"""


# ---------------------------------------------------------------- figures
def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def fig_alignment(scores: pd.DataFrame, path) -> None:
    lags = lf.best_lags(scores)
    s = scores[scores["lag"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, clock in zip(axes, ("GMT", "BST")):
        g = s[s["clock"] == clock]
        for b in ("High", "Low"):
            ax.plot(g["lag"], g[f"resid_{b}_pct"], color=COLORS[b], lw=2, label=f"{b}-price half-hours")
        ax.axvline(lags[clock], color=INK2, lw=1, ls="--")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"{clock} half-hours of 2013 (best L = {lags[clock]}, dashed)", color=INK, fontsize=10,
                     loc="left")
        ax.set_xlabel("row offset L (consumption row = price row + L)", color=INK2, fontsize=9)
        style(ax)
    axes[0].set_ylabel("mean load residual (%)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle("Load responds to price only after offsetting rows; the offset differs by 1 h with DST",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def fig_windows(summ: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(WINDOWS))
    for j, b in enumerate(("High", "Low")):
        g = summ[summ["band"] == b].set_index("window").reindex(list(WINDOWS))
        ax.errorbar(x + (j - 0.5) * 0.18, g["pooled_pct"],
                    yerr=[g["pooled_pct"] - g["pooled_lo"], g["pooled_hi"] - g["pooled_pct"]],
                    fmt="o", ms=7, color=COLORS[b], elinewidth=2, capsize=0, label=f"{b} events")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [WINDOW_LABELS[w] for w in WINDOWS])
    ax.set_ylabel("vs matched Normal-only days (%)", color=INK2, fontsize=9)
    ax.set_title("Descriptive, not causal: dToU load around events vs matched days (95% CI)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raw = lf.read_raw()
    scores = lf.estimate_alignment_lag(raw)
    df = lf.load_lcl_features()
    df_all = lf.load_lcl_features(period="all", validate=False)
    ev = lf.find_events(df)
    daily = lf.daily_table(df)

    K: dict = {}
    s1 = section_data(raw, df, df_all, scores, ev, K)
    s2 = section_schedule(df, ev, daily, K)
    s3 = section_weather(ev, daily, rng, K)
    s4, summ, tab = section_response(df, ev, K)
    report = "\n".join([
        "# LCL dToU 2013: price-schedule audit and first descriptive look\n",
        "Generated by `python scripts/audit_lcl_schedule.py`. Descriptive only: the source file has "
        "no control group. Times are local (Europe/London) unless stated.\n",
        section_key(K), s1, s2, s3, s4, section_implications(K),
        "Outputs: `lcl_alignment_lag.png`, `lcl_matched_windows.png`, `lcl_events.csv`, "
        "`lcl_matched_windows.csv` (all in `outputs/`).\n",
    ])
    (OUTPUTS / "lcl_schedule_summary.md").write_text(report)
    ev.drop(columns=["i0", "i1"]).to_csv(OUTPUTS / "lcl_events.csv", index=False)
    tab.to_csv(OUTPUTS / "lcl_matched_windows.csv", index=False)
    fig_alignment(scores, OUTPUTS / "lcl_alignment_lag.png")
    fig_windows(summ, OUTPUTS / "lcl_matched_windows.png")
    print(report)


if __name__ == "__main__":
    main()
