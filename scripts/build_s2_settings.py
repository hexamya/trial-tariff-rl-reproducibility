"""S2 settings fixed before S3: reward weights, constraint caps and the forecasts the agents see.

Run:  python scripts/build_s2_settings.py   (~1 min; needs build_london_env.py, build_ercot_env.py)

Writes data/processed/s2_settings.json (ttrl.env.rewards.S2Settings) and outputs/s2_settings.md
(copied to docs/reports). The report identifies each input and its source. See ``ttrl.env.rewards``
for the definitions.

* Dollar year 2024: BEA GDP implicit price deflator (FRED A191RD3A086NBEA, annual, cached in data/raw/fred).
* Gas: delivered cost of natural gas to the Texas electric power sector, 2024, by month (EIA fuel
  receipts aggregates, PUDL ``core_eia__yearly_fuel_receipts_costs_aggs``).
* CO2: Rennert et al. (2022) mean social cost of CO2, deflated from 2020 to 2024 US$; their 5-95%
  range as the sensitivity.
* Capacity: EIA AEO2026 industrial-frame CT in region TRE, annualised (``rewards.capacity_value_per_kw_yr``).
* Month weights: share of each calendar month in the 1% highest net-load hours (EIA-930 demand minus
  wind and solar) of each year 2019-2023.
* London caps: revenue +/-3% and segment mean bill change at most +3% (Kale et al.); segments = quartiles
  of pre-trial evening consumption share; the 2013 schedule's household and segment bill changes for
  reference.
* Forecast checks: ERCOT's day-ahead forecast (EIA-930) against demand, 2019-2024; the London
  day-ahead forecast (fitted on pre-trial days) on the 2013 trial days against two naive forecasts.
"""
from __future__ import annotations

import io
import shutil
import urllib.request

import numpy as np
import pandas as pd

from build_london_env import schedule_on_grid
from ttrl.data import eia930, pudl
from ttrl.env import ercot as er
from ttrl.env import london as ld
from ttrl.env import rewards as rw
from ttrl.paths import OUTPUTS, RAW, ROOT
from ttrl.reporting import md_table

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=A191RD3A086NBEA"
FRED_PATH = RAW / "fred" / "A191RD3A086NBEA.csv"
DOCS_REPORTS = ROOT / "docs" / "reports"
CT_AERO_TRE_USD_2025 = 2058.0          # EIA AEO2026 EMM Table 4, CT--aeroderivative, 1 TRE (high case)
CT_AERO_FOM_USD_2025 = 9.99            # EIA AEO2026 EMM Table 3
SC_CO2_RANGE_USD_2020 = (44.0, 413.0)  # Rennert et al. 2022, 5-95% range


def gdp_deflator() -> pd.Series:
    if not FRED_PATH.exists():
        FRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FRED_URL, timeout=60) as r:
            FRED_PATH.write_bytes(r.read())
    d = pd.read_csv(FRED_PATH)
    d.columns = ["date", "index"]
    return d.set_index(pd.to_datetime(d["date"]).dt.year)["index"].astype(float)


def gas_prices() -> tuple[dict[int, float], float, pd.DataFrame]:
    a = pd.read_parquet(pudl.download_table("core_eia__yearly_fuel_receipts_costs_aggs"))
    a = a[(a["fuel_agg"] == "natural_gas") & (a["geo_agg"] == "TX") & (a["sector_agg"] == "all_electric_power")]
    a = a.assign(year=pd.to_datetime(a["report_date"]).dt.year, month=pd.to_datetime(a["report_date"]).dt.month)
    m = a[(a["temporal_agg"] == "monthly") & (a["year"] == rw.DOLLAR_YEAR)].sort_values("month")
    annual = a[a["temporal_agg"] == "annual"].set_index("year")["fuel_cost_per_mmbtu"]
    if len(m) != 12:
        raise ValueError("expected 12 monthly gas prices")
    return dict(zip(m["month"].astype(int), m["fuel_cost_per_mmbtu"].astype(float))), float(annual.loc[rw.DOLLAR_YEAR]), \
        annual.loc[2019:2025].rename("annual $/MMBtu").to_frame()


def month_weights(col: str = "net") -> tuple[dict[int, float], int]:
    rows = []
    for y in rw.WEIGHT_YEARS:
        h = er.local_year_hours(y)
        d = eia930.load_ba_hourly(y).set_index("datetime_utc").reindex(h)
        v = d["demand"] - d["wind"].fillna(0) - d["solar"].fillna(0) if col == "net" else d["demand"]
        top = v >= v.quantile(1 - rw.PEAK_HOUR_SHARE)
        rows.append(pd.Series((h - er.LST_OFFSET).month[top.to_numpy()]))
    months = pd.concat(rows)
    w = months.value_counts().reindex(range(1, 13), fill_value=0) / len(months)
    return {int(k): float(v) for k, v in w.items()}, len(months)


def london_caps() -> tuple[float, dict]:
    inp = ld.LondonInputs.load()
    tdays = pd.DatetimeIndex(inp.days["tday"])
    blocks, _ = schedule_on_grid(tdays)
    trial = (inp.days["period"] == "trial").to_numpy()
    idx = np.flatnonzero(trial)
    zero = np.zeros((len(idx), ld.HH_PER_DAY))
    bills0 = ld.bill_changes(blocks[trial], inp.households[:, idx], zero)
    seg = ld.exposure_segments(inp.households, inp.days)
    month = tdays[trial].to_period("M").astype(str)
    monthly = {}
    for m in sorted(set(month)):
        b = ld.bill_changes(blocks[trial][month == m], inp.households[:, idx[month == m]], zero[month == m])
        monthly[m] = {"cvar95": ld.cvar(b), "segment_max": float(np.max(ld.segment_bill_means(b, seg))),
                      "revenue_ratio": ld.revenue_ratio(blocks[trial][month == m], inp.baseline[idx[month == m]], zero[month == m])}
    return rw.LONDON_SEGMENT_BILL_CAP, {"households": int(inp.households.shape[0]), "cvar95": ld.cvar(bills0),
                                        "p95": ld.percentile(bills0, 95), "segments": ld.segment_bill_means(bills0, seg).tolist(),
                                        "monthly": monthly, "revenue_ratio": ld.revenue_ratio(blocks[trial], inp.baseline[idx], zero)}


def ercot_forecast_check() -> pd.DataFrame:
    rows = []
    for y in range(2019, 2025):
        h = er.local_year_hours(y)
        f = eia930.load_ba_forecast(y).set_index("datetime_utc").reindex(h)
        loc = h.tz_localize("UTC").tz_convert(er.TZ).tz_localize(None).normalize()
        ok = ~np.isin(loc, pd.DatetimeIndex(er.curtailed_days(y)["date"]))
        ape = (f["demand_forecast"] - f["demand"]).abs() / f["demand"]
        pk = (f["demand_forecast"].groupby(loc).max() / f["demand"].groupby(loc).max() - 1)
        pk = pk[~pk.index.isin(pd.DatetimeIndex(er.curtailed_days(y)["date"]))]
        rows.append({"year": y, "hours": len(h), "missing forecast hours": int(f["demand_forecast"].isna().sum()),
                     "MAPE % (non-curtailed hours)": 100 * float(ape[ok].mean()),
                     "daily peak error, mean abs %": 100 * float(pk.abs().mean()),
                     "daily peak bias %": 100 * float(pk.mean())})
    return pd.DataFrame(rows)


def london_forecast_check() -> pd.DataFrame:
    inp = ld.LondonInputs.load()
    base, days = inp.baseline, inp.days
    fc = ld.day_ahead_forecast(base, days)
    tday = pd.DatetimeIndex(days["tday"])
    pos = pd.Index(tday)
    trial = (days["period"] == "trial").to_numpy()
    out = []
    for name, f in (("day-ahead forecast (fit on pre-trial days)", fc),
                    ("same half-hour 7 days earlier", base[np.maximum(pos.get_indexer(tday - pd.Timedelta(days=7)), 0)]),
                    ("same half-hour 2 days earlier", base[np.maximum(pos.get_indexer(tday - pd.Timedelta(days=2)), 0)])):
        e = (f[trial] - base[trial]) / base[trial]
        pk = f[trial].max(axis=1) / base[trial].max(axis=1) - 1
        hp = np.abs(np.argmax(f[trial], axis=1) - np.argmax(base[trial], axis=1))
        out.append({"forecast": name, "trial days": int(trial.sum()), "MAPE %": 100 * float(np.abs(e).mean()),
                    "daily peak error, mean abs %": 100 * float(np.abs(pk).mean()),
                    "peak half-hour within 1 h %": 100 * float((hp <= 2).mean())})
    return pd.DataFrame(out)


def ercot_episode_check(st: rw.S2Settings) -> pd.DataFrame:
    """Reward components of one fixed schedule (weekday High 17-20 h, kernel estimate) in months of the
    validation year: orders of magnitude of fuel, CO2 and capacity terms."""
    from ttrl.analysis import lcl_kernel as kn
    from ttrl.env import ercot_env as ee
    from ttrl.env import fastkernel as fk
    from ttrl.env import tariff as tf
    from ttrl.env.responses import TrialResponse
    spec = kn.KernelSpec.from_json((OUTPUTS / "lcl_kernel_spec.json").read_text(encoding="utf-8"))
    terms = pd.read_csv(OUTPUTS / "lcl_kernel_terms.csv").set_index("term")["coef"]
    resp = TrialResponse(fk.from_coefficients(spec, terms))
    days = ee.ErcotDays.load(2023)
    rows = []
    for mon in ("2023-01", "2023-04", "2023-07", "2023-08"):
        env = ee.ErcotTariffEnv(days, resp, ee.ErcotConfig.from_settings(st, [mon], np.array([0])), seed=0)
        env.reset(options={"month": mon, "member": 0})
        total, done, k = 0.0, False, 0
        while not done:
            day = np.zeros(tf.BLOCKS_PER_DAY, np.int8)
            if pd.Timestamp(days.tdays[env.day_idx[k]]).dayofweek < 5:
                day[5] = tf.HIGH
            _, r, done, _, info = env.step(day)
            total += r
            k += 1
        m = info["metrics"]
        rows.append({"month": mon, "reward $": total, "fuel $": m["fuel_cost"], "CO2 $": m["co2_cost"],
                     "capacity $": m["capacity_cost"], "fuel MMBtu": m["d_fuel_mmbtu"],
                     "net-load peak change MW": m["monthly_net_peak_change_mw"], "revenue ratio": m["revenue_ratio"],
                     "masked days": m["masked_days"]})
    return pd.DataFrame(rows)


def main() -> None:
    defl = gdp_deflator()
    to24 = lambda v, y: v * defl.loc[rw.DOLLAR_YEAR] / defl.loc[y]   # noqa: E731
    gas_m, gas_y, gas_years = gas_prices()
    co2 = to24(rw.SC_CO2_USD_2020, 2020)
    cap_kw_2025 = rw.capacity_value_per_kw_yr()
    cap = to24(cap_kw_2025, 2025) * 1000
    weights, n_hours = month_weights("net")
    w_demand, _ = month_weights("demand")
    bill_cap, lon = london_caps()
    sens = {"co2_price_rennert_5pct": to24(SC_CO2_RANGE_USD_2020[0], 2020), "co2_price_rennert_95pct": to24(SC_CO2_RANGE_USD_2020[1], 2020),
            "co2_price_zero": 0.0, "gas_price_texas_power_2022": float(gas_years.loc[2022].iloc[0]),
            "capacity_value_20yr": to24(rw.capacity_value_per_kw_yr(years=20), 2025) * 1000,
            "capacity_value_aeroderivative": to24(rw.capacity_value_per_kw_yr(CT_AERO_TRE_USD_2025, CT_AERO_FOM_USD_2025), 2025) * 1000,
            "peak_month_weight_demand": w_demand}
    sources = {
        "dollar_year": "BEA GDP implicit price deflator, FRED A191RD3A086NBEA (annual, 2017=100)",
        "gas": "EIA fuel receipts aggregates, natural gas, Texas, all electric power, monthly 2024 (PUDL core_eia__yearly_fuel_receipts_costs_aggs)",
        "co2": "Rennert et al. (2022) Nature 610:687-692, doi:10.1038/s41586-022-05224-9: $185/tCO2 (2020 US$), 2% near-term discount rate",
        "capacity": "EIA, Assumptions to the Annual Energy Outlook 2026: Electricity Market Module (April 2026), Table 3, Table 4 (1 TRE), p. 9",
        "peak_month_weight": "EIA-930 ERCO demand minus wind and solar, top 1% hours of each year 2019-2023",
        "london_revenue_tol": "Kale et al. (revenue within +/-3% of flat-normal)",
        "london_segment_bill_cap": "Kale et al. (segment expected bills at most +3%); segments = quartiles of pre-trial 17-23 h consumption share",
    }
    st = rw.S2Settings(rw.DOLLAR_YEAR, gas_m, co2, cap, weights, rw.LONDON_REVENUE_TOL, bill_cap, sens, sources)
    st.save()
    ef, lf = ercot_forecast_check(), london_forecast_check()
    epi = ercot_episode_check(st)

    env24 = pd.read_parquet(er.inputs_path(2024), columns=["fuel_mmbtu_per_mwh", "co2_t_per_mwh"])
    fuel_mwh, co2_mwh = float(env24["fuel_mmbtu_per_mwh"].mean()), float(env24["co2_t_per_mwh"].mean())
    gas_tab = pd.DataFrame({"month": list(gas_m), "$/MMBtu": list(gas_m.values())})
    w_tab = pd.DataFrame({"month": range(1, 13), "net-load weight %": [100 * weights[m] for m in range(1, 13)],
                          "demand weight % (sensitivity)": [100 * w_demand[m] for m in range(1, 13)]})
    crf, idc = rw.crf(rw.EIA_DISCOUNT_RATE, rw.RECOVERY_YEARS), rw.idc_factor(rw.EIA_DISCOUNT_RATE, rw.CT_LEAD_YEARS)
    lines = [
        "# S2 settings fixed before S3: reward weights, constraint caps, forecasts",
        "",
        "Generated by `python scripts/build_s2_settings.py`; values in `data/processed/s2_settings.json` "
        "(`ttrl.env.rewards.S2Settings`). Money in 2024 US$ (BEA GDP deflator: "
        f"2020 {defl.loc[2020]:.3f}, 2024 {defl.loc[2024]:.3f}, 2025 {defl.loc[2025]:.3f}).",
        "",
        "## ERCOT reward: fuel + CO2 + capacity",
        "",
        f"* **Gas price**: delivered cost to the Texas electric power sector, 2024 by month (annual {gas_y:.2f} $/MMBtu), "
        "the prices the test-year fleet paid. Applied by calendar month in every weather year.",
        "",
        md_table(gas_tab, floatfmt="{:.2f}"),
        "",
        "  Annual Texas electric-power gas cost, for context (same source): "
        + ", ".join(f"{y} {v:.2f}" for y, v in gas_years["annual $/MMBtu"].items()) + " $/MMBtu.",
        f"* **CO2 price**: {co2:.1f} $/t (Rennert et al. 2022: {rw.SC_CO2_USD_2020:.0f} $/t in 2020 US$ at a 2% near-term "
        "discount rate): the peer-reviewed estimate, independent of US policy; Executive Order 14154 (January 2025) "
        "disbanded the Interagency Working Group and withdrew its estimates (CRS IF12916).",
        f"* **Capacity value**: {cap:,.0f} $ per MW-yr = new industrial-frame CT in ERCOT (EIA AEO2026, region TRE): "
        f"overnight {rw.CT_OVERNIGHT_TRE_USD_2025:,.0f} $/kW x interest during construction {idc:.4f} ({rw.CT_LEAD_YEARS}-year "
        f"lead time, mid-year spending) x capital recovery factor {crf:.4f} ({rw.EIA_DISCOUNT_RATE:.1%} EIA discount rate, "
        f"no risk adder for CTs; {rw.RECOVERY_YEARS} years) + fixed O&M {rw.CT_FOM_USD_2025:.2f} $/kW-yr = "
        f"{cap_kw_2025:.2f} $/kW-yr in 2025 US$. Taxes are not modelled.",
        f"* **Month weights of the capacity value**: share of each calendar month in the 1% highest net-load hours of each "
        f"year 2019-2023 ({n_hours} hours). The capacity charge of a monthly episode is capacity value x weight x change of "
        "the month's net-load peak.",
        "",
        md_table(w_tab, floatfmt="{:.1f}"),
        "",
        f"For orientation (2024 surrogate means: {fuel_mwh:.2f} MMBtu and {co2_mwh:.3f} t CO2 per MWh of gas output): "
        f"fuel {gas_y * fuel_mwh:.1f} $/MWh and CO2 {co2 * co2_mwh:.1f} $/MWh at the annual gas price; one MW off the "
        f"August net-load peak is worth {cap * weights[8]:,.0f} $.",
        "",
        "Sensitivity values (recorded in the settings file, not used for training): "
        + "; ".join(f"{k} = {v:,.2f}" if not isinstance(v, dict) else f"{k} (table above)" for k, v in sens.items()) + ".",
        "",
        "Reward components of one fixed schedule (weekday High 17-20 h, kernel estimate, member 0) in months of the "
        "validation year 2023 (negative cost = saving; the reward is minus the sum):",
        "",
        md_table(epi, floatfmt="{:,.2f}"),
        "",
        "## London constraints",
        "",
        f"* Revenue within +/-{rw.LONDON_REVENUE_TOL:.0%} of the flat tariff each month (Kale et al.). Reward: mean daily "
        "peak reduction, Kale et al.'s objective, for comparability.",
        f"* Bill risk: the mean monthly bill change of each household segment at most **+{100 * bill_cap:.0f}%** "
        "(Kale et al.'s segment cap), with segments the quartiles of the households' pre-trial share of consumption at "
        "17:00-23:00 (the households most exposed to evening High prices), for households that do not respond (so the "
        "constraint is the same whatever response an agent trains on). Household P95 and CVaR95 are reported, not capped: "
        "within one month events fall unevenly on households, and a first choice, a household CVaR95 cap at the 2013 "
        "schedule's annual value, could not be met together with monthly revenue neutrality.",
        f"* The 2013 schedule over the trial year on the environment's {lon['households']} households (no response): revenue "
        f"ratio {lon['revenue_ratio']:.4f}, household P95 {100 * lon['p95']:.2f}%, CVaR95 {100 * lon['cvar95']:.2f}%, segment means "
        + ", ".join(f"{100 * v:+.2f}%" for v in lon["segments"]) + ". By month it balanced revenue over the year, not each "
        "month (revenue ratio " + f"{min(v['revenue_ratio'] for v in lon['monthly'].values()):.3f} to "
        f"{max(v['revenue_ratio'] for v in lon['monthly'].values()):.3f}; worst segment {100 * min(v['segment_max'] for v in lon['monthly'].values()):+.1f}% "
        f"to {100 * max(v['segment_max'] for v in lon['monthly'].values()):+.1f}%).",
        "",
        "## Forecasts in the agents' observations",
        "",
        "ERCOT: ERCOT's own day-ahead demand forecast as published in EIA-930 (scaled like demand in the training "
        "years); curtailed-demand days excluded from the errors.",
        "",
        md_table(ef, floatfmt="{:.2f}"),
        "",
        "London: per half-hour regression on the load 7 and 2 days earlier, the previous day's temperature and heating "
        "degrees, and day types, fitted on the pre-trial days (July-December 2012) and applied to 2013; compared on the "
        "trial days with two naive forecasts.",
        "",
        md_table(lf, floatfmt="{:.2f}"),
        "",
        "Both forecasts use only what is known when the next day's tariff is posted (the previous day's temperature, "
        "not the day's own).",
    ]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / "s2_settings.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy(out, DOCS_REPORTS / out.name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
