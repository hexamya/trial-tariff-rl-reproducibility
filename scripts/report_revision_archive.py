"""Reanalyse saved simulations for the manuscript revision.

This aggregates recorded outcomes and does not evaluate a policy. Average
over response members and seeds within month before resampling months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table


OUT = OUTPUTS / "revision" / "archive"


def paired(frame: pd.DataFrame, metric: str, label: str, sign: float) -> dict:
    monthly = frame.groupby(["month", "world"], observed=True)[metric].mean().unstack("world")
    monthly = sign * monthly[["self-assessed", "realised"]].dropna()
    design, evaluated = monthly.to_numpy().T
    gap = design - evaluated
    rng = np.random.default_rng(20261123)
    index = rng.integers(0, len(gap), (5000, len(gap)))
    result = {"comparison": label, "months": len(gap), "projection": design.mean(),
              "evaluation": evaluated.mean(), "optimism gap": gap.mean()}
    for name, values in [("evaluation", evaluated), ("gap", gap)]:
        lower, upper = np.quantile(values[index].mean(axis=1), (0.025, 0.975))
        result |= {f"{name} lo": lower, f"{name} hi": upper}
    return result


def breaches(frame: pd.DataFrame, label: str, world: str) -> dict:
    revenue = frame["cost_revenue"].fillna(0)
    segment = frame.get("cost_bill_segment", pd.Series(0.0, index=frame.index)).fillna(0)
    return {"method": label, "world": world, "episodes": len(frame),
            "any limit %": 100 * frame.violated.mean(), "revenue %": 100 * revenue.gt(1e-12).mean(),
            "revenue excess pp": 100 * revenue.mean(), "bill %": 100 * segment.gt(1e-12).mean(),
            "bill excess pp": 100 * segment.mean()}


def main() -> None:
    rl = pd.read_parquet(OUTPUTS / "s3_rl_episodes.parquet")
    london_rl = rl[(rl.case == "london") & ~rl.ppo]
    assumed = pd.read_parquet(OUTPUTS / "s3_planners_assumed_london_episodes.parquet")
    selected = pd.read_parquet(OUTPUTS / "sensitivity" / "L_main.parquet")
    original = assumed[assumed.planner.eq("MIP (oracle month)")]
    mpc = assumed[assumed.planner.eq("MPC (day-ahead)")]
    london_pairs = [(f"PPO {agent}", group) for agent, group in london_rl.groupby("agent")]
    london_pairs.extend([("Monthly MIP B (July checked)", selected), ("Monthly MIP B (original)", original),
                         ("Day-ahead MPC B", mpc)])
    london = pd.DataFrame([paired(data, "daily_peak_cut_pct", label, 1.0) for label, data in london_pairs])
    london_breach = pd.DataFrame([breaches(group, label, world) for label, data in london_pairs
                                 for world, group in data.groupby("world")])

    ercot = pd.concat([pd.read_parquet(file) for file in sorted((OUTPUTS / "test2024").glob("*.parquet"))], ignore_index=True)
    ercot["method"] = ercot.method.str.replace(r"_s\d+$", "", regex=True)
    base = ercot[ercot.m.eq(1)]
    physical = pd.DataFrame([paired(group, "s4_fuel", label, -1 / 1e6) for label, group in base.groupby("method")
                             if group.world.nunique() == 2])
    ercot_breach = pd.DataFrame([breaches(group, label, world) for (label, world), group in base.groupby(["method", "world"])])
    evaluated = base[base.world.eq("realised")].groupby("method")
    totals = evaluated[["fuel_cost", "co2_cost", "capacity_cost", "reward"]].mean() / 1e6
    assert np.allclose(totals.reward, -totals[["fuel_cost", "co2_cost", "capacity_cost"]].sum(axis=1))
    totals.columns = ["fuel cost", "carbon cost", "capacity cost", "reward"]
    operations = evaluated[["s4_fuel", "s4_co2_t", "monthly_net_peak_change_mw"]].mean()
    operations["s4_fuel"] /= 1e6
    operations["s4_co2_t"] /= 1e3
    operations.columns = ["fuel change (million MMBtu)", "CO2 change (kt)", "net peak change (MW)"]
    operations["revenue violations %"] = evaluated.violated.mean() * 100
    heat = ercot[ercot.world.eq("realised") & ercot.method.isin(["mip_B", "mpc_B", "rl_B"])].groupby(["method", "m"])[
        ["d_fuel_mmbtu", "d_co2_t", "violated"]].mean().reset_index()
    heat["d_fuel_mmbtu"] /= 1e6
    heat["d_co2_t"] /= 1e3
    heat["violated"] *= 100
    heat.columns = ["method", "hot-day multiplier", "fuel increase (million MMBtu)", "CO2 increase (kt)", "violations %"]
    monthly = base[base.world.eq("realised")].groupby(["method", "month"])["s4_fuel"].mean().gt(0).groupby(level=0).sum()

    OUT.mkdir(parents=True, exist_ok=True)
    tables = [("london_gaps", london), ("london_limits", london_breach), ("ercot_full_fleet_gaps", physical),
              ("ercot_limits", ercot_breach), ("ercot_reward_breakdown", totals.reset_index()),
              ("ercot_evaluated_operations", operations.reset_index()),
              ("ercot_heat_surrogate", heat), ("ercot_months_with_increases", monthly.reset_index(name="months with added fuel"))]
    for name, table in tables:
        table.to_csv(OUT / (name + ".csv"), index=False)
    text = ["# Revision analysis from recorded simulations", "",
            "Generated by `scripts/report_revision_archive.py` from saved runs. The ERCOT policy test was not rerun. "
            "The point estimates give equal weight to months after averaging over members and seeds. The paired "
            "5,000-draw bootstrap resamples the same month in the two assessment environments and keeps seeds "
            "from a given month together. Its intervals are conditional on the fitted response and gas models. "
            "London outcomes are peak cuts in percent, and gaps are in percentage points. ERCOT paired outcomes "
            "use full-fleet fuel savings in million MMBtu per month. Positive gaps mean greater optimism in design. "
            "Mean constraint excess is over all episodes, including zero excess in compliant episodes. The reward "
            "breakdown is in million 2024 US dollars per month. The hot-day table uses reduced-model fuel and CO2, "
            "with increases positive. Full-fleet estimates were saved only at multiplier one.", ""]
    for name, table in tables:
        text += [f"## {name.replace('_', ' ').capitalize()}", "", md_table(table, floatfmt="{:.3f}"), ""]
    report = "\n".join(text)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "revision_archive.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
