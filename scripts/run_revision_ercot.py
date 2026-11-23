"""Fixed-tariff fuel diagnostics on ERCOT 2023 validation loads.

The tariff sequence is generated once by each archived assumed-response PPO
policy in its own response model. Replay the same sequence with the frozen
point response and specified transfer scenarios through the calibrated fleet.
These are validation-year diagnostics, separate from the archived 2024 test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from evaluate_s3 import S4Eval, ercot_envs
from run_revision_london import point_kernel
from run_s3_baselines import kernel_arrays
from run_s3_rl import FINAL_DIR
from ttrl.env import ercot_env as ee
from ttrl.env import london_env as le
from ttrl.env.responses import TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import harness as hs
from ttrl.s3 import rl


OUT = OUTPUTS / "revision" / "ercot2023"


@dataclass(frozen=True)
class Scenario:
    heat: float = 1.0
    participation: float = 1.0
    level_shift: float = 0.0
    gas: str = "all"


SCENARIOS = {
    "Reference": Scenario(),
    "Hot 0.5": Scenario(heat=0.5),
    "Hot 1.5": Scenario(heat=1.5),
    "Hot 2": Scenario(heat=2.0),
    "Participation 10%": Scenario(participation=0.1),
    "Participation 50%": Scenario(participation=0.5),
    "Level −3.2%": Scenario(level_shift=-0.032),
    "Level +3.2%": Scenario(level_shift=0.032),
    "Gas 50%": Scenario(gas="half"),
    "Hourly gas share": Scenario(gas="observed"),
}


def matched_hours(days: ee.ErcotDays, day_idx: np.ndarray, masked: np.ndarray, effect: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    hours, percentages = [], []
    for position, day in enumerate(day_idx):
        if masked[position]:
            continue
        hourly = ee.hourly_effect(effect[position])
        for grid in (days.row[day], days.row2[day]):
            ok = grid >= 0
            hours.append(grid[ok])
            percentages.append(hourly[ok])
    indices = np.concatenate(hours)
    order = np.argsort(indices)
    return indices[order], np.concatenate(percentages)[order]


def fleet_fuel(fleet: S4Eval, indices: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
    year = fleet.days.year
    y = fleet.model.impact(delta, year.hours[indices], temp_c=fleet.temps[indices], base_mw=year.base_mw[indices]).totals
    return float(y.at["fuel_mmbtu", "corrected"]), float(y.at["co2_tonnes", "corrected"])


def scenarios(seed: int, months_option: list[str] | None) -> None:
    arr = kernel_arrays()
    tr, _ = le.member_split(arr.n_members)
    trial_env, own, months = ercot_envs(arr, tr, S2Settings.load())
    months = months_option or months
    if not set(months).issubset({f"2023-{i:02d}" for i in range(1, 13)}):
        raise ValueError("Use 2023 validation months for the reviewer analysis.")
    fleet = S4Eval(trial_env.days)
    year = fleet.days.year
    point = TrialResponse(point_kernel())
    file = FINAL_DIR / f"ercot_B_all_s{seed}.pt"
    policy = rl.load_agent(file, own)
    OUT.mkdir(parents=True, exist_ok=True)

    for month in months:
        path = OUT / f"ppo_B_s{seed}_{month}.parquet"
        if path.exists():
            continue
        start = time.perf_counter()
        episode = hs.run_month(own, policy, month, 0)
        idx, design_effect = matched_hours(fleet.days, episode["day_idx"], own.masked, episode["effect"])
        design_delta = year.residential[idx] * design_effect / 100
        design_fuel, design_co2 = fleet_fuel(fleet, idx, design_delta)
        raw = point.effect(episode["blocks"])
        hot = fleet.days.temp_day[episode["day_idx"]] > ee.KERNEL_TEMP_MAX
        base = year.residential[idx]
        rows = [{"scenario": "Design projection", "electricity_mwh": design_delta.sum(),
                 "gas_mwh_assigned": design_delta.sum(), "fuel_mmbtu": design_fuel, "co2_t": design_co2}]

        for label, spec in SCENARIOS.items():
            effect = raw * np.where(hot, spec.heat, 1)[:, None]
            candidate_indices, effect = matched_hours(fleet.days, episode["day_idx"], own.masked, effect)
            np.testing.assert_array_equal(candidate_indices, idx)
            # Hypothetical multiplicative shift for participating loads on all tariff hours.
            delta = base * spec.participation * ((1 + effect / 100) * (1 + spec.level_shift) - 1)
            if spec.gas == "half":
                gas = delta * 0.5
            elif spec.gas == "observed":
                gas = delta * np.clip(year.base_mw[idx] / year.demand[idx], 0, 1)
            else:
                gas = delta
            fuel, co2 = fleet_fuel(fleet, idx, gas)
            record = {"scenario": label, "electricity_mwh": delta.sum(), "gas_mwh_assigned": gas.sum(),
                      "fuel_mmbtu": fuel, "co2_t": co2}
            if label == "Reference":
                level_profile = base / base.sum() * delta.sum()
                level, _ = fleet_fuel(fleet, idx, level_profile)
                nonzero_steps = np.abs(delta[delta != 0])
                record |= {"fuel_level_mmbtu": level, "fuel_shape_residual_mmbtu": fuel - level,
                           "below_500_mw_share": float((nonzero_steps < 500).mean()),
                           "below_1000_mw_share": float((nonzero_steps < 1000).mean())}
            rows.append(record)

        frame = pd.DataFrame(rows).assign(month=month, seed=seed, hot_days=int(hot.sum()))
        partial = path.with_suffix(".partial")
        frame.to_parquet(partial, index=False)
        partial.replace(path)
        ref = frame[frame.scenario.eq("Reference")].iloc[0]
        print(f"2023 seed {seed} {month}: {time.perf_counter()-start:.1f}s; reference fuel {ref.fuel_mmbtu:,.0f} MMBtu.", flush=True)


def report(out: Path) -> None:
    data = pd.concat([pd.read_parquet(file) for file in out.glob("*.parquet")], ignore_index=True)
    cols = ["electricity_mwh", "gas_mwh_assigned", "fuel_mmbtu", "co2_t"]
    by_month = data.groupby(["scenario", "month"])[cols].mean()
    stats = by_month.groupby("scenario").mean().reindex(["Design projection", *SCENARIOS])
    stats["months with additional fuel"] = by_month["fuel_mmbtu"].gt(0).groupby(level=0).sum()
    stats["fuel_mmbtu"] /= 1e6
    stats["co2_t"] /= 1000
    stats["electricity_mwh"] /= 1000
    stats["gas_mwh_assigned"] /= 1000
    stats.columns = ["load change GWh", "allocated gas GWh", "fuel change million MMBtu", "CO2 change kt",
                     "months with added fuel"]
    monthly_fuel = by_month["fuel_mmbtu"].unstack("scenario").reindex(columns=stats.index) / 1e6
    rng = np.random.default_rng(20261123)
    draws = rng.integers(0, len(monthly_fuel), size=(5000, len(monthly_fuel)))
    lo, hi = np.quantile(monthly_fuel.to_numpy()[draws].mean(axis=1), (0.025, 0.975), axis=0)
    stats["fuel 2.5%"] = lo
    stats["fuel 97.5%"] = hi
    ref = data[data.scenario.eq("Reference")].groupby("month").mean(numeric_only=True)
    decomp = pd.DataFrame({"total fuel change MMBtu": ref.fuel_mmbtu,
                           "energy-volume component MMBtu": ref.fuel_level_mmbtu,
                           "load-shape residual MMBtu": ref.fuel_shape_residual_mmbtu,
                           "nonzero steps below 0.5 GW %": 100 * ref.below_500_mw_share,
                           "nonzero steps below 1 GW %": 100 * ref.below_1000_mw_share})
    stats.reset_index().to_csv(out / "stress.csv", index=False)
    decomp.to_csv(out / "fuel_decomposition.csv")
    counts = data[data.scenario.eq("Reference")].groupby("seed").month.nunique()
    report_text = (
        "# Revision analysis: 2023 fixed-tariff gas-fleet stress tests\n\n"
        "Generated by `scripts/run_revision_ercot.py` for ERCOT's validation year. Each month and seed "
        "uses the tariff schedule produced by the archived PPO B policy under its design response. "
        "All physical scenarios use the same tariff sequence. Evaluations apply the frozen point kernel, "
        "except Design projection, which uses the assumed elasticity matrix. Price response is scaled on "
        "hot days and by a hypothetical participating share. The level cases multiply participating load "
        "by 0.968 or 1.032. These are diagnostic perturbations of the counterfactual baseline, not "
        "identified price responses. Gas shares allocate part of the load change to the CEMS-calibrated "
        "non-CHP gas fleet; the hourly case uses its observed share of ERCOT demand as an allocation "
        "proxy. Emissions of other resources are outside this calculation. Positive physical changes "
        "are increases from the flat-rate load. Values are conditional on the fixed tariff schedules and "
        "are descriptive stress scenarios. The 95% intervals resample calendar months after "
        "averaging seeds within each month and condition on the assumed scenario and fitted models. "
        "The 2024 test is archived separately.\n\n"
        f"Validation months: {len(ref)}. Seeds with results: {', '.join(map(str, counts.index))}.\n\n"
        + md_table(stats.reset_index(), floatfmt="{:.3f}") + "\n\n"
        "The energy-volume component replays the net monthly MWh change allocated in proportion to the "
        "baseline residential profile. The shape residual includes redistribution across hours and nonlinear "
        "interaction with the volume component. Step shares use nonzero hourly steps, averaged over seed and month.\n\n"
        + md_table(decomp.reset_index(), floatfmt="{:.2f}") + "\n\n"
        + md_table(decomp.mean().to_frame("mean").reset_index(names="metric"), floatfmt="{:.2f}") + "\n"
    )
    (out / "report.md").write_text(report_text, encoding="utf-8")
    if len(counts) == 3 and counts.eq(12).all():
        (ROOT / "docs" / "reports" / "revision_ercot2023.md").write_text(report_text, encoding="utf-8")
    print(report_text.replace("−", "-"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(3), default=0)
    parser.add_argument("--months", nargs="*")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report(OUT)
    else:
        scenarios(args.seed, args.months)


if __name__ == "__main__":
    main()
