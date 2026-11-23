"""Reviewer analysis on London 2013 with the frozen trial response.

Build two responses that match the same two event-band means as the frozen
kernel, then hold the planner, prices and bill and revenue limits fixed.
This uses the historical tariff calendar and the training-sample response
fit, with no calibration on the evaluation members. The ERCOT test is separate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from build_london_env import schedule_on_grid
from run_s3_baselines import kernel_arrays
from ttrl.analysis import lcl_kernel as kn
from ttrl.env import assumed as am
from ttrl.env import fastkernel as fk
from ttrl.env import london as ld
from ttrl.env import london_env as le
from ttrl.env import tariff as tf
from ttrl.env.responses import AssumedResponse, TrialResponse
from ttrl.env.rewards import S2Settings
from ttrl.paths import OUTPUTS, ROOT
from ttrl.reporting import md_table
from ttrl.s3 import harness as hs
from ttrl.s3 import patterns as pt
from ttrl.s3 import planners as pl
from ttrl.s3.baselines import ScheduleReplay


OUT = OUTPUTS / "revision" / "london"
BANDS = (tf.HIGH, tf.LOW)
CASES = ("aalami", "power_matrix", "matched_bands", "matched_matrix", "robust_matched",
         "observed_aalami", "observed_matched_matrix")


@dataclass(frozen=True)
class PowerMatrix:
    matrix: am.ElasticityMatrix

    def effect(self, blocks: np.ndarray) -> np.ndarray:
        prices = tf.slot_prices(blocks).reshape(-1, am.HOURS, 2).mean(axis=2)
        log_change = np.log(prices / self.matrix.p0) @ self.matrix.matrix().T
        return 100 * np.repeat(np.expm1(log_change), 2, axis=1)


@dataclass(frozen=True)
class MatchedBands:
    effects: np.ndarray

    def effect(self, blocks: np.ndarray) -> np.ndarray:
        bands = tf.slot_bands(blocks)
        return np.select([bands == b for b in BANDS], self.effects, default=0.0)


@dataclass(frozen=True)
class MatchedMatrix:
    cross_only: am.ElasticityMatrix
    cross_scale: float
    own_elasticity: np.ndarray

    def effect(self, blocks: np.ndarray) -> np.ndarray:
        bands = tf.slot_bands(blocks)
        own = np.select([bands == b for b in BANDS], self.own_elasticity, default=0.0)
        price_change = tf.slot_prices(blocks) / tf.PRICE[tf.NORMAL] - 1
        return 100 * own * price_change + self.cross_scale * self.cross_only.effect(blocks)


@dataclass(frozen=True)
class ModelEnsemble:
    models: tuple

    @property
    def n_members(self) -> int:
        return len(self.models)

    def effect(self, blocks: np.ndarray, member: int = 0) -> np.ndarray:
        return self.models[int(member)].effect(blocks)


def event_means(effect: np.ndarray, history: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    bands = tf.slot_bands(history)
    return np.array([np.average(effect[bands == b], weights=baseline[bands == b]) for b in BANDS])


def point_kernel() -> fk.KernelArrays:
    spec = kn.KernelSpec.from_json((OUTPUTS / "lcl_kernel_spec.json").read_text(encoding="utf-8"))
    coefficients = pd.read_csv(OUTPUTS / "lcl_kernel_terms.csv").set_index("term")["coef"]
    return fk.from_coefficients(spec, coefficients)


def calibrate(inputs: ld.LondonInputs, history: np.ndarray, trial_days: np.ndarray,
              point: TrialResponse, *, save: bool) -> dict:
    base = inputs.baseline[trial_days]
    matrix = am.aalami_2010(am.categories_by_load(inputs.baseline[inputs.days["period"].eq("pre")]))
    target = event_means(point.effect(history), history, base)
    cross_only = am.ElasticityMatrix(np.zeros_like(matrix.own), matrix.cross, matrix.categories, tf.PRICE[tf.NORMAL])
    cross_means = event_means(cross_only.effect(history), history, base)
    same_sign = target * cross_means > 0
    upper_scale = min(1.0, *np.abs(target[same_sign] / cross_means[same_sign])) if same_sign.any() else 1.0
    cross_scale = upper_scale / 2
    relative_prices = np.array([tf.PRICE[b] / tf.PRICE[tf.NORMAL] - 1 for b in BANDS])
    own_elasticity = (target - cross_scale * cross_means) / (100 * relative_prices)
    if not np.all(own_elasticity < 0):
        raise ValueError(f"Moment matching produced a nonnegative own-price elasticity: {own_elasticity}")
    models = {"aalami": matrix, "power_matrix": PowerMatrix(matrix),
              "matched_bands": MatchedBands(target),
              "matched_matrix": MatchedMatrix(cross_only, cross_scale, own_elasticity)}
    rows = [{"design response": "frozen kernel", "high event %": target[0], "low event %": target[1]}]
    for label, model in models.items():
        high, low = event_means(am.clip_effect(model.effect(history)), history, base)
        rows.append({"design response": label, "high event %": high, "low event %": low})
    np.testing.assert_allclose(event_means(models["matched_matrix"].effect(history), history, base), target)
    log_elasticities = np.log1p(target / 100) / np.log([tf.PRICE[b] / tf.PRICE[tf.NORMAL] for b in BANDS])
    print(md_table(pd.DataFrame(rows), floatfmt="{:.3f}"), flush=True)
    print(f"Historical high and low log elasticities: {log_elasticities}. "
          f"Matrix cross scale: {cross_scale:.6f}. Matrix own elasticities: {own_elasticity}.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if save:
        pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
        pd.DataFrame({"band": ["High", "Low"], "log elasticity": log_elasticities,
                      "matrix own elasticity": own_elasticity,
                      "matrix cross scale": cross_scale}).to_csv(OUT / "elasticities.csv", index=False)
    return models


def event_keys(blocks: np.ndarray) -> list[tuple[int, int, int]]:
    return [(band, (start % tf.HH_PER_DAY) // tf.HH_PER_BLOCK, fk.duration_class(end - start + 1))
            for band, start, end in fk.event_runs(tf.slot_bands(blocks).ravel())]


def support(plans: np.ndarray, history: np.ndarray) -> dict:
    known_patterns = {tuple(pattern) for pattern in history}
    known_events = set(event_keys(history))
    active = [tuple(pattern) for pattern in plans if np.any(pattern != tf.NORMAL)]
    events = event_keys(plans)
    return {"active days": len(active), "historical active-day patterns": sum(p in known_patterns for p in active),
            "events": len(events), "historical event cells": sum(e in known_events for e in events),
            "long high events": sum(e[0] == tf.HIGH and e[2] == 2 for e in events)}


def check_schedules(inputs: ld.LondonInputs, history: np.ndarray, point: TrialResponse, matrix: am.ElasticityMatrix,
                    months: list[str]) -> None:
    from evaluate_s3 import london_envs
    from run_s3_rl import FINAL_DIR, run_name
    from ttrl.s3 import rl

    settings = S2Settings.load()
    _, assumed_env, _, fold_of = london_envs(point.arrays, [0], settings)
    config = le.LondonConfig.from_settings(settings, months, [0])
    trial = le.LondonTariffEnv(inputs, point, config, seed=0)
    cases = [("A", "Trial point"), ("B", "Trial point"), ("B", "Design")]
    rows = []
    for agent, world in cases:
        for fold in sorted(set(fold_of.values())):
            env = assumed_env(fold) if world == "Design" else trial
            for seed in range(3):
                file = FINAL_DIR / (run_name("london", agent, fold, seed, False) + ".pt")
                policy = rl.load_agent(file, env)
                for month in (m for m in months if fold_of[m] == fold):
                    episode = hs.run_month(env, policy, month, 0)
                    rows.append({"policy": f"PPO {agent}", "world": world, "seed": seed, "month": month,
                                 **support(episode["blocks"], history)})
    coverage = pd.DataFrame(rows).groupby(["policy", "world"], sort=False)
    sums = coverage[["active days", "historical active-day patterns", "events", "historical event cells", "long high events"]].sum()
    sums["exact active days %"] = 100 * sums["historical active-day patterns"] / sums["active days"]
    sums["matched event cells %"] = 100 * sums["historical event cells"] / sums["events"]
    sums.to_csv(OUT / "ppo_support.csv")

    rows = []
    for slope in (-0.02, -0.05, -0.10, -0.15):
        response = AssumedResponse(am.ElasticityMatrix(np.full_like(matrix.own, slope), matrix.cross, matrix.categories, matrix.p0))
        env = le.LondonTariffEnv(inputs, response, config, seed=0)
        for month in months:
            blocks = pd.read_csv(OUT / f"aalami_{month}.csv").to_numpy()
            days = inputs.days[inputs.days.month.eq(month)].tday
            replay = ScheduleReplay(dict(zip(map(pd.Timestamp, days), blocks)), name="Fixed imported-matrix schedule")
            episode = hs.run_month(env, replay, month, 0)
            rows.append({"own elasticity": slope, "peak cut %": episode["daily_peak_cut_pct"],
                         "revenue ratio": episode["revenue_ratio"], "revenue breaches %": 100 * (episode["cost_revenue"] > 1e-12)})
    pd.DataFrame(rows).groupby("own elasticity", sort=False).mean().to_csv(OUT / "fixed_slope_sweep.csv")


def solve_case(case: str, inputs: ld.LondonInputs, history: np.ndarray, point: TrialResponse, models: dict,
               months: list[str], time_limit: float) -> None:
    arr = kernel_arrays()
    tr, ev = le.member_split(arr.n_members)
    cfg = le.LondonConfig.from_settings(S2Settings.load(), months, tr)
    trial = le.LondonTariffEnv(inputs, TrialResponse(arr), cfg, seed=0)
    patterns = pt.candidate_patterns()
    if case.startswith("observed_"):
        observed = {tuple(pattern) for pattern in history}
        patterns = patterns[[tuple(pattern) in observed for pattern in patterns]]
    if case == "robust_matched":
        own_models = (point, AssumedResponse(models["matched_bands"]), AssumedResponse(models["matched_matrix"]))
        own_response = ModelEnsemble(own_models)
        effects = np.concatenate((pt.pattern_effects(patterns, point.arrays),
                                  *(m.effect(patterns)[None] for m in own_models[1:])))
    else:
        own_response = AssumedResponse(models[case.removeprefix("observed_")])
        effects = own_response.effect(patterns)[None]
    own = le.LondonTariffEnv(inputs, own_response, cfg, seed=0)
    planner = pl.MonthlyMIP(effects, patterns, time_limit=time_limit)

    for month in months:
        path = OUT / f"{case}_{month}.parquet"
        if path.exists():
            continue
        timer = time.perf_counter()
        design = hs.evaluate(own, planner, [month], np.arange(own_response.n_members)).assign(world="design")
        evaluated = hs.evaluate(trial, planner, [month], ev).assign(world="evaluation")
        info = planner.log[-1]
        design = pd.concat([design, evaluated], ignore_index=True).drop(columns=["blocks", "effect", "day_idx"])
        aux = support(planner.cache[month], history)
        for name, value in aux.items():
            design[name] = value
        design = design.assign(case=case, solver_status=info["status"], objective=info["objective"], mip_gap=info["mip_gap"])
        pd.DataFrame(planner.cache[month], columns=tf.BLOCK_LABELS).to_csv(path.with_suffix(".csv"), index=False)
        partial = path.with_suffix(".partial")
        design.to_parquet(partial, index=False)
        partial.replace(path)
        print(f"{case} {month}: {time.perf_counter()-timer:.1f}s, status={info['status']}, "
              f"gap={info['mip_gap']}, d/e peak cut={design.iloc[:own_response.n_members].daily_peak_cut_pct.mean():.3f}/"
              f"{evaluated.daily_peak_cut_pct.mean():.3f}%", flush=True)


def report(out: Path) -> None:
    data = pd.concat([pd.read_parquet(file) for file in out.glob("*.parquet")], ignore_index=True)
    rows = []
    stat = []
    for case, group in data.groupby("case", sort=False):
        by_month = group.groupby(["world", "month"])["daily_peak_cut_pct"].mean().unstack("world")
        design = group[group.world.eq("design")]
        evaluated = group[group.world.eq("evaluation")]
        diagnostics = design.drop_duplicates("month")
        found = diagnostics[diagnostics.objective.notna()]
        def coverage(num: str, den: str) -> float:
            total = found[den].sum()
            return 100 * found[num].sum() / total if total else np.nan
        stat.append({"response": case, "months": len(diagnostics), "incumbents": len(found),
                     "proved infeasible": int(diagnostics.solver_status.eq(2).sum()),
                     "at limit": int(diagnostics.solver_status.eq(1).sum()),
                     "maximum solver gap %": 100 * found.mip_gap.max(),
                     "historical active-day patterns %": coverage("historical active-day patterns", "active days"),
                     "historical event cells %": coverage("historical event cells", "events"),
                     "long high events": int(found["long high events"].sum())})
        by_month = by_month.reindex(found.month)
        if by_month.empty:
            continue
        gaps = (by_month.design - by_month.evaluation).to_numpy()
        rng = np.random.default_rng(20261123)
        indices = rng.integers(0, len(gaps), (5000, len(gaps)))
        samples = gaps[indices].mean(axis=1)
        left, right = np.quantile(samples, [0.025, 0.975])
        eval_left, eval_right = np.quantile(by_month.evaluation.to_numpy()[indices].mean(axis=1), [0.025, 0.975])
        rows.append({"response": case, "incumbent months": len(gaps), "design cut %": by_month.design.mean(),
                     "evaluation cut %": by_month.evaluation.mean(), "gap pp": gaps.mean(),
                     "gap 2.5%": left, "gap 97.5%": right,
                     "evaluation 2.5%": eval_left, "evaluation 97.5%": eval_right,
                     "design violations %": 100 * design[design.month.isin(found.month)].violated.mean(),
                     "evaluation violations %": 100 * evaluated[evaluated.month.isin(found.month)].violated.mean()})
    result = pd.DataFrame(rows)
    result.to_csv(out / "comparisons.csv", index=False)
    diagnostics = pd.DataFrame(stat)
    diagnostics.to_csv(out / "schedule_support.csv", index=False)
    caption = (
        "# Revision analysis: matched response models in London\n\n"
        "Generated by `scripts/run_revision_london.py`. The models match baseline-load-weighted event means "
        "under the historical tariff calendar from the frozen fit on training households. The two-band "
        "model is equivalent to a two-slope log-linear model on the three price levels. The matched matrix "
        "retains the signs and proportions of cross elasticities, normalises the response at the Normal rate, "
        "and fits separate own-price terms for High and Low. The cross scale is half the lesser of one "
        "and the largest scale permitting both own-price terms to remain negative. "
        "The power matrix preserves the original elasticities and flat-rate reference in log-price form. "
        "The robust design gives equal weight to the trial point estimate and both matched models and "
        "checks revenue in all three. The London bill cap, revenue band, action set and evaluation ensemble "
        "are shared in the full-menu comparisons. The two observed-menu checks restrict the action set "
        "to recorded tariff-day patterns. The MIP has perfect monthly load information.\n\n"
        "The paired gap bootstrap resamples calendar months. Results refer to months with a solver incumbent "
        "and can be compared directly across methods only over shared months. Support uses recorded active "
        "tariff-day patterns and (band, starting block, duration class) event cells. These are diagnostics of "
        "calendar overlap and do not establish exposure to the whole intervened sequence. The PPO diagnostic "
        "uses saved policies, the frozen point response for evaluation, and the three archived seeds. The "
        "elasticity sweep keeps the reproduced imported-model monthly schedules and all off-diagonal coefficients fixed.\n\n"
    )
    calibration = pd.read_csv(out / "calibration.csv")
    elasticities = pd.read_csv(out / "elasticities.csv")
    report_tables = [calibration, elasticities, result, diagnostics]
    for source in ("ppo_support.csv", "fixed_slope_sweep.csv"):
        file = out / source
        if file.exists():
            report_tables.append(pd.read_csv(file))
    report_text = caption + "\n\n".join(md_table(table, floatfmt="{:.3f}") for table in report_tables) + "\n"
    (out / "report.md").write_text(report_text, encoding="utf-8")
    if all(diagnostics.set_index("response")["months"].reindex(CASES).eq(12)):
        (ROOT / "docs" / "reports" / "revision_london.md").write_text(report_text, encoding="utf-8")
    print(report_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--months", nargs="*")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--schedule-checks", action="store_true")
    args = parser.parse_args()
    if args.report:
        report(OUT)
        return
    inputs = ld.LondonInputs.load()
    trial = inputs.days["period"].eq("trial").to_numpy()
    history, _ = schedule_on_grid(pd.DatetimeIndex(inputs.days.loc[trial, "tday"]))
    point = TrialResponse(point_kernel())
    models = calibrate(inputs, history, trial, point, save=args.case is None)
    months = args.months or sorted(inputs.days.loc[trial, "month"].unique())
    if args.case:
        solve_case(args.case, inputs, history, point, models, months, args.time_limit)
    if args.schedule_checks:
        check_schedules(inputs, history, point, models["aalami"], months)


if __name__ == "__main__":
    main()
