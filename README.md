# Dynamic electricity tariff reproducibility package

This package accompanies *Peak and fuel impacts of dynamic electricity tariffs under trial-estimated and assumed demand response* by Hesam Jafari and Hadi Sahebi. It contains analysis code for the Low Carbon London household response, the London tariff-planning comparison and the ERCOT gas-fleet assessment.

## Rebuild the reported comparisons

Use Python 3.13. The pinned direct dependencies describe the environment used to check this release.

```sh
python -m venv .venv
```

Activate `.venv` in your shell, then install the dependencies and the local package.

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Extract `trial-tariff-rl-reproducibility-archived-results-v1.0.0.zip` from the same release into this directory. The archive preserves the `outputs/` paths that the report scripts read. Its `ARCHIVED-RESULTS.sha256` lists the checksums of the supplied files.

```sh
python -X utf8 scripts/report_revision_archive.py
python -X utf8 scripts/run_revision_london.py --report
python -X utf8 scripts/run_revision_ercot.py --report
python -X utf8 scripts/run_s3_test2024.py --report
python -X utf8 scripts/run_s3_sensitivity.py --report
python -m pytest -q tests
```

The first command reconstructs the archived appraisal gaps, violations and reward breakdowns. The next two reconstruct the new London response comparisons and the 2023 gas-fleet stress tests. `--report` reads recorded simulation outcomes. The primary 2024 comparison uses the frozen evaluation archive. The paired month intervals reflect variation across the supplied months under the fitted response and fleet models. The integration tests for the upstream workbook, household index and plant-temperature cache run when those inputs are available.

## Rebuild the models from public inputs

The analysis scripts provide the upstream pipeline. The London meter panel is large and its queries use DuckDB. Keep public upstream downloads in `data/raw/`. `curl` is needed for the London meter download. The full planning and policy-training pipeline is substantially slower than rebuilding the report tables.

```sh
python scripts/fetch_lcl_features.py
python scripts/fetch_lcl_meters.py
python scripts/build_lcl_panel.py
python scripts/build_lcl_split.py
python scripts/estimate_lcl_did.py
python scripts/check_lcl_controls.py
python scripts/estimate_lcl_kernel.py
python scripts/build_london_env.py
python scripts/run_s4.py
python scripts/build_ercot_weather_years.py
python scripts/build_ercot_residential.py
python scripts/build_ercot_env.py
python scripts/build_s2_settings.py
```

`scripts/run_s3_planners.py`, `scripts/run_s3_planners_assumed.py` and `scripts/launch_s3_rl.py` supply the planning and training entry points. `scripts/evaluate_s3.py` evaluates the saved policies on their validation months. `scripts/run_revision_london.py --case` regenerates a planning case from the frozen kernel and `--schedule-checks` computes the additional exposure checks from saved policy weights. `scripts/run_revision_ercot.py --seed` regenerates the fixed-tariff 2023 stress scenarios. Time-limited MIP incumbents can vary across machines and runs. Use the archived incumbents when comparing the paper's exact tabulated results.

## Sources and citation

- UK Power Networks supplied the Low Carbon London meter data and tariff calendar under Creative Commons Attribution. The derived response fits use the 2013 dynamic time-of-use trial. [London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
- ERCOT load, generation and program inputs use EIA-930, EIA-861 and the PUDL `v2026.9.0` data snapshot. The gas-fleet inputs use EIA generator characteristics and EPA CEMS. [EIA grid monitor](https://www.eia.gov/electricity/gridmonitor/) · [EIA-861](https://www.eia.gov/electricity/data/eia861/) · [PUDL CEMS documentation](https://docs.catalyst.coop/pudl/en/latest/data_sources/epacems.html)
- Hourly residential load and weather use ResStock 2024 Release 2 and NOAA Integrated Surface Database observations. [ResStock data](https://resstock.nlr.gov/datasets) · [NOAA Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)
- The constrained policy optimisation follows Schulman et al. (2017), Ray, Achiam and Amodei (2019), and Stooke, Achiam and Abbeel (2020). The implementation centres constraint advantages and uses a PID update for the multipliers. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) · [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods](https://proceedings.mlr.press/v119/stooke20a.html)

Use `CITATION.cff` to cite the software and retain upstream dataset attribution. The analysis code is licensed under MIT.
