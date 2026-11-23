"""Fuel, CO2, CT-hour and heat-rate effect of a profile of load changes (S4 -> RL).

This is the function the tariff environment calls. Given an hourly profile
of load changes (MW, + or -) with the hours' temperatures, :meth:`S4Model.impact`
re-dispatches the ERCOT non-CHP gas fleet with and without the change and
returns

- delta fuel (MMBtu), delta CO2 (short tons, as CEMS reports, and tonnes),
- delta simple-cycle CT operating hours (unit-hours with output > 0),
- delta fleet heat rate (non-CHP fleet fuel / gross output over the hours),

each with a band. The band comes from :class:`ttrl.supply.errorband.ErrorBand`:
the validated error of the model's marginal fuel, CO2 and CT response by
stratum (season x load tercile x hot hour) and the share of the model's
hour-level variation that CEMS confirms, as block-bootstrap draws. For each
draw the hour's change becomes the CEMS stratum marginal plus the confirmed
share of the model's deviation from its stratum mean, times the gas-output
change (:meth:`ttrl.supply.errorband.ErrorBand.correct`). Bands with a
step-size part also need the model's +/- 500 MW marginal in each hour (two
more dispatch runs): the deviation is split into the hour-level part, scaled
by beta, and the part that depends on the size of the change (curvature,
commitment jumps), scaled by the share CEMS confirms for that size. The band therefore
carries the model's systematic error in the strata the profile touches, the
noise in its hour-level answers and the sampling error of the CEMS
benchmark. ``corrected`` is the median over draws. Results from two calls
that use the same band are draw-aligned, so :func:`compare` gives the band
of their difference (e.g. two tariffs).

Assumptions (see the S4 reports): the load change is met by the non-CHP gas
fleet in proportion ``gas_share`` (default 1: gas is marginal); the rest of
the change, and its fuel, is outside the model. The change is applied to
gross gas output (no loss or auxiliary-load factor). Each hour is dispatched
independently (no start-up fuel, ramp limits or multi-hour commitment).

Hours inside the model's CEMS year (2024) default to that year's actual
non-CHP output as the baseline, the actual plant temperatures and the actual
monthly unit availability. Other hours need ``base_mw`` and ``temp_c``; they
use the 2024 availability of the same local month and, in configuration
C-sx, the 2024 mean output of the held suspect CC units for that month and
hour of day.

Usage::

    from ttrl.supply.impact import S4Model, shift_profile
    model = S4Model.load()                       # temperature-aware, 2024
    prof = shift_profile(["2024-08-20"], 500.0)  # 16-19 h -> 22-01 h local
    res = model.impact(prof.to_numpy(), prof.index)
    res.totals                                   # estimate, corrected, lo, hi
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ttrl.data import pudl
from ttrl.paths import INTERIM, PROCESSED
from ttrl.supply import dispatch as D
from ttrl.supply.errorband import ErrorBand
from ttrl.supply.fleetmodel import FleetData, FleetModel, build_fleet_data
from ttrl.supply.units import LST_OFFSET

TZ = "America/Chicago"
KG_PER_SHORT_TON = 907.18474
TONNE_PER_SHORT_TON = KG_PER_SHORT_TON / 1000.0
METRICS = ["fuel_mmbtu", "co2_short_tons", "co2_tonnes", "ct_unit_hours",
           "fleet_heat_rate_mmbtu_per_mwh", "fleet_heat_rate_pct"]


def model_spec_path(year: int = 2024) -> Path:
    """JSON written by ``scripts/integrate_s4.py`` that describes the S4 model."""
    return PROCESSED / f"s4_model_{year}.json"


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
@dataclass
class ImpactResult:
    """Effect of a load-change profile.

    ``totals``: one row per metric of :data:`METRICS` with ``estimate`` (the
    model), ``corrected`` (median over band draws), ``lo``/``hi`` (band) and
    ``unit``. ``hourly``: per hour the change of gas output, temperature,
    stratum, base output and the model's change of fuel, CO2, CT units on
    and units on in total (dispatched units).
    ``draws``: (n_draws, metrics) corrected totals, aligned across calls that
    use the same band. ``meta``: model, band, gas share, feasibility counts.
    """
    totals: pd.DataFrame
    hourly: pd.DataFrame
    draws: pd.DataFrame
    meta: dict

    def __getitem__(self, metric: str) -> float:
        return float(self.totals.at[metric, "estimate"])

    def interval(self, metric: str) -> tuple[float, float]:
        """(lo, hi) of the band for ``metric``."""
        return float(self.totals.at[metric, "lo"]), float(self.totals.at[metric, "hi"])


def _totals(est: dict, draws: pd.DataFrame, level: float) -> pd.DataFrame:
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    units = {"fuel_mmbtu": "MMBtu", "co2_short_tons": "short tons", "co2_tonnes": "t",
             "ct_unit_hours": "unit-hours", "fleet_heat_rate_mmbtu_per_mwh": "MMBtu/MWh",
             "fleet_heat_rate_pct": "%"}
    rows = {m: {"estimate": est[m], "corrected": float(np.median(draws[m])),
                "lo": float(np.quantile(draws[m], lo_q)), "hi": float(np.quantile(draws[m], hi_q)),
                "unit": units[m]} for m in METRICS}
    return pd.DataFrame.from_dict(rows, orient="index")


def compare(a: ImpactResult, b: ImpactResult, level: float = 0.95) -> pd.DataFrame:
    """Difference a - b of two results, with the band of the difference.

    Both must come from the same model and band (draw-aligned), e.g. two
    tariffs evaluated by one :class:`S4Model`.
    """
    if a.meta.get("band_id") != b.meta.get("band_id") or len(a.draws) != len(b.draws):
        raise ValueError("results use different error bands; their draws are not aligned")
    est = {m: a.totals.at[m, "estimate"] - b.totals.at[m, "estimate"] for m in METRICS}
    return _totals(est, a.draws - b.draws, level)


# --------------------------------------------------------------------------
# Profiles in local time
# --------------------------------------------------------------------------
def to_utc_hours(hours) -> pd.DatetimeIndex:
    """tz-naive UTC DatetimeIndex (tz-aware input is converted; naive input is taken as UTC)."""
    idx = pd.DatetimeIndex(hours)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("us")


def local_hours(dates, start_hour: int, end_hour: int, tz: str = TZ) -> pd.DatetimeIndex:
    """UTC hour-beginning stamps of the local clock hours [start_hour, end_hour) of each date.

    ``end_hour <= start_hour`` wraps past midnight (22 -> 1 gives 22:00,
    23:00 and 00:00 of the next day). Ambiguous or missing local hours (DST
    changes) raise.
    """
    n = (end_hour - start_hour) % 24 or 24
    stamps = [pd.Timestamp(pd.Timestamp(d).date()) + pd.Timedelta(hours=start_hour + k)
              for d in dates for k in range(n)]
    loc = pd.DatetimeIndex(stamps).tz_localize(tz, ambiguous="raise", nonexistent="raise")
    return to_utc_hours(loc)


def shift_profile(dates, mw: float, from_hours: tuple[int, int] = (16, 19),
                  to_hours: tuple[int, int] = (22, 1), tz: str = TZ) -> pd.Series:
    """Load shift: -``mw`` in local hours ``from_hours`` and +``mw`` in ``to_hours`` of each date.

    Returns a Series of MW indexed by UTC hour (sorted; hours in both
    windows are summed). Energy is conserved when both windows have the same
    length.
    """
    out = pd.concat([pd.Series(-float(mw), index=local_hours(dates, *from_hours, tz=tz)),
                     pd.Series(float(mw), index=local_hours(dates, *to_hours, tz=tz))])
    return out.groupby(level=0).sum().sort_index()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class S4Model:
    """S4 gas-fleet model behind the tariff environment.

    Holds the fleet's 2024 operation (:class:`FleetData`), a fitted
    :class:`FleetModel` (curves + headroom) and the :class:`ErrorBand` of
    that model. Build it with :meth:`load` after ``scripts/integrate_s4.py``
    has run. The P15-weighted mean plant temperature defines the fleet
    temperature used for the band's strata (hot hour: >= 30 C).
    """

    def __init__(self, data: FleetData, model: FleetModel, band: ErrorBand, meta: dict | None = None):
        if list(model.table["unit_id"]) != list(data.unit_ids):
            raise ValueError("curve table rows must follow FleetData.unit_ids")
        self.data, self.model, self.band = data, model, band
        self.meta = dict(meta or {})
        self.weights = model.table["p15"].to_numpy(float)
        ft = data.fleet_temp(self.weights)
        self.temp_offset = (data.temp - ft[:, None]).mean(axis=0)       # (N,) plant minus fleet
        m = data.month - 1
        self.avail_month = np.stack([data.on[m == k].any(axis=0) for k in range(12)])
        exo = ~model.disp
        key = m * 24 + data.hour_lst
        cnt = np.bincount(key, minlength=288)
        self.exo_energy_mh = np.bincount(key, weights=data.energy[:, exo].sum(axis=1),
                                         minlength=288) / np.maximum(cnt, 1)
        self.exo_heat_mh = np.bincount(key, weights=data.heat[:, exo].sum(axis=1),
                                       minlength=288) / np.maximum(cnt, 1)

    @classmethod
    def load(cls, year: int = 2024, mode: str = "temperature", band: str = "cv",
             spec_path: str | Path | None = None) -> "S4Model":
        """Model and band written by ``scripts/integrate_s4.py``.

        ``mode``: "temperature" (main) or "static" (the temperature-free
        comparison model). ``band``: "cv" (12-fold month-blocked
        cross-validation, all seasons) or "holdout" (Jan-Aug fit, Sep-Dec).
        """
        spec = json.loads(Path(spec_path or model_spec_path(year)).read_text())
        table = pd.read_parquet(PROCESSED / spec["curves"])
        fm = FleetModel(table, mode=mode, config=spec["config"],
                        headroom=tuple(spec["headroom"][mode]))
        eb = ErrorBand.load(PROCESSED / spec["bands"][mode][band])
        paths = pudl.interim_paths(year)
        data = build_fleet_data(pd.read_parquet(paths["cems"]), pd.read_parquet(paths["units"]),
                                pd.read_parquet(INTERIM / f"plant_temperature_{year}.parquet"), year)
        return cls(data, fm, eb, meta={"year": year, "mode": mode, "band": band,
                                       "config": spec["config"], "spec": spec})

    # -- inputs ------------------------------------------------------------
    def plant_temperatures(self, hours: pd.DatetimeIndex, temp_c=None) -> np.ndarray:
        """(H, N) plant temperatures for ``hours``.

        ``temp_c`` None: the actual plant temperatures (hours must lie in the
        model year). (H, N): used as given. (H,): a fleet temperature; each
        plant keeps its 2024 deviation from the fleet in that hour (model-year
        hours) or its mean 2024 deviation (other hours).
        """
        pos = self.data.hours.get_indexer(hours)
        if temp_c is None:
            if (pos < 0).any():
                raise ValueError("temp_c is required for hours outside the model year")
            return self.data.temp[pos]
        tc = np.asarray(temp_c, float)
        if tc.ndim == 2:
            if tc.shape != (len(hours), len(self.data.unit_ids)):
                raise ValueError("temp_c must be (hours,) or (hours, units)")
            return tc
        pattern = np.where((pos >= 0)[:, None], self.data.temp[np.maximum(pos, 0)],
                           self.temp_offset[None, :])
        pft = (pattern * self.weights).sum(axis=1) / self.weights.sum()
        return pattern + (tc - pft)[:, None]

    # -- the function ------------------------------------------------------
    def impact(self, delta_mw, hours, temp_c=None, base_mw=None, gas_share: float = 1.0,
               level: float = 0.95) -> ImpactResult:
        """Effect of load changes ``delta_mw`` (MW, one value per hour of ``hours``).

        Parameters
        ----------
        delta_mw : (H,) load change in each hour, MW (= MWh), + more load.
        hours : (H,) hour-beginning timestamps; tz-aware, or tz-naive UTC.
        temp_c : None, (H,) fleet temperature or (H, N) plant temperatures, C
            (see :meth:`plant_temperatures`).
        base_mw : (H,) baseline non-CHP gas gross output, MW; default the
            model year's actual output (hours must lie in that year).
        gas_share : share of the load change met by the non-CHP gas fleet.
        level : coverage of the band (default 95%).
        """
        hours = to_utc_hours(hours)
        delta = np.asarray(delta_mw, float).reshape(-1)
        if len(delta) != len(hours):
            raise ValueError("delta_mw and hours must have the same length")
        d, fm = self.data, self.model
        pos = d.hours.get_indexer(hours)
        inside = pos >= 0
        p = np.maximum(pos, 0)
        temp = self.plant_temperatures(hours, temp_c)
        ft = (temp * self.weights).sum(axis=1) / self.weights.sum()
        if base_mw is None:
            if not inside.all():
                raise ValueError("base_mw is required for hours outside the model year")
            base = d.target[pos]
        else:
            base = np.asarray(base_mw, float).reshape(-1)
        lst = hours - LST_OFFSET
        month = lst.month.to_numpy()
        mh = (month - 1) * 24 + lst.hour.to_numpy()
        avail = np.where(inside[:, None], d.avail[p], self.avail_month[month - 1])
        exo = ~fm.disp
        exo_e = np.where(inside, d.energy[p][:, exo].sum(axis=1), self.exo_energy_mh[mh])
        exo_h = np.where(inside, d.heat[p][:, exo].sum(axis=1), self.exo_heat_mh[mh])
        target = np.maximum(base - exo_e, 0.0)
        d_gas = gas_share * delta
        new = np.maximum(target + d_gas, 0.0)
        r0 = fm.run(target, temp, avail)
        r1 = fm.run(new, temp, avail)
        d_gas_eff = new - target                     # after the floor at zero output
        dF, dC = r1.heat - r0.heat, r1.co2 - r0.co2
        dN = (r1.n_ct - r0.n_ct).astype(float)

        st = self.band.stratum(ft, base, month)
        m_hour = None
        if self.band.beta2 is not None:
            mg = fm.marginal(target, temp, avail, float(self.band.meta.get("delta_mw", 500.0)))
            m_hour = np.column_stack([mg["fuel"], mg["co2"], mg["ct"]])
        corr = self.band.correct(np.column_stack([dF, dC, dN]), d_gas_eff, st, m_hour)   # (B, 3)
        f0 = r0.heat.sum() + exo_h.sum()
        e0 = (target + exo_e).sum()
        e1 = e0 + d_gas_eff.sum()
        hr0 = f0 / e0

        def hr_change(dfuel):
            return (f0 + dfuel) / e1 - hr0
        est = {"fuel_mmbtu": dF.sum(), "co2_short_tons": dC.sum(),
               "co2_tonnes": dC.sum() * TONNE_PER_SHORT_TON, "ct_unit_hours": dN.sum(),
               "fleet_heat_rate_mmbtu_per_mwh": hr_change(dF.sum())}
        est["fleet_heat_rate_pct"] = 100 * est["fleet_heat_rate_mmbtu_per_mwh"] / hr0
        fuel_b, co2_b = corr[:, 0], corr[:, 1]
        draws = pd.DataFrame({"fuel_mmbtu": fuel_b, "co2_short_tons": co2_b,
                              "co2_tonnes": co2_b * TONNE_PER_SHORT_TON,
                              "ct_unit_hours": corr[:, 2],
                              "fleet_heat_rate_mmbtu_per_mwh": hr_change(fuel_b)})
        draws["fleet_heat_rate_pct"] = 100 * draws["fleet_heat_rate_mmbtu_per_mwh"] / hr0
        labels = np.array(self.band.labels)
        hourly = pd.DataFrame({
            "hour_utc": hours, "delta_mw": delta, "d_gas_mwh": d_gas_eff, "fleet_temp_c": ft,
            "base_mw": base, "stratum": labels[st], "d_fuel_mmbtu": dF,
            "d_co2_short_tons": dC, "d_ct_units": dN,
            "d_units_on": (r1.p > 0).sum(axis=1) - (r0.p > 0).sum(axis=1),
            "model_marginal_fuel": np.nan if m_hour is None else m_hour[:, 0],
            "status_base": [D.STATUS_NAMES[int(s)] for s in r0.status],
            "status_new": [D.STATUS_NAMES[int(s)] for s in r1.status]})
        bad = (r0.status != D.STATUS_OK) | (r1.status != D.STATUS_OK)
        meta = {**{k: v for k, v in self.meta.items() if k != "spec"},
                "band_id": self.band.meta.get("id"), "gas_share": gas_share, "level": level,
                "hours": len(hours), "hours_outside_model_year": int((~inside).sum()),
                "hours_not_solved": int(bad.sum()),
                "hours_clipped_at_zero": int((target + d_gas < 0).sum()),
                "base_fleet_heat_rate": hr0, "base_fuel_mmbtu": f0, "base_output_mwh": e0}
        return ImpactResult(_totals(est, draws, level), hourly, draws, meta)


# --------------------------------------------------------------------------
# Share of a demand change met by the gas fleet (EIA-930)
# --------------------------------------------------------------------------
def gas_response_share(data: FleetData, lag_hours: int = 1, hac_lags: int = 24) -> tuple[float, float]:
    """Change of non-CHP CEMS gas output per MWh change of ERCO demand.

    OLS of the hour-to-hour change of the non-CHP gross output on the
    changes of EIA-930 ERCO demand, wind and solar (the method of
    ``scripts/validate_dispatch.py``); the EIA-930 stamp marks the end of
    the hour, hence ``lag_hours`` = 1. Returns (coefficient, HAC SE).
    """
    import statsmodels.api as sm
    from ttrl.data import eia930

    e = eia930.load_ba_hourly(data.year).set_index("datetime_utc").sort_index()
    h = e.reindex(data.hours + pd.Timedelta(hours=lag_hours))
    df = pd.DataFrame({"y": data.target, "demand": h["demand"].to_numpy(),
                       "wind": h["wind"].to_numpy(), "solar": h["solar"].to_numpy(),
                       "gas": h["gas"].to_numpy()}).diff()
    df = df.dropna(subset=["y", "demand", "wind", "solar", "gas"])
    r = sm.OLS(df["y"], sm.add_constant(df[["demand", "wind", "solar"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return float(r.params["demand"]), float(r.bse["demand"])
