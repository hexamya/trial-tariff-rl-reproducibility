"""Reward weights and constraint caps of the S2 environments, fixed before S3 training.

Built by ``scripts/build_s2_settings.py`` into ``data/processed/s2_settings.json``; the report
``docs/reports/s2_settings.md`` lists each input and its source. Money is in 2024 US dollars
(BEA GDP implicit price deflator, FRED series
A191RD3A086NBEA).

ERCOT reward (``ttrl.env.ercot_env``): minus the cost of the gas fuel, CO2 and capacity that a
tariff changes,

* fuel: delivered price of natural gas to the Texas electric power sector, by calendar month of 2024
  (EIA fuel receipts aggregates via PUDL), the prices the test-year fleet paid;
* CO2: social cost of CO2, Rennert et al. (2022, Nature 610, 687-692): mean $185 per tonne in 2020 US$
  at a 2% near-term discount rate; the peer-reviewed estimate, independent of US policy (Executive
  Order 14154 of January 2025 disbanded the Interagency Working Group and withdrew its estimates as no
  longer representative of government policy; CRS IF12916, 20 February 2025);
* capacity: the annualised cost of a new industrial-frame combustion turbine in the ERCOT region (EIA,
  Assumptions to AEO2026, Electricity Market Module, April 2026: Table 4 overnight cost in region TRE,
  Table 3 fixed O&M and lead time, discount rate of the "New construction financing" section),
  charged on the change of each month's peak of net load (demand minus wind and solar) times the
  month's share of the capacity value: the share of the 1% highest net-load hours of 2019-2023 that
  fall in that calendar month (a probability-of-peak allocation; pre-test years only).

London constraints (``ttrl.env.london_env``): revenue within +/-3% of the flat tariff and the mean bill
change of each household segment at most +3%, both as in Kale et al.; segments are the quartiles of the
households' pre-trial evening (17-23 h) consumption share, bills those of households that do not respond
(response-independent, so the same for agents trained on any response). A first choice, a cap on the
household CVaR95 at the 2013 schedule's annual value (4.41%), could not be met together with monthly
revenue neutrality.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ttrl.paths import PROCESSED

SETTINGS_PATH = PROCESSED / "s2_settings.json"
DOLLAR_YEAR = 2024

# Valuation inputs from the original source documents.
SC_CO2_USD_2020 = 185.0                 # Rennert et al. 2022, abstract: preferred mean, 2020 US$, 2% discount
CT_OVERNIGHT_TRE_USD_2025 = 1033.0      # EIA AEO2026 EMM Table 4, CT--industrial frame, 1 TRE, 2025$/kW
CT_FOM_USD_2025 = 7.18                  # EIA AEO2026 EMM Table 3, fixed O&M, 2025$/kW-yr
CT_LEAD_YEARS = 2                       # EIA AEO2026 EMM Table 3, lead time
EIA_DISCOUNT_RATE = 0.071               # EIA AEO2026 EMM p. 9: 60/40 debt/equity, 2025-2050; no adder for CTs
RECOVERY_YEARS = 30                     # cost recovery period (an ATB 2024 option); 20 years as a sensitivity
PEAK_HOUR_SHARE = 0.01                  # top 1% of net-load hours per year carry the capacity value
WEIGHT_YEARS = (2019, 2020, 2021, 2022, 2023)
LONDON_REVENUE_TOL = 0.03               # Kale et al.: revenue within +/-3% of flat-normal
LONDON_SEGMENT_BILL_CAP = 0.03          # Kale et al.: segment expected bills at most +3%


def crf(rate: float, years: int) -> float:
    """Capital recovery factor."""
    g = (1 + rate) ** years
    return rate * g / (g - 1)


def idc_factor(rate: float, lead_years: int) -> float:
    """Interest during construction: spending spread evenly over the lead time (mid-year), compounded to
    the start of operation."""
    return sum((1 + rate) ** (lead_years - k - 0.5) for k in range(lead_years)) / lead_years


def capacity_value_per_kw_yr(overnight: float = CT_OVERNIGHT_TRE_USD_2025, fom: float = CT_FOM_USD_2025,
                             rate: float = EIA_DISCOUNT_RATE, years: int = RECOVERY_YEARS,
                             lead: int = CT_LEAD_YEARS) -> float:
    """Annualised cost of new peaking capacity, in the dollar year of the inputs ($/kW-yr)."""
    return overnight * idc_factor(rate, lead) * crf(rate, years) + fom


@dataclass
class S2Settings:
    dollar_year: int
    gas_price_by_month: dict[int, float]        # $/MMBtu by calendar month
    co2_price: float                            # $/t CO2
    capacity_value: float                       # $ per MW-yr of capacity-relevant peak
    peak_month_weight: dict[int, float]         # share of the capacity value per calendar month (sums to 1)
    london_revenue_tol: float
    london_segment_bill_cap: float
    sensitivity: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "S2Settings":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["gas_price_by_month"] = {int(k): float(v) for k, v in d["gas_price_by_month"].items()}
        d["peak_month_weight"] = {int(k): float(v) for k, v in d["peak_month_weight"].items()}
        return cls(**d)
