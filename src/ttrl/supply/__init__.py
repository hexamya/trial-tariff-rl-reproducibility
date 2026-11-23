"""Supply side of stage S4: ERCOT gas-fleet unit curves and dispatch.

- :mod:`ttrl.supply.units`: per-unit heat-input curves H(P), output limits and
  CO2 factors fitted on EPA CEMS, behind an interface that can later take
  ambient temperature.
- :mod:`ttrl.supply.dispatch`: merit-order commitment and equal-incremental
  heat-rate dispatch of the fleet for a given hourly output target.
- :mod:`ttrl.supply.fleet`: actual CEMS operation as (hours, units) arrays.
- :mod:`ttrl.supply.heatrate`: temperature-dependent heat input H(P, T) and
  capacity derating Pavail(T) per unit.
- :mod:`ttrl.supply.fleetmodel`: the dispatch model with temperature-free or
  temperature-aware curves (and the plant temperature of every unit-hour).
- :mod:`ttrl.supply.errorband`: the S4 error band of the model's marginal fuel,
  CO2 and CT response, from out-of-sample CEMS hour pairs.
- :mod:`ttrl.supply.impact`: the function the tariff environment calls: fuel,
  CO2, CT hours and fleet heat rate of a profile of load changes, with a band.
"""
