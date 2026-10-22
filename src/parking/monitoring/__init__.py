"""Monitoring: degradation against persistence, drift and sensor coverage.

The three signals phase 7 of the plan asks for, each answering a different
question:

- `backtest`  -> is the model still beating persistence on today's data?
- `drift`     -> does the occupancy distribution still look like training?
- `coverage`  -> how much of the city do we still have data for?

A system can look perfect on the first two while the third collapses, because
the metrics are computed over whichever sensors are still alive.
"""

from parking.monitoring.backtest import BacktestResult, backtest, coverage_trend, realised_targets
from parking.monitoring.drift import (
    DriftResult,
    occupancy_drift,
    population_stability_index,
)

__all__ = [
    "BacktestResult",
    "DriftResult",
    "backtest",
    "coverage_trend",
    "occupancy_drift",
    "population_stability_index",
    "realised_targets",
]
