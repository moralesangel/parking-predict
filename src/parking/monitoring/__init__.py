"""Monitorizacion: degradacion frente a la persistencia, drift y cobertura.

Las tres senales que pide la Fase 7 del plan, y las tres responden a preguntas
distintas:

- `backtest`  -> el modelo, ¿sigue batiendo a la persistencia sobre datos de hoy?
- `drift`     -> la distribucion de ocupacion, ¿se parece a la de entrenamiento?
- `coverage`  -> ¿sobre cuanta ciudad seguimos teniendo datos?

Un modelo puede ir perfecto en las dos primeras mientras la tercera se hunde,
porque las metricas se calculan sobre los sensores que quedan vivos.
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
