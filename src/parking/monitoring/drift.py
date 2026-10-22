"""Detección de drift en la distribución de ocupación.

El modelo se entrenó con una distribución concreta. Si la de hoy se aleja, las
probabilidades dejan de estar calibradas aunque el modelo siga ordenando bien,
y en este producto la calibración **es** el entregable.

Se usa el **PSI** (Population Stability Index), que es el estándar operativo
para esto: compara dos histogramas sobre los mismos cortes y resume la
diferencia en un número.

    PSI = suma_i (actual_i - referencia_i) * ln(actual_i / referencia_i)

Umbrales habituales en la industria, que se adoptan aquí tal cual:

| PSI | Lectura |
|---|---|
| < 0,10 | estable |
| 0,10 a 0,25 | cambio moderado, vigilar |
| > 0,25 | cambio fuerte, revisar el modelo |

Son convención, no una verdad estadística: no hay un contraste de hipótesis
detrás. Sirven para disparar una revisión, no para concluir nada por sí solos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

#: Cortes del histograma de ocupación. Fijos y no cuantiles de la muestra
#: actual: con cortes móviles, dos distribuciones distintas podrían dar PSI
#: cero porque cada una define sus propios tramos.
OCCUPANCY_BINS = (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0001)

STABLE_BELOW = 0.10
INVESTIGATE_ABOVE = 0.25

#: Evita el log de cero cuando un tramo queda vacío en una de las dos muestras.
_EPSILON = 1e-6


@dataclass(frozen=True)
class DriftResult:
    """Comparación de dos distribuciones."""

    psi: float
    reference_n: int
    current_n: int
    bins: pl.DataFrame

    @property
    def level(self) -> str:
        if self.psi < STABLE_BELOW:
            return "estable"
        if self.psi < INVESTIGATE_ABOVE:
            return "moderado"
        return "fuerte"

    @property
    def needs_attention(self) -> bool:
        return self.psi >= STABLE_BELOW


def _histogram(values: np.ndarray, bins: tuple[float, ...]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=list(bins))
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), _EPSILON)
    return np.maximum(counts / total, _EPSILON)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: tuple[float, ...] = OCCUPANCY_BINS,
) -> DriftResult:
    """PSI entre una muestra de referencia y una actual.

    `reference` suele ser la distribución de entrenamiento y `current` la de la
    ventana reciente.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    reference_share = _histogram(reference, bins)
    current_share = _histogram(current, bins)
    contributions = (current_share - reference_share) * np.log(current_share / reference_share)

    detail = pl.DataFrame(
        {
            "bin_low": list(bins[:-1]),
            "bin_high": [min(edge, 1.0) for edge in bins[1:]],
            "reference_pct": (reference_share * 100).round(2),
            "current_pct": (current_share * 100).round(2),
            "contribution": contributions.round(5),
        }
    )
    return DriftResult(
        psi=float(contributions.sum()),
        reference_n=int(reference.size),
        current_n=int(current.size),
        bins=detail,
    )


def occupancy_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    column: str = "occupancy_rate",
) -> DriftResult:
    """Drift de la tasa de ocupación entre dos conjuntos.

    Se compara la **ocupación** y no la variable objetivo porque es la señal que
    domina el modelo: el 90 % de la ganancia sale del estado actual y su media
    reciente (ver `docs/results.md` §5). Si esa distribución se mueve, el modelo
    se ha quedado sin suelo.
    """
    for frame, name in ((reference, "referencia"), (current, "actual")):
        if column not in frame.columns:
            raise ValueError(f"falta la columna {column!r} en el conjunto de {name}")
    return population_stability_index(
        reference[column].drop_nulls().to_numpy(),
        current[column].drop_nulls().to_numpy(),
    )
