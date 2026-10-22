"""Backtest del modelo en producción contra lo que realmente pasó.

Es la pieza de monitorización que de verdad responde "¿sigue funcionando?".
Las métricas de la Fase 5 se midieron sobre 2019; esto las vuelve a medir sobre
los datos que el poller va acumulando, que es el único sitio donde el modelo se
enfrenta al presente.

Funciona porque **la etiqueta llega sola**: la predicción hecha en `t` para
`t+h` se puede comprobar en cuanto el poller capture `t+h`. No hace falta
anotar nada a mano, solo esperar `h` minutos.

Se compara siempre contra la persistencia calculada sobre los mismos datos. Un
Brier de 0,03 no significa nada por sí solo; lo que importa es si sigue por
debajo de la referencia que habría costado cero esfuerzo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from parking.models.evaluation import brier_score, log_loss, skill_score


@dataclass(frozen=True)
class BacktestResult:
    """Métricas del modelo frente a la persistencia sobre datos ya realizados."""

    horizon_minutes: int
    n: int
    base_rate: float
    model_brier: float
    persistence_brier: float
    model_log_loss: float

    @property
    def skill_vs_persistence(self) -> float:
        return skill_score(self.model_brier, self.persistence_brier)

    @property
    def is_degraded(self) -> bool:
        """True si el modelo ya no bate a la persistencia.

        Es la alarma que importa: el momento en que el pipeline de ML deja de
        pagar lo que cuesta.
        """
        return self.model_brier >= self.persistence_brier

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "n": self.n,
            "base_rate": round(self.base_rate, 4),
            "model_brier": round(self.model_brier, 5),
            "persistence_brier": round(self.persistence_brier, 5),
            "model_log_loss": round(self.model_log_loss, 5),
            "skill_vs_persistence": round(self.skill_vs_persistence, 4),
            "is_degraded": self.is_degraded,
        }


def realised_targets(
    features: pl.DataFrame, horizon_minutes: int, *, grid_minutes: int = 5
) -> pl.DataFrame:
    """Añade el objetivo observado: ¿había sitio `h` minutos después?

    Es exactamente `dataset.add_targets`, pero aplicado a la serie en vivo y a
    posteriori. Las filas cuyo `t+h` todavía no ha ocurrido quedan con objetivo
    nulo y se descartan al evaluar: son el futuro, no un dato que falte.
    """
    steps = horizon_minutes // grid_minutes
    return features.sort("segment_id", "timestamp").with_columns(
        realised_free_bays=pl.col("free_bays").shift(-steps).over("segment_id"),
        realised_target=(pl.col("free_bays").shift(-steps).over("segment_id") >= 1),
    )


def backtest(
    features: pl.DataFrame,
    probabilities: np.ndarray,
    horizon_minutes: int,
    *,
    grid_minutes: int = 5,
    exclude_interpolated: bool = True,
) -> BacktestResult | None:
    """Evalúa unas predicciones contra lo que realmente ocurrió.

    `features` debe ser la serie completa (no solo la última fila por tramo) y
    `probabilities` las predicciones alineadas con ella.

    `exclude_interpolated` descarta las filas que el relleno de huecos inventó.
    Evaluar sobre ellas mediría la calidad del relleno, no la del modelo: una
    fila arrastrada del instante anterior hace que la persistencia parezca
    perfecta y que el modelo parezca peor de lo que es.

    Devuelve `None` si no queda nada evaluable, en lugar de métricas sobre cero
    filas.
    """
    frame = features.with_columns(probability=pl.Series(probabilities))
    frame = realised_targets(frame, horizon_minutes, grid_minutes=grid_minutes)
    frame = frame.filter(pl.col("realised_target").is_not_null())
    if exclude_interpolated and "is_interpolated" in frame.columns:
        frame = frame.filter(~pl.col("is_interpolated"))
    if frame.is_empty():
        return None

    y_true = frame["realised_target"].to_numpy().astype(np.float64)
    model = frame["probability"].to_numpy()
    persistence = (frame["free_bays"].to_numpy() >= 1).astype(np.float64)

    return BacktestResult(
        horizon_minutes=horizon_minutes,
        n=frame.height,
        base_rate=float(np.mean(y_true)),
        model_brier=brier_score(y_true, model),
        persistence_brier=brier_score(y_true, persistence),
        model_log_loss=log_loss(y_true, model),
    )


def coverage_trend(series: pl.DataFrame, *, bucket: str = "1h") -> pl.DataFrame:
    """Cobertura de sensores a lo largo del tiempo.

    Es la tercera señal que pide el plan y la más fácil de pasar por alto: un
    modelo puede seguir perfecto mientras la mitad de los sensores se apagan,
    porque las métricas se calculan sobre lo que queda. La cobertura cayendo
    significa que el producto cubre menos ciudad, aunque acierte igual.
    """
    if series.is_empty():
        return series
    return (
        series.with_columns(window=pl.col("feed_lastupdated").dt.truncate(bucket))
        .group_by("window")
        .agg(
            bays=pl.len(),
            reliable=pl.col("is_reliable").sum(),
            batches=pl.col("feed_lastupdated").n_unique(),
        )
        .with_columns(coverage_pct=(pl.col("reliable") / pl.col("bays") * 100).round(1))
        .sort("window")
    )
