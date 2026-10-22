"""Baselines obligatorios, en el orden que fija el plan.

El proyecto entero existe para responder a una pregunta: ¿sirve de algo predecir
si ya hay datos en tiempo real? La respuesta honesta exige una referencia dura,
y esa referencia es la **persistencia**: contestar con el estado actual.

Se implementan tres, de menos a más exigente:

1. `PersistenceBaseline` — la del plan, literal: si ahora hay sitio, habrá sitio.
2. `CalibratedPersistence` — la misma idea pero devolviendo la probabilidad
   empírica de que el estado se mantenga. Es la referencia *justa*, y la
   difícil de batir.
3. `HistoricalMeanBaseline` — media histórica por tramo, día de la semana y
   franja horaria.

Todos exponen la misma interfaz: `fit(frame, target)` y `predict_proba(frame)`,
que devuelve P(hay al menos una plaza libre en `t+h`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

#: Las probabilidades se recortan antes de calcular log loss. Un baseline que
#: dice 0 o 1 y se equivoca una sola vez daría log loss infinita, y comparar
#: infinitos no informa de nada. El recorte es parte de la evaluación, no una
#: corrección del modelo: se aplica igual a todos para que la comparación sea
#: justa, y se documenta porque penaliza a los baselines duros.
PROBABILITY_EPSILON = 1e-6


def clip_probabilities(values: np.ndarray, epsilon: float = PROBABILITY_EPSILON) -> np.ndarray:
    """Lleva las probabilidades a `[epsilon, 1-epsilon]`."""
    return np.clip(values, epsilon, 1.0 - epsilon)


@dataclass
class PersistenceBaseline:
    """El estado en `t+h` es el estado actual. La referencia del plan.

    Devuelve 1,0 si el tramo tiene ahora al menos una plaza libre y 0,0 si no.
    Es determinista: no aprende nada y `fit` no hace nada.

    Es durísima de batir a horizontes cortos, y ése es el punto. Si a 5 minutos
    nadie la mejora, el resultado válido es decirlo.
    """

    def fit(self, frame: pl.DataFrame, target: str) -> PersistenceBaseline:
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        return (frame["free_bays"].to_numpy() >= 1).astype(np.float64)


@dataclass
class CalibratedPersistence:
    """Persistencia que devuelve la probabilidad empírica de que el estado dure.

    En vez de contestar 0 o 1, contesta "de las veces que un tramo tenía sitio,
    ¿en qué fracción seguía habiéndolo `h` minutos después?". Esa fracción se
    estima **solo con los datos de entrenamiento**.

    Es la referencia justa: tiene la misma información que la persistencia cruda
    —el estado actual, nada más— pero expresada como probabilidad, así que no la
    castiga el recorte del log loss. Un modelo que no bata a ésta no está
    aportando nada que no estuviera ya en el estado actual.
    """

    rates: dict[bool, float] = field(default_factory=dict)
    fallback: float = 0.5

    def fit(self, frame: pl.DataFrame, target: str) -> CalibratedPersistence:
        usable = frame.filter(pl.col(target).is_not_null())
        self.fallback = float(usable[target].mean())
        grouped = (
            usable.with_columns(has_space=pl.col("free_bays") >= 1)
            .group_by("has_space")
            .agg(rate=pl.col(target).mean())
        )
        self.rates = {
            bool(row["has_space"]): float(row["rate"]) for row in grouped.iter_rows(named=True)
        }
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        has_space = frame["free_bays"].to_numpy() >= 1
        with_space = self.rates.get(True, self.fallback)
        without = self.rates.get(False, self.fallback)
        return np.where(has_space, with_space, without)


@dataclass
class HistoricalMeanBaseline:
    """Media histórica por tramo, día de la semana y franja horaria.

    Captura el patrón regular —los martes a las 11 esta calle suele estar
    llena— pero **ignora por completo el estado actual**. Es el contrapunto
    exacto de la persistencia, que solo usa el estado actual y ningún patrón.
    Comparar las dos dice cuánta de la señal es rutina y cuánta es la situación
    del momento.

    `bucket_minutes` agrupa la hora en franjas. Con franjas de 60 min y un año
    de datos, cada combinación (tramo, día, franja) tiene ~52 observaciones;
    bajar a 15 min las dejaría en ~13 y la media sería ruido.

    Las combinaciones no vistas en entrenamiento caen por una jerarquía de
    respaldo —tramo+franja, luego tramo, luego global— en lugar de a un 0,5
    arbitrario. Un tramo que siempre está lleno no tiene un 50 % de
    probabilidad solo porque falte su martes.
    """

    bucket_minutes: int = 60
    by_segment_day_bucket: dict[tuple[str, int, int], float] = field(default_factory=dict)
    by_segment_bucket: dict[tuple[str, int], float] = field(default_factory=dict)
    by_segment: dict[str, float] = field(default_factory=dict)
    global_rate: float = 0.5

    def _with_bucket(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            bucket=(pl.col("minute_of_day") // self.bucket_minutes).cast(pl.Int32)
        )

    def fit(self, frame: pl.DataFrame, target: str) -> HistoricalMeanBaseline:
        usable = self._with_bucket(frame.filter(pl.col(target).is_not_null()))
        self.global_rate = float(usable[target].mean())

        finest = usable.group_by("segment_id", "day_of_week", "bucket").agg(
            rate=pl.col(target).mean()
        )
        self.by_segment_day_bucket = {
            (row["segment_id"], int(row["day_of_week"]), int(row["bucket"])): float(row["rate"])
            for row in finest.iter_rows(named=True)
        }
        coarse = usable.group_by("segment_id", "bucket").agg(rate=pl.col(target).mean())
        self.by_segment_bucket = {
            (row["segment_id"], int(row["bucket"])): float(row["rate"])
            for row in coarse.iter_rows(named=True)
        }
        per_segment = usable.group_by("segment_id").agg(rate=pl.col(target).mean())
        self.by_segment = {
            row["segment_id"]: float(row["rate"]) for row in per_segment.iter_rows(named=True)
        }
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        prepared = self._with_bucket(frame)
        segments = prepared["segment_id"].to_list()
        days = prepared["day_of_week"].to_list()
        buckets = prepared["bucket"].to_list()

        predictions = np.empty(len(segments), dtype=np.float64)
        for index, (segment, day, bucket) in enumerate(zip(segments, days, buckets, strict=True)):
            value = self.by_segment_day_bucket.get((segment, int(day), int(bucket)))
            if value is None:
                value = self.by_segment_bucket.get((segment, int(bucket)))
            if value is None:
                value = self.by_segment.get(segment)
            predictions[index] = self.global_rate if value is None else value
        return predictions
