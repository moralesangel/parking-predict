"""Construcción del dataset de entrenamiento.

Define el problema de predicción de la Fase 4:

> dado el estado de un tramo en `t`, ¿cuál es la probabilidad de que haya **al
> menos una plaza libre** en `t + h`, con `h` ∈ {5, 10, 15, 30} minutos?

Decisiones que condicionan todo lo demás:

- **El objetivo se mide en `t + h`, no en `t`.** Suena obvio y es el punto del
  proyecto: la referencia a batir es la persistencia, que responde con el
  estado actual.
- **Un objetivo por horizonte**, en columnas separadas (`target_h5`, `target_h10`…).
  Permite entrenar y evaluar cada horizonte por separado, que es como pide
  desglosar los resultados la Fase 5.
- **También se guarda `free_bays` futuro** (`free_bays_h5`…). Es la alternativa
  que el plan manda evaluar: predecir el número de plazas y derivar de ahí la
  probabilidad, en vez de clasificar directamente.
- **Ninguna feature mira al futuro.** Todas se calculan con información
  disponible en `t`. Es el error que infla las métricas sin que nada falle.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

#: Horizontes de predicción en minutos, según el plan.
DEFAULT_HORIZONS = (5, 10, 15, 30)

#: Retardos en minutos para las features de historia reciente.
DEFAULT_LAGS = (5, 10, 15, 30, 60)

#: Festivos de Victoria (Australia) en 2019, que es el estado de Melbourne.
#: Tomados del registro público de festivos victorianos. **Conviene
#: verificarlos contra la fuente oficial antes de sacar conclusiones de la
#: Fase 5**: un festivo mal puesto no rompe nada, solo mete ruido en la feature.
#: Los de fecha móvil son los que más riesgo tienen (el viernes previo a la
#: final de la AFL y la Melbourne Cup).
VICTORIA_HOLIDAYS_2019: tuple[dt.date, ...] = (
    dt.date(2019, 1, 1),  # Año nuevo
    dt.date(2019, 1, 28),  # Australia Day (trasladado: el 26 cayó en sábado)
    dt.date(2019, 3, 11),  # Labour Day
    dt.date(2019, 4, 19),  # Viernes Santo
    dt.date(2019, 4, 20),  # Sábado de Pascua
    dt.date(2019, 4, 21),  # Domingo de Pascua
    dt.date(2019, 4, 22),  # Lunes de Pascua
    dt.date(2019, 4, 25),  # ANZAC Day
    dt.date(2019, 6, 10),  # Queen's Birthday
    dt.date(2019, 9, 27),  # Viernes previo a la final de la AFL
    dt.date(2019, 11, 5),  # Melbourne Cup
    dt.date(2019, 12, 25),  # Navidad
    dt.date(2019, 12, 26),  # Boxing Day
)


@dataclass(frozen=True)
class TemporalSplit:
    """Partición temporal del dataset.

    Nunca aleatoria. Un split aleatorio pondría en entrenamiento el instante
    `t+5` y en test el `t`, que están correlacionadísimos: el modelo parecería
    excelente y sería incapaz de predecir el futuro real. Con series temporales
    el único corte honesto es por tiempo.
    """

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    train_end: dt.datetime
    validation_end: dt.datetime

    def summary(self) -> pl.DataFrame:
        rows = []
        for name, frame in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            rows.append(
                {
                    "split": name,
                    "rows": frame.height,
                    "segments": frame["segment_id"].n_unique() if frame.height else 0,
                    "first": frame["timestamp"].min() if frame.height else None,
                    "last": frame["timestamp"].max() if frame.height else None,
                }
            )
        return pl.DataFrame(rows)


def add_time_features(frame: pl.DataFrame, *, holidays: tuple[dt.date, ...] = ()) -> pl.DataFrame:
    """Features de calendario.

    `minute_of_day` va como seno y coseno además de crudo: la medianoche está a
    un minuto de las 23:59, y un entero plano le dice al modelo que están a
    1439 de distancia. Los árboles pueden aprender el salto partiendo el rango,
    pero no gratis.
    """
    minute_of_day = pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
    angle = 2 * 3.141592653589793 * minute_of_day / 1440
    return frame.with_columns(
        hour=pl.col("timestamp").dt.hour().cast(pl.Int8),
        minute_of_day=minute_of_day.cast(pl.Int16),
        minute_sin=angle.sin(),
        minute_cos=angle.cos(),
        day_of_week=pl.col("timestamp").dt.weekday().cast(pl.Int8),  # 1 = lunes
        is_weekend=(pl.col("timestamp").dt.weekday() >= 6),
        is_holiday=pl.col("timestamp").dt.date().is_in(list(holidays)),
    )


def add_lag_features(
    frame: pl.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    grid_minutes: int = 5,
) -> pl.DataFrame:
    """Historia reciente de cada tramo: retardos, tendencias y medias móviles.

    Los retardos se expresan en minutos y se traducen a número de filas usando
    `grid_minutes`. Eso **asume una rejilla sin huecos**, que es lo que produce
    `occupancy.occupancy_series`: ahí cada tramo tiene una fila por instante,
    incluso sin eventos. Sobre una rejilla con huecos, `shift` saltaría a un
    instante arbitrario y la feature mentiría en silencio.
    """
    expressions: list[pl.Expr] = []
    for lag in lags:
        steps = lag // grid_minutes
        expressions.append(
            pl.col("free_bays").shift(steps).over("segment_id").alias(f"free_bays_lag{lag}")
        )
        expressions.append(
            pl.col("occupancy_rate")
            .shift(steps)
            .over("segment_id")
            .alias(f"occupancy_rate_lag{lag}")
        )

    frame = frame.with_columns(expressions)

    # Tendencias: cuanto ha cambiado respecto a hace N minutos. Un tramo que se
    # esta vaciando y otro que se esta llenando pueden tener la misma ocupacion
    # ahora y futuros opuestos.
    trends = [
        (pl.col("free_bays") - pl.col(f"free_bays_lag{lag}")).alias(f"free_bays_trend{lag}")
        for lag in lags
    ]
    rolling = [
        pl.col("free_bays")
        .rolling_mean(window_size=window // grid_minutes, min_samples=1)
        .over("segment_id")
        .alias(f"free_bays_mean{window}")
        for window in (15, 30, 60)
    ]
    return frame.with_columns(trends + rolling)


def add_neighbour_features(
    frame: pl.DataFrame, adjacency: dict[str, frozenset[str]]
) -> pl.DataFrame:
    """Ocupación agregada de los tramos vecinos en el mismo instante.

    Es información legítima en `t`: el estado de la manzana de al lado se
    conoce ahora, no en el futuro. Y es la que distingue "esta zona está llena
    pero la de al lado vacía" de "todo el barrio está lleno", que llevan a
    decisiones distintas.

    Los tramos sin vecinos conocidos reciben nulos, no ceros: un cero diría
    "los vecinos están llenos", que es una afirmación, no una ausencia.
    """
    pairs = [
        {"segment_id": segment, "neighbour_id": neighbour}
        for segment, neighbours in adjacency.items()
        for neighbour in neighbours
    ]
    if not pairs:
        return frame.with_columns(
            neighbour_free_bays=pl.lit(None, pl.Int64),
            neighbour_occupancy_rate=pl.lit(None, pl.Float64),
            neighbour_count=pl.lit(0, pl.UInt32),
        )

    edges = pl.DataFrame(pairs)
    neighbour_state = frame.select(
        neighbour_id="segment_id",
        timestamp="timestamp",
        neighbour_free="free_bays",
        neighbour_rate="occupancy_rate",
    )
    aggregated = (
        edges.join(neighbour_state, on="neighbour_id", how="inner")
        .group_by("segment_id", "timestamp")
        .agg(
            neighbour_free_bays=pl.col("neighbour_free").sum(),
            neighbour_occupancy_rate=pl.col("neighbour_rate").mean(),
            neighbour_count=pl.len(),
        )
    )
    return frame.join(aggregated, on=["segment_id", "timestamp"], how="left")


def add_targets(
    frame: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    grid_minutes: int = 5,
) -> pl.DataFrame:
    """Objetivos por horizonte: plazas libres en `t+h` y si hay al menos una.

    Se guardan ambos. `target_hN` es la clasificación que pide el plan;
    `free_bays_hN` permite la alternativa de predecir el recuento y derivar la
    probabilidad, que la Fase 5 debe comparar.

    Las últimas filas de cada tramo quedan con objetivo nulo porque su futuro
    cae fuera de la ventana. Se marcan, no se rellenan: imputarlas inventaría
    etiquetas.
    """
    expressions: list[pl.Expr] = []
    for horizon in horizons:
        steps = horizon // grid_minutes
        future = pl.col("free_bays").shift(-steps).over("segment_id")
        expressions.append(future.alias(f"free_bays_h{horizon}"))
        expressions.append((future >= 1).alias(f"target_h{horizon}"))
    return frame.with_columns(expressions)


def build_dataset(
    occupancy: pl.DataFrame,
    adjacency: dict[str, frozenset[str]],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    grid_minutes: int = 5,
    holidays: tuple[dt.date, ...] = VICTORIA_HOLIDAYS_2019,
) -> pl.DataFrame:
    """Ensambla el dataset completo a partir de la serie de ocupación.

    El orden importa: primero `occupancy_rate`, del que dependen los retardos y
    las features de vecinos; después los objetivos, que miran al futuro y deben
    construirse sobre la rejilla ya completa.
    """
    frame = occupancy.sort("segment_id", "timestamp").with_columns(
        occupancy_rate=pl.when(pl.col("total_bays") > 0)
        .then(pl.col("occupied_bays") / pl.col("total_bays"))
        .otherwise(None)
    )
    frame = add_time_features(frame, holidays=holidays)
    frame = add_lag_features(frame, lags=lags, grid_minutes=grid_minutes)
    frame = add_neighbour_features(frame, adjacency)
    frame = add_targets(frame, horizons=horizons, grid_minutes=grid_minutes)
    return frame.sort("segment_id", "timestamp")


def temporal_split(
    frame: pl.DataFrame, *, train_end: dt.datetime, validation_end: dt.datetime
) -> TemporalSplit:
    """Parte el dataset por tiempo. Nunca al azar.

    Los cortes son por `timestamp`, de modo que cada fila de validación y de
    test es posterior a todo el entrenamiento.

    Aviso que la Fase 5 debe tener presente: con horizonte `h`, la etiqueta de
    la última fila de entrenamiento procede del instante `train_end + h`, que
    cae ya en validación. Es una fuga de `h` minutos en la frontera. Con
    `h ≤ 30 min` sobre meses de datos es despreciable, pero si se acortan los
    splits hay que descartar las últimas `h` filas de cada bloque.
    """
    if validation_end <= train_end:
        raise ValueError("validation_end debe ser posterior a train_end")

    return TemporalSplit(
        train=frame.filter(pl.col("timestamp") < train_end),
        validation=frame.filter(
            (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < validation_end)
        ),
        test=frame.filter(pl.col("timestamp") >= validation_end),
        train_end=train_end,
        validation_end=validation_end,
    )


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Columnas utilizables como entrada del modelo.

    Excluye los identificadores, el instante y **todas** las columnas de
    objetivo. Que esta lista se derive de los datos y no se escriba a mano es lo
    que evita que una columna futura se cuele como feature al añadir un
    horizonte nuevo.
    """
    excluded = {"segment_id", "timestamp"}
    return [
        name
        for name in frame.columns
        if name not in excluded
        and not name.startswith("target_h")
        and not name.startswith("free_bays_h")
    ]
