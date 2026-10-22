"""Clasificación de fiabilidad de cada lectura de sensor.

El feed falla de dos maneras distintas, y confundirlas sesga cualquier métrica
de ocupación:

1. **Sensor fuera de línea.** No aparece en el lote actual del feed. Se detecta
   de forma exacta: `lastupdated < max(lastupdated)` del snapshot. Son relés de
   red caídos, y caen en bloque (ver `docs/data_schema.md` §1.2).

2. **Sensor atascado.** Sí aparece en el lote —el relé lo reporta— pero su
   estado no cambia desde hace mucho. El sensor de suelo está congelado. Es el
   fallo más numeroso y el que no se ve si solo se mira `lastupdated`: 1707 de
   5264 plazas en línea, con una mediana de 150 días sin cambiar.

Regla del plan que este módulo respeta de forma estructural: **nunca se borra
nada en silencio**. Todas las funciones *añaden* columnas; ninguna filtra filas.
Quien consuma los datos decide qué hacer con las lecturas no fiables, y la
cobertura se puede medir precisamente porque las filas malas siguen ahí.
"""

from __future__ import annotations

import polars as pl

#: Estados de fiabilidad de una lectura. `pl.Enum` en lugar de texto libre para
#: dejar el conjunto cerrado: castear un valor que no esté aquí lanza
#: `InvalidOperationError`, así que no se puede escribir un estado inventado.
#: No protege del typo al *leer* (`col("quality") == "okay"` devuelve 0 filas en
#: silencio); para eso están estas constantes.
QUALITY_VALUES = ("ok", "stuck", "offline")
QUALITY_DTYPE = pl.Enum(QUALITY_VALUES)

#: Columnas que añade `annotate_quality`.
QUALITY_COLUMNS = ("status_age_hours", "quality", "is_reliable")


def annotate_quality(frame: pl.DataFrame, *, stuck_after_hours: float = 24.0) -> pl.DataFrame:
    """Añade el diagnóstico de calidad a un snapshot.

    Columnas añadidas:

    - `status_age_hours`: horas desde el último cambio de estado, medidas contra
      el sello del lote (`feed_lastupdated`) y no contra la hora actual, para
      que el resultado sea reproducible al reprocesar datos antiguos.
    - `quality`: `offline`, `stuck` u `ok`. El orden importa: una plaza fuera de
      línea se marca como tal aunque además lleve mucho sin cambiar, porque la
      causa raíz es el relé.
    - `is_reliable`: `True` solo para `ok`. Es la columna que deben usar las
      features y las métricas.

    No filtra ninguna fila (regla del plan: nunca borrar en silencio).
    """
    missing = {"lastupdated", "feed_lastupdated", "status_timestamp", "is_live"} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"al snapshot le faltan columnas: {sorted(missing)}")

    age_hours = (
        pl.col("feed_lastupdated") - pl.col("status_timestamp")
    ).dt.total_seconds() / 3600.0

    return (
        frame.with_columns(status_age_hours=age_hours)
        .with_columns(
            quality=pl.when(~pl.col("is_live"))
            .then(pl.lit("offline"))
            .when(pl.col("status_age_hours") > stuck_after_hours)
            .then(pl.lit("stuck"))
            .otherwise(pl.lit("ok"))
            .cast(QUALITY_DTYPE)
        )
        .with_columns(is_reliable=pl.col("quality") == pl.lit("ok").cast(QUALITY_DTYPE))
    )


def quality_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Recuento y porcentaje por estado de calidad, para informes y logs."""
    if "quality" not in frame.columns:
        raise ValueError("llama antes a annotate_quality")

    total = frame.height
    return (
        frame.group_by("quality")
        .agg(bays=pl.len(), occupied=pl.col("is_occupied").sum())
        .with_columns(
            share_pct=(pl.col("bays") / total * 100).round(1),
            occupancy_pct=(pl.col("occupied") / pl.col("bays") * 100).round(1),
        )
        .sort("quality")
    )


def occupancy(frame: pl.DataFrame, *, reliable_only: bool = True) -> float | None:
    """Ocupación media, por defecto solo sobre lecturas fiables.

    `reliable_only=False` existe para poder medir el sesgo de no filtrar, que no
    es constante: las lecturas atascadas tienen una ocupación *congelada*
    (39,3 % en el feed real) que no sigue el ciclo diario, así que el sesgo es
    proporcional a cuánto se aleja la ocupación real de ese valor. Medido sobre
    snapshots individuales: +7,6 puntos a las 10:04 UTC y -0,1 a las 11:46. Es
    decir, el sesgo es peor en hora punta, justo cuando la predicción importa.

    Devuelve `None` si no queda ninguna lectura utilizable, en lugar de 0.0: un
    cero significaría "todo libre", que es exactamente lo contrario de "no se
    sabe".
    """
    subset = frame.filter(pl.col("is_reliable")) if reliable_only else frame
    if subset.height == 0:
        return None
    return float(subset["is_occupied"].mean())
