"""Métricas de cobertura de sensores por zona y por hora.

La cobertura es la fracción de plazas con lectura fiable. Importa por dos
razones distintas:

- **Por zona**, decide dónde se puede predecir. Una zona con 3 de 20 plazas
  fiables no permite estimar "¿hay sitio?" con ninguna confianza, y el producto
  debería decir "no hay datos" en lugar de inventar una probabilidad.
- **Por hora**, detecta patrones de degradación horaria. Si la cobertura cae
  todas las noches, cualquier métrica nocturna está medida sobre otro parque de
  sensores que la diurna, y comparar horas sin corregirlo es un error.
"""

from __future__ import annotations

import polars as pl


def coverage_by_zone(frame: pl.DataFrame, *, min_bays: int = 1) -> pl.DataFrame:
    """Cobertura por `zone_number`.

    Las plazas sin zona (520 de 6324 en el feed real) se agrupan bajo zona nula
    en lugar de descartarse: son plazas reales y su falta de zona es en sí un
    problema de calidad que conviene ver.

    `min_bays` filtra zonas demasiado pequeñas para que el porcentaje signifique
    algo. Filtra el *informe*, no los datos.

    Acepta tanto un snapshot como una serie de varios. `bays` cuenta plazas
    distintas y `readings` cuenta lecturas, que con N snapshots son N veces más:
    confundirlos inflaría el tamaño aparente de cada zona.
    """
    _require_quality(frame)
    return (
        frame.group_by("zone_number")
        .agg(
            bays=pl.col("kerbsideid").n_unique(),
            readings=pl.len(),
            reliable=pl.col("is_reliable").sum(),
            offline=(pl.col("quality") == "offline").sum(),
            stuck=(pl.col("quality") == "stuck").sum(),
        )
        .filter(pl.col("bays") >= min_bays)
        .with_columns(coverage_pct=(pl.col("reliable") / pl.col("readings") * 100).round(1))
        .sort("coverage_pct", "bays")
    )


def coverage_by_hour(frame: pl.DataFrame) -> pl.DataFrame:
    """Cobertura por hora UTC del sello del lote.

    Espera un frame con varios snapshots concatenados (la serie que acumula el
    poller). Con un solo snapshot devuelve una fila, que no dice nada sobre
    patrones horarios: hace falta al menos un día completo para eso.
    """
    _require_quality(frame)
    return (
        frame.with_columns(hour_utc=pl.col("feed_lastupdated").dt.hour())
        .group_by("hour_utc")
        .agg(
            snapshots=pl.col("feed_lastupdated").n_unique(),
            readings=pl.len(),
            reliable=pl.col("is_reliable").sum(),
        )
        .with_columns(coverage_pct=(pl.col("reliable") / pl.col("readings") * 100).round(1))
        .sort("hour_utc")
    )


def flapping_sensors(frame: pl.DataFrame, *, min_snapshots: int = 3) -> pl.DataFrame:
    """Plazas que alternan entre fiable y no fiable a lo largo de la serie.

    Un sensor que entra y sale del lote es peor que uno caído del todo: el caído
    se excluye y se acabó, pero el intermitente mete y saca plazas del
    denominador de cada zona, y eso mueve la ocupación agregada sin que haya
    cambiado ningún coche. Son los candidatos a excluir de las features.
    """
    _require_quality(frame)
    return (
        # El orden temporal es imprescindible: `transitions` compara cada lectura
        # con la anterior, y sobre filas desordenadas contaria cambios ficticios.
        frame.sort("feed_lastupdated")
        .group_by("kerbsideid")
        .agg(
            snapshots=pl.len(),
            reliable=pl.col("is_reliable").sum(),
            transitions=(pl.col("is_reliable") != pl.col("is_reliable").shift(1))
            .sum()
            .fill_null(0),
        )
        .filter((pl.col("snapshots") >= min_snapshots) & (pl.col("transitions") > 0))
        .with_columns(reliable_pct=(pl.col("reliable") / pl.col("snapshots") * 100).round(1))
        .sort("transitions", descending=True)
    )


def _require_quality(frame: pl.DataFrame) -> None:
    if "is_reliable" not in frame.columns:
        raise ValueError("llama antes a parking.quality.flags.annotate_quality")
