"""Features de tramo a partir de los snapshots del poller.

El modelo se entrenó sobre el histórico de 2019, cuyo esquema es de intervalos.
El tiempo real es otra cosa: una foto del estado cada 2 minutos. Este módulo
produce, a partir de esas fotos, **exactamente las mismas columnas** que
`parking.features.dataset`, para que el modelo entrenado pueda puntuarlas.

Que eso sea posible depende de un detalle de la Fase 4: `segment_id` **no es
una feature**. El modelo solo mira ocupación, historia reciente, vecinos y
calendario, así que puntúa igual un tramo que no existía en 2019. De los 278
tramos vivos solo 130 aparecen en el dataset de entrenamiento, y aun así los
278 son puntuables.

El puente entre ambos mundos es `kerbsideid → segment_id`, vía el dataset de
geometría (`docs/id_crosswalk.md`): cubre el 84,9 % de las plazas vivas.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import polars as pl

from parking.features.dataset import (
    DEFAULT_LAGS,
    VICTORIA_HOLIDAYS_2019,
    add_lag_features,
    add_neighbour_features,
    add_time_features,
)
from parking.features.segments import build_adjacency, parse_segment_id, segment_id
from parking.ingest.crosswalk import segment_key_from_description

logger = logging.getLogger(__name__)

#: Paso de la rejilla en vivo. El feed publica cada 2 min, pero el modelo se
#: entrenó con retardos expresados sobre una rejilla de 5, así que la serie se
#: remuestrea a 5 para que `free_bays_lag5` signifique lo mismo en ambos lados.
LIVE_GRID_MINUTES = 5


def load_bay_segments(path: Path) -> dict[int, str]:
    """Mapa `kerbsideid → segment_id` desde el export de geometría.

    Se descarta cualquier plaza cuya descripción de tramo no encaje en el
    patrón `X between A and B`: sin tramo no hay dónde agregarla, y meterla en
    uno aproximado contaminaría el recuento de ese tramo.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    for row in rows:
        kerbsideid = row.get("kerbsideid")
        if kerbsideid is None or not str(kerbsideid).isdigit():
            continue
        key = segment_key_from_description(row.get("roadsegmentdescription"))
        if key is not None:
            mapping[int(kerbsideid)] = segment_id(key)
    return mapping


def load_bay_coordinates(path: Path) -> pl.DataFrame:
    """Centroide de cada tramo, para poder buscar por latitud y longitud."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        key = segment_key_from_description(row.get("roadsegmentdescription"))
        if key is None or row.get("latitude") is None or row.get("longitude") is None:
            continue
        records.append(
            {
                "segment_id": segment_id(key),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
        )
    if not records:
        return pl.DataFrame(
            schema={"segment_id": pl.String, "latitude": pl.Float64, "longitude": pl.Float64}
        )
    return (
        pl.DataFrame(records)
        .group_by("segment_id")
        .agg(
            latitude=pl.col("latitude").mean(),
            longitude=pl.col("longitude").mean(),
            bays=pl.len(),
        )
    )


def segment_occupancy(series: pl.DataFrame, bay_segments: dict[int, str]) -> pl.DataFrame:
    """Agrega los snapshots por tramo e instante.

    Solo cuentan las lecturas **fiables**. Es la regla de la Fase 2 aplicada
    aquí: una plaza con el sensor congelado no es una plaza libre, y sumarla al
    denominador inflaría la oferta aparente del tramo.

    La rejilla se lleva a 5 minutos para que coincida con la del entrenamiento.
    Si en una ventana de 5 min caen dos lotes del feed, se toma el último: es el
    estado al final del intervalo, que es lo que el histórico representaba.
    """
    if series.is_empty():
        return pl.DataFrame()

    mapping = pl.DataFrame(
        {"kerbsideid": list(bay_segments.keys()), "segment_id": list(bay_segments.values())}
    )
    bucketed = (
        series.filter(pl.col("is_reliable"))
        .join(mapping, on="kerbsideid", how="inner")
        .with_columns(timestamp=pl.col("feed_lastupdated").dt.truncate(f"{LIVE_GRID_MINUTES}m"))
    )
    if bucketed.is_empty():
        return pl.DataFrame()

    # Último lote dentro de cada ventana de 5 min, por plaza.
    latest = (
        bucketed.sort("feed_lastupdated")
        .group_by("segment_id", "timestamp", "kerbsideid")
        .agg(is_occupied=pl.col("is_occupied").last())
    )
    return (
        latest.group_by("segment_id", "timestamp")
        .agg(
            total_bays=pl.len(),
            occupied_bays=pl.col("is_occupied").sum(),
        )
        .with_columns(free_bays=pl.col("total_bays") - pl.col("occupied_bays"))
        .sort("segment_id", "timestamp")
    )


def densify(occupancy: pl.DataFrame) -> pl.DataFrame:
    """Rellena los instantes que falten, para que los retardos no mientan.

    `add_lag_features` traduce minutos a número de filas, lo que **asume una
    rejilla sin huecos**. Si el poller perdió lotes, `shift(1)` saltaría a un
    instante arbitrario y `free_bays_lag5` contendría un valor de hace media
    hora sin que nada lo indique.

    Los huecos se rellenan hacia delante (el último estado conocido) y se marcan
    con `is_interpolated`, para que el consumidor sepa que esa fila no se
    observó.
    """
    if occupancy.is_empty():
        return occupancy

    start, end = occupancy["timestamp"].min(), occupancy["timestamp"].max()
    grid = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start, end, f"{LIVE_GRID_MINUTES}m", closed="both", eager=True
            )
        }
    )
    segments = occupancy.select("segment_id").unique()
    scaffold = segments.join(grid, how="cross")

    return (
        scaffold.join(occupancy, on=["segment_id", "timestamp"], how="left")
        .sort("segment_id", "timestamp")
        .with_columns(is_interpolated=pl.col("total_bays").is_null())
        .with_columns(
            [
                pl.col(column).forward_fill().over("segment_id")
                for column in ("total_bays", "occupied_bays", "free_bays")
            ]
        )
        .filter(pl.col("total_bays").is_not_null())
    )


def build_live_features(
    occupancy: pl.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    holidays: tuple[dt.date, ...] = VICTORIA_HOLIDAYS_2019,
) -> pl.DataFrame:
    """Construye las mismas columnas que el dataset de entrenamiento.

    No añade objetivos: en vivo no existe el futuro todavía. Esa es la única
    diferencia con `dataset.build_dataset`.
    """
    if occupancy.is_empty():
        return occupancy

    frame = occupancy.sort("segment_id", "timestamp").with_columns(
        occupancy_rate=pl.when(pl.col("total_bays") > 0)
        .then(pl.col("occupied_bays") / pl.col("total_bays"))
        .otherwise(None)
    )
    frame = add_time_features(frame, holidays=holidays)
    frame = add_lag_features(frame, lags=lags, grid_minutes=LIVE_GRID_MINUTES)

    adjacency = {
        segment_id(key): frozenset(segment_id(n) for n in neighbours)
        for key, neighbours in build_adjacency(
            [parse_segment_id(value) for value in frame["segment_id"].unique().to_list()]
        ).items()
    }
    return add_neighbour_features(frame, adjacency)


def latest_features(features: pl.DataFrame) -> pl.DataFrame:
    """Última fila de cada tramo: el presente, que es desde donde se predice."""
    if features.is_empty():
        return features
    return features.sort("timestamp").group_by("segment_id").last()
