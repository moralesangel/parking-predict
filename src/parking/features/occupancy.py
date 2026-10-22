"""De eventos de estancia a serie temporal de ocupación por tramo.

El histórico es un esquema de **intervalos**: cada fila cubre el periodo entre
dos transiciones de una plaza. El modelo necesita lo contrario, el **estado** de
un tramo en instantes regulares. Esta es la conversión.

**`VehiclePresent` decide qué es cada intervalo, y confundirlo arruina el
dataset.** La descripción de la fuente lo dice: un evento indica si había o no
vehículo, *"which can determine stay time and vacancy time"*. Es decir:

- `VehiclePresent = true`  → la plaza estuvo **ocupada** en ese intervalo
- `VehiclePresent = false` → la plaza estuvo **libre** en ese intervalo

No son estancias y ruido: son **dos particiones complementarias del tiempo**.
Medido sobre la semana del 1 al 8 de marzo de 2019, los intervalos ocupados
cubren el 43,7 % del tiempo-plaza y los vacíos el 47,9 %; juntos, el 91,6 %.
Tratar todas las filas como estancias da una ocupación del 91,6 % —plana a lo
largo del día, porque es la suma de ocupado y vacío— en vez del 43,7 % real.
Ese 43,7 % además concuerda con lo que mide hoy el feed en tiempo real.

El método evita el producto cartesiano obvio (para cada instante, contar
estancias que lo cubren), que sobre 42,7 M de eventos no termina. En su lugar:

1. Cada estancia se parte en dos deltas: `+1` al llegar, `-1` al salir.
2. Los deltas se ordenan por tiempo dentro de cada tramo.
3. La **suma acumulada** da el número de plazas ocupadas tras cada evento.
4. Un `join_asof` lleva ese valor a la rejilla de instantes.

Coste: una ordenación, no un cruce. La ocupación en cualquier instante sale de
la última suma acumulada anterior a él.
"""

from __future__ import annotations

import datetime as dt
import warnings

import polars as pl

#: Formato de fecha del CSV histórico: `11/03/2019 03:00:56 AM` es el 3 de
#: noviembre, no el 11 de marzo. Se declara explícito porque una inferencia
#: automática acierta en los días > 12 y falla en silencio en el resto.
US_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

#: Paso de la rejilla. 5 min es el horizonte de predicción más corto del plan:
#: una rejilla más gruesa no podría expresar el objetivo a 5 min.
DEFAULT_GRID_MINUTES = 5


def parse_events(source: pl.LazyFrame) -> pl.LazyFrame:
    """Normaliza el CSV histórico crudo a estancias con tipos correctos.

    `StreetMarker` es la clave de plaza, **no `DeviceId`**: en 2019 hay 5013
    marcas de calzada pero 6665 identificadores de sensor, porque un sensor
    averiado se sustituye y la plaza física sigue siendo la misma. Agregar por
    `DeviceId` contaría una plaza dos veces tras cada sustitución.
    """
    return source.select(
        bay=pl.col("StreetMarker"),
        street=pl.col("StreetName"),
        cross_a=pl.col("BetweenStreet1"),
        cross_b=pl.col("BetweenStreet2"),
        vehicle_present=pl.col("VehiclePresent").cast(pl.Boolean),
        arrival=pl.col("ArrivalTime").str.strptime(
            pl.Datetime("us"), US_DATETIME_FORMAT, strict=False
        ),
        departure=pl.col("DepartureTime").str.strptime(
            pl.Datetime("us"), US_DATETIME_FORMAT, strict=False
        ),
    )


def clean_events(events: pl.LazyFrame) -> pl.LazyFrame:
    """Descarta las estancias imposibles de interpretar.

    Sobre los 42,7 M de eventos de 2019 esto quita:

    - **166 estancias con salida anterior a la llegada.** Un intervalo negativo
      restaría ocupación donde no la hubo, y arrastraría el error por toda la
      suma acumulada posterior de ese tramo.
    - **21 046 estancias de duración cero.** No aportan ocupación en ninguna
      rejilla de 5 min y sus dos deltas se anulan.
    - Filas sin fecha parseable o sin alguno de los nombres de calle, que no se
      pueden situar en ningún tramo.

    Son el 0,05 % del total. Se descartan aquí y la cifra se registra: a
    diferencia de una lectura de sensor dudosa, una estancia con el tiempo al
    revés no admite ninguna interpretación que conservarla haga más honesta.
    """
    return events.filter(
        pl.col("arrival").is_not_null()
        & pl.col("departure").is_not_null()
        & (pl.col("departure") > pl.col("arrival"))
        & pl.col("bay").is_not_null()
        & pl.col("street").is_not_null()
        & pl.col("cross_a").is_not_null()
        & pl.col("cross_b").is_not_null()
        & pl.col("vehicle_present").is_not_null()
    )


def _normalise(column: str) -> pl.Expr:
    """Mayúsculas y espacios colapsados, igual que `crosswalk.normalise_street`."""
    return pl.col(column).str.strip_chars().str.replace_all(r"\s+", " ").str.to_uppercase()


def with_segment_id(events: pl.LazyFrame) -> pl.LazyFrame:
    """Añade `segment_id`, la clave del tramo de calle.

    Las dos calles delimitadoras se ordenan alfabéticamente antes de concatenar,
    porque el mismo tramo aparece con los dos órdenes en los datos. Es la misma
    regla que `parking.ingest.crosswalk.segment_key`, aquí expresada en polars
    para no salir del motor con 42,7 M de filas.
    """
    street, first, second = _normalise("street"), _normalise("cross_a"), _normalise("cross_b")
    low = pl.min_horizontal(first, second)
    high = pl.max_horizontal(first, second)
    return events.with_columns(segment_id=pl.concat_str([street, low, high], separator="|")).drop(
        "street", "cross_a", "cross_b"
    )


def time_grid(
    start: dt.datetime, end: dt.datetime, *, minutes: int = DEFAULT_GRID_MINUTES
) -> pl.DataFrame:
    """Rejilla regular de instantes `[start, end)`."""
    return pl.DataFrame(
        {"timestamp": pl.datetime_range(start, end, f"{minutes}m", closed="left", eager=True)}
    )


def occupancy_series(
    events: pl.LazyFrame,
    start: dt.datetime,
    end: dt.datetime,
    *,
    minutes: int = DEFAULT_GRID_MINUTES,
) -> pl.DataFrame:
    """Ocupación por tramo en cada instante de la rejilla `[start, end)`.

    `events` debe traer ya `segment_id`, `bay`, `arrival` y `departure`.

    Se incluyen las estancias que **solapan** la ventana, no solo las que
    empiezan en ella: un coche que aparcó antes de `start` y sigue allí ocupa
    una plaza durante toda la ventana, y filtrar por `arrival >= start` lo
    perdería y haría parecer el tramo más vacío de lo que estaba.

    Devuelve `segment_id`, `timestamp`, `occupied_bays`, `total_bays` y
    `free_bays`.

    Los dos conteos usan subconjuntos distintos a propósito:

    - `occupied_bays` sale **solo** de los intervalos con `vehicle_present`.
    - `total_bays` cuenta las plazas vistas en **cualquiera** de los dos
      estados. Una plaza que estuvo libre toda la ventana no genera ningún
      intervalo ocupado, pero existe y hay que contarla en el denominador;
      derivar el inventario de los intervalos ocupados la perdería justo en los
      tramos más vacíos, que son los interesantes.

    `total_bays` sigue siendo el número de plazas **observadas**: una plaza sin
    ningún intervalo de ningún tipo es invisible, y no se puede distinguir de un
    sensor muerto. Es la limitación principal de esta reconstrucción y está
    documentada en `docs/dataset.md`.
    """
    overlapping = events.filter((pl.col("arrival") < end) & (pl.col("departure") > start))
    occupied = overlapping.filter(pl.col("vehicle_present"))

    arrivals = occupied.select("segment_id", t=pl.col("arrival"), delta=pl.lit(1, pl.Int32))
    departures = occupied.select("segment_id", t=pl.col("departure"), delta=pl.lit(-1, pl.Int32))

    # La suma acumulada se calcula desde el principio del flujo de deltas, que
    # incluye las llegadas anteriores a `start`: así el valor en el primer
    # instante de la rejilla ya refleja los coches que estaban allí antes.
    cumulative = (
        pl.concat([arrivals, departures])
        .sort("segment_id", "t")
        .with_columns(occupied_bays=pl.col("delta").cum_sum().over("segment_id"))
        .select("segment_id", "t", "occupied_bays")
        .collect()
    )

    totals = overlapping.group_by("segment_id").agg(total_bays=pl.col("bay").n_unique()).collect()

    grid = time_grid(start, end, minutes=minutes)
    segments = totals.select("segment_id")
    # Producto tramo x instante: es la forma final del dataset, no un paso
    # intermedio, asi que aqui si es el cruce que se quiere.
    scaffold = segments.join(grid, how="cross").sort("segment_id", "timestamp")

    # polars avisa de que no puede comprobar la ordenación cuando se agrupa con
    # `by`. Ambos lados se acaban de ordenar justo arriba, así que el aviso no
    # aporta nada aquí y dejarlo saldría en cada ejecución.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        joined = scaffold.join_asof(
            cumulative.sort("t"),
            left_on="timestamp",
            right_on="t",
            by="segment_id",
            strategy="backward",
        )

    series = (
        joined.with_columns(
            # Sin ningun evento previo en la ventana, el tramo estaba vacio.
            occupied_bays=pl.col("occupied_bays").fill_null(0).clip(lower_bound=0)
        )
        .join(totals, on="segment_id", how="left")
        .with_columns(
            free_bays=(pl.col("total_bays") - pl.col("occupied_bays")).clip(lower_bound=0)
        )
        .drop("t")
    )
    return series.sort("segment_id", "timestamp")
