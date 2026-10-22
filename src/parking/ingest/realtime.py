"""Ingesta del feed de sensores en tiempo real.

El dataset de origen se **sobreescribe** en cada ciclo de publicación: solo
expone el estado actual. Conservar la serie temporal es el trabajo de este
módulo, y es el dato con el que se entrenará el modelo de producción (ver
`docs/id_crosswalk.md` §5).

Dos decisiones vienen medidas de los datos, no supuestas (ver
`docs/data_schema.md`):

- Se usa `/exports/json?limit=-1`: una sola petición por snapshot, con
  coordenadas ya en lon/lat. El export Parquet entrega `location` como WKB
  binario y el endpoint paginado necesitaría 64 peticiones.
- La identidad de un snapshot es `max(lastupdated)`, el sello del lote del feed,
  no la hora del reloj. Ver `snapshot_path`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from parking.config import Settings

logger = logging.getLogger(__name__)

#: `status_description` solo toma estos dos valores (verificado: 6324/6324 filas).
#: `Present` describe el COCHE, no la plaza: significa que la plaza está ocupada.
_OCCUPANCY_BY_STATUS = {"Present": True, "Unoccupied": False}

_UTC_TIMESTAMP = pl.Datetime(time_unit="us", time_zone="UTC")

#: Esquema explícito del snapshot. Se fija a mano en lugar de inferirlo para que
#: un cambio de tipos en origen falle al escribir, no semanas después al leer.
#: Los tres últimos campos los añadimos nosotros, no vienen del origen.
SNAPSHOT_SCHEMA: dict[str, pl.DataType] = {
    "kerbsideid": pl.Int64(),
    "is_occupied": pl.Boolean(),
    "status_description": pl.String(),
    "status_timestamp": _UTC_TIMESTAMP,
    "lastupdated": _UTC_TIMESTAMP,
    "zone_number": pl.Int64(),
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
    "feed_lastupdated": _UTC_TIMESTAMP,
    "is_live": pl.Boolean(),
    "fetched_at": _UTC_TIMESTAMP,
}


class SnapshotError(RuntimeError):
    """El feed devolvió algo que no se puede tratar como snapshot válido."""


@dataclass(frozen=True)
class SnapshotResult:
    """Resultado de capturar un snapshot."""

    path: Path
    rows: int
    live_rows: int
    feed_lastupdated: dt.datetime
    written: bool

    @property
    def skipped_as_duplicate(self) -> bool:
        """True si el lote ya estaba en disco y no se reescribió."""
        return not self.written


def export_url(settings: Settings) -> str:
    """URL del export JSON completo del dataset de sensores."""
    return (
        f"{settings.api_base_url}/api/explore/v2.1/catalog/datasets"
        f"/{settings.realtime_dataset}/exports/json"
    )


def _is_retryable(exc: BaseException) -> bool:
    """Reintentar solo fallos transitorios: red, timeouts y 5xx/429."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def fetch_records(client: httpx.Client, settings: Settings) -> list[dict]:
    """Descarga el export completo del feed, con reintentos en fallos transitorios.

    Un 4xx distinto de 429 no se reintenta: indica que la petición está mal y
    repetirla solo gasta cuota.
    """

    @retry(
        stop=stop_after_attempt(settings.max_retries + 1),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _get() -> list[dict]:
        response = client.get(export_url(settings), params={"limit": -1})
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if not _is_retryable(exc):
                raise SnapshotError(
                    f"el feed respondió HTTP {exc.response.status_code}; no se reintenta"
                ) from exc
            raise
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SnapshotError("el feed no devolvió JSON válido") from exc
        if not isinstance(payload, list):
            raise SnapshotError(
                f"se esperaba una lista de registros, llegó {type(payload).__name__}"
            )
        return payload

    return _get()


def _parse_timestamp(value: str | None, field: str, kerbsideid: object) -> dt.datetime:
    if not value:
        raise SnapshotError(f"{field} vacío en kerbsideid={kerbsideid!r}")
    # La API entrega ISO-8601 con offset explícito (+00:00).
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise SnapshotError(f"{field} sin zona horaria en kerbsideid={kerbsideid!r}: {value!r}")
    return parsed.astimezone(dt.UTC)


def build_snapshot(records: list[dict], fetched_at: dt.datetime) -> pl.DataFrame:
    """Normaliza los registros crudos del feed a una tabla con `SNAPSHOT_SCHEMA`.

    Decisiones de normalización:

    - `status_description` se traduce a `is_occupied` booleano. Un valor
      desconocido es un error, no un `None` silencioso: si la fuente añade un
      tercer estado, hay que enterarse.
    - `location` se parte en `latitude`/`longitude`, que es lo que consume el
      frontend y lo que permite cruzar por coordenadas.
    - Se calcula `feed_lastupdated = max(lastupdated)` y la bandera
      `is_live`. Una plaza cuyo `lastupdated` es anterior al del lote no
      reportó en este ciclo (`docs/data_schema.md` §1.2). La bandera se
      **marca**, no se filtra: borrar en silencio impediría medir la cobertura.
    """
    if not records:
        raise SnapshotError("el feed devolvió 0 registros")

    rows: list[dict] = []
    for record in records:
        kerbsideid = record.get("kerbsideid")
        if kerbsideid is None:
            raise SnapshotError(f"registro sin kerbsideid: {record!r}")

        status = record.get("status_description")
        if status not in _OCCUPANCY_BY_STATUS:
            raise SnapshotError(
                f"status_description desconocido {status!r} en kerbsideid={kerbsideid!r}; "
                f"esperados {sorted(_OCCUPANCY_BY_STATUS)}"
            )

        location = record.get("location") or {}
        latitude, longitude = location.get("lat"), location.get("lon")
        if latitude is None or longitude is None:
            raise SnapshotError(f"location incompleta en kerbsideid={kerbsideid!r}: {location!r}")

        rows.append(
            {
                "kerbsideid": int(kerbsideid),
                "is_occupied": _OCCUPANCY_BY_STATUS[status],
                "status_description": status,
                "status_timestamp": _parse_timestamp(
                    record.get("status_timestamp"), "status_timestamp", kerbsideid
                ),
                "lastupdated": _parse_timestamp(
                    record.get("lastupdated"), "lastupdated", kerbsideid
                ),
                "zone_number": None
                if record.get("zone_number") is None
                else int(record["zone_number"]),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )

    duplicates = len(rows) - len({row["kerbsideid"] for row in rows})
    if duplicates:
        raise SnapshotError(f"{duplicates} kerbsideid duplicados en el snapshot")

    feed_lastupdated = max(row["lastupdated"] for row in rows)
    for row in rows:
        row["feed_lastupdated"] = feed_lastupdated
        row["is_live"] = row["lastupdated"] == feed_lastupdated
        row["fetched_at"] = fetched_at.astimezone(dt.UTC)

    return pl.from_dicts(rows, schema=SNAPSHOT_SCHEMA)


def snapshot_path(settings: Settings, feed_lastupdated: dt.datetime) -> Path:
    """Ruta del snapshot, derivada del sello del lote del feed.

    Nombrar por `feed_lastupdated` y no por la hora del reloj es lo que da
    idempotencia: el feed publica cada ~2 min y el poller sondea cada 5, pero
    un reintento o un reinicio pueden volver a leer el mismo lote. Con esta
    ruta, ese caso reescribe la misma celda en lugar de añadir un duplicado que
    luego habría que deduplicar en las features.

    Partición `date=YYYY-MM-DD` según el plan, en UTC para que no se mueva con
    el horario de verano de Melbourne.
    """
    stamp = feed_lastupdated.astimezone(dt.UTC)
    return (
        settings.realtime_raw_dir
        / f"date={stamp:%Y-%m-%d}"
        / f"snapshot_{stamp:%Y%m%dT%H%M%SZ}.parquet"
    )


def write_snapshot(frame: pl.DataFrame, path: Path, *, overwrite: bool = False) -> bool:
    """Escribe el snapshot. Devuelve `False` si ya existía y no se sobreescribe.

    La escritura es atómica: se escribe a un temporal en el mismo directorio y
    se renombra. Así un poller interrumpido a media escritura no deja un
    Parquet truncado que rompa las lecturas posteriores.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        logger.info("snapshot %s ya existe; no se reescribe", path.name)
        return False

    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def capture_snapshot(
    client: httpx.Client,
    settings: Settings,
    *,
    now: dt.datetime | None = None,
) -> SnapshotResult:
    """Captura un snapshot del feed y lo persiste. Idempotente por lote."""
    fetched_at = now or dt.datetime.now(dt.UTC)
    records = fetch_records(client, settings)
    frame = build_snapshot(records, fetched_at)

    feed_lastupdated = frame["feed_lastupdated"].item(0)
    live_rows = int(frame["is_live"].sum())
    path = snapshot_path(settings, feed_lastupdated)
    written = write_snapshot(frame, path)

    logger.info(
        "snapshot lote=%s plazas=%d vivas=%d (%.1f%%) -> %s%s",
        feed_lastupdated.isoformat(),
        frame.height,
        live_rows,
        live_rows / frame.height * 100,
        path.name,
        "" if written else " (duplicado, omitido)",
    )
    return SnapshotResult(
        path=path,
        rows=frame.height,
        live_rows=live_rows,
        feed_lastupdated=feed_lastupdated,
        written=written,
    )
