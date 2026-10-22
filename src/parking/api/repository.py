"""Acceso al último snapshot capturado por el poller.

La API no consulta la fuente: lee lo que el poller ya guardó. Así una caída de
la API no interrumpe la captura de la serie, y una caída del feed no tumba la
API, que seguirá sirviendo el último estado conocido **diciendo lo viejo que
es** (ver `SnapshotMeta.age_seconds`, que la API expone como `is_stale`).

Esto último es consecuencia directa de la Fase 2: si el poller muere, la API
serviría indefinidamente un snapshot viejo sin que nadie lo note. Que el
endpoint publique la antigüedad convierte un fallo silencioso en uno visible.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from parking.config import Settings
from parking.quality.flags import annotate_quality
from parking.quality.gaps import snapshot_timestamps

logger = logging.getLogger(__name__)


class NoSnapshotError(RuntimeError):
    """No hay ningún snapshot en disco: el poller no ha corrido nunca."""


@dataclass(frozen=True)
class SnapshotMeta:
    """Metadatos del snapshot servido."""

    feed_lastupdated: dt.datetime
    path: Path
    served_at: dt.datetime

    @property
    def age_seconds(self) -> float:
        return (self.served_at - self.feed_lastupdated).total_seconds()


@dataclass(frozen=True)
class LiveSnapshot:
    """Snapshot anotado con calidad, más sus metadatos."""

    frame: pl.DataFrame
    meta: SnapshotMeta


class SnapshotRepository:
    """Lee y cachea el snapshot más reciente del disco.

    El poller escribe un fichero cada ~2 min, así que releer el Parquet en cada
    petición sería gasto puro. La caché se invalida por ruta: el nombre del
    fichero contiene el sello del lote, así que un lote nuevo es una ruta nueva
    y no hace falta mirar `mtime` ni fijar un TTL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cached: LiveSnapshot | None = None

    def latest_path(self) -> Path:
        """Ruta del snapshot más reciente, por sello de lote y no por `mtime`.

        Ordenar por `mtime` daría el orden equivocado si se recupera un
        histórico o se copian ficheros entre máquinas.
        """
        stamps = snapshot_timestamps(self._settings.realtime_raw_dir)
        if not stamps:
            raise NoSnapshotError(
                f"no hay snapshots en {self._settings.realtime_raw_dir}; "
                "arranca el poller (python -m parking.ingest.poller)"
            )
        newest = stamps[-1]
        return (
            self._settings.realtime_raw_dir
            / f"date={newest:%Y-%m-%d}"
            / f"snapshot_{newest:%Y%m%dT%H%M%SZ}.parquet"
        )

    def load(self, *, now: dt.datetime | None = None) -> LiveSnapshot:
        """Devuelve el snapshot más reciente, anotado con calidad.

        No filtra las plazas no fiables: el mapa necesita pintarlas como "sin
        dato" y omitirlas sería mentir por omisión (regla de la Fase 2).
        """
        served_at = now or dt.datetime.now(dt.UTC)
        path = self.latest_path()

        if self._cached is not None and self._cached.meta.path == path:
            # Mismo lote: se reutiliza el frame y solo se actualiza la hora de
            # servicio, que es lo que hace crecer `age_seconds`.
            cached = self._cached
            return LiveSnapshot(
                frame=cached.frame,
                meta=SnapshotMeta(cached.meta.feed_lastupdated, path, served_at),
            )

        logger.info("cargando snapshot %s", path.name)
        frame = annotate_quality(
            pl.read_parquet(path), stuck_after_hours=self._settings.stuck_after_hours
        )
        feed_lastupdated = frame["feed_lastupdated"].item(0)
        snapshot = LiveSnapshot(frame=frame, meta=SnapshotMeta(feed_lastupdated, path, served_at))
        self._cached = snapshot
        return snapshot

    def invalidate(self) -> None:
        """Olvida la caché. Para los tests y para recargas manuales."""
        self._cached = None
