"""Detección de huecos en la serie de snapshots.

El feed publica un lote cada 120 s. Un salto mayor entre dos lotes consecutivos
significa que faltan datos, y hay que saberlo antes de calcular lags o
tendencias: una feature de "cambio en los últimos 10 min" calculada sobre un
hueco de 2 h no es una tendencia, es ruido.

Importa distinguir la causa, porque se arreglan de forma distinta:

- **El poller no capturó** (caído, sin red, reiniciado). Se corrige operando.
- **La fuente no publicó** nuevos lotes. No se puede corregir; hay que marcar
  el periodo como no observado.

Ambas se ven igual en los snapshots guardados, pero la segunda deja rastro: si
el poller siguió corriendo y pidiendo, el lote repetido se descartó por
idempotencia y no hay fichero. Distinguirlas de forma automática requiere los
logs del poller, así que este módulo reporta el hueco sin atribuir causa.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Gap:
    """Hueco entre dos lotes consecutivos."""

    previous: dt.datetime
    following: dt.datetime
    missing_cycles: int

    @property
    def duration(self) -> dt.timedelta:
        return self.following - self.previous


def snapshot_timestamps(realtime_dir: Path) -> list[dt.datetime]:
    """Sellos de lote de los snapshots en disco, ordenados.

    Se leen del nombre del fichero y no del contenido: con miles de snapshots,
    abrir cada Parquet para sacar una fecha que ya está en la ruta es un gasto
    innecesario. El nombre es `snapshot_YYYYMMDDTHHMMSSZ.parquet`.
    """
    stamps: list[dt.datetime] = []
    for path in sorted(realtime_dir.rglob("snapshot_*.parquet")):
        stem = path.stem.removeprefix("snapshot_")
        try:
            stamps.append(dt.datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC))
        except ValueError:
            continue  # fichero ajeno al patrón: se ignora, no se rompe el informe
    return sorted(stamps)


def find_gaps(
    timestamps: list[dt.datetime],
    *,
    expected_interval_seconds: float = 120.0,
    tolerance_seconds: float = 360.0,
) -> list[Gap]:
    """Huecos entre lotes consecutivos que superan `tolerance_seconds`.

    La tolerancia por defecto son 3 ciclos: el feed se retrasa de vez en cuando
    sin que falte nada, y marcar cada retraso de un ciclo como hueco llenaría el
    informe de falsos positivos.
    """
    gaps: list[Gap] = []
    for previous, following in itertools.pairwise(timestamps):
        delta = (following - previous).total_seconds()
        if delta > tolerance_seconds:
            missing = max(0, round(delta / expected_interval_seconds) - 1)
            gaps.append(Gap(previous, following, missing))
    return gaps


def continuity_report(
    timestamps: list[dt.datetime],
    *,
    expected_interval_seconds: float = 120.0,
    tolerance_seconds: float = 360.0,
) -> dict[str, object]:
    """Resumen de continuidad de la serie capturada.

    `completeness_pct` compara los lotes capturados con los que cabían en la
    ventana observada. Es la cifra que dice si la serie sirve para entrenar.
    """
    if not timestamps:
        return {
            "snapshots": 0,
            "first": None,
            "last": None,
            "expected": 0,
            "completeness_pct": None,
            "gaps": [],
        }

    first, last = timestamps[0], timestamps[-1]
    span = (last - first).total_seconds()
    expected = int(span // expected_interval_seconds) + 1
    gaps = find_gaps(
        timestamps,
        expected_interval_seconds=expected_interval_seconds,
        tolerance_seconds=tolerance_seconds,
    )
    return {
        "snapshots": len(timestamps),
        "first": first,
        "last": last,
        "expected": expected,
        "completeness_pct": round(len(timestamps) / expected * 100, 1) if expected else None,
        "gaps": gaps,
    }


def load_series(realtime_dir: Path, *, columns: list[str] | None = None) -> pl.DataFrame:
    """Carga todos los snapshots como un único frame.

    Usa `scan_parquet` sobre el glob para no materializar los ficheros uno a uno.
    Devuelve un frame vacío con esquema si no hay nada que cargar, para que quien
    llame no tenga que distinguir ese caso.
    """
    paths = sorted(realtime_dir.rglob("snapshot_*.parquet"))
    if not paths:
        return pl.DataFrame()
    lazy = pl.scan_parquet(paths)
    if columns is not None:
        lazy = lazy.select(columns)
    return lazy.collect()
