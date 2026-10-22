"""Descarga del histórico anual de sensores.

Los datasets históricos no son consultables por la API: `has_records` es `false`
y las filas solo existen como ZIP de archivo en S3 (717 MB para 2019). Ver
`docs/data_schema.md` §3.

El S3 responde `Accept-Ranges: bytes`, así que la descarga se reanuda en lugar
de reempezar: a 717 MB, un corte de red a mitad no debería costar la descarga
entera.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# URL verificada el 2026-10-22, extraída del href de la descripción del dataset
# `on-street-car-parking-sensor-data-2019` (la API no la expone como campo).
HISTORICAL_ARCHIVES: dict[str, str] = {
    "2019": "https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip",
}

_CHUNK_BYTES = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DownloadResult:
    """Resultado de una descarga."""

    path: Path
    total_bytes: int
    downloaded_bytes: int
    resumed: bool

    @property
    def already_complete(self) -> bool:
        """True si no hubo que pedir nada porque el fichero ya estaba entero."""
        return self.downloaded_bytes == 0


def remote_size(url: str, client: httpx.Client) -> int | None:
    """Tamaño declarado por el servidor, o `None` si no lo declara."""
    head = client.head(url, follow_redirects=True)
    head.raise_for_status()
    length = head.headers.get("content-length")
    return int(length) if length is not None else None


def download_archive(
    url: str,
    destination: Path,
    client: httpx.Client,
    *,
    progress_every_bytes: int = 64 << 20,
) -> DownloadResult:
    """Descarga `url` a `destination`, reanudando si hay una descarga parcial.

    Idempotente: si `destination` ya tiene el tamaño que declara el servidor, no
    se pide nada. Si es más corto, se piden solo los bytes que faltan con una
    cabecera `Range`. Si es más largo que lo declarado, se descarta y se
    reempieza: un fichero sobredimensionado no se puede reparar por rango.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url, client)
    have = destination.stat().st_size if destination.exists() else 0

    if total is not None and have > total:
        logger.warning(
            "%s es mayor que el remoto (%d > %d); se descarta y se reempieza",
            destination.name,
            have,
            total,
        )
        destination.unlink()
        have = 0

    if total is not None and have == total:
        logger.info("%s ya está completo (%d bytes)", destination.name, total)
        return DownloadResult(destination, total, 0, resumed=False)

    resumed = have > 0
    headers = {"Range": f"bytes={have}-"} if resumed else {}
    if resumed:
        logger.info("reanudando %s en el byte %d de %s", destination.name, have, total)

    written = 0
    next_report = progress_every_bytes
    with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        # Si se pidió un rango pero el servidor devuelve 200, está mandando el
        # fichero entero: hay que sobreescribir, no concatenar.
        append = resumed and response.status_code == 206
        if resumed and not append:
            logger.warning(
                "el servidor ignoró el Range (HTTP %d); se reempieza", response.status_code
            )
            have = 0
        with destination.open("ab" if append else "wb") as handle:
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                handle.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    logger.info("%s: %.0f MB descargados", destination.name, (have + written) / 1e6)
                    next_report += progress_every_bytes

    final = destination.stat().st_size
    if total is not None and final != total:
        raise OSError(
            f"descarga incompleta de {destination.name}: {final} bytes de {total} esperados"
        )
    logger.info("%s descargado (%d bytes)", destination.name, final)
    return DownloadResult(destination, total if total is not None else final, written, resumed)


def download_year(
    year: str,
    destination_dir: Path,
    *,
    client: httpx.Client | None = None,
    timeout_seconds: float = 60.0,
) -> DownloadResult:
    """Descarga el ZIP del histórico de `year` en `destination_dir`."""
    if year not in HISTORICAL_ARCHIVES:
        known = ", ".join(sorted(HISTORICAL_ARCHIVES))
        raise KeyError(f"año {year!r} sin URL verificada; conocidos: {known}")

    url = HISTORICAL_ARCHIVES[year]
    destination = destination_dir / f"sensor-data-{year}.zip"
    owns_client = client is None
    # read=None: el stream va chunk a chunk y un fichero de 717 MB puede tardar
    # mucho más que el timeout de conexión sin estar colgado.
    client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, read=None))
    try:
        return download_archive(url, destination, client)
    finally:
        if owns_client:
            client.close()


def _is_real_entry(name: str) -> bool:
    """Descarta los metadatos de macOS que el ZIP de origen trae dentro.

    El archivo de 2019 se empaquetó en macOS e incluye
    `__MACOSX/._On-street_Car_Parking_Sensor_Data_-_2019.csv`, un resource fork
    que no es un CSV aunque la extensión lo sugiera.
    """
    return not name.startswith("__MACOSX/") and not Path(name).name.startswith("._")


def iter_csv_lines(zip_path: Path, *, max_bytes: int | None = None) -> Iterator[str]:
    """Itera las líneas del único CSV de un ZIP del histórico.

    Trabaja en streaming con `zipfile`, sin materializar los 1,6 GB que ocupa el
    CSV descomprimido. `max_bytes` limita la lectura para muestreo en EDA.

    Ojo: el CSV viene ordenado por `DeviceId`, no por tiempo, así que un prefijo
    es un subconjunto de sensores y no una muestra temporal (ver
    `docs/data_schema.md` §3).
    """
    import io
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv") and _is_real_entry(n)]
        if len(names) != 1:
            raise ValueError(f"se esperaba 1 CSV en {zip_path.name}, hay {len(names)}: {names}")
        read = 0
        with archive.open(names[0]) as raw:
            # utf-8-sig: el CSV de 2019 empieza con BOM.
            for line in io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""):
                if max_bytes is not None and read >= max_bytes:
                    return
                read += len(line)
                yield line.rstrip("\r\n")
