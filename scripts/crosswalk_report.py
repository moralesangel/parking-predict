"""Recalcula el informe de cruce de identificadores de `docs/id_crosswalk.md`.

Reimprime las tablas de solape descargando las entradas de la API. Las cifras
del documento salen de aquí, así que cualquiera puede comprobarlas.

    uv run python scripts/crosswalk_report.py --help

Por defecto usa una muestra de 150 MB del ZIP histórico (lo descarga por rango
si no está en disco). Con `--full` recorre el fichero completo, que ya debería
estar descargado por `parking.ingest.historical`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import logging
import struct
import sys
import zlib
from collections.abc import Iterator
from pathlib import Path

import httpx

from parking.config import get_settings
from parking.ingest.crosswalk import segment_key_from_description, segment_key_from_historical_row
from parking.ingest.historical import HISTORICAL_ARCHIVES, iter_csv_lines

logger = logging.getLogger("crosswalk")

API = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
SAMPLE_BYTES = 150 << 20


def fetch_export(dataset: str, select: str, client: httpx.Client) -> list[dict]:
    """Descarga un dataset completo por el endpoint de export JSON."""
    logger.info("descargando %s", dataset)
    response = client.get(
        f"{API}/{dataset}/exports/json", params={"limit": -1, "select": select}, timeout=180
    )
    response.raise_for_status()
    return response.json()


def iter_sample_rows(archive_prefix: Path) -> Iterator[dict[str, str]]:
    """Itera las filas del CSV inflando el prefijo de un ZIP parcial.

    Un ZIP truncado no tiene directorio central, así que `zipfile` no lo abre:
    se lee la cabecera local y se infla el stream deflate a mano. Es lo que
    permite inspeccionar 42,7 M de filas sin bajar los 717 MB.
    """
    raw = archive_prefix.read_bytes()
    if raw[:4] != b"PK\x03\x04":
        raise ValueError(f"{archive_prefix} no empieza por una cabecera local de ZIP")
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    body = raw[30 + name_len + extra_len :]

    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    chunks: list[bytes] = []
    step = 1 << 22
    for offset in range(0, len(body), step):
        try:
            chunks.append(decompressor.decompress(body[offset : offset + step]))
        except zlib.error:
            break  # fin del prefijo disponible
    text = b"".join(chunks).decode("utf-8-sig", "replace")
    # La última línea del prefijo casi seguro está cortada a medias.
    yield from csv.DictReader(io.StringIO(text[: text.rfind("\n")]))


def download_sample(destination: Path, client: httpx.Client) -> Path:
    """Baja los primeros `SAMPLE_BYTES` del ZIP de 2019 por petición `Range`."""
    if destination.exists() and destination.stat().st_size >= SAMPLE_BYTES:
        logger.info("muestra ya en disco: %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("descargando muestra de %d MB por Range", SAMPLE_BYTES >> 20)
    with client.stream(
        "GET",
        HISTORICAL_ARCHIVES["2019"],
        headers={"Range": f"bytes=0-{SAMPLE_BYTES - 1}"},
        timeout=httpx.Timeout(60, read=None),
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
    return destination


def overlap(a: set, b: set) -> tuple[int, float]:
    common = a & b
    smaller = min(len(a), len(b)) or 1
    return len(common), len(common) / smaller * 100


def report_line(left_name: str, left: set, right_name: str, right: set) -> None:
    count, pct = overlap(left, right)
    verdict = "OK " if pct > 50 else ("~  " if pct > 5 else "NO ")
    print(f"  {verdict} {left_name:<22} n {right_name:<22} {count:>5} comunes = {pct:5.1f} %")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="recorre el ZIP completo en vez de la muestra de 150 MB",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    settings = get_settings()
    with httpx.Client(follow_redirects=True) as client:
        realtime = fetch_export(settings.realtime_dataset, "kerbsideid", client)
        geometry = fetch_export(
            "on-street-parking-bays", "kerbsideid,roadsegmentid,roadsegmentdescription", client
        )
        restrictions = fetch_export("on-street-car-park-bay-restrictions", "bayid,deviceid", client)

        if args.full:
            archive = settings.historical_raw_dir / "sensor-data-2019.zip"
            if not archive.exists():
                parser.error(f"falta {archive}; ejecuta antes la descarga del histórico")
            lines = iter_csv_lines(archive)
            header = next(iter(lines)).split(",")
            rows: Iterator[dict[str, str]] = (
                dict(zip(header, line.split(","), strict=False)) for line in lines
            )
        else:
            sample = download_sample(settings.data_dir / "tmp" / "hist_2019_sample.bin", client)
            rows = iter_sample_rows(sample)

        hist_devices: set[int] = set()
        hist_bays: set[int] = set()
        hist_segments: collections.Counter = collections.Counter()
        hist_marker_segment: dict[str, object] = {}
        events = 0
        for row in rows:
            events += 1
            if row.get("DeviceId", "").isdigit():
                hist_devices.add(int(row["DeviceId"]))
            if row.get("BayId", "").isdigit():
                hist_bays.add(int(row["BayId"]))
            key = segment_key_from_historical_row(row)
            if key is not None:
                hist_segments[key] += 1
                hist_marker_segment.setdefault(row.get("StreetMarker", ""), key)

    rt_kerb = {r["kerbsideid"] for r in realtime if r.get("kerbsideid") is not None}
    geo_kerb = {
        int(r["kerbsideid"])
        for r in geometry
        if r.get("kerbsideid") and str(r["kerbsideid"]).isdigit()
    }
    restr_bay = {r["bayid"] for r in restrictions if r.get("bayid") is not None}
    restr_dev = {r["deviceid"] for r in restrictions if r.get("deviceid") is not None}

    geo_segments: dict[object, set[int]] = collections.defaultdict(set)
    unparsed = 0
    for row in geometry:
        key = segment_key_from_description(row.get("roadsegmentdescription"))
        if key is None:
            unparsed += 1
            continue
        geo_segments[key].add(row["roadsegmentid"])

    print(
        f"\nmuestra histórico: {events:,} eventos, {len(hist_devices)} DeviceId, "
        f"{len(hist_bays)} BayId, {len(hist_segments)} tramos"
    )
    print(f"geometría: {len(geometry):,} filas, {unparsed} descripciones no parseables\n")

    print("=== A) histórico -> tiempo real, directo ===")
    report_line("hist.DeviceId", hist_devices, "realtime.kerbsideid", rt_kerb)
    report_line("hist.BayId", hist_bays, "realtime.kerbsideid", rt_kerb)
    report_line("hist.DeviceId", hist_devices, "bays.kerbsideid", geo_kerb)
    report_line("hist.BayId", hist_bays, "bays.kerbsideid", geo_kerb)

    print("\n=== B) histórico -> restricciones (puente candidato) ===")
    report_line("hist.DeviceId", hist_devices, "restr.deviceid", restr_dev)
    report_line("hist.BayId", hist_bays, "restr.bayid", restr_bay)

    print("\n=== C) restricciones -> tiempo real (cerraría el puente) ===")
    report_line("restr.deviceid", restr_dev, "realtime.kerbsideid", rt_kerb)
    report_line("restr.bayid", restr_bay, "realtime.kerbsideid", rt_kerb)

    print("\n=== D) control: geometría -> tiempo real (debe cruzar) ===")
    report_line("realtime.kerbsideid", rt_kerb, "bays.kerbsideid", geo_kerb)

    crossed = set(hist_segments) & set(geo_segments)
    covered = sum(n for key, n in hist_segments.items() if key in crossed)
    markers = sum(1 for key in hist_marker_segment.values() if key in crossed)
    ambiguous = sum(1 for key in crossed if len(geo_segments[key]) > 1)
    print("\n=== CRUCE POR TRAMO DE CALLE ===")
    print(
        f"  tramos que cruzan : {len(crossed)}/{len(hist_segments)} = "
        f"{len(crossed) / max(len(hist_segments), 1) * 100:.1f} %"
    )
    print(f"  eventos cubiertos : {covered:,}/{events:,} = {covered / max(events, 1) * 100:.1f} %")
    print(
        f"  plazas cubiertas  : {markers}/{len(hist_marker_segment)} = "
        f"{markers / max(len(hist_marker_segment), 1) * 100:.1f} %"
    )
    print(f"  tramos ambiguos (>1 roadsegmentid): {ambiguous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
