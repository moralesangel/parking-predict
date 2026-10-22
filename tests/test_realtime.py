"""Tests de `parking.ingest.realtime`.

Los registros de ejemplo copian la forma real del feed, incluidos los casos que
se observaron en producción: `zone_number` nulo y plazas con `lastupdated`
anterior al del lote.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx

from parking.config import Settings
from parking.ingest.realtime import (
    SNAPSHOT_SCHEMA,
    SnapshotError,
    build_snapshot,
    capture_snapshot,
    export_url,
    fetch_records,
    snapshot_path,
    write_snapshot,
)

FETCHED_AT = dt.datetime(2026, 10, 22, 9, 30, tzinfo=dt.UTC)
FEED_STAMP = "2026-10-22T09:26:50+00:00"


def record(
    kerbsideid: int,
    status: str = "Unoccupied",
    *,
    lastupdated: str = FEED_STAMP,
    status_timestamp: str = "2026-10-22T08:00:00+00:00",
    zone_number: int | None = 7219,
    lat: float = -37.82,
    lon: float = 144.95,
) -> dict:
    return {
        "lastupdated": lastupdated,
        "status_timestamp": status_timestamp,
        "zone_number": zone_number,
        "status_description": status,
        "kerbsideid": kerbsideid,
        "location": {"lon": lon, "lat": lat},
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path)


# --- build_snapshot ---------------------------------------------------------


def test_present_means_occupied() -> None:
    """`Present` describe el coche, no la plaza: la plaza está ocupada."""
    frame = build_snapshot([record(1, "Present"), record(2, "Unoccupied")], FETCHED_AT)

    occupied = dict(zip(frame["kerbsideid"], frame["is_occupied"], strict=True))
    assert occupied == {1: True, 2: False}


def test_snapshot_matches_declared_schema() -> None:
    frame = build_snapshot([record(1)], FETCHED_AT)

    assert dict(frame.schema) == SNAPSHOT_SCHEMA


def test_location_is_split_into_lat_lon() -> None:
    frame = build_snapshot([record(1, lat=-37.8203, lon=144.9546)], FETCHED_AT)

    assert frame["latitude"].to_list() == [-37.8203]
    assert frame["longitude"].to_list() == [144.9546]


def test_null_zone_number_is_preserved() -> None:
    """520 de 6324 plazas reales no tienen zona; es un nulo legítimo, no un error."""
    frame = build_snapshot([record(1, zone_number=None)], FETCHED_AT)

    assert frame["zone_number"].to_list() == [None]


def test_stale_sensors_are_flagged_not_dropped() -> None:
    """Una plaza con `lastupdated` anterior al del lote no reportó en este ciclo."""
    records = [
        record(1, lastupdated=FEED_STAMP),
        record(2, lastupdated=FEED_STAMP),
        record(3, lastupdated="2025-10-30T02:45:40+00:00"),
    ]

    frame = build_snapshot(records, FETCHED_AT)

    assert frame.height == 3, "las plazas caídas se marcan, nunca se borran"
    live = dict(zip(frame["kerbsideid"], frame["is_live"], strict=True))
    assert live == {1: True, 2: True, 3: False}


def test_feed_lastupdated_is_the_batch_maximum() -> None:
    records = [
        record(1, lastupdated="2025-10-30T02:45:40+00:00"),
        record(2, lastupdated=FEED_STAMP),
    ]

    frame = build_snapshot(records, FETCHED_AT)

    expected = dt.datetime(2026, 10, 22, 9, 26, 50, tzinfo=dt.UTC)
    assert set(frame["feed_lastupdated"].to_list()) == {expected}


def test_timestamps_are_normalised_to_utc() -> None:
    """El feed da offset explícito; se guarda siempre en UTC."""
    frame = build_snapshot(
        [record(1, lastupdated="2026-10-22T20:26:50+11:00", status_timestamp=FEED_STAMP)],
        FETCHED_AT,
    )

    assert frame["lastupdated"].to_list() == [dt.datetime(2026, 10, 22, 9, 26, 50, tzinfo=dt.UTC)]


def test_unknown_status_is_an_error() -> None:
    """Un tercer estado cambiaría la semántica del target: hay que enterarse."""
    with pytest.raises(SnapshotError, match="status_description desconocido"):
        build_snapshot([record(1, "Occupied")], FETCHED_AT)


def test_empty_feed_is_an_error() -> None:
    with pytest.raises(SnapshotError, match="0 registros"):
        build_snapshot([], FETCHED_AT)


def test_duplicate_kerbsideid_is_an_error() -> None:
    """`kerbsideid` es clave única en el feed real (6324/6324)."""
    with pytest.raises(SnapshotError, match="duplicados"):
        build_snapshot([record(1), record(1)], FETCHED_AT)


def test_missing_location_is_an_error() -> None:
    broken = record(1)
    broken["location"] = {"lon": 144.95}

    with pytest.raises(SnapshotError, match="location incompleta"):
        build_snapshot([broken], FETCHED_AT)


def test_naive_timestamp_is_an_error() -> None:
    with pytest.raises(SnapshotError, match="sin zona horaria"):
        build_snapshot([record(1, lastupdated="2026-10-22T09:26:50")], FETCHED_AT)


# --- rutas y escritura ------------------------------------------------------


def test_snapshot_path_is_derived_from_the_feed_stamp(settings: Settings) -> None:
    stamp = dt.datetime(2026, 10, 22, 9, 26, 50, tzinfo=dt.UTC)

    path = snapshot_path(settings, stamp)

    assert path.parent.name == "date=2026-10-22"
    assert path.name == "snapshot_20261022T092650Z.parquet"
    assert path.parent.parent == settings.realtime_raw_dir


def test_snapshot_path_partitions_in_utc(settings: Settings) -> None:
    """La partición no debe moverse con el horario de verano de Melbourne."""
    stamp = dt.datetime(2026, 10, 23, 9, 0, tzinfo=dt.timezone(dt.timedelta(hours=11)))

    path = snapshot_path(settings, stamp)

    assert path.parent.name == "date=2026-10-22"


def test_write_snapshot_roundtrips(settings: Settings) -> None:
    frame = build_snapshot([record(1, "Present"), record(2)], FETCHED_AT)
    path = snapshot_path(settings, FETCHED_AT)

    assert write_snapshot(frame, path) is True
    assert pl.read_parquet(path).equals(frame)


def test_write_snapshot_is_idempotent(settings: Settings) -> None:
    frame = build_snapshot([record(1)], FETCHED_AT)
    path = snapshot_path(settings, FETCHED_AT)
    write_snapshot(frame, path)
    mtime = path.stat().st_mtime_ns

    assert write_snapshot(frame, path) is False
    assert path.stat().st_mtime_ns == mtime, "no debe reescribir el fichero"


def test_write_snapshot_leaves_no_temp_files(settings: Settings) -> None:
    frame = build_snapshot([record(1)], FETCHED_AT)
    path = snapshot_path(settings, FETCHED_AT)

    write_snapshot(frame, path)

    assert [p.name for p in path.parent.iterdir()] == [path.name]


# --- capa HTTP --------------------------------------------------------------


def test_export_url_uses_the_verified_endpoint(settings: Settings) -> None:
    assert export_url(settings) == (
        "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
        "/on-street-parking-bay-sensors/exports/json"
    )


@respx.mock
def test_capture_snapshot_writes_once_per_feed_batch(settings: Settings) -> None:
    """Dos sondeos del mismo lote producen un solo fichero."""
    respx.get(export_url(settings)).respond(json=[record(1, "Present"), record(2)])

    with httpx.Client() as client:
        first = capture_snapshot(client, settings, now=FETCHED_AT)
        second = capture_snapshot(client, settings, now=FETCHED_AT + dt.timedelta(minutes=5))

    assert first.written is True
    assert second.skipped_as_duplicate is True
    assert first.path == second.path
    assert list(settings.realtime_raw_dir.rglob("*.parquet")) == [first.path]


@respx.mock
def test_capture_snapshot_counts_live_rows(settings: Settings) -> None:
    respx.get(export_url(settings)).respond(
        json=[
            record(1),
            record(2),
            record(3, lastupdated="2025-10-30T02:45:40+00:00"),
        ]
    )

    with httpx.Client() as client:
        result = capture_snapshot(client, settings, now=FETCHED_AT)

    assert (result.rows, result.live_rows) == (3, 2)


@respx.mock
def test_fetch_records_retries_transient_failures(settings: Settings) -> None:
    route = respx.get(export_url(settings)).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=[record(1)]),
        ]
    )

    with httpx.Client() as client:
        records = fetch_records(client, Settings(_env_file=None, max_retries=2))

    assert route.call_count == 2
    assert len(records) == 1


@respx.mock
def test_fetch_records_does_not_retry_client_errors(settings: Settings) -> None:
    """Un 404 no se arregla repitiendo: solo gastaría cuota."""
    route = respx.get(export_url(settings)).respond(404)

    with httpx.Client() as client, pytest.raises(SnapshotError, match="HTTP 404"):
        fetch_records(client, settings)

    assert route.call_count == 1


@respx.mock
def test_fetch_records_rejects_non_list_payload(settings: Settings) -> None:
    respx.get(export_url(settings)).respond(json={"error": "nope"})

    with httpx.Client() as client, pytest.raises(SnapshotError, match="lista de registros"):
        fetch_records(client, settings)
