"""Tests for `parking.ingest.realtime`.

The sample records mirror the real feed's shape, including the cases observed
in production: a null `zone_number`, and bays whose `lastupdated` predates the
batch stamp.
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
    """`Present` describes the car, not the bay: the bay is taken."""
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
    """520 of 6324 real bays have no zone; that is a legitimate null, not an error."""
    frame = build_snapshot([record(1, zone_number=None)], FETCHED_AT)

    assert frame["zone_number"].to_list() == [None]


def test_stale_sensors_are_flagged_not_dropped() -> None:
    """A bay whose `lastupdated` predates the batch did not report this cycle."""
    records = [
        record(1, lastupdated=FEED_STAMP),
        record(2, lastupdated=FEED_STAMP),
        record(3, lastupdated="2025-10-30T02:45:40+00:00"),
    ]

    frame = build_snapshot(records, FETCHED_AT)

    assert frame.height == 3, "offline bays are flagged, never deleted"
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
    """The feed gives an explicit offset; everything is stored in UTC."""
    frame = build_snapshot(
        [record(1, lastupdated="2026-10-22T20:26:50+11:00", status_timestamp=FEED_STAMP)],
        FETCHED_AT,
    )

    assert frame["lastupdated"].to_list() == [dt.datetime(2026, 10, 22, 9, 26, 50, tzinfo=dt.UTC)]


def test_unknown_status_is_an_error() -> None:
    """A third state would change the target's meaning: we need to find out."""
    with pytest.raises(SnapshotError, match="unknown status_description"):
        build_snapshot([record(1, "Occupied")], FETCHED_AT)


def test_empty_feed_is_an_error() -> None:
    with pytest.raises(SnapshotError, match="0 records"):
        build_snapshot([], FETCHED_AT)


def test_duplicate_kerbsideid_is_an_error() -> None:
    """`kerbsideid` is a unique key in the real feed (6324/6324)."""
    with pytest.raises(SnapshotError, match="duplicated"):
        build_snapshot([record(1), record(1)], FETCHED_AT)


def test_missing_location_is_an_error() -> None:
    broken = record(1)
    broken["location"] = {"lon": 144.95}

    with pytest.raises(SnapshotError, match="incomplete location"):
        build_snapshot([broken], FETCHED_AT)


def test_naive_timestamp_is_an_error() -> None:
    with pytest.raises(SnapshotError, match="no timezone"):
        build_snapshot([record(1, lastupdated="2026-10-22T09:26:50")], FETCHED_AT)


# --- paths and writing ------------------------------------------------------


def test_snapshot_path_is_derived_from_the_feed_stamp(settings: Settings) -> None:
    stamp = dt.datetime(2026, 10, 22, 9, 26, 50, tzinfo=dt.UTC)

    path = snapshot_path(settings, stamp)

    assert path.parent.name == "date=2026-10-22"
    assert path.name == "snapshot_20261022T092650Z.parquet"
    assert path.parent.parent == settings.realtime_raw_dir


def test_snapshot_path_partitions_in_utc(settings: Settings) -> None:
    """The partition must not shift with Melbourne's daylight saving."""
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
    assert path.stat().st_mtime_ns == mtime, "the file must not be rewritten"


def test_write_snapshot_leaves_no_temp_files(settings: Settings) -> None:
    frame = build_snapshot([record(1)], FETCHED_AT)
    path = snapshot_path(settings, FETCHED_AT)

    write_snapshot(frame, path)

    assert [p.name for p in path.parent.iterdir()] == [path.name]


# --- HTTP layer -------------------------------------------------------------


def test_export_url_uses_the_verified_endpoint(settings: Settings) -> None:
    assert export_url(settings) == (
        "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
        "/on-street-parking-bay-sensors/exports/json"
    )


@respx.mock
def test_capture_snapshot_writes_once_per_feed_batch(settings: Settings) -> None:
    """Two polls of the same batch produce a single file."""
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
    """A 404 is not fixed by repeating: it would only burn quota."""
    route = respx.get(export_url(settings)).respond(404)

    with httpx.Client() as client, pytest.raises(SnapshotError, match="HTTP 404"):
        fetch_records(client, settings)

    assert route.call_count == 1


@respx.mock
def test_fetch_records_rejects_non_list_payload(settings: Settings) -> None:
    respx.get(export_url(settings)).respond(json={"error": "nope"})

    with httpx.Client() as client, pytest.raises(SnapshotError, match="list of records"):
        fetch_records(client, settings)
