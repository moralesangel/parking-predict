"""Tests for `parking.ingest.poller`.

What matters here is resilience: a failing cycle must not bring the loop down,
because a 2-minute hole in the series is far cheaper than a dead poller nobody
restarts.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
import respx

from parking.config import Settings
from parking.ingest.poller import StopSignal, run, run_once
from parking.ingest.realtime import export_url

FEED_STAMP = "2026-10-22T09:26:50+00:00"


def record(kerbsideid: int, *, lastupdated: str = FEED_STAMP) -> dict:
    return {
        "lastupdated": lastupdated,
        "status_timestamp": "2026-10-22T08:00:00+00:00",
        "zone_number": 7219,
        "status_description": "Unoccupied",
        "kerbsideid": kerbsideid,
        "location": {"lon": 144.95, "lat": -37.82},
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # 60 is the minimum Settings accepts; the tests use max_cycles so they
    # never actually wait for it.
    return Settings(_env_file=None, data_dir=tmp_path, poll_interval_seconds=60, max_retries=0)


# --- run_once ---------------------------------------------------------------


@respx.mock
def test_run_once_reports_success(settings: Settings) -> None:
    respx.get(export_url(settings)).respond(json=[record(1)])

    with httpx.Client() as client:
        assert run_once(client, settings) is True

    assert len(list(settings.realtime_raw_dir.rglob("*.parquet"))) == 1


@respx.mock
def test_run_once_survives_a_bad_payload(settings: Settings) -> None:
    """An unknown state is a data error: logged, not propagated."""
    broken = record(1) | {"status_description": "Vandalised"}
    respx.get(export_url(settings)).respond(json=[broken])

    with httpx.Client() as client:
        assert run_once(client, settings) is False


@respx.mock
def test_run_once_survives_network_failure(settings: Settings) -> None:
    respx.get(export_url(settings)).mock(side_effect=httpx.ConnectError("no network"))

    with httpx.Client() as client:
        assert run_once(client, settings) is False


@respx.mock
def test_run_once_survives_server_error(settings: Settings) -> None:
    respx.get(export_url(settings)).respond(503)

    with httpx.Client() as client:
        assert run_once(client, settings) is False


# --- loop -------------------------------------------------------------------


@respx.mock
def test_run_stops_after_max_cycles(settings: Settings) -> None:
    route = respx.get(export_url(settings)).respond(json=[record(1)])

    failures = run(settings, max_cycles=3, interval_seconds=0)

    assert route.call_count == 3
    assert failures == 0


@respx.mock
def test_run_does_not_sleep_after_the_last_cycle(settings: Settings) -> None:
    """After the final cycle it must exit without waiting out the interval."""
    respx.get(export_url(settings)).respond(json=[record(1)])

    started = time.monotonic()
    run(settings, max_cycles=1, interval_seconds=120)

    assert time.monotonic() - started < 10


@respx.mock
def test_run_keeps_going_after_a_failure(settings: Settings) -> None:
    """This is the central property: the loop survives a failed cycle."""
    route = respx.get(export_url(settings)).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=[record(1)]),
            httpx.Response(200, json=[record(1)]),
        ]
    )

    failures = run(settings, max_cycles=3, interval_seconds=0)

    assert route.call_count == 3
    assert failures == 0, "the counter resets once a cycle succeeds again"


@respx.mock
def test_run_counts_consecutive_failures(settings: Settings) -> None:
    respx.get(export_url(settings)).respond(500)

    assert run(settings, max_cycles=2, interval_seconds=0) == 2


@respx.mock
def test_run_exits_when_stop_is_requested(settings: Settings) -> None:
    route = respx.get(export_url(settings)).respond(json=[record(1)])
    stop = StopSignal()
    stop._event.set()

    run(settings, max_cycles=5, stop=stop, interval_seconds=0)

    assert route.call_count == 0, "no cycle should start once a stop was requested"


# --- StopSignal -------------------------------------------------------------


def test_stop_signal_starts_unset() -> None:
    assert StopSignal().requested is False


def test_stop_signal_is_set_by_request() -> None:
    import signal

    stop = StopSignal()
    stop.request(signal.SIGTERM, None)

    assert stop.requested is True


def test_stop_signal_wait_returns_immediately_once_set() -> None:
    import signal

    stop = StopSignal()
    stop.request(signal.SIGINT, None)

    started = time.monotonic()
    assert stop.wait(30) is True
    assert time.monotonic() - started < 1
