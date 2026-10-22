"""Polling loop for the real-time feed.

Designed to run as a service (`python -m parking.ingest.poller`). A failing
cycle does not bring the loop down: it is logged and the next one is awaited,
because a 2-minute hole in the series is far cheaper than a dead poller.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
import threading
import time
from types import FrameType

import httpx

from parking.config import Settings, get_settings
from parking.ingest.realtime import SnapshotError, capture_snapshot

logger = logging.getLogger(__name__)


class StopSignal:
    """Graceful shutdown on SIGINT/SIGTERM.

    Docker sends SIGTERM when stopping a container. Without this the process
    would die mid-write; with it, the current cycle finishes and then exits.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self, signum: int, _frame: FrameType | None) -> None:
        logger.info("received %s; stopping after the current cycle", signal.Signals(signum).name)
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Wait `seconds` or until the signal arrives. True means stop."""
        return self._event.wait(seconds)

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request)


def run_once(client: httpx.Client, settings: Settings) -> bool:
    """One cycle. Returns `True` when the capture succeeded."""
    try:
        capture_snapshot(client, settings)
    except SnapshotError:
        # The feed returned something unexpected. That is a data problem, not a
        # network one, so it is logged with a traceback for schema review.
        logger.exception("invalid snapshot; skipping this cycle")
        return False
    except httpx.HTTPError:
        logger.exception("network failure after exhausting retries; skipping this cycle")
        return False
    except OSError:
        logger.exception("write failure; skipping this cycle")
        return False
    return True


def run(
    settings: Settings | None = None,
    *,
    max_cycles: int | None = None,
    stop: StopSignal | None = None,
    interval_seconds: float | None = None,
) -> int:
    """Poll until a stop signal arrives or `max_cycles` is reached.

    `max_cycles` and `interval_seconds` exist for tests and one-off runs; in
    service both are left as `None`, the loop is open-ended and the cadence
    comes from `Settings`, where it has a 60 s floor that protects the upstream
    API.
    """
    settings = settings or get_settings()
    stop = stop or StopSignal()
    interval = settings.poll_interval_seconds if interval_seconds is None else interval_seconds

    logger.info(
        "poller started: dataset=%s interval=%ds destination=%s",
        settings.realtime_dataset,
        interval,
        settings.realtime_raw_dir,
    )

    failures = 0
    cycles = 0
    with httpx.Client(timeout=settings.request_timeout_seconds) as client:
        while not stop.requested and (max_cycles is None or cycles < max_cycles):
            started = time.monotonic()
            if run_once(client, settings):
                failures = 0
            else:
                failures += 1
                logger.warning("consecutive failed cycles: %d", failures)
            cycles += 1

            if max_cycles is not None and cycles >= max_cycles:
                break
            # Subtract the cycle's own duration so the cadence does not drift.
            remaining = max(0.0, interval - (time.monotonic() - started))
            if stop.wait(remaining):
                break

    logger.info("poller stopped after %d cycles", cycles)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Poll the parking sensor feed.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="capture a single snapshot and exit (useful to check ingestion)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        stream=sys.stdout,
    )
    # UTC in the logs: the data is in UTC, and mixing zones while debugging a
    # timestamp mismatch is a free source of mistakes.
    logging.Formatter.converter = time.gmtime

    settings = get_settings()
    stop = StopSignal()
    stop.install()

    started = dt.datetime.now(dt.UTC)
    failures = run(settings, max_cycles=1 if args.once else None, stop=stop)
    logger.info("total runtime: %s", dt.datetime.now(dt.UTC) - started)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
