"""Access to the most recent snapshot captured by the poller.

The API never queries the source: it reads what the poller already stored. That
way an API outage does not interrupt capture, and a feed outage does not bring
the API down - it keeps serving the last known state **while saying how old it
is** (see `SnapshotMeta.age_seconds`, which the API exposes as `is_stale`).

That last part is a direct consequence of phase 2: if the poller dies, the API
would serve a stale snapshot indefinitely without anyone noticing. Publishing
the age turns a silent failure into a visible one.
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
    """No snapshot on disk: the poller has never run."""


@dataclass(frozen=True)
class SnapshotMeta:
    """Metadata of the snapshot being served."""

    feed_lastupdated: dt.datetime
    path: Path
    served_at: dt.datetime

    @property
    def age_seconds(self) -> float:
        return (self.served_at - self.feed_lastupdated).total_seconds()


@dataclass(frozen=True)
class LiveSnapshot:
    """A quality-annotated snapshot plus its metadata."""

    frame: pl.DataFrame
    meta: SnapshotMeta


class SnapshotRepository:
    """Reads and caches the most recent snapshot from disk.

    The poller writes a file every ~2 min, so re-reading the Parquet on every
    request would be pure waste. The cache is invalidated by path: the file
    name contains the batch stamp, so a new batch is a new path and there is no
    need to check `mtime` or set a TTL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cached: LiveSnapshot | None = None

    def latest_path(self) -> Path:
        """Path of the most recent snapshot, by batch stamp rather than `mtime`.

        Sorting by `mtime` would give the wrong order when restoring an archive
        or copying files between machines.
        """
        stamps = snapshot_timestamps(self._settings.realtime_raw_dir)
        if not stamps:
            raise NoSnapshotError(
                f"no snapshots in {self._settings.realtime_raw_dir}; "
                "start the poller (python -m parking.ingest.poller)"
            )
        newest = stamps[-1]
        return (
            self._settings.realtime_raw_dir
            / f"date={newest:%Y-%m-%d}"
            / f"snapshot_{newest:%Y%m%dT%H%M%SZ}.parquet"
        )

    def load(self, *, now: dt.datetime | None = None) -> LiveSnapshot:
        """Return the most recent snapshot, annotated with quality.

        Unreliable bays are not filtered out: the map needs to paint them as
        "no data", and omitting them would be lying by omission (phase 2 rule).
        """
        served_at = now or dt.datetime.now(dt.UTC)
        path = self.latest_path()

        if self._cached is not None and self._cached.meta.path == path:
            # Same batch: the frame is reused and only the serving time is
            # updated, which is what makes `age_seconds` grow.
            cached = self._cached
            return LiveSnapshot(
                frame=cached.frame,
                meta=SnapshotMeta(cached.meta.feed_lastupdated, path, served_at),
            )

        logger.info("loading snapshot %s", path.name)
        frame = annotate_quality(
            pl.read_parquet(path), stuck_after_hours=self._settings.stuck_after_hours
        )
        feed_lastupdated = frame["feed_lastupdated"].item(0)
        snapshot = LiveSnapshot(frame=frame, meta=SnapshotMeta(feed_lastupdated, path, served_at))
        self._cached = snapshot
        return snapshot

    def invalidate(self) -> None:
        """Forget the cache. For tests and manual reloads."""
        self._cached = None
