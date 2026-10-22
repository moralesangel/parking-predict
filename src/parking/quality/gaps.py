"""Detection of gaps in the snapshot series.

The feed publishes a batch every 120 s. A longer jump between two consecutive
batches means data is missing, and that has to be known before computing lags
or trends: a "change over the last 10 minutes" feature computed across a
two-hour gap is not a trend, it is noise.

Telling the two causes apart matters, because they are fixed differently:

- **The poller did not capture** (crashed, lost network, restarted). Fixed by
  operating it.
- **The source did not publish** new batches. Cannot be fixed; the period has
  to be marked as unobserved.

Both look identical in the stored snapshots, but the second leaves a trace: if
the poller kept running and requesting, the repeated batch was discarded by
idempotency and no file exists. Separating them automatically would need the
poller's logs, so this module reports the gap without attributing a cause.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Gap:
    """A gap between two consecutive batches."""

    previous: dt.datetime
    following: dt.datetime
    missing_cycles: int

    @property
    def duration(self) -> dt.timedelta:
        return self.following - self.previous


def snapshot_timestamps(realtime_dir: Path) -> list[dt.datetime]:
    """Batch stamps of the snapshots on disk, sorted.

    Read from the file name rather than the contents: with thousands of
    snapshots, opening every Parquet to extract a date that is already in the
    path is wasted work. The name is `snapshot_YYYYMMDDTHHMMSSZ.parquet`.
    """
    stamps: list[dt.datetime] = []
    for path in sorted(realtime_dir.rglob("snapshot_*.parquet")):
        stem = path.stem.removeprefix("snapshot_")
        try:
            stamps.append(dt.datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC))
        except ValueError:
            continue  # foreign file: ignored rather than breaking the report
    return sorted(stamps)


def find_gaps(
    timestamps: list[dt.datetime],
    *,
    expected_interval_seconds: float = 120.0,
    tolerance_seconds: float = 360.0,
) -> list[Gap]:
    """Gaps between consecutive batches longer than `tolerance_seconds`.

    The default tolerance is three cycles: the feed occasionally runs late
    without anything being missing, and flagging every one-cycle delay as a gap
    would fill the report with false positives.
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
    """Continuity summary of the captured series.

    `completeness_pct` compares captured batches against those that fit in the
    observed window. It is the number that says whether the series is usable
    for training.
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
    """Load every snapshot as a single frame.

    Uses `scan_parquet` over the glob so the files are not materialised one by
    one. Returns an empty frame when there is nothing to load, so callers do
    not have to special-case that.
    """
    paths = sorted(realtime_dir.rglob("snapshot_*.parquet"))
    if not paths:
        return pl.DataFrame()
    lazy = pl.scan_parquet(paths)
    if columns is not None:
        lazy = lazy.select(columns)
    return lazy.collect()
