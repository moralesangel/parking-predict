"""Ingestion of the real-time sensor feed.

The source dataset is **overwritten** on every publication cycle: it only ever
exposes the current state. Preserving the time series is this module's job, and
that series is what the production model will be trained on (see
`docs/id_crosswalk.md` §5).

Two decisions come measured from the data rather than assumed (see
`docs/data_schema.md`):

- `/exports/json?limit=-1` is used: one request per snapshot, with coordinates
  already split into lon/lat. The Parquet export returns `location` as binary
  WKB and the paginated endpoint would need 64 requests.
- A snapshot's identity is `max(lastupdated)`, the feed's batch stamp, not the
  wall clock. See `snapshot_path`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from parking.config import Settings

logger = logging.getLogger(__name__)

#: `status_description` only ever takes these two values (verified: 6324/6324
#: rows). `Present` describes the CAR, not the bay: it means the bay is taken.
_OCCUPANCY_BY_STATUS = {"Present": True, "Unoccupied": False}

_UTC_TIMESTAMP = pl.Datetime(time_unit="us", time_zone="UTC")

#: Explicit snapshot schema. Pinned by hand instead of inferred so that a type
#: change upstream fails at write time, not weeks later at read time.
#: The last three fields are ours; they do not come from the source.
SNAPSHOT_SCHEMA: dict[str, pl.DataType] = {
    "kerbsideid": pl.Int64(),
    "is_occupied": pl.Boolean(),
    "status_description": pl.String(),
    "status_timestamp": _UTC_TIMESTAMP,
    "lastupdated": _UTC_TIMESTAMP,
    "zone_number": pl.Int64(),
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
    "feed_lastupdated": _UTC_TIMESTAMP,
    "is_live": pl.Boolean(),
    "fetched_at": _UTC_TIMESTAMP,
}


class SnapshotError(RuntimeError):
    """The feed returned something that cannot be treated as a valid snapshot."""


@dataclass(frozen=True)
class SnapshotResult:
    """Outcome of capturing one snapshot."""

    path: Path
    rows: int
    live_rows: int
    feed_lastupdated: dt.datetime
    written: bool

    @property
    def skipped_as_duplicate(self) -> bool:
        """True when the batch was already on disk and was not rewritten."""
        return not self.written


def export_url(settings: Settings) -> str:
    """URL of the full JSON export of the sensor dataset."""
    return (
        f"{settings.api_base_url}/api/explore/v2.1/catalog/datasets"
        f"/{settings.realtime_dataset}/exports/json"
    )


def _is_retryable(exc: BaseException) -> bool:
    """Retry only transient failures: network, timeouts and 5xx/429."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def fetch_records(client: httpx.Client, settings: Settings) -> list[dict]:
    """Fetch the feed's full export, retrying transient failures.

    A 4xx other than 429 is not retried: it means the request is wrong, and
    repeating it only burns quota.
    """

    @retry(
        stop=stop_after_attempt(settings.max_retries + 1),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _get() -> list[dict]:
        response = client.get(export_url(settings), params={"limit": -1})
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if not _is_retryable(exc):
                raise SnapshotError(
                    f"the feed answered HTTP {exc.response.status_code}; not retrying"
                ) from exc
            raise
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SnapshotError("the feed did not return valid JSON") from exc
        if not isinstance(payload, list):
            raise SnapshotError(f"expected a list of records, got {type(payload).__name__}")
        return payload

    return _get()


def _parse_timestamp(value: str | None, field: str, kerbsideid: object) -> dt.datetime:
    if not value:
        raise SnapshotError(f"empty {field} for kerbsideid={kerbsideid!r}")
    # The API returns ISO-8601 with an explicit offset (+00:00).
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise SnapshotError(f"{field} has no timezone for kerbsideid={kerbsideid!r}: {value!r}")
    return parsed.astimezone(dt.UTC)


def build_snapshot(records: list[dict], fetched_at: dt.datetime) -> pl.DataFrame:
    """Normalise raw feed records into a table matching `SNAPSHOT_SCHEMA`.

    Normalisation decisions:

    - `status_description` becomes the boolean `is_occupied`. An unknown value
      is an error, not a silent `None`: if the source adds a third state we
      need to find out.
    - `location` is split into `latitude`/`longitude`, which is what the
      frontend consumes and what allows joining by coordinates.
    - `feed_lastupdated = max(lastupdated)` and the `is_live` flag are
      computed. A bay whose `lastupdated` predates the batch stamp did not
      report in this cycle (`docs/data_schema.md` §1.2). The flag **marks**
      rather than filters: dropping them silently would make coverage
      impossible to measure.
    """
    if not records:
        raise SnapshotError("the feed returned 0 records")

    rows: list[dict] = []
    for record in records:
        kerbsideid = record.get("kerbsideid")
        if kerbsideid is None:
            raise SnapshotError(f"record without kerbsideid: {record!r}")

        status = record.get("status_description")
        if status not in _OCCUPANCY_BY_STATUS:
            raise SnapshotError(
                f"unknown status_description {status!r} for kerbsideid={kerbsideid!r}; "
                f"expected one of {sorted(_OCCUPANCY_BY_STATUS)}"
            )

        location = record.get("location") or {}
        latitude, longitude = location.get("lat"), location.get("lon")
        if latitude is None or longitude is None:
            raise SnapshotError(f"incomplete location for kerbsideid={kerbsideid!r}: {location!r}")

        rows.append(
            {
                "kerbsideid": int(kerbsideid),
                "is_occupied": _OCCUPANCY_BY_STATUS[status],
                "status_description": status,
                "status_timestamp": _parse_timestamp(
                    record.get("status_timestamp"), "status_timestamp", kerbsideid
                ),
                "lastupdated": _parse_timestamp(
                    record.get("lastupdated"), "lastupdated", kerbsideid
                ),
                "zone_number": None
                if record.get("zone_number") is None
                else int(record["zone_number"]),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )

    duplicates = len(rows) - len({row["kerbsideid"] for row in rows})
    if duplicates:
        raise SnapshotError(f"{duplicates} duplicated kerbsideid values in the snapshot")

    feed_lastupdated = max(row["lastupdated"] for row in rows)
    for row in rows:
        row["feed_lastupdated"] = feed_lastupdated
        row["is_live"] = row["lastupdated"] == feed_lastupdated
        row["fetched_at"] = fetched_at.astimezone(dt.UTC)

    return pl.from_dicts(rows, schema=SNAPSHOT_SCHEMA)


def snapshot_path(settings: Settings, feed_lastupdated: dt.datetime) -> Path:
    """Snapshot path, derived from the feed's batch stamp.

    Naming by `feed_lastupdated` rather than by wall clock is what gives
    idempotency: the feed publishes every ~2 min while the poller polls every
    2 min too, but a retry or a restart can re-read the same batch. With this
    path, that case rewrites the same cell instead of adding a duplicate that
    would later have to be de-duplicated during feature building.

    Partitioned as `date=YYYY-MM-DD` per the plan, in UTC so the partition does
    not shift with Melbourne's daylight saving.
    """
    stamp = feed_lastupdated.astimezone(dt.UTC)
    return (
        settings.realtime_raw_dir
        / f"date={stamp:%Y-%m-%d}"
        / f"snapshot_{stamp:%Y%m%dT%H%M%SZ}.parquet"
    )


def write_snapshot(frame: pl.DataFrame, path: Path, *, overwrite: bool = False) -> bool:
    """Write the snapshot. Returns `False` if it already existed and is kept.

    The write is atomic: a temporary file in the same directory is written and
    then renamed. That way a poller interrupted mid-write does not leave a
    truncated Parquet that breaks every later read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        logger.info("snapshot %s already exists; not rewriting", path.name)
        return False

    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def capture_snapshot(
    client: httpx.Client,
    settings: Settings,
    *,
    now: dt.datetime | None = None,
) -> SnapshotResult:
    """Capture one snapshot of the feed and persist it. Idempotent per batch."""
    fetched_at = now or dt.datetime.now(dt.UTC)
    records = fetch_records(client, settings)
    frame = build_snapshot(records, fetched_at)

    feed_lastupdated = frame["feed_lastupdated"].item(0)
    live_rows = int(frame["is_live"].sum())
    path = snapshot_path(settings, feed_lastupdated)
    written = write_snapshot(frame, path)

    logger.info(
        "snapshot batch=%s bays=%d live=%d (%.1f%%) -> %s%s",
        feed_lastupdated.isoformat(),
        frame.height,
        live_rows,
        live_rows / frame.height * 100,
        path.name,
        "" if written else " (duplicate, skipped)",
    )
    return SnapshotResult(
        path=path,
        rows=frame.height,
        live_rows=live_rows,
        feed_lastupdated=feed_lastupdated,
        written=written,
    )
