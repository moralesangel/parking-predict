"""Reliability classification for each sensor reading.

The feed fails in two distinct ways, and conflating them biases every
occupancy metric:

1. **Offline sensor.** It is absent from the feed's current batch. Detected
   exactly: `lastupdated < max(lastupdated)` within the snapshot. These are
   dead network relays, and they fail in blocks (see `docs/data_schema.md`
   §1.2).

2. **Stuck sensor.** It *is* in the batch — the relay reports it — but its
   state has not changed in a very long time. The in-ground sensor is frozen.
   This is the more numerous failure and the one you cannot see by watching
   `lastupdated` alone: 1707 bays out of 5264 online, with a median of 150
   days without a state change.

Plan rule this module enforces structurally: **nothing is ever dropped
silently.** Every function *adds* columns; none filters rows. Consumers decide
what to do with unreliable readings, and coverage stays measurable precisely
because the bad rows are still there.
"""

from __future__ import annotations

import polars as pl

#: Reliability states of a reading. `pl.Enum` rather than free text so the set
#: stays closed: casting a value that is not listed here raises
#: `InvalidOperationError`, so an invented state cannot be written.
#: It does *not* protect against a typo when *reading*
#: (`col("quality") == "okay"` silently returns 0 rows); these constants do.
QUALITY_VALUES = ("ok", "stuck", "offline")
QUALITY_DTYPE = pl.Enum(QUALITY_VALUES)

#: Columns added by `annotate_quality`.
QUALITY_COLUMNS = ("status_age_hours", "quality", "is_reliable")


def annotate_quality(frame: pl.DataFrame, *, stuck_after_hours: float = 24.0) -> pl.DataFrame:
    """Add the quality diagnosis to a snapshot.

    Columns added:

    - `status_age_hours`: hours since the last state change, measured against
      the batch stamp (`feed_lastupdated`) rather than the current time, so
      reprocessing old data stays reproducible.
    - `quality`: `offline`, `stuck` or `ok`. Order matters: a bay that is out
      of the batch is marked offline even when it has also been unchanged for
      ages, because the root cause is the relay.
    - `is_reliable`: `True` only for `ok`. This is the column features and
      metrics must use.

    Drops no rows (plan rule: never delete silently).
    """
    missing = {"lastupdated", "feed_lastupdated", "status_timestamp", "is_live"} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"snapshot is missing columns: {sorted(missing)}")

    age_hours = (
        pl.col("feed_lastupdated") - pl.col("status_timestamp")
    ).dt.total_seconds() / 3600.0

    return (
        frame.with_columns(status_age_hours=age_hours)
        .with_columns(
            quality=pl.when(~pl.col("is_live"))
            .then(pl.lit("offline"))
            .when(pl.col("status_age_hours") > stuck_after_hours)
            .then(pl.lit("stuck"))
            .otherwise(pl.lit("ok"))
            .cast(QUALITY_DTYPE)
        )
        .with_columns(is_reliable=pl.col("quality") == pl.lit("ok").cast(QUALITY_DTYPE))
    )


def quality_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Counts and shares per quality state, for reports and logs."""
    if "quality" not in frame.columns:
        raise ValueError("call annotate_quality first")

    total = frame.height
    return (
        frame.group_by("quality")
        .agg(bays=pl.len(), occupied=pl.col("is_occupied").sum())
        .with_columns(
            share_pct=(pl.col("bays") / total * 100).round(1),
            occupancy_pct=(pl.col("occupied") / pl.col("bays") * 100).round(1),
        )
        .sort("quality")
    )


def occupancy(frame: pl.DataFrame, *, reliable_only: bool = True) -> float | None:
    """Mean occupancy, by default over reliable readings only.

    `reliable_only=False` exists so the bias of not filtering can be measured,
    and that bias is not constant: stuck readings carry a *frozen* occupancy
    (39.3% in the real feed) that does not follow the daily cycle, so the bias
    is proportional to how far true occupancy sits from that value. Measured on
    individual snapshots: +7.6 points at 10:04 UTC and -0.1 at 11:46. In other
    words the bias is worst at peak hours, exactly when prediction matters.

    Returns `None` when no usable reading is left, rather than 0.0: a zero
    would mean "everything free", which is the opposite of "unknown".
    """
    subset = frame.filter(pl.col("is_reliable")) if reliable_only else frame
    if subset.height == 0:
        return None
    return float(subset["is_occupied"].mean())
