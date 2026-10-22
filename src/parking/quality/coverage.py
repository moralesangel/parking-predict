"""Sensor coverage metrics by zone and by hour.

Coverage is the share of bays with a reliable reading. It matters for two
different reasons:

- **By zone**, it decides where prediction is possible at all. A zone with 3
  reliable bays out of 20 cannot support any confident "is there space?"
  answer, and the product should say "no data" instead of inventing a
  probability.
- **By hour**, it detects hourly degradation patterns. If coverage drops every
  night, any night-time metric is measured over a different sensor fleet than
  the daytime one, and comparing hours without correcting for that is wrong.
"""

from __future__ import annotations

import polars as pl


def coverage_by_zone(frame: pl.DataFrame, *, min_bays: int = 1) -> pl.DataFrame:
    """Coverage per `zone_number`.

    Bays without a zone (520 out of 6324 in the real feed) are grouped under a
    null zone rather than dropped: they are real bays, and their missing zone
    is itself a quality problem worth seeing.

    `min_bays` filters out zones too small for the percentage to mean anything.
    It filters the *report*, not the data.

    Accepts a single snapshot or a series of many. `bays` counts distinct bays
    and `readings` counts rows, which with N snapshots are N times as many:
    conflating them would inflate each zone's apparent size.
    """
    _require_quality(frame)
    return (
        frame.group_by("zone_number")
        .agg(
            bays=pl.col("kerbsideid").n_unique(),
            readings=pl.len(),
            reliable=pl.col("is_reliable").sum(),
            offline=(pl.col("quality") == "offline").sum(),
            stuck=(pl.col("quality") == "stuck").sum(),
        )
        .filter(pl.col("bays") >= min_bays)
        .with_columns(coverage_pct=(pl.col("reliable") / pl.col("readings") * 100).round(1))
        .sort("coverage_pct", "bays")
    )


def coverage_by_hour(frame: pl.DataFrame) -> pl.DataFrame:
    """Coverage per UTC hour of the batch stamp.

    Expects a frame with several snapshots concatenated (the series the poller
    accumulates). With a single snapshot it returns one row, which says nothing
    about hourly patterns: at least a full day is needed for that.
    """
    _require_quality(frame)
    return (
        frame.with_columns(hour_utc=pl.col("feed_lastupdated").dt.hour())
        .group_by("hour_utc")
        .agg(
            snapshots=pl.col("feed_lastupdated").n_unique(),
            readings=pl.len(),
            reliable=pl.col("is_reliable").sum(),
        )
        .with_columns(coverage_pct=(pl.col("reliable") / pl.col("readings") * 100).round(1))
        .sort("hour_utc")
    )


def flapping_sensors(frame: pl.DataFrame, *, min_snapshots: int = 3) -> pl.DataFrame:
    """Bays that alternate between reliable and unreliable along the series.

    A sensor that drops in and out of the batch is worse than one that is down
    for good: the dead one is excluded and that is that, while the flapping one
    pushes bays in and out of each zone's denominator, moving aggregate
    occupancy without a single car having moved. These are the candidates to
    exclude from features.
    """
    _require_quality(frame)
    return (
        # Temporal order is essential: `transitions` compares each reading with
        # the previous one, and over unsorted rows it would count phantom flips.
        frame.sort("feed_lastupdated")
        .group_by("kerbsideid")
        .agg(
            snapshots=pl.len(),
            reliable=pl.col("is_reliable").sum(),
            transitions=(pl.col("is_reliable") != pl.col("is_reliable").shift(1))
            .sum()
            .fill_null(0),
        )
        .filter((pl.col("snapshots") >= min_snapshots) & (pl.col("transitions") > 0))
        .with_columns(reliable_pct=(pl.col("reliable") / pl.col("snapshots") * 100).round(1))
        .sort("transitions", descending=True)
    )


def _require_quality(frame: pl.DataFrame) -> None:
    if "is_reliable" not in frame.columns:
        raise ValueError("call parking.quality.flags.annotate_quality first")
