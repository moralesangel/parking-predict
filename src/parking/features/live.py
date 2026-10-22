"""Segment-level features built from the poller's snapshots.

The model was trained on the 2019 archive, whose schema is intervals. The
real-time feed is something else: a snapshot of the state every 2 minutes. This
module turns those snapshots into **exactly the same columns** as
`parking.features.dataset`, so the trained model can score them.

That this is possible at all rests on a phase 4 detail: `segment_id` **is not a
feature**. The model only looks at occupancy, recent history, neighbours and
calendar, so it scores a segment that did not exist in 2019 just as well. Of
the 278 live segments only 130 appear in the training dataset, and all 278 are
still scorable.

The bridge between the two worlds is `kerbsideid -> segment_id`, via the
geometry dataset (`docs/id_crosswalk.md`): it covers 84.9% of live bays.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import polars as pl

from parking.features.dataset import (
    DEFAULT_LAGS,
    VICTORIA_HOLIDAYS_2019,
    add_lag_features,
    add_neighbour_features,
    add_time_features,
)
from parking.features.segments import build_adjacency, parse_segment_id, segment_id
from parking.ingest.crosswalk import segment_key_from_description

logger = logging.getLogger(__name__)

#: Live grid step. The feed publishes every 2 min, but the model was trained
#: with lags expressed over a 5-minute grid, so the series is resampled to 5 so
#: that `free_bays_lag5` means the same thing on both sides.
LIVE_GRID_MINUTES = 5


def load_bay_segments(path: Path) -> dict[int, str]:
    """Map `kerbsideid -> segment_id` from the geometry export.

    Any bay whose segment description does not match the `X between A and B`
    pattern is dropped: with no segment there is nowhere to aggregate it, and
    putting it in an approximate one would contaminate that segment's counts.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    for row in rows:
        kerbsideid = row.get("kerbsideid")
        if kerbsideid is None or not str(kerbsideid).isdigit():
            continue
        key = segment_key_from_description(row.get("roadsegmentdescription"))
        if key is not None:
            mapping[int(kerbsideid)] = segment_id(key)
    return mapping


def load_bay_coordinates(path: Path) -> pl.DataFrame:
    """Centroid of each segment, so it can be looked up by latitude/longitude."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        key = segment_key_from_description(row.get("roadsegmentdescription"))
        if key is None or row.get("latitude") is None or row.get("longitude") is None:
            continue
        records.append(
            {
                "segment_id": segment_id(key),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
        )
    if not records:
        return pl.DataFrame(
            schema={"segment_id": pl.String, "latitude": pl.Float64, "longitude": pl.Float64}
        )
    return (
        pl.DataFrame(records)
        .group_by("segment_id")
        .agg(
            latitude=pl.col("latitude").mean(),
            longitude=pl.col("longitude").mean(),
            bays=pl.len(),
        )
    )


def segment_occupancy(series: pl.DataFrame, bay_segments: dict[int, str]) -> pl.DataFrame:
    """Aggregate the snapshots by segment and instant.

    Only **reliable** readings count. That is the phase 2 rule applied here: a
    bay with a frozen sensor is not a free bay, and adding it to the
    denominator would inflate the segment's apparent supply.

    The grid is bucketed to 5 minutes to match training. If two feed batches
    land in the same 5-minute window, the last one wins: it is the state at the
    end of the interval, which is what the archive represented.
    """
    if series.is_empty():
        return pl.DataFrame()

    mapping = pl.DataFrame(
        {"kerbsideid": list(bay_segments.keys()), "segment_id": list(bay_segments.values())}
    )
    bucketed = (
        series.filter(pl.col("is_reliable"))
        .join(mapping, on="kerbsideid", how="inner")
        .with_columns(timestamp=pl.col("feed_lastupdated").dt.truncate(f"{LIVE_GRID_MINUTES}m"))
    )
    if bucketed.is_empty():
        return pl.DataFrame()

    # Last batch within each 5-minute window, per bay.
    latest = (
        bucketed.sort("feed_lastupdated")
        .group_by("segment_id", "timestamp", "kerbsideid")
        .agg(is_occupied=pl.col("is_occupied").last())
    )
    return (
        latest.group_by("segment_id", "timestamp")
        .agg(
            total_bays=pl.len(),
            occupied_bays=pl.col("is_occupied").sum(),
        )
        .with_columns(free_bays=pl.col("total_bays") - pl.col("occupied_bays"))
        .sort("segment_id", "timestamp")
    )


def densify(occupancy: pl.DataFrame) -> pl.DataFrame:
    """Fill in missing instants so the lags do not lie.

    `add_lag_features` translates minutes into a row count, which **assumes a
    gap-free grid**. If the poller missed batches, `shift(1)` would jump to an
    arbitrary instant and `free_bays_lag5` would hold a value from half an hour
    ago with nothing signalling it.

    Gaps are forward-filled (the last known state) and flagged with
    `is_interpolated`, so consumers know that row was not observed.
    """
    if occupancy.is_empty():
        return occupancy

    start, end = occupancy["timestamp"].min(), occupancy["timestamp"].max()
    grid = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start, end, f"{LIVE_GRID_MINUTES}m", closed="both", eager=True
            )
        }
    )
    segments = occupancy.select("segment_id").unique()
    scaffold = segments.join(grid, how="cross")

    return (
        scaffold.join(occupancy, on=["segment_id", "timestamp"], how="left")
        .sort("segment_id", "timestamp")
        .with_columns(is_interpolated=pl.col("total_bays").is_null())
        .with_columns(
            [
                pl.col(column).forward_fill().over("segment_id")
                for column in ("total_bays", "occupied_bays", "free_bays")
            ]
        )
        .filter(pl.col("total_bays").is_not_null())
    )


def build_live_features(
    occupancy: pl.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    holidays: tuple[dt.date, ...] = VICTORIA_HOLIDAYS_2019,
) -> pl.DataFrame:
    """Build the same columns as the training dataset.

    No targets are added: live, the future does not exist yet. That is the only
    difference from `dataset.build_dataset`.
    """
    if occupancy.is_empty():
        return occupancy

    frame = occupancy.sort("segment_id", "timestamp").with_columns(
        occupancy_rate=pl.when(pl.col("total_bays") > 0)
        .then(pl.col("occupied_bays") / pl.col("total_bays"))
        .otherwise(None)
    )
    frame = add_time_features(frame, holidays=holidays)
    frame = add_lag_features(frame, lags=lags, grid_minutes=LIVE_GRID_MINUTES)

    adjacency = {
        segment_id(key): frozenset(segment_id(n) for n in neighbours)
        for key, neighbours in build_adjacency(
            [parse_segment_id(value) for value in frame["segment_id"].unique().to_list()]
        ).items()
    }
    return add_neighbour_features(frame, adjacency)


def latest_features(features: pl.DataFrame) -> pl.DataFrame:
    """Last row per segment: the present, which is where prediction starts."""
    if features.is_empty():
        return features
    return features.sort("timestamp").group_by("segment_id").last()
