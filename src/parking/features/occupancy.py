"""From stay events to a per-segment occupancy time series.

The historical archive is an **interval** schema: each row covers the period
between two transitions of one bay. The model needs the opposite, the **state**
of a segment at regular instants. This is that conversion.

**`VehiclePresent` decides what each interval is, and getting it wrong ruins
the dataset.** The source description says so: an event records whether a
vehicle was present or not, *"which can determine stay time and vacancy time"*.
That is:

- `VehiclePresent = true`  -> the bay was **occupied** during that interval
- `VehiclePresent = false` -> the bay was **free** during that interval

They are not stays plus noise: they are **two complementary partitions of
time**. Measured over the week of 1-8 March 2019, occupied intervals cover
43.7% of bay-time and vacant ones 47.9%; together, 91.6%. Treating every row as
a stay yields 91.6% occupancy -- flat across the day, because it is the sum of
occupied and free -- instead of the real 43.7%. That 43.7% also agrees with
what the real-time feed measures today.

The method avoids the obvious cartesian product (for each instant, count the
intervals covering it), which does not terminate over 42.7M rows. Instead:

1. Each occupied interval is split into two deltas: `+1` on arrival, `-1` on
   departure.
2. The deltas are sorted by time within each segment.
3. The **cumulative sum** gives the number of occupied bays after each event.
4. A `join_asof` carries that value onto the grid of instants.

Cost: one sort, not a join. Occupancy at any instant is the last cumulative sum
before it.
"""

from __future__ import annotations

import datetime as dt
import warnings

import polars as pl

#: Date format of the historical CSV: `11/03/2019 03:00:56 AM` is 3 November,
#: not 11 March. Declared explicitly because automatic inference succeeds on
#: days > 12 and fails silently on the rest.
US_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

#: Grid step. 5 min is the plan's shortest prediction horizon: a coarser grid
#: could not express the 5-minute target at all.
DEFAULT_GRID_MINUTES = 5


def parse_events(source: pl.LazyFrame) -> pl.LazyFrame:
    """Normalise the raw historical CSV into typed intervals.

    `StreetMarker` is the bay key, **not `DeviceId`**: 2019 has 5013 kerb
    markers but 6665 sensor identifiers, because a faulty sensor gets replaced
    while the physical bay stays the same. Aggregating by `DeviceId` would
    count one bay twice after every replacement.
    """
    return source.select(
        bay=pl.col("StreetMarker"),
        street=pl.col("StreetName"),
        cross_a=pl.col("BetweenStreet1"),
        cross_b=pl.col("BetweenStreet2"),
        vehicle_present=pl.col("VehiclePresent").cast(pl.Boolean),
        arrival=pl.col("ArrivalTime").str.strptime(
            pl.Datetime("us"), US_DATETIME_FORMAT, strict=False
        ),
        departure=pl.col("DepartureTime").str.strptime(
            pl.Datetime("us"), US_DATETIME_FORMAT, strict=False
        ),
    )


def clean_events(events: pl.LazyFrame) -> pl.LazyFrame:
    """Drop the intervals that cannot be interpreted at all.

    Over the 42.7M events of 2019 this removes:

    - **166 intervals whose departure precedes the arrival.** A negative
      interval would subtract occupancy where there was none, and the error
      would propagate through that segment's entire cumulative sum.
    - **21,046 zero-length intervals.** They contribute no occupancy on any
      5-minute grid and their two deltas cancel out.
    - Rows with an unparseable date or a missing street name, which cannot be
      placed in any segment.

    That is 0.05% of the total. They are dropped here and the figure is
    recorded: unlike a doubtful sensor reading, an interval with time running
    backwards admits no interpretation that keeping it would make more honest.
    """
    return events.filter(
        pl.col("arrival").is_not_null()
        & pl.col("departure").is_not_null()
        & (pl.col("departure") > pl.col("arrival"))
        & pl.col("bay").is_not_null()
        & pl.col("street").is_not_null()
        & pl.col("cross_a").is_not_null()
        & pl.col("cross_b").is_not_null()
        & pl.col("vehicle_present").is_not_null()
    )


def _normalise(column: str) -> pl.Expr:
    """Uppercase with collapsed spaces, same as `crosswalk.normalise_street`."""
    return pl.col(column).str.strip_chars().str.replace_all(r"\s+", " ").str.to_uppercase()


def with_segment_id(events: pl.LazyFrame) -> pl.LazyFrame:
    """Add `segment_id`, the street-segment key.

    The two bounding streets are sorted alphabetically before concatenation,
    because the same segment appears in both orders in the data. It is the same
    rule as `parking.ingest.crosswalk.segment_key`, expressed in polars here so
    that 42.7M rows never leave the engine.
    """
    street, first, second = _normalise("street"), _normalise("cross_a"), _normalise("cross_b")
    low = pl.min_horizontal(first, second)
    high = pl.max_horizontal(first, second)
    return events.with_columns(segment_id=pl.concat_str([street, low, high], separator="|")).drop(
        "street", "cross_a", "cross_b"
    )


def time_grid(
    start: dt.datetime, end: dt.datetime, *, minutes: int = DEFAULT_GRID_MINUTES
) -> pl.DataFrame:
    """Regular grid of instants over `[start, end)`."""
    return pl.DataFrame(
        {"timestamp": pl.datetime_range(start, end, f"{minutes}m", closed="left", eager=True)}
    )


def occupancy_series(
    events: pl.LazyFrame,
    start: dt.datetime,
    end: dt.datetime,
    *,
    minutes: int = DEFAULT_GRID_MINUTES,
) -> pl.DataFrame:
    """Per-segment occupancy at every instant of the grid `[start, end)`.

    `events` must already carry `segment_id`, `bay`, `arrival` and `departure`.

    Intervals that **overlap** the window are included, not only those starting
    inside it: a car parked before `start` still occupies a bay for the whole
    window, and filtering on `arrival >= start` would lose it and make the
    segment look emptier than it was.

    Returns `segment_id`, `timestamp`, `occupied_bays`, `total_bays` and
    `free_bays`.

    The two counts deliberately use different subsets:

    - `occupied_bays` comes **only** from intervals with `vehicle_present`.
    - `total_bays` counts bays seen in **either** state. A bay that was free
      for the whole window produces no occupied interval, but it exists and
      belongs in the denominator; deriving the inventory from occupied
      intervals alone would lose it precisely in the emptiest segments, which
      are the interesting ones.

    `total_bays` is still the **observed** bay count: a bay with no interval of
    either kind is invisible, and indistinguishable from a dead sensor. That is
    this reconstruction's main limitation and it is documented in
    `docs/dataset.md`.
    """
    overlapping = events.filter((pl.col("arrival") < end) & (pl.col("departure") > start))
    occupied = overlapping.filter(pl.col("vehicle_present"))

    arrivals = occupied.select("segment_id", t=pl.col("arrival"), delta=pl.lit(1, pl.Int32))
    departures = occupied.select("segment_id", t=pl.col("departure"), delta=pl.lit(-1, pl.Int32))

    # The cumulative sum runs from the start of the delta stream, which
    # includes arrivals predating `start`: that way the value at the grid's
    # first instant already reflects the cars that were there beforehand.
    cumulative = (
        pl.concat([arrivals, departures])
        .sort("segment_id", "t")
        .with_columns(occupied_bays=pl.col("delta").cum_sum().over("segment_id"))
        .select("segment_id", "t", "occupied_bays")
        .collect()
    )

    totals = overlapping.group_by("segment_id").agg(total_bays=pl.col("bay").n_unique()).collect()

    grid = time_grid(start, end, minutes=minutes)
    segments = totals.select("segment_id")
    # Segment x instant product: this is the dataset's final shape, not an
    # intermediate step, so here the cross join is exactly what we want.
    scaffold = segments.join(grid, how="cross").sort("segment_id", "timestamp")

    # polars warns that it cannot verify sortedness when grouping with `by`.
    # Both sides were just sorted above, so the warning adds nothing here and
    # leaving it would print on every run.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        joined = scaffold.join_asof(
            cumulative.sort("t"),
            left_on="timestamp",
            right_on="t",
            by="segment_id",
            strategy="backward",
        )

    series = (
        joined.with_columns(
            # With no prior event in the window, the segment was empty.
            occupied_bays=pl.col("occupied_bays").fill_null(0).clip(lower_bound=0)
        )
        .join(totals, on="segment_id", how="left")
        .with_columns(
            free_bays=(pl.col("total_bays") - pl.col("occupied_bays")).clip(lower_bound=0)
        )
        .drop("t")
    )
    return series.sort("segment_id", "timestamp")
