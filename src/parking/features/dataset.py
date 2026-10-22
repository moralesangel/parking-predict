"""Construction of the training dataset.

Defines the phase 4 prediction problem:

> given a segment's state at `t`, what is the probability that **at least one
> bay is free** at `t + h`, for `h` in {5, 10, 15, 30} minutes?

Decisions that shape everything downstream:

- **The target is measured at `t + h`, not at `t`.** Obvious as it sounds, it
  is the whole point of the project: the baseline to beat is persistence,
  which answers with the current state.
- **One target per horizon**, in separate columns (`target_h5`, `target_h10`,
  ...). That allows training and evaluating each horizon independently, which
  is how phase 5 asks for the results to be broken down.
- **The future `free_bays` count is stored too** (`free_bays_h5`, ...). That is
  the alternative the plan requires evaluating: predicting the count and
  deriving the probability from it instead of classifying directly.
- **No feature looks into the future.** Every one is computed from information
  available at `t`. That is the mistake that inflates metrics without anything
  failing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

#: Prediction horizons in minutes, per the plan.
DEFAULT_HORIZONS = (5, 10, 15, 30)

#: Lags in minutes for the recent-history features.
DEFAULT_LAGS = (5, 10, 15, 30, 60)

#: Victoria (Australia) public holidays in 2019 -- Melbourne's state. Taken
#: from the public holiday register. **They should be verified against the
#: official source before drawing conclusions from phase 5**: a wrong holiday
#: breaks nothing, it only adds noise to one feature. The movable ones carry
#: the most risk (the Friday before the AFL Grand Final, and Melbourne Cup).
VICTORIA_HOLIDAYS_2019: tuple[dt.date, ...] = (
    dt.date(2019, 1, 1),  # New Year's Day
    dt.date(2019, 1, 28),  # Australia Day (observed: the 26th fell on a Saturday)
    dt.date(2019, 3, 11),  # Labour Day
    dt.date(2019, 4, 19),  # Good Friday
    dt.date(2019, 4, 20),  # Easter Saturday
    dt.date(2019, 4, 21),  # Easter Sunday
    dt.date(2019, 4, 22),  # Easter Monday
    dt.date(2019, 4, 25),  # ANZAC Day
    dt.date(2019, 6, 10),  # Queen's Birthday
    dt.date(2019, 9, 27),  # Friday before the AFL Grand Final
    dt.date(2019, 11, 5),  # Melbourne Cup
    dt.date(2019, 12, 25),  # Christmas Day
    dt.date(2019, 12, 26),  # Boxing Day
)


@dataclass(frozen=True)
class TemporalSplit:
    """Temporal partition of the dataset.

    Never random. A random split would put instant `t+5` in training and `t` in
    test, and those are overwhelmingly correlated: the model would look
    excellent and be unable to predict the actual future. With time series the
    only honest cut is by time.
    """

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    train_end: dt.datetime
    validation_end: dt.datetime

    def summary(self) -> pl.DataFrame:
        rows = []
        for name, frame in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            rows.append(
                {
                    "split": name,
                    "rows": frame.height,
                    "segments": frame["segment_id"].n_unique() if frame.height else 0,
                    "first": frame["timestamp"].min() if frame.height else None,
                    "last": frame["timestamp"].max() if frame.height else None,
                }
            )
        return pl.DataFrame(rows)


def add_time_features(frame: pl.DataFrame, *, holidays: tuple[dt.date, ...] = ()) -> pl.DataFrame:
    """Calendar features.

    `minute_of_day` is added as sine and cosine as well as raw: midnight is one
    minute away from 23:59, and a flat integer tells the model they are 1439
    apart. Trees can learn the wrap by splitting the range, but not for free.
    """
    minute_of_day = pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
    angle = 2 * 3.141592653589793 * minute_of_day / 1440
    return frame.with_columns(
        hour=pl.col("timestamp").dt.hour().cast(pl.Int8),
        minute_of_day=minute_of_day.cast(pl.Int16),
        minute_sin=angle.sin(),
        minute_cos=angle.cos(),
        day_of_week=pl.col("timestamp").dt.weekday().cast(pl.Int8),  # 1 = Monday
        is_weekend=(pl.col("timestamp").dt.weekday() >= 6),
        is_holiday=pl.col("timestamp").dt.date().is_in(list(holidays)),
    )


def add_lag_features(
    frame: pl.DataFrame,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    grid_minutes: int = 5,
) -> pl.DataFrame:
    """Recent history per segment: lags, trends and rolling means.

    Lags are expressed in minutes and translated into a row count using
    `grid_minutes`. That **assumes a gap-free grid**, which is what
    `occupancy.occupancy_series` produces: every segment gets one row per
    instant, even with no events. Over a grid with holes, `shift` would jump to
    an arbitrary instant and the feature would lie silently.
    """
    expressions: list[pl.Expr] = []
    for lag in lags:
        steps = lag // grid_minutes
        expressions.append(
            pl.col("free_bays").shift(steps).over("segment_id").alias(f"free_bays_lag{lag}")
        )
        expressions.append(
            pl.col("occupancy_rate")
            .shift(steps)
            .over("segment_id")
            .alias(f"occupancy_rate_lag{lag}")
        )

    frame = frame.with_columns(expressions)

    # Trends: how much has changed since N minutes ago. A segment that is
    # emptying and one that is filling can show the same occupancy now and
    # opposite futures.
    trends = [
        (pl.col("free_bays") - pl.col(f"free_bays_lag{lag}")).alias(f"free_bays_trend{lag}")
        for lag in lags
    ]
    rolling = [
        pl.col("free_bays")
        .rolling_mean(window_size=window // grid_minutes, min_samples=1)
        .over("segment_id")
        .alias(f"free_bays_mean{window}")
        for window in (15, 30, 60)
    ]
    return frame.with_columns(trends + rolling)


def add_neighbour_features(
    frame: pl.DataFrame, adjacency: dict[str, frozenset[str]]
) -> pl.DataFrame:
    """Aggregated occupancy of neighbouring segments at the same instant.

    This is legitimate information at `t`: the state of the next block is known
    now, not in the future. And it is what distinguishes "this zone is full but
    the next one is empty" from "the whole neighbourhood is full", which lead
    to different decisions.

    Segments with no known neighbours get nulls, not zeros: a zero would say
    "the neighbours are full", which is a claim rather than an absence.
    """
    pairs = [
        {"segment_id": segment, "neighbour_id": neighbour}
        for segment, neighbours in adjacency.items()
        for neighbour in neighbours
    ]
    if not pairs:
        return frame.with_columns(
            neighbour_free_bays=pl.lit(None, pl.Int64),
            neighbour_occupancy_rate=pl.lit(None, pl.Float64),
            neighbour_count=pl.lit(0, pl.UInt32),
        )

    edges = pl.DataFrame(pairs)
    neighbour_state = frame.select(
        neighbour_id="segment_id",
        timestamp="timestamp",
        neighbour_free="free_bays",
        neighbour_rate="occupancy_rate",
    )
    aggregated = (
        edges.join(neighbour_state, on="neighbour_id", how="inner")
        .group_by("segment_id", "timestamp")
        .agg(
            neighbour_free_bays=pl.col("neighbour_free").sum(),
            neighbour_occupancy_rate=pl.col("neighbour_rate").mean(),
            neighbour_count=pl.len(),
        )
    )
    return frame.join(aggregated, on=["segment_id", "timestamp"], how="left")


def add_targets(
    frame: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    grid_minutes: int = 5,
) -> pl.DataFrame:
    """Targets per horizon: free bays at `t+h` and whether at least one exists.

    Both are stored. `target_hN` is the classification the plan asks for;
    `free_bays_hN` enables the alternative of predicting the count and deriving
    the probability, which phase 5 must compare.

    The last rows of each segment end up with a null target because their
    future falls outside the window. They are marked, not filled: imputing them
    would invent labels.
    """
    expressions: list[pl.Expr] = []
    for horizon in horizons:
        steps = horizon // grid_minutes
        future = pl.col("free_bays").shift(-steps).over("segment_id")
        expressions.append(future.alias(f"free_bays_h{horizon}"))
        expressions.append((future >= 1).alias(f"target_h{horizon}"))
    return frame.with_columns(expressions)


def build_dataset(
    occupancy: pl.DataFrame,
    adjacency: dict[str, frozenset[str]],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    grid_minutes: int = 5,
    holidays: tuple[dt.date, ...] = VICTORIA_HOLIDAYS_2019,
) -> pl.DataFrame:
    """Assemble the full dataset from the occupancy series.

    Order matters: `occupancy_rate` first, since lags and neighbour features
    depend on it; targets last, since they look into the future and must be
    built on the already complete grid.
    """
    frame = occupancy.sort("segment_id", "timestamp").with_columns(
        occupancy_rate=pl.when(pl.col("total_bays") > 0)
        .then(pl.col("occupied_bays") / pl.col("total_bays"))
        .otherwise(None)
    )
    frame = add_time_features(frame, holidays=holidays)
    frame = add_lag_features(frame, lags=lags, grid_minutes=grid_minutes)
    frame = add_neighbour_features(frame, adjacency)
    frame = add_targets(frame, horizons=horizons, grid_minutes=grid_minutes)
    return frame.sort("segment_id", "timestamp")


def temporal_split(
    frame: pl.DataFrame, *, train_end: dt.datetime, validation_end: dt.datetime
) -> TemporalSplit:
    """Split the dataset by time. Never at random.

    The cuts are on `timestamp`, so every validation and test row is later than
    all of training.

    Caveat phase 5 must keep in mind: with horizon `h`, the label of the last
    training row comes from `train_end + h`, which already falls inside
    validation. That is an `h`-minute leak at the boundary. With `h <= 30 min`
    over months of data it is negligible, but if the splits get shorter the
    last `h` rows of each block must be dropped.
    """
    if validation_end <= train_end:
        raise ValueError("validation_end must be later than train_end")

    return TemporalSplit(
        train=frame.filter(pl.col("timestamp") < train_end),
        validation=frame.filter(
            (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < validation_end)
        ),
        test=frame.filter(pl.col("timestamp") >= validation_end),
        train_end=train_end,
        validation_end=validation_end,
    )


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Columns usable as model input.

    Excludes the identifiers, the timestamp and **every** target column. That
    this list is derived from the data rather than hand-written is what stops a
    future column from sneaking in as a feature when a new horizon is added.
    """
    excluded = {"segment_id", "timestamp"}
    return [
        name
        for name in frame.columns
        if name not in excluded
        and not name.startswith("target_h")
        and not name.startswith("free_bays_h")
    ]
