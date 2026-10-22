"""Build the training dataset from the raw historical archive.

    uv run python scripts/build_dataset.py --months 2019-01:2019-12

Processes **month by month** and writes partitioned Parquet into
`data/processed/dataset/month=YYYY-MM/`. Chunking by month keeps memory
bounded: the full year is ~29M rows and does not fit at once.

Each month is computed with **padding** on both sides and trimmed afterwards.
Without that padding, the first rows of each month would have null lags and the
last ones null targets, and the model would see an artificial gap on the 1st of
every month.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import polars as pl

from parking.config import get_settings
from parking.features.dataset import (
    DEFAULT_HORIZONS,
    DEFAULT_LAGS,
    VICTORIA_HOLIDAYS_2019,
    build_dataset,
    feature_columns,
)
from parking.features.occupancy import (
    DEFAULT_GRID_MINUTES,
    clean_events,
    occupancy_series,
    parse_events,
    with_segment_id,
)
from parking.features.segments import build_adjacency, parse_segment_id, segment_id

logger = logging.getLogger("build_dataset")


def month_range(spec: str) -> list[tuple[dt.datetime, dt.datetime]]:
    """Turn `2019-01:2019-03` into each month's boundaries."""
    first, last = spec.split(":")
    start = dt.datetime.strptime(first, "%Y-%m")
    stop = dt.datetime.strptime(last, "%Y-%m")
    months: list[tuple[dt.datetime, dt.datetime]] = []
    current = start
    while current <= stop:
        following = (current.replace(day=28) + dt.timedelta(days=7)).replace(day=1)
        months.append((current, following))
        current = following
    return months


def adjacency_for(segment_ids: list[str]) -> dict[str, frozenset[str]]:
    """Adjacency graph over the segments present."""
    keys = [parse_segment_id(value) for value in segment_ids]
    return {
        segment_id(key): frozenset(segment_id(n) for n in neighbours)
        for key, neighbours in build_adjacency(keys).items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default="2019-01:2019-12", help="range YYYY-MM:YYYY-MM")
    parser.add_argument("--grid-minutes", type=int, default=DEFAULT_GRID_MINUTES)
    parser.add_argument(
        "--min-bays",
        type=int,
        default=3,
        help="drop segments with fewer observed bays: with 1 or 2, "
        "'at least one is free' is almost the state of a single bay",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    settings = get_settings()
    csv = settings.data_dir / "interim" / "sensor-data-2019.csv"
    if not csv.exists():
        parser.error(f"missing {csv}; extract the historical CSV from the ZIP first")

    events = with_segment_id(
        clean_events(parse_events(pl.scan_csv(csv, infer_schema_length=10000)))
    )
    out_root = settings.processed_dir / "dataset"
    out_root.mkdir(parents=True, exist_ok=True)

    # The padding covers the longest lag on the left and the longest horizon on
    # the right, so that trimming to the month leaves no null that is not real.
    pad_before = dt.timedelta(minutes=max(DEFAULT_LAGS))
    pad_after = dt.timedelta(minutes=max(DEFAULT_HORIZONS))

    total_rows = 0
    for start, end in month_range(args.months):
        label = f"{start:%Y-%m}"
        occupancy = occupancy_series(
            events, start - pad_before, end + pad_after, minutes=args.grid_minutes
        )
        occupancy = occupancy.filter(pl.col("total_bays") >= args.min_bays)
        if occupancy.is_empty():
            logger.warning("%s has no data; skipping", label)
            continue

        adjacency = adjacency_for(occupancy["segment_id"].unique().to_list())
        frame = build_dataset(
            occupancy,
            adjacency,
            horizons=DEFAULT_HORIZONS,
            lags=DEFAULT_LAGS,
            grid_minutes=args.grid_minutes,
            holidays=VICTORIA_HOLIDAYS_2019,
        )
        # Trim to the month: the padding rows have served their purpose.
        frame = frame.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

        destination = out_root / f"month={label}"
        destination.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(destination / "part.parquet", compression="zstd")
        total_rows += frame.height
        logger.info(
            "%s  %8d rows  %3d segments  target h5 = %.2f%% positive",
            label,
            frame.height,
            frame["segment_id"].n_unique(),
            frame["target_h5"].mean() * 100,
        )

    logger.info("total: %d rows in %s", total_rows, out_root)
    logger.info("features: %d", len(feature_columns(frame)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
