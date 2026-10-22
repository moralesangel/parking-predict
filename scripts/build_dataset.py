"""Construye el dataset de entrenamiento desde el histórico crudo.

    uv run python scripts/build_dataset.py --months 2019-01:2019-12

Procesa **mes a mes** y escribe Parquet particionado en
`data/processed/dataset/month=YYYY-MM/`. Trocear por meses mantiene la memoria
acotada: el año entero son ~32 M de filas y no cabe de una vez.

Cada mes se calcula con un **margen** a ambos lados y se recorta después. Sin
ese margen, las primeras filas de cada mes tendrían los retardos nulos y las
últimas el objetivo nulo, y el modelo vería un hueco artificial cada 1 de mes.
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
    """Convierte `2019-01:2019-03` en los límites de cada mes."""
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
    """Grafo de vecindad entre los tramos presentes."""
    keys = [parse_segment_id(value) for value in segment_ids]
    return {
        segment_id(key): frozenset(segment_id(n) for n in neighbours)
        for key, neighbours in build_adjacency(keys).items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default="2019-01:2019-12", help="rango YYYY-MM:YYYY-MM")
    parser.add_argument("--grid-minutes", type=int, default=DEFAULT_GRID_MINUTES)
    parser.add_argument(
        "--min-bays",
        type=int,
        default=3,
        help="descarta tramos con menos plazas observadas: con 1 o 2, "
        "'hay al menos una libre' es casi el estado de una sola plaza",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    settings = get_settings()
    csv = settings.data_dir / "interim" / "sensor-data-2019.csv"
    if not csv.exists():
        parser.error(f"falta {csv}; extrae antes el CSV del ZIP histórico")

    events = with_segment_id(
        clean_events(parse_events(pl.scan_csv(csv, infer_schema_length=10000)))
    )
    out_root = settings.processed_dir / "dataset"
    out_root.mkdir(parents=True, exist_ok=True)

    # El margen cubre el retardo más largo por la izquierda y el horizonte más
    # largo por la derecha, de modo que al recortar al mes no quede ningún
    # nulo que no sea real.
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
            logger.warning("%s sin datos; se omite", label)
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
        # Recorte al mes: las filas del margen ya cumplieron su papel.
        frame = frame.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))

        destination = out_root / f"month={label}"
        destination.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(destination / "part.parquet", compression="zstd")
        total_rows += frame.height
        logger.info(
            "%s  %8d filas  %3d tramos  objetivo h5 = %.2f%% positivos",
            label,
            frame.height,
            frame["segment_id"].n_unique(),
            frame["target_h5"].mean() * 100,
        )

    logger.info("total: %d filas en %s", total_rows, out_root)
    logger.info("features: %d", len(feature_columns(frame)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
