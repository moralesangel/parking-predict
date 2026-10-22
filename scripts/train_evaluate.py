"""Train the baselines and the model, and evaluate all four horizons.

    uv run python scripts/train_evaluate.py

Writes the tables into `data/processed/results/` and saves the models into
`data/processed/models/`. The figures in `docs/results.md` come from here.

Fixed temporal split, never random:

- training:   2019-01 .. 2019-09
- validation: 2019-10          (LightGBM early stopping)
- test:       2019-11 .. 2019-12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import polars as pl

from parking.config import get_settings
from parking.features.dataset import DEFAULT_HORIZONS, feature_columns
from parking.models.baselines import (
    CalibratedPersistence,
    HistoricalMeanBaseline,
    PersistenceBaseline,
)
from parking.models.evaluation import (
    breakdown,
    calibration_curve,
    comparison_table,
    evaluate,
    occupancy_bucket,
)
from parking.models.gbm import GradientBoostingModel

logger = logging.getLogger("train")

TRAIN_MONTHS = [f"2019-{m:02d}" for m in range(1, 10)]
VALIDATION_MONTHS = ["2019-10"]
TEST_MONTHS = ["2019-11", "2019-12"]


def load_months(root, months: list[str], *, every_nth: int = 1) -> pl.DataFrame:
    """Load the given monthly partitions.

    `every_nth` subsamples in time, keeping one row out of every N per segment.
    It is done **by position within each segment** rather than at random, so
    the subsampling neither breaks the temporal structure nor mixes segments.
    """
    paths = [root / f"month={month}" / "part.parquet" for month in months]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing partitions: {[str(p) for p in missing]}")

    frame = pl.read_parquet(paths)
    if every_nth > 1:
        frame = (
            frame.sort("segment_id", "timestamp")
            .with_columns(row=pl.int_range(pl.len()).over("segment_id"))
            .filter(pl.col("row") % every_nth == 0)
            .drop("row")
        )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-every-nth",
        type=int,
        default=4,
        help="temporal subsampling of the training set (4 = one row in four, 20 min)",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    settings = get_settings()
    root = settings.processed_dir / "dataset"
    out = settings.processed_dir / "results"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("loading data")
    train = load_months(root, TRAIN_MONTHS, every_nth=args.train_every_nth)
    validation = load_months(root, VALIDATION_MONTHS, every_nth=args.train_every_nth)
    test = load_months(root, TEST_MONTHS)
    logger.info(
        "train %d | validation %d | test %d rows", train.height, validation.height, test.height
    )

    features = feature_columns(train)
    test = test.with_columns(occupancy_bucket())

    summary_rows: list[dict] = []
    importances: dict[int, pl.DataFrame] = {}

    for horizon in args.horizons:
        target = f"target_h{horizon}"
        logger.info("--- horizon %d min ---", horizon)

        evaluable = test.filter(pl.col(target).is_not_null())
        y_true = evaluable[target].to_numpy().astype(np.float64)

        models = {
            "persistence": PersistenceBaseline(),
            "calibrated_persistence": CalibratedPersistence(),
            "historical_mean": HistoricalMeanBaseline(),
            "lightgbm": GradientBoostingModel(features=features),
        }

        predictions: dict[str, np.ndarray] = {}
        results: dict[str, dict[str, float]] = {}
        for name, model in models.items():
            if isinstance(model, GradientBoostingModel):
                model.fit(train, target, validation=validation)
                importances[horizon] = model.importances()
                model.save(settings.processed_dir / "models", f"lightgbm_h{horizon}")
            else:
                model.fit(train, target)
            probabilities = model.predict_proba(evaluable)
            predictions[name] = probabilities
            results[name] = evaluate(y_true, probabilities)
            logger.info(
                "  %-24s brier %.5f  logloss %.5f",
                name,
                results[name]["brier"],
                results[name]["log_loss"],
            )

        table = comparison_table(results).with_columns(horizon_minutes=pl.lit(horizon))
        table.write_csv(out / f"comparison_h{horizon}.csv")
        summary_rows.extend(table.to_dicts())

        for name, probabilities in predictions.items():
            calibration_curve(y_true, probabilities).write_csv(
                out / f"calibration_h{horizon}_{name}.csv"
            )

        breakdown(evaluable, y_true, predictions, by="hour").write_csv(
            out / f"by_hour_h{horizon}.csv"
        )
        breakdown(evaluable, y_true, predictions, by="occupancy_bucket").write_csv(
            out / f"by_occupancy_h{horizon}.csv"
        )

    summary = pl.DataFrame(summary_rows)
    summary.write_csv(out / "summary.csv")
    for horizon, frame in importances.items():
        frame.write_csv(out / f"importances_h{horizon}.csv")

    (out / "split.json").write_text(
        json.dumps(
            {
                "train_months": TRAIN_MONTHS,
                "validation_months": VALIDATION_MONTHS,
                "test_months": TEST_MONTHS,
                "train_every_nth": args.train_every_nth,
                "train_rows": train.height,
                "test_rows": test.height,
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("results in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
