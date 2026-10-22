"""Retraining job over the data the poller has accumulated.

    uv run python scripts/retrain.py --horizon 15

Meant to run on a schedule (cron, Task Scheduler or the compose service). It
does four things, in this order:

1. Build the dataset from the accumulated snapshots.
2. **Check there is enough data and refuse to train if there is not.**
3. Train on the older part and evaluate on the recent part (temporal split).
4. Promote the model **only if it beats persistence** and does not regress
   against what is already in production.

Step 2 matters as much as the rest. A job that trains on whatever is there
will, sooner or later, produce a model based on six hours of a Tuesday; one
that refuses and says so is the correct behaviour, not a failure.

A non-zero exit code only means the job **failed**. Not promoting is not a
failure: it is a decision, and it is recorded in the registry's history.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import numpy as np
import polars as pl

from parking.config import get_settings
from parking.features.dataset import add_targets, feature_columns
from parking.features.live import (
    LIVE_GRID_MINUTES,
    build_live_features,
    densify,
    load_bay_segments,
    segment_occupancy,
)
from parking.models.baselines import CalibratedPersistence
from parking.models.evaluation import brier_score
from parking.models.gbm import GradientBoostingModel
from parking.models.registry import ModelRegistry, decide
from parking.quality.flags import annotate_quality
from parking.quality.gaps import load_series

logger = logging.getLogger("retrain")

#: Minimums for retraining to mean anything. These are judgement calls, not
#: thresholds derived from the data, which is why they are parameters:
#: - 14 days covers two full weekly cycles. With less, the model has not seen
#:   a weekend often enough to tell it apart from noise.
#: - 50,000 labelled rows avoids training on a handful of segments during a
#:   feed outage.
MIN_DAYS = 14
MIN_LABELLED_ROWS = 50_000


def build_training_frame(settings) -> tuple[pl.DataFrame, dict]:
    """Dataset with targets, built from the accumulated snapshots."""
    raw = load_series(settings.realtime_raw_dir)
    if raw.is_empty():
        raise SystemExit("no snapshots; start the poller")

    series = annotate_quality(raw, stuck_after_hours=settings.stuck_after_hours)
    mapping = load_bay_segments(settings.data_dir / "reference" / "parking_bays.json")
    occupancy = segment_occupancy(series, mapping)
    if occupancy.is_empty():
        raise SystemExit("no reliable bay could be assigned to a segment")

    dense = densify(occupancy)
    features = build_live_features(dense)
    span = features["timestamp"].max() - features["timestamp"].min()
    return features, {
        "first": features["timestamp"].min(),
        "last": features["timestamp"].max(),
        "days": span.total_seconds() / 86400,
        "segments": features["segment_id"].n_unique(),
        "rows": features.height,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=15, choices=[5, 10, 15, 30])
    parser.add_argument("--min-days", type=float, default=MIN_DAYS)
    parser.add_argument("--min-rows", type=int, default=MIN_LABELLED_ROWS)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25,
        help="FINAL fraction of the window held out for evaluation",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    settings = get_settings()
    registry = ModelRegistry(settings.processed_dir / "registry")
    version = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")

    features, span = build_training_frame(settings)
    logger.info(
        "accumulated data: %.2f days | %d segments | %d rows (%s -> %s)",
        span["days"],
        span["segments"],
        span["rows"],
        span["first"],
        span["last"],
    )

    target = f"target_h{args.horizon}"
    labelled = add_targets(
        features, horizons=(args.horizon,), grid_minutes=LIVE_GRID_MINUTES
    ).filter(pl.col(target).is_not_null())

    # --- the "is there enough data" gate ---
    reasons = []
    if span["days"] < args.min_days:
        reasons.append(f"only {span['days']:.2f} days of data (minimum {args.min_days})")
    if labelled.height < args.min_rows:
        reasons.append(f"only {labelled.height} labelled rows (minimum {args.min_rows})")
    if reasons:
        reason = "insufficient data: " + "; ".join(reasons)
        logger.warning("NOT retraining. %s", reason)
        logger.warning(
            "This is the correct behaviour: a model trained on this window "
            "would not have seen enough weekly cycles."
        )
        registry.record(
            decide(float("inf"), 0.0, None),
            version=version,
            extra={
                "skipped": True,
                "reason_detail": reason,
                "span": {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in span.items()
                },
            },
        )
        return 0

    # --- temporal split ---
    cutoff = labelled["timestamp"].quantile(1 - args.holdout_fraction)
    train = labelled.filter(pl.col("timestamp") < cutoff)
    holdout = labelled.filter(pl.col("timestamp") >= cutoff)
    logger.info("train %d rows | holdout %d rows (cutoff %s)", train.height, holdout.height, cutoff)

    y_true = holdout[target].to_numpy().astype(np.float64)
    features_list = [c for c in feature_columns(train) if c != "is_interpolated"]

    candidate = GradientBoostingModel(features=features_list).fit(train, target)
    candidate_brier = brier_score(y_true, candidate.predict_proba(holdout))
    persistence_brier = brier_score(
        y_true, CalibratedPersistence().fit(train, target).predict_proba(holdout)
    )

    production = registry.production_metrics()
    production_brier = production.get("brier") if production else None

    decision = decide(candidate_brier, persistence_brier, production_brier)
    logger.info(
        "candidate %.5f | persistence %.5f | production %s",
        candidate_brier,
        persistence_brier,
        f"{production_brier:.5f}" if production_brier is not None else "none",
    )
    logger.info("%s: %s", "PROMOTING" if decision.promoted else "NOT promoting", decision.reason)

    staging = settings.processed_dir / "candidates" / version
    candidate.save(staging, f"lightgbm_h{args.horizon}")
    registry.record(decision, version=version, extra={"horizon_minutes": args.horizon})

    if decision.promoted:
        registry.promote(
            staging,
            version=version,
            metrics={
                "brier": candidate_brier,
                "persistence_brier": persistence_brier,
                "horizon_minutes": args.horizon,
                "trained_at": version,
                "train_rows": train.height,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
