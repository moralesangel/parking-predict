"""Backtest of the production model against what actually happened.

This is the monitoring piece that really answers "is it still working?". The
phase 5 metrics were measured on 2019; this measures them again on the data the
poller keeps accumulating, which is the only place the model meets the present.

It works because **the label arrives on its own**: the prediction made at `t`
for `t+h` can be checked as soon as the poller captures `t+h`. Nothing has to
be annotated by hand, only `h` minutes waited.

It is always compared against persistence computed on the same data. A Brier of
0.03 means nothing by itself; what matters is whether it is still below the
reference that would have cost zero effort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from parking.models.evaluation import brier_score, log_loss, skill_score


@dataclass(frozen=True)
class BacktestResult:
    """Model metrics against persistence over already-realised data."""

    horizon_minutes: int
    n: int
    base_rate: float
    model_brier: float
    persistence_brier: float
    model_log_loss: float

    @property
    def skill_vs_persistence(self) -> float:
        return skill_score(self.model_brier, self.persistence_brier)

    @property
    def is_degraded(self) -> bool:
        """True when the model no longer beats persistence.

        This is the alarm that matters: the moment the ML pipeline stops paying
        for itself.
        """
        return self.model_brier >= self.persistence_brier

    def as_dict(self) -> dict:
        return {
            "horizon_minutes": self.horizon_minutes,
            "n": self.n,
            "base_rate": round(self.base_rate, 4),
            "model_brier": round(self.model_brier, 5),
            "persistence_brier": round(self.persistence_brier, 5),
            "model_log_loss": round(self.model_log_loss, 5),
            "skill_vs_persistence": round(self.skill_vs_persistence, 4),
            "is_degraded": self.is_degraded,
        }


def realised_targets(
    features: pl.DataFrame, horizon_minutes: int, *, grid_minutes: int = 5
) -> pl.DataFrame:
    """Add the observed target: was there space `h` minutes later?

    This is exactly `dataset.add_targets`, applied to the live series after the
    fact. Rows whose `t+h` has not happened yet keep a null target and are
    dropped at evaluation time: they are the future, not missing data.
    """
    steps = horizon_minutes // grid_minutes
    return features.sort("segment_id", "timestamp").with_columns(
        realised_free_bays=pl.col("free_bays").shift(-steps).over("segment_id"),
        realised_target=(pl.col("free_bays").shift(-steps).over("segment_id") >= 1),
    )


def backtest(
    features: pl.DataFrame,
    probabilities: np.ndarray,
    horizon_minutes: int,
    *,
    grid_minutes: int = 5,
    exclude_interpolated: bool = True,
) -> BacktestResult | None:
    """Evaluate predictions against what actually happened.

    `features` must be the full series (not just the last row per segment) and
    `probabilities` the predictions aligned with it.

    `exclude_interpolated` drops the rows the gap-filling invented. Evaluating
    on them would measure the quality of the filling rather than the model: a
    row carried over from the previous instant makes persistence look perfect
    and the model look worse than it is.

    Returns `None` when nothing evaluable is left, rather than metrics over
    zero rows.
    """
    frame = features.with_columns(probability=pl.Series(probabilities))
    frame = realised_targets(frame, horizon_minutes, grid_minutes=grid_minutes)
    frame = frame.filter(pl.col("realised_target").is_not_null())
    if exclude_interpolated and "is_interpolated" in frame.columns:
        frame = frame.filter(~pl.col("is_interpolated"))
    if frame.is_empty():
        return None

    y_true = frame["realised_target"].to_numpy().astype(np.float64)
    model = frame["probability"].to_numpy()
    persistence = (frame["free_bays"].to_numpy() >= 1).astype(np.float64)

    return BacktestResult(
        horizon_minutes=horizon_minutes,
        n=frame.height,
        base_rate=float(np.mean(y_true)),
        model_brier=brier_score(y_true, model),
        persistence_brier=brier_score(y_true, persistence),
        model_log_loss=log_loss(y_true, model),
    )


def coverage_trend(series: pl.DataFrame, *, bucket: str = "1h") -> pl.DataFrame:
    """Sensor coverage over time.

    This is the third signal the plan asks for and the easiest to overlook: a
    model can look perfect while half the sensors go dark, because the metrics
    are computed over whatever is left. Falling coverage means the product
    covers less of the city, even if it is just as accurate.
    """
    if series.is_empty():
        return series
    return (
        series.with_columns(window=pl.col("feed_lastupdated").dt.truncate(bucket))
        .group_by("window")
        .agg(
            bays=pl.len(),
            reliable=pl.col("is_reliable").sum(),
            batches=pl.col("feed_lastupdated").n_unique(),
        )
        .with_columns(coverage_pct=(pl.col("reliable") / pl.col("bays") * 100).round(1))
        .sort("window")
    )
