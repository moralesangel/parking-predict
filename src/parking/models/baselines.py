"""The mandatory baselines, in the order the plan sets out.

The whole project exists to answer one question: is there any point predicting
when real-time data already exists? An honest answer needs a hard reference,
and that reference is **persistence**: answering with the current state.

Three are implemented, from weakest to most demanding:

1. `PersistenceBaseline` - the plan's, literally: if there is space now, there
   will be space.
2. `CalibratedPersistence` - the same idea, but returning the empirical
   probability that the state holds. This is the *fair* reference, and the hard
   one to beat.
3. `HistoricalMeanBaseline` - historical mean per segment, weekday and hour
   bucket.

All three share the same interface: `fit(frame, target)` and
`predict_proba(frame)`, which returns P(at least one free bay at `t+h`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

#: Probabilities are clipped before computing log loss. A baseline that answers
#: 0 or 1 and gets one case wrong would score infinite log loss, and comparing
#: infinities says nothing. The clip is part of the evaluation, not a fix to
#: the model: it is applied identically to all of them so the comparison stays
#: fair, and it is documented because it penalises the hard baselines.
PROBABILITY_EPSILON = 1e-6


def clip_probabilities(values: np.ndarray, epsilon: float = PROBABILITY_EPSILON) -> np.ndarray:
    """Clamp probabilities into `[epsilon, 1-epsilon]`."""
    return np.clip(values, epsilon, 1.0 - epsilon)


@dataclass
class PersistenceBaseline:
    """The state at `t+h` is the state now. The plan's reference.

    Returns 1.0 when the segment currently has at least one free bay and 0.0
    otherwise. It is deterministic: it learns nothing and `fit` is a no-op.

    It is brutally hard to beat at short horizons, and that is the point. If
    nobody improves on it at 5 minutes, the valid result is to say so.
    """

    def fit(self, frame: pl.DataFrame, target: str) -> PersistenceBaseline:
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        return (frame["free_bays"].to_numpy() >= 1).astype(np.float64)


@dataclass
class CalibratedPersistence:
    """Persistence returning the empirical probability that the state holds.

    Instead of answering 0 or 1, it answers "of the times a segment had space,
    in what fraction was there still space `h` minutes later?". That fraction
    is estimated **only from the training data**.

    This is the fair reference: it has exactly the same information as raw
    persistence - the current state, nothing more - but expressed as a
    probability, so the log-loss clip does not punish it. A model that cannot
    beat this one is adding nothing that was not already in the current state.
    """

    rates: dict[bool, float] = field(default_factory=dict)
    fallback: float = 0.5

    def fit(self, frame: pl.DataFrame, target: str) -> CalibratedPersistence:
        usable = frame.filter(pl.col(target).is_not_null())
        self.fallback = float(usable[target].mean())
        grouped = (
            usable.with_columns(has_space=pl.col("free_bays") >= 1)
            .group_by("has_space")
            .agg(rate=pl.col(target).mean())
        )
        self.rates = {
            bool(row["has_space"]): float(row["rate"]) for row in grouped.iter_rows(named=True)
        }
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        has_space = frame["free_bays"].to_numpy() >= 1
        with_space = self.rates.get(True, self.fallback)
        without = self.rates.get(False, self.fallback)
        return np.where(has_space, with_space, without)


@dataclass
class HistoricalMeanBaseline:
    """Historical mean per segment, weekday and hour bucket.

    It captures the regular pattern - this street is usually full on Tuesdays
    at 11 - but **completely ignores the current state**. It is the exact
    counterpoint to persistence, which uses only the current state and no
    pattern at all. Comparing the two says how much of the signal is routine
    and how much is the situation right now.

    `bucket_minutes` groups the hour into bands. With 60-minute bands and a
    year of data each (segment, weekday, bucket) combination holds ~52
    observations; dropping to 15 minutes would leave ~13 and the mean would be
    noise.

    Combinations unseen in training fall through a backoff hierarchy -
    segment+bucket, then segment, then global - rather than to an arbitrary
    0.5. A segment that is always full does not have a 50% probability just
    because its Tuesday is missing.
    """

    bucket_minutes: int = 60
    by_segment_day_bucket: dict[tuple[str, int, int], float] = field(default_factory=dict)
    by_segment_bucket: dict[tuple[str, int], float] = field(default_factory=dict)
    by_segment: dict[str, float] = field(default_factory=dict)
    global_rate: float = 0.5

    def _with_bucket(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            bucket=(pl.col("minute_of_day") // self.bucket_minutes).cast(pl.Int32)
        )

    def fit(self, frame: pl.DataFrame, target: str) -> HistoricalMeanBaseline:
        usable = self._with_bucket(frame.filter(pl.col(target).is_not_null()))
        self.global_rate = float(usable[target].mean())

        finest = usable.group_by("segment_id", "day_of_week", "bucket").agg(
            rate=pl.col(target).mean()
        )
        self.by_segment_day_bucket = {
            (row["segment_id"], int(row["day_of_week"]), int(row["bucket"])): float(row["rate"])
            for row in finest.iter_rows(named=True)
        }
        coarse = usable.group_by("segment_id", "bucket").agg(rate=pl.col(target).mean())
        self.by_segment_bucket = {
            (row["segment_id"], int(row["bucket"])): float(row["rate"])
            for row in coarse.iter_rows(named=True)
        }
        per_segment = usable.group_by("segment_id").agg(rate=pl.col(target).mean())
        self.by_segment = {
            row["segment_id"]: float(row["rate"]) for row in per_segment.iter_rows(named=True)
        }
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        prepared = self._with_bucket(frame)
        segments = prepared["segment_id"].to_list()
        days = prepared["day_of_week"].to_list()
        buckets = prepared["bucket"].to_list()

        predictions = np.empty(len(segments), dtype=np.float64)
        for index, (segment, day, bucket) in enumerate(zip(segments, days, buckets, strict=True)):
            value = self.by_segment_day_bucket.get((segment, int(day), int(bucket)))
            if value is None:
                value = self.by_segment_bucket.get((segment, int(bucket)))
            if value is None:
                value = self.by_segment.get(segment)
            predictions[index] = self.global_rate if value is None else value
        return predictions
