"""Prediction service: from the latest snapshot to a probability per zone.

Chains everything before it: reads the poller's snapshots, aggregates them by
segment, builds the same features the model was trained on, and applies it.

**Warning returned in every response:** the model is trained on the 2019
archive and applied to today's data. Why that is not production-grade is
documented in `docs/id_crosswalk.md` §5 - temporal drift and sensor fleet
renewal - and the consequence is that this phase is a **demonstration of the
full flow**, not a reliable service. The warning travels inside the JSON so it
does not get lost along the way.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from parking.config import Settings
from parking.features.live import (
    build_live_features,
    densify,
    latest_features,
    load_bay_coordinates,
    load_bay_segments,
    segment_occupancy,
)
from parking.models.gbm import GradientBoostingModel
from parking.quality.flags import annotate_quality
from parking.quality.gaps import load_series

logger = logging.getLogger(__name__)

#: Horizons with a trained model. An arbitrary ETA snaps to the nearest one.
AVAILABLE_HORIZONS = (5, 10, 15, 30)

#: Earth radius in metres, for the haversine distance.
_EARTH_RADIUS_M = 6_371_000.0

MODEL_CAVEAT = (
    "Model trained on the 2019 historical archive and applied to current data. "
    "This is a demonstration of the full flow, not a reliable service: see "
    "docs/id_crosswalk.md section 5."
)


class PredictionUnavailableError(RuntimeError):
    """Cannot predict: models, geometry or enough history are missing."""


@dataclass(frozen=True)
class SegmentPrediction:
    """Probability of finding a space in one segment."""

    segment_id: str
    probability: float
    free_bays: int
    total_bays: int
    latitude: float
    longitude: float
    distance_m: float
    is_interpolated: bool


def snap_horizon(eta_minutes: float) -> int:
    """Snap the ETA to the nearest trained horizon.

    It snaps rather than interpolating between models: interpolating two
    probabilities from different models produces a number no model predicted
    and whose calibration has never been measured.
    """
    return min(AVAILABLE_HORIZONS, key=lambda h: abs(h - eta_minutes))


def haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres. Vectorised over the second point."""
    phi1, phi2 = math.radians(lat1), np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


class PredictionService:
    """Computes and caches the live features, and applies the models.

    The cache is invalidated by the latest batch stamp: building the features
    walks the whole accumulated series, and doing that per request would not
    scale.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[int, GradientBoostingModel] = {}
        self._bay_segments: dict[int, str] | None = None
        self._centroids: pl.DataFrame | None = None
        self._cached_batch: dt.datetime | None = None
        self._cached_features: pl.DataFrame | None = None

    # --- resources loaded once ---

    @property
    def _reference_path(self) -> Path:
        return self._settings.data_dir / "reference" / "parking_bays.json"

    def bay_segments(self) -> dict[int, str]:
        if self._bay_segments is None:
            if not self._reference_path.exists():
                raise PredictionUnavailableError(
                    f"missing {self._reference_path}; run scripts/fetch_reference.py"
                )
            self._bay_segments = load_bay_segments(self._reference_path)
            logger.info("kerbsideid->segment map: %d bays", len(self._bay_segments))
        return self._bay_segments

    def centroids(self) -> pl.DataFrame:
        if self._centroids is None:
            self._centroids = load_bay_coordinates(self._reference_path)
        return self._centroids

    def model(self, horizon: int) -> GradientBoostingModel:
        if horizon not in self._models:
            directory = self._settings.processed_dir / "models"
            name = f"lightgbm_h{horizon}"
            if not (directory / f"{name}.txt").exists():
                raise PredictionUnavailableError(
                    f"missing model {name}; run scripts/train_evaluate.py"
                )
            self._models[horizon] = GradientBoostingModel.load(directory, name)
        return self._models[horizon]

    # --- live features ---

    def features(self) -> pl.DataFrame:
        """Last feature row per segment, cached by feed batch."""
        # The check comes BEFORE annotating: over an empty directory
        # `load_series` returns a frame with no columns and `annotate_quality`
        # would fail with a schema error instead of this message, which says
        # what to do.
        raw = load_series(self._settings.realtime_raw_dir)
        if raw.is_empty():
            raise PredictionUnavailableError(
                "no snapshots; start the poller (python -m parking.ingest.poller)"
            )
        series = annotate_quality(raw, stuck_after_hours=self._settings.stuck_after_hours)

        batch = series["feed_lastupdated"].max()
        if self._cached_batch == batch and self._cached_features is not None:
            return self._cached_features

        occupancy = segment_occupancy(series, self.bay_segments())
        if occupancy.is_empty():
            raise PredictionUnavailableError(
                "no reliable bay could be assigned to a street segment"
            )
        features = latest_features(build_live_features(densify(occupancy)))

        self._cached_batch = batch
        self._cached_features = features
        logger.info("live features rebuilt: %d segments (batch %s)", features.height, batch)
        return features

    # --- prediction ---

    def predict_near(
        self,
        latitude: float,
        longitude: float,
        eta_minutes: float,
        *,
        radius_m: float = 500.0,
        limit: int = 10,
    ) -> tuple[int, list[SegmentPrediction]]:
        """Segments within `radius_m` and their probability, nearest first.

        Returning **several segments rather than only the best one** is
        deliberate: if the product pointed at a single free bay, it would send
        everyone who asks to the same spot. A list of zones with their
        probabilities spreads the load and reflects the real uncertainty.
        """
        horizon = snap_horizon(eta_minutes)
        features = self.features()
        centroids = self.centroids()

        located = features.join(centroids.drop("bays"), on="segment_id", how="inner")
        if located.is_empty():
            raise PredictionUnavailableError("no segment with features has coordinates")

        distances = haversine_m(
            latitude,
            longitude,
            located["latitude"].to_numpy(),
            located["longitude"].to_numpy(),
        )
        located = located.with_columns(distance_m=pl.Series(distances)).filter(
            pl.col("distance_m") <= radius_m
        )
        if located.is_empty():
            return horizon, []

        located = located.sort("distance_m").head(limit)
        probabilities = self.model(horizon).predict_proba(located)

        return horizon, [
            SegmentPrediction(
                segment_id=row["segment_id"],
                probability=float(probability),
                free_bays=int(row["free_bays"]),
                total_bays=int(row["total_bays"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                distance_m=round(float(row["distance_m"]), 1),
                is_interpolated=bool(row["is_interpolated"]),
            )
            for row, probability in zip(located.iter_rows(named=True), probabilities, strict=True)
        ]

    def predict_all(self, eta_minutes: float) -> tuple[int, pl.DataFrame]:
        """Probability for every segment, for painting the whole map."""
        horizon = snap_horizon(eta_minutes)
        features = self.features()
        located = features.join(self.centroids().drop("bays"), on="segment_id", how="inner")
        if located.is_empty():
            raise PredictionUnavailableError("no segment with features has coordinates")
        probabilities = self.model(horizon).predict_proba(located)
        return horizon, located.with_columns(probability=pl.Series(probabilities))
