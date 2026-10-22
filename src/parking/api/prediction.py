"""Servicio de predicción: del snapshot más reciente a una probabilidad por zona.

Encadena todo lo anterior: lee los snapshots del poller, los agrega por tramo,
construye las mismas features con las que se entrenó el modelo y lo aplica.

**Advertencia que el endpoint devuelve en cada respuesta:** el modelo está
entrenado con el histórico de 2019 y se aplica a datos de hoy. Las razones por
las que eso no es apto para producción están en `docs/id_crosswalk.md` §5
—deriva temporal y renovación del parque de sensores— y la consecuencia es que
esta fase es una **demostración del flujo completo**, no un servicio fiable. La
advertencia viaja en el JSON para que no se pierda por el camino.
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

#: Horizontes con modelo entrenado. Un ETA cualquiera se ajusta al más cercano.
AVAILABLE_HORIZONS = (5, 10, 15, 30)

#: Radio de la Tierra en metros, para la distancia haversine.
_EARTH_RADIUS_M = 6_371_000.0

MODEL_CAVEAT = (
    "Modelo entrenado con el histórico de 2019 y aplicado a datos actuales. "
    "Es una demostración del flujo completo, no un servicio fiable: ver "
    "docs/id_crosswalk.md §5."
)


class PredictionUnavailableError(RuntimeError):
    """No se puede predecir: faltan modelos, geometría o historia suficiente."""


@dataclass(frozen=True)
class SegmentPrediction:
    """Probabilidad de encontrar sitio en un tramo."""

    segment_id: str
    probability: float
    free_bays: int
    total_bays: int
    latitude: float
    longitude: float
    distance_m: float
    is_interpolated: bool


def snap_horizon(eta_minutes: float) -> int:
    """Ajusta el ETA al horizonte entrenado más cercano.

    Se ajusta en lugar de interpolar entre modelos: interpolar dos
    probabilidades de modelos distintos produce un número que ningún modelo
    predijo y cuya calibración no se ha medido.
    """
    return min(AVAILABLE_HORIZONS, key=lambda h: abs(h - eta_minutes))


def haversine_m(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distancia en metros sobre la esfera. Vectorizada sobre el segundo punto."""
    phi1, phi2 = math.radians(lat1), np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


class PredictionService:
    """Calcula y cachea las features en vivo, y aplica los modelos.

    La caché se invalida por el sello del último lote: construir las features
    recorre toda la serie acumulada, y hacerlo en cada petición no escalaría.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[int, GradientBoostingModel] = {}
        self._bay_segments: dict[int, str] | None = None
        self._centroids: pl.DataFrame | None = None
        self._cached_batch: dt.datetime | None = None
        self._cached_features: pl.DataFrame | None = None

    # --- recursos cargados una vez ---

    @property
    def _reference_path(self) -> Path:
        return self._settings.data_dir / "reference" / "parking_bays.json"

    def bay_segments(self) -> dict[int, str]:
        if self._bay_segments is None:
            if not self._reference_path.exists():
                raise PredictionUnavailableError(
                    f"falta {self._reference_path}; ejecuta scripts/fetch_reference.py"
                )
            self._bay_segments = load_bay_segments(self._reference_path)
            logger.info("mapa kerbsideid->tramo: %d plazas", len(self._bay_segments))
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
                    f"falta el modelo {name}; ejecuta scripts/train_evaluate.py"
                )
            self._models[horizon] = GradientBoostingModel.load(directory, name)
        return self._models[horizon]

    # --- features en vivo ---

    def features(self) -> pl.DataFrame:
        """Última fila de features por tramo, cacheada por lote del feed."""
        # La comprobación va ANTES de anotar: sobre un directorio vacío,
        # `load_series` devuelve un frame sin columnas y `annotate_quality`
        # fallaría con un error de esquema en vez de con este mensaje, que
        # dice qué hacer.
        raw = load_series(self._settings.realtime_raw_dir)
        if raw.is_empty():
            raise PredictionUnavailableError(
                "no hay snapshots; arranca el poller (python -m parking.ingest.poller)"
            )
        series = annotate_quality(raw, stuck_after_hours=self._settings.stuck_after_hours)

        batch = series["feed_lastupdated"].max()
        if self._cached_batch == batch and self._cached_features is not None:
            return self._cached_features

        occupancy = segment_occupancy(series, self.bay_segments())
        if occupancy.is_empty():
            raise PredictionUnavailableError(
                "ninguna plaza fiable se pudo asignar a un tramo de calle"
            )
        features = latest_features(build_live_features(densify(occupancy)))

        self._cached_batch = batch
        self._cached_features = features
        logger.info("features en vivo recalculadas: %d tramos (lote %s)", features.height, batch)
        return features

    # --- predicción ---

    def predict_near(
        self,
        latitude: float,
        longitude: float,
        eta_minutes: float,
        *,
        radius_m: float = 500.0,
        limit: int = 10,
    ) -> tuple[int, list[SegmentPrediction]]:
        """Tramos dentro de `radius_m`, con su probabilidad, del más cercano al más lejano.

        Devolver **varios tramos y no solo el mejor** es deliberado: si el
        producto señalara una única plaza libre, mandaría a todo el que
        pregunte al mismo hueco. Una lista de zonas con su probabilidad reparte
        y además refleja la incertidumbre real.
        """
        horizon = snap_horizon(eta_minutes)
        features = self.features()
        centroids = self.centroids()

        located = features.join(centroids.drop("bays"), on="segment_id", how="inner")
        if located.is_empty():
            raise PredictionUnavailableError("ningún tramo con features tiene coordenadas")

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
        """Probabilidad de todos los tramos, para pintar el mapa entero."""
        horizon = snap_horizon(eta_minutes)
        features = self.features()
        located = features.join(self.centroids().drop("bays"), on="segment_id", how="inner")
        if located.is_empty():
            raise PredictionUnavailableError("ningún tramo con features tiene coordenadas")
        probabilities = self.model(horizon).predict_proba(located)
        return horizon, located.with_columns(probability=pl.Series(probabilities))
