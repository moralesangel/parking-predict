"""Job de reentrenamiento con los datos acumulados por el poller.

    uv run python scripts/retrain.py --horizon 15

Pensado para correr programado (cron, Task Scheduler o el servicio del
compose). Hace cuatro cosas, en este orden:

1. Construye el dataset desde los snapshots acumulados.
2. **Comprueba que hay datos suficientes y se niega a entrenar si no.**
3. Entrena sobre la parte antigua y evalúa sobre la reciente (split temporal).
4. Promociona el modelo **solo si bate a la persistencia** y no empeora al que
   ya está en producción.

El punto 2 importa tanto como el resto. Un job que entrena con lo que haya
produce, antes o después, un modelo basado en seis horas de un martes; que se
niegue y lo diga es el comportamiento correcto, no un fallo.

Salida distinta de cero solo si el job **falla**. Que no promocione no es un
fallo: es una decisión, y se registra en el histórico del registro.
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

#: Mínimos para que reentrenar signifique algo. Son juicios, no umbrales
#: derivados de los datos, y por eso van como parámetros:
#: - 14 días cubren dos ciclos semanales completos. Con menos, el modelo no ha
#:   visto un fin de semana suficientes veces para distinguirlo del ruido.
#: - 50 000 filas con objetivo observado evitan entrenar sobre un puñado de
#:   tramos durante una caída del feed.
MIN_DAYS = 14
MIN_LABELLED_ROWS = 50_000


def build_training_frame(settings) -> tuple[pl.DataFrame, dict]:
    """Dataset con objetivos a partir de los snapshots acumulados."""
    raw = load_series(settings.realtime_raw_dir)
    if raw.is_empty():
        raise SystemExit("no hay snapshots; arranca el poller")

    series = annotate_quality(raw, stuck_after_hours=settings.stuck_after_hours)
    mapping = load_bay_segments(settings.data_dir / "reference" / "parking_bays.json")
    occupancy = segment_occupancy(series, mapping)
    if occupancy.is_empty():
        raise SystemExit("ninguna plaza fiable se pudo asignar a un tramo")

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
        help="fraccion FINAL de la ventana que se reserva para evaluar",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    settings = get_settings()
    registry = ModelRegistry(settings.processed_dir / "registry")
    version = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")

    features, span = build_training_frame(settings)
    logger.info(
        "datos acumulados: %.2f dias | %d tramos | %d filas (%s -> %s)",
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

    # --- la puerta de "hay datos suficientes" ---
    reasons = []
    if span["days"] < args.min_days:
        reasons.append(f"solo {span['days']:.2f} dias de datos (minimo {args.min_days})")
    if labelled.height < args.min_rows:
        reasons.append(f"solo {labelled.height} filas etiquetadas (minimo {args.min_rows})")
    if reasons:
        reason = "datos insuficientes: " + "; ".join(reasons)
        logger.warning("NO se reentrena. %s", reason)
        logger.warning(
            "Es el comportamiento correcto: un modelo entrenado con esta ventana "
            "no habria visto suficientes ciclos semanales."
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

    # --- split temporal ---
    cutoff = labelled["timestamp"].quantile(1 - args.holdout_fraction)
    train = labelled.filter(pl.col("timestamp") < cutoff)
    holdout = labelled.filter(pl.col("timestamp") >= cutoff)
    logger.info(
        "train %d filas | holdout %d filas (corte %s)", train.height, holdout.height, cutoff
    )

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
        "candidato %.5f | persistencia %.5f | produccion %s",
        candidate_brier,
        persistence_brier,
        f"{production_brier:.5f}" if production_brier is not None else "ninguno",
    )
    logger.info("%s: %s", "PROMOCIONA" if decision.promoted else "NO promociona", decision.reason)

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
