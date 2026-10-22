"""Informe de monitorización sobre los datos acumulados.

    uv run python scripts/monitor.py

Las tres señales que pide la Fase 7, cada una respondiendo a una pregunta
distinta:

1. **Degradación** — ¿el modelo sigue batiendo a la persistencia *hoy*?
2. **Drift** — ¿la ocupación de hoy se parece a la de entrenamiento?
3. **Cobertura** — ¿sobre cuánta ciudad seguimos teniendo datos?

Escribe `data/processed/monitoring/report.json`, que es lo que sirve el
endpoint `/monitoring`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import polars as pl

from parking.api.prediction import AVAILABLE_HORIZONS
from parking.config import get_settings
from parking.features.live import (
    build_live_features,
    densify,
    load_bay_segments,
    segment_occupancy,
)
from parking.models.gbm import GradientBoostingModel
from parking.models.registry import ModelRegistry
from parking.monitoring import backtest, coverage_trend, occupancy_drift
from parking.quality.coverage import coverage_by_zone
from parking.quality.flags import annotate_quality, quality_summary
from parking.quality.gaps import continuity_report, load_series, snapshot_timestamps

logger = logging.getLogger("monitor")


def jsonable(value):
    """Fechas a ISO, para que el informe sea JSON válido."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-month", default="2019-06", help="mes de referencia para el drift"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    settings = get_settings()
    report: dict = {}

    # --- datos en vivo ---
    raw = load_series(settings.realtime_raw_dir)
    if raw.is_empty():
        print("no hay snapshots; arranca el poller", file=sys.stderr)
        return 1
    series = annotate_quality(raw, stuck_after_hours=settings.stuck_after_hours)

    stamps = snapshot_timestamps(settings.realtime_raw_dir)
    continuity = continuity_report(
        stamps,
        expected_interval_seconds=settings.poll_interval_seconds,
        tolerance_seconds=settings.gap_after_seconds,
    )
    report["continuity"] = jsonable(
        {k: v for k, v in continuity.items() if k != "gaps"}
        | {"gap_count": len(continuity["gaps"])}
    )

    # --- 3. cobertura ---
    latest = series.filter(pl.col("feed_lastupdated") == series["feed_lastupdated"].max())
    zones = coverage_by_zone(latest, min_bays=10)
    report["coverage"] = jsonable(
        {
            "bays_total": latest.height,
            "bays_reliable": int(latest["is_reliable"].sum()),
            "coverage_pct": round(float(latest["is_reliable"].mean()) * 100, 1),
            "zones_measured": zones.height,
            "zones_without_data": int((zones["coverage_pct"] == 0).sum()),
            "by_quality": quality_summary(latest).to_dicts(),
            "trend": coverage_trend(series).to_dicts(),
        }
    )
    logger.info(
        "cobertura: %s%% (%s de %s plazas)",
        report["coverage"]["coverage_pct"],
        report["coverage"]["bays_reliable"],
        report["coverage"]["bays_total"],
    )

    # --- features en vivo, base de las otras dos señales ---
    mapping = load_bay_segments(settings.data_dir / "reference" / "parking_bays.json")
    occupancy = segment_occupancy(series, mapping)
    live = build_live_features(densify(occupancy))

    # --- 2. drift ---
    reference_path = (
        settings.processed_dir / "dataset" / f"month={args.reference_month}" / "part.parquet"
    )
    if reference_path.exists():
        reference = pl.read_parquet(reference_path, columns=["occupancy_rate"])
        drift = occupancy_drift(reference, live)
        report["drift"] = jsonable(
            {
                "reference": args.reference_month,
                "psi": round(drift.psi, 4),
                "level": drift.level,
                "needs_attention": drift.needs_attention,
                "reference_n": drift.reference_n,
                "current_n": drift.current_n,
                "bins": drift.bins.to_dicts(),
            }
        )
        logger.info("drift de ocupacion: PSI %.4f (%s)", drift.psi, drift.level)
    else:
        report["drift"] = {"error": f"falta {reference_path}"}

    # --- 1. degradacion ---
    registry = ModelRegistry(settings.processed_dir / "registry")
    model_dir = (
        registry.production_dir
        if (registry.production_dir / f"lightgbm_h{AVAILABLE_HORIZONS[0]}.txt").exists()
        else settings.processed_dir / "models"
    )
    report["model_source"] = str(model_dir.name)

    degradation = []
    for horizon in AVAILABLE_HORIZONS:
        try:
            model = GradientBoostingModel.load(model_dir, f"lightgbm_h{horizon}")
        except FileNotFoundError:
            continue
        result = backtest(live, model.predict_proba(live), horizon)
        if result is None:
            continue
        degradation.append(result.as_dict())
        logger.info(
            "h=%2d  modelo %.5f  persistencia %.5f  skill %+.1f%%%s",
            horizon,
            result.model_brier,
            result.persistence_brier,
            result.skill_vs_persistence * 100,
            "  <-- DEGRADADO" if result.is_degraded else "",
        )
    report["degradation"] = degradation
    report["registry_history"] = registry.history()[-10:]

    destination = settings.processed_dir / "monitoring"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("informe en %s", destination / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
