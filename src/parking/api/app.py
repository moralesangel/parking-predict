"""API del mapa en tiempo real.

Sirve el último snapshot que capturó el poller, nunca consulta la fuente
directamente. Tres endpoints:

- `GET /bays/live`        estado de cada plaza con su bandera de fiabilidad
- `GET /predict`          probabilidad de encontrar sitio al llegar, por zona
- `GET /zones/coverage`   cobertura por zona, para saber dónde no hay datos
- `GET /health`           si los datos que sirve están frescos

Además monta el frontend estático en `/`.
"""

from __future__ import annotations

import logging
from typing import Annotated

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from parking.api.prediction import (
    MODEL_CAVEAT,
    PredictionService,
    PredictionUnavailableError,
)
from parking.api.repository import LiveSnapshot, NoSnapshotError, SnapshotRepository
from parking.api.schemas import (
    Bay,
    BayState,
    HealthResponse,
    LiveResponse,
    PredictResponse,
    Quality,
    QualityBreakdown,
    SnapshotInfo,
    Summary,
    ZoneCoverage,
    ZonePrediction,
    ZonesResponse,
)
from parking.config import Settings, get_settings
from parking.quality.coverage import coverage_by_zone

logger = logging.getLogger(__name__)

ATTRIBUTION = (
    "Datos de City of Melbourne Open Data (https://data.melbourne.vic.gov.au/), licencia CC BY 4.0."
)

#: Un lote más viejo que esto sugiere que el poller no está escribiendo. Son 10
#: ciclos del feed (120 s): tolera varios fallos de red seguidos, que el poller
#: registra y omite sin morir, pero no una caída prolongada.
STALE_AFTER_SECONDS = 1200.0


def get_repository(request: Request) -> SnapshotRepository:
    """Repositorio compartido, guardado en el estado de la aplicación."""
    return request.app.state.repository


def get_predictor(request: Request) -> PredictionService:
    """Servicio de predicción compartido, con su caché de features."""
    return request.app.state.predictor


# A nivel de módulo a propósito: con `from __future__ import annotations` las
# anotaciones son cadenas y FastAPI las resuelve contra los globales del módulo.
# Definido dentro de `create_app` sería invisible para él y las rutas pedirían
# `repository` como parámetro de query.
RepositoryDep = Annotated[SnapshotRepository, Depends(get_repository)]
PredictorDep = Annotated[PredictionService, Depends(get_predictor)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construye la aplicación. Recibe `settings` para poder testearla aislada."""
    settings = settings or get_settings()
    repository = SnapshotRepository(settings)

    app = FastAPI(
        title="Parking Predict API",
        version="0.1.0",
        description=(
            "Ocupación de aparcamiento en calle en Melbourne. "
            "Fase 3: estado en tiempo real, sin predicción todavía.\n\n"
            f"{ATTRIBUTION}"
        ),
    )
    app.state.settings = settings
    app.state.repository = repository
    app.state.predictor = PredictionService(settings)

    def _snapshot_info(snapshot: LiveSnapshot) -> SnapshotInfo:
        age = snapshot.meta.age_seconds
        return SnapshotInfo(
            feed_lastupdated=snapshot.meta.feed_lastupdated,
            served_at=snapshot.meta.served_at,
            age_seconds=round(age, 1),
            is_stale=age > STALE_AFTER_SECONDS,
        )

    def _load(repository: SnapshotRepository) -> LiveSnapshot:
        try:
            return repository.load()
        except NoSnapshotError as exc:
            # 503 y no 404: el recurso existe conceptualmente, lo que falta es
            # que el poller haya producido datos. Es un problema operativo.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/bays/live", response_model=LiveResponse, tags=["tiempo real"])
    def bays_live(
        repository: RepositoryDep,
        reliable_only: Annotated[
            bool,
            Query(
                description="Si es true, omite las plazas sin lectura fiable. "
                "Por defecto se devuelven TODAS, porque el mapa necesita "
                "pintar 'sin dato' como un estado propio."
            ),
        ] = False,
    ) -> LiveResponse:
        snapshot = _load(repository)
        frame = snapshot.frame

        summary = _build_summary(frame)
        if reliable_only:
            frame = frame.filter(pl.col("is_reliable"))

        bays = [
            Bay(
                kerbsideid=row["kerbsideid"],
                state=_bay_state(row["is_reliable"], row["is_occupied"]),
                quality=Quality(row["quality"]),
                is_reliable=row["is_reliable"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                zone_number=row["zone_number"],
                status_age_hours=round(row["status_age_hours"], 2),
            )
            for row in frame.select(
                "kerbsideid",
                "is_reliable",
                "is_occupied",
                "quality",
                "latitude",
                "longitude",
                "zone_number",
                "status_age_hours",
            ).iter_rows(named=True)
        ]

        return LiveResponse(
            snapshot=_snapshot_info(snapshot),
            summary=summary,
            bays=bays,
            attribution=ATTRIBUTION,
        )

    @app.get("/predict", response_model=PredictResponse, tags=["prediccion"])
    def predict(
        repository: RepositoryDep,
        predictor: PredictorDep,
        lat: Annotated[float, Query(ge=-90, le=90, description="Latitud del destino")],
        lon: Annotated[float, Query(ge=-180, le=180, description="Longitud del destino")],
        eta_minutes: Annotated[
            float,
            Query(
                ge=0,
                le=120,
                description="Minutos hasta la llegada. Se ajusta al horizonte "
                "entrenado mas cercano (5, 10, 15 o 30).",
            ),
        ] = 15,
        radius_m: Annotated[float, Query(ge=50, le=3000)] = 500,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> PredictResponse:
        """Probabilidad de encontrar plaza en las zonas cercanas al destino.

        Devuelve **varias zonas y no la mejor**: señalar una sola mandaría a
        todo el que pregunte al mismo hueco. Van ordenadas por distancia, no
        por probabilidad, por la misma razón.
        """
        snapshot = _load(repository)
        try:
            horizon, zones = predictor.predict_near(
                lat, lon, eta_minutes, radius_m=radius_m, limit=limit
            )
        except PredictionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return PredictResponse(
            snapshot=_snapshot_info(snapshot),
            eta_minutes=eta_minutes,
            horizon_minutes=horizon,
            zones=[
                ZonePrediction(
                    segment_id=zone.segment_id,
                    label=_segment_label(zone.segment_id),
                    probability=round(zone.probability, 4),
                    free_bays=zone.free_bays,
                    total_bays=zone.total_bays,
                    latitude=zone.latitude,
                    longitude=zone.longitude,
                    distance_m=zone.distance_m,
                    is_interpolated=zone.is_interpolated,
                )
                for zone in zones
            ],
            model_caveat=MODEL_CAVEAT,
            attribution=ATTRIBUTION,
        )

    @app.get("/zones/coverage", response_model=ZonesResponse, tags=["calidad"])
    def zones_coverage(
        repository: RepositoryDep,
        min_bays: Annotated[int, Query(ge=1, description="Omite zonas más pequeñas")] = 1,
    ) -> ZonesResponse:
        snapshot = _load(repository)
        coverage = coverage_by_zone(snapshot.frame, min_bays=min_bays)

        zones = [
            ZoneCoverage(
                zone_number=row["zone_number"],
                bays=row["bays"],
                reliable=row["reliable"],
                stuck=row["stuck"],
                offline=row["offline"],
                coverage_pct=row["coverage_pct"],
            )
            for row in coverage.iter_rows(named=True)
        ]
        return ZonesResponse(
            snapshot=_snapshot_info(snapshot),
            zones_total=len(zones),
            zones_without_data=sum(1 for zone in zones if zone.reliable == 0),
            zones=zones,
            attribution=ATTRIBUTION,
        )

    @app.get("/health", response_model=HealthResponse, tags=["operación"])
    def health(repository: RepositoryDep) -> HealthResponse:
        """Estado de la API. Nunca devuelve error: informa.

        Un 200 con `status: "stale"` es más útil que un 503 para un panel de
        monitorización, que necesita distinguir "la API está caída" de "la API
        está viva pero los datos no avanzan".
        """
        try:
            snapshot = repository.load()
        except NoSnapshotError as exc:
            return HealthResponse(status="no_data", detail=str(exc))

        info = _snapshot_info(snapshot)
        if info.is_stale:
            return HealthResponse(
                status="stale",
                detail=(
                    f"el último lote tiene {info.age_seconds:.0f} s "
                    f"(umbral {STALE_AFTER_SECONDS:.0f} s); revisa el poller"
                ),
                snapshot=info,
            )
        return HealthResponse(
            status="ok", detail=f"lote de hace {info.age_seconds:.0f} s", snapshot=info
        )

    if settings.web_dir.is_dir():
        # html=True sirve index.html en la raíz. El montaje va al final para que
        # no tape las rutas de la API, que se registraron antes.
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    else:  # pragma: no cover - solo si falta el directorio en el despliegue
        # No se monta nada: las rutas de la API siguen funcionando y `/` da 404,
        # que es más honesto que servir una página vacía.
        logger.warning("no existe %s; el frontend no se sirve", settings.web_dir)

    return app


def _segment_label(segment_id: str) -> str:
    """`COLLINS STREET|EXHIBITION STREET|RUSSELL STREET` -> texto legible."""
    parts = segment_id.split("|")
    if len(parts) != 3:
        return segment_id
    street, first, second = (part.title() for part in parts)
    return f"{street}, entre {first} y {second}"


def _bay_state(is_reliable: bool, is_occupied: bool) -> BayState:
    """Estado para el mapa.

    Sin lectura fiable el estado es `unknown`, nunca `free`: una plaza cuyo
    sensor lleva meses congelado puede estar ocupada ahora mismo.
    """
    if not is_reliable:
        return BayState.UNKNOWN
    return BayState.OCCUPIED if is_occupied else BayState.FREE


def _build_summary(frame: pl.DataFrame) -> Summary:
    """Resumen calculado siempre sobre el frame completo.

    Se construye antes de cualquier filtrado para que `coverage_pct` siga
    significando lo mismo cuando el cliente pide `reliable_only=true`.
    """
    total = frame.height
    reliable = frame.filter(pl.col("is_reliable"))
    occupied = int(reliable["is_occupied"].sum()) if reliable.height else 0
    counts = dict(frame.group_by("quality").agg(n=pl.len()).iter_rows())

    return Summary(
        total_bays=total,
        reliable_bays=reliable.height,
        free_bays=reliable.height - occupied,
        occupied_bays=occupied,
        occupancy_pct=round(occupied / reliable.height * 100, 1) if reliable.height else None,
        coverage_pct=round(reliable.height / total * 100, 1) if total else 0.0,
        by_quality=QualityBreakdown(
            ok=counts.get("ok", 0),
            stuck=counts.get("stuck", 0),
            offline=counts.get("offline", 0),
        ),
    )


def main() -> None:  # pragma: no cover - punto de entrada
    """Arranca uvicorn. Para desarrollo; en Docker lo lanza el ENTRYPOINT."""
    import time

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    # UTC en los logs, igual que en el poller: los datos están en UTC.
    logging.Formatter.converter = time.gmtime
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
