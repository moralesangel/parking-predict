"""Real-time map API.

Serves the latest snapshot the poller captured; it never queries the source
directly. Endpoints:

- `GET /bays/live`        state of every bay with its reliability flag
- `GET /predict`          probability of finding a space on arrival, per zone
- `GET /zones/coverage`   coverage per zone, to know where there is no data
- `GET /health`           whether the data being served is fresh
- `GET /monitoring`       latest operations report (degradation, drift, coverage)

It also mounts the static frontend at `/`.
"""

from __future__ import annotations

import datetime as dt
import json
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
    "Data from City of Melbourne Open Data (https://data.melbourne.vic.gov.au/), "
    "licensed CC BY 4.0."
)

#: A batch older than this suggests the poller has stopped writing. It is 10
#: feed cycles (120 s): tolerant of several consecutive network failures, which
#: the poller logs and skips without dying, but not of a prolonged outage.
STALE_AFTER_SECONDS = 1200.0


def get_repository(request: Request) -> SnapshotRepository:
    """Shared repository, kept on the application state."""
    return request.app.state.repository


def get_predictor(request: Request) -> PredictionService:
    """Shared prediction service, with its feature cache."""
    return request.app.state.predictor


# At module level on purpose: with `from __future__ import annotations` the
# annotations are strings and FastAPI resolves them against the module globals.
# Defined inside `create_app` they would be invisible to FastAPI and the routes
# would ask for `repository` as a query parameter.
RepositoryDep = Annotated[SnapshotRepository, Depends(get_repository)]
PredictorDep = Annotated[PredictionService, Depends(get_predictor)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Takes `settings` so it can be tested in isolation."""
    settings = settings or get_settings()
    repository = SnapshotRepository(settings)

    app = FastAPI(
        title="Parking Predict API",
        version="0.1.0",
        description=(
            "On-street parking occupancy in Melbourne, with arrival-time "
            "prediction.\n\n"
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
            # 503 rather than 404: the resource exists conceptually, what is
            # missing is for the poller to have produced data. Operational.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/bays/live", response_model=LiveResponse, tags=["real time"])
    def bays_live(
        repository: RepositoryDep,
        reliable_only: Annotated[
            bool,
            Query(
                description="When true, omit bays without a reliable reading. "
                "By default ALL bays are returned, because the map needs to "
                "paint 'no data' as a state of its own."
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

    @app.get("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(
        repository: RepositoryDep,
        predictor: PredictorDep,
        lat: Annotated[float, Query(ge=-90, le=90, description="Destination latitude")],
        lon: Annotated[float, Query(ge=-180, le=180, description="Destination longitude")],
        eta_minutes: Annotated[
            float,
            Query(
                ge=0,
                le=120,
                description="Minutes until arrival. Snapped to the nearest "
                "trained horizon (5, 10, 15 or 30).",
            ),
        ] = 15,
        radius_m: Annotated[float, Query(ge=50, le=3000)] = 500,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> PredictResponse:
        """Probability of finding a space in the zones near the destination.

        Returns **several zones rather than the best one**: pointing at a
        single spot would send everyone who asks to the same place. They are
        ordered by distance, not by probability, for the same reason.
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

    @app.get("/zones/coverage", response_model=ZonesResponse, tags=["quality"])
    def zones_coverage(
        repository: RepositoryDep,
        min_bays: Annotated[int, Query(ge=1, description="Skip smaller zones")] = 1,
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

    @app.get("/monitoring", tags=["operations"])
    def monitoring() -> dict:
        """Latest monitoring report, exactly as `scripts/monitor.py` left it.

        The API **does not recompute it**: doing so would walk the whole
        accumulated series inside an HTTP request. The scheduled job produces
        the report and this only serves it, so `generated_at` is included to
        make it visible when the job has stopped running.
        """
        path = settings.processed_dir / "monitoring" / "report.json"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"no report at {path}; run scripts/monitor.py",
            )
        report = json.loads(path.read_text(encoding="utf-8"))
        generated = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
        age = (dt.datetime.now(dt.UTC) - generated).total_seconds()
        return {
            "generated_at": generated.isoformat(),
            "age_seconds": round(age, 1),
            # A stale monitoring report is itself a symptom: it means the
            # scheduled job is not running.
            "is_stale": age > 24 * 3600,
            **report,
        }

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(repository: RepositoryDep) -> HealthResponse:
        """API status. Never returns an error: it reports.

        A 200 with `status: "stale"` is more useful than a 503 for a monitoring
        dashboard, which needs to tell "the API is down" apart from "the API is
        alive but the data has stopped moving".
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
                    f"the latest batch is {info.age_seconds:.0f} s old "
                    f"(threshold {STALE_AFTER_SECONDS:.0f} s); check the poller"
                ),
                snapshot=info,
            )
        return HealthResponse(
            status="ok", detail=f"batch from {info.age_seconds:.0f} s ago", snapshot=info
        )

    if settings.web_dir.is_dir():
        # html=True serves index.html at the root. The mount goes last so it
        # does not shadow the API routes, which were registered above.
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    else:  # pragma: no cover - only when the directory is missing in a deployment
        # Nothing is mounted: the API routes keep working and `/` returns 404,
        # which is more honest than serving an empty page.
        logger.warning("%s does not exist; the frontend is not served", settings.web_dir)

    return app


def _segment_label(segment_id: str) -> str:
    """`COLLINS STREET|EXHIBITION STREET|RUSSELL STREET` -> readable text."""
    parts = segment_id.split("|")
    if len(parts) != 3:
        return segment_id
    street, first, second = (part.title() for part in parts)
    return f"{street}, between {first} and {second}"


def _bay_state(is_reliable: bool, is_occupied: bool) -> BayState:
    """State for the map.

    Without a reliable reading the state is `unknown`, never `free`: a bay
    whose sensor has been frozen for months may well be occupied right now.
    """
    if not is_reliable:
        return BayState.UNKNOWN
    return BayState.OCCUPIED if is_occupied else BayState.FREE


def _build_summary(frame: pl.DataFrame) -> Summary:
    """Summary always computed over the complete frame.

    Built before any filtering so that `coverage_pct` keeps meaning the same
    thing when the client asks for `reliable_only=true`.
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


def main() -> None:  # pragma: no cover - entry point
    """Start uvicorn. For development; in Docker the CMD launches it."""
    import time

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    # UTC in the logs, same as the poller: the data is in UTC.
    logging.Formatter.converter = time.gmtime
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
