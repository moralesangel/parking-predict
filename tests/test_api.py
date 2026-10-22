"""Tests de la API del mapa en tiempo real.

Lo que más importa proteger aquí:

- Una plaza sin lectura fiable nunca se devuelve como `free`. Es el error que
  convertiría el producto en un mentiroso.
- El endpoint devuelve por defecto TODAS las plazas, incluidas las no fiables,
  porque el mapa necesita pintar "sin dato" como estado propio.
- La API publica la antigüedad de sus propios datos: si el poller muere, un
  snapshot viejo tiene que notarse.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from parking.api.app import STALE_AFTER_SECONDS, create_app
from parking.api.repository import NoSnapshotError, SnapshotRepository
from parking.config import Settings

BATCH = dt.datetime(2026, 10, 22, 12, 0, 49, tzinfo=dt.UTC)


def reading(
    kerbsideid: int,
    *,
    occupied: bool = False,
    live: bool = True,
    status_age_hours: float = 1.0,
    zone: int | None = 7219,
    batch: dt.datetime = BATCH,
) -> dict:
    return {
        "kerbsideid": kerbsideid,
        "is_occupied": occupied,
        "status_description": "Present" if occupied else "Unoccupied",
        "status_timestamp": batch - dt.timedelta(hours=status_age_hours),
        "lastupdated": batch if live else batch - dt.timedelta(days=30),
        "feed_lastupdated": batch,
        "is_live": live,
        "zone_number": zone,
        "latitude": -37.82,
        "longitude": 144.95,
        "fetched_at": batch,
    }


def write_snapshot(root: Path, *readings: dict, batch: dt.datetime = BATCH) -> Path:
    day = root / "raw" / "realtime" / f"date={batch:%Y-%m-%d}"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"snapshot_{batch:%Y%m%dT%H%M%SZ}.parquet"
    pl.DataFrame(list(readings)).write_parquet(path)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    write_snapshot(
        settings.data_dir,
        reading(1, occupied=False),
        reading(2, occupied=True),
        reading(3, occupied=True, live=False),
        reading(4, occupied=False, status_age_hours=5000),
    )
    return TestClient(create_app(settings))


# --- /bays/live -------------------------------------------------------------


def test_live_returns_every_bay_including_the_unreliable(client: TestClient) -> None:
    body = client.get("/bays/live").json()

    assert len(body["bays"]) == 4
    assert body["summary"]["total_bays"] == 4


def test_unreliable_bays_are_unknown_never_free(client: TestClient) -> None:
    """Un sensor congelado puede estar sobre un coche: no es una plaza libre."""
    bays = {bay["kerbsideid"]: bay for bay in client.get("/bays/live").json()["bays"]}

    assert bays[1]["state"] == "free"
    assert bays[2]["state"] == "occupied"
    assert bays[3]["state"] == "unknown"  # offline, aunque su lectura diga "ocupada"
    assert bays[4]["state"] == "unknown"  # stuck, aunque su lectura diga "libre"


def test_quality_reason_is_exposed(client: TestClient) -> None:
    """`offline` y `stuck` son ambos 'sin dato', pero su causa es distinta."""
    bays = {bay["kerbsideid"]: bay for bay in client.get("/bays/live").json()["bays"]}

    assert bays[3]["quality"] == "offline"
    assert bays[4]["quality"] == "stuck"
    assert bays[1]["quality"] == "ok"


def test_summary_counts_only_reliable_bays_as_free(client: TestClient) -> None:
    summary = client.get("/bays/live").json()["summary"]

    assert summary["reliable_bays"] == 2
    assert summary["free_bays"] == 1
    assert summary["occupied_bays"] == 1
    assert summary["occupancy_pct"] == 50.0
    assert summary["coverage_pct"] == 50.0
    assert summary["by_quality"] == {"ok": 2, "stuck": 1, "offline": 1}


def test_reliable_only_filters_bays_but_not_the_summary(client: TestClient) -> None:
    """La cobertura debe seguir significando lo mismo aunque se filtren las plazas."""
    body = client.get("/bays/live", params={"reliable_only": True}).json()

    assert len(body["bays"]) == 2
    assert all(bay["is_reliable"] for bay in body["bays"])
    assert body["summary"]["total_bays"] == 4
    assert body["summary"]["coverage_pct"] == 50.0


def test_occupancy_is_null_when_nothing_is_reliable(settings: Settings) -> None:
    """`null` significa 'no se sabe'; un 0 significaría 'todo libre'."""
    write_snapshot(settings.data_dir, reading(1, live=False), reading(2, live=False))
    client = TestClient(create_app(settings))

    summary = client.get("/bays/live").json()["summary"]

    assert summary["occupancy_pct"] is None
    assert summary["reliable_bays"] == 0


def test_response_carries_the_data_attribution(client: TestClient) -> None:
    """La licencia CC BY obliga a atribuir también en la respuesta de la API."""
    body = client.get("/bays/live").json()

    assert "City of Melbourne" in body["attribution"]
    assert "CC BY" in body["attribution"]


def test_bays_carry_coordinates_and_zone(client: TestClient) -> None:
    bay = client.get("/bays/live").json()["bays"][0]

    assert bay["latitude"] == -37.82
    assert bay["longitude"] == 144.95
    assert bay["zone_number"] == 7219


def test_null_zone_is_served_as_null(settings: Settings) -> None:
    write_snapshot(settings.data_dir, reading(1, zone=None))
    client = TestClient(create_app(settings))

    assert client.get("/bays/live").json()["bays"][0]["zone_number"] is None


# --- frescura ---------------------------------------------------------------


def test_snapshot_age_is_reported(client: TestClient) -> None:
    snapshot = client.get("/bays/live").json()["snapshot"]

    assert snapshot["age_seconds"] > 0
    assert snapshot["feed_lastupdated"].startswith("2026-10-22T12:00:49")


def test_old_snapshot_is_flagged_as_stale(settings: Settings) -> None:
    """Si el poller muere, servir datos viejos en silencio sería el peor fallo."""
    old = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=STALE_AFTER_SECONDS + 600)
    write_snapshot(settings.data_dir, reading(1, batch=old), batch=old)
    client = TestClient(create_app(settings))

    assert client.get("/bays/live").json()["snapshot"]["is_stale"] is True


def test_fresh_snapshot_is_not_stale(settings: Settings) -> None:
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    write_snapshot(settings.data_dir, reading(1, batch=now), batch=now)
    client = TestClient(create_app(settings))

    assert client.get("/bays/live").json()["snapshot"]["is_stale"] is False


# --- /zones/coverage --------------------------------------------------------


def test_zones_coverage_reports_zones_without_data(settings: Settings) -> None:
    write_snapshot(
        settings.data_dir,
        reading(1, zone=7001),
        reading(2, zone=7002, live=False),
        reading(3, zone=7002, live=False),
    )
    client = TestClient(create_app(settings))

    body = client.get("/zones/coverage").json()

    assert body["zones_total"] == 2
    assert body["zones_without_data"] == 1
    dead = next(zone for zone in body["zones"] if zone["zone_number"] == 7002)
    assert (dead["coverage_pct"], dead["bays"], dead["offline"]) == (0.0, 2, 2)


def test_zones_coverage_min_bays_filters(settings: Settings) -> None:
    write_snapshot(
        settings.data_dir, reading(1, zone=7001), reading(2, zone=7002), reading(3, zone=7002)
    )
    client = TestClient(create_app(settings))

    body = client.get("/zones/coverage", params={"min_bays": 2}).json()

    assert [zone["zone_number"] for zone in body["zones"]] == [7002]


def test_zones_coverage_rejects_min_bays_below_one(client: TestClient) -> None:
    assert client.get("/zones/coverage", params={"min_bays": 0}).status_code == 422


# --- /health ----------------------------------------------------------------


def test_health_is_ok_with_a_fresh_snapshot(settings: Settings) -> None:
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    write_snapshot(settings.data_dir, reading(1, batch=now), batch=now)
    client = TestClient(create_app(settings))

    body = client.get("/health").json()

    assert body["status"] == "ok"


def test_health_reports_stale_without_failing(settings: Settings) -> None:
    """200 con status 'stale' distingue 'API caída' de 'datos parados'."""
    old = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=STALE_AFTER_SECONDS + 600)
    write_snapshot(settings.data_dir, reading(1, batch=old), batch=old)
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "stale"
    assert "poller" in response.json()["detail"]


def test_health_reports_missing_data(settings: Settings) -> None:
    client = TestClient(create_app(settings))

    body = client.get("/health").json()

    assert body["status"] == "no_data"
    assert body["snapshot"] is None


# --- sin datos --------------------------------------------------------------


def test_endpoints_return_503_without_snapshots(settings: Settings) -> None:
    """503 y no 404: el recurso existe, lo que falta es que el poller produzca."""
    client = TestClient(create_app(settings))

    assert client.get("/bays/live").status_code == 503
    assert client.get("/zones/coverage").status_code == 503


# --- frontend ---------------------------------------------------------------


def test_frontend_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>" in response.text


def test_frontend_does_not_shadow_the_api(client: TestClient) -> None:
    """El montaje estático va en `/`; las rutas de la API deben seguir ganando."""
    assert client.get("/bays/live").status_code == 200
    assert client.get("/health").status_code == 200


def test_api_still_works_without_the_frontend(settings: Settings, tmp_path: Path) -> None:
    """Un despliegue sin `web/` debe servir la API igual y dar 404 en `/`."""
    write_snapshot(settings.data_dir, reading(1))
    headless = settings.model_copy(update={"web_dir": tmp_path / "no-existe"})
    client = TestClient(create_app(headless))

    assert client.get("/bays/live").status_code == 200
    assert client.get("/").status_code == 404


# --- repositorio ------------------------------------------------------------


def test_repository_reuses_the_cached_frame(settings: Settings) -> None:
    write_snapshot(settings.data_dir, reading(1))
    repository = SnapshotRepository(settings)

    first = repository.load()
    second = repository.load()

    assert first.frame is second.frame, "el mismo lote no debe releerse del disco"


def test_repository_age_grows_between_calls(settings: Settings) -> None:
    """La caché reutiliza el frame pero la antigüedad debe seguir avanzando."""
    write_snapshot(settings.data_dir, reading(1))
    repository = SnapshotRepository(settings)
    now = dt.datetime.now(dt.UTC)

    first = repository.load(now=now)
    second = repository.load(now=now + dt.timedelta(seconds=300))

    assert second.meta.age_seconds == pytest.approx(first.meta.age_seconds + 300)


def test_repository_picks_up_a_new_batch(settings: Settings) -> None:
    write_snapshot(settings.data_dir, reading(1))
    repository = SnapshotRepository(settings)
    repository.load()

    later = BATCH + dt.timedelta(minutes=2)
    write_snapshot(settings.data_dir, reading(1, batch=later), reading(2, batch=later), batch=later)

    assert repository.load().frame.height == 2


def test_repository_orders_by_batch_not_mtime(settings: Settings) -> None:
    """Ordenar por mtime daría el orden equivocado al recuperar un histórico."""
    newer = BATCH + dt.timedelta(minutes=10)
    write_snapshot(settings.data_dir, reading(1, batch=newer), batch=newer)
    # Escrito después, pero su lote es anterior.
    write_snapshot(settings.data_dir, reading(2, batch=BATCH), batch=BATCH)

    assert SnapshotRepository(settings).load().meta.feed_lastupdated == newer


def test_repository_without_snapshots_raises(settings: Settings) -> None:
    with pytest.raises(NoSnapshotError, match="poller"):
        SnapshotRepository(settings).load()
