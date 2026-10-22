"""Modelos de respuesta de la API.

Se declaran explícitamente en lugar de devolver diccionarios para que el esquema
de OpenAPI documente la semántica —en particular la de `quality`, que es la
parte que un consumidor puede malinterpretar con más facilidad.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field


class Quality(StrEnum):
    """Fiabilidad de la lectura de una plaza.

    Los tres valores vienen de `parking.quality.flags`. Se exponen tal cual
    porque el cliente necesita distinguirlos: `offline` y `stuck` son ambos
    "sin dato fiable", pero su causa es distinta y conviene poder mostrarla.
    """

    OK = "ok"
    STUCK = "stuck"
    OFFLINE = "offline"


class BayState(StrEnum):
    """Estado de una plaza tal como lo pinta el mapa.

    `UNKNOWN` es un estado de primera clase, no un hueco: 2767 de las 6324
    plazas del feed no dan lectura fiable, y pintarlas como libres sería
    inventarse plazas que quizá no existen.
    """

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class Bay(BaseModel):
    """Una plaza en el snapshot actual."""

    kerbsideid: int = Field(description="Identificador de la plaza en el feed de origen")
    state: BayState = Field(description="free, occupied o unknown (sin lectura fiable)")
    quality: Quality = Field(description="Por qué la lectura es o no fiable")
    is_reliable: bool = Field(description="True solo si quality == ok")
    latitude: float
    longitude: float
    zone_number: int | None = Field(
        default=None, description="Zona de restricción; nula en 520 de las 6324 plazas"
    )
    status_age_hours: float = Field(
        description="Horas desde el último cambio de estado, medidas contra el sello del lote"
    )


class QualityBreakdown(BaseModel):
    """Recuento de plazas por estado de calidad."""

    ok: int
    stuck: int
    offline: int


class Summary(BaseModel):
    """Resumen del snapshot. Es lo que permite leer la cifra correcta de un vistazo."""

    total_bays: int = Field(description="Plazas que publica el feed")
    reliable_bays: int = Field(description="Plazas con lectura utilizable")
    free_bays: int = Field(description="Plazas libres entre las fiables")
    occupied_bays: int = Field(description="Plazas ocupadas entre las fiables")
    occupancy_pct: float | None = Field(
        description="Ocupación sobre plazas fiables. Nula si no hay ninguna: "
        "un 0 significaría 'todo libre', que es lo contrario de 'no se sabe'"
    )
    coverage_pct: float = Field(description="Porcentaje de plazas con lectura fiable")
    by_quality: QualityBreakdown


class SnapshotInfo(BaseModel):
    """De cuándo son los datos que se están sirviendo."""

    feed_lastupdated: dt.datetime = Field(description="Sello del lote del feed, en UTC")
    served_at: dt.datetime
    age_seconds: float = Field(description="Antigüedad del lote en el momento de servir")
    is_stale: bool = Field(
        description="True si el lote es más viejo de lo tolerable: el poller puede estar caído"
    )


class LiveResponse(BaseModel):
    """Respuesta de `GET /bays/live`."""

    snapshot: SnapshotInfo
    summary: Summary
    bays: list[Bay]
    attribution: str


class ZoneCoverage(BaseModel):
    """Cobertura de sensores de una zona."""

    zone_number: int | None
    bays: int
    reliable: int
    stuck: int
    offline: int
    coverage_pct: float


class ZonesResponse(BaseModel):
    """Respuesta de `GET /zones/coverage`."""

    snapshot: SnapshotInfo
    zones_total: int
    zones_without_data: int = Field(description="Zonas sin ni una lectura fiable")
    zones: list[ZoneCoverage]
    attribution: str


class HealthResponse(BaseModel):
    """Respuesta de `GET /health`."""

    status: str = Field(description="ok, stale o no_data")
    detail: str
    snapshot: SnapshotInfo | None = None


class ZonePrediction(BaseModel):
    """Probabilidad de encontrar sitio en un tramo de calle."""

    segment_id: str = Field(description="CALLE|CRUCE_A|CRUCE_B, las dos delimitadoras ordenadas")
    label: str = Field(description="El mismo tramo en texto legible")
    probability: float = Field(
        ge=0, le=1, description="P(al menos una plaza libre al llegar), calibrada"
    )
    free_bays: int = Field(description="Plazas libres AHORA entre las fiables")
    total_bays: int = Field(description="Plazas con lectura fiable en el tramo")
    latitude: float
    longitude: float
    distance_m: float = Field(description="Distancia desde el destino consultado")
    is_interpolated: bool = Field(
        description="True si el ultimo estado se arrastro de un lote anterior por un hueco"
    )


class PredictResponse(BaseModel):
    """Respuesta de `GET /predict`."""

    snapshot: SnapshotInfo
    eta_minutes: float = Field(description="Lo que pidio el cliente")
    horizon_minutes: int = Field(
        description="Horizonte realmente usado: se ajusta al modelo entrenado mas cercano"
    )
    zones: list[ZonePrediction] = Field(
        description="Tramos cercanos ordenados por distancia, no por probabilidad: "
        "ordenarlos por probabilidad mandaria a todo el mundo al mismo sitio"
    )
    model_caveat: str = Field(description="Por que esto es una demostracion y no un servicio")
    attribution: str
